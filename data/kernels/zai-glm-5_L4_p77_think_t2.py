import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Scale + BatchNorm + Global Average Pooling
fused_bn_pool_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Compute spatial sums and sums of squares for each (N, C) channel
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ sum_out,
    scalar_t* __restrict__ sum_sq_out,
    const int N, const int C, const int spatial_size,
    const scalar_t scale_factor) {
    
    int nc = blockIdx.x; 
    int n = nc / C;
    int c = nc % C;
    
    int tid = threadIdx.x;
    
    scalar_t sum = 0.0f;
    scalar_t sum_sq = 0.0f;
    
    const scalar_t* x_ptr = input + nc * spatial_size;
    
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        scalar_t val = x_ptr[i] * scale_factor;
        sum += val;
        sum_sq += val * val;
    }
    
    __shared__ scalar_t s_sum[256];
    __shared__ scalar_t s_sum_sq[256];
    
    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        sum_out[nc] = s_sum[0];
        sum_sq_out[nc] = s_sum_sq[0];
    }
}

// Kernel 2: Finalize Batch Normalization and Global Pooling
template <typename scalar_t>
__global__ void finalize_bn_pool_kernel(
    const scalar_t* __restrict__ spatial_sums,
    const scalar_t* __restrict__ spatial_sums_sq,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ running_mean,
    scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    int N, int C, int spatial_size,
    scalar_t eps, scalar_t momentum, bool training) {
    
    int c = blockIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory for spatial sums (N), spatial sums sq (N), and batch stats (mean, var)
    extern __shared__ char shared_mem[];
    scalar_t* s_sums = (scalar_t*)shared_mem;
    scalar_t* s_sums_sq = s_sums + N;
    scalar_t* s_batch_mean = s_sums_sq + N;
    scalar_t* s_batch_var = s_batch_mean + 1;
    
    // Load spatial sums to shared memory
    if (tid < N) {
        s_sums[tid] = spatial_sums[tid * C + c];
        s_sums_sq[tid] = spatial_sums_sq[tid * C + c];
    }
    __syncthreads();
    
    scalar_t mean, var;
    
    if (training) {
        // Thread 0 computes batch statistics
        if (tid == 0) {
            scalar_t batch_sum = 0;
            scalar_t batch_sum_sq = 0;
            for (int i = 0; i < N; ++i) {
                batch_sum += s_sums[i];
                batch_sum_sq += s_sums_sq[i];
            }
            
            int total_count = N * spatial_size;
            mean = batch_sum / total_count;
            // Biased variance for normalization
            var = batch_sum_sq / total_count - mean * mean;
            
            // Update running stats
            // PyTorch uses unbiased variance for running stats update
            scalar_t var_unbiased = var * static_cast<scalar_t>(total_count) / (total_count - 1);
            
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var_unbiased;
            
            s_batch_mean[0] = mean;
            s_batch_var[0] = var;
        }
        __syncthreads();
        
        mean = s_batch_mean[0];
        var = s_batch_var[0];
    } else {
        // Evaluation mode
        if (tid == 0) {
            s_batch_mean[0] = running_mean[c];
            s_batch_var[0] = running_var[c];
        }
        __syncthreads();
        mean = s_batch_mean[0];
        var = s_batch_var[0];
    }
    
    // Compute output
    if (tid < N) {
        scalar_t sample_mean = s_sums[tid] / spatial_size;
        scalar_t norm_val = (sample_mean - mean) / std::sqrt(var + eps);
        out[tid * C + c] = norm_val * weight[c] + bias[c];
    }
}

torch::Tensor fused_bn_pool_cuda(
    torch::Tensor input, 
    float scale_factor,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps, float momentum, bool training) {
    
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int spatial_size = sizes[2] * sizes[3] * sizes[4];
    
    auto options = input.options();
    auto spatial_sums = torch::zeros({N, C}, options);
    auto spatial_sums_sq = torch::zeros({N, C}, options);
    auto out = torch::zeros({N, C}, options);
    
    int threads = 256;
    int blocks = N * C;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "compute_stats_kernel", ([&] {
        compute_stats_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            spatial_sums.data_ptr<scalar_t>(),
            spatial_sums_sq.data_ptr<scalar_t>(),
            N, C, spatial_size,
            static_cast<scalar_t>(scale_factor)
        );
    }));
    
    int blocks2 = C;
    int threads2 = 256;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "finalize_bn_pool_kernel", ([&] {
        size_t shared_mem_size = (2 * N + 2) * sizeof(scalar_t);
        finalize_bn_pool_kernel<scalar_t><<<blocks2, threads2, shared_mem_size>>>(
            spatial_sums.data_ptr<scalar_t>(),
            spatial_sums_sq.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            running_mean.data_ptr<scalar_t>(),
            running_var.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            N, C, spatial_size,
            static_cast<scalar_t>(eps),
            static_cast<scalar_t>(momentum),
            training
        );
    }));
    
    return out.view({N, C, 1, 1, 1});
}
"""

fused_bn_pool_cpp_source = (
    "torch::Tensor fused_bn_pool_cuda(torch::Tensor input, float scale_factor, "
    "torch::Tensor running_mean, torch::Tensor running_var, "
    "torch::Tensor weight, torch::Tensor bias, "
    "float eps, float momentum, bool training);"
)

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_bn_pool_cpp_source,
    cuda_sources=fused_bn_pool_source,
    functions=["fused_bn_pool_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        
        if self.training:
            self.num_batches_tracked += 1
            
        x = self.fused_ops.fused_bn_pool_cuda(
            x, self.scale_factor, 
            self.running_mean, self.running_var, 
            self.weight, self.bias, 
            self.eps, self.momentum, self.training
        )
        return x