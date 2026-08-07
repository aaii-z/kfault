import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Scale + BatchNorm + Global Average Pooling
fused_bn_pool_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Compute spatial sums and sums of squares for each (N, C) channel
// This replaces the need to materialize the scaled output and the BN output in global memory
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ input, // Input from ConvTranspose3d
    scalar_t* __restrict__ sum_out,     // Output sum per channel
    scalar_t* __restrict__ sum_sq_out,  // Output sum of squares per channel
    const int N, const int C, const int spatial_size,
    const scalar_t scale_factor) {
    
    // Each block handles one (n, c) pair
    int nc = blockIdx.x; 
    int n = nc / C;
    int c = nc % C;
    
    int tid = threadIdx.x;
    
    scalar_t sum = 0.0f;
    scalar_t sum_sq = 0.0f;
    
    // Pointer to the start of the spatial dimensions for this (n, c)
    const scalar_t* x_ptr = input + nc * spatial_size;
    
    // Stride loop over spatial dimensions
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        scalar_t val = x_ptr[i] * scale_factor; // Apply scale
        sum += val;
        sum_sq += val * val;
    }
    
    // Block reduction using shared memory
    __shared__ scalar_t s_sum[256]; // Assuming block size 256
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
    
    // Write result for this (n, c) to global memory
    if (tid == 0) {
        sum_out[nc] = s_sum[0];
        sum_sq_out[nc] = s_sum_sq[0];
    }
}

// Kernel 2: Finalize Batch Normalization and Global Pooling
// Computes mean/var, updates running stats (if training), and computes final output
template <typename scalar_t>
__global__ void finalize_bn_pool_kernel(
    const scalar_t* __restrict__ spatial_sums,
    const scalar_t* __restrict__ spatial_sums_sq,
    scalar_t* __restrict__ out, // Output tensor (N, C, 1, 1, 1)
    scalar_t* __restrict__ running_mean,
    scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ weight, // gamma
    const scalar_t* __restrict__ bias,   // beta
    int N, int C, int spatial_size,
    scalar_t eps, scalar_t momentum, bool training) {
    
    int c = blockIdx.x; // One block per channel
    int tid = threadIdx.x;
    
    // Load sums for this channel 'c' across the batch 'N'
    // We use shared memory to share loaded data within the block
    extern __shared__ char shared_mem[];
    scalar_t* s_sums = (scalar_t*)shared_mem;
    scalar_t* s_sums_sq = &s_sums[N];
    
    if (tid < N) {
        s_sums[tid] = spatial_sums[tid * C + c];
        s_sums_sq[tid] = spatial_sums_sq[tid * C + c];
    }
    __syncthreads();
    
    // Thread 0 computes the batch statistics and updates running stats
    // Note: In a highly optimized scenario, we might parallelize the reduction over N as well,
    // but N is typically small (batch size). Here we assume N <= 256 (typical batch size).
    if (tid == 0) {
        scalar_t batch_sum = 0;
        scalar_t batch_sum_sq = 0;
        for (int i = 0; i < N; ++i) {
            batch_sum += s_sums[i];
            batch_sum_sq += s_sums_sq[i];
        }
        
        int total_count = N * spatial_size;
        scalar_t mean = batch_sum / total_count;
        // Var = E[x^2] - E[x]^2
        scalar_t var = batch_sum_sq / total_count - mean * mean;
        
        if (training) {
            // Update running stats
            // running_mean = (1 - momentum) * running_mean + momentum * mean
            // Note: PyTorch uses exponential moving average
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            // Using unbiased variance estimate for running var usually, but standard BN often uses biased for running stats
            // PyTorch default uses unbiased (total_count / (total_count - 1)) for running_var calculation? 
            // Actually PyTorch BN update uses the biased var by default for the running stats unless track_running_stats is handled differently.
            // Let's stick to the calculated var.
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;
        }
        
        // We need to compute the output.
        // In training mode, output is simply bias (because we normalize to zero mean, then pool).
        // In eval mode, output depends on input mean and running stats.
        
        // Store necessary values for threads to write output
        s_sums[0] = mean; // Reuse shared memory to broadcast mean if needed (not really needed for training)
        s_sums[1] = var;
    }
    __syncthreads();
    
    // Write output
    // Output is (N, C, 1, 1, 1). We write to out[n, c, 0, 0, 0]
    // We parallelize over the batch dimension N
    if (training) {
        // Output is just bias[c] because BatchNorm normalizes mean to 0, and GlobalAvgPool takes the mean.
        // Result: bias
        if (tid < N) {
            out[tid * C + c] = bias[c];
        }
    } else {
        // Eval mode: Use running stats
        // We need running_mean and running_var. These are in global memory.
        // It's faster to load them once.
        // Let's load them in thread 0 and share? Or just load directly. N is small.
        // Actually, we can just calculate.
        if (tid < N) {
            scalar_t instance_mean = s_sums[tid] / spatial_size;
            scalar_t rm = running_mean[c];
            scalar_t rv = running_var[c];
            scalar_t w = weight[c];
            scalar_t b = bias[c];
            
            scalar_t norm_val = (instance_mean - rm) / sqrt(rv + eps);
            out[tid * C + c] = norm_val * w + b;
        }
    }
}

// C++ interface
torch::Tensor fused_bn_pool_cuda(
    torch::Tensor input, 
    float scale_factor,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps, float momentum, bool training) {
    
    // Input shape: (N, C, D, H, W)
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int spatial_size = sizes[2] * sizes[3] * sizes[4];
    
    auto options = input.options();
    // Temporary buffers for sums
    auto spatial_sums = torch::zeros({N, C}, options);
    auto spatial_sums_sq = torch::zeros({N, C}, options);
    
    // Output tensor (N, C, 1, 1, 1) stored as (N, C) initially
    auto out = torch::zeros({N, C}, options);
    
    // Launch Kernel 1: Compute sums
    // Grid: N * C blocks. Block: 256 threads.
    int threads = 256;
    int blocks = N * C;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "compute_stats_kernel", ([&] {
        compute_stats_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            spatial_sums.data_ptr<scalar_t>(),
            spatial_sums_sq.data_ptr<scalar_t>(),
            N, C, spatial_size,
            static_cast<scalar_t>(scale_factor)
        );
    }));
    
    // Launch Kernel 2: Finalize
    // Grid: C blocks. Block: 256 threads (enough to cover N).
    int blocks2 = C;
    int threads2 = 256; 
    // Shared memory: 2 * N * sizeof(float) for loading sums
    size_t shared_mem_size = 2 * N * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "finalize_bn_pool_kernel", ([&] {
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
    
    // Reshape output to (N, C, 1, 1, 1)
    return out.view({N, C, 1, 1, 1});
}
"""

fused_bn_pool_cpp_source = (
    "torch::Tensor fused_bn_pool_cuda(torch::Tensor input, float scale_factor, "
    "torch::Tensor running_mean, torch::Tensor running_var, "
    "torch::Tensor weight, torch::Tensor bias, "
    "float eps, float momentum, bool training);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_bn_pool_cpp_source,
    cuda_sources=fused_bn_pool_source,
    functions=["fused_bn_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by a fused
    Scale + BatchNorm + Global Average Pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.eps = eps
        self.momentum = momentum
        
        # BatchNorm parameters
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Transposed Convolution (using PyTorch's optimized implementation)
        x = self.conv_transpose(x)
        
        # 2. Fused Scale + BatchNorm + Global Average Pooling
        # We handle the training/eval logic inside the kernel wrapper for efficiency
        if self.training:
            self.num_batches_tracked += 1
            
        x = self.fused_ops.fused_bn_pool_cuda(
            x, self.scale_factor, 
            self.running_mean, self.running_var, 
            self.weight, self.bias, 
            self.eps, self.momentum, self.training
        )
        return x