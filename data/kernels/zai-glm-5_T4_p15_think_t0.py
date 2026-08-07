import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Compute statistics
// Computes sum per (N, C) for spatial mean, and sum/sum_sq per (C) for batch mean/var
__global__ void stats_kernel(
    const float* __restrict__ input,
    float* __restrict__ sum_spatial, // Shape (N, C)
    float* __restrict__ sum_batch,  // Shape (C)
    float* __restrict__ sum_sq_batch, // Shape (C)
    int N, int C, int spatial_size) {

    int nc = blockIdx.x; // Grid size is N*C
    int n = nc / C;
    int c = nc % C;

    const float* x_ptr = input + nc * spatial_size;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Block-stride loop
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = x_ptr[i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Block reduction
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    
    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum_spatial[nc] = s_sum[0];
        // Atomic add to global batch stats
        atomicAdd(&sum_batch[c], s_sum[0]);
        atomicAdd(&sum_sq_batch[c], s_sum_sq[0]);
    }
}

// Kernel 2: Apply fused BN and Centering
// out = weight * (x - mean_spatial) / std
__global__ void apply_kernel(
    const float* __restrict__ input,
    const float* __restrict__ sum_spatial,
    const float* __restrict__ weight,
    const float* __restrict__ std, // Precomputed 1/std or std
    float* __restrict__ output,
    int N, int C, int spatial_size, float eps) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * C * spatial_size;
    if (idx >= total_size) return;

    int nc = idx / spatial_size;
    int c = nc % C;
    int s = idx % spatial_size;

    float mean_s = sum_spatial[nc] / spatial_size;
    float std_val = std[c];
    float w = weight[c];

    float val = input[idx];
    // Formula: weight * (x - mean_spatial) / std
    // Note: std_val passed in is sqrt(var + eps)
    output[idx] = w * (val - mean_s) / std_val;
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor sum_batch,
    torch::Tensor sum_sq_batch,
    bool training,
    float momentum,
    float eps) {

    int N = input.size(0);
    int C = input.size(1);
    int spatial_size = input.size(2) * input.size(3) * input.size(4);

    auto output = torch::empty_like(input);
    auto sum_spatial = torch::empty({N, C}, input.options()).contiguous();

    // Ensure buffers are contiguous and correct shape
    sum_batch.fill_(0);
    sum_sq_batch.fill_(0);

    int threads = 256;
    int blocks = N * C;

    // Kernel 1: Compute Stats
    stats_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        sum_spatial.data_ptr<float>(),
        sum_batch.data_ptr<float>(),
        sum_sq_batch.data_ptr<float>(),
        N, C, spatial_size
    );

    // CPU Post-processing for stats (small tensors, negligible overhead)
    // Compute mean_spatial is deferred to kernel 2 (passed as sum)
    // Compute batch stats and update running vars
    auto sum_batch_cpu = sum_batch.cpu();
    auto sum_sq_batch_cpu = sum_sq_batch.cpu();
    auto running_mean_cpu = running_mean.cpu();
    auto running_var_cpu = running_var.cpu();
    
    auto std_dev = torch::empty({C}, input.options());

    for (int c = 0; c < C; ++c) {
        float mean = sum_batch_cpu[c].item<float>() / (N * spatial_size);
        float var = sum_sq_batch_cpu[c].item<float>() / (N * spatial_size) - mean * mean;
        
        if (training) {
            running_mean_cpu[c] = (1 - momentum) * running_mean_cpu[c].item<float>() + momentum * mean;
            running_var_cpu[c] = (1 - momentum) * running_var_cpu[c].item<float>() + momentum * var;
            std_dev[c] = sqrtf(var + eps);
        } else {
            std_dev[c] = sqrtf(running_var_cpu[c].item<float>() + eps);
        }
    }
    
    // Copy back updated running stats
    running_mean.copy_(running_mean_cpu);
    running_var.copy_(running_var_cpu);

    // Kernel 2: Apply
    int total_elements = N * C * spatial_size;
    int apply_blocks = (total_elements + threads - 1) / threads;

    apply_kernel<<<apply_blocks, threads>>>(
        input.data_ptr<float>(),
        sum_spatial.data_ptr<float>(),
        weight.data_ptr<float>(),
        std_dev.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, spatial_size, eps
    );

    return output;
}
"""

cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor sum_batch,
    torch::Tensor sum_sq_batch,
    bool training,
    float momentum,
    float eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        
        # BatchNorm parameters
        self.weight = nn.Parameter(torch.ones(out_channels))
        # Bias is mathematically cancelled out by the centering operation, so we don't need it.
        
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # Buffers for intermediate stats to avoid allocation every forward pass
        self.register_buffer('sum_batch', torch.zeros(out_channels))
        self.register_buffer('sum_sq_batch', torch.zeros(out_channels))
        
        self.momentum = 0.1
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Use custom fused kernel
        x = fused_ops.fused_op_cuda(
            x, 
            self.weight, 
            self.running_mean, 
            self.running_var,
            self.sum_batch,
            self.sum_sq_batch,
            self.training,
            self.momentum,
            self.eps
        )
        return x