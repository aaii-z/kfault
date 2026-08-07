import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_op_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Compute Mean and Variance of the input tensor per channel
// Input shape: (N, C, D, H, W)
// Output: mean (N, C), var (N, C)
__global__ void compute_stats_kernel(const float* input, float* mean, float* var, int N, int C, int spatial_size) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    int base_idx = (n * C + c) * spatial_size;
    
    // Each thread processes a subset of the spatial dimensions
    for (int i = tid; i < spatial_size; i += block_size) {
        float val = input[base_idx + i];
        sum += val;
        sum_sq += val * val;
    }
    
    // Shared memory for reduction
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    
    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();
    
    // Parallel reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the result
    if (tid == 0) {
        float m = s_sum[0] / spatial_size;
        mean[n * C + c] = m;
        // Var = E[X^2] - E[X]^2
        float v = s_sum_sq[0] / spatial_size - m * m;
        // Clamp variance to avoid numerical issues
        if (v < 0.0f) v = 0.0f;
        var[n * C + c] = v;
    }
}

// Kernel 2: Fused Mul -> InstanceNorm -> Clamp -> Mul -> Max
// Input: input (N, C, D, H, W), multiplier (C), mean (N, C), var (N, C)
// Output: out (N, D, H, W)
__global__ void fused_op_kernel(const float* input, const float* multiplier, const float* mean, const float* var, 
                                float* out, int N, int C, int spatial_size, 
                                float clamp_min, float clamp_max, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = N * spatial_size;
    
    if (idx < total_spatial) {
        int n = idx / spatial_size;
        int spatial_idx = idx % spatial_size;
        
        float max_val = -INFINITY;
        
        // Iterate over channels
        for (int c = 0; c < C; ++c) {
            int input_idx = (n * C + c) * spatial_size + spatial_idx;
            
            float x = input[input_idx];
            float m = multiplier[c];
            
            // 1. Multiplication
            float val = x * m;
            
            // 2. Instance Normalization
            // We computed stats of 'input', but we need stats of 'input * m'
            // mean(val) = m * mean(x)
            // var(val) = m^2 * var(x)
            float mu_x = mean[n * C + c];
            float var_x = var[n * C + c];
            
            float mu_val = m * mu_x;
            float var_val = m * m * var_x;
            
            float norm = (val - mu_val) / sqrtf(var_val + eps);
            
            // 3. Clamp
            float clamped = fmaxf(clamp_min, fminf(clamp_max, norm));
            
            // 4. Multiplication
            float final_val = clamped * m;
            
            // 5. Max reduction
            if (final_val > max_val) {
                max_val = final_val;
            }
        }
        out[idx] = max_val;
    }
}

// C++ wrapper function
torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor multiplier, float clamp_min, float clamp_max) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int spatial_size = D * H * W;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto mean = torch::empty({N, C}, options);
    auto var = torch::empty({N, C}, options);
    
    // Launch stats kernel
    // Grid: (N, C), Block: 256
    dim3 grid_stats(N, C);
    dim3 block_stats(256);
    
    compute_stats_kernel<<<grid_stats, block_stats>>>(
        input.data_ptr<float>(), 
        mean.data_ptr<float>(), 
        var.data_ptr<float>(), 
        N, C, spatial_size
    );
    
    // Launch fused kernel
    // Output shape: (N, D, H, W)
    auto out = torch::empty({N, D, H, W}, options);
    int total_spatial = N * spatial_size;
    int threads = 256;
    int blocks = (total_spatial + threads - 1) / threads;
    
    float eps = 1e-5; // Default epsilon for InstanceNorm
    
    fused_op_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, spatial_size,
        clamp_min, clamp_max, eps
    );
    
    return out;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor multiplier, float clamp_min, float clamp_max);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel fusing multiplication, instance normalization, 
    clamping, multiplication, and max reduction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        # The custom kernel handles:
        # 1. x * self.multiplier
        # 2. InstanceNorm3d
        # 3. Clamp
        # 4. x * self.multiplier
        # 5. Max(dim=1)
        x = self.fused_op.fused_op_cuda(x, self.multiplier.view(-1), self.clamp_min, self.clamp_max)
        return x