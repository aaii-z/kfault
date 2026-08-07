import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Group Normalization, Min, Clamp, and Dropout
fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel to compute sum and sum of squares for Group Normalization
// Using vectorized loads for better memory throughput
__global__ void compute_stats_kernel(const float* __restrict__ x, float* sum_out, float* sum_sq_out, int N, int C, int D, int H, int W, int G, int C_per_g) {
    int n = blockIdx.x; // Batch index
    int g = blockIdx.y; // Group index
    
    int group_size = C_per_g * D * H * W;
    int offset = n * C * D * H * W + g * C_per_g * D * H * W;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Use float4 for vectorized memory access
    // Assuming memory is aligned and size is divisible by 4 for simplicity in this specific kernel context
    // or handling remainder. Given dimensions (16, 64, 64), it is likely aligned.
    int vec_size = 4;
    int num_vec = group_size / vec_size;
    
    const float4* x_vec = reinterpret_cast<const float4*>(x + offset);
    
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 val = x_vec[i];
        sum += val.x + val.y + val.z + val.w;
        sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }
    
    // Handle remaining elements
    int remainder_start = num_vec * vec_size;
    for (int i = remainder_start + threadIdx.x; i < group_size; i += blockDim.x) {
        float val = x[offset + i];
        sum += val;
        sum_sq += val * val;
    }
    
    // Block reduction using shared memory
    __shared__ float shared_sum[256]; // Assuming block_size <= 256
    __shared__ float shared_sum_sq[256];
    
    shared_sum[threadIdx.x] = sum;
    shared_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
            shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        int out_idx = n * G + g;
        sum_out[out_idx] = shared_sum[0];
        sum_sq_out[out_idx] = shared_sum_sq[0];
    }
}

// Kernel for fused normalization, min, clamp, and dropout
__global__ void fused_norm_min_clamp_dropout_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ mean, 
    const float* __restrict__ var, 
    const float* __restrict__ gamma, 
    const float* __restrict__ beta, 
    const float* __restrict__ dropout_mask,
    float* __restrict__ out, 
    int N, int C, int D, int H, int W, 
    int G, int C_per_g, 
    float min_val, float max_val, float dropout_p, bool training) {
    
    int total_elements = N * C * D * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Calculate indices
        int w = idx % W;
        int h = (idx / W) % H;
        int d = (idx / (W * H)) % D;
        int c = (idx / (W * H * D)) % C;
        int n = idx / (W * H * D * C);
        
        int g = c / C_per_g;
        
        // Fetch stats
        float mu = mean[n * G + g];
        float sigma2 = var[n * G + g];
        
        // Normalize
        float val = x[idx];
        float normalized = (val - mu) / sqrtf(sigma2 + 1e-5f);
        
        // Scale and shift
        float y = normalized * gamma[c] + beta[c];
        
        // Min operation: torch.min(x, min_value)
        y = fminf(y, min_val);
        
        // Clamp operation: torch.clamp(x, min=min_value, max=max_value)
        y = fmaxf(fminf(y, max_val), min_val);
        
        // Dropout
        if (training) {
            if (dropout_mask[idx] < dropout_p) {
                y = 0.0f;
            } else {
                y = y / (1.0f - dropout_p);
            }
        }
        
        out[idx] = y;
    }
}

// C++ wrapper functions
std::tuple<torch::Tensor, torch::Tensor> compute_stats_cuda(torch::Tensor x, int G, int C_per_g) {
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto sum_out = torch::zeros({N, G}, options);
    auto sum_sq_out = torch::zeros({N, G}, options);
    
    const int block_size = 256;
    dim3 grid(N, G);
    
    compute_stats_kernel<<<grid, block_size>>>(
        x.data_ptr<float>(), 
        sum_out.data_ptr<float>(), 
        sum_sq_out.data_ptr<float>(), 
        N, C, D, H, W, G, C_per_g
    );
    
    return std::make_tuple(sum_out, sum_sq_out);
}

torch::Tensor fused_norm_min_clamp_dropout_cuda(
    torch::Tensor x, torch::Tensor mean, torch::Tensor var, 
    torch::Tensor gamma, torch::Tensor beta, torch::Tensor dropout_mask,
    int G, int C_per_g, float min_val, float max_val, float dropout_p, bool training) {
    
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    
    auto out = torch::empty_like(x);
    
    int total_elements = x.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_norm_min_clamp_dropout_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        mean.data_ptr<float>(), 
        var.data_ptr<float>(), 
        gamma.data_ptr<float>(), 
        beta.data_ptr<float>(), 
        dropout_mask.data_ptr<float>(),
        out.data_ptr<float>(), 
        N, C, D, H, W, G, C_per_g, 
        min_val, max_val, dropout_p, training
    );
    
    return out;
}
"""

fused_ops_cpp_source = """
std::tuple<torch::Tensor, torch::Tensor> compute_stats_cuda(torch::Tensor x, int G, int C_per_g);
torch::Tensor fused_norm_min_clamp_dropout_cuda(
    torch::Tensor x, torch::Tensor mean, torch::Tensor var, 
    torch::Tensor gamma, torch::Tensor beta, torch::Tensor dropout_mask,
    int G, int C_per_g, float min_val, float max_val, float dropout_p, bool training);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["compute_stats_cuda", "fused_norm_min_clamp_dropout_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, applies Group Normalization, minimum, clamp, and dropout.
    The post-conv operations are fused into a single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        
        # GroupNorm parameters
        self.groups = groups
        self.num_channels = out_channels
        self.register_buffer('weight', torch.ones(out_channels))
        self.register_buffer('bias', torch.zeros(out_channels))
        
        # Constants
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Conv3d (using PyTorch's optimized implementation)
        x = self.conv(x)
        
        # 2. Fused GroupNorm + Min + Clamp + Dropout
        N, C, D, H, W = x.shape
        G = self.groups
        C_per_g = C // G
        
        # Compute mean and variance
        # Note: This launches a custom kernel to compute sums
        sum_out, sum_sq_out = self.fused_ops.compute_stats_cuda(x, G, C_per_g)
        
        group_size = C_per_g * D * H * W
        mean = sum_out / group_size
        # Var = E[X^2] - E[X]^2
        var = (sum_sq_out / group_size) - (mean * mean)
        
        # Generate dropout mask
        dropout_mask = torch.rand_like(x)
        
        # Launch fused kernel
        x = self.fused_ops.fused_norm_min_clamp_dropout_cuda(
            x, mean, var, self.weight, self.bias, dropout_mask, 
            G, C_per_g, self.min_value, self.max_value, self.dropout_p, self.training
        )
        
        return x