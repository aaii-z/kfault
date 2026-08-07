import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, BiasAdd, Hardtanh, and Mish
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel for BiasAdd, Hardtanh, and Mish
__global__ void fused_bias_hardtanh_mish_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ out, int size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int feature_idx = idx % out_features;
        float val = x[idx] + bias[feature_idx];
        
        // Hardtanh
        val = fmaxf(-1.0f, fminf(1.0f, val));
        
        // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        // softplus(x) = ln(1 + exp(x))
        float sp = log1pf(expf(val));
        out[idx] = val * tanhf(sp);
    }
}

// Kernel to compute mean and variance for GroupNorm
__global__ void group_norm_stats_kernel(const float* __restrict__ x, float* __restrict__ mean, float* __restrict__ var, int batch_size, int channels, int group_size) {
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    
    if (batch_idx < batch_size) {
        int start_channel = group_idx * group_size;
        int end_channel = start_channel + group_size;
        
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int c = start_channel; c < end_channel; ++c) {
            float val = x[batch_idx * channels + c];
            sum += val;
            sum_sq += val * val;
        }
        
        float mean_val = sum / group_size;
        float var_val = sum_sq / group_size - mean_val * mean_val;
        
        mean[batch_idx * gridDim.y + group_idx] = mean_val;
        var[batch_idx * gridDim.y + group_idx] = var_val;
    }
}

// Kernel to apply GroupNorm
__global__ void group_norm_kernel(const float* __restrict__ x, const float* __restrict__ mean, const float* __restrict__ var, 
                                  const float* __restrict__ weight, const float* __restrict__ bias, float* __restrict__ out, 
                                  int batch_size, int channels, int group_size, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels;
    
    if (idx < total_size) {
        int batch_idx = idx / channels;
        int channel_idx = idx % channels;
        int group_idx = channel_idx / group_size;
        
        float mean_val = mean[batch_idx * (channels / group_size) + group_idx];
        float var_val = var[batch_idx * (channels / group_size) + group_idx];
        
        float x_norm = (x[idx] - mean_val) / sqrtf(var_val + eps);
        out[idx] = x_norm * weight[channel_idx] + bias[channel_idx];
    }
}

torch::Tensor fused_gemm_bias_hardtanh_mish_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias_vec) {
    // x: (batch_size, in_features)
    // weight: (out_features, in_features)
    // bias_vec: (out_features,)
    
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(0);
    
    // Perform GEMM using cuBLAS (handled by PyTorch's matmul which is optimized)
    auto x_gemm = torch::matmul(x, weight.t());
    
    auto out = torch::empty_like(x_gemm);
    int size = x_gemm.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_bias_hardtanh_mish_kernel<<<num_blocks, block_size>>>(
        x_gemm.data_ptr<float>(), bias_vec.data_ptr<float>(), out.data_ptr<float>(), size, out_features
    );
    
    return out;
}

torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    int group_size = channels / num_groups;
    
    auto mean = torch::empty({batch_size, num_groups}, x.options());
    auto var = torch::empty({batch_size, num_groups}, x.options());
    auto out = torch::empty_like(x);
    
    dim3 blocks(batch_size, num_groups);
    
    group_norm_stats_kernel<<<blocks, 1>>>(
        x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), batch_size, channels, group_size
    );
    
    int total_size = batch_size * channels;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    group_norm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
        batch_size, channels, group_size, eps
    );
    
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_gemm_bias_hardtanh_mish_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias_vec);\n"
    "torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_gemm_bias_hardtanh_mish_cuda", "group_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for fused operations.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        # Initialize weights similar to nn.Linear
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # GroupNorm parameters
        self.num_groups = num_groups;
        self.eps = 1e-5;
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Fused GEMM + Bias + Hardtanh + Mish
        x = self.fused_ops.fused_gemm_bias_hardtanh_mish_cuda(x, self.weight, self.bias)
        
        # GroupNorm
        x = self.fused_ops.group_norm_cuda(x, self.gamma, self.beta, self.num_groups, self.eps)
        
        return x