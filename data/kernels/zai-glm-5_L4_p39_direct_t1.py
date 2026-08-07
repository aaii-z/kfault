import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Scale + BatchNorm
fused_gemm_scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: BatchNorm Inference Forward
// Computes: y = (x * scale - mean) * rsqrt(var + eps) * weight + bias
// Here x is the output of (GEMM * Scale)
// We fuse the scaling into the GEMM output processing
// Grid: (out_features / block_size)
// Block: (block_size)

__global__ void fused_gemm_scale_bn_inference_kernel(
    const float* __restrict__ x, 
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ scale, // The learned scale parameter from the model
    float eps,
    int batch_size,
    int out_features) {
    
    int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (feat_idx < out_features) {
        // Load BatchNorm parameters and Model Scale for this feature
        float w = weight[feat_idx];
        float b = bias[feat_idx];
        float m = mean[feat_idx];
        float v = var[feat_idx];
        float s = scale[feat_idx]; // Model scale
        
        // Pre-compute the combined scale factor
        // Original: y = (x * s - m) / sqrt(v + eps) * w + b
        // Fused: y = x * (s * w / sqrt(v + eps)) + (b - m * w / sqrt(v + eps))
        // Let inv_std = 1.0 / sqrt(v + eps)
        // fused_scale = s * w * inv_std
        // fused_bias = b - m * w * inv_std
        
        float inv_std = rsqrtf(v + eps);
        float fused_scale = s * w * inv_std;
        float fused_bias = b - m * w * inv_std;
        
        // Process the column
        for (int i = 0; i < batch_size; ++i) {
            int idx = i * out_features + feat_idx;
            out[idx] = x[idx] * fused_scale + fused_bias;
        }
    }
}

// C++ wrapper function
torch::Tensor fused_gemm_scale_bn_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, // GEMM bias
    torch::Tensor scale, // Model scale
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    float eps) {
    
    // 1. Perform GEMM
    // input: (batch_size, in_features)
    // weight: (out_features, in_features)
    // output: (batch_size, out_features)
    auto gemm_output = torch::matmul(input, weight.t());
    
    // Add GEMM bias
    gemm_output = gemm_output + bias;
    
    // Allocate output tensor
    auto output = torch::empty_like(gemm_output);
    
    int batch_size = input.size(0);
    int out_features = gemm_output.size(1);
    
    // 2. Launch fused kernel
    const int block_size = 256;
    const int num_blocks = (out_features + block_size - 1) / block_size;
    
    fused_gemm_scale_bn_inference_kernel<<<num_blocks, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
        gemm_output.data_ptr<float>(),
        output.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        scale.data_ptr<float>(),
        eps,
        batch_size,
        out_features
    );
    
    return output;
}
"""

fused_gemm_scale_bn_cpp_source = (
    "torch::Tensor fused_gemm_scale_bn_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor scale, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var, float eps);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_gemm_scale_bn",
    cpp_sources=fused_gemm_scale_bn_cpp_source,
    cuda_sources=fused_gemm_scale_bn_source,
    functions=["fused_gemm_scale_bn_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for fused GEMM + Scale + BatchNorm.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # We keep the parameters but remove the original modules to avoid overhead
        # We need to manage the weights manually for the custom kernel
        
        # GEMM parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Scale parameter
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('bn_mean', torch.zeros(out_features))
        self.register_buffer('bn_var', torch.ones(out_features))
        
        self.eps = eps
        self.fused_ops = fused_ops

    def forward(self, x):
        # Call the custom fused kernel
        # Note: In a real training scenario, we would need to handle the update of running mean/var
        # and the backward pass. Since the prompt implies inference optimization (static BN),
        # we use the stored mean and variance directly.
        return self.fused_ops.fused_gemm_scale_bn_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.scale, 
            self.bn_weight, 
            self.bn_bias, 
            self.bn_mean, 
            self.bn_var, 
            self.eps
        )