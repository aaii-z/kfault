import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias addition and clamping
fused_post_process_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_process_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ bias_conv, 
    const float* __restrict__ bias_explicit, 
    float* __restrict__ out, 
    int N, int C, int H, int W, 
    float inv_scale) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    
    if (idx < total_elements) {
        // Calculate channel index
        // Tensor is NCHW, so channel stride is H*W
        int c_idx = (idx / (H * W)) % C;
        
        float val = x[idx];
        val += bias_conv[c_idx];
        val += bias_explicit[c_idx];
        
        // Fused logic: clamp(val, 0, inv_scale)
        // Original logic: clamp(val, 0, 1) * scale -> clamp(res, 0, 1) / scale
        // Simplifies to clamp(val, 0, 1/scale)
        val = fmaxf(0.0f, val);
        val = fminf(inv_scale, val);
        
        out[idx] = val;
    }
}

torch::Tensor fused_post_process_cuda(
    torch::Tensor x, 
    torch::Tensor bias_conv, 
    torch::Tensor bias_explicit, 
    float inv_scale) {
    
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    
    auto out = torch::empty_like(x);
    
    int total_elements = x.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_post_process_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias_conv.data_ptr<float>(),
        bias_explicit.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        inv_scale
    );
    
    return out;
}
"""

fused_post_process_cpp_source = (
    "torch::Tensor fused_post_process_cuda(torch::Tensor x, torch::Tensor bias_conv, torch::Tensor bias_explicit, float inv_scale);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_post_process_cpp_source,
    cuda_sources=fused_post_process_source,
    functions=["fused_post_process_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution followed by a fused
    bias addition and clamping operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose2d, which is highly optimized in cuDNN
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        
        # The explicit bias parameter from the original architecture
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        # Perform transposed convolution
        x = self.conv_transpose(x)
        
        # Apply fused post-processing:
        # Adds both the convolution's bias and the explicit bias,
        # then applies the optimized clamp logic.
        # Logic: clamp(x + bias_conv + bias_explicit, 0, 1/scaling_factor)
        x = self.fused_ops.fused_post_process_cuda(x, self.conv_transpose.bias, self.bias, 1.0 / self.scaling_factor)
        
        return x