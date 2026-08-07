import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom kernel for fused softmax normalization and scale multiplication
softmax_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_scale_kernel(
    const float* exp_input,
    const float* scale,
    const float* sum_val,
    float* output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * spatial_size) return;
    
    int c = (idx / spatial_size) % channels;
    int b = idx / (channels * spatial_size);
    
    float factor = scale[c] / sum_val[b * channels + c];
    output[idx] = exp_input[idx] * factor;
}

torch::Tensor softmax_scale_cuda(
    torch::Tensor exp_input,
    torch::Tensor scale,
    torch::Tensor sum_val
) {
    auto b = exp_input.size(0);
    auto c = exp_input.size(1);
    auto n = exp_input.size(2);
    
    auto output = torch::empty_like(exp_input);
    
    // Reshape scale to (c) and sum_val to (b*c)
    auto scale_flat = scale.view({c});
    auto sum_val_flat = sum_val.view({b * c});
    
    const int block_size = 256;
    const int num_blocks = (b * c * n + block_size - 1) / block_size;
    
    softmax_scale_kernel<<<num_blocks, block_size>>>(
        exp_input.data_ptr<float>(),
        scale_flat.data_ptr<float>(),
        sum_val_flat.data_ptr<float>(),
        output.data_ptr<float>(),
        b, c, n
    );
    
    return output;
}
"""

softmax_scale_cpp = "torch::Tensor softmax_scale_cuda(torch::Tensor exp_input, torch::Tensor scale, torch::Tensor sum_val);"

# Compile the custom CUDA kernel
softmax_scale = load_inline(
    name="softmax_scale",
    cpp_sources=softmax_scale_cpp,
    cuda_sources=softmax_scale_source,
    functions=["softmax_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3 --use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that replaces softmax and scale multiplication with a fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.softmax_scale = softmax_scale

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1)  # flatten spatial dimensions
        
        # Compute max for numerical stability
        max_val = torch.max(x, dim=2, keepdim=True)[0]
        x = x - max_val
        x = torch.exp(x)
        sum_val = torch.sum(x, dim=2, keepdim=True)
        
        # Fused normalization and scale multiplication
        x = self.softmax_scale.softmax_scale_cuda(x, self.scale, sum_val)
        
        x = x.view(b, c, d, h, w)
        return x

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 32, 64, 64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]