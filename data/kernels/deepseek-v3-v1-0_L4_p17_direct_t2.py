import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Conv2D + InstanceNorm2d + Division CUDA kernel
fused_conv_norm_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void instance_norm_div_kernel(float* data, const float* mean, const float* var, 
                                        float divide_by, int batch_size, int channels, 
                                        int height, int width, float epsilon) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;
    
    if (n < batch_size && c < channels && h < height && w < width) {
        int spatial_idx = h * width + w;
        int feature_idx = n * channels + c;
        int data_idx = n * channels * height * width + c * height * width + spatial_idx;
        
        float normalized = (data[data_idx] - mean[feature_idx]) / sqrtf(var[feature_idx] + epsilon);
        data[data_idx] = normalized / divide_by;
    }
}

torch::Tensor fused_conv_norm_div_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divide_by,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    // Perform convolution
    auto conv_output = torch::conv2d(input, weight, bias, stride, padding, dilation, groups);
    
    int batch_size = conv_output.size(0);
    int channels = conv_output.size(1);
    int height = conv_output.size(2);
    int width = conv_output.size(3);
    
    // Calculate mean and variance for instance normalization
    auto mean = conv_output.view({batch_size, channels, -1}).mean(-1);
    auto var = conv_output.view({batch_size, channels, -1}).var(-1, false);
    
    float epsilon = 1e-5f;
    
    // Configure kernel launch parameters
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (height + block_dim.x - 1) / block_dim.x,
        (width + block_dim.y - 1) / block_dim.y,
        batch_size * channels
    );
    
    // Apply instance normalization and division in a single kernel
    instance_norm_div_kernel<<<grid_dim, block_dim>>>(
        conv_output.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        divide_by,
        batch_size, channels, height, width, epsilon
    );
    
    cudaDeviceSynchronize();
    
    return conv_output;
}
"""

fused_conv_norm_div_cpp_source = """
torch::Tensor fused_conv_norm_div_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divide_by,
    int stride,
    int padding,
    int dilation,
    int groups);
"""

# Compile the inline CUDA code
fused_conv_norm_div = load_inline(
    name="fused_conv_norm_div",
    cpp_sources=fused_conv_norm_div_cpp_source,
    cuda_sources=fused_conv_norm_div_source,
    functions=["fused_conv_norm_div_forward"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.divide_by = divide_by
        
        # Store convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2
        self.dilation = 1
        self.groups = 1
        
        # Initialize convolution weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias)
        
        self.fused_op = fused_conv_norm_div

    def forward(self, x):
        return self.fused_op.fused_conv_norm_div_forward(
            x, self.weight, self.bias, self.divide_by, 
            self.stride, self.padding, self.dilation, self.groups
        )