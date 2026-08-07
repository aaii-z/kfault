import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Conv2D + InstanceNorm2d + Division CUDA kernel
fused_conv_norm_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cudnn.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void instance_norm_div_kernel(float* data, const float* mean, const float* var, 
                                        float gamma, float beta, float divide_by, 
                                        int batch_size, int channels, int height, int width, 
                                        float epsilon) {
    int c = blockIdx.x;
    int n = blockIdx.y;
    int idx = n * channels * height * width + c * height * width + threadIdx.x * width + threadIdx.y;
    
    if (c < channels && n < batch_size && threadIdx.x < height && threadIdx.y < width) {
        float normalized = gamma * (data[idx] - mean[n * channels + c]) / sqrtf(var[n * channels + c] + epsilon) + beta;
        data[idx] = normalized / divide_by;
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
    float gamma = 1.0f;
    float beta = 0.0f;
    
    // Configure kernel launch parameters
    dim3 blocks(channels, batch_size);
    dim3 threads(height, width);
    
    // Apply instance normalization and division in a single kernel
    instance_norm_div_kernel<<<blocks, threads>>>(
        conv_output.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        gamma, beta, divide_by,
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