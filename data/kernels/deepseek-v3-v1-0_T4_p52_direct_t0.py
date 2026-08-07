import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define fused Conv + Activation + BatchNorm CUDA kernel
fused_conv_activation_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float softplus_activation(float x) {
    return log1pf(expf(x));
}

__device__ __forceinline__ float tanh_activation(float x) {
    return tanhf(x);
}

__global__ void fused_conv_activation_bn_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight_bn,
    const float* __restrict__ bias_bn,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    float eps,
    float momentum) {
    
    const int output_h = height - kernel_size + 1;
    const int output_w = width - kernel_size + 1;
    const int output_size = batch_size * out_channels * output_h * output_w;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;
    
    // Calculate output indices
    int n = idx / (out_channels * output_h * output_w);
    int remainder = idx % (out_channels * output_h * output_w);
    int c = remainder / (output_h * output_w);
    remainder %= (output_h * output_w);
    int h_out = remainder / output_w;
    int w_out = remainder % output_w;
    
    float sum = 0.0f;
    
    // Perform convolution
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            for (int ic = 0; ic < in_channels; ++ic) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;
                
                int input_idx = ((n * in_channels + ic) * height + h_in) * width + w_in;
                int weight_idx = ((c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // Add bias
    if (bias) {
        sum += bias[c];
    }
    
    // Apply activation: softplus -> tanh -> multiply
    float activated = sum * tanh_activation(softplus_activation(sum));
    
    // Apply batch normalization
    float normalized = (activated - running_mean[c]) / sqrtf(running_var[c] + eps);
    normalized = normalized * weight_bn[c] + bias_bn[c];
    
    output[idx] = normalized;
}

torch::Tensor fused_conv_activation_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    float eps,
    float momentum) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    CHECK_INPUT(running_mean);
    CHECK_INPUT(running_var);
    CHECK_INPUT(weight_bn);
    CHECK_INPUT(bias_bn);
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int output_h = height - kernel_size + 1;
    int output_w = width - kernel_size + 1;
    int output_size = batch_size * out_channels * output_h * output_w;
    
    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;
    
    fused_conv_activation_bn_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight_bn.data_ptr<float>(),
        bias_bn.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        eps,
        momentum);
    
    return output;
}
"""

fused_conv_activation_bn_cpp_source = """
torch::Tensor fused_conv_activation_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    float eps,
    float momentum);
"""

# Compile the inline CUDA code
fused_conv_activation_bn = load_inline(
    name="fused_conv_activation_bn",
    cpp_sources=fused_conv_activation_bn_cpp_source,
    cuda_sources=fused_conv_activation_bn_source,
    functions=["fused_conv_activation_bn_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.momentum = momentum
        
        # Initialize convolution parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # Initialize batch normalization parameters
        self.running_mean = nn.Parameter(torch.empty(out_channels), requires_grad=False)
        self.running_var = nn.Parameter(torch.empty(out_channels), requires_grad=False)
        self.weight_bn = nn.Parameter(torch.empty(out_channels))
        self.bias_bn = nn.Parameter(torch.empty(out_channels))
        
        # Initialize parameters
        self.reset_parameters()
        
        # Load custom CUDA operator
        self.fused_op = fused_conv_activation_bn

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.ones_(self.weight_bn)
        nn.init.zeros_(self.bias_bn)
        nn.init.zeros_(self.running_mean)
        nn.init.ones_(self.running_var)

    def forward(self, x):
        return self.fused_op.fused_conv_activation_bn_cuda(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.weight_bn, self.bias_bn, self.eps, self.momentum
        )