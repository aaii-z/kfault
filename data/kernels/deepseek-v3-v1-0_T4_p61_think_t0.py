import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose3d + ReLU
fused_conv_transpose_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_transpose_relu_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int kernel_size,
    int output_d, int output_h, int output_w
) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int d = threadIdx.x * blockDim.y + threadIdx.y;
    int h = blockIdx.z % output_h;
    int w = blockIdx.z / output_h;

    if (d >= output_d || h >= output_h || w >= output_w) return;

    float value = 0.0f;

    for (int i = 0; i < in_channels; i++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int input_d_idx = d - kd;
                    int input_h_idx = h - kh;
                    int input_w_idx = w - kw;

                    if (input_d_idx >= 0 && input_d_idx < input_d &&
                        input_h_idx >= 0 && input_h_idx < input_h &&
                        input_w_idx >= 0 && input_w_idx < input_w) {
                        int input_idx = n * in_channels * input_d * input_h * input_w +
                                      i * input_d * input_h * input_w +
                                      input_d_idx * input_h * input_w +
                                      input_h_idx * input_w +
                                      input_w_idx;
                        
                        int weight_idx = i * out_channels * kernel_size * kernel_size * kernel_size +
                                      c * kernel_size * kernel_size * kernel_size +
                                      kd * kernel_size * kernel_size +
                                      kh * kernel_size +
                                      kw;
                        
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Apply ReLU activation
    value = fmaxf(0.0f, value);
    
    int output_idx = n * out_channels * output_d * output_h * output_w +
                   c * output_d * output_h * output_w +
                   d * output_h * output_w +
                   h * output_w +
                   w;
    
    output[output_idx] = value;
}

torch::Tensor fused_conv_transpose_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int output_d = input_d + kernel_size - 1;
    int output_h = input_h + kernel_size - 1;
    int output_w = input_w + kernel_size - 1;

    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, input.options());

    dim3 blocks(batch_size, out_channels, output_h * output_w);
    dim3 threads(32, 32);
    
    fused_conv_transpose_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        kernel_size,
        output_d, output_h, output_w
    );

    return output;
}
"""

fused_conv_transpose_relu_cpp_source = """
torch::Tensor fused_conv_transpose_relu_cuda(torch::Tensor input, torch::Tensor weight);
"""

# Compile the inline CUDA code
fused_conv_transpose_relu = load_inline(
    name="fused_conv_transpose_relu",
    cpp_sources=fused_conv_transpose_relu_cpp_source,
    cuda_sources=fused_conv_transpose_relu_source,
    functions=["fused_conv_transpose_relu_cuda"],
    verbose=True
)

class FusedConvTransposeReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(FusedConvTransposeReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = fused_conv_transpose_relu.fused_conv_transpose_relu_cuda(x, self.weight)
        if self.bias is not None:
            x = x + self.bias.view(1, self.out_channels, 1, 1, 1)
        return x

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.fused_conv_transpose_relu = FusedConvTransposeReLU(in_channels, out_channels, kernel_size, bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        x = self.fused_conv_transpose_relu(x)
        x = self.group_norm(x)
        return x