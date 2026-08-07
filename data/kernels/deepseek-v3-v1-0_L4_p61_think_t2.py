import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

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
    int stride,
    int padding,
    int output_padding,
    int output_d, int output_h, int output_w
) {
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;

    int n = idx / (out_channels * output_d * output_h * output_w);
    int rem = idx % (out_channels * output_d * output_h * output_w);
    int c = rem / (output_d * output_h * output_w);
    rem = rem % (output_d * output_h * output_w);
    int d = rem / (output_h * output_w);
    rem = rem % (output_h * output_w);
    int h = rem / output_w;
    int w = rem % output_w;

    float value = 0.0f;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int input_d_idx = d - kd + padding;
                    int input_h_idx = h - kh + padding;
                    int input_w_idx = w - kw + padding;

                    if (input_d_idx >= 0 && input_d_idx < input_d &&
                        input_h_idx >= 0 && input_h_idx < input_h &&
                        input_w_idx >= 0 && input_w_idx < input_w) {
                        
                        int input_idx = n * in_channels * input_d * input_h * input_w +
                                      ic * input_d * input_h * input_w +
                                      input_d_idx * input_h * input_w +
                                      input_h_idx * input_w +
                                      input_w_idx;
                        
                        int weight_idx = ic * out_channels * kernel_size * kernel_size * kernel_size +
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

    value = fmaxf(0.0f, value);
    
    output[idx] = value;
}

torch::Tensor fused_conv_transpose_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(1);
    
    int output_d = (input_d - 1) * stride + kernel_size - 2 * padding + output_padding;
    int output_h = (input_h - 1) * stride + kernel_size - 2 * padding + output_padding;
    int output_w = (input_w - 1) * stride + kernel_size - 2 * padding + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, input.options());

    int total_elements = output.numel();
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    fused_conv_transpose_relu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        kernel_size,
        stride,
        padding,
        output_padding,
        output_d, output_h, output_w
    );

    return output;
}
"""

fused_conv_transpose_relu_cpp_source = """
torch::Tensor fused_conv_transpose_relu_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int output_padding);
"""

fused_conv_transpose_relu = load_inline(
    name="fused_conv_transpose_relu",
    cpp_sources=fused_conv_transpose_relu_cpp_source,
    cuda_sources=fused_conv_transpose_relu_source,
    functions=["fused_conv_transpose_relu_cuda"],
    verbose=True
)

class FusedConvTransposeReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=False):
        super(FusedConvTransposeReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
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
        x = fused_conv_transpose_relu.fused_conv_transpose_relu_cuda(
            x, self.weight, self.kernel_size, self.stride, self.padding, self.output_padding
        )
        if self.bias is not None:
            x = x + self.bias.view(1, self.out_channels, 1, 1, 1)
        return x

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.fused_conv_transpose_relu = FusedConvTransposeReLU(
            in_channels, out_channels, kernel_size, 
            stride=1, padding=0, output_padding=0, bias=bias
        )
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        x = self.fused_conv_transpose_relu(x)
        x = self.group_norm(x)
        return x