import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for transposed convolution + bias add + HardSwish
fused_conv_transpose_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_transpose_bias_swish_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, const float* add_input,
    int batch_size, int in_channels, int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w) {
    
    const int output_size = batch_size * out_channels * output_d * output_h * output_w;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size) {
        // Decompose index
        int n = idx / (out_channels * output_d * output_h * output_w);
        int remainder = idx % (out_channels * output_d * output_h * output_w);
        int c = remainder / (output_d * output_h * output_w);
        remainder = remainder % (output_d * output_h * output_w);
        int d = remainder / (output_h * output_w);
        remainder = remainder % (output_h * output_w);
        int h = remainder / output_w;
        int w = remainder % output_w;
        
        float sum = 0.0f;
        
        // Perform transposed convolution
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int input_d_pos = (d + padding_d - kd) / stride_d;
                    int input_h_pos = (h + padding_h - kh) / stride_h;
                    int input_w_pos = (w + padding_w - kw) / stride_w;
                    
                    if ((d + padding_d - kd) % stride_d == 0 &&
                        (h + padding_h - kh) % stride_h == 0 &&
                        (w + padding_w - kw) % stride_w == 0 &&
                        input_d_pos >= 0 && input_d_pos < input_d &&
                        input_h_pos >= 0 && input_h_pos < input_h &&
                        input_w_pos >= 0 && input_w_pos < input_w) {
                        
                        for (int ic = 0; ic < in_channels; ic++) {
                            int input_idx = n * (in_channels * input_d * input_h * input_w) +
                                          ic * (input_d * input_h * input_w) +
                                          input_d_pos * (input_h * input_w) +
                                          input_h_pos * input_w +
                                          input_w_pos;
                            
                            int weight_idx = c * (in_channels * kernel_d * kernel_h * kernel_w) +
                                           ic * (kernel_d * kernel_h * kernel_w) +
                                           kd * (kernel_h * kernel_w) +
                                           kh * kernel_w +
                                           kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias (broadcasted to output shape)
        int bias_idx = c;
        sum += bias[bias_idx];
        
        // Add input tensor
        sum += add_input[idx];
        
        // Apply HardSwish: x * hardtanh(x/3 + 0.5, 0, 1) * 3
        float swish = sum * min(max(sum / 3.0f + 0.5f, 0.0f), 1.0f) * 3.0f;
        
        output[idx] = swish;
    }
}

torch::Tensor fused_conv_transpose_bias_swish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor add_input,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(add_input);
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_d = input.size(2);
    const int input_h = input.size(3);
    const int input_w = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    const int output_d = (input_d - 1) * stride_d + kernel_d - 2 * padding_d + output_padding_d;
    const int output_h = (input_h - 1) * stride_h + kernel_h - 2 * padding_h + output_padding_h;
    const int output_w = (input_w - 1) * stride_w + kernel_w - 2 * padding_w + output_padding_w;
    
    auto output = torch::empty({batch_size, out_channels, output_d, output_h, output_w}, input.options());
    
    const int output_size = batch_size * out_channels * output_d * output_h * output_w;
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;
    
    fused_conv_transpose_bias_swish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), add_input.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w);
    
    return output;
}
"""

fused_conv_transpose_swish_cpp_source = """
torch::Tensor fused_conv_transpose_bias_swish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor add_input,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w);
"""

# Compile the inline CUDA code
fused_conv_swish = load_inline(
    name="fused_conv_transpose_swish",
    cpp_sources=fused_conv_transpose_swish_cpp_source,
    cuda_sources=fused_conv_transpose_swish_source,
    functions=["fused_conv_transpose_bias_swish_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Initialize weight and bias parameters (note: weight shape is [out_channels, in_channels, k, k, k])
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store the custom CUDA operator
        self.fused_op = fused_conv_swish

    def forward(self, x, add_input):
        return self.fused_op.fused_conv_transpose_bias_swish_cuda(
            x, self.weight, self.bias, add_input,
            self.stride, self.stride, self.stride,
            self.padding, self.padding, self.padding,
            self.output_padding, self.output_padding, self.output_padding
        )