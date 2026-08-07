import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_conv_transpose_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_transpose_ops_kernel(
    const float* input, const float* weight, const float* bias,
    float scale1, float scale2, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride, int padding) {
    
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_depth;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_height;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_width;
    
    const int pooled_depth = output_depth / 2;
    const int pooled_height = output_height / 2;
    const int pooled_width = output_width / 2;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    
    if (idx >= total_outputs) return;
    
    int n = idx / (out_channels * pooled_depth * pooled_height * pooled_width);
    int rem = idx % (out_channels * pooled_depth * pooled_height * pooled_width);
    int c_out = rem / (pooled_depth * pooled_height * pooled_width);
    rem = rem % (pooled_depth * pooled_height * pooled_width);
    int d_pool = rem / (pooled_height * pooled_width);
    rem = rem % (pooled_height * pooled_width);
    int h_pool = rem / pooled_width;
    int w_pool = rem % pooled_width;
    
    int d_out = d_pool * 2;
    int h_out = h_pool * 2;
    int w_out = w_pool * 2;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kd = 0; kd < kernel_depth; kd++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int d_in = d_out - kd * stride + padding;
                    int h_in = h_out - kh * stride + padding;
                    int w_in = w_out - kw * stride + padding;
                    
                    if (d_in >= 0 && d_in < input_depth && 
                        h_in >= 0 && h_in < input_height && 
                        w_in >= 0 && w_in < input_width) {
                        
                        int input_idx = ((n * in_channels + c_in) * input_depth + d_in) * input_height * input_width + 
                                      h_in * input_width + w_in;
                        
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_depth + kd) * kernel_height * kernel_width + 
                                       kh * kernel_width + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    float pooled_sum = 0.0f;
    int count = 0;
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int d = d_out + i;
                int h = h_out + j;
                int w = w_out + k;
                
                if (d < output_depth && h < output_height && w < output_width) {
                    float conv_output = sum * scale1;
                    pooled_sum += conv_output;
                    count++;
                }
            }
        }
    }
    
    if (count > 0) {
        output[idx] = (pooled_sum / count + bias[c_out]) * scale2;
    } else {
        output[idx] = (bias[c_out]) * scale2;
    }
}

torch::Tensor fused_conv_transpose_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scale1, float scale2, int stride, int padding) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_depth = weight.size(2);
    const int kernel_height = weight.size(3);
    const int kernel_width = weight.size(4);
    
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_depth;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_height;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_width;
    
    const int pooled_depth = output_depth / 2;
    const int pooled_height = output_height / 2;
    const int pooled_width = output_width / 2;
    
    auto output = torch::zeros({batch_size, out_channels, pooled_depth, pooled_height, pooled_width}, 
                              input.options());
    
    const int output_size = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;
    
    fused_conv_transpose_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        scale1, scale2, output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_depth, kernel_height, kernel_width,
        stride, padding);
    
    return output;
}
"""

fused_conv_transpose_ops_cpp_source = """
torch::Tensor fused_conv_transpose_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scale1, float scale2, int stride, int padding);
"""

fused_conv_transpose_ops = load_inline(
    name="fused_conv_transpose_ops",
    cpp_sources=fused_conv_transpose_ops_cpp_source,
    cuda_sources=fused_conv_transpose_ops_source,
    functions=["fused_conv_transpose_ops_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        self.fused_op = fused_conv_transpose_ops

    def forward(self, x):
        return self.fused_op.fused_conv_transpose_ops_cuda(
            x, self.weight, self.bias,
            self.scale1.item(), self.scale2.item(),
            self.stride, self.padding
        )