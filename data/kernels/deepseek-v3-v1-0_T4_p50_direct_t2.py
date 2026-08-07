import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3d + Scale + AvgPool + Bias + Scale
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
    int output_depth, int output_height, int output_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width) {
    
    const int pooled_depth = output_depth / 2;
    const int pooled_height = output_height / 2;
    const int pooled_width = output_width / 2;
    const int output_size = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= output_size) return;
    
    // Decompose global index for pooled output
    int n = idx / (out_channels * pooled_depth * pooled_height * pooled_width);
    int remainder = idx % (out_channels * pooled_depth * pooled_height * pooled_width);
    int c_out = remainder / (pooled_depth * pooled_height * pooled_width);
    remainder = remainder % (pooled_depth * pooled_height * pooled_width);
    int d_pool = remainder / (pooled_height * pooled_width);
    remainder = remainder % (pooled_height * pooled_width);
    int h_pool = remainder / pooled_width;
    int w_pool = remainder % pooled_width;
    
    // Convert pooled coordinates to original output coordinates
    int d_out = d_pool * 2;
    int h_out = h_pool * 2;
    int w_out = w_pool * 2;
    
    float sum = 0.0f;
    
    // Compute input coordinates for transposed convolution
    int d_in = d_out - padding_depth;
    int h_in = h_out - padding_height;
    int w_in = w_out - padding_width;
    
    // Iterate over kernel and input channels
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kd = 0; kd < kernel_depth; kd++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int d_pos = d_in + kd * stride_depth;
                    int h_pos = h_in + kh * stride_height;
                    int w_pos = w_in + kw * stride_width;
                    
                    if (d_pos >= 0 && d_pos < input_depth && 
                        h_pos >= 0 && h_pos < input_height && 
                        w_pos >= 0 && w_pos < input_width) {
                        
                        int input_idx = ((n * in_channels + c_in) * input_depth + d_pos) * input_height * input_width + 
                                      h_pos * input_width + w_pos;
                        
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_depth + kd) * kernel_height * kernel_width + 
                                       kh * kernel_width + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Apply scale1, add bias, and apply scale2
    output[idx] = (sum * scale1 + bias[c_out]) * scale2;
}

torch::Tensor fused_conv_transpose_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scale1, float scale2, 
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width) {
    
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
    
    // Calculate output dimensions for transposed convolution
    const int output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth + kernel_depth;
    const int output_height = (input_height - 1) * stride_height - 2 * padding_height + kernel_height;
    const int output_width = (input_width - 1) * stride_width - 2 * padding_width + kernel_width;
    
    // Output after average pooling (halves each dimension)
    const int pooled_depth = output_depth / 2;
    const int pooled_height = output_height / 2;
    const int pooled_width = output_width / 2;
    
    auto output = torch::zeros({batch_size, out_channels, pooled_depth, pooled_height, pooled_width}, 
                              torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int output_size = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;
    
    fused_conv_transpose_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        scale1, scale2, output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_depth, kernel_height, kernel_width,
        stride_depth, stride_height, stride_width,
        padding_depth, padding_height, padding_width);
    
    return output;
}
"""

fused_conv_transpose_ops_cpp_source = """
torch::Tensor fused_conv_transpose_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scale1, float scale2, 
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width);
"""

# Compile the inline CUDA code
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
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        self.fused_op = fused_conv_transpose_ops

    def forward(self, x):
        return self.fused_op.fused_conv_transpose_ops_cuda(
            x, self.weight, self.bias,
            self.scale1.item(), self.scale2.item(),
            self.stride, self.stride, self.stride,
            self.padding, self.padding, self.padding
        )