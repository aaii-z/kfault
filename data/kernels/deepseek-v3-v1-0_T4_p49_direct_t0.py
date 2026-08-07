import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3D + Softmax + Sigmoid
fused_conv_softmax_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void fused_conv_softmax_sigmoid_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int D, int H, int W, int kernel_size, int stride, int padding, int output_padding) {
    
    // Calculate output dimensions
    int out_D = (D - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_H = (H - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_W = (W - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int output_size = batch_size * out_channels * out_D * out_H * out_W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= output_size) return;
    
    // Decompose global index
    int n = idx / (out_channels * out_D * out_H * out_W);
    int remainder = idx % (out_channels * out_D * out_H * out_W);
    int c = remainder / (out_D * out_H * out_W);
    remainder = remainder % (out_D * out_H * out_W);
    int d = remainder / (out_H * out_W);
    remainder = remainder % (out_H * out_W);
    int h = remainder / out_W;
    int w = remainder % out_W;
    
    // Perform transposed convolution
    float conv_value = 0.0f;
    if (bias != nullptr) {
        conv_value = bias[c];
    }
    
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int input_d = (d + padding - kd) / stride;
                int input_h = (h + padding - kh) / stride;
                int input_w = (w + padding - kw) / stride;
                
                if ((d + padding - kd) % stride == 0 && 
                    (h + padding - kh) % stride == 0 && 
                    (w + padding - kw) % stride == 0 &&
                    input_d >= 0 && input_d < D &&
                    input_h >= 0 && input_h < H &&
                    input_w >= 0 && input_w < W) {
                    
                    for (int ic = 0; ic < in_channels; ic++) {
                        int input_idx = n * (in_channels * D * H * W) + 
                                      ic * (D * H * W) + 
                                      input_d * (H * W) + 
                                      input_h * W + 
                                      input_w;
                        
                        int weight_idx = ic * (out_channels * kernel_size * kernel_size * kernel_size) + 
                                       c * (kernel_size * kernel_size * kernel_size) + 
                                       kd * (kernel_size * kernel_size) + 
                                       kh * kernel_size + 
                                       kw;
                        
                        conv_value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[idx] = conv_value;
}

__global__ void softmax_sigmoid_kernel(
    float* data, int batch_size, int out_channels, int out_D, int out_H, int out_W, int dim) {
    
    int spatial_size = out_D * out_H * out_W;
    int elements_per_sample = out_channels * spatial_size;
    
    for (int n = 0; n < batch_size; n++) {
        for (int spatial_idx = 0; spatial_idx < spatial_size; spatial_idx++) {
            // Find max for numerical stability
            float max_val = -FLT_MAX;
            for (int c = 0; c < out_channels; c++) {
                int idx = n * elements_per_sample + c * spatial_size + spatial_idx;
                max_val = fmaxf(max_val, data[idx]);
            }
            
            // Compute sum of exponentials
            float sum_exp = 0.0f;
            for (int c = 0; c < out_channels; c++) {
                int idx = n * elements_per_sample + c * spatial_size + spatial_idx;
                data[idx] = expf(data[idx] - max_val);
                sum_exp += data[idx];
            }
            
            // Apply softmax and sigmoid
            for (int c = 0; c < out_channels; c++) {
                int idx = n * elements_per_sample + c * spatial_size + spatial_idx;
                float softmax_val = data[idx] / sum_exp;
                data[idx] = 1.0f / (1.0f + expf(-softmax_val));  // Sigmoid
            }
        }
    }
}

torch::Tensor fused_conv_softmax_sigmoid_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding, int output_padding) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto out_channels = weight.size(1);
    
    // Calculate output dimensions
    int out_D = (D - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_H = (H - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_W = (W - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, out_D, out_H, out_W}, input.options());
    
    // Launch convolution kernel
    int conv_output_size = batch_size * out_channels * out_D * out_H * out_W;
    const int block_size = 256;
    const int num_blocks_conv = (conv_output_size + block_size - 1) / block_size;
    
    fused_conv_softmax_sigmoid_kernel<<<num_blocks_conv, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        D, H, W, kernel_size, stride, padding, output_padding);
    
    // Launch softmax + sigmoid kernel
    dim3 blocks(batch_size, (spatial_size + 31) / 32);
    dim3 threads(32, 32);
    
    softmax_sigmoid_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        batch_size, out_channels, out_D, out_H, out_W, 1);
    
    return output;
}
"""

fused_conv_softmax_sigmoid_cpp_source = """
torch::Tensor fused_conv_softmax_sigmoid_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding, int output_padding);
"""

# Compile the inline CUDA code
fused_conv_softmax_sigmoid = load_inline(
    name="fused_conv_softmax_sigmoid",
    cpp_sources=fused_conv_softmax_sigmoid_cpp_source,
    cuda_sources=fused_conv_softmax_sigmoid_source,
    functions=["fused_conv_softmax_sigmoid_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(
            in_channels, out_channels, kernel_size, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.fused_op = fused_conv_softmax_sigmoid

    def forward(self, x):
        return self.fused_op.fused_conv_softmax_sigmoid_cuda(
            x, self.weight, self.bias if self.bias is not None else torch.Tensor(),
            self.kernel_size, self.stride, self.padding, self.output_padding
        )