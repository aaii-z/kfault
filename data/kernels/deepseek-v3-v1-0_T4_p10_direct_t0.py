import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose2d + MaxPool2d + Hardtanh + Mean + Tanh
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_transpose_maxpool_hardtanh_mean_tanh_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_size, int stride, int padding,
    int maxpool_kernel_size, int maxpool_stride,
    float hardtanh_min, float hardtanh_max) {
    
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;
    const int pooled_height = (output_height - maxpool_kernel_size) / maxpool_stride + 1;
    const int pooled_width = (output_width - maxpool_kernel_size) / maxpool_stride + 1;
    
    // Each block processes one output channel
    const int oc = blockIdx.x;
    const int n = blockIdx.y;
    
    if (n >= batch_size || oc >= out_channels) return;
    
    // Shared memory for accumulating mean
    __shared__ float channel_sum;
    if (threadIdx.x == 0) {
        channel_sum = 0.0f;
    }
    __syncthreads();
    
    // Process spatial dimensions
    for (int ph = threadIdx.y; ph < pooled_height; ph += blockDim.y) {
        for (int pw = threadIdx.x; pw < pooled_width; pw += blockDim.x) {
            float max_val = -FLT_MAX;
            
            // Max pooling window
            for (int kh = 0; kh < maxpool_kernel_size; ++kh) {
                for (int kw = 0; kw < maxpool_kernel_size; ++kw) {
                    const int h = ph * maxpool_stride + kh;
                    const int w = pw * maxpool_stride + kw;
                    
                    if (h < output_height && w < output_width) {
                        float conv_val = 0.0f;
                        
                        // Transposed convolution
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int ky = 0; ky < kernel_size; ++ky) {
                                for (int kx = 0; kx < kernel_size; ++kx) {
                                    const int input_h = (h + padding - ky) / stride;
                                    const int input_w = (w + padding - kx) / stride;
                                    
                                    if ((h + padding - ky) % stride == 0 && 
                                        (w + padding - kx) % stride == 0 &&
                                        input_h >= 0 && input_h < input_height &&
                                        input_w >= 0 && input_w < input_width) {
                                        
                                        const int input_idx = n * in_channels * input_height * input_width + 
                                                             ic * input_height * input_width + 
                                                             input_h * input_width + input_w;
                                        
                                        const int weight_idx = oc * in_channels * kernel_size * kernel_size + 
                                                              ic * kernel_size * kernel_size + 
                                                              ky * kernel_size + kx;
                                        
                                        conv_val += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                        
                        // Add bias
                        if (bias) {
                            conv_val += bias[oc];
                        }
                        
                        // Apply Hardtanh
                        conv_val = fminf(fmaxf(conv_val, hardtanh_min), hardtanh_max);
                        
                        // Update max value
                        max_val = fmaxf(max_val, conv_val);
                    }
                }
            }
            
            // Atomic add to channel sum
            atomicAdd(&channel_sum, max_val);
        }
    }
    
    __syncthreads();
    
    // Compute mean and apply tanh (only thread 0 per block)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        const float mean_val = channel_sum / (pooled_height * pooled_width);
        output[n * out_channels + oc] = tanhf(mean_val);
    }
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int maxpool_kernel_size,
    int maxpool_stride,
    float hardtanh_min,
    float hardtanh_max) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(1);
    
    // Calculate output dimensions
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;
    const int pooled_height = (output_height - maxpool_kernel_size) / maxpool_stride + 1;
    const int pooled_width = (output_width - maxpool_kernel_size) / maxpool_stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels}, input.options());
    
    // Grid and block dimensions
    dim3 grid(out_channels, batch_size);
    dim3 block(32, 32);  // Optimal for spatial processing
    
    fused_conv_transpose_maxpool_hardtanh_mean_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_size, stride, padding,
        maxpool_kernel_size, maxpool_stride,
        hardtanh_min, hardtanh_max);
    
    return output.unsqueeze(-1).unsqueeze(-1);  // Add spatial dimensions for keepdim=True
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int maxpool_kernel_size,
    int maxpool_stride,
    float hardtanh_min,
    float hardtanh_max);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Initialize weights and bias for transposed convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Load the fused operator
        self.fused_op = fused_operator

    def forward(self, x):
        return self.fused_op.fused_operator_cuda(
            x, 
            self.weight, 
            self.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.maxpool_kernel_size,
            self.maxpool_stride,
            self.hardtanh_min,
            self.hardtanh_max
        )