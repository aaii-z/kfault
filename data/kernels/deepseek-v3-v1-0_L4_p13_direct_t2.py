import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3D + MeanPool + BiasAdd + Softmax + Tanh + Scale
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void fused_operator_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float scaling_factor,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int output_depth, int output_height, int output_width) {
    
    // Calculate output spatial indices
    int n = blockIdx.z;  // batch index
    int c = blockIdx.y;  // output channel index
    int d = blockIdx.x * blockDim.x + threadIdx.x; // output depth index
    int h = blockIdx.x * blockDim.y + threadIdx.y; // output height index
    int w = blockIdx.x * blockDim.z + threadIdx.z; // output width index
    
    if (n >= batch_size || c >= out_channels || d >= output_depth || h >= output_height || w >= output_width) {
        return;
    }
    
    // Initialize convolution result
    float conv_result = 0.0f;
    
    // Perform transposed convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int input_d = d - kd + padding;
                    int input_h = h - kh + padding;
                    int input_w = w - kw + padding;
                    
                    if (input_d >= 0 && input_d < depth && input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                        int input_idx = n * (in_channels * depth * height * width) +
                                      ic * (depth * height * width) +
                                      input_d * (height * width) +
                                      input_h * width +
                                      input_w;
                        
                        int weight_idx = c * (in_channels * kernel_d * kernel_h * kernel_w) +
                                       ic * (kernel_d * kernel_h * kernel_w) +
                                       kd * (kernel_h * kernel_w) +
                                       kh * kernel_w +
                                       kw;
                        
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Store convolution result
    int output_idx = n * (out_channels * output_depth * output_height * output_width) +
                   c * (output_depth * output_height * output_width) +
                   d * (output_height * output_width) +
                   h * output_width +
                   w;
    output[output_idx] = conv_result;
    
    __syncthreads();
    
    // Mean pooling across depth dimension (only for thread 0 in each spatial position)
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && d == 0) {
        for (int h_pos = 0; h_pos < output_height; ++h_pos) {
            for (int w_pos = 0; w_pos < output_width; ++w_pos) {
                float mean_pool = 0.0f;
                for (int od = 0; od < output_depth; ++od) {
                    int pool_idx = n * (out_channels * output_depth * output_height * output_width) +
                                 c * (output_depth * output_height * output_width) +
                                 od * (output_height * output_width) +
                                 h_pos * output_width +
                                 w_pos;
                    mean_pool += output[pool_idx];
                }
                mean_pool /= output_depth;
                
                // Add bias
                mean_pool += bias[c];
                
                // Store for softmax computation
                int store_idx = n * (out_channels * output_height * output_width) +
                              c * (output_height * output_width) +
                              h_pos * output_width +
                              w_pos;
                output[store_idx] = mean_pool;
            }
        }
    }
    
    __syncthreads();
    
    // Softmax across channels (only for thread 0 in each spatial position)
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && d == 0) {
        for (int h_pos = 0; h_pos < output_height; ++h_pos) {
            for (int w_pos = 0; w_pos < output_width; ++w_pos) {
                // First pass: find max value for numerical stability
                float max_val = -FLT_MAX;
                for (int oc = 0; oc < out_channels; ++oc) {
                    int idx = n * (out_channels * output_height * output_width) +
                             oc * (output_height * output_width) +
                             h_pos * output_width +
                             w_pos;
                    max_val = fmaxf(max_val, output[idx]);
                }
                
                // Second pass: compute exponentials and sum
                float sum_exp = 0.0f;
                for (int oc = 0; oc < out_channels; ++oc) {
                    int idx = n * (out_channels * output_height * output_width) +
                             oc * (output_height * output_width) +
                             h_pos * output_width +
                             w_pos;
                    float val = output[idx] - max_val;
                    output[idx] = expf(val);
                    sum_exp += output[idx];
                }
                
                // Third pass: normalize and apply tanh + scaling
                for (int oc = 0; oc < out_channels; ++oc) {
                    int idx = n * (out_channels * output_height * output_width) +
                             oc * (output_height * output_width) +
                             h_pos * output_width +
                             w_pos;
                    float softmax_val = output[idx] / sum_exp;
                    float tanh_val = tanhf(softmax_val);
                    output[idx] = tanh_val * scaling_factor;
                }
            }
        }
    }
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, int kernel_size, int stride, int padding) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto out_channels = weight.size(1);
    
    int kernel_d = kernel_size;
    int kernel_h = kernel_size;
    int kernel_w = kernel_size;
    
    // Calculate output spatial dimensions for transposed convolution
    int output_depth = (depth - 1) * stride + kernel_d - 2 * padding;
    int output_height = (height - 1) * stride + kernel_h - 2 * padding;
    int output_width = (width - 1) * stride + kernel_w - 2 * padding;
    
    // Create output tensor (after mean pooling, we only keep depth=1)
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());
    
    // Launch kernel with proper grid and block dimensions
    dim3 blocks((output_depth + 7) / 8, (output_height + 7) / 8, (output_width + 7) / 8);
    dim3 threads(8, 8, 8);
    
    fused_operator_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), scaling_factor,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_d, kernel_h, kernel_w,
        stride, padding, output_depth, output_height, output_width);
    
    // Return only the first depth slice (after mean pooling)
    return output.narrow(2, 0, 1);
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, int kernel_size, int stride, int padding);
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scaling_factor = scaling_factor
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias, self.scaling_factor, 
            self.kernel_size, self.stride, self.padding
        )