import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for conv3d + scaling + tanh + bias scaling + sigmoid
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv3d_activation_kernel(
    const float* input, const float* weight, const float* bias,
    const float* scaling_factor, const float* bias_scale,
    float* output, 
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width, int kernel_size) {
    
    const int output_depth = depth - kernel_size + 1;
    const int output_height = height - kernel_size + 1;
    const int output_width = width - kernel_size + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (idx < total_elements) {
        // Calculate output indices
        int n = idx / (out_channels * output_depth * output_height * output_width);
        int remainder = idx % (out_channels * output_depth * output_height * output_width);
        int c_out = remainder / (output_depth * output_height * output_width);
        remainder = remainder % (output_depth * output_height * output_width);
        int d_out = remainder / (output_height * output_width);
        remainder = remainder % (output_height * output_width);
        int h_out = remainder / output_width;
        int w_out = remainder % output_width;
        
        float sum = 0.0f;
        
        // 3D convolution
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kd = 0; kd < kernel_size; kd++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int d_in = d_out + kd;
                        int h_in = h_out + kh;
                        int w_in = w_out + kw;
                        
                        int input_idx = n * (in_channels * depth * height * width) +
                                      c_in * (depth * height * width) +
                                      d_in * (height * width) +
                                      h_in * width +
                                      w_in;
                        
                        int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                       c_in * (kernel_size * kernel_size * kernel_size) +
                                       kd * (kernel_size * kernel_size) +
                                       kh * kernel_size +
                                       kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        sum += bias[c_out];
        
        // Scale by scaling factor (broadcasted)
        float scale_val = scaling_factor[c_out * scaling_factor[1] + 0 * scaling_factor[2] + 0 * scaling_factor[3] + 0];
        sum *= scale_val;
        
        // Apply tanh
        sum = tanhf(sum);
        
        // Scale by bias scale (broadcasted)
        float bias_val = bias_scale[c_out * bias_scale[1] + 0 * bias_scale[2] + 0 * bias_scale[3] + 0];
        sum *= bias_val;
        
        // Apply sigmoid
        sum = 1.0f / (1.0f + expf(-sum));
        
        // Write output
        int output_idx = n * (out_channels * output_depth * output_height * output_width) +
                        c_out * (output_depth * output_height * output_width) +
                        d_out * (output_height * output_width) +
                        h_out * output_width +
                        w_out;
        
        output[output_idx] = sum;
    }
}

torch::Tensor fused_conv3d_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor scaling_factor, torch::Tensor bias_scale) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    const int output_depth = depth - kernel_size + 1;
    const int output_height = height - kernel_size + 1;
    const int output_width = width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, 
                              input.options());
    
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv3d_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        scaling_factor.data_ptr<float>(), bias_scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth, height, width, kernel_size);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_operator_cpp_source = "torch::Tensor fused_conv3d_activation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor scaling_factor, torch::Tensor bias_scale);"

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_conv3d_activation_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias_scale = nn.Parameter(torch.randn(bias_shape))
        
        self.fused_op = fused_operator

    def forward(self, x):
        return self.fused_op.fused_conv3d_activation_cuda(
            x, self.weight, self.conv_bias, self.scaling_factor, self.bias_scale
        )