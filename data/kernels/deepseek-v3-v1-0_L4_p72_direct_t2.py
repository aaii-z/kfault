import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3d + BatchNorm3d
fused_conv_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void fused_conv_bn_kernel(
    const float* input, const float* weight, const float* bias, 
    const float* running_mean, const float* running_var, 
    const float* weight_bn, const float* bias_bn, 
    float* output, 
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width,
    int kernel_size, int stride, int padding, 
    float eps) {
    
    int n = blockIdx.w;
    int c_out = blockIdx.z;
    int d_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (n >= batch_size || c_out >= out_channels || d_out >= output_depth || h_out >= output_height) {
        return;
    }
    
    for (int w_out = 0; w_out < output_width; w_out++) {
        float sum = 0.0f;
        
        // Transposed convolution
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kd = 0; kd < kernel_size; kd++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int d_in = (d_out - kd + padding) / stride;
                        int h_in = (h_out - kh + padding) / stride;
                        int w_in = (w_out - kw + padding) / stride;
                        
                        if (d_in >= 0 && d_in < input_depth && 
                            h_in >= 0 && h_in < input_height && 
                            w_in >= 0 && w_in < input_width &&
                            (d_out - kd + padding) % stride == 0 &&
                            (h_out - kh + padding) % stride == 0 &&
                            (w_out - kw + padding) % stride == 0) {
                            
                            int input_idx = ((n * in_channels + c_in) * input_depth + d_in) * input_height * input_width + 
                                          h_in * input_width + w_in;
                            int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kd) * kernel_size * kernel_size + 
                                           kh * kernel_size + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias
        if (bias) {
            sum += bias[c_out];
        }
        
        // Batch normalization
        float mean = running_mean[c_out];
        float var = running_var[c_out];
        float normalized = (sum - mean) / sqrtf(var + eps);
        float scaled = normalized * weight_bn[c_out] + bias_bn[c_out];
        
        int output_idx = ((n * out_channels + c_out) * output_depth + d_out) * output_height * output_width + 
                       h_out * output_width + w_out;
        output[output_idx] = scaled;
    }
}

torch::Tensor fused_conv_bn_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight_bn, torch::Tensor bias_bn,
    int kernel_size, int stride, int padding, float eps) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(0);
    
    int output_depth = (input_depth - 1) * stride + kernel_size - 2 * padding;
    int output_height = (input_height - 1) * stride + kernel_size - 2 * padding;
    int output_width = (input_width - 1) * stride + kernel_size - 2 * padding;
    
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, 
                              input.options());
    
    dim3 block_size(8, 8);
    dim3 grid_size(
        (output_depth + block_size.x - 1) / block_size.x,
        (output_height + block_size.y - 1) / block_size.y,
        out_channels,
        batch_size
    );
    
    fused_conv_bn_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        weight_bn.data_ptr<float>(), bias_bn.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_size, stride, padding, eps
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return output;
}
"""

fused_conv_bn_cpp_source = "torch::Tensor fused_conv_bn_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight_bn, torch::Tensor bias_bn, int kernel_size, int stride, int padding, float eps);"

# Compile the inline CUDA code
fused_conv_bn = load_inline(
    name="fused_conv_bn",
    cpp_sources=fused_conv_bn_cpp_source,
    cuda_sources=fused_conv_bn_source,
    functions=["fused_conv_bn_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # BatchNorm parameters
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.weight_bn = nn.Parameter(torch.ones(out_channels))
        self.bias_bn = nn.Parameter(torch.zeros(out_channels))
        
        self.eps = 1e-5
        
        self.fused_op = fused_conv_bn
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        x = self.fused_op.fused_conv_bn_cuda(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.weight_bn, self.bias_bn,
            self.kernel_size, self.stride, self.padding,
            self.eps
        )
        x = self.avg_pool1(x)
        x = self.avg_pool2(x)
        return x