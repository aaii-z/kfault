import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv3D + Mish + Tanh
fused_conv3d_mish_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Mish activation function
__device__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

// Fused Conv3D + Mish + Tanh kernel
__global__ void fused_conv3d_mish_tanh_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, 
    int batch_size, int in_channels, int out_channels,
    int D, int H, int W, int kernel_size, int stride, int padding,
    int out_D, int out_H, int out_W) {
    
    // Calculate output indices using 3D grid
    int n = blockIdx.z;  // batch
    int c_out = blockIdx.y;  // output channel
    int d_out = blockIdx.x;  // depth
    
    // Calculate thread indices for height and width
    int h_out = threadIdx.y;
    int w_out = threadIdx.x;
    
    if (n >= batch_size || c_out >= out_channels || d_out >= out_D || 
        h_out >= out_H || w_out >= out_W) {
        return;
    }
    
    // Calculate input indices with padding
    int d_in_start = d_out * stride - padding;
    int h_in_start = h_out * stride - padding;
    int w_in_start = w_out * stride - padding;
    
    float sum = 0.0f;
    
    // Convolution operation
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int d_in = d_in_start + kd;
                    int h_in = h_in_start + kh;
                    int w_in = w_in_start + kw;
                    
                    // Check boundaries
                    if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int input_idx = ((n * in_channels + c_in) * D + d_in) * H * W + h_in * W + w_in;
                        int weight_idx = (((c_out * in_channels + c_in) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    // Apply Mish activation
    float mish_result = mish_activation(sum);
    
    // Apply Tanh activation
    float final_result = tanhf(mish_result);
    
    // Write output
    int output_idx = ((n * out_channels + c_out) * out_D + d_out) * out_H * out_W + h_out * out_W + w_out;
    output[output_idx] = final_result;
}

torch::Tensor fused_conv3d_mish_tanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto out_channels = weight.size(0);
    
    // Calculate output dimensions
    int out_D = (D + 2 * padding - kernel_size) / stride + 1;
    int out_H = (H + 2 * padding - kernel_size) / stride + 1;
    int out_W = (W + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_D, out_H, out_W}, input.options());
    
    // Configure grid and block dimensions properly
    dim3 grid_dim(out_D, out_channels, batch_size);
    dim3 block_dim(out_W, out_H, 1);
    
    // Check if block dimensions are valid
    if (block_dim.x > 1024 || block_dim.y > 1024 || block_dim.z > 64) {
        // Fall back to smaller block size if needed
        block_dim = dim3(16, 16, 1);
    }
    
    fused_conv3d_mish_tanh_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        D, H, W, kernel_size, stride, padding,
        out_D, out_H, out_W);
    
    cudaDeviceSynchronize();
    
    return output;
}
"""

fused_conv3d_mish_tanh_cpp_source = """
torch::Tensor fused_conv3d_mish_tanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code
fused_conv3d_mish_tanh = load_inline(
    name="fused_conv3d_mish_tanh",
    cpp_sources=fused_conv3d_mish_tanh_cpp_source,
    cuda_sources=fused_conv3d_mish_tanh_source,
    functions=["fused_conv3d_mish_tanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    """
    Optimized model that performs fused 3D convolution with Mish and Tanh activations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize weights using Kaiming initialization for better convergence
        nn.init.kaiming_uniform_(self.weight, a=5.0**0.5)
        if self.bias is not None:
            fan_in = in_channels * kernel_size * kernel_size * kernel_size
            bound = 1.0 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch::Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        return fused_conv3d_mish_tanh.fused_conv3d_mish_tanh_cuda(
            x, self.weight, self.bias, self.kernel_size, self.stride, self.padding
        )