import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused CUDA kernel for Conv3D + MaxPool3D + LogSumExp + ReLU
fused_conv3d_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define CUDA_CHECK(call) {\\
    cudaError_t err = call;\\
    if (err != cudaSuccess) {\\
        fprintf(stderr, "CUDA error at %s:%d: %s\\\\n", __FILE__, __LINE__, cudaGetErrorString(err));\\
        exit(EXIT_FAILURE);\\
    }\\
}

__global__ void fused_conv3d_maxpool_logsumexp_relu_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, 
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    // Output dimensions after convolution
    int out_depth = (depth + 2 * pad_d - kernel_d) / stride_d + 1;
    int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // MaxPool dimensions (pooling with kernel 2, stride 2)
    int pool_depth = (out_depth + 1) / 2;
    int pool_height = (out_height + 1) / 2;
    int pool_width = (out_width + 1) / 2;
    
    // Calculate output position
    int n = blockIdx.x;  // batch index
    int c = blockIdx.y;  // output channel
    int d = threadIdx.x + blockIdx.z * blockDim.x;  // depth
    int h = threadIdx.y + blockIdx.w * blockDim.y;  // height
    int w = threadIdx.z;  // width
    
    if (n >= batch_size || c >= out_channels || d >= pool_depth || h >= pool_height || w >= pool_width) {
        return;
    }
    
    // Calculate input position for max pooling (2x2x2 window)
    int conv_d_start = d * 2;
    int conv_h_start = h * 2;
    int conv_w_start = w * 2;
    
    int conv_d_end = min(conv_d_start + 2, out_depth);
    int conv_h_end = min(conv_h_start + 2, out_height);
    int conv_w_end = min(conv_w_start + 2, out_width);
    
    // For LogSumExp: accumulate max and sum of exp values across max pooling window
    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;
    
    // Iterate over max pooling window
    for (int pd = conv_d_start; pd < conv_d_end; pd++) {
        for (int ph = conv_h_start; ph < conv_h_end; ph++) {
            for (int pw = conv_w_start; pw < conv_w_end; pw++) {
                
                // Perform convolution for this position
                float conv_val = 0.0f;
                if (bias) {
                    conv_val = bias[c];
                }
                
                // Convolution operation
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kd = 0; kd < kernel_d; kd++) {
                        for (int kh = 0; kh < kernel_h; kh++) {
                            for (int kw = 0; kw < kernel_w; kw++) {
                                int input_d = pd * stride_d + kd - pad_d;
                                int input_h = ph * stride_h + kh - pad_h;
                                int input_w = pw * stride_w + kw - pad_w;
                                
                                if (input_d >= 0 && input_d < depth && 
                                    input_h >= 0 && input_h < height && 
                                    input_w >= 0 && input_w < width) {
                                    
                                    int input_idx = ((n * in_channels + ic) * depth + input_d) * height * width + 
                                                  input_h * width + input_w;
                                    
                                    int weight_idx = ((c * in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w + 
                                                   kh * kernel_w + kw;
                                    
                                    conv_val += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
                
                // Track max for LogSumExp
                if (conv_val > max_val) {
                    max_val = conv_val;
                }
            }
        }
    }
    
    // Second pass for LogSumExp: calculate sum of exp(values - max_val)
    for (int pd = conv_d_start; pd < conv_d_end; pd++) {
        for (int ph = conv_h_start; ph < conv_h_end; ph++) {
            for (int pw = conv_w_start; pw < conv_w_end; pw++) {
                
                // Perform convolution again
                float conv_val = 0.0f;
                if (bias) {
                    conv_val = bias[c];
                }
                
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kd = 0; kd < kernel_d; kd++) {
                        for (int kh = 0; kh < kernel_h; kh++) {
                            for (int kw = 0; kw < kernel_w; kw++) {
                                int input_d = pd * stride_d + kd - pad_d;
                                int input_h = ph * stride_h + kh - pad_h;
                                int input_w = pw * stride_w + kw - pad_w;
                                
                                if (input_d >= 0 && input_d < depth && 
                                    input_h >= 0 && input_h < height && 
                                    input_w >= 0 && input_w < width) {
                                    
                                    int input_idx = ((n * in_channels + ic) * depth + input_d) * height * width + 
                                                  input_h * width + input_w;
                                    
                                    int weight_idx = ((c * in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w + 
                                                   kh * kernel_w + kw;
                                    
                                    conv_val += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
                
                sum_exp += expf(conv_val - max_val);
            }
        }
    }
    
    // Calculate LogSumExp: max_val + log(sum_exp)
    float logsumexp_val = max_val + logf(sum_exp);
    
    // Apply ReLU
    float result = fmaxf(0.0f, logsumexp_val);
    
    // Write output
    int output_idx = ((n * out_channels + c) * pool_depth + d) * pool_height * pool_width + 
                    h * pool_width + w;
    output[output_idx] = result;
}

torch::Tensor fused_conv3d_maxpool_logsumexp_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto out_channels = weight.size(0);
    
    // Calculate output dimensions after convolution
    int out_depth = (depth + 2 * pad_d - kernel_d) / stride_d + 1;
    int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // MaxPool dimensions (pooling with kernel 2, stride 2)
    int pool_depth = (out_depth + 1) / 2;
    int pool_height = (out_height + 1) / 2;
    int pool_width = (out_width + 1) / 2;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, pool_depth, pool_height, pool_width}, 
                              torch::dtype(torch::kFloat32).device(input.device()));
    
    // Configure kernel launch parameters
    dim3 block_size(4, 4, 4);
    int grid_x = batch_size;
    int grid_y = out_channels;
    int grid_z = (pool_depth + block_size.x - 1) / block_size.x;
    int grid_w = (pool_height + block_size.y - 1) / block_size.y;
    
    dim3 grid_dims(grid_x, grid_y, grid_z, grid_w);
    
    // Launch kernel
    fused_conv3d_maxpool_logsumexp_relu_kernel<<<grid_dims, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w);
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

fused_conv3d_ops_cpp_source = """
torch::Tensor fused_conv3d_maxpool_logsumexp_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w);
"""

# Compile the inline CUDA code
fused_conv3d_ops = load_inline(
    name="fused_conv3d_ops",
    cpp_sources=fused_conv3d_ops_cpp_source,
    cuda_sources=fused_conv3d_ops_source,
    functions=["fused_conv3d_maxpool_logsumexp_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs fused 3D convolution, max pooling, log sum exp, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Convert kernel_size, stride, padding to tuples if they are integers
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        
        # Initialize weights and bias like nn.Conv3d
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, 
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
        ))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width')
        """
        return fused_conv3d_ops.fused_conv3d_maxpool_logsumexp_relu_cuda(
            x, self.weight, self.bias,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2]
        )