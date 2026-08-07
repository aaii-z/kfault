import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3d + MaxPool3d + MaxPool3d + Sum
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cub/cub.cuh>

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        printf("%s in %s at line %d\\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

template<typename T>
__global__ void fused_conv_transpose_maxpool_sum_kernel(
    const T* input, const T* weight, const T* bias,
    T* output, 
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width,
    int kernel_size, int stride, int padding,
    int pool_kernel1, int pool_kernel2) {
    
    // Calculate output dimensions after transpose conv
    int conv_out_d = (input_depth - 1) * stride + kernel_size - 2 * padding;
    int conv_out_h = (input_height - 1) * stride + kernel_size - 2 * padding;
    int conv_out_w = (input_width - 1) * stride + kernel_size - 2 * padding;
    
    // Calculate output dimensions after first maxpool
    int pool1_out_d = (conv_out_d - pool_kernel1) / pool_kernel1 + 1;
    int pool1_out_h = (conv_out_h - pool_kernel1) / pool_kernel1 + 1;
    int pool1_out_w = (conv_out_w - pool_kernel1) / pool_kernel1 + 1;
    
    // Calculate output dimensions after second maxpool
    int pool2_out_d = (pool1_out_d - pool_kernel2) / pool_kernel2 + 1;
    int pool2_out_h = (pool1_out_h - pool_kernel2) / pool_kernel2 + 1;
    int pool2_out_w = (pool1_out_w - pool_kernel2) / pool_kernel2 + 1;
    
    // Final output dimensions
    int final_depth = pool2_out_d;
    int final_height = pool2_out_h;
    int final_width = pool2_out_w;
    
    // Thread indexing
    int n = blockIdx.x;
    int oc = blockIdx.y;
    int od = blockIdx.z / (final_height * final_width);
    int oh = (blockIdx.z % (final_height * final_width)) / final_width;
    int ow = blockIdx.z % final_width;
    
    if (n >= batch_size || oc >= out_channels || od >= final_depth || oh >= final_height || ow >= final_width) {
        return;
    }
    
    // Calculate input indices for second maxpool
    int pool2_start_d = od * pool_kernel2;
    int pool2_start_h = oh * pool_kernel2;
    int pool2_start_w = ow * pool_kernel2;
    
    T max_val1 = -FLT_MAX;
    T max_val2 = -FLT_MAX;
    T final_max = -FLT_MAX;
    
    // First maxpool window
    for (int pd1 = 0; pd1 < pool_kernel2; ++pd1) {
        for (int ph1 = 0; ph1 < pool_kernel2; ++ph1) {
            for (int pw1 = 0; pw1 < pool_kernel2; ++pw1) {
                
                int pool1_d = pool2_start_d + pd1;
                int pool1_h = pool2_start_h + ph1;
                int pool1_w = pool2_start_w + pw1;
                
                if (pool1_d < pool1_out_d && pool1_h < pool1_out_h && pool1_w < pool1_out_w) {
                    
                    // Calculate input indices for first maxpool
                    int conv_start_d = pool1_d * pool_kernel1;
                    int conv_start_h = pool1_h * pool_kernel1;
                    int conv_start_w = pool1_w * pool_kernel1;
                    
                    max_val2 = -FLT_MAX;
                    
                    // Second maxpool window
                    for (int pd2 = 0; pd2 < pool_kernel1; ++pd2) {
                        for (int ph2 = 0; ph2 < pool_kernel1; ++ph2) {
                            for (int pw2 = 0; pw2 < pool_kernel1; ++pw2) {
                                
                                int conv_d = conv_start_d + pd2;
                                int conv_h = conv_start_h + ph2;
                                int conv_w = conv_start_w + pw2;
                                
                                if (conv_d < conv_out_d && conv_h < conv_out_h && conv_w < conv_out_w) {
                                    
                                    T conv_val = bias[oc];
                                    
                                    // Transposed convolution
                                    for (int ic = 0; ic < in_channels; ++ic) {
                                        for (int kd = 0; kd < kernel_size; ++kd) {
                                            for (int kh = 0; kh < kernel_size; ++kh) {
                                                for (int kw = 0; kw < kernel_size; ++kw) {
                                                    
                                                    int input_d = (conv_d + padding - kd) / stride;
                                                    int input_h = (conv_h + padding - kh) / stride;
                                                    int input_w = (conv_w + padding - kw) / stride;
                                                    
                                                    if (input_d >= 0 && input_d < input_depth &&
                                                        input_h >= 0 && input_h < input_height &&
                                                        input_w >= 0 && input_w < input_width &&
                                                        (conv_d + padding - kd) % stride == 0 &&
                                                        (conv_h + padding - kh) % stride == 0 &&
                                                        (conv_w + padding - kw) % stride == 0) {
                                                        
                                                        int input_idx = ((n * in_channels + ic) * input_depth + input_d) * input_height * input_width +
                                                                        input_h * input_width + input_w;
                                                        int weight_idx = ((oc * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size +
                                                                        kh * kernel_size + kw;
                                                        
                                                        conv_val += input[input_idx] * weight[weight_idx];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    
                                    max_val2 = max(max_val2, conv_val);
                                }
                            }
                        }
                    }
                    
                    max_val1 = max(max_val1, max_val2);
                }
            }
        }
    }
    
    final_max = max_val1;
    
    // Atomic max for sum reduction across channels
    int output_idx = ((n * 1 + 0) * final_depth + od) * final_height * final_width +
                     oh * final_width + ow;
    
    atomicMax(reinterpret_cast<int*>(&output[output_idx]), 
              reinterpret_cast<int&>(final_max));
}

torch::Tensor fused_conv_transpose_maxpool_sum_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding, int pool_kernel1, int pool_kernel2) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);
    auto out_channels = weight.size(0);
    
    // Calculate output dimensions
    int conv_out_d = (input_depth - 1) * stride + kernel_size - 2 * padding;
    int conv_out_h = (input_height - 1) * stride + kernel_size - 2 * padding;
    int conv_out_w = (input_width - 1) * stride + kernel_size - 2 * padding;
    
    int pool1_out_d = (conv_out_d - pool_kernel1) / pool_kernel1 + 1;
    int pool1_out_h = (conv_out_h - pool_kernel1) / pool_kernel1 + 1;
    int pool1_out_w = (conv_out_w - pool_kernel1) / pool_kernel1 + 1;
    
    int pool2_out_d = (pool1_out_d - pool_kernel2) / pool_kernel2 + 1;
    int pool2_out_h = (pool1_out_h - pool_kernel2) / pool_kernel2 + 1;
    int pool2_out_w = (pool1_out_w - pool_kernel2) / pool_kernel2 + 1;
    
    // Create output tensor with keepdim=True
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, 1, pool2_out_d, pool2_out_h, pool2_out_w}, options);
    
    // Configure kernel launch parameters
    dim3 grid_size(batch_size, out_channels, pool2_out_d * pool2_out_h * pool2_out_w);
    int block_size = 1;
    
    // Launch kernel
    fused_conv_transpose_maxpool_sum_kernel<float><<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        conv_out_d, conv_out_h, conv_out_w,
        kernel_size, stride, padding,
        pool_kernel1, pool_kernel2);
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_conv_transpose_maxpool_sum_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding, int pool_kernel1, int pool_kernel2);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv_transpose_maxpool_sum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcub"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias for transposed convolution
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size, kernel_size
        ))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_conv_transpose_maxpool_sum_cuda(
            x, self.weight, self.bias,
            self.kernel_size, self.stride, self.padding, 2, 3
        )