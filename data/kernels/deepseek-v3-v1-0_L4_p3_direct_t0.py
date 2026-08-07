import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3d + Add + LayerNorm + AvgPool3d + GELU
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_add_norm_pool_gelu_kernel(
    const float* input, const float* weight, const float* bias, 
    float sum_weight, float* gamma, float* beta, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int output_padding_depth, int output_padding_height, int output_padding_width,
    int pool_kernel_depth, int pool_kernel_height, int pool_kernel_width) {
    
    // Calculate output dimensions after pooling
    int pooled_depth = (output_depth + pool_kernel_depth - 1) / pool_kernel_depth;
    int pooled_height = (output_height + pool_kernel_height - 1) / pool_kernel_height;
    int pooled_width = (output_width + pool_kernel_width - 1) / pool_kernel_width;
    
    int n = blockIdx.x;
    int c = blockIdx.y;
    int d = threadIdx.x * blockDim.y + threadIdx.y;
    int h = blockIdx.z % pooled_height;
    int w = blockIdx.z / pooled_height;
    
    if (n >= batch_size || c >= out_channels || d >= pooled_depth || h >= pooled_height || w >= pooled_width) {
        return;
    }
    
    // Layer normalization statistics
    float mean = 0.0f;
    float variance = 0.0f;
    int count = 0;
    
    // Calculate mean and variance for this channel across spatial dimensions
    for (int pd = d * pool_kernel_depth; pd < min((d + 1) * pool_kernel_depth, output_depth); pd++) {
        for (int ph = h * pool_kernel_height; ph < min((h + 1) * pool_kernel_height, output_height); ph++) {
            for (int pw = w * pool_kernel_width; pw < min((w + 1) * pool_kernel_width, output_width); pw++) {
                float conv_value = 0.0f;
                
                // Perform transposed convolution
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kd = 0; kd < kernel_depth; kd++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                            for (int kw = 0; kw < kernel_width; kw++) {
                                int input_d = (pd - kd + padding_depth) / stride_depth;
                                int input_h = (ph - kh + padding_height) / stride_height;
                                int input_w = (pw - kw + padding_width) / stride_width;
                                
                                if (input_d >= 0 && input_d < input_depth && 
                                    input_h >= 0 && input_h < input_height && 
                                    input_w >= 0 && input_w < input_width &&
                                    (pd - kd + padding_depth) % stride_depth == 0 &&
                                    (ph - kh + padding_height) % stride_height == 0 &&
                                    (pw - kw + padding_width) % stride_width == 0) {
                                    
                                    int input_idx = ((n * in_channels + ic) * input_depth + input_d) * input_height * input_width + 
                                                  input_h * input_width + input_w;
                                    int weight_idx = ((c * in_channels + ic) * kernel_depth + kd) * kernel_height * kernel_width + 
                                                   kh * kernel_width + kw;
                                    
                                    conv_value += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
                
                // Add bias and sum_weight
                conv_value += bias[c] + sum_weight;
                mean += conv_value;
                variance += conv_value * conv_value;
                count++;
            }
        }
    }
    
    mean /= count;
    variance = variance / count - mean * mean;
    float inv_std = 1.0f / sqrtf(variance + 1e-5f);
    
    // Apply layer normalization, pooling, and GELU
    float pooled_value = 0.0f;
    int pool_count = 0;
    
    for (int pd = d * pool_kernel_depth; pd < min((d + 1) * pool_kernel_depth, output_depth); pd++) {
        for (int ph = h * pool_kernel_height; ph < min((h + 1) * pool_kernel_height, output_height); ph++) {
            for (int pw = w * pool_kernel_width; pw < min((w + 1) * pool_kernel_width, output_width); pw++) {
                float conv_value = 0.0f;
                
                // Perform transposed convolution again (could be optimized with shared memory)
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kd = 0; kd < kernel_depth; kd++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                            for (int kw = 0; kw < kernel_width; kw++) {
                                int input_d = (pd - kd + padding_depth) / stride_depth;
                                int input_h = (ph - kh + padding_height) / stride_height;
                                int input_w = (pw - kw + padding_width) / stride_width;
                                
                                if (input_d >= 0 && input_d < input_depth && 
                                    input_h >= 0 && input_h < input_height && 
                                    input_w >= 0 && input_w < input_width &&
                                    (pd - kd + padding_depth) % stride_depth == 0 &&
                                    (ph - kh + padding_height) % stride_height == 0 &&
                                    (pw - kw + padding_width) % stride_width == 0) {
                                    
                                    int input_idx = ((n * in_channels + ic) * input_depth + input_d) * input_height * input_width + 
                                                  input_h * input_width + input_w;
                                    int weight_idx = ((c * in_channels + ic) * kernel_depth + kd) * kernel_height * kernel_width + 
                                                   kh * kernel_width + kw;
                                    
                                    conv_value += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
                
                // Add bias and sum_weight, then apply layer normalization and GELU
                conv_value = (conv_value + bias[c] + sum_weight - mean) * inv_std * gamma[c] + beta[c];
                conv_value = gelu(conv_value);
                pooled_value += conv_value;
                pool_count++;
            }
        }
    }
    
    // Average pooling
    int output_idx = ((n * out_channels + c) * pooled_depth + d) * pooled_height * pooled_width + 
                    h * pooled_width + w;
    output[output_idx] = pooled_value / pool_count;
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    float sum_weight, torch::Tensor gamma, torch::Tensor beta,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int output_padding_depth, int output_padding_height, int output_padding_width,
    int pool_kernel_depth, int pool_kernel_height, int pool_kernel_width) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(1);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);
    
    // Calculate output dimensions for transposed convolution
    int output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth;
    int output_height = (input_height - 1) * stride_height - 2 * padding_height + kernel_height + output_padding_height;
    int output_width = (input_width - 1) * stride_width - 2 * padding_width + kernel_width + output_padding_width;
    
    // Calculate pooled dimensions
    int pooled_depth = (output_depth + pool_kernel_depth - 1) / pool_kernel_depth;
    int pooled_height = (output_height + pool_kernel_height - 1) / pool_kernel_height;
    int pooled_width = (output_width + pool_kernel_width - 1) / pool_kernel_width;
    
    auto output = torch::zeros({batch_size, out_channels, pooled_depth, pooled_height, pooled_width}, 
                              torch::dtype(torch::kFloat32).device(input.device()));
    
    dim3 blocks(batch_size, out_channels, pooled_height * pooled_width);
    dim3 threads(16, 16); // 256 threads per block
    
    fused_conv_transpose_add_norm_pool_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        sum_weight, gamma.data_ptr<float>(), beta.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_depth, kernel_height, kernel_width,
        stride_depth, stride_height, stride_width,
        padding_depth, padding_height, padding_width,
        output_padding_depth, output_padding_height, output_padding_width,
        pool_kernel_depth, pool_kernel_height, pool_kernel_width);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    float sum_weight, torch::Tensor gamma, torch::Tensor beta,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int output_padding_depth, int output_padding_height, int output_padding_width,
    int pool_kernel_depth, int pool_kernel_height, int pool_kernel_width);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.pool_kernel_size = pool_kernel_size
        
        # Initialize convolution weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize layer normalization parameters
        self.gamma = nn.Parameter(torch.ones(norm_shape))
        self.beta = nn.Parameter(torch.zeros(norm_shape))
        
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias, self.sum_weight.item(), 
            self.gamma, self.beta,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2]
        )