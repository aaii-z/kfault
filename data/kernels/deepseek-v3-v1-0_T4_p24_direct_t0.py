import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv3D + Min + Softmax
fused_conv_min_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_min_softmax_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int D, int H, int W, int kernel_size, int min_dim) {
    
    const int output_H = H - kernel_size + 1;
    const int output_W = W - kernel_size + 1;
    const int output_D = D - kernel_size + 1;
    
    // Calculate 3D grid coordinates
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = threadIdx.x + blockIdx.z * blockDim.x;
    int w = threadIdx.y + blockIdx.w * blockDim.y;
    
    if (b >= batch_size || c >= out_channels || h >= output_H || w >= output_W) {
        return;
    }
    
    // Convolution operation
    float conv_result = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int input_d = h + kd;
                    int input_h = w + kh;
                    int input_w = threadIdx.z + kw;
                    
                    if (input_d < D && input_h < H && input_w < W) {
                        int input_idx = b * (in_channels * D * H * W) +
                                      ic * (D * H * W) +
                                      input_d * (H * W) +
                                      input_h * W +
                                      input_w;
                        
                        int weight_idx = c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                       ic * (kernel_size * kernel_size * kernel_size) +
                                       kd * (kernel_size * kernel_size) +
                                       kh * kernel_size +
                                       kw;
                        
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    conv_result += bias[c];
    
    // Store convolution result
    int output_idx = b * (out_channels * output_D * output_H * output_W) +
                    c * (output_D * output_H * output_W) +
                    threadIdx.z * (output_H * output_W) +
                    h * output_W +
                    w;
    
    output[output_idx] = conv_result;
}

__global__ void min_reduction_kernel(
    const float* input, float* min_output, int batch_size, int out_channels,
    int D, int H, int W, int reduction_dim) {
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    int c = threadIdx.x;
    
    if (b >= batch_size || c >= out_channels || h >= H || w >= W) {
        return;
    }
    
    float min_val = INFINITY;
    for (int d = 0; d < D; ++d) {
        int idx = b * (out_channels * D * H * W) +
                 c * (D * H * W) +
                 d * (H * W) +
                 h * W +
                 w;
        min_val = fminf(min_val, input[idx]);
    }
    
    int output_idx = b * (out_channels * H * W) +
                    c * (H * W) +
                    h * W +
                    w;
    
    min_output[output_idx] = min_val;
}

__global__ void softmax_kernel(
    float* input, int batch_size, int channels, int H, int W) {
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    int c = threadIdx.x;
    
    if (b >= batch_size || c >= channels || h >= H || w >= W) {
        return;
    }
    
    __shared__ float max_val;
    __shared__ float sum_exp;
    
    // Find max value for numerical stability
    if (threadIdx.x == 0) {
        max_val = -INFINITY;
        for (int cc = 0; cc < channels; ++cc) {
            int idx = b * (channels * H * W) +
                     cc * (H * W) +
                     h * W +
                     w;
            max_val = fmaxf(max_val, input[idx]);
        }
    }
    __syncthreads();
    
    // Calculate sum of exponentials
    float exp_val = 0.0f;
    int idx = b * (channels * H * W) +
             c * (H * W) +
             h * W +
             w;
    
    exp_val = expf(input[idx] - max_val);
    
    // Atomic add to sum
    atomicAdd(&sum_exp, exp_val);
    __syncthreads();
    
    // Normalize
    input[idx] = exp_val / sum_exp;
}

torch::Tensor fused_conv_min_softmax_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int min_dim) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int out_channels = weight.size(0);
    
    int output_D = D - kernel_size + 1;
    int output_H = H - kernel_size + 1;
    int output_W = W - kernel_size + 1;
    
    // Create output tensor for convolution
    auto conv_output = torch::zeros({batch_size, out_channels, output_D, output_H, output_W}, 
                                   torch::dtype(torch::kFloat32).device(input.device()));
    
    // Launch convolution kernel
    dim3 conv_blocks(batch_size, out_channels, (output_H + 7) / 8, (output_W + 7) / 8);
    dim3 conv_threads(8, 8, output_D);
    
    fused_conv_min_softmax_kernel<<<conv_blocks, conv_threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        conv_output.data_ptr<float>(), batch_size, in_channels, out_channels,
        D, H, W, kernel_size, min_dim);
    
    // Apply min reduction along specified dimension
    auto min_output = torch::zeros({batch_size, out_channels, output_H, output_W}, 
                                  torch::dtype(torch::kFloat32).device(input.device()));
    
    dim3 min_blocks(batch_size, output_H, output_W);
    dim3 min_threads(out_channels);
    
    min_reduction_kernel<<<min_blocks, min_threads>>>(
        conv_output.data_ptr<float>(), min_output.data_ptr<float>(),
        batch_size, out_channels, output_D, output_H, output_W, min_dim);
    
    // Apply softmax along channel dimension
    dim3 softmax_blocks(batch_size, output_H, output_W);
    dim3 softmax_threads(out_channels);
    
    softmax_kernel<<<softmax_blocks, softmax_threads>>>(
        min_output.data_ptr<float>(), batch_size, out_channels, output_H, output_W);
    
    return min_output;
}
"""

fused_conv_min_softmax_cpp_source = """
torch::Tensor fused_conv_min_softmax_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, int min_dim);
"""

# Compile the inline CUDA code
fused_conv_min_softmax = load_inline(
    name="fused_conv_min_softmax",
    cpp_sources=fused_conv_min_softmax_cpp_source,
    cuda_sources=fused_conv_min_softmax_source,
    functions=["fused_conv_min_softmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dim = dim
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_op = fused_conv_min_softmax

    def forward(self, x):
        return self.fused_op.fused_conv_min_softmax_cuda(
            x, self.weight, self.bias, self.kernel_size, self.dim
        )