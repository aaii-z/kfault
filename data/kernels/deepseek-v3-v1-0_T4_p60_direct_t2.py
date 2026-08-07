import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3d + Swish + GroupNorm + HardSwish
fused_conv_gn_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float swish_activation(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float hardswish_activation(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_conv_transpose_swish_gn_hardswish_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, const float* gamma, const float* beta,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int groups, int group_size, float eps) {
    
    const int out_depth = (depth - 1) * stride_d + kernel_d - 2 * padding_d;
    const int out_height = (height - 1) * stride_h + kernel_h - 2 * padding_h;
    const int out_width = (width - 1) * stride_w + kernel_w - 2 * padding_w;
    
    const int n = blockIdx.x;
    const int c = blockIdx.y;
    const int g = c / group_size;
    const int local_c = c % group_size;
    
    // Shared memory for group statistics
    __shared__ float shared_mean;
    __shared__ float shared_variance;
    
    // First pass: compute convolution + swish and accumulate for group statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = out_depth * out_height * out_width;
    
    for (int d = 0; d < out_depth; d++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                float conv_result = 0.0f;
                
                // Transposed convolution operation
                for (int kd = 0; kd < kernel_d; kd++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int in_d = (d - kd + padding_d) / stride_d;
                            int in_h = (h - kh + padding_h) / stride_h;
                            int in_w = (w - kw + padding_w) / stride_w;
                            
                            if ((d - kd + padding_d) % stride_d == 0 &&
                                (h - kh + padding_h) % stride_h == 0 &&
                                (w - kw + padding_w) % stride_w == 0 &&
                                in_d >= 0 && in_d < depth &&
                                in_h >= 0 && in_h < height &&
                                in_w >= 0 && in_w < width) {
                                
                                for (int ic = 0; ic < in_channels; ic++) {
                                    int weight_idx = ((c * in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                                    int input_idx = ((n * in_channels + ic) * depth + in_d) * height * width + in_h * width + in_w;
                                    conv_result += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
                
                if (bias != nullptr) {
                    conv_result += bias[c];
                }
                
                // Apply Swish activation
                float activated = swish_activation(conv_result);
                
                // Store intermediate result and accumulate statistics
                int output_idx = ((n * out_channels + c) * out_depth + d) * out_height * out_width + h * out_width + w;
                output[output_idx] = activated;
                
                sum += activated;
                sum_sq += activated * activated;
            }
        }
    }
    
    // Compute mean and variance for the group
    float mean = sum / count;
    float variance = (sum_sq / count) - (mean * mean);
    
    if (threadIdx.x == 0) {
        shared_mean = mean;
        shared_variance = sqrtf(variance + eps);
    }
    __syncthreads();
    
    // Second pass: apply group normalization and hardswish
    for (int d = 0; d < out_depth; d++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                int output_idx = ((n * out_channels + c) * out_depth + d) * out_height * out_width + h * out_width + w;
                float normalized = (output[output_idx] - shared_mean) / shared_variance;
                normalized = normalized * gamma[c] + beta[c];
                output[output_idx] = hardswish_activation(normalized);
            }
        }
    }
}

torch::Tensor fused_conv_transpose_swish_gn_hardswish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int groups, float eps) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    int out_channels = weight.size(0);
    int group_size = out_channels / groups;
    
    int out_depth = (depth - 1) * stride_d + kernel_d - 2 * padding_d;
    int out_height = (height - 1) * stride_h + kernel_h - 2 * padding_h;
    int out_width = (width - 1) * stride_w + kernel_w - 2 * padding_w;
    
    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, 
                              input.options());
    
    dim3 grid(batch_size, out_channels);
    dim3 block(1);
    
    fused_conv_transpose_swish_gn_hardswish_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        groups, group_size, eps);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_conv_gn_activation_cpp_source = """
torch::Tensor fused_conv_transpose_swish_gn_hardswish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int groups, float eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_conv_gn_activation_cpp_source,
    cuda_sources=fused_conv_gn_activation_source,
    functions=["fused_conv_transpose_swish_gn_hardswish_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.eps = eps
        
        # Initialize weights and biases for transposed convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters for group normalization
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize the custom CUDA operation
        self.fused_op = fused_ops

    def forward(self, x):
        return self.fused_op.fused_conv_transpose_swish_gn_hardswish_cuda(
            x, self.weight, self.bias, self.gamma, self.beta,
            self.kernel_size, self.kernel_size, self.kernel_size,
            self.stride, self.stride, self.stride,
            self.padding, self.padding, self.padding,
            self.groups, self.eps
        )