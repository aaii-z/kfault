import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3d + ReLU + GroupNorm
fused_conv_transpose_relu_gn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at %s:%d\\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void fused_conv_transpose_relu_gn_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, 
    int batch_size, int in_channels, int out_channels,
    int D, int H, int W, int kernel_size, int groups,
    int out_D, int out_H, int out_W) {
    
    const int group_size = out_channels / groups;
    const int spatial_size = out_D * out_H * out_W;
    
    // Calculate thread indices
    const int n = blockIdx.z;
    const int g = blockIdx.y;
    const int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n >= batch_size || g >= groups || spatial_idx >= spatial_size) return;
    
    const int c_start = g * group_size;
    const int d = spatial_idx / (out_H * out_W);
    const int h = (spatial_idx % (out_H * out_W)) / out_W;
    const int w = spatial_idx % out_W;
    
    // First phase: ConvTranspose3d + ReLU for each channel in the group
    for (int c = c_start; c < c_start + group_size; c++) {
        float sum = 0.0f;
        
        // ConvTranspose3d operation
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int in_d = d - kd + kernel_size / 2;
                    const int in_h = h - kh + kernel_size / 2;
                    const int in_w = w - kw + kernel_size / 2;
                    
                    if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                        for (int ic = 0; ic < in_channels; ic++) {
                            const int input_idx = ((n * in_channels + ic) * D + in_d) * H * W + in_h * W + in_w;
                            const int weight_idx = ((ic * out_channels + c) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        // Apply ReLU
        sum = fmaxf(0.0f, sum);
        
        const int output_idx = ((n * out_channels + c) * out_D + d) * out_H * out_W + h * out_W + w;
        output[output_idx] = sum;
    }
}

__global__ void compute_group_stats_kernel(
    const float* input, float* mean, float* variance,
    int batch_size, int out_channels, int groups,
    int out_D, int out_H, int out_W) {
    
    const int group_size = out_channels / groups;
    const int group_elements = group_size * out_D * out_H * out_W;
    const int n = blockIdx.x;
    const int g = blockIdx.y;
    
    if (n >= batch_size || g >= groups) return;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const int c_start = g * group_size;
    
    for (int c = c_start; c < c_start + group_size; c++) {
        for (int d = 0; d < out_D; d++) {
            for (int h = 0; h < out_H; h++) {
                for (int w = 0; w < out_W; w++) {
                    const int idx = ((n * out_channels + c) * out_D + d) * out_H * out_W + h * out_W + w;
                    const float val = input[idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }
        }
    }
    
    mean[n * groups + g] = sum / group_elements;
    variance[n * groups + g] = (sum_sq / group_elements) - (mean[n * groups + g] * mean[n * groups + g]);
}

__global__ void apply_group_norm_kernel(
    const float* input, float* output, const float* mean, const float* inv_std,
    int batch_size, int out_channels, int groups,
    int out_D, int out_H, int out_W) {
    
    const int group_size = out_channels / groups;
    const int spatial_size = out_D * out_H * out_W;
    
    const int n = blockIdx.z;
    const int g = blockIdx.y;
    const int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n >= batch_size || g >= groups || spatial_idx >= spatial_size) return;
    
    const int c_start = g * group_size;
    const int d = spatial_idx / (out_H * out_W);
    const int h = (spatial_idx % (out_H * out_W)) / out_W;
    const int w = spatial_idx % out_W;
    
    const float group_mean = mean[n * groups + g];
    const float group_inv_std = inv_std[n * groups + g];
    
    for (int c = c_start; c < c_start + group_size; c++) {
        const int idx = ((n * out_channels + c) * out_D + d) * out_H * out_W + h * out_W + w;
        output[idx] = (input[idx] - group_mean) * group_inv_std;
    }
}

torch::Tensor fused_conv_transpose_relu_gn_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int groups, float eps) {
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto D = input.size(2);
    const auto H = input.size(3);
    const auto W = input.size(4);
    const auto out_channels = weight.size(0);
    
    const int out_D = D + kernel_size - 1;
    const int out_H = H + kernel_size - 1;
    const int out_W = W + kernel_size - 1;
    const int spatial_size = out_D * out_H * out_W;
    
    auto output = torch::zeros({batch_size, out_channels, out_D, out_H, out_W}, input.options());
    
    // Launch conv transpose + ReLU kernel
    const int conv_threads = 256;
    const int conv_blocks = (spatial_size + conv_threads - 1) / conv_threads;
    dim3 conv_grid(conv_blocks, groups, batch_size);
    
    fused_conv_transpose_relu_gn_kernel<<<conv_grid, conv_threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        D, H, W, kernel_size, groups,
        out_D, out_H, out_W);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate mean and variance for group normalization
    auto mean = torch::zeros({batch_size, groups}, input.options());
    auto variance = torch::zeros({batch_size, groups}, input.options());
    auto inv_std = torch::zeros({batch_size, groups}, input.options());
    
    dim3 stats_grid(batch_size, groups);
    compute_group_stats_kernel<<<stats_grid, 1>>>(
        output.data_ptr<float>(), mean.data_ptr<float>(), variance.data_ptr<float>(),
        batch_size, out_channels, groups, out_D, out_H, out_W);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Compute inverse standard deviation
    for (int n = 0; n < batch_size; n++) {
        for (int g = 0; g < groups; g++) {
            inv_std.data_ptr<float>()[n * groups + g] = 1.0f / sqrtf(variance.data_ptr<float>()[n * groups + g] + eps);
        }
    }
    
    // Apply group normalization
    const int norm_threads = 256;
    const int norm_blocks = (spatial_size + norm_threads - 1) / norm_threads;
    dim3 norm_grid(norm_blocks, groups, batch_size);
    
    apply_group_norm_kernel<<<norm_grid, norm_threads>>>(
        output.data_ptr<float>(), output.data_ptr<float>(),
        mean.data_ptr<float>(), inv_std.data_ptr<float>(),
        batch_size, out_channels, groups, out_D, out_H, out_W);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return output;
}
"""

fused_conv_transpose_relu_gn_cpp_source = """
torch::Tensor fused_conv_transpose_relu_gn_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int groups, float eps);
"""

# Compile the inline CUDA code
fused_conv_transpose_relu_gn = load_inline(
    name="fused_conv_transpose_relu_gn",
    cpp_sources=fused_conv_transpose_relu_gn_cpp_source,
    cuda_sources=fused_conv_transpose_relu_gn_source,
    functions=["fused_conv_transpose_relu_gn_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = 1e-5
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using Kaiming initialization for ReLU
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        
        self.fused_op = fused_conv_transpose_relu_gn

    def forward(self, x):
        return self.fused_op.fused_conv_transpose_relu_gn_cuda(
            x, self.weight, self.bias, self.kernel_size, self.groups, self.eps
        )