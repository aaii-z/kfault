import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3d + ReLU + GroupNorm
fused_conv_transpose_relu_gn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cub/cub.cuh>

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at %s:%d\\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void fused_conv_transpose_relu_gn_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* workspace,
    int batch_size, int in_channels, int out_channels,
    int D, int H, int W, int kernel_size, int groups,
    int out_D, int out_H, int out_W,
    float eps) {
    
    extern __shared__ float shared_mem[];
    
    const int group_size = out_channels / groups;
    const int spatial_size = out_D * out_H * out_W;
    const int group_spatial_size = spatial_size * group_size;
    
    // Calculate thread indices
    const int n = blockIdx.x;
    const int g = blockIdx.y;
    const int c_start = g * group_size;
    const int c_end = c_start + group_size;
    
    // First phase: ConvTranspose3d + ReLU
    for (int c = c_start + threadIdx.x; c < c_end; c += blockDim.x) {
        for (int d = 0; d < out_D; d++) {
            for (int h = 0; h < out_H; h++) {
                for (int w = 0; w < out_W; w++) {
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
                    
                    if (bias) {
                        sum += bias[c];
                    }
                    
                    // Apply ReLU
                    sum = fmaxf(0.0f, sum);
                    
                    const int output_idx = ((n * out_channels + c) * out_D + d) * out_H * out_W + h * out_W + w;
                    output[output_idx] = sum;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Second phase: GroupNorm
    // Calculate mean and variance for this group
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int c = c_start; c < c_end; c++) {
            for (int d = 0; d < out_D; d++) {
                for (int h = 0; h < out_H; h++) {
                    for (int w = 0; w < out_W; w++) {
                        const int idx = ((n * out_channels + c) * out_D + d) * out_H * out_W + h * out_W + w;
                        const float val = output[idx];
                        sum += val;
                        sum_sq += val * val;
                    }
                }
            }
        }
        
        const int num_elements = group_size * spatial_size;
        const float mean = sum / num_elements;
        const float variance = (sum_sq / num_elements) - (mean * mean);
        const float inv_std = 1.0f / sqrtf(variance + eps);
        
        // Store mean and inv_std for this group
        const int group_idx = n * groups * 2 + g * 2;
        workspace[group_idx] = mean;
        workspace[group_idx + 1] = inv_std;
    }
    
    __syncthreads();
    
    // Apply normalization
    const int group_idx = n * groups * 2 + g * 2;
    const float mean = workspace[group_idx];
    const float inv_std = workspace[group_idx + 1];
    
    for (int c = c_start + threadIdx.x; c < c_end; c += blockDim.x) {
        for (int d = 0; d < out_D; d++) {
            for (int h = 0; h < out_H; h++) {
                for (int w = 0; w < out_W; w++) {
                    const int idx = ((n * out_channels + c) * out_D + d) * out_H * out_W + h * out_W + w;
                    output[idx] = (output[idx] - mean) * inv_std;
                }
            }
        }
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
    const auto out_channels = weight.size(1);
    
    const int out_D = D + kernel_size - 1;
    const int out_H = H + kernel_size - 1;
    const int out_W = W + kernel_size - 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_D, out_H, out_W}, input.options());
    
    // Allocate workspace for mean and inv_std
    auto workspace = torch::zeros({batch_size * groups * 2}, input.options());
    
    const int threads_per_block = 256;
    dim3 grid(batch_size, groups);
    
    const size_t shared_mem_size = 0; // No dynamic shared memory needed
    
    fused_conv_transpose_relu_gn_kernel<<<grid, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), workspace.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        D, H, W, kernel_size, groups,
        out_D, out_H, out_W, eps);
    
    CUDA_CHECK(cudaGetLastError());
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
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcub"],
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
            in_channels, out_channels, kernel_size, kernel_size, kernel_size
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