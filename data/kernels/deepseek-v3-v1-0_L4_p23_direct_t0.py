import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Conv3D + GroupNorm + Mean custom CUDA kernel
fused_conv_gn_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cub/cub.cuh>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_gn_mean_forward_kernel(
    const float* input, const float* weight, const float* bias,
    const float* gamma, const float* beta, float* output,
    int batch_size, int in_channels, int out_channels,
    int D, int H, int W, int kernel_size, int num_groups,
    int groups_per_block, float eps) {
    
    extern __shared__ float shared_mem[];
    
    const int group_size = out_channels / num_groups;
    const int spatial_size = D * H * W;
    const int total_elements = batch_size * out_channels * spatial_size;
    
    // Block handles multiple groups
    const int group_idx = blockIdx.x / groups_per_block;
    const int batch_idx = blockIdx.x % groups_per_block;
    
    if (group_idx >= num_groups || batch_idx >= batch_size) return;
    
    const int group_start = group_idx * group_size;
    const int group_end = group_start + group_size;
    
    float* group_means = shared_mem;
    float* group_vars = shared_mem + group_size;
    float* temp_storage = shared_mem + 2 * group_size;
    
    // Initialize shared memory
    if (threadIdx.x < group_size) {
        group_means[threadIdx.x] = 0.0f;
        group_vars[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // Compute mean and variance for each channel in the group
    for (int c = group_start; c < group_end; c++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int count = 0;
        
        for (int d = 0; d < D; d++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    // Simplified convolution (for demonstration)
                    float conv_val = 0.0f;
                    for (int kd = 0; kd < kernel_size; kd++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int in_d = d + kd - kernel_size/2;
                                int in_h = h + kh - kernel_size/2;
                                int in_w = w + kw - kernel_size/2;
                                
                                if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                                    for (int in_c = 0; in_c < in_channels; in_c++) {
                                        int input_idx = ((batch_idx * in_channels + in_c) * D + in_d) * H * W + in_h * W + in_w;
                                        int weight_idx = ((c * in_channels + in_c) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                                        conv_val += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                    conv_val += bias[c];
                    
                    sum += conv_val;
                    sum_sq += conv_val * conv_val;
                    count++;
                }
            }
        }
        
        float mean = sum / count;
        float var = (sum_sq / count) - (mean * mean);
        
        if (threadIdx.x == c - group_start) {
            group_means[threadIdx.x] = mean;
            group_vars[threadIdx.x] = var;
        }
    }
    __syncthreads();
    
    // Apply group normalization and compute mean
    float channel_sum = 0.0f;
    int channel_idx = group_start + threadIdx.x;
    
    if (channel_idx < group_end) {
        float mean = group_means[threadIdx.x];
        float var = group_vars[threadIdx.x];
        float inv_std = 1.0f / sqrtf(var + eps);
        
        for (int d = 0; d < D; d++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    // Simplified convolution (same as above)
                    float conv_val = 0.0f;
                    for (int kd = 0; kd < kernel_size; kd++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int in_d = d + kd - kernel_size/2;
                                int in_h = h + kh - kernel_size/2;
                                int in_w = w + kw - kernel_size/2;
                                
                                if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                                    for (int in_c = 0; in_c < in_channels; in_c++) {
                                        int input_idx = ((batch_idx * in_channels + in_c) * D + in_d) * H * W + in_h * W + in_w;
                                        int weight_idx = ((channel_idx * in_channels + in_c) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                                        conv_val += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                    conv_val += bias[channel_idx];
                    
                    // Group normalization
                    float normalized = (conv_val - mean) * inv_std;
                    float scaled = normalized * gamma[channel_idx] + beta[channel_idx];
                    
                    channel_sum += scaled;
                }
            }
        }
    }
    
    // Reduce within warp
    for (int offset = 16; offset > 0; offset /= 2) {
        channel_sum += __shfl_down_sync(0xffffffff, channel_sum, offset);
    }
    
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&output[batch_idx], channel_sum / (spatial_size * group_size));
    }
}

torch::Tensor fused_conv_gn_mean_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int kernel_size, int num_groups, float eps = 1e-5) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size}, input.options());
    
    const int groups_per_block = 4;
    const int num_blocks = batch_size * num_groups;
    const int threads_per_block = 256;
    
    size_t shared_mem_size = 2 * (out_channels / num_groups) * sizeof(float) + 1024;
    
    fused_conv_gn_mean_forward_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, D, H, W, kernel_size, num_groups,
        groups_per_block, eps);
    
    return output;
}
"""

fused_conv_gn_mean_cpp_source = """
torch::Tensor fused_conv_gn_mean_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int kernel_size, int num_groups, float eps = 1e-5);
"""

# Compile the inline CUDA code
fused_conv_gn_mean = load_inline(
    name="fused_conv_gn_mean",
    cpp_sources=fused_conv_gn_mean_cpp_source,
    cuda_sources=fused_conv_gn_mean_source,
    functions=["fused_conv_gn_mean_forward"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas", "-lcudnn"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        
        # Initialize parameters manually
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.gamma, 1)
        nn.init.constant_(self.beta, 0)
        
        self.fused_op = fused_conv_gn_mean

    def forward(self, x):
        return self.fused_op.fused_conv_gn_mean_forward(
            x, self.weight, self.bias, self.gamma, self.beta,
            self.kernel_size, self.num_groups
        )