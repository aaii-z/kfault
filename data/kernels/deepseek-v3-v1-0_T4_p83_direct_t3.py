import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Conv3D + GroupNorm + Min + Clamp + Dropout
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <vector>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void compute_group_stats_kernel(
    const float* input, float* means, float* variances,
    int batch_size, int out_channels, int spatial_size, int groups) {
    
    const int group_size = out_channels / groups;
    const int group_elements = group_size * spatial_size;
    const int total_groups = batch_size * groups;
    
    CUDA_KERNEL_LOOP(idx, total_groups) {
        int n = idx / groups;
        int g = idx % groups;
        
        float mean = 0.0f;
        for (int c = g * group_size; c < (g + 1) * group_size; c++) {
            for (int s = 0; s < spatial_size; s++) {
                int input_idx = (n * out_channels + c) * spatial_size + s;
                mean += input[input_idx];
            }
        }
        mean /= group_elements;
        
        float var = 0.0f;
        for (int c = g * group_size; c < (g + 1) * group_size; c++) {
            for (int s = 0; s < spatial_size; s++) {
                int input_idx = (n * out_channels + c) * spatial_size + s;
                float diff = input[input_idx] - mean;
                var += diff * diff;
            }
        }
        var /= group_elements;
        
        means[idx] = mean;
        variances[idx] = var;
    }
}

__global__ void fused_conv_norm_min_clamp_dropout_kernel(
    const float* input, const float* weight, const float* bias, 
    const float* means, const float* variances,
    float* output, int batch_size, int in_channels, int out_channels, 
    int depth, int height, int width, int kernel_size,
    int groups, float min_value, float max_value, float dropout_p,
    float epsilon, bool training, unsigned long long seed) {
    
    const int spatial_size = depth * height * width;
    const int group_size = out_channels / groups;
    const int k_half = kernel_size / 2;
    const int total_elements = batch_size * out_channels * spatial_size;
    
    CUDA_KERNEL_LOOP(idx, total_elements) {
        int n = idx / (out_channels * spatial_size);
        int c = (idx % (out_channels * spatial_size)) / spatial_size;
        int s = idx % spatial_size;
        
        int d = s / (height * width);
        int h = (s % (height * width)) / width;
        int w = s % width;
        
        int g = c / group_size;
        int group_idx = n * groups + g;
        float mean = means[group_idx];
        float var = variances[group_idx];
        
        // 3D convolution with optimized memory access
        float conv_result = bias[c];
        for (int kd = 0; kd < kernel_size; kd++) {
            int od = d + kd - k_half;
            if (od < 0 || od >= depth) continue;
            
            for (int kh = 0; kh < kernel_size; kh++) {
                int oh = h + kh - k_half;
                if (oh < 0 || oh >= height) continue;
                
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ow = w + kw - k_half;
                    if (ow < 0 || ow >= width) continue;
                    
                    for (int ic = 0; ic < in_channels; ic++) {
                        int input_idx = ((n * in_channels + ic) * depth + od) * height * width + oh * width + ow;
                        int weight_idx = (((c * in_channels + ic) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Apply Group Normalization
        float normalized = (conv_result - mean) / sqrtf(var + epsilon);
        
        // Min operation
        normalized = fminf(normalized, min_value);
        
        // Clamp operation
        normalized = fmaxf(fminf(normalized, max_value), min_value);
        
        // Dropout with thread-local random state
        if (training && dropout_p > 0.0f) {
            curandState local_state;
            curand_init(seed + idx, 0, 0, &local_state);
            float random = curand_uniform(&local_state);
            
            if (random < dropout_p) {
                normalized = 0.0f;
            } else {
                normalized /= (1.0f - dropout_p);
            }
        }
        
        output[idx] = normalized;
    }
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int groups, float min_value, float max_value, float dropout_p,
    bool training) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    auto spatial_size = depth * height * width;
    
    auto output = torch::zeros({batch_size, out_channels, depth, height, width}, input.options());
    
    // Precompute group statistics
    auto means = torch::zeros({batch_size, groups}, input.options());
    auto variances = torch::zeros({batch_size, groups}, input.options());
    
    const int block_size = 256;
    int num_blocks;
    
    // Compute group statistics
    num_blocks = (batch_size * groups + block_size - 1) / block_size;
    compute_group_stats_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), means.data_ptr<float>(), variances.data_ptr<float>(),
        batch_size, out_channels, spatial_size, groups);
    
    // Main fused operation
    num_blocks = (batch_size * out_channels * spatial_size + block_size - 1) / block_size;
    unsigned long long seed = time(NULL);
    
    fused_conv_norm_min_clamp_dropout_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        means.data_ptr<float>(), variances.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels, 
        depth, height, width, kernel_size, groups, min_value, max_value, 
        dropout_p, 1e-5f, training, seed);
    
    return output;
}
"""

fused_operator_cpp_source = "torch::Tensor fused_operator_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int groups, float min_value, float max_value, float dropout_p, bool training);"

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcurand"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        
        # Initialize weights and bias for convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias, self.groups, 
            self.min_value, self.max_value, self.dropout_p, self.training
        )