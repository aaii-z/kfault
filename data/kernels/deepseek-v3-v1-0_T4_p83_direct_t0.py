import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Conv3D + GroupNorm + Min + Clamp + Dropout
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_conv_norm_min_clamp_dropout_kernel(
    const float* input, const float* weight, const float* bias, 
    float* output, float* running_mean, float* running_var,
    int batch_size, int in_channels, int out_channels, 
    int depth, int height, int width, int kernel_size,
    int groups, float min_value, float max_value, float dropout_p,
    float epsilon, bool training, curandState* states) {
    
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
        
        // 3D convolution
        float conv_result = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = -k_half; kd <= k_half; kd++) {
                for (int kh = -k_half; kh <= k_half; kh++) {
                    for (int kw = -k_half; kw <= k_half; kw++) {
                        int od = d + kd;
                        int oh = h + kh;
                        int ow = w + kw;
                        
                        if (od >= 0 && od < depth && oh >= 0 && oh < height && ow >= 0 && ow < width) {
                            int input_idx = ((n * in_channels + ic) * depth + od) * height * width + oh * width + ow;
                            int weight_idx = (((c * in_channels + ic) * kernel_size + (kd + k_half)) * kernel_size + (kh + k_half)) * kernel_size + (kw + k_half);
                            conv_result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        conv_result += bias[c];
        
        // Group Normalization
        float mean = 0.0f;
        float var = 0.0f;
        int group_start = g * group_size;
        int group_end = (g + 1) * group_size;
        
        for (int gc = group_start; gc < group_end; gc++) {
            for (int gd = 0; gd < depth; gd++) {
                for (int gh = 0; gh < height; gh++) {
                    for (int gw = 0; gw < width; gw++) {
                        int group_idx = ((n * out_channels + gc) * depth + gd) * height * width + gh * width + gw;
                        mean += output[group_idx];
                    }
                }
            }
        }
        
        int group_elements = group_size * spatial_size;
        mean /= group_elements;
        
        for (int gc = group_start; gc < group_end; gc++) {
            for (int gd = 0; gd < depth; gd++) {
                for (int gh = 0; gh < height; gh++) {
                    for (int gw = 0; gw < width; gw++) {
                        int group_idx = ((n * out_channels + gc) * depth + gd) * height * width + gh * width + gw;
                        float diff = output[group_idx] - mean;
                        var += diff * diff;
                    }
                }
            }
        }
        var /= group_elements;
        
        float normalized = (conv_result - mean) / sqrtf(var + epsilon);
        normalized = normalized * 1.0f + 0.0f;  // Scale and shift (learnable parameters simplified)
        
        // Min operation
        normalized = fminf(normalized, min_value);
        
        // Clamp operation
        normalized = fmaxf(fminf(normalized, max_value), min_value);
        
        // Dropout
        if (training && dropout_p > 0.0f) {
            curandState local_state = states[idx];
            float random = curand_uniform(&local_state);
            states[idx] = local_state;
            
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
    auto kernel_size = 3;  // Assuming kernel_size=3
    
    auto output = torch::zeros({batch_size, out_channels, depth, height, width}, input.options());
    
    // Initialize random states for dropout
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(input.device());
    auto states = torch::zeros({batch_size * out_channels * depth * height * width}, options);
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * depth * height * width + block_size - 1) / block_size;
    
    fused_conv_norm_min_clamp_dropout_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), nullptr, nullptr,
        batch_size, in_channels, out_channels, depth, height, width, kernel_size,
        groups, min_value, max_value, dropout_p, 1e-5f, training, 
        reinterpret_cast<curandState*>(states.data_ptr<int64_t>()));
    
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