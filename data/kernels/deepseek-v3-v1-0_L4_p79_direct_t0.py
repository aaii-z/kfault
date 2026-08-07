import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for conv, multiplication, instance norm, clamp, and max
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cub/cub.cuh>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_mult_norm_clamp_max_kernel(
    const float* input, const float* weight, const float* bias, const float* multiplier,
    float* output, int batch_size, int in_channels, int out_channels,
    int depth, int height, int width, int kernel_size, float clamp_min, float clamp_max) {
    
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (d >= depth || h >= height || w >= width) return;
    
    const int spatial_size = depth * height * width;
    const int kernel_radius = kernel_size / 2;
    
    // Shared memory for statistics calculation
    __shared__ float shared_mean[256];
    __shared__ float shared_variance[256];
    __shared__ int shared_count[256];
    
    // First pass: convolution and multiplication
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float conv_result = 0.0f;
            
            // 3D convolution
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kd = -kernel_radius; kd <= kernel_radius; ++kd) {
                    for (int kh = -kernel_radius; kh <= kernel_radius; ++kh) {
                        for (int kw = -kernel_radius; kw <= kernel_radius; ++kw) {
                            int input_d = d + kd;
                            int input_h = h + kh;
                            int input_w = w + kw;
                            
                            if (input_d >= 0 && input_d < depth &&
                                input_h >= 0 && input_h < height &&
                                input_w >= 0 && input_w < width) {
                                
                                int input_idx = ((b * in_channels + ic) * depth + input_d) * height * width +
                                              input_h * width + input_w;
                                int weight_idx = ((oc * in_channels + ic) * kernel_size + (kd + kernel_radius)) *
                                               kernel_size * kernel_size +
                                               (kh + kernel_radius) * kernel_size + (kw + kernel_radius);
                                
                                conv_result += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            
            conv_result += bias[oc];
            
            // Multiply by multiplier
            float multiplied = conv_result * multiplier[oc];
            
            // Store intermediate result
            int output_idx = ((b * out_channels + oc) * depth + d) * height * width + h * width + w;
            output[output_idx] = multiplied;
        }
    }
    
    __syncthreads();
    
    // Second pass: instance normalization and clamping
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            // Calculate mean and variance for this instance and channel
            float sum = 0.0f;
            float sum_sq = 0.0f;
            int count = 0;
            
            for (int d2 = 0; d2 < depth; ++d2) {
                for (int h2 = 0; h2 < height; ++h2) {
                    for (int w2 = 0; w2 < width; ++w2) {
                        int idx = ((b * out_channels + oc) * depth + d2) * height * width + h2 * width + w2;
                        float val = output[idx];
                        sum += val;
                        sum_sq += val * val;
                        count++;
                    }
                }
            }
            
            float mean = sum / count;
            float variance = (sum_sq / count) - (mean * mean);
            float inv_std = 1.0f / sqrtf(variance + 1e-5f);
            
            // Apply normalization, clamp, and final multiplication
            for (int d2 = 0; d2 < depth; ++d2) {
                for (int h2 = 0; h2 < height; ++h2) {
                    for (int w2 = 0; w2 < width; ++w2) {
                        int idx = ((b * out_channels + oc) * depth + d2) * height * width + h2 * width + w2;
                        float normalized = (output[idx] - mean) * inv_std;
                        float clamped = fminf(fmaxf(normalized, clamp_min), clamp_max);
                        output[idx] = clamped * multiplier[oc];
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Third pass: max reduction across channels
    for (int b = 0; b < batch_size; ++b) {
        for (int d2 = 0; d2 < depth; ++d2) {
            for (int h2 = 0; h2 < height; ++h2) {
                for (int w2 = 0; w2 < width; ++w2) {
                    float max_val = -FLT_MAX;
                    
                    for (int oc = 0; oc < out_channels; ++oc) {
                        int idx = ((b * out_channels + oc) * depth + d2) * height * width + h2 * width + w2;
                        max_val = fmaxf(max_val, output[idx]);
                    }
                    
                    // Store final result
                    int final_idx = (b * depth + d2) * height * width + h2 * width + w2;
                    output[final_idx] = max_val;
                }
            }
        }
    }
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor multiplier,
    int kernel_size, float clamp_min, float clamp_max) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(multiplier);
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto depth = input.size(2);
    const auto height = input.size(3);
    const auto width = input.size(4);
    const auto out_channels = weight.size(0);
    
    auto output = torch::empty({batch_size, depth, height, width}, input.options());
    
    dim3 block_size(4, 4, 4);
    dim3 grid_size(
        (depth + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y,
        (width + block_size.z - 1) / block_size.z
    );
    
    fused_conv_mult_norm_clamp_max_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), multiplier.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        depth, height, width, kernel_size, clamp_min, clamp_max);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor multiplier,
    int kernel_size, float clamp_min, float clamp_max);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcub"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Initialize convolution parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias, self.multiplier, 
            self.kernel_size, self.clamp_min, self.clamp_max
        )