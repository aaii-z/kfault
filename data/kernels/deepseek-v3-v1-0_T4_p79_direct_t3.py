import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_mult_norm_clamp_max_kernel(
    const float* input, const float* weight, const float* bias, const float* multiplier,
    float* output, int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width, int kernel_size, 
    int output_depth, int output_height, int output_width, float clamp_min, float clamp_max) {
    
    const int d = blockIdx.z;
    const int h = blockIdx.y;
    const int w = blockIdx.x;
    
    if (d >= output_depth || h >= output_height || w >= output_width) return;
    
    const int kernel_radius = kernel_size / 2;
    const int spatial_size = output_depth * output_height * output_width;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float conv_result = 0.0f;
            
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kd = -kernel_radius; kd <= kernel_radius; ++kd) {
                    for (int kh = -kernel_radius; kh <= kernel_radius; ++kh) {
                        for (int kw = -kernel_radius; kw <= kernel_radius; ++kw) {
                            int input_d = d + kd;
                            int input_h = h + kh;
                            int input_w = w + kw;
                            
                            if (input_d >= 0 && input_d < input_depth &&
                                input_h >= 0 && input_h < input_height &&
                                input_w >= 0 && input_w < input_width) {
                                
                                int input_idx = ((b * in_channels + ic) * input_depth + input_d) * input_height * input_width +
                                              input_h * input_width + input_w;
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
            float multiplied = conv_result * multiplier[oc];
            
            int output_idx = ((b * out_channels + oc) * output_depth + d) * output_height * output_width + h * output_width + w;
            output[output_idx] = multiplied;
        }
    }
}

__global__ void instance_norm_clamp_mult_kernel(
    float* data, const float* multiplier, int batch_size, int out_channels,
    int depth, int height, int width, float clamp_min, float clamp_max) {
    
    const int b = blockIdx.x;
    const int oc = blockIdx.y;
    
    if (b >= batch_size || oc >= out_channels) return;
    
    const int spatial_size = depth * height * width;
    const int base_idx = (b * out_channels + oc) * spatial_size;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = 0; i < spatial_size; ++i) {
        float val = data[base_idx + i];
        sum += val;
        sum_sq += val * val;
    }
    
    float mean = sum / spatial_size;
    float variance = (sum_sq / spatial_size) - (mean * mean);
    float inv_std = 1.0f / sqrtf(variance + 1e-5f);
    
    for (int i = 0; i < spatial_size; ++i) {
        float normalized = (data[base_idx + i] - mean) * inv_std;
        float clamped = fminf(fmaxf(normalized, clamp_min), clamp_max);
        data[base_idx + i] = clamped * multiplier[oc];
    }
}

__global__ void max_reduction_kernel(
    const float* input, float* output, int batch_size, int out_channels,
    int depth, int height, int width) {
    
    const int b = blockIdx.z;
    const int d = blockIdx.y;
    const int h = blockIdx.x;
    const int w = threadIdx.x;
    
    if (b >= batch_size || d >= depth || h >= height || w >= width) return;
    
    float max_val = -FLT_MAX;
    for (int oc = 0; oc < out_channels; ++oc) {
        int idx = ((b * out_channels + oc) * depth + d) * height * width + h * width + w;
        max_val = fmaxf(max_val, input[idx]);
    }
    
    int output_idx = (b * depth + d) * height * width + h * width + w;
    output[output_idx] = max_val;
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
    const auto input_depth = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);
    const auto out_channels = weight.size(0);
    
    const int output_depth = input_depth - kernel_size + 1;
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;
    
    auto intermediate = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, input.options());
    auto output = torch::empty({batch_size, output_depth, output_height, output_width}, input.options());
    
    dim3 conv_grid(output_width, output_height, output_depth);
    fused_conv_mult_norm_clamp_max_kernel<<<conv_grid, 1>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), multiplier.data_ptr<float>(),
        intermediate.data_ptr<float>(), batch_size, in_channels, out_channels,
        input_depth, input_height, input_width, kernel_size,
        output_depth, output_height, output_width, clamp_min, clamp_max);
    
    dim3 norm_grid(batch_size, out_channels);
    instance_norm_clamp_mult_kernel<<<norm_grid, 1>>>(
        intermediate.data_ptr<float>(), multiplier.data_ptr<float>(), batch_size, out_channels,
        output_depth, output_height, output_width, clamp_min, clamp_max);
    
    dim3 max_grid(batch_size, output_depth, output_height);
    max_reduction_kernel<<<max_grid, output_width>>>(
        intermediate.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_channels,
        output_depth, output_height, output_width);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor multiplier,
    int kernel_size, float clamp_min, float clamp_max);
"""

fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias, self.multiplier, 
            self.kernel_size, self.clamp_min, self.clamp_max
        )