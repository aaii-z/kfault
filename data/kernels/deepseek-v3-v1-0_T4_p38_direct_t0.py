import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3d + Clamp + Softmax + Scale
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void fused_conv_transpose_clamp_softmax_scale_kernel(
    const float* input, const float* weight, const float* bias, 
    float* output, const float* scale,
    int batch_size, int in_channels, int out_channels,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    float clamp_min, float clamp_max) {
    
    // Calculate output spatial dimensions
    const int output_spatial_size = output_d * output_h * output_w;
    const int n = blockIdx.x;
    const int c = blockIdx.y;
    const int spatial_idx = threadIdx.x + blockIdx.z * blockDim.x;
    
    if (n >= batch_size || c >= out_channels || spatial_idx >= output_spatial_size) {
        return;
    }
    
    // Convert flat spatial index to 3D coordinates
    const int d_out = spatial_idx / (output_h * output_w);
    const int h_out = (spatial_idx % (output_h * output_w)) / output_w;
    const int w_out = spatial_idx % output_w;
    
    float result = 0.0f;
    if (bias) {
        result = bias[c];
    }
    
    // Iterate over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Iterate over kernel dimensions
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    // Calculate input coordinates
                    int d_in = d_out + padding_d - kd * stride_d;
                    int h_in = h_out + padding_h - kh * stride_h;
                    int w_in = w_out + padding_w - kw * stride_w;
                    
                    // Check if input coordinates are valid
                    if (d_in >= 0 && d_in < input_d && 
                        h_in >= 0 && h_in < input_h && 
                        w_in >= 0 && w_in < input_w) {
                        
                        // Calculate indices
                        int input_idx = ((n * in_channels + ic) * input_d + d_in) * input_h * input_w + 
                                       h_in * input_w + w_in;
                        int weight_idx = ((c * in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w + 
                                        kh * kernel_w + kw;
                        
                        result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Apply clamp
    result = fminf(fmaxf(result, clamp_min), clamp_max);
    output[(n * out_channels + c) * output_spatial_size + spatial_idx] = result;
}

__global__ void spatial_softmax_and_scale_kernel(
    float* data, const float* scale,
    int batch_size, int channels, int spatial_size) {
    
    const int n = blockIdx.x;
    const int spatial_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (n >= batch_size || spatial_idx >= spatial_size) {
        return;
    }
    
    // First pass: find max for each channel across spatial dimensions
    __shared__ float max_vals[256]; // Assuming max 256 channels
    if (threadIdx.x == 0) {
        for (int c = 0; c < channels; ++c) {
            float max_val = -FLT_MAX;
            for (int s = 0; s < spatial_size; ++s) {
                int idx = (n * channels + c) * spatial_size + s;
                max_val = fmaxf(max_val, data[idx]);
            }
            max_vals[c] = max_val;
        }
    }
    __syncthreads();
    
    // Second pass: compute exponentials and sum
    __shared__ float exp_sums[256];
    if (threadIdx.x == 0) {
        for (int c = 0; c < channels; ++c) {
            exp_sums[c] = 0.0f;
        }
    }
    __syncthreads();
    
    for (int c = 0; c < channels; ++c) {
        int idx = (n * channels + c) * spatial_size + spatial_idx;
        float val = data[idx] - max_vals[c];
        float exp_val = expf(val);
        data[idx] = exp_val;
        atomicAdd(&exp_sums[c], exp_val);
    }
    __syncthreads();
    
    // Third pass: normalize and scale
    for (int c = 0; c < channels; ++c) {
        int idx = (n * channels + c) * spatial_size + spatial_idx;
        data[idx] = (data[idx] / exp_sums[c]) * scale[c];
    }
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor scale, int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    float clamp_min, float clamp_max) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_d = input.size(2);
    auto input_h = input.size(3);
    auto input_w = input.size(4);
    
    auto out_channels = weight.size(1);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    // Calculate output dimensions
    int output_d = (input_d - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    int output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;
    
    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, 
                              torch::dtype(torch::kFloat32).device(input.device()));
    
    const int spatial_size = output_d * output_h * output_w;
    
    // Launch conv transpose + clamp kernel
    dim3 blocks_conv(batch_size, out_channels, (spatial_size + 255) / 256);
    dim3 threads_conv(256);
    
    fused_conv_transpose_clamp_softmax_scale_kernel<<<blocks_conv, threads_conv>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), scale.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        clamp_min, clamp_max);
    
    // Reshape for softmax (flatten spatial dimensions)
    auto output_reshaped = output.view({batch_size, out_channels, spatial_size});
    
    // Launch softmax + scale kernel
    dim3 blocks_softmax(batch_size, (spatial_size + 255) / 256);
    dim3 threads_softmax(256);
    
    spatial_softmax_and_scale_kernel<<<blocks_softmax, threads_softmax>>>(
        output_reshaped.data_ptr<float>(), scale.data_ptr<float>(),
        batch_size, out_channels, spatial_size);
    
    // Reshape back to original shape
    return output_reshaped.view({batch_size, out_channels, output_d, output_h, output_w});
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor scale, int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    float clamp_min, float clamp_max);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Store convolution parameters
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        
        # Load custom CUDA operator
        self.fused_operator = fused_operator

    def forward(self, x):
        x = self.avg_pool(x)
        
        # Use fused custom CUDA operator
        x = self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias, self.scale,
            self.stride, self.stride, self.stride,
            self.padding, self.padding, self.padding,
            self.output_padding, self.output_padding, self.output_padding,
            self.clamp_min, self.clamp_max
        )
        
        return x