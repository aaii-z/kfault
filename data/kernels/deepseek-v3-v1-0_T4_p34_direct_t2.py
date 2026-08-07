import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused CUDA kernel for ConvTranspose3D + LayerNorm + GELU + Scaling
fused_conv_transpose_ln_gelu_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_1D_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

constexpr int kBlockSize = 256;

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_ln_gelu_scale_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, const float* gamma, const float* beta,
    int batch_size, int in_channels, int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    float eps, float scaling_factor) {
    
    const int output_size = batch_size * out_channels * output_d * output_h * output_w;
    const int spatial_size = output_d * output_h * output_w;
    const int channel_spatial_size = out_channels * spatial_size;
    
    // First kernel: transposed convolution
    CUDA_1D_KERNEL_LOOP(idx, output_size) {
        const int b = idx / channel_spatial_size;
        const int c = (idx % channel_spatial_size) / spatial_size;
        const int d = (idx % spatial_size) / (output_h * output_w);
        const int h = (idx % (output_h * output_w)) / output_w;
        const int w = idx % output_w;
        
        float sum = 0.0f;
        
        // Perform transposed convolution
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_d; ++kd) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int input_d_pos = d - kd + padding_d;
                        int input_h_pos = h - kh + padding_h;
                        int input_w_pos = w - kw + padding_w;
                        
                        if (input_d_pos >= 0 && input_d_pos < input_d * stride_d &&
                            input_h_pos >= 0 && input_h_pos < input_h * stride_h &&
                            input_w_pos >= 0 && input_w_pos < input_w * stride_w) {
                            
                            if (input_d_pos % stride_d == 0 && input_h_pos % stride_h == 0 && input_w_pos % stride_w == 0) {
                                input_d_pos /= stride_d;
                                input_h_pos /= stride_h;
                                input_w_pos /= stride_w;
                                
                                if (input_d_pos < input_d && input_h_pos < input_h && input_w_pos < input_w) {
                                    int input_idx = ((b * in_channels + ic) * input_d + input_d_pos) * input_h * input_w +
                                                  input_h_pos * input_w + input_w_pos;
                                    
                                    int weight_idx = ((ic * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                                    weight_idx = weight_idx * out_channels + c;
                                    
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Add bias if present
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        output[idx] = sum;
    }
    
    __syncthreads();
    
    // Second kernel: layer normalization, GELU, and scaling
    CUDA_1D_KERNEL_LOOP(idx, batch_size * spatial_size) {
        const int b = idx / spatial_size;
        const int spatial_idx = idx % spatial_size;
        const int channel_start = b * channel_spatial_size;
        const int spatial_offset = channel_start + spatial_idx;
        
        // Calculate mean and variance for this spatial position across all channels
        float mean = 0.0f;
        float variance = 0.0f;
        
        for (int c = 0; c < out_channels; ++c) {
            int current_idx = spatial_offset + c * spatial_size;
            mean += output[current_idx];
        }
        mean /= out_channels;
        
        for (int c = 0; c < out_channels; ++c) {
            int current_idx = spatial_offset + c * spatial_size;
            float diff = output[current_idx] - mean;
            variance += diff * diff;
        }
        variance = variance / out_channels + eps;
        float inv_std = 1.0f / sqrtf(variance);
        
        // Apply layer normalization, GELU, and scaling
        for (int c = 0; c < out_channels; ++c) {
            int current_idx = spatial_offset + c * spatial_size;
            float normalized = (output[current_idx] - mean) * inv_std;
            if (gamma != nullptr && beta != nullptr) {
                normalized = normalized * gamma[c] + beta[c];
            }
            output[current_idx] = gelu(normalized) * scaling_factor;
        }
    }
}

torch::Tensor fused_conv_transpose_ln_gelu_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    float eps, float scaling_factor) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_d = input.size(2);
    auto input_h = input.size(3);
    auto input_w = input.size(4);
    auto out_channels = weight.size(0);
    
    int output_d = (input_d - 1) * stride_d + kernel_d - 2 * padding_d;
    int output_h = (input_h - 1) * stride_h + kernel_h - 2 * padding_h;
    int output_w = (input_w - 1) * stride_w + kernel_w - 2 * padding_w;
    
    auto output = torch::empty({batch_size, out_channels, output_d, output_h, output_w}, input.options());
    
    const int output_size = batch_size * out_channels * output_d * output_h * output_w;
    const int num_blocks_conv = (output_size + kBlockSize - 1) / kBlockSize;
    const int num_blocks_ln = (batch_size * output_d * output_h * output_w + kBlockSize - 1) / kBlockSize;
    
    fused_conv_transpose_ln_gelu_scale_kernel<<<num_blocks_conv, kBlockSize>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        gamma.defined() ? gamma.data_ptr<float>() : nullptr,
        beta.defined() ? beta.data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        eps, scaling_factor);
    
    return output;
}
"""

fused_conv_transpose_ln_gelu_scale_cpp_source = """
torch::Tensor fused_conv_transpose_ln_gelu_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    float eps, float scaling_factor);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_conv_transpose_ln_gelu_scale",
    cpp_sources=fused_conv_transpose_ln_gelu_scale_cpp_source,
    cuda_sources=fused_conv_transpose_ln_gelu_scale_source,
    functions=["fused_conv_transpose_ln_gelu_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.eps = eps
        self.scaling_factor = scaling_factor
        
        # Initialize weights and biases (note: weight shape is [in_channels, out_channels, kd, kh, kw])
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
        # LayerNorm parameters
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_conv_transpose_ln_gelu_scale_cuda(
            x, self.weight, self.bias, self.gamma, self.beta,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.eps, self.scaling_factor
        )