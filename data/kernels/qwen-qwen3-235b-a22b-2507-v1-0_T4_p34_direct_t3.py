import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 3D transposed convolution, layer norm, GELU, and scaling
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU activation using tanh approximation
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

// Fast inverse square root
__device__ float rsqrt(float x) {
    return rsqrtf(x);
}

// Fused kernel: 3D transposed convolution + layer norm + GELU + scale
// Each block handles one output spatial location and multiple channels
__global__ void fused_convtrans3d_layernorm_gelu_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_D, int input_H, int input_W,
    int output_D, int output_H, int output_W,
    int kernel_size,
    int stride,
    int padding,
    float eps,
    float scaling_factor
) {
    int spatial_idx = blockIdx.x;
    int c = blockIdx.y;
    int batch = blockIdx.z;

    if (spatial_idx >= output_D * output_H * output_W || c >= out_channels || batch >= batch_size) return;

    int wo = spatial_idx % output_W;
    int ho = (spatial_idx / output_W) % output_H;
    int do_ = spatial_idx / (output_W * output_H);

    // Compute transposed convolution
    float conv_val = 0.0f;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int di = do_ * stride - padding + kd;
                int hi = ho * stride - padding + kh;
                int wi = wo * stride - padding + kw;

                if (di >= 0 && di < input_D && hi >= 0 && hi < input_H && wi >= 0 && wi < input_W) {
                    for (int ci = 0; ci < in_channels; ++ci) {
                        int input_idx = batch * in_channels * input_D * input_H * input_W +
                                        ci * input_D * input_H * input_W +
                                        di * input_H * input_W + hi * input_W + wi;
                        int weight_idx = c * in_channels * kernel_size * kernel_size * kernel_size +
                                         ci * kernel_size * kernel_size * kernel_size +
                                         kd * kernel_size * kernel_size + kh * kernel_size + kw;
                        conv_val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        conv_val += bias[c];
    }

    int output_idx = batch * out_channels * output_D * output_H * output_W +
                     c * output_D * output_H * output_W +
                     do_ * output_H * output_W + ho * output_W + wo;

    // Store conv result temporarily
    output[output_idx] = conv_val;

    // Synchronize to ensure all channels have written their conv results for this spatial location
    __syncthreads();

    // Compute mean over channels for layer norm
    float mean = 0.0f;
    for (int cc = 0; cc < out_channels; ++cc) {
        int idx = batch * out_channels * output_D * output_H * output_W +
                  cc * output_D * output_H * output_W +
                  do_ * output_H * output_W + ho * output_W + wo;
        mean += output[idx];
    }
    mean /= out_channels;

    // Compute variance
    float var = 0.0f;
    for (int cc = 0; cc < out_channels; ++cc) {
        int idx = batch * out_channels * output_D * output_H * output_W +
                  cc * output_D * output_H * output_W +
                  do_ * output_H * output_W + ho * output_W + wo;
        float diff = output[idx] - mean;
        var += diff * diff;
    }
    var /= out_channels;
    float inv_std = rsqrt(var + eps);

    // Apply layer norm, GELU, and scaling
    float x = (conv_val - mean) * inv_std;
    float gelu_val = gelu(x);
    output[output_idx] = gelu_val * scaling_factor;
}
"""

fused_cpp_source = """
torch::Tensor fused_convtrans3d_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    float scaling_factor);
"""

# Compile the fused CUDA kernel
fused_op = load_inline(
    name="fused_convtrans3d_layernorm_gelu_scale",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_convtrans3d_layernorm_gelu_scale_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for 3D transposed convolution, layer normalization, GELU, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.eps = eps
        self.scaling_factor = scaling_factor
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)
        
        # Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)
        
        self.fused_op = fused_op.fused_convtrans3d_layernorm_gelu_scale_cuda

    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        return self.fused_op(x, self.weight, self.bias_param, self.eps, self.scaling_factor)