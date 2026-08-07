import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused 3D convolution + ReLU + LeakyReLU + GELU + Sigmoid + Bias
fused_conv_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused GELU activation: x * 0.5 * (1.0 + erf(x * 0.7071067811865476))
__device__ float gelu(float x) {
    return x * 0.5f * (1.0f + ::erff(x * 0.7071067811865476f));
}

__global__ void fused_conv_act_bias_kernel(
    const float* input,
    const float* weight,
    const float* bias_param,
    float* output,
    int batch_size,
    int out_channels,
    int in_channels,
    int depth, int height, int width,
    int out_d, int out_h, int out_w,
    int kernel_size,
    float leaky_relu_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % out_w;
    int h = (idx / out_w) % out_h;
    int d = (idx / (out_w * out_h)) % out_d;
    int c = (idx / (out_w * out_h * out_d)) % out_channels;
    int b = idx / (out_w * out_h * out_d * out_channels);

    int kernel_radius = kernel_size / 2;
    float val = 0.0f;

    // 3D Convolution
    for (int ci = 0; ci < in_channels; ++ci) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_d = d + kd - kernel_radius;
                    int in_h = h + kh - kernel_radius;
                    int in_w = w + kw - kernel_radius;

                    if (in_d >= 0 && in_d < depth && in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        int in_idx = b * in_channels * depth * height * width +
                                     ci * depth * height * width +
                                     in_d * height * width +
                                     in_h * width +
                                     in_w;

                        int weight_idx = c * in_channels * kernel_size * kernel_size * kernel_size +
                                         ci * kernel_size * kernel_size * kernel_size +
                                         kd * kernel_size * kernel_size +
                                         kh * kernel_size +
                                         kw;

                        val += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add per-channel bias from conv
    val += bias_param[c];

    // Apply ReLU
    val = fmaxf(0.0f, val);

    // Apply LeakyReLU
    val = (val >= 0.0f) ? val : val * leaky_relu_slope;

    // Apply GELU
    val = gelu(val);

    // Apply Sigmoid
    val = 1.0f / (1.0f + expf(-val));

    output[idx] = val;
}

// CUDA C++ wrapper called from Python
torch::Tensor fused_conv_act_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_param,
    int out_d, int out_h, int out_w,
    float leaky_relu_slope) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    auto output = torch::empty({batch_size, out_channels, out_d, out_h, out_w}, input.options());

    const int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_act_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_param.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        in_channels,
        depth, height, width,
        out_d, out_h, out_w,
        kernel_size,
        leaky_relu_slope
    );

    return output;
}
"""

fused_conv_activation_cpp = """
torch::Tensor fused_conv_act_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_param,
    int out_d, int out_h, int out_w,
    float leaky_relu_slope);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_act_bias",
    cpp_sources=fused_conv_activation_cpp,
    cuda_sources=fused_conv_activation_source,
    functions=["fused_conv_act_bias_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused 3D convolution and activation functions (ReLU, LeakyReLU, GELU, Sigmoid) + bias in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))  # per-channel bias
        self.leaky_relu_slope = 0.01
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        batch_size, _, depth, height, width = x.shape
        out_d = depth - self.kernel_size + 1
        out_h = height - self.kernel_size + 1
        out_w = width - self.kernel_size + 1

        return fused_op.fused_conv_act_bias_cuda(
            x,
            self.weight,
            self.bias.view(self.out_channels),
            out_d, out_h, out_w,
            self.leaky_relu_slope
        )