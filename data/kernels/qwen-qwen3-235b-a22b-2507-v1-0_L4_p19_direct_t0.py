import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose2d + GELU + GroupNorm
fused_conv_gelu_gnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_gelu_gnorm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float* gamma,
    float* beta,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size, int stride,
    int groups,
    int num_groups,
    int group_size,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_elements) return;

    int n = idx / (out_channels * output_height * output_width);
    int residual = idx % (out_channels * output_height * output_width);
    int c = residual / (output_height * output_width);
    int hw = residual % (output_height * output_width);
    int h = hw / output_width;
    int w = hw % output_width;

    int group_id = c / (out_channels / groups);
    int local_c = c % (out_channels / groups);

    float val = 0.0f;
    int weight_offset = group_id * (out_channels / groups) * in_channels / groups * kernel_size * kernel_size;
    
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int ih_start = h * stride - (kernel_size / 2);
            int iw_start = w * stride - (kernel_size / 2);
            int ih = ih_start + i;
            int iw = iw_start + j;

            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                for (int ic = 0; ic < in_channels / groups; ic++) {
                    int input_idx = n * in_channels * input_height * input_width +
                                    (group_id * (in_channels / groups) + ic) * input_height * input_width +
                                    ih * input_width + iw;
                    int weight_idx = weight_offset + local_c * (in_channels / groups) * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size + i * kernel_size + j;
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    val += bias[c];

    // Apply GELU
    val = gelu(val);

    // GroupNorm: compute mean and variance within the group
    int group_start_channel = (c / group_size) * group_size;
    float mean = 0.0f, var = 0.0f;
    int count = group_size * output_height * output_width;
    int base_idx = n * out_channels * output_height * output_width;
    
    for (int gc = 0; gc < group_size; gc++) {
        for (int gh = 0; gh < output_height; gh++) {
            for (int gw = 0; gw < output_width; gw++) {
                int temp_idx = base_idx + (group_start_channel + gc) * output_height * output_width + gh * output_width + gw;
                float x_temp = (gc == 0 && gh == 0 && gw == 0) ? val : output[temp_idx];
                mean += x_temp;
            }
        }
    }
    mean /= count;

    for (int gc = 0; gc < group_size; gc++) {
        for (int gh = 0; gh < output_height; gh++) {
            for (int gw = 0; gw < output_width; gw++) {
                int temp_idx = base_idx + (group_start_channel + gc) * output_height * output_width + gh * output_width + gw;
                float x_temp = (gc == 0 && gh == 0 && gw == 0) ? val : output[temp_idx];
                var += (x_temp - mean) * (x_temp - mean);
            }
        }
    }
    var /= count;
    float inv_std = 1.0f / sqrt(var + eps);

    // Normalize and apply gamma (scale) and beta (bias)
    int channel_in_group = c - group_start_channel;
    float gamma_val = gamma[channel_in_group];
    float beta_val = beta[channel_in_group];
    val = (val - mean) * inv_std * gamma_val + beta_val;

    output[idx] = val;
}
"""

fused_conv_gelu_gnorm_cpp = """
torch::Tensor fused_conv_transpose_gelu_gnorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_size, int stride,
    int groups,
    int num_groups,
    float eps
);
"""

fused_op = load_inline(
    name="fused_conv_gelu_gnorm",
    cpp_sources=fused_conv_gelu_gnorm_cpp,
    cuda_sources=fused_conv_gelu_gnorm_source,
    functions=["fused_conv_transpose_gelu_gnorm_cuda"],
    extra_cuda_cflags=["-use_fast_math"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.num_groups = num_groups
        self.group_size = out_channels // num_groups
        self.eps = 1e-5

        # Transposed convolution parameters
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # GroupNorm parameters
        self.gamma = nn.Parameter(torch.ones(num_groups, self.group_size))
        self.beta = nn.Parameter(torch.zeros(num_groups, self.group_size))

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')

        # Register buffer for output size calculation
        self.register_buffer('dummy', torch.empty(0))

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        # Calculate output dimensions for transposed convolution
        output_height = (height - 1) * self.stride - 2 * (self.kernel_size // 2) + self.kernel_size
        output_width = (width - 1) * self.stride - 2 * (self.kernel_size // 2) + self.kernel_size

        # Flatten weight to match expected format in CUDA kernel
        weight_flat = self.weight.permute(1, 0, 2, 3).contiguous().flatten()
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=x.dtype)

        # Launch fused kernel
        n_threads = 256
        total_elements = batch_size * self.out_channels * output_height * output_width
        n_blocks = (total_elements + n_threads - 1) // n_threads

        fused_op.fused_conv_transpose_gelu_gnorm_cuda(
            x, self.weight, self.bias, self.gamma, self.beta,
            self.in_channels, self.out_channels,
            height, width,
            output_height, output_width,
            self.kernel_size, self.stride,
            self.groups, self.num_groups,
            self.group_size,
            self.eps
        )

        return output