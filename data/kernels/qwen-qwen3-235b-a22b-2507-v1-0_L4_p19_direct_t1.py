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
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_gelu_gnorm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* gamma,
    const float* beta,
    float* output,
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

    // ConvTranspose2d computation
    float val = 0.0f;
    int ic_per_group = in_channels / groups;
    int oc_per_group = out_channels / groups;
    int group_id = c / oc_per_group;
    int local_c = c % oc_per_group;

    int weight_offset = group_id * oc_per_group * ic_per_group * kernel_size * kernel_size;

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int ih_start = h * stride - (kernel_size / 2);
            int iw_start = w * stride - (kernel_size / 2);
            int ih = ih_start + i;
            int iw = iw_start + j;

            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                for (int ic = 0; ic < ic_per_group; ++ic) {
                    int input_idx = n * in_channels * input_height * input_width +
                                    (group_id * ic_per_group + ic) * input_height * input_width +
                                    ih * input_width + iw;
                    int weight_idx = weight_offset + local_c * ic_per_group * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size + i * kernel_size + j;
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    val += bias[c];

    // GELU activation
    val = gelu(val);

    // GroupNorm: compute mean and variance over group
    int group_idx = c / group_size;
    int base_ch = group_idx * group_size;
    int num_elements_in_group = group_size * output_height * output_width;

    float mean = 0.0f;
    float var = 0.0f;

    // Use shared memory for reduction
    extern __shared__ float s_data[];
    float* s_mean = s_data;
    float* s_var = s_data + blockDim.x;

    int local_idx = threadIdx.x;
    int global_ch, gh, gw;
    float sum_val = 0.0f;

    // Compute mean
    for (int i = 0; i < num_elements_in_group; i += blockDim.x) {
        int offset = i + local_idx;
        if (offset < num_elements_in_group) {
            global_ch = base_ch + (offset / (output_height * output_width));
            gh = (offset % (output_height * output_width)) / output_width;
            gw = (offset % (output_height * output_width)) % output_width;
            int temp_idx = n * out_channels * output_height * output_width +
                           global_ch * output_height * output_width + gh * output_width + gw;
            float temp_val = (global_ch == c && gh == h && gw == w) ? val : output[temp_idx];
            sum_val += temp_val;
        }
    }
    s_mean[local_idx] = sum_val;
    __syncthreads();

    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_idx < s) {
            s_mean[local_idx] += s_mean[local_idx + s];
        }
        __syncthreads();
    }

    mean = s_mean[0] / num_elements_in_group;

    // Compute variance
    sum_val = 0.0f;
    for (int i = 0; i < num_elements_in_group; i += blockDim.x) {
        int offset = i + local_idx;
        if (offset < num_elements_in_group) {
            global_ch = base_ch + (offset / (output_height * output_width));
            gh = (offset % (output_height * output_width)) / output_width;
            gw = (offset % (output_height * output_width)) % output_width;
            int temp_idx = n * out_channels * output_height * output_width +
                           global_ch * output_height * output_width + gh * output_width + gw;
            float temp_val = (global_ch == c && gh == h && gw == w) ? val : output[temp_idx];
            float diff = temp_val - mean;
            sum_val += diff * diff;
        }
    }
    s_var[local_idx] = sum_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_idx < s) {
            s_var[local_idx] += s_var[local_idx + s];
        }
        __syncthreads();
    }

    var = s_var[0] / num_elements_in_group;
    float inv_std = rsqrtf(var + eps);

    // Normalize and apply affine transform
    int c_in_group = c - base_ch;
    float gamma_val = gamma[c_in_group];
    float beta_val = beta[c_in_group];
    val = (val - mean) * inv_std * gamma_val + beta_val;

    output[idx] = val;
}

torch::Tensor fused_conv_transpose_gelu_gnorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size, int stride,
    int groups, int num_groups, float eps) {
    
    auto batch_size = input.size(0);
    auto total_elements = batch_size * out_channels * output_height * output_width;
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    const int block_size = 256;
    const int n_blocks = (total_elements + block_size - 1) / block_size;
    const int smem_size = block_size * 2 * sizeof(float);

    fused_conv_transpose_gelu_gnorm_kernel<<<n_blocks, block_size, smem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size, stride,
        groups, num_groups,
        out_channels / num_groups,
        eps
    );

    return output;
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
    int output_height, int output_width,
    int kernel_size, int stride,
    int groups, int num_groups, float eps);
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
        self.eps = 1e-5

        # Learnable parameters for transposed convolution
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Learnable parameters for GroupNorm
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape

        # Calculate output dimensions for transposed convolution
        output_height = (height - 1) * self.stride - 2 * (self.kernel_size // 2) + self.kernel_size
        output_width = (width - 1) * self.stride - 2 * (self.kernel_size // 2) + self.kernel_size

        # Reshape weight to match NCHW format expected in kernel
        weight_flat = self.weight
        gamma_flat = self.gamma
        beta_flat = self.beta

        # Launch fused kernel
        output = fused_op.fused_conv_transpose_gelu_gnorm_cuda(
            x, weight_flat, self.bias, gamma_flat, beta_flat,
            self.in_channels, self.out_channels,
            height, width,
            output_height, output_width,
            self.kernel_size, self.stride,
            self.groups, self.num_groups,
            self.eps
        )

        return output