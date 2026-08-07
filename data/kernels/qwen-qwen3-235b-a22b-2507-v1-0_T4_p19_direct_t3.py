import torch
import torch.nn as nn
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
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size, int stride,
    int groups, int num_groups,
    float eps
) {
    const int group_size = out_channels / num_groups;
    const int oc_per_group = out_channels / groups;
    const int ic_per_group = in_channels / groups;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    if (idx >= total_elements) return;

    int n = idx / (out_channels * output_height * output_width);
    int residual = idx % (out_channels * output_height * output_width);
    int c = residual / (output_height * output_width);
    int h = (residual % (output_height * output_width)) / output_width;
    int w = residual % output_width;

    // ConvTranspose2d: compute output element (n, c, h, w)
    float conv_val = 0.0f;
    int group_id = c / oc_per_group;
    int local_oc = c % oc_per_group;
    int weight_offset = group_id * oc_per_group * ic_per_group * kernel_size * kernel_size +
                        local_oc * ic_per_group * kernel_size * kernel_size;

    for (int ki = 0; ki < kernel_size; ++ki) {
        for (int kj = 0; kj < kernel_size; ++kj) {
            int ih = h * stride - (kernel_size / 2) + ki;
            int iw = w * stride - (kernel_size / 2) + kj;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int input_nc_offset = n * in_channels * input_height * input_width;
                int weight_kij = ki * kernel_size + kj;
                for (int ic = 0; ic < ic_per_group; ++ic) {
                    int ic_g = group_id * ic_per_group + ic;
                    int input_idx = input_nc_offset + ic_g * input_height * input_width + ih * input_width + iw;
                    int weight_idx = weight_offset + ic * kernel_size * kernel_size + weight_kij;
                    conv_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    conv_val += bias[c];

    // GELU activation
    float act_val = gelu(conv_val);

    // GroupNorm: normalize over (N, H, W) within group
    int group_idx = c / group_size;
    int c_in_group = c % group_size;
    int num_channels_in_group = group_size;
    int num_elements_in_group = batch_size * num_channels_in_group * output_height * output_width;

    // Use shared memory for mean and variance reduction
    extern __shared__ float s_data[];
    float* s_sum = s_data;
    float* s_sq_sum = s_data + blockDim.x;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // Each thread accumulates over all spatial and batch positions in the group
    for (int i = threadIdx.x; i < num_elements_in_group; i += blockDim.x) {
        int b = i / (num_channels_in_group * output_height * output_width);
        int rem = i % (num_channels_in_group * output_height * output_width);
        int cg = rem / (output_height * output_width);
        int hw = rem % (output_height * output_width);
        int h_gn = hw / output_width;
        int w_gn = hw % output_width;
        int ch = group_idx * group_size + cg;
        int src_idx = b * out_channels * output_height * output_width + ch * output_height * output_width + h_gn * output_width + w_gn;
        float val = (b == n && ch == c && h_gn == h && w_gn == w) ? act_val : __ldg(&((const float*)output)[src_idx]);
        local_sum += val;
        local_sq_sum += val * val;
    }

    s_sum[threadIdx.x] = local_sum;
    s_sq_sum[threadIdx.x] = local_sq_sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sq_sum[threadIdx.x] += s_sq_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum = s_sum[0];
    float sq_sum = s_sq_sum[0];
    float mean = sum / num_elements_in_group;
    float var = fmaxf(sq_sum / num_elements_in_group - mean * mean, 0.0f);
    float inv_std = rsqrtf(var + eps);

    // Apply normalization and affine transform
    float gamma_val = gamma[c_in_group];
    float beta_val = beta[c_in_group];
    float result = (act_val - mean) * inv_std * gamma_val + beta_val;

    output[idx] = result;
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
    name="fused_conv_transpose_gelu_gnorm",
    cpp_sources=fused_conv_gelu_gnorm_cpp,
    cuda_sources=fused_conv_gelu_gnorm_source,
    functions=["fused_conv_transpose_gelu_gnorm_cuda"],
    extra_cuda_cflags=["-O3", "--use-fast-math"],
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

        # Transposed conv weights: (in_channels, out_channels // groups, k, k)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # GroupNorm parameters
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))

        # Init
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape

        # Output spatial dimensions for ConvTranspose2d
        output_height = (height - 1) * self.stride - 2 * (self.kernel_size // 2) + self.kernel_size
        output_width = (width - 1) * self.stride - 2 * (self.kernel_size // 2) + self.kernel_size

        # Flatten parameters
        weight_flat = self.weight.view(-1).contiguous()
        gamma_flat = self.gamma.contiguous()
        beta_flat = self.beta.contiguous()

        # Shared memory size: 2 * block_size floats
        block_size = 256
        smem_size = 2 * block_size * sizeof(float);

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