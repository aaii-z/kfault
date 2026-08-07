import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose2d + GELU + GroupNorm with optimized parallelism and memory access
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

    // Each thread handles one output element: (n, c, h, w)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    if (idx >= total_elements) return;

    int n = idx / (out_channels * output_height * output_width);
    int residual = idx % (out_channels * output_height * output_width);
    int c = residual / (output_height * output_width);
    int h = (residual % (output_height * output_width)) / output_width;
    int w = residual % output_width;

    // ConvTranspose2d: compute single output element
    float conv_val = 0.0f;
    int group_id = c / oc_per_group;
    int local_oc = c % oc_per_group;
    int weight_offset = (group_id * oc_per_group * ic_per_group + local_oc * ic_per_group) * kernel_size * kernel_size;

    for (int ki = 0; ki < kernel_size; ++ki) {
        for (int kj = 0; kj < kernel_size; ++kj) {
            int ih = h * stride - (kernel_size / 2) + ki;
            int iw = w * stride - (kernel_size / 2) + kj;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int input_offset = n * in_channels * input_height * input_width + group_id * ic_per_group * input_height * input_width;
                int input_idx_base = input_offset + ih * input_width + iw;
                int weight_idx_base = weight_offset + ki * kernel_size + kj;
                for (int ic = 0; ic < ic_per_group; ++ic) {
                    float input_val = input[input_idx_base + ic * input_height * input_width];
                    float weight_val = weight[weight_idx_base + ic * kernel_size * kernel_size];
                    conv_val += input_val * weight_val;
                }
            }
        }
    }
    conv_val += bias[c];

    // GELU activation
    float gelu_val = gelu(conv_val);

    // GroupNorm: per-group mean and variance computed via block-wide reduction
    int group_idx = c / group_size;
    int c_in_group = c - group_idx * group_size;
    int base_ch = group_idx * group_size;
    int num_ch_in_group = group_size;
    int hw = output_height * output_width;
    int num_elements_in_group = num_ch_in_group * hw;

    // Shared memory for reduction
    extern __shared__ float s_data[];
    float* s_sum = s_data;
    float* s_sq_sum = s_data + blockDim.x;

    float local_val = gelu_val;
    float local_sum = local_val;
    float local_sq_sum = local_val * local_val;

    // Use warp shuffle for reduction within thread block
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sq_sum += __shfl_down_sync(0xffffffff, local_sq_sum, offset);
    }

    // Write warp results to shared memory
    if (threadIdx.x % 32 == 0) {
        s_sum[threadIdx.x / 32] = local_sum;
        s_sq_sum[threadIdx.x / 32] = local_sq_sum;
    }
    __syncthreads();

    // Final reduction in first warp
    if (threadIdx.x < 32) {
        float warp_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? s_sum[threadIdx.x] : 0.0f;
        float warp_sq_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? s_sq_sum[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            warp_sq_sum += __shfl_down_sync(0xffffffff, warp_sq_sum, offset);
        }
        if (threadIdx.x == 0) {
            float mean = warp_sum / num_elements_in_group;
            float var = warp_sq_sum / num_elements_in_group - mean * mean;
            float inv_std = rsqrtf(var + eps);

            // Store normalization parameters in shared memory for reuse
            s_sum[0] = mean;
            s_sq_sum[0] = inv_std;
        }
    }
    __syncthreads();

    float mean = s_sum[0];
    float inv_std = s_sq_sum[0];

    // Apply GroupNorm
    float norm_val = (gelu_val - mean) * inv_std;
    norm_val = norm_val * gamma[c_in_group] + beta[c_in_group];

    output[idx] = norm_val;
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

# Compile the fused CUDA kernel
fused_op = load_inline(
    name="fused_conv_gelu_gnorm",
    cpp_sources=fused_conv_gelu_gnorm_cpp,
    cuda_sources=fused_conv_gelu_gnorm_source,
    functions=["fused_conv_transpose_gelu_gnorm_cuda"],
    extra_cuda_cflags=["-use_fast_math", "--expt-relaxed-constexpr"],
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

        # Transposed convolution parameters
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # GroupNorm parameters
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))

        # Weight initialization
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape

        # Output spatial dimensions for ConvTranspose2d
        output_height = (height - 1) * self.stride - 2 * (self.kernel_size // 2) + self.kernel_size
        output_width = (width - 1) * self.stride - 2 * (self.kernel_size // 2) + self.kernel_size

        # Reshape weight to (in_channels, out_channels//groups, k, k) as expected by kernel
        weight_reshaped = self.weight

        # Shared memory size: 2 * number of warps (max 8 warps for 256 threads)
        int smem_size = 2 * ((256 + 31) // 32) * sizeof(float);

        # Launch configuration
        int total_elements = batch_size * self.out_channels * output_height * output_width
        int block_size = 256
        int grid_size = (total_elements + block_size - 1) // block_size

        output = fused_op.fused_conv_transpose_gelu_gnorm_cuda(
            x, weight_reshaped, self.bias, self.gamma, self.beta,
            self.in_channels, self.out_channels,
            height, width,
            output_height, output_width,
            self.kernel_size, self.stride,
            self.groups, self.num_groups,
            self.eps
        )

        return output