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

    // ConvTranspose2d: compute output element
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
                int input_group_offset = n * in_channels * input_height * input_width + group_id * ic_per_group * input_height * input_width;
                int weight_ki_kj = ki * kernel_size + kj;
                for (int ic = 0; ic < ic_per_group; ++ic) {
                    int input_idx = input_group_offset + ic * input_height * input_width + ih * input_width + iw;
                    int weight_idx = weight_offset + ic * kernel_size * kernel_size + weight_ki_kj;
                    conv_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    conv_val += bias[c];

    // GELU activation
    float act_val = gelu(conv_val);

    // GroupNorm: compute mean and variance for the group
    int group_idx = c / group_size;
    int c_in_group = c % group_size;
    int num_channels_in_group = group_size;
    int hw = output_height * output_width;
    int num_elements_in_group = num_channels_in_group * hw;

    float sum = 0.0f, sq_sum = 0.0f;
    for (int cc = 0; cc < num_channels_in_group; ++cc) {
        for (int ph = 0; ph < output_height; ++ph) {
            for (int pw = 0; pw < output_width; ++pw) {
                int idx_g = n * out_channels * output_height * output_width +
                            (group_idx * num_channels_in_group + cc) * output_height * output_width +
                            ph * output_width + pw;
                float val = __ldg(&act_val); // Placeholder: we need to load actual GELU output
                // We cannot recompute GELU here efficiently, so we must recompute conv+gelu for all elements in the group
                // This kernel design is flawed for GroupNorm because we need all values in the group
                // Therefore, we abandon per-element full fusion and instead do two kernels
            }
        }
    }

    // Due to complexity of fusing GroupNorm in a single kernel with proper reduction,
    // we instead split into two kernels: one for ConvTranspose2d + GELU, then GroupNorm separately
    // This version will not fully fuse, but we provide a corrected syntax version
}

// We instead define a simpler fused kernel: ConvTranspose2d + GELU only
__global__ void fused_conv_transpose_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size, int stride,
    int groups
) {
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
                int input_group_offset = n * in_channels * input_height * input_width + group_id * ic_per_group * input_height * input_width;
                int weight_ki_kj = ki * kernel_size + kj;
                for (int ic = 0; ic < ic_per_group; ++ic) {
                    int input_idx = input_group_offset + ic * input_height * input_width + ih * input_width + iw;
                    int weight_idx = weight_offset + ic * kernel_size * kernel_size + weight_ki_kj;
                    conv_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    conv_val += bias[c];

    // GELU
    float x = conv_val;
    float tanh_input = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    float gelu_val = 0.5f * x * (1.0f + tanhf(tanh_input));

    output[idx] = gelu_val;
}

// Host function for fused ConvTranspose2d + GELU
torch::Tensor fused_conv_transpose_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size, int stride,
    int groups) {
    
    auto output = torch::empty({input.size(0), out_channels, output_height, output_width}, input.options());
    
    const int total_elements = output.numel();
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    fused_conv_transpose_gelu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0), in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size, stride,
        groups
    );
    
    return output;
}
"""

fused_conv_gelu_gnorm_cpp = """
torch::Tensor fused_conv_transpose_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size, int stride,
    int groups);
"""

# Compile the fused CUDA kernel (ConvTranspose2d + GELU only)
fused_op = load_inline(
    name="fused_conv_transpose_gelu",
    cpp_sources=fused_conv_gelu_gnorm_cpp,
    cuda_sources=fused_conv_gelu_gnorm_source,
    functions=["fused_conv_transpose_gelu_cuda"],
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

        # Learnable parameters for transposed convolution
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # GroupNorm parameters
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        # Weight initialization
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape

        # Output spatial dimensions for ConvTranspose2d
        output_height = (height - 1) * self.stride - 2 * (self.kernel_size // 2) + self.kernel_size
        output_width = (width - 1) * self.stride - 2 * (self.kernel_size // 2) + self.kernel_size

        # Reshape weight to match expected format in CUDA kernel: (out_channels, in_channels//groups, k, k)
        weight_reshaped = self.weight

        # Launch fused ConvTranspose2d + GELU kernel
        x = fused_op.fused_conv_transpose_gelu_cuda(
            x, weight_reshaped, self.bias,
            self.in_channels, self.out_channels,
            height, width,
            output_height, output_width,
            self.kernel_size, self.stride,
            self.groups
        )

        # Reshape to [batch_size, out_channels, H, W] for GroupNorm
        x = x.view(batch_size, self.out_channels, output_height, output_width)

        # Apply GroupNorm (using PyTorch's optimized kernel)
        x = self.group_norm(x)

        return x