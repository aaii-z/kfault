import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + Swish + GroupNorm + HardSwish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utility function to get linear index in 5D tensor: (n, c, d, h, w)
__device__ int get_index(int n, int c, int d, int h, int w, 
                         int size_c, int size_d, int size_h, int size_w) {
    return ((n * size_c + c) * size_d + d) * size_h * size_w + h * size_w + w;
}

// Transposed convolution with Swish activation fused kernel
__global__ void fused_convtranspose_swish_gn_hardswish(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    // Input dimensions
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions
    int out_batch_size, int out_channels, int out_depth, int out_height, int out_width,
    // Conv params
    int kernel_size, int stride, int padding,
    // Group norm params
    int groups, int group_size,
    const float* gn_weight,  // per-channel scale
    const float* gn_bias,    // per-channel shift
    float eps,
    // Weight dimensions: (in_channels, out_channels, k, k, k)
    int weight_offset
) {
    int total_elements = out_batch_size * out_depth * out_height * out_width * out_channels;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decode linear index
    int temp = idx;
    int w = temp % out_width; temp /= out_width;
    int h = temp % out_height; temp /= out_height;
    int d = temp % out_depth; temp /= out_depth;
    int c = temp % out_channels; temp /= out_channels;
    int n = temp;

    int kernel_radius = kernel_size / 2;
    float sum = 0.0f;

    // Perform transposed convolution (also known as deconvolution or fractionally-strided conv)
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int id = d * stride - padding + kd;
                int ih = h * stride - padding + kh;
                int iw = w * stride - padding + kw;

                if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        float input_val = input[get_index(n, ic, id, ih, iw, in_channels, in_depth, in_height, in_width)];
                        float weight_val = weight[(((ic * out_channels) + c) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw];
                        sum += input_val * weight_val;
                    }
                }
            }
        }
    }

    // Add bias
    sum += bias[c];

    // Swish: x * sigmoid(x)
    float swish = sum * (1.0f / (1.0f + expf(-sum)));

    // Group Normalization: apply per group
    int group_id = c / group_size;
    int c_within_group = c % group_size;

    // Compute mean and variance within group across D, H, W
    float mean = 0.0f, var = 0.0f;
    int group_elements = out_depth * out_height * out_width * group_size;
    for (int i = 0; i < group_elements; ++i) {
        int cw = i % group_size;
        int ww = (i / group_size) % out_width;
        int hh = (i / (group_size * out_width)) % out_height;
        int dd = i / (group_size * out_width * out_height);
        float val = output[get_index(n, group_id * group_size + cw, dd, hh, ww, out_channels, out_depth, out_height, out_width)];
        mean += val;
        var += val * val;
    }
    mean /= group_elements;
    var = var / group_elements - mean * mean;
    float inv_std = rsqrtf(var + eps);

    float gn_output = (swish - mean) * inv_std * gn_weight[c] + gn_bias[c];

    // HardSwish: x * relu6(x + 3) / 6
    float relu6 = fmaxf(0.0f, fminf(6.0f, gn_output + 3.0f));
    float hardswish = gn_output * relu6 / 6.0f;

    output[idx] = hardswish;
}

// Host wrapper
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int in_channels, int out_channels, int kernel_size, int stride, int padding,
    int groups, float eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(gn_weight);
    CHECK_INPUT(gn_bias);

    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_depth = input_sizes[2];
    int in_height = input_sizes[3];
    int in_width = input_sizes[4];

    // Output spatial dimensions for transposed conv
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, 
                               input.options()).contiguous();

    int total_output_elements = output.numel();
    int block_size = 256;
    int grid_size = (total_output_elements + block_size - 1) / block_size;

    int group_size = out_channels / groups;

    fused_convtranspose_swish_gn_hardswish<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        batch_size, out_channels, out_depth, out_height, out_width,
        kernel_size, stride, padding,
        groups, group_size,
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        eps,
        0  // weight_offset placeholder
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int in_channels, int out_channels, int kernel_size, int stride, int padding,
    int groups, float eps
);
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_convtranspose_swish_gn_hardswish",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for:
    ConvTranspose3d + Swish + GroupNorm + HardSwish
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.eps = eps

        # Define the transposed convolution weights and bias
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # GroupNorm parameters
        self.group_norm_weight = nn.Parameter(torch.ones(out_channels))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_channels))

        # Store fused CUDA op
        self.fused_op = fused_ops.fused_op_cuda

    def forward(self, x):
        bias = self.bias if self.bias is not None else torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        return self.fused_op(
            x,
            self.weight,
            bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.groups,
            self.eps
        )