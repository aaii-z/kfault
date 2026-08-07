import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + Swish
conv_transpose3d_swish_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Helper function to compute 5D tensor index: (n, c, d, h, w)
__device__ int index_5d(int n, int c, int d, int h, int w,
                        int C, int D, int H, int W) {
    return ((n * C + c) * D + d) * H * W + h * W + w;
}

__global__ void conv_transpose3d_swish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    // Input shape
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output shape
    int out_channels, int out_depth, int out_height, int out_width,
    // Conv params
    int kernel_size, int stride, int padding
) {
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decode output index
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int d = (idx / (out_width * out_height)) % out_depth;
    int c = (idx / (out_width * out_height * out_depth)) % out_channels;
    int n = idx / (out_width * out_height * out_depth * out_channels);

    float val = 0.0f;

    // ConvTranspose3d: sum over input channels and kernel positions
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_d = d * stride - padding + kd;
                    int in_h = h * stride - padding + kh;
                    int in_w = w * stride - padding + kw;

                    if (in_d >= 0 && in_d < in_depth &&
                        in_h >= 0 && in_h < in_height &&
                        in_w >= 0 && in_w < in_width) {
                        int input_idx = index_5d(n, ic, in_d, in_h, in_w, in_channels, in_depth, in_height, in_width);
                        int weight_idx = ((ic * out_channels + c) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias
    val += bias[c];

    // Swish: x * sigmoid(x), with clamping to avoid overflow
    float sigmoid = 1.0f / (1.0f + expf(-fmaxf(-15.0f, fminf(15.0f, val))));
    output[idx] = val * sigmoid;
}

torch::Tensor conv_transpose3d_swish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride, int padding, int kernel_size
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int in_depth = input_sizes[2];
    int in_height = input_sizes[3];
    int in_width = input_sizes[4];

    int out_channels = weight.size(1);
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;

    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width},
                               input.options());

    int total_elements = output.numel();
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    conv_transpose3d_swish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, out_depth, out_height, out_width,
        kernel_size, stride, padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}
"""

conv_transpose3d_swish_cpp_source = """
torch::Tensor conv_transpose3d_swish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride, int padding, int kernel_size
);
"""

# Compile the fused CUDA kernel
fused_ops = load_inline(
    name="conv_transpose3d_swish",
    cpp_sources=conv_transpose3d_swish_cpp_source,
    cuda_sources=conv_transpose3d_swish_cuda_source,
    functions=["conv_transpose3d_swish_cuda"],
    verbose=False,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for:
    ConvTranspose3d + Swish, followed by PyTorch's efficient GroupNorm and HardSwish
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

        # Initialize weights and bias (same as ConvTranspose3d)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Use PyTorch's optimized GroupNorm and Hardswish
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.hardswish = nn.Hardswish()

        # Fused CUDA op
        self.fused_conv_swish = fused_ops.conv_transpose3d_swish_cuda

    def forward(self, x):
        bias = self.bias if self.bias is not None else torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        x = self.fused_conv_swish(x, self.weight, bias, self.stride, self.padding, self.kernel_size)
        x = self.group_norm(x)
        x = self.hardswish(x)
        return x