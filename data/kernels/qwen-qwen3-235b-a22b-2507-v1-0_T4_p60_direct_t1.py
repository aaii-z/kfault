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

// Forward kernel: each thread handles one output element
__global__ void fused_convtranspose_swish_gn_hardswish(
    const float* input,
    const float* weight,
    const float* bias,
    const float* gn_weight,
    const float* gn_bias,
    float* output,
    // Input dimensions
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions
    int out_batch_size, int out_channels, int out_depth, int out_height, int out_width,
    // Conv params
    int kernel_size, int stride, int padding,
    // Group norm params
    int groups, int group_size,
    float eps
) {
    int total_elements = out_batch_size * out_channels * out_depth * out_height * out_width;
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

    // Perform transposed convolution
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
    float swish = sum * (1.0f / (1.0f + expf(-fmaxf(-15.0f, fminf(15.0f, sum)))));

    // Group Normalization: compute mean and variance within the channel group for this sample
    int group_id = c / group_size;
    float mean = 0.0f, var = 0.0f;
    int num_elements_in_group = out_depth * out_height * out_width * group_size;

    // Use a small block-reduction to compute mean and variance
    // We use a simple loop over all spatial locations and channels in the group
    for (int dc = 0; dc < group_size; ++dc) {
        int group_c = group_id * group_size + dc;
        for (int dd = 0; dd < out_depth; ++dd) {
            for (int hh = 0; hh < out_height; ++hh) {
                for (int ww = 0; ww < out_width; ++ww) {
                    float val = (group_c == c && dd == d && hh == h && ww == w) ? swish : 
                                __ldg(&input[get_index(n, group_c, dd, hh, ww, in_channels, in_depth, in_height, in_width)]);
                    // This is a placeholder: we cannot read from output or input here correctly
                    // Instead, we must split into two kernels or precompute stats
                }
            }
        }
    }

    // Instead: we abandon full fusion and only fuse conv + swish + hardswish
    // and leave group norm as a separate efficient PyTorch op
    // So this kernel will only do: conv_transpose + swish
    // Then we do group norm and hardswish in PyTorch
    output[idx] = swish;
}
"""

# Instead, we do a safer fusion: only ConvTranspose3d + Swish in CUDA
# Then use PyTorch's optimized GroupNorm and HardSwish
conv_swish_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ int get_index_5d(int n, int c, int d, int h, int w, 
                            int C, int D, int H, int W) {
    return ((n * C + c) * D + d) * H * W + h * W + w;
}

__global__ void conv_transpose3d_swish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int out_depth, int out_height, int out_width,
    int kernel_size, int stride, int padding
) {
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int temp = idx;
    int w = temp % out_width; temp /= out_width;
    int h = temp % out_height; temp /= out_height;
    int d = temp % out_depth; temp /= out_depth;
    int c = temp % out_channels; temp /= out_channels;
    int n = temp;

    float sum = 0.0f;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int id = d * stride - padding + kd;
                int ih = h * stride - padding + kh;
                int iw = w * stride - padding + kw;

                if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        int in_idx = get_index_5d(n, ic, id, ih, iw, in_channels, in_depth, in_height, in_width);
                        int weight_idx = ((ic * out_channels + c) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    sum += bias[c];

    // Swish: x * sigmoid(x), clamped to avoid overflow
    float sigmoid = 1.0f / (1.0f + expf(-fmaxf(-15.0f, fminf(15.0f, sum))));
    output[idx] = sum * sigmoid;
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

    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int in_channels = sizes[1];
    int in_depth = sizes[2];
    int in_height = sizes[3];
    int in_width = sizes[4];

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

conv_swish_cpp_source = """
torch::Tensor conv_transpose3d_swish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride, int padding, int kernel_size
);
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="conv_transpose3d_swish",
    cpp_sources=conv_swish_cpp_source,
    cuda_sources=conv_swish_cuda_source,
    functions=["conv_transpose3d_swish_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
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

        # Transposed convolution weights: (in_channels, out_channels, k, k, k)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # GroupNorm and Hardswish remain as standard PyTorch ops (highly optimized)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.hardswish = torch.nn.Hardswish()

        # Fused CUDA op for conv_transpose + swish
        self.fused_conv_swish = fused_ops.conv_transpose3d_swish_cuda

    def forward(self, x):
        bias = self.bias if self.bias is not None else torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        x = self.fused_conv_swish(x, self.weight, bias, self.stride, self.padding, self.kernel_size)
        x = self.group_norm(x)
        x = self.hardswish(x)
        return x