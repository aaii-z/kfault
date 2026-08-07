import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sub_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void sub_hardswish_kernel(const float* input, float* output, int size, float subtract_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - subtract_value;
        float temp = x + 3.0f;
        float relu6 = (temp < 0) ? 0.0f : (temp > 6.0f) ? 6.0f : temp;
        output[idx] = x * relu6 / 6.0f;
    }
}

torch::Tensor sub_hardswish_cuda(torch::Tensor input, float subtract_value) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    sub_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        subtract_value
    );

    return output;
}
"""

sub_hardswish_cpp = "torch::Tensor sub_hardswish_cuda(torch::Tensor input, float subtract_value);"

sub_hardswish = load_inline(
    name="sub_hardswish",
    cpp_sources=sub_hardswish_cpp,
    cuda_sources=sub_hardswish_source,
    functions=["sub_hardswish_cuda"],
    verbose=True
)

maxpool_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

int compute_output_size(int input_size, int kernel_size, int padding, int stride, int dilation) {
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

__global__ void maxpool_mish_kernel(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int dilationH, int dilationW,
    int H_out, int W_out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H_out * W_out) return;

    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    int h_start = oh * strideH - padH;
    int w_start = ow * strideW - padW;
    float max_val = -1e30f;

    for (int kh = 0; kh < kH; ++kh) {
        int h = h_start + kh * dilationH;
        if (h < 0 || h >= H) continue;
        for (int kw = 0; kw < kW; ++kw) {
            int w = w_start + kw * dilationW;
            if (w < 0 || w >= W) continue;
            float val = input[n * C * H * W + c * H * W + h * W + w];
            if (val > max_val) {
                max_val = val;
            }
        }
    }

    float sp = log1pf(expf(max_val));
    float tanh_sp = tanhf(sp);
    float mish_val = max_val * tanh_sp;

    output[n * C * H_out * W_out + c * H_out * W_out + oh * W_out + ow] = mish_val;
}

torch::Tensor maxpool_mish_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    int kH = kernel_size;
    int kW = kernel_size;
    int strideH = stride;
    int strideW = stride;
    int padH = padding;
    int padW = padding;
    int dilationH = dilation;
    int dilationW = dilation;

    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    int H_out = compute_output_size(H, kH, padH, strideH, dilationH);
    int W_out = compute_output_size(W, kW, padW, strideW, dilationW);

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    int total_elements = N * C * H_out * W_out;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    maxpool_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kH, kW,
        strideH, strideW,
        padH, padW,
        dilationH, dilationW,
        H_out, W_out
    );

    return output;
}
"""

maxpool_mish_cpp = "torch::Tensor maxpool_mish_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"

maxpool_mish = load_inline(
    name="maxpool_mish",
    cpp_sources=maxpool_mish_cpp,
    cuda_sources=maxpool_mish_source,
    functions=["maxpool_mish_cuda"],
    verbose=True
)

class MaxPoolMish(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, input):
        return maxpool_mish.maxpool_mish_cuda(input, self.kernel_size, self.stride, self.padding, self.dilation)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.maxpool_mish = MaxPoolMish(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = sub_hardswish.sub_hardswish_cuda(x, self.subtract_value)
        x = self.maxpool_mish(x)
        return x