import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

logsumexp_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void logsumexp_relu_kernel(
    const float* input,
    float* output,
    int B, int C, int D, int H, int W,
    int input_b_stride, int input_c_stride, int input_d_stride, int input_h_stride, int input_w_stride,
    int output_b_stride, int output_c_stride, int output_d_stride, int output_h_stride, int output_w_stride
) {
    int idx = blockIdx.x;
    int b = idx / (D * H * W);
    int rest = idx % (D * H * W);
    int d = rest / (H * W);
    rest = rest % (H * W);
    int h = rest / W;
    int w = rest % W;

    float max_val = -INFINITY;
    for (int c = 0; c < C; c++) {
        int input_idx = b * input_b_stride + c * input_c_stride + d * input_d_stride + h * input_h_stride + w * input_w_stride;
        float x = input[input_idx];
        if (x > max_val) max_val = x;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < C; c++) {
        int input_idx = b * input_b_stride + c * input_c_stride + d * input_d_stride + h * input_h_stride + w * input_w_stride;
        float x = input[input_idx];
        sum_exp += expf(x - max_val);
    }

    float logsum = max_val + logf(sum_exp);
    float result = logsum > 0.0f ? logsum : 0.0f;

    int output_idx = b * output_b_stride + 0 * output_c_stride + d * output_d_stride + h * output_h_stride + w * output_w_stride;
    output[output_idx] = result;
}

torch::Tensor logsumexp_relu_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 5, "Input must be 5-dimensional");
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::empty({B, 1, D, H, W}, input.options());

    auto input_strides = input.strides();
    int input_b_stride = input_strides[0];
    int input_c_stride = input_strides[1];
    int input_d_stride = input_strides[2];
    int input_h_stride = input_strides[3];
    int input_w_stride = input_strides[4];

    auto output_strides = output.strides();
    int output_b_stride = output_strides[0];
    int output_c_stride = output_strides[1];
    int output_d_stride = output_strides[2];
    int output_h_stride = output_strides[3];
    int output_w_stride = output_strides[4];

    int grid_size = B * D * H * W;
    int block_size = 1;

    logsumexp_relu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W,
        input_b_stride, input_c_stride, input_d_stride, input_h_stride, input_w_stride,
        output_b_stride, output_c_stride, output_d_stride, output_h_stride, output_w_stride
    );

    return output;
}
"""

logsumexp_relu_cpp = "torch::Tensor logsumexp_relu_cuda(torch::Tensor input);"

logsumexp_relu = load_inline(
    name="logsumexp_relu",
    cpp_sources=logsumexp_relu_cpp,
    cuda_sources=logsumexp_relu_source,
    functions=["logsumexp_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.logsumexp_relu = logsumexp_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.logsumexp_relu.logsumexp_relu_cuda(x)
        return x