import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

double_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void double_avg_pool_kernel(
    const float* input,
    float* output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * D_out * H_out * W_out) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int d = (idx / (W_out * H_out)) % D_out;
    int c = (idx / (W_out * H_out * D_out)) % C;
    int n = idx / (W_out * H_out * D_out * C);

    float sum = 0.0f;
    int count = 0;

    int start_d = d * 4;
    int start_h = h * 4;
    int start_w = w * 4;

    for (int kd = 0; kd < 4; ++kd) {
        for (int kh = 0; kh < 4; ++kh) {
            for (int kw = 0; kw < 4; ++kw) {
                int d_in = start_d + kd;
                int h_in = start_h + kh;
                int w_in = start_w + kw;

                if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = ((n * C + c) * D_in + d_in) * H_in * W_in + h_in * W_in + w_in;
                    sum += input[input_idx];
                    count++;
                }
            }
        }
    }

    output[idx] = count > 0 ? sum / count : 0.0f;
}

torch::Tensor double_avg_pool_cuda(torch::Tensor input) {
    auto input_sizes = input.sizes();
    int N = input_sizes[0];
    int C = input_sizes[1];
    int D_in = input_sizes[2];
    int H_in = input_sizes[3];
    int W_in = input_sizes[4];

    int D_out = (D_in - 4) / 4 + 1;
    int H_out = (H_in - 4) / 4 + 1;
    int W_out = (W_in - 4) / 4 + 1;

    auto output = torch::zeros({N, C, D_out, H_out, W_out}, input.options());

    int total_elements = N * C * D_out * H_out * W_out;
    if (total_elements == 0) {
        return output;
    }

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    double_avg_pool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out
    );

    return output;
}
"""

double_avg_pool_cpp_source = "torch::Tensor double_avg_pool_cuda(torch::Tensor input);"

double_avg_pool = load_inline(
    name="double_avg_pool",
    cpp_sources=double_avg_pool_cpp_source,
    cuda_sources=double_avg_pool_source,
    functions=["double_avg_pool_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.double_avg_pool = double_avg_pool

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.double_avg_pool.double_avg_pool_cuda(x)
        return x