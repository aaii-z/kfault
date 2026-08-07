import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_avg_affine_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_avg_affine_kernel(
    const float* input,
    float* output,
    float combined_scale,
    const float* combined_bias,
    int batch,
    int channels,
    int input_D,
    int input_H,
    int input_W,
    int output_D,
    int output_H,
    int output_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * output_D * output_H * output_W;
    if (idx >= total_elements) return;

    int w = idx % output_W;
    int h = (idx / output_W) % output_H;
    int d = (idx / (output_W * output_H)) % output_D;
    int c = (idx / (output_W * output_H * output_D)) % channels;
    int b = idx / (output_W * output_H * output_D * channels);

    int input_start_d = d * 2;
    int input_start_h = h * 2;
    int input_start_w = w * 2;

    float sum = 0.0f;
    int count = 0;
    for (int dz = 0; dz < 2; dz++) {
        int input_d = input_start_d + dz;
        if (input_d >= input_D) continue;
        for (int dh = 0; dh < 2; dh++) {
            int input_h = input_start_h + dh;
            if (input_h >= input_H) continue;
            for (int dw = 0; dw < 2; dw++) {
                int input_w = input_start_w + dw;
                if (input_w >= input_W) continue;
                int input_idx = b * (channels * input_D * input_H * input_W) +
                               c * (input_D * input_H * input_W) +
                               input_d * (input_H * input_W) +
                               input_h * input_W +
                               input_w;
                sum += input[input_idx];
                count++;
            }
        }
    }

    float avg = (count > 0) ? sum / count : 0.0f;
    float result = avg * combined_scale + combined_bias[c];
    output[idx] = result;
}

torch::Tensor custom_avg_affine_cuda(
    torch::Tensor input,
    float combined_scale,
    torch::Tensor combined_bias
) {
    if (input.dim() != 5) {
        throw std::runtime_error("Input must be 5-dimensional (NCDHW)");
    }

    int batch = input.size(0);
    int channels = input.size(1);
    int input_D = input.size(2);
    int input_H = input.size(3);
    int input_W = input.size(4);

    int output_D = input_D / 2;
    int output_H = input_H / 2;
    int output_W = input_W / 2;

    auto output = torch::empty({batch, channels, output_D, output_H, output_W}, input.options());

    if (output.numel() == 0) {
        return output;
    }

    int total_elements = output.numel();
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    custom_avg_affine_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        combined_scale,
        combined_bias.data_ptr<float>(),
        batch,
        channels,
        input_D,
        input_H,
        input_W,
        output_D,
        output_H,
        output_W
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

custom_avg_affine_cpp = "torch::Tensor custom_avg_affine_cuda(torch::Tensor input, float combined_scale, torch::Tensor combined_bias);"

custom_avg_affine = load_inline(
    name="custom_avg_affine",
    cpp_sources=custom_avg_affine_cpp,
    cuda_sources=custom_avg_affine_source,
    functions=["custom_avg_affine_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.custom_op = custom_avg_affine

    def forward(self, x):
        x = self.conv_transpose(x)
        combined_scale = self.scale1.item() * self.scale2.item()
        combined_bias = self.bias * self.scale2
        x = self.custom_op.custom_avg_affine_cuda(x, combined_scale, combined_bias)
        return x