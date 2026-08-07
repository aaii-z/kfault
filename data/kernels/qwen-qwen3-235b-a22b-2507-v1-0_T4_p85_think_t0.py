import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void maxpool_clamp_kernel(
    const float* input, float* output,
    int batch_size, int channels, int input_height, int input_width,
    int output_height, int output_width,
    int input_stride_b, int input_stride_c, int input_stride_h, int input_stride_w,
    float clamp_min, float clamp_max) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_height * output_width;
    if (idx >= total_elements) return;

    int oj = idx % output_width;
    int oi = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % channels;
    int b = idx / (output_width * output_height * channels);

    int start_i = oi * 4;
    int start_j = oj * 4;

    float max_val = -INFINITY;
    for (int di = 0; di < 4; di++) {
        for (int dj = 0; dj < 4; dj++) {
            int i_pos = start_i + di;
            int j_pos = start_j + dj;
            if (i_pos < input_height && j_pos < input_width) {
                int input_idx = b * input_stride_b +
                                c * input_stride_c +
                                i_pos * input_stride_h +
                                j_pos * input_stride_w;
                float val = input[input_idx];
                if (val > max_val) max_val = val;
            }
        }
    }

    if (max_val < clamp_min) max_val = clamp_min;
    else if (max_val > clamp_max) max_val = clamp_max;

    output[idx] = max_val;
}

torch::Tensor maxpool_clamp_cuda(torch::Tensor input, float clamp_min, float clamp_max) {
    auto input_size = input.sizes();
    int batch_size = input_size[0];
    int channels = input_size[1];
    int input_height = input_size[2];
    int input_width = input_size[3];

    int output_height = (input_height - 4) / 4 + 1;
    int output_width = (input_width - 4) / 4 + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    int input_stride_b = input.stride(0);
    int input_stride_c = input.stride(1);
    int input_stride_h = input.stride(2);
    int input_stride_w = input.stride(3);

    int total_elements = batch_size * channels * output_height * output_width;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    maxpool_clamp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, input_height, input_width,
        output_height, output_width,
        input_stride_b, input_stride_c, input_stride_h, input_stride_w,
        clamp_min, clamp_max
    );

    return output;
}
"""

maxpool_clamp_cpp = "torch::Tensor maxpool_clamp_cuda(torch::Tensor input, float clamp_min, float clamp_max);"

maxpool_clamp = load_inline(
    name="maxpool_clamp",
    cpp_sources=maxpool_clamp_cpp,
    cuda_sources=maxpool_clamp_source,
    functions=["maxpool_clamp_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.maxpool_clamp = maxpool_clamp

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = x * self.scale
        x = self.maxpool_clamp.maxpool_clamp_cuda(x, self.clamp_min, self.clamp_max)
        return x