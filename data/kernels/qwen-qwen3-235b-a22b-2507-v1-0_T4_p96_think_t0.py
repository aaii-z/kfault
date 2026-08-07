import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

global_avg_pool_scale_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_avg_pool_scale_clamp_kernel(const float* input, float* output, 
                                            int batch, int channels, 
                                            int depth, int height, int width,
                                            float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels) return;

    int n = idx / channels;
    int c = idx % channels;

    float sum = 0.0f;
    int count = depth * height * width;

    for (int d = 0; d < depth; d++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int input_idx = ((n * channels + c) * depth + d) * height * width + h * width + w;
                sum += input[input_idx];
            }
        }
    }

    float avg = sum / count;
    float scaled_avg = avg * scale;
    output[idx] = fmaxf(0.0f, fminf(1.0f, scaled_avg));
}

torch::Tensor global_avg_pool_scale_clamp_cuda(torch::Tensor input, float scale) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    auto output = torch::empty({batch, channels, 1, 1, 1}, input.options());

    int total_threads = batch * channels;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    global_avg_pool_scale_clamp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, depth, height, width,
        scale
    );

    return output;
}
"""

global_avg_pool_scale_clamp_cpp = "torch::Tensor global_avg_pool_scale_clamp_cuda(torch::Tensor input, float scale);"

global_avg_pool_scale_clamp = load_inline(
    name="global_avg_pool_scale_clamp",
    cpp_sources=global_avg_pool_scale_clamp_cpp,
    cuda_sources=global_avg_pool_scale_clamp_source,
    functions=["global_avg_pool_scale_clamp_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool_scale_clamp = global_avg_pool_scale_clamp

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = self.global_avg_pool_scale_clamp.global_avg_pool_scale_clamp_cuda(x, self.scale)
        return x