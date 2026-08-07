import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void min_reduction_softmax_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width
) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (h >= height || w >= width) return;

    float min_vals[24];
    float softmax_vals[24];

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            float min_val = FLT_MAX;
            for (int d = 0; d < depth; ++d) {
                int idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
                min_val = fminf(min_val, input[idx]);
            }
            min_vals[c] = min_val;
        }

        float max_val = -FLT_MAX;
        for (int c = 0; c < channels; ++c) {
            max_val = fmaxf(max_val, min_vals[c]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < channels; ++c) {
            softmax_vals[c] = expf(min_vals[c] - max_val);
            sum_exp += softmax_vals[c];
        }

        for (int c = 0; c < channels; ++c) {
            int out_idx = (b * channels + c) * height * width + h * width + w;
            output[out_idx] = softmax_vals[c] / sum_exp;
        }
    }
}

torch::Tensor min_reduction_softmax_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];
    
    auto output = torch::empty({batch_size, channels, height, width}, input.options());

    dim3 blocks(1, (height + 7) / 8, (width + 7) / 8);
    dim3 threads(1, 8, 8);

    min_reduction_softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width
    );

    return output;
}
"""

min_softmax_cpp_source = "torch::Tensor min_reduction_softmax_cuda(torch::Tensor input);"

min_softmax = load_inline(
    name="min_softmax",
    cpp_sources=min_softmax_cpp_source,
    cuda_sources=min_softmax_source,
    functions=["min_reduction_softmax_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.min_softmax = min_softmax

    def forward(self, x):
        x = self.conv(x)
        if self.dim == 2:
            x = self.min_softmax.min_reduction_softmax_cuda(x.contiguous())
        else:
            x = torch.min(x, dim=self.dim)[0]
            x = torch.softmax(x, dim=1)
        return x