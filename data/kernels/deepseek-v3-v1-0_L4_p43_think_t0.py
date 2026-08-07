import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused logsumexp + ReLU
logsumexp_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

__global__ void logsumexp_relu_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_channels,
    int depth,
    int height,
    int width
) {
    int d = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    int b = blockIdx.w;
    
    if (d >= depth || h >= height || w >= width || b >= batch_size) return;

    int spatial_size = height * width;
    int slice_size = spatial_size * depth;
    int input_idx = b * slice_size * num_channels + d * spatial_size + h * width + w;
    int output_idx = b * slice_size + d * spatial_size + h * width + w;

    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    for (int c = 0; c < num_channels; ++c) {
        float val = input[input_idx + c * slice_size];
        if (val > max_val) max_val = val;
    }

    for (int c = 0; c < num_channels; ++c) {
        float val = input[input_idx + c * slice_size];
        sum_exp += expf(val - max_val);
    }

    float result = max_val + logf(sum_exp);
    output[output_idx] = fmaxf(0.0f, result);
}

torch::Tensor logsumexp_relu_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int num_channels = sizes[1];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];
    
    auto output = torch::empty({batch_size, 1, depth, height, width}, 
                              torch::device(torch::kCUDA).dtype(torch::kFloat32));
    
    dim3 grid(depth, height, width, batch_size);
    logsumexp_relu_kernel<<<grid, 1>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        depth,
        height,
        width
    );
    
    return output;
}
"""

logsumexp_relu_cpp_source = "torch::Tensor logsumexp_relu_cuda(torch::Tensor input);"

# Compile the inline CUDA code
logsumexp_relu = load_inline(
    name="logsumexp_relu",
    cpp_sources=logsumexp_relu_cpp_source,
    cuda_sources=logsumexp_relu_source,
    functions=["logsumexp_relu_cuda"],
    verbose=True
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