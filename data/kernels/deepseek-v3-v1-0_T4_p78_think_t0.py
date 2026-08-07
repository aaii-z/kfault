import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused max pooling and sum reduction
fused_maxpool_sum_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_maxpool_sum_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    int pool_size
) {
    int output_depth = depth / pool_size;
    int output_height = height / pool_size;
    int output_width = width / pool_size;
    
    int n = blockIdx.x;
    int d = blockIdx.y;
    int h = blockIdx.z;
    int w = threadIdx.x;

    if (n >= batch_size || d >= output_depth || h >= output_height || w >= output_width) {
        return;
    }

    float sum_val = 0.0f;
    for (int c = 0; c < channels; ++c) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {
                for (int k = 0; k < pool_size; ++k) {
                    int input_d = d * pool_size + i;
                    int input_h = h * pool_size + j;
                    int input_w = w * pool_size + k;
                    int input_idx = ((n * channels + c) * depth + input_d) * height * width + input_h * width + input_w;
                    float val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
        sum_val += max_val;
    }

    int output_idx = ((n * 1 + 0) * output_depth + d) * output_height * output_width + h * output_width + w;
    output[output_idx] = sum_val;
}

torch::Tensor fused_maxpool_sum_cuda(torch::Tensor input, int pool_size) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    
    int output_depth = depth / pool_size;
    int output_height = height / pool_size;
    int output_width = width / pool_size;
    
    auto output = torch::zeros({batch_size, 1, output_depth, output_height, output_width}, input.options());
    
    dim3 grid(batch_size, output_depth, output_height);
    dim3 block(output_width);
    
    fused_maxpool_sum_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        pool_size
    );
    
    return output;
}
"""

fused_maxpool_sum_cpp_source = "torch::Tensor fused_maxpool_sum_cuda(torch::Tensor input, int pool_size);"

# Compile the inline CUDA code
fused_maxpool_sum = load_inline(
    name="fused_maxpool_sum",
    cpp_sources=fused_maxpool_sum_cpp_source,
    cuda_sources=fused_maxpool_sum_source,
    functions=["fused_maxpool_sum_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.pool_size = 6  # Combined pool size from original max pools (2*3=6)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_maxpool_sum.fused_maxpool_sum_cuda(x, self.pool_size)
        return x