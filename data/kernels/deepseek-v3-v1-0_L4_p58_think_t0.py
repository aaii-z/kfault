import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_operation_kernel(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width
) {
    int total_spatial = depth * height * width;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * total_spatial) return;

    int b = index / total_spatial;
    int spatial_idx = index % total_spatial;
    int d = spatial_idx / (height * width);
    int h = (spatial_idx % (height * width)) / width;
    int w = spatial_idx % width;

    // Reduce over channels (logsumexp)
    float max_val = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
        int input_idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        max_val = fmaxf(max_val, input[input_idx]);
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < channels; ++c) {
        int input_idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        sum_exp += expf(input[input_idx] - max_val);
    }
    float logsumexp = logf(sum_exp) + max_val;

    // HardSwish approximation: x * sigmoid(x+3) / 6
    float swished = logsumexp * sigmoid(logsumexp + 3.0f) / 6.0f;

    // Subtract bias and clamp
    float result = swished - bias[0];
    result = fminf(fmaxf(result, -1.0f), 1.0f);

    int output_idx = (b * depth + d) * height * width + h * width + w;
    output[output_idx] = result;
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    
    auto output = torch::empty({batch_size, 1, depth, height, width}, input.options());
    
    int total_elements = batch_size * depth * height * width;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    fused_operation_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
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

fused_operation_cpp_source = "torch::Tensor fused_operation_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_operation_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operation_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.fused_op = fused_operation

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_operation_cuda(x, self.bias)
        return x