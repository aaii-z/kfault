import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused softmax, bias add, scaling, and sigmoid
fused_softmax_bias_scale_sigmoid_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 256

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void fused_softmax_bias_scale_sigmoid_kernel(
    const float* input,
    const float* bias,
    float scaling_factor,
    float* output,
    int num_channels,
    int spatial_size
) {
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (spatial_idx >= spatial_size) return;

    extern __shared__ float shared_mem[];
    float* max_vals = shared_mem;
    float* sum_vals = shared_mem + gridDim.y;

    int batch = blockIdx.y;
    int base_idx = (batch * num_channels + threadIdx.y) * spatial_size + spatial_idx;

    float max_val = -INFINITY;
    for (int c = threadIdx.y; c < num_channels; c += blockDim.y) {
        int idx = (batch * num_channels + c) * spatial_size + spatial_idx;
        max_val = fmaxf(max_val, input[idx]);
    }
    max_val = warpReduceMax(max_val);
    if (threadIdx.x == 0) {
        max_vals[threadIdx.y] = max_val;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        float block_max = -INFINITY;
        for (int i = 0; i < blockDim.y; i++) {
            block_max = fmaxf(block_max, max_vals[i]);
        }
        max_vals[0] = block_max;
    }
    __syncthreads();
    max_val = max_vals[0];

    float sum_val = 0.0f;
    for (int c = threadIdx.y; c < num_channels; c += blockDim.y) {
        int idx = (batch * num_channels + c) * spatial_size + spatial_idx;
        float exp_val = expf(input[idx] - max_val);
        sum_val += exp_val;
    }
    sum_val = warpReduceSum(sum_val);
    if (threadIdx.x == 0) {
        sum_vals[threadIdx.y] = sum_val;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < blockDim.y; i++) {
            block_sum += sum_vals[i];
        }
        sum_vals[0] = block_sum;
    }
    __syncthreads();
    sum_val = sum_vals[0];

    for (int c = threadIdx.y; c < num_channels; c += blockDim.y) {
        int idx = (batch * num_channels + c) * spatial_size + spatial_idx;
        float val = expf(input[idx] - max_val) / sum_val;
        val = (val + bias[c]) * scaling_factor;
        val = 1.0f / (1.0f + expf(-val));
        output[idx] = val;
    }
}

torch::Tensor fused_softmax_bias_scale_sigmoid_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor
) {
    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int spatial_size = input.size(2) * input.size(3);

    dim3 grid_dim(
        (spatial_size + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size
    );
    dim3 block_dim(BLOCK_SIZE, 32);

    size_t shared_mem_size = 2 * block_dim.y * sizeof(float);

    fused_softmax_bias_scale_sigmoid_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        num_channels,
        spatial_size
    );

    return output;
}
"""

fused_softmax_bias_scale_sigmoid_cpp_source = """
torch::Tensor fused_softmax_bias_scale_sigmoid_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);
"""

# Compile the inline CUDA code
fused_softmax_bias_scale_sigmoid = load_inline(
    name="fused_softmax_bias_scale_sigmoid",
    cpp_sources=fused_softmax_bias_scale_sigmoid_cpp_source,
    cuda_sources=fused_softmax_bias_scale_sigmoid_source,
    functions=["fused_softmax_bias_scale_sigmoid_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_softmax_bias_scale_sigmoid

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_softmax_bias_scale_sigmoid_cuda(x, self.bias, self.scaling_factor)
        return x