import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear+GELU+Softmax
fused_linear_gelu_softmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

const int BLOCK_SIZE = 256;
const int NUM_BLOCKS = 8192 / BLOCK_SIZE;

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void linear_gelu_softmax_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int in_features,
    int out_features,
    int batch_size
) {
    extern __shared__ float shared_data[];
    float* row_buffer = shared_data;
    float* reduction_buffer = shared_data + out_features;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Load input row into shared memory
    if (tid < in_features) {
        row_buffer[tid] = input[row * in_features + tid];
    }
    __syncthreads();

    // Compute linear transformation
    float sum = 0.0f;
    for (int i = tid; i < in_features; i += blockDim.x) {
        sum += row_buffer[i] * weight[blockIdx.y * in_features + i];
    }

    // Reduce within warp
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in warp writes result
    if (tid % 32 == 0) {
        reduction_buffer[tid / 32] = sum + bias[blockIdx.y];
    }
    __syncthreads();

    // Apply GELU and prepare for softmax
    if (tid < out_features) {
        float val = reduction_buffer[tid];
        float gelu_val = gelu(val);
        reduction_buffer[tid] = gelu_val;
    }
    __syncthreads();

    // Softmax reduction
    float max_val = -INFINITY;
    for (int i = tid; i < out_features; i += blockDim.x) {
        max_val = fmaxf(max_val, reduction_buffer[i]);
    }

    // Warp reduce max
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    float sum_exp = 0.0f;
    for (int i = tid; i < out_features; i += blockDim.x) {
        sum_exp += expf(reduction_buffer[i] - max_val);
    }

    // Warp reduce sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }

    // Compute final softmax
    if (tid < out_features) {
        output[row * out_features + tid] = expf(reduction_buffer[tid] - max_val) / sum_exp;
    }
}

torch::Tensor fused_linear_gelu_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    dim3 grid(batch_size, out_features);
    dim3 block(BLOCK_SIZE);

    size_t shared_mem_size = (out_features + BLOCK_SIZE) * sizeof(float);

    linear_gelu_softmax_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        in_features,
        out_features,
        batch_size
    );

    return output;
}
"""

fused_linear_gelu_softmax_cpp_source = """
torch::Tensor fused_linear_gelu_softmax_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_linear_gelu_softmax = load_inline(
    name="fused_linear_gelu_softmax",
    cpp_sources=fused_linear_gelu_softmax_cpp_source,
    cuda_sources=fused_linear_gelu_softmax_source,
    functions=["fused_linear_gelu_softmax_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_ldflags=["-lcub"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_linear_gelu_softmax.fused_linear_gelu_softmax_cuda(x, self.weight, self.bias)