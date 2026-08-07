import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_scale_add_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_scale_add_clamp_kernel(const float* input, float* output, int size, float scale_factor, float clamp_min, float clamp_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        val = 2.0f * scale_factor * val;
        if (val < clamp_min) val = clamp_min;
        if (val > clamp_max) val = clamp_max;
        output[idx] = val;
    }
}

torch::Tensor fused_scale_add_clamp_cuda(torch::Tensor input, float scale_factor, float clamp_min, float clamp_max) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_scale_add_clamp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        scale_factor,
        clamp_min,
        clamp_max
    );

    return output;
}
"""

fused_scale_add_clamp_cpp = "torch::Tensor fused_scale_add_clamp_cuda(torch::Tensor input, float scale_factor, float clamp_min, float clamp_max);"

logsumexp_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

#define BLOCK_SIZE 256

__global__ void logsumexp_mish_kernel(const float* input, float* output, int batch_size, int hidden_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    for (int j = tid; j < hidden_size; j += num_threads) {
        float val = input[row * hidden_size + j];
        if (val > max_val) max_val = val;
    }

    __shared__ float s_max[BLOCK_SIZE];
    s_max[tid] = max_val;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    max_val = s_max[0];

    float local_sum = 0.0f;
    for (int j = tid; j < hidden_size; j += num_threads) {
        float val = input[row * hidden_size + j];
        local_sum += expf(val - max_val);
    }

    __shared__ float s_sum[BLOCK_SIZE];
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    sum_exp = s_sum[0];

    float logsum = max_val + logf(sum_exp);
    float softplus = log1pf(expf(logsum));
    float tanh_val = tanhf(softplus);
    float mish_val = logsum * tanh_val;
    float result = logsum * mish_val;

    if (tid == 0) {
        output[row] = result;
    }
}

torch::Tensor logsumexp_mish_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int hidden_size = input.size(1);
    auto output = torch::empty({batch_size, 1}, input.options());

    int block_size = BLOCK_SIZE;
    int num_blocks = batch_size;

    logsumexp_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size
    );

    return output;
}
"""

logsumexp_mish_cpp = "torch::Tensor logsumexp_mish_cuda(torch::Tensor input);"

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=[fused_scale_add_clamp_cpp, logsumexp_mish_cpp],
    cuda_sources=[fused_scale_add_clamp_source, logsumexp_mish_source],
    functions=["fused_scale_add_clamp_cuda", "logsumexp_mish_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_ops.fused_scale_add_clamp_cuda(x, self.scale_factor, self.clamp_min, self.clamp_max)
        x = self.fused_ops.logsumexp_mish_cuda(x)
        return x