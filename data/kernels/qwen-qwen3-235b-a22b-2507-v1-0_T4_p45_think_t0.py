import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_add_sigmoid_kernel(const float* matmul, const float* bias, float* out, int size, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int i = idx / hidden_size;
        int j = idx % hidden_size;
        float val = matmul[idx] + bias[j];
        out[idx] = 1.0f / (1.0f + expf(-val));
    }
}

torch::Tensor fused_add_sigmoid_cuda(torch::Tensor matmul, torch::Tensor bias) {
    auto batch_size = matmul.size(0);
    auto hidden_size = matmul.size(1);
    auto size = batch_size * hidden_size;
    auto out = torch::empty_like(matmul);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_add_sigmoid_kernel<<<num_blocks, block_size>>>(matmul.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), size, hidden_size);
    return out;
}

__global__ void logsumexp_kernel(const float* matmul, const float* bias, float* out, int batch_size, int output_size) {
    int i = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared[];

    float val = matmul[i * output_size + tid] + bias[tid];
    shared[tid] = val;
    __syncthreads();

    for (int stride = output_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = shared[tid + stride];
            if (other > shared[tid]) {
                shared[tid] = other;
            }
        }
        __syncthreads();
    }
    float max_val = shared[0];

    float exp_val = expf(val - max_val);
    shared[tid] = exp_val;
    __syncthreads();

    for (int stride = output_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = shared[0];

    if (tid == 0) {
        out[i] = logf(sum_exp) + max_val;
    }
}

torch::Tensor custom_logsumexp_cuda(torch::Tensor matmul, torch::Tensor bias) {
    auto batch_size = matmul.size(0);
    auto output_size = matmul.size(1);
    auto out = torch::empty({batch_size}, matmul.options());

    const int block_size = output_size;
    const int num_blocks = batch_size;

    logsumexp_kernel<<<num_blocks, block_size, output_size * sizeof(float)>>>(
        matmul.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch_size, output_size);
    return out;
}
"""

cpp_sources = """
torch::Tensor fused_add_sigmoid_cuda(torch::Tensor matmul, torch::Tensor bias);
torch::Tensor custom_logsumexp_cuda(torch::Tensor matmul, torch::Tensor bias);
"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_add_sigmoid_cuda", "custom_logsumexp_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.fused_add_sigmoid = custom_ops.fused_add_sigmoid_cuda
        self.custom_logsumexp = custom_ops.custom_logsumexp_cuda

    def forward(self, x):
        x = x @ self.linear1.weight.t()
        x = self.fused_add_sigmoid(x, self.linear1.bias)
        x = x @ self.linear2.weight.t()
        x = self.custom_logsumexp(x, self.linear2.bias)
        return x