import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <float.h>

__global__ void logsumexp_kernel(const float* input, float* output, int batch_size, int N) {
    int i = blockIdx.x;
    if (i >= batch_size) return;
    
    const float* row = input + i * N;
    float max_val = -INFINITY;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float val = row[j];
        if (val > max_val) max_val = val;
    }
    
    __shared__ float shared_max[256];
    shared_max[threadIdx.x] = max_val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = shared_max[threadIdx.x + stride];
            if (other > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }
    max_val = shared_max[0];
    
    float sum_exp = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float val = expf(row[j] - max_val);
        sum_exp += val;
    }
    
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum_exp;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    sum_exp = shared_sum[0];
    
    float result = logf(sum_exp) + max_val;
    output[i] = result;
}

torch::Tensor logsumexp_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int N = input.size(1);
    auto output = torch::empty({batch_size, 1}, input.options());
    
    const int block_size = 256;
    logsumexp_kernel<<<batch_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, N);
    return output;
}
"""

logsumexp_cpp = "torch::Tensor logsumexp_cuda(torch::Tensor input);"

fused_activations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_activations_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // First LeakyReLU
        val = (val > 0) ? val : 0.01f * val;
        // Second LeakyReLU
        val = (val > 0) ? val : 0.01f * val;
        // First GELU
        float c = 0.044715f;
        float k = 0.7978845608028654f;
        float x3 = val * val * val;
        float inner = k * (val + c * x3);
        float tanh_inner = tanhf(inner);
        val = 0.5f * val * (1.0f + tanh_inner);
        // Second GELU
        x3 = val * val * val;
        inner = k * (val + c * x3);
        tanh_inner = tanhf(inner);
        val = 0.5f * val * (1.0f + tanh_inner);
        output[idx] = val;
    }
}

torch::Tensor fused_activations_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_activations_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

fused_activations_cpp = "torch::Tensor fused_activations_cuda(torch::Tensor input);"

logsumexp = load_inline(
    name="logsumexp",
    cpp_sources=logsumexp_cpp,
    cuda_sources=logsumexp_source,
    functions=["logsumexp_cuda"],
    verbose=False,
)

fused_activations = load_inline(
    name="fused_activations",
    cpp_sources=fused_activations_cpp,
    cuda_sources=fused_activations_source,
    functions=["fused_activations_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.logsumexp = logsumexp
        self.fused_activations = fused_activations

    def forward(self, x):
        x = self.linear(x)
        x = self.logsumexp.logsumexp_cuda(x)
        x = self.fused_activations.fused_activations_cuda(x)
        return x