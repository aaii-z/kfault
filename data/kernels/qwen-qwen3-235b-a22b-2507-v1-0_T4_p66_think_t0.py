import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for fused linear layer + softmax
fused_linear_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

__global__ void fused_linear_softmax_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int batch_size,
    int in_features,
    int out_features
) {
    int i = blockIdx.x;  // batch index
    if (i >= batch_size) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Each thread processes a chunk of output features
    int j_per_thread = (out_features + num_threads - 1) / num_threads;
    int j_start = tid * j_per_thread;
    int j_end = min((tid + 1) * j_per_thread, out_features);

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    // First pass: compute linear output and find local max
    for (int j = j_start; j < j_end; j++) {
        float val = bias[j];
        for (int k = 0; k < in_features; k++) {
            val += x[i * in_features + k] * weight[j * in_features + k];
        }
        if (val > local_max) {
            local_max = val;
        }
    }

    // Reduce to find global max
    __shared__ float shared_max[256];
    shared_max[tid] = local_max;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = shared_max[tid + stride];
            if (other > shared_max[tid]) {
                shared_max[tid] = other;
            }
        }
        __syncthreads();
    }
    float global_max = shared_max[0];

    // Second pass: compute exp(x - max) and local sum
    local_sum = 0.0f;
    for (int j = j_start; j < j_end; j++) {
        float val = bias[j];
        for (int k = 0; k < in_features; k++) {
            val += x[i * in_features + k] * weight[j * in_features + k];
        }
        float exp_val = expf(val - global_max);
        local_sum += exp_val;
        out[i * out_features + j] = exp_val;
    }

    // Reduce to find global sum
    __shared__ float shared_sum[256];
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    float global_sum = shared_sum[0];

    // Third pass: normalize by sum
    for (int j = j_start; j < j_end; j++) {
        out[i * out_features + j] /= global_sum;
    }
}

torch::Tensor fused_linear_softmax_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = bias.size(0);

    auto out = torch::empty({batch_size, out_features}, x.options());

    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);

    fused_linear_softmax_kernel<<<grid, block, 2 * block_size * sizeof(float)>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return out;
}
"""

fused_linear_softmax_cpp_source = (
    "torch::Tensor fused_linear_softmax_cuda("
    "    torch::Tensor x,"
    "    torch::Tensor weight,"
    "    torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_linear_softmax = load_inline(
    name="fused_linear_softmax",
    cpp_sources=fused_linear_softmax_cpp_source,
    cuda_sources=fused_linear_softmax_source,
    functions=["fused_linear_softmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses linear layer and softmax into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.dropout = nn.Dropout(dropout_p)
        
        # Initialize parameters (same as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_linear_softmax = fused_linear_softmax

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.fused_linear_softmax.fused_linear_softmax_cuda(x, self.weight, self.bias)
        x = self.dropout(x)
        return x

batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, dropout_p]