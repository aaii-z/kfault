import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear+sigmoid+sum
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

__global__ void fused_linear_sigmoid_sum_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size) {
    
    extern __shared__ float shared_mem[];
    float* row_cache = shared_mem;
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Load input row into shared memory
    if (tid < input_size) {
        row_cache[tid] = input[row * input_size + tid];
    }
    __syncthreads();

    // Compute matrix multiplication and sigmoid for this row
    float sum = 0.0f;
    for (int col = tid; col < hidden_size; col += stride) {
        float dot_product = 0.0f;
        for (int k = 0; k < input_size; k++) {
            dot_product += row_cache[k] * weight[col * input_size + k];
        }
        float activated = 1.0f / (1.0f + expf(-(dot_product + bias[col])));
        sum += activated;
    }

    // Parallel reduction within block
    __shared__ float block_sum[256];
    block_sum[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_sum[tid] += block_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = block_sum[0];
    }
}

torch::Tensor fused_linear_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);

    auto output = torch::zeros({batch_size, 1}, input.options());

    dim3 blocks(batch_size);
    dim3 threads(256);
    size_t shared_mem_size = input_size * sizeof(float);

    fused_linear_sigmoid_sum_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_linear_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and bias like nn.Linear
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return fused_op.fused_linear_sigmoid_sum_cuda(x, self.weight, self.bias)