import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LogSumExp and activations
fused_op_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <int BLOCK_SIZE>
__global__ void fused_logsumexp_act_kernel(const float* __restrict__ input, float* __restrict__ output, int batch_size, int features) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;
    const float* row_in = input + batch_idx * features;

    // Online LogSumExp
    float max_val = -INFINITY;
    float sum_val = 0.0f;

    // Grid-stride loop (though here it's block-stride for the row)
    for (int i = tid; i < features; i += BLOCK_SIZE) {
        float val = __ldg(&row_in[i]); // Use __ldg for read-only cache
        float new_max = fmaxf(max_val, val);
        
        // Efficient update
        if (new_max > max_val) {
            sum_val *= expf(max_val - new_max);
            sum_val += expf(val - new_max);
            max_val = new_max;
        } else {
            sum_val += expf(val - max_val);
        }
    }

    // Shared memory reduction
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];

    s_max[tid] = max_val;
    s_sum[tid] = sum_val;

    __syncthreads();

    // Reduce
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float my_max = s_max[tid];
            float other_max = s_max[tid + s];
            float my_sum = s_sum[tid];
            float other_sum = s_sum[tid + s];

            float final_max = fmaxf(my_max, other_max);
            float final_sum = my_sum * expf(my_max - final_max) + other_sum * expf(other_max - final_max);

            s_max[tid] = final_max;
            s_sum[tid] = final_sum;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float lse = logf(s_sum[0]) + s_max[0];
        
        // Activations
        float x = lse;
        
        // LeakyReLU 1
        x = (x > 0.0f) ? x : 0.01f * x;
        // LeakyReLU 2
        x = (x > 0.0f) ? x : 0.01f * x;
        
        // GELU 1
        // x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
        x = x * 0.5f * (1.0f + erff(x * 0.7071067811865475f));
        
        // GELU 2
        x = x * 0.5f * (1.0f + erff(x * 0.7071067811865475f));
        
        output[batch_idx] = x;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input) {
    // input is (Batch, Features)
    int batch_size = input.size(0);
    int features = input.size(1);
    
    auto output = torch::zeros({batch_size, 1}, input.options());
    
    const int block_size = 256;
    const int num_blocks = batch_size; // One block per row
    
    fused_logsumexp_act_kernel<block_size><<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        features
    );
    
    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fused_op = fused_op

    def forward(self, x):
        # Gemm
        x = self.linear(x)
        # Fused LogSumExp + LeakyReLU + LeakyReLU + GELU + GELU
        x = self.fused_op.fused_op_cuda(x)
        return x