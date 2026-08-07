import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: Matmul -> AvgPool -> GELU -> Scale -> Max
# Corrected to handle AvgPool1d behavior (padding=0, stride=kernel_size) and proper GELU approximation.
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Constants for GELU approximation (matching PyTorch behavior closely)
constexpr float GELU_K = 0.7978845608028654f; // sqrt(2/pi)
constexpr float GELU_C = 0.044715f;

// Device function for approximate GELU
__device__ __forceinline__ float gelu_approx(float x) {
    // x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    float inner = GELU_K * (x + GELU_C * x3);
    return x * 0.5f * (1.0f + tanhf(inner));
}

// Kernel for the fused operation
__global__ void fused_op_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, 
    int in_features, 
    int out_features, 
    int pool_kernel_size, 
    float scale_factor) {
    
    // Each block handles one sample in the batch
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Shared memory for the intermediate result after matmul
    extern __shared__ float s_matmul[];

    // Thread index
    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    // Pointers for the current batch sample
    const float* input_row = input + batch_idx * in_features;
    
    // Step 1: Matmul (Linear)
    // Distribute 'out_features' across threads in the block.
    for (int o_idx = tid; o_idx < out_features; o_idx += block_dim) {
        float sum = 0.0f;
        const float* weight_row = weight + o_idx * in_features;
        
        // Dot product
        // Using a simple loop. Vectorization could be added for more speed.
        for (int k = 0; k < in_features; ++k) {
            sum += input_row[k] * weight_row[k];
        }
        sum += bias[o_idx];
        s_matmul[o_idx] = sum;
    }
    
    __syncthreads();
    
    // Step 2, 3, 4: AvgPool -> GELU -> Scale -> Max
    // AvgPool1d with default parameters divides dimension by pool_kernel_size.
    // It computes the average of non-overlapping windows.
    int pooled_out_features = out_features / pool_kernel_size;
    
    // Each thread computes one or more pooled output elements, applies GELU+Scale,
    // and updates a local max.
    float local_max = -INFINITY;
    
    for (int p_idx = tid; p_idx < pooled_out_features; p_idx += block_dim) {
        // AvgPool
        float pool_sum = 0.0f;
        int start_idx = p_idx * pool_kernel_size;
        for (int k = 0; k < pool_kernel_size; ++k) {
            pool_sum += s_matmul[start_idx + k];
        }
        float avg = pool_sum / pool_kernel_size;
        
        // GELU
        float g = gelu_approx(avg);
        
        // Scale
        float scaled = g * scale_factor;
        
        // Update local max
        if (scaled > local_max) {
            local_max = scaled;
        }
    }
    
    // Step 5: Final Max reduction across threads in the block
    // We reuse the beginning of s_matmul for this reduction.
    float* s_partial_maxes = s_matmul; 
    
    s_partial_maxes[tid] = local_max;
    __syncthreads();
    
    // Parallel reduction
    // Assuming block_dim is power of 2 (256 or 512)
    for (unsigned int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial_maxes[tid] = fmaxf(s_partial_maxes[tid], s_partial_maxes[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[batch_idx] = s_partial_maxes[0];
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int pool_kernel_size, 
    float scale_factor) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0); // weight is (out_features, in_features)
    
    auto output = torch::empty({batch_size}, input.options());
    
    // Shared memory size: matmul output
    size_t shared_mem_size = out_features * sizeof(float);
    
    const int block_size = 256; 
    const int num_blocks = batch_size;
    
    fused_op_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        pool_kernel_size,
        scale_factor
    );
    
    return output;
}
"""

fused_kernel_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int pool_kernel_size, float scale_factor);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        # Define weights to match nn.Linear (out_features, in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        
        # Initialize parameters similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(x, self.weight, self.bias, self.pool_kernel_size, self.scale_factor)