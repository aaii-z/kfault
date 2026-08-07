import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: GEMM + Swish + Divide + Clamp + Tanh + Clamp
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Block size for CUDA kernels
constexpr int BLOCK_SIZE = 256;

// Fused kernel for Swish, Divide, Clamp, Tanh, Clamp
__global__ void fused_post_gemm_kernel(const float* __restrict__ gemm_out, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = gemm_out[idx];
        // Swish: x * sigmoid(x)
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        val = val * sigmoid_val;
        // Divide by 2.0
        val = val * 0.5f;
        // Clamp between -1.0 and 1.0
        val = fmaxf(-1.0f, fminf(1.0f, val));
        // Tanh
        val = tanhf(val);
        // Clamp between -1.0 and 1.0 (Note: tanh output is already in [-1, 1], so this is redundant but kept for correctness)
        val = fmaxf(-1.0f, fminf(1.0f, val));
        out[idx] = val;
    }
}

// Fused kernel for GEMM + Post-processing
// Uses shared memory and register blocking for optimized GEMM
__global__ void fused_gemm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    // Shared memory for weight tile
    __shared__ float s_weight[BLOCK_SIZE * BLOCK_SIZE];

    int batch_idx = blockIdx.y;
    int out_idx_local = threadIdx.x;

    // Global output index for this thread
    int out_idx_global = blockIdx.x * BLOCK_SIZE + out_idx_local;

    if (batch_idx >= batch_size) return;

    // Pointers to the current row of input and current column of weight
    const float* input_row = input + batch_idx * in_features;
    const float* weight_col = weight + out_idx_global * in_features;

    float sum = 0.0f;
    if (bias != nullptr) {
        sum = bias[out_idx_global];
    }

    // Loop over input features in tiles
    for (int i = 0; i < in_features; i += BLOCK_SIZE) {
        // Load weight tile into shared memory
        // Each thread loads one element
        int weight_row = out_idx_global;
        int weight_col_idx = i + threadIdx.x;
        
        if (weight_row < out_features && weight_col_idx < in_features) {
            s_weight[threadIdx.y * BLOCK_SIZE + threadIdx.x] = weight[weight_row * in_features + weight_col_idx];
        } else {
            s_weight[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute partial dot product
        // Each thread computes one output element
        // We iterate over the loaded tile
        if (out_idx_global < out_features) {
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                int input_idx = i + k;
                if (input_idx < in_features) {
                    sum += input_row[input_idx] * s_weight[threadIdx.x * BLOCK_SIZE + k]; // Transposed access logic needs care
                    // Correct logic: weight is (out_features, in_features). We are computing dot product of input row and weight column.
                    // s_weight loaded as (BLOCK_SIZE rows, BLOCK_SIZE cols) block from weight.
                    // Wait, the loading logic above is slightly flawed for standard GEMM tiling.
                    // Let's simplify: We are computing output[batch_idx, out_idx_global].
                    // This is dot(input[batch_idx, :], weight[out_idx_global, :]).
                    // We can just iterate through the weight row directly without complex tiling for simplicity and correctness,
                    // or use a standard tiled GEMM approach.
                    // Given the constraints and potential for complexity, a simpler vectorized loop is safer to implement correctly in raw CUDA here.
                }
            }
        }
        __syncthreads();
    }
    // The tiling logic above is getting complicated to write correctly in a short time without errors.
    // Let's fallback to a simpler, yet effective approach: 
    // Each thread computes one output element. We iterate through the input/weight vectors.
    // This is memory-bound but correct. We rely on L1/L2 cache.
    // We will use the post-processed kernel separately for clarity and correctness, or fuse it here.
    // Let's fuse it here.
}

// Simpler, correct fused kernel
__global__ void fused_gemm_swish_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int row = idx / out_features;
        int col = idx % out_features;
        
        float sum = 0.0f;
        const float* input_row = input + row * in_features;
        const float* weight_row = weight + col * in_features; // Weight is transposed in memory layout for Linear? No, Linear weight is (out, in).
        
        // Dot product
        for (int k = 0; k < in_features; ++k) {
            sum += input_row[k] * weight_row[k];
        }
        
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Fused post-processing
        // Swish
        float sigmoid_val = 1.0f / (1.0f + expf(-sum));
        sum = sum * sigmoid_val;
        // Divide
        sum = sum * 0.5f;
        // Clamp
        sum = fmaxf(-1.0f, fminf(1.0f, sum));
        // Tanh
        sum = tanhf(sum);
        // Clamp
        sum = fmaxf(-1.0f, fminf(1.0f, sum));
        
        output[idx] = sum;
    }
}

torch::Tensor fused_gemm_swish_tanh_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    const int block_size = 256;
    int total_elements = batch_size * out_features;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_gemm_swish_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_gemm_swish_tanh_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_gemm_swish_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel fusing GEMM, Swish, Divide, Clamp, Tanh, and Clamp.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight and bias parameters similar to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_gemm_swish_tanh_cuda(x, self.weight, self.bias)

import math # Ensure math is imported for initialization