import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear -> Sigmoid -> Sum
fused_linear_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define TILE_SIZE 32 // Each block processes TILE_SIZE hidden units

// Helper function for warp-level reduction
__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fused_linear_sigmoid_sum_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int batch_size, int input_size, int hidden_size) {
    
    // Block index
    int batch_idx = blockIdx.x;
    int hidden_tile_idx = blockIdx.y;
    
    // Thread index
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Number of warps in block
    int num_warps = blockDim.x / WARP_SIZE; // Should be TILE_SIZE

    // Current hidden unit index handled by this warp
    int hidden_idx_base = hidden_tile_idx * TILE_SIZE;
    int hidden_idx = hidden_idx_base + warp_id;

    if (batch_idx >= batch_size) return;

    // Accumulator for the dot product
    float dot = 0.0f;
    
    // Pointers
    const float* input_row = input + batch_idx * input_size;
    
    // Compute dot product for the assigned hidden unit
    // Only compute if hidden_idx is valid
    if (hidden_idx < hidden_size) {
        const float* weight_row = weight + hidden_idx * input_size;
        
        // Strided loop over input dimension
        // Threads in a warp access consecutive memory locations for coalescing
        for (int k = lane_id; k < input_size; k += WARP_SIZE) {
            dot += input_row[k] * weight_row[k];
        }

        // Reduce within warp
        dot = warp_reduce_sum(dot);
    }

    // Compute sigmoid and store in shared memory
    // Shared memory to hold partial sums for the block
    __shared__ float shared_sums[TILE_SIZE];
    
    if (lane_id == 0) {
        if (hidden_idx < hidden_size) {
            if (bias != nullptr) {
                dot += bias[hidden_idx];
            }
            // Sigmoid activation
            float sigmoid_val = 1.0f / (1.0f + expf(-dot));
            shared_sums[warp_id] = sigmoid_val;
        } else {
            shared_sums[warp_id] = 0.0f;
        }
    }
    __syncthreads();

    // Reduce across warps (block reduction)
    // Use the first warp to sum up results from shared memory
    if (warp_id == 0) {
        float block_sum = 0.0f;
        // Each lane in the first warp sums a subset of the shared memory
        // Since TILE_SIZE == WARP_SIZE (32), we can map lane_id to warp_id index
        if (lane_id < num_warps) {
             block_sum = shared_sums[lane_id];
        }
        
        // Reduce within the warp
        block_sum = warp_reduce_sum(block_sum);
        
        // Write to global memory using atomic add
        if (lane_id == 0) {
            atomicAdd(output + batch_idx, block_sum);
        }
    }
}

torch::Tensor fused_linear_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0); // weight is (hidden_size, input_size)

    // Initialize output tensor with zeros
    auto output = torch::zeros({batch_size, 1}, input.options());

    const int block_size = TILE_SIZE * WARP_SIZE; // 32 * 32 = 1024 threads
    const int num_tiles = (hidden_size + TILE_SIZE - 1) / TILE_SIZE;
    
    dim3 grid(batch_size, num_tiles);
    dim3 block(block_size);

    // Launch kernel
    fused_linear_sigmoid_sum_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr, 
        output.data_ptr<float>(),
        batch_size, input_size, hidden_size
    );

    return output;
}
"""

fused_linear_sigmoid_sum_cpp_source = (
    "torch::Tensor fused_linear_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_linear_sigmoid_sum = load_inline(
    name="fused_linear_sigmoid_sum",
    cpp_sources=fused_linear_sigmoid_sum_cpp_source,
    cuda_sources=fused_linear_sigmoid_sum_source,
    functions=["fused_linear_sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused Linear -> Sigmoid -> Sum.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.fused_op = fused_linear_sigmoid_sum

    def forward(self, x):
        return self.fused_op.fused_linear_sigmoid_sum_cuda(x, self.linear.weight, self.linear.bias)