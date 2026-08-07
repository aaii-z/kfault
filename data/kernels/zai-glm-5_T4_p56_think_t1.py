python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear -> Sigmoid -> Sum
fused_linear_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Define warp size for clarity
#define WARP_SIZE 32

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
    
    // Each block processes one row of the batch (one sample)
    int row_idx = blockIdx.x;
    if (row_idx >= batch_size) return;

    // Thread identification
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Pointers for the current row
    const float* input_row = input + row_idx * input_size;
    
    // Register for accumulating the sum of sigmoid outputs
    float total_sum = 0.0f;

    // Iterate over the hidden dimension (columns of weight matrix / output features)
    // We parallelize this loop across warps
    // Each warp computes the sigmoid output for one hidden unit
    for (int hidden_idx = warp_id; hidden_idx < hidden_size; hidden_idx += num_warps) {
        const float* weight_row = weight + hidden_idx * input_size;
        
        // Compute dot product: input . weight_row
        float dot = 0.0f;
        
        // Iterate over input dimension (K) in steps of warp size
        // Each thread processes a strided subset of the input/weight elements
        for (int k = lane_id; k < input_size; k += WARP_SIZE) {
            dot += input_row[k] * weight_row[k];
        }

        // Reduce within the warp to get the full dot product
        dot = warp_reduce_sum(dot);

        // The first lane of the warp computes the sigmoid and accumulates
        if (lane_id == 0) {
            float val = dot;
            if (bias != nullptr) {
                val += bias[hidden_idx];
            }
            // Sigmoid activation
            float sigmoid_val = 1.0f / (1.0f + expf(-val));
            total_sum += sigmoid_val;
        }
    }

    // Now we need to sum the partial sums from each warp
    // Use shared memory to gather results from each warp leader
    __shared__ float shared_sums[32]; // Assuming max 32 warps (1024 threads)
    
    // Only the first lane of each warp writes to shared memory
    if (lane_id == 0) {
        shared_sums[warp_id] = total_sum;
    }
    __syncthreads();

    // Let the first warp perform the final reduction
    if (warp_id == 0) {
        float final_sum = 0.0f;
        // Read from shared memory using the first warp
        // Note