import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul, dropout, and softmax
fused_matmul_dropout_softmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>

// Helper function to calculate the linear index for a given row and column
__device__ __forceinline__ int get_linear_index(int row, int col, int stride) {
    return row * stride + col;
}

// Kernel for fused matrix multiplication, dropout, and softmax
__global__ void fused_matmul_dropout_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float dropout_prob,
    unsigned long long seed
) {
    // Each block handles one row of the output
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Shared memory for the row data to perform softmax
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;

    // Pointers to the current row of input and output
    const float* input_row = input + row * in_features;
    float* output_row = output + row * out_features;

    // Initialize cuRAND state for dropout
    curandState state;
    curand_init(seed, row, tid, &state);

    // Step 1: Matrix Multiplication (output_row = input_row * weight^T + bias)
    // Each thread computes a portion of the output row
    for (int col = tid; col < out_features; col += blockDim.x) {
        float sum = 0.0f;
        if (bias != nullptr) {
            sum = bias[col];
        }
        
        // Dot product of input_row and weight column
        for (int k = 0; k < in_features; ++k) {
            sum += input_row[k] * weight[col * in_features + k];
        }

        // Step 2: Apply Dropout
        // We apply dropout before softmax as per the original model logic
        if (dropout_prob > 0.0f) {
            float rand_val = curand_uniform(&state);
            if (rand_val < dropout_prob) {
                sum = 0.0f;
            } else {
                // Scale the output during training to maintain expected value
                sum /= (1.0f - dropout_prob);
            }
        }
        
        shared_data[col] = sum;
    }
    __syncthreads();

    // Step 3: Softmax
    // Find the maximum value in the row for numerical stability
    float max_val = -FLT_MAX;
    for (int i = tid; i < out_features; i += blockDim.x) {
        max_val = max(max_val, shared_data[i]);
    }

    // Reduce max_val within the block
    __shared__ float block_max;
    if (tid == 0) block_max = -FLT_MAX;
    __syncthreads();

    // Atomic max operation to find the global max for the row
    atomicMax((int*)&block_max, __float_as_int(max_val));
    __syncthreads();

    // Broadcast the max value to all threads
    max_val = block_max;

    // Compute exponentials and sum them up
    float sum_exp = 0.0f;
    for (int i = tid; i < out_features; i += blockDim.x) {
        shared_data[i] = expf(shared_data[i] - max_val);
        sum_exp += shared_data[i];
    }

    // Reduce sum_exp within the block
    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();

    // Atomic add to find the global sum for the row
    atomicAdd(&block_sum, sum_exp);
    __syncthreads();

    // Normalize by the sum
    for (int i = tid; i < out_features; i += blockDim.x) {
        output_row[i] = shared_data[i] / block_sum;
    }
}

torch::Tensor fused_matmul_dropout_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float dropout_prob
) {
    // Check inputs
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");

    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    // Create output tensor
    auto output = torch::empty({batch_size, out_features}, input.options());

    // Launch kernel
    int threads_per_block = 256; // Can be tuned
    int blocks = batch_size;
    size_t shared_mem_size = out_features * sizeof(float);

    // Generate a random seed for dropout
    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();

    fused_matmul_dropout_softmax_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        dropout_prob,
        seed
    );

    return output;
}
"""

fused_matmul_dropout_softmax_cpp_source = (
    "torch::Tensor fused_matmul_dropout_softmax_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float dropout_prob);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_matmul_dropout_softmax_cpp_source,
    cuda_sources=fused_matmul_dropout_softmax_source,
    functions=["fused_matmul_dropout_softmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcurand"],
)


class ModelNew(nn.Module):
    """
    Optimized model that replaces matmul, dropout, and softmax with a single custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        # We still define the weight and bias as parameters so they are learned during training.
        # We transpose the weight here because the custom kernel expects it in a specific layout.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.dropout_p = dropout_p
        
        # Initialize parameters (mimicking nn.Linear's default initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Call the custom CUDA kernel
        return self.fused_ops.fused_matmul_dropout_softmax_cuda(x, self.weight, self.bias, self.dropout_p)

import math