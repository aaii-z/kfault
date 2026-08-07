import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul, dropout, and softmax
fused_matmul_dropout_softmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <limits>

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

    // Dynamic shared memory for the row data
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    // Pointers to the current row of input and output
    const float* input_row = input + row * in_features;
    float* output_row = output + row * out_features;

    // Initialize cuRAND state for dropout
    curandState state;
    curand_init(seed, row, tid, &state);

    // Step 1: Matrix Multiplication (output_row = input_row * weight^T + bias)
    // Each thread computes a portion of the output row
    for (int col = tid; col < out_features; col += block_dim) {
        float sum = 0.0f;
        if (bias != nullptr) {
            sum = bias[col];
        }
        
        // Dot product of input_row and weight column
        for (int k = 0; k < in_features; ++k) {
            sum += input_row[k] * weight[col * in_features + k];
        }

        // Step 2: Apply Dropout
        if (dropout_prob > 0.0f) {
            float rand_val = curand_uniform(&state);
            if (rand_val < dropout_prob) {
                sum = 0.0f;
            } else {
                sum /= (1.0f - dropout_prob);
            }
        }
        
        shared_data[col] = sum;
    }
    __syncthreads();

    // Step 3: Softmax
    // Find the maximum value in the row for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = tid; i < out_features; i += block_dim) {
        max_val = fmaxf(max_val, shared_data[i]);
    }

    // Reduce max_val within the warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    
    // Broadcast the max value within the warp (lane 0 has the value)
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // If block_dim > 32, we need to reduce across warps.
    // We use the first column of shared_data (which is already processed) as scratchpad.
    if (block_dim > 32) {
        if (tid % 32 == 0) {
             shared_data[tid / 32] = max_val;
        }
        __syncthreads();
        
        // The first warp reads the partial maxes
        if (tid < block_dim / 32) {
            max_val = shared_data[tid];
        } else {
            max_val = -std::numeric_limits<float>::infinity();
        }
        
        // Reduce within the first warp again
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        
        // Broadcast final max to all threads in the block
        if (tid == 0) {
            shared_data[0] = max_val;
        }
        __syncthreads();
        max_val = shared_data[0];
    }

    // Compute exponentials and sum them up
    float sum_exp = 0.0f;
    for (int i = tid; i < out_features; i += block_dim) {
        shared_data[i] = expf(shared_data[i] - max_val);
        sum_exp += shared_data[i];
    }

    // Reduce sum_exp within the warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

    // If block_dim > 32, reduce across warps
    if (block_dim > 32) {
        if (tid % 32 == 0) {
             shared_data[tid / 32] = sum_exp;
        }
        __syncthreads();
        
        if (tid < block_dim / 32) {
            sum_exp = shared_data[tid];
        } else {
            sum_exp = 0.0f;
        }
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
        }
        
        if (tid == 0) {
            shared_data[0] = sum_exp;
        }
        __syncthreads();
        sum_exp = shared_data[0];
    }

    // Normalize by the sum
    for (int i = tid; i < out_features; i += block_dim) {
        output_row[i] = shared_data[i] / sum_exp;
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
    int threads_per_block = 256; 
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