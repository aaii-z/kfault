import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear, min, and subtraction
fused_linear_min_sub_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Fused kernel: output = min(x * W^T + b, constant) - constant
// Using a tiled matrix multiplication approach.
// Block size: 16x16 threads.
// Grid: (batch_size / 16, out_features / 16)
// Shared memory used for tiling input and weight matrices.

template <int TILE_SIZE>
__global__ void fused_linear_min_sub_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float constant,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    // Shared memory for input tile (TILE_SIZE x TILE_SIZE)
    __shared__ float s_input[TILE_SIZE][TILE_SIZE];
    
    // Shared memory for weight tile (TILE_SIZE x TILE_SIZE)
    // Padding by 1 to avoid bank conflicts during transposed access
    __shared__ float s_weight[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global indices for the output element this thread computes
    int batch_idx = blockIdx.x * TILE_SIZE + ty;
    int out_feat_idx = blockIdx.y * TILE_SIZE + tx;

    float acc = 0.0f;

    // Loop over in_features in tiles
    for (int k = 0; k < in_features; k += TILE_SIZE) {
        // Load input tile into shared memory
        // Input is (batch, in_features)
        // Thread (ty, tx) loads input[batch_idx, k + tx]
        if (batch_idx < batch_size && (k + tx) < in_features) {
            s_input[ty][tx] = input[batch_idx * in_features + k + tx];
        } else {
            s_input[ty][tx] = 0.0f;
        }

        // Load weight tile into shared memory
        // Weight is (out_features, in_features)
        // We want to load weight[out_feat_idx, k + i] for the computation.
        // To ensure coalesced global memory access, we load a row of the weight matrix.
        // Thread (ty, tx) loads weight[blockIdx.y * TILE_SIZE + ty, k + tx].
        // This corresponds to s_weight[ty][tx].
        int weight_row = blockIdx.y * TILE_SIZE + ty;
        if (weight_row < out_features && (k + tx) < in_features) {
            s_weight[ty][tx] = weight[weight_row * in_features + k + tx];
        } else {
            s_weight[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        // We need sum(input[batch_idx, k + i] * weight[out_feat_idx, k + i])
        // s_input[ty][i] holds input[batch_idx, k + i]
        // s_weight[tx][i] holds weight[out_feat_idx, k + i] (accessed transposed)
        // Note: out_feat_idx corresponds to tx, so we access row tx of s_weight.
        // The padding in s_weight avoids bank conflicts here.
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += s_input[ty][i] * s_weight[tx][i];
        }

        __syncthreads();
    }

    // Add bias and apply fused operations
    if (batch_idx < batch_size && out_feat_idx < out_features) {
        float val = acc;
        if (bias != nullptr) {
            val += bias[out_feat_idx];
        }
        
        // Fused min and subtraction
        // val = min(val, constant) - constant
        // Equivalent to: if val < constant, val - constant; else 0.
        // Using fminf for robustness
        val = fminf(val, constant) - constant;
        
        output[batch_idx * out_features + out_feat_idx] = val;
    }
}

// Helper function to determine grid size
int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// C interface function
torch::Tensor fused_linear_min_sub_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float constant) {
    // Ensure inputs are contiguous and on CUDA
    input = input.contiguous();
    weight = weight.contiguous();
    // Bias might be optional or empty, handle safely
    const float* bias_ptr = bias.defined() && bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;

    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_features}, input.options());

    // Launch kernel
    const int TILE_SIZE = 16; // 16x16 block
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(ceil_div(batch_size, TILE_SIZE), ceil_div(out_features, TILE_SIZE));

    fused_linear_min_sub_kernel<TILE_SIZE><<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        constant,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_linear_min_sub_cpp_source = (
    "torch::Tensor fused_linear_min_sub_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float constant);"
)

# Compile the inline CUDA code
fused_linear_min_sub = load_inline(
    name="fused_linear_min_sub",
    cpp_sources=fused_linear_min_sub_cpp_source,
    cuda_sources=fused_linear_min_sub_source,
    functions=["fused_linear_min_sub_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.constant = constant
        
        # Define weights and bias as parameters to mimic nn.Linear
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters (mimicking nn.Linear initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fused_op = fused_linear_min_sub

    def forward(self, x):
        # Call the custom CUDA kernel
        return self.fused_op.fused_linear_min_sub_cuda(x, self.weight, self.bias, self.constant)