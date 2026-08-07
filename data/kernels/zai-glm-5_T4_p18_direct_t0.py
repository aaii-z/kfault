import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Tiling and shared memory approach for efficient Matrix Multiplication
// We compute C = A * B^T. 
// A is (batch_size, in_features), B is (out_features, in_features) -> B^T is (in_features, out_features).
// C is (batch_size, out_features).
// We use a tile size of 32x32.

#define TILE_SIZE 32

__global__ void fused_op_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    const float* __restrict__ bias,
    int batch_size, int in_features, int out_features) {
    
    // 1. Matrix Multiplication: C = A * B^T + bias
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Coordinate of the element in C that this thread is responsible for
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Shared memory for the tile of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float Cvalue = 0.0f;

    // Loop over all the tiles of A and B required to compute the C element
    for (int t = 0; t < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load one tile from A into shared memory
        // A is row-major: A[row][t*TILE_SIZE + tx]
        if (row < batch_size && (t * TILE_SIZE + tx) < in_features) {
            As[ty][tx] = A[row * in_features + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load one tile from B into shared memory
        // B is stored as (out_features, in_features). We need B^T conceptually, 
        // which corresponds to accessing B[col][t*TILE_SIZE + ty] if we treat B as row-major.
        // B[col] is the col-th row of B.
        if (col < out_features && (t * TILE_SIZE + ty) < in_features) {
            Bs[ty][tx] = B[col * in_features + t * TILE_SIZE + ty];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Compute the partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory
    if (row < batch_size && col < out_features) {
        C[row * out_features + col] = Cvalue + bias[col];
    }
}

// Kernel for the reduction operations
// Since the output of the first step is (batch_size, out_features) and subsequent ops reduce to (batch_size, 1),
// we can parallelize over the batch dimension.
// Each block handles one batch element.
__global__ void reduction_kernel(const float* __restrict__ input, float* __restrict__ output, int out_features) {
    // input is (batch_size, out_features)
    // output is (batch_size, 1)
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Pointers to the current row
    const float* row_ptr = input + batch_idx * out_features;
    
    // 2. Summation
    // Using parallel reduction in shared memory
    __shared__ float sdata[1024]; // Max threads per block
    
    float my_sum = 0.0f;
    // Stride loop to handle out_features > block size
    for (int i = tid; i < out_features; i += blockDim.x) {
        my_sum += row_ptr[i];
    }
    sdata[tid] = my_sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float sum_val = sdata[0];
    
    // 3. Max
    // Note: The logic for max(sum_val) over dim 1 of a (batch, 1) tensor is just the value itself.
    // The logic for mean(sum_val) is sum_val / 1.
    // The logic for logsumexp(sum_val) is log(exp(sum_val)) = sum_val.
    // Since the tensor becomes (batch_size, 1) after the first sum, the subsequent operations
    // on dimension 1 are effectively identity operations on the scalar value.
    // max(v) = v, mean(v) = v, logsumexp(v) = v.
    // So we just write the sum_val to output.
    
    if (tid == 0) {
        output[batch_idx] = sum_val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0); // weight is (out_features, in_features)

    // Allocate intermediate tensor for MatMul result
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto matmul_out = torch::empty({batch_size, out_features}, options);
    
    // 1. Launch Matrix Multiplication Kernel
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((out_features + TILE_SIZE - 1) / TILE_SIZE, 
                 (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_op_kernel<<<dimGrid, dimBlock>>>(
        x.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        matmul_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size, in_features, out_features
    );

    // Allocate final output tensor
    auto out = torch::empty({batch_size, 1}, options);

    // 2. Launch Reduction Kernel
    // We use a block size that fits in shared memory and covers the reduction efficiently.
    int threads = 1024; 
    int blocks = batch_size;
    
    reduction_kernel<<<blocks, threads>>>(
        matmul_out.data_ptr<float>(), 
        out.data_ptr<float>(), 
        out_features
    );

    return out;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # We keep the Linear layer to hold the weights and bias parameters.
        # We will use its weights in our custom CUDA kernel.
        self.linear = nn.Linear(in_features, out_features)
        self.fused_op = fused_op

    def forward(self, x):
        # Call the custom fused CUDA kernel
        # We pass the weight and bias from the Linear layer
        return self.fused_op.fused_op_cuda(x, self.linear.weight, self.linear.bias)