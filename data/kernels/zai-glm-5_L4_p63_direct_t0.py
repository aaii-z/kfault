import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear, ReLU, and Division
fused_linear_relu_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for ReLU and Division
__global__ void relu_div_kernel(const float* __restrict__ x, float* __restrict__ out, int size, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        val = val > 0.0f ? val : 0.0f;
        out[idx] = val / divisor;
    }
}

// Kernel for fused GEMM (Matrix Multiplication)
// Computes C = A * B^T
// A: (M, K), B: (N, K) -> B is stored transposed conceptually or physically. 
// PyTorch Linear stores weights as (out_features, in_features), so B is (N, K) in memory.
// We compute C(m, n) = sum_k A(m, k) * B(n, k)
__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Calculate the row and column of the C element to process
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension

    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Optimization: Use shared memory for B tile (optional but good for performance)
        // For simplicity and robustness in a single file without complex tuning, 
        // we will use a slightly optimized global memory access pattern or just direct access.
        // Given the constraints, a standard grid-stride or block-wise loop is sufficient.
        // However, for a basic high-performance example, we assume coalesced access on A.
        
        // Pointers to the start of the row in A and column in B
        const float* A_row = A + row * K;
        const float* B_row = B + col * K; // B is (N, K), so row 'col' of B corresponds to column 'col' of result

        // Vectorized memory access might be tricky without alignment guarantees, 
        // so we stick to float operations.
        for (int k = 0; k < K; ++k) {
            sum += A_row[k] * B_row[k];
        }
        
        // We write the result of GEMM to C. 
        // The ReLU/Div kernel will process C.
        // Alternatively, we could fuse it here, but separating allows modularity and checking.
        // Let's fuse it here to save a kernel launch and memory bandwidth.
        
        // Fused ReLU and Division
        sum = sum > 0.0f ? sum : 0.0f;
        C[row * N + col] = sum; // Division applied in a separate pass or here?
        // The prompt implies replacing operators. 
        // Let's keep the structure clean: GEMM writes to a buffer, then ReLU/Div kernel.
        // Actually, fusing ReLU/Div into GEMM is better (write once).
        // But let's stick to the prompt's structure of replacing operators.
        // Wait, the prompt asks to replace PyTorch operators.
        // Replacing `linear` + `relu` + `div` with a fused kernel is the best approach.
        
        // Let's implement the fused version directly in this kernel.
        // C[row * N + col] = (sum > 0.0f ? sum : 0.0f) / divisor; 
        // We need to pass divisor here. 
    }
}

// Fused Kernel: GEMM + ReLU + Division
__global__ void fused_gemm_relu_div_kernel(const float* __restrict__ A, const float* __restrict__ B, const float* __restrict__ bias, float* __restrict__ C, int M, int N, int K, float divisor) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N

    if (row < M && col < N) {
        float sum = 0.0f;
        const float* A_row = A + row * K;
        const float* B_row = B + col * K; // B is stored as (N, K) in memory (out_features, in_features)

        // Loop over K
        // Unrolling might help, but compiler usually does a decent job.
        // Using restrict helps compiler optimize loads.
        for (int k = 0; k < K; ++k) {
            sum += A_row[k] * B_row[k];
        }
        
        // Add bias
        if (bias != nullptr) {
            sum += bias[col];
        }

        // ReLU
        sum = sum > 0.0f ? sum : 0.0f;
        
        // Division
        C[row * N + col] = sum / divisor;
    }
}

// Helper function to launch the fused kernel
torch::Tensor fused_linear_relu_div_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float divisor) {
    // x: (M, K)
    // weight: (N, K) - PyTorch Linear weight shape is (out_features, in_features)
    // bias: (N)
    // output: (M, N)
    
    int M = x.size(0);
    int K = x.size(1);
    int N = weight.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(16, 16); // 256 threads per block
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Check if bias is defined (not None)
    // In PyTorch C++, a undefined tensor is like a None.
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    fused_gemm_relu_div_kernel<<<grid, block>>>(
        x.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias_ptr,
        out.data_ptr<float>(), 
        M, N, K, 
        divisor
    );

    return out;
}
"""

fused_linear_relu_div_cpp_source = (
    "torch::Tensor fused_linear_relu_div_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float divisor);"
)

# Compile the inline CUDA code
fused_linear_relu_div = load_inline(
    name="fused_linear_relu_div",
    cpp_sources=fused_linear_relu_div_cpp_source,
    cuda_sources=fused_linear_relu_div_source,
    functions=["fused_linear_relu_div_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for fused Linear, ReLU, and Division.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        # We still define the linear layer to hold the weights and bias parameters
        # so that the initialization matches the original model.
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        
        # Load the custom operator
        self.fused_op = fused_linear_relu_div

    def forward(self, x):
        # Use the custom fused CUDA kernel
        # We pass the weight and bias from the linear layer
        return self.fused_op.fused_linear_relu_div_cuda(x, self.linear.weight, self.linear.bias, self.divisor)