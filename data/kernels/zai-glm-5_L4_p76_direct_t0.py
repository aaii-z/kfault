import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Bias + ReLU
fused_gemm_bias_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>

// Kernel for Bias addition and ReLU
__global__ void bias_relu_kernel(float* out, const float* bias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        float val = out[row * cols + col] + bias[col];
        out[row * cols + col] = val > 0.0f ? val : 0.0f;
    }
}

// Helper function to handle cuBLAS errors
void checkCublasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(msg);
    }
}

torch::Tensor fused_gemm_bias_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    // x: (batch_size, in_features)
    // weight: (out_features, in_features)
    // bias: (out_features)
    
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    // Allocate output tensor
    auto out = torch::empty({batch_size, out_features}, x.options());

    // cuBLAS handle management
    // Note: In a production scenario, the handle should be cached.
    // For this context, we create and destroy it per call for simplicity,
    // or we could use a global static handle. 
    // Let's use a static handle for better performance in repeated calls.
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");
    }

    // cuBLAS GEMM parameters
    // C = alpha * op(A) * op(B) + beta * C
    // We want: out = x * weight^T
    // cuBLAS is column-major by default.
    // Row-major (batch, in) * (in, out) -> (batch, out)
    // In column-major terms: C^T = B^T * A^T
    // Let A = x (batch, in), B = weight (out, in)
    // We compute C (batch, out) = A (batch, in) * B^T (in, out)
    // Using cublasSgemm:
    // op(B) = N (no transpose), op(A) = T (transpose)
    // This doesn't map directly to standard row-major matmul without transposition logic.
    // Easier approach: Treat row-major matrices as column-major transposed matrices.
    // We want RowMajor C = RowMajor A * RowMajor B^T
    // This is equivalent to ColMajor C^T = ColMajor B * ColMajor A^T
    // So we can call cublasSgemm with A=weight, B=x, C=out, transa=CUBLAS_OP_N, transb=CUBLAS_OP_T
    // Wait, let's stick to the standard trick:
    // Row-major A (M, K) * B (K, N) -> C (M, N)
    // In cuBLAS (col-major): A is (M, K), B is (K, N), C is (M, N)
    // We have: x (M, K), weight (N, K). We want out (M, N).
    // We want x * weight^T.
    // Let's map: A_cublas = weight^T (K, N), B_cublas = x (M, K) ?? No.
    // Standard trick: A_row * B_row^T = (B_col * A_col^T)^T
    // So we compute: out^T = weight * x^T
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, 1.0f, weight, N, x, M, 0.0f, out, N)
    // M = batch_size, K = in_features, N = out_features
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Leading dimensions:
    // weight is (out_features, in_features), in column-major ld is out_features
    // x is (batch_size, in_features), in column-major ld is batch_size
    // out is (batch_size, out_features), in column-major ld is batch_size
    
    // We compute out^T = weight * x^T
    // Result out^T will be (out_features, batch_size)
    // So the output 'out' allocated as (batch_size, out_features) will store the result correctly in row-major.
    
    checkCublasError(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                    out_features, batch_size, in_features, 
                    &alpha, 
                    weight.data_ptr<float>(), out_features, 
                    x.data_ptr<float>(), batch_size, 
                    &beta, 
                    out.data_ptr<float>(), out_features),
        "cuBLAS GEMM failed"
    );

    // Launch Bias + ReLU kernel
    // Grid and Block dimensions
    // We operate on a (batch_size, out_features) matrix
    int threads_x = 32;
    int threads_y = 32;
    dim3 block(threads_x, threads_y);
    dim3 grid((out_features + threads_x - 1) / threads_x, (batch_size + threads_y - 1) / threads_y);

    bias_relu_kernel<<<grid, block>>>(out.data_ptr<float>(), bias.data_ptr<float>(), batch_size, out_features);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

fused_gemm_bias_relu_cpp_source = (
    "torch::Tensor fused_gemm_bias_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_gemm_bias_relu = load_inline(
    name="fused_gemm_bias_relu",
    cpp_sources=fused_gemm_bias_relu_cpp_source,
    cuda_sources=fused_gemm_bias_relu_source,
    functions=["fused_gemm_bias_relu_cuda"],
    verbose=True,
    extra_cflags=["-lcublas"],
    extra_ldflags=["-lcublas"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a fused GEMM + Bias + ReLU operation.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        # We define the weight parameter manually to match nn.Linear behavior (bias=False)
        # nn.Linear stores weights as (out_features, in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Initialize the custom op
        self.fused_op = fused_gemm_bias_relu

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        return self.fused_op.fused_gemm_bias_relu_cuda(x, self.weight, self.bias)