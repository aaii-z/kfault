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
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");
    }

    // cuBLAS GEMM parameters
    // We compute: out = x * weight^T
    // Dimensions: (M, K) * (K, N) -> (M, N)
    // M = batch_size, K = in_features, N = out_features
    
    // cuBLAS uses column-major layout.
    // We interpret the row-major matrices as column-major transposed matrices.
    // Row-major A (M, K) * B (K, N) = C (M, N)
    // is equivalent to:
    // Col-major C^T (N, M) = Col-major B^T (N, K) * Col-major A^T (K, M)
    
    // Mapping:
    // A (row-major) = x (batch_size, in_features)
    // B (row-major) = weight (out_features, in_features)
    // C (row-major) = out (batch_size, out_features)
    
    // In cuBLAS call:
    // op(A_cublas) = B^T  -> weight (N, K), transa=N (op is identity on B^T which is weight)
    // op(B_cublas) = A^T  -> x (M, K), transb=T (op is transpose on A^T which is x^T)
    // Wait, let's stick to the standard trick:
    // We want C = A * B^T
    // cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // We compute C^T = B * A^T
    // A_cublas = weight (N, K), lda = N
    // B_cublas = x (M, K), ldb = M. We need A^T, so transb=CUBLAS_OP_T. 
    // Wait, B_cublas is x. x is (M, K). x^T is (K, M).
    // We want B * A^T = weight * x^T.
    // weight is (N, K). x is (M, K).
    // weight * x^T -> (N, K) * (K, M) -> (N, M).
    // This matches C^T (N, M).
    
    // So:
    // m = N (out_features)
    // n = M (batch_size)
    // k = K (in_features)
    // A = weight pointer
    // lda = N (out_features)
    // transa = CUBLAS_OP_N
    // B = x pointer
    // ldb = M (batch_size)
    // transb = CUBLAS_OP_T
    // C = out pointer
    // ldc = N (out_features)
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
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