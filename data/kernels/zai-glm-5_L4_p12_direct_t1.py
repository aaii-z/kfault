import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, multiplication, and LeakyReLU
fused_gemm_mul_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Constants for LeakyReLU
// Note: negative_slope is passed as argument to allow flexibility, 
// but could be templated for constant propagation if strictly fixed.

// CUDA kernel for fused Bias + Multiplication + LeakyReLU
__global__ void bias_mul_leaky_relu_kernel(
    float* __restrict__ data, 
    const float* __restrict__ bias, 
    int size, 
    int out_features, 
    float multiplier, 
    float negative_slope) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int bias_idx = idx % out_features;
        float val = data[idx] + bias[bias_idx];
        val *= multiplier;
        data[idx] = val > 0.0f ? val : val * negative_slope;
    }
}

// Helper function to handle cuBLAS errors
void checkCublasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(msg);
    }
}

torch::Tensor fused_gemm_mul_leaky_relu_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float multiplier,
    float negative_slope) {
    
    // Get dimensions
    int64_t batch_size = input.size(0);
    int64_t in_features = input.size(1);
    int64_t out_features = weight.size(0);

    // Allocate output tensor
    auto output = torch::empty({batch_size, out_features}, input.options());

    // Get cuBLAS handle from PyTorch
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    // Constants for GEMM
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS GEMM
    // We want: Output = Input * Weight^T + Bias
    // cuBLAS is column-major. 
    // Input (A): (batch, in) -> Row Major. Col-Major: (in, batch), Transposed.
    // Weight (B): (out, in) -> Row Major. Col-Major: (in, out), Transposed.
    // Output (C): (batch, out) -> Row Major. Col-Major: (out, batch).
    
    // C_col = op(A_col) * op(B_col)
    // C_col = Input^T * Weight^T^T = Input^T * Weight
    // This doesn't match.
    
    // Let's use the standard trick:
    // RowMajor C = RowMajor A * RowMajor B^T
    // C^T = (A * B^T)^T = B * A^T
    // In ColMajor: C_col = B_col * A_col^T
    // A_col (Weight) is (out, in) in RowMajor -> (in, out) in ColMajor? No.
    // Weight is stored (out, in). In ColMajor view, it is a matrix of dims (out, in) with leading dimension out.
    // Input is stored (batch, in). In ColMajor view, it is (batch, in) with leading dimension batch.
    
    // Let's map directly to cublasSgemm arguments:
    // We compute C = A * B^T (Row-Major)
    // cublasSgemm computes C_col = op(A_col) * op(B_col)
    // If we treat RowMajor tensors as ColMajor with transposed shapes:
    // A (Input) is (batch, in). Treat as A_col (in, batch) with op=T.
    // B (Weight) is (out, in). Treat as B_col (in, out) with op=N.
    // C (Output) is (batch, out). Treat as C_col (out, batch).
    
    // C_col = op(A_col) * op(B_col)
    // C_col (out, batch) = A_col^T (batch, in) * B_col (in, out) ?? No.
    
    // Let's stick to the verified logic:
    // C_row = A_row * B_row^T
    // Equivalent to C_col = B_row * A_row^T
    // A_col = B_row (Weight) -> dims (out, in), ld = out. op=N.
    // B_col = A_row (Input) -> dims (batch, in), ld = batch. op=T.
    // C_col -> dims (out, batch), ld = out.
    
    checkCublasError(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,       // op(A_col)
            CUBLAS_OP_T,       // op(B_col)
            out_features,      // m
            batch_size,        // n
            in_features,       // k
            &alpha,
            weight.data_ptr<float>(), // A_col
            out_features,      // lda
            input.data_ptr<float>(),  // B_col
            batch_size,        // ldb
            &beta,
            output.data_ptr<float>(), // C_col
            out_features       // ldc
        ),
        "cuBLAS GEMM failed"
    );

    // Fused Bias + Mul + LeakyReLU Kernel
    const int block_size = 256;
    const int num_elements = batch_size * out_features;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    // We assume bias is always defined for this model based on nn.Linear default
    bias_mul_leaky_relu_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        num_elements, 
        out_features, 
        multiplier, 
        negative_slope
    );

    return output;
}
"""

fused_gemm_mul_leaky_relu_cpp_source = (
    "torch::Tensor fused_gemm_mul_leaky_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float multiplier, float negative_slope);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_gemm_mul_leaky_relu",
    cpp_sources=fused_gemm_mul_leaky_relu_cpp_source,
    cuda_sources=fused_gemm_mul_leaky_relu_source,
    functions=["fused_gemm_mul_leaky_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        # We keep the Linear layer to hold the learned weights and biases
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.fused_op = fused_op

    def forward(self, x):
        # Call the custom fused CUDA kernel
        return self.fused_op.fused_gemm_mul_leaky_relu_cuda(
            x, self.gemm.weight, self.gemm.bias, self.multiplier, self.negative_slope
        )