import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, multiplication, and LeakyReLU
fused_gemm_mul_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
    // cublasSgemm computes C = alpha * op(A) * op(B) + beta * C
    // We treat RowMajor tensors as ColMajor with transposed logic:
    // C_row = A_row * B_row^T
    // C_col = B_row * A_row^T
    // A_col (Weight) is (out, in) in memory. op=N. dims (out, in). lda=out.
    // B_col (Input) is (batch, in) in memory. op=T. dims (batch, in). ldb=batch.
    // C_col (Output) is (out, batch) in memory. dims (out, batch). ldc=out.
    
    cublasStatus_t status = cublasSgemm(
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
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        // Handle error appropriately in production code
    }

    // Fused Bias + Mul + LeakyReLU Kernel
    const int block_size = 256;
    const int num_elements = batch_size * out_features;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

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