import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear (GEMM) + Bias + ReLU + Division
fused_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Kernel for Bias addition, ReLU, and Division
__global__ void bias_relu_div_kernel(float* out, const float* bias, int M, int N, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;
    if (idx < total_elements) {
        int col = idx % M; // Bias index corresponds to the output feature dimension
        float val = out[idx];
        if (bias != nullptr) {
            val += bias[col];
        }
        val = fmaxf(0.0f, val); // ReLU
        out[idx] = val / divisor;
    }
}

// Fused forward function
torch::Tensor fused_linear_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor) {
    // Shapes
    int N = input.size(0); // Batch size
    int K = input.size(1); // In features
    int M = weight.size(0); // Out features
    
    // Output tensor
    auto output = torch::empty({N, M}, input.options());
    
    // cuBLAS handle
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    // Pointers
    float* input_ptr = input.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Constants
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // GEMM: output = input * weight^T
    // In cuBLAS column-major terms:
    // We want row-major: C_rm = A_rm * B_rm^T
    // Equivalent to col-major: C_cm = B_cm^T * A_cm^T
    // We compute C_cm = op(A) * op(B) where:
    // op(A) = weight (transposed) -> op(A) is (M, K)
    // op(B) = input (not transposed) -> op(B) is (K, N)
    // Result C_cm is (M, N)
    
    // Arguments for cublasSgemm:
    // m = M, n = N, k = K
    // A = weight, lda = K (leading dim of weight storage in col-major view)
    // B = input, ldb = K (leading dim of input storage in col-major view)
    // C = output, ldc = M (leading dim of output storage in col-major view)
    
    cublasSgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N, 
                M, N, K, 
                &alpha, 
                weight_ptr, K, 
                input_ptr, K, 
                &beta, 
                output_ptr, M);
    
    // Post-process kernel: Bias + ReLU + Div
    int total_elements = N * M;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    bias_relu_div_kernel<<<num_blocks, block_size>>>(output_ptr, bias_ptr, M, N, divisor);
    
    return output;
}
"""

fused_linear_cpp_source = (
    "torch::Tensor fused_linear_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);"
)

# Compile the inline CUDA code
fused_linear_op = load_inline(
    name="fused_linear",
    cpp_sources=fused_linear_cpp_source,
    cuda_sources=fused_linear_source,
    functions=["fused_linear_forward_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_ldflags=["-lcublas"],
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for fused Linear + ReLU + Division.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        # We keep the linear layer to hold the weights and bias parameters
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        self.fused_linear_op = fused_linear_op

    def forward(self, x):
        # Call the custom fused CUDA kernel
        return self.fused_linear_op.fused_linear_forward_cuda(x, self.linear.weight, self.linear.bias, self.divisor)