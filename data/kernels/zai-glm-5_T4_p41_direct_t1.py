import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + BatchNorm + GELU + ReLU
# Optimization Strategy:
# 1. Use cuBLAS for GEMM (highly optimized matrix multiply).
# 2. Fuse BatchNorm, GELU, and ReLU into a single element-wise kernel.
# 3. This avoids launching separate kernels for each operation and minimizes memory traffic.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

// Error checking macro for cuBLAS calls
#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      printf("cuBLAS error at line %d: %d\\n", __LINE__, status);              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Kernel for fused BatchNorm + GELU + ReLU
// Each thread processes one element.
__global__ void bn_gelu_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    float* __restrict__ out,
    int batch_size,
    int out_features,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int row = idx / out_features;
        int col = idx % out_features;
        
        float val = x[idx];
        
        // BatchNorm
        float mean = bn_mean[col];
        float var = bn_var[col];
        float gamma = bn_weight[col];
        float beta = bn_bias[col];
        
        // Normalize and affine
        float normalized = (val - mean) * rsqrtf(var + eps);
        float bn_out = normalized * gamma + beta;
        
        // GELU (approximation)
        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float c = 0.7978845608028654f; // sqrt(2/pi)
        float x_val = bn_out;
        float x3 = x_val * x_val * x_val;
        float inner = c * (x_val + 0.044715f * x3);
        float gelu_out = 0.5f * x_val * (1.0f + tanhf(inner));
        
        // ReLU
        out[idx] = fmaxf(0.0f, gelu_out);
    }
}

// Helper function to create cuBLAS handle (thread-safe-ish for this context)
// Note: In a production scenario, the handle should be managed more carefully.
cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}

torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    // Allocate output tensor
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    // 1. GEMM using cuBLAS
    // Note: PyTorch Linear is y = x * W^T + b
    // cuBLAS gemm expects column-major, so we treat row-major as column-major transposed.
    // We compute output = input * weight^T
    // In cuBLAS column-major terms: C = B * A^T
    // Where A is weight (out_features x in_features), B is input (batch_size x in_features)
    // C is output (batch_size x out_features)
    
    cublasHandle_t handle = get_cublas_handle();
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // input (B): batch_size x in_features
    // weight (A): out_features x in_features
    // We want C = B * A^T
    // cublasSgemm: C (m x n) = alpha * op(A) (m x k) * op(B) (k x n) + beta * C
    // Let op(A) be input (not transposed in logic, but passed as B to cublas)
    // Let op(B) be weight (transposed in logic, passed as A to cublas)
    // m = batch_size, n = out_features, k = in_features
    // op(A) = input (batch_size x in_features) -> m=batch_size, k=in_features
    // op(B) = weight (out_features x in_features) -> k=in_features, n=out_features
    
    // Correct mapping for cublasSgemm (Row-major logic mapped to Column-major API):
    // C = A * B^T
    // A = input (batch_size x in_features)
    // B = weight (out_features x in_features)
    // m = batch_size, n = out_features, k = in_features
    // We pass A to cublas as B matrix (transposed logic)
    // We pass B to cublas as A matrix (transposed logic)
    // Actually, simpler view:
    // We want C = input * weight^T
    // In Cublas: C = op(A) * op(B)
    // Set op(A) = input (N), op(B) = weight (T)
    // A (input) is (batch_size, in_features) -> passed as (in_features, batch_size) lda=in_features
    // B (weight) is (out_features, in_features) -> passed as (in_features, out_features) ldb=in_features
    // C (output) is (batch_size, out_features) -> passed as (out_features, batch_size) ldc=out_features
    
    // Wait, let's stick to standard row-major interpretation for cublas:
    // To compute C = A * B^T (Row major)
    // We compute C^T = B * A^T
    // So we call cublasSgemm with:
    // op(A) = B (weight), transposed -> N (no transpose) because we want B^T in the math, but cublas sees col-major.
    // Let's use the standard trick:
    // C (row-major) = A (row-major) * B (row-major)^T
    // Cublas computes: C (col-major) = A (col-major) * B (col-major)
    // If we pass A as B and B as A and transpose both:
    // C^T = B^T * A^T
    // This is getting confusing. Let's use the direct mapping.
    
    // We want: output = input * weight^T
    // Cublas parameters for C = op(A) * op(B) + C
    // We want C (batch_size, out_features)
    // A = input (batch_size, in_features)
    // B = weight (out_features, in_features)
    
    // Cublas is column major.
    // Input memory is row-major (batch_size, in_features). Treat as (in_features, batch_size) col-major.
    // Weight memory is row-major (out_features, in_features). Treat as (in_features, out_features) col-major.
    // Output memory is row-major (batch_size, out_features). Treat as (out_features, batch_size) col-major.
    
    // We want to compute Output = Input * Weight^T
    // In col-major terms: Output^T = Weight * Input^T
    // So we compute C = A * B where:
    // A = Weight (out_features, in_features) -> passed as is (lda = in_features)
    // B = Input (batch_size, in_features) -> passed transposed (ldb = in_features)
    // C = Output (batch_size, out_features) -> passed transposed (ldc = out_features)
    
    // Wait, if A=Weight, B=Input^T, then C = Weight * Input^T.
    // C will be (out_features, batch_size).
    // This matches Output^T.
    
    // So:
    // m = out_features (rows of A)
    // n = batch_size (cols of B)
    // k = in_features (cols of A / rows of B)
    // op(A) = N (no transpose), A is weight
    // op(B) = T (transpose), B is input
    // lda = in_features
    // ldb = in_features
    // ldc = out_features
    
    // Let's verify dimensions:
    // A (weight): out_features x in_features
    // B (input): batch_size x in_features -> Transposed: in_features x batch_size
    // C = A * B = (out_features x in_features) * (in_features x batch_size) = out_features x batch_size
    // This is Output^T. Correct.
    
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, // op(A)
        CUBLAS_OP_T, // op(B)
        out_features, // m
        batch_size, // n
        in_features, // k
        &alpha,
        weight.data_ptr<float>(), in_features, // A, lda
        input.data_ptr<float>(), in_features, // B, ldb
        &beta,
        output.data_ptr<float>(), out_features // C, ldc
    ));
    
    // 2. Add bias (fused into the kernel or separate? Let's fuse it into the element-wise kernel)
    // Actually, bias addition is just an element-wise add. We can do it in the BN kernel.
    // We need to pass bias to the kernel.
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;
    
    bn_gelu_relu_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), // Input is the output of GEMM
        bias.data_ptr<float>(), // Bias
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        output.data_ptr<float>(), // Output (in-place)
        batch_size,
        out_features,
        1e-5f
    );
    
    return output;
}

// Updated kernel to include bias addition
__global__ void bn_gelu_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    float* __restrict__ out,
    int batch_size,
    int out_features,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int row = idx / out_features;
        int col = idx % out_features;
        
        // Add bias first
        float val = x[idx] + bias[col];
        
        // BatchNorm
        float mean = bn_mean[col];
        float var = bn_var[col];
        float gamma = bn_weight[col];
        float beta = bn_bias[col];
        
        // Normalize and affine
        float normalized = (val - mean) * rsqrtf(var + eps);
        float bn_out = normalized * gamma + beta;
        
        // GELU (approximation)
        float c = 0.7978845608028654f;
        float x_val = bn_out;
        float x3 = x_val * x_val * x_val;
        float inner = c * (x_val + 0.044715f * x3);
        float gelu_out = 0.5f * x_val * (1.0f + tanhf(inner));
        
        // ReLU
        out[idx] = fmaxf(0.0f, gelu_out);
    }
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_gemm_bn_gelu_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a fused GEMM + BatchNorm + GELU + ReLU.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define parameters manually to match nn.Linear and nn.BatchNorm1d
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.empty(out_features))
        self.bn_bias = nn.Parameter(torch.empty(out_features))
        
        # Register buffers for BatchNorm running stats
        self.register_buffer('running_mean', torch.empty(out_features))
        self.register_buffer('running_var', torch.empty(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.ones_(self.bn_weight)
        nn.init.zeros_(self.bn_bias)
        nn.init.zeros_(self.running_mean)
        nn.init.ones_(self.running_var)
        
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_gemm_bn_gelu_relu_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.running_mean, 
            self.running_var, 
            self.bn_weight, 
            self.bn_bias
        )