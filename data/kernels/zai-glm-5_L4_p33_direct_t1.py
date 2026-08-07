import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, scaling, and batch normalization
fused_gemm_scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

// Kernel for scaling and computing sum and sum of squares for Batch Normalization statistics
__global__ void scale_bn_stats_kernel(const float* __restrict__ x, const float* __restrict__ scale, 
                                     float* __restrict__ sum_out, float* __restrict__ sum_sq_out, 
                                     int batch_size, int out_features) {
    // Use shared memory to reduce global memory access for the partial sums
    extern __shared__ float shared_data[]; 
    float* s_sum = shared_data;
    float* s_sum_sq = shared_data + blockDim.x;

    int feature_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Initialize shared memory
    s_sum[tid] = 0.0f;
    s_sum_sq[tid] = 0.0f;
    __syncthreads();

    // Pointers to the specific column (feature) for this block
    const float* x_col = x + feature_idx;
    float scale_val = scale[feature_idx];

    // Grid-stride loop to process all rows for this feature
    for (int i = tid; i < batch_size; i += blockDim.x) {
        float val = x_col[i * out_features] * scale_val;
        // Accumulate in shared memory
        s_sum[tid] += val;
        s_sum_sq[tid] += val * val;
    }
    __syncthreads();

    // Reduce shared memory values to a single value per block
    // Only thread 0 does the final accumulation to global memory
    if (tid == 0) {
        float block_sum = 0.0f;
        float block_sum_sq = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            block_sum += s_sum[i];
            block_sum_sq += s_sum_sq[i];
        }
        // Atomic add to global memory
        atomicAdd(&sum_out[feature_idx], block_sum);
        atomicAdd(&sum_sq_out[feature_idx], block_sum_sq);
    }
}

// Kernel for the final Batch Normalization forward pass
__global__ void bn_forward_kernel(const float* __restrict__ x, const float* __restrict__ scale, 
                                  const float* __restrict__ mean, const float* __restrict__ inv_std, 
                                  const float* __restrict__ gamma, const float* __restrict__ beta, 
                                  float* __restrict__ out, int batch_size, int out_features) {
    int feature_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    float s = scale[feature_idx];
    float g = gamma[feature_idx];
    float b = beta[feature_idx];
    float m = mean[feature_idx];
    float i_std = inv_std[feature_idx];

    // Process rows in a grid-stride loop
    for (int i = tid; i < batch_size; i += blockDim.x) {
        int idx = i * out_features + feature_idx;
        float val = x[idx];
        // Apply scaling (from the model), then BN logic: gamma * (x - mean) * inv_std + beta
        val = val * s; // Scale
        val = (val - m) * i_std; // Normalize
        val = val * g + b; // Affine
        out[idx] = val;
    }
}

// Helper to handle cublas
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}

torch::Tensor fused_gemm_scale_bn_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                                       torch::Tensor scale, torch::Tensor running_mean, 
                                       torch::Tensor running_var, torch::Tensor gamma, 
                                       torch::Tensor beta, bool training, float eps, float momentum) {
    // 1. GEMM
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(0);
    
    auto gemm_out = torch::empty({batch_size, out_features}, x.options());
    
    // Perform GEMM: out = x * W^T + bias
    // Note: PyTorch Linear stores weights as (out_features, in_features), so we need transpose
    float alpha = 1.0f;
    float beta_val = 0.0f; 
    
    cublasHandle_t handle = get_cublas_handle();
    
    // cuBLAS assumes column-major storage.
    // We want to compute C = x * W^T.
    // x is (Batch, In) row-major. In cuBLAS col-major, this is x^T (In, Batch).
    // W is (Out, In) row-major. In cuBLAS col-major, this is W^T (In, Out).
    // We want C (Batch, Out) row-major. In cuBLAS col-major, this is C^T (Out, Batch).
    // So we want to compute C^T = W^T^T * x^T = W * x^T? No.
    // Let's look at the operation: C = A * B^T.
    // In col-major notation: C^T = (A * B^T)^T = B * A^T.
    // Let A = x (Batch x In). Let B = W (Out x In).
    // We want C = x * W^T.
    // C^T = (x * W^T)^T = W * x^T.
    // So we can perform GEMM with op(A) = N (W), op(B) = N (x).
    // A is W (Out x In) stored row-major -> Col-major (In x Out) transposed? 
    // Let's stick to the standard trick:
    // To compute RowMajor C = RowMajor A * RowMajor B^T:
    // ColMajor C^T = ColMajor B * ColMajor A^T.
    // A = x (Batch x In). B = W (Out x In).
    // We compute C^T = W * x^T.
    // cuBLAS arguments: transa=N, transb=N, m=Out, n=Batch, k=In, A=W, lda=In, B=x, ldb=In, C=gemm_out, ldc=Out.
    // Wait, W is (Out x In) row-major. In cuBLAS col-major, the leading dimension is the first dimension, which is Out.
    // But the data is laid out row-by-row. 
    // Let's use the standard interpretation:
    // RowMajor Matrix M (R, C) is viewed as ColMajor M^T (C, R) with leading dimension R.
    // So x (Batch, In) is x^T (In, Batch) with ld=Batch.
    // W (Out, In) is W^T (In, Out) with ld=Out.
    // We want C (Batch, Out) which is C^T (Out, Batch) with ld=Batch.
    // Operation: C = x * W^T.
    // C^T = (x * W^T)^T = W * x^T.
    // In cuBLAS: C = op(A) * op(B).
    // We want C^T = W * x^T.
    // A = W^T (In, Out) with ld=Out. op(A) = T -> A^T = W (Out, In).
    // B = x^T (In, Batch) with ld=Batch. op(B) = N -> B = x^T.
    // Result C (Out, Batch) with ld=Batch.
    // This matches C^T.
    
    cublasSgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N, 
                out_features, batch_size, in_features, 
                &alpha, 
                weight.data_ptr<float>(), out_features, // W is (Out, In) row-major. Treated as (In, Out) col-major with ld=Out. Transpose it to get W.
                x.data_ptr<float>(), batch_size,         // x is (Batch, In) row-major. Treated as (In, Batch) col-major with ld=Batch.
                &beta_val, 
                gemm_out.data_ptr<float>(), batch_size); // Output (Batch, Out) row-major. Treated as (Out, Batch) col-major with ld=Batch.

    // Add bias
    if (bias.defined() && bias.numel() > 0) {
        // bias is (Out). We need to add it to each row.
        // gemm_out += bias (broadcasted).
        // We can fuse this into the BN kernel or do it here.
        // Since BN kernel reads gemm_out, let's keep it separate or add it in BN kernel.
        // Adding it in BN kernel is cleaner (one less read/write).
        // But wait, the BN kernel uses gemm_out.data_ptr.
        // I will pass bias to the BN kernel.
    }
    
    // 2. Fused Scale + BN
    auto out = torch::empty_like(gemm_out);
    
    const int block_size = 256;
    const int num_blocks = out_features;
    
    if (training) {
        // Compute stats
        auto sum = torch::zeros(out_features, x.options());
        auto sum_sq = torch::zeros(out_features, x.options());
        
        size_t shared_mem_size = 2 * block_size * sizeof(float);
        scale_bn_stats_kernel<<<num_blocks, block_size, shared_mem_size>>>(
            gemm_out.data_ptr<float>(), scale.data_ptr<float>(),
            sum.data_ptr<float>(), sum_sq.data_ptr<float>(),
            batch_size, out_features
        );
        
        // Compute mean and var on CPU or via a small kernel (or torch ops)
        auto mean = sum / batch_size;
        auto var = sum_sq / batch_size - mean * mean;
        auto inv_std = 1.0 / torch::sqrt(var + eps);
        
        // Update running stats
        torch::lerp_out(running_mean, running_mean, mean, momentum);
        torch::lerp_out(running_var, running_var, var, momentum);
        
        // Launch forward kernel
        bn_forward_kernel<<<num_blocks, block_size>>>(
            gemm_out.data_ptr<float>(), scale.data_ptr<float>(),
            mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            out.data_ptr<float>(), batch_size, out_features
        );
        
    } else {
        // Inference mode
        auto mean = running_mean;
        auto inv_std = 1.0 / torch::sqrt(running_var + eps);
        
        bn_forward_kernel<<<num_blocks, block_size>>>(
            gemm_out.data_ptr<float>(), scale.data_ptr<float>(),
            mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            out.data_ptr<float>(), batch_size, out_features
        );
    }
    
    return out;
}
"""

fused_gemm_scale_bn_cpp_source = (
    "torch::Tensor fused_gemm_scale_bn_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor scale, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor gamma, "
    "torch::Tensor beta, bool training, float eps, float momentum);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_gemm_scale_bn_cpp_source,
    cuda_sources=fused_gemm_scale_bn_source,
    functions=["fused_gemm_scale_bn_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Parameters for GEMM (Linear)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Parameter for Scaling
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        # Parameters for BatchNorm
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        
        self.eps = eps
        self.momentum = momentum
        
        # Initialize weights (simple initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fused_op = fused_ops.fused_gemm_scale_bn_cuda

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias, self.scale, 
                             self.running_mean, self.running_var, self.gamma, self.beta, 
                             self.training, self.eps, self.momentum)