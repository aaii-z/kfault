import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the entire forward pass
fused_gemm_norm_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

// Kernel for fused GroupNorm + Swish + Mul + Swish
// Assumes input x is (batch_size, out_features) row-major
// Assumes weight/bias/param are (out_features)
__global__ void fused_norm_swish_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    const float* __restrict__ mul_weight, 
    float* __restrict__ out,
    int batch_size, 
    int out_features, 
    int num_groups) {
    
    int group_size = out_features / num_groups;
    
    // Each block handles one group instance (one group within a batch item)
    // Or we can map blocks to (batch, group) pairs.
    // Let's map blockIdx.x to the batch index, and blockIdx.y to the group index.
    // Actually, standard 1D block mapping is easier for reduction.
    // Let's map blockIdx.x to (batch_idx * num_groups + group_idx).
    
    int instance_idx = blockIdx.x; // 0 to batch_size * num_groups - 1
    int batch_idx = instance_idx / num_groups;
    int group_idx = instance_idx % num_groups;
    
    int start_feature = group_idx * group_size;
    int end_feature = start_feature + group_size;
    
    // Shared memory for reduction
    // We need sum and sum of squares.
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = sdata + blockDim.x;
    
    // 1. Calculate Mean and Variance
    float l_sum = 0.0f;
    float l_sum_sq = 0.0f;
    
    // Stride loop to handle group_size > blockDim.x
    // Use vectorized loads if possible? Maybe too complex for generic sizing. 
    // Stick to scalar for robustness.
    
    int tid = threadIdx.x;
    
    for (int i = start_feature + tid; i < end_feature; i += blockDim.x) {
        float val = x[batch_idx * out_features + i];
        l_sum += val;
        l_sum_sq += val * val;
    }
    
    s_sum[tid] = l_sum;
    s_sum_sq[tid] = l_sum_sq;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 computes mean and var
    __shared__ float mean, var;
    if (tid == 0) {
        mean = s_sum[0] / group_size;
        var = s_sum_sq[0] / group_size - mean * mean;
        var = max(var, 0.0f); // Clamp variance
    }
    __syncthreads();
    
    // 2. Normalize, Apply Swish, Mul, Swish
    float inv_std = rsqrtf(var + 1e-5f);
    
    for (int i = start_feature + tid; i < end_feature; i += blockDim.x) {
        int idx = batch_idx * out_features + i;
        float val = x[idx];
        
        // GroupNorm
        float normalized = (val - mean) * inv_std;
        float w = weight[i];
        float b = bias[i];
        float normed = normalized * w + b;
        
        // Swish 1
        float sig = 1.0f / (1.0f + expf(-normed));
        float swish1 = normed * sig;
        
        // Multiply
        float mul = swish1 * mul_weight[i];
        
        // Swish 2
        float sig2 = 1.0f / (1.0f + expf(-mul));
        float swish2 = mul * sig2;
        
        out[idx] = swish2;
    }
}

// GEMM kernel (SGEMM-like)
// Simplified approach: Use cuBLAS for GEMM as it is highly optimized.
// We will write a wrapper that calls cuBLAS.
// However, the prompt asks for "custom CUDA operators". 
// Writing a GEMM from scratch that beats cuBLAS is hard.
// But we can fuse the GEMM output into the Norm kernel.
// We can launch the Norm kernel on the output of GEMM.
// Since GEMM is async, we can just call cuBLAS GEMM and then our kernel.
// Or, we can implement a simple GEMM if we want to avoid cuBLAS dependency in the custom op,
// but usually custom ops wrap cuBLAS.
// Let's assume we use cuBLAS for the GEMM part for performance, 
// and fuse the subsequent operations.

torch::Tensor forward_cuda(
    torch::Tensor x, 
    torch::Tensor gemm_weight, 
    torch::Tensor gemm_bias,
    torch::Tensor norm_weight, 
    torch::Tensor norm_bias, 
    torch::Tensor mul_weight,
    int num_groups) {
    
    // 1. GEMM
    // x: (batch, in), gemm_weight: (out, in)
    // result: (batch, out)
    // We use torch::matmul which dispatches to cuBLAS.
    // To make it a "custom operator", we define the logic here.
    // If we strictly need a CUDA kernel for GEMM, it would be very long.
    // Given the constraints, using torch's built-in GEMM is the most practical approach 
    // for the matrix multiplication part, while fusing the post-processing.
    // However, to strictly follow "custom CUDA operators", I will use torch::matmul 
    // inside the C++ extension. This is standard practice.
    
    auto x_contiguous = x.contiguous();
    auto gemm_out = torch::addmm(gemm_bias, x_contiguous, gemm_weight.t());
    
    // 2. Fused Norm + Swish + Mul + Swish
    auto out = torch::empty_like(gemm_out);
    int batch_size = gemm_out.size(0);
    int out_features = gemm_out.size(1);
    
    int group_size = out_features / num_groups;
    
    // We want enough threads to cover the group size for reduction efficiency.
    // Threads per block should be power of 2.
    int threads = 256; 
    if (group_size < 256) {
        // Round up to nearest power of 2? Or just use group_size.
        // For small groups, we might have idle threads, but reduction logic handles it.
        // Let's stick to 256 or 512. 256 is usually safe.
        threads = 256;
    }
    
    // Grid size: one block per (batch, group)
    int blocks = batch_size * num_groups;
    
    // Shared memory size: 2 * threads * sizeof(float)
    size_t shared_mem_size = 2 * threads * sizeof(float);
    
    fused_norm_swish_kernel<<<blocks, threads, shared_mem_size>>>(
        gemm_out.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        mul_weight.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_features,
        num_groups
    );
    
    return out;
}
"""

fused_gemm_norm_swish_cpp_source = """
torch::Tensor forward_cuda(
    torch::Tensor x, 
    torch::Tensor gemm_weight, 
    torch::Tensor gemm_bias,
    torch::Tensor norm_weight, 
    torch::Tensor norm_bias, 
    torch::Tensor mul_weight,
    int num_groups);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_gemm_norm_swish_cpp_source,
    cuda_sources=fused_gemm_norm_swish_source,
    functions=["forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        # We keep the parameters but remove the nn.Module wrappers to avoid overhead
        # and to use them directly in our custom op.
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Initialize weights similar to nn.Linear
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.randn(out_features))
        
        # Initialize weights for GroupNorm
        self.norm_weight = nn.Parameter(torch.ones(out_features))
        self.norm_bias = nn.Parameter(torch.zeros(out_features))
        
        # Initialize multiply weight
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.forward_cuda(
            x, 
            self.gemm_weight, 
            self.gemm_bias, 
            self.norm_weight, 
            self.norm_bias, 
            self.multiply_weight, 
            self.num_groups
        )