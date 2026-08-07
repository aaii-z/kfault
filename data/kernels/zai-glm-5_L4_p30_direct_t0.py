import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + GroupNorm + HardTanh
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

// Define EPS for numerical stability
#define EPS 1e-5f

// Kernel for fused Group Normalization and HardTanh
// This kernel is launched after the GEMM operation.
// It processes the output of GEMM in-place.
__global__ void fused_group_norm_hardtanh_kernel(
    float* __restrict__ data,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size, int out_features, int num_groups) {
    
    // Calculate indices
    // We assign each thread to a specific element in the output tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int n = idx / out_features; // Batch index
        int f = idx % out_features; // Feature index
        
        int group_size = out_features / num_groups;
        int g = f / group_size; // Group index
        
        // Calculate mean and variance for the group
        // Note: This naive implementation has race conditions if multiple threads write to shared memory
        // A more optimized version would use shared memory reduction.
        // However, for simplicity and to avoid complex synchronization, we will use a two-pass approach
        // or rely on the fact that for small group sizes, we can iterate.
        // Given the constraints, a highly optimized reduction is complex to write inline.
        // Instead, we will perform the reduction in a separate kernel or assume the group size is small enough
        // to be handled by a single thread block (which is not generally true).
        // 
        // CORRECTION: To make this functional and correct without excessive complexity for this prompt,
        // we will perform the normalization using a standard approach:
        // 1. Compute mean/variance. This requires a reduction.
        // 2. Normalize.
        // 3. Apply HardTanh.
        //
        // Since writing a full high-performance Group Norm kernel is lengthy, we will use a simpler
        // block-based reduction strategy or atomic operations if needed, but for the purpose of this
        // exercise, we will focus on the fusion logic.
        //
        // Let's use a standard parallel reduction for Group Norm.
        // We need to launch a kernel where each block handles one (batch, group) pair.
        
        // However, the current kernel is element-wise.
        // Let's change the launch configuration.
        // We will launch a kernel where each block calculates the mean/var for a specific (batch, group).
        // Then threads in that block normalize their respective elements.
        
        // Re-designing the kernel launch:
        // Grid: (batch_size, num_groups)
        // Block: (group_size) -- assuming group_size <= 1024. If larger, we need more complex reduction.
        // The problem states out_features=8192, num_groups=16 -> group_size = 512.
        // This fits within a single block (max 1024 threads).
        
        // So we will use a different kernel for the normalization part.
    }
}

// Optimized Kernel for Group Normalization + HardTanh
// Grid: (batch_size, num_groups)
// Block: (group_size) -> Assumes group_size <= 1024
__global__ void group_norm_hardtanh_kernel(
    float* __restrict__ data,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size, int out_features, int num_groups, int group_size,
    float hardtanh_min, float hardtanh_max) {
    
    // We use shared memory for reduction
    extern __shared__ float s_data[]; // Size: 2 * blockDim.x (for sum and sum of squares)
    
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    
    // Global feature index for the start of this group
    int feature_start = group_idx * group_size;
    
    // Thread index within the block
    int tid = threadIdx.x;
    int f = feature_start + tid; // Global feature index
    
    // Load data to register and shared memory
    float val = 0.0f;
    if (f < out_features) {
        int idx = batch_idx * out_features + f;
        val = data[idx];
    }
    
    s_data[tid] = val;
    s_data[blockDim.x + tid] = val * val;
    __syncthreads();
    
    // Reduction for sum and sum of squares
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
            s_data[blockDim.x + tid] += s_data[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    
    // Compute mean and variance
    float mean = s_data[0] / group_size;
    float var = (s_data[blockDim.x] / group_size) - (mean * mean);
    float inv_std = rsqrtf(var + EPS);
    
    // Normalize, scale, shift, and apply HardTanh
    if (f < out_features) {
        int idx = batch_idx * out_features + f;
        float normalized = (val - mean) * inv_std;
        float scaled = normalized * weight[f] + bias[f];
        
        // HardTanh
        scaled = fmaxf(hardtanh_min, fminf(hardtanh_max, scaled));
        
        data[idx] = scaled;
    }
}

// The GEMM part will be handled by cuBLAS, but we need to wrap it.
// We can fuse the operations by:
// 1. Performing GEMM (output to a buffer).
// 2. Launching the normalization kernel on that buffer.

// Wrapper function
torch::Tensor fused_gemm_group_norm_hardtanh_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gn_weight, torch::Tensor gn_bias,
    int num_groups, float hardtanh_min, float hardtanh_max) {
    
    // 1. GEMM
    // We use torch's matmul which is efficient and handles shapes.
    // To truly fuse, we would need to write a GEMM kernel or use cuBLAS directly.
    // Using torch::matmul is safer and cleaner for the GEMM part.
    // The optimization comes from fusing the subsequent ops.
    
    auto out = torch::matmul(x, weight.transpose(0, 1));
    // Add bias for GEMM
    out = out + bias; // Broadcasting handles this efficiently
    
    // 2. Fused Group Norm + HardTanh
    int batch_size = out.size(0);
    int out_features = out.size(1);
    int group_size = out_features / num_groups;
    
    // Launch configuration
    dim3 grid(batch_size, num_groups);
    dim3 block(group_size);
    size_t shared_mem_size = 2 * group_size * sizeof(float);
    
    group_norm_hardtanh_kernel<<<grid, block, shared_mem_size>>>(
        out.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        batch_size, out_features, num_groups, group_size,
        hardtanh_min, hardtanh_max
    );
    
    // Check for launch errors
    C10_CUDA_CHECK(cudaGetLastError());
    
    return out;
}
"""

fused_kernel_cpp_source = (
    "torch::Tensor fused_gemm_group_norm_hardtanh_cuda("
    "torch::Tensor x, torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor gn_weight, torch::Tensor gn_bias, "
    "int num_groups, float hardtanh_min, float hardtanh_max);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_group_norm_hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # We keep the parameters from the original model to initialize weights
        # We use nn.Linear to initialize weights and bias, then extract them.
        # This ensures we match the initialization of the original model.
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        
        # We extract the weights and biases to pass to the CUDA kernel
        # We register them as buffers so they are part of the state_dict but not trained by PyTorch's optimizers directly
        # (The original model trains them, so we just need to keep them as parameters)
        # Actually, we should keep them as Parameters if we want to train them.
        # But the custom op expects tensors.
        
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.num_groups = num_groups
        
        # We will use the weights from the defined layers directly in the forward pass
        # No need to re-register.

    def forward(self, x):
        # Call the custom fused kernel
        # We pass the weights and biases from the nn.Module components
        return fused_ops.fused_gemm_group_norm_hardtanh_cuda(
            x, 
            self.gemm.weight, 
            self.gemm.bias, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.num_groups, 
            self.hardtanh_min, 
            self.hardtanh_max
        )