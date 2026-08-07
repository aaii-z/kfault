import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the entire forward pass
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel for: Gemm -> Subtract -> GlobalAvgPool -> LogSumExp -> GELU -> ResidualAdd
// Note: Gemm is handled by cuBLAS, this kernel handles the element-wise and reduction ops.

__global__ void fused_post_gemm_kernel(
    const float* __restrict__ x_gemm, // Input from Gemm (batch_size, out_features)
    const float* __restrict__ subtract_param, // (out_features)
    const float* __restrict__ original_x, // Residual input (batch_size, in_features)
    float* __restrict__ out, // Output (batch_size, 1)
    int batch_size,
    int out_features
) {
    // Using shared memory for reduction (GlobalAvgPool and LogSumExp)
    extern __shared__ float s_data[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Pointers for the current batch item
    const float* x_gemm_row = x_gemm + batch_idx * out_features;
    const float* original_x_row = original_x + batch_idx * out_features; // Assuming in_features == out_features for residual add

    // Step 1 & 2: Subtract and compute sum for GlobalAvgPool
    // Each thread processes multiple elements to cover out_features
    float sum_val = 0.0f;
    float max_val = -INFINITY; // For LogSumExp stability
    
    // Grid stride loop would be better if out_features was huge, but here we assume
    // threads per block covers the dimension or we iterate.
    // We use a fixed block size (e.g., 256) and iterate.
    
    for (int i = tid; i < out_features; i += blockDim.x) {
        float val = x_gemm_row[i] - subtract_param[i];
        
        // We need to store 'val' back to global memory if we were doing this separately, 
        // but here we just need the mean and logsumexp.
        // However, we need the subtracted values for the mean.
        // Wait, GlobalAvgPool is on the subtracted result.
        
        sum_val += val;
        
        // For LogSumExp, we need max. 
        // But LogSumExp is on the result of GlobalAvgPool? 
        // The PyTorch code says:
        // x = torch.mean(x, dim=1, keepdim=True) -> (batch_size, 1)
        // x = torch.logsumexp(x, dim=1, keepdim=True) -> (batch_size, 1)
        // Wait, LogSumExp on a (batch_size, 1) tensor along dim=1?
        // If x is (batch_size, 1), logsumexp along dim=1 reduces it to (batch_size, 1).
        // logsumexp(x) where x is size 1 is just x.
        // Is dim=1 correct? 
        // Input to mean: (batch_size, out_features). Mean dim=1 -> (batch_size, 1).
        // Input to logsumexp: (batch_size, 1). LogSumExp dim=1 -> (batch_size, 1).
        // This seems redundant. logsumexp([val]) = val.
        // Unless the user meant logsumexp on the features before mean? 
        // No, the code is sequential.
        // Let's assume the code is exactly as written.
        // So we compute mean. Then logsumexp(mean). Since mean is a scalar per row, logsumexp is identity.
        // Then GELU(scalar). Then Add original_x (broadcasting?).
        // original_x is (batch_size, in_features). 
        // If x is (batch_size, 1), x + original_x broadcasts to (batch_size, in_features).
        
        // So the output is (batch_size, out_features).
        // My kernel signature assumed output was (batch_size, 1). I need to fix this.
        // The output should be the same size as original_x.
    }
    
    // Reduction for Sum
    s_data[tid] = sum_val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    
    float mean_val = s_data[0] / out_features;
    
    // Step 4: LogSumExp
    // Since mean_val is a scalar, logsumexp is just the value itself.
    float lse_val = mean_val; 
    
    // Step 5: GELU
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float gelu_val = 0.5f * lse_val * (1.0f + tanhf(0.7978845608f * (lse_val + 0.044715f * lse_val * lse_val * lse_val)));
    
    // Step 6: Residual Add
    // out = gelu_val + original_x
    // We need to write this to every element of the output row.
    
    for (int i = tid; i < out_features; i += blockDim.x) {
        out[batch_idx * out_features + i] = gelu_val + original_x_row[i];
    }
}

torch::Tensor fused_post_gemm_cuda(
    torch::Tensor x_gemm, 
    torch::Tensor subtract_param, 
    torch::Tensor original_x
) {
    auto batch_size = x_gemm.size(0);
    auto out_features = x_gemm.size(1);
    
    // Allocate output tensor
    auto out = torch::empty({batch_size, out_features}, x_gemm.options());

    const int block_size = 256;
    const int num_blocks = batch_size;
    const size_t shared_mem_size = block_size * sizeof(float);

    fused_post_gemm_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x_gemm.data_ptr<float>(),
        subtract_param.data_ptr<float>(),
        original_x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_features
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_post_gemm_cuda(torch::Tensor x_gemm, torch::Tensor subtract_param, torch::Tensor original_x);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_post_gemm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.fused_ops = fused_ops

    def forward(self, x):
        original_x = x
        # Gemm
        x = self.gemm(x)
        
        # Fused: Subtract -> GlobalAvgPool -> LogSumExp -> GELU -> ResidualAdd
        # Note: We pass original_x directly. The kernel handles the broadcasting add.
        # We assume in_features == out_features for the residual add to work element-wise
        # based on the kernel logic, or the kernel logic broadcasts.
        # The kernel logic broadcasts the scalar result of GELU to the shape of original_x.
        return self.fused_ops.fused_post_gemm_cuda(x, self.subtract, original_x)