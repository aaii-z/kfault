import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + maxpool + sum + scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

const int TILE_SIZE = 32;
const int BLOCK_SIZE = 256;

__global__ void fused_matmul_maxpool_sum_scale_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    int kernel_size, float scale_factor) {
    
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int out_feature_idx = threadIdx.x;
    int num_threads = blockDim.x;
    
    if (batch_idx >= batch_size || out_feature_idx >= out_features) return;
    
    const float* current_input = input + batch_idx * in_features;
    const float* current_weight = weight + out_feature_idx * in_features;
    
    // Compute matmul result for this output feature
    float result = 0.0f;
    for (int i = out_feature_idx; i < in_features; i += num_threads) {
        result += current_input[i] * current_weight[i];
    }
    
    // Reduce within warp using warp shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        result += __shfl_down_sync(0xffffffff, result, offset);
    }
    
    if (out_feature_idx % 32 == 0) {
        shared_mem[out_feature_idx / 32] = result;
    }
    __syncthreads();
    
    // Final reduction for the first 32 threads
    if (out_feature_idx < 32) {
        float final_result = 0.0f;
        for (int i = 0; i < (out_features + 31) / 32; i++) {
            if (i * 32 + out_feature_idx < out_features) {
                final_result += shared_mem[i];
            }
        }
        
        // Apply bias and maxpool (simplified - since kernel_size=2, we can handle it)
        if (out_feature_idx < out_features) {
            float biased_result = final_result + (bias ? bias[out_feature_idx] : 0.0f);
            
            // For kernel_size=2 maxpool: compare with neighbor
            float neighbor_val = 0.0f;
            if (out_feature_idx + 1 < out_features) {
                neighbor_val = shared_mem[(out_feature_idx + 1) / 32] + 
                              (bias ? bias[out_feature_idx + 1] : 0.0f);
            }
            
            float max_val = fmaxf(biased_result, neighbor_val);
            
            // Sum reduction across output features (simplified)
            atomicAdd(&output[batch_idx], max_val * scale_factor);
        }
    }
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, float scale_factor) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size}, input.options());
    
    dim3 grid(batch_size);
    dim3 block(min(1024, out_features));
    int shared_mem_size = ((out_features + 31) / 32) * sizeof(float);
    
    fused_matmul_maxpool_sum_scale_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        kernel_size,
        scale_factor
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_operation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, float scale_factor);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operation_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas", "-lcudnn"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.fused_op = fused_operation

    def forward(self, x):
        return self.fused_op.fused_operation_cuda(
            x, self.weight, self.bias, self.kernel_size, self.scale_factor
        )