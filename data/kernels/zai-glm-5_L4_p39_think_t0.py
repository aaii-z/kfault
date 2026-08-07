import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Scale + BatchNorm
fused_scale_bn_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel for fused Scale + BatchNorm
// Grid: (features,)
// Block: (block_size,)
// Each block processes one feature channel.
__global__ void fused_scale_bn_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ scale,
    const float* __restrict__ bn_weight, 
    const float* __restrict__ bn_bias,   
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    float* __restrict__ output,
    int batch_size, int features, float eps, float momentum, bool training) {
    
    int c = blockIdx.x; // Feature index
    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    if (c >= features) return;

    // Load parameters for this feature
    float s = scale[c];
    float gamma = bn_weight[c];
    float beta = bn_bias[c];
    
    // Shared memory for reduction
    extern __shared__ float s_mem[];
    float* s_sum = s_mem;
    float* s_sum_sq = s_mem + block_dim;

    float mean_scaled = 0.0f;
    float inv_std = 0.0f;

    // Pass 1: Compute mean and variance (if training)
    if (training) {
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;

        for (int i = tid; i < batch_size; i += block_dim) {
            float val = __ldg(&input[i * features + c]);
            local_sum += val;
            local_sum_sq += val * val;
        }

        s_sum[tid] = local_sum;
        s_sum_sq[tid] = local_sum_sq;
        __syncthreads();

        // Block reduction
        for (int s = block_dim / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_sum[tid] += s_sum[tid + s];
                s_sum_sq[tid] += s_sum_sq[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float mean = s_sum[0] / batch_size;
            float var = s_sum_sq[0] / batch_size - mean * mean;
            
            // Apply scale to statistics
            mean_scaled = mean * s;
            float var_scaled = var * s * s;

            // Update running stats
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean_scaled;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var_scaled;
            
            // Store results for Pass 2
            s_sum[0] = mean_scaled;
            s_sum_sq[0] = 1.0f / sqrtf(var_scaled + eps);
        }
        __syncthreads();
        
        mean_scaled = s_sum[0];
        inv_std = s_sum_sq[0];
    } else {
        // Eval mode: use running stats
        if (tid == 0) {
            mean_scaled = running_mean[c];
            float var_scaled = running_var[c];
            inv_std = 1.0f / sqrtf(var_scaled + eps);
            
            s_sum[0] = mean_scaled;
            s_sum_sq[0] = inv_std;
        }
        __syncthreads();
        
        mean_scaled = s_sum[0];
        inv_std = s_sum_sq[0];
    }
    
    // Pass 2: Normalize
    for (int i = tid; i < batch_size; i += block_dim) {
        float val = __ldg(&input[i * features + c]);
        float val_scaled = val * s;
        float val_norm = (val_scaled - mean_scaled) * inv_std;
        output[i * features + c] = val_norm * gamma + beta;
    }
}

torch::Tensor fused_scale_bn_cuda(
    torch::Tensor input, torch::Tensor scale, 
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    float eps, float momentum, bool training) {
    
    auto batch_size = input.size(0);
    auto features = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 1024;
    const int num_blocks = features;
    const int shared_mem_size = 2 * block_size * sizeof(float);

    fused_scale_bn_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, features, eps, momentum, training
    );

    return output;
}
"""

fused_scale_bn_cpp_source = (
    "torch::Tensor fused_scale_bn_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor running_mean, torch::Tensor running_var, float eps, float momentum, bool training);"
)

# Compile the inline CUDA code
fused_scale_bn = load_inline(
    name="fused_scale_bn",
    cpp_sources=fused_scale_bn_cpp_source,
    cuda_sources=fused_scale_bn_source,
    functions=["fused_scale_bn_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, scales the result, and applies batch normalization.
    The scaling and batch normalization are fused into a single custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.fused_scale_bn = fused_scale_bn

    def forward(self, x):
        x = self.gemm(x)
        # Fused Scale + BatchNorm
        x = self.fused_scale_bn.fused_scale_bn_cuda(
            x, 
            self.scale, 
            self.bn.weight, 
            self.bn.bias, 
            self.bn.running_mean, 
            self.bn.running_var, 
            self.bn.eps, 
            self.bn.momentum, 
            self.training
        )
        return x