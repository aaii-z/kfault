import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LayerNorm (interpreting InstanceNorm on features), Add, and Mul
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_ln_add_mul_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ y, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ out, 
    int batch_size, int features, float eps) {
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= batch_size) return;

    const float* x_row = x + row * features;
    const float* y_row = y + row * features;
    float* out_row = out + row * features;
    
    // Shared memory for reduction and caching
    // Size: features (for x cache) + 2 * blockDim.x (for partial sums)
    extern __shared__ float s_mem[];
    float* s_x = s_mem; 
    float* s_sums = s_mem + features; // Partial sums for mean
    float* s_sum_sqs = s_mem + features + blockDim.x; // Partial sums for variance

    // 1. Load x to shared memory and compute partial sums
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = tid; i < features; i += blockDim.x) {
        float val = x_row[i];
        s_x[i] = val;
        sum += val;
        sum_sq += val * val;
    }
    
    s_sums[tid] = sum;
    s_sum_sqs[tid] = sum_sq;
    __syncthreads();

    // 2. Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sums[tid] += s_sums[tid + s];
            s_sum_sqs[tid] += s_sum_sqs[tid + s];
        }
        __syncthreads();
    }

    // 3. Compute mean and variance (only thread 0 needs to do this, then broadcast? 
    // Or just let all threads read s_sums[0] since it's synced)
    float mean = s_sums[0] / features;
    float var = s_sum_sqs[0] / features - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // 4. Normalize, affine, add, mul, and store
    for (int i = tid; i < features; i += blockDim.x) {
        float val = s_x[i];
        float norm_val = (val - mean) * inv_std;
        
        // Apply affine transform (weight/bias from InstanceNorm)
        norm_val = norm_val * weight[i] + bias[i];
        
        // Add and Mul with y
        float y_val = y_row[i];
        float res = (norm_val + y_val) * y_val;
        
        out_row[i] = res;
    }
}

torch::Tensor fused_ln_add_mul_cuda(
    torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor bias, float eps) {
    
    auto batch_size = x.size(0);
    auto features = x.size(1);
    auto out = torch::empty_like(x);

    const int block_size = 1024; 
    // Shared memory: features * sizeof(float) + 2 * block_size * sizeof(float)
    // For features=8192, this is 32768 + 8192 = 40960 bytes. Fits in 48KB limit.
    size_t shared_mem_size = features * sizeof(float) + 2 * block_size * sizeof(float);

    fused_ln_add_mul_kernel<<<batch_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), 
        y.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size, features, eps
    );

    return out;
}
"""

fused_kernel_cpp_source = (
    "torch::Tensor fused_ln_add_mul_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor bias, float eps);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_ln_add_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a batch matrix multiplication, 
    fused normalization/add/mul operation.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        # We keep the InstanceNorm2d module to hold the weights and biases (affine parameters)
        # and to maintain compatibility with the original interface.
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.fused_ops = fused_ops

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # 1. Linear layer
        x = self.bmm(x)
        
        # 2. Fused Normalization + Add + Mul
        # The original code performed InstanceNorm2d on (B, F, 1, 1) which is degenerate.
        # We replace it with LayerNorm-style normalization over the feature dimension,
        # fused with the subsequent addition and multiplication.
        # We use the weights and biases from the InstanceNorm module.
        
        # Ensure inputs are contiguous for the CUDA kernel
        x = x.contiguous()
        y = y.contiguous()
        
        x = self.fused_ops.fused_ln_add_mul_cuda(
            x, 
            y, 
            self.instance_norm.weight, 
            self.instance_norm.bias, 
            self.instance_norm.eps
        )
        
        return x