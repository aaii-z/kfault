import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm, Scale, and Softmax
fused_bn_scale_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for Batch Normalization
// Performs in-place Batch Normalization
__global__ void batch_norm_kernel(
    float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    int N, int C, float eps, float momentum, bool training) {
    
    int c = blockIdx.x; // One block per feature channel
    int tid = threadIdx.x;
    
    // Shared memory for reduction
    extern __shared__ float s_data[];
    float* s_sum = s_data;
    float* s_sum_sq = s_data + blockDim.x;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Strided access to handle N dimension
    // Input shape is (N, C), accessing (n, c)
    // x[n * C + c]
    
    for (int n = tid; n < N; n += blockDim.x) {
        float val = x[n * C + c];
        sum += val;
        sum_sq += val * val;
    }
    
    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();
    
    // Block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float mean = s_sum[0] / N;
        float var = s_sum_sq[0] / N - mean * mean;
        
        if (training) {
            // Update running stats
            // running_mean = (1 - momentum) * running_mean + momentum * mean
            // Note: PyTorch uses exponential moving average
            // The user provided momentum matches PyTorch's default logic usually
            // But PyTorch BN momentum is 1 - momentum_factor usually? 
            // PyTorch docs: x_new = (1 - momentum) * x + momentum * x_observed
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            // Use unbiased variance for running_var
            float var_unbiased = (N > 1) ? (var * N / (N - 1)) : var;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var_unbiased;
        } else {
            // Inference mode: use running stats
            mean = running_mean[c];
            var = running_var[c];
        }
        
        // Store mean and var to shared memory for other threads
        s_sum[0] = mean;
        s_sum[1] = var;
    }
    __syncthreads();
    
    float mean = s_sum[0];
    float var = s_sum[1];
    
    float inv_std = 1.0f / sqrtf(var + eps);
    float w = weight ? weight[c] : 1.0f;
    float b = bias ? bias[c] : 0.0f;
    
    // Apply normalization in-place
    for (int n = tid; n < N; n += blockDim.x) {
        float val = x[n * C + c];
        x[n * C + c] = (val - mean) * inv_std * w + b;
    }
}

// Kernel for Scale and Softmax
// Performs in-place scaling and softmax
template <int BLOCK_SIZE>
__global__ void scale_softmax_kernel(
    float* __restrict__ x,
    const float* __restrict__ scale,
    int N, int C) {
    
    int n = blockIdx.x; // One block per row (batch item)
    int tid = threadIdx.x;
    
    float scale_val = scale[0];
    
    // Pointers
    float* row_x = x + n * C;
    
    // 1. Find Max (for numerical stability)
    // 2. Compute Sum of Exp
    // 3. Normalize
    
    float max_val = -INFINITY;
    
    // Vectorized load for better throughput
    // Assuming C is divisible by 4 (8192 is)
    int vec_C = C / 4;
    float4* row_x_vec = reinterpret_cast<float4*>(row_x);
    
    // Find Max
    for (int i = tid; i < vec_C; i += BLOCK_SIZE) {
        float4 val_vec = row_x_vec[i];
        val_vec.x *= scale_val;
        val_vec.y *= scale_val;
        val_vec.z *= scale_val;
        val_vec.w *= scale_val;
        
        // Write back scaled values (we need to do this anyway for softmax input)
        // Actually, let's write back later. Just find max first.
        // Wait, we need the scaled values for exp.
        // Let's find max of scaled values.
        max_val = fmaxf(max_val, val_vec.x);
        max_val = fmaxf(max_val, val_vec.y);
        max_val = fmaxf(max_val, val_vec.z);
        max_val = fmaxf(max_val, val_vec.w);
    }
    
    __shared__ float s_max;
    __shared__ float s_sum;
    
    // Reduce max within block
    // Using shared memory reduction
    __shared__ float s_data[BLOCK_SIZE];
    s_data[tid] = max_val;
    __syncthreads();
    
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        s_max = s_data[0];
    }
    __syncthreads();
    
    max_val = s_max;
    
    // Compute Exp and Sum
    float sum_exp = 0.0f;
    
    for (int i = tid; i < vec_C; i += BLOCK_SIZE) {
        float4 val_vec = row_x_vec[i];
        val_vec.x = expf(val_vec.x * scale_val - max_val);
        val_vec.y = expf(val_vec.y * scale_val - max_val);
        val_vec.z = expf(val_vec.z * scale_val - max_val);
        val_vec.w = expf(val_vec.w * scale_val - max_val);
        
        // Write back exp values to global memory temporarily?
        // Or keep in registers? 
        // We need to read them again to normalize.
        // Since we are doing in-place, we can write them back to x.
        row_x_vec[i] = val_vec;
        
        sum_exp += val_vec.x + val_vec.y + val_vec.z + val_vec.w;
    }
    
    s_data[tid] = sum_exp;
    __syncthreads();
    
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        s_sum = s_data[0];
    }
    __syncthreads();
    
    float inv_sum = 1.0f / s_sum;
    
    // Normalize
    for (int i = tid; i < vec_C; i += BLOCK_SIZE) {
        float4 val_vec = row_x_vec[i];
        val_vec.x *= inv_sum;
        val_vec.y *= inv_sum;
        val_vec.z *= inv_sum;
        val_vec.w *= inv_sum;
        row_x_vec[i] = val_vec;
    }
}

// C++ wrapper function
torch::Tensor fused_bn_scale_softmax_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    bool training,
    float eps,
    float momentum) {
    
    int N = x.size(0);
    int C = x.size(1);
    
    // Ensure x is contiguous and float
    x = x.contiguous();
    
    // Launch BatchNorm kernel
    // Grid: C blocks (one per feature)
    // Block: 256 threads (enough to cover N=1024 with loop)
    int threads_bn = 256;
    int blocks_bn = C;
    size_t shared_mem_bn = 2 * threads_bn * sizeof(float);
    
    batch_norm_kernel<<<blocks_bn, threads_bn, shared_mem_bn>>>(
        x.data_ptr<float>(),
        bn_weight.defined() ? bn_weight.data_ptr<float>() : nullptr,
        bn_bias.defined() ? bn_bias.data_ptr<float>() : nullptr,
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        N, C, eps, momentum, training
    );
    
    // Launch Softmax kernel
    // Grid: N blocks (one per row)
    // Block: 256 threads
    int threads_sm = 256;
    int blocks_sm = N;
    
    scale_softmax_kernel<256><<<blocks_sm, threads_sm>>>(
        x.data_ptr<float>(),
        scale.data_ptr<float>(),
        N, C
    );
    
    return x;
}
"""

fused_bn_scale_softmax_cpp_source = (
    "torch::Tensor fused_bn_scale_softmax_cuda("
    "torch::Tensor x, torch::Tensor scale, torch::Tensor bn_weight, torch::Tensor bn_bias, "
    "torch::Tensor bn_running_mean, torch::Tensor bn_running_var, "
    "bool training, float eps, float momentum);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_bn_scale_softmax_cpp_source,
    cuda_sources=fused_bn_scale_softmax_source,
    functions=["fused_bn_scale_softmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Gemm followed by a fused
    BatchNorm + Scale + Softmax operation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        # We keep the original BatchNorm1d layer to handle the state (running_mean, running_var)
        # and weight/bias initialization, but we won't call its forward method.
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)
        
        self.fused_ops = fused_ops
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

    def forward(self, x):
        # 1. Gemm
        x = self.gemm(x)
        
        # 2. Fused BN + Scale + Softmax
        # We use the custom CUDA kernel here.
        # Note: We pass self.training to the kernel so it can update running stats if needed.
        x = self.fused_ops.fused_bn_scale_softmax_cuda(
            x, 
            self.scale,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.training,
            self.bn_eps,
            self.bn_momentum
        )
        return x