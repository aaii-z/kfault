import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for fused Batch Norm, Bias, Division, and Swish
# This version uses shared memory to cache the input data for a channel,
# enabling a numerically stable two-pass variance calculation and reducing global memory traffic.
fused_bn_swish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel for fused operations: BN + Bias + Div + Swish
// Optimized for row-major input (N, C) where N is batch size, C is features
// Each block handles one channel (feature). Threads in a block handle the batch dimension.
// Uses extern shared memory to cache the input data for the channel.
__global__ void fused_bn_swish_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output,
    const float* __restrict__ bn_weight, 
    const float* __restrict__ bn_bias,
    float* __restrict__ running_mean, 
    float* __restrict__ running_var,
    const float* __restrict__ custom_bias, 
    float divide_value, 
    float eps, 
    float momentum, 
    int N, int C, 
    bool training) {
    
    int c = blockIdx.x;
    if (c >= C) return;

    // Load parameters for this channel
    float gamma = bn_weight[c];
    float beta = bn_bias[c];
    float bias_val = custom_bias[0]; // Shape (1,)

    // Dynamic shared memory for data caching
    extern __shared__ float s_data[];
    
    // Shared memory for reduction results and broadcast values
    __shared__ float s_mean;
    __shared__ float s_var;

    // Phase 1: Load data to shared memory
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        s_data[n] = input[n * C + c];
    }
    __syncthreads();

    float mean_val, var_val;

    if (training) {
        // Phase 2a: Calculate Mean
        double local_sum = 0.0;
        for (int n = threadIdx.x; n < N; n += blockDim.x) {
            local_sum += s_data[n];
        }
        
        // Reduction for sum
        // Using simple atomic add or reduction tree? 
        // Since N is small (1024), we can use a simple approach or standard reduction.
        // Let's use standard reduction in shared mem for sum.
        // We need a temporary shared buffer for reduction.
        // We can reuse s_data? No, s_data holds values.
        // We can use the first N elements of a separate shared buffer, or just atomicAdd for simplicity given N is small?
        // Or better: since we have s_data, we can sum in registers and reduce.
        
        // Reduction in shared memory requires a buffer. 
        // Let's declare a small buffer in the extern shared mem if possible, 
        // or just use a separate __shared__ float s_reduce[1024] (fixed max N).
        // Given N=1024, fixed size is okay.
        __shared__ float s_reduce[1024]; 

        s_reduce[threadIdx.x] = (threadIdx.x < N) ? local_sum : 0.0f;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                s_reduce[threadIdx.x] += s_reduce[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float mean = s_reduce[0] / N;
            s_mean = mean;
        }
        __syncthreads();
        mean_val = s_mean;

        // Phase 2b: Calculate Variance (using the stable two-pass algorithm)
        double local_sum_sq = 0.0;
        for (int n = threadIdx.x; n < N; n += blockDim.x) {
            float diff = s_data[n] - mean_val;
            local_sum_sq += diff * diff;
        }
        
        s_reduce[threadIdx.x] = (threadIdx.x < N) ? local_sum_sq : 0.0f;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                s_reduce[threadIdx.x] += s_reduce[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float var = s_reduce[0] / N;
            s_var = var;
            
            // Update running mean and var
            // running_mean = (1 - momentum) * running_mean + momentum * mean
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean_val;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;
        }
        __syncthreads();
        var_val = s_var;
    } 
    else {
        // Inference mode: Use running statistics
        mean_val = running_mean[c];
        var_val = running_var[c];
    }

    // Phase 3: Normalize and apply activations
    // Use 1.0f / sqrtf for better precision than rsqrtf
    float inv_std = 1.0f / sqrtf(var_val + eps);
    
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float val = s_data[n];
        // Batch Norm
        float x_norm = (val - mean_val) * inv_std;
        float y = x_norm * gamma + beta;
        // Bias Add
        y = y + bias_val;
        // Division
        y = y / divide_value;
        // Swish: x * sigmoid(x)
        float sig = 1.0f / (1.0f + expf(-y));
        y = y * sig;
        output[n * C + c] = y;
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input, 
    torch::Tensor bn_weight, torch::Tensor bn_bias, 
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor custom_bias, 
    double divide_value, double eps, double momentum, 
    bool training) {
    
    // input is (N, C)
    int N = input.size(0);
    int C = input.size(1);
    
    auto output = torch::empty_like(input);
    
    // Launch configuration
    // One block per channel (feature)
    int threads = 256; 
    // If N is small, we don't need many threads. If N is large, we loop.
    // 1024 is max threads per block. 256 is a good balance.
    int blocks = C;
    
    // Shared memory size: N floats for data caching + N floats for reduction buffer (worst case)
    // Actually, we can put the reduction buffer in the same shared memory block or fixed size.
    // The kernel uses `extern __shared__ float s_data[]` and `__shared__ float s_reduce[1024]`.
    // Note: Fixed size shared memory inside kernel counts towards limit.
    // Dynamic shared memory is passed as 3rd arg to launch.
    size_t shared_mem_size = N * sizeof(float);
    
    fused_bn_swish_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(),
        bn_weight.data_ptr<float>(), 
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), 
        running_var.data_ptr<float>(),
        custom_bias.data_ptr<float>(), 
        (float)divide_value, 
        (float)eps, 
        (float)momentum, 
        N, C, 
        training);
        
    return output;
}
"""

fused_bn_swish_cpp_source = (
    "torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor custom_bias, double divide_value, double eps, double momentum, bool training);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_bn_swish",
    cpp_sources=fused_bn_swish_cpp_source,
    cuda_sources=fused_bn_swish_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication followed by a fused
    Batch Normalization, bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # MatMul weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_linear = nn.Parameter(torch.Tensor(out_features))
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.Tensor(out_features))
        self.bn_bias = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # Custom bias and divide value
        self.bias_custom = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        
        # Configuration
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        
        # Initialize parameters
        self.reset_parameters()
        
        # Load custom op
        self.fused_op = fused_op

    def reset_parameters(self):
        # Initialize Linear weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_linear is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_linear, -bound, bound)
            
        # Initialize BatchNorm weights and bias
        nn.init.ones_(self.bn_weight)
        nn.init.zeros_(self.bn_bias)

    def forward(self, x):
        # 1. Matrix Multiplication (Linear)
        # We keep the matmul as PyTorch's implementation is highly optimized (cuBLAS)
        x = F.linear(x, self.weight, self.bias_linear)
        
        # 2. Fused Operation: BN + Bias + Div + Swish
        # We handle the training/eval mode logic for BN inside the kernel
        x = self.fused_op.fused_forward_cuda(
            x, 
            self.bn_weight, 
            self.bn_bias, 
            self.running_mean, 
            self.running_var,
            self.bias_custom, 
            self.divide_value, 
            self.bn_eps, 
            self.bn_momentum, 
            self.training
        )
        
        return x