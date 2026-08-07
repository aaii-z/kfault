import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for fused Batch Norm, Bias, Division, and Swish
fused_bn_swish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel for fused operations: BN + Bias + Div + Swish
// Optimized for row-major input (N, C) where N is batch size, C is features
// Each block handles one channel (feature). Threads in a block handle the batch dimension.
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
    
    // Shared memory for reduction (sum and sum of squares)
    // Using a struct or separate arrays. Max block size assumed 1024.
    __shared__ float s_sum[1024];
    __shared__ float s_sum_sq[1024];

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Phase 1: Calculate Sum and Sum of Squares (for training) or just load stats (for inference)
    // Loop over the batch dimension (N) with stride blockDim.x
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float val = input[n * C + c];
        if (training) {
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    if (training) {
        s_sum[threadIdx.x] = local_sum;
        s_sum_sq[threadIdx.x] = local_sum_sq;
        __syncthreads();

        // Parallel reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
                s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
            }
            __syncthreads();
        }

        // Thread 0 computes mean, variance, and updates running stats
        if (threadIdx.x == 0) {
            float mean = s_sum[0] / N;
            // Variance = E[x^2] - E[x]^2
            float var = s_sum_sq[0] / N - mean * mean;
            
            // Update running mean and var
            // running_mean = (1 - momentum) * running_mean + momentum * mean
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;
            
            // Store back to shared memory for other threads to use
            s_sum[0] = mean;
            s_sum[1] = var;
        }
        __syncthreads();
        
        // Broadcast mean and var from shared memory
        float mean = s_sum[0];
        float var = s_sum[1];
        
        // Compute inverse standard deviation
        float inv_std = rsqrtf(var + eps);
        
        // Phase 2: Normalize and apply activations
        for (int n = threadIdx.x; n < N; n += blockDim.x) {
            float val = input[n * C + c];
            // Batch Norm
            float x_norm = (val - mean) * inv_std;
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
    else {
        // Inference mode: Use running statistics
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);

        for (int n = threadIdx.x; n < N; n += blockDim.x) {
            float val = input[n * C + c];
            // Batch Norm
            float x_norm = (val - mean) * inv_std;
            float y = x_norm * gamma + beta;
            // Bias Add
            y = y + bias_val;
            // Division
            y = y / divide_value;
            // Swish
            float sig = 1.0f / (1.0f + expf(-y));
            y = y * sig;
            output[n * C + c] = y;
        }
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
    
    fused_bn_swish_kernel<<<blocks, threads>>>(
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