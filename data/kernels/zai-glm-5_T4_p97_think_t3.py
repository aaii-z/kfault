import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for Matrix Multiplication: y = x * w^T + b
// x: (batch, in_features), w: (out_features, in_features) -> stored as (in_features, out_features) for coalescing
// We expect w to be transposed in the model initialization to (in_features, out_features)
// So we compute y = x * w
// x: (batch, in), w: (in, out), b: (out), y: (batch, out)

__global__ void matmul_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ y, int batch, int in, int out) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * 32 + tx; // out dimension
    int row = blockIdx.y * 32 + ty; // batch dimension

    float val = 0.0f;

    // Loop over tiles in the K dimension (in_features)
    for (int k = 0; k < in; k += 32) {
        __shared__ float s_x[32][33]; // Padding to avoid bank conflicts
        __shared__ float s_w[32][33];

        // Load x tile
        // x is (batch, in)
        if (row < batch && (k + tx) < in) {
            s_x[ty][tx] = x[row * in + k + tx];
        } else {
            s_x[ty][tx] = 0.0f;
        }

        // Load w tile
        // w is (in, out)
        if ((k + ty) < in && col < out) {
            s_w[ty][tx] = w[(k + ty) * out + col];
        } else {
            s_w[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < 32; ++i) {
            val += s_x[ty][i] * s_w[i][tx];
        }
        __syncthreads();
    }

    if (row < batch && col < out) {
        y[row * out + col] = val + b[col];
    }
}

// Kernel for Fused Batch Normalization + Bias + Divide + Swish
// y: (batch, out)
// One block per feature (out_features)
// Block size = batch_size (1024)

__global__ void bn_act_kernel(const float* __restrict__ y, float* __restrict__ out, 
                               const float* __restrict__ gamma, const float* __restrict__ beta, 
                               float* __restrict__ running_mean, float* __restrict__ running_var, 
                               const float* __restrict__ bias, float divide_val, 
                               float eps, float momentum, int batch, int out_size) {
    int j = blockIdx.x; // feature index (0 to out_size-1)
    int i = threadIdx.x; // batch index (0 to batch-1)
    
    // Shared memory for reduction
    // We use a simple approach: load value, reduce in shared memory
    __shared__ float s_data[1024]; // Assuming batch_size is 1024
    
    float val = 0.0f;
    if (i < batch) {
        val = y[i * out_size + j];
    }
    s_data[i] = val;
    __syncthreads();

    // Reduction for Mean
    // Assuming batch_size is power of 2 (1024)
    for (unsigned int s = batch / 2; s > 0; s >>= 1) {
        if (i < s) {
            s_data[i] += s_data[i + s];
        }
        __syncthreads();
    }
    
    float mean = s_data[0] / batch;
    
    // Recompute variance
    // Reload data to s_data as (x - mean)^2
    if (i < batch) {
        float diff = y[i * out_size + j] - mean;
        s_data[i] = diff * diff;
    }
    __syncthreads();
    
    // Reduction for Variance
    for (unsigned int s = batch / 2; s > 0; s >>= 1) {
        if (i < s) {
            s_data[i] += s_data[i + s];
        }
        __syncthreads();
    }
    
    float var_biased = s_data[0] / batch;
    
    // Update running stats (only thread 0)
    if (i == 0) {
        // PyTorch uses unbiased variance for running stats update: var_unbiased = var_biased * N / (N-1)
        float var_unbiased = var_biased * (float)batch / (float)(batch - 1);
        running_mean[j] = (1.0f - momentum) * running_mean[j] + momentum * mean;
        running_var[j] = (1.0f - momentum) * running_var[j] + momentum * var_unbiased;
    }
    
    // Normalization and Activation
    float inv_std = rsqrtf(var_biased + eps);
    float x_norm = (val - mean) * inv_std;
    
    // Apply BN parameters
    float res = gamma[j] * x_norm + beta[j];
    
    // Add bias
    res += bias[0]; // bias is (1,)
    
    // Divide
    res /= divide_val;
    
    // Swish: x * sigmoid(x)
    // Sigmoid approximation or direct expf
    float sig = 1.0f / (1.0f + expf(-res));
    res = res * sig;
    
    if (i < batch) {
        out[i * out_size + j] = res;
    }
}

// C++ interface
torch::Tensor forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b_linear, 
                           torch::Tensor gamma, torch::Tensor beta, 
                           torch::Tensor running_mean, torch::Tensor running_var, 
                           torch::Tensor bias, float divide_val, float eps, float momentum) {
    int batch = x.size(0);
    int in_features = x.size(1);
    int out_features = w.size(1); // w is (in, out)
    
    auto y = torch::empty({batch, out_features}, x.options());
    auto out = torch::empty({batch, out_features}, x.options());
    
    // Launch MatMul kernel
    dim3 block(32, 32);
    dim3 grid((out_features + 31) / 32, (batch + 31) / 32);
    matmul_kernel<<<grid, block>>>(x.data_ptr<float>(), w.data_ptr<float>(), b_linear.data_ptr<float>(), 
                                   y.data_ptr<float>(), batch, in_features, out_features);
    
    // Launch BN+Act kernel
    // Assuming batch_size is 1024 as per prompt
    int threads = 1024; 
    // If batch size is dynamic, we need to handle reduction differently, 
    // but prompt implies fixed batch size 1024.
    // We clamp threads to batch size just in case.
    if (batch < 1024) threads = batch;
    
    dim3 grid2(out_features);
    bn_act_kernel<<<grid2, threads>>>(y.data_ptr<float>(), out.data_ptr<float>(), 
                                      gamma.data_ptr<float>(), beta.data_ptr<float>(), 
                                      running_mean.data_ptr<float>(), running_var.data_ptr<float>(), 
                                      bias.data_ptr<float>(), divide_val, eps, momentum, batch, out_features);
    
    return out;
}
"""

cpp_source = """
torch::Tensor forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b_linear, 
                           torch::Tensor gamma, torch::Tensor beta, 
                           torch::Tensor running_mean, torch::Tensor running_var, 
                           torch::Tensor bias, float divide_val, float eps, float momentum);
"""

# Compile
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["forward_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.divide_value = divide_value

        # Weights for Linear
        # Note: We store weights as (in_features, out_features) to optimize memory access in CUDA kernel
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias_linear = nn.Parameter(torch.empty(out_features))
        
        # Parameters for BatchNorm
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        
        # Buffers for BatchNorm running stats
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        
        # Extra bias
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Initialize weights similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_linear is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_linear, -bound, bound)

    def forward(self, x):
        # Ensure inputs are contiguous
        x = x.contiguous()
        
        return custom_ops.forward_cuda(
            x, 
            self.weight, 
            self.bias_linear, 
            self.bn_weight, 
            self.bn_bias, 
            self.running_mean, 
            self.running_var, 
            self.bias, 
            self.divide_value, 
            self.bn_eps, 
            self.bn_momentum
        )

import math