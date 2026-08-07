import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Gemm + BatchNorm1d + Scale + Softmax
fused_gemm_bn_scale_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void fused_gemm_bn_scale_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ bn_running_mean,
    const float* __restrict__ bn_running_var,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float bn_eps
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Shared memory for reduction in GEMM
    extern __shared__ float s_input[];

    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        s_input[threadIdx.x] = input[row * in_features + k];
        __syncthreads();

        // Compute GEMM: sum += weight[j][k] * input[row][k]
        for (int i = 0; i < blockDim.x && i < in_features; ++i) {
            int k_idx = threadIdx.x - i;
            if (k_idx >= 0 && k_idx < in_features) {
                sum += weight[threadIdx.x * in_features + k_idx] * s_input[k_idx];
            }
        }
        __syncthreads();
    }

    // Add bias
    sum += bias[threadIdx.x];

    // Apply BatchNorm: (sum - mean) / sqrt(var + eps) * bn_weight + bn_bias
    float mean = bn_running_mean[threadIdx.x];
    float var_inv = rsqrtf(bn_running_var[threadIdx.x] + bn_eps);
    float bn_out = (sum - mean) * var_inv * bn_weight[threadIdx.x] + bn_bias[threadIdx.x];

    // Scale
    float scaled = scale[0] * bn_out;

    // Store pre-softmax values in shared memory for softmax computation
    __shared__ float s_output[1024]; // Assuming out_features <= 1024
    s_output[threadIdx.x] = scaled;
    __syncthreads();

    // Softmax: subtract max for numerical stability
    float max_val = s_output[0];
    for (int i = 1; i < out_features; ++i) {
        max_val = fmaxf(max_val, s_output[i]);
    }
    __syncthreads();

    // Compute exp(x - max) and sum
    float exp_val = expf(s_output[threadIdx.x] - max_val);
    s_output[threadIdx.x] = exp_val;
    __syncthreads();

    float sum_exp = 0.0f;
    for (int i = 0; i < out_features; ++i) {
        sum_exp += s_output[i];
    }

    // Normalize
    output[row * out_features + threadIdx.x] = exp_val / sum_exp;
}
"""

fused_gemm_bn_scale_softmax_cpp_source = """
torch::Tensor fused_gemm_bn_scale_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor scale,
    float bn_eps,
    int batch_size,
    int in_features,
    int out_features);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_gemm_bn_scale_softmax",
    cpp_sources=fused_gemm_bn_scale_softmax_cpp_source,
    cuda_sources=fused_gemm_bn_scale_softmax_source,
    functions=["fused_gemm_bn_scale_softmax_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "--extended-lambda"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Gemm + BatchNorm1d + Scale + Softmax in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps

        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.fused_op = fused_op

    def forward(self, x):
        # Extract all required parameters
        weight = self.gemm.weight
        bias = self.gemm.bias
        bn_running_mean = self.bn.running_mean
        bn_running_var = self.bn.running_var
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        scale = self.scale

        # Ensure all tensors are contiguous and on the same device
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        bn_running_mean = bn_running_mean.contiguous()
        bn_running_var = bn_running_var.contiguous()
        bn_weight = bn_weight.contiguous()
        bn_bias = bn_bias.contiguous()
        scale = scale.contiguous()

        # Launch fused kernel
        output = self.fused_op.fused_gemm_bn_scale_softmax_cuda(
            x, weight, bias, bn_running_mean, bn_running_var,
            bn_weight, bn_bias, scale,
            self.bn_eps, x.size(0), self.in_features, self.out_features
        )
        return output