import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Gemm + BatchNorm1d + Scale + Softmax
fused_gemm_bn_scale_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16
#define MAX_THREADS_PER_BLOCK 1024

// Efficiently compute batched GEMV with fused BatchNorm, Scale, and Softmax
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

    // Shared memory for input tile
    extern __shared__ float s_input[];
    float s_scale[1];

    // Load scale factor
    if (threadIdx.x == 0) {
        s_scale[0] = scale[0];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < in_features; k += blockDim.x) {
        int k_idx = k + threadIdx.x;
        if (k_idx < in_features) {
            s_input[threadIdx.x] = input[row * in_features + k_idx];
        } else {
            s_input[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < blockDim.x && (k + i) < in_features; ++i) {
            sum += weight[threadIdx.x * in_features + (k + i)] * s_input[i];
        }
        __syncthreads();
    }

    // Add bias
    sum += bias[threadIdx.x];

    // Apply BatchNorm: (sum - mean) / sqrt(var + eps) * weight + bias
    float mean = bn_running_mean[threadIdx.x];
    float inv_std = rsqrtf(bn_running_var[threadIdx.x] + bn_eps);
    float bn_val = (sum - mean) * inv_std * bn_weight[threadIdx.x] + bn_bias[threadIdx.x];

    // Scale
    float scaled_val = s_scale[0] * bn_val;

    // Store in shared memory for softmax
    extern __shared__ float s_buf[];
    float* s_output = s_buf;
    s_output[threadIdx.x] = scaled_val;
    __syncthreads();

    // Compute max for numerical stability
    float max_val = scaled_val;
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        float temp = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        max_val = fmaxf(max_val, temp);
    }
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);

    // Compute exp(x - max)
    float exp_val = expf(scaled_val - max_val);
    s_output[threadIdx.x] = exp_val;
    __syncthreads();

    // Block-wide sum reduction
    float sum_exp = exp_val;
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }
    sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp, 0);

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
    int out_features) {
    
    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = std::min(1024, out_features);
    const int grid_size = batch_size;
    const int shared_mem_size = (in_features + out_features) * sizeof(float);

    void* kernel_args[] = {
        (void*)&input.data_ptr<float>()[0],
        (void*)&weight.data_ptr<float>()[0],
        (void*)&bias.data_ptr<float>()[0],
        (void*)&bn_running_mean.data_ptr<float>()[0],
        (void*)&bn_running_var.data_ptr<float>()[0],
        (void*)&bn_weight.data_ptr<float>()[0],
        (void*)&bn_bias.data_ptr<float>()[0],
        (void*)&scale.data_ptr<float>()[0],
        (void*)output.data_ptr<float>(),
        (void*)&batch_size,
        (void*)&in_features,
        (void*)&out_features,
        (void*)&bn_eps
    };

    auto stream = at::cuda::getCurrentCUDAStream();

    void (*kernel_ptr)(const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, float*, int, int, int, float) = 
        fused_gemm_bn_scale_softmax_kernel;

    cudaLaunchKernel(
        (void*)kernel_ptr,
        dim3(grid_size),
        dim3(block_size),
        kernel_args,
        shared_mem_size,
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_gemm_bn_scale_softmax",
    cpp_sources=[fused_gemm_bn_scale_softmax_cpp_source],
    cuda_sources=[fused_gemm_bn_scale_softmax_source],
    functions=["fused_gemm_bn_scale_softmax_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"]
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