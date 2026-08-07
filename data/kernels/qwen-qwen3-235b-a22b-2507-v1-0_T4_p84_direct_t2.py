import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Gemm + BatchNorm1d + Scale + Softmax
fused_gemm_bn_scale_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

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
    int tid = threadIdx.x;

    if (row >= batch_size) return;

    // Shared memory for input and output values
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_vals = &shared_mem[in_features];

    float sum = 0.0f;

    // Load input vector tile by tile
    for (int k = tid; k < in_features; k += blockDim.x) {
        s_input[k] = input[row * in_features + k];
    }
    __syncthreads();

    // Compute GEMM: output[j] = sum_i (weight[j][i] * input[i])
    for (int i = 0; i < in_features; ++i) {
        sum += weight[tid * in_features + i] * s_input[i];
    }

    // Add bias
    sum += bias[tid];

    // BatchNorm: (sum - mean) / sqrt(var + eps) * gamma + beta
    float bn_val = (sum - bn_running_mean[tid]) * rsqrtf(bn_running_var[tid] + bn_eps);
    bn_val = bn_val * bn_weight[tid] + bn_bias[tid];

    // Scale
    float scaled_val = scale[0] * bn_val;
    s_vals[tid] = scaled_val;
    __syncthreads();

    // Softmax: subtract max for numerical stability
    float max_val = scaled_val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);

    float exp_val = expf(scaled_val - max_val);
    s_vals[tid] = exp_val;
    __syncthreads();

    // Block-wide reduction for sum of exps
    float sum_exp = exp_val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }
    sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp, 0);

    // Normalize
    output[row * out_features + tid] = exp_val / sum_exp;
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

    fused_gemm_bn_scale_softmax_kernel<<<grid_size, block_size, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        bn_eps
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