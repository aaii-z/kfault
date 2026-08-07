import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + BiasAdd + Hardtanh + Mish + GroupNorm
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fast approximate tanh using polynomial approximation
__device__ float fast_tanh(float x) {
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

// Fast approximate exp using LUT-free method
__device__ float fast_exp(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) return 3.4e38f;

    float fx = x + 121.2711680312f;
    int i = __float2int_rn(fx);
    fx = fx - i;

    const float c0 = 0.99992520f;
    const float c1 = 0.69583354f;
    const float c2 = 0.22606716f;
    const float c3 = 0.07802452f;
    float y = c0 + fx * (c1 + fx * (c2 + fx * c3));

    int exponent = i - 127;
    return __ldexpf(y, exponent);
}

// Fast approximate log(1 + exp(x)) for softplus
__device__ float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return fast_exp(x);
    return logf(1.0f + fast_exp(x));
}

// Fused forward kernel
__global__ void fused_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int batch_idx = idx / out_features;
    int feat_idx = idx % out_features;
    int group_size = out_features / num_groups;
    int group_idx = feat_idx / group_size;
    int local_idx = feat_idx % group_size;

    // GEMM computation
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += input[batch_idx * in_features + k] * weight[feat_idx * in_features + k];
    }
    sum += bias[feat_idx];

    // Hardtanh: clamp between -1 and 1
    float ht = fmaxf(-1.0f, fminf(1.0f, sum));

    // Mish: x * tanh(softplus(x))
    float sp = softplus(ht);
    float ms = ht * fast_tanh(sp);

    // Store mish result in shared memory for group norm reduction
    extern __shared__ float shared_data[];
    float* s_mish = shared_data;
    float* s_mean = &shared_data[32 * 32]; // assuming max group_size <= 1024, use conservative offset
    float* s_var = &shared_data[32 * 32 + 1];

    s_mish[local_idx] = ms;
    __syncthreads();

    // Compute mean within group
    if (local_idx == 0) {
        float mean = 0.0f;
        for (int i = 0; i < group_size; ++i) {
            mean += s_mish[i];
        }
        *s_mean = mean / group_size;
    }
    __syncthreads();

    // Compute variance
    if (local_idx == 0) {
        float mean = *s_mean;
        float var = 0.0f;
        for (int i = 0; i < group_size; ++i) {
            float diff = s_mish[i] - mean;
            var += diff * diff;
        }
        *s_var = rsqrtf(var / group_size + 1e-5f);
    }
    __syncthreads();

    // Normalize and apply affine transform
    float mean = *s_mean;
    float inv_var = *s_var;
    float norm = (ms - mean) * inv_var;
    output[idx] = norm * gamma[group_idx] + beta[group_idx];
}
"""

fused_cpp_source = """
torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups
);
"""

# Host wrapper function (defined in CUDA source via pybind11)
fused_wrapper = """
torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    const int grid_size = (batch_size * out_features + block_size - 1) / block_size;
    const int max_group_size = 1024;
    const int shared_mem_size = max_group_size * sizeof(float) + 2 * sizeof(float); // for mean, var

    fused_forward_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        num_groups
    );

    return output;
}
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=[fused_cpp_source, fused_wrapper],
    cuda_sources=[fused_kernel_source],
    functions=["fused_forward"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + BiasAdd + Hardtanh + Mish + GroupNorm in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.num_groups = num_groups
        self.fused_forward = fused_ops.fused_forward

    def forward(self, x):
        return self.fused_forward(x, self.linear.weight, self.bias, self.groupnorm.weight, self.groupnorm.bias, self.num_groups)