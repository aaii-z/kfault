import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + BiasAdd + Hardtanh + Mish + GroupNorm with high parallelism and memory coalescing
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256
#define TILE_SIZE 16

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

// Shared memory for partial sums in GEMM
extern __shared__ float shared_mem[];

// Fused forward kernel with high parallelism
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
    const int group_size = out_features / num_groups;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // Each thread handles multiple elements in input dim for GEMM
    float thread_input[BLOCK_SIZE / 16]; // register blocking
    float thread_weight[BLOCK_SIZE / 16];
    float thread_sum = 0.0f;

    // GEMM + BiasAdd + Hardtanh + Mish in one pass
    for (int idx = bid; idx < batch_size * out_features; idx += total_threads) {
        int batch_idx = idx / out_features;
        int feat_idx = idx % out_features;
        int group_idx = feat_idx / group_size;

        // GEMM with loop unrolling and coalesced access
        float sum = 0.0f;
        int k = 0;
        if (in_features % 4 == 0) {
            for (; k < in_features - 3; k += 4) {
                int in_offset = batch_idx * in_features + k;
                int w_offset = feat_idx * in_features + k;
                sum += input[in_offset + 0] * weight[w_offset + 0] +
                       input[in_offset + 1] * weight[w_offset + 1] +
                       input[in_offset + 2] * weight[w_offset + 2] +
                       input[in_offset + 3] * weight[w_offset + 3];
            }
        }
        for (; k < in_features; ++k) {
            sum += input[batch_idx * in_features + k] * weight[feat_idx * in_features + k];
        }
        sum += bias[feat_idx];

        // Hardtanh
        float ht = fmaxf(-1.0f, fminf(1.0f, sum));

        // Mish: x * tanh(softplus(x))
        float sp = softplus(ht);
        float ms = ht * fast_tanh(sp);

        // Store temporary Mish output to shared memory for GroupNorm reduction
        __shared__ float s_mish[TILE_SIZE * TILE_SIZE];
        __shared__ float s_mean, s_var;

        int local_idx = feat_idx % group_size;
        s_mish[local_idx] = ms;
        __syncthreads();

        // GroupNorm: compute mean within group
        if (local_idx == 0) {
            float mean = 0.0f;
            for (int i = 0; i < group_size; ++i) {
                mean += s_mish[i];
            }
            s_mean = mean / group_size;
        }
        __syncthreads();

        // Compute variance
        if (local_idx == 0) {
            float var = 0.0f;
            for (int i = 0; i < group_size; ++i) {
                float diff = s_mish[i] - s_mean;
                var += diff * diff;
            }
            s_var = rsqrtf(var / group_size + 1e-5f);
        }
        __syncthreads();

        // Normalize and apply affine
        float norm = (ms - s_mean) * s_var;
        output[idx] = norm * gamma[group_idx] + beta[group_idx];
    }
}

// Host wrapper function
torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    const int grid_size = min(65535, (batch_size * out_features + block_size - 1) / block_size);
    const int shared_mem_size = 0; // We use small shared memory per block, but not dynamically allocated

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

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
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