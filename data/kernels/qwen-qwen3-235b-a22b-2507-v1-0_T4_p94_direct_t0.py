import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + BiasAdd + Hardtanh + Mish + GroupNorm
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Forward kernel for fused GEMM + BiasAdd + Hardtanh + Mish + GroupNorm
__global__ void fused_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups,
    int group_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx >= total_elements) return;

    int batch_idx = idx / out_features;
    int feat_idx = idx % out_features;
    int group_idx = feat_idx / group_size;

    // GEMM: compute matmul + bias
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += input[batch_idx * in_features + k] * weight[feat_idx * in_features + k];
    }
    sum += bias[feat_idx];

    // Hardtanh: clamp between -1 and 1
    float hardtanh_out = fmaxf(-1.0f, fminf(1.0f, sum));

    // Mish: x * tanh(softplus(x))
    float softplus = logf(1.0f + expf(hardtanh_out));
    float mish_out = hardtanh_out * tanhf(softplus);

    // GroupNorm: per-group normalization
    // First, compute mean for the group
    float mean = 0.0f;
    for (int i = 0; i < group_size; ++i) {
        int offset = group_idx * group_size + i;
        float x = output[batch_idx * out_features + offset];
        // Reuse output buffer temporarily to store pre-activation
        if (offset == feat_idx) {
            x = mish_out;
        } else {
            float s = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                s += input[batch_idx * in_features + k] * weight[offset * in_features + k];
            }
            s += bias[offset];
            s = fmaxf(-1.0f, fminf(1.0f, s));
            float sp = logf(1.0f + expf(s));
            x = s * tanhf(sp);
        }
        mean += x;
    }
    mean /= group_size;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < group_size; ++i) {
        int offset = group_idx * group_size + i;
        float x = output[batch_idx * out_features + offset];
        if (offset == feat_idx) {
            x = mish_out;
        } else {
            float s = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                s += input[batch_idx * in_features + k] * weight[offset * in_features + k];
            }
            s += bias[offset];
            s = fmaxf(-1.0f, fminf(1.0f, s));
            float sp = logf(1.0f + expf(s));
            x = s * tanhf(sp);
        }
        float diff = x - mean;
        var += diff * diff;
    }
    var /= group_size;
    float inv_std = rsqrtf(var + 1e-5f);

    // Normalize and apply affine transform
    float norm = (mish_out - mean) * inv_std;
    output[idx] = norm * gamma[group_idx] + beta[group_idx];
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
    auto group_size = out_features / num_groups;

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    const int total_elements = batch_size * out_features;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        num_groups,
        group_size
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