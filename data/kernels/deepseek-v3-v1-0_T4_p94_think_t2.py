import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused BiasAdd + Hardtanh + Mish
fused_act_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Hardtanh clamp values
const float hardtanh_min_val = -1.0f;
const float hardtanh_max_val = 1.0f;

// Stable Mish activation function
__device__ float stable_mish(float x) {
    if (x > 20.0f) {
        return x;
    } else if (x < -20.0f) {
        return expf(x);
    } else {
        return log1pf(expf(x));
    }
}

__global__ void bias_act_kernel(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int bias_idx = idx % out_features;
        float val = input[idx] + bias[bias_idx];
        
        // Apply Hardtanh
        val = fmaxf(hardtanh_min_val, fminf(hardtanh_max_val, val));
        
        // Apply Mish
        float sp = stable_mish(val);
        output[idx] = val * tanhf(sp);
    }
}

torch::Tensor fused_bias_hardtanh_mish(
    torch::Tensor input,
    torch::Tensor bias
) {
    CHECK_INPUT(input);
    CHECK_INPUT(bias);

    int batch_size = input.size(0);
    int out_features = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (batch_size * out_features + threads - 1) / threads;
    
    bias_act_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );

    return output;
}
"""

fused_act_cpp_source = """
torch::Tensor fused_bias_hardtanh_mish(
    torch::Tensor input,
    torch::Tensor bias
);
"""

# Load the custom CUDA extension
fused_act = load_inline(
    name="fused_act",
    cpp_sources=fused_act_cpp_source,
    cuda_sources=fused_act_source,
    functions=["fused_bias_hardtanh_mish"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.fused_act = fused_act

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_act.fused_bias_hardtanh_mish(x, self.bias)
        x = self.groupnorm(x)
        return x