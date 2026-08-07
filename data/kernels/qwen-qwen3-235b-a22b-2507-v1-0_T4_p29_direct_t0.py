import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused linear + mish + mish
fused_linear_mish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float mish_activation(float x) {
    const float threshold = 20.0f;
    if (x > threshold) {
        return x;
    }
    float exp_x = expf(x);
    float softplus = (x > threshold) ? x : logf(1.0f + exp_x);
    float tanh_softplus = tanhf(softplus);
    return x * tanh_softplus;
}

__global__ void fused_linear_mish_mish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // batch index
    int col = blockIdx.y * blockDim.y + threadIdx.y; // output feature index

    if (row < batch_size && col < out_features) {
        float sum = bias[col];
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        // First Mish
        float activated1 = mish_activation(sum);
        // Second Mish
        float activated2 = mish_activation(activated1);
        
        output[row * out_features + col] = activated2;
    }
}

torch::Tensor fused_linear_mish_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = bias.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const dim3 block_size(16, 16);
    const dim3 grid_size(
        (batch_size + block_size.x - 1) / block_size.x,
        (out_features + block_size.y - 1) / block_size.y
    );

    fused_linear_mish_mish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_linear_mish_mish_cpp_source = """
torch::Tensor fused_linear_mish_mish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA kernel
fused_linear_mish_mish = load_inline(
    name="fused_linear_mish_mish",
    cpp_sources=fused_linear_mish_mish_cpp_source,
    cuda_sources=fused_linear_mish_mish_source,
    functions=["fused_linear_mish_mish_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused linear + mish + mish in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_op = fused_linear_mish_mish.fused_linear_mish_mish_cuda

    def forward(self, x):
        return self.fused_op(x, self.linear.weight, self.linear.bias)