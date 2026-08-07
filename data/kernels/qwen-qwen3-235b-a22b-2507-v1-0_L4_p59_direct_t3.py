import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + swish + scale
fused_matmul_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_matmul_swish_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float scaling_factor,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= out_features) return;

    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];
    float sigmoid = 1.0f / (1.0f + expf(-sum));
    float swish = sum * sigmoid;
    output[row * out_features + col] = swish * scaling_factor;
}

torch::Tensor fused_matmul_swish_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    dim3 block_size(16, 16);
    dim3 grid_size((out_features + block_size.x - 1) / block_size.x,
                   (batch_size + block_size.y - 1) / block_size.y);

    fused_matmul_swish_scale_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_matmul_swish_scale_cpp_source = """
torch::Tensor fused_matmul_swish_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor);
"""

fused_op = load_inline(
    name="fused_matmul_swish_scale",
    cpp_sources=fused_matmul_swish_scale_cpp_source,
    cuda_sources=fused_matmul_swish_scale_source,
    functions=["fused_matmul_swish_scale_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused matmul + swish activation + scaling in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_matmul_swish_scale_cuda(
            x, self.linear.weight, self.linear.bias, self.scaling_factor
        )