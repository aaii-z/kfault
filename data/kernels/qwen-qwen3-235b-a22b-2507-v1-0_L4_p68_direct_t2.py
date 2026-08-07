import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + min + subtract operation
fused_linear_min_sub_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_min_sub_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float constant_val,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx >= total_elements) return;

    int row = idx / out_features;
    int col = idx % out_features;

    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];

    // Apply min(sum, constant) - constant
    float result = fminf(sum, constant_val) - constant_val;
    output[idx] = result;
}

torch::Tensor fused_linear_min_sub_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor constant
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Copy constant value to host once
    float constant_val = constant.item<float>();

    fused_linear_min_sub_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.contiguous().data_ptr<float>(),
        bias.contiguous().data_ptr<float>(),
        constant_val,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    // Synchronize to catch any kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_linear_min_sub_cpp_source = """
torch::Tensor fused_linear_min_sub_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor constant
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_linear_min_sub",
    cpp_sources=fused_linear_min_sub_cpp_source,
    cuda_sources=fused_linear_min_sub_source,
    functions=["fused_linear_min_sub_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with fused linear + min + subtract in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.constant = nn.Parameter(torch.tensor(constant))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        fan_in = in_features
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_linear_min_sub_cuda(x, self.weight, self.bias, self.constant)