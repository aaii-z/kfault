import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + ReLU + division operation
fused_linear_relu_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_relu_div_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float divisor
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output feature index

    if (row >= batch_size || col >= out_features) return;

    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];
    sum = fmaxf(0.0f, sum); // ReLU
    sum /= divisor; // Division

    output[row * out_features + col] = sum;
}

torch::Tensor fused_linear_relu_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const dim3 block_size(16, 16);
    const dim3 grid_size(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );

    fused_linear_relu_div_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.contiguous().data_ptr<float>(),
        bias.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        divisor
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return output;
}
"""

fused_linear_relu_div_cpp_source = """
torch::Tensor fused_linear_relu_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_linear_relu_div",
    cpp_sources=fused_linear_relu_div_cpp_source,
    cuda_sources=fused_linear_relu_div_source,
    functions=["fused_linear_relu_div_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused linear, ReLU, and division operations in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.divisor = divisor
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_op.fused_linear_relu_div_cuda(x, self.weight, self.bias, self.divisor)