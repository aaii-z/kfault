import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Linear + Mish + Mish CUDA kernel
fused_linear_mish2_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Mish activation function
__device__ float mish_func(float x) {
    float sp = log1pf(expf(x));
    if (isinf(sp)) sp = x;  // Handle large values
    return x * tanhf(sp);
}

// Fused kernel for Linear + 2x Mish
__global__ void linear_mish2_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        if (bias) {
            sum += bias[col];
        }
        
        // Apply Mish twice
        float temp = mish_func(sum);
        output[row * out_features + col] = mish_func(temp);
    }
}

torch::Tensor fused_linear_mish2_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);

    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    dim3 blocks(
        (out_features + 15) / 16,
        (batch_size + 15) / 16
    );
    dim3 threads(16, 16);

    linear_mish2_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_linear_mish2_cpp_source = """
torch::Tensor fused_linear_mish2_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Load the custom CUDA extension
fused_linear_mish2 = load_inline(
    name="fused_linear_mish2",
    cpp_sources=fused_linear_mish2_cpp_source,
    cuda_sources=fused_linear_mish2_source,
    functions=["fused_linear_mish2_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_linear_mish2.fused_linear_mish2_cuda(x, self.weight, self.bias)