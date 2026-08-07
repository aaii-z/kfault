import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min, sum, GELU, and bias addition
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <float.h>

// Kernel to compute min over channels, sum over height, GELU, and add bias
extern "C" __global__ void fused_min_sum_gelu_bias_kernel(
    const float* input,
    const float* bias,
    float* output,
    int B, int C, int H, int W) {
    int b = blockIdx.y;
    int w = blockIdx.x;
    if (b >= B || w >= W) return;

    extern __shared__ float shared_min[];
    int tid = threadIdx.x;

    // Compute min over channels for each height position
    if (tid < H) {
        float min_val = FLT_MAX;
        for (int c = 0; c < C; c++) {
            int index = b * C * H * W + c * H * W + tid * W + w;
            float val = input[index];
            if (val < min_val) min_val = val;
        }
        shared_min[tid] = min_val;
    }
    __syncthreads();

    // Sum all min values across height
    float total = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < H; i++) {
            total += shared_min[i];
        }
        // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x = total;
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        float gelu_out = x * cdf;
        float bias_val = *bias;
        output[b * W + w] = gelu_out + bias_val;
    }
}

torch::Tensor fused_min_sum_gelu_bias_cuda(torch::Tensor input, torch::Tensor bias) {
    // Validate input dimensions
    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor");
    TORCH_CHECK(bias.dim() == 3 && bias.size(0) == 1 && bias.size(1) == 1 && bias.size(2) == 1,
                "Bias must be 3D tensor with shape (1,1,1)");

    // Ensure tensors are contiguous
    input = input.contiguous();
    bias = bias.contiguous();

    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Create output tensor of shape [B, 1, 1, W]
    auto output = torch::empty({B, 1, 1, W}, input.options());

    // Configure kernel launch
    dim3 grid(W, B);  // blockIdx.x = w, blockIdx.y = b
    dim3 block(H);     // One thread per height
    size_t shared_mem = H * sizeof(float);

    // Launch kernel
    fused_min_sum_gelu_bias_kernel<<<grid, block, shared_mem>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
    );

    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_min_sum_gelu_bias_cuda(torch::Tensor input, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_min_sum_gelu_bias = load_inline(
    name="fused_min_sum_gelu_bias",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_min_sum_gelu_bias_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_min_sum_gelu_bias

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_min_sum_gelu_bias_cuda(x, self.bias)
        return x