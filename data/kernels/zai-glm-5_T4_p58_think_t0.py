import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LogSumExp, HardSwish, Sub, Clamp
fused_op_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int D, int H, int W) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D * H * W;

    if (index < total_elements) {
        // Calculate indices for the 5D tensor
        int w_idx = index % W;
        int h_idx = (index / W) % H;
        int d_idx = (index / (W * H)) % D;
        int n_idx = index / (W * H * D);

        // Calculate strides for input tensor (N, C, D, H, W)
        // We use long long to avoid overflow for large tensors
        long long n_stride = (long long)C * D * H * W;
        long long c_stride = (long long)D * H * W;
        long long d_stride = (long long)H * W;
        long long h_stride = (long long)W;
        
        long long base_offset = (long long)n_idx * n_stride + (long long)d_idx * d_stride + (long long)h_idx * h_stride + w_idx;

        // 1. LogSumExp reduction over channels
        float max_val = -INFINITY;
        for (int c = 0; c < C; ++c) {
            float val = input[base_offset + c * c_stride];
            if (val > max_val) {
                max_val = val;
            }
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            float val = input[base_offset + c * c_stride];
            sum_exp += expf(val - max_val);
        }

        float result = logf(sum_exp) + max_val;

        // 2. HardSwish: x * sigmoid(x + 3) / 6
        float x_plus_3 = result + 3.0f;
        float sig = 1.0f / (1.0f + expf(-x_plus_3));
        result = result * sig / 6.0f;

        // 3. Subtraction
        result -= bias[0];

        // 4. Clamp
        result = fmaxf(-1.0f, fminf(result, 1.0f));

        output[index] = result;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor bias) {
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    // Output shape is (N, 1, D, H, W)
    std::vector<int64_t> output_sizes = {N, 1, D, H, W};
    auto output = torch::empty(output_sizes, output_options);

    int total_elements = N * D * H * W;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W
    );

    return output;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the inline CUDA code
fused_op_module = load_inline(
    name="fused_op_module",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution followed by a fused 
    LogSumExp, HardSwish, subtraction, and clamp operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.fused_op = fused_op_module

    def forward(self, x):
        x = self.conv_transpose(x)
        # The fused kernel handles LogSumExp, HardSwish, Sub, Clamp
        x = self.fused_op.fused_op_cuda(x, self.bias)
        return x