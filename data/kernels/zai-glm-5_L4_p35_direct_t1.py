import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operator: Sub -> HardSwish -> MaxPool -> Mish
# We fuse the post-convolution operations into a single kernel to minimize memory bandwidth usage.
# Conv2d is kept as a PyTorch primitive for high performance and correctness.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>
#include <cmath>

// Helper function for HardSwish
__device__ __forceinline__ float hardswish_func(float x) {
    // HardSwish definition: x * relu6(x + 3) / 6
    // relu6(x+3) is min(max(x+3, 0), 6)
    if (x < -3.0f) return 0.0f;
    if (x > 3.0f) return x;
    return x * (x + 3.0f) / 6.0f;
}

// Helper function for Mish
// Mish definition: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
__device__ __forceinline__ float mish_func(float x) {
    // Using tanhf and expf for single precision
    // softplus(x) = log(1 + exp(x))
    // For numerical stability with large x, ln(1+exp(x)) approx x
    // However, tanh(x) saturates to 1, so Mish approx x for large x.
    // We can just compute directly.
    return x * tanhf(log1pf(expf(x)));
}

// Kernel for fused Sub, HardSwish, MaxPool, and Mish
__global__ void fused_sub_hardswish_maxpool_mish_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int pool_kernel_size,
    const float subtract_value) {

    // Calculate output coordinates
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels; // Channels packed into z-dimension
    int n = blockIdx.z / channels; // Batch index

    if (n < batch_size && c < channels && oh < out_height && ow < out_width) {
        float max_val = -FLT_MAX;

        // Iterate over the pooling window
        for (int ph = 0; ph < pool_kernel_size; ++ph) {
            for (int pw = 0; pw < pool_kernel_size; ++pw) {
                int ih = oh * pool_kernel_size + ph;
                int iw = ow * pool_kernel_size + pw;

                // Input index (NCHW format)
                int input_idx = n * (channels * in_height * in_width) +
                                c * (in_height * in_width) +
                                ih * in_width +
                                iw;

                float val = input[input_idx];

                // Apply Subtract
                val -= subtract_value;

                // Apply HardSwish
                val = hardswish_func(val);

                // Update max
                if (val > max_val) {
                    max_val = val;
                }
            }
        }

        // Apply Mish to the result of MaxPool
        float result = mish_func(max_val);

        // Output index (NCHW format)
        int output_idx = n * (channels * out_height * out_width) +
                         c * (out_height * out_width) +
                         oh * out_width +
                         ow;

        output[output_idx] = result;
    }
}

// C++ interface
torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    float subtract_value,
    int pool_kernel_size) {

    // Check input dimensions
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);

    auto out_height = in_height / pool_kernel_size;
    auto out_width = in_width / pool_kernel_size;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());

    // Launch configuration
    // Using 2D thread blocks for better mapping to the 2D output spatial dimensions
    dim3 block(16, 16); 
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch_size * channels
    );

    fused_sub_hardswish_maxpool_mish_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        pool_kernel_size,
        subtract_value
    );

    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, float subtract_value, int pool_kernel_size);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution followed by a fused CUDA kernel
    for Sub, HardSwish, MaxPool, and Mish.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_ops_cuda(x, self.subtract_value, self.pool_kernel_size)
        return x