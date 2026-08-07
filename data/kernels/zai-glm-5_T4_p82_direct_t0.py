import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused tanh, scaling, bias addition, and max pooling
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel for tanh, scaling, bias addition, and max pooling
__global__ void fused_tanh_scale_bias_maxpool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int pool_size,
    const float scaling_factor) {

    // Calculate output pixel coordinates
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % in_channels; // Output channel same as input channel for this op
    int n = blockIdx.z / in_channels;

    if (n >= batch_size || oc >= in_channels || oh >= out_height || ow >= out_width) return;

    // Calculate the top-left corner of the pooling window in the input
    int ih_start = oh * pool_size;
    int iw_start = ow * pool_size;

    float max_val = -INFINITY;

    // Iterate over the pooling window
    for (int ph = 0; ph < pool_size; ++ph) {
        int ih = ih_start + ph;
        // Boundary check for height (though grid sizing should prevent out-of-bounds if calculated correctly)
        if (ih >= in_height) continue;

        for (int pw = 0; pw < pool_size; ++pw) {
            int iw = iw_start + pw;
            if (iw >= in_width) continue;

            // Calculate input index (N, C, H, W)
            long input_idx = (long)n * in_channels * in_height * in_width +
                             (long)oc * in_height * in_width +
                             (long)ih * in_width + iw;

            // Load value
            float val = input[input_idx];

            // Apply operations: tanh, scale, bias
            // tanhf is standard, approximations like __tanhf might be faster but less precise
            val = tanhf(val);
            val = val * scaling_factor;
            val = val + bias[oc]; // Bias is (C, 1, 1)

            // Max pooling reduction
            if (val > max_val) {
                max_val = val;
            }
        }
    }

    // Store result
    if (max_val != -INFINITY) {
        long output_idx = (long)n * in_channels * out_height * out_width +
                          (long)oc * out_height * out_width +
                          (long)oh * out_width + ow;
        output[output_idx] = max_val;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    int pool_size,
    float scaling_factor) {

    // Input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    // Output dimensions
    int out_height = in_height / pool_size;
    int out_width = in_width / pool_size;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, in_channels, out_height, out_width}, input.options());

    // Grid and block dimensions
    // Block: 16x16 threads for spatial dimensions
    dim3 blockDim(16, 16);
    // Grid: covers output width, height, and (batch * channel)
    dim3 gridDim(
        (out_width + blockDim.x - 1) / blockDim.x,
        (out_height + blockDim.y - 1) / blockDim.y,
        batch_size * in_channels
    );

    // Launch kernel
    fused_tanh_scale_bias_maxpool_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        pool_size,
        scaling_factor
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, int pool_size, float scaling_factor);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        # Keep the convolution as is, since it's a standard optimized library call
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        # Bias is created as a parameter, matching the original logic
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        # Load the custom op
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Convolution (using PyTorch's optimized implementation)
        x = self.conv(x)

        # 2. Fused Tanh + Scale + Bias + MaxPool
        # We pass the bias flattened or as is, the kernel handles broadcasting logic
        x = self.fused_ops.fused_ops_cuda(x, self.bias, self.pool_kernel_size, self.scaling_factor)

        return x