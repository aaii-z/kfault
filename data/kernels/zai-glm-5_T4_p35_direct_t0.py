import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operator: Conv2d -> Sub -> HardSwish -> MaxPool -> Mish
fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>

// Define constants for HardSwish and Mish
#define HARD_SWISH_THRESHOLD 3.0f
#define MISH_THRESHOLD 20.0f

// Helper function for HardSwish
__device__ __forceinline__ float hardswish_func(float x) {
    if (x <= -HARD_SWISH_THRESHOLD) return 0.0f;
    if (x >= HARD_SWISH_THRESHOLD) return x;
    return x * (x + 3.0f) / 6.0f;
}

// Helper function for Mish
__device__ __forceinline__ float mish_func(float x) {
    float sp = 1.0f / (1.0f + expf(-x));
    return x * tanhf(sp * x);
}

// Kernel for fused Sub, HardSwish, MaxPool, and Mish
// Assumes input tensor is contiguous and dimensions are aligned as per the problem
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

    // Calculate output dimensions
    int n = blockIdx.z; // Batch
    int c = blockIdx.y; // Channel
    int oh = blockIdx.x / out_width; // Output Height
    int ow = blockIdx.x % out_width; // Output Width

    if (n < batch_size && c < channels && oh < out_height && ow < out_width) {
        // MaxPool window
        float max_val = -FLT_MAX;
        
        // Iterate over the pooling window
        for (int ph = 0; ph < pool_kernel_size; ++ph) {
            for (int pw = 0; pw < pool_kernel_size; ++pw) {
                int ih = oh * pool_kernel_size + ph;
                int iw = ow * pool_kernel_size + pw;
                
                // Input index
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
        
        // Output index
        int output_idx = n * (channels * out_height * out_width) +
                         c * (out_height * out_width) +
                         oh * out_width +
                         ow;
        
        output[output_idx] = result;
    }
}

// Helper function to get output size
std::vector<int64_t> get_output_size(std::vector<int64_t> input_size, int kernel_size) {
    std::vector<int64_t> output_size(4);
    output_size[0] = input_size[0];
    output_size[1] = input_size[1];
    output_size[2] = input_size[2] / kernel_size;
    output_size[3] = input_size[3] / kernel_size;
    return output_size;
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

    // Launch kernel
    const int block_size_x = 16; // Tunable, but 16 is a reasonable default for 2D-like ops
    const int blocks_x = (out_height * out_width + block_size_x - 1) / block_size_x;
    
    dim3 grid(blocks_x, channels, batch_size);
    dim3 block(block_size_x);

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