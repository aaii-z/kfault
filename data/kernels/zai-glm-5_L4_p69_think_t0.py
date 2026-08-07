import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused HardSwish and ReLU
# Mathematical derivation:
# HardSwish(x) = x * ReLU6(x + 3) / 6
# ReLU6(x + 3) is:
#   0 if x < -3
#   x + 3 if -3 <= x <= 3
#   6 if x > 3
# So HardSwish(x) is:
#   0 if x < -3
#   x * (x + 3) / 6 if -3 <= x <= 3
#   x if x > 3
#
# Applying ReLU to HardSwish(x):
# If x < -3: HardSwish is 0. ReLU(0) = 0.
# If -3 <= x < 0: HardSwish is x(x+3)/6. Since x is neg and x+3 is pos, result is neg. ReLU(neg) = 0.
# If x >= 0: HardSwish is >= 0. ReLU(HardSwish) = HardSwish.
#
# Conclusion: The fused operation is equivalent to HardSwish(ReLU(x)).
# We can implement this efficiently by checking if x < 0.

fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish_relu_op(float x) {
    if (x < 0.0f) {
        return 0.0f;
    }
    // x >= 0
    if (x < 3.0f) {
        return x * (x + 3.0f) / 6.0f;
    }
    // x >= 3
    return x;
}

__global__ void fused_hardswish_relu_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time using vectorized loads/stores for better memory bandwidth
    // We assume the tensor is contiguous, which it is coming out of Conv2d
    int vectorized_size = size / 4;
    int vectorized_stride = stride;
    
    // Vectorized processing
    for (int i = tid; i < vectorized_size; i += vectorized_stride) {
        float4 in_val = reinterpret_cast<const float4*>(input)[i];
        float4 out_val;
        
        out_val.x = hardswish_relu_op(in_val.x);
        out_val.y = hardswish_relu_op(in_val.y);
        out_val.z = hardswish_relu_op(in_val.z);
        out_val.w = hardswish_relu_op(in_val.w);
        
        reinterpret_cast<float4*>(output)[i] = out_val;
    }
    
    // Handle remaining elements
    int remainder_start = vectorized_size * 4;
    for (int i = remainder_start + tid; i < size; i += stride) {
        output[i] = hardswish_relu_op(input[i]);
    }
}

torch::Tensor fused_hardswish_relu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size * 4 - 1) / (block_size * 4); // Adjusted for vectorization

    fused_hardswish_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );
    
    return output;
}
"""

fused_kernel_cpp_source = "torch::Tensor fused_hardswish_relu_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_hardswish_relu",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_hardswish_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, followed by a fused HardSwish and ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        x = self.fused_op.fused_hardswish_relu_cuda(x)
        return x