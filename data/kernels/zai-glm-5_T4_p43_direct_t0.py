import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused logsumexp and ReLU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// Kernel for fused logsumexp(dim=1) and ReLU
// Input: (N, C, D, H, W)
// Output: (N, 1, D, H, W)
// Algorithm: Online LogSumExp to minimize numerical error and register pressure
__global__ void fused_logsumexp_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W) {

    // Each thread handles one spatial location (n, d, h, w)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = N * D * H * W;
    
    if (idx < total_spatial) {
        // Decode linear index to (n, d, h, w)
        int w = idx % W;
        int temp = idx / W;
        int h = temp % H;
        temp /= H;
        int d = temp % D;
        int n = temp / D;

        // Calculate base offset for the input pointer
        // Input layout: (N, C, D, H, W)
        // Offset = n * (C*D*H*W) + c * (D*H*W) + d * (H*W) + h * W + w
        // We iterate over c, so the stride for c is D*H*W
        size_t base_offset = (size_t)n * C * D * H * W + (size_t)d * H * W + h * W + w;
        size_t stride_c = (size_t)D * H * W;

        // Online LogSumExp algorithm
        // max_val tracks the max(x_i) seen so far
        // sum_exp tracks the sum(exp(x_i - max_val)) seen so far
        float max_val = -INFINITY;
        float sum_exp = 0.0f;

        // Use float2 for vectorized loads if possible (coalesced access)
        // Assuming C is large enough, we can process in chunks.
        // However, to keep it simple and robust for any C, we iterate scalar.
        // Optimization: Loop unrolling or vectorization could be added if C is known to be large.
        
        const float* ptr = input + base_offset;
        
        // First pass: Find max
        // It's better to do one pass for max and one for sum to avoid high register pressure
        // or complex control flow inside a single loop, but online algorithm does it in one.
        // Let's use the online algorithm:
        // new_max = max(old_max, x)
        // if old_max == new_max: sum_exp += exp(x - new_max)
        // else: sum_exp = sum_exp * exp(old_max - new_max) + exp(x - new_max)
        
        // To speed up, we can vectorize loads.
        // Let's stick to scalar for correctness and simplicity, 
        // assuming compiler might unroll or optimize.
        
        for (int c = 0; c < C; ++c) {
            float val = ptr[c * stride_c];
            
            if (val > max_val) {
                // Rescale sum_exp
                // If max_val was -infinity, sum_exp is 0, so exp(-inf - val) is 0.
                if (max_val > -INFINITY) {
                    float scale = expf(max_val - val);
                    sum_exp *= scale;
                }
                max_val = val;
                // Add the current element's contribution (which is exp(0) = 1)
                // But wait, if we just updated max_val to val, then exp(val - max_val) is exp(0) = 1.
                // However, the online algorithm usually adds exp(x - new_max).
                // Since x == new_max here, we add 1.
                // But we need to be careful: did we already count 'val' in sum_exp? No.
                // So we add exp(val - max_val) = 1.
                sum_exp += 1.0f;
            } else {
                sum_exp += expf(val - max_val);
            }
        }

        float lse = logf(sum_exp) + max_val;
        
        // Fused ReLU
        // output layout: (N, 1, D, H, W)
        size_t out_offset = (size_t)n * D * H * W + (size_t)d * H * W + h * W + w;
        output[out_offset] = fmaxf(lse, 0.0f);
    }
}

torch::Tensor fused_logsumexp_relu_cuda(torch::Tensor input) {
    // Input shape: (N, C, D, H, W)
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];

    // Output shape: (N, 1, D, H, W)
    auto output = torch::zeros({N, 1, D, H, W}, input.options());

    int total_spatial = N * D * H * W;
    const int block_size = 256;
    const int num_blocks = (total_spatial + block_size - 1) / block_size;

    fused_logsumexp_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_logsumexp_relu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_logsumexp_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, max pooling, and a fused log sum exp + ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width')
        """
        x = self.conv(x)
        x = self.max_pool(x)
        # Use the custom fused kernel for logsumexp and relu
        x = self.fused_ops.fused_logsumexp_relu_cuda(x)
        return x