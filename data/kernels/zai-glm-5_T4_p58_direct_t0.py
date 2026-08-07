import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LogSumExp, HardSwish, Subtraction, and Clamp
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// Kernel for LogSumExp reduction
// We perform reduction along dim=1 (channels). 
// Input shape: (N, C, D, H, W)
// Output shape: (N, 1, D, H, W)
__global__ void logsumexp_kernel(const float* __restrict__ x, float* __restrict__ out, int N, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = N * D * H * W;
    
    if (idx < total_spatial) {
        int n = idx / (D * H * W);
        int spatial_idx = idx % (D * H * W);
        int d = spatial_idx / (H * W);
        int hw = spatial_idx % (H * W);
        int h = hw / W;
        int w = hw % W;

        float max_val = -INFINITY;
        
        // Find max for numerical stability
        for (int c = 0; c < C; ++c) {
            int input_idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
            float val = x[input_idx];
            if (val > max_val) {
                max_val = val;
            }
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            int input_idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
            sum_exp += expf(x[input_idx] - max_val);
        }

        out[idx] = logf(sum_exp) + max_val;
    }
}

// Kernel for fused HardSwish, Subtraction, and Clamp
// Operates element-wise on the output of LogSumExp
__global__ void fused_post_ops_kernel(float* __restrict__ x, const float* __restrict__ bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        
        // HardSwish: x * sigmoid(x + 3) / 6
        // Sigmoid implementation: 1 / (1 + exp(-x))
        float sigmoid_arg = val + 3.0f;
        float sigmoid_val = 1.0f / (1.0f + expf(-sigmoid_arg));
        val = val * sigmoid_val / 6.0f;
        
        // Subtraction
        val = val - bias[0];
        
        // Clamp
        val = fmaxf(-1.0f, fminf(1.0f, val));
        
        x[idx] = val;
    }
}

// Helper function to launch kernels
torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias) {
    // x shape: (N, C, D, H, W)
    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    // Output for LogSumExp: (N, 1, D, H, W)
    auto out_options = x.options();
    std::vector<int64_t> out_sizes = {N, 1, D, H, W};
    auto out = torch::empty(out_sizes, out_options);

    int total_spatial = N * D * H * W;
    const int block_size = 256;
    const int num_blocks_logsumexp = (total_spatial + block_size - 1) / block_size;

    // Launch LogSumExp kernel
    logsumexp_kernel<<<num_blocks_logsumexp, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, C, D, H, W
    );

    // Launch fused post-processing kernel (HardSwish, Sub, Clamp)
    // Operates on 'out' in-place
    int total_elements_out = out.numel();
    const int num_blocks_fused = (total_elements_out + block_size - 1) / block_size;

    fused_post_ops_kernel<<<num_blocks_fused, block_size>>>(
        out.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        total_elements_out
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias);"
)

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
    Optimized Model that performs a 3D transposed convolution followed by a fused CUDA kernel
    for LogSumExp, HardSwish, subtraction, and clamp operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias)
        return x