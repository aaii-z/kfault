import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Min and Softmax
fused_min_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void fused_min_softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int N, int C, int D, int H, int W) {
    int w = threadIdx.x;
    int c = threadIdx.y;
    int h = blockIdx.y;
    int n = blockIdx.x;

    extern __shared__ float s_data[];

    // Bounds check
    if (n < N && h < H && w < W && c < C) {
        // 1. Min reduction along D (dim=2)
        float min_val = std::numeric_limits<float>::infinity();
        
        // Pointer arithmetic for coalesced reads
        // Base pointer for (n, c, 0, h, w)
        const float* ptr = input + (n * C * D * H * W + c * D * H * W + h * W + w);
        int stride_d = H * W;

        for (int d = 0; d < D; ++d) {
            float val = *ptr;
            min_val = fminf(min_val, val);
            ptr += stride_d;
        }

        // Store min result in shared memory
        // Layout: [C, W] -> s_data[c * W + w]
        s_data[c * W + w] = min_val;
    }

    __syncthreads();

    // 2. Softmax along C (dim=1)
    // Only threads with c=0 perform the reduction and write output
    if (c == 0 && w < W && n < N && h < H) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < C; ++k) {
            float val = s_data[k * W + w];
            max_val = fmaxf(max_val, val);
        }

        float sum_exp = 0.0f;
        for (int k = 0; k < C; ++k) {
            float val = s_data[k * W + w];
            sum_exp += expf(val - max_val);
        }

        for (int k = 0; k < C; ++k) {
            float val = s_data[k * W + w];
            float res = expf(val - max_val) / sum_exp;
            
            // Output index: (N, C, H, W)
            int out_idx = n * C * H * W + k * H * W + h * W + w;
            output[out_idx] = res;
        }
    }
}

torch::Tensor fused_min_softmax_cuda(torch::Tensor input, int dim_min) {
    // This kernel is optimized for dim_min == 2 (Depth dimension)
    // TORCH_CHECK(dim_min == 2, "Custom kernel only supports min reduction along dim=2");
    
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::zeros({N, C, H, W}, input.options());

    // Block size: (W, C)
    // Assumes W and C are small enough to fit in a block (e.g., W=30, C=24)
    dim3 block(W, C);
    dim3 grid(N, H);

    size_t shared_mem_size = W * C * sizeof(float);

    fused_min_softmax_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N, C, D, H, W
    );

    return output;
}
"""

fused_min_softmax_cpp_source = (
    "torch::Tensor fused_min_softmax_cuda(torch::Tensor input, int dim_min);"
)

# Compile the inline CUDA code
fused_min_softmax = load_inline(
    name="fused_min_softmax",
    cpp_sources=fused_min_softmax_cpp_source,
    cuda_sources=fused_min_softmax_source,
    functions=["fused_min_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax using a custom fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.fused_op = fused_min_softmax

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        x = self.conv(x)
        # Use the custom fused kernel for min and softmax
        # The kernel assumes min is along dim=2 and softmax is along dim=1
        if self.dim == 2:
            x = self.fused_op.fused_min_softmax_cuda(x, self.dim)
        else:
            # Fallback for other dimensions (should not happen based on problem description)
            x = torch.min(x, dim=self.dim)[0]
            x = torch.softmax(x, dim=1)
        return x