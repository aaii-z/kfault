import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom softmax CUDA kernel optimized for dim=1 (channel dimension)
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

__global__ void softmax_kernel(const float* input, float* output, int64_t B, int64_t C, int64_t D, int64_t H, int64_t W) {
    int total_spatial = D * H * W;
    int idx = blockIdx.x;
    if (idx >= B * total_spatial) return;

    int b = idx / total_spatial;
    int spatial_idx = idx % total_spatial;
    int d = spatial_idx / (H * W);
    int hw = spatial_idx % (H * W);
    int h = hw / W;
    int w = hw % W;

    extern __shared__ float shared_data[];
    int c = threadIdx.x;

    if (c < C) {
        int64_t input_offset = b * C * D * H * W + 
                               c * D * H * W + 
                               d * H * W + 
                               h * W + 
                               w;
        shared_data[c] = input[input_offset];
    }
    __syncthreads();

    float max_val = -INFINITY;
    for (int i = 0; i < C; i++) {
        if (shared_data[i] > max_val) max_val = shared_data[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < C; i++) {
        float exp_val = expf(shared_data[i] - max_val);
        sum_exp += exp_val;
    }

    float inv_sum_exp = 1.0f / (sum_exp + 1e-10f);
    if (c < C) {
        int64_t output_offset = b * C * D * H * W + 
                                c * D * H * W + 
                                d * H * W + 
                                h * W + 
                                w;
        output[output_offset] = expf(shared_data[c] - max_val) * inv_sum_exp;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 5, "Input must be 5-dimensional (B, C, D, H, W)");
    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t D = input.size(2);
    int64_t H = input.size(3);
    int64_t W = input.size(4);

    auto output = torch::empty_like(input);

    // Ensure input is contiguous
    input = input.contiguous();

    const int total_spatial = D * H * W;
    const int grid_size = B * total_spatial;
    const int block_size = C;

    // Launch kernel with dynamic shared memory
    softmax_kernel<<<grid_size, block_size, C * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W
    );

    return output;
}
"""

softmax_cpp = "torch::Tensor softmax_cuda(torch::Tensor input);"

# Compile the custom softmax kernel
softmax_op = load_inline(
    name="softmax_cuda",
    cpp_sources=softmax_cpp,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses two max pooling operations and uses a custom softmax CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # Combine two MaxPool3d operations into a single pooling with equivalent kernel size
        combined_kernel = pool_kernel_size * pool_kernel_size
        self.pool = nn.MaxPool3d(kernel_size=combined_kernel, stride=combined_kernel)
        self.softmax_op = softmax_op

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax_op.softmax_cuda(x)
        x = self.pool(x)
        return x