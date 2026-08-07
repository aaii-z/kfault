import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused matmul + maxpool1d + sum + scale
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_matmul_maxpool_sum_scale_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    float sum_pooled = 0.0f;

    // Process each pooling window
    for (int start = 0; start < out_features; start += kernel_size) {
        int end = min(start + kernel_size, out_features);
        float max_val = -FLT_MAX;

        // Compute linear transformation (matmul + bias) for each output feature in the window
        for (int j = start; j < end; j++) {
            float val = bias[j];
            for (int i = 0; i < in_features; i++) {
                val += input[batch_idx * in_features + i] * weight[j * in_features + i];
            }
            max_val = fmaxf(max_val, val);
        }

        sum_pooled += max_val;
    }

    output[batch_idx] = sum_pooled * scale_factor;
}

torch::Tensor fused_op_launcher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    float scale_factor
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size}, input.options());

    const int block_size = 1;
    const int grid_size = batch_size;

    fused_matmul_maxpool_sum_scale_kernel<<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        kernel_size,
        scale_factor
    );

    return output;
}
"""

cpp_source = """
torch::Tensor fused_op_launcher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    float scale_factor
);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_matmul_maxpool_sum_scale",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_op_launcher"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for fused matmul, maxpool, sum, and scaling.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return fused_op.fused_op_launcher(
            x, self.weight, self.bias, self.kernel_size, self.scale_factor
        )