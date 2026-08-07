import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear + Mish + Mish
fused_linear_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Tiled MatMul + Mish + Mish Kernel
// Block size: 16x16 threads. Each thread computes 1 output element.
// Shared memory tiles: 16x16.

__global__ void fused_linear_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) 
{
    // Shared memory tiles
    // Padding to avoid bank conflicts
    __shared__ float input_tile[16][16 + 1];
    __shared__ float weight_tile[16][16 + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Output coordinates
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;

    float sum = 0.0f;
    
    // Loop over tiles in the k-dimension (in_features)
    int num_tiles = (in_features + 15) / 16;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Load input tile
        // input is (batch_size, in_features)
        // We load input[row, t*16 + tx]
        int in_col = t * 16 + tx;
        if (row < batch_size && in_col < in_features) {
            input_tile[ty][tx] = input[row * in_features + in_col];
        } else {
            input_tile[ty][tx] = 0.0f;
        }

        // Load weight tile
        // weight is (out_features, in_features)
        // We need weight[col, k]. 
        // We load weight[col, t*16 + ty] into weight_tile[ty][tx].
        // Note: We use the column index 'col' which corresponds to the output feature index.
        int w_col = t * 16 + ty;
        if (col < out_features && w_col < in_features) {
            weight_tile[ty][tx] = weight[col * in_features + w_col];
        } else {
            weight_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        // sum += input[row, k] * weight[col, k]
        // input_tile[ty][k] holds input[row, t*16 + k]
        // weight_tile[k][tx] holds weight[col, t*16 + k]
        
        for (int k = 0; k < 16; ++k) {
            sum += input_tile[ty][k] * weight_tile[k][tx];
        }

        __syncthreads();
    }

    // Boundary check for output dimensions
    if (row >= batch_size || col >= out_features) return;

    // Add bias
    float val = sum + bias[col];

    // Apply Mish twice
    // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    
    // First Mish
    float sp = logf(1.0f + expf(val));
    float th = tanhf(sp);
    val = val * th;

    // Second Mish
    sp = logf(1.0f + expf(val));
    th = tanhf(sp);
    val = val * th;

    // Write result
    output[row * out_features + col] = val;
}

torch::Tensor fused_linear_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) 
{
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::zeros({batch_size, out_features}, input.options());

    dim3 block_size(16, 16);
    dim3 grid_size((out_features + 15) / 16, (batch_size + 15) / 16);

    fused_linear_mish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_linear_mish_cpp_source = (
    "torch::Tensor fused_linear_mish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_linear_mish_op = load_inline(
    name="fused_linear_mish",
    cpp_sources=fused_linear_mish_cpp_source,
    cuda_sources=fused_linear_mish_source,
    functions=["fused_linear_mish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom fused CUDA kernel for Linear + Mish + Mish.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # We keep the weight and bias as standard Parameters so that loading weights
        # from the original model works seamlessly.
        # We just won't use the standard nn.Linear forward pass.
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.fused_op = fused_linear_mish_op

    def forward(self, x):
        # Call the custom fused kernel
        return self.fused_op.fused_linear_mish_cuda(x, self.weight, self.bias)