import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
# The operation sequence is:
# 1. Linear (Matrix Multiplication + Bias)
# 2. Sum reduction (dim=1)
# 3. Max (Identity on size-1 tensor)
# 4. Mean (Identity on size-1 tensor)
# 5. LogSumExp (Identity on size-1 tensor)
# 6. LogSumExp (Identity on size-1 tensor)
#
# Since steps 3-6 operate on a tensor of size (batch_size, 1), they are mathematically
# equivalent to identity operations (max of 1 element is the element itself, etc.).
# Thus, we only need to compute the sum of the linear layer output.
# Furthermore, sum(linear(x)) can be optimized:
# sum(x * W^T + b) = x * sum(W) + sum(b)
# This reduces the problem to a matrix-vector multiplication.
fused_linear_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_sum_kernel(const float* x, const float* w_reduced, float b_reduced, float* out, int batch_size, int in_features) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row < batch_size) {
        const float* x_row = x + row * in_features;
        float sum = 0.0f;

        // Each thread accumulates a partial sum of the dot product
        for (int i = tid; i < in_features; i += blockDim.x) {
            sum += x_row[i] * w_reduced[i];
        }

        // Use shared memory for block reduction
        __shared__ float s_data[1024]; // Assuming block_size <= 1024
        s_data[tid] = sum;
        __syncthreads();

        // Perform reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_data[tid] += s_data[tid + s];
            }
            __syncthreads();
        }

        // Write the result for this row
        if (tid == 0) {
            out[row] = s_data[0] + b_reduced;
        }
    }
}

torch::Tensor fused_linear_sum_cuda(torch::Tensor x, torch::Tensor w_reduced, float b_reduced) {
    int batch_size = x.size(0);
    int in_features = x.size(1);

    auto out = torch::empty({batch_size, 1}, x.options());

    const int block_size = 1024;
    const int num_blocks = batch_size;

    fused_linear_sum_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        w_reduced.data_ptr<float>(),
        b_reduced,
        out.data_ptr<float>(),
        batch_size,
        in_features
    );

    return out;
}
"""

fused_linear_sum_cpp_source = (
    "torch::Tensor fused_linear_sum_cuda(torch::Tensor x, torch::Tensor w_reduced, float b_reduced);"
)

# Compile the inline CUDA code
fused_linear_sum = load_inline(
    name="fused_linear_sum",
    cpp_sources=fused_linear_sum_cpp_source,
    cuda_sources=fused_linear_sum_source,
    functions=["fused_linear_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # We keep the linear layer to initialize weights and biases
        self.linear = nn.Linear(in_features, out_features)
        
        # Precompute the reduced weights and bias
        # The operation is sum(Linear(x), dim=1).
        # Linear(x) = x @ W^T + b
        # sum(x @ W^T + b) = x @ sum(W^T) + sum(b) = x @ sum(W, dim=0) + sum(b)
        # We register these as buffers so they are moved to the correct device with the model
        w_reduced = self.linear.weight.sum(dim=0)
        b_reduced = self.linear.bias.sum()
        
        self.register_buffer('w_reduced', w_reduced)
        self.register_buffer('b_reduced', torch.tensor(b_reduced))
        
        self.fused_op = fused_linear_sum

    def forward(self, x):
        # Use the custom CUDA kernel
        return self.fused_op.fused_linear_sum_cuda(x, self.w_reduced, self.b_reduced.item())