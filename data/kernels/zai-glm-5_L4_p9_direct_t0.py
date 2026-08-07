import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear, subtract, multiply, and ReLU
fused_linear_sub_mul_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Fused kernel: (Linear + Subtract + Multiply + ReLU)
// Computes: y = ReLU((x * W^T + b) - sub_val) * mul_val
// We can rewrite this as: y = ReLU(x * W^T + (b - sub_val)) * mul_val
// To maximize throughput, we perform the ReLU and scaling in-place on the output tile in registers.

// Block size 128 provides a good balance for 8192 dimensions
#define BLOCK_SIZE 128

__global__ void fused_linear_sub_mul_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float subtract_value,
    float multiply_value,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) {

    // Compute the adjusted bias once
    const float adjusted_bias = bias[0] - subtract_value;

    // Shared memory for input tile (batch_size x BLOCK_SIZE)
    // We load a block of input rows to shared memory for coalesced access
    extern __shared__ float input_tile[];

    int batch_idx = blockIdx.x; // One block per row in the batch
    int tid = threadIdx.x;

    // Pointers to the current row
    const float* row_input = input + batch_idx * in_features;
    float* row_output = output + batch_idx * out_features;

    // Load input row segment into shared memory
    for (int i = tid; i < in_features; i += blockDim.x) {
        input_tile[i] = row_input[i];
    }
    __syncthreads();

    // Each thread computes multiple output elements
    // Grid-stride loop over output features
    for (int out_idx = tid; out_idx < out_features; out_idx += blockDim.x) {
        float sum = 0.0f;

        // Dot product: input row . weight column
        // Weight is stored as (out_features, in_features)
        const float* weight_row = weight + out_idx * in_features;

        // Manual unrolling for better ILP
        int i = 0;
        for (; i <= in_features - 4; i += 4) {
            sum += input_tile[i] * weight_row[i];
            sum += input_tile[i+1] * weight_row[i+1];
            sum += input_tile[i+2] * weight_row[i+2];
            sum += input_tile[i+3] * weight_row[i+3];
        }
        for (; i < in_features; ++i) {
            sum += input_tile[i] * weight_row[i];
        }

        // Add adjusted bias
        sum += adjusted_bias;

        // ReLU
        sum = fmaxf(0.0f, sum);

        // Multiply
        sum *= multiply_value;

        // Write result
        row_output[out_idx] = sum;
    }
}

torch::Tensor fused_linear_sub_mul_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value) {

    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    // Launch configuration
    int block_size = BLOCK_SIZE;
    int num_blocks = batch_size; // One block per batch item
    size_t shared_mem_size = in_features * sizeof(float);

    fused_linear_sub_mul_relu_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        subtract_value,
        multiply_value,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_linear_sub_mul_relu_cpp_source = (
    "torch::Tensor fused_linear_sub_mul_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float subtract_value, float multiply_value);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_linear_sub_mul_relu",
    cpp_sources=fused_linear_sub_mul_relu_cpp_source,
    cuda_sources=fused_linear_sub_mul_relu_source,
    functions=["fused_linear_sub_mul_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a fused matrix multiplication, subtraction, multiplication, and ReLU activation.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        # We manually handle weights to pass to CUDA kernel
        # We register them as parameters so they are recognized by .parameters(), .cuda(), etc.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters (mimicking nn.Linear initialization)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_linear_sub_mul_relu_cuda(
            x, self.weight, self.bias, self.subtract_value, self.multiply_value
        )