import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, Swish, and Scaling
fused_gemm_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Define block size for the kernel
#define BLOCK_SIZE 256

__global__ void fused_gemm_swish_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int K, int N, float scaling_factor) {
    
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;

    if (idx < total_elements) {
        // Determine row (m) and column (n) indices
        int m = idx / N;
        int n = idx % N;

        // Compute dot product for GEMM: output[m, n] = input[m, :] dot weight[n, :] + bias[n]
        float sum = 0.0f;
        
        // Pointers to the start of the relevant row in input and row in weight
        const float* input_row = input + m * K;
        const float* weight_row = weight + n * K; // Weight is (out_features, in_features)

        // Unroll loop for better instruction level parallelism
        for (int k = 0; k < K; ++k) {
            sum += input_row[k] * weight_row[k];
        }

        // Add bias if present
        if (bias != nullptr) {
            sum += bias[n];
        }

        // Apply Swish activation: x * sigmoid(x)
        float sig = 1.0f / (1.0f + expf(-sum));
        float swish = sum * sig;

        // Apply scaling
        output[idx] = swish * scaling_factor;
    }
}

torch::Tensor fused_gemm_swish_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor) {
    
    // Get dimensions
    int64_t M = input.size(0);
    int64_t K = input.size(1);
    int64_t N = weight.size(0); // Weight shape is (out_features, in_features)

    // Allocate output tensor
    auto output = torch::empty({M, N}, input.options());

    // Calculate grid and block dimensions
    int total_elements = M * N;
    const int block_size = BLOCK_SIZE;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Launch kernel
    fused_gemm_swish_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        M, K, N, scaling_factor
    );

    // Check for launch errors
    C10_CUDA_CHECK(cudaGetLastError());

    return output;
}
"""

fused_gemm_swish_scale_cpp_source = (
    "torch::Tensor fused_gemm_swish_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code
fused_gemm_swish_scale_op = load_inline(
    name="fused_gemm_swish_scale",
    cpp_sources=fused_gemm_swish_scale_cpp_source,
    cuda_sources=fused_gemm_swish_scale_source,
    functions=["fused_gemm_swish_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for fused GEMM + Swish + Scaling.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # We still define weights as Parameters so they are trained/loaded correctly
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scaling_factor = scaling_factor
        
        # Initialize weights (mimicking nn.Linear default initialization)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # Load the custom op
        self.fused_op = fused_gemm_swish_scale_op

    def forward(self, x):
        # Call the custom fused kernel
        return self.fused_op.fused_gemm_swish_scale_cuda(x, self.weight, self.bias, self.scaling_factor)