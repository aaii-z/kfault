import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul, scaling, and residual addition
fused_matmul_scale_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Define block size for kernel launch
#define BLOCK_SIZE 32

// Kernel for fused matrix multiplication, scaling, and residual addition
// Computes: output = (input * weight^T + bias) * scaling_factor + (input * weight^T + bias)
// Which simplifies to: output = (input * weight^T + bias) * (1 + scaling_factor)
// However, to strictly follow the original model logic which clones the intermediate result:
// intermediate = input * weight^T + bias
// output = intermediate * scaling_factor + intermediate
// We calculate intermediate first, then apply the scaling and residual.

// Note: For large matrices, cuBLAS is generally more efficient than a manual kernel.
// However, to demonstrate a custom CUDA implementation as requested:

__global__ void fused_matmul_scale_residual_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int batch_size, 
    int in_features, 
    int out_features, 
    float scaling_factor) {
    
    // Compute the coordinates of the element to process
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output feature index

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        
        // Compute dot product for matrix multiplication
        // input: (batch_size, in_features)
        // weight: (out_features, in_features) - stored transposed in Linear layer logic usually, 
        // but PyTorch Linear stores (out_features, in_features) and does input * weight^T.
        // Here we assume weight is (out_features, in_features) and we access it as weight[col, k].
        
        // Using shared memory for input tile optimization could be added here for more complex scenarios,
        // but for a concise custom kernel demonstration:
        
        for (int k = 0; k < in_features; ++k) {
            // input[row, k] * weight[col, k]
            // Weight is stored row-major: weight[col * in_features + k]
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        // Add bias if it exists
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Apply scaling and residual addition
        // x = x * scaling_factor + original_x
        // output[row, col] = sum * scaling_factor + sum
        output[row * out_features + col] = sum * scaling_factor + sum;
    }
}

torch::Tensor fused_matmul_scale_residual_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    c10::optional<torch::Tensor> bias_opt, 
    float scaling_factor) {
    
    // Check inputs
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    // Handle optional bias
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        bias_ptr = bias_opt.value().data_ptr<float>();
    }
    
    // Grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    fused_matmul_scale_residual_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scaling_factor
    );
    
    return output;
}
"""

fused_matmul_scale_residual_cpp_source = (
    "torch::Tensor fused_matmul_scale_residual_cuda(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt, float scaling_factor);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_matmul_scale_residual",
    cpp_sources=fused_matmul_scale_residual_cpp_source,
    cuda_sources=fused_matmul_scale_residual_source,
    functions=["fused_matmul_scale_residual_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for fused matmul, scaling, and residual addition.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # We keep the Linear layer definition to hold the weights and bias parameters
        # so that the weight initialization matches the original model.
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        """
        Forward pass using the custom CUDA kernel.
        """
        # Call the custom fused kernel
        # We pass the weight and bias from the Linear layer directly to the kernel
        return self.fused_op.fused_matmul_scale_residual_cuda(x, self.matmul.weight, self.matmul.bias, self.scaling_factor)