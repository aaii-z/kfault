import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the entire forward pass
fused_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

// Kernel for the fused operation
__global__ void fused_forward_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size, 
    const int input_size, 
    const int hidden_size, 
    const float scale_factor, 
    const float clamp_min, 
    const float clamp_max) {
    
    // Each block handles one row of the output (one sample in the batch)
    int row_idx = blockIdx.x;
    if (row_idx >= batch_size) return;

    // Dynamic shared memory for the intermediate row results
    extern __shared__ float shared_row_data[];

    // Pointers to the current row
    const float* x_row = input + row_idx * input_size;

    // Thread cooperation for matrix-vector multiplication
    // Each thread computes multiple elements of the output row
    for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        float sum = 0.0f;
        // Weight is stored as (hidden_size, input_size) in PyTorch Linear
        // We compute dot product of input row and weight[col, :]
        const float* w_row = weight + col * input_size;
        
        // Unroll loop for potential optimization (compiler usually handles this)
        for (int k = 0; k < input_size; ++k) {
            sum += x_row[k] * w_row[k];
        }
        
        // Add bias
        sum += bias[col];
        
        // Scale
        sum *= scale_factor;
        
        // Residual connection (x + x)
        sum += sum;
        
        // Clamp
        sum = fmaxf(clamp_min, fminf(clamp_max, sum));
        
        shared_row_data[col] = sum;
    }

    // Synchronize threads to ensure shared_row_data is fully populated
    __syncthreads();

    // Parallel reduction for LogSumExp
    // 1. Find max value in the row
    __shared__ float shared_max; 
    float local_max = -std::numeric_limits<float>::infinity();
    
    for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        local_max = fmaxf(local_max, shared_row_data[col]);
    }
    
    // Reduce local maxes within the block
    // Using a simple approach: atomicMax or reduction in shared mem
    // Since we have multiple threads, we need to combine their results.
    // Let's use shared memory reduction for max
    __shared__ float s_max_vec[256]; // Assuming blockDim.x <= 256
    s_max_vec[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_max_vec[threadIdx.x] = fmaxf(s_max_vec[threadIdx.x], s_max_vec[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    float row_max = s_max_vec[0];
    
    // 2. Compute sum of exp(x - max)
    __shared__ float s_sum_vec[256];
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        local_sum += expf(shared_row_data[col] - row_max);
    }
    s_sum_vec[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum_vec[threadIdx.x] += s_sum_vec[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float row_sum = s_sum_vec[0];
    float logsumexp_val = logf(row_sum) + row_max;

    // Mish activation and final multiplication
    // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    // Result = x * Mish(x)
    if (threadIdx.x == 0) {
        float val = logsumexp_val;
        // softplus(val) = ln(1 + exp(val))
        // log1pf(exp(val)) is stable for large val
        float sp = log1pf(expf(val)); 
        float th = tanhf(sp);
        float mish_val = val * th;
        float result = val * mish_val; // x * mish(x)
        
        output[row_idx] = result;
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int64_t input_size, 
    int64_t hidden_size, 
    double scale_factor, 
    double clamp_min, 
    double clamp_max) {
    
    int64_t batch_size = input.size(0);
    
    // Output tensor shape is (Batch, 1) because of the LogSumExp reduction
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({batch_size, 1}, options);
    
    const int block_size = 256; 
    const int num_blocks = batch_size;
    
    // Calculate shared memory size for the intermediate row data
    size_t shared_mem_size = hidden_size * sizeof(float);
    
    // Launch kernel
    fused_forward_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        static_cast<float>(scale_factor),
        static_cast<float>(clamp_min),
        static_cast<float>(clamp_max)
    );
    
    return output;
}
"""

fused_forward_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int64_t input_size, 
    int64_t hidden_size, 
    double scale_factor, 
    double clamp_min, 
    double clamp_max);
"""

# Compile the inline CUDA code
fused_forward_op = load_inline(
    name="fused_forward",
    cpp_sources=fused_forward_cpp_source,
    cuda_sources=fused_forward_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # We keep the Linear layer to hold the weights and bias, 
        # but we will bypass its forward method.
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fused_forward = fused_forward_op

    def forward(self, x):
        # Call the custom CUDA kernel
        # We pass the weights and bias from the Linear layer directly
        return self.fused_forward.fused_forward_cuda(
            x, 
            self.matmul.weight, 
            self.matmul.bias, 
            self.input_size, 
            self.hidden_size, 
            self.scale_factor, 
            self.clamp_min, 
            self.clamp_max
        )