import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for fused Linear + Sigmoid
// Computes: output = sigmoid(input * weight^T + bias)
__global__ void fused_linear_sigmoid_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int row = idx / out_features; // batch index
        int col = idx % out_features; // output feature index
        
        float sum = 0.0f;
        // Compute dot product for this output element
        // weight is (out_features, in_features), we access weight[col, k]
        // input is (batch_size, in_features), we access input[row, k]
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        sum += bias[col];
        
        // Sigmoid activation
        output[idx] = 1.0f / (1.0f + expf(-sum));
    }
}

// Kernel for fused Linear + LogSumExp
// Computes: output[i] = log(sum_j(exp(input[i, :] * weight^T[j, :] + bias[j])))
__global__ void fused_linear_logsumexp_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int in_features, int out_features) {
    int row = blockIdx.x; // One block per batch item
    int tid = threadIdx.x;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    
    // Strided loop to handle out_features > blockDim.x
    // Each thread computes partial max/sum for a subset of output features
    for (int k = tid; k < out_features; k += blockDim.x) {
        float val = 0.0f;
        // Compute matmul for this specific output feature k
        for (int i = 0; i < in_features; ++i) {
            val += input[row * in_features + i] * weight[k * in_features + i];
        }
        val += bias[k];
        
        // Online softmax/max reduction algorithm
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val); // Rescale previous sum
            local_max = val;
            local_sum += 1.0f; // Add the new element (exp(0))
        } else {
            local_sum += expf(val - local_max);
        }
    }
    
    // Load local results into shared memory
    sdata[tid] = local_max;
    __syncthreads();
    
    // Reduce max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float global_max = sdata[0];
    __syncthreads();
    
    // Rescale sums and store in shared memory
    // We need to scale the local_sum by exp(local_max - global_max)
    // If local_max was -infinity (empty), handle carefully, though unlikely here
    if (local_max > -INFINITY) {
         sdata[tid] = local_sum * expf(local_max - global_max);
    } else {
         sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    // Reduce sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[row] = logf(sdata[0]) + global_max;
    }
}

// C++ wrapper functions
torch::Tensor fused_linear_sigmoid_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_linear_sigmoid_kernel<<<num_blocks, block_size>>>(
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

torch::Tensor fused_linear_logsumexp_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size}, input.options());
    
    const int block_size = 256;
    // Shared memory size for float reduction
    size_t shared_mem_size = block_size * sizeof(float);
    
    fused_linear_logsumexp_kernel<<<batch_size, block_size, shared_mem_size>>>(
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

cpp_source = """
torch::Tensor fused_linear_sigmoid_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor fused_linear_logsumexp_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_linear_sigmoid_cuda", "fused_linear_logsumexp_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernels for fused operations.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        # Store weights as Parameters to mimic nn.Linear behavior
        # Note: nn.Linear stores weights as (out_features, in_features)
        self.linear1_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.linear1_bias = nn.Parameter(torch.randn(hidden_size))
        
        self.linear2_weight = nn.Parameter(torch.randn(output_size, hidden_size))
        self.linear2_bias = nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        # Layer 1: Linear + Sigmoid
        x = fused_ops.fused_linear_sigmoid_cuda(x, self.linear1_weight, self.linear1_bias)
        
        # Layer 2: Linear + LogSumExp
        x = fused_ops.fused_linear_logsumexp_cuda(x, self.linear2_weight, self.linear2_bias)
        
        return x