import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + groupnorm + leaky_relu + add
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void fused_matmul_gn_leaky_relu_add_kernel(
    const float* input, const float* weight, const float* bias,
    const float* gamma, const float* beta, float* output,
    int batch_size, int input_size, int hidden_size,
    int num_groups, float eps, float negative_slope) {
    
    const int group_size = hidden_size / num_groups;
    const int group_id = blockIdx.y;
    const int element_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (element_id >= batch_size) return;
    
    extern __shared__ float shared_mem[];
    float* group_means = shared_mem;
    float* group_vars = shared_mem + num_groups;
    
    // Compute matmul for this element
    float sum = 0.0f;
    for (int i = 0; i < input_size; ++i) {
        sum += input[element_id * input_size + i] * weight[group_id * group_size * input_size + i];
    }
    sum += bias[group_id * group_size];
    
    // Store in shared memory for reduction
    __shared__ float thread_sums[256];
    thread_sums[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduce to compute group mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            thread_sums[threadIdx.x] += thread_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        group_means[group_id] = thread_sums[0] / batch_size;
    }
    __syncthreads();
    
    // Compute variance
    float diff = sum - group_means[group_id];
    thread_sums[threadIdx.x] = diff * diff;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            thread_sums[threadIdx.x] += thread_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        group_vars[group_id] = thread_sums[0] / batch_size + eps;
    }
    __syncthreads();
    
    // Apply group normalization
    float normalized = (sum - group_means[group_id]) / sqrtf(group_vars[group_id]);
    float scaled = normalized * gamma[group_id * group_size + threadIdx.x % group_size] + beta[group_id * group_size + threadIdx.x % group_size];
    
    // Apply leaky ReLU
    float activated = (scaled > 0.0f) ? scaled : scaled * negative_slope;
    
    // Element-wise add (x + x)
    float result = activated + activated;
    
    output[element_id * hidden_size + group_id * group_size + threadIdx.x % group_size] = result;
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int num_groups, float eps, float negative_slope) {
    
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, hidden_size}, input.options());
    
    const int group_size = hidden_size / num_groups;
    dim3 blocks((batch_size + 255) / 256, num_groups);
    dim3 threads(256);
    
    size_t shared_mem_size = 2 * num_groups * sizeof(float);
    
    fused_matmul_gn_leaky_relu_add_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, input_size, hidden_size,
        num_groups, eps, negative_slope);
    
    return output;
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int num_groups, float eps, float negative_slope);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        
        # Initialize parameters manually
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        
        self.fused_op = fused_operator

    def forward(self, x):
        return self.fused_op.fused_operator_cuda(
            x, self.weight, self.bias, self.gamma, self.beta,
            self.num_groups, self.eps, self.negative_slope
        )