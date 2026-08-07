import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void optimized_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= batch_size || col >= out_features) return;

    extern __shared__ float shared_mem[];
    float* row_cache = shared_mem;
    
    float sum = 0.0f;
    for (int k = 0; k < in_features; k += blockDim.x) {
        int load_idx = k + threadIdx.x;
        if (load_idx < in_features) {
            row_cache[threadIdx.x] = input[row * in_features + load_idx];
        }
        __syncthreads();
        
        for (int i = 0; i < blockDim.x; i++) {
            int feat_idx = k + i;
            if (feat_idx < in_features) {
                sum += row_cache[i] * weight[col * in_features + feat_idx];
            }
        }
        __syncthreads();
    }
    
    sum += bias[col];
    
    atomicAdd(&output[row], sum);
}

__global__ void reduction_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float val = input[idx];
    float max_val = val;
    float mean_val = val;
    float lse1 = logf(expf(mean_val));
    float lse2 = logf(expf(lse1));
    
    output[idx] = lse2;
}

torch::Tensor optimized_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto intermediate = torch::zeros({batch_size}, input.options());
    auto output = torch::empty({batch_size, 1}, input.options());
    
    dim3 block(32, 32);
    dim3 grid((out_features + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
    
    optimized_forward_kernel<<<grid, block, sizeof(float) * 32>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    dim3 reduction_block(256);
    dim3 reduction_grid((batch_size + reduction_block.x - 1) / reduction_block.x);
    
    reduction_kernel<<<reduction_grid, reduction_block>>>(
        intermediate.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor optimized_forward_cuda(torch::Tensor, torch::Tensor, torch::Tensor);"

optimized_forward = load_inline(
    name='optimized_forward',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['optimized_forward_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return optimized_forward.optimized_forward_cuda(x, self.weight, self.bias)