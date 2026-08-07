import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel for min, clamp, and dropout
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void fused_min_clamp_dropout_kernel(
    const float* input,
    float* output,
    int num_elements,
    float min_val,
    float max_val,
    float dropout_p,
    bool training,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    float value = input[idx];
    value = min(value, min_val);
    value = clamp(value, min_val, max_val);

    if (training) {
        float random = curand_uniform(&state);
        if (random < dropout_p) {
            value = 0.0f;
        } else {
            value = value / (1.0f - dropout_p);
        }
    }
    output[idx] = value;
}

torch::Tensor fused_min_clamp_dropout_cuda(
    torch::Tensor input,
    float min_val,
    float max_val,
    float dropout_p,
    bool training
) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    unsigned long long seed = curand() % 1000000000ULL;
    
    fused_min_clamp_dropout_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        min_val,
        max_val,
        dropout_p,
        training,
        seed
    );
    
    return output;
}
"""

fused_kernel_cpp = "torch::Tensor fused_min_clamp_dropout_cuda(torch::Tensor, float, float, float, bool);"

fused_op = load_inline(
    name="fused_min_clamp_dropout",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_min_clamp_dropout_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.fused_op.fused_min_clamp_dropout_cuda(
            x, 
            self.min_value, 
            self.max_value, 
            self.dropout_p,
            self.training
        )
        return x