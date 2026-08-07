import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_mish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_mish_mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        
        // First Mish
        float sp1 = (x > 20.0f) ? x : log1pf(expf(x));
        float tanh_sp1 = tanhf(sp1);
        float y = x * tanh_sp1;
        
        // Second Mish
        float sp2 = (y > 20.0f) ? y : log1pf(expf(y));
        float tanh_sp2 = tanhf(sp2);
        float z = y * tanh_sp2;
        
        output[idx] = z;
    }
}

torch::Tensor fused_mish_mish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_mish_mish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    
    return output;
}
"""

fused_mish_mish_cpp = "torch::Tensor fused_mish_mish_cuda(torch::Tensor input);"

fused_mish_mish = load_inline(
    name="fused_mish_mish",
    cpp_sources=fused_mish_mish_cpp,
    cuda_sources=fused_mish_mish_source,
    functions=["fused_mish_mish_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_mish_mish = fused_mish_mish
    
    def forward(self, x):
        x = self.linear(x)
        x = self.fused_mish_mish.fused_mish_mish_cuda(x)
        return x