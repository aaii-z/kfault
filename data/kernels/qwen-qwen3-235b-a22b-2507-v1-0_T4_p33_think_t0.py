import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_gemm_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_kernel(
    const float* input,
    const float* weight,
    const float* A,
    const float* bias_fused,
    float* output,
    int in_features,
    int out_features,
    int total) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    int i = idx / out_features;
    int j = idx % out_features;
    
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[i * in_features + k] * weight[j * in_features + k];
    }
    output[idx] = sum * A[j] + bias_fused[j];
}

torch::Tensor fused_gemm_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor A,
    torch::Tensor bias_fused,
    int in_features,
    int out_features) {
    
    auto batch_size = input.size(0);
    auto total = batch_size * out_features;
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_gemm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        A.data_ptr<float>(),
        bias_fused.data_ptr<float>(),
        output.data_ptr<float>(),
        in_features,
        out_features,
        total
    );
    
    return output;
}
"""

fused_gemm_bn_cpp = "torch::Tensor fused_gemm_bn_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor A, torch::Tensor bias_fused, int in_features, int out_features);"

fused_gemm_bn = load_inline(
    name="fused_gemm_bn",
    cpp_sources=fused_gemm_bn_cpp,
    cuda_sources=fused_gemm_bn_source,
    functions=["fused_gemm_bn_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        
        with torch.no_grad():
            std = torch.sqrt(self.bn.running_var + self.bn.eps)
            scale_bn = self.bn.weight / std
            bias_bn = self.bn.bias - self.bn.running_mean * scale_bn
            A_fused = self.scale * scale_bn
            bias_fused_val = self.gemm.bias * A_fused + bias_bn
            
        self.register_buffer('A_fused', A_fused)
        self.register_buffer('bias_fused', bias_fused_val)
        self.in_features = in_features
        self.out_features = out_features
        self.fused_op = fused_gemm_bn

    def forward(self, x):
        return self.fused_op.fused_gemm_bn_cuda(
            x, 
            self.gemm.weight, 
            self.A_fused, 
            self.bias_fused, 
            self.in_features, 
            self.out_features
        )