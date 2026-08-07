import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_bias_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_relu_kernel(const float* x, const float* bias, float* out, int total_elements, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int j = idx % out_features;
        float val = x[idx] + bias[j];
        out[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

torch::Tensor fused_bias_relu_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D tensor");
    TORCH_CHECK(x.size(1) == bias.size(0), "Bias size mismatch");
    
    auto batch_size = x.size(0);
    auto out_features = x.size(1);
    auto total_elements = batch_size * out_features;
    
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_bias_relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        total_elements, 
        out_features
    );
    
    return out;
}
"""

fused_bias_relu_cpp_source = (
    "torch::Tensor fused_bias_relu_cuda(torch::Tensor x, torch::Tensor bias);"
)

fused_bias_relu = load_inline(
    name="fused_bias_relu",
    cpp_sources=fused_bias_relu_cpp_source,
    cuda_sources=fused_bias_relu_source,
    functions=["fused_bias_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses bias addition and ReLU into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_bias_relu = fused_bias_relu

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.fused_bias_relu.fused_bias_relu_cuda(x, self.bias)
        return x