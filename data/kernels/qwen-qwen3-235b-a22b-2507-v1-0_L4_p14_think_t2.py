import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_forward_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reduce_weight_kernel(const float* weight, float* col_sum, int hidden_size, int input_size, int stride0, int stride1) {
    int k = blockIdx.x;
    if (k >= input_size) return;
    
    float sum = 0.0f;
    for (int j = 0; j < hidden_size; j++) {
        sum += weight[j * stride0 + k * stride1];
    }
    col_sum[k] = sum;
}

__global__ void matvec_scale_kernel(const float* x, int x_stride0, int x_stride1, const float* col_sum, float* out, int batch_size, int input_size, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    
    float sum = 0.0f;
    for (int k = 0; k < input_size; k++) {
        sum += x[i * x_stride0 + k * x_stride1] * col_sum[k];
    }
    out[i] = sum * scale;
}

torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor) {
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat, "weight must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    
    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = weight.size(0);
    TORCH_CHECK(weight.size(1) == input_size, "weight size mismatch");
    
    auto col_sum = torch::zeros({input_size}, x.options());
    
    dim3 grid_reduce(input_size);
    if (input_size > 0 && hidden_size > 0) {
        int stride0_weight = weight.stride(0);
        int stride1_weight = weight.stride(1);
        reduce_weight_kernel<<<grid_reduce, 1>>>(
            weight.data_ptr<float>(), 
            col_sum.data_ptr<float>(), 
            hidden_size, 
            input_size,
            stride0_weight,
            stride1_weight
        );
    }
    
    auto out_temp = torch::zeros({batch_size}, x.options());
    float scale_val = scaling_factor / 2.0f;
    
    int block_size_matvec = 256;
    int grid_size_matvec = (batch_size + block_size_matvec - 1) / block_size_matvec;
    if (batch_size > 0 && input_size > 0) {
        int x_stride0 = x.stride(0);
        int x_stride1 = x.stride(1);
        matvec_scale_kernel<<<grid_size_matvec, block_size_matvec>>>(
            x.data_ptr<float>(),
            x_stride0,
            x_stride1,
            col_sum.data_ptr<float>(),
            out_temp.data_ptr<float>(),
            batch_size,
            input_size,
            scale_val
        );
    }
    
    return out_temp.unsqueeze(1);
}
'''

fused_forward_cpp_source = "torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor);"

fused_forward = load_inline(
    name="fused_forward",
    cpp_sources=fused_forward_cpp_source,
    cuda_sources=fused_forward_source,
    functions=["fused_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_forward = fused_forward

    def forward(self, x):
        return self.fused_forward.fused_forward_cuda(x, self.weight, self.scaling_factor)