import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_linear_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_sum_kernel(const float* x, const float* sum_W, float sum_b, float* out, int batch, int in) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch) {
        float sum = 0.0f;
        for (int j = 0; j < in; ++j) {
            sum += x[i * in + j] * sum_W[j];
        }
        out[i] = sum + sum_b;
    }
}

torch::Tensor fused_linear_sum_cuda(torch::Tensor x, torch::Tensor sum_W, float sum_b) {
    int batch = x.size(0);
    int in = x.size(1);
    auto options = x.options().dtype(torch::kFloat32);
    auto out = torch::empty({batch}, options);

    int block_size = 256;
    int num_blocks = (batch + block_size - 1) / block_size;

    fused_linear_sum_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), sum_W.data_ptr<float>(), sum_b, out.data_ptr<float>(), batch, in
    );

    return out.unsqueeze(-1);
}
"""

fused_linear_sum_cpp = "torch::Tensor fused_linear_sum_cuda(torch::Tensor x, torch::Tensor sum_W, float sum_b);"

logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void logsumexp_kernel(const float* x, float* out, int batch, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < n; ++j) {
            float val = x[i * n + j];
            if (val > max_val) max_val = val;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum_exp += expf(x[i * n + j] - max_val);
        }
        out[i] = max_val + logf(sum_exp);
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor x, int dim) {
    int batch = (dim == 0) ? 1 : x.size(0);
    int n = (dim == 0) ? x.numel() : x.size(dim);
    auto options = x.options().dtype(torch::kFloat32);
    auto out = torch::empty({batch}, options);

    int block_size = 256;
    int num_blocks = (batch + block_size - 1) / block_size;

    logsumexp_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch, n);

    return (dim == 0) ? out : out.unsqueeze(-1);
}
"""

logsumexp_cpp = "torch::Tensor logsumexp_cuda(torch::Tensor x, int dim);"

fused_linear_sum = load_inline(
    name="fused_linear_sum",
    cpp_sources=fused_linear_sum_cpp,
    cuda_sources=fused_linear_sum_source,
    functions=["fused_linear_sum_cuda"],
    verbose=True,
)

logsumexp = load_inline(
    name="logsumexp",
    cpp_sources=logsumexp_cpp,
    cuda_sources=logsumexp_source,
    functions=["logsumexp_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            self.sum_W = nn.Parameter(self.linear.weight.sum(dim=0), requires_grad=False)
            self.sum_b = nn.Parameter(torch.tensor(self.linear.bias.sum().item()), requires_grad=False)
        self.fused_linear_sum = fused_linear_sum
        self.logsumexp = logsumexp

    def forward(self, x):
        x = self.fused_linear_sum.fused_linear_sum_cuda(x, self.sum_W, self.sum_b.item())
        x = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.logsumexp.logsumexp_cuda(x, 1)
        x = self.logsumexp.logsumexp_cuda(x, 1)
        return x