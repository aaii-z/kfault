import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define PI 3.14159265358979323846f

__global__ void fused_kernel(const float* input, const float* scale, const float* shift,
                            float* output, int out_features, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int i = idx % out_features;
    int j = idx / out_features;

    float x = input[idx];
    float s = scale[i];
    float sh = shift[i];
    float z = x * s + sh;

    // GELU calculation
    const float c = 0.044715f;
    const float k = sqrtf(2.0f / PI);
    float z3 = z * z * z;
    float inner = k * (z + c * z3);
    float tanh_inner = tanhf(inner);
    float gelu = 0.5f * z * (1.0f + tanh_inner);

    // ReLU
    float result = (gelu > 0.0f) ? gelu : 0.0f;

    output[idx] = result;
}

torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor shift) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(scale.dim() == 1 && shift.dim() == 1, "Scale and shift must be 1D");
    TORCH_CHECK(scale.size(0) == input.size(1) && shift.size(0) == input.size(1),
                "Scale and shift size must match input features");

    int out_features = input.size(1);
    int total_elements = input.numel();

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        shift.data_ptr<float>(),
        output.data_ptr<float>(),
        out_features,
        total_elements
    );

    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor shift);"
)

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-lineinfo", "-maxrregcount=60"],
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses BatchNorm, GELU, and ReLU into a single CUDA kernel after GEMM.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        
        # Precompute scale and shift for fused BatchNorm (inference mode)
        self.scale = nn.Parameter(
            self.batch_norm.weight / torch.sqrt(self.batch_norm.running_var + self.batch_norm.eps),
            requires_grad=False
        )
        self.shift = nn.Parameter(
            self.batch_norm.bias - self.batch_norm.running_mean * self.scale,
            requires_grad=False
        )
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.fused_op.fused_op_cuda(x, self.scale, self.shift)
        return x