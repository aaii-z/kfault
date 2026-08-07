import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Sigmoid, Scaling, and Residual Addition
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_forward_kernel(const float* __restrict__ x, float* __restrict__ out, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float sig = 1.0f / (1.0f + expf(-val));
        out[idx] = sig * scale + val;
    }
}

__global__ void fused_backward_kernel(const float* __restrict__ grad_output, const float* __restrict__ x, float* __restrict__ grad_input, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float go = grad_output[idx];
        float val = x[idx];
        float sig = 1.0f / (1.0f + expf(-val));
        float sig_deriv = sig * (1.0f - sig);
        grad_input[idx] = go * (sig_deriv * scale + 1.0f);
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor x, float scale) {
    auto out = torch::empty_like(x);
    int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_forward_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), scale, size
    );
    return out;
}

torch::Tensor fused_backward_cuda(torch::Tensor grad_output, torch::Tensor x, float scale) {
    auto grad_input = torch::empty_like(x);
    int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_backward_kernel<<<num_blocks, block_size>>>(
        grad_output.data_ptr<float>(), x.data_ptr<float>(), grad_input.data_ptr<float>(), scale, size
    );
    return grad_input;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_forward_cuda(torch::Tensor x, float scale);\n"
    "torch::Tensor fused_backward_cuda(torch::Tensor grad_output, torch::Tensor x, float scale);"
)

# Compile the inline CUDA code
fused_op_module = load_inline(
    name="fused_op_module",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_forward_cuda", "fused_backward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the autograd function to support backward pass
class FusedSigmoidScaleAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x)
        ctx.scale = scale
        return fused_op_module.fused_forward_cuda(x, scale)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        scale = ctx.scale
        grad_input = fused_op_module.fused_backward_cuda(grad_output, x, scale)
        return grad_input, None

class ModelNew(nn.Module):
    """
    Optimized Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    Uses a custom CUDA kernel for the fused element-wise operations.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        self.fused_op = FusedSigmoidScaleAdd.apply

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = self.gemm(x)
        x = self.fused_op(x, self.scaling_factor)
        return x