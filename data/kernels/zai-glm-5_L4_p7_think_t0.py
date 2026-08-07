import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Constants for GELU
const float SQRT2_INV = 0.7071067811865475f; // 1/sqrt(2)
const float SQRT2PI_INV = 0.3989422804014327f; // 1/sqrt(2*pi)

__global__ void fused_forward_kernel(const float* __restrict__ input, const float* __restrict__ bias, float* __restrict__ output, int size, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        
        // 1. ReLU
        x = fmaxf(0.0f, x);
        
        // 2. LeakyReLU (slope 0.01)
        // Note: Since input to LeakyReLU is output of ReLU (non-negative),
        // LeakyReLU(x) = max(0, x) + 0.01 * min(0, x).
        // If x >= 0, LeakyReLU(x) = x.
        // If x < 0 (impossible here), LeakyReLU(x) = 0.01*x.
        // So LeakyReLU(Relu(x)) is effectively just ReLU(x).
        // We skip explicit calculation.

        // 3. GELU (exact approximation using erf)
        // GELU(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        float cdf = 0.5f * (1.0f + erff(x * SQRT2_INV));
        x = x * cdf;

        // 4. Sigmoid
        x = 1.0f / (1.0f + expf(-x));

        // 5. Add Bias
        // Bias is (C, 1, 1, 1), input is (N, C, D, H, W)
        // We need to map the linear index to the channel index.
        // spatial_size = D * H * W
        int c = (idx / spatial_size) % channels;
        x += bias[c];

        output[idx] = x;
    }
}

__global__ void fused_backward_kernel(const float* __restrict__ grad_output, const float* __restrict__ input, float* __restrict__ grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float go = grad_output[idx];
        float x = input[idx];

        // Recompute forward pass intermediates for derivatives
        // u = ReLU(x)
        float u = fmaxf(0.0f, x);

        // v = GELU(u)
        float cdf = 0.5f * (1.0f + erff(u * SQRT2_INV));
        float pdf = expf(-0.5f * u * u) * SQRT2PI_INV;
        float v = u * cdf;

        // w = Sigmoid(v)
        float w = 1.0f / (1.0f + expf(-v));

        // Backprop chain rule
        // d_loss/d_x = d_loss/d_w * d_w/d_v * d_v/d_u * d_u/d_x
        
        // d_w/d_v (Sigmoid derivative)
        float dsigmoid = w * (1.0f - w);

        // d_v/d_u (GELU derivative)
        // d/du (u * Phi(u)) = Phi(u) + u * phi(u)
        float dgelu = cdf + u * pdf;

        // d_u/d_x (ReLU derivative)
        float drelu = (x > 0.0f) ? 1.0f : 0.0f;

        grad_input[idx] = go * dsigmoid * dgelu * drelu;
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor bias) {
    auto sizes = input.sizes();
    int size = input.numel();
    int channels = sizes[1];
    int spatial_size = sizes[2] * sizes[3] * sizes[4];
    
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size, channels, spatial_size
    );
    
    return output;
}

torch::Tensor fused_backward_cuda(torch::Tensor grad_output, torch::Tensor input) {
    auto grad_input = torch::empty_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_backward_kernel<<<num_blocks, block_size>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        size
    );
    
    return grad_input;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor bias);
torch::Tensor fused_backward_cuda(torch::Tensor grad_output, torch::Tensor input);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_forward_cuda", "fused_backward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the autograd function
class FusedActivationBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input)
        # bias is (C, 1, 1, 1), flatten to (C) for kernel
        return fused_ops.fused_forward_cuda(input, bias.view(-1))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = fused_ops.fused_backward_cuda(grad_output, input)
        
        # grad_bias: sum over batch and spatial dims
        # grad_output shape: (N, C, D, H, W)
        # bias shape: (C, 1, 1, 1)
        grad_bias = grad_output.sum(dim=(0, 2, 3, 4), keepdim=True)
        
        return grad_input, grad_bias

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution followed by a fused activation and bias add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.fused_op = FusedActivationBias.apply

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op(x, self.bias)
        return x