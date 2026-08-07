import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LeakyReLU, Add, Clamp, and GELU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Approximation for GELU (Gaussian Error Linear Unit)
__device__ __forceinline__ float gelu_approx(float x) {
    return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x)));
}

__global__ void fused_ops_kernel(const float* __restrict__ x, const float* __restrict__ sum_tensor, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // LeakyReLU with negative_slope=0.2
        val = (val >= 0.0f) ? val : 0.2f * val;
        // Add sum_tensor (broadcasting is handled by the caller or assumed shape match)
        val += sum_tensor[idx]; // Assuming sum_tensor is broadcasted to match x shape or passed correctly
        // Clamp between -1.0 and 1.0
        val = fmaxf(-1.0f, fminf(1.0f, val));
        // GELU activation
        out[idx] = gelu_approx(val);
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor sum_tensor) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_ops_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), sum_tensor.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor sum_tensor);"
)

# Compile the inline CUDA code for fused operations
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    The post-convolution operations are fused into a single custom CUDA kernel for performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        # The sum_tensor has shape (out_channels, 1, 1, 1). 
        # PyTorch broadcasts this automatically during addition.
        # We rely on PyTorch's broadcasting mechanism to expand sum_tensor to match x's shape before passing to the kernel.
        # Alternatively, we could handle broadcasting in the kernel, but passing the expanded tensor is simpler and leverages PyTorch's efficient broadcasting.
        # However, to avoid an explicit allocation for the expanded tensor, we can pass the original tensor and handle broadcasting logic in the kernel or assume the input tensor is contiguous.
        # For simplicity and correctness with the provided kernel, we expand the sum_tensor to match x's shape.
        # Note: In a highly optimized scenario, we would modify the kernel to handle the broadcasting logic directly to avoid the overhead of expanding the tensor.
        # But given the constraints, we'll expand it here.
        # Actually, since the kernel assumes pointers match, we should expand.
        # Let's expand the sum_tensor to match the shape of x.
        # This might introduce overhead, but it ensures correctness with the simple kernel above.
        # A better approach for the kernel would be to accept strides or dimensions, but that complicates the kernel.
        # Let's stick to the expanded approach for now, or better yet, modify the kernel to handle broadcasting.
        
        # Let's modify the kernel call to handle the broadcasting implicitly by passing the original tensor.
        # We need to update the kernel to handle the broadcasting.
        # Re-writing the kernel to handle broadcasting:
        pass # Placeholder for the updated kernel logic below.

        # Updated kernel with broadcasting logic
        fused_ops_broadcast_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float gelu_approx(float x) {
    return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x)));
}

__global__ void fused_ops_broadcast_kernel(const float* __restrict__ x, const float* __restrict__ sum_tensor, float* __restrict__ out, int N, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * C * D * H * W;
    if (idx < total_size) {
        // Decompose linear index into (n, c, d, h, w)
        int w = idx % W;
        int h = (idx / W) % H;
        int d = (idx / (W * H)) % D;
        int c = (idx / (W * H * D)) % C;
        int n = idx / (W * H * D * C);

        float val = x[idx];
        // LeakyReLU
        val = (val >= 0.0f) ? val : 0.2f * val;
        
        // Add sum_tensor. sum_tensor has shape (C, 1, 1, 1) (broadcasted to N, C, D, H, W)
        // The index in sum_tensor corresponds to channel c.
        // sum_tensor is stored as (C, 1, 1, 1), so its linear index is just c.
        val += sum_tensor[c];
        
        // Clamp
        val = fmaxf(-1.0f, fminf(1.0f, val));
        
        // GELU
        out[idx] = gelu_approx(val);
    }
}

torch::Tensor fused_ops_broadcast_cuda(torch::Tensor x, torch::Tensor sum_tensor) {
    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    auto out = torch::empty_like(x);
    int total_size = x.numel();

    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;

    fused_ops_broadcast_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), sum_tensor.data_ptr<float>(), out.data_ptr<float>(), N, C, D, H, W);

    return out;
}
"""
        fused_ops_broadcast_cpp_source = (
            "torch::Tensor fused_ops_broadcast_cuda(torch::Tensor x, torch::Tensor sum_tensor);"
        )

        self.fused_ops_broadcast = load_inline(
            name="fused_ops_broadcast",
            cpp_sources=fused_ops_broadcast_cpp_source,
            cuda_sources=fused_ops_broadcast_source,
            functions=["fused_ops_broadcast_cuda"],
            verbose=True,
            extra_cflags=[""],
            extra_ldflags=[""],
        )
        
        return self.fused_ops_broadcast.fused_ops_broadcast_cuda(x, self.sum_tensor)