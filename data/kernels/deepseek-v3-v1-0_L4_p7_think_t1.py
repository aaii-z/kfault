import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused activation and bias CUDA kernel
fused_activation_bias_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(1); \
    } \
} while (0)

template <typename T>
__device__ T gelu_approximate(T x) {
    const T sqrt_2_over_pi = 0.7978845608028654;
    const T approx_const = 0.044715;
    T x_cube = x * x * x;
    T inner = sqrt_2_over_pi * (x + approx_const * x_cube);
    return 0.5 * x * (1.0 + tanh(inner));
}

template <typename T>
__global__ void fused_activation_bias_kernel(
    const T* input,
    const T* bias,
    T* output,
    int64_t total_elements,
    int channels,
    int depth,
    int height,
    int width
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int dhw = depth * height * width;
        int c_dhw = channels * dhw;
        int c_idx = (idx % c_dhw) / dhw;
        T val = input[idx];
        
        // ReLU
        val = val > 0 ? val : 0;
        
        // GELU
        val = gelu_approximate(val);
        
        // Sigmoid
        val = 1.0 / (1.0 + exp(-val));
        
        // Add bias
        val += bias[c_idx];
        
        output[idx] = val;
    }
}

torch::Tensor fused_activation_bias_cuda(
    torch::Tensor input,
    torch::Tensor bias
) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];
    
    int64_t total_elements = batch_size * channels * depth * height * width;
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "fused_activation_bias_kernel",
        [&] {
            fused_activation_bias_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                total_elements,
                channels,
                depth,
                height,
                width
            );
        }
    );
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

fused_activation_bias_cpp_source = """
torch::Tensor fused_activation_bias_cuda(torch::Tensor input, torch::Tensor bias);
"""

# Load the custom CUDA extension
fused_activation_bias = load_inline(
    name="fused_activation_bias",
    cpp_sources=fused_activation_bias_cpp_source,
    cuda_sources=fused_activation_bias_source,
    functions=["fused_activation_bias_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_activation_bias = fused_activation_bias

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation_bias.fused_activation_bias_cuda(x, self.bias)
        return x