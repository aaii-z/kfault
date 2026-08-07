import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused pointwise operations CUDA kernel
fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

namespace {

template <typename T>
__device__ __forceinline__ T gelu(T x) {
    const float scale = 0.7978845608028654f; // sqrt(2/pi)
    const float approx = 0.044715f;
    float x_flt = static_cast<float>(x);
    float cube = x_flt * x_flt * x_flt;
    float inner = scale * (x_flt + approx * cube);
    float tanh_inner = tanhf(inner);
    return 0.5f * x_flt * (1.0f + tanh_inner);
}

template <typename T>
__global__ void fused_ops_kernel(
    const T* input,
    const T* sum_tensor,
    T* output,
    const int num_elements,
    const int channels,
    const int spatial_size,
    const float negative_slope) {
    
    CUDA_1D_KERNEL_LOOP(idx, num_elements) {
        const int c = (idx / spatial_size) % channels;
        
        float val = static_cast<float>(input[idx]);
        // LeakyReLU
        val = val < 0 ? val * negative_slope : val;
        // Add sum tensor
        val += static_cast<float>(sum_tensor[c]);
        // Clamp
        val = fminf(1.0f, fmaxf(-1.0f, val));
        // GELU
        val = gelu(val);
        
        output[idx] = static_cast<T>(val);
    }
}

} // namespace

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor sum_tensor,
    float negative_slope) {
    
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    int channels = input.size(1);
    int spatial_size = input.numel() / (input.size(0) * channels);
    
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "fused_ops_kernel",
        [&] {
            fused_ops_kernel<scalar_t><<<num_blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                sum_tensor.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                num_elements,
                channels,
                spatial_size,
                negative_slope);
        });
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor sum_tensor, float negative_slope);
"""

# Load the custom CUDA extension
fused_ops_ext = load_inline(
    name='fused_ops',
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=['fused_ops_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.negative_slope = 0.2

    def forward(self, x):
        x = self.conv(x)
        x = fused_ops_ext.fused_ops_cuda(x, self.sum_tensor, self.negative_slope)
        return x