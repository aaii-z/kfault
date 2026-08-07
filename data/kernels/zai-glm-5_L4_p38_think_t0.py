import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Clamp + Softmax + Scale
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void softmax_clamp_scale_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ scale,
    float clamp_min, float clamp_max,
    int batch_size, int channels, int depth, int height, int width) {

    int b = blockIdx.y;
    int c = blockIdx.x;
    
    if (b >= batch_size || c >= channels) return;

    int spatial_size = depth * height * width;
    
    // Assuming contiguous memory layout (NCDHW)
    size_t base_offset = (b * channels + c) * spatial_size;
    const scalar_t* input_slice = input + base_offset;
    scalar_t* output_slice = output + base_offset;
    
    extern __shared__ char s_mem_char[];
    scalar_t* s_mem = reinterpret_cast<scalar_t*>(s_mem_char);
    
    int tid = threadIdx.x;
    
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    scalar_t local_sum = 0.0;
    
    // Pass 1: Online Softmax (Max + Sum)
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        scalar_t val = input_slice[i];
        // Clamp
        val = max(static_cast<scalar_t>(clamp_min), min(static_cast<scalar_t>(clamp_max), val));
        
        scalar_t new_max = max(local_max, val);
        local_sum = local_sum * exp(local_max - new_max) + exp(val - new_max);
        local_max = new_max;
    }
    
    s_mem[tid] = local_max;
    s_mem[blockDim.x + tid] = local_sum;
    __syncthreads();
    
    // Block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scalar_t m1 = s_mem[tid];
            scalar_t m2 = s_mem[tid + s];
            scalar_t s1 = s_mem[blockDim.x + tid];
            scalar_t s2 = s_mem[blockDim.x + tid + s];
            
            scalar_t new_max = max(m1, m2);
            scalar_t new_sum = s1 * exp(m1 - new_max) + s2 * exp(m2 - new_max);
            
            s_mem[tid] = new_max;
            s_mem[blockDim.x + tid] = new_sum;
        }
        __syncthreads();
    }
    
    scalar_t global_max = s_mem[0];
    scalar_t global_sum = s_mem[blockDim.x];
    scalar_t scale_val = scale[c];
    
    // Pass 2: Normalize and write
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        scalar_t val = input_slice[i];
        val = max(static_cast<scalar_t>(clamp_min), min(static_cast<scalar_t>(clamp_max), val));
        scalar_t exp_val = exp(val - global_max);
        scalar_t softmax_val = exp_val / global_sum;
        output_slice[i] = softmax_val * scale_val;
    }
}

torch::Tensor softmax_clamp_scale_cuda(torch::Tensor input, torch::Tensor scale, float clamp_min, float clamp_max) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];
    
    auto output = torch::empty_like(input);
    
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "Output must be contiguous");

    dim3 grid(channels, batch_size);
    int threads = 256;
    int shared_mem_size = 2 * threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "softmax_clamp_scale_cuda", ([&] {
        softmax_clamp_scale_kernel<scalar_t><<<grid, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            scale.data_ptr<scalar_t>(),
            clamp_min, clamp_max,
            batch_size, channels, depth, height, width
        );
    }));
    
    return output;
}
"""

cpp_source = "torch::Tensor softmax_clamp_scale_cuda(torch::Tensor input, torch::Tensor scale, float clamp_min, float clamp_max);"

# Compile the inline CUDA code
softmax_op = load_inline(
    name="softmax_clamp_scale",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["softmax_clamp_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.softmax_op = softmax_op

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = self.softmax_op.softmax_clamp_scale_cuda(x, self.scale, self.clamp_min, self.clamp_max)
        return x