import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Softmax+Sigmoid
softmax_sigmoid_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

template <typename T>
__device__ void warpReduceMax(volatile T* sdata, int tid) {
    sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
    sdata[tid] = fmaxf(sdata[tid], sdata[tid + 16]);
    sdata[tid] = fmaxf(sdata[tid], sdata[tid + 8]);
    sdata[tid] = fmaxf(sdata[tid], sdata[tid + 4]);
    sdata[tid] = fmaxf(sdata[tid], sdata[tid + 2]);
    sdata[tid] = fmaxf(sdata[tid], sdata[tid + 1]);
}

template <typename T>
__device__ void warpReduceSum(volatile T* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void softmax_sigmoid_kernel(const float* input, float* output, 
                                      int num_channels, int spatial_size) {
    extern __shared__ float sdata[];
    float* max_vals = sdata;
    float* sums = &sdata[blockDim.x];

    int spatial_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (spatial_idx >= spatial_size) return;

    const float* channel_ptr = input + spatial_idx * num_channels;
    float* out_ptr = output + spatial_idx * num_channels;

    // Each thread loads one element
    float val = (tid < num_channels) ? channel_ptr[tid] : -FLT_MAX;
    max_vals[tid] = val;
    __syncthreads();

    // Parallel reduction for max value
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + s]);
        }
        __syncthreads();
    }
    if (tid < 32) warpReduceMax(max_vals, tid);
    __syncthreads();

    float max_val = max_vals[0];
    __syncthreads();

    // Compute exponentials and sum
    float exp_val = (tid < num_channels) ? expf(val - max_val) : 0.0f;
    sums[tid] = exp_val;
    __syncthreads();

    // Parallel reduction for sum
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sums[tid] += sums[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) warpReduceSum(sums, tid);
    __syncthreads();

    float sum_val = sums[0];
    
    // Compute softmax and sigmoid
    if (tid < num_channels) {
        float softmax_val = exp_val / sum_val;
        out_ptr[tid] = 1.0f / (1.0f + expf(-softmax_val));
    }
}

torch::Tensor softmax_sigmoid_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    int num_channels = sizes[1];
    int spatial_size = sizes[0] * sizes[2] * sizes[3] * sizes[4];
    
    auto output = torch::empty_like(input);
    
    dim3 blocks(spatial_size);
    dim3 threads((num_channels + 31) / 32 * 32);  // Round up to nearest multiple of 32
    int shared_mem = 2 * threads.x * sizeof(float);
    
    softmax_sigmoid_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(),
        num_channels,
        spatial_size
    );
    
    return output;
}
"""

softmax_sigmoid_cpp = """
torch::Tensor softmax_sigmoid_cuda(torch::Tensor input);
"""

# Load the custom CUDA extension
softmax_sigmoid_ext = load_inline(
    name='softmax_sigmoid_ext',
    cpp_sources=softmax_sigmoid_cpp,
    cuda_sources=softmax_sigmoid_kernel,
    functions=['softmax_sigmoid_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding, bias=bias
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = softmax_sigmoid_ext.softmax_sigmoid_cuda(x)
        return x