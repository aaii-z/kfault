import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused operations
scaled_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

template<typename T>
__global__ void scaled_mean_kernel(
    const T* input,
    T* output,
    int batch_size,
    int channels,
    int spatial_size,
    float multiplier
) {
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    
    if (batch >= batch_size || channel >= channels) return;
    
    const T* batch_channel_ptr = input + (batch * channels + channel) * spatial_size;
    T sum = 0.0f;
    
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        sum += batch_channel_ptr[i];
    }
    
    __shared__ T shared_sum[1024];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        output[batch * channels + channel] = static_cast<T>(shared_sum[0] * multiplier);
    }
}

torch::Tensor scaled_mean_cuda(
    torch::Tensor input,
    float multiplier
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int spatial_size = input.size(2) * input.size(3);
    
    auto output = torch::empty({batch_size, channels, 1, 1}, input.options());
    
    dim3 grid(batch_size, channels);
    int block_size = 1024;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "scaled_mean_kernel", [&] {
        scaled_mean_kernel<scalar_t><<<grid, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            spatial_size,
            multiplier
        );
    });
    
    return output;
}
"""

scaled_mean_cpp_source = "torch::Tensor scaled_mean_cuda(torch::Tensor input, float multiplier);"

# Compile the inline CUDA code
scaled_mean = load_inline(
    name='scaled_mean',
    cpp_sources=scaled_mean_cpp_source,
    cuda_sources=scaled_mean_source,
    functions=['scaled_mean_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding
        )
        self.multiplier = multiplier
        self.scaled_mean = scaled_mean

    def forward(self, x):
        x = self.conv_transpose(x)
        # Calculate spatial size for normalization
        spatial_size = x.size(2) * x.size(3)
        # Use custom fused kernel for scaling and mean reduction
        x = self.scaled_mean.scaled_mean_cuda(x, self.multiplier / spatial_size)
        return x