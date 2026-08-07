import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused clamp, spatial softmax, and scaling
clamp_softmax_scale_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

__global__ void clamp_spatial_softmax_scale_kernel(
    const float* input,
    float* output,
    const float* scale,
    float clamp_min,
    float clamp_max,
    int batch_size,
    int channels,
    int spatial_size
) {
    extern __shared__ float shared_data[];
    int batch = blockIdx.x / channels;
    int channel = blockIdx.x % channels;
    int offset = (batch * channels + channel) * spatial_size;

    float* shared_max = shared_data;
    float* shared_sum = shared_data + blockDim.x;

    // Step 1: Clamp and find max
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = input[offset + i];
        val = fmaxf(fminf(val, clamp_max), clamp_min);
        if (val > max_val) max_val = val;
    }

    shared_max[threadIdx.x] = max_val;
    __syncthreads();

    // Reduce max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_max[threadIdx.x + s] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    max_val = shared_max[0];
    __syncthreads();

    // Step 2: Compute sum of exponentials
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = input[offset + i];
        val = fmaxf(fminf(val, clamp_max), clamp_min);
        sum_exp += expf(val - max_val);
    }

    shared_sum[threadIdx.x] = sum_exp;
    __syncthreads();

    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_exp = shared_sum[0];
    __syncthreads();

    // Step 3: Normalize and scale
    float scale_val = scale[channel];
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = input[offset + i];
        val = fmaxf(fminf(val, clamp_max), clamp_min);
        output[offset + i] = expf(val - max_val) / sum_exp * scale_val;
    }
}

torch::Tensor clamp_spatial_softmax_scale_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    float clamp_min,
    float clamp_max
) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int spatial_size = sizes[2] * sizes[3] * sizes[4];
    auto output = torch::empty_like(input);

    dim3 grid(batch_size * channels);
    dim3 block(1024);
    size_t shared_mem_size = 2 * block.x * sizeof(float);

    clamp_spatial_softmax_scale_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scale.data_ptr<float>(),
        clamp_min,
        clamp_max,
        batch_size,
        channels,
        spatial_size
    );

    return output;
}
"""

clamp_softmax_scale_cpp_source = """
torch::Tensor clamp_spatial_softmax_scale_cuda(torch::Tensor input, torch::Tensor scale, float clamp_min, float clamp_max);
"""

# Compile the inline CUDA code
clamp_softmax_scale = load_inline(
    name="clamp_softmax_scale",
    cpp_sources=clamp_softmax_scale_cpp_source,
    cuda_sources=clamp_softmax_scale_source,
    functions=["clamp_spatial_softmax_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.clamp_softmax_scale = clamp_softmax_scale

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = self.clamp_softmax_scale.clamp_spatial_softmax_scale_cuda(x, self.scale.squeeze(0).squeeze(-1).squeeze(-1).squeeze(-1), self.clamp_min, self.clamp_max)
        return x