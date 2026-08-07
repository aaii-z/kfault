import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
custom_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_reduction_kernel(
    const float* conv_output,
    const float* bias,
    float* output,
    int batch_size,
    int out_channels,
    int H,
    int W
) {
    int batch_idx = blockIdx.x;
    int channel_idx = threadIdx.x;

    if (batch_idx >= batch_size || channel_idx >= out_channels) return;

    // Compute global average for this channel
    float sum = 0.0f;
    int spatial_size = H * W;
    int offset = (batch_idx * out_channels + channel_idx) * spatial_size;
    
    for (int i = 0; i < spatial_size; ++i) {
        sum += conv_output[offset + i];
    }
    float avg = sum / spatial_size;
    float biased_avg = avg + bias[channel_idx];

    // Store intermediate result in shared memory
    extern __shared__ float shared_avgs[];
    shared_avgs[channel_idx] = biased_avg;
    __syncthreads();

    // Compute logsumexp across channels (first find max)
    if (channel_idx == 0) {
        float max_val = -INFINITY;
        for (int c = 0; c < out_channels; ++c) {
            if (shared_avgs[c] > max_val) {
                max_val = shared_avgs[c];
            }
        }

        // Compute sum of exponentials
        float exp_sum = 0.0f;
        for (int c = 0; c < out_channels; ++c) {
            exp_sum += expf(shared_avgs[c] - max_val);
        }

        // Final computation
        float result = logf(exp_sum) + max_val;
        output[batch_idx] = result * 10.0f;
    }
}

torch::Tensor fused_operations_cuda(
    torch::Tensor conv_output,
    torch::Tensor bias
) {
    int batch_size = conv_output.size(0);
    int out_channels = conv_output.size(1);
    int H = conv_output.size(2);
    int W = conv_output.size(3);
    
    auto output = torch::empty({batch_size, 1}, conv_output.options());
    
    dim3 blocks(batch_size);
    dim3 threads(out_channels);
    size_t shared_mem = out_channels * sizeof(float);
    
    fused_reduction_kernel<<<blocks, threads, shared_mem>>>(
        conv_output.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        H,
        W
    );
    
    return output;
}
"""

custom_kernel_cpp = "torch::Tensor fused_operations_cuda(torch::Tensor conv_output, torch::Tensor bias);"

# Load the custom CUDA extension
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=custom_kernel_cpp,
    cuda_sources=custom_kernel_source,
    functions=['fused_operations_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape).squeeze())  # Squeeze to 1D

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_ops.fused_operations_cuda(x, self.bias)
        return x