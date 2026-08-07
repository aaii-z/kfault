import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softmax and Sigmoid fusion
softmax_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// Kernel for Softmax followed by Sigmoid
// Optimized for FP32 precision
__global__ void softmax_sigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, int channels, int spatial_size) {
    int spatial_idx = blockIdx.x; // Each block handles one spatial location
    int tid = threadIdx.x;        // Thread ID within the block
    
    if (spatial_idx >= spatial_size) return;

    // Pointers to the start of the channel dimension for this spatial location
    const float* x_ptr = input + spatial_idx * channels;
    float* out_ptr = output + spatial_idx * channels;

    // Step 1: Find the maximum value for numerical stability
    // Use shared memory for reduction
    extern __shared__ float s_data[];
    float local_max = -INFINITY;
    
    // Grid-stride loop to find max
    for (int c = tid; c < channels; c += blockDim.x) {
        local_max = fmaxf(local_max, x_ptr[c]);
    }
    s_data[tid] = local_max;
    __syncthreads();

    // Block reduction to find the max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }
    float max_val = s_data[0];
    
    // Step 2: Compute exponentials and sum
    float local_sum = 0.0f;
    for (int c = tid; c < channels; c += blockDim.x) {
        float exp_val = expf(x_ptr[c] - max_val);
        // Store exp values temporarily in output to avoid re-computation or extra memory
        // Or just compute sum here. Since we need exp(x - max) later, we can write it to output buffer
        // But output buffer is float, expf is float. 
        // Let's write exp values to output buffer first, then normalize.
        out_ptr[c] = exp_val;
        local_sum += exp_val;
    }
    
    // Reduce sum
    s_data[tid] = local_sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    float sum_val = s_data[0];

    // Step 3: Normalize (Softmax) and apply Sigmoid
    // Softmax: exp(x - max) / sum
    // Sigmoid: 1 / (1 + exp(-softmax))
    // Combined: 1 / (1 + exp(-(exp(x - max) / sum)))
    for (int c = tid; c < channels; c += blockDim.x) {
        float softmax_val = out_ptr[c] / sum_val;
        out_ptr[c] = 1.0f / (1.0f + expf(-softmax_val));
    }
}

torch::Tensor softmax_sigmoid_cuda(torch::Tensor x) {
    // x shape: (batch, channels, D, H, W)
    auto sizes = x.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int spatial_size = batch_size * sizes[2] * sizes[3] * sizes[4]; // Flatten batch and spatial dims
    
    auto output = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = spatial_size;
    // Shared memory size for reduction (max and sum)
    size_t shared_mem_size = block_size * sizeof(float);

    softmax_sigmoid_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        channels, 
        spatial_size
    );

    return output;
}
"""

softmax_sigmoid_cpp_source = (
    "torch::Tensor softmax_sigmoid_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code
softmax_sigmoid_op = load_inline(
    name="softmax_sigmoid",
    cpp_sources=softmax_sigmoid_cpp_source,
    cuda_sources=softmax_sigmoid_source,
    functions=["softmax_sigmoid_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by a fused Softmax and Sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax_sigmoid = softmax_sigmoid_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = self.softmax_sigmoid.softmax_sigmoid_cuda(x)
        return x