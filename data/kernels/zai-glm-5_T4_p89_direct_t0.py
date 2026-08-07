import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused softmax, subtract, swish, and max reduction
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// Kernel for Softmax + Subtract + Swish
// We process one spatial position (d, h, w) per block. Each thread handles multiple channels.
__global__ void fused_softmax_sub_swish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ subtract_param,
    float* __restrict__ intermediate_out,
    int num_channels, int depth, int height, int width) {
    
    // Calculate global spatial index
    int d = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    
    if (d >= depth || h >= height || w >= width) return;

    int spatial_idx = (d * height + h) * width + w;
    int offset = spatial_idx * num_channels;
    
    // 1. Find Max for numerical stability (using shared memory reduction)
    extern __shared__ float s_data[];
    float local_max = -INFINITY;
    
    // Loop over channels to find max
    for (int c = threadIdx.x; c < num_channels; c += blockDim.x) {
        float val = input[offset + c];
        local_max = fmaxf(local_max, val);
    }
    s_data[threadIdx.x] = local_max;
    __syncthreads();
    
    // Reduce max in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] = fmaxf(s_data[threadIdx.x], s_data[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float max_val = s_data[0];
    
    // 2. Compute exp and sum
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < num_channels; c += blockDim.x) {
        float val = expf(input[offset + c] - max_val);
        // We can reuse the input buffer if it's not needed, but here we write to intermediate_out
        // Actually, we need to store the exp values temporarily or recompute. 
        // Let's recompute to save memory bandwidth, or store in registers if possible.
        // Storing in intermediate_out directly: softmax_out
        intermediate_out[offset + c] = val;
        local_sum += val;
    }
    
    // Reduce sum in shared memory
    s_data[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_val = s_data[0];
    
    // 3. Finalize Softmax, Subtract, Swish
    for (int c = threadIdx.x; c < num_channels; c += blockDim.x) {
        float softmax_val = intermediate_out[offset + c] / sum_val;
        float sub_val = softmax_val - subtract_param[c];
        // Swish: x * sigmoid(x)
        float swish_val = sub_val * (1.0f / (1.0f + expf(-sub_val)));
        intermediate_out[offset + c] = swish_val;
    }
}

// Kernel for Max reduction across channels
__global__ void max_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_channels, int depth, int height, int width) {
    
    int d = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;

    if (d >= depth || h >= height || w >= width) return;

    int spatial_idx = (d * height + h) * width + w;
    int offset = spatial_idx * num_channels;
    
    float max_val = -INFINITY;
    for (int c = 0; c < num_channels; ++c) {
        max_val = fmaxf(max_val, input[offset + c]);
    }
    output[spatial_idx] = max_val;
}

// Host function
torch::Tensor fused_ops_cuda(
    torch::Tensor x, 
    torch::Tensor subtract_param) {
    
    // x shape: (Batch, Channel, Depth, Height, Width)
    // Output shape: (Batch, Depth, Height, Width)
    
    auto sizes = x.sizes();
    int batch_size = sizes[0];
    int num_channels = sizes[1];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];
    
    auto out_options = x.options();
    // Intermediate tensor for Softmax -> Subtract -> Swish result
    // Shape same as input: (B, C, D, H, W)
    auto intermediate_tensor = torch::empty_like(x);
    // Final output tensor: (B, D, H, W)
    auto output_tensor = torch::empty({batch_size, depth, height, width}, out_options);

    const int block_size = 256; // Max threads per block
    size_t shared_mem_size = block_size * sizeof(float);

    dim3 grid_dim(depth, height, width);
    
    // Iterate over batch
    for (int b = 0; b < batch_size; ++b) {
        auto x_b = x[b];
        auto intermediate_b = intermediate_tensor[b];
        auto output_b = output_tensor[b];
        
        // Launch Softmax/Subtract/Swish kernel
        fused_softmax_sub_swish_kernel<<<grid_dim, block_size, shared_mem_size>>>(
            x_b.data_ptr<float>(),
            subtract_param.data_ptr<float>(),
            intermediate_b.data_ptr<float>(),
            num_channels, depth, height, width
        );
        
        // Launch Max reduction kernel
        max_reduce_kernel<<<grid_dim, block_size>>>(
            intermediate_b.data_ptr<float>(),
            output_b.data_ptr<float>(),
            num_channels, depth, height, width
        );
    }
    
    return output_tensor;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor subtract_param);"
)

# Compile the inline CUDA code
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        # Keep subtract as a parameter
        self.subtract = nn.Parameter(torch.randn(out_channels))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        # Apply fused CUDA kernel for Softmax, Subtract, Swish, and Max
        x = self.fused_ops.fused_ops_cuda(x, self.subtract)
        return x