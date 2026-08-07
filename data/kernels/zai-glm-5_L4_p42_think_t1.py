import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Spatial Sum Reduction
// Reduces (N, C, H, W) to (N, C) by summing over H and W
__global__ void spatial_sum_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int n = blockIdx.y;
    int c = blockIdx.x;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Calculate the starting index for this (n, c)
    int base_idx = (n * C + c) * H * W;
    
    float sum = 0.0f;
    
    // Grid stride loop to sum all elements in the spatial dimensions
    for (int i = tid; i < H * W; i += block_size) {
        sum += input[base_idx + i];
    }
    
    // Block reduction using shared memory
    __shared__ float s_data[256];
    s_data[tid] = sum;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this (n, c)
    if (tid == 0) {
        output[n * C + c] = s_data[0];
    }
}

// Kernel 2: Fused Post-Processing
// Fuses: MatMul (via dot product), Bias Add, Average Pooling (scaling), LogSumExp, Sum, Mul
// Input sum_x: (N, C_in)
// Input sum_w: (C_in, C_out)
// Input bias: (C_out) - Combined bias from ConvTranspose and the separate bias layer
// Output: (N)
__global__ void fused_post_kernel(const float* sum_x, const float* sum_w, const float* bias, float* out, int N, int C_in, int C_out, float area_inv) {
    int n = blockIdx.x; // One block per sample in batch
    int tid = threadIdx.x;
    
    // Dynamic shared memory allocation
    // Layout: [sum_x (C_in)] [y_vals (C_out)] [reduction_buf (blockDim.x)]
    extern __shared__ float s_mem[];
    float* s_sum_x = s_mem;
    float* s_y = s_mem + C_in;
    float* s_buf = s_y + C_out; // Used for max reduction and sum reduction
    
    // 1. Load sum_x[n, :] into shared memory
    for (int i = tid; i < C_in; i += blockDim.x) {
        s_sum_x[i] = sum_x[n * C_in + i];
    }
    __syncthreads(); // Ensure sum_x is loaded before usage
    
    // 2. Compute MatMul result, apply scaling (avg pool) and bias
    // Each thread computes a subset of the C_out channels
    // y[c] = (sum_x . sum_w[:, c]) * area_inv + bias[c]
    for (int c = tid; c < C_out; c += blockDim.x) {
        float dot_prod = 0.0f;
        for (int k = 0; k < C_in; ++k) {
            // sum_w is (C_in, C_out)
            dot_prod += s_sum_x[k] * sum_w[k * C_out + c];
        }
        float val = dot_prod * area_inv + bias[c];
        s_y[c] = val;
    }
    __syncthreads();
    
    // 3. LogSumExp over channels (dim 1)
    // First, find max value in s_y
    float local_max = -INFINITY;
    for (int c = tid; c < C_out; c += blockDim.x) {
        local_max = fmaxf(local_max, s_y[c]);
    }
    s_buf[tid] = local_max;
    __syncthreads();
    
    // Block reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_buf[tid] = fmaxf(s_buf[tid], s_buf[tid + s]);
        }
        __syncthreads();
    }
    float global_max = s_buf[0];
    __syncthreads();
    
    // Second, compute sum(exp(y - max))
    float local_sum = 0.0f;
    for (int c = tid; c < C_out; c += blockDim.x) {
        local_sum += expf(s_y[c] - global_max);
    }
    s_buf[tid] = local_sum;
    __syncthreads();
    
    // Block reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_buf[tid] += s_buf[tid + s];
        }
        __syncthreads();
    }
    
    // 4. Final result: log(sum) + max, then multiply by 10
    if (tid == 0) {
        float lse = logf(s_buf[0]) + global_max;
        out[n] = lse * 10.0f;
    }
}

// C++ interface
torch::Tensor custom_forward_cuda(
    torch::Tensor x, 
    torch::Tensor weight_sum, 
    torch::Tensor bias, 
    int64_t output_area
) {
    // x shape: (N, C_in, H, W)
    // weight_sum shape: (C_in, C_out)
    // bias shape: (C_out)
    
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int C_out = weight_sum.size(1);
    
    // Output for spatial sum: (N, C_in)
    auto sum_x_opts = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor sum_x = torch::empty({N, C_in}, sum_x_opts);
    
    // Output for final result: (N)
    torch::Tensor out = torch::empty({N}, sum_x_opts);
    
    // Launch Kernel 1: Spatial Sum
    // Grid: (C_in, N), Block: 256
    dim3 grid1(C_in, N);
    dim3 block1(256);
    spatial_sum_kernel<<<grid1, block1>>>(
        x.data_ptr<float>(), 
        sum_x.data_ptr<float>(), 
        N, C_in, H, W
    );
    
    // Launch Kernel 2: Fused Post-Processing
    // Grid: N, Block: 256
    // Shared memory: C_in + C_out + 256 floats
    size_t shared_mem_size = (C_in + C_out + 256) * sizeof(float);
    dim3 grid2(N);
    dim3 block2(256);
    
    float area_inv = 1.0f / static_cast<float>(output_area);
    
    fused_post_kernel<<<grid2, block2, shared_mem_size>>>(
        sum_x.data_ptr<float>(),
        weight_sum.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, C_out, area_inv
    );
    
    return out.view({N, 1});
}
"""

cpp_source = """
torch::Tensor custom_forward_cuda(torch::Tensor x, torch::Tensor weight_sum, torch::Tensor bias, int64_t output_area);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["custom_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # Keep the original layers to maintain weight shapes and initialization
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Calculate output spatial dimensions
        # H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        # Assuming default stride=1, padding=0 for ConvTranspose2d
        _, _, H, W = x.shape
        kH, kW = self.conv_transpose.kernel_size
        H_out = H + kH - 1
        W_out = W + kW - 1
        output_area = H_out * W_out
        
        # Compute weight_sum: sum of weights over spatial dimensions
        # Weight shape: (C_in, C_out, kH, kW) -> (C_in, C_out)
        weight_sum = self.conv_transpose.weight.sum(dim=(2, 3))
        
        # Combine biases
        # self.conv_transpose.bias is (C_out)
        # self.bias is (C_out, 1, 1) -> squeeze to (C_out)
        bias_combined = self.conv_transpose.bias + self.bias.squeeze()
        
        # Call the custom CUDA kernel
        return custom_ops.custom_forward_cuda(x, weight_sum, bias_combined, output_area)