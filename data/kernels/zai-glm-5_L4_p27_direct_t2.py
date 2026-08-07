import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv3D + HardSwish + GroupNorm + Mean
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: Conv3D + HardSwish + GroupNorm + Mean
// Grid: (N * D_out * H_out * W_out) blocks.
// Block: 16 threads (one per output channel).
// Each block handles one spatial location (n, d, h, w).
// 1. Each thread computes Conv3D output for its channel.
// 2. Apply HardSwish.
// 3. GroupNorm using shared memory.
// 4. AtomicAdd to global output sum for Mean.

__global__ void fused_conv_gn_mean_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    const float* __restrict__ gamma, 
    const float* __restrict__ beta, 
    float* __restrict__ out_sum,
    int N, int C_in, int C_out, 
    int D_in, int H_in, int W_in,
    int K_d, int K_h, int K_w,
    int D_out, int H_out, int W_out,
    int num_groups) {
    
    // Shared memory for GroupNorm
    __shared__ float s_val[16]; // C_out = 16
    __shared__ float s_mean[4];  // num_groups = 4
    __shared__ float s_var[4];

    int spatial_size = D_out * H_out * W_out;
    int n = blockIdx.x / spatial_size;
    int rem = blockIdx.x % spatial_size;
    int d_out = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int h_out = rem / W_out;
    int w_out = rem % W_out;
    
    int c_out = threadIdx.x; // 0 to 15

    if (n >= N) return;

    // 1. Conv3D
    float val = 0.0f;
    
    // Weight shape: (C_out, C_in, K_d, K_h, K_w)
    // Input shape: (N, C_in, D_in, H_in, W_in)
    
    const float* weight_ptr = weight + c_out * (C_in * K_d * K_h * K_w);
    const float* input_base = input + n * (C_in * D_in * H_in * W_in);
    
    // Loops
    #pragma unroll
    for (int c_in = 0; c_in < 3; c_in++) {
        #pragma unroll
        for (int kd = 0; kd < 4; kd++) {
            #pragma unroll
            for (int kh = 0; kh < 4; kh++) {
                #pragma unroll
                for (int kw = 0; kw < 4; kw++) {
                    float w = weight_ptr[c_in * 64 + kd * 16 + kh * 4 + kw];
                    
                    int d_in = d_out + kd;
                    int h_in = h_out + kh;
                    int w_in = w_out + kw;
                    
                    long long in_idx = c_in * (D_in * H_in * W_in) + d_in * (H_in * W_in) + h_in * W_in + w_in;
                    float x = input_base[in_idx];
                    
                    val += x * w;
                }
            }
        }
    }
    
    // Add bias
    val += bias[c_out];
    
    // 2. HardSwish
    // x * min(max(x + 3, 0), 6) / 6
    float relu6 = fminf(fmaxf(val + 3.0f, 0.0f), 6.0f);
    val = val * relu6 / 6.0f;
    
    // Store to shared memory for GroupNorm
    s_val[c_out] = val;
    __syncthreads();
    
    // 3. GroupNorm
    // Groups = 4. Channels per group = 4.
    int g = c_out / 4; // Group index 0..3
    int lane = c_out % 4; // Lane in group 0..3
    
    // Compute mean and var for the group
    if (lane == 0) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int start_c = g * 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float v = s_val[start_c + i];
            sum += v;
            sum_sq += v * v;
        }
        float mean = sum / 4.0f;
        float var = sum_sq / 4.0f - mean * mean;
        s_mean[g] = mean;
        s_var[g] = var;
    }
    __syncthreads();
    
    // Normalize
    float mean = s_mean[g];
    float var = s_var[g];
    float inv_std = rsqrtf(var + 1e-5f);
    float x_hat = (val - mean) * inv_std;
    val = gamma[c_out] * x_hat + beta[c_out];
    
    // 4. Mean Pooling
    // Accumulate to global output
    long long out_idx = n * C_out + c_out;
    atomicAdd(&out_sum[out_idx], val);
}

__global__ void divide_kernel(float* data, int size, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] /= divisor;
    }
}

// Entry point
torch::Tensor fused_op_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta, int num_groups) {
    
    // Shapes
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int C_out = weight.size(0);
    int K_d = weight.size(2);
    int K_h = weight.size(3);
    int K_w = weight.size(4);
    
    int D_out = D_in - K_d + 1;
    int H_out = H_in - K_h + 1;
    int W_out = W_in - K_w + 1;
    
    // Output tensor for sum
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    torch::Tensor out_sum = torch::zeros({N, C_out}, options);
    
    // Launch kernel
    int spatial_size = D_out * H_out * W_out;
    int total_blocks = N * spatial_size;
    int threads = C_out; // 16
    
    fused_conv_gn_mean_kernel<<<total_blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out_sum.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        K_d, K_h, K_w,
        D_out, H_out, W_out,
        num_groups
    );
    
    // Divide by spatial size
    int output_size = N * C_out;
    int block_size_div = 256;
    int num_blocks_div = (output_size + block_size_div - 1) / block_size_div;
    
    divide_kernel<<<num_blocks_div, block_size_div>>>(
        out_sum.data_ptr<float>(), output_size, (float)spatial_size
    );
    
    return out_sum;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta, int num_groups);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        # We still define the Conv3d and GroupNorm layers to hold the parameters
        # but we will not use their forward methods.
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_ops = fused_ops

    def forward(self, x):
        # Extract parameters
        weight = self.conv.weight
        bias = self.conv.bias
        if bias is None:
            bias = torch.zeros(self.conv.out_channels, device=x.device, dtype=x.dtype)
        
        # GroupNorm parameters
        gamma = self.group_norm.weight
        beta = self.group_norm.bias
        
        # Call the fused CUDA kernel
        return self.fused_ops.fused_op_cuda(x, weight, bias, gamma, beta, self.group_norm.num_groups)