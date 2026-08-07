import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: GroupNorm + Tanh + HardSwish + Residual Add
__global__ void fused_gn_tanh_hardswish_add_kernel(
    const float* __restrict__ x_conv,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int H, int W, int G, int C_per_G, float eps) {
    
    int n = blockIdx.x;
    int c = blockIdx.y;
    int hw = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (n >= N || c >= C || hw >= H * W) return;

    int g = c / C_per_G;
    int c_in_g = c % C_per_G;
    
    // Calculate mean and variance for the group
    // Each block handles one (n, c) pair, but we need to reduce over (H, W) and channels in the group.
    // This mapping is tricky for a single kernel launch.
    // Alternative approach: Use a two-pass or more granular approach.
    // However, for a single kernel launch to handle this, we need shared memory reduction.
    
    // Let's try a different mapping:
    // Grid: (N, G, HW_blocks)
    // Block: threads for reduction and writing.
    // This is complex to implement in a single kernel without multiple launches or complex sync.
    
    // Given the constraints, let's implement a "Fused Post-Norm" kernel.
    // We assume the input x_conv is already computed.
    // We launch a kernel where each thread block computes the Group Norm for one (N, G) instance,
    // and then applies the element-wise ops.
    
    // Re-mapping for efficiency:
    // Grid: (N, G)
    // Block: 256 or 512 threads. Each thread handles multiple pixels.
    // This requires loading data, reducing in shared memory, computing norm, then writing back.
    
    // Let's stick to a simpler implementation that fuses the element-wise ops after Group Norm stats are computed.
    // But we want to fuse everything.
    
    // Optimized Kernel Strategy:
    // 1. Each block handles one (N, G) group.
    // 2. Threads in block iterate over (C_per_G, H, W) to compute sum and sum_sq.
    // 3. Compute mean and var.
    // 4. Threads iterate again to apply norm, tanh, hardswish, and add.
    
    extern __shared__ float shared_data[]; // Size: 2 * blockDim.x
    
    int group_size = C_per_G * H * W;
    int group_start_c = g * C_per_G;
    
    // Pointers
    const float* x_conv_n = x_conv + n * C * H * W;
    float* out_n = out + n * C * H * W;
    
    // Step 1: Compute Sum and Sum of Squares
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Iterate over channels in the group
    for (int c_offset = 0; c_offset < C_per_G; ++c_offset) {
        int c_idx = group_start_c + c_offset;
        // Iterate over HW
        for (int i = threadIdx.x; i < H * W; i += blockDim.x) {
            float val = x_conv_n[c_idx * H * W + i];
            sum += val;
            sum_sq += val * val;
        }
    }
    
    // Reduction in shared memory
    shared_data[threadIdx.x] = sum;
    shared_data[blockDim.x + threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
            shared_data[blockDim.x + threadIdx.x] += shared_data[blockDim.x + threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float mean = shared_data[0] / group_size;
    float var = shared_data[blockDim.x] / group_size - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    // Step 2: Apply Normalization, Weight, Bias, Tanh, HardSwish, and Add
    for (int c_offset = 0; c_offset < C_per_G; ++c_offset) {
        int c_idx = group_start_c + c_offset;
        float w = weight[c_idx];
        float b = bias[c_idx];
        
        for (int i = threadIdx.x; i < H * W; i += blockDim.x) {
            float val = x_conv_n[c_idx * H * W + i];
            
            // Group Norm
            float norm = (val - mean) * inv_std;
            float x_norm = norm * w + b;
            
            // Tanh
            float x_tanh = tanhf(x_norm);
            
            // HardSwish
            float x_hard_swish;
            if (x_tanh < -3.0f) x_hard_swish = 0.0f;
            else if (x_tanh > 3.0f) x_hard_swish = x_tanh;
            else x_hard_swish = x_tanh * (x_tanh + 3.0f) / 6.0f;
            
            // Residual Add
            out_n[c_idx * H * W + i] = val + x_hard_swish;
        }
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor x_conv, torch::Tensor weight, torch::Tensor bias, 
    int G, float eps) {
    
    auto N = x_conv.size(0);
    auto C = x_conv.size(1);
    auto H = x_conv.size(2);
    auto W = x_conv.size(3);
    auto C_per_G = C / G;
    
    auto out = torch::empty_like(x_conv);
    
    // Launch configuration
    dim3 grid(N, G);
    int threads = 256; // Power of 2 for reduction
    size_t shared_mem = 2 * threads * sizeof(float);
    
    fused_gn_tanh_hardswish_add_kernel<<<grid, threads, shared_mem>>>(
        x_conv.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W, G, C_per_G, eps
    );
    
    return out;
}

// LogSumExp Kernel
__global__ void logsumexp_kernel(const float* __restrict__ input, float* __restrict__ output, int N, int C, int HW) {
    int n = blockIdx.x;
    int hw = threadIdx.x;
    
    // Max value for numerical stability
    float max_val = -INFINITY;
    
    // Find max
    for (int c = 0; c < C; ++c) {
        float val = input[n * C * HW + c * HW + hw];
        if (val > max_val) max_val = val;
    }
    
    // Sum of exps
    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        float val = input[n * C * HW + c * HW + hw];
        sum_exp += expf(val - max_val);
    }
    
    // Log
    float result = logf(sum_exp) + max_val;
    
    // Store result (only one thread per block writes, or use atomic if needed, but here 1 thread per pixel)
    // Wait, the grid is (N, 1) and block is (HW). Each thread handles one pixel.
    if (hw < HW) {
        output[n * HW + hw] = result;
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor x) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto HW = H * W;
    
    auto out = torch::empty({N, 1, H, W}, x.options());
    
    dim3 grid(N);
    dim3 block(HW > 1024 ? 1024 : HW); // Max threads per block
    
    logsumexp_kernel<<<grid, block>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N, C, HW
    );
    
    return out;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_cuda(torch::Tensor x_conv, torch::Tensor weight, torch::Tensor bias, int G, float eps);
torch::Tensor logsumexp_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda", "logsumexp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # We still define GroupNorm to get the weights and bias, but we won't use its forward
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.groups = groups
        self.eps = eps
        self.fused_ops = fused_ops

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        
        # Fused GroupNorm + Tanh + HardSwish + Residual Add
        # We pass the weight and bias from the GroupNorm layer
        x_res = self.fused_ops.fused_ops_cuda(
            x_conv, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps
        )
        
        # LogSumExp
        x_logsumexp = self.fused_ops.logsumexp_cuda(x_res)
        
        return x_logsumexp