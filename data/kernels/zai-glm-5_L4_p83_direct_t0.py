import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: GroupNorm + Min + Clamp + Dropout
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Fused kernel for GroupNorm, Min, Clamp, and Dropout
__global__ void fused_gn_min_clamp_dropout_kernel(
    const float* __restrict__ x, 
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int D, int H, int W, int groups,
    float min_val, float max_val, float dropout_p, bool training) {
    
    // Calculate dimensions
    int total_spatial = D * H * W;
    int total_elements = N * C * total_spatial;
    
    // GroupNorm parameters
    int channels_per_group = C / groups;
    int group_size = channels_per_group * total_spatial;
    
    // Indices
    int nc = blockIdx.x; // N*C blocks
    int c = nc % C;
    int g = c / channels_per_group;
    int n = nc / C;
    
    // Pointers for the current N, C slice
    const float* x_nc = x + nc * total_spatial;
    float* out_nc = out + nc * total_spatial;
    
    // 1. Calculate Mean and Variance for the group (using parallel reduction would be better, 
    // but for simplicity and to avoid complex block synchronization in a single kernel, 
    // we use a standard approach where threads cooperate or we iterate.
    // Given the complexity of a full optimized GN in a fused kernel, we will use a simpler 
    // per-thread accumulation if group_size is small, or shared memory reduction.
    // For this implementation, we assume a block processes one (N, Group) instance.
    
    // However, to keep the kernel launch simple (N*C blocks), we compute stats per channel 
    // and combine? No, GN stats are across channels in a group.
    // A full high-performance GN requires careful reduction. 
    // Here we implement a functional version where threads collaborate.
    
    // Let's map threads to spatial locations.
    int tid = threadIdx.x;
    int spatial_idx = tid;
    
    // Shared memory for reduction
    extern __shared__ float s_data[]; // Size: 2 * blockDim.x
    float* s_sum = s_data;
    float* s_sum_sq = &s_data[blockDim.x];
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Iterate over channels in the group and spatial locations
    // This mapping is tricky. 
    // Alternative: One block per (N, Group). Grid: N * groups.
    // Let's stick to the launch config: N*C blocks, 1D thread block for spatial dims.
    // We need to sync across channels in the group. This requires grid sync or 
    // launching blocks differently.
    
    // Re-designing for performance and correctness:
    // Launch N * groups blocks. Each block handles one group instance.
    // Block size covers the spatial dimensions, iterating if necessary.
    // This allows easy reduction within the group.
    
    // BUT, the launch config in the code below uses N*C.
    // Let's stick to the provided structure but optimize the logic.
    // If we launch N*C blocks, each block handles one channel.
    // We cannot efficiently compute GN stats (which span channels) without global sync.
    // 
    // CORRECTION: To fuse this correctly, we must launch N * groups blocks.
    // Let's assume the launch config is changed to N * groups.
    // If we are forced to use N*C blocks, we can't fuse GN efficiently.
    // I will implement the kernel assuming N*groups launch, and adjust the C++ caller.
    
    // Wait, the prompt implies I write the kernel. I can choose the launch config.
    
    // Let's implement the kernel assuming:
    // Grid: (N * groups)
    // Block: covers spatial dimensions.
    
    // Re-mapping indices based on N * groups launch:
    int ng = blockIdx.x; 
    n = ng / groups;
    g = ng % groups;
    
    int c_start = g * channels_per_group;
    int c_end = c_start + channels_per_group;
    
    // Thread mapping: 1D block covering spatial locations
    // Each thread iterates over channels in the group and spatial locations
    // to compute sum and sum_sq.
    
    float mean = 0.0f;
    float var = 0.0f;
    
    // Reduction for Mean and Var
    for (int i = tid; i < group_size; i += blockDim.x) {
        int c_local = i / total_spatial;
        int s_local = i % total_spatial;
        int c_abs = c_start + c_local;
        
        float val = x[n * C * total_spatial + c_abs * total_spatial + s_local];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    // Reduce in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float sum = s_sum[0];
        float sum_sq = s_sum_sq[0];
        mean = sum / group_size;
        var = sum_sq / group_size - mean * mean;
        // Store to shared memory for all threads to see
        s_sum[0] = mean;
        s_sum[1] = var;
    }
    __syncthreads();
    
    mean = s_sum[0];
    var = s_sum[1];
    
    // 2. Normalize, Affine, Min, Clamp, Dropout
    float inv_std = rsqrtf(var + 1e-5f);
    
    // Setup random state for dropout
    curandState state;
    if (training && dropout_p > 0.0f) {
        // Simple seeding
        curand_init(clock64(), blockIdx.x, tid, &state);
    }
    
    // Iterate over all elements in this group instance
    for (int i = tid; i < group_size; i += blockDim.x) {
        int c_local = i / total_spatial;
        int s_local = i % total_spatial;
        int c_abs = c_start + c_local;
        
        int idx = n * C * total_spatial + c_abs * total_spatial + s_local;
        
        float val = x[idx];
        
        // Norm
        val = (val - mean) * inv_std;
        // Affine
        val = val * weight[c_abs] + bias[c_abs];
        
        // Min
        val = fminf(val, min_val);
        
        // Clamp
        val = fmaxf(min_val, fminf(val, max_val));
        
        // Dropout
        if (training && dropout_p > 0.0f) {
            float rand = curand_uniform(&state);
            if (rand < dropout_p) {
                val = 0.0f;
            } else {
                val *= 1.0f / (1.0f - dropout_p);
            }
        }
        
        out[idx] = val;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor x, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int groups, 
    float min_val, 
    float max_val, 
    float dropout_p, 
    bool training) {
    
    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    auto out = torch::empty_like(x);
    
    int total_spatial = D * H * W;
    int channels_per_group = C / groups;
    int group_size = channels_per_group * total_spatial;
    
    // Launch N * groups blocks
    int blocks = N * groups;
    int threads = 256; // Optimize for typical spatial size
    // Shared memory for reduction (sum, sum_sq)
    size_t shared_mem = 2 * threads * sizeof(float);
    
    fused_gn_min_clamp_dropout_kernel<<<blocks, threads, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, D, H, W, groups,
        min_val, max_val, dropout_p, training
    );
    
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int groups, float min_val, float max_val, float dropout_p, bool training);"
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
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # We keep the GroupNorm module to hold the weight and bias parameters
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout_p = dropout_p
        self.min_value = min_value
        self.max_value = max_value
        self.groups = groups
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Convolution (using PyTorch's optimized implementation)
        x = self.conv(x)
        
        # 2. Fused GroupNorm + Min + Clamp + Dropout
        # Pass the weight and bias from the GroupNorm module
        x = self.fused_ops.fused_ops_cuda(
            x, 
            self.norm.weight, 
            self.norm.bias, 
            self.groups, 
            self.min_value, 
            self.max_value, 
            self.dropout_p, 
            self.training
        )
        
        return x