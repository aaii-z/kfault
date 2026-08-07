import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Swish, Bias, and GroupNorm
fused_swish_bias_groupnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Device helper for min to avoid std::min issues in CUDA kernel
__device__ __host__ int my_min(int a, int b) {
    return a < b ? a : b;
}

// Kernel 1: Compute partial sums (sum and sum of squares) for each group chunk
// This kernel fuses: Input loading -> Swish -> Bias Add -> Partial Reduction
__global__ void compute_stats_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_sum_sq,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int chunk_size) {

    const int g = blockIdx.x;
    const int chunk_idx = blockIdx.y;
    
    const int start_row = chunk_idx * chunk_size;
    const int end_row = my_min(start_row + chunk_size, batch_size);
    
    if (start_row >= batch_size) return;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    __shared__ float s_sum[256]; // Assuming block size 256
    __shared__ float s_sum_sq[256];

    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;

    // Iterate over rows assigned to this block
    for (int r = start_row; r < end_row; ++r) {
        // Iterate over channels within the group
        for (int c = tid; c < channels_per_group; c += block_threads) {
            const int global_ch = g * channels_per_group + c;
            const int idx = r * num_channels + global_ch;
            
            // Load
            float val = input[idx];
            
            // Swish: x * sigmoid(x)
            val = val / (1.0f + expf(-val));
            
            // Bias Add
            val += bias[global_ch];
            
            // Accumulate
            sum += val;
            sum_sq += val * val;
        }
    }

    // Block reduction
    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();

    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    // Write partial results to global memory
    if (tid == 0) {
        const int out_idx = g * gridDim.y + chunk_idx;
        partial_sum[out_idx] = s_sum[0];
        partial_sum_sq[out_idx] = s_sum_sq[0];
    }
}

// Kernel 2: Finalize stats and Normalize
// Reads partial sums, computes mean/var, then normalizes the data
__global__ void normalize_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    const float* __restrict__ gamma, // GroupNorm weight
    const float* __restrict__ beta,  // GroupNorm bias
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_sum_sq,
    float* __restrict__ output,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int chunk_size,
    const float eps) {

    const int g = blockIdx.x;
    const int chunk_idx = blockIdx.y;

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    
    // 1. Aggregate partial sums for this group
    // Only thread 0 performs the aggregation to avoid redundant memory traffic
    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        float total_sum_sq = 0.0f;
        for (int i = 0; i < gridDim.y; ++i) {
            total_sum += partial_sum[g * gridDim.y + i];
            total_sum_sq += partial_sum_sq[g * gridDim.y + i];
        }
        
        const int total_elements = batch_size * channels_per_group;
        const float mean = total_sum / total_elements;
        const float var = total_sum_sq / total_elements - mean * mean;
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();
    
    const float mean = s_mean;
    const float inv_std = s_inv_std;

    // 2. Normalize
    const int start_row = chunk_idx * chunk_size;
    const int end_row = my_min(start_row + chunk_size, batch_size);
    
    if (start_row >= batch_size) return;

    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;

    for (int r = start_row; r < end_row; ++r) {
        for (int c = tid; c < channels_per_group; c += block_threads) {
            const int global_ch = g * channels_per_group + c;
            const int idx = r * num_channels + global_ch;
            
            float val = input[idx];
            
            // Fused Swish
            val = val / (1.0f + expf(-val));
            
            // Fused Bias
            val += bias[global_ch];
            
            // Normalize
            val = (val - mean) * inv_std;
            
            // Scale and Shift (GroupNorm)
            val = val * gamma[global_ch] + beta[global_ch];
            
            output[idx] = val;
        }
    }
}

// C++ interface
torch::Tensor fused_swish_bias_groupnorm_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    double eps) {

    const int batch_size = input.size(0);
    const int num_channels = input.size(1);
    const int channels_per_group = num_channels / num_groups;

    auto output = torch::empty_like(input);
    
    // Tuning parameter: number of chunks to split the Batch dimension into
    // We want enough blocks to saturate the GPU.
    const int target_blocks = 4096;
    const int num_chunks = (target_blocks + num_groups - 1) / num_groups;
    
    // Allocate temporary buffers for partial sums
    // Shape: (num_groups, num_chunks)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto partial_sum = torch::empty({num_groups, num_chunks}, options);
    auto partial_sum_sq = torch::empty({num_groups, num_chunks}, options);

    // Calculate chunk size for the kernels
    const int chunk_size = (batch_size + num_chunks - 1) / num_chunks;

    // Launch Kernel 1: Compute Stats
    const int threads = 256;
    dim3 blocks(num_groups, num_chunks);
    
    compute_stats_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_sum_sq.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        channels_per_group,
        chunk_size
    );

    // Launch Kernel 2: Normalize
    normalize_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_sum_sq.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        channels_per_group,
        chunk_size,
        static_cast<float>(eps)
    );

    return output;
}
"""

fused_swish_bias_groupnorm_cpp_source = """
torch::Tensor fused_swish_bias_groupnorm_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    double eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_swish_bias_groupnorm",
    cpp_sources=fused_swish_bias_groupnorm_cpp_source,
    cuda_sources=fused_swish_bias_groupnorm_source,
    functions=["fused_swish_bias_groupnorm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused Swish, Bias, and GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        # Keep the Linear layer as is, cuBLAS is highly optimized.
        self.matmul = nn.Linear(in_features, out_features)
        
        # Custom bias parameter
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # GroupNorm layer to maintain state_dict compatibility and learnable parameters
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        
        # Load the custom op
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Linear (MatMul + Bias)
        x = self.matmul(x)
        
        # 2. Fused Swish + Bias + GroupNorm
        # We use the parameters from the self.group_norm module to ensure correctness
        # with loading weights and maintaining learnable parameters.
        x = self.fused_ops.fused_swish_bias_groupnorm_cuda(
            x, 
            self.bias, 
            self.group_norm.weight