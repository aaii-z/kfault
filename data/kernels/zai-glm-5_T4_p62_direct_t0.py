import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + GroupNorm + LeakyReLU + Add
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Fused kernel: GEMM + GroupNorm + LeakyReLU + Add
// Assumes inputs are contiguous and shapes match the problem description.
// FP32 precision.

__global__ void fused_gemm_gn_lrelu_add_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size, 
    int input_size, 
    int hidden_size, 
    int num_groups, 
    int group_size, 
    float eps, 
    float negative_slope) {
    
    // 1. Matrix Multiplication (GEMM)
    // Each block computes a tile of the output matrix.
    // We use block size (256, 1, 1) and grid (hidden_size, batch_size)
    // blockIdx.x = hidden_dim_index, blockIdx.y = batch_index
    
    int row = blockIdx.y;
    int col = blockIdx.x;
    
    // Bounds check
    if (row >= batch_size || col >= hidden_size) return;

    // Compute dot product
    float val = 0.0f;
    const float* input_row = input + row * input_size;
    const float* weight_col = weight + col;
    
    // Unrolled or vectorized load could be used here, but keeping it simple for correctness
    // with potential loop unrolling by compiler
    for (int k = 0; k < input_size; ++k) {
        val += input_row[k] * weight_col[k * hidden_size];
    }
    
    // Add bias
    if (bias != nullptr) {
        val += bias[col];
    }

    // 2. Group Normalization
    // We need mean and variance of the group that 'col' belongs to.
    // Since we computed one element, we need to compute stats for the whole group.
    // This is inefficient if done naively per thread.
    // Strategy: 
    // We can't easily share memory between blocks without a second kernel launch.
    // However, we can optimize by having the first thread in the group compute stats?
    // No, blocks are independent.
    // Alternative: Use cooperative groups or just recompute?
    // Given the constraints, we perform a two-pass approach within the kernel logic 
    // or assume the user accepts the overhead of computing stats per output element 
    // if we were to do it naively (which is wrong).
    // Correct approach for a single kernel without shared memory cooperation across rows:
    // We can't efficiently compute GN across the hidden dimension in a single kernel launch 
    // configured as (hidden_size, batch_size) without reduction.
    // 
    // Let's re-evaluate the grid. 
    // If we launch (batch_size, num_groups) blocks, each block computes one group.
    // Inside the block, threads compute the GEMM for that group, then cooperate to find mean/var.
    
    // Re-designing for (batch_size, num_groups) grid:
    // blockIdx.y = batch_index (0 to batch_size-1)
    // blockIdx.x = group_index (0 to num_groups-1)
    // Block size = 256.
    
    // This requires re-writing the kernel logic.
    // Let's stick to the previous grid idea but use a reduction.
    // Actually, the prompt implies replacing operators. 
    // A standard optimization is fusing the activation.
    // Fusing GN + LeakyReLU + Add is easy.
    // Fusing GEMM + GN is hard because GN requires global reduction over hidden_size.
    // 
    // However, we can implement a "Welford" style reduction or a two-stage kernel.
    // But we are limited to "custom CUDA operators". We can define a single operator.
    // Let's try to implement the GEMM + GN fusion using block-level stats.
    
    // Grid: (batch_size, num_groups)
    // Block: (group_size / 1) -> limited by 1024. group_size = 16 here.
    // We have group_size = 16. This is small.
    // We can launch blocks of size 16 or 32.
    
    // Let's use the previous grid (batch, hidden) but we need to synchronize.
    // Since we can't synchronize across blocks easily, we must compute stats within a block.
    // This implies one block computes one group.
    
    // Correct Kernel Implementation:
    // Grid: dim3(batch_size, num_groups)
    // Block: dim3(group_size) -> 16 threads per block.
    // Each thread computes one element of the group.
    
    // Mapping:
    // row = blockIdx.y
    // group_idx = blockIdx.x
    // tid = threadIdx.x
    
    // This mapping is valid if group_size <= 1024. Here group_size=16.
    
    // Wait, the previous grid was (batch, hidden). That was for GEMM.
    // If we change grid to (batch, num_groups), we need to compute GEMM results 
    // inside the kernel.
    
    // Let's proceed with Grid (batch, num_groups), Block (group_size).
    // Each thread computes the GEMM for one output element.
    // Then threads in the block cooperate to compute mean/var.
    // Then apply GN, LeakyReLU, and Add.
    
    // Re-defining variables for this logic:
    int b_idx = blockIdx.y;
    int g_idx = blockIdx.x;
    int lane_id = threadIdx.x; // 0 to group_size-1
    
    if (lane_id >= group_size) return; // Should not happen if block dim matches
    
    int h_idx = g_idx * group_size + lane_id; // The specific column in hidden_size
    
    // 1. GEMM for this element
    float gemm_out = 0.0f;
    const float* input_ptr = input + b_idx * input_size;
    const float* weight_ptr = weight + h_idx; // Column major access for weight
    
    // Dot product
    // input_size = 8192. Loop is long.
    // We should vectorize or unroll.
    // Since we are inside a small block (16 threads), memory access might be uncoalesced if we iterate k.
    // Input is row-major. Weight is... PyTorch Linear weight is (out_features, in_features).
    // So weight[h_idx][k] is weight[h_idx * input_size + k].
    // Access pattern: input[b][k] (coalesced across threads if threads map to different h_idx)
    // weight[h_idx][k].
    // Threads in warp access different h_idx, same k.
    // input[b][k] is broadcast/read-only.
    // weight[h_idx][k] is strided? No, weight is (hidden, input).
    // weight[h_idx][k] -> weight + h_idx * input_size + k.
    // Threads 0..15 access k, h_idx=0..15.
    // Addresses: (0*in+k), (1*in+k), ...
    // This is strided access. Bad for global memory.
    // However, weight is likely cached.
    // Also we can't change the data layout.
    
    // Optimization: Loop unrolling
    #pragma unroll 4
    for (int k = 0; k < input_size; k++) {
        gemm_out += input_ptr[k] * weight[h_idx * input_size + k];
    }
    
    if (bias != nullptr) {
        gemm_out += bias[h_idx];
    }
    
    // 2. Group Normalization Stats
    // We need mean and var of `gemm_out` across the group (threads in the block).
    // We use shared memory for reduction.
    
    __shared__ float s_mean;
    __shared__ float s_var;
    
    // Local copies for reduction
    float val_for_stats = gemm_out;
    
    // Reduction for sum and sum_sq
    // Using standard parallel reduction in a block
    // We need to handle arbitrary group_size (power of 2 usually, here 16).
    
    // Block size is group_size. We assume block dimension x is power of 2 >= group_size.
    // We can use shuffle instructions (warp level) since group_size <= 32 (warp size).
    
    float sum = val_for_stats;
    float sum_sq = val_for_stats * val_for_stats;
    
    // Warp reduce
    #pragma unroll
    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        // We can use __shfl_down_sync
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }
    
    // Thread 0 has the sums
    if (lane_id == 0) {
        float mean = sum / group_size;
        float var = sum_sq / group_size - mean * mean;
        s_mean = mean;
        s_var = var;
    }
    
    // Sync to broadcast
    __syncthreads();
    
    // 3. Normalize, LeakyReLU, Add
    float mean = s_mean;
    float var = s_var;
    
    float normalized = (gemm_out - mean) / sqrtf(var + eps);
    
    // Apply gamma/beta
    float gn_out = normalized * gamma[h_idx] + beta[h_idx];
    
    // LeakyReLU
    float lrelu_out = gn_out > 0 ? gn_out : gn_out * negative_slope;
    
    // Add
    float final_out = lrelu_out + lrelu_out;
    
    // Write output
    output[b_idx * hidden_size + h_idx] = final_out;
}

torch::Tensor fused_op_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    torch::Tensor gamma, 
    torch::Tensor beta, 
    int num_groups, 
    float eps, 
    float negative_slope) {
    
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);
    int group_size = hidden_size / num_groups;
    
    auto output = torch::empty({batch_size, hidden_size}, input.options());
    
    // Grid and Block
    dim3 grid(num_groups, batch_size);
    dim3 block(group_size);
    
    // Launch kernel
    fused_gemm_gn_lrelu_add_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        num_groups,
        group_size,
        eps,
        negative_slope
    );
    
    return output;
}
"""

fused_kernel_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps, float negative_slope);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        # We need to maintain the weights and parameters to match the original model's behavior
        # but we will bypass them in the forward pass.
        
        # We create a Linear layer just to initialize the weights (weight and bias)
        self.fc = nn.Linear(input_size, hidden_size)
        # We create a GroupNorm layer just to initialize the weights (gamma and beta)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        self.fused_op = fused_op

    def forward(self, x):
        # Call the custom fused CUDA kernel
        # We pass the weights directly from the PyTorch parameters
        return self.fused_op.fused_op_cuda(
            x, 
            self.fc.weight, 
            self.fc.bias, 
            self.gn.weight, 
            self.gn.bias, 
            self.num_groups, 
            self.eps, 
            self.negative_slope
        )