import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: Add, LayerNorm, AvgPool3d, GELU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Block size for reduction. W=64 in this architecture, so 64 threads per block is optimal.
#define BLOCK_SIZE 64

__device__ float gelu_approx(float x) {
    // Approximation of GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_ops_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output,
    float sum_weight,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, int D, int H, int W,
    int D_out, int H_out, int W_out) {
    
    // Shared memory for LayerNorm reduction and data sharing
    // We need to store 4 lines of data (for the 2x2 pooling in D/H) + reduction space
    extern __shared__ float smem[];
    float* s_data[4];
    // Allocate shared memory segments
    // Total shared memory: 4 * W floats
    s_data[0] = smem;
    s_data[1] = smem + W;
    s_data[2] = smem + 2 * W;
    s_data[3] = smem + 3 * W;

    // Reduction shared memory (can overlay or be separate, here separate for clarity)
    // Since we do reduction sequentially for 4 lines, we can reuse space or just use s_data for reduction
    // Let's use s_data lines for reduction sums to save shared mem if needed, 
    // but here we have enough. We will use registers for partial sums.

    int tid = threadIdx.x; // 0 to W-1

    // Grid dimensions mapping:
    // blockIdx.x: H_out
    // blockIdx.y: unused (1)
    // blockIdx.z: N * C * D_out
    
    int h_out = blockIdx.x;
    int idx = blockIdx.z;
    int n = idx / (C * D_out);
    int rem = idx % (C * D_out);
    int c = rem / D_out;
    int d_out = rem % D_out;

    int d_in_base = d_out * 2;
    int h_in_base = h_out * 2;

    // Input tensor strides
    // Input shape: (N, C, D, H, W)
    long input_stride_N = C * D * H * W;
    long input_stride_C = D * H * W;
    long input_stride_D = H * W;
    long input_stride_H = W;
    long input_stride_W = 1;

    // Base pointer for the current (n, c)
    const float* input_nc = input + n * input_stride_N + c * input_stride_C;

    // Load 4 lines corresponding to the pooling window (2x2 in D,H)
    // Lines are: (d_in_base, h_in_base), (d_in_base, h_in_base+1), 
    //            (d_in_base+1, h_in_base), (d_in_base+1, h_in_base+1)
    
    // We use float4 for coalesced loads if W is multiple of 4. W=64 is multiple of 4.
    // However, with 64 threads, each thread loads 1 float.
    // Coalescing is perfect: Thread i loads address Base + i.

    // Line 0
    float val0 = input_nc[(d_in_base) * input_stride_D + (h_in_base) * input_stride_H + tid] + sum_weight;
    s_data[0][tid] = val0;

    // Line 1
    float val1 = input_nc[(d_in_base) * input_stride_D + (h_in_base + 1) * input_stride_H + tid] + sum_weight;
    s_data[1][tid] = val1;

    // Line 2
    float val2 = input_nc[(d_in_base + 1) * input_stride_D + (h_in_base) * input_stride_H + tid] + sum_weight;
    s_data[2][tid] = val2;

    // Line 3
    float val3 = input_nc[(d_in_base + 1) * input_stride_D + (h_in_base + 1) * input_stride_H + tid] + sum_weight;
    s_data[3][tid] = val3;

    __syncthreads();

    // LayerNorm for each line
    // We process each line sequentially to reuse the same threads for reduction
    // W=64, BLOCK_SIZE=64. We perform warp reduction.
    // Since 64 threads is 2 warps, we need inter-warp reduction via shared mem.

    float mean, var;
    float inv_std;

    // Helper lambda for reduction (conceptually)
    // We will manually unroll or write a helper block reduction
    
    // Process Line k
    for (int k = 0; k < 4; ++k) {
        float val = s_data[k][tid];
        
        // Block reduction for sum
        // Warp reduce
        for (int offset = 32; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        // Only first lane of second warp writes to shared mem? No.
        // We have 2 warps. Warp 0 reduces to lane 0. Warp 1 reduces to lane 0.
        // Lane 0 of each warp writes to shared mem.
        
        __shared__ float s_warp_sums[2]; 
        if (tid < 32) { // Lane 0 of each warp (tid 0 and tid 32)
             if ((tid & 31) == 0) {
                 s_warp_sums[tid / 32] = val;
             }
        }
        __syncthreads();
        
        // Combine warp sums (only thread 0 does this)
        if (tid == 0) {
            float total_sum = s_warp_sums[0] + s_warp_sums[1];
            mean = total_sum / W;
            s_warp_sums[0] = mean; // Reuse to broadcast
        }
        __syncthreads();
        mean = s_warp_sums[0]; // Broadcast mean

        // Calculate variance
        float diff = s_data[k][tid] - mean;
        float diff_sq = diff * diff;
        
        // Reduce diff_sq
        for (int offset = 32; offset > 0; offset /= 2) {
            diff_sq += __shfl_down_sync(0xffffffff, diff_sq, offset);
        }
        
        if (tid < 32) {
             if ((tid & 31) == 0) {
                 s_warp_sums[tid / 32] = diff_sq;
             }
        }
        __syncthreads();

        if (tid == 0) {
            float total_var = s_warp_sums[0] + s_warp_sums[1];
            var = total_var / W;
            inv_std = rsqrtf(var + 1e-5f);
            s_warp_sums[0] = inv_std;
        }
        __syncthreads();
        inv_std = s_warp_sums[0];

        // Apply LayerNorm
        // y = (x - mean) / std * gamma + beta
        s_data[k][tid] = (s_data[k][tid] - mean) * inv_std * gamma[tid] + beta[tid];
    }
    
    __syncthreads();

    // AvgPool3d (2x2x2) + GELU
    // Output size: W_out = W/2 = 32
    // Each thread computes one output element for W_out
    // We have 64 threads, so each thread can compute 2 output elements? 
    // Or we just have 32 threads active?
    // Let's make 32 threads active (tid < 32).
    
    if (tid < W_out) {
        float sum_pool = 0.0f;
        // Pool over W (2 elements)
        // Pool over H (2 lines: k=0,1 vs k=2,3? No, lines are D,H combinations)
        // Lines: 0:(d0,h0), 1:(d0,h1), 2:(d1,h0), 3:(d1,h1)
        // We need to pool over 2x2x2 block.
        // Indices in W: 2*tid, 2*tid+1
        // Indices in Lines: 0, 1, 2, 3
        
        sum_pool += s_data[0][2*tid] + s_data[0][2*tid+1];
        sum_pool += s_data[1][2*tid] + s_data[1][2*tid+1];
        sum_pool += s_data[2][2*tid] + s_data[2][2*tid+1];
        sum_pool += s_data[3][2*tid] + s_data[3][2*tid+1];
        
        float avg = sum_pool / 8.0f;
        float out_val = gelu_approx(avg);

        // Write output
        // Output shape: (N, C, D_out, H_out, W_out)
        // Strides for output
        long output_stride_N = C * D_out * H_out * W_out;
        long output_stride_C = D_out * H_out * W_out;
        long output_stride_D = H_out * W_out;
        long output_stride_H = W_out;
        long output_stride_W = 1;

        output[n * output_stride_N + c * output_stride_C + d_out * output_stride_D + h_out * output_stride_H + tid] = out_val;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input, 
    float sum_weight, 
    torch::Tensor gamma, 
    torch::Tensor beta,
    int D_out, int H_out, int W_out) {
    
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];

    auto output = torch::zeros({N, C, D_out, H_out, W_out}, input.options());

    // Grid configuration
    // blockIdx.x: H_out
    // blockIdx.z: N * C * D_out
    dim3 grid(H_out, 1, N * C * D_out);
    dim3 block(BLOCK_SIZE);
    
    // Shared memory size: 4 lines of W floats
    size_t shared_mem_size = 4 * W * sizeof(float);

    fused_ops_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        sum_weight,
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N, C, D, H, W,
        D_out, H_out, W_out
    );

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor input, float sum_weight, torch::Tensor gamma, torch::Tensor beta, int D_out, int H_out, int W_out);
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape) # Keep this to hold gamma/beta parameters
        self.pool_kernel_size = pool_kernel_size
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Calculate output dimensions for pooling
        # Assuming pool_kernel_size is (2,2,2) and stride is same as kernel_size (default for AvgPool)
        # D_out = floor(D_in / 2), etc.
        # But we need to pass these to CUDA kernel
        D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
        D_out = D_in // self.pool_kernel_size[0]
        H_out = H_in // self.pool_kernel_size[1]
        W_out = W_in // self.pool_kernel_size[2]
        
        # Call fused kernel
        # Note: self.norm.weight and self.norm.bias are gamma and beta
        x = self.fused_ops.fused_op_cuda(
            x, 
            self.sum_weight.item(), 
            self.norm.weight, 
            self.norm.bias, 
            D_out, H_out, W_out
        )
        return x