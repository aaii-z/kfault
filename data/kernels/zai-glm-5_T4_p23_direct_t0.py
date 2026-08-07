import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv3D + GroupNorm + Mean
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Block size for reduction operations
#define BLOCK_SIZE 256

// Kernel for fused Conv3D + GroupNorm + Mean
// This kernel computes the convolution, applies group normalization, and then computes the mean
// for a specific output channel 'oc' and batch element 'b'.
// It uses shared memory to reduce the partial sums for mean calculation.
__global__ void fused_conv_gn_mean_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias_conv,
    const float* __restrict__ gamma, 
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int Kd, int Kh, int Kw,
    int num_groups,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    // Each block handles one (batch, out_channel) pair
    int b = blockIdx.x; // Batch index
    int oc = blockIdx.y; // Output channel index
    
    // Calculate which group this channel belongs to
    int channels_per_group = C_out / num_groups;
    int g = oc / channels_per_group;
    
    // Thread index
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Total number of elements in the output volume for this channel
    int total_elements = D_out * H_out * W_out;
    
    // Shared memory for reduction
    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_sum_sq[BLOCK_SIZE];
    
    // Initialize accumulators for this thread
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Loop over all spatial positions in the output volume
    // Each thread processes a subset of the elements
    for (int idx = tid; idx < total_elements; idx += block_size) {
        // Convert linear index to 3D coordinates
        int od = idx / (H_out * W_out);
        int oh = (idx % (H_out * W_out)) / W_out;
        int ow = idx % W_out;
        
        // Compute convolution for this position
        float val = 0.0f;
        
        // Loop over input channels
        for (int ic = 0; ic < C_in; ++ic) {
            // Loop over kernel dimensions
            for (int kd = 0; kd < Kd; ++kd) {
                for (int kh = 0; kh < Kh; ++kh) {
                    for (int kw = 0; kw < Kw; ++kw) {
                        // Calculate input coordinates
                        int id = od * stride_d - pad_d + kd;
                        int ih = oh * stride_h - pad_h + kh;
                        int iw = ow * stride_w - pad_w + kw;
                        
                        // Check bounds
                        if (id >= 0 && id < D_in && ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                            // Input index
                            int input_idx = ((b * C_in + ic) * D_in + id) * H_in * W_in + ih * W_in + iw;
                            // Weight index (oc, ic, kd, kh, kw)
                            int weight_idx = ((oc * C_in + ic) * Kd + kd) * Kh * Kw + kh * Kw + kw;
                            val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias
        if (bias_conv != nullptr) {
            val += bias_conv[oc];
        }
        
        // Accumulate for GroupNorm statistics
        sum += val;
        sum_sq += val * val;
    }
    
    // Store in shared memory
    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 computes the mean and variance for the group
    if (tid == 0) {
        float group_sum = 0.0f;
        float group_sum_sq = 0.0f;
        
        // We need to aggregate across all channels in the group.
        // This is tricky in a single kernel. 
        // For a high-performance implementation, we would typically use a two-kernel approach:
        // 1. Compute Conv and partial sums per channel.
        // 2. Reduce across channels to get GN stats, then apply GN and Mean.
        // However, for a single kernel demonstration, we will perform a global reduction.
        // Since we are in a single block, we can't easily communicate with other blocks.
        // We will use a simpler approach: each block computes its part, and we use a separate kernel for the final reduction.
        // But to keep it fused, we will use a global memory reduction for the group stats.
        
        // For now, let's assume we are doing Instance Norm (GroupNorm with num_groups = C_out) for simplicity in a single kernel,
        // or we accept that this kernel computes partial stats.
        // Given the prompt asks for a custom CUDA operator, a full fused implementation is complex.
        // A more practical approach for a single kernel is to compute the Conv and then the Mean, 
        // but GroupNorm requires stats across multiple channels.
        
        // Let's implement a two-stage approach within the constraints.
        // Stage 1: Compute Conv output and partial sums.
        // Stage 2: Apply GN and Mean.
        // But we want it fused.
        
        // Let's use a simpler, more direct approach for the example.
        // We will compute the conv result and the mean of the conv result.
        // GroupNorm is skipped in this specific kernel for simplicity of the single-kernel fusion,
        // as doing it correctly requires inter-block communication.
        // However, the prompt asks for GroupNorm.
        
        // Let's output the partial sum and sum_sq to global memory for a second reduction.
        // But we want one kernel.
        
        // Let's try to do it all.
        // We will use atomicAdd to global memory for the group statistics.
        // This is slow but correct and fused.
        
        // Global memory pointers for group statistics
        // We need to allocate these outside.
        // Let's assume we have a workaround.
        
        // Actually, let's just compute the Conv and Mean.
        // The GroupNorm is the hard part to fuse.
        // Let's provide a kernel that does Conv + Mean.
        // And we will use PyTorch for GroupNorm.
        // No, the prompt asks to replace the operators.
        
        // Let's implement a kernel that computes the Conv, and then we will use a separate kernel for GN+Mean.
        // This is still "custom CUDA operators".
        
        // Let's output the Conv result to global memory.
        // No, that defeats the purpose.
        
        // Let's go with the atomicAdd approach for GroupNorm stats.
        // We will use global counters for each group.
        // This will be slow but demonstrates the fusion.
        
        // Wait, the prompt asks for speedups. AtomicAdd is slow.
        // The best way is to launch one kernel for Conv + Partial Stats, and a second for Final Stats + GN + Mean.
        // Let's define two kernels.
    }
}

// Kernel 1: Compute Conv3D and partial sums for GroupNorm
__global__ void conv_gn_stats_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias_conv,
    float* __restrict__ conv_out,
    float* __restrict__ partial_sum, 
    float* __restrict__ partial_sum_sq,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int Kd, int Kh, int Kw,
    int num_groups,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int spatial_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    int total_spatial = D_out * H_out * W_out;
    if (spatial_idx >= total_spatial) return;
    
    int od = spatial_idx / (H_out * W_out);
    int oh = (spatial_idx % (H_out * W_out)) / W_out;
    int ow = spatial_idx % W_out;
    
    float val = 0.0f;
    
    for (int ic = 0; ic < C_in; ++ic) {
        for (int kd = 0; kd < Kd; ++kd) {
            for (int kh = 0; kh < Kh; ++kh) {
                for (int kw = 0; kw < Kw; ++kw) {
                    int id = od * stride_d - pad_d + kd;
                    int ih = oh * stride_h - pad_h + kh;
                    int iw = ow * stride_w - pad_w + kw;
                    
                    if (id >= 0 && id < D_in && ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                        int input_idx = ((b * C_in + ic) * D_in + id) * H_in * W_in + ih * W_in + iw;
                        int weight_idx = ((oc * C_in + ic) * Kd + kd) * Kh * Kw + kh * Kw + kw;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    if (bias_conv != nullptr) {
        val += bias_conv[oc];
    }
    
    // Store conv output
    int out_idx = ((b * C_out + oc) * D_out + od) * H_out * W_out + oh * W_out + ow;
    conv_out[out_idx] = val;
    
    // Accumulate partial sums for GroupNorm
    // We accumulate per (batch, group)
    int channels_per_group = C_out / num_groups;
    int g = oc / channels_per_group;
    int stats_idx = b * num_groups + g;
    
    atomicAdd(&partial_sum[stats_idx], val);
    atomicAdd(&partial_sum_sq[stats_idx], val * val);
}

// Kernel 2: Apply GroupNorm and compute Mean
__global__ void gn_mean_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_sum_sq,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C_out, int D_out, int H_out, int W_out,
    int num_groups) {
    
    int b = blockIdx.x;
    int oc = threadIdx.x;
    
    if (oc >= C_out) return;
    
    int channels_per_group = C_out / num_groups;
    int g = oc / channels_per_group;
    int stats_idx = b * num_groups + g;
    
    // Number of elements per group
    int count = channels_per_group * D_out * H_out * W_out;
    
    // Compute mean and variance
    float mean = partial_sum[stats_idx] / count;
    float var = partial_sum_sq[stats_idx] / count - mean * mean;
    
    // Apply GroupNorm
    float scale = gamma[oc] / sqrtf(var + 1e-5f);
    float shift = beta[oc] - mean * scale;
    
    // Compute sum for the final mean
    float sum = 0.0f;
    int total_spatial = D_out * H_out * W_out;
    
    for (int i = 0; i < total_spatial; ++i) {
        int idx = ((b * C_out + oc) * D_out) * H_out * W_out + i;
        float val = conv_out[idx] * scale + shift;
        sum += val;
    }
    
    // Store partial sum for the final mean reduction
    // We need to reduce across channels as well.
    // Let's use shared memory for this.
    __shared__ float s_mean[BLOCK_SIZE]; // Assuming C_out <= BLOCK_SIZE
    
    s_mean[oc] = sum;
    __syncthreads();
    
    // Reduction
    if (oc == 0) {
        float total_sum = 0.0f;
        for (int c = 0; c < C_out; ++c) {
            total_sum += s_mean[c];
        }
        output[b] = total_sum / (C_out * D_out * H_out * W_out);
    }
}

// Helper function to launch the kernels
torch::Tensor fused_conv_gn_mean_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias_conv,
    torch::Tensor gamma, 
    torch::Tensor beta,
    int num_groups) {
    
    // Input dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    // Output dimensions
    int C_out = weight.size(0);
    int Kd = weight.size(2);
    int Kh = weight.size(3);
    int Kw = weight.size(4);
    
    // Assuming stride=1, padding=0 for simplicity, or we can pass them
    // The model uses default Conv3d which has stride=1, padding=0
    int stride_d = 1, stride_h = 1, stride_w = 1;
    int pad_d = 0, pad_h = 0, pad_w = 0;
    
    int D_out = D_in - Kd + 1;
    int H_out = H_in - Kh + 1;
    int W_out = W_in - Kw + 1;
    
    // Allocate output tensor
    auto output = torch::zeros({N}, input.options());
    auto conv_out = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());
    
    // Allocate partial sums for GroupNorm
    auto partial_sum = torch::zeros({N, num_groups}, input.options());
    auto partial_sum_sq = torch::zeros({N, num_groups}, input.options());
    
    // Launch kernel 1
    int total_spatial = D_out * H_out * W_out;
    int threads = 256;
    int blocks_spatial = (total_spatial + threads - 1) / threads;
    
    dim3 grid(N, C_out, blocks_spatial);
    
    conv_gn_stats_kernel<<<grid, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias_conv.defined() && bias_conv.numel() > 0) ? bias_conv.data_ptr<float>() : nullptr,
        conv_out.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_sum_sq.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        Kd, Kh, Kw,
        num_groups,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    );
    
    // Launch kernel 2
    // We need one block per batch
    // Threads per block = C_out
    dim3 grid2(N, 1);
    int threads2 = C_out;
    
    gn_mean_kernel<<<grid2, threads2>>>(
        conv_out.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_sum_sq.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_out, D_out, H_out, W_out,
        num_groups
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_conv_gn_mean_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias_conv, torch::Tensor gamma, torch::Tensor beta, int num_groups);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv_gn_mean_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_ops = fused_ops

    def forward(self, x):
        # Extract weights and biases
        weight = self.conv.weight
        bias_conv = self.conv.bias if self.conv.bias is not None else torch.Tensor()
        gamma = self.group_norm.weight
        beta = self.group_norm.bias
        
        # Call the custom CUDA kernel
        return self.fused_ops.fused_conv_gn_mean_cuda(x, weight, bias_conv, gamma, beta, self.group_norm.num_groups)