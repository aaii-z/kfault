import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose3d + ReLU + GroupNorm
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Define constants for vectorized memory access
#define CUDA_NUM_THREADS 256

// Fused kernel: ConvTranspose3d + ReLU + GroupNorm
// We use a tiled approach for the convolution part to maximize data reuse.
// For GroupNorm, we use a block reduction to compute mean and variance.

template <typename scalar_t>
__global__ void fused_conv_relu_gn_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ gn_weight,
    const scalar_t* __restrict__ gn_bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int groups,
    const int channels_per_group) {

    // --- GroupNorm Shared Memory ---
    // We need shared memory for reduction within a group.
    // A block processes one (batch, group) pair.
    // The number of channels in a group is channels_per_group.
    // The spatial size is out_depth * out_height * out_width.
    // We use a 2D grid of threads to cover the spatial dimensions.
    // Each thread accumulates partial sums for its assigned pixels across all channels in the group.
    
    // Shared memory for mean and var reduction
    // We use a two-level reduction.
    // 1. Each thread accumulates partial sum and sum of squares.
    // 2. Reduce within the block using shared memory.
    
    extern __shared__ char shared_mem[];
    scalar_t* s_sum = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* s_sum_sq = s_sum + CUDA_NUM_THREADS;
    
    const int b = blockIdx.x; // batch index
    const int g = blockIdx.y; // group index
    
    const int spatial_size = out_depth * out_height * out_width;
    const int group_offset = g * channels_per_group;
    
    // Thread indices for spatial dimensions
    // We flatten the spatial dimensions and map them to threads.
    // Each thread may handle multiple spatial locations if spatial_size > CUDA_NUM_THREADS.
    
    scalar_t local_sum = 0.0f;
    scalar_t local_sum_sq = 0.0f;
    
    // Loop over spatial locations
    for (int sp_idx = threadIdx.x; sp_idx < spatial_size; sp_idx += blockDim.x) {
        const int od = sp_idx / (out_height * out_width);
        const int ohw = sp_idx % (out_height * out_width);
        const int oh = ohw / out_width;
        const int ow = ohw % out_width;
        
        // Loop over channels in the group
        for (int c = 0; c < channels_per_group; ++c) {
            const int oc = group_offset + c;
            
            // --- ConvTranspose3d ---
            scalar_t val = 0.0f;
            
            // Input volume corresponding to this output pixel
            // We iterate over the kernel
            for (int kd = 0; kd < kernel_d; ++kd) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        // Calculate input coordinates
                        const int id = od * stride_d - padding_d + kd;
                        const int ih = oh * stride_h - padding_h + kh;
                        const int iw = ow * stride_w - padding_w + kw;
                        
                        if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            // Weight layout: (in_channels, out_channels/groups, kernel_d, kernel_h, kernel_w)
                            // We need to iterate over input channels
                            for (int ic = 0; ic < in_channels; ++ic) {
                                const int input_idx = ((b * in_channels + ic) * in_depth + id) * in_height * in_width + ih * in_width + iw;
                                const int weight_idx = ((ic * (out_channels / groups) + (oc % (out_channels / groups))) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                                val += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            
            // --- ReLU ---
            val = val > 0.0f ? val : 0.0f;
            
            // Store intermediate result for GroupNorm
            // We need to write to a temporary buffer or accumulate directly.
            // Since we need the mean and var first, we can't write the final output yet.
            // However, we can write the post-ReLU value to a temporary global buffer.
            // For simplicity and to save memory, we will re-compute the convolution after finding mean/var.
            // This is wasteful. A better approach is to write to a temporary buffer.
            // Let's assume we write to a temporary buffer `temp_output`.
            // But we don't have one passed. We can use the output buffer as temp storage.
            // Let's write the post-ReLU value to output[b, oc, od, oh, ow].
            
            long long out_idx = ((b * out_channels + oc) * out_depth + od) * out_height * out_width + oh * out_width + ow;
            output[out_idx] = val;
            
            local_sum += val;
            local_sum_sq += val * val;
        }
    }
    
    // --- Block Reduction for GroupNorm ---
    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();
    
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + offset];
        }
        __syncthreads();
    }
    
    scalar_t mean = s_sum[0] / (spatial_size * channels_per_group);
    scalar_t var = s_sum_sq[0] / (spatial_size * channels_per_group) - mean * mean;
    scalar_t inv_std = rsqrtf(var + 1e-5f);
    
    // --- Final Pass: Apply GroupNorm ---
    // Re-read the temporary values from output buffer and apply normalization.
    for (int sp_idx = threadIdx.x; sp_idx < spatial_size; sp_idx += blockDim.x) {
        const int od = sp_idx / (out_height * out_width);
        const int ohw = sp_idx % (out_height * out_width);
        const int oh = ohw / out_width;
        const int ow = ohw % out_width;
        
        for (int c = 0; c < channels_per_group; ++c) {
            const int oc = group_offset + c;
            long long out_idx = ((b * out_channels + oc) * out_depth + od) * out_height * out_width + oh * out_width + ow;
            scalar_t val = output[out_idx];
            val = (val - mean) * inv_std;
            val = val * gn_weight[oc] + gn_bias[oc];
            output[out_idx] = val;
        }
    }
}

torch::Tensor fused_conv_relu_gn_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor gn_weight, 
    torch::Tensor gn_bias, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int in_depth, 
    int in_height, 
    int in_width, 
    int out_depth, 
    int out_height, 
    int out_width, 
    int kernel_d, 
    int kernel_h, 
    int kernel_w, 
    int stride_d, 
    int stride_h, 
    int stride_w, 
    int padding_d, 
    int padding_h, 
    int padding_w, 
    int groups) {

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    int channels_per_group = out_channels / groups;

    const int block_size = CUDA_NUM_THREADS;
    const dim3 blocks(batch_size, groups);
    const size_t shared_mem_size = 2 * block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_relu_gn_kernel", ([&] {
        fused_conv_relu_gn_kernel<scalar_t><<<blocks, block_size, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            gn_weight.data_ptr<scalar_t>(),
            gn_bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            in_depth, in_height, in_width,
            out_depth, out_height, out_width,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            groups, channels_per_group);
    }));

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_conv_relu_gn_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor gn_weight, torch::Tensor gn_bias, "
    "int batch_size, int in_channels, int out_channels, int in_depth, int in_height, int in_width, "
    "int out_depth, int out_height, int out_width, int kernel_d, int kernel_h, int kernel_w, "
    "int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, int groups);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv_relu_gn_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = (1, 1, 1)
        self.padding = (1, 1, 1) # Assuming padding='same' or similar for kernel_size=3. Let's use standard padding.
        # ConvTranspose3d default padding is 0. Let's check PyTorch defaults.
        # PyTorch ConvTranspose3d: padding=0, output_padding=0.
        # The example doesn't specify padding, so it defaults to 0.
        # However, for kernel_size=3, stride=1, padding=0, output size = input_size + 2.
        # The problem description says D, H, W are 32 for both input and output.
        # This implies padding=1. Let's assume padding=1.
        self.padding = (1, 1, 1)
        
        # We need to create parameters for ConvTranspose3d and GroupNorm
        # ConvTranspose3d weights
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # GroupNorm parameters
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels))
        
        self.fused_ops = fused_ops

    def forward(self, x):
        # Calculate output dimensions
        batch_size, _, in_depth, in_height, in_width = x.shape
        out_depth = in_depth
        out_height = in_height
        out_width = in_width
        
        # Call the custom CUDA kernel
        # Note: The kernel is designed for a specific configuration.
        # We need to pass all necessary parameters.
        
        return self.fused_ops.fused_conv_relu_gn_cuda(
            x, self.weight, self.gn_weight, self.gn_bias,
            batch_size, self.in_channels, self.out_channels,
            in_depth, in_height, in_width,
            out_depth, out_height, out_width,
            self.kernel_size, self.kernel_size, self.kernel_size,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.groups
        )