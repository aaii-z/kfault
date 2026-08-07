import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LayerNorm, GELU, and Scaling
# This kernel fuses the normalization, activation, and scaling operations
# to reduce memory bandwidth usage and kernel launch overhead.
fused_ln_gelu_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Approximation of GELU using the tanh approximation
__device__ __forceinline__ float gelu_approx(float x) {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Constants
    const float SQRT_2_PI = 0.7978845608028654f;
    const float COEFF = 0.044715f;
    float x3 = x * x * x;
    float inner = SQRT_2_PI * (x + COEFF * x3);
    float tanh_inner = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_inner);
}

// Kernel for LayerNorm + GELU + Scale
// We process the channel dimension (C) as the normalized dimension.
// Input/Output shape is assumed to be (N, C, D, H, W) or flattened (TotalElements / C, C)
// We launch blocks for each independent normalization instance.
__global__ void fused_ln_gelu_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float scaling_factor,
    int num_instances,
    int channels,
    int num_spatial_elements
) {
    // Each block handles one (N, D, H, W) instance
    int instance_idx = blockIdx.x;
    if (instance_idx >= num_instances) return;

    int offset = instance_idx * channels;
    
    // 1. Calculate Mean and Variance (Reduction over channels)
    // Using sequential addressing for reduction within the block
    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Use shared memory for reduction if needed, but channels might be small (e.g., 64).
    // For simplicity and robustness across different channel sizes, we use a loop.
    // If channels is large, we might need parallel reduction, but for typical CNNs (64-512), a loop is often efficient enough per thread block if we parallelize the loop.
    // However, here we have 1 block per instance. We need threads to cooperate.
    
    // Parallel reduction within the block
    // We assign threads to cover the channel dimension
    int tid = threadIdx.x;
    int block_dim = blockDim.x;
    
    __shared__ float s_mean;
    __shared__ float s_var;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Loop over channels with stride
    for (int c = tid; c < channels; c += block_dim) {
        float val = input[offset + c];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Reduce local sums to global sums using shared memory
    // Using standard parallel reduction pattern would be complex to inline for variable block sizes.
    // Instead, we can use atomicAdd or simply loop if block size is 1 (not efficient).
    // Better approach: Use warp shuffle instructions or just atomicAdd for simplicity since it's one-off per block.
    // Or, since we have a small fixed block size (e.g., 256) and channels might be small, let's use atomicAdd.
    
    // Note: atomicAdd on float is supported on modern GPUs.
    // However, to avoid atomics, we can do a simple reduction in shared mem.
    
    __shared__ float s_sum[256]; // Max block size assumption
    __shared__ float s_sum_sq[256];

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Reduction in shared memory
    for (int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = s_sum[0] / channels;
        float variance = s_sum_sq[0] / channels - mean * mean;
        s_mean = mean;
        s_var = variance;
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = rsqrtf(s_var + 1e-5f);

    // 2. Normalize, Activate, Scale, and Write
    for (int c = tid; c < channels; c += block_dim) {
        float val = input[offset + c];
        float normalized = (val - mean) * inv_std;
        
        // Apply weight and bias
        float w = weight[c];
        float b = bias[c];
        float ln_out = normalized * w + b;
        
        // Apply GELU
        float gelu_out = gelu_approx(ln_out);
        
        // Apply scaling
        output[offset + c] = gelu_out * scaling_factor;
    }
}

torch::Tensor fused_ln_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor
) {
    // input shape: (N, C, D, H, W)
    // output shape: (N, C, D, H, W)
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    int num_spatial = D * H * W;
    int num_instances = N * num_spatial;
    
    auto output = torch::empty_like(input);
    
    const int block_size = 256; // Typical block size
    const int num_blocks = num_instances; // One block per instance
    
    // Launch kernel
    fused_ln_gelu_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        num_instances,
        C,
        num_spatial
    );
    
    return output;
}
"""

fused_ln_gelu_scale_cpp_source = (
    "torch::Tensor fused_ln_gelu_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ln_gelu_scale",
    cpp_sources=fused_ln_gelu_scale_cpp_source,
    cuda_sources=fused_ln_gelu_scale_source,
    functions=["fused_ln_gelu_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution followed by a fused
    LayerNorm, GELU activation, and scaling operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        
        # We keep the LayerNorm module to hold the learnable parameters (weight and bias)
        # and to handle the initialization, but we will not call its forward pass.
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
        
        # Load the custom op
        self.fused_op = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        # 1. Perform Transposed Convolution
        x = self.conv_transpose(x)
        
        # 2. Perform Fused LayerNorm + GELU + Scaling
        # The custom kernel handles the normalization over the channel dimension,
        # applies GELU, and scales the result.
        # We pass the weight and bias from the LayerNorm module.
        x = self.fused_op.fused_ln_gelu_scale_cuda(
            x, 
            self.layer_norm.weight, 
            self.layer_norm.bias, 
            self.scaling_factor
        )
        
        return x