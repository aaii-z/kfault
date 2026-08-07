import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + ReLU + Bias
fused_conv_relu_bias_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

// Define alignment for shared memory access
#define ALIGN_BYTES 16

// Tiling configuration
// Block size: 16x16 pixels
// Each thread computes 1 output pixel
#define TILE_DIM 16
#define FILTER_RADIUS 1
#define CACHE_WIDTH (TILE_DIM + 2 * FILTER_RADIUS)

// Kernel for Convolution + ReLU + Bias
// Assumes kernel_size=3, stride=1, padding=0 (implicit in input indexing)
// Supports arbitrary batch size, in_channels, out_channels, height, width
__global__ void conv_relu_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width) {

    // Shared memory for input tile (load 2D patch)
    __shared__ float shared_input[CACHE_WIDTH][CACHE_WIDTH];

    // Weights are accessed via global memory (texture cache usually helps here)
    // Optimization: Using float4 for weight access might be beneficial but complex indexing
    // For simplicity and robustness, we use direct global loads for weights.

    int batch_idx = blockIdx.z;
    int out_channel_idx = blockIdx.y; // One block per output channel per batch slice
    int tile_x = blockIdx.x * TILE_DIM;
    int tile_y = blockIdx.y * TILE_DIM; // Note: blockIdx.y is used for channel, so we map differently

    // Correct block mapping:
    // blockIdx.z = batch index
    // blockIdx.y = output channel index
    // blockIdx.x = tile index in width dimension (flattened 2D grid for H/W)
    
    // Let's map H/W to a 2D grid of blocks
    // Grid: (batch_size * out_channels, (out_height + TILE_DIM - 1) / TILE_DIM, (out_width + TILE_DIM - 1) / TILE_DIM)
    // Wait, max grid dim z is limited. Better:
    // Grid: (batch_size, out_channels, tiles_per_image)
    // Or flatten batch and channel: blockIdx.z = batch * out_channels
    
    // Let's stick to a standard mapping:
    // blockIdx.z: batch index
    // blockIdx.y: output channel index
    // blockIdx.x: tile index in the spatial dimension (linearized 16x16 tiles)
    
    int b = blockIdx.z;
    int oc = blockIdx.y;
    
    // Calculate spatial tile start
    int tiles_per_row = (out_width + TILE_DIM - 1) / TILE_DIM;
    int tile_idx = blockIdx.x;
    int tile_start_y = (tile_idx / tiles_per_row) * TILE_DIM;
    int tile_start_x = (tile_idx % tiles_per_row) * TILE_DIM;

    int ty = threadIdx.y; // 0 to 15
    int tx = threadIdx.x; // 0 to 15

    int out_y = tile_start_y + ty;
    int out_x = tile_start_x + tx;

    // Load input tile into shared memory
    // We need to load a (TILE_DIM + 2) x (TILE_DIM + 2) region
    // to compute a TILE_DIM x TILE_DIM output.
    
    // Input coordinates for loading
    // We need to handle boundaries (padding with 0)
    // Each thread loads one element. We have 16x16 = 256 threads.
    // We need to load 18x18 = 324 elements.
    // We need to perform multiple loads per thread or use a strided loop.
    
    // Strided load loop
    int shared_y = ty;
    int shared_x = tx;
    
    // Loop to cover the cache area
    for (int y_off = 0; y_off < CACHE_WIDTH; y_off += TILE_DIM) {
        for (int x_off = 0; x_off < CACHE_WIDTH; x_off += TILE_DIM) {
            
            shared_y = ty + y_off;
            shared_x = tx + x_off;

            if (shared_y < CACHE_WIDTH && shared_x < CACHE_WIDTH) {
                int in_y = tile_start_y - FILTER_RADIUS + shared_y;
                int in_x = tile_start_x - FILTER_RADIUS + shared_x;

                float val = 0.0f;
                // Check bounds
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    // We need to load for all input channels later? 
                    // No, shared memory is too small to cache all channels.
                    // We can only cache the spatial patch for the current input channel being processed.
                    // This implies an outer loop over in_channels.
                    // So we don't load to shared memory here yet. We do it inside the channel loop.
                }
                // Placeholder, logic moved inside channel loop
            }
        }
    }
    // Actually, caching input in shared memory is tricky because of channel dimension.
    // Standard optimization: Cache weights in shared memory? No, too big.
    // Cache input tile? Yes, but we iterate over in_channels.
    // We can load a tile of (in_y, in_x) for a specific channel.
    // Since threads cooperate to load, we need synchronization.
    
    float sum = 0.0f;
    
    // Loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        
        // Load (CACHE_WIDTH x CACHE_WIDTH) patch for channel ic into shared memory
        // Each thread loads multiple elements if needed (strided loop)
        for (int i = threadIdx.y; i < CACHE_WIDTH; i += TILE_DIM) {
            for (int j = threadIdx.x; j < CACHE_WIDTH; j += TILE_DIM) {
                int in_y = tile_start_y - FILTER_RADIUS + i;
                int in_x = tile_start_x - FILTER_RADIUS + j;
                
                float val = 0.0f;
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    // Input tensor layout: (N, C, H, W)
                    // input[b, ic, in_y, in_x]
                    val = input[(b * in_channels + ic) * in_height * in_width + in_y * in_width + in_x];
                }
                shared_input[i][j] = val;
            }
        }
        
        // Synchronize to ensure the tile is loaded
        __syncthreads();
        
        // Compute convolution for this channel
        // Only compute if output pixel is valid
        if (out_y < out_height && out_x < out_width) {
            // Weights layout: (out_channels, in_channels, 3, 3)
            // weight[oc, ic, ky, kx]
            const float* weight_ptr = weight + (oc * in_channels + ic) * 9;
            
            // Unroll 3x3 filter
            #pragma unroll
            for (int ky = 0; ky < 3; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    // Access shared memory
                    // The shared memory corresponds to the tile.
                    // Thread (ty, tx) computes output at (out_y, out_x).
                    // The input patch needed is (out_y-1 + ky, out_x-1 + kx).
                    // In shared memory coordinates:
                    // shared_y = ty + ky (since shared_input[0][0] corresponds to in_y = tile_start_y - 1)
                    // shared_x = tx + kx
                    
                    // Wait, shared_input[0][0] is (tile_start_y - 1, tile_start_x - 1).
                    // We want input at (out_y - 1 + ky, out_x - 1 + kx).
                    // out_y = tile_start_y + ty.
                    // So we want (tile_start_y + ty - 1 + ky).
                    // In shared memory, index 0 corresponds to tile_start_y - 1.
                    // So index is ty + ky.
                    
                    float in_val = shared_input[ty + ky][tx + kx];
                    float w_val = weight_ptr[ky * 3 + kx];
                    sum += in_val * w_val;
                }
            }
        }
        
        // Synchronize before loading next channel
        __syncthreads();
    }
    
    // Write output
    if (out_y < out_height && out_x < out_width) {
        // Add bias
        // Bias shape: (out_channels, 1, 1)
        float b_val = bias[oc];
        sum += b_val;
        
        // ReLU
        sum = fmaxf(0.0f, sum);
        
        // Output layout: (N, C, H, W)
        output[(b * out_channels + oc) * out_height * out_width + out_y * out_width + out_x] = sum;
    }
}

torch::Tensor fused_conv_relu_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    // Input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    // Weight dimensions
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // Assuming square kernel
    
    // Output dimensions
    int out_height = in_height - kernel_size + 1;
    int out_width = in_width - kernel_size + 1;
    
    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    // Kernel launch configuration
    // Block size: TILE_DIM x TILE_DIM (16x16)
    dim3 block_size(TILE_DIM, TILE_DIM);
    
    // Grid size
    // We need enough blocks to cover the output spatial dimensions
    int tiles_per_row = (out_width + TILE_DIM - 1) / TILE_DIM;
    int tiles_per_col = (out_height + TILE_DIM - 1) / TILE_DIM;
    int spatial_tiles = tiles_per_row * tiles_per_col;
    
    dim3 grid_size(spatial_tiles, out_channels, batch_size);
    
    // Launch kernel
    conv_relu_bias_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width);
    
    return output;
}
"""

fused_conv_relu_bias_cpp_source = (
    "torch::Tensor fused_conv_relu_bias_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_conv_relu_bias",
    cpp_sources=fused_conv_relu_bias_cpp_source,
    cuda_sources=fused_conv_relu_bias_source,
    functions=["fused_conv_relu_bias_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for fused Conv2d + ReLU + Bias.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # We keep the weight and bias as Parameters to maintain the interface
        # but we will use them directly in the custom kernel.
        # Note: We assume kernel_size=3 based on the problem description.
        # We initialize weights similar to nn.Conv2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Initialize weights using Kaiming uniform (default for Conv2d)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        # Bias initialization (Conv2d default is uniform bound calculated from fan_in)
        # We'll just use randn as per original model code, or we can match Conv2d exactly.
        # The original model used torch.randn for bias.
        
        # Load the custom op
        self.fused_op = fused_op

    def forward(self, x):
        # The custom kernel handles convolution, ReLU, and bias addition
        # We need to ensure inputs are contiguous and on the correct device
        return self.fused_op.fused_conv_relu_bias_cuda(x, self.weight, self.bias)