import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + ReLU + Bias
fused_conv_relu_bias_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

// Tiling configuration
#define TILE_DIM 16
#define FILTER_RADIUS 1
#define CACHE_WIDTH (TILE_DIM + 2 * FILTER_RADIUS)

// Kernel for Convolution + ReLU + Bias
// Assumes kernel_size=3, stride=1, padding=0
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

    int b = blockIdx.z;
    int oc = blockIdx.y;
    
    // Calculate spatial tile start
    int tiles_per_row = (out_width + TILE_DIM - 1) / TILE_DIM;
    int tile_idx = blockIdx.x;
    int tile_start_y = (tile_idx / tiles_per_row) * TILE_DIM;
    int tile_start_x = (tile_idx % tiles_per_row) * TILE_DIM;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int out_y = tile_start_y + ty;
    int out_x = tile_start_x + tx;

    float sum = 0.0f;
    
    // Loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        
        // Load (CACHE_WIDTH x CACHE_WIDTH) patch for channel ic into shared memory
        // Strided loop to cover the shared memory block
        for (int i = threadIdx.y; i < CACHE_WIDTH; i += TILE_DIM) {
            for (int j = threadIdx.x; j < CACHE_WIDTH; j += TILE_DIM) {
                int in_y = tile_start_y - FILTER_RADIUS + i;
                int in_x = tile_start_x - FILTER_RADIUS + j;
                
                float val = 0.0f;
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    val = input[(b * in_channels + ic) * in_height * in_width + in_y * in_width + in_x];
                }
                shared_input[i][j] = val;
            }
        }
        
        __syncthreads();
        
        // Compute convolution for this channel
        if (out_y < out_height && out_x < out_width) {
            const float* weight_ptr = weight + (oc * in_channels + ic) * 9;
            
            #pragma unroll
            for (int ky = 0; ky < 3; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    float in_val = shared_input[ty + ky][tx + kx];
                    float w_val = weight_ptr[ky * 3 + kx];
                    sum += in_val * w_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    if (out_y < out_height && out_x < out_width) {
        float b_val = bias[oc];
        sum += b_val;
        sum = fmaxf(0.0f, sum);
        output[(b * out_channels + oc) * out_height * out_width + out_y * out_width + out_x] = sum;
    }
}

torch::Tensor fused_conv_relu_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = in_height - kernel_size + 1;
    int out_width = in_width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    dim3 block_size(TILE_DIM, TILE_DIM);
    
    int tiles_per_row = (out_width + TILE_DIM - 1) / TILE_DIM;
    int tiles_per_col = (out_height + TILE_DIM - 1) / TILE_DIM;
    int spatial_tiles = tiles_per_row * tiles_per_col;
    
    dim3 grid_size(spatial_tiles, out_channels, batch_size);
    
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
        # Define weights and bias as parameters to be trained/loaded
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Initialize weights using Kaiming uniform to match nn.Conv2d default initialization
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        # Bias initialization: The original model used randn, but Conv2d default is uniform.
        # To match the functional behavior of the original model exactly, 
        # we should usually match the initialization.
        # However, the original model code used torch.randn for bias in __init__.
        # Let's stick to the provided logic: randn was used in original.
        # But wait, the original model used `self.bias = nn.Parameter(torch.randn(bias_shape))`.
        # So we should reset it to randn if we want to match the "random" state, 
        # but usually we want to match the architecture definition.
        # The original code: `self.bias = nn.Parameter(torch.randn(bias_shape))`
        # My code above uses randn. That is correct.
        
        self.fused_op = fused_op

    def forward(self, x):
        # Ensure weight is contiguous for the kernel access pattern
        return self.fused_op.fused_conv_relu_bias_cuda(x, self.weight.contiguous(), self.bias)