import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution, scaling, and min reduction
fused_conv_scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Define constant memory for convolution weights and bias
// Assuming max kernel_size 7 for demonstration, adjust as needed
__constant__ float c_weight[128*64*7*7]; 
__constant__ float c_bias[128];

// Tiled convolution kernel with scaling and min reduction
__global__ void fused_conv_scale_min_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output,
    const int batch_size, const int in_channels, 
    const int out_channels, const int height, const int width,
    const int kernel_size, const float scale_factor,
    const int padding, const int stride) {
    
    // Shared memory for input tile
    // Tile size needs to cover output block + kernel_size - 1
    // Assuming block size 16x16 for output
    const int TILE_SIZE = 16;
    const int IN_TILE_SIZE = TILE_SIZE + 7 - 1; // Max kernel size 7
    
    __shared__ float shared_input[IN_TILE_SIZE][IN_TILE_SIZE];
    
    // Output dimensions
    const int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    // Batch and output channel index
    const int b = blockIdx.z / out_channels;
    const int oc = blockIdx.z % out_channels;
    
    // Output pixel coordinates
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Initialize min_val with a large value
    float min_val = FLT_MAX;
    
    // Loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Load input tile into shared memory
        // Calculate input tile coordinates
        const int in_x_start = blockIdx.x * blockDim.x * stride - padding;
        const int in_y_start = blockIdx.y * blockDim.y * stride - padding;
        
        // Cooperative loading of input tile
        for (int i = threadIdx.y; i < IN_TILE_SIZE; i += blockDim.y) {
            for (int j = threadIdx.x; j < IN_TILE_SIZE; j += blockDim.x) {
                int in_x = in_x_start + j;
                int in_y = in_y_start + i;
                
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    shared_input[i][j] = input[b * in_channels * height * width + 
                                               ic * height * width + 
                                               in_y * width + in_x];
                } else {
                    shared_input[i][j] = 0.0f; // Zero padding
                }
            }
        }
        __syncthreads();
        
        // Compute convolution for this input channel
        if (ox < out_w && oy < out_h) {
            float conv_val = c_bias[oc]; // Start with bias
            
            // Convolution loop
            #pragma unroll
            for (int ky = 0; ky < kernel_size; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < kernel_size; ++kx) {
                    // Weight index: (oc, ic, ky, kx)
                    int weight_idx = oc * in_channels * kernel_size * kernel_size + 
                                    ic * kernel_size * kernel_size + 
                                    ky * kernel_size + kx;
                    
                    // Input index in shared memory
                    int shared_y = threadIdx.y * stride + ky;
                    int shared_x = threadIdx.x * stride + kx;
                    
                    conv_val += shared_input[shared_y][shared_x] * c_weight[weight_idx];
                }
            }
            
            // Scale
            conv_val *= scale_factor;
            
            // Update min
            min_val = fminf(min_val, conv_val);
        }
        __syncthreads();
    }
    
    // Write output
    if (ox < out_w && oy < out_h) {
        // Output shape: (batch_size, 1, out_h, out_w)
        output[b * out_h * out_w + oy * out_w + ox] = min_val;
    }
}

torch::Tensor fused_conv_scale_min_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int64_t kernel_size, 
    double scale_factor, 
    int64_t padding, 
    int64_t stride) {
    
    // Get dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    
    // Calculate output dimensions
    const int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    // Allocate output tensor
    auto output = torch::zeros({batch_size, 1, out_h, out_w}, input.options());
    
    // Copy weights and bias to constant memory
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), weight.numel() * sizeof(float));
    cudaMemcpyToSymbol(c_bias, bias.data_ptr<float>(), bias.numel() * sizeof(float));
    
    // Launch kernel
    const int block_size = 16;
    dim3 threads(block_size, block_size);
    dim3 blocks((out_w + block_size - 1) / block_size, 
                (out_h + block_size - 1) / block_size, 
                batch_size * out_channels);
    
    fused_conv_scale_min_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width,
        kernel_size, static_cast<float>(scale_factor),
        padding, stride
    );
    
    return output;
}
"""

fused_conv_scale_min_cpp_source = (
    "torch::Tensor fused_conv_scale_min_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int64_t kernel_size, double scale_factor, int64_t padding, int64_t stride);"
)

# Compile the inline CUDA code for fused convolution, scaling, and min reduction
fused_conv_scale_min = load_inline(
    name="fused_conv_scale_min",
    cpp_sources=fused_conv_scale_min_cpp_source,
    cuda_sources=fused_conv_scale_min_source,
    functions=["fused_conv_scale_min_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, scales the output, and then applies a minimum operation.
    All operations are fused into a single CUDA kernel for better performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        # Store convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        
        # Initialize weights and bias similar to nn.Conv2d
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Load the fused CUDA kernel
        self.fused_conv_scale_min = fused_conv_scale_min

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, height, width).
        """
        # Call the fused CUDA kernel
        return self.fused_conv_scale_min.fused_conv_scale_min_cuda(
            x, self.weight, self.bias, self.kernel_size, self.scale_factor, 
            padding=1, stride=1
        )