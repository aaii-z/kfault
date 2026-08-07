import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution + min + tanh operations
fused_conv_min_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output, 
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_size, int padding, int stride,
    int output_h, int output_w) {
    
    // Shared memory for input tile and weights
    __shared__ float input_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float weight_tile[TILE_SIZE][TILE_SIZE];
    
    const int n = blockIdx.z;  // batch index
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;  // output height
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;  // output width
    
    if (n >= batch_size || oh >= output_h || ow >= output_w) return;
    
    // Initialize min value for this spatial position
    float min_val = INFINITY;
    
    // Process output channels in tiles
    for (int oc_tile = 0; oc_tile < out_channels; oc_tile += TILE_SIZE) {
        int oc_end = min(oc_tile + TILE_SIZE, out_channels);
        
        // For each output channel in current tile
        for (int oc = oc_tile; oc < oc_end; oc++) {
            float conv_result = 0.0f;
            
            // Process input channels in tiles
            for (int ic_tile = 0; ic_tile < in_channels; ic_tile += TILE_SIZE) {
                int ic_end = min(ic_tile + TILE_SIZE, in_channels);
                
                // Load weights for current tile
                if (threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE) {
                    int w_ic = ic_tile + threadIdx.y;
                    int w_oc = oc_tile + threadIdx.x;
                    if (w_ic < in_channels && w_oc < out_channels) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int weight_idx = w_oc * (in_channels * kernel_size * kernel_size) + 
                                               w_ic * (kernel_size * kernel_size) + 
                                               kh * kernel_size + kw;
                                weight_tile[threadIdx.y][threadIdx.x] = weight[weight_idx];
                            }
                        }
                    }
                }
                __syncthreads();
                
                // Process kernel elements
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int ih = oh * stride + kh - padding;
                        int iw = ow * stride + kw - padding;
                        
                        if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                            // Load input tile
                            for (int ic = ic_tile; ic < ic_end; ic++) {
                                int input_idx = n * (in_channels * input_h * input_w) + 
                                             ic * (input_h * input_w) + 
                                             ih * input_w + iw;
                                input_tile[ic - ic_tile][0] = input[input_idx];
                            }
                            __syncthreads();
                            
                            // Accumulate convolution result
                            for (int ic = ic_tile; ic < ic_end; ic++) {
                                int weight_idx = oc * (in_channels * kernel_size * kernel_size) + 
                                               ic * (kernel_size * kernel_size) + 
                                               kh * kernel_size + kw;
                                conv_result += input_tile[ic - ic_tile][0] * weight[weight_idx];
                            }
                            __syncthreads();
                        }
                    }
                }
            }
            
            // Add bias
            if (bias != nullptr) {
                conv_result += bias[oc];
            }
            
            // Update min value
            min_val = fminf(min_val, conv_result);
        }
    }
    
    // Apply double tanh
    float tanh1 = tanhf(min_val);
    float tanh2 = tanhf(tanh1);
    
    // Write output
    int output_idx = n * (output_h * output_w) + oh * output_w + ow;
    output[output_idx] = tanh2;
}

torch::Tensor fused_conv_min_tanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int padding, int stride) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_h = input.size(2);
    auto input_w = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    const int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, 1, output_h, output_w}, input.options());
    
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim(
        (output_w + block_dim.x - 1) / block_dim.x,
        (output_h + block_dim.y - 1) / block_dim.y,
        batch_size
    );
    
    fused_conv_min_tanh_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        input_h, input_w, kernel_size, padding, stride, output_h, output_w);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_conv_min_tanh_cpp_source = """
torch::Tensor fused_conv_min_tanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int padding, int stride);
"""

# Compile the inline CUDA code
fused_conv_min_tanh = load_inline(
    name="fused_conv_min_tanh",
    cpp_sources=fused_conv_min_tanh_cpp_source,
    cuda_sources=fused_conv_min_tanh_source,
    functions=["fused_conv_min_tanh_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that performs fused convolution + min + tanh operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = 1
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_op = fused_conv_min_tanh

    def forward(self, x):
        return self.fused_op.fused_conv_min_tanh_cuda(
            x, self.weight, self.bias, self.padding, self.stride
        )