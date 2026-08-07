import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + AvgPool2d + Sigmoid + Sum
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Constant memory for convolution weights and bias (limited size, but faster)
// Assuming out_channels <= 1024 and in_channels <= 16 for constant memory usage optimization
// For generic sizes, we pass pointers. Here we stick to pointer access for generality.

__global__ void fused_conv_pool_sigmoid_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int kernel_size, int pool_size)
{
    // Calculate dimensions
    int out_h = in_h - kernel_size + 1;
    int out_w = in_w - kernel_size + 1;
    int pool_out_h = out_h / pool_size;
    int pool_out_w = out_w / pool_size;

    // Each block handles one (batch, out_channel) pair
    // blockIdx.x: batch index
    // blockIdx.y: out_channel index
    int b = blockIdx.x;
    int oc = blockIdx.y;

    // Shared memory for the partial sums of the pooling window
    // We reduce the spatial dimensions into a single sum.
    // We use a parallel reduction approach within the block.
    // Each thread accumulates a portion of the spatial sum.
    
    float local_sum = 0.0f;

    // Iterate over the pooled output grid
    // Each thread takes a stride over the pooled grid
    int pool_pixels = pool_out_h * pool_out_w;
    
    for (int i = threadIdx.x; i < pool_pixels; i += blockDim.x) {
        int ph = i / pool_out_w;
        int pw = i % pool_out_w;

        // Coordinates in the convolution output (before pooling)
        int h_start = ph * pool_size;
        int w_start = pw * pool_size;

        float pool_sum = 0.0f;

        // Average Pooling loop
        // Unrolling might help, but loop is safer for generic pool_size
        for (int kh = 0; kh < pool_size; ++kh) {
            for (int kw = 0; kw < pool_size; ++kw) {
                int h = h_start + kh;
                int w = w_start + kw;

                // Convolution calculation for this specific output pixel (oc, h, w)
                float conv_val = 0.0f;
                
                // Input offset for this batch
                const float* input_ptr = input + b * in_channels * in_h * in_w;
                
                // Convolution loops
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int ih = h + ky;
                            int iw = w + kx;
                            
                            // Weight index: (oc, ic, ky, kx)
                            // We assume valid convolution (no padding)
                            // Bounds check implicitly handled by output dimensions
                            float w_val = weight[oc * in_channels * kernel_size * kernel_size + 
                                                  ic * kernel_size * kernel_size + 
                                                  ky * kernel_size + kx];
                            
                            float i_val = input_ptr[ic * in_h * in_w + ih * in_w + iw];
                            conv_val += w_val * i_val;
                        }
                    }
                }
                
                // Add bias
                conv_val += bias[oc];
                
                // Accumulate for average pool
                pool_sum += conv_val;
            }
        }
        
        // Average
        float avg_val = pool_sum / (float)(pool_size * pool_size);
        
        // Sigmoid
        float sig_val = 1.0f / (1.0f + expf(-avg_val));
        
        // Accumulate to local sum
        local_sum += sig_val;
    }

    // Parallel reduction in shared memory
    __shared__ float s_data[256];
    s_data[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write result
    if (threadIdx.x == 0) {
        output[b * out_channels + oc] = s_data[0];
    }
}

torch::Tensor fused_conv_pool_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int kernel_size, int pool_size)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto output = torch::zeros({batch_size, out_channels}, options);

    int out_h = in_h - kernel_size + 1;
    int out_w = in_w - kernel_size + 1;
    int pool_out_h = out_h / pool_size;
    int pool_out_w = out_w / pool_size;

    // Grid: (batch_size, out_channels)
    dim3 grid(batch_size, out_channels);
    // Block: enough threads to cover the pooled spatial dimensions for reduction
    // 256 is a safe bet for block size
    int threads = 256; 

    fused_conv_pool_sigmoid_sum_kernel<<<grid, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_h, in_w, kernel_size, pool_size
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_conv_pool_sigmoid_sum_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "int batch_size, int in_channels, int out_channels, "
    "int in_h, int in_w, int kernel_size, int pool_size);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv_pool_sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with a custom CUDA kernel fusing Conv2d, AvgPool2d, Sigmoid, and Sum.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        # We still define the Conv2d layer to hold weights and bias, 
        # but we won't use its forward pass.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.fused_ops = fused_ops

    def forward(self, x):
        # Extract parameters
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0] # Assuming square kernel
        
        # Call the custom fused kernel
        return self.fused_ops.fused_conv_pool_sigmoid_sum_cuda(
            x, 
            self.conv.weight, 
            self.conv.bias,
            batch_size, in_channels, out_channels,
            in_h, in_w, kernel_size, self.pool_kernel_size
        )