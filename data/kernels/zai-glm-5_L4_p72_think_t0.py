import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm and Average Pooling
# We observe that AvgPool3d(kernel_size=2) applied twice is equivalent to
# AvgPool3d(kernel_size=4, stride=4) on the valid region.
# We fuse Batch Normalization and this effective pooling operation into a single kernel.

fused_bn_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bn_pool_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output,
    const float* __restrict__ mean, 
    const float* __restrict__ var, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    int N, int C, int D, int H, int W, 
    int D_out, int H_out, int W_out, 
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D_out * H_out * W_out;
    
    if (idx < total_elements) {
        // Calculate output coordinates (n, c, d_out, h_out, w_out)
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int d_out = (idx / (W_out * H_out)) % D_out;
        int c = (idx / (W_out * H_out * D_out)) % C;
        int n = idx / (W_out * H_out * D_out * C);
        
        // Retrieve BatchNorm parameters for this channel
        float m = mean[c];
        float v = var[c];
        float w_bn = weight[c];
        float b_bn = bias[c];
        float inv_std = 1.0f / sqrtf(v + eps);
        
        float sum = 0.0f;
        
        // Calculate input window start indices
        // The two original AvgPool3d(kernel=2) layers result in a receptive field of 4
        int d_start = d_out * 4;
        int h_start = h_out * 4;
        int w_start = w_out * 4;
        
        // Strides for input tensor (assuming contiguous NCDHW layout)
        int stride_n = C * D * H * W;
        int stride_c = D * H * W;
        int stride_d = H * W;
        int stride_h = W;
        int stride_w = 1;
        
        int base_n = n * stride_n;
        int base_c = c * stride_c;
        
        // Sum over the 4x4x4 window
        // We assume the input dimensions are such that the window fits (e.g., D=63 -> D_out=15, 15*4=60 < 63)
        // The original architecture drops the tail elements (floor division logic)
        for (int kd = 0; kd < 4; ++kd) {
            int d_in = d_start + kd;
            int base_d = d_in * stride_d;
            for (int kh = 0; kh < 4; ++kh) {
                int h_in = h_start + kh;
                int base_h = h_in * stride_h;
                for (int kw = 0; kw < 4; ++kw) {
                    int w_in = w_start + kw;
                    
                    int in_idx = base_n + base_c + base_d + base_h + w_in;
                    float val = input[in_idx];
                    
                    // Apply BatchNorm
                    val = (val - m) * inv_std * w_bn + b_bn;
                    sum += val;
                }
            }
        }
        
        // Average and write to output
        output[idx] = sum * 0.015625f; // 1/64
    }
}

torch::Tensor fused_bn_pool_cuda(
    torch::Tensor input, 
    torch::Tensor mean, 
    torch::Tensor var, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    double eps) {
    
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    // Output dimensions after two AvgPool3d(kernel_size=2) layers
    // This is equivalent to integer division by 4
    int D_out = D / 4;
    int H_out = H / 4;
    int W_out = W / 4;
    
    auto output = torch::zeros({N, C, D_out, H_out, W_out}, input.options());
    
    int total_elements = output.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_bn_pool_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, D, H, W,
        D_out, H_out, W_out,
        static_cast<float>(eps)
    );
    
    return output;
}
"""

fused_bn_pool_cpp_source = (
    "torch::Tensor fused_bn_pool_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var, torch::Tensor weight, torch::Tensor bias, double eps);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_bn_pool",
    cpp_sources=fused_bn_pool_cpp_source,
    cuda_sources=fused_bn_pool_source,
    functions=["fused_bn_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel to fuse BatchNorm3d and two AvgPool3d layers.
    The ConvTranspose3d layer is kept as is, as it is computationally intensive and 
    optimized in PyTorch.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        # We remove the separate pooling layers and use the fused kernel instead
        self.fused_op = fused_op

    def forward(self, x):
        # Step 1: Transposed Convolution
        x = self.conv_transpose(x)
        
        # Step 2: Fused BatchNorm + AvgPool
        # We extract the running statistics and parameters from the batch_norm layer
        # to pass to our custom kernel.
        x = self.fused_op.fused_bn_pool_cuda(
            x, 
            self.batch_norm.running_mean, 
            self.batch_norm.running_var, 
            self.batch_norm.weight, 
            self.batch_norm.bias, 
            self.batch_norm.eps
        )
        
        return x