import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose3d + BatchNorm + AvgPool
fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>

// Helper function to calculate output dimensions
__host__ __device__ int calc_out_dim(int in_dim, int kernel_size, int stride, int padding, int dilation=1) {
    return (in_dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// Kernel for ConvTranspose3d + BatchNorm + AvgPool fusion
// We compute the output for the final pooled output.
// Output size: (N, C, D_out/4, H_out/4, W_out/4)
// AvgPool kernel is 2x2x2, stride 2.
// We compute the sum of 8 input patches for the pooling.
// Formula: y[n, c, d, h, w] = (1/8) * sum_{kd,kh,kw} BN( ConvT(X) )
// BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
// We can rewrite this as: y = A * x + B
// where A = gamma / sqrt(var + eps), B = beta - mean * A.
// So for each output element of the pool, we sum 8 (A*x + B) terms and divide by 8.
// y = (1/8) * sum( A*x_i + B ) = (A/8) * sum(x_i) + B
// Note: A and B depend on the channel 'c'.

__global__ void fused_conv_bn_pool_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, // ConvTranspose weights (C_in, C_out, kD, kH, kW)
    const float* __restrict__ bias,   // ConvTranspose bias (C_out)
    const float* __restrict__ bn_mean, // BN mean (C_out)
    const float* __restrict__ bn_var,  // BN var (C_out)
    const float* __restrict__ bn_gamma, // BN gamma (C_out)
    const float* __restrict__ bn_beta,  // BN beta (C_out)
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out, // ConvT output dims
    int kD, int kH, int kW,
    int stride, int padding) {

    // Output dimensions are pooled dimensions
    int D_pooled = (D_out + 1) / 2; // Assuming kernel 2, stride 2
    int H_pooled = (H_out + 1) / 2;
    int W_pooled = (W_out + 1) / 2;

    int n = blockIdx.x; // Batch index
    int c = blockIdx.y; // Channel index
    
    // Spatial index in pooled output
    int spatial_idx = blockIdx.z * blockDim.x + threadIdx.x;
    int d_pool = spatial_idx / (H_pooled * W_pooled);
    int hw_pool = spatial_idx % (H_pooled * W_pooled);
    int h_pool = hw_pool / W_pooled;
    int w_pool = hw_pool % W_pooled;

    if (d_pool >= D_pooled || h_pool >= H_pooled || w_pool >= W_pooled) return;

    // Pre-compute BN parameters for this channel
    float mean = bn_mean[c];
    float var = bn_var[c];
    float gamma = bn_gamma[c];
    float beta = bn_beta[c];
    float eps = 1e-5f;
    
    float A = gamma / sqrtf(var + eps);
    float B = beta - mean * A;

    // ConvTranspose bias term contribution to the sum
    // Each of the 8 positions in the pool window receives the bias.
    // So sum of biases = 8 * bias[c].
    // Then we apply the BN transformation logic: sum( A*(val+bias) + B )? 
    // No, bias is added in ConvT. So val = dot_prod + bias.
    // BN(val) = A * val + B.
    // We want sum(BN(val)) for the 8 values.
    // sum(A * (dot_prod + bias) + B) = A * sum(dot_prod) + 8 * A * bias + 8 * B.
    // Final output is average: (sum / 8).
    // Result = (A/8) * sum(dot_prod) + A * bias + B.
    
    float conv_bias_term = A * bias[c] + B;
    
    float sum_val = 0.0f;

    // Iterate over the 2x2x2 pooling window
    // Pooling input is the output of ConvTranspose
    for (int pd = 0; pd < 2; ++pd) {
        for (int ph = 0; ph < 2; ++ph) {
            for (int pw = 0; pw < 2; ++pw) {
                int d_convt = d_pool * 2 + pd;
                int h_convt = h_pool * 2 + ph;
                int w_convt = w_pool * 2 + pw;

                // Bounds check for ConvTranspose output (if D_out/H_out/W_out are odd)
                if (d_convt >= D_out || h_convt >= H_out || w_convt >= W_out) continue;

                // Compute ConvTranspose output at (d_convt, h_convt, w_convt)
                // We need to iterate over input volume and kernel volume
                // ConvTranspose: input contributes to output if input coord aligns with kernel
                
                float dot_prod = 0.0f;

                // Iterate over input channels
                for (int ic = 0; ic < C_in; ++ic) {
                    // Iterate over kernel
                    for (int kd = 0; kd < kD; ++kd) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                // Calculate input coordinates corresponding to this output pixel and kernel element
                                // Formula: i = (o + 2*pad - k) / stride
                                // We check if (o + 2*pad - k) is divisible by stride
                                
                                int d_tmp = d_convt + padding - kd;
                                int h_tmp = h_convt + padding - kh;
                                int w_tmp = w_convt + padding - kw;

                                if (d_tmp % stride == 0 && h_tmp % stride == 0 && w_tmp % stride == 0) {
                                    int d_in = d_tmp / stride;
                                    int h_in = h_tmp / stride;
                                    int w_in = w_tmp / stride;

                                    if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                        // Weight index: ConvTranspose weight layout is (C_in, C_out, kD, kH, kW)
                                        // Note: PyTorch ConvTranspose weights are stored as (C_in, C_out, kD, kH, kW)
                                        // but the mathematical operation uses the transposed kernel.
                                        // However, the logic above effectively implements the sparse matrix multiplication.
                                        // We need to pick the correct weight.
                                        // If we are at output (d_convt, h_convt, w_convt) and kernel (kd, kh, kw),
                                        // and input (d_in, h_in, w_in), the weight connects input ic to output c.
                                        // The weight value is weight[ic, c, kd, kh, kw].
                                        
                                        int weight_idx = ((ic * C_out + c) * kD + kd) * kH * kW + kh * kW + kw;
                                        int input_idx = ((n * C_in + ic) * D_in + d_in) * H_in * W_in + h_in * W_in + w_in;
                                        
                                        dot_prod += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
                sum_val += dot_prod;
            }
        }
    }

    // Final calculation
    // Output = (A/8) * sum_val + conv_bias_term
    // Note: sum_val is sum of dot products over the 8 locations.
    // We need to divide by 8 for the average pooling.
    // However, we accumulated sum_val which is sum of dot products.
    // The term (A/8) * sum_val handles the scaling of the convolution result.
    // conv_bias_term already handles the bias and BN shift for the average.
    
    float result = (A / 8.0f) * sum_val + conv_bias_term;
    
    // Write output
    // Output layout: (N, C, D_pooled, H_pooled, W_pooled)
    int out_idx = ((n * C_out + c) * D_pooled + d_pool) * H_pooled * W_pooled + h_pool * W_pooled + w_pool;
    output[out_idx] = result;
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    torch::Tensor bn_mean, 
    torch::Tensor bn_var, 
    torch::Tensor bn_gamma, 
    torch::Tensor bn_beta,
    int stride, int padding) {

    // Input dims
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    // Weight dims: (C_in, C_out, kD, kH, kW)
    int C_out = weight.size(1);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    // Calculate ConvTranspose output dims
    int D_out = calc_out_dim(D_in, kD, stride, padding);
    int H_out = calc_out_dim(H_in, kH, stride, padding);
    int W_out = calc_out_dim(W_in, kW, stride, padding);

    // Calculate Pooled output dims
    int D_pooled = (D_out + 1) / 2;
    int H_pooled = (H_out + 1) / 2;
    int W_pooled = (W_out + 1) / 2;

    auto output = torch::zeros({N, C_out, D_pooled, H_pooled, W_pooled}, input.options());

    // Grid and Block dimensions
    // We map each (n, c, spatial_index) to a thread.
    // Max threads per block is 1024. We use a 1D block for spatial dimensions.
    // Grid: (N, C_out, num_spatial_blocks)
    
    int spatial_size = D_pooled * H_pooled * W_pooled;
    int threads_per_block = 256; // Typical size
    int num_spatial_blocks = (spatial_size + threads_per_block - 1) / threads_per_block;

    dim3 grid(N, C_out, num_spatial_blocks);
    dim3 block(threads_per_block);

    fused_conv_bn_pool_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        bn_gamma.data_ptr<float>(),
        bn_beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride, padding
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor bn_mean, torch::Tensor bn_var, torch::Tensor bn_gamma, torch::Tensor bn_beta, "
    "int stride, int padding);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        # We keep the original modules to hold the parameters and state dicts
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        
        # We don't need avg_pool modules as they are fused
        self.stride = stride
        self.padding = padding
        
        self.fused_ops = fused_ops

    def forward(self, x):
        # Use the fused kernel
        # We need to extract the parameters from the modules
        # ConvTranspose3d weight shape: (in_channels, out_channels, kD, kH, kW)
        # BatchNorm3d parameters: weight (gamma), bias (beta), running_mean, running_var
        
        # Ensure parameters are contiguous and on the correct device
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()
        
        bn_gamma = self.batch_norm.weight.contiguous()
        bn_beta = self.batch_norm.bias.contiguous()
        bn_mean = self.batch_norm.running_mean.contiguous()
        bn_var = self.batch_norm.running_var.contiguous()
        
        return self.fused_ops.fused_ops_cuda(
            x, weight, bias, 
            bn_mean, bn_var, bn_gamma, bn_beta, 
            self.stride, self.padding
        )