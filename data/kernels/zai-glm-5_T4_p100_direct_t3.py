import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose3d + Clamp + Div
fused_conv_clamp_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Kernel for fused operations: ConvTranspose3d + Clamp + Division
// Optimized to iterate over input tiles to compute output
__global__ void fused_conv_transpose_3d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int N, int C_in, int C_out, 
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    float min_val, float divisor)
{
    // Calculate global thread index for the output tensor (N, C_out, D_out, H_out, W_out)
    // We flatten the first 3 dimensions (N, C_out, D_out) into the z-dimension of the grid
    // And map H_out, W_out to y and x dimensions.
    
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    
    // blockIdx.z encodes (n * C_out * D_out + c_out * D_out + d_out)
    int n_c_d_idx = blockIdx.z;
    
    int d_out = n_c_d_idx % D_out;
    int c_out = (n_c_d_idx / D_out) % C_out;
    int n = n_c_d_idx / (D_out * C_out);

    if (w_out < W_out && h_out < H_out) {
        float sum = 0.0f;
        
        // Iterate over input channels
        for (int c_in = 0; c_in < C_in; ++c_in) {
            // Iterate over kernel depth
            for (int kd = 0; kd < kD; ++kd) {
                // Iterate over kernel height
                for (int kh = 0; kh < kH; ++kh) {
                    // Iterate over kernel width
                    for (int kw = 0; kw < kW; ++kw) {
                        
                        // Calculate input coordinates based on transposed convolution formula
                        // d_in = (d_out + padD - kd) / strideD
                        // We need to check if the input coordinate is valid
                        
                        int d_in = (d_out + padD - kd);
                        int h_in = (h_out + padH - kh);
                        int w_in = (w_out + padW - kw);

                        // Check if the indices align with the stride
                        if (d_in % strideD == 0 && h_in % strideH == 0 && w_in % strideW == 0) {
                            d_in /= strideD;
                            h_in /= strideH;
                            w_in /= strideW;

                            // Check bounds
                            if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                // Input tensor layout: (N, C_in, D_in, H_in, W_in)
                                int input_idx = ((n * C_in + c_in) * D_in + d_in) * H_in * W_in + h_in * W_in + w_in;
                                
                                // Weight tensor layout: (C_in, C_out, kD, kH, kW)
                                int weight_idx = ((c_in * C_out + c_out) * kD + kd) * kH * kW + kh * kW + kw;
                                
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }

        // Add bias if present
        if (bias != nullptr) {
            sum += bias[c_out];
        }

        // Clamp
        sum = fmaxf(sum, min_val);

        // Division
        float inv_divisor = 1.0f / divisor;
        sum *= inv_divisor;

        // Write to output
        int output_idx = ((n * C_out + c_out) * D_out + d_out) * H_out * W_out + h_out * W_out + w_out;
        output[output_idx] = sum;
    }
}

torch::Tensor fused_conv_transpose_3d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> output_size, 
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    float min_val, float divisor) 
{
    auto output = torch::zeros(output_size, input.options());
    
    const float* bias_ptr = bias_opt.has_value() ? bias_opt.value().data_ptr<float>() : nullptr;

    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int C_out = weight.size(1); // Weight shape (in_channels, out_channels, ...)
    int D_out = output_size[2];
    int H_out = output_size[3];
    int W_out = output_size[4];

    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    // Grid and Block dimensions
    // Block: 16x16 threads for spatial dimensions (W_out, H_out)
    // Grid Z: N * C_out * D_out
    
    dim3 blockDim(16, 16, 1);
    
    // Calculate grid dimensions ensuring we cover the output tensor
    dim3 gridDim(
        (W_out + blockDim.x - 1) / blockDim.x,
        (H_out + blockDim.y - 1) / blockDim.y,
        N * C_out * D_out
    );

    fused_conv_transpose_3d_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        strideD, strideH, strideW,
        padD, padH, padW,
        min_val, divisor
    );

    return output;
}
"""

fused_conv_clamp_div_cpp_source = """
torch::Tensor fused_conv_transpose_3d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> output_size, 
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    float min_val, float divisor);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_conv_clamp_div_cpp_source,
    cuda_sources=fused_conv_clamp_div_source,
    functions=["fused_conv_transpose_3d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for fused ConvTranspose3d + Clamp + Div.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        # We still define the ConvTranspose3d to hold the weights and bias
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        self.fused_ops = fused_ops

    def forward(self, x):
        # Calculate output shape manually to pass to the custom kernel
        # The formula for output shape in ConvTranspose3d is:
        # D_out = (D_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
        # Note: output_padding is assumed 0 here as it's not in the original model init.
        
        N, C_in, D_in, H_in, W_in = x.shape
        C_out = self.conv_transpose.out_channels
        
        # Handle kernel_size, stride, padding being int or tuple
        if isinstance(self.conv_transpose.kernel_size, int):
            kD = kH = kW = self.conv_transpose.kernel_size
        else:
            kD, kH, kW = self.conv_transpose.kernel_size
            
        if isinstance(self.conv_transpose.stride, int):
            sD = sH = sW = self.conv_transpose.stride
        else:
            sD, sH, sW = self.conv_transpose.stride
            
        if isinstance(self.conv_transpose.padding, int):
            pD = pH = pW = self.conv_transpose.padding
        else:
            pD, pH, pW = self.conv_transpose.padding

        D_out = (D_in - 1) * sD - 2 * pD + kD
        H_out = (H_in - 1) * sH - 2 * pH + kH
        W_out = (W_in - 1) * sW - 2 * pW + kW
        
        output_size = (N, C_out, D_out, H_out, W_out)

        # Call the custom fused kernel
        # We pass the weights and bias directly from the conv_transpose layer
        return self.fused_ops.fused_conv_transpose_3d_cuda(
            x, 
            self.conv_transpose.weight, 
            self.conv_transpose.bias,
            output_size,
            sD, sH, sW,
            pD, pH, pW,
            self.min_value, 
            self.divisor
        )