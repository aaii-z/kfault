import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + BatchNorm2d + scaling
fused_conv_bn_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function to calculate 4D index from flattened index
__device__ int get_index(int n, int c, int h, int w, int C, int H, int W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

__global__ void fused_conv_bn_scale_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* bn_running_mean,
    const float* bn_running_var,
    const float* bn_weight,
    const float* bn_bias,
    float* output,
    float scaling_factor,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (output_width * output_height * out_channels);

    int pad = kernel_size / 2;
    float sum = 0.0f;

    // Perform convolution
    for (int ki = 0; ki < kernel_size; ++ki) {
        for (int kj = 0; kj < kernel_size; ++kj) {
            for (int ic = 0; ic < in_channels; ++ic) {
                int i_h = h + pad - ki;
                int i_w = w + pad - kj;

                if (i_h >= 0 && i_h < input_height && i_w >= 0 && i_w < input_width) {
                    int in_idx = get_index(n, ic, i_h, i_w, in_channels, input_height, input_width);
                    int weight_idx = c * in_channels * kernel_size * kernel_size + 
                                     ic * kernel_size * kernel_size + 
                                     ki * kernel_size + kj;
                    sum += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias
    sum += bias[c];

    // Apply BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    float mean = bn_running_mean[c];
    float inv_std = 1.0f / sqrtf(bn_running_var[c] + eps);
    float bn_output = (sum - mean) * inv_std;
    bn_output = bn_output * bn_weight[c] + bn_bias[c];

    // Apply scaling
    bn_output *= scaling_factor;

    // Write output
    int out_idx = get_index(n, c, h, w, out_channels, output_height, output_width);
    output[out_idx] = bn_output;
}

torch::Tensor fused_conv_bn_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    float scaling_factor) {
    
    // Validate input
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int output_height = input_height;
    int output_width = input.size(3);  // Fixed: was incorrectly using .width()

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, 
                               input.options());

    const int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Launch kernel
    fused_conv_bn_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size,
        1e-5f  // eps
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_conv_bn_scale_cpp_source = """
torch::Tensor fused_conv_bn_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    float scaling_factor);
"""

# Compile the inline CUDA kernel
fused_conv_bn_scale = load_inline(
    name="fused_conv_bn_scale",
    cpp_sources=[fused_conv_bn_scale_cpp_source],
    cuda_sources=[fused_conv_bn_scale_source],
    functions=["fused_conv_bn_scale_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with fused Conv2d + BatchNorm2d + scaling in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = scaling_factor
        
        # Define the same layers to retain parameters
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Register the fused CUDA function
        self.fused_op = fused_conv_bn_scale

    def forward(self, x):
        # Extract all necessary parameters
        weight = self.conv.weight
        bias = self.conv.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        scaling_factor = self.scaling_factor

        # Launch fused CUDA kernel
        return self.fused_op.fused_conv_bn_scale_cuda(
            x, weight, bias, running_mean, running_var, bn_weight, bn_bias, scaling_factor
        )