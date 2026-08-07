import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused 3x3 convolution, division by constant, and leaky_relu
fused_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_div_leaky_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float divisor,
    int N, int C_in, int H, int W, int C_out, int H_out, int W_out) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;
    if (idx >= total_elements) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    float sum = bias[c_out];
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int h_in = h + kh;
                int w_in = w + kw;
                int input_idx = n * C_in * H * W + c_in * H * W + h_in * W + w_in;
                int weight_idx = c_out * (C_in * 9) + c_in * 9 + kh * 3 + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    // Apply division by constant and leaky_relu in one step
    sum = sum / divisor;
    output[idx] = (sum > 0.0f) ? sum : 0.01f * sum;
}

torch::Tensor fused_conv_div_leaky_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor) {
    
    // Validate input dimensions
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OIHW)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(weight.size(2) == 3 && weight.size(3) == 3, "Only 3x3 kernels supported");
    
    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(0);
    
    // Calculate output dimensions for 3x3 conv with stride=1, padding=0
    int H_out = H - 2;
    int W_out = W - 2;
    
    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());
    int total_elements = N * C_out * H_out * W_out;
    
    if (total_elements == 0) {
        return output;
    }

    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_div_leaky_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor,
        N, C_in, H, W, C_out, H_out, W_out
    );
    
    return output;
}
"""

fused_conv_cpp = "torch::Tensor fused_conv_div_leaky_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);"

# Compile the custom CUDA operator
fused_conv_op = load_inline(
    name="fused_conv_op",
    cpp_sources=fused_conv_cpp,
    cuda_sources=fused_conv_source,
    functions=["fused_conv_div_leaky_relu_cuda"],
    verbose=False,
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses convolution, division by constant, and LeakyReLU into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        # Store divisor for use in forward pass
        self.divisor = divisor
        # Initialize convolution layer (weights/bias will be used directly in custom op)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        
    def forward(self, x):
        # Use custom fused operator instead of separate operations
        return fused_conv_op.fused_conv_div_leaky_relu_cuda(
            x, 
            self.conv.weight, 
            self.conv.bias, 
            self.divisor
        )