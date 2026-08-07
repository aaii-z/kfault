import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LeakyReLU + Mul + LeakyReLU + MaxPool3d
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void fused_kernel(const float* __restrict__ input, 
                             const float* __restrict__ multiplier, 
                             float* __restrict__ output, 
                             int N, int C, 
                             int D_in, int H_in, int W_in, 
                             int D_out, int H_out, int W_out, 
                             float negative_slope) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D_out * H_out * W_out;

    if (idx < total_elements) {
        // Decompose linear index into (n, c, d_out, h_out, w_out)
        // Layout is NCDHW
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int d_out = (idx / (W_out * H_out)) % D_out;
        int c = (idx / (W_out * H_out * D_out)) % C;
        int n = idx / (W_out * H_out * D_out * C);

        float max_val = -1e20f; // Sufficiently small
        float mult = multiplier[c];

        // MaxPool3d with kernel_size=2, stride=2
        // Input window: (d_out*2 : d_out*2+2, ...)
        for (int kd = 0; kd < 2; ++kd) {
            for (int kh = 0; kh < 2; ++kh) {
                for (int kw = 0; kw < 2; ++kw) {
                    int d_in = d_out * 2 + kd;
                    int h_in = h_out * 2 + kh;
                    int w_in = w_out * 2 + kw;

                    // Boundary check (though dimensions are even here)
                    if (d_in < D_in && h_in < H_in && w_in < W_in) {
                        // Calculate input index assuming contiguous NCDHW
                        long long in_idx = (long long)n * C * D_in * H_in * W_in + 
                                           (long long)c * D_in * H_in * W_in + 
                                           (long long)d_in * H_in * W_in + 
                                           (long long)h_in * W_in + 
                                           w_in;
                        
                        float val = input[in_idx];
                        
                        // LeakyReLU 1
                        val = val > 0 ? val : val * negative_slope;
                        
                        // Multiply
                        val = val * mult;
                        
                        // LeakyReLU 2
                        val = val > 0 ? val : val * negative_slope;
                        
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
            }
        }
        
        // Write output
        output[idx] = max_val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor multiplier, float negative_slope) {
    // Input shape: (N, C, D_in, H_in, W_in)
    // Multiplier shape: (C, 1, 1, 1)
    
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    // MaxPool3d with kernel=2, stride=2
    int D_out = D_in / 2;
    int H_out = H_in / 2;
    int W_out = W_in / 2;

    auto output = torch::zeros({N, C, D_out, H_out, W_out}, input.options());

    int total_elements = N * C * D_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        multiplier.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N, C, D_in, H_in, W_in, 
        D_out, H_out, W_out, 
        negative_slope
    );

    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor multiplier, float negative_slope);"
)

# Compile the inline CUDA code
fused_op_module = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution followed by a fused
    LeakyReLU -> Mul -> LeakyReLU -> MaxPool3d operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.fused_op = fused_op_module
        self.negative_slope = 0.2

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused LeakyReLU -> Mul -> LeakyReLU -> MaxPool3d
        x = self.fused_op.fused_op_cuda(x, self.multiplier, self.negative_slope)
        return x