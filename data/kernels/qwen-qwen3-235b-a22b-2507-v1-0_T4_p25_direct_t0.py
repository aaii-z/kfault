import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + Min(dim=1) + Tanh + Tanh
fused_conv_min_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function to get 4D index from flattened index
__device__ int get_index(int n, int c, int h, int w, int C, int H, int W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

__global__ void fused_conv_min_tanh_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int pad,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_height * output_width;
    
    if (idx >= total_elements) return;

    int n = idx / (output_height * output_width);
    int h_out = (idx % (output_height * output_width)) / output_width;
    int w_out = (idx % (output_height * output_width)) % output_width;

    int h_start = h_out * stride - pad;
    int w_start = w_out * stride - pad;
    int h_end = h_start + kernel_size;
    int w_end = w_start + kernel_size;

    // Compute convolution output for all output channels
    float* conv_out = new float[out_channels];
    for (int co = 0; co < out_channels; co++) {
        float acc = bias[co];
        for (int ci = 0; ci < in_channels; ci++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int h = h_start + kh;
                    int w = w_start + kw;
                    if (h >= 0 && h < input_height && w >= 0 && w < input_width) {
                        int in_idx = get_index(n, ci, h, w, in_channels, input_height, input_width);
                        int w_idx = co * in_channels * kernel_size * kernel_size + ci * kernel_size * kernel_size + kh * kernel_size + kw;
                        acc += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        conv_out[co] = acc;
    }

    // Apply min over channel dimension (keepdim=True)
    float min_val = FLT_MAX;
    for (int co = 0; co < out_channels; co++) {
        conv_out[co] = tanhf(tanhf(conv_out[co]));  // fused tanh(tanh(x)) -> tanhf applied twice
        if (conv_out[co] < min_val) {
            min_val = conv_out[co];
        }
    }

    // Output has shape (batch, 1, height, width), so write to single channel
    int out_idx = n * (1 * output_height * output_width) + 0 * (output_height * output_width) + h_out * output_width + w_out;
    output[out_idx] = min_val;

    delete[] conv_out;
}
"""

fused_conv_min_tanh_cpp = """
torch::Tensor fused_conv_min_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int pad,
    int stride
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_min_tanh",
    cpp_sources=[fused_conv_min_tanh_cpp],
    cuda_sources=[fused_conv_min_tanh_source],
    functions=["fused_conv_min_tanh_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv2d + Min(dim=1) + Tanh + Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2
        self.output_height = None
        self.output_width = None

        # Create the convolution weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='tanh')
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        batch_size, _, input_height, input_width = x.shape

        # Compute output dimensions
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Allocate output tensor: (batch_size, 1, output_height, output_width)
        output = torch.empty(batch_size, 1, output_height, output_width, device=x.device, dtype=x.dtype)

        # Launch fused CUDA kernel
        block_size = 256
        total_elements = batch_size * output_height * output_width
        grid_size = (total_elements + block_size - 1) // block_size

        fused_op.fused_conv_min_tanh_cuda(
            x,
            self.weight,
            self.bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            self.kernel_size,
            self.padding,
            self.stride
        )

        # Call the CUDA kernel
        fused_op.fused_conv_min_tanh_cuda(
            x, self.weight, self.bias,
            batch_size, self.in_channels, self.out_channels,
            input_height, input_width, output_height, output_width,
            self.kernel_size, self.padding, self.stride
        )

        return output