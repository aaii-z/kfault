import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + Softplus + Tanh + Multiply (Swish-like activation)
fused_conv_softplus_tanh_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_THREADS 1024

__device__ float softplus(float x) {
    const float threshold = 20.0f;
    if (x > threshold) {
        return x;
    }
    return log1pf(expf(x));
}

__global__ void fused_conv_softplus_tanh_mul_forward(
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
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_elements) return;

    int n = idx / (out_channels * output_height * output_width);
    int remainder = idx % (out_channels * output_height * output_width);
    int c = remainder / (output_height * output_width);
    remainder = remainder % (output_height * output_width);
    int h = remainder / output_width;
    int w = remainder % output_width;

    int input_h = input_height;
    int input_w = input_width;

    float val = 0.0f;
    int weight_h = kernel_size;
    int weight_w = kernel_size;

    for (int ci = 0; ci < in_channels; ++ci) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = h * stride - pad + kh;
                int iw = w * stride - pad + kw;

                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                    int input_idx = n * in_channels * input_h * input_w +
                                    ci * input_h * input_w +
                                    ih * input_w + iw;
                    int weight_idx = c * in_channels * kernel_size * kernel_size +
                                     ci * kernel_size * kernel_size +
                                     kh * kernel_size + kw;
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias
    val += bias[c];

    // Apply tanh(softplus(x)) * x (Swish-like with tanh(softplus))
    float sp = softplus(val);
    float tanh_sp = tanhf(sp);
    output[idx] = tanh_sp * val;
}

torch::Tensor fused_conv_softplus_tanh_mul_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int pad,
    int stride,
    int output_height,
    int output_width
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    int total_elements = batch_size * out_channels * output_height * output_width;
    int block_size = MAX_THREADS;
    int grid_size = (total_elements + block_size - 1) / block_size;

    fused_conv_softplus_tanh_mul_forward<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        pad,
        stride
    );

    return output;
}
"""

fused_conv_softplus_tanh_mul_cpp_source = """
torch::Tensor fused_conv_softplus_tanh_mul_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int pad,
    int stride,
    int output_height,
    int output_width);
"""

# Compile the inline CUDA kernel
fused_conv_op = load_inline(
    name="fused_conv_softplus_tanh_mul",
    cpp_sources=fused_conv_softplus_tanh_mul_cpp_source,
    cuda_sources=fused_conv_softplus_tanh_mul_source,
    functions=["fused_conv_softplus_tanh_mul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Conv2d + tanh(softplus(x)) * x using custom CUDA kernel.
    BatchNorm2d remains as-is since it's already efficient and hard to beat without cuDNN fusion.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = (kernel_size - 1) // 2
        self.eps = eps
        self.momentum = momentum

        # Define parameters manually to avoid using nn.Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        # BatchNorm layer remains standard
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

        # Bind the compiled CUDA function
        self.fused_conv_act = fused_conv_op.fused_conv_softplus_tanh_mul_cuda

    def forward(self, x):
        batch_size, _, input_height, input_width = x.shape
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Use fused CUDA kernel for conv + tanh(softplus(x)) * x
        x = self.fused_conv_act(
            x,
            self.weight,
            self.bias,
            self.padding,
            self.stride,
            output_height,
            output_width
        )

        # Apply BatchNorm (already optimized in PyTorch)
        x = self.bn(x)
        return x