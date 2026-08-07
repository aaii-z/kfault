import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused transposed convolution, add, clamp, GELU, and multiply
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU activation approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__device__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void fused_convtranspose_add_clamp_gelu_mul_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    float add_value,
    float multiply_value
) {
    int total_elements = batch_size * out_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Decode index
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);

    float sum = 0.0f;

    // ConvTranspose2d computation
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = h - kh * stride;
            int iw = w - kw * stride;
            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    int input_idx = n * in_channels * in_height * in_width +
                                    ic * in_height * in_width +
                                    ih * in_width + iw;
                    int weight_idx = c * in_channels * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size +
                                     kh * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias
    sum += bias[c];

    // Add constant value
    sum += add_value;

    // Clamp to min(0.0) -> this is min(x, 0), so negative clamping
    sum = fminf(sum, 0.0f);

    // GELU activation
    sum = gelu(sum);

    // Multiply by constant
    sum *= multiply_value;

    output[idx] = sum;
}

torch::Tensor fused_convtranspose_add_clamp_gelu_mul_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    float add_value,
    float multiply_value) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());

    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_convtranspose_add_clamp_gelu_mul_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        stride,
        add_value,
        multiply_value
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_convtranspose_add_clamp_gelu_mul_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    float add_value,
    float multiply_value);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_convtranspose_add_clamp_gelu_mul",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_convtranspose_add_clamp_gelu_mul_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for transposed convolution, add, clamp, GELU, and multiply.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.add_value = add_value
        self.multiply_value = multiply_value

        # Initialize transposed convolution weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

        # Register the fused CUDA operation
        self.fused_op = fused_op.fused_convtranspose_add_clamp_gelu_mul_cuda

    def forward(self, x):
        batch_size, _, in_height, in_width = x.shape

        # Calculate output spatial dimensions (same as ConvTranspose2d)
        out_height = (in_height - 1) * self.stride + self.kernel_size
        out_width = (in_width - 1) * self.stride + self.kernel_size

        # Launch fused CUDA kernel
        output = self.fused_op(
            x,
            self.weight,
            self.bias,
            in_height,
            in_width,
            out_height,
            out_width,
            self.kernel_size,
            self.stride,
            self.add_value,
            self.multiply_value
        )

        return output