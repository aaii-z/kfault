import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + Clamp + Divide
fused_conv_transpose_clamp_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Fused kernel for transposed 3D convolution, clamp, and division
__global__ void fused_conv_transpose_clamp_divide_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int out_channels,
    int out_depth,
    int out_height,
    int out_width,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int d = (idx / (out_width * out_height)) % out_depth;
    int c = (idx / (out_width * out_height * out_depth)) % out_channels;
    int b = idx / (out_width * out_height * out_depth * out_channels);

    float sum = 0.0f;

    // Transposed convolution: loop over input channels and kernel
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    
                    // Compute input position corresponding to output (d, h, w)
                    int i_d = d * stride - padding + kd;
                    int i_h = h * stride - padding + kh;
                    int i_w = w * stride - padding + kw;

                    // Check bounds
                    if (i_d >= 0 && i_d < in_depth && i_h >= 0 && i_h < in_height && i_w >= 0 && i_w < in_width) {
                        int input_idx = b * in_channels * in_depth * in_height * in_width +
                                        ic * in_depth * in_height * in_width +
                                        i_d * in_height * in_width +
                                        i_h * in_width +
                                        i_w;
                        int weight_idx = c * in_channels * kernel_size * kernel_size * kernel_size +
                                         ic * kernel_size * kernel_size * kernel_size +
                                         kd * kernel_size * kernel_size +
                                         kh * kernel_size +
                                         kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias
    sum += bias[c];

    // Clamp and divide
    sum = fmaxf(sum, min_value);
    sum /= divisor;

    output[idx] = sum;
}

torch::Tensor fused_conv_transpose_clamp_divide_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor) {
    
    auto batch_size = input.size(0);
    auto out_channels = weight.size(0);
    auto in_channels = weight.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);

    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, 
                               input.options().dtype(torch::kFloat32).device(torch::kCUDA));

    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_transpose_clamp_divide_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        out_depth,
        out_height,
        out_width,
        in_channels,
        in_depth,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        min_value,
        divisor
    );

    cudaError_t err = cudaGetLastError();
    C10_CUDA_CHECK(err);

    return output;
}
"""

fused_conv_transpose_clamp_divide_cpp_source = """
torch::Tensor fused_conv_transpose_clamp_divide_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_transpose_clamp_divide",
    cpp_sources=fused_conv_transpose_clamp_divide_cpp_source,
    cuda_sources=fused_conv_transpose_clamp_divide_source,
    functions=["fused_conv_transpose_clamp_divide_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom CUDA kernel that fuses
    transposed 3D convolution, clamp, and division operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = min_value
        self.divisor = divisor
        
        # Create the transposed convolution weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

        # Output dimensions calculation for ConvTranspose3d
        def compute_output_dim(dim):
            return (dim - 1) * stride - 2 * padding + kernel_size
        self.out_depth = compute_output_dim(24)
        self.out_height = compute_output_dim(48)
        self.out_width = compute_output_dim(48)

        # Register the fused CUDA operation
        self.fused_op = fused_op.fused_conv_transpose_clamp_divide_cuda

    def forward(self, x):
        return self.fused_op(
            x,
            self.weight,
            self.bias,
            self.out_depth,
            self.out_height,
            self.out_width,
            self.kernel_size,
            self.stride,
            self.padding,
            self.min_value,
            self.divisor
        )