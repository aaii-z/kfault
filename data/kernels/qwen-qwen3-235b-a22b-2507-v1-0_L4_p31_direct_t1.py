import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d + Min + Add Bias + Multiply Scaling Factor
fused_conv_min_add_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utility function to get 4D index: (n, c, h, w)
__device__ int get_index(int n, int c, int H, int W, int h, int w) {
    return ((n * H + h) * W + w) * 1 + c;
}

__global__ void fused_conv_min_add_mul_kernel(
    const float* input,
    const float* weight,
    const float* bias_vec,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    float constant_value,
    float scaling_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int c_out = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (output_width * output_height * out_channels);

    int k_half = kernel_size / 2;

    // Perform convolution
    float conv_val = 0.0f;
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int in_h = h + i - k_half;
                int in_w = w + j - k_half;

                if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                    int in_idx = get_index(n, c_in, input_height, input_width, in_h, in_w);
                    int weight_idx = (c_out * in_channels + c_in) * kernel_size * kernel_size + (i * kernel_size + j);
                    conv_val += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias (bias shape: [out_channels, 1, 1])
    conv_val += bias_vec[c_out];

    // Apply min(clamp) with constant_value
    conv_val = fminf(conv_val, constant_value);

    // Scale
    conv_val *= scaling_factor;

    // Write output
    int out_idx = get_index(n, c_out, output_height, output_width, h, w);
    output[out_idx] = conv_val;
}

torch::Tensor fused_conv_min_add_mul_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_vec,
    float constant_value,
    float scaling_factor
) {
    // Input validation
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias_vec);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    // Correctly compute output height and width for conv with stride=1, padding=0, dilation=1
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;

    auto output = torch::zeros(
        {batch_size, out_channels, output_height, output_width},
        input.options()
    );

    int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_min_add_mul_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_vec.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        constant_value,
        scaling_factor
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}
"""

fused_conv_min_add_mul_cpp_source = """
torch::Tensor fused_conv_min_add_mul_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_vec,
    float constant_value,
    float scaling_factor
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_min_add_mul",
    cpp_sources=fused_conv_min_add_mul_cpp_source,
    cuda_sources=fused_conv_min_add_mul_source,
    functions=["fused_conv_min_add_mul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model using a fused CUDA kernel for Conv2d + Min + Add Bias + Scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight)

        # Fused CUDA function
        self.fused_op = fused_op.fused_conv_min_add_mul_cuda

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias.squeeze(), self.constant_value, self.scaling_factor)