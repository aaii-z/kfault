import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused conv2d, subtract, tanh, subtract, and avgpool
fused_conv_tanh_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel for fused Conv2d -> Sub1 -> Tanh -> Sub2 -> AvgPool2d
__global__ void fused_conv_tanh_pool_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size_h, int kernel_size_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int pool_kernel_h, int pool_kernel_w,
    float subtract1_value,
    float subtract2_value,
    int output_height_pooled,
    int output_width_pooled
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height_pooled * output_width_pooled;
    
    if (idx >= total_elements) return;

    // Decode index
    int w_out_p = idx % output_width_pooled;
    int h_out_p = (idx / output_width_pooled) % output_height_pooled;
    int c_out = (idx / (output_width_pooled * output_height_pooled)) % out_channels;
    int n = idx / (output_width_pooled * output_height_pooled * out_channels);

    // Pooling region start in unpooled output space
    int h_start = h_out_p * pool_kernel_h;
    int w_start = w_out_p * pool_kernel_w;
    int h_end = min(h_start + pool_kernel_h, out_height);
    int w_end = min(w_start + pool_kernel_w, out_width);

    float sum = 0.0f;
    int count = 0;

    for (int ph = h_start; ph < h_end; ++ph) {
        for (int pw = w_start; pw < w_end; ++pw) {
            float conv_val = 0.0f;

            // Convolution computation at (ph, pw)
            for (int k = 0; k < in_channels; ++k) {
                for (int kh = 0; kh < kernel_size_h; ++kh) {
                    for (int kw = 0; kw < kernel_size_w; ++kw) {
                        int h_in = ph * stride_h - padding_h + kh * dilation_h;
                        int w_in = pw * stride_w - padding_w + kw * dilation_w;

                        if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                            int in_idx = ((n * in_channels + k) * in_height + h_in) * in_width + w_in;
                            int weight_idx = ((c_out * in_channels + k) * kernel_size_h + kh) * kernel_size_w + kw;
                            conv_val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }

            // Add bias
            conv_val += bias[c_out];

            // Subtract1, Tanh, Subtract2
            conv_val = tanhf(conv_val - subtract1_value) - subtract2_value;

            sum += conv_val;
            ++count;
        }
    }

    output[idx] = sum / count;
}

torch::Tensor fused_conv_tanh_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract1_value,
    float subtract2_value,
    int kernel_size_h,
    int kernel_size_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int pool_kernel_h,
    int pool_kernel_w
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);

    const auto out_channels = weight.size(0);
    const auto kernel_size_h_ = weight.size(2);
    const auto kernel_size_w_ = weight.size(3);

    // Output size after convolution
    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_size_h_ - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_size_w_ - 1) - 1) / stride_w + 1;

    // Output size after average pooling
    int output_height_pooled = (out_height - pool_kernel_h) / pool_kernel_h + 1;
    int output_width_pooled = (out_width - pool_kernel_w) / pool_kernel_w + 1;

    auto output = torch::empty(
        {batch_size, out_channels, output_height_pooled, output_width_pooled},
        input.options()
    );

    const int total_elements = batch_size * out_channels * output_height_pooled * output_width_pooled;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_tanh_pool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_size_h_, kernel_size_w_,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        pool_kernel_h, pool_kernel_w,
        subtract1_value,
        subtract2_value,
        output_height_pooled,
        output_width_pooled
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_conv_tanh_pool_cpp_source = """
torch::Tensor fused_conv_tanh_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract1_value,
    float subtract2_value,
    int kernel_size_h,
    int kernel_size_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int pool_kernel_h,
    int pool_kernel_w
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_tanh_pool",
    cpp_sources=fused_conv_tanh_pool_cpp_source,
    cuda_sources=fused_conv_tanh_pool_source,
    functions=["fused_conv_tanh_pool_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv2d -> Sub -> Tanh -> Sub -> AvgPool2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

        # Create the convolution parameters manually
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='tanh')
        nn.init.zeros_(self.bias)

        # Conv parameters (assuming default values used by PyTorch Conv2d)
        self.stride = (1, 1)
        self.padding = (kernel_size // 2, kernel_size // 2)  # Standard padding for preserving size
        self.dilation = (1, 1)

        # Bind the fused CUDA function
        self.fused_op = fused_op.fused_conv_tanh_pool_cuda

    def forward(self, x):
        return self.fused_op(
            x,
            self.weight,
            self.bias,
            self.subtract1_value,
            self.subtract2_value,
            self.kernel_size,
            self.kernel_size,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.kernel_size_pool,
            self.kernel_size_pool
        )