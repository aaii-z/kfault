import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused transposed convolution, max pooling, hardtanh, mean, and tanh
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Compute output spatial dimensions for transposed convolution
int compute_output_size(int input_size, int padding, int kernel_size, int stride) {
    return (input_size - 1) * stride - 2 * padding + kernel_size;
}

__global__ void fused_conv_transpose_maxpool_hardtanh_mean_tanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size, int in_channels, int input_height, int input_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    int pool_kernel_size, int pool_stride,
    float hardtanh_min, float hardtanh_max
) {
    int batch = blockIdx.x;
    int chan = blockIdx.y;

    if (batch >= batch_size || chan >= out_channels) return;

    int pooled_out_h = (out_height + pool_stride - 1) / pool_stride;
    int pooled_out_w = (out_width + pool_stride - 1) / pool_stride;
    int pool_area = pooled_out_h * pooled_out_w;

    float sum = 0.0f;

    for (int ph = 0; ph < pooled_out_h; ++ph) {
        for (int pw = 0; pw < pooled_out_w; ++pw) {
            float max_val = -FLT_MAX;

            int h_start = ph * pool_stride;
            int w_start = pw * pool_stride;

            for (int ph_k = 0; ph_k < pool_kernel_size; ++ph_k) {
                int h = h_start + ph_k;
                if (h >= out_height) continue;

                for (int pw_k = 0; pw_k < pool_kernel_size; ++pw_k) {
                    int w = w_start + pw_k;
                    if (w >= out_width) continue;

                    float val = 0.0f;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            int ih = h * stride - padding + kh;
                            if (ih < 0 || ih >= input_height) continue;
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int iw = w * stride - padding + kw;
                                if (iw < 0 || iw >= input_width) continue;

                                int in_idx = batch * in_channels * input_height * input_width +
                                           ic * input_height * input_width +
                                           ih * input_width + iw;
                                int w_idx = chan * in_channels * kernel_size * kernel_size +
                                          ic * kernel_size * kernel_size +
                                          kh * kernel_size + kw;
                                val += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    val += bias[chan];
                    val = fmaxf(hardtanh_min, fminf(hardtanh_max, val));
                    max_val = fmaxf(max_val, val);
                }
            }
            sum += max_val;
        }
    }

    float mean_val = sum / pool_area;
    mean_val = tanhf(mean_val);

    int out_idx = batch * out_channels + chan;
    output[out_idx] = mean_val;
}

// C++ wrapper function that gets called from Python
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int input_height, int input_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    int pool_kernel_size, int pool_stride,
    float hardtanh_min, float hardtanh_max
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    int batch_size = input.size(0);
    auto output = torch::empty({batch_size * out_channels}, options);

    dim3 grid(batch_size, out_channels);
    dim3 block(1);

    fused_conv_transpose_maxpool_hardtanh_mean_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, input_height, input_width,
        out_channels, out_height, out_width,
        kernel_size, stride, padding,
        pool_kernel_size, pool_stride,
        hardtanh_min, hardtanh_max
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int input_height, int input_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    int pool_kernel_size, int pool_stride,
    float hardtanh_min, float hardtanh_max
);
"""

# Compile the fused CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_op_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for:
    ConvTranspose2d -> MaxPool2d -> Hardtanh -> Mean -> Tanh
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        # Define transposed convolution parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

        # Store fused CUDA op
        self.fused_op = fused_op

    def forward(self, x):
        batch_size, _, input_height, input_width = x.shape

        # Compute output size of ConvTranspose2d
        out_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size

        # Launch fused kernel
        output = self.fused_op.fused_op_cuda(
            x, self.weight, self.bias,
            self.in_channels, input_height, input_width,
            self.out_channels, out_height, out_width,
            self.kernel_size, self.stride, self.padding,
            self.maxpool_kernel_size, self.maxpool_stride,
            self.hardtanh_min, self.hardtanh_max
        )

        # Reshape to (batch, out_channels, 1, 1)
        output = output.view(batch_size, self.out_channels, 1, 1)
        return output