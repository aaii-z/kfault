import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d + Tanh + Scale + Bias + MaxPool2d
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Utility function to get 4D index from linear index
__device__ int get_index(int n, int c, int h, int w, int C, int H, int W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

__global__ void fused_conv_tanh_scale_bias_pool_kernel(
    const float* input,
    const float* weight,
    const float* bias_vec,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int padding, int stride,
    float scaling_factor,
    int pool_kernel_size, int pool_stride) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_elements) return;

    // Decode output index
    int w_out = out_idx % out_width;
    int h_out = (out_idx / out_width) % out_height;
    int c_out = (out_idx / (out_width * out_height)) % out_channels;
    int n = out_idx / (out_width * out_height * out_channels);

    // Compute max-pool input region (before activation)
    int h_pool_start = h_out * pool_stride;
    int w_pool_start = w_out * pool_stride;
    int h_pool_end = min(h_pool_start + pool_kernel_size, in_height - 2 * padding);
    int w_pool_end = min(w_pool_start + pool_kernel_size, in_width - 2 * padding);

    // Perform convolution (with padding assumed to be kernel_size//2 = 1)
    float conv_val = 0.0f;
    int pad = kernel_size / 2;

    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_pool_start + kh;
                int w_in = w_pool_start + kw;

                if (h_in >= pad && h_in < in_height - pad && w_in >= pad && w_in < in_width - pad) {
                    int h_conv = h_in - pad;
                    int w_conv = w_in - pad;
                    int in_idx = get_index(n, c_in, h_conv, w_conv, in_channels, in_height, in_width);
                    int weight_idx = get_index(c_out, c_in, kh, kw, in_channels, kernel_size, kernel_size);
                    conv_val += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias (from conv layer bias will be handled in bias_vec separately)
    conv_val += bias_vec[c_out];

    // Apply tanh, scaling, and then max-pool over the pool region
    float max_val = -FLT_MAX;

    for (int ph = h_pool_start; ph < h_pool_end; ph++) {
        for (int pw = w_pool_start; pw < w_pool_end; pw++) {
            float val = 0.0f;
            for (int c_in = 0; c_in < in_channels; c_in++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int h_in = ph + kh - pad;
                        int w_in = pw + kw - pad;
                        if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                            int h_conv = h_in;
                            int w_conv = w_in;
                            int in_idx = get_index(n, c_in, h_conv, w_conv, in_channels, in_height, in_width);
                            int weight_idx = get_index(c_out, c_in, kh, kw, in_channels, kernel_size, kernel_size);
                            val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            val += bias_vec[c_out];
            val = tanhf(val);           // tanh activation
            val = val * scaling_factor; // scaling
            val = val + bias_vec[c_out]; // add bias term (assuming same bias reused or separate)
            if (val > max_val) max_val = val;
        }
    }

    output[out_idx] = max_val;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_tanh_scale_bias_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias_vec,
    float scaling_factor,
    int kernel_size,
    int padding,
    int stride,
    int pool_kernel_size,
    int pool_stride,
    int out_height,
    int out_width);
"""

# Note: The above kernel is a simplified version. In practice, we would separate the conv and pool.
# But for fusion, we aim to fuse all operations.

# Actually, fusing conv and pooling with activation in between is non-trivial due to data dependencies.
# Instead, we do: Conv -> Tanh -> Scale -> Bias -> MaxPool as separate kernels but fuse Tanh+Scale+Bias+MaxPool
# But even better: we can fuse the pointwise operations and use PyTorch's optimized conv and maxpool,
# or write a custom fused kernel for the entire pipeline.

# However, due to complexity, we instead choose to fuse Tanh + Scale + Bias + MaxPool
# But MaxPool cannot be fused directly unless we recompute conv.

# Alternative: We keep Conv and MaxPool as PyTorch ops, but fuse Tanh + Scale + Bias into one CUDA kernel.

tanh_scale_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_scale_bias_kernel(const float* x, float* out, const float* bias, float scale, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C * H * W) {
        int c = (idx / (H * W)) % C;
        out[idx] = tanhf(x[idx]) * scale + bias[c];
    }
}
"""

tanh_scale_bias_cpp = """
torch::Tensor tanh_scale_bias_cuda(torch::Tensor x, torch::Tensor bias, float scale);
"""

tanh_scale_bias_module = load_inline(
    name="tanh_scale_bias",
    cpp_sources=tanh_scale_bias_cpp,
    cuda_sources=tanh_scale_bias_source,
    functions=["tanh_scale_bias_cuda"],
    verbose=False,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.tanh_scale_bias_op = tanh_scale_bias_module.tanh_scale_bias_cuda

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh_scale_bias_op(x, self.bias, self.scaling_factor)
        x = self.max_pool(x)
        return x