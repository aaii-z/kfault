import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

logsumexp_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void logsumexp_3d_kernel(
    const float* x, float* y,
    int batch, int channels, int depth, int height, int width,
    int stride_b, int stride_c, int stride_d, int stride_h, int stride_w,
    int out_stride_b, int out_stride_d, int out_stride_h, int out_stride_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * depth * height * width) return;

    int b = idx / (depth * height * width);
    int rem = idx % (depth * height * width);
    int d = rem / (height * width);
    int rem2 = rem % (height * width);
    int h = rem2 / width;
    int w = rem2 % width;

    float max_val = -FLT_MAX;
    for (int c = 0; c < channels; c++) {
        int x_index = b * stride_b + c * stride_c + d * stride_d + h * stride_h + w * stride_w;
        float val = x[x_index];
        if (val > max_val) {
            max_val = val;
        }
    }

    if (max_val == -FLT_MAX) {
        int y_index = b * out_stride_b + d * out_stride_d + h * out_stride_h + w * out_stride_w;
        y[y_index] = -FLT_MAX;
    } else {
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; c++) {
            int x_index = b * stride_b + c * stride_c + d * stride_d + h * stride_h + w * stride_w;
            float val = x[x_index];
            sum_exp += expf(val - max_val);
        }
        float result = logf(sum_exp) + max_val;
        int y_index = b * out_stride_b + d * out_stride_d + h * out_stride_h + w * out_stride_w;
        y[y_index] = result;
    }
}

torch::Tensor logsumexp_3d_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 5, "Input must be 5D tensor");
    int batch = x.size(0);
    int channels = x.size(1);
    int depth = x.size(2);
    int height = x.size(3);
    int width = x.size(4);
    
    auto x_strides = x.strides();
    int stride_b = x_strides[0];
    int stride_c = x_strides[1];
    int stride_d = x_strides[2];
    int stride_h = x_strides[3];
    int stride_w = x_strides[4];
    
    auto y = torch::empty({batch, 1, depth, height, width}, x.options());
    auto y_strides = y.strides();
    int out_stride_b = y_strides[0];
    int out_stride_d = y_strides[2];
    int out_stride_h = y_strides[3];
    int out_stride_w = y_strides[4];
    
    const int size = batch * depth * height * width;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    logsumexp_3d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(),
        batch, channels, depth, height, width,
        stride_b, stride_c, stride_d, stride_h, stride_w,
        out_stride_b, out_stride_d, out_stride_h, out_stride_w
    );
    
    return y;
}
"""

logsumexp_3d_cpp = "torch::Tensor logsumexp_3d_cuda(torch::Tensor x);"

hard_swish_sub_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void hard_swish_sub_clamp_kernel(const float* x, float* out, float bias_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-(val + 3.0f)));
        float hard_swish = val * sigmoid_val / 6.0f;
        float temp = hard_swish - bias_val;
        if (temp < -1.0f) temp = -1.0f;
        else if (temp > 1.0f) temp = 1.0f;
        out[idx] = temp;
    }
}

torch::Tensor hard_swish_sub_clamp_cuda(torch::Tensor x, float bias_val) {
    auto y = torch::empty_like(x);
    const int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    hard_swish_sub_clamp_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), bias_val, size
    );
    
    return y;
}
"""

hard_swish_sub_clamp_cpp = "torch::Tensor hard_swish_sub_clamp_cuda(torch::Tensor x, float bias_val);"

logsumexp_3d = load_inline(
    name="logsumexp_3d",
    cpp_sources=logsumexp_3d_cpp,
    cuda_sources=logsumexp_3d_source,
    functions=["logsumexp_3d_cuda"],
    verbose=False,
    extra_cflags=["-O3"]
)

hard_swish_sub_clamp = load_inline(
    name="hard_swish_sub_clamp",
    cpp_sources=hard_swish_sub_clamp_cpp,
    cuda_sources=hard_swish_sub_clamp_source,
    functions=["hard_swish_sub_clamp_cuda"],
    verbose=False,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operators for logsumexp and fused hard_swish/subtract/clamp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.logsumexp_3d = logsumexp_3d
        self.hard_swish_sub_clamp = hard_swish_sub_clamp

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.logsumexp_3d.logsumexp_3d_cuda(x)
        bias_val = self.bias.item()
        x = self.hard_swish_sub_clamp.hard_swish_sub_clamp_cuda(x, bias_val)
        return x