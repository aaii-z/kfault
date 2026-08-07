import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused MaxPool3D
max_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void max_pool3d_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    int kernel_size,
    int stride
) {
    int output_d = (depth - kernel_size) / stride + 1;
    int output_h = (height - kernel_size) / stride + 1;
    int output_w = (width - kernel_size) / stride + 1;
    
    int output_size = batch_size * channels * output_d * output_h * output_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        int n = idx / (channels * output_d * output_h * output_w);
        int c = (idx / (output_d * output_h * output_w)) % channels;
        int d = (idx / (output_h * output_w)) % output_d;
        int h = (idx / output_w) % output_h;
        int w = idx % output_w;

        int start_d = d * stride;
        int start_h = h * stride;
        int start_w = w * stride;

        float max_val = -FLT_MAX;
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int curr_d = start_d + kd;
                    int curr_h = start_h + kh;
                    int curr_w = start_w + kw;
                    if (curr_d < depth && curr_h < height && curr_w < width) {
                        int input_idx = n * channels * depth * height * width + 
                                      c * depth * height * width + 
                                      curr_d * height * width + 
                                      curr_h * width + 
                                      curr_w;
                        max_val = fmaxf(max_val, input[input_idx]);
                    }
                }
            }
        }

        output[idx] = max_val;
    }
}

torch::Tensor max_pool3d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride
) {
    CHECK_INPUT(input);

    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    int output_d = (depth - kernel_size) / stride + 1;
    int output_h = (height - kernel_size) / stride + 1;
    int output_w = (width - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, output_d, output_h, output_w}, input.options());

    int output_size = batch_size * channels * output_d * output_h * output_w;
    int block_size = 256;
    int grid_size = (output_size + block_size - 1) / block_size;

    max_pool3d_forward_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        kernel_size,
        stride
    );

    return output;
}
"""

max_pool_cpp_source = "torch::Tensor max_pool3d_forward(torch::Tensor input, int kernel_size, int stride);"

# Load custom CUDA operations
max_pool3d = load_inline(
    name="max_pool3d",
    cpp_sources=max_pool_cpp_source,
    cuda_sources=max_pool_source,
    functions=["max_pool3d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_kernel_size
        self.max_pool3d = max_pool3d

    def forward(self, x):
        x = self.conv(x)
        x = torch.softmax(x, dim=1)
        # First MaxPool3D
        x = self.max_pool3d.max_pool3d_forward(x, self.pool_kernel_size, self.pool_stride)
        # Second MaxPool3D
        x = self.max_pool3d.max_pool3d_forward(x, self.pool_kernel_size, self.pool_stride)
        return x