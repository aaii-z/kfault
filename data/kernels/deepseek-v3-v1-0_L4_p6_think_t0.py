import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv3D + Softmax
conv_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv_softmax_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding
) {
    int output_size = batch_size * out_channels * depth * height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size) {
        // Calculate output indices
        int n = idx / (out_channels * depth * height * width);
        int c = (idx / (depth * height * width)) % out_channels;
        int d = (idx / (height * width)) % depth;
        int h = (idx / width) % height;
        int w = idx % width;

        float sum = 0.0f;
        float max_val = -FLT_MAX;
        float* out_ptr = &output[n * out_channels * depth * height * width + c * depth * height * width + d * height * width + h * width + w];

        // Convolution operation
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int id = d * stride - padding + kd;
                        int ih = h * stride - padding + kh;
                        int iw = w * stride - padding + kw;
                        
                        if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            int input_idx = n * in_channels * depth * height * width + 
                                          ic * depth * height * width + 
                                          id * height * width + 
                                          ih * width + 
                                          iw;
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
        sum += bias[c];

        // Online softmax preparation
        max_val = fmaxf(max_val, sum);
        *out_ptr = sum;
    }

    __syncthreads();

    // Softmax implementation
    if (idx < output_size) {
        int n = idx / (out_channels * depth * height * width);
        int c = (idx / (depth * height * width)) % out_channels;
        int d = (idx / (height * width)) % depth;
        int h = (idx / width) % height;
        int w = idx % width;

        float* out_ptr = &output[n * out_channels * depth * height * width + c * depth * height * width + d * height * width + h * width + w];
        float exp_val = expf(*out_ptr - max_val);
        *out_ptr = exp_val;
    }
}

torch::Tensor conv_softmax_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    int out_channels = weight.size(0);

    auto output = torch::zeros({batch_size, out_channels, depth, height, width}, input.options());

    int output_size = batch_size * out_channels * depth * height * width;
    int block_size = 256;
    int grid_size = (output_size + block_size - 1) / block_size;

    conv_softmax_forward_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        kernel_size,
        stride,
        padding
    );

    return output;
}
"""

conv_softmax_cpp_source = "torch::Tensor conv_softmax_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, int stride, int padding);"

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
                    int input_idx = n * channels * depth * height * width + 
                                  c * depth * height * width + 
                                  curr_d * height * width + 
                                  curr_h * width + 
                                  curr_w;
                    max_val = fmaxf(max_val, input[input_idx]);
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
conv_softmax = load_inline(
    name="conv_softmax",
    cpp_sources=conv_softmax_cpp_source,
    cuda_sources=conv_softmax_source,
    functions=["conv_softmax_forward"],
    verbose=True
)

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.stride = 1
        self.padding = kernel_size // 2
        
        # Initialize weights and bias for convolution
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size, kernel_size
        ))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize custom operations
        self.conv_softmax = conv_softmax
        self.max_pool3d = max_pool3d

    def forward(self, x):
        # Fused Conv3D + Softmax
        x = self.conv_softmax.conv_softmax_forward(
            x, 
            self.weight, 
            self.bias, 
            self.kernel_size, 
            self.stride, 
            self.padding
        )
        
        # First MaxPool3D
        x = self.max_pool3d.max_pool3d_forward(x, self.pool_kernel_size, self.pool_kernel_size)
        
        # Second MaxPool3D
        x = self.max_pool3d.max_pool3d_forward(x, self.pool_kernel_size, self.pool_kernel_size)
        
        return x