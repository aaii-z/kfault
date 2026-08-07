import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 256

__device__ float hardtanh(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

__global__ void fused_operation_kernel(
    const float* input,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int maxpool_kernel_size,
    int maxpool_stride,
    float hardtanh_min,
    float hardtanh_max,
    const float* weight,
    const float* bias
) {
    // Compute indices
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int h_out = blockIdx.x * blockDim.y + threadIdx.y;
    int w_out = threadIdx.x;

    if (h_out >= output_height || w_out >= output_width) return;

    float output_val = 0.0f;

    // ConvTranspose2d operation
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int input_idx = n * in_channels * input_height * input_width +
                                  c_in * input_height * input_width +
                                  h_in * input_width + w_in;
                    int weight_idx = c_in * out_channels * kernel_size * kernel_size +
                                   c_out * kernel_size * kernel_size +
                                   kh * kernel_size + kw;
                    output_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    if (bias) {
        output_val += bias[c_out];
    }

    // MaxPool2d operation
    int pool_h = h_out / maxpool_stride;
    int pool_w = w_out / maxpool_stride;
    if (h_out % maxpool_stride == 0 && w_out % maxpool_stride == 0) {
        float max_val = -FLT_MAX;
        for (int ph = 0; ph < maxpool_kernel_size; ph++) {
            for (int pw = 0; pw < maxpool_kernel_size; pw++) {
                int h_idx = pool_h * maxpool_stride + ph;
                int w_idx = pool_w * maxpool_stride + pw;
                if (h_idx < output_height && w_idx < output_width) {
                    max_val = fmaxf(max_val, output_val);
                }
            }
        }
        output_val = max_val;
    } else {
        output_val = -FLT_MAX;
    }

    // Hardtanh activation
    output_val = hardtanh(output_val, hardtanh_min, hardtanh_max);

    // Write to output
    int output_idx = n * out_channels * output_height * output_width +
                   c_out * output_height * output_width +
                   h_out * output_width + w_out;
    output[output_idx] = output_val;
}

__global__ void reduce_mean_tanh_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    extern __shared__ float sdata[];

    // Each thread loads one element into shared memory
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    for (int i = tid; i < height * width; i += blockDim.x) {
        int h = i / width;
        int w = i % width;
        int idx = n * channels * height * width + c * height * width + h * width + w;
        sdata[tid] += input[idx];
    }
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        float mean_val = sdata[0] / (height * width);
        output[n * channels + c] = tanhf(mean_val);
    }
}

torch::Tensor fused_conv_transpose_maxpool_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int maxpool_kernel_size,
    int maxpool_stride,
    float hardtanh_min,
    float hardtanh_max
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(1);

    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 blocks(
        (output_height + 15) / 16,
        out_channels,
        batch_size
    );
    dim3 threads(16, 16);

    fused_operation_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min,
        hardtanh_max,
        weight.data_ptr<float>(),
        bias.data_ptr<float>()
    );

    return output;
}

torch::Tensor reduce_mean_tanh_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::empty({batch_size, channels}, input.options());

    dim3 blocks(batch_size, channels);
    int threads = 1024;
    int shared_mem = threads * sizeof(float);

    reduce_mean_tanh_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );

    return output.unsqueeze(-1).unsqueeze(-1);
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_conv_transpose_maxpool_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int maxpool_kernel_size,
    int maxpool_stride,
    float hardtanh_min,
    float hardtanh_max
);
torch::Tensor reduce_mean_tanh_cuda(torch::Tensor input);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=[
        "fused_conv_transpose_maxpool_hardtanh_cuda",
        "reduce_mean_tanh_cuda"
    ],
    verbose=True,
    extra_cflags=["-O2"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
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
        
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.fused_ops.fused_conv_transpose_maxpool_hardtanh_cuda(
            x, self.weight, self.bias,
            self.kernel_size, self.stride, self.padding,
            self.maxpool_kernel_size, self.maxpool_stride,
            self.hardtanh_min, self.hardtanh_max
        )
        x = self.fused_ops.reduce_mean_tanh_cuda(x)
        return x