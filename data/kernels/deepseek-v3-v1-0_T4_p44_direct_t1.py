import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for conv_transpose + scaling + pooling
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_transpose_scale_pool_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float multiplier,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size, int stride, int padding, int output_padding) {
    
    int n = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (n >= batch_size || oc >= out_channels || oh >= output_height || ow >= output_width) {
        return;
    }
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = n * in_channels * input_height * input_width +
                                  ic * input_height * input_width +
                                  ih * input_width + iw;
                    
                    int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                   ic * kernel_size * kernel_size +
                                   kh * kernel_size + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    sum *= multiplier;
    
    int output_idx = n * out_channels * output_height * output_width +
                    oc * output_height * output_width +
                    oh * output_width + ow;
    output[output_idx] = sum;
}

__global__ void global_avg_pool_kernel(
    const float* input, float* output,
    int batch_size, int channels, int height, int width) {
    
    int n = blockIdx.x;
    int c = blockIdx.y;
    
    if (n >= batch_size || c >= channels) {
        return;
    }
    
    float sum = 0.0f;
    int input_idx = (n * channels + c) * height * width;
    
    for (int i = 0; i < height * width; ++i) {
        sum += input[input_idx + i];
    }
    
    output[n * channels + c] = sum / (height * width);
}

torch::Tensor fused_conv_transpose_scale_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float multiplier, int stride, int padding, int output_padding) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) {
        CHECK_INPUT(bias);
    }
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());
    
    dim3 grid_dim((output_height + 15) / 16, out_channels, batch_size);
    dim3 block_dim(16, 16);
    
    fused_conv_transpose_scale_pool_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), multiplier,
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size, stride, padding, output_padding);
    
    auto pooled1 = torch::empty({batch_size, out_channels, 1, 1}, input.options());
    global_avg_pool_kernel<<<dim3(batch_size, out_channels), dim3(1, 1)>>>(
        output.data_ptr<float>(), pooled1.data_ptr<float>(),
        batch_size, out_channels, output_height, output_width);
    
    auto pooled2 = torch::empty({batch_size, out_channels, 1, 1}, input.options());
    global_avg_pool_kernel<<<dim3(batch_size, out_channels), dim3(1, 1)>>>(
        pooled1.data_ptr<float>(), pooled2.data_ptr<float>(),
        batch_size, out_channels, 1, 1);
    
    return pooled2;
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_conv_transpose_scale_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float multiplier, int stride, int padding, int output_padding);
"""

fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_conv_transpose_scale_pool_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.multiplier = multiplier
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_conv_transpose_scale_pool_cuda(
            x, self.weight, self.bias, self.multiplier, 
            self.stride, self.padding, self.output_padding
        )