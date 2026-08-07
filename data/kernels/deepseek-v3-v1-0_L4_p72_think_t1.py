import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Custom CUDA kernel for fused ConvTranspose3d + BatchNorm3d
fused_conv_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_bn_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    const float* weight_bn,
    const float* bias_bn,
    float eps,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int output_depth,
    int output_height,
    int output_width
) {
    int output_size = output_depth * output_height * output_width;
    int output_total = batch_size * out_channels * output_size;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_total) return;

    int n = idx / (out_channels * output_size);
    int c = (idx % (out_channels * output_size)) / output_size;
    int spatial_idx = idx % output_size;
    int d = spatial_idx / (output_height * output_width);
    int h = (spatial_idx % (output_height * output_width)) / output_width;
    int w = spatial_idx % output_width;

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int input_d = d * stride - padding + kd;
                    int input_h = h * stride - padding + kh;
                    int input_w = w * stride - padding + kw;

                    if (input_d >= 0 && input_d < input_depth &&
                        input_h >= 0 && input_h < input_height &&
                        input_w >= 0 && input_w < input_width) {
                        int input_idx = ((n * in_channels + ic) * input_depth + input_d) * input_height * input_width +
                                        input_h * input_width + input_w;
                        int weight_idx = ((ic * out_channels + c) * kernel_size + kd) * kernel_size * kernel_size +
                                        kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias) {
        sum += bias[c];
    }

    // BatchNorm
    float mean = running_mean[c];
    float var = running_var[c];
    float bn_scale = weight_bn[c] / sqrtf(var + eps);
    float bn_shift = bias_bn[c] - mean * bn_scale;
    output[idx] = sum * bn_scale + bn_shift;
}

torch::Tensor fused_conv_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    float eps,
    int kernel_size,
    int stride,
    int padding
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(running_mean);
    CHECK_INPUT(running_var);
    CHECK_INPUT(weight_bn);
    CHECK_INPUT(bias_bn);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    int out_channels = weight.size(1);

    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    int total_output_size = batch_size * out_channels * output_depth * output_height * output_width;
    const int threads = 256;
    int blocks = (total_output_size + threads - 1) / threads;

    fused_conv_bn_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight_bn.data_ptr<float>(),
        bias_bn.data_ptr<float>(),
        eps,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        output_depth,
        output_height,
        output_width
    );

    return output;
}
"""

fused_conv_bn_cpp_source = """
torch::Tensor fused_conv_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    float eps,
    int kernel_size,
    int stride,
    int padding
);
"""

fused_conv_bn = load_inline(
    name="fused_conv_bn",
    cpp_sources=fused_conv_bn_cpp_source,
    cuda_sources=fused_conv_bn_source,
    functions=["fused_conv_bn_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize parameters for fused operation
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.running_mean = nn.Parameter(torch.Tensor(out_channels), requires_grad=False)
        self.running_var = nn.Parameter(torch.Tensor(out_channels), requires_grad=False)
        self.weight_bn = nn.Parameter(torch.Tensor(out_channels))
        self.bias_bn = nn.Parameter(torch.Tensor(out_channels))
        self.eps = 1e-5
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.ones_(self.weight_bn)
        nn.init.zeros_(self.bias_bn)
        nn.init.zeros_(self.running_mean)
        nn.init.ones_(self.running_var)

        # Replace two average poolings with one equivalent operation
        self.avg_pool = nn.AvgPool3d(kernel_size=4, stride=4)

    def forward(self, x):
        x = fused_conv_bn.fused_conv_bn_cuda(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.weight_bn,
            self.bias_bn,
            self.eps,
            self.kernel_size,
            self.stride,
            self.padding
        )
        x = self.avg_pool(x)
        return x