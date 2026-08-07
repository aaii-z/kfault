import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused ConvTranspose3d + Scale + BatchNorm + GlobalAvgPool
fused_conv_bn_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_bn_pool_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    const float* weight_bn,
    const float* bias_bn,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int depth, int height, int width,
    int out_depth, int out_height, int out_width,
    float scale_factor,
    float eps,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    // Simplified implementation: In practice, this would use optimized CUDA libraries like CUTLASS or cuDNN
    // This is a placeholder showing the kernel structure
    extern __shared__ float shared[];
    
    int n = blockIdx.x;
    int c = blockIdx.y;
    int d = threadIdx.x;
    int h = threadIdx.y;
    int w = threadIdx.z;
    
    if (n >= batch_size || c >= out_channels || d >= out_depth || h >= out_height || w >= out_width) {
        return;
    }
    
    // Compute transposed convolution output
    float val = 0.0f;
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Simplified transposed convolution calculation
                int input_d = d - kd + padding;
                int input_h = h - kh + padding;
                int input_w = w - kw + padding;
                
                if (input_d >= 0 && input_d < depth && input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    for (int ic = 0; ic < in_channels; ic++) {
                        int input_idx = ((n * in_channels + ic) * depth + input_d) * height * width + input_h * width + input_w;
                        int weight_idx = ((ic * out_channels + c) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    if (bias) {
        val += bias[c];
    }
    
    // Apply scale factor
    val *= scale_factor;
    
    // Apply batch normalization
    float mean = running_mean[c];
    float var = running_var[c];
    float bn_weight = weight_bn ? weight_bn[c] : 1.0f;
    float bn_bias = bias_bn ? bias_bn[c] : 0.0f;
    
    val = (val - mean) / sqrtf(var + eps) * bn_weight + bn_bias;
    
    // Global average pooling
    atomicAdd(&output[n * out_channels + c], val / (out_depth * out_height * out_width));
}

torch::Tensor fused_conv_bn_pool_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    float scale_factor,
    float eps,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
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
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int out_depth = (depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_height = (height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_width = (width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    auto output = torch::zeros({batch_size, out_channels, 1, 1, 1}, input.options());
    
    dim3 blocks(batch_size, out_channels, 1);
    dim3 threads(out_depth, out_height, out_width);
    int shared_mem = out_depth * out_height * out_width * sizeof(float);
    
    fused_conv_bn_pool_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight_bn.data_ptr<float>(),
        bias_bn.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        depth, height, width,
        out_depth, out_height, out_width,
        scale_factor,
        eps,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
    
    return output;
}
"""

fused_conv_bn_pool_cpp_source = """
torch::Tensor fused_conv_bn_pool_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    float scale_factor,
    float eps,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
);
"""

fused_conv_bn_pool = load_inline(
    name='fused_conv_bn_pool',
    cpp_sources=fused_conv_bn_pool_cpp_source,
    cuda_sources=fused_conv_bn_pool_source,
    functions=['fused_conv_bn_pool_forward'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.eps = eps
        
        # Initialize weights and biases for transposed convolution
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize batch normalization parameters
        self.running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)
        self.weight_bn = nn.Parameter(torch.randn(out_channels))
        self.bias_bn = nn.Parameter(torch.randn(out_channels))
        
        # Set default conv parameters
        self.stride = 1
        self.padding = 0
        self.output_padding = 0
        self.groups = 1
        self.dilation = 1

    def forward(self, x):
        return fused_conv_bn_pool.fused_conv_bn_pool_forward(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.weight_bn,
            self.bias_bn,
            self.scale_factor,
            self.eps,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation
        )