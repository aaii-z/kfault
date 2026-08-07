import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused ConvTranspose3D + BatchNorm + Mean Subtraction
fused_conv_bn_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_bn_mean_kernel(
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var,
    const float* weight_bn, const float* bias_bn,
    float* output, float* mean_buffer,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    float eps) {
    
    // Calculate output spatial dimensions
    int out_depth = (depth - 1) * stride_d - 2 * pad_d + kernel_d;
    int out_height = (height - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_width = (width - 1) * stride_w - 2 * pad_w + kernel_w;
    
    // Calculate total output elements per channel
    int spatial_size = out_depth * out_height * out_width;
    int total_elements = batch_size * out_channels * spatial_size;
    
    // Thread indexing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Calculate indices
        int b = idx / (out_channels * spatial_size);
        int c = (idx % (out_channels * spatial_size)) / spatial_size;
        int spatial_idx = idx % spatial_size;
        int d_out = spatial_idx / (out_height * out_width);
        int h_out = (spatial_idx % (out_height * out_width)) / out_width;
        int w_out = spatial_idx % out_width;
        
        float sum = 0.0f;
        
        // Perform transposed convolution
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int d_in = (d_out + pad_d - kd) / stride_d;
                    int h_in = (h_out + pad_h - kh) / stride_h;
                    int w_in = (w_out + pad_w - kw) / stride_w;
                    
                    if (d_in >= 0 && d_in < depth && h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        if ((d_out + pad_d - kd) % stride_d == 0 && 
                            (h_out + pad_h - kh) % stride_h == 0 && 
                            (w_out + pad_w - kw) % stride_w == 0) {
                            
                            for (int ic = 0; ic < in_channels; ++ic) {
                                int input_idx = ((b * in_channels + ic) * depth + d_in) * height * width + 
                                              h_in * width + w_in;
                                int weight_idx = ((c * in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w + 
                                               kh * kernel_w + kw;
                                
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
        
        // Add bias if present
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        // Apply batch normalization: (x - running_mean) / sqrt(running_var + eps) * weight_bn + bias_bn
        float normalized = (sum - running_mean[c]) / sqrtf(running_var[c] + eps);
        normalized = normalized * weight_bn[c] + bias_bn[c];
        
        // Store result
        output[idx] = normalized;
        
        // Accumulate for mean calculation
        atomicAdd(&mean_buffer[c], normalized);
    }
}

__global__ void subtract_mean_kernel(
    float* output, const float* mean_buffer, 
    int batch_size, int out_channels, int spatial_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * spatial_size;
    
    if (idx < total_elements) {
        int c = (idx % (out_channels * spatial_size)) / spatial_size;
        float mean_val = mean_buffer[c] / (batch_size * spatial_size);
        output[idx] -= mean_val;
    }
}

torch::Tensor fused_conv_transpose_bn_mean_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight_bn, torch::Tensor bias_bn,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    float eps) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
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
    
    // Calculate output spatial dimensions
    int out_depth = (depth - 1) * stride_d - 2 * pad_d + kernel_d;
    int out_height = (height - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_width = (width - 1) * stride_w - 2 * pad_w + kernel_w;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, 
                              input.options());
    auto mean_buffer = torch::zeros({out_channels}, input.options());
    
    int spatial_size = out_depth * out_height * out_width;
    int total_elements = batch_size * out_channels * spatial_size;
    
    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // First kernel: ConvTranspose + BatchNorm + Accumulate mean
    fused_conv_bn_mean_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        weight_bn.data_ptr<float>(), bias_bn.data_ptr<float>(),
        output.data_ptr<float>(), mean_buffer.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        eps);
    
    // Second kernel: Subtract mean
    subtract_mean_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), mean_buffer.data_ptr<float>(),
        batch_size, out_channels, spatial_size);
    
    return output;
}
"""

fused_conv_bn_mean_cpp_source = """
torch::Tensor fused_conv_transpose_bn_mean_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight_bn, torch::Tensor bias_bn,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    float eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_conv_bn_mean_cpp_source,
    cuda_sources=fused_conv_bn_mean_source,
    functions=["fused_conv_transpose_bn_mean_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Store parameters as buffers for CUDA kernel
        self.register_buffer('weight', torch.randn(
            in_channels, out_channels, kernel_size, kernel_size, kernel_size
        ))
        if bias:
            self.register_buffer('bias', torch.randn(out_channels))
        else:
            self.register_buffer('bias', torch.Tensor())
        
        # BatchNorm parameters
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('weight_bn', torch.ones(out_channels))
        self.register_buffer('bias_bn', torch.zeros(out_channels))
        
        self.eps = 1e-5
        
        self.fused_op = fused_ops

    def forward(self, x):
        return self.fused_op.fused_conv_transpose_bn_mean_cuda(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.weight_bn, self.bias_bn,
            self.kernel_size, self.kernel_size, self.kernel_size,
            self.stride, self.stride, self.stride,
            self.padding, self.padding, self.padding,
            self.eps
        )