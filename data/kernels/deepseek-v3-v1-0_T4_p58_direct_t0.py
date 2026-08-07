import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3D + LogSumExp + HardSwish + Bias + Clamp
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ float hard_swish(float x) {
    return x * __fdividef(__float2half_rn(1.0), __float2half_rn(6.0)) * min(max(x + 3.0f, 0.0f), 6.0f);
}

__global__ void fused_conv_transpose_logsumexp_swish_bias_clamp_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* bias_param,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_depth, int out_height, int out_width) {
    
    int n = blockIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int w = threadIdx.x;
    
    if (n >= batch_size || d >= out_depth || h >= out_height || w >= out_width) return;
    
    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;
    
    // Compute LogSumExp across channels
    for (int c = 0; c < out_channels; c++) {
        float conv_val = 0.0f;
        
        // 3D transposed convolution
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int in_d = (d - kd + pad_d) / stride_d;
                    int in_h = (h - kh + pad_h) / stride_h;
                    int in_w = (w - kw + pad_w) / stride_w;
                    
                    if (in_d >= 0 && in_d < depth && in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        if ((d - kd + pad_d) % stride_d == 0 && (h - kh + pad_h) % stride_h == 0 && (w - kw + pad_w) % stride_w == 0) {
                            int input_idx = n * (in_channels * depth * height * width) + 
                                          in_d * (height * width) + in_h * width + in_w;
                            int weight_idx = c * (in_channels * kernel_d * kernel_h * kernel_w) + 
                                           kd * (kernel_h * kernel_w) + kh * kernel_w + kw;
                            conv_val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias if present
        if (bias != nullptr) {
            conv_val += bias[c];
        }
        
        // Update max and sum for LogSumExp
        if (conv_val > max_val) {
            max_val = conv_val;
        }
    }
    
    // Second pass for stable LogSumExp
    for (int c = 0; c < out_channels; c++) {
        float conv_val = 0.0f;
        
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int in_d = (d - kd + pad_d) / stride_d;
                    int in_h = (h - kh + pad_h) / stride_h;
                    int in_w = (w - kw + pad_w) / stride_w;
                    
                    if (in_d >= 0 && in_d < depth && in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        if ((d - kd + pad_d) % stride_d == 0 && (h - kh + pad_h) % stride_h == 0 && (w - kw + pad_w) % stride_w == 0) {
                            int input_idx = n * (in_channels * depth * height * width) + 
                                          in_d * (height * width) + in_h * width + in_w;
                            int weight_idx = c * (in_channels * kernel_d * kernel_h * kernel_w) + 
                                           kd * (kernel_h * kernel_w) + kh * kernel_w + kw;
                            conv_val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            conv_val += bias[c];
        }
        
        sum_exp += expf(conv_val - max_val);
    }
    
    float logsumexp = max_val + logf(sum_exp);
    
    // Apply HardSwish: x * sigmoid(x + 3) / 6
    float swished = hard_swish(logsumexp);
    
    // Subtract bias parameter and clamp
    float result = swished - bias_param[0];
    result = fminf(fmaxf(result, -1.0f), 1.0f);
    
    int output_idx = n * (out_depth * out_height * out_width) + 
                    d * (out_height * out_width) + h * out_width + w;
    output[output_idx] = result;
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor bias_param,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    // Calculate output dimensions
    int out_depth = (depth - 1) * stride_d - 2 * pad_d + kernel_d;
    int out_height = (height - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_width = (width - 1) * stride_w - 2 * pad_w + kernel_w;
    
    auto output = torch::zeros({batch_size, 1, out_depth, out_height, out_width}, 
                              torch::dtype(torch::kFloat32).device(input.device()));
    
    dim3 block_size(32, 4, 4);
    dim3 grid_size(
        (batch_size + block_size.x - 1) / block_size.x,
        (out_depth + block_size.y - 1) / block_size.y,
        (out_height + block_size.z - 1) / block_size.z
    );
    
    fused_conv_transpose_logsumexp_swish_bias_clamp_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), bias_param.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_depth, out_height, out_width
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor bias_param,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w);
"""

# Compile the fused CUDA operations
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias for transposed convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_ops_cuda(
            x, self.weight, self.conv_bias, self.bias,
            self.stride, self.stride, self.stride,
            self.padding, self.padding, self.padding
        )