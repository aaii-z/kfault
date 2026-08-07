import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3d + MaxPool3d + Softmax + Subtract + Swish + Max
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>

__device__ float swish(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

__global__ void conv_transpose_maxpool_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* intermediate,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding
) {
    int d_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.z * blockDim.z + threadIdx.z;
    int batch = blockIdx.w / out_channels;
    int channel = blockIdx.w % out_channels;
    
    if (d_out >= output_depth || h_out >= output_height || w_out >= output_width || batch >= batch_size || channel >= out_channels) {
        return;
    }
    
    int d_in_start = d_out * stride - padding;
    int h_in_start = h_out * stride - padding;
    int w_in_start = w_out * stride - padding;
    
    float conv_result = bias[channel];
    
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int d_in = d_in_start + kd;
                int h_in = h_in_start + kh;
                int w_in = w_in_start + kw;
                
                if (d_in >= 0 && d_in < input_depth && h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                        int input_idx = ((batch * in_channels + in_ch) * input_depth + d_in) * input_height * input_width + h_in * input_width + w_in;
                        int weight_idx = ((in_ch * out_channels + channel) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    int pool_d_start = d_out * pool_stride - pool_padding;
    int pool_h_start = h_out * pool_stride - pool_padding;
    int pool_w_start = w_out * pool_stride - pool_padding;
    
    float pool_max = -FLT_MAX;
    for (int pd = 0; pd < pool_kernel_size; pd++) {
        for (int ph = 0; ph < pool_kernel_size; ph++) {
            for (int pw = 0; pw < pool_kernel_size; pw++) {
                int pool_d = pool_d_start + pd;
                int pool_h = pool_d_start + ph;
                int pool_w = pool_d_start + pw;
                
                if (pool_d >= 0 && pool_d < output_depth && pool_h >= 0 && pool_h < output_height && pool_w >= 0 && pool_w < output_width) {
                    pool_max = fmaxf(pool_max, conv_result);
                }
            }
        }
    }
    
    int intermediate_idx = ((batch * out_channels + channel) * output_depth + d_out) * output_height * output_width + h_out * output_width + w_out;
    intermediate[intermediate_idx] = pool_max;
}

__global__ void softmax_subtract_swish_max_kernel(
    const float* intermediate,
    const float* subtract_param,
    float* final_output,
    int batch_size,
    int out_channels,
    int output_depth,
    int output_height,
    int output_width
) {
    int d = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    int batch = blockIdx.w;
    
    if (d >= output_depth || h >= output_height || w >= output_width || batch >= batch_size) {
        return;
    }
    
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    float max_val = -FLT_MAX;
    for (int ch = 0; ch < out_channels; ch++) {
        int idx = ((batch * out_channels + ch) * output_depth + d) * output_height * output_width + h * output_width + w;
        max_val = fmaxf(max_val, intermediate[idx]);
    }
    
    if (threadIdx.x == 0) {
        shared_max = max_val;
    }
    __syncthreads();
    
    float sum_exp = 0.0f;
    for (int ch = 0; ch < out_channels; ch++) {
        int idx = ((batch * out_channels + ch) * output_depth + d) * output_height * output_width + h * output_width + w;
        float val = intermediate[idx];
        sum_exp += expf(val - shared_max);
    }
    
    if (threadIdx.x == 0) {
        shared_sum = sum_exp;
    }
    __syncthreads();
    
    float max_final = -FLT_MAX;
    for (int ch = 0; ch < out_channels; ch++) {
        int idx = ((batch * out_channels + ch) * output_depth + d) * output_height * output_width + h * output_width + w;
        float val = intermediate[idx];
        
        float softmax_val = expf(val - shared_max) / shared_sum;
        softmax_val -= subtract_param[ch];
        softmax_val = swish(softmax_val);
        
        max_final = fmaxf(max_final, softmax_val);
    }
    
    if (threadIdx.x == 0) {
        int final_idx = (batch * output_depth + d) * output_height * output_width + h * output_width + w;
        final_output[final_idx] = max_final;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor subtract_param,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    int out_channels = weight.size(1);
    
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto intermediate = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());
    
    dim3 block_size(4, 4, 4);
    dim3 grid_size(
        (output_depth + block_size.x - 1) / block_size.x,
        (output_height + block_size.y - 1) / block_size.y,
        (output_width + block_size.z - 1) / block_size.z,
        batch_size * out_channels
    );
    
    conv_transpose_maxpool_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        pool_stride,
        pool_padding
    );
    
    auto final_output = torch::zeros({batch_size, output_depth, output_height, output_width}, input.options());
    
    dim3 final_block(1);
    dim3 final_grid(output_depth, output_height, output_width, batch_size);
    
    softmax_subtract_swish_max_kernel<<<final_grid, final_block>>>(
        intermediate.data_ptr<float>(),
        subtract_param.data_ptr<float>(),
        final_output.data_ptr<float>(),
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width
    );
    
    return final_output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor subtract_param,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding
);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.subtract = nn.Parameter(torch.randn(out_channels))
        
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_ops_cuda(
            x,
            self.weight,
            self.bias,
            self.subtract,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.pool_kernel_size,
            self.pool_stride,
            self.pool_padding
        )