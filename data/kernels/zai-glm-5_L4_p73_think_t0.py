python
import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + BatchNorm + Scale
fused_conv_bn_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_bn_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_running_mean,
    const float* __restrict__ bn_running_var,
    float scaling_factor,
    float eps,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_channels, int out_h, int out_w,
    int kernel_size) {
    
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    int n_c_out_idx = blockIdx.z;
    int n = n_c_out_idx / out_channels;
    int c_out = n_c_out_idx % out_channels;

    // Shared memory for input tile
    extern __shared__ float shared_input[];

    int tile_h = blockDim.y + kernel_size - 1;
    int tile_w = blockDim.x + kernel_size - 1;
    int tile_size = tile_h * tile_w * in_channels;
    
    int h_in_start = blockIdx.y * blockDim.y;
    int w_in_start = blockIdx.x * blockDim.x;

    // Cooperative loading of input tile
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < tile_size; i += blockDim.x * blockDim.y) {
        int local_idx = i;
        int w_local = local_idx % tile_w;
        local_idx /= tile_w;
        int h_local = local_idx % tile_h;
        int c_local = local_idx / tile_h;
        
        int h_in = h_in_start + h_local;
        int w_in = w_in_start + w_local;
        
        float val = 0.0f;
        if (h_in < in_h && w_in < in_w) {
            val = input[n * (in_channels * in_h * in_w) + c_local * (in_h * in_w) + h_in * in_w + w_in];
        }
        shared_input[i] = val;
    }
    
    __syncthreads();

    if (h_out >= out_h || w_out >= out_w) return;

    // Convolution
    float sum = 0.0f;
    if (conv_bias != nullptr) {
        sum = conv_bias[c_out];
    }

    int weight_base = c_out * (in_channels * kernel_size * kernel_size);

    for (int c = 0; c < in_channels; ++c) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int sh_idx = c * (tile_h * tile_w) + (threadIdx.y + kh) * tile_w + (threadIdx.x + kw);
                float in_val = shared_input[sh_idx];
                
                int w_idx = weight_base + c * (kernel_size * kernel_size) + kh * kernel_size + kw;
                float w_val = conv_weight[w_idx];
                
                sum += in_val * w_val;
            }
        }
    }

    // BatchNorm (Inference mode)
    float mean = bn_running_mean[c_out];
    float var = bn_running_var[c_out];
    float gamma = bn_weight[c_out];
    float beta = bn_bias[c_out];
    
    float bn_out = (sum - mean) / sqrtf(var + eps) * gamma + beta;

    // Scale
    float final_out = bn_out * scaling_factor;

    int out_idx = n * (out_channels * out_h * out_w) + c_out * (out_h * out_w) + h_out * out_w + w_out;
    output[out_idx] = final_out;
}

torch::Tensor fused_conv_bn_scale_cuda(
    torch::Tensor input, 
    torch::Tensor conv_weight, torch::Tensor conv_bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_running_mean, torch::Tensor bn_running_var,
    float scaling_factor, float eps,
    int kernel_size) {
    
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    int C_out = conv_weight.size(0);
    int H_out = H_in - kernel_size + 1;
    int W_out = W_in - kernel_size + 1;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    dim3 block(16, 16);
    dim3 grid((W_out + 15) / 16, (H_out + 15) / 16, N * C_out);

    int tile_h = 16 + kernel_size - 1;
    int tile_w = 16 + kernel_size - 1;
    size_t shared_mem_size = tile_h * tile_w * C_in * sizeof(float);

    fused_conv_bn_scale_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        (conv_bias.defined() && conv_bias.numel() > 0) ? conv_bias.data_ptr<float>() : nullptr,
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        scaling_factor,
        eps,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_size
    );

    return output;
}
"""

fused_conv_bn_scale_cpp_source = (
    "torch::Tensor fused_conv_bn_scale_cuda(torch::Tensor input, torch::Tensor conv_weight, torch::Tensor conv_bias, "
    "torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_running_mean, torch::Tensor bn_running_var, "
    "float scaling_factor, float eps, int kernel_size);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_conv_bn_scale_cpp_source,
    cuda_sources=fused_conv_bn_scale_source,
    functions=["fused_conv_bn_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        
        self.register_buffer('bn_running_mean', torch.zeros(out_channels))
        self.register_buffer('bn_running_var', torch.ones(out_channels))
        
        self.scaling_factor = scaling_factor
        self.kernel_size = kernel_size
        self.eps = 1e-5
        
        # Initialize weights similar to PyTorch defaults
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.conv_bias, -bound, bound)

    def forward(self, x):
        return fused_ops.fused_conv_bn_scale_cuda(
            x, 
            self.conv_weight, 
            self.conv_bias,
            self.bn_weight, 
            self.bn_bias,
            self.bn_running_mean, 
            self.bn_running_var,
            self.scaling_factor,
            self.eps,
            self.kernel_size
        )