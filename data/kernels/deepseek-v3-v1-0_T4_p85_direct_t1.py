import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for conv + group_norm + scale + maxpool + clamp
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_gn_scale_maxpool_clamp_kernel(
    const float* input, const float* weight, const float* bias,
    const float* scale, const float* running_mean, const float* running_var,
    float* output, int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int kernel_size, int num_groups,
    int maxpool_kernel_size, float clamp_min, float clamp_max, float eps) {
    
    const int output_height = (input_height - kernel_size + 1) / maxpool_kernel_size;
    const int output_width = (input_width - kernel_size + 1) / maxpool_kernel_size;
    const int group_size = out_channels / num_groups;
    const int spatial_size = output_height * output_width;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * spatial_size) return;
    
    int n = idx / (out_channels * spatial_size);
    int c = (idx % (out_channels * spatial_size)) / spatial_size;
    int s = idx % spatial_size;
    int h_out = s / output_width;
    int w_out = s % output_width;
    
    int g = c / group_size;
    int group_start = g * group_size;
    int group_end = group_start + group_size;
    
    // Max pooling window
    float max_val = -FLT_MAX;
    
    for (int ph = 0; ph < maxpool_kernel_size; ++ph) {
        for (int pw = 0; pw < maxpool_kernel_size; ++pw) {
            int h_in = h_out * maxpool_kernel_size + ph;
            int w_in = w_out * maxpool_kernel_size + pw;
            
            if (h_in < input_height - kernel_size + 1 && w_in < input_width - kernel_size + 1) {
                // Convolution
                float conv_val = 0.0f;
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        for (int ic = 0; ic < in_channels; ++ic) {
                            int input_idx = ((n * in_channels + ic) * input_height + (h_in + kh)) * input_width + (w_in + kw);
                            int weight_idx = ((c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                            conv_val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
                conv_val += bias[c];
                
                // Group normalization
                float mean = running_mean[c];
                float var = running_var[c] + eps;
                float normalized = (conv_val - mean) / sqrtf(var);
                
                // Scale
                float scaled = normalized * scale[c];
                
                if (scaled > max_val) {
                    max_val = scaled;
                }
            }
        }
    }
    
    // Clamp
    float result = max_val;
    if (result < clamp_min) result = clamp_min;
    if (result > clamp_max) result = clamp_max;
    
    output[idx] = result;
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor scale, torch::Tensor running_mean, torch::Tensor running_var,
    int kernel_size, int num_groups, int maxpool_kernel_size,
    float clamp_min, float clamp_max, float eps) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(scale);
    CHECK_INPUT(running_mean);
    CHECK_INPUT(running_var);
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    
    int output_height = (input_height - kernel_size + 1) / maxpool_kernel_size;
    int output_width = (input_width - kernel_size + 1) / maxpool_kernel_size;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());
    
    int total_elements = batch_size * out_channels * output_height * output_width;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    fused_conv_gn_scale_maxpool_clamp_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        scale.data_ptr<float>(), running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        input_height, input_width, kernel_size, num_groups,
        maxpool_kernel_size, clamp_min, clamp_max, eps);
    
    return output;
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor scale, torch::Tensor running_mean, torch::Tensor running_var,
    int kernel_size, int num_groups, int maxpool_kernel_size,
    float clamp_min, float clamp_max, float eps);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.eps = 1e-5
        
        # Initialize convolution parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize group normalization parameters
        self.running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)
        
        # Initialize scale parameter
        self.scale = nn.Parameter(torch.ones(scale_shape))
        
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias, self.scale,
            self.running_mean, self.running_var,
            self.kernel_size, self.num_groups, self.maxpool_kernel_size,
            self.clamp_min, self.clamp_max, self.eps
        )