import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_conv_norm_min_clamp_dropout_kernel(
    const float* input, const float* weight, const float* bias, 
    float* output, int batch_size, int in_channels, int out_channels, 
    int input_depth, int input_height, int input_width, int kernel_size,
    int groups, float min_value, float max_value, float dropout_p,
    bool training, unsigned long long seed) {
    
    const int output_depth = input_depth - kernel_size + 1;
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;
    const int spatial_size = output_depth * output_height * output_width;
    const int group_size = out_channels / groups;
    const int k_half = kernel_size / 2;
    const int total_elements = batch_size * out_channels * spatial_size;
    
    CUDA_KERNEL_LOOP(idx, total_elements) {
        int n = idx / (out_channels * spatial_size);
        int c = (idx % (out_channels * spatial_size)) / spatial_size;
        int s = idx % spatial_size;
        
        int d = s / (output_height * output_width);
        int h = (s % (output_height * output_width)) / output_width;
        int w = s % output_width;
        
        int g = c / group_size;
        
        // 3D convolution
        float conv_result = bias[c];
        const float* weight_ptr = &weight[c * in_channels * kernel_size * kernel_size * kernel_size];
        
        for (int ic = 0; ic < in_channels; ic++) {
            const float* input_ptr = &input[((n * in_channels + ic) * input_depth) * input_height * input_width];
            
            for (int kd = 0; kd < kernel_size; kd++) {
                int od = d + kd;
                if (od >= input_depth) continue;
                
                for (int kh = 0; kh < kernel_size; kh++) {
                    int oh = h + kh;
                    if (oh >= input_height) continue;
                    
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int ow = w + kw;
                        if (ow >= input_width) continue;
                        
                        int input_idx = (od * input_height + oh) * input_width + ow;
                        int weight_idx = (((ic * kernel_size + kd) * kernel_size + kh) * kernel_size + kw);
                        
                        conv_result += input_ptr[input_idx] * weight_ptr[weight_idx];
                    }
                }
            }
        }
        
        // Min operation
        conv_result = fminf(conv_result, min_value);
        
        // Clamp operation
        conv_result = fmaxf(fminf(conv_result, max_value), min_value);
        
        // Dropout
        if (training && dropout_p > 0.0f) {
            curandState local_state;
            curand_init(seed + idx, 0, 0, &local_state);
            float random = curand_uniform(&local_state);
            if (random < dropout_p) {
                conv_result = 0.0f;
            } else {
                conv_result /= (1.0f - dropout_p);
            }
        }
        
        output[idx] = conv_result;
    }
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int groups, float min_value, float max_value, float dropout_p,
    bool training) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    int output_depth = input_depth - kernel_size + 1;
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());
    
    const int block_size = 256;
    const int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    unsigned long long seed = time(NULL);
    
    fused_conv_norm_min_clamp_dropout_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels, 
        input_depth, input_height, input_width, kernel_size, groups, min_value, max_value, 
        dropout_p, training, seed);
    
    return output;
}
"""

fused_operator_cpp_source = "torch::Tensor fused_operator_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int groups, float min_value, float max_value, float dropout_p, bool training);"

fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcurand"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias, self.groups, 
            self.min_value, self.max_value, self.dropout_p, self.training
        )