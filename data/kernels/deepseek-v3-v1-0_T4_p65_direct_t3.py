import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for conv + avg_pool + sigmoid + sum
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_pool_sigmoid_sum_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int kernel_size, int pool_kernel_size) {
    
    const int output_height = (input_height - kernel_size + 1) / pool_kernel_size;
    const int output_width = (input_width - kernel_size + 1) / pool_kernel_size;
    const int output_spatial_size = output_height * output_width;
    
    int batch = blockIdx.x;
    int out_c = blockIdx.y;
    int spatial_idx = threadIdx.x;
    
    if (batch >= batch_size || out_c >= out_channels || spatial_idx >= output_spatial_size) return;
    
    int h = spatial_idx / output_width;
    int w = spatial_idx % output_width;
    
    float pool_sum = 0.0f;
    int pool_count = 0;
    
    for (int ph = 0; ph < pool_kernel_size; ph++) {
        for (int pw = 0; pw < pool_kernel_size; pw++) {
            int input_h = h * pool_kernel_size + ph;
            int input_w = w * pool_kernel_size + pw;
            
            if (input_h < input_height && input_w < input_width) {
                float conv_val = bias[out_c];
                
                for (int in_c = 0; in_c < in_channels; in_c++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int conv_h = input_h + kh;
                            int conv_w = input_w + kw;
                            
                            if (conv_h < input_height && conv_w < input_width) {
                                int input_idx = ((batch * in_channels + in_c) * input_height + conv_h) * input_width + conv_w;
                                int weight_idx = ((out_c * in_channels + in_c) * kernel_size + kh) * kernel_size + kw;
                                conv_val += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                
                pool_sum += conv_val;
                pool_count++;
            }
        }
    }
    
    float avg_val = pool_count > 0 ? pool_sum / pool_count : 0.0f;
    float sigmoid_val = 1.0f / (1.0f + expf(-avg_val));
    
    atomicAdd(&output[batch * out_channels + out_c], sigmoid_val);
}

torch::Tensor fused_conv_pool_sigmoid_sum_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int pool_kernel_size) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    
    const int output_height = (input_height - kernel_size + 1) / pool_kernel_size;
    const int output_width = (input_width - kernel_size + 1) / pool_kernel_size;
    const int output_spatial_size = output_height * output_width;
    
    auto output = torch::zeros({batch_size, out_channels}, input.options());
    
    dim3 grid_dim(batch_size, out_channels);
    dim3 block_dim((output_spatial_size + 31) / 32 * 32);
    
    fused_conv_pool_sigmoid_sum_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        pool_kernel_size
    );
    
    return torch::sum(output, 1);
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_conv_pool_sigmoid_sum_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int pool_kernel_size);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_conv_pool_sigmoid_sum_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        
        # Initialize weights and bias manually
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_op = fused_operator

    def forward(self, x):
        return self.fused_op.fused_conv_pool_sigmoid_sum_cuda(
            x, self.weight, self.bias, self.kernel_size, self.pool_kernel_size
        )