import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_pool_sigmoid_sum_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output, 
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int kernel_size, int pool_kernel_size) {
    
    const int output_height = (input_height - kernel_size + 1) / pool_kernel_size;
    const int output_width = (input_width - kernel_size + 1) / pool_kernel_size;
    const int spatial_size = output_height * output_width;
    
    extern __shared__ float shared_mem[];
    float* conv_shared = shared_mem;
    
    int tid = threadIdx.x;
    int batch = blockIdx.x;
    int out_c = blockIdx.y;
    
    if (batch >= batch_size || out_c >= out_channels) return;
    
    float channel_sum = 0.0f;
    
    for (int h_start = 0; h_start < output_height; h_start += gridDim.z) {
        for (int w_start = 0; w_start < output_width; w_start += blockDim.x) {
            int h = h_start + blockIdx.z;
            int w = w_start + tid;
            
            if (h < output_height && w < output_width) {
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
                
                if (pool_count > 0) {
                    float avg_val = pool_sum / pool_count;
                    float sigmoid_val = 1.0f / (1.0f + expf(-avg_val));
                    channel_sum += sigmoid_val;
                }
            }
        }
    }
    
    __shared__ float block_sum[256];
    block_sum[tid] = channel_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_sum[tid] += block_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        int output_idx = batch * out_channels + out_c;
        output[output_idx] = block_sum[0];
    }
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
    
    auto output = torch::zeros({batch_size, out_channels}, input.options());
    
    dim3 grid_dim(batch_size, out_channels, (input_height - kernel_size + 1) / pool_kernel_size);
    dim3 block_dim(256);
    
    size_t shared_mem_size = 0;
    
    fused_conv_pool_sigmoid_sum_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
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
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_op = fused_operator

    def forward(self, x):
        return self.fused_op.fused_conv_pool_sigmoid_sum_cuda(
            x, self.weight, self.bias, self.kernel_size, self.pool_kernel_size
        )