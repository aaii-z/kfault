import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_add_norm_pool_gelu_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float sum_weight, 
    const float* __restrict__ gamma, 
    const float* __restrict__ beta, 
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int output_padding_depth, int output_padding_height, int output_padding_width,
    int pool_kernel_depth, int pool_kernel_height, int pool_kernel_width) {
    
    extern __shared__ float shared_mem[];
    
    int pooled_depth = (output_depth + pool_kernel_depth - 1) / pool_kernel_depth;
    int pooled_height = (output_height + pool_kernel_height - 1) / pool_kernel_height;
    int pooled_width = (output_width + pool_kernel_width - 1) / pool_kernel_width;
    
    int n = blockIdx.z;
    int c = blockIdx.y;
    int d_pool = blockIdx.x / (pooled_height * pooled_width);
    int hw_idx = blockIdx.x % (pooled_height * pooled_width);
    int h_pool = hw_idx / pooled_width;
    int w_pool = hw_idx % pooled_width;
    
    if (n >= batch_size || c >= out_channels || d_pool >= pooled_depth || h_pool >= pooled_height || w_pool >= pooled_width) {
        return;
    }
    
    int d_start = d_pool * pool_kernel_depth;
    int d_end = min(d_start + pool_kernel_depth, output_depth);
    int h_start = h_pool * pool_kernel_height;
    int h_end = min(h_start + pool_kernel_height, output_height);
    int w_start = w_pool * pool_kernel_width;
    int w_end = min(w_start + pool_kernel_width, output_width);
    
    float* conv_results = shared_mem;
    float* norm_results = shared_mem + pool_kernel_depth * pool_kernel_height * pool_kernel_width;
    
    int thread_id = threadIdx.x;
    int total_threads = blockDim.x;
    int total_elements = (d_end - d_start) * (h_end - h_start) * (w_end - w_start);
    
    for (int idx = thread_id; idx < total_elements; idx += total_threads) {
        int d_offset = idx / ((h_end - h_start) * (w_end - w_start));
        int h_offset = (idx % ((h_end - h_start) * (w_end - w_start))) / (w_end - w_start);
        int w_offset = idx % (w_end - w_start);
        
        int d = d_start + d_offset;
        int h = h_start + h_offset;
        int w = w_start + w_offset;
        
        float conv_value = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_depth; kd++) {
                for (int kh = 0; kh < kernel_height; kh++) {
                    for (int kw = 0; kw < kernel_width; kw++) {
                        int input_d = (d - kd + padding_depth) / stride_depth;
                        int input_h = (h - kh + padding_height) / stride_height;
                        int input_w = (w - kw + padding_width) / stride_width;
                        
                        if (input_d >= 0 && input_d < input_depth && 
                            input_h >= 0 && input_h < input_height && 
                            input_w >= 0 && input_w < input_width &&
                            (d - kd + padding_depth) % stride_depth == 0 &&
                            (h - kh + padding_height) % stride_height == 0 &&
                            (w - kw + padding_width) % stride_width == 0) {
                            
                            int input_idx = ((n * in_channels + ic) * input_depth + input_d) * input_height * input_width + 
                                          input_h * input_width + input_w;
                            int weight_idx = ((c * in_channels + ic) * kernel_depth + kd) * kernel_height * kernel_width + 
                                           kh * kernel_width + kw;
                            
                            conv_value += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        conv_value += bias[c] + sum_weight;
        conv_results[idx] = conv_value;
    }
    
    __syncthreads();
    
    if (thread_id == 0) {
        float mean = 0.0f;
        float variance = 0.0f;
        
        for (int i = 0; i < total_elements; i++) {
            mean += conv_results[i];
            variance += conv_results[i] * conv_results[i];
        }
        
        mean /= total_elements;
        variance = variance / total_elements - mean * mean;
        float inv_std = 1.0f / sqrtf(variance + 1e-5f);
        
        for (int i = 0; i < total_elements; i++) {
            norm_results[i] = (conv_results[i] - mean) * inv_std * gamma[c] + beta[c];
            norm_results[i] = gelu(norm_results[i]);
        }
        
        float pooled_value = 0.0f;
        for (int i = 0; i < total_elements; i++) {
            pooled_value += norm_results[i];
        }
        
        int output_idx = ((n * out_channels + c) * pooled_depth + d_pool) * pooled_height * pooled_width + 
                        h_pool * pooled_width + w_pool;
        output[output_idx] = pooled_value / total_elements;
    }
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    float sum_weight, torch::Tensor gamma, torch::Tensor beta,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int output_padding_depth, int output_padding_height, int output_padding_width,
    int pool_kernel_depth, int pool_kernel_height, int pool_kernel_width) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);
    
    int output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth;
    int output_height = (input_height - 1) * stride_height - 2 * padding_height + kernel_height + output_padding_height;
    int output_width = (input_width - 1) * stride_width - 2 * padding_width + kernel_width + output_padding_width;
    
    int pooled_depth = (output_depth + pool_kernel_depth - 1) / pool_kernel_depth;
    int pooled_height = (output_height + pool_kernel_height - 1) / pool_kernel_height;
    int pooled_width = (output_width + pool_kernel_width - 1) / pool_kernel_width;
    
    auto output = torch::zeros({batch_size, out_channels, pooled_depth, pooled_height, pooled_width}, 
                              torch::dtype(torch::kFloat32).device(input.device()));
    
    int max_pool_elements = pool_kernel_depth * pool_kernel_height * pool_kernel_width;
    int shared_mem_size = 2 * max_pool_elements * sizeof(float);
    
    dim3 blocks(pooled_depth * pooled_height * pooled_width, out_channels, batch_size);
    int threads = min(256, max_pool_elements);
    
    fused_conv_transpose_add_norm_pool_gelu_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        sum_weight, gamma.data_ptr<float>(), beta.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_depth, kernel_height, kernel_width,
        stride_depth, stride_height, stride_width,
        padding_depth, padding_height, padding_width,
        output_padding_depth, output_padding_height, output_padding_width,
        pool_kernel_depth, pool_kernel_height, pool_kernel_width);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    float sum_weight, torch::Tensor gamma, torch::Tensor beta,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int output_padding_depth, int output_padding_height, int output_padding_width,
    int pool_kernel_depth, int pool_kernel_height, int pool_kernel_width);
"""

fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.pool_kernel_size = pool_kernel_size
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.gamma = nn.Parameter(torch.ones(norm_shape))
        self.beta = nn.Parameter(torch.zeros(norm_shape))
        
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias, self.sum_weight.item(), 
            self.gamma, self.beta,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2]
        )