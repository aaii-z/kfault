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
    int pool_kernel_depth, int pool_kernel_height, int pool_kernel_width) {
    
    extern __shared__ float shared_mem[];
    float* conv_output = shared_mem;
    float* norm_stats = &shared_mem[output_depth * output_height * output_width];
    
    int n = blockIdx.x;
    int c = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Phase 1: Transposed convolution
    for (int idx = tid; idx < output_depth * output_height * output_width; idx += num_threads) {
        int d = idx / (output_height * output_width);
        int h = (idx % (output_height * output_width)) / output_width;
        int w = idx % output_width;
        
        float conv_value = bias[c];
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_depth; kd++) {
                for (int kh = 0; kh < kernel_height; kh++) {
                    for (int kw = 0; kw < kernel_width; kw++) {
                        int input_d = d - kd + padding_depth;
                        int input_h = h - kh + padding_height;
                        int input_w = w - kw + padding_width;
                        
                        if (input_d >= 0 && input_d < input_depth * stride_depth && 
                            input_h >= 0 && input_h < input_height * stride_height && 
                            input_w >= 0 && input_w < input_width * stride_width) {
                            
                            if (input_d % stride_depth == 0 &&
                                input_h % stride_height == 0 &&
                                input_w % stride_width == 0) {
                                
                                input_d /= stride_depth;
                                input_h /= stride_height;
                                input_w /= stride_width;
                                
                                if (input_d < input_depth && input_h < input_height && input_w < input_width) {
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
            }
        }
        
        conv_output[idx] = conv_value + sum_weight;
    }
    
    __syncthreads();
    
    // Phase 2: Layer normalization statistics
    if (tid == 0) {
        float mean = 0.0f;
        float variance = 0.0f;
        int count = output_depth * output_height * output_width;
        
        for (int idx = 0; idx < count; idx++) {
            mean += conv_output[idx];
        }
        mean /= count;
        
        for (int idx = 0; idx < count; idx++) {
            float diff = conv_output[idx] - mean;
            variance += diff * diff;
        }
        variance /= count;
        
        norm_stats[0] = mean;
        norm_stats[1] = 1.0f / sqrtf(variance + 1e-5f);
    }
    
    __syncthreads();
    
    float mean = norm_stats[0];
    float inv_std = norm_stats[1];
    
    // Phase 3: Layer normalization, pooling, and GELU
    int pooled_depth = (output_depth + pool_kernel_depth - 1) / pool_kernel_depth;
    int pooled_height = (output_height + pool_kernel_height - 1) / pool_kernel_height;
    int pooled_width = (output_width + pool_kernel_width - 1) / pool_kernel_width;
    
    for (int idx = tid; idx < pooled_depth * pooled_height * pooled_width; idx += num_threads) {
        int pd = idx / (pooled_height * pooled_width);
        int ph = (idx % (pooled_height * pooled_width)) / pooled_width;
        int pw = idx % pooled_width;
        
        float sum = 0.0f;
        int count = 0;
        
        for (int d = pd * pool_kernel_depth; d < min((pd + 1) * pool_kernel_depth, output_depth); d++) {
            for (int h = ph * pool_kernel_height; h < min((ph + 1) * pool_kernel_height, output_height); h++) {
                for (int w = pw * pool_kernel_width; w < min((pw + 1) * pool_kernel_width, output_width); w++) {
                    int conv_idx = d * output_height * output_width + h * output_width + w;
                    float norm_value = (conv_output[conv_idx] - mean) * inv_std * gamma[c] + beta[c];
                    sum += gelu(norm_value);
                    count++;
                }
            }
        }
        
        int output_idx = ((n * out_channels + c) * pooled_depth + pd) * pooled_height * pooled_width + 
                        ph * pooled_width + pw;
        output[output_idx] = sum / count;
    }
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    float sum_weight, torch::Tensor gamma, torch::Tensor beta,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
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
    
    int output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth + kernel_depth;
    int output_height = (input_height - 1) * stride_height - 2 * padding_height + kernel_height;
    int output_width = (input_width - 1) * stride_width - 2 * padding_width + kernel_width;
    
    int pooled_depth = (output_depth + pool_kernel_depth - 1) / pool_kernel_depth;
    int pooled_height = (output_height + pool_kernel_height - 1) / pool_kernel_height;
    int pooled_width = (output_width + pool_kernel_width - 1) / pool_kernel_width;
    
    auto output = torch::zeros({batch_size, out_channels, pooled_depth, pooled_height, pooled_width}, 
                              torch::dtype(torch::kFloat32).device(input.device()));
    
    size_t shared_mem_size = (output_depth * output_height * output_width + 2) * sizeof(float);
    
    dim3 blocks(batch_size, out_channels);
    int threads = 256;
    
    fused_conv_transpose_add_norm_pool_gelu_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        sum_weight, gamma.data_ptr<float>(), beta.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_depth, kernel_height, kernel_width,
        stride_depth, stride_height, stride_width,
        padding_depth, padding_height, padding_width,
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
            self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2]
        )