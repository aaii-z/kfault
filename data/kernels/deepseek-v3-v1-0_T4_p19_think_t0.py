import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom fused kernel for ConvTranspose2d + GELU + GroupNorm
fused_conv_gelu_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_GROUP_SIZE = 1024;

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / 1.41421356237f));
}

__global__ void fused_conv_transpose_gelu_norm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int groups,
    int num_groups,
    float eps
) {
    int n = blockIdx.z;
    int group = blockIdx.y;
    int group_size = out_channels / groups;
    int group_start = group * group_size;
    
    extern __shared__ float shared[];
    float* group_sum = shared;
    float* group_sqsum = &shared[MAX_GROUP_SIZE];
    
    // Initialize shared memory for reduction
    if (threadIdx.x < MAX_GROUP_SIZE) {
        group_sum[threadIdx.x] = 0.0f;
        group_sqsum[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // Compute deconvolution and GELU
    for (int c = group_start; c < group_start + group_size; c++) {
        for (int h = threadIdx.x; h < out_height; h += blockDim.x) {
            for (int w = threadIdx.y; w < out_width; w += blockDim.y) {
                float val = 0.0f;
                
                // Deconvolution computation
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int in_h = h - kh;
                        int in_w = w - kw;
                        if (in_h >= 0 && in_w >= 0 && in_h < in_height && in_w < in_width) {
                            for (int ic = 0; ic < in_channels; ic++) {
                                int weight_idx = ((c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                int input_idx = ((n * in_channels + ic) * in_height + in_h) * in_width + in_w;
                                val += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                
                // Add bias and apply GELU
                if (bias) val += bias[c];
                val = gelu(val);
                
                // Store intermediate result and accumulate statistics
                int out_idx = ((n * out_channels + c) * out_height + h) * out_width + w;
                output[out_idx] = val;
                
                // Accumulate statistics for GroupNorm
                atomicAdd(&group_sum[threadIdx.x], val);
                atomicAdd(&group_sqsum[threadIdx.x], val * val);
            }
        }
    }
    __syncthreads();
    
    // Compute mean and variance for GroupNorm
    float mean = 0.0f, variance = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        mean += group_sum[i];
        variance += group_sqsum[i];
    }
    mean /= (group_size * out_height * out_width);
    variance = variance / (group_size * out_height * out_width) - mean * mean;
    float inv_std = rsqrtf(variance + eps);
    
    // Apply GroupNorm normalization
    for (int c = group_start; c < group_start + group_size; c++) {
        for (int h = threadIdx.x; h < out_height; h += blockDim.x) {
            for (int w = threadIdx.y; w < out_width; w += blockDim.y) {
                int out_idx = ((n * out_channels + c) * out_height + h) * out_width + w;
                float val = (output[out_idx] - mean) * inv_std;
                if (gamma && beta) {
                    val = val * gamma[c] + beta[c];
                }
                output[out_idx] = val;
            }
        }
    }
}

torch::Tensor fused_conv_transpose_gelu_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int stride,
    int groups,
    int num_groups,
    float eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    if (gamma.defined()) CHECK_INPUT(gamma);
    if (beta.defined()) CHECK_INPUT(beta);
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height - 1) * stride + kernel_size;
    int out_width = (in_width - 1) * stride + kernel_size;
    
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());
    
    dim3 blocks(num_groups, groups, batch_size);
    dim3 threads(32, 32);
    size_t shared_mem = 2 * MAX_GROUP_SIZE * sizeof(float);
    
    fused_conv_transpose_gelu_norm_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        gamma.defined() ? gamma.data_ptr<float>() : nullptr,
        beta.defined() ? beta.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        groups,
        num_groups,
        eps
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_conv_gelu_norm_cpp_source = """
torch::Tensor fused_conv_transpose_gelu_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int stride,
    int groups,
    int num_groups,
    float eps
);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_conv_gelu_norm_cpp_source,
    cuda_sources=fused_conv_gelu_norm_source,
    functions=["fused_conv_transpose_gelu_norm_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.num_groups = num_groups
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.eps = 1e-5

    def forward(self, x):
        return fused_ops.fused_conv_transpose_gelu_norm_cuda(
            x, self.weight, self.bias, self.gamma, self.beta,
            self.stride, self.groups, self.num_groups, self.eps
        )