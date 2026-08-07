import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int TILE_SIZE = 16;
constexpr int BLOCK_SIZE = 256;

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / 1.41421356237f));
}

__global__ void conv_transpose_gelu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride
) {
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (h_out >= out_height || w_out >= out_width) return;

    int n = blockIdx.x;
    int c_out = threadIdx.x;

    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_out - kh;
                int w_in = w_out - kw;
                
                if (h_in >= 0 && w_in >= 0 && h_in < in_height && w_in < in_width) {
                    if (h_in % stride == 0 && w_in % stride == 0) {
                        int input_idx = ((n * in_channels + c_in) * in_height + h_in/stride) * in_width + w_in/stride;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias) sum += bias[c_out];
    sum = gelu(sum);

    int output_idx = ((n * out_channels + c_out) * out_height + h_out) * out_width + w_out;
    output[output_idx] = sum;
}

__global__ void group_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int num_groups,
    float eps
) {
    int n = blockIdx.x;
    int g = blockIdx.y;
    int group_size = channels / num_groups;
    int c_start = g * group_size;
    int c_end = c_start + group_size;
    
    extern __shared__ float shared[];
    float* sum_shared = shared;
    float* sqsum_shared = &shared[blockDim.x];

    int tid = threadIdx.x;
    sum_shared[tid] = 0.0f;
    sqsum_shared[tid] = 0.0f;

    for (int c = c_start; c < c_end; c++) {
        for (int i = tid; i < height * width; i += blockDim.x) {
            int h = i / width;
            int w = i % width;
            int idx = ((n * channels + c) * height + h) * width + w;
            float val = input[idx];
            sum_shared[tid] += val;
            sqsum_shared[tid] += val * val;
        }
    }
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_shared[tid] += sum_shared[tid + s];
            sqsum_shared[tid] += sqsum_shared[tid + s];
        }
        __syncthreads();
    }

    float mean = sum_shared[0] / (group_size * height * width);
    float variance = sqsum_shared[0] / (group_size * height * width) - mean * mean;
    float inv_std = rsqrtf(variance + eps);

    for (int c = c_start; c < c_end; c++) {
        for (int i = tid; i < height * width; i += blockDim.x) {
            int h = i / width;
            int w = i % width;
            int idx = ((n * channels + c) * height + h) * width + w;
            float val = (input[idx] - mean) * inv_std;
            output[idx] = val * gamma[c] + beta[c];
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
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height - 1) * stride + kernel_size;
    int out_width = (in_width - 1) * stride + kernel_size;

    auto intermediate = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());
    auto output = torch::empty_like(intermediate);

    dim3 block_conv(out_channels, 1, 1);
    dim3 grid_conv(batch_size, (out_height + TILE_SIZE - 1) / TILE_SIZE, (out_width + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threads_conv(1, TILE_SIZE, TILE_SIZE);
    
    conv_transpose_gelu_kernel<<<grid_conv, threads_conv>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        intermediate.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid_norm(batch_size, num_groups);
    group_norm_kernel<<<grid_norm, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float)>>>(
        intermediate.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        out_height,
        out_width,
        num_groups,
        eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_ops_cpp_source = """
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
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
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