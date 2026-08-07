import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused BatchNorm and mean subtraction
fused_bn_mean_subtract_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) do { if (err != cudaSuccess) { printf("CUDA error: %s\\n", cudaGetErrorString(err)); exit(1); } } while (0)

__global__ void fused_bn_mean_subtract_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    float eps,
    int num_elements,
    int spatial_size,
    int channels) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int c = (idx / spatial_size) % channels;
    
    float mean = running_mean[c];
    float var = running_var[c];
    float w = weight ? weight[c] : 1.0f;
    float b = bias ? bias[c] : 0.0f;
    
    float scale = w / sqrtf(var + eps);
    float shift = b - mean * scale;
    
    output[idx] = input[idx] * scale + shift;
}

__global__ void compute_mean_kernel(
    const float* input,
    float* means,
    int num_elements,
    int spatial_size,
    int channels) {
    
    extern __shared__ float shared_means[];
    
    int tid = threadIdx.x;
    int c = blockIdx.x % channels;
    int batch = blockIdx.x / channels;
    
    if (tid < channels) {
        shared_means[tid] = 0.0f;
    }
    __syncthreads();
    
    float sum = 0.0f;
    int start = (batch * channels + c) * spatial_size;
    int end = start + spatial_size;
    
    for (int i = start + tid; i < end; i += blockDim.x) {
        sum += input[i];
    }
    
    atomicAdd(&shared_means[c], sum);
    __syncthreads();
    
    if (tid == 0) {
        means[blockIdx.x] = shared_means[c] / spatial_size;
    }
}

__global__ void subtract_mean_kernel(
    float* data,
    const float* means,
    int num_elements,
    int spatial_size,
    int channels) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    int batch_channel = idx / spatial_size;
    data[idx] -= means[batch_channel];
}

torch::Tensor fused_bn_mean_subtract_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {
    
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    int spatial_size = input.size(2) * input.size(3) * input.size(4);
    int channels = input.size(1);
    int batch_size = input.size(0);
    
    const int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    
    fused_bn_mean_subtract_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.numel() > 0 ? weight.data_ptr<float>() : nullptr,
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        eps,
        num_elements,
        spatial_size,
        channels);
    
    CUDA_CHECK(cudaGetLastError());
    
    auto means = torch::zeros({batch_size * channels}, input.options());
    int smem_size = channels * sizeof(float);
    
    compute_mean_kernel<<<batch_size * channels, 256, smem_size>>>(
        output.data_ptr<float>(),
        means.data_ptr<float>(),
        num_elements,
        spatial_size,
        channels);
    
    CUDA_CHECK(cudaGetLastError());
    
    subtract_mean_kernel<<<grid_size, block_size>>>(
        output.data_ptr<float>(),
        means.data_ptr<float>(),
        num_elements,
        spatial_size,
        channels);
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

fused_bn_mean_subtract_cpp_source = """
torch::Tensor fused_bn_mean_subtract_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_bn_mean_subtract_cpp_source,
    cuda_sources=fused_bn_mean_subtract_source,
    functions=["fused_bn_mean_subtract_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_bn_mean_subtract_cuda(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        return x