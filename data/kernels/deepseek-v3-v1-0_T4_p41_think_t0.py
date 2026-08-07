import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm+GELU+ReLU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__inline__ __device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__global__ void fused_bn_gelu_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int num_features,
    int total_elements
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int feature_idx = index % num_features;
        float mean = running_mean[feature_idx];
        float var = running_var[feature_idx];
        float gamma = weight ? weight[feature_idx] : 1.0f;
        float beta = bias ? bias[feature_idx] : 0.0f;
        
        float inv_std = 1.0f / sqrtf(var + eps);
        float normalized = (input[index] - mean) * inv_std;
        float scaled = normalized * gamma + beta;
        
        float gelu_val = gelu(scaled);
        output[index] = fmaxf(0.0f, gelu_val);
    }
}

torch::Tensor fused_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(running_mean);
    CHECK_INPUT(running_var);

    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    int num_features = input.size(1);

    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_bn_gelu_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        num_features,
        total_elements
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
);
"""

# Compile the inline CUDA code
fused_bn_gelu_relu = load_inline(
    name="fused_bn_gelu_relu",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_bn_gelu_relu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.fused_bn_gelu_relu = fused_bn_gelu_relu

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_bn_gelu_relu.fused_bn_gelu_relu_cuda(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        return x