import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Scale-BatchNorm CUDA kernel
scale_bn_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void scale_bn_kernel(
    const float* input,
    const float* scale,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int num_elements,
    int num_features,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int feature_idx = idx % num_features;
        float inv_std = 1.0f / sqrtf(running_var[feature_idx] + eps);
        float normalized = (input[idx] - running_mean[feature_idx]) * inv_std;
        output[idx] = normalized * weight[feature_idx] * scale[feature_idx] + bias[feature_idx];
    }
}

torch::Tensor scale_bn_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
) {
    int batch_size = input.size(0);
    int num_features = input.size(1);
    int num_elements = batch_size * num_features;
    
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    scale_bn_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        num_features,
        eps
    );

    return output;
}
"""

scale_bn_cpp_source = """
torch::Tensor scale_bn_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
);
"""

# Compile the inline CUDA code
scale_bn_op = load_inline(
    name="scale_bn_op",
    cpp_sources=scale_bn_cpp_source,
    cuda_sources=scale_bn_kernel_source,
    functions=["scale_bn_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.gemm(x)
        if self.training:
            x = x * self.scale
            x = self.bn(x)
        else:
            x = scale_bn_op.scale_bn_cuda(
                x,
                self.scale,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps
            )
        return x