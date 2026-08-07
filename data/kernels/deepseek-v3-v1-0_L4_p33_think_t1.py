import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Scale-BatchNorm CUDA kernel for inference
scale_bn_inference_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void scale_bn_inference_kernel(
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
        float bn_scale = weight[feature_idx] * inv_std;
        float bn_shift = bias[feature_idx] - running_mean[feature_idx] * bn_scale;
        output[idx] = input[idx] * scale[feature_idx] * bn_scale + bn_shift;
    }
}

torch::Tensor scale_bn_inference_cuda(
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

    scale_bn_inference_kernel<<<num_blocks, block_size>>>(
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

scale_bn_inference_cpp_source = """
torch::Tensor scale_bn_inference_cuda(
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
scale_bn_inference_op = load_inline(
    name="scale_bn_inference_op",
    cpp_sources=scale_bn_inference_cpp_source,
    cuda_sources=scale_bn_inference_kernel_source,
    functions=["scale_bn_inference_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        x = self.gemm(x)
        if self.training:
            x = x * self.scale
            x = torch.nn.functional.batch_norm(
                x, 
                self.running_mean, 
                self.running_var, 
                self.bn_weight, 
                self.bn_bias,
                training=True,
                momentum=self.momentum,
                eps=self.eps
            )
        else:
            x = scale_bn_inference_op.scale_bn_inference_cuda(
                x,
                self.scale,
                self.bn_weight,
                self.bn_bias,
                self.running_mean,
                self.running_var,
                self.eps
            )
        return x