import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bn_bias_divide_swish_kernel(
    const float* input,
    const float* gamma, const float* beta, 
    const float* running_mean, const float* running_var,
    const float* bias_tensor, float divide_value, float eps,
    float* output,
    int total_elements, int out_features) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int j = idx % out_features;

    float x_val = input[idx];
    float mean = running_mean[j];
    float var = running_var[j];
    float g = gamma[j];
    float b = beta[j];
    float bias_val = bias_tensor[0];

    float normalized = g * (x_val - mean) / sqrtf(var + eps) + b;
    normalized += bias_val;
    normalized /= divide_value;

    float sigmoid_val = 1.0f / (1.0f + expf(-normalized));
    output[idx] = normalized * sigmoid_val;
}

torch::Tensor fused_bn_bias_divide_swish_cuda(
    torch::Tensor input,
    torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor bias_tensor, float divide_value, float eps) {

    auto total_elements = input.numel();
    auto out_features = running_mean.size(0);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bn_bias_divide_swish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        bias_tensor.data_ptr<float>(), divide_value, eps,
        output.data_ptr<float>(),
        total_elements, out_features
    );

    return output;
}
"""

fused_bn_cpp_source = (
    "torch::Tensor fused_bn_bias_divide_swish_cuda("
    "torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, "
    "torch::Tensor running_mean, torch::Tensor running_var, "
    "torch::Tensor bias_tensor, float divide_value, float eps);"
)

fused_op = load_inline(
    name="fused_bn_op",
    cpp_sources=fused_bn_cpp_source,
    cuda_sources=fused_bn_source,
    functions=["fused_bn_bias_divide_swish_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses batch normalization, bias addition, division, and Swish activation into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.fused_op = fused_op

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_op.fused_bn_bias_divide_swish_cuda(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bias,
            self.divide_value,
            self.bn.eps
        )
        return x