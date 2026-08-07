import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused matmul + batchnorm + bias + divide + swish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_matmul_bn_bias_div_swish(const float* input,
                                               const float* weight,
                                               const float* bias_matmul,
                                               const float* bn_running_mean,
                                               const float* bn_running_var,
                                               const float* bn_weight,
                                               const float* bn_bias,
                                               const float* bias_param,
                                               float divide_value,
                                               float eps,
                                               float* output,
                                               int batch_size,
                                               int in_features,
                                               int out_features) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= out_features) return;

    // Perform matrix multiplication: output[row][col] = sum(input[row][k] * weight[col][k])
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias_matmul[col];

    // BatchNorm: (sum - mean) / sqrt(var + eps) * weight + bias
    float bn_mean = bn_running_mean[col];
    float bn_var = bn_running_var[col];
    float bn_w = (bn_weight == nullptr) ? 1.0f : bn_weight[col];
    float bn_b = (bn_bias == nullptr) ? 0.0f : bn_bias[col];
    float bn_out = (sum - bn_mean) / sqrtf(bn_var + eps) * bn_w + bn_b;

    // Add learned bias and divide
    float biased = bn_out + bias_param[0];
    float divided = biased / divide_value;

    // Swish: divided * sigmoid(divided)
    float sigmoid = 1.0f / (1.0f + expf(-divided));
    float result = divided * sigmoid;

    output[row * out_features + col] = result;
}

torch::Tensor fused_op_cuda(torch::Tensor input,
                            torch::Tensor weight,
                            torch::Tensor bias_matmul,
                            torch::Tensor bn_running_mean,
                            torch::Tensor bn_running_var,
                            torch::Tensor bn_weight,
                            torch::Tensor bn_bias,
                            torch::Tensor bias_param,
                            float divide_value,
                            float eps) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    dim3 block_dim(block_size);
    dim3 grid_dim((out_features + block_size - 1) / block_size, batch_size);

    fused_matmul_bn_bias_div_swish<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_matmul.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bias_param.data_ptr<float>(),
        divide_value,
        eps,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor input,
                            torch::Tensor weight,
                            torch::Tensor bias_matmul,
                            torch::Tensor bn_running_mean,
                            torch::Tensor bn_running_var,
                            torch::Tensor bn_weight,
                            torch::Tensor bn_bias,
                            torch::Tensor bias_param,
                            float divide_value,
                            float eps);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matmul + batchnorm + bias + divide + swish.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.bn_eps = bn_eps
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.bias,
            self.divide_value,
            self.bn_eps
        )