import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_bn_scale_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

__global__ void fused_bn_scale_softmax_kernel(
    const float* Y,
    const float* bn_weight,
    const float* bn_bias,
    const float* running_mean,
    const float* running_var,
    float scale,
    float eps,
    float* output,
    int batch_size,
    int out_features) 
{
    int i = blockIdx.x;
    if (i >= batch_size) return;

    extern __shared__ float s_data[];
    float* s_max = s_data;
    float* s_sum = s_data + blockDim.x;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    float max_val = -FLT_MAX;
    float total_sum = 0.0f;

    // Compute Z[j] = new_gamma[j] * Y[i][j] + new_beta[j] and find max
    for (int j = tid; j < out_features; j += num_threads) {
        float std_val = sqrtf(running_var[j] + eps);
        float new_gamma = scale * bn_weight[j] / std_val;
        float new_beta = scale * (bn_bias[j] - bn_weight[j] * running_mean[j] / std_val);
        float z_val = new_gamma * Y[i * out_features + j] + new_beta;
        if (z_val > max_val) max_val = z_val;
    }

    // Reduce max within block
    s_max[tid] = max_val;
    __syncthreads();
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }
    max_val = s_max[0];

    // Compute exp(z_val - max_val) and sum
    float exp_sum = 0.0f;
    for (int j = tid; j < out_features; j += num_threads) {
        float std_val = sqrtf(running_var[j] + eps);
        float new_gamma = scale * bn_weight[j] / std_val;
        float new_beta = scale * (bn_bias[j] - bn_weight[j] * running_mean[j] / std_val);
        float z_val = new_gamma * Y[i * out_features + j] + new_beta;
        exp_sum += expf(z_val - max_val);
    }

    // Reduce sum within block
    s_sum[tid] = exp_sum;
    __syncthreads();
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    total_sum = s_sum[0];

    // Compute output[j] = exp(z_val - max_val) / total_sum
    for (int j = tid; j < out_features; j += num_threads) {
        float std_val = sqrtf(running_var[j] + eps);
        float new_gamma = scale * bn_weight[j] / std_val;
        float new_beta = scale * (bn_bias[j] - bn_weight[j] * running_mean[j] / std_val);
        float z_val = new_gamma * Y[i * out_features + j] + new_beta;
        output[i * out_features + j] = expf(z_val - max_val) / total_sum;
    }
}

torch::Tensor fused_bn_scale_softmax_cuda(
    torch::Tensor Y,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor scale,
    float eps) 
{
    int batch_size = Y.size(0);
    int out_features = Y.size(1);
    
    auto output = torch::zeros_like(Y);
    
    const int block_size = 256;
    int num_blocks = batch_size;
    size_t shared_mem_size = 2 * block_size * sizeof(float);

    fused_bn_scale_softmax_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        Y.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        scale.item<float>(),
        eps,
        output.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    return output;
}
"""

fused_bn_scale_softmax_cpp = "torch::Tensor fused_bn_scale_softmax_cuda(torch::Tensor Y, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor scale, float eps);"

fused_bn_scale_softmax = load_inline(
    name="fused_bn_scale_softmax",
    cpp_sources=fused_bn_scale_softmax_cpp,
    cuda_sources=fused_bn_scale_softmax_source,
    functions=["fused_bn_scale_softmax_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses BatchNorm, scaling, and Softmax into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.bn_eps = bn_eps
        self.fused_op = fused_bn_scale_softmax

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_op.fused_bn_scale_softmax_cuda(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.scale,
            self.bn_eps
        )
        return x