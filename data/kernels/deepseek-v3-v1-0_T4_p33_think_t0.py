import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Scale-BatchNorm CUDA kernel
scale_bn_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void scale_bn_forward_kernel(
    const float* input,
    const float* scale,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    float* save_mean,
    float* save_var,
    int batch_size,
    int num_features,
    float eps,
    float momentum
) {
    __shared__ float shared_mean[BLOCK_SIZE];
    __shared__ float shared_var[BLOCK_SIZE];

    int feature_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Compute mean and variance for this feature
    float mean = 0.0f;
    float var = 0.0f;

    for (int i = tid; i < batch_size; i += blockDim.x) {
        float val = input[i * num_features + feature_idx] * scale[feature_idx];
        mean += val;
        var += val * val;
    }

    // Parallel reduction for mean
    shared_mean[tid] = mean;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mean[tid] += shared_mean[tid + s];
        }
        __syncthreads();
    }
    mean = shared_mean[0] / batch_size;

    // Parallel reduction for variance
    shared_var[tid] = var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_var[tid] += shared_var[tid + s];
        }
        __syncthreads();
    }
    var = shared_var[0] / batch_size - mean * mean;

    if (tid == 0) {
        save_mean[feature_idx] = mean;
        save_var[feature_idx] = var;
        
        // Update running statistics
        running_mean[feature_idx] = (1 - momentum) * running_mean[feature_idx] + momentum * mean;
        running_var[feature_idx] = (1 - momentum) * running_var[feature_idx] + momentum * var;
    }

    __syncthreads();

    // Normalize and transform
    float inv_std = 1.0f / sqrtf(var + eps);
    float bn_scale = weight[feature_idx] * inv_std;
    float bn_shift = bias[feature_idx] - mean * bn_scale;

    for (int i = tid; i < batch_size; i += blockDim.x) {
        int idx = i * num_features + feature_idx;
        output[idx] = input[idx] * scale[feature_idx] * bn_scale + bn_shift;
    }
}

torch::Tensor scale_bn_forward_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    float momentum
) {
    int batch_size = input.size(0);
    int num_features = input.size(1);
    
    auto output = torch::empty_like(input);
    auto save_mean = torch::empty_like(scale);
    auto save_var = torch::empty_like(scale);

    dim3 blocks(num_features);
    dim3 threads(BLOCK_SIZE);

    scale_bn_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        save_mean.data_ptr<float>(),
        save_var.data_ptr<float>(),
        batch_size,
        num_features,
        eps,
        momentum
    );

    return output;
}
"""

scale_bn_cpp_source = """
torch::Tensor scale_bn_forward_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    float momentum
);
"""

# Compile the inline CUDA code
scale_bn_op = load_inline(
    name="scale_bn_op",
    cpp_sources=scale_bn_cpp_source,
    cuda_sources=scale_bn_kernel_source,
    functions=["scale_bn_forward_cuda"],
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
        x = scale_bn_op.scale_bn_forward_cuda(
            x,
            self.scale,
            self.bn_weight,
            self.bn_bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum
        )
        return x