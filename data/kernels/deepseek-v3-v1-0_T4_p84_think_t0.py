import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for BatchNorm+Scale+Softmax
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

const int BLOCK_SIZE = 256;
const int MAX_BLOCK_DIM = 1024;

template<typename T>
__global__ void fused_bn_scale_softmax_forward_kernel(
    T* output,
    const T* input,
    const T* weight,
    const T* bias,
    const T* scale,
    const T* running_mean,
    const T* running_var,
    T* save_mean,
    T* save_inv_std,
    int batch_size,
    int num_features,
    T eps,
    T momentum,
    bool training) {
    
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_[];
    T* shared = reinterpret_cast<T*>(shared_memory_);
    
    int tid = threadIdx.x;
    int feature_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    
    if (feature_idx >= num_features || batch_idx >= batch_size) return;
    
    // Load running statistics
    T mean_val, var_val;
    if (training) {
        // Compute mean and variance across batch
        __shared__ T mean_shared, var_shared;
        
        if (tid == 0) {
            T sum = 0.0;
            T sq_sum = 0.0;
            for (int i = 0; i < batch_size; ++i) {
                T val = input[i * num_features + feature_idx];
                sum += val;
                sq_sum += val * val;
            }
            mean_shared = sum / batch_size;
            var_shared = (sq_sum / batch_size) - (mean_shared * mean_shared);
        }
        __syncthreads();
        
        mean_val = mean_shared;
        var_val = var_shared;
        
        if (tid == 0 && save_mean && save_inv_std) {
            save_mean[feature_idx] = mean_val;
            save_inv_std[feature_idx] = 1.0 / sqrt(var_val + eps);
        }
    } else {
        mean_val = running_mean[feature_idx];
        var_val = running_var[feature_idx];
    }
    
    // Apply BatchNorm and Scale
    T inv_std = 1.0 / sqrt(var_val + eps);
    T bn_scale = weight[feature_idx] * scale[0] * inv_std;
    T bn_shift = bias[feature_idx] * scale[0] - mean_val * bn_scale;
    
    int idx = batch_idx * num_features + feature_idx;
    T x = input[idx];
    T normalized = x * bn_scale + bn_shift;
    
    // Store normalized value and prepare for softmax
    output[idx] = normalized;
    shared[tid] = normalized;
    __syncthreads();
    
    // Softmax reduction within block
    typedef cub::BlockReduce<T, MAX_BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    // Find max value
    T max_val = normalized;
    for (int i = tid; i < num_features; i += blockDim.x) {
        max_val = max(max_val, output[batch_idx * num_features + i]);
    }
    max_val = BlockReduce(temp_storage).Reduce(max_val, cub::Max());
    
    if (tid == 0) {
        shared[MAX_BLOCK_DIM] = max_val;
    }
    __syncthreads();
    max_val = shared[MAX_BLOCK_DIM];
    
    // Compute exp and sum
    T exp_val = exp(normalized - max_val);
    T sum_val = exp_val;
    for (int i = tid + blockDim.x; i < num_features; i += blockDim.x) {
        sum_val += exp(output[batch_idx * num_features + i] - max_val);
    }
    sum_val = BlockReduce(temp_storage).Reduce(sum_val, cub::Sum());
    
    if (tid == 0) {
        shared[MAX_BLOCK_DIM + 1] = sum_val;
    }
    __syncthreads();
    sum_val = shared[MAX_BLOCK_DIM + 1];
    
    // Normalize
    output[idx] = exp_val / sum_val;
}

torch::Tensor fused_bn_scale_softmax_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor save_mean,
    torch::Tensor save_inv_std,
    double eps,
    double momentum,
    bool training) {
    
    auto batch_size = input.size(0);
    auto num_features = input.size(1);
    auto output = torch::empty_like(input);
    
    dim3 blocks(num_features, batch_size);
    int shared_mem_size = (MAX_BLOCK_DIM + 2) * sizeof(float);
    
    fused_bn_scale_softmax_forward_kernel<float><<<blocks, MAX_BLOCK_DIM, shared_mem_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        save_mean.data_ptr<float>(),
        save_inv_std.data_ptr<float>(),
        batch_size,
        num_features,
        eps,
        momentum,
        training);
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_bn_scale_softmax_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor save_mean,
    torch::Tensor save_inv_std,
    double eps,
    double momentum,
    bool training);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=['fused_bn_scale_softmax_forward'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.register_buffer('save_mean', torch.zeros(out_features))
        self.register_buffer('save_inv_std', torch.zeros(out_features))
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

    def forward(self, x):
        x = self.gemm(x)
        
        x = fused_ops.fused_bn_scale_softmax_forward(
            x,
            self.bn.weight,
            self.bn.bias,
            self.scale,
            self.bn.running_mean,
            self.bn.running_var,
            self.save_mean,
            self.save_inv_std,
            self.bn_eps,
            self.bn_momentum,
            self.training
        )
        
        return x