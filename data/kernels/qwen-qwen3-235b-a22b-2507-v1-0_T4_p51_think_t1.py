import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void fused_ops_kernel(
    const float* gemm_output,
    const float* subtract,
    const float* original_x,
    float* output,
    int batch,
    int out_features,
    int in_features
) {
    int i = blockIdx.x;
    if (i >= batch) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Compute partial sum for the row (after subtract)
    float sum = 0.0f;
    for (int k = tid; k < out_features; k += num_threads) {
        sum += gemm_output[i * out_features + k] - subtract[k];
    }

    __shared__ float sdata[BLOCK_SIZE];
    sdata[tid] = sum;
    __syncthreads();

    // Reduce to get total sum for the row
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Compute mean (same as logsumexp for single element)
    __shared__ float s_mean;
    if (tid == 0) {
        s_mean = sdata[0] / out_features;
    }
    __syncthreads();
    float mean_val = s_mean;

    // Compute GELU
    float x = mean_val;
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    float gelu_val = x * cdf;

    // Add to original_x
    for (int j = tid; j < in_features; j += num_threads) {
        output[i * in_features + j] = original_x[i * in_features + j] + gelu_val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor gemm_output, torch::Tensor subtract, torch::Tensor original_x) {
    int batch = gemm_output.size(0);
    int out_features = gemm_output.size(1);
    int in_features = original_x.size(1);
    
    auto output = torch::empty_like(original_x);
    
    const int block_size = BLOCK_SIZE;
    dim3 grid(batch);
    dim3 block(block_size);
    
    fused_ops_kernel<<<grid, block>>>(
        gemm_output.data_ptr<float>(),
        subtract.data_ptr<float>(),
        original_x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        out_features,
        in_features
    );
    
    return output;
}
"""

fused_ops_cpp = "torch::Tensor fused_ops_cuda(torch::Tensor gemm_output, torch::Tensor subtract, torch::Tensor original_x);"

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.fused_ops = fused_ops

    def forward(self, x):
        original_x = x.clone().detach()
        x = self.gemm(x)
        x = self.fused_ops.fused_ops_cuda(x, self.subtract, original_x)
        return x