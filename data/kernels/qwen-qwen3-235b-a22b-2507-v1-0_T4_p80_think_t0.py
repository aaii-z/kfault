import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <float.h>

__global__ void fused_max_mean_sub_gelu_kernel(
    const float* input,
    float* output,
    int B,
    int O
) {
    int i = blockIdx.x;
    if (i >= B) return;

    float max_val = -FLT_MAX;
    float sum_val = 0.0f;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int j = tid; j < O; j += stride) {
        float val = input[i * O + j];
        if (val > max_val) max_val = val;
        sum_val += val;
    }

    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];
    shared_max[tid] = max_val;
    shared_sum[tid] = sum_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
            }
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float row_max = shared_max[0];
        float row_mean = shared_sum[0] / static_cast<float>(O);
        float result = row_max - row_mean;
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (result + 0.044715f * result * result * result)));
        output[i] = result * cdf;
    }
}

torch::Tensor fused_max_mean_sub_gelu_cuda(torch::Tensor input) {
    int B = input.size(0);
    int O = input.size(1);
    auto output = torch::empty({B, 1}, input.options());

    dim3 block_size(256);
    dim3 grid_size(B);

    fused_max_mean_sub_gelu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        O
    );

    return output;
}
"""

fused_op_cpp = "torch::Tensor fused_max_mean_sub_gelu_cuda(torch::Tensor input);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp,
    cuda_sources=fused_op_source,
    functions=["fused_max_mean_sub_gelu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.fused_op = fused_op

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_op.fused_max_mean_sub_gelu_cuda(x)
        return x