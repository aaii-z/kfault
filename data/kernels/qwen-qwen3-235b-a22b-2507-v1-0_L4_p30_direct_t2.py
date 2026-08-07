import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + GroupNorm + HardTanh
fused_gemm_gn_hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline void cudaCheckError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
}

__global__ void fused_gemm_gn_hardtanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ weight_gn,
    const float* __restrict__ bias_gn,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups,
    float hardtanh_min,
    float hardtanh_max
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= out_features) return;

    // GEMM: Compute dot product for this row and column
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    float result = sum + bias[col];

    int group_size = out_features / num_groups;
    int group_id = col / group_size;

    // Shared memory for group statistics: [data][mean][var]
    extern __shared__ float sdata[];
    float* group_data = sdata;
    float* mean_ptr = &sdata[group_size];
    float* var_ptr = &sdata[group_size + 1];

    // Each thread in the group loads one element
    int local_idx = col % group_size;
    group_data[local_idx] = result;
    __syncthreads();

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < group_size; ++i) {
        mean += group_data[i];
    }
    mean /= group_size;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < group_size; ++i) {
        float diff = group_data[i] - mean;
        var += diff * diff;
    }
    var /= group_size;

    // One thread writes mean and variance
    if (local_idx == 0) {
        *mean_ptr = mean;
        *var_ptr = var;
    }
    __syncthreads();

    mean = *mean_ptr;
    var = *var_ptr;

    // Normalize
    float eps = 1e-5f;
    float inv_std = rsqrtf(var + eps);
    float norm_val = (result - mean) * inv_std;

    // Apply affine transform
    norm_val = norm_val * weight_gn[group_id] + bias_gn[group_id];

    // Apply HardTanh
    norm_val = fmaxf(hardtanh_min, fminf(hardtanh_max, norm_val));

    // Write output
    output[row * out_features + col] = norm_val;
}

torch::Tensor fused_gemm_gn_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    int64_t num_groups,
    float hardtanh_min,
    float hardtanh_max
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(weight_gn);
    CHECK_INPUT(bias_gn);

    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    const int grid_x = (out_features + block_size - 1) / block_size;
    const int grid_y = batch_size;
    dim3 grid(grid_x, grid_y);
    dim3 block(block_size);

    int group_size = out_features / num_groups;
    size_t shared_mem_size = group_size * sizeof(float) + 2 * sizeof(float);

    fused_gemm_gn_hardtanh_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        weight_gn.data_ptr<float>(),
        bias_gn.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        num_groups,
        hardtanh_min,
        hardtanh_max
    );

    cudaCheckError();
    return output;
}
"""

fused_gemm_gn_hardtanh_cpp_source = """
torch::Tensor fused_gemm_gn_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    int64_t num_groups,
    float hardtanh_min,
    float hardtanh_max);
"""

fused_op = load_inline(
    name="fused_gemm_gn_hardtanh",
    cpp_sources=fused_gemm_gn_hardtanh_cpp_source,
    cuda_sources=fused_gemm_gn_hardtanh_source,
    functions=["fused_gemm_gn_hardtanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with fused GEMM + GroupNorm + HardTanh in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        self.linear = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_gemm_gn_hardtanh_cuda(
            x,
            self.linear.weight,
            self.linear.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.num_groups,
            self.hardtanh_min,
            self.hardtanh_max
        )