import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

group_norm_hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void group_norm_hardtanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int out_features,
    int num_groups,
    float eps,
    float hardtanh_min,
    float hardtanh_max
) {
    int group_size = out_features / num_groups;
    int sample_id = blockIdx.y;
    int group_id = blockIdx.x;
    int group_start = group_id * group_size;
    
    int tid = threadIdx.x;
    int idx = sample_id * out_features + group_start + tid;
    
    if (tid >= group_size) return;
    
    float x_val = input[idx];
    
    extern __shared__ float s_data[];
    s_data[tid] = x_val;
    __syncthreads();
    
    float mean = 0.0f;
    float M2 = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < group_size; i++) {
            float x = s_data[i];
            float delta = x - mean;
            mean += delta / (i + 1.0f);
            float delta2 = x - mean;
            M2 += delta * delta2;
        }
        float variance = M2 / group_size;
        float inv_std = 1.0f / sqrtf(variance + eps);
        s_data[0] = mean;
        s_data[1] = inv_std;
    }
    __syncthreads();
    
    mean = s_data[0];
    float inv_std = s_data[1];
    
    float centered = x_val - mean;
    float normalized = centered * inv_std;
    normalized = normalized * weight[group_start + tid] + bias[group_start + tid];
    normalized = fmaxf(hardtanh_min, fminf(hardtanh_max, normalized));
    
    output[idx] = normalized;
}

torch::Tensor group_norm_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    float hardtanh_min,
    float hardtanh_max,
    int num_groups
) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 1, "Weight must be 1D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    
    int batch_size = input.size(0);
    int out_features = input.size(1);
    TORCH_CHECK(out_features % num_groups == 0, "out_features must be divisible by num_groups");
    int group_size = out_features / num_groups;
    
    TORCH_CHECK(weight.size(0) == out_features, "Weight size mismatch");
    TORCH_CHECK(bias.size(0) == out_features, "Bias size mismatch");
    
    auto output = torch::empty_like(input);
    
    dim3 grid(num_groups, batch_size);
    dim3 block(group_size);
    size_t shared_mem = group_size * sizeof(float);
    
    group_norm_hardtanh_kernel<<<grid, block, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        num_groups,
        eps,
        hardtanh_min,
        hardtanh_max
    );
    
    return output;
}
"""

group_norm_hardtanh_cpp_source = (
    "torch::Tensor group_norm_hardtanh_cuda("
    "    torch::Tensor input,"
    "    torch::Tensor weight,"
    "    torch::Tensor bias,"
    "    float eps,"
    "    float hardtanh_min,"
    "    float hardtanh_max,"
    "    int num_groups);"
)

group_norm_hardtanh = load_inline(
    name="group_norm_hardtanh",
    cpp_sources=group_norm_hardtanh_cpp_source,
    cuda_sources=group_norm_hardtanh_source,
    functions=["group_norm_hardtanh_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.num_groups = num_groups
        self.eps = 1e-5
        self.group_norm_hardtanh = group_norm_hardtanh

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm_hardtanh.group_norm_hardtanh_cuda(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias,
            self.eps,
            self.hardtanh_min,
            self.hardtanh_max,
            self.num_groups
        )
        return x