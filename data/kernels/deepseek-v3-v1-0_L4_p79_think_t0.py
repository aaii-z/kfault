import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

#define CUDA_CHECK(err) do { \\
    cudaError_t err_ = (err); \\
    if (err_ != cudaSuccess) { \\
        printf("CUDA error %d at %s:%d\\n", err_, __FILE__, __LINE__); \\
        throw std::runtime_error(cudaGetErrorString(err_)); \\
    } \\
} while (0)

__global__ void mean_var_kernel(
    const float* input,
    const float* multiplier,
    float* mean,
    float* var,
    int B,
    int C,
    int D,
    int H,
    int W
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int total_spatial = D * H * W;

    extern __shared__ float sdata[];
    float* sdata_sum = sdata;
    float* sdata_sum_sq = (float*)&sdata_sum[blockDim.x];

    int tid = threadIdx.x;
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;

    for (int i = tid; i < total_spatial; i += blockDim.x) {
        int d = i / (H * W);
        int remainder = i % (H * W);
        int h = remainder / W;
        int w = remainder % W;

        int idx = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        float val = input[idx] * multiplier[c];
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    sdata_sum[tid] = thread_sum;
    sdata_sum_sq[tid] = thread_sum_sq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_sum[tid] += sdata_sum[tid + s];
            sdata_sum_sq[tid] += sdata_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = sdata_sum[0];
        float total_sum_sq = sdata_sum_sq[0];
        float mean_val = total_sum / total_spatial;
        float variance_val = total_sum_sq / total_spatial - mean_val * mean_val;
        mean[b * C + c] = mean_val;
        var[b * C + c] = variance_val;
    }
}

__global__ void normalize_clamp_multiply_max_kernel(
    const float* input,
    const float* multiplier,
    const float* mean,
    const float* var,
    float clamp_min,
    float clamp_max,
    float* output,
    int B,
    int C,
    int D,
    int H,
    int W
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial_batch = B * D * H * W;
    if (index >= total_spatial_batch) return;

    int b = index / (D * H * W);
    int remainder = index % (D * H * W);
    int d = remainder / (H * W);
    remainder = remainder % (H * W);
    int h = remainder / W;
    int w = remainder % W;

    float max_val = -FLT_MAX;
    for (int c = 0; c < C; c++) {
        int input_idx = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        float val = input[input_idx] * multiplier[c];
        float mean_val = mean[b * C + c];
        float var_val = var[b * C + c];
        float normalized = (val - mean_val) / sqrtf(var_val + 1e-5f);
        normalized = fminf(fmaxf(normalized, clamp_min), clamp_max);
        normalized = normalized * multiplier[c];
        if (normalized > max_val) {
            max_val = normalized;
        }
    }
    output[index] = max_val;
}

torch::Tensor compute_mean_var_cuda(
    torch::Tensor input,
    torch::Tensor multiplier,
    torch::Tensor mean,
    torch::Tensor var
) {
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    dim3 grid(B, C);
    int block_size = 256;
    int shared_mem_size = 2 * block_size * sizeof(float);

    mean_var_kernel<<<grid, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        B, C, D, H, W
    );
    CUDA_CHECK(cudaGetLastError());
    return torch::cat({mean.unsqueeze(-1), var.unsqueeze(-1)}, -1);
}

torch::Tensor compute_normalize_clamp_multiply_max_cuda(
    torch::Tensor input,
    torch::Tensor multiplier,
    torch::Tensor mean,
    torch::Tensor var,
    float clamp_min,
    float clamp_max
) {
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int total_spatial_batch = B * D * H * W;
    int block_size = 256;
    int grid_size = (total_spatial_batch + block_size - 1) / block_size;

    auto output = torch::empty({B, D, H, W}, input.options());

    normalize_clamp_multiply_max_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        clamp_min,
        clamp_max,
        output.data_ptr<float>(),
        B, C, D, H, W
    );
    CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

cpp_source = """
torch::Tensor compute_mean_var_cuda(torch::Tensor input, torch::Tensor multiplier, torch::Tensor mean, torch::Tensor var);
torch::Tensor compute_normalize_clamp_multiply_max_cuda(torch::Tensor input, torch::Tensor multiplier, torch::Tensor mean, torch::Tensor var, float clamp_min, float clamp_max);
"""

custom_ops = load_inline(
    name='custom_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        'compute_mean_var_cuda',
        'compute_normalize_clamp_multiply_max_cuda'
    ],
    verbose=True,
    extra_cflags=['-O2'],
    extra_ldflags=['-lcudart']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        B, C, D, H, W = x.shape
        mean = torch.empty(B, C, device=x.device, dtype=x.dtype)
        var = torch.empty(B, C, device=x.device, dtype=x.dtype)
        multiplier_flat = self.multiplier.view(-1)
        
        custom_ops.compute_mean_var_cuda(x, multiplier_flat, mean, var)
        x = custom_ops.compute_normalize_clamp_multiply_max_cuda(
            x, multiplier_flat, mean, var, self.clamp_min, self.clamp_max
        )
        return x