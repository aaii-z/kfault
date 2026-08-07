import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_instance_norm_and_divide_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

using namespace torch;

__global__ void compute_mean_variance_kernel(
    const float* input, 
    float* mean, 
    float* variance, 
    int N, int C, int H, int W,
    int stride_n, int stride_c, int stride_h, int stride_w)
{
    int c = blockIdx.x;
    int n = blockIdx.y;
    if (c >= C || n >= N) return;

    int size = H * W;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int index = tid; index < size; index += num_threads) {
        int h = index / W;
        int w = index % W;
        float val = input[n * stride_n + c * stride_c + h * stride_h + w * stride_w];
        local_sum += val;
        local_sum_sq += val * val;
    }

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = sdata + blockDim.x;

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = s_sum[0];
        float total_sum_sq = s_sum_sq[0];
        float mean_val = total_sum / size;
        float var_val = (total_sum_sq / size) - (mean_val * mean_val);
        if (var_val < 0.0f) var_val = 0.0f;
        mean[n * C + c] = mean_val;
        variance[n * C + c] = var_val;
    }
}

__global__ void normalize_kernel(
    const float* input, 
    float* output, 
    const float* mean, 
    const float* variance, 
    int N, int C, int H, int W,
    float divide_by,
    float eps,
    int stride_n, int stride_c, int stride_h, int stride_w)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C * H * W) return;

    int w = index % W;
    int h = (index / W) % H;
    int c = (index / (W * H)) % C;
    int n = index / (W * H * C);

    float mean_val = mean[n * C + c];
    float var_val = variance[n * C + c];
    float factor = 1.0f / (sqrtf(var_val + eps) * divide_by);
    float val = input[n * stride_n + c * stride_c + h * stride_h + w * stride_w];
    output[n * stride_n + c * stride_c + h * stride_h + w * stride_w] = (val - mean_val) * factor;
}

Tensor custom_instance_norm_and_divide_forward(
    Tensor input, 
    float divide_by, 
    float eps) 
{
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = at::empty_like(input);
    auto mean = at::empty({N, C}, input.options());
    auto variance = at::empty({N, C}, input.options());

    dim3 block_reduction(256);
    dim3 grid_reduction(C, N);
    size_t shared_mem = 2 * block_reduction.x * sizeof(float);

    compute_mean_variance_kernel<<<grid_reduction, block_reduction, shared_mem>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        variance.data_ptr<float>(),
        N, C, H, W,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3)
    );

    int total_elements = N * C * H * W;
    int block_size_norm = 256;
    int grid_size_norm = (total_elements + block_size_norm - 1) / block_size_norm;

    normalize_kernel<<<grid_size_norm, block_size_norm>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        variance.data_ptr<float>(),
        N, C, H, W,
        divide_by,
        eps,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3)
    );

    return output;
}
"""

custom_instance_norm_and_divide_cpp_source = (
    "Tensor custom_instance_norm_and_divide_forward(Tensor input, float divide_by, float eps);"
)

custom_instance_norm_and_divide = load_inline(
    name="custom_instance_norm_and_divide",
    cpp_sources=custom_instance_norm_and_divide_cpp_source,
    cuda_sources=custom_instance_norm_and_divide_source,
    functions=["custom_instance_norm_and_divide_forward"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by
        self.eps = 1e-5
        self.custom_op = custom_instance_norm_and_divide

    def forward(self, x):
        x = self.conv(x)
        x = self.custom_op.custom_instance_norm_and_divide_forward(x, self.divide_by, self.eps)
        return x