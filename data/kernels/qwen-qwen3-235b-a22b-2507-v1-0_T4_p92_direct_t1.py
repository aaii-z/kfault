import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused GroupNorm + Tanh + HardSwish
fused_gn_act_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float hard_swish(float x) {
    float tmp = fmaxf(0.0f, fminf(3.0f, x + 3.0f));
    return x * tmp / 6.0f;
}

__global__ void fused_gn_tanh_hardswish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int num_channels,
    int height,
    int width,
    int groups,
    float eps) {

    int total_elements = batch_size * num_channels * height * width;
    CUDA_KERNEL_LOOP(idx, total_elements) {
        int n = idx / (num_channels * height * width);
        int c = (idx / (height * width)) % num_channels;
        int h = (idx / width) % height;
        int w = idx % width;

        int group_id = c * groups / num_channels;
        int channels_per_group = num_channels / groups;
        int group_start = group_id * channels_per_group;
        int group_size = channels_per_group * height * width;

        float mean = 0.0f, var = 0.0f;

        // Compute mean
        for (int i = 0; i < group_size; ++i) {
            int c_inner = i / (height * width);
            int h_inner = (i / width) % height;
            int w_inner = i % width;
            int offset = n * num_channels * height * width +
                        (group_start + c_inner) * height * width + h_inner * width + w_inner;
            mean += input[offset];
        }
        mean /= group_size;

        // Compute variance
        for (int i = 0; i < group_size; ++i) {
            int c_inner = i / (height * width);
            int h_inner = (i / width) % height;
            int w_inner = i % width;
            int offset = n * num_channels * height * width +
                        (group_start + c_inner) * height * width + h_inner * width + w_inner;
            float diff = input[offset] - mean;
            var += diff * diff;
        }
        var /= group_size;
        float inv_std = rsqrtf(var + eps);

        // Apply GroupNorm, Tanh, and HardSwish
        int input_idx = n * num_channels * height * width + c * height * width + h * width + w;
        float x = input[input_idx];
        x = (x - mean) * inv_std;
        x = gamma[c] * x + beta[c];
        x = tanhf(x);
        x = hard_swish(x);
        output[input_idx] = x;
    }
}

// LogSumExp kernel
__global__ void logsumexp_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_channels,
    int height,
    int width) {

    int total_elements = batch_size * height * width;
    CUDA_KERNEL_LOOP(idx, total_elements) {
        int n = idx / (height * width);
        int h = (idx / width) % height;
        int w = idx % width;

        float max_val = -INFINITY;
        for (int c = 0; c < num_channels; ++c) {
            int in_idx = n * num_channels * height * width + c * height * width + h * width + w;
            max_val = fmaxf(max_val, input[in_idx]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < num_channels; ++c) {
            int in_idx = n * num_channels * height * width + c * height * width + h * width + w;
            sum_exp += expf(input[in_idx] - max_val);
        }

        output[n * height * width + h * width + w] = max_val + logf(sum_exp);
    }
}

torch::Tensor fused_gn_tanh_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float eps) {
    
    auto stream = at::cuda::getCurrentCUDAStream();

    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = input.clone();

    const int threads_per_block = 256;
    const int total_elements = batch_size * num_channels * height * width;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    dim3 grid(std::min(65535, num_blocks));
    dim3 block(threads_per_block);

    fused_gn_tanh_hardswish_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, num_channels, height, width, groups, eps
    );

    return output;
}

torch::Tensor logsumexp_cuda(torch::Tensor input) {
    auto stream = at::cuda::getCurrentCUDAStream();

    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::zeros({batch_size, 1, height, width}, input.options());

    const int threads_per_block = 256;
    const int total_elements = batch_size * height * width;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    dim3 grid(std::min(65535, num_blocks));
    dim3 block(threads_per_block);

    logsumexp_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, num_channels, height, width
    );

    return output;
}
"""

fused_gn_act_cpp = """
torch::Tensor fused_gn_tanh_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float eps);

torch::Tensor logsumexp_cuda(torch::Tensor input);
"""

# Compile the inline CUDA kernels
fused_ops = load_inline(
    name="fused_gn_act",
    cpp_sources=fused_gn_act_cpp,
    cuda_sources=fused_gn_act_source,
    functions=["fused_gn_tanh_hardswish_cuda", "logsumexp_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels for GroupNorm + Tanh + HardSwish and LogSumExp.
    Convolution remains in PyTorch for correctness and compatibility.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.groups = groups
        self.eps = eps
        self.fused_gn_act = fused_ops.fused_gn_tanh_hardswish_cuda
        self.fused_logsumexp = fused_ops.logsumexp_cuda

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Fused GroupNorm + Tanh + HardSwish
        x_act = self.fused_gn_act(
            x_conv, self.group_norm.weight, self.group_norm.bias,
            self.group_norm.weight, self.group_norm.bias,
            self.groups, self.eps
        )
        # Residual Addition
        x_res = x_conv + x_act
        # LogSumExp
        x_logsumexp = self.fused_logsumexp(x_res)
        return x_logsumexp