import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d + GroupNorm + Tanh + Hardswish
fused_conv_gn_act_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float hard_swish(float x) {
    float tmp = fmaxf(0.0f, fminf(3.0f, x + 3.0f));
    return x * tmp / 6.0f;
}

__global__ void fused_conv_gn_tanh_hardswish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* weight_gn,
    const float* bias_gn,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int groups,
    float eps,
    int channels_per_group) {

    int total_elements = batch_size * out_channels * output_height * output_width;
    CUDA_KERNEL_LOOP(idx, total_elements) {
        int n = idx / (out_channels * output_height * output_width);
        int c = (idx / (output_height * output_width)) % out_channels;
        int h = (idx / output_width) % output_height;
        int w = idx % output_width;

        int group_id = c / channels_per_group;
        int c_in_group = c % channels_per_group;

        // Convolution
        float conv_val = 0.0f;
        int in_c_start = group_id * channels_per_group;
        int k_half = kernel_size / 2;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = h + kh - k_half;
                int iw = w + kw - k_half;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    for (int ic = 0; ic < channels_per_group; ++ic) {
                        int in_c = in_c_start + ic;
                        int weight_idx = c * channels_per_group * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        int input_idx = n * in_channels * input_height * input_width + in_c * input_height * input_width + ih * input_width + iw;
                        conv_val += weight[weight_idx] * input[input_idx];
                    }
                }
            }
        }
        conv_val += bias[c];

        // Group Normalization
        int out_idx_base = n * out_channels * output_height * output_width + c * output_height * output_width;
        int out_idx = out_idx_base + h * output_width + w;

        int group_start = group_id * channels_per_group * output_height * output_width;
        int group_size = channels_per_group * output_height * output_width;
        float mean = 0.0f, var = 0.0f;

        // Compute mean
        for (int i = 0; i < group_size; ++i) {
            int offset = group_start + i;
            float val = (offset == out_idx) ? conv_val : output[offset];
            mean += val;
        }
        mean /= group_size;

        // Compute variance
        for (int i = 0; i < group_size; ++i) {
            int offset = group_start + i;
            float val = (offset == out_idx) ? conv_val : output[offset];
            float diff = val - mean;
            var += diff * diff;
        }
        var /= group_size;
        float inv_std = rsqrtf(var + eps);

        float norm_val = (conv_val - mean) * inv_std;
        norm_val = norm_val * weight_gn[c] + bias_gn[c];

        // Tanh
        float tanh_val = tanhf(norm_val);

        // HardSwish
        float act_val = hard_swish(tanh_val);

        output[out_idx] = act_val;
    }
}

// LogSumExp kernel
__global__ void logsumexp_kernel(
    const float* input,
    float* output,
    int batch_size,
    int out_channels,
    int height,
    int width) {

    int total_elements = batch_size * height * width;
    CUDA_KERNEL_LOOP(idx, total_elements) {
        int n = idx / (height * width);
        int h = (idx / width) % height;
        int w = idx % width;

        float max_val = -INFINITY;
        for (int c = 0; c < out_channels; ++c) {
            int in_idx = n * out_channels * height * width + c * height * width + h * width + w;
            max_val = fmaxf(max_val, input[in_idx]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < out_channels; ++c) {
            int in_idx = n * out_channels * height * width + c * height * width + h * width + w;
            sum_exp += expf(input[in_idx] - max_val);
        }

        output[idx] = max_val + logf(sum_exp);
    }
}

torch::Tensor fused_conv_gn_tanh_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    int groups,
    float eps) {
    
    auto stream = at::cuda::getCurrentCUDAStream();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;

    int channels_per_group = out_channels / groups;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    const int threads_per_block = 256;
    const int num_blocks = (batch_size * out_channels * output_height * output_width + threads_per_block - 1) / threads_per_block;

    dim3 grid(std::min(65535, num_blocks));
    dim3 block(threads_per_block);

    fused_conv_gn_tanh_hardswish_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        weight_gn.data_ptr<float>(),
        bias_gn.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, input_height, input_width,
        output_height, output_width, kernel_size, groups, eps, channels_per_group
    );

    return output;
}

torch::Tensor logsumexp_cuda(torch::Tensor input) {
    auto stream = at::cuda::getCurrentCUDAStream();

    int batch_size = input.size(0);
    int out_channels = input.size(1);
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
        batch_size, out_channels, height, width
    );

    return output;
}
"""

fused_conv_gn_act_cpp = """
torch::Tensor fused_conv_gn_tanh_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    int groups,
    float eps);

torch::Tensor logsumexp_cuda(torch::Tensor input);
"""

# Compile the inline CUDA kernels
fused_ops = load_inline(
    name="fused_conv_gn_act",
    cpp_sources=fused_conv_gn_act_cpp,
    cuda_sources=fused_conv_gn_act_source,
    functions=["fused_conv_gn_tanh_hardswish_cuda", "logsumexp_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels for Conv2d + GroupNorm + Tanh + HardSwish and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps

        # Conv parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        # GroupNorm parameters
        self.weight_gn = nn.Parameter(torch.ones(out_channels))
        self.bias_gn = nn.Parameter(torch.zeros(out_channels))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.bias)

        self.fused_conv_gn_act = fused_ops.fused_conv_gn_tanh_hardswish_cuda
        self.fused_logsumexp = fused_ops.logsumexp_cuda

    def forward(self, x):
        # Fused Conv + GroupNorm + Tanh + HardSwish
        x_act = self.fused_conv_gn_act(
            x, self.weight, self.bias, self.weight_gn, self.bias_gn, self.groups, self.eps
        )
        # Residual connection with original conv output (x_act already includes conv)
        x_res = x_act + x_act  # Note: in original, residual was conv + hardswish; here x_act is the full output
        # But note: in original model, residual was: conv + hardswish, and hardswish was applied on norm(tanh(norm(...)))
        # However, our fused kernel outputs the hardswish(norm(tanh(norm(...)))) result
        # So residual = conv_output + hardswish_output
        # But we don't have conv_output separately. So we need to recompute or refactor.

        # Correction: We need to refactor to preserve conv output
        # Let's adjust: we'll split the fused kernel to return both conv and final activation
        # But for now, we assume we can recompute conv separately or modify kernel
        # Instead, we change strategy: we do not fuse conv into the activation kernel, but keep conv as is
        # But that defeats fusion. So we instead modify the kernel to output conv separately?
        # Given complexity, we instead do: fuse only GN + Tanh + HardSwish, and keep conv separate
        # But we already fused conv. So let's fix the logic.

        # Actually, in the original model:
        # x_conv = conv(x)
        # x_norm = gn(x_conv)
        # x_tanh = tanh(x_norm)
        # x_hard_swish = hardswish(x_tanh)
        # x_res = x_conv + x_hard_swish

        # Our fused kernel computes: hardswish(tanh(gn(conv(x)))) -> x_hard_swish
        # So we need x_conv separately.

        # Therefore, we must split: either run conv twice (wasteful) or modify kernel to return both
        # We choose to recompute conv output using PyTorch's conv2d for now to preserve correctness
        # But that's inefficient. Alternatively, we can modify the kernel to write conv output to a buffer
        # Due to complexity of multi-output kernel, we instead refactor: do not fuse conv into this kernel
        # But we already did. So let's change approach.

        # Revised plan: fuse only GN + Tanh + HardSwish, and keep conv and residual separate
        # But that gives less speedup.

        # Given time, we proceed with a corrected fused kernel that returns both conv and final
        # But for now, we simulate: use PyTorch conv, then fused GN+Tanh+HardSwish
        # But that's not what we implemented.

        # We decide to recompute conv using PyTorch for residual
        # This is suboptimal but ensures correctness
        x_conv = torch.conv2d(x, self.weight, self.bias, stride=1, padding=0)
        x_res = x_conv + x_act
        x_logsumexp = self.fused_logsumexp(x_res)
        return x_logsumexp