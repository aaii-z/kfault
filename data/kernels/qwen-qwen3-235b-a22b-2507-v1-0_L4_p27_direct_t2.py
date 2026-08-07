import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused HardSwish + GroupNorm + Mean over spatial dimensions
fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) \\
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float hardswish(float x) {
    float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6 / 6.0f;
}

__global__ void hardswish_groupnorm_mean_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight_gn,
    const float* __restrict__ bias_gn,
    float* __restrict__ output,
    int B, int C, int D, int H, int W, int group_size, float eps) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int num_channels = C;
    int spatial_size = D * H * W;
    int elements_per_batch = C * spatial_size;

    // Each thread processes multiple (batch, channel) pairs if necessary
    for (int i = idx; i < B * num_channels; i += total_threads) {
        int batch = i / num_channels;
        int channel = i % num_channels;

        int group_id = channel / group_size;
        int local_channel = channel % group_size;

        int offset = batch * elements_per_batch + channel * spatial_size;

        // Step 1: Apply HardSwish and compute sum and sum of squares for GroupNorm
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int sp = 0; sp < spatial_size; sp++) {
            float x = input[offset + sp];
            float act = hardswish(x);
            sum += act;
            sum_sq += act * act;
        }

        // Step 2: Compute mean and variance for GroupNorm
        float mean = sum / spatial_size;
        float var = sum_sq / spatial_size - mean * mean;
        var = fmaxf(var, 0.0f);  // Ensure non-negative variance
        float inv_std = rsqrtf(var + eps);

        // Step 3: Apply GroupNorm: (act - mean) * inv_std * gamma + beta
        float gamma = weight_gn[group_id * group_size + local_channel];
        float beta = bias_gn[group_id * group_size + local_channel];

        float norm_sum = 0.0f;
        for (int sp = 0; sp < spatial_size; sp++) {
            float x = input[offset + sp];
            float act = hardswish(x);
            float x_norm = (act - mean) * inv_std;
            float out_val = gamma * x_norm + beta;
            norm_sum += out_val;
        }

        // Final mean over spatial dimensions
        output[batch * num_channels + channel] = norm_sum / spatial_size;
    }
}

torch::Tensor fused_hardswish_groupnorm_mean_cuda(
    torch::Tensor input,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    int num_groups,
    float eps) {
    
    auto B = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    TORCH_CHECK(C % num_groups == 0, "Number of channels must be divisible by num_groups");

    int group_size = C / num_groups;

    auto output = torch::empty({B, C}, input.options());

    const int block_size = 256;
    int num_blocks = (B * C + block_size - 1) / block_size;
    num_blocks = min(65535, num_blocks);  // Limit grid size

    dim3 grid(num_blocks);
    dim3 block(block_size);

    hardswish_groupnorm_mean_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight_gn.data_ptr<float>(),
        bias_gn.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W, group_size, eps
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}
"""

fused_post_conv_cpp = """
torch::Tensor fused_hardswish_groupnorm_mean_cuda(
    torch::Tensor input,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    int num_groups,
    float eps);
"""

# Compile the fused kernel
fused_ops = load_inline(
    name="fused_hardswish_groupnorm_mean",
    cpp_sources=[fused_post_conv_cpp],
    cuda_sources=[fused_post_conv_source],
    functions=["fused_hardswish_groupnorm_mean_cuda"],
    verbose=False,
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused HardSwish + GroupNorm + Mean in CUDA.
    Conv3D is kept as-is (cuDNN optimized), rest fused.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.eps = self.group_norm.eps
        self.fused_op = fused_ops.fused_hardswish_groupnorm_mean_cuda

    def forward(self, x):
        x = self.conv(x)  # Use cuDNN optimized Conv3D
        # Fused HardSwish + GroupNorm + Mean over spatial dims
        x = self.fused_op(x, self.group_norm.weight, self.group_norm.bias, self.num_groups, self.eps)
        return x