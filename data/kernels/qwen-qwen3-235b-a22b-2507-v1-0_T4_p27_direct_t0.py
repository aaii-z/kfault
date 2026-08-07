import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv3D + HardSwish + GroupNorm + Mean Pooling
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// HardSwish activation: x * relu6(x + 3) / 6
__device__ float hardswish(float x) {
    float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6 / 6.0f;
}

// Kernel for fused Conv3D -> HardSwish -> GroupNorm -> Mean Pooling
// We perform all operations in one fused kernel to minimize memory bandwidth and kernel launches.
__global__ void fused_conv_hardswish_groupnorm_mean_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ weight_gn,
    const float* __restrict__ bias_gn,
    float* __restrict__ output,
    int B, int C_out, int C_in, int D, int H, int W, int D_k, int H_k, int W_k,
    int num_groups, float eps
) {
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;

    if (batch_idx >= B || out_ch >= C_out) return;

    int group_id = out_ch / (C_out / num_groups);
    int local_ch = out_ch % (C_out / num_groups);

    // Step 1: Conv3D + HardSwish
    float conv_val = 0.0f;
    int weight_offset = out_ch * C_in * D_k * H_k * W_k;

    for (int c = 0; c < C_in; c++) {
        for (int d = 0; d < D_k; d++) {
            for (int h = 0; h < H_k; h++) {
                for (int w = 0; w < W_k; w++) {
                    int input_idx = batch_idx * C_in * D * H * W +
                                    c * D * H * W +
                                    d * H * W + h * W + w;
                    int weight_idx = weight_offset + c * D_k * H_k * W_k + d * H_k * W_k + h * W_k + w;
                    conv_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    if (bias) {
        conv_val += bias[out_ch];
    }
    float activated = hardswish(conv_val);

    // Atomically add to mean accumulator for this (batch, out_ch)
    atomicAdd(&output[batch_idx * C_out + out_ch], activated);

    // Synchronize all threads before GroupNorm (we need full reduction first)
    __syncthreads();

    // Only one thread per (batch, out_ch) should proceed to normalization
    if (threadIdx.x == 0) {
        float* mean_ptr = &output[batch_idx * C_out];
        
        // Compute mean over spatial dimensions (already accumulated, now divide)
        // But note: we haven't divided yet. We accumulated C_out * D * H * W values?
        // Actually, we are doing pointwise conv with kernel_size, so we need to compute full output spatial size?
        // Wait: our current approach is flawed because we're not computing full feature map.

        // Let's change strategy: we cannot fuse full Conv3D with spatial reduction unless we compute full output.
        // So we must compute full output first. We need to restructure.
    }
}
"""

# Revised plan: We cannot easily fuse full Conv3D with spatial mean without computing all spatial outputs.
# Instead, we can fuse HardSwish + GroupNorm + Mean in a separate kernel, but Conv3D is best left to cuDNN.
# However, we can fuse HardSwish + GroupNorm + Mean over spatial dims in a custom kernel applied after Conv3D.

# So we do:
# 1. Use standard Conv3D (cuDNN is highly optimized)
# 2. Fused kernel for: HardSwish -> GroupNorm -> Mean(dim=[2,3,4])

fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) \
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
    int B, int C, int D, int H, int W, int num_groups, float eps) {

    int batch = blockIdx.y;
    int channel = blockIdx.x;
    if (batch >= B || channel >= C) return;

    int spatial_size = D * H * W;
    int total_elements = B * C * spatial_size;
    int gid = batch * C * spatial_size + channel * spatial_size;

    // Compute mean and variance for GroupNorm over spatial dims for this group
    int group_size = C / num_groups;
    int group_id = channel / group_size;
    int local_channel = channel % group_size;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = 0; i < spatial_size; i++) {
        float val = input[gid + i];
        float act = hardswish(val);
        sum += act;
        sum_sq += act * act;
    }

    float mean = sum / spatial_size;
    float var = sum_sq / spatial_size - mean * mean;

    // Normalize
    float gamma = weight_gn[group_id * group_size + local_channel];
    float beta = bias_gn[group_id * group_size + local_channel];
    float inv_std = rsqrtf(var + eps);

    float norm_val = 0.0f;
    for (int i = 0; i < spatial_size; i++) {
        float val = input[gid + i];
        float act = hardswish(val);
        float x_norm = (act - mean) * inv_std;
        norm_val += gamma * x_norm + beta;
    }
    norm_val /= spatial_size;  // Average over spatial dimensions

    output[batch * C + channel] = norm_val;
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

    auto output = torch::empty({B, C}, input.options());

    const int block_size = 256;
    dim3 grid(C, B);  // each channel and batch gets a block
    dim3 block(block_size);

    hardswish_groupnorm_mean_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight_gn.data_ptr<float>(),
        bias_gn.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W, num_groups, eps
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