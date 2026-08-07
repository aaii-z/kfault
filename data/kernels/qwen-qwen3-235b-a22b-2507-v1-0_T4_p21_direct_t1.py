import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d + Bias + Scale + Sigmoid + GroupNorm
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utility function to get 4D index: (n, c, h, w)
__device__ int idx_4d(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

__global__ void fused_conv_bias_scale_sigmoid_gn_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* scale,
    const float* gamma,
    const float* beta,
    float* output,
    int B, int C_in, int C_out, int H, int W, int H_out, int W_out, int K,
    int groups, int C_per_group, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C_out * H_out * W_out;
    
    if (idx >= total_elements) return;

    int w_idx = idx % W_out;
    int h_idx = (idx / W_out) % H_out;
    int c_idx = (idx / (W_out * H_out)) % C_out;
    int b_idx = idx / (W_out * H_out * C_out);

    int group_id = c_idx / C_per_group;
    int local_c_idx = c_idx % C_per_group;

    // Convolution computation
    float conv_val = 0.0f;
    int k_half = K / 2;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            for (int c_in = 0; c_in < C_in; c_in++) {
                int h_im = h_idx + i - k_half;
                int w_im = w_idx + j - k_half;
                if (h_im >= 0 && h_im < H && w_im >= 0 && w_im < W) {
                    int input_idx = idx_4d(b_idx, c_in, h_im, w_im, C_in, H, W);
                    int weight_idx = idx_4d(c_idx, c_in, i, j, C_in, K, K);
                    conv_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias and scale
    conv_val += bias[c_idx];
    conv_val *= scale[c_idx];
    float x = 1.0f / (1.0f + expf(-conv_val)); // sigmoid

    // Group Normalization: compute mean and variance for the group
    int group_start = group_id * C_per_group;
    float mean = 0.0f, var = 0.0f;
    int spatial_size = H_out * W_out;
    int group_size = C_per_group * spatial_size;

    // Compute mean within the group
    for (int c = 0; c < C_per_group; c++) {
        for (int hw = 0; hw < spatial_size; hw++) {
            int c_abs = group_start + c;
            int idx_inner = idx_4d(b_idx, c_abs, hw / W_out, hw % W_out, C_out, H_out, W_out);
            float val = 1.0f / (1.0f + expf(-(
                input[idx_4d(b_idx, c_abs / C_per_group * C_in + c_abs % C_per_group, (hw / W_out) + k_half, (hw % W_out) + k_half, C_in, H, W)] *
                weight[idx_4d(c_abs, c_abs / C_per_group * C_in + c_abs % C_per_group, (hw / W_out), (hw % W_out), C_in, K, K)] +
                bias[c_abs]
            ) * scale[c_abs]));
            mean += val;
        }
    }
    mean /= group_size;

    // Compute variance
    for (int c = 0; c < C_per_group; c++) {
        for (int hw = 0; hw < spatial_size; hw++) {
            int c_abs = group_start + c;
            int idx_inner = idx_4d(b_idx, c_abs, hw / W_out, hw % W_out, C_out, H_out, W_out);
            float val = 1.0f / (1.0f + expf(-(
                input[idx_4d(b_idx, c_abs / C_per_group * C_in + c_abs % C_per_group, (hw / W_out) + k_half, (hw % W_out) + k_half, C_in, H, W)] *
                weight[idx_4d(c_abs, c_abs / C_per_group * C_in + c_abs % C_per_group, (hw / W_out), (hw % W_out), C_in, K, K)] +
                bias[c_abs]
            ) * scale[c_abs]));
            float diff = val - mean;
            var += diff * diff;
        }
    }
    var /= group_size;

    float inv_std = rsqrtf(var + eps);
    float gn_val = (x - mean) * inv_std * gamma[group_id] + beta[group_id];

    output[idx] = gn_val;
}

torch::Tensor fused_conv_bias_scale_sigmoid_gn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(scale);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);

    const auto B = input.size(0);
    const auto C_in = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const auto C_out = weight.size(0);
    const auto K = weight.size(2);
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int C_per_group = C_out / groups;

    auto output = torch::empty({B, C_out, H_out, W_out}, input.options());

    const int total_elements = B * C_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_bias_scale_sigmoid_gn_kernel<<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, C_out, H, W, H_out, W_out, K,
        groups, C_per_group, eps
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_bias_scale_sigmoid_gn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float eps
);
"""

# Compile the inline CUDA module
fused_ops = load_inline(
    name="fused_conv_bias_scale_sigmoid_gn",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_bias_scale_sigmoid_gn_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv2d + Bias + Scale + Sigmoid + GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.bias_shape = bias_shape
        self.scale_shape = scale_shape

        # Conv weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        # GroupNorm parameters (gamma and beta)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.eps = self.group_norm.eps

        # Fused CUDA kernel
        self.fused_kernel = fused_ops.fused_conv_bias_scale_sigmoid_gn_cuda

    def forward(self, x):
        if self.training:
            x = self.conv(x)
            x = x + self.bias
            x = x * self.scale
            x = torch.sigmoid(x)
            x = self.group_norm(x)
            return x
        else:
            return self.fused_kernel(
                x,
                self.weight,
                self.bias.view(-1),
                self.scale.view(-1),
                self.group_norm.weight,
                self.group_norm.bias,
                self.num_groups,
                self.eps
            )

    def train(self, mode=True):
        was_training = self.training
        super().train(mode)
        if was_training and not mode:
            # No need to do anything special
            pass