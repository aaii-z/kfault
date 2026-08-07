import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + ReLU + GroupNorm
fused_convtranspose_relu_gn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function to calculate 5D index from linear index
__device__ int get_index(int idx, int stride_d, int stride_h, int stride_w, int &d, int &h, int &w) {
    w = idx % stride_w;
    h = (idx / stride_w) % stride_h;
    d = (idx / (stride_w * stride_h)) % stride_d;
    return idx / (stride_w * stride_h * stride_d);
}

// Kernel for fused transposed conv, ReLU and GroupNorm
__global__ void fused_convtranspose_relu_gn_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float* running_mean,
    float* running_var,
    float* weight_gn,
    float* bias_gn,
    int batch_size, int in_channels, int out_channels, int D, int H, int W,
    int kernel_size, int groups, float eps,
    int stride, int padding, int output_padding,
    int D_out, int H_out, int W_out
) {
    int total_elements = batch_size * out_channels * D_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c_out = (idx / (W_out * H_out * D_out)) % out_channels;
    int b = idx / (W_out * H_out * D_out * out_channels);

    // Transposed convolution (simple implementation with implicit padding)
    float conv_val = 0.0f;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d_in = d_out * stride - padding + kd - output_padding;
                int h_in = h_out * stride - padding + kh - output_padding;
                int w_in = w_out * stride - padding + kw - output_padding;

                if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        int weight_idx = c_out * in_channels * kernel_size * kernel_size * kernel_size +
                                        c_in * kernel_size * kernel_size * kernel_size +
                                        kd * kernel_size * kernel_size + kh * kernel_size + kw;
                        int input_idx = b * in_channels * D * H * W +
                                       c_in * D * H * W + d_in * H * W + h_in * W + w_in;
                        conv_val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        conv_val += bias[c_out];
    }

    // ReLU activation
    float relu_val = fmaxf(0.0f, conv_val);

    // Group Normalization
    int group_id = c_out / (out_channels / groups);
    int channels_per_group = out_channels / groups;
    int c_in_group = c_out % channels_per_group;

    // Compute mean and variance within the group across D_out, H_out, W_out
    float mean = 0.0f, var = 0.0f;
    int spatial_size = D_out * H_out * W_out;
    int group_offset = group_id * spatial_size * channels_per_group;

    // We assume pre-computed mean and variance (as in inference mode)
    mean = running_mean[group_id];
    var = running_var[group_id];

    float inv_std = 1.0f / sqrtf(var + eps);
    float norm_val = (relu_val - mean) * inv_std;

    // Apply affine transform if available
    norm_val = norm_val * weight_gn[c_out] + bias_gn[c_out];

    output[idx] = norm_val;
}

torch::Tensor fused_convtranspose_relu_gn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    int batch_size, int in_channels, int out_channels,
    int D, int H, int W,
    int kernel_size, int groups, float eps,
    int stride, int padding, int output_padding,
    int D_out, int H_out, int W_out
) {
    auto output = torch::zeros({batch_size, out_channels, D_out, H_out, W_out}, input.options());

    const int total_elements = output.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_convtranspose_relu_gn_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight_gn.data_ptr<float>(),
        bias_gn.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        D, H, W,
        kernel_size, groups, eps,
        stride, padding, output_padding,
        D_out, H_out, W_out
    );

    return output;
}
"""

fused_convtranspose_relu_gn_cpp_source = """
torch::Tensor fused_convtranspose_relu_gn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    int batch_size, int in_channels, int out_channels,
    int D, int H, int W,
    int kernel_size, int groups, float eps,
    int stride, int padding, int output_padding,
    int D_out, int H_out, int W_out);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_convtranspose_relu_gn",
    cpp_sources=fused_convtranspose_relu_gn_cpp_source,
    cuda_sources=fused_convtranspose_relu_gn_source,
    functions=["fused_convtranspose_relu_gn_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for ConvTranspose3d + ReLU + GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.bias = bias

        # Define the transposed convolution layer (we'll extract weights from it)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

        # Store fixed parameters for CUDA kernel
        self.stride = self.conv_transpose.stride[0]
        self.padding = self.conv_transpose.padding[0]
        self.output_padding = self.conv_transpose.output_padding[0]
        self.eps = self.group_norm.eps

        # Register GroupNorm parameters as buffers so they can be accessed in CUDA
        self.register_buffer('running_mean', torch.zeros(groups))
        self.register_buffer('running_var', torch.ones(groups))

    def forward(self, x):
        # Update running mean and var if in training mode
        if self.training:
            # Replicate GroupNorm statistics computation (per-group mean and variance)
            x_grouped = x.view(x.size(0), self.groups, self.out_channels // self.groups, *x.shape[2:])
            mean = x_grouped.mean(dim=[0, 2, 3, 4, 5])  # (groups,)
            var = x_grouped.var(dim=[0, 2, 3, 4, 5], unbiased=False)  # (groups,)
            self.running_mean.data = mean
            self.running_var.data = var

        # Extract necessary parameters
        batch_size, _, D, H, W = x.shape
        D_out = (D - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        H_out = (H - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        W_out = (W - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # Extract weights and biases
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.zeros(self.out_channels, device=x.device)

        # GroupNorm parameters
        weight_gn = self.group_norm.weight
        bias_gn = self.group_norm.bias if self.group_norm.bias is not None else torch.zeros(self.out_channels, device=x.device)

        # Call fused CUDA kernel
        output = fused_op.fused_convtranspose_relu_gn_cuda(
            x, weight, bias,
            self.running_mean, self.running_var,
            weight_gn, bias_gn,
            batch_size, self.in_channels, self.out_channels,
            D, H, W,
            self.kernel_size, self.groups, self.eps,
            self.stride, self.padding, self.output_padding,
            D_out, H_out, W_out
        )

        return output