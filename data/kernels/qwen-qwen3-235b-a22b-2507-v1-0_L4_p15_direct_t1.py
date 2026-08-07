import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + BatchNorm3d + mean subtraction
fused_conv_bn_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function to calculate 5D tensor index: (n, c, d, h, w)
__device__ int get_index(int n, int c, int d, int h, int w, 
                         int N, int C, int D, int H, int W) {
    return ((n * C + c) * D + d) * H * W + h * W + w;
}

__global__ void fused_conv_transpose_bn_mean_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* bn_weight,
    const float* bn_bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int kernel_size, int stride, int padding,
    float eps,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Reverse index mapping: from output index to input
    int n = idx / (C_out * D_out * H_out * W_out);
    int c_out = (idx / (D_out * H_out * W_out)) % C_out;
    int d_out = (idx / (H_out * W_out)) % D_out;
    int h_out = (idx / W_out) % H_out;
    int w_out = idx % W_out;

    float conv_sum = 0.0f;

    // ConvTranspose3d: iterate over kernel and input
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d_in = d_out * stride - padding + kd;
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;

                if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    for (int c_in = 0; c_in < C_in; ++c_in) {
                        int input_idx = get_index(n, c_in, d_in, h_in, w_in, N, C_in, D_in, H_in, W_in);
                        int weight_idx = ((c_out * C_in + c_in) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                        conv_sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias
    conv_sum += bias[c_out];

    // BatchNorm: y = bn_weight * (x - running_mean) / sqrt(running_var + eps) + bn_bias
    float bn_output = bn_weight[c_out] * (conv_sum - running_mean[c_out]) / sqrtf(running_var[c_out] + eps) + bn_bias[c_out];

    // Store BN output
    output[idx] = bn_output;
}

// Second kernel to subtract spatial mean
__global__ void subtract_spatial_mean_kernel(
    float* data,
    int N, int C_out, int D_out, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * D_out * H_out * W_out) return;

    int n = idx / (C_out * D_out * H_out * W_out);
    int c = (idx / (D_out * H_out * W_out)) % C_out;

    // Compute mean over D, H, W for each (n, c)
    float sum = 0.0f;
    for (int d = 0; d < D_out; ++d) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                int spatial_idx = get_index(n, c, d, h, w, N, C_out, D_out, H_out, W_out);
                sum += data[spatial_idx];
            }
        }
    }
    float mean_val = sum / (D_out * H_out * W_out);

    // Subtract mean from each spatial location
    for (int d = 0; d < D_out; ++d) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                int spatial_idx = get_index(n, c, d, h, w, N, C_out, D_out, H_out, W_out);
                data[spatial_idx] -= mean_val;
            }
        }
    }
}

// Host function to perform the full fused operation
torch::Tensor fused_conv_transpose_bn_mean_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int kernel_size,
    int stride,
    int padding,
    int D_out, int H_out, int W_out
) {
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);
    auto C_out = weight.size(0);

    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    int total_elements = N * C_out * D_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Launch fused conv_transpose + bn
    fused_conv_transpose_bn_mean_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        kernel_size, stride, padding,
        1e-5f,  // eps
        total_elements
    );

    // Synchronize before next kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    // Subtract spatial mean in-place
    subtract_spatial_mean_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        N, C_out, D_out, H_out, W_out
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in mean subtraction: %s\\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return output;
}
"""

fused_conv_bn_mean_cpp_source = """
torch::Tensor fused_conv_transpose_bn_mean_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int kernel_size,
    int stride,
    int padding,
    int D_out, int H_out, int W_out);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_bn_mean",
    cpp_sources=fused_conv_bn_mean_cpp_source,
    cuda_sources=fused_conv_bn_mean_source,
    functions=["fused_conv_transpose_bn_mean_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with fused ConvTranspose3d + BatchNorm3d + mean subtraction in CUDA.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Define parameters manually to pass to CUDA kernel
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias_param = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)

        self.fused_op = fused_op.fused_conv_transpose_bn_mean_cuda

    def forward(self, x):
        # Calculate output dimensions manually
        D_out = (x.size(2) - 1) * self.stride - 2 * self.padding + self.kernel_size
        H_out = (x.size(3) - 1) * self.stride - 2 * self.padding + self.kernel_size
        W_out = (x.size(4) - 1) * self.stride - 2 * self.padding + self.kernel_size

        bias_tensor = self.bias_param if self.bias_param is not None else torch.zeros(self.out_channels, device=x.device)

        return self.fused_op(
            x,
            self.weight,
            bias_tensor,
            self.bn_weight,
            self.bn_bias,
            self.running_mean,
            self.running_var,
            self.kernel_size,
            self.stride,
            self.padding,
            D_out, H_out, W_out
        )