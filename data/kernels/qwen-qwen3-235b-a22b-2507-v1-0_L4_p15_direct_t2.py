import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + BatchNorm3d + mean subtraction with high parallelism and coalesced memory access
fused_conv_bn_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cooperative_groups.h>

using namespace cooperative_groups;

// 5D tensor index: (n, c, d, h, w)
__device__ int get_index(int n, int c, int d, int h, int w, int N, int C, int D, int H, int W) {
    return ((n * C + c) * D + d) * H * W + h * W + w;
}

// Efficiently compute output spatial dimensions
__device__ void compute_output_dims(
    int D_in, int H_in, int W_in,
    int kernel_size, int stride, int padding,
    int &D_out, int &H_out, int &W_out) {
    D_out = (D_in - 1) * stride - 2 * padding + kernel_size;
    H_out = (H_in - 1) * stride - 2 * padding + kernel_size;
    W_out = (W_in - 1) * stride - 2 * padding + kernel_size;
}

// Fast reduction for computing mean across spatial dimensions
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

// Kernel 1: Fused ConvTranspose3d + BatchNorm3d
__global__ void fused_conv_transpose_bn_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int kernel_size, int stride, int padding,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * D_out * H_out * W_out;
    if (idx >= total_elements) return;

    int n = idx / (C_out * D_out * H_out * W_out);
    int c_out = (idx / (D_out * H_out * W_out)) % C_out;
    int d_out = (idx / (H_out * W_out)) % D_out;
    int h_out = (idx / W_out) % H_out;
    int w_out = idx % W_out;

    float conv_sum = 0.0f;

    // ConvTranspose3d computation with coalesced access
    for (int kd = 0; kd < kernel_size; ++kd) {
        int d_in = d_out * stride - padding + kd;
        if (d_in < 0 || d_in >= D_in) continue;

        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_in = h_out * stride - padding + kh;
            if (h_in < 0 || h_in >= H_in) continue;

            for (int kw = 0; kw < kernel_size; ++kw) {
                int w_in = w_out * stride - padding + kw;
                if (w_in < 0 || w_in >= W_in) continue;

                int weight_idx_base = (c_out * C_in * kernel_size + 0) * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw;
                int input_idx_base = get_index(n, 0, d_in, h_in, w_in, N, C_in, D_in, H_in, W_in);

                // Accumulate over input channels
                for (int c_in = 0; c_in < C_in; ++c_in) {
                    float input_val = input[input_idx_base + c_in];
                    float weight_val = weight[weight_idx_base + c_in * kernel_size * kernel_size * kernel_size];
                    conv_sum += input_val * weight_val;
                }
            }
        }
    }

    conv_sum += bias[c_out];

    // BatchNorm
    float bn_output = bn_weight[c_out] * (conv_sum - running_mean[c_out]) / sqrtf(running_var[c_out] + eps) + bn_bias[c_out];
    output[idx] = bn_output;
}

// Kernel 2: Subtract spatial mean using efficient per-(n,c) reduction
__global__ void subtract_spatial_mean_kernel(
    float* __restrict__ data,
    int N, int C_out, int D_out, int H_out, int W_out
) {
    int nc_id = blockIdx.x;
    int tid = threadIdx.x;
    int spatial_size = D_out * H_out * W_out;
    int total_threads = blockDim.x;

    int n = nc_id / C_out;
    int c = nc_id % C_out;

    float sum = 0.0f;

    // Grid-stride loop over spatial dimensions
    for (int idx = tid; idx < spatial_size; idx += total_threads) {
        int d = idx / (H_out * W_out);
        int h = (idx / W_out) % H_out;
        int w = idx % W_out;
        int data_idx = get_index(n, c, d, h, w, N, C_out, D_out, H_out, W_out);
        sum += data[data_idx];
    }

    // Block-wide reduction
    float mean_val = block_reduce_sum(sum) / spatial_size;

    // Only thread 0 writes the mean and broadcasts
    __shared__ float shared_mean;
    if (tid == 0) {
        shared_mean = mean_val;
    }
    __syncthreads();

    // Subtract mean from all spatial locations
    for (int idx = tid; idx < spatial_size; idx += total_threads) {
        int d = idx / (H_out * W_out);
        int h = (idx / W_out) % H_out;
        int w = idx % W_out;
        int data_idx = get_index(n, c, d, h, w, N, C_out, D_out, H_out, W_out);
        data[data_idx] -= shared_mean;
    }
}

// Host function
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

    const int block_size = 256;
    const int total_elements = N * C_out * D_out * H_out * W_out;
    const int num_blocks_conv = (total_elements + block_size - 1) / block_size;

    // Launch fused conv_transpose + batchnorm
    fused_conv_transpose_bn_kernel<<<num_blocks_conv, block_size>>>(
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
        1e-5f
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in conv+bn: %s\\n", cudaGetErrorString(err));
    }

    // Use one block per (n, c) for mean subtraction
    const int num_nc = N * C_out;
    subtract_spatial_mean_kernel<<<num_nc, block_size>>>(
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

fused_op = load_inline(
    name="fused_conv_bn_mean",
    cpp_sources=fused_conv_bn_mean_cpp_source,
    cuda_sources=fused_conv_bn_mean_source,
    functions=["fused_conv_transpose_bn_mean_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-use_fast_math"],
    extra_cuda_cflags=["-O3", "--use-fast-math", "--ptxas-options=-v"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with fused ConvTranspose3d + BatchNorm3d + mean subtraction in CUDA.
    Uses high parallelism, coalesced memory access, and efficient reductions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias_param = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))

        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)

        self.fused_op = fused_op.fused_conv_transpose_bn_mean_cuda

    def forward(self, x):
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