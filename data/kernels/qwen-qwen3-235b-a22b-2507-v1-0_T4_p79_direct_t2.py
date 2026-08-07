import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 3D convolution, element-wise multiply, instance norm, clamp, multiply, and max
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function to calculate 3D index
__device__ int get_index(int d, int h, int w, int D, int H, int W) {
    return d * H * W + h * W + w;
}

__global__ void fused_conv_mul_norm_clamp_max_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* multiplier,
    float* output,
    int B, int C_in, int C_out, int D, int H, int W,
    int kD, int kH, int kW,
    int pad_d, int pad_h, int pad_w,
    float clamp_min, float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * D * H * W;
    
    if (idx >= total_elements) return;

    int b = idx / (D * H * W);
    int d = (idx / (H * W)) % D;
    int h = (idx / W) % H;
    int w = idx % W;

    int spatial_idx = d * H * W + h * W + w;

    // Shared memory for storing per-channel conv results for this spatial location
    extern __shared__ float shared_data[];
    float* conv_results = shared_data;

    // Each thread handles one output channel
    int c_out = threadIdx.x;
    float sum_val = 0.0f;

    // Perform convolution for this channel
    if (c_out < C_out) {
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kd = 0; kd < kD; ++kd) {
                for (int kh = 0; kh < kH; ++kh) {
                    for (int kw = 0; kw < kW; ++kw) {
                        int in_d = d + kd - pad_d;
                        int in_h = h + kh - pad_h;
                        int in_w = w + kw - pad_w;
                        if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                            int in_idx = b * C_in * D * H * W + c_in * D * H * W + in_d * H * W + in_h * W + in_w;
                            int weight_idx = c_out * C_in * kD * kH * kW + c_in * kD * kH * kW + kd * kH * kW + kh * kW + kw;
                            sum_val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        sum_val += bias[c_out];
        conv_results[c_out] = sum_val * multiplier[c_out];
    }

    __syncthreads();

    // Compute mean and variance for instance norm
    float mean = 0.0f, var = 0.0f;
    if (threadIdx.x == 0) {
        for (int c = 0; c < C_out; ++c) {
            mean += conv_results[c];
        }
        mean /= C_out;

        for (int c = 0; c < C_out; ++c) {
            float diff = conv_results[c] - mean;
            var += diff * diff;
        }
        var /= C_out;
    }

    __shared__ float shared_mean, shared_inv_std;
    if (threadIdx.x == 0) {
        float eps = 1e-5f;
        shared_inv_std = rsqrtf(var + eps);
        shared_mean = mean;
    }
    __syncthreads();

    // Apply instance norm, clamp, multiply again, and compute max in parallel
    float x = 0.0f;
    if (c_out < C_out) {
        x = (conv_results[c_out] - shared_mean) * shared_inv_std;
        x = fmaxf(clamp_min, fminf(clamp_max, x));
        x = x * multiplier[c_out];
    }

    // Parallel reduction for max across channels
    for (int stride = C_out / 2; stride > 0; stride >>= 1) {
        if (c_out < stride && c_out + stride < C_out) {
            float other = x;
            if (threadIdx.x + stride < C_out) {
                float temp = (conv_results[threadIdx.x + stride] - shared_mean) * shared_inv_std;
                temp = fmaxf(clamp_min, fminf(clamp_max, temp));
                temp = temp * multiplier[threadIdx.x + stride];
                other = temp;
            }
            x = fmaxf(x, other);
        }
        __syncthreads();
    }

    // Write result
    if (c_out == 0) {
        output[b * D * H * W + spatial_idx] = x;
    }
}

torch::Tensor fused_conv_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    float clamp_min,
    float clamp_max
) {
    const int B = input.size(0);
    const int C_in = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);

    const int C_out = weight.size(0);
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    const int pad_d = kD / 2;
    const int pad_h = kH / 2;
    const int pad_w = kW / 2;

    auto output = torch::zeros({B, D, H, W}, input.options());

    const int total_elements = B * D * H * W;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    const size_t shared_mem_size = C_out * sizeof(float);

    fused_conv_mul_norm_clamp_max_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, C_out, D, H, W,
        kD, kH, kW,
        pad_d, pad_h, pad_w,
        clamp_min, clamp_max
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    float clamp_min,
    float clamp_max
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_op"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model using fused CUDA kernel for:
    Conv3d -> mul -> InstanceNorm3d -> clamp -> mul -> max(dim=1)
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Initialize weights and bias like Conv3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

        # Bind the fused CUDA op
        self.fused_op = fused_ops.fused_conv_op

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias, self.multiplier, self.clamp_min, self.clamp_max)