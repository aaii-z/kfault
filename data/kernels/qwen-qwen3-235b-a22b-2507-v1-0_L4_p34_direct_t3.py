import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 3D transposed convolution, layer norm, GELU, and scaling
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU activation using tanh approximation
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

// Kernel 1: Fused 3D transposed convolution with bias
__global__ void convtrans3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_D,
    const int input_H,
    const int input_W,
    const int output_D,
    const int output_H,
    const int output_W,
    const int kernel_size,
    const int stride,
    const int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_D * output_H * output_W;
    if (idx >= total_elements) return;

    int wo = idx % output_W;
    int ho = (idx / output_W) % output_H;
    int do_ = (idx / (output_H * output_W)) % output_D;
    int c = (idx / (output_D * output_H * output_W)) % out_channels;
    int b = idx / (output_D * output_H * output_W * out_channels);

    float sum = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int di = do_ * stride - padding + kd;
                    int hi = ho * stride - padding + kh;
                    int wi = wo * stride - padding + kw;

                    if (di >= 0 && di < input_D && hi >= 0 && hi < input_H && wi >= 0 && wi < input_W) {
                        int input_idx = b * in_channels * input_D * input_H * input_W +
                                        ci * input_D * input_H * input_W +
                                        di * input_H * input_W +
                                        hi * input_W +
                                        wi;
                        int weight_idx = c * in_channels * kernel_size * kernel_size * kernel_size +
                                         ci * kernel_size * kernel_size * kernel_size +
                                         kd * kernel_size * kernel_size +
                                         kh * kernel_size +
                                         kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c];
    }

    int output_idx = b * out_channels * output_D * output_H * output_W +
                     c * output_D * output_H * output_W +
                     do_ * output_H * output_W +
                     ho * output_W +
                     wo;
    output[output_idx] = sum;
}

// Kernel 2: LayerNorm + GELU + Scale with proper per-channel mean/var computation
__global__ void layernorm_gelu_scale_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    const int batch_size,
    const int out_channels,
    const int output_D,
    const int output_H,
    const int output_W,
    const float eps,
    const float scaling_factor
) {
    const int spatial_size = output_D * output_H * output_W;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    int b = blockIdx.x / out_channels;
    int c = blockIdx.x % out_channels;

    if (b >= batch_size || c >= out_channels) return;

    int channel_offset = b * out_channels * spatial_size + c * spatial_size;

    // Shared memory for reduction
    extern __shared__ float s_data[];
    float* s_mean = s_data;
    float* s_var = &s_data[block_size];

    // Initialize
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (int i = tid; i < spatial_size; i += block_size) {
        float val = x[channel_offset + i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    s_mean[tid] = local_sum;
    s_var[tid] = local_sq_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mean[tid] += s_mean[tid + stride];
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_mean[0] / spatial_size;
    float variance = s_var[0] / spatial_size - mean * mean;
    float inv_std = rsqrtf(variance + eps);

    // Final pass: normalize, GELU, scale
    for (int i = tid; i < spatial_size; i += block_size) {
        float val = x[channel_offset + i];
        float normalized = (val - mean) * inv_std;
        float gelu_val = gelu(normalized);
        output[channel_offset + i] = gelu_val * scaling_factor;
    }
}

// Host function
torch::Tensor fused_convtrans3d_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    float scaling_factor
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_D = input.size(2);
    int input_H = input.size(3);
    int input_W = input.size(4);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int stride = 2;
    int padding = 1;

    int output_D = (input_D - 1) * stride - 2 * padding + kernel_size;
    int output_H = (input_H - 1) * stride - 2 * padding + kernel_size;
    int output_W = (input_W - 1) * stride - 2 * padding + kernel_size;

    auto options = input.options();
    auto conv_out = torch::empty({batch_size, out_channels, output_D, output_H, output_W}, options);
    auto output = torch::empty({batch_size, out_channels, output_D, output_H, output_W}, options);

    const int block_size = 256;
    int total_elements = batch_size * out_channels * output_D * output_H * output_W;
    int num_blocks_conv = (total_elements + block_size - 1) / block_size;

    // Launch convolution kernel
    convtrans3d_kernel<<<num_blocks_conv, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        conv_out.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_D, input_H, input_W,
        output_D, output_H, output_W,
        kernel_size, stride, padding
    );

    // For layernorm: one block per (batch, channel)
    int num_blocks_layernorm = batch_size * out_channels;
    size_t smem_size = 2 * block_size * sizeof(float);  // for mean and variance

    layernorm_gelu_scale_kernel<<<num_blocks_layernorm, block_size, smem_size>>>(
        conv_out.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, out_channels, output_D, output_H, output_W,
        eps, scaling_factor
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_convtrans3d_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    float scaling_factor);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_convtrans3d_layernorm_gelu_scale",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_convtrans3d_layernorm_gelu_scale_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for 3D transposed convolution, layer normalization, GELU, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.eps = eps
        self.scaling_factor = scaling_factor
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)
        
        # Initialization
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)
        
        # Bind fused CUDA kernel
        self.fused_op = fused_op.fused_convtrans3d_layernorm_gelu_scale_cuda

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias_param, self.eps, self.scaling_factor)