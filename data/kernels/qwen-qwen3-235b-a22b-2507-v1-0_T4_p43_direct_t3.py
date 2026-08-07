import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv3d + MaxPool3d + LogSumExp + ReLU with high parallelism
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Helper function to calculate 4D index: (batch, channel, depth, height, width)
__device__ int get_index(int b, int c, int d, int h, int w, 
                         int batch_size, int channels, int depth, int height, int width) {
    return b * (channels * depth * height * width) + 
           c * (depth * height * width) + 
           d * (height * width) + 
           h * width + 
           w;
}

__global__ void fused_conv_maxpool_logsumexp_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    // Input dimensions
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions after conv and maxpool
    int out_channels, int out_depth, int out_height, int out_width,
    // Conv parameters
    int kernel_size, int conv_stride, int padding
) {
    int batch = blockIdx.x;
    int od = blockIdx.y;
    int oh = blockIdx.z / out_width;
    int ow = blockIdx.z % out_width;

    if (batch >= batch_size || od >= out_depth || oh >= out_height || ow >= out_width) return;

    // Each output voxel after maxpool corresponds to a 2x2x2 region in the conv output
    int base_d = od * 2;
    int base_h = oh * 2;
    int base_w = ow * 2;

    // Shared memory for logsumexp reduction across channels
    extern __shared__ float shared_mem[];
    float* thread_vals = shared_mem;
    int tid = threadIdx.x;

    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    // Process output channels in chunks to allow reduction in shared memory
    int chunk_size = (out_channels + gridDim.y * gridDim.z - 1) / (gridDim.y * gridDim.z);
    int start_oc = (blockIdx.y * gridDim.z + blockIdx.z) * chunk_size;
    int end_oc = min(start_oc + chunk_size, out_channels);

    // Each thread block handles a spatial location; threads loop over output channels
    for (int oc = start_oc + tid; oc < end_oc; oc += blockDim.x) {
        float best = -FLT_MAX;

        // Max pooling over 2x2x2 region in conv output for this output channel
        for (int kd_pool = 0; kd_pool < 2; ++kd_pool) {
            for (int kh_pool = 0; kh_pool < 2; ++kh_pool) {
                for (int kw_pool = 0; kw_pool < 2; ++kw_pool) {
                    int conv_d = base_d + kd_pool;
                    int conv_h = base_h + kh_pool;
                    int conv_w = base_w + kw_pool;

                    // Perform convolution at (conv_d, conv_h, conv_w)
                    float conv_val = bias[oc];
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kd = 0; kd < kernel_size; ++kd) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int id = conv_d * conv_stride + kd - padding;
                                    int ih = conv_h * conv_stride + kh - padding;
                                    int iw = conv_w * conv_stride + kw - padding;

                                    if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                        int in_idx = get_index(batch, ic, id, ih, iw,
                                                               batch_size, in_channels, in_depth, in_height, in_width);
                                        int w_idx = oc * (in_channels * kernel_size * kernel_size * kernel_size) +
                                                   ic * (kernel_size * kernel_size * kernel_size) +
                                                   kd * (kernel_size * kernel_size) +
                                                   kh * kernel_size + kw;
                                        conv_val += input[in_idx] * weight[w_idx];
                                    }
                                }
                            }
                        }
                    }

                    if (conv_val > best) best = conv_val;
                }
            }
        }

        // Store max-pooled value for this channel
        thread_vals[tid] = best;
        __syncthreads();

        // Perform block-wide reduction to compute max and sum of exps
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = thread_vals[tid + s];
                if (other > thread_vals[tid]) {
                    thread_vals[tid] = other;
                }
            }
            __syncthreads();
        }

        max_val = thread_vals[0];
        __syncthreads();

        sum_exp = 0.0f;
        for (int oc2 = start_oc + tid; oc2 < end_oc; oc2 += blockDim.x) {
            float val = thread_vals[tid];
            sum_exp += expf(val - max_val);
            __syncthreads();
        }

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sum_exp += shared_mem[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float lse = max_val + logf(sum_exp);
            float relu_out = fmaxf(0.0f, lse);
            int out_idx = get_index(batch, 0, od, oh, ow,
                                    batch_size, 1, out_depth, out_height, out_width);
            output[out_idx] = relu_out;
        }
    }
}

torch::Tensor fused_conv_mp_lse_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int out_depth, int out_height, int out_width,
    int kernel_size, int conv_stride, int padding) {
    
    auto output = torch::empty({batch_size, 1, out_depth, out_height, out_width}, 
                               input.options());

    const int block_size = 256;
    dim3 grid(batch_size, out_depth, out_height * out_width);
    size_t shared_mem_size = block_size * sizeof(float);

    fused_conv_maxpool_logsumexp_relu_kernel<<<grid, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, out_depth, out_height, out_width,
        kernel_size, conv_stride, padding
    );

    return output;
}
"""

cpp_source = """
torch::Tensor fused_conv_mp_lse_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int out_depth, int out_height, int out_width,
    int kernel_size, int conv_stride, int padding);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_conv_mp_lse_relu",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_mp_lse_relu"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Keep the original conv layer to use its parameters
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, x):
        batch_size, _, in_depth, in_height, in_width = x.shape

        # Compute output dimensions after conv and maxpool
        conv_out_depth = (in_depth + 2 * self.padding - self.kernel_size) // self.stride + 1
        conv_out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        conv_out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        out_depth = conv_out_depth // 2
        out_height = conv_out_height // 2
        out_width = conv_out_width // 2

        output = fused_op.fused_conv_mp_lse_relu(
            x, self.conv.weight, self.conv.bias,
            batch_size, self.in_channels, in_depth, in_height, in_width,
            self.out_channels, out_depth, out_height, out_width,
            self.kernel_size, self.stride, self.padding
        )

        return output