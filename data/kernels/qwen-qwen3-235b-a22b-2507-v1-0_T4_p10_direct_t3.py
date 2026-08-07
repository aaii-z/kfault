import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused transposed convolution, max pooling, hardtanh, mean, and tanh
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__device__ float hardtanh(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, x));
}

__global__ void fused_conv_transpose_maxpool_hardtanh_mean_tanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    int pool_kernel_size, int pool_stride,
    float hardtanh_min, float hardtanh_max
) {
    int batch = blockIdx.x;
    int chan = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (batch >= batch_size || chan >= out_channels) return;

    // Shared memory for reduction
    extern __shared__ float sdata[];

    int pooled_out_h = (out_height + pool_stride - 1) / pool_stride;
    int pooled_out_w = (out_width + pool_stride - 1) / pool_stride;
    int total_pooled = pooled_out_h * pooled_out_w;

    float local_sum = 0.0f;
    int work_per_thread = (total_pooled + block_size - 1) / block_size;

    for (int idx = tid; idx < total_pooled; idx += block_size) {
        int ph = idx / pooled_out_w;
        int pw = idx % pooled_out_w;

        int h_start = ph * pool_stride;
        int w_start = pw * pool_stride;

        float window_max = -FLT_MAX;

        for (int ph_k = 0; ph_k < pool_kernel_size; ++ph_k) {
            for (int pw_k = 0; pw_k < pool_kernel_size; ++pw_k) {
                int h = h_start + ph_k;
                int w = w_start + pw_k;
                if (h >= out_height || w >= out_width) continue;

                float val = 0.0f;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = h * stride - padding + kh;
                            int iw = w * stride - padding + kw;
                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                int in_idx = batch * in_channels * in_height * in_width +
                                           ic * in_height * in_width + ih * in_width + iw;
                                int w_idx = chan * in_channels * kernel_size * kernel_size +
                                          ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                val += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                }
                val += bias[chan];
                val = hardtanh(val, hardtanh_min, hardtanh_max);
                window_max = fmaxf(window_max, val);
            }
        }
        local_sum += window_max;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction to sum all contributions
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float total_sum = sdata[0];
    float mean_val = total_sum / total_pooled;
    mean_val = tanhf(mean_val);

    int out_idx = batch * out_channels + chan;
    output[out_idx] = mean_val;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    int pool_kernel_size, int pool_stride,
    float hardtanh_min, float hardtanh_max
);
"""

# Compile the fused CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for:
    ConvTranspose2d -> MaxPool2d -> Hardtanh -> Mean -> Tanh
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        # Define transposed convolution parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

        # Store fused CUDA op
        self.fused_op = fused_op

    def forward(self, x):
        batch_size, _, input_height, input_width = x.shape

        # Output size of ConvTranspose2d
        out_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size

        # Use fixed block size
        int block_size = 256;

        // Calculate grid and shared memory
        size_t shared_mem = block_size * sizeof(float);

        // Launch fused kernel
        output = self.fused_op.fused_op_cuda(
            x, self.weight, self.bias,
            self.in_channels, input_height, input_width,
            self.out_channels, out_height, out_width,
            self.kernel_size, self.stride, self.padding,
            self.maxpool_kernel_size, self.maxpool_stride,
            self.hardtanh_min, self.hardtanh_max
        )

        // Reshape to (batch, out_channels, 1, 1)
        output = output.view(batch_size, self.out_channels, 1, 1)
        return output