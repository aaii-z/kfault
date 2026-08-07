import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 3D convolution, Group Normalization, and mean computation
fused_conv_gn_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_THREADS 1024

// Helper function to calculate linear index for 5D tensor (batch, channel, depth, height, width)
__device__ int get_index(int b, int c, int d, int h, int w, 
                         int B, int C, int D, int H, int W) {
    return b * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
}

__global__ void fused_conv_gn_mean_forward(
    const float* input,
    const float* weight,
    const float* bias,
    const float* weight_gn,
    const float* bias_gn,
    float* output,
    // Input tensor shape
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output channels and kernel size
    int out_channels, int kernel_size,
    // Output spatial dimensions
    int out_depth, int out_height, int out_width,
    // Group norm params
    int num_groups,
    // Conv parameters
    int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;

    // Decode index to 5D coordinates
    int temp = idx;
    int w = temp % out_width; temp /= out_width;
    int h = temp % out_height; temp /= out_height;
    int d = temp % out_depth; temp /= out_depth;
    int c = temp % out_channels; temp /= out_channels;
    int b = temp;

    // Perform convolution
    float conv_val = 0.0f;
    int kernel_radius = kernel_size / 2;
    
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_d = d * stride + kd - padding;
                int in_h = h * stride + kh - padding;
                int in_w = w * stride + kw - padding;
                
                if (in_d >= 0 && in_d < in_depth && 
                    in_h >= 0 && in_h < in_height && 
                    in_w >= 0 && in_w < in_width) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        int in_idx = get_index(b, ic, in_d, in_h, in_w, 
                                               batch_size, in_channels, in_depth, in_height, in_width);
                        int weight_idx = c * (in_channels * kernel_size * kernel_size * kernel_size) + 
                                         ic * (kernel_size * kernel_size * kernel_size) + 
                                         kd * (kernel_size * kernel_size) + kh * kernel_size + kw;
                        conv_val += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    conv_val += bias[c];

    // Store conv result in shared memory equivalent via global temp array (we'll use the same buffer)
    // Now do group norm: first compute mean and variance over group
    int group_size = out_channels / num_groups;
    int group_id = c / group_size;
    int group_start = group_id * group_size;
    int group_end = group_start + group_size;

    // We need to compute mean and variance across the entire group for this spatial location
    float mean = 0.0f;
    for (int gc = group_start; gc < group_end; ++gc) {
        int conv_idx = get_index(b, gc, d, h, w, batch_size, out_channels, out_depth, out_height, out_width);
        // Re-compute conv value for this channel
        float val = 0.0f;
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_d = d * stride + kd - padding;
                    int in_h = h * stride + kh - padding;
                    int in_w = w * stride + kw - padding;
                    
                    if (in_d >= 0 && in_d < in_depth && 
                        in_h >= 0 && in_h < in_height && 
                        in_w >= 0 && in_w < in_width) {
                        for (int ic = 0; ic < in_channels; ++ic) {
                            int in_idx = get_index(b, ic, in_d, in_h, in_w, 
                                                   batch_size, in_channels, in_depth, in_height, in_width);
                            int weight_idx = gc * (in_channels * kernel_size * kernel_size * kernel_size) + 
                                             ic * (kernel_size * kernel_size * kernel_size) + 
                                             kd * (kernel_size * kernel_size) + kh * kernel_size + kw;
                            val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        val += bias[gc];
        mean += val;
    }
    mean /= group_size;

    float var = 0.0f;
    for (int gc = group_start; gc < group_end; ++gc) {
        float val = 0.0f;
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_d = d * stride + kd - padding;
                    int in_h = h * stride + kh - padding;
                    int in_w = w * stride + kw - padding;
                    
                    if (in_d >= 0 && in_d < in_depth && 
                        in_h >= 0 && in_h < in_height && 
                        in_w >= 0 && in_w < in_width) {
                        for (int ic = 0; ic < in_channels; ++ic) {
                            int in_idx = get_index(b, ic, in_d, in_h, in_w, 
                                                   batch_size, in_channels, in_depth, in_height, in_width);
                            int weight_idx = gc * (in_channels * kernel_size * kernel_size * kernel_size) + 
                                             ic * (kernel_size * kernel_size * kernel_size) + 
                                             kd * (kernel_size * kernel_size) + kh * kernel_size + kw;
                            val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        val += bias[gc];
        float diff = val - mean;
        var += diff * diff;
    }
    var /= group_size;
    float eps = 1e-5f;
    float inv_std = rsqrtf(var + eps);

    // Apply group norm
    int c_local = c - group_start;
    float gn_weight = weight_gn[group_id * group_size + c_local];
    float gn_bias = bias_gn[group_id * group_size + c_local];
    float gn_output = (conv_val - mean) * inv_std * gn_weight + gn_bias;

    // Write normalized value back
    int norm_idx = get_index(b, c, d, h, w, batch_size, out_channels, out_depth, out_height, out_width);
    float* temp_norm = (float*)output;
    temp_norm[norm_idx] = gn_output;

    // Synchronize all threads (not possible directly, so we rely on grid stride and later kernel)
    // We launch another kernel for mean reduction or do it separately
}
"""

fused_reduction_source = """
__global__ void reduce_to_scalar(const float* input, float* output, 
                                 int batch_size, int out_channels, 
                                 int out_depth, int out_height, int out_width) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int total_spatial = out_channels * out_depth * out_height * out_width;
    int start = b * total_spatial;
    int end = start + total_spatial;

    float sum = 0.0f;
    for (int i = start + tid; i < end; i += blockDim.x) {
        sum += input[i];
    }

    __shared__ float shared_sum[1024];
    shared_sum[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b] = shared_sum[0] / total_spatial;  // mean
    }
}
"""

# Full fused function that calls both kernels
fused_main_source = """
torch::Tensor fused_conv_gn_mean_launcher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    int kernel_size, int stride, int padding,
    int num_groups) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto out_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
    auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    // Allocate output buffer large enough for activations
    auto options = input.options();
    auto norm_output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, options);
    auto result = torch::empty({batch_size}, options);

    const int total_conv_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_conv_elements + block_size - 1) / block_size;

    // Launch fused conv + GN kernel
    fused_conv_gn_mean_forward<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        weight_gn.data_ptr<float>(),
        bias_gn.data_ptr<float>(),
        norm_output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_size,
        out_depth, out_height, out_width,
        num_groups,
        stride, padding
    );

    // Launch reduction kernel
    reduce_to_scalar<<<batch_size, 256>>>(
        norm_output.data_ptr<float>(),
        result.data_ptr<float>(),
        batch_size, out_channels, out_depth, out_height, out_width
    );

    return result;
}
"""

# Combine all CUDA sources
cuda_sources = fused_conv_gn_mean_source + fused_reduction_source + fused_main_source
cpp_sources = "torch::Tensor fused_conv_gn_mean_launcher( \
    torch::Tensor input, \
    torch::Tensor weight, \
    torch::Tensor bias, \
    torch::Tensor weight_gn, \
    torch::Tensor bias_gn, \
    int kernel_size, int stride, int padding, \
    int num_groups);"

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_conv_gn_mean",
    cpp_sources=[cpp_sources],
    cuda_sources=[cuda_sources],
    functions=["fused_conv_gn_mean_launcher"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.stride = 1
        self.padding = kernel_size // 2

        # Define parameters manually to access data pointers
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.group_norm_weight = nn.Parameter(torch.empty(out_channels))
        self.group_norm_bias = nn.Parameter(torch.empty(out_channels))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.group_norm_weight)
        nn.init.zeros_(self.group_norm_bias)

        # Store num_groups for group norm
        self.num_groups = num_groups

        # Register the fused CUDA kernel
        self.fused_kernel = fused_ops.fused_conv_gn_mean_launcher

    def forward(self, x):
        return self.fused_kernel(
            x,
            self.weight,
            self.bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.num_groups
        )