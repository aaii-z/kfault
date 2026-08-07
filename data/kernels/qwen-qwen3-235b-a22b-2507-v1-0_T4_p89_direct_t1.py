import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused ConvTranspose3d + MaxPool3d + Softmax + Subtract + Swish + Max
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Helper function to calculate 3D index
__device__ int get_index(int d, int h, int w, int D, int H, int W) {
    return d * H * W + h * W + w;
}

__global__ void fused_conv_maxpool_softmax_subtract_swish_max_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* subtract_val,
    float* output,
    // Input dimensions
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions after conv_transpose
    int out_depth_t, int out_height_t, int out_width_t,
    // After maxpool
    int out_depth, int out_height, int out_width,
    // ConvTranspose params
    int kernel_size_d, int kernel_size_h, int kernel_size_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    // MaxPool params
    int pool_kernel_size_d, int pool_kernel_size_h, int pool_kernel_size_w,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_padding_d, int pool_padding_h, int pool_padding_w,
    // Other
    int out_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_depth * out_height * out_width;
    if (idx >= total_elements) return;

    int n = idx / (out_depth * out_height * out_width);
    int temp = idx % (out_depth * out_height * out_width);
    int d = temp / (out_height * out_width);
    int h = (temp / out_width) % out_height;
    int w = temp % out_width;

    // Allocate shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* transposed = shared_mem;
    float* pooled = &transposed[out_channels * out_depth_t * out_height_t * out_width_t];
    float* softmax_out = &pooled[out_channels * out_depth * out_height * out_width];
    float* swish_out = &softmax_out[out_channels * out_depth * out_height * out_width];

    // ConvTranspose3d
    for (int oc = 0; oc < out_channels; oc++) {
        for (int od = 0; od < out_depth_t; od++) {
            for (int oh = 0; oh < out_height_t; oh++) {
                for (int ow = 0; ow < out_width_t; ow++) {
                    float val = 0.0f;
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kd = 0; kd < kernel_size_d; kd++) {
                            for (int kh = 0; kh < kernel_size_h; kh++) {
                                for (int kw = 0; kw < kernel_size_w; kw++) {
                                    int id = od * stride_d - padding_d + kd;
                                    int ih = oh * stride_h - padding_h + kh;
                                    int iw = ow * stride_w - padding_w + kw;

                                    // Apply output padding
                                    if (output_padding_d > 0 && (od % stride_d) == 0 && od > 0) id += output_padding_d;
                                    if (output_padding_h > 0 && (oh % stride_h) == 0 && oh > 0) ih += output_padding_h;
                                    if (output_padding_w > 0 && (ow % stride_w) == 0 && ow > 0) iw += output_padding_w;

                                    if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                        int in_idx = n * in_channels * in_depth * in_height * in_width +
                                                     ic * in_depth * in_height * in_width +
                                                     id * in_height * in_width + ih * in_width + iw;
                                        int weight_idx = oc * in_channels * kernel_size_d * kernel_size_h * kernel_size_w +
                                                         ic * kernel_size_d * kernel_size_h * kernel_size_w +
                                                         kd * kernel_size_h * kernel_size_w + kh * kernel_size_w + kw;
                                        val += input[in_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                    val += bias[oc];
                    int t_idx = oc * (out_depth_t * out_height_t * out_width_t) +
                                od * (out_height_t * out_width_t) + oh * out_width_t + ow;
                    transposed[t_idx] = val;
                }
            }
        }
    }

    // MaxPool3d
    for (int oc = 0; oc < out_channels; oc++) {
        for (int od = 0; od < out_depth; od++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float max_val = -FLT_MAX;
                    for (int pd = 0; pd < pool_kernel_size_d; pd++) {
                        for (int ph = 0; ph < pool_kernel_size_h; ph++) {
                            for (int pw = 0; pw < pool_kernel_size_w; pw++) {
                                int id = od * pool_stride_d - pool_padding_d + pd;
                                int ih = oh * pool_stride_h - pool_padding_h + ph;
                                int iw = ow * pool_stride_w - pool_padding_w + pw;
                                if (id >= 0 && id < out_depth_t && ih >= 0 && ih < out_height_t && iw >= 0 && iw < out_width_t) {
                                    int t_idx = oc * (out_depth_t * out_height_t * out_width_t) +
                                                id * (out_height_t * out_width_t) + ih * out_width_t + iw;
                                    float val = transposed[t_idx];
                                    if (val > max_val) max_val = val;
                                }
                            }
                        }
                    }
                    int p_idx = oc * (out_depth * out_height * out_width) +
                                od * (out_height * out_width) + oh * out_width + ow;
                    pooled[p_idx] = max_val;
                }
            }
        }
    }

    // Softmax across channel dimension (dim=1)
    for (int od = 0; od < out_depth; od++) {
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                float max_val = -FLT_MAX;
                for (int oc = 0; oc < out_channels; oc++) {
                    int p_idx = oc * (out_depth * out_height * out_width) +
                                od * (out_height * out_width) + oh * out_width + ow;
                    float val = pooled[p_idx];
                    if (val > max_val) max_val = val;
                }
                float sum = 0.0f;
                for (int oc = 0; oc < out_channels; oc++) {
                    int p_idx = oc * (out_depth * out_height * out_width) +
                                od * (out_height * out_width) + oh * out_width + ow;
                    float val = expf(pooled[p_idx] - max_val);
                    int s_idx = oc * (out_depth * out_height * out_width) +
                                od * (out_height * out_width) + oh * out_width + ow;
                    softmax_out[s_idx] = val;
                    sum += val;
                }
                for (int oc = 0; oc < out_channels; oc++) {
                    int s_idx = oc * (out_depth * out_height * out_width) +
                                od * (out_height * out_width) + oh * out_width + ow;
                    softmax_out[s_idx] /= sum;
                }
            }
        }
    }

    // Subtract and Swish
    for (int oc = 0; oc < out_channels; oc++) {
        float sub_val = subtract_val[oc];
        for (int od = 0; od < out_depth; od++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    int s_idx = oc * (out_depth * out_height * out_width) +
                                od * (out_height * out_width) + oh * out_width + ow;
                    float val = softmax_out[s_idx] - sub_val;
                    float sigmoid_val = 1.0f / (1.0f + expf(-val));
                    int sw_idx = oc * (out_depth * out_height * out_width) +
                                 od * (out_height * out_width) + oh * out_width + ow;
                    swish_out[sw_idx] = val * sigmoid_val;
                }
            }
        }
    }

    // Max across channels
    float max_val = -FLT_MAX;
    for (int oc = 0; oc < out_channels; oc++) {
        int sw_idx = oc * (out_depth * out_height * out_width) +
                     d * (out_height * out_width) + h * out_width + w;
        float val = swish_out[sw_idx];
        if (val > max_val) max_val = val;
    }
    output[idx] = max_val;
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor subtract_val,
    int in_channels, int out_channels,
    int kernel_size_d, int kernel_size_h, int kernel_size_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int pool_kernel_size_d, int pool_kernel_size_h, int pool_kernel_size_w,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_padding_d, int pool_padding_h, int pool_padding_w,
    int out_depth_t, int out_height_t, int out_width_t,
    int out_depth, int out_height, int out_width
) {
    auto batch_size = input.size(0);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);

    auto output = torch::zeros({batch_size, out_depth, out_height, out_width}, input.options());

    const int total_elements = batch_size * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Shared memory size: transposed + pooled + softmax_out + swish_out
    int shared_mem_size = (out_channels * out_depth_t * out_height_t * out_width_t +
                           out_channels * out_depth * out_height * out_width +
                           out_channels * out_depth * out_height * out_width +
                           out_channels * out_depth * out_height * out_width) * sizeof(float);

    fused_conv_maxpool_softmax_subtract_swish_max_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        subtract_val.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_depth_t, out_height_t, out_width_t,
        out_depth, out_height, out_width,
        kernel_size_d, kernel_size_h, kernel_size_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        pool_kernel_size_d, pool_kernel_size_h, pool_kernel_size_w,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_padding_d, pool_padding_h, pool_padding_w,
        out_channels
    );

    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor subtract_val,
    int in_channels, int out_channels,
    int kernel_size_d, int kernel_size_h, int kernel_size_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int pool_kernel_size_d, int pool_kernel_size_h, int pool_kernel_size_w,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_padding_d, int pool_padding_h, int pool_padding_w,
    int out_depth_t, int out_height_t, int out_width_t,
    int out_depth, int out_height, int out_width
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=[fused_kernel_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding
        )
        self.subtract = nn.Parameter(torch.randn(out_channels))

        # Precompute output shapes
        def compute_transpose_output_shape(dim, k, s, p, op):
            return (dim - 1) * s - 2 * p + k + op
        def compute_pooled_output_shape(dim, k, s, p):
            return (dim + 2 * p - k) // s + 1

        self.out_depth_t = compute_transpose_output_shape(16, kernel_size, stride, padding, output_padding)
        self.out_height_t = compute_transpose_output_shape(32, kernel_size, stride, padding, output_padding)
        self.out_width_t = compute_transpose_output_shape(32, kernel_size, stride, padding, output_padding)

        self.out_depth = compute_pooled_output_shape(self.out_depth_t, pool_kernel_size, pool_stride, pool_padding)
        self.out_height = compute_pooled_output_shape(self.out_height_t, pool_kernel_size, pool_stride, pool_padding)
        self.out_width = compute_pooled_output_shape(self.out_width_t, pool_kernel_size, pool_stride, pool_padding)

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        subtract_val = self.subtract

        output = fused_op.fused_op_cuda(
            x, weight, bias, subtract_val,
            self.in_channels, self.out_channels,
            self.kernel_size, self.kernel_size, self.kernel_size,
            self.stride, self.stride, self.stride,
            self.padding, self.padding, self.padding,
            self.output_padding, self.output_padding, self.output_padding,
            self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size,
            self.pool_stride, self.pool_stride, self.pool_stride,
            self.pool_padding, self.pool_padding, self.pool_padding,
            self.out_depth_t, self.out_height_t, self.out_width_t,
            self.out_depth, self.out_height, self.out_width
        )
        return output