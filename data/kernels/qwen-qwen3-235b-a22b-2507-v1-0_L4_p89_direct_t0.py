import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused ConvTranspose3d + MaxPool3d + Softmax + Subtract + Swish + Max
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

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

    // Output volume after conv_transpose
    float* transposed = new float[out_channels * out_depth_t * out_height_t * out_width_t];
    for (int oc = 0; oc < out_channels; oc++) {
        for (int od = 0; od < out_depth_t; od++) {
            for (int oh = 0; oh < out_height_t; oh++) {
                for (int ow = 0; ow < out_width_t; ow++) {
                    float val = 0.0f;
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kd = 0; kd < kernel_size_d; kd++) {
                            for (int kh = 0; kh < kernel_size_h; kh++) {
                                for (int kw = 0; kw < kernel_size_w; kw++) {
                                    int id = od * stride_d - padding_d + kd + (output_padding_d > 0 ? (od % stride_d == 0 ? output_padding_d : 0) : 0);
                                    int ih = oh * stride_h - padding_h + kh + (output_padding_h > 0 ? (oh % stride_h == 0 ? output_padding_h : 0) : 0);
                                    int iw = ow * stride_w - padding_w + kw + (output_padding_w > 0 ? (ow % stride_w == 0 ? output_padding_w : 0) : 0);

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
                    transposed[oc * out_depth_t * out_height_t * out_width_t +
                               od * out_height_t * out_width_t + oh * out_width_t + ow] = val;
                }
            }
        }
    }

    // MaxPool3d
    float* pooled = new float[out_channels * out_depth * out_height * out_width];
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
                                    float val = transposed[oc * out_depth_t * out_height_t * out_width_t +
                                                           id * out_height_t * out_width_t + ih * out_width_t + iw];
                                    if (val > max_val) max_val = val;
                                }
                            }
                        }
                    }
                    pooled[oc * out_depth * out_height * out_width +
                           od * out_height * out_width + oh * out_width + ow] = max_val;
                }
            }
        }
    }

    // Softmax across channel dimension (dim=1)
    float* softmax_out = new float[out_channels * out_depth * out_height * out_width];
    for (int od = 0; od < out_depth; od++) {
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                float max_val = -FLT_MAX;
                for (int oc = 0; oc < out_channels; oc++) {
                    float val = pooled[oc * out_depth * out_height * out_width +
                                       od * out_height * out_width + oh * out_width + ow];
                    if (val > max_val) max_val = val;
                }
                float sum = 0.0f;
                for (int oc = 0; oc < out_channels; oc++) {
                    float val = expf(pooled[oc * out_depth * out_height * out_width +
                                            od * out_height * out_width + oh * out_width + ow] - max_val);
                    softmax_out[oc * out_depth * out_height * out_width +
                                od * out_height * out_width + oh * out_width + ow] = val;
                    sum += val;
                }
                for (int oc = 0; oc < out_channels; oc++) {
                    softmax_out[oc * out_depth * out_height * out_width +
                                od * out_height * out_width + oh * out_width + ow] /= sum;
                }
            }
        }
    }

    // Subtract and Swish
    float* swish_out = new float[out_channels * out_depth * out_height * out_width];
    for (int oc = 0; oc < out_channels; oc++) {
        float sub_val = subtract_val[oc];
        for (int od = 0; oc < out_depth; od++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float val = softmax_out[oc * out_depth * out_height * out_width +
                                            od * out_height * out_width + oh * out_width + ow] - sub_val;
                    float sigmoid_val = 1.0f / (1.0f + expf(-val));
                    swish_out[oc * out_depth * out_height * out_width +
                              od * out_height * out_width + oh * out_width + ow] = val * sigmoid_val;
                }
            }
        }
    }

    // Max across channels
    float max_val = -FLT_MAX;
    for (int oc = 0; oc < out_channels; oc++) {
        float val = swish_out[oc * out_depth * out_height * out_width +
                              d * out_height * out_width + h * out_width + w];
        if (val > max_val) max_val = val;
    }
    output[idx] = max_val;

    delete[] transposed;
    delete[] pooled;
    delete[] softmax_out;
    delete[] swish_out;
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
            return (dim + 2 * p - k) / s + 1

        self.out_depth_t = compute_transpose_output_shape(16, kernel_size, stride, padding, output_padding)
        self.out_height_t = compute_transpose_output_shape(32, kernel_size, stride, padding, output_padding)
        self.out_width_t = compute_transpose_output_shape(32, kernel_size, stride, padding, output_padding)

        self.out_depth = int(compute_pooled_output_shape(self.out_depth_t, pool_kernel_size, pool_stride, pool_padding))
        self.out_height = int(compute_pooled_output_shape(self.out_height_t, pool_kernel_size, pool_stride, pool_padding))
        self.out_width = int(compute_pooled_output_shape(self.out_width_t, pool_kernel_size, pool_stride, pool_padding))

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