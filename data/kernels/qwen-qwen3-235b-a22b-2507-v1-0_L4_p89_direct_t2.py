import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused ConvTranspose3d + MaxPool3d + Softmax + Subtract + Swish + Max
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Efficient 3D index calculation
__device__ int get_idx(int n, int c, int d, int h, int w, 
                      int N, int C, int D, int H, int W) {
    return ((n * C + c) * D + d) * H * W + h * W + w;
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
    int kernel_size, int stride, int padding, int output_padding,
    // MaxPool params
    int pool_kernel_size, int pool_stride, int pool_padding,
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

    // Shared memory for intermediate feature maps
    extern __shared__ float shared_mem[];
    float* transposed = shared_mem;
    float* pooled = transposed + batch_size * out_channels * out_depth_t * out_height_t * out_width_t;
    float* softmax_out = pooled + batch_size * out_channels * out_depth * out_height * out_width;
    float* swish_out = softmax_out + batch_size * out_channels * out_depth * out_height * out_width;

    // ConvTranspose3d: compute all output points for this sample
    if (threadIdx.x == 0) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int od = 0; od < out_depth_t; od++) {
                for (int oh = 0; oh < out_height_t; oh++) {
                    for (int ow = 0; ow < out_width_t; ow++) {
                        float val = 0.0f;
                        for (int ic = 0; ic < in_channels; ic++) {
                            for (int kd = 0; kd < kernel_size; kd++) {
                                for (int kh = 0; kh < kernel_size; kh++) {
                                    for (int kw = 0; kw < kernel_size; kw++) {
                                        int id = od * stride - padding + kd;
                                        int ih = oh * stride - padding + kh;
                                        int iw = ow * stride - padding + kw;

                                        // Apply output padding every 'stride' steps
                                        if (output_padding > 0) {
                                            if ((od > 0) && (od % stride == 0)) id += output_padding;
                                            if ((oh > 0) && (oh % stride == 0)) ih += output_padding;
                                            if ((ow > 0) && (ow % stride == 0)) iw += output_padding;
                                        }

                                        if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                            int in_idx = ((n * in_channels + ic) * in_depth + id) * in_height * in_width + ih * in_width + iw;
                                            int w_idx = ((oc * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                                            val += input[in_idx] * weight[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        val += bias[oc];
                        int t_idx = ((n * out_channels + oc) * out_depth_t + od) * out_height_t * out_width_t + oh * out_width_t + ow;
                        transposed[t_idx] = val;
                    }
                }
            }
        }
    }
    __syncthreads();

    // MaxPool3d: reduce over pooling window
    for (int oc = 0; oc < out_channels; oc++) {
        for (int od = 0; od < out_depth; od++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float max_val = -FLT_MAX;
                    for (int pd = 0; pd < pool_kernel_size; pd++) {
                        for (int ph = 0; ph < pool_kernel_size; ph++) {
                            for (int pw = 0; pw < pool_kernel_size; pw++) {
                                int id = od * pool_stride - pool_padding + pd;
                                int ih = oh * pool_stride - pool_padding + ph;
                                int iw = ow * pool_stride - pool_padding + pw;
                                if (id >= 0 && id < out_depth_t && ih >= 0 && ih < out_height_t && iw >= 0 && iw < out_width_t) {
                                    int t_idx = ((n * out_channels + oc) * out_depth_t + id) * out_height_t * out_width_t + ih * out_width_t + iw;
                                    float val = transposed[t_idx];
                                    if (val > max_val) max_val = val;
                                }
                            }
                        }
                    }
                    int p_idx = ((n * out_channels + oc) * out_depth + od) * out_height * out_width + oh * out_width + ow;
                    pooled[p_idx] = max_val;
                }
            }
        }
    }
    __syncthreads();

    // Softmax across channel dimension (dim=1) - per spatial location
    for (int od = 0; od < out_depth; od++) {
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                float max_val = -FLT_MAX;
                for (int oc = 0; oc < out_channels; oc++) {
                    int p_idx = ((n * out_channels + oc) * out_depth + od) * out_height * out_width + oh * out_width + ow;
                    float val = pooled[p_idx];
                    if (val > max_val) max_val = val;
                }
                float sum = 0.0f;
                for (int oc = 0; oc < out_channels; oc++) {
                    int p_idx = ((n * out_channels + oc) * out_depth + od) * out_height * out_width + oh * out_width + ow;
                    float val = expf(pooled[p_idx] - max_val);
                    int s_idx = ((n * out_channels + oc) * out_depth + od) * out_height * out_width + oh * out_width + ow;
                    softmax_out[s_idx] = val;
                    sum += val;
                }
                for (int oc = 0; oc < out_channels; oc++) {
                    int s_idx = ((n * out_channels + oc) * out_depth + od) * out_height * out_width + oh * out_width + ow;
                    softmax_out[s_idx] /= sum;
                }
            }
        }
    }
    __syncthreads();

    // Subtract and Swish: x = (x - b) * sigmoid(x - b)
    for (int oc = 0; oc < out_channels; oc++) {
        float sub_val = subtract_val[oc];
        for (int od = 0; oc < out_channels && od < out_depth; od++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    int s_idx = ((n * out_channels + oc) * out_depth + od) * out_height * out_width + oh * out_width + ow;
                    float val = softmax_out[s_idx] - sub_val;
                    float sigmoid_val = 1.0f / (1.0f + expf(-val));
                    int sw_idx = ((n * out_channels + oc) * out_depth + od) * out_height * out_width + oh * out_width + ow;
                    swish_out[sw_idx] = val * sigmoid_val;
                }
            }
        }
    }
    __syncthreads();

    // Final Max across channels at (d, h, w)
    float max_val = -FLT_MAX;
    for (int oc = 0; oc < out_channels; oc++) {
        int sw_idx = ((n * out_channels + oc) * out_depth + d) * out_height * out_width + h * out_width + w;
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
    int kernel_size, int stride, int padding, int output_padding,
    int pool_kernel_size, int pool_stride, int pool_padding,
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

    // Shared memory: transposed + pooled + softmax_out + swish_out
    int shared_mem_size = (batch_size * out_channels * out_depth_t * out_height_t * out_width_t +
                          batch_size * out_channels * out_depth * out_height * out_width +
                          batch_size * out_channels * out_depth * out_height * out_width +
                          batch_size * out_channels * out_depth * out_height * out_width) * sizeof(float);

    fused_conv_maxpool_softmax_subtract_swish_max_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        subtract_val.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_depth_t, out_height_t, out_width_t,
        out_depth, out_height, out_width,
        kernel_size, stride, padding, output_padding,
        pool_kernel_size, pool_stride, pool_padding,
        out_channels
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\\n", cudaGetErrorString(err));
    }

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
    int kernel_size, int stride, int padding, int output_padding,
    int pool_kernel_size, int pool_stride, int pool_padding,
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
            self.kernel_size, self.stride, self.padding, self.output_padding,
            self.pool_kernel_size, self.pool_stride, self.pool_padding,
            self.out_depth_t, self.out_height_t, self.out_width_t,
            self.out_depth, self.out_height, self.out_width
        )
        return output