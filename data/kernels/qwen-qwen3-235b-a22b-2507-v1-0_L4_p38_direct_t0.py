import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused AvgPool3d + ConvTranspose3d + Clamp + Softmax + Scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Efficient fused kernel for AvgPool3d -> ConvTranspose3d -> Clamp -> Softmax -> Scale
__global__ void fused_op_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* scale,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int pool_kernel_size,
    int kernel_size, int stride, int padding, int output_padding,
    float clamp_min, float clamp_max,
    int input_pooled_depth, int input_pooled_height, int input_pooled_width,
    int weight_offset, int bias_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int d = (idx / (out_width * out_height)) % out_depth;
    int c = (idx / (out_width * out_height * out_depth)) % out_channels;
    int b = idx / (out_width * out_height * out_depth * out_channels);

    // Compute AvgPool3d input indices (reverse of pooling)
    int pool_d_start = d * stride - padding;
    int pool_h_start = h * stride - padding;
    int pool_w_start = w * stride - padding;

    float conv_sum = 0.0f;

    // ConvTranspose3d: loop over kernel and input
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int id = pool_d_start + kd;
                int ih = pool_h_start + kh;
                int iw = pool_w_start + kw;

                // Check bounds after accounting for output padding
                if (id >= 0 && id < input_pooled_depth && 
                    ih >= 0 && ih < input_pooled_height && 
                    iw >= 0 && iw < input_pooled_width) {

                    for (int ic = 0; ic < in_channels; ++ic) {
                        // Pooling: average over pool_kernel_size in each dim
                        float pool_sum = 0.0f;
                        int pool_count = 0;

                        for (int pd = 0; pd < pool_kernel_size; ++pd) {
                            for (int ph = 0; ph < pool_kernel_size; ++ph) {
                                for (int pw = 0; pw < pool_kernel_size; ++pw) {
                                    int orig_d = id * pool_kernel_size + pd;
                                    int orig_h = ih * pool_kernel_size + ph;
                                    int orig_w = iw * pool_kernel_size + pw;

                                    if (orig_d < in_depth && orig_h < in_height && orig_w < in_width) {
                                        int input_idx = b * in_channels * in_depth * in_height * in_width +
                                                       ic * in_depth * in_height * in_width +
                                                       orig_d * in_height * in_width +
                                                       orig_h * in_width + orig_w;
                                        pool_sum += input[input_idx];
                                        ++pool_count;
                                    }
                                }
                            }
                        }

                        float pooled_val = pool_count > 0 ? pool_sum / (pool_kernel_size * pool_kernel_size * pool_kernel_size) : 0.0f;

                        int weight_idx = c * in_channels * kernel_size * kernel_size * kernel_size +
                                        ic * kernel_size * kernel_size * kernel_size +
                                        kd * kernel_size * kernel_size + kh * kernel_size + kw;
                        conv_sum += pooled_val * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias
    conv_sum += bias[c];

    // Clamp
    float clamped = fmaxf(clamp_min, fminf(clamp_max, conv_sum));

    // Store clamped value temporarily in output (will be overwritten later by softmax)
    output[idx] = clamped;
}

// Second kernel for softmax (spatial softmax over DHW per channel)
__global__ void softmax_kernel(
    float* data,
    int batch_size, int out_channels, int out_depth, int out_height, int out_width
) {
    int bc = blockIdx.x;
    int b = bc / out_channels;
    int c = bc % out_channels;

    int spatial_size = out_depth * out_height * out_width;
    int offset = b * out_channels * spatial_size + c * spatial_size;

    // Shared memory for block reduction
    extern __shared__ float sdata[];

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        thread_sum += expf(data[offset + i]);
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum = sdata[0];
    __syncthreads();

    // Normalize
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        data[offset + i] = expf(data[offset + i]) / (sum + 1e-8f);
    }
}

// Third kernel for scaling
__global__ void scale_kernel(float* data, const float* scale, int total_elements, int out_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int c = (idx / (data_stride)) % out_channels;  // Note: data_stride not defined in kernel
    // Fix: compute c manually
    int stride_w = 1;
    int stride_h = out_width;
    int stride_d = out_width * out_height;
    int stride_c = out_width * out_height * out_depth;
    int c_idx = (idx / stride_c) % out_channels;

    data[idx] *= scale[c_idx];
}

// Fixed scale kernel
__global__ void scale_kernel_fixed(float* data, const float* scale, int total_elements, int out_channels, int out_depth, int out_height, int out_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int rem = idx;
    int w = rem % out_width; rem /= out_width;
    int h = rem % out_height; rem /= out_height;
    int d = rem % out_depth; rem /= out_depth;
    int c = rem % out_channels;
    // b = rem / out_channels;

    data[idx] *= scale[c];
}
"""

fused_op_cpp_source = """
torch::Tensor fused_model_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_padding,
    int64_t pool_kernel_size,
    float clamp_min, float clamp_max,
    int64_t out_depth, int64_t out_height, int64_t out_width
);
"""

# Full corrected CUDA source with proper kernels
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel 1: Fused AvgPool3d + ConvTranspose3d + Clamp
__global__ void fused_pool_transpose_clamp_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int pool_kernel_size,
    int kernel_size, int stride, int padding, int output_padding,
    float clamp_min, float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;

    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int d = (idx / (out_width * out_height)) % out_depth;
    int c = (idx / (out_width * out_height * out_depth)) % out_channels;
    int b = idx / (out_width * out_height * out_depth * out_channels);

    // Compute input indices for transposed conv
    int pool_d_start = d * stride - padding;
    int pool_h_start = h * stride - padding;
    int pool_w_start = w * stride - padding;

    float conv_sum = 0.0f;

    // ConvTranspose3d with implicit pooling
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int id = pool_d_start + kd;
                int ih = pool_h_start + kh;
                int iw = pool_w_start + kw;

                if (id >= 0 && id < (in_depth + pool_kernel_size - 1) / pool_kernel_size && 
                    ih >= 0 && ih < (in_height + pool_kernel_size - 1) / pool_kernel_size && 
                    iw >= 0 && iw < (in_width + pool_kernel_size - 1) / pool_kernel_size) {

                    for (int ic = 0; ic < in_channels; ++ic) {
                        // AvgPool3d: average over pool_kernel_size
                        float pool_sum = 0.0f;
                        int pool_count = 0;

                        for (int pd = 0; pd < pool_kernel_size; ++pd) {
                            for (int ph = 0; ph < pool_kernel_size; ++ph) {
                                for (int pw = 0; pw < pool_kernel_size; ++pw) {
                                    int orig_d = id * pool_kernel_size + pd;
                                    int orig_h = ih * pool_kernel_size + ph;
                                    int orig_w = iw * pool_kernel_size + pw;

                                    if (orig_d < in_depth && orig_h < in_height && orig_w < in_width) {
                                        int input_idx = b * in_channels * in_depth * in_height * in_width +
                                                       ic * in_depth * in_height * in_width +
                                                       orig_d * in_height * in_width +
                                                       orig_h * in_width + orig_w;
                                        pool_sum += input[input_idx];
                                        ++pool_count;
                                    }
                                }
                            }
                        }

                        float pooled_val = pool_count > 0 ? pool_sum / (pool_kernel_size * pool_kernel_size * pool_kernel_size) : 0.0f;

                        int weight_idx = c * in_channels * kernel_size * kernel_size * kernel_size +
                                        ic * kernel_size * kernel_size * kernel_size +
                                        kd * kernel_size * kernel_size + kh * kernel_size + kw;
                        conv_sum += pooled_val * weight[weight_idx];
                    }
                }
            }
        }
    }

    conv_sum += bias[c];
    output[idx] = fmaxf(clamp_min, fminf(clamp_max, conv_sum));
}

// Kernel 2: In-place spatial softmax over DHW per channel
__global__ void softmax_kernel(
    float* data,
    int batch_size, int out_channels, int spatial_size
) {
    int bc = blockIdx.x;
    int offset = bc * spatial_size;

    extern __shared__ float sdata[];

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        thread_sum += expf(data[offset + i]);
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum = sdata[0];
    __syncthreads();

    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        data[offset + i] = expf(data[offset + i]) / fmaxf(sum, 1e-8f);
    }
}

// Kernel 3: Scale multiplication
__global__ void scale_kernel(
    float* data,
    const float* scale,
    int total_elements,
    int out_channels,
    int out_depth, int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int rem = idx;
    int w = rem % out_width; rem /= out_width;
    int h = rem % out_height; rem /= out_height;
    int d = rem % out_depth; rem /= out_depth;
    int c = rem % out_channels;

    data[idx] *= scale[c];
}

torch::Tensor fused_model_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_padding,
    int64_t pool_kernel_size,
    float clamp_min, float clamp_max,
    int64_t out_depth, int64_t out_height, int64_t out_width
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(scale);

    int batch_size = input.size(0);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, 
                               input.options());

    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_pool_transpose_clamp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        pool_kernel_size,
        kernel_size, stride, padding, output_padding,
        clamp_min, clamp_max
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in fused_pool_transpose_clamp_kernel: %s\\n", cudaGetErrorString(err));
    }

    // Softmax: one block per (batch, channel)
    int spatial_size = out_depth * out_height * out_width;
    int bc_blocks = batch_size * out_channels;
    int smem_size = block_size * sizeof(float);
    softmax_kernel<<<bc_blocks, block_size, smem_size>>>(
        output.data_ptr<float>(),
        batch_size, out_channels, spatial_size
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in softmax_kernel: %s\\n", cudaGetErrorString(err));
    }

    // Scale
    scale_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        scale.data_ptr<float>(),
        total_elements,
        out_channels,
        out_depth, out_height, out_width
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in scale_kernel: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_op = load_inline(
    name="fused_model",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_model_forward"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        # Compute output shape manually
        def deconv_output_length(dim, kernel_size, stride, padding, output_padding):
            return (dim - 1) * stride - 2 * padding + kernel_size + output_padding

        out_depth = deconv_output_length(
            x.shape[2] // self.pool_kernel_size, 
            self.kernel_size, self.stride, self.padding, self.output_padding
        )
        out_height = deconv_output_length(
            x.shape[3] // self.pool_kernel_size, 
            self.kernel_size, self.stride, self.padding, self.output_padding
        )
        out_width = deconv_output_length(
            x.shape[4] // self.pool_kernel_size, 
            self.kernel_size, self.stride, self.padding, self.output_padding
        )

        return fused_op.fused_model_forward(
            x, 
            self.conv_transpose.weight, 
            self.conv_transpose.bias,
            self.scale.squeeze(),
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.pool_kernel_size,
            self.clamp_min,
            self.clamp_max,
            out_depth,
            out_height,
            out_width
        )