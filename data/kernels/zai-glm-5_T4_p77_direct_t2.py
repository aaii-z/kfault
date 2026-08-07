import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused transposed convolution, scaling, batch norm, and global average pooling
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function to calculate output size for ConvTranspose3d
void calculate_output_size(
    int input_size, int kernel_size, int stride, int padding, int output_padding, int dilation, int* output_size) {
    *output_size = (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
}

// Kernel for fused ConvTranspose3d + Scale + BatchNorm + Global Average Pooling
__global__ void conv_transpose_scale_bn_gap_kernel(
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var, const float* bn_weight, const float* bn_bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int output_pad_d, int output_pad_h, int output_pad_w,
    float scale_factor, float eps) {

    // Shared memory for reduction
    extern __shared__ float shared_data[];

    int n = blockIdx.x; // Batch index
    int oc = blockIdx.y; // Output channel index
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Pointers for the current batch and channel
    const float* input_n = input + n * in_channels * in_depth * in_height * in_width;
    float* output_nc = output + n * out_channels + oc; // Output is (N, C, 1, 1, 1)

    // Initialize local sum for Global Average Pooling
    float local_sum = 0.0f;

    // ConvTranspose3d
    // Each thread block handles one (n, oc) pair.
    // Threads in the block iterate over input channels and kernel elements.
    for (int ic = 0; ic < in_channels; ++ic) {
        const float* weight_ic = weight + ic * out_channels * kernel_d * kernel_h * kernel_w;
        const float* weight_ic_oc = weight_ic + oc * kernel_d * kernel_h * kernel_w;
        const float* input_nic = input_n + ic * in_depth * in_height * in_width;

        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    float weight_val = weight_ic_oc[kd * kernel_h * kernel_w + kh * kernel_w + kw];
                    
                    // Iterate over input spatial dimensions
                    // Each thread handles a subset of the input pixels
                    int total_input_pixels = in_depth * in_height * in_width;
                    for (int i = tid; i < total_input_pixels; i += num_threads) {
                        int id = i / (in_height * in_width);
                        int ih = (i % (in_height * in_width)) / in_width;
                        int iw = (i % (in_height * in_width)) % in_width;

                        int od = id * stride_d - pad_d + kd * dilation_d;
                        int oh = ih * stride_h - pad_h + kh * dilation_h;
                        int ow = iw * stride_w - pad_w + kw * dilation_w;

                        if (od >= 0 && od < out_depth && oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                            float input_val = input_nic[id * in_height * in_width + ih * in_width + iw];
                            // Accumulate directly to local sum for Global Average Pooling
                            local_sum += input_val * weight_val;
                        }
                    }
                }
            }
        }
    }

    // Reduction in shared memory
    shared_data[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = shared_data[0];
        // Add bias, scale, and apply BatchNorm
        float mean = running_mean[oc];
        float var = running_var[oc];
        float w = bn_weight[oc];
        float b = bn_bias[oc];
        float conv_bias = bias[oc];

        float val = sum + conv_bias;
        val *= scale_factor;
        val = (val - mean) / sqrtf(var + eps);
        val = val * w + b;
        
        // Divide by number of elements for Global Average Pooling
        int num_elements = out_depth * out_height * out_width;
        val /= num_elements;
        
        output_nc[0] = val;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor bn_weight, torch::Tensor bn_bias,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int output_pad_d, int output_pad_h, int output_pad_w,
    float scale_factor, float eps) {

    // Calculate output dimensions
    int out_depth, out_height, out_width;
    calculate_output_size(in_depth, kernel_d, stride_d, pad_d, output_pad_d, dilation_d, &out_depth);
    calculate_output_size(in_height, kernel_h, stride_h, pad_h, output_pad_h, dilation_h, &out_height);
    calculate_output_size(in_width, kernel_w, stride_w, pad_w, output_pad_w, dilation_w, &out_width);

    // Allocate output tensor (N, C, 1, 1, 1)
    auto output = torch::zeros({batch_size, out_channels, 1, 1, 1}, input.options());

    const int block_size = 256;
    dim3 grid(batch_size, out_channels);
    size_t shared_mem_size = block_size * sizeof(float);

    conv_transpose_scale_bn_gap_kernel<<<grid, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(), bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        output_pad_d, output_pad_h, output_pad_w,
        scale_factor, eps);

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor bn_weight, torch::Tensor bn_bias, "
    "int batch_size, int in_channels, int out_channels, "
    "int in_depth, int in_height, int in_width, "
    "int kernel_d, int kernel_h, int kernel_w, "
    "int stride_d, int stride_h, int stride_w, "
    "int pad_d, int pad_h, int pad_w, "
    "int dilation_d, int dilation_h, int dilation_d, "
    "int output_pad_d, int output_pad_h, int output_pad_w, "
    "float scale_factor, float eps);"
)

# Compile the inline CUDA code for fused operations
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # We still need the original layers to hold the weights and running stats
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fused_ops = fused_ops

    def forward(self, x):
        # Extract parameters from the original layers
        batch_size, in_channels, in_depth, in_height, in_width = x.shape
        out_channels = self.conv_transpose.out_channels
        kernel_size = self.conv_transpose.kernel_size
        stride = self.conv_transpose.stride
        padding = self.conv_transpose.padding
        dilation = self.conv_transpose.dilation
        output_padding = self.conv_transpose.output_padding
        
        # Call the fused CUDA kernel
        return self.fused_ops.fused_ops_cuda(
            x, 
            self.conv_transpose.weight, 
            self.conv_transpose.bias,
            self.batch_norm.running_mean, 
            self.batch_norm.running_var, 
            self.batch_norm.weight, 
            self.batch_norm.bias,
            batch_size, in_channels, out_channels,
            in_depth, in_height, in_width,
            kernel_size[0], kernel_size[1], kernel_size[2],
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
            output_padding[0], output_padding[1], output_padding[2],
            self.scale_factor, 
            self.batch_norm.eps
        )