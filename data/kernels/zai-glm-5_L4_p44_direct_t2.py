import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused transposed convolution, scalar multiplication, and global average pooling
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Kernel for fused ConvTranspose2d + Mul + Global Average Pooling
__global__ void fused_conv_transpose_mul_gap_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int output_pad_h, int output_pad_w,
    float multiplier) {

    int n = blockIdx.x; // Batch index
    int oc = blockIdx.y; // Output channel index
    
    // Shared memory for reduction
    __shared__ float s_sum[256]; 
    
    int tid = threadIdx.x;
    s_sum[tid] = 0.0f;
    
    int spatial_size = out_h * out_w;
    
    // Strided loop over output pixels
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int oh = i / out_w;
        int ow = i % out_w;
        
        float val = 0.0f;
        
        // Compute ConvTranspose2d for this specific output pixel (oh, ow)
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int ih_candidate = oh + pad_h - kh;
                    int iw_candidate = ow + pad_w - kw;
                    
                    if (ih_candidate % stride_h == 0 && iw_candidate % stride_w == 0) {
                        int ih = ih_candidate / stride_h;
                        int iw = iw_candidate / stride_w;
                        
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            int input_idx = n * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
                            // Corrected weight indexing: weight layout is (in_channels, out_channels, kH, kW)
                            int weight_idx = ic * (out_channels * kernel_h * kernel_w) + oc * (kernel_h * kernel_w) + kh * kernel_w + kw;
                            val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            val += bias[oc];
        }
        
        val *= multiplier;
        s_sum[tid] += val;
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        int output_idx = n * out_channels + oc;
        output[output_idx] = s_sum[0] / spatial_size;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int output_pad_h, int output_pad_w,
    float multiplier) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    
    // Weight shape for ConvTranspose2d is (in_channels, out_channels, kH, kW)
    int out_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;
    
    auto output = torch::zeros({batch_size, out_channels, 1, 1}, input.options());
    
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        bias_ptr = bias_opt.value().data_ptr<float>();
    }
    
    const int block_size = 256;
    dim3 grid(batch_size, out_channels);
    
    fused_conv_transpose_mul_gap_kernel<<<grid, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_h, in_w,
        out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        output_pad_h, output_pad_w,
        multiplier
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias, int stride_h, int stride_w, int pad_h, int pad_w, int output_pad_h, int output_pad_w, float multiplier);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        self.fused_op = fused_ops

    def forward(self, x):
        stride = self.conv_transpose.stride
        padding = self.conv_transpose.padding
        output_padding = self.conv_transpose.output_padding
        
        return self.fused_op.fused_op_cuda(
            x, 
            self.conv_transpose.weight, 
            self.conv_transpose.bias, 
            stride[0], stride[1], 
            padding[0], padding[1], 
            output_padding[0], output_padding[1], 
            self.multiplier
        )