import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose2d + Mul + Global Average Pooling
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for fused ConvTranspose2d, Scalar Multiplication, and Global Average Pooling
// This kernel computes the output for one (n, c_out) pair.
// Each thread block handles one output channel for one batch item.
__global__ void fused_conv_transpose_mul_gap_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    float multiplier,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kH, int kW, int stride, int padding) {

    int n = blockIdx.x;
    int c_out = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Shared memory for weights and reduction
    // We need space for weights (C_in * kH * kW) and partial sums (block_size)
    // To simplify, we use extern shared memory. 
    // Layout: [weights (C_in * kH * kW)] followed by [partial_sums (block_size)]
    extern __shared__ char shared_mem[];
    float* s_weights = reinterpret_cast<float*>(shared_mem);
    float* s_partial_sums = reinterpret_cast<float*>(shared_mem + C_in * kH * kW * sizeof(float));

    // Load weights for this output channel into shared memory
    // Weight layout: (C_in, C_out, kH, kW)
    int weight_size = C_in * kH * kW;
    for (int i = tid; i < weight_size; i += block_size) {
        int c_in = i / (kH * kW);
        int k_idx = i % (kH * kW);
        int w_idx = c_in * C_out * kH * kW + c_out * kH * kW + k_idx;
        s_weights[i] = weight[w_idx];
    }
    __syncthreads();

    float local_sum = 0.0f;
    float b = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    int total_pixels = H_out * W_out;

    // Iterate over output pixels assigned to this thread
    for (int pix_idx = tid; pix_idx < total_pixels; pix_idx += block_size) {
        int h_out = pix_idx / W_out;
        int w_out = pix_idx % W_out;

        float val = 0.0f;

        // Compute ConvTranspose for this output pixel
        // Iterate over input channels and kernel
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    // Calculate input indices based on ConvTranspose relation
                    // h_out = (h_in - 1) * stride - 2 * padding + kh
                    // => h_in = (h_out + 2 * padding - kh) / stride
                    // We check if the mapping is valid (aligned and within bounds)
                    
                    int h_in_candidate = h_out + 2 * padding - kh;
                    int w_in_candidate = w_out + 2 * padding - kw;

                    if (h_in_candidate % stride == 0 && w_in_candidate % stride == 0) {
                        int h_in = h_in_candidate / stride;
                        int w_in = w_in_candidate / stride;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int input_idx = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in;
                            int weight_idx = c_in * kH * kW + kh * kW + kw; // Index in shared memory
                            val += input[input_idx] * s_weights[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias
        val += b;
        
        // Accumulate for Global Average Pooling
        // We sum the values. The division happens later.
        local_sum += val;
    }

    // Reduction in shared memory
    s_partial_sums[tid] = local_sum;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial_sums[tid] += s_partial_sums[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        float total_sum = s_partial_sums[0];
        // Apply multiplier
        total_sum *= multiplier;
        // Average
        float mean_val = total_sum / total_pixels;
        
        // Output shape is (N, C_out, 1, 1)
        output[n * C_out + c_out] = mean_val;
    }
}

torch::Tensor fused_conv_transpose_mul_gap_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    c10::optional<torch::Tensor> bias_opt, 
    float multiplier,
    int64_t stride, int64_t padding, int64_t output_padding) {

    // Assumptions based on problem: dilation=1, groups=1
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int C_out = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);

    // Calculate output dimensions
    // H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding
    int H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding;

    auto output = torch::zeros({N, C_out, 1, 1}, input.options());
    
    torch::Tensor bias;
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        bias_ptr = bias.data_ptr<float>();
    }

    int block_size = 256;
    dim3 grid(N, C_out);
    
    // Shared memory size: weights + partial sums
    size_t shared_mem_size = (C_in * kH * kW + block_size) * sizeof(float);

    fused_conv_transpose_mul_gap_kernel<<<grid, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        multiplier,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW, stride, padding
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_conv_transpose_mul_gap_cuda(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias, float multiplier, int64_t stride, int64_t padding, int64_t output_padding);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv_transpose_mul_gap_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        # Define parameters to mimic the original layer's weights and bias
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # Initialize weights and bias similar to ConvTranspose2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.multiplier = multiplier
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        self.fused_ops = fused_ops

    def forward(self, x):
        # The second global average pooling in the original model operates on a (N, C, 1, 1) tensor,
        # which is an identity operation. We skip it.
        return self.fused_ops.fused_conv_transpose_mul_gap_cuda(
            x, self.weight, self.bias, self.multiplier, self.stride, self.padding, self.output_padding
        )

import math # Ensure math is imported for init