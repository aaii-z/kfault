import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused transposed convolution, scalar multiplication, and global average pooling
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Define constant memory for weights (assuming kernel_size <= 5 for this example, adjust if needed)
// For simplicity and generality in this specific request, we will pass weights directly.
// A more optimized version might use constant memory or shared memory tiling.

// Kernel for fused ConvTranspose2d + Mul + Global Average Pooling
// This kernel computes one output pixel (h, w) for a given batch and channel.
// It then accumulates the sum for the global average pooling.
// To handle the two global average poolings, we can compute the first one, and the second one is trivial (mean of a single value).
// Actually, the second mean is mean(x, dim=[2,3]) on a tensor of shape (N, C, 1, 1), which is identity.
// So we just need to compute the first global average pooling.

__global__ void fused_conv_transpose_mul_gap_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias, // Can be null
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

    // We use a 3D grid: (batch * out_channel) as the major dimension, and (out_h, out_w) as the other two.
    // This allows each block to handle a spatial location for a specific (N, C_out).
    // Or better: each thread handles one output pixel.
    
    int n = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x / out_w;
    int ow = blockIdx.x % out_w;
    
    // threadIdx.x can be used for reduction if we were doing reduction in shared memory,
    // but here we calculate the output value directly.
    // To optimize, we need to consider memory access patterns.
    
    // Let's map threads more directly.
    // Grid: (batch_size, out_channels, out_h)
    // Block: (out_w)
    // This ensures coalesced writes if out_w is the innermost dimension.
    
    // Correct mapping for coalescing:
    // blockIdx.z = n
    // blockIdx.y = oc
    // blockIdx.x = oh * out_w + ow (flattened spatial dim)
    // This is what was roughly written above.
    
    // Wait, the output of GAP is (N, C, 1, 1).
    // So we need to sum over (oh, ow).
    // We can use atomicAdd to accumulate the sum into the output tensor (N, C, 1, 1).
    // Or, we can have each block compute one (N, C) output and use shared memory reduction.
    
    // Let's try the atomicAdd approach first for simplicity, or better:
    // Map (N, C) to blocks. Each block computes the sum over the spatial dimensions.
    // Block size = 256 or 512.
    // Each thread iterates over a subset of the output spatial pixels.
    
    // Let's stick to the thread-per-output-pixel for the ConvTranspose part,
    // and then use atomics or a second kernel for reduction?
    // No, fusion is key.
    
    // Optimized approach:
    // 1. Each thread computes one pixel of the ConvTranspose output.
    // 2. It multiplies by scalar.
    // 3. It atomically adds to the global accumulator for that (N, C).
    // This is efficient enough if the output spatial size is large.
    // Output spatial size: ~256x256. 65536 threads per (N,C). Atomic contention might be high.
    
    // Better approach: Block-level reduction.
    // Grid: (batch_size, out_channels).
    // Block: covers the spatial dimensions.
    // Each thread sums a portion of the spatial dimensions, then reduces in shared memory.
    
    // Let's implement the Block-level reduction version.
    // Grid: (batch_size, out_channels).
    // Block: 1D, covering the spatial pixels.
    
    int n = blockIdx.x; // Batch index
    int oc = blockIdx.y; // Output channel index
    
    // Shared memory for reduction
    __shared__ float s_sum[256]; // Assuming block_size <= 256
    __shared__ float s_count; // To normalize
    
    int tid = threadIdx.x;
    s_sum[tid] = 0.0f;
    
    // Total number of output pixels for this (n, oc)
    int spatial_size = out_h * out_w;
    
    // Strided loop over output pixels
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int oh = i / out_w;
        int ow = i % out_w;
        
        float val = 0.0f;
        
        // Compute ConvTranspose2d for this specific output pixel (oh, ow)
        // Formula: input[...] * weight[...] + bias
        // ConvTranspose logic:
        // Iterate over input channels and kernel
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    // Calculate input position
                    // ih = oh * stride - pad + kh  (Standard Conv logic)
                    // For ConvTranspose:
                    // We need to find which input pixels contribute to this output pixel.
                    // This is the "backward" pass of convolution.
                    // Output (oh, ow) depends on Input (ih, iw) if:
                    // (oh + pad - kh) % stride == 0
                    // ih = (oh + pad - kh) / stride
                    // Wait, the formula for ConvTranspose output coordinate mapping is:
                    // ih = (oh - kh + pad) / stride  (integer division)
                    // We need to iterate over kernel and check if the corresponding input pixel is valid.
                    
                    // Let's use the standard ConvTranspose loop structure:
                    // Iterate over kernel positions.
                    // Calculate input coordinates.
                    int ih = oh - kh + pad_h; // This is not quite right for stride.
                    
                    // Correct logic for ConvTranspose:
                    // The output pixel (oh, ow) is connected to input pixels at:
                    // ih = (oh - kh) / stride  (if valid)
                    // Actually, it's easier to iterate over the kernel and check validity.
                    
                    // Let's use the relation:
                    // oh = ih * stride + kh - pad
                    // We want to find (ic, ih, iw) that contribute to (oc, oh, ow).
                    // This is equivalent to a convolution of the input with the flipped kernel.
                    // But iterating over the kernel is standard.
                    
                    // For a given output pixel (oh, ow), we sum over input channels and kernel:
                    // out[n, oc, oh, ow] = bias[oc] + sum_ic sum_kh sum_kw (input[n, ic, ih, iw] * weight[oc, ic, kh, kw])
                    // where (ih, iw) is determined by:
                    // ih = (oh + pad_h - kh) / stride_h
                    // iw = (ow + pad_w - kw) / stride_w
                    // And we must check:
                    // 1. (oh + pad_h - kh) % stride_h == 0
                    // 2. 0 <= ih < in_h
                    
                    int ih_candidate = oh + pad_h - kh;
                    int iw_candidate = ow + pad_w - kw;
                    
                    if (ih_candidate % stride_h == 0 && iw_candidate % stride_w == 0) {
                        int ih = ih_candidate / stride_h;
                        int iw = iw_candidate / stride_w;
                        
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            // Input index
                            // Input tensor is (N, C_in, H_in, W_in)
                            // We need to access input[n, ic, ih, iw]
                            int input_idx = n * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
                            
                            // Weight index
                            // Weight tensor is (C_out, C_in, kH, kW)
                            int weight_idx = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + kh * kernel_w + kw;
                            
                            val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias if present
        if (bias != nullptr) {
            val += bias[oc];
        }
        
        // Multiply by scalar
        val *= multiplier;
        
        // Accumulate for GAP
        s_sum[tid] += val;
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    // Assuming blockDim.x is power of 2
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        // Output is (N, C_out, 1, 1)
        // The second mean is identity on (N, C, 1, 1), so we just write the mean.
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

    // Shapes
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    // Calculate output dimensions
    // H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;
    
    // Output tensor for the final result (N, C, 1, 1)
    auto output = torch::zeros({batch_size, out_channels, 1, 1}, input.options());
    
    // Bias handling
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        bias_ptr = bias_opt.value().data_ptr<float>();
    }
    
    // Launch kernel
    // Grid: (batch_size, out_channels)
    // Block: 256 threads (tunable)
    
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
        # We still define the ConvTranspose2d to hold the weights and bias
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        self.fused_op = fused_ops

    def forward(self, x):
        # Extract parameters
        stride = self.conv_transpose.stride
        padding = self.conv_transpose.padding
        output_padding = self.conv_transpose.output_padding
        
        # Call the custom CUDA kernel
        # Note: We pass the weight and bias from the conv_transpose layer
        return self.fused_op.fused_op_cuda(
            x, 
            self.conv_transpose.weight, 
            self.conv_transpose.bias, 
            stride[0], stride[1], 
            padding[0], padding[1], 
            output_padding[0], output_padding[1], 
            self.multiplier
        )