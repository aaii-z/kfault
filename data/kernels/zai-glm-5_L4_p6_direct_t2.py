import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv3d + Softmax + MaxPool
fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>
#include <cmath>

// Kernel 1: Conv3d
// Grid: (batch, out_channels, spatial_blocks)
// Block: threads per block processing the spatial dimensions
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int out_depth, int out_height, int out_width) {

    int b = blockIdx.x;
    int oc = blockIdx.y;
    int spatial_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    int spatial_size = out_depth * out_height * out_width;
    if (spatial_idx >= spatial_size) return;

    int od = spatial_idx / (out_height * out_width);
    int rem = spatial_idx % (out_height * out_width);
    int oh = rem / out_width;
    int ow = rem % out_width;

    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int id = od * stride_d - padding_d + kd;
                    int ih = oh * stride_h - padding_h + kh;
                    int iw = ow * stride_w - padding_w + kw;

                    if (id >= 0 && id < in_depth && 
                        ih >= 0 && ih < in_height && 
                        iw >= 0 && iw < in_width) {
                        
                        int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                         ic * (in_depth * in_height * in_width) +
                                         id * (in_height * in_width) +
                                         ih * in_width + iw;
                                         
                        int weight_idx = oc * (in_channels * kernel_d * kernel_h * kernel_w) +
                                         ic * (kernel_d * kernel_h * kernel_w) +
                                         kd * (kernel_h * kernel_w) +
                                         kh * kernel_w + kw;
                                         
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[b * (out_channels * spatial_size) + oc * spatial_size + spatial_idx] = sum + bias[oc];
}

// Kernel 2: Softmax + MaxPool
// Grid: (batch, out_depth_pool, out_height_pool, out_width_pool)
// Block: out_channels (one thread per channel)
__global__ void softmax_pool_kernel(
    const float* __restrict__ conv_out,
    float* __restrict__ final_out,
    int batch_size, int out_channels,
    int conv_d, int conv_h, int conv_w,
    int pool_kd, int pool_kh, int pool_kw,
    int stride_pd, int stride_ph, int stride_pw,
    int out_d, int out_h, int out_w) {

    // Shared memory for the pooling window values
    // Size: pool_size * out_channels
    // Max pool size 2x2x2 = 8. Max channels 16. 8*16 = 256 floats.
    __shared__ float pool_window_vals[8 * 16]; 
    __shared__ float max_vals[8]; // Max value for softmax stability per spatial location
    __shared__ float sum_vals[8]; // Sum of exponentials per spatial location

    int b = blockIdx.x;
    int od_p = blockIdx.y;
    int oh_p = blockIdx.z / out_w;
    int ow_p = blockIdx.z % out_w;

    int tid = threadIdx.x; // Channel index
    
    // Load data into shared memory
    // Each thread handles one channel for all 8 spatial locations in the window
    // Assuming blockDim.x = out_channels
    if (tid < out_channels) {
        for (int kd = 0; kd < pool_kd; ++kd) {
            for (int kh = 0; kh < pool_kh; ++kh) {
                for (int kw = 0; kw < pool_kw; ++kw) {
                    int id = od_p * stride_pd + kd;
                    int ih = oh_p * stride_ph + kh;
                    int iw = ow_p * stride_pw + kw;
                    
                    int win_idx = kd * (pool_kh * pool_kw) + kh * pool_kw + kw;
                    
                    // Boundary check
                    float val = -INFINITY;
                    if (id < conv_d && ih < conv_h && iw < conv_w) {
                        int input_idx = b * (out_channels * conv_d * conv_h * conv_w) +
                                        tid * (conv_d * conv_h * conv_w) +
                                        id * (conv_h * conv_w) +
                                        ih * conv_w + iw;
                        val = conv_out[input_idx];
                    }
                    pool_window_vals[win_idx * out_channels + tid] = val;
                }
            }
        }
    }
    __syncthreads();

    // Compute Softmax
    // Thread local storage for the 8 locations
    float thread_vals[8];
    float thread_exps[8];
    
    int pool_size = pool_kd * pool_kh * pool_kw;
    
    // Load values
    for (int i = 0; i < pool_size; ++i) {
        thread_vals[i] = pool_window_vals[i * out_channels + tid];
    }
    
    // Compute max for each location (reduction over threads)
    if (tid < pool_size) max_vals[tid] = -INFINITY;
    __syncthreads();
    
    for (int i = 0; i < pool_size; ++i) {
        atomicMax((int*)&max_vals[i], __float_as_int(thread_vals[i]));
    }
    __syncthreads();
    
    // Compute exp(val - max)
    for (int i = 0; i < pool_size; ++i) {
        thread_exps[i] = expf(thread_vals[i] - max_vals[i]);
        pool_window_vals[i * out_channels + tid] = thread_exps[i]; // Store exp back to shared mem
    }
    
    // Compute sum of exps (reduction over threads)
    if (tid < pool_size) sum_vals[tid] = 0.0f;
    __syncthreads();
    
    for (int i = 0; i < pool_size; ++i) {
        atomicAdd(&sum_vals[i], thread_exps[i]);
    }
    __syncthreads();
    
    // Compute Softmax values and store back to thread_vals
    for (int i = 0; i < pool_size; ++i) {
        thread_vals[i] = thread_exps[i] / sum_vals[i];
    }
    
    // Now perform MaxPool
    // We need max over the 8 locations for each channel.
    float max_pool_val = -INFINITY;
    for (int i = 0; i < pool_size; ++i) {
        max_pool_val = fmaxf(max_pool_val, thread_vals[i]);
    }
    
    // Write output
    if (tid < out_channels) {
        int output_idx = b * (out_channels * out_d * out_h * out_w) +
                         tid * (out_d * out_h * out_w) +
                         od_p * (out_h * out_w) +
                         oh_p * out_w + ow_p;
        final_out[output_idx] = max_pool_val;
    }
}

// Wrapper function
torch::Tensor fused_ops_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int pool_kd, int pool_kh, int pool_kw) {
    
    // Calculate dimensions
    int conv_out_depth = (in_depth + 2 * padding_d - kernel_d) / stride_d + 1;
    int conv_out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int conv_out_width = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;
    
    // Apply pooling twice
    int pool1_out_depth = conv_out_depth / pool_kd;
    int pool1_out_height = conv_out_height / pool_kh;
    int pool1_out_width = conv_out_width / pool_kw;
    
    int pool_out_depth = pool1_out_depth / pool_kd;
    int pool_out_height = pool1_out_height / pool_kh;
    int pool_out_width = pool1_out_width / pool_kw;
    
    // Allocate intermediate tensor for Conv3d output
    auto conv_out_options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto conv_out = torch::empty({batch_size, out_channels, conv_out_depth, conv_out_height, conv_out_width}, conv_out_options);
    
    // Allocate final output tensor
    auto final_out = torch::empty({batch_size, out_channels, pool_out_depth, pool_out_height, pool_out_width}, conv_out_options);
    
    // Launch Conv3d kernel
    int conv_spatial_size = conv_out_depth * conv_out_height * conv_out_width;
    int threads_per_block = 256;
    int spatial_blocks = (conv_spatial_size + threads_per_block - 1) / threads_per_block;
    
    dim3 conv_grid(batch_size, out_channels, spatial_blocks);
    
    conv3d_kernel<<<conv_grid, threads_per_block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        conv_out.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        conv_out_depth, conv_out_height, conv_out_width);
    
    // Launch Softmax + MaxPool kernel
    // Note: Assuming pool stride is equal to kernel size (standard pooling)
    int pool_stride_d = pool_kd;
    int pool_stride_h = pool_kh;
    int pool_stride_w = pool_kw;
    
    dim3 pool_grid(batch_size, pool_out_depth, pool_out_height * pool_out_width);
    int pool_block_size = out_channels; 
    
    softmax_pool_kernel<<<pool_grid, pool_block_size>>>(
        conv_out.data_ptr<float>(), final_out.data_ptr<float>(),
        batch_size, out_channels,
        conv_out_depth, conv_out_height, conv_out_width,
        pool_kd, pool_kh, pool_kw,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_out_depth, pool_out_height, pool_out_width);
        
    return final_out;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int batch_size, int in_channels, int out_channels, int in_depth, int in_height, int in_width, int kernel_d, int kernel_h, int kernel_w, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, int pool_kd, int pool_kh, int pool_kw);"

# Compile the inline CUDA code
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
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        # Define parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        
        # We need to handle kernel_size being an int or a tuple
        if isinstance(kernel_size, int):
            self.kernel_d = kernel_size
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_d, self.kernel_h, self.kernel_w = kernel_size
            
        if isinstance(pool_kernel_size, int):
            self.pool_kd = pool_kernel_size
            self.pool_kh = pool_kernel_size
            self.pool_kw = pool_kernel_size
        else:
            self.pool_kd, self.pool_kh, self.pool_kw = pool_kernel_size

        # Stride for Conv3d is 1 by default in PyTorch
        self.stride = 1
        self.stride_d = 1
        self.stride_h = 1
        self.stride_w = 1
        
        # Padding for Conv3d is 0 by default in PyTorch
        self.padding = 0
        self.padding_d = 0
        self.padding_h = 0
        self.padding_w = 0

        # Create a Conv3d layer just to initialize weights
        # We will extract the weights and use them in our custom kernel
        self.conv_ref = nn.Conv3d(in_channels, out_channels, kernel_size)
        
        # Load the custom CUDA operators
        self.fused_ops = fused_ops

    def forward(self, x):
        # Extract weights from the reference Conv3d layer
        weight = self.conv_ref.weight
        bias = self.conv_ref.bias
        
        # Get input dimensions
        batch_size, in_channels, in_depth, in_height, in_width = x.shape
        
        # Call the custom CUDA kernel
        return self.fused_ops.fused_ops_cuda(
            x, weight, bias,
            batch_size, in_channels, self.out_channels,
            in_depth, in_height, in_width,
            self.kernel_d, self.kernel_h, self.kernel_w,
            self.stride_d, self.stride_h, self.stride_w,
            self.padding_d, self.padding_h, self.padding_w,
            self.pool_kd, self.pool_kh, self.pool_kw
        )