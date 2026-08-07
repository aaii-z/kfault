import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3d + Scale + BatchNorm + GlobalAvgPool
fused_conv_transpose_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cub/cub.cuh>

#define CUDA_CHECK(call) {\\
    cudaError_t err = call;\\
    if (err != cudaSuccess) {\\
        fprintf(stderr, "CUDA error at %s:%d: %s\\\\n", __FILE__, __LINE__, cudaGetErrorString(err));\\
        exit(EXIT_FAILURE);\\
    }\\
}

template<typename T>
__global__ void fused_conv_transpose_scale_bn_pool_kernel(
    const T* input, const T* weight, const T* bias, 
    const T* running_mean, const T* running_var, 
    T* output, T scale_factor, T eps,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int output_depth, int output_height, int output_width) {
    
    // Calculate output spatial dimensions
    const int stride = 1;
    const int padding = (kernel_depth - 1) / 2;
    const int output_padding = 0;
    
    const int output_size = output_depth * output_height * output_width;
    const int total_output_elements = batch_size * out_channels * output_size;
    
    // Global index for this thread
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_output_elements) return;
    
    // Decompose global index
    const int n = idx / (out_channels * output_size);
    const int c = (idx % (out_channels * output_size)) / output_size;
    const int spatial_idx = idx % output_size;
    const int d = spatial_idx / (output_height * output_width);
    const int h = (spatial_idx % (output_height * output_width)) / output_width;
    const int w = spatial_idx % output_width;
    
    T result = 0.0;
    
    // Perform transposed convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_depth; ++kd) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int input_d = d - kd + padding;
                    int input_h = h - kh + padding;
                    int input_w = w - kw + padding;
                    
                    if (input_d >= 0 && input_d < input_depth &&
                        input_h >= 0 && input_h < input_height &&
                        input_w >= 0 && input_w < input_width) {
                        
                        int input_idx = ((n * in_channels + ic) * input_depth + input_d) * input_height * input_width +
                                       input_h * input_width + input_w;
                        
                        int weight_idx = ((ic * out_channels + c) * kernel_depth + kd) * kernel_height * kernel_width +
                                        kh * kernel_width + kw;
                        
                        result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias if present
    if (bias != nullptr) {
        result += bias[c];
    }
    
    // Apply scale factor
    result *= scale_factor;
    
    // Apply batch normalization
    if (running_mean != nullptr && running_var != nullptr) {
        T mean = running_mean[c];
        T var = running_var[c];
        result = (result - mean) / sqrt(var + eps);
    }
    
    // Store intermediate result for pooling
    output[idx] = result;
}

template<typename T>
__global__ void global_avg_pool_kernel(
    const T* input, T* output,
    int batch_size, int out_channels,
    int output_depth, int output_height, int output_width) {
    
    const int spatial_size = output_depth * output_height * output_width;
    const int total_elements = batch_size * out_channels;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    const int n = idx / out_channels;
    const int c = idx % out_channels;
    
    T sum = 0.0;
    const T* channel_data = input + (n * out_channels + c) * spatial_size;
    
    for (int i = 0; i < spatial_size; ++i) {
        sum += channel_data[i];
    }
    
    output[n * out_channels + c] = sum / static_cast<T>(spatial_size);
}

torch::Tensor fused_conv_transpose_scale_bn_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    float scale_factor, float eps) {
    
    // Get input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    // Get weight dimensions
    int out_channels = weight.size(1);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);
    
    // Calculate output dimensions for transposed convolution
    int output_depth = input_depth;
    int output_height = input_height;
    int output_width = input_width;
    
    const int output_size = output_depth * output_height * output_width;
    const int total_output_elements = batch_size * out_channels * output_size;
    
    // Allocate intermediate output
    auto intermediate_output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, 
                                           input.options());
    
    // Launch kernel for fused conv transpose + scale + BN
    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;
    
    fused_conv_transpose_scale_bn_pool_kernel<float><<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        running_mean.defined() ? running_mean.data_ptr<float>() : nullptr,
        running_var.defined() ? running_var.data_ptr<float>() : nullptr,
        intermediate_output.data_ptr<float>(),
        scale_factor, eps,
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_depth, kernel_height, kernel_width,
        output_depth, output_height, output_width);
    
    CUDA_CHECK(cudaGetLastError());
    
    // Perform global average pooling
    auto final_output = torch::empty({batch_size, out_channels, 1, 1, 1}, input.options());
    const int pool_elements = batch_size * out_channels;
    const int pool_blocks = (pool_elements + block_size - 1) / block_size;
    
    global_avg_pool_kernel<float><<<pool_blocks, block_size>>>(
        intermediate_output.data_ptr<float>(),
        final_output.data_ptr<float>(),
        batch_size, out_channels,
        output_depth, output_height, output_width);
    
    CUDA_CHECK(cudaGetLastError());
    
    return final_output;
}
"""

fused_conv_transpose_ops_cpp_source = """
torch::Tensor fused_conv_transpose_scale_bn_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    float scale_factor, float eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv_transpose_ops",
    cpp_sources=fused_conv_transpose_ops_cpp_source,
    cuda_sources=fused_conv_transpose_ops_source,
    functions=["fused_conv_transpose_scale_bn_pool_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcub"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.scale_factor = scale_factor
        self.eps = eps
        
        # Initialize weights and parameters manually
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # BatchNorm parameters
        self.running_mean = nn.Parameter(torch.empty(out_channels), requires_grad=False)
        self.running_var = nn.Parameter(torch.empty(out_channels), requires_grad=False)
        self.num_batches_tracked = nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)
        
        # Initialize parameters
        self.reset_parameters()
        
        # Load custom CUDA operations
        self.fused_ops = fused_ops

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.zeros_(self.running_mean)
        nn.init.ones_(self.running_var)
        self.num_batches_tracked.zero_()

    def forward(self, x):
        return self.fused_ops.fused_conv_transpose_scale_bn_pool_cuda(
            x, self.weight, self.bias, 
            self.running_mean, self.running_var,
            self.scale_factor, self.eps
        )