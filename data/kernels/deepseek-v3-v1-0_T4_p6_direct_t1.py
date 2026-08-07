import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Conv3D + Softmax + MaxPool3D kernel
fused_conv_softmax_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

#define CUDA_CHECK(call) {\\
    cudaError_t err = call;\\
    if (err != cudaSuccess) {\\
        fprintf(stderr, "CUDA error at %s:%d: %s\\\\n", __FILE__, __LINE__, cudaGetErrorString(err));\\
        exit(EXIT_FAILURE);\\
    }\\
}

__global__ void fused_conv_softmax_pool_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_size, int pool_kernel_size,
    int output_depth, int output_height, int output_width,
    int pooled_depth, int pooled_height, int pooled_width) {
    
    int d_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.z * blockDim.z + threadIdx.z;
    int batch = blockIdx.w / out_channels;
    int channel = blockIdx.w % out_channels;
    
    if (d_out >= output_depth || h_out >= output_height || w_out >= output_width || batch >= batch_size || channel >= out_channels) return;
    
    // Convolution operation
    float conv_result = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int d_in = d_out + kd;
                    int h_in = h_out + kh;
                    int w_in = w_out + kw;
                    
                    if (d_in < input_depth && h_in < input_height && w_in < input_width) {
                        int input_idx = ((batch * in_channels + c_in) * input_depth + d_in) * input_height * input_width + 
                                      h_in * input_width + w_in;
                        int weight_idx = ((channel * in_channels + c_in) * kernel_size + kd) * kernel_size * kernel_size + 
                                       kh * kernel_size + kw;
                        
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    conv_result += bias[channel];
    
    // Store convolution result for softmax
    __shared__ float shared_conv[256];
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    shared_conv[tid] = conv_result;
    __syncthreads();
    
    // Find max for softmax stability
    float max_val = -FLT_MAX;
    for (int i = 0; i < blockDim.x * blockDim.y * blockDim.z; i++) {
        if (shared_conv[i] > max_val) {
            max_val = shared_conv[i];
        }
    }
    
    // Compute softmax
    float exp_sum = 0.0f;
    for (int i = 0; i < blockDim.x * blockDim.y * blockDim.z; i++) {
        exp_sum += expf(shared_conv[i] - max_val);
    }
    
    float softmax_result = expf(conv_result - max_val) / exp_sum;
    
    // Max pooling
    int d_pool = d_out / pool_kernel_size;
    int h_pool = h_out / pool_kernel_size;
    int w_pool = w_out / pool_kernel_size;
    
    if (d_pool < pooled_depth && h_pool < pooled_height && w_pool < pooled_width) {
        int output_idx = ((batch * out_channels + channel) * pooled_depth + d_pool) * pooled_height * pooled_width + 
                       h_pool * pooled_width + w_pool;
        
        atomicMax((unsigned int*)&output[output_idx], __float_as_uint(softmax_result));
    }
}

torch::Tensor fused_conv_softmax_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int pool_kernel_size) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);
    auto out_channels = weight.size(0);
    
    // Calculate output dimensions
    int output_depth = input_depth - kernel_size + 1;
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    
    int pooled_depth = (output_depth + pool_kernel_size - 1) / pool_kernel_size;
    int pooled_height = (output_height + pool_kernel_size - 1) / pool_kernel_size;
    int pooled_width = (output_width + pool_kernel_size - 1) / pool_kernel_size;
    
    // Create output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto output = torch::zeros({batch_size, out_channels, pooled_depth, pooled_height, pooled_width}, options);
    
    // Configure kernel launch parameters
    dim3 block_size(4, 4, 4);
    dim3 grid_size(
        (output_depth + block_size.x - 1) / block_size.x,
        (output_height + block_size.y - 1) / block_size.y,
        (output_width + block_size.z - 1) / block_size.z
    );
    grid_size.w = batch_size * out_channels;
    
    // Launch kernel
    fused_conv_softmax_pool_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_size, pool_kernel_size,
        output_depth, output_height, output_width,
        pooled_depth, pooled_height, pooled_width);
    
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}
"""

fused_conv_softmax_pool_cpp_source = "torch::Tensor fused_conv_softmax_pool_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, int pool_kernel_size);"

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_conv_softmax_pool",
    cpp_sources=fused_conv_softmax_pool_cpp_source,
    cuda_sources=fused_conv_softmax_pool_source,
    functions=["fused_conv_softmax_pool_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_op = fused_operator

    def forward(self, x):
        return self.fused_op.fused_conv_softmax_pool_cuda(
            x, self.weight, self.bias, self.kernel_size, self.pool_kernel_size
        )