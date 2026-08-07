import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for the entire forward pass
fused_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void fused_forward_kernel(
    const float* input, 
    const float* weight, 
    const float* bias_conv, 
    const float* bias_add, 
    float* output, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int input_height, 
    int input_width, 
    int kernel_size, 
    int output_height, 
    int output_width
) {
    // Calculate output position
    int n = blockIdx.z; // batch index
    int oc = blockIdx.y; // output channel
    int oh = blockIdx.x * blockDim.y + threadIdx.y; // output height
    int ow = threadIdx.x; // output width
    
    if (n >= batch_size || oc >= out_channels || oh >= output_height || ow >= output_width) {
        return;
    }
    
    extern __shared__ float shared_mem[];
    float* channel_sums = shared_mem;
    
    // Initialize shared memory for reduction
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        channel_sums[oc] = 0.0f;
    }
    __syncthreads();
    
    // Perform transposed convolution
    float conv_result = 0.0f;
    
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh - kh + kernel_size / 2;
                int iw = ow - kw + kernel_size / 2;
                
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = n * in_channels * input_height * input_width + 
                                  ic * input_height * input_width + 
                                  ih * input_width + iw;
                    int weight_idx = oc * in_channels * kernel_size * kernel_size + 
                                   ic * kernel_size * kernel_size + 
                                   kh * kernel_size + kw;
                    
                    conv_result += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add convolution bias
    conv_result += bias_conv[oc];
    
    // Global average pooling (implicit in the reduction)
    atomicAdd(&channel_sums[oc], conv_result / (output_height * output_width));
    __syncthreads();
    
    // Add the second bias and compute log-sum-exp
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        float pooled_value = channel_sums[oc] + bias_add[oc];
        float exp_value = expf(pooled_value);
        
        // Use atomic operations for log-sum-exp reduction across channels
        atomicAdd(&shared_mem[out_channels], exp_value); // sum of exps
        atomicMax(&shared_mem[out_channels + 1], pooled_value); // max value
    }
    __syncthreads();
    
    // Final computation (only one thread per batch)
    if (oc == 0 && oh == 0 && ow == 0) {
        float max_val = shared_mem[out_channels + 1];
        float sum_exp = shared_mem[out_channels];
        float log_sum_exp = logf(sum_exp) + max_val;
        
        // Multiply by 10.0 and store final result
        output[n] = log_sum_exp * 10.0f;
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias_conv, 
    torch::Tensor bias_add
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_size = sqrt(weight.size(3));
    
    int output_height = input_height;
    int output_width = input_width;
    
    auto output = torch::zeros({batch_size}, input.options());
    
    dim3 block_dim(32, 8);
    dim3 grid_dim(
        (output_height + block_dim.y - 1) / block_dim.y,
        out_channels,
        batch_size
    );
    
    int shared_mem_size = (out_channels + 2) * sizeof(float);
    
    fused_forward_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_conv.data_ptr<float>(),
        bias_add.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        output_height,
        output_width
    );
    
    return output;
}
"""

fused_forward_cpp_source = "torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias_conv, torch::Tensor bias_add);"

# Compile the inline CUDA code
fused_forward = load_inline(
    name="fused_forward",
    cpp_sources=fused_forward_cpp_source,
    cuda_sources=fused_forward_source,
    functions=["fused_forward_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias_conv = nn.Parameter(torch.randn(out_channels))
        self.bias_add = nn.Parameter(torch.randn(bias_shape[0], 1, 1))
        
        self.fused_forward = fused_forward

    def forward(self, x):
        return self.fused_forward.fused_forward_cuda(x, self.weight, self.bias_conv, self.bias_add)