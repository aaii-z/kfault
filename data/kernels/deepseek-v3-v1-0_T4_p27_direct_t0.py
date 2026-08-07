import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the fused Conv3D + HardSwish + GroupNorm + MeanPool CUDA kernel
fused_conv_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define CUDA_CHECK(call) {\\
    cudaError_t err = call;\\
    if (err != cudaSuccess) {\\
        fprintf(stderr, "CUDA error at %s:%d: %s\\\\n", __FILE__, __LINE__, cudaGetErrorString(err));\\
        exit(EXIT_FAILURE);\\
    }\\
}

__device__ float hardswish_activation(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_conv3d_hardswish_groupnorm_mean_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* group_mean, float* group_var,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width, int kernel_size,
    int num_groups, int group_size, int spatial_size) {
    
    extern __shared__ float shared_mem[];
    
    int group_id = blockIdx.x;
    int group_channel = threadIdx.x;
    int spatial_idx = threadIdx.y;
    
    if (group_channel >= group_size || spatial_idx >= spatial_size) return;
    
    int out_c = group_id * group_size + group_channel;
    if (out_c >= out_channels) return;
    
    // Compute convolution and hardswish for this output channel and spatial position
    float conv_result = 0.0f;
    if (bias) {
        conv_result = bias[out_c];
    }
    
    for (int in_c = 0; in_c < in_channels; in_c++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int input_d = spatial_idx / (height * width);
                    int input_h = (spatial_idx % (height * width)) / width;
                    int input_w = spatial_idx % width;
                    
                    int d = input_d + kd - kernel_size / 2;
                    int h = input_h + kh - kernel_size / 2;
                    int w = input_w + kw - kernel_size / 2;
                    
                    if (d >= 0 && d < depth && h >= 0 && h < height && w >= 0 && w < width) {
                        int input_idx = in_c * depth * height * width + d * height * width + h * width + w;
                        int weight_idx = out_c * in_channels * kernel_size * kernel_size * kernel_size + 
                                       in_c * kernel_size * kernel_size * kernel_size + 
                                       kd * kernel_size * kernel_size + kh * kernel_size + kw;
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Apply HardSwish activation
    float activated = hardswish_activation(conv_result);
    
    // Store in shared memory for group norm computation
    int shared_idx = group_channel * spatial_size + spatial_idx;
    shared_mem[shared_idx] = activated;
    __syncthreads();
    
    // Compute group mean (within the group)
    if (spatial_idx == 0) {
        float sum = 0.0f;
        for (int s = 0; s < spatial_size; s++) {
            sum += shared_mem[group_channel * spatial_size + s];
        }
        group_mean[group_id * group_size + group_channel] = sum / spatial_size;
    }
    __syncthreads();
    
    // Compute group variance (within the group)
    if (spatial_idx == 0) {
        float mean_val = group_mean[group_id * group_size + group_channel];
        float var_sum = 0.0f;
        for (int s = 0; s < spatial_size; s++) {
            float diff = shared_mem[group_channel * spatial_size + s] - mean_val;
            var_sum += diff * diff;
        }
        group_var[group_id * group_size + group_channel] = var_sum / spatial_size;
    }
    __syncthreads();
    
    // Apply group normalization and store final result
    float mean_val = group_mean[group_id * group_size + group_channel];
    float var_val = group_var[group_id * group_size + group_channel];
    float normalized = (activated - mean_val) / sqrtf(var_val + 1e-5f);
    
    // Atomic add to output (mean pooling)
    atomicAdd(&output[group_id * group_size + group_channel], normalized / spatial_size);
}

torch::Tensor fused_conv3d_hardswish_groupnorm_mean_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int num_groups) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto out_channels = weight.size(0);
    
    int spatial_size = depth * height * width;
    int group_size = out_channels / num_groups;
    
    auto output = torch::zeros({batch_size, out_channels}, input.options());
    auto group_mean = torch::zeros({batch_size, out_channels}, input.options());
    auto group_var = torch::zeros({batch_size, out_channels}, input.options());
    
    dim3 block_dim(group_size, min(1024 / group_size, spatial_size));
    dim3 grid_dim(num_groups);
    
    size_t shared_mem_size = group_size * spatial_size * sizeof(float);
    
    for (int b = 0; b < batch_size; b++) {
        fused_conv3d_hardswish_groupnorm_mean_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
            input[b].data_ptr<float>(), weight.data_ptr<float>(), 
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output[b].data_ptr<float>(), group_mean[b].data_ptr<float>(), 
            group_var[b].data_ptr<float>(),
            batch_size, in_channels, out_channels, depth, height, width, 
            kernel_size, num_groups, group_size, spatial_size);
    }
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

fused_conv_ops_cpp_source = """
torch::Tensor fused_conv3d_hardswish_groupnorm_mean_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int num_groups);
"""

# Compile the inline CUDA code
fused_conv_ops = load_inline(
    name="fused_conv_ops",
    cpp_sources=fused_conv_ops_cpp_source,
    cuda_sources=fused_conv_ops_source,
    functions=["fused_conv3d_hardswish_groupnorm_mean_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_conv_ops

    def forward(self, x):
        return self.fused_op.fused_conv3d_hardswish_groupnorm_mean_cuda(
            x, self.weight, self.bias, self.kernel_size, self.num_groups)