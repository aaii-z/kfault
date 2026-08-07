import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose2d + GELU + GroupNorm
fused_conv_gelu_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) {\\
    cudaError_t err = call;\\
    if (err != cudaSuccess) {\\
        fprintf(stderr, "CUDA error at %s:%d: %s\\\\n", __FILE__, __LINE__, cudaGetErrorString(err));\\
        exit(EXIT_FAILURE);\\
    }\\
}

constexpr int TILE_DIM = 16;
constexpr int BLOCK_ROWS = 8;

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void fused_conv_transpose_gelu_norm_kernel(
    const float* input, const float* weight, const float* bias,
    float* intermediate,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_size, int stride) {
    
    int output_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int output_y = blockIdx.y * TILE_DIM + threadIdx.y;
    int c = blockIdx.z % out_channels;
    int batch = blockIdx.z / out_channels;
    
    if (output_x >= output_width || output_y >= output_height || batch >= batch_size || c >= out_channels) return;
    
    float sum = 0.0f;
    
    for (int k = 0; k < in_channels; k++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int input_x = output_x - kx;
                int input_y = output_y - ky;
                
                if (input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height) {
                    if (input_x % stride == 0 && input_y % stride == 0) {
                        int input_idx = ((batch * in_channels + k) * input_height + (input_y / stride)) * input_width + (input_x / stride);
                        int weight_idx = ((k * out_channels + c) * kernel_size + ky) * kernel_size + kx;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias and apply GELU
    if (bias) {
        sum += bias[c];
    }
    sum = gelu_approx(sum);
    
    // Store intermediate result
    int output_idx = ((batch * out_channels + c) * output_height + output_y) * output_width + output_x;
    intermediate[output_idx] = sum;
}

__global__ void compute_group_stats_kernel(
    const float* input, float* mean, float* var,
    int batch_size, int out_channels, int output_height, int output_width,
    int num_groups, float eps) {
    
    int group_size = out_channels / num_groups;
    int group_id = blockIdx.x % num_groups;
    int batch = blockIdx.x / num_groups;
    
    if (batch >= batch_size) return;
    
    int channel_start = group_id * group_size;
    int channel_end = channel_start + group_size;
    
    float group_sum = 0.0f;
    float group_sq_sum = 0.0f;
    int num_elements = group_size * output_height * output_width;
    
    for (int c = channel_start; c < channel_end; c++) {
        for (int y = 0; y < output_height; y++) {
            for (int x = 0; x < output_width; x++) {
                int idx = ((batch * out_channels + c) * output_height + y) * output_width + x;
                float val = input[idx];
                group_sum += val;
                group_sq_sum += val * val;
            }
        }
    }
    
    float group_mean = group_sum / num_elements;
    float group_var = (group_sq_sum / num_elements) - (group_mean * group_mean) + eps;
    
    mean[blockIdx.x] = group_mean;
    var[blockIdx.x] = group_var;
}

__global__ void apply_group_norm_kernel(
    const float* input, const float* mean, const float* var,
    const float* gamma, const float* beta, float* output,
    int batch_size, int out_channels, int output_height, int output_width,
    int num_groups) {
    
    int group_size = out_channels / num_groups;
    int group_id = blockIdx.z % num_groups;
    int batch = blockIdx.z / num_groups;
    
    int output_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int output_y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if (output_x >= output_width || output_y >= output_height || batch >= batch_size) return;
    
    int stats_idx = batch * num_groups + group_id;
    float group_mean = mean[stats_idx];
    float group_std = sqrtf(var[stats_idx]);
    
    for (int c = group_id * group_size; c < (group_id + 1) * group_size; c++) {
        int input_idx = ((batch * out_channels + c) * output_height + output_y) * output_width + output_x;
        float val = input[input_idx];
        float normalized = (val - group_mean) / group_std;
        
        if (gamma && beta) {
            normalized = normalized * gamma[c] + beta[c];
        }
        
        output[input_idx] = normalized;
    }
}

torch::Tensor fused_conv_transpose_gelu_norm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int kernel_size, int stride, int num_groups, float eps=1e-5) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto out_channels = weight.size(1);
    
    int output_height = (input_height - 1) * stride + kernel_size;
    int output_width = (input_width - 1) * stride + kernel_size;
    
    auto intermediate = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());
    auto output = torch::empty_like(intermediate);
    
    // Step 1: ConvTranspose + GELU
    dim3 block_dim1(TILE_DIM, TILE_DIM);
    dim3 grid_dim1(
        (output_width + TILE_DIM - 1) / TILE_DIM,
        (output_height + TILE_DIM - 1) / TILE_DIM,
        batch_size * out_channels
    );
    
    fused_conv_transpose_gelu_norm_kernel<<<grid_dim1, block_dim1>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        intermediate.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width, output_height, output_width,
        kernel_size, stride);
    
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: Compute group statistics
    auto mean = torch::empty({batch_size * num_groups}, input.options());
    auto var = torch::empty({batch_size * num_groups}, input.options());
    
    dim3 grid_dim2(batch_size * num_groups);
    compute_group_stats_kernel<<<grid_dim2, 1>>>(
        intermediate.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        batch_size, out_channels, output_height, output_width, num_groups, eps);
    
    CUDA_CHECK(cudaGetLastError());
    
    // Step 3: Apply group normalization
    dim3 block_dim3(TILE_DIM, TILE_DIM);
    dim3 grid_dim3(
        (output_width + TILE_DIM - 1) / TILE_DIM,
        (output_height + TILE_DIM - 1) / TILE_DIM,
        batch_size * num_groups
    );
    
    apply_group_norm_kernel<<<grid_dim3, block_dim3>>>(
        intermediate.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, out_channels, output_height, output_width, num_groups);
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

fused_conv_gelu_norm_cpp_source = """
torch::Tensor fused_conv_transpose_gelu_norm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int kernel_size, int stride, int num_groups, float eps=1e-5);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv_gelu_norm",
    cpp_sources=fused_conv_gelu_norm_cpp_source,
    cuda_sources=fused_conv_gelu_norm_source,
    functions=["fused_conv_transpose_gelu_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.num_groups = num_groups
        
        # Initialize weights and bias for transposed convolution
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize scale and shift parameters for group normalization
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        torch.nn.init.zeros_(self.bias)
        torch.nn.init.ones_(self.gamma)
        torch.nn.init.zeros_(self.beta)

    def forward(self, x):
        return fused_ops.fused_conv_transpose_gelu_norm_cuda(
            x, self.weight, self.bias, self.gamma, self.beta,
            self.kernel_size, self.stride, self.num_groups
        )