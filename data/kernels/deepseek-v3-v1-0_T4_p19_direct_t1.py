import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose2d + GELU + GroupNorm
fused_conv_gelu_norm_source = """
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

constexpr int TILE_SIZE = 16;
constexpr int BLOCK_SIZE = 256;

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
    float* output, float* mean, float* var,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_size, int stride, int groups, int num_groups,
    float eps) {
    
    extern __shared__ float shared_mem[];
    
    int group_size = out_channels / num_groups;
    int group_id = blockIdx.z % num_groups;
    int channel_start = group_id * group_size;
    int channel_end = channel_start + group_size;
    
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (output_idx >= output_width || output_idy >= output_height) return;
    
    int batch = blockIdx.z / num_groups;
    if (batch >= batch_size) return;
    
    float* group_shared = shared_mem + threadIdx.y * blockDim.x + threadIdx.x;
    
    // Compute convolution
    float sum = 0.0f;
    int input_start_x = output_idx - kernel_size + 1;
    int input_start_y = output_idy - kernel_size + 1;
    
    for (int k = 0; k < in_channels; ++k) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int input_x = input_start_x + kx;
                int input_y = input_start_y + ky;
                
                if (input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height) {
                    int input_idx = ((batch * in_channels + k) * input_height + input_y) * input_width + input_x;
                    int weight_idx = ((k * out_channels + (channel_start + threadIdx.y)) * kernel_size + ky) * kernel_size + kx;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias and apply GELU
    if (bias) {
        sum += bias[channel_start + threadIdx.y];
    }
    sum = gelu_approx(sum);
    
    // Store in shared memory for group norm computation
    *group_shared = sum;
    __syncthreads();
    
    // Compute mean for this group
    float group_sum = 0.0f;
    for (int i = 0; i < group_size; ++i) {
        group_sum += shared_mem[i];
    }
    float group_mean = group_sum / (group_size * output_height * output_width);
    
    // Compute variance for this group
    float group_var_sum = 0.0f;
    for (int i = 0; i < group_size; ++i) {
        float diff = shared_mem[i] - group_mean;
        group_var_sum += diff * diff;
    }
    float group_var = group_var_sum / (group_size * output_height * output_width) + eps;
    
    // Apply group normalization
    float normalized = (*group_shared - group_mean) / sqrtf(group_var);
    
    // Store result
    int output_idx_final = ((batch * out_channels + (channel_start + threadIdx.y)) * output_height + output_idy) * output_width + output_idx;
    output[output_idx_final] = normalized;
}

torch::Tensor fused_conv_transpose_gelu_norm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int groups, int num_groups, float eps=1e-5) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto out_channels = weight.size(1);
    
    int output_height = (input_height - 1) * stride + kernel_size;
    int output_width = (input_width - 1) * stride + kernel_size;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());
    auto mean = torch::empty({batch_size, num_groups}, input.options());
    auto var = torch::empty({batch_size, num_groups}, input.options());
    
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim(
        (output_width + block_dim.x - 1) / block_dim.x,
        (output_height + block_dim.y - 1) / block_dim.y,
        batch_size * num_groups
    );
    
    size_t shared_mem_size = TILE_SIZE * TILE_SIZE * sizeof(float);
    
    fused_conv_transpose_gelu_norm_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width, output_height, output_width,
        kernel_size, stride, groups, num_groups, eps);
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

fused_conv_gelu_norm_cpp_source = """
torch::Tensor fused_conv_transpose_gelu_norm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int groups, int num_groups, float eps=1e-5);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv_gelu_norm",
    cpp_sources=fused_conv_gelu_norm_cpp_source,
    cuda_sources=fused_conv_gelu_norm_source,
    functions=["fused_conv_transpose_gelu_norm_cuda"],
    verbose=True,
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
        self.gamma = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        torch.nn.init.zeros_(self.bias)
        torch.nn.init.ones_(self.gamma)
        torch.nn.init.zeros_(self.beta)

    def forward(self, x):
        # Use fused CUDA kernel for conv_transpose + gelu + group_norm
        x = fused_ops.fused_conv_transpose_gelu_norm_cuda(
            x, self.weight, self.bias,
            self.kernel_size, self.stride, self.groups, self.num_groups
        )
        
        # Apply final scale and shift (this could also be fused, but kept separate for simplicity)
        return x * self.gamma + self.beta