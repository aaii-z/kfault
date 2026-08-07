import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose2d + GELU + GroupNorm
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

// GELU approximation
__device__ __forceinline__ float gelu_approx(float x) {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Fused Kernel: ConvTranspose2d + GELU + GroupNorm
// Each block handles one (N, G) pair. Threads in block iterate over (C/G, H, W).
__global__ void fused_conv_gelu_norm_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, // Can be null
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding,
    int num_groups, int C_per_group) {

    int n = blockIdx.y;
    int g = blockIdx.x;

    int c_start = g * C_per_group;
    int c_end = c_start + C_per_group;

    // Shared memory for reduction
    __shared__ float smem[32 * 3]; // mean, M2, count for warp reduction
    
    int total_pixels = C_per_group * H_out * W_out;
    
    float sum_mean = 0.0f;
    float sum_var = 0.0f;
    int count = 0;
    
    // First pass: Compute ConvTranspose + GELU and accumulate statistics
    for (int i = threadIdx.x; i < total_pixels; i += blockDim.x) {
        int c_out_local = i / (H_out * W_out);
        int h_out = (i / W_out) % H_out;
        int w_out = i % W_out;
        
        int c_out = c_start + c_out_local;

        float sum = 0.0f;
        
        // ConvTranspose logic
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        float val = input[n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in];
                        float w = weight[c_in * C_out * K * K + c_out * K * K + kh * K + kw];
                        sum += val * w;
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        
        // GELU
        float activated = gelu_approx(sum);
        
        // Store activated value
        output[n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out] = activated;
        
        // Welford update (simplified for single thread)
        count++;
        float delta = activated - sum_mean;
        sum_mean += delta / count;
        float delta2 = activated - sum_mean;
        sum_var += delta * delta2;
    }
    
    // Block reduction for Mean and Var
    // Using standard parallel reduction approach
    __shared__ float sh_mean;
    __shared__ float sh_var;
    
    // Reduce within warps
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_mean = __shfl_down_sync(0xffffffff, sum_mean, offset);
        float other_var = __shfl_down_sync(0xffffffff, sum_var, offset);
        int other_count = __shfl_down_sync(0xffffffff, count, offset);
        
        if (other_count > 0) {
            float delta = other_mean - sum_mean;
            float new_count = count + other_count;
            sum_mean += delta * other_count / new_count;
            sum_var += other_var + delta * delta * count * other_count / new_count;
            count = new_count;
        }
    }

    if (lane == 0) {
        smem[wid] = sum_mean;
        smem[32 + wid] = sum_var;
        smem[64 + wid] = (float)count;
    }
    __syncthreads();

    if (wid == 0) {
        sum_mean = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[threadIdx.x] : 0.0f;
        sum_var = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[32 + threadIdx.x] : 0.0f;
        count = (threadIdx.x < (blockDim.x + 31) / 32) ? (int)smem[64 + threadIdx.x] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_mean = __shfl_down_sync(0xffffffff, sum_mean, offset);
            float other_var = __shfl_down_sync(0xffffffff, sum_var, offset);
            int other_count = __shfl_down_sync(0xffffffff, count, offset);
            
            if (other_count > 0) {
                float delta = other_mean - sum_mean;
                float new_count = count + other_count;
                sum_mean += delta * other_count / new_count;
                sum_var += other_var + delta * delta * count * other_count / new_count;
                count = new_count;
            }
        }
    }

    if (threadIdx.x == 0) {
        sh_mean = sum_mean;
        sh_var = (count > 1) ? sum_var / count : 0.0f;
    }
    __syncthreads();
    
    float mean = sh_mean;
    float var = sh_var;
    float inv_std = rsqrtf(var + 1e-5f);
    
    // Second pass: Normalize
    for (int i = threadIdx.x; i < total_pixels; i += blockDim.x) {
        int c_out_local = i / (H_out * W_out);
        int h_out = (i / W_out) % H_out;
        int w_out = i % W_out;
        
        int c_out = c_start + c_out_local;
        
        float val = output[n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out];
        
        float norm_val = (val - mean) * inv_std;
        norm_val = norm_val * gamma[c_out] + beta[c_out];
        
        output[n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out] = norm_val;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    c10::optional<torch::Tensor> bias_opt,
    torch::Tensor gamma, 
    torch::Tensor beta,
    int64_t stride, int64_t padding, int64_t num_groups) {
    
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Beta must be a CUDA tensor");
    
    torch::Tensor bias = bias_opt.value_or(torch::Tensor());
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    int C_out = weight.size(1);
    int K = weight.size(2);
    
    // Output size calculation for ConvTranspose2d
    // H_out = (H_in - 1) * stride - 2 * padding + kernel_size
    int H_out = (H_in - 1) * stride - 2 * padding + K;
    int W_out = (W_in - 1) * stride - 2 * padding + K;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    int C_per_group = C_out / num_groups;
    
    // Grid: (num_groups, N)
    dim3 grid(num_groups, N);
    // Block: enough to cover C_per_group * H_out * W_out
    int block_size = 512; // Optimized for typical occupancy
    
    fused_conv_gelu_norm_kernel<<<grid, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding,
        num_groups, C_per_group
    );
    
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    c10::optional<torch::Tensor> bias,
    torch::Tensor gamma, 
    torch::Tensor beta,
    int64_t stride, int64_t padding, int64_t num_groups);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        # We keep the original ConvTranspose2d to initialize weights
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        self.fused_ops = fused_ops
        self.stride = stride
        self.padding = self.conv_transpose.padding[0] # Assuming symmetric padding
        self.num_groups = num_groups

    def forward(self, x):
        return self.fused_ops.fused_ops_cuda(
            x, 
            self.conv_transpose.weight, 
            self.conv_transpose.bias, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.stride, 
            self.padding, 
            self.num_groups
        )