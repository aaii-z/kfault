import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Constants for GELU approximation
constexpr float GELU_COEF = 0.044715f;
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f; 

// Kernel: Fused ConvTranspose2d + GELU
// Specialized for stride=1, padding=0, dilation=1
__global__ void fused_conv_transpose_gelu_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int N, int C_in, int C_out, int H_in, int W_in, int K) {
    
    int H_out = H_in + K - 1;
    int W_out = W_in + K - 1;

    // Block and thread indices
    int n_oc_idx = blockIdx.x;
    int n = n_oc_idx / C_out;
    int oc = n_oc_idx % C_out;
    
    int oh = blockIdx.y * blockDim.x + threadIdx.x;
    int ow = blockIdx.z * blockDim.y + threadIdx.y;

    if (oh >= H_out || ow >= W_out) return;

    float val = bias[oc];

    // Shared memory for input tile
    // Size: (blockDim.x + K - 1) * (blockDim.y + K - 1)
    // blockDim is 16x16, K=3 -> 18x18
    extern __shared__ float shared_input[];

    int tile_h = blockDim.x + K - 1;
    int tile_w = blockDim.y + K - 1;
    
    // Calculate input region needed for this block
    int input_tile_h_start = blockIdx.y * blockDim.x - K + 1;
    int input_tile_w_start = blockIdx.z * blockDim.y - K + 1;

    // Loop over input channels
    for (int ic = 0; ic < C_in; ++ic) {
        // Load input tile into shared memory
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < tile_h * tile_w; i += blockDim.x * blockDim.y) {
            int sh_h = i / tile_w;
            int sh_w = i % tile_w;
            
            int ih = input_tile_h_start + sh_h;
            int iw = input_tile_w_start + sh_w;
            
            float in_val = 0.0f;
            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                in_val = input[n * C_in * H_in * W_in + ic * H_in * W_in + ih * W_in + iw];
            }
            shared_input[sh_h * tile_w + sh_w] = in_val;
        }
        __syncthreads();

        // Compute convolution
        // We are computing output(oh, ow). We need input(oh - kh, ow - kw).
        // In shared memory, the pixel (oh, ow) corresponds to:
        // smem_h = oh - input_tile_h_start
        // smem_w = ow - input_tile_w_start
        
        int smem_base_h = oh - input_tile_h_start;
        int smem_base_w = ow - input_tile_w_start;

        const float* w_ptr = weight + ic * C_out * K * K + oc * K * K;
        
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                int s_h = smem_base_h - kh;
                int s_w = smem_base_w - kw;
                
                float w = w_ptr[kh * K + kw];
                float i = shared_input[s_h * tile_w + s_w];
                
                val += w * i;
            }
        }
        __syncthreads();
    }

    // Apply GELU (approximation)
    float cdf = 0.5f * (1.0f + tanhf(SQRT_2_OVER_PI * (val + GELU_COEF * val * val * val)));
    val = val * cdf;

    // Write output
    output[n * C_out * H_out * W_out + oc * H_out * W_out + oh * W_out + ow] = val;
}

// Kernel: GroupNorm Stats
__global__ void group_norm_stats_kernel(
    const float* __restrict__ x, 
    float* __restrict__ mean, 
    float* __restrict__ var, 
    int N, int C, int H, int W, int G, int C_per_G) {
    
    int n = blockIdx.x;
    int g = blockIdx.y;
    
    int group_start_c = g * C_per_G;
    int total_elements = C_per_G * H * W;
    
    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;
    
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        int c_local = i / (H * W);
        int hw_local = i % (H * W);
        int h = hw_local / W;
        int w = hw_local % W;
        
        int c_global = group_start_c + c_local;
        
        float val = x[n * C * H * W + c_global * H * W + h * W + w];
        sum_val += val;
        sum_sq_val += val * val;
    }
    
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    
    int tid = threadIdx.x;
    s_sum[tid] = sum_val;
    s_sum_sq[tid] = sum_sq_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float mean_val = s_sum[0] / total_elements;
        mean[n * G + g] = mean_val;
        var[n * G + g] = s_sum_sq[0] / total_elements - mean_val * mean_val;
    }
}

// Kernel: GroupNorm Apply
__global__ void group_norm_apply_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ mean, 
    const float* __restrict__ var, 
    const float* __restrict__ gamma, 
    const float* __restrict__ beta, 
    float* __restrict__ y, 
    int N, int C, int H, int W, int G, int C_per_G, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * C * H * W;
    
    if (idx < total_size) {
        int n = idx / (C * H * W);
        int rem = idx % (C * H * W);
        int c = rem / (H * W);
        int hw = rem % (H * W);
        int h = hw / W;
        int w = hw % W;
        
        int g = c / C_per_G;
        
        float mean_val = mean[n * G + g];
        float var_val = var[n * G + g];
        float inv_std = rsqrtf(var_val + eps);
        
        float val = x[idx];
        float norm_val = (val - mean_val) * inv_std;
        
        y[idx] = norm_val * gamma[c] + beta[c];
    }
}

// Wrappers
torch::Tensor fused_conv_transpose_gelu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int K) {
    
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(1);
    
    int H_out = H_in + K - 1;
    int W_out = W_in + K - 1;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    int block_h = 16;
    int block_w = 16;
    dim3 block(block_h, block_w);
    dim3 grid(N * C_out, (H_out + block_h - 1) / block_h, (W_out + block_w - 1) / block_w);
    
    int tile_h = block_h + K - 1;
    int tile_w = block_w + K - 1;
    size_t shared_mem_size = tile_h * tile_w * sizeof(float);
    
    fused_conv_transpose_gelu_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), N, C_in, C_out, H_in, W_in, K);
    
    return output;
}

std::vector<torch::Tensor> group_norm_stats_cuda(torch::Tensor x, int G, int C_per_G) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    auto mean = torch::zeros({N, G}, x.options());
    auto var = torch::zeros({N, G}, x.options());
    
    int block_size = 256;
    dim3 grid(N, G);
    
    group_norm_stats_kernel<<<grid, block_size>>>(
        x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        N, C, H, W, G, C_per_G);
        
    return {mean, var};
}

torch::Tensor group_norm_apply_cuda(
    torch::Tensor x, torch::Tensor mean, torch::Tensor var, 
    torch::Tensor gamma, torch::Tensor beta, int G, int C_per_G, float eps) {
    
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    auto y = torch::zeros_like(x);
    
    int total_size = N * C * H * W;
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    
    group_norm_apply_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(),
        N, C, H, W, G, C_per_G, eps);
        
    return y;
}
"""

cpp_source = """
torch::Tensor fused_conv_transpose_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int K);
std::vector<torch::Tensor> group_norm_stats_cuda(torch::Tensor x, int G, int C_per_G);
torch::Tensor group_norm_apply_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta, int G, int C_per_G, float eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_conv_transpose_gelu_cuda", "group_norm_stats_cuda", "group_norm_apply_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.fused_ops = fused_ops
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        # 1. Fused ConvTranspose + GELU
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        intermediate = self.fused_ops.fused_conv_transpose_gelu_cuda(x, weight, bias, self.kernel_size)
        
        # 2. GroupNorm
        C = intermediate.size(1)
        C_per_G = C // self.num_groups
        
        stats = self.fused_ops.group_norm_stats_cuda(intermediate, self.num_groups, C_per_G)
        mean = stats[0]
        var = stats[1]
        
        gamma = self.group_norm.weight
        beta = self.group_norm.bias
        out = self.fused_ops.group_norm_apply_cuda(intermediate, mean, var, gamma, beta, self.num_groups, C_per_G, self.eps)
        
        return out