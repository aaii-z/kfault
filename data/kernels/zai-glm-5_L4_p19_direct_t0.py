import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose2d + GELU + GroupNorm
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper struct for Welford's online algorithm for variance/mean
template <typename T>
struct WelfordData {
    T mean;
    T M2;
    int count;

    __device__ WelfordData() : mean(0), M2(0), count(0) {}
};

template <typename T>
__device__ void welford_update(WelfordData<T>& acc, T val) {
    acc.count++;
    T delta = val - acc.mean;
    acc.mean += delta / acc.count;
    T delta2 = val - acc.mean;
    acc.M2 += delta * delta2;
}

// Block reduction for WelfordData
template <typename T>
__device__ void block_reduce_welford(WelfordData<T>& acc) {
    __shared__ T mean_shared[32];
    __shared__ T m2_shared[32];
    __shared__ int count_shared[32];

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 8) { // warp size is 32
        T other_mean = __shfl_down_sync(0xffffffff, acc.mean, offset);
        T other_m2 = __shfl_down_sync(0xffffffff, acc.M2, offset);
        int other_count = __shfl_down_sync(0xffffffff, acc.count, offset);
        
        if (other_count > 0) {
            T delta = other_mean - acc.mean;
            T new_count = acc.count + other_count;
            acc.mean += delta * other_count / new_count;
            acc.M2 += other_m2 + delta * delta * acc.count * other_count / new_count;
            acc.count = new_count;
        }
    }

    if (lane == 0) {
        mean_shared[wid] = acc.mean;
        m2_shared[wid] = acc.M2;
        count_shared[wid] = acc.count;
    }
    __syncthreads();

    if (wid == 0) {
        acc.mean = (lane < blockDim.x >> 5) ? mean_shared[lane] : T(0);
        acc.M2 = (lane < blockDim.x >> 5) ? m2_shared[lane] : T(0);
        acc.count = (lane < blockDim.x >> 5) ? count_shared[lane] : 0;

        for (int offset = 16; offset > 0; offset >>= 8) {
            T other_mean = __shfl_down_sync(0xffffffff, acc.mean, offset);
            T other_m2 = __shfl_down_sync(0xffffffff, acc.M2, offset);
            int other_count = __shfl_down_sync(0xffffffff, acc.count, offset);
            
            if (other_count > 0) {
                T delta = other_mean - acc.mean;
                T new_count = acc.count + other_count;
                acc.mean += delta * other_count / new_count;
                acc.M2 += other_m2 + delta * delta * acc.count * other_count / new_count;
                acc.count = new_count;
            }
        }
    }
    __syncthreads();
    
    // Broadcast result
    acc.mean = mean_shared[0];
    acc.M2 = m2_shared[0];
    acc.count = count_shared[0];
}


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

    // Pointers for this block
    // Input: (N, C_in, H_in, W_in)
    // Weight: (C_in, C_out, K, K) - Note: ConvTranspose weight layout
    // Output: (N, C_out, H_out, W_out)
    
    // We compute the output for channels [c_start, c_end)
    // We need to iterate over all input channels
    
    // Shared memory for partial sums? No, too large. 
    // We will compute output pixels and atomically add? No, standard output is too large for shared memory if we try to do all.
    // We will map threads to output pixels.
    
    int total_pixels = C_per_group * H_out * W_out;
    
    WelfordData<float> stat_acc;
    
    // First pass: Compute ConvTranspose + GELU and accumulate statistics
    for (int i = threadIdx.x; i < total_pixels; i += blockDim.x) {
        int c_out_local = i / (H_out * W_out);
        int h_out = (i / W_out) % H_out;
        int w_out = i % W_out;
        
        int c_out = c_start + c_out_local;

        float sum = 0.0f;
        
        // ConvTranspose logic
        // Output pixel (h_out, w_out) depends on input pixels
        // h_in = h_out * stride + k - padding
        // We iterate over kernel and input channels
        
        // Optimization: Iterate over input channels
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        float val = input[n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in];
                        // Weight index: ConvTranspose2d weight is (in_channels, out_channels, kH, kW)
                        // But PyTorch documentation says: weight (in_channels, out_channels, kH, kW)
                        // Wait, let's check PyTorch docs.
                        // ConvTranspose2d weight: (in_channels, out_channels, kH, kW)
                        // Yes.
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
        
        // Store temporarily? We need to store to compute mean/var.
        // We can't store the whole output in shared memory.
        // We must write to global memory, then sync, then normalize.
        // But we can fuse the stats accumulation here.
        
        // Write activated value to output (we will overwrite it later with normalized)
        // Or write to a temporary buffer? Output buffer is fine, we just need to read it back.
        // Let's write to output.
        output[n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out] = activated;
        
        welford_update(stat_acc, activated);
    }
    
    block_reduce_welford(stat_acc);
    
    __shared__ float sh_mean;
    __shared__ float sh_var;
    
    if (threadIdx.x == 0) {
        sh_mean = stat_acc.mean;
        sh_var = (stat_acc.count > 1) ? stat_acc.M2 / stat_acc.count : 0.0f;
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
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    auto bias = bias_opt.value_or(torch::Tensor());
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    int C_out = weight.size(1);
    int K = weight.size(2);
    
    int H_out = (H_in - 1) * stride - 2 * padding + K;
    int W_out = (W_in - 1) * stride - 2 * padding + K;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    int C_per_group = C_out / num_groups;
    
    // Grid: (num_groups, N)
    dim3 grid(num_groups, N);
    // Block: enough to cover C_per_group * H_out * W_out
    int block_size = 512; // Optimized for typical occupancy
    
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
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
        
        # We extract weights and bias from the conv layer
        # and gamma/beta from the norm layer to pass to our custom kernel.
        # We register them as buffers or parameters if they need gradients.
        # Since we are replacing the forward, we just use the existing parameters.
        
        self.fused_ops = fused_ops
        self.stride = stride
        self.padding = self.conv_transpose.padding[0] # Assuming symmetric padding
        self.num_groups = num_groups

    def forward(self, x):
        # We use the weights from the defined layers
        # Note: ConvTranspose2d padding defaults to 0 if not specified, but here we use the layer's padding
        # The original model didn't specify padding, so it defaults to 0.
        # However, ConvTranspose2d output size formula: H_out = (H_in - 1)*stride - 2*padding + K
        # If padding is 0, H_out = (H_in - 1)*stride + K.
        # Let's check the original model's output size logic.
        # Original: H_out = (256 - 1) * 1 + 3 = 258.
        # My kernel uses padding. I need to match the padding.
        
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