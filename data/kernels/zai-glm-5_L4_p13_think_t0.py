import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void fused_op_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int B, int C, int D, int H, int W,
    scalar_t scaling_factor) {

    // Block indices
    int tx = threadIdx.x; // 0..15 (W dimension tile)
    int ty = threadIdx.y; // 0..63 (C dimension tile)
    
    // Grid indices
    int w_idx = blockIdx.x * 16 + tx;
    int h_idx = blockIdx.y;
    int b_idx = blockIdx.z;

    if (w_idx >= W) return;

    // Shared memory for reduction (Max and Sum)
    // Size: 64 threads * 2 values (max, sum) -> 128 floats. 
    // We use 2 separate arrays or a struct. Let's use a simple array padded to avoid bank conflicts.
    // Layout: [ty * 32 + tx]. We use 32 width to match blockDim.x.
    extern __shared__ char s_mem_char[];
    scalar_t* s_data = reinterpret_cast<scalar_t*>(s_mem_char);

    // Each thread handles multiple channels if C > 64
    // We assume C <= 1024 for register usage, typical for CNNs.
    // Max channels per thread = ceil(C / 64)
    int num_ch_per_thread = (C + 63) / 64;
    
    scalar_t vals[16]; // Register storage for channel values (max 16 * 64 = 1024 channels)
    scalar_t local_max = -INFINITY;
    
    // 1. Mean over D and Add Bias
    // Input stride for D is H*W
    int stride_d = H * W;
    int base_idx = b_idx * (C * D * H * W) + h_idx * W + w_idx;
    
    // Loop over channels assigned to this thread
    for (int i = 0; i < num_ch_per_thread; ++i) {
        int c = ty + i * 64;
        if (c < C) {
            scalar_t sum_d = 0.0f;
            const scalar_t* in_ptr = input + base_idx + c * (D * H * W);
            
            // Loop over Depth
            // Unrolling might help, but D=32 is small enough
            for (int d = 0; d < D; ++d) {
                sum_d += in_ptr[d * stride_d];
            }
            
            scalar_t mean_val = sum_d / static_cast<scalar_t>(D);
            mean_val += bias[c]; // Add bias
            vals[i] = mean_val;
            local_max = fmaxf(local_max, mean_val);
        } else {
            vals[i] = -INFINITY; // Padding for reduction
        }
    }

    // 2. Max Reduction (Softmax part 1)
    // Load local max to shared memory
    s_data[ty * 32 + tx] = local_max;
    __syncthreads();

    // Reduce within the block (64 threads in y)
    // Assuming blockDim.y = 64
    for (int s = 32; s > 0; s >>= 1) {
        if (ty < s) {
            s_data[ty * 32 + tx] = fmaxf(s_data[ty * 32 + tx], s_data[(ty + s) * 32 + tx]);
        }
        __syncthreads();
    }
    scalar_t global_max = s_data[tx]; // ty=0 holds the max
    __syncthreads();

    // 3. Compute Exp and Local Sum
    scalar_t local_sum = 0.0f;
    for (int i = 0; i < num_ch_per_thread; ++i) {
        int c = ty + i * 64;
        if (c < C) {
            scalar_t exp_val = expf(vals[i] - global_max);
            vals[i] = exp_val; // Store exp result back to register
            local_sum += exp_val;
        }
    }

    // 4. Sum Reduction (Softmax part 2)
    s_data[ty * 32 + tx] = local_sum;
    __syncthreads();

    for (int s = 32; s > 0; s >>= 1) {
        if (ty < s) {
            s_data[ty * 32 + tx] += s_data[(ty + s) * 32 + tx];
        }
        __syncthreads();
    }
    scalar_t global_sum = s_data[tx];
    
    // 5. Finalize: Softmax, Tanh, Scale
    scalar_t inv_sum = 1.0f / global_sum;
    
    for (int i = 0; i < num_ch_per_thread; ++i) {
        int c = ty + i * 64;
        if (c < C) {
            scalar_t softmax_val = vals[i] * inv_sum;
            scalar_t tanh_val = tanhf(softmax_val);
            scalar_t out_val = tanh_val * scaling_factor;
            
            // Write output
            // Output shape: (B, C, 1, H, W)
            // Strides: C*H*W, H*W, H*W, W, 1
            output[b_idx * (C * H * W) + c * (H * W) + h_idx * W + w_idx] = out_val;
        }
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input, 
    torch::Tensor bias, 
    float scaling_factor) {
    
    auto sizes = input.sizes();
    int B = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];

    // Output shape: (B, C, 1, H, W)
    auto output = torch::zeros({B, C, 1, H, W}, input.options());

    // Grid: (W_tiles, H, B)
    // Block: (16, 64) -> 16 for W, 64 for C
    dim3 block(16, 64);
    dim3 grid((W + 15) / 16, H, B);

    // Shared memory size: 64 * 32 floats for reduction
    size_t shared_mem_size = 64 * 32 * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_op_kernel", ([&] {
        fused_op_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, C, D, H, W,
            static_cast<scalar_t>(scaling_factor)
        );
    }));

    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);
"""

# Compile the inline CUDA code
fused_op_module = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op_module

    def forward(self, x):
        x = self.conv_transpose(x)
        # Use custom fused kernel
        # Note: The bias in the original model is (1, C, 1, 1, 1). 
        # The kernel expects a 1D bias tensor of size C for efficient access.
        bias_1d = self.bias.squeeze() # Shape (C)
        x = self.fused_op.fused_op_cuda(x, bias_1d, self.scaling_factor)
        return x