import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused post-convolution operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for fused Mean Pooling (Depth), Bias Add, Softmax, Tanh, and Scaling
// Input Tensor shape: (N, C, D, H, W)
// Bias shape: (1, C, 1, 1, 1)
// Output Tensor shape: (N, C, 1, H, W)
__global__ void fused_ops_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int N, int C, int D, int H, int W, 
    float scaling_factor) {
    
    // Calculate global thread index
    // We map threads to output elements: (n, c, h, w)
    // Output size is N * C * H * W
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;

    if (idx < total_elements) {
        // Decompose idx into (n, c, h, w)
        int w_idx = idx % W;
        int h_idx = (idx / W) % H;
        int c_idx = (idx / (W * H)) % C;
        int n_idx = idx / (W * H * C);

        // 1. Mean Pooling over Depth (D)
        // Input offset for this (n, c, h, w) starting at d=0
        // Input layout is contiguous: input[n, c, d, h, w]
        long long input_offset = (long long)n_idx * C * D * H * W + 
                                 (long long)c_idx * D * H * W + 
                                 (long long)h_idx * W + 
                                 w_idx;
        
        // Stride for D dimension
        long long d_stride = H * W;

        float sum_val = 0.0f;
        // Use loop unrolling or pragma unroll for potential speedup
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            sum_val += input[input_offset + d * d_stride];
        }
        float mean_val = sum_val / D;

        // 2. Add Bias
        // Bias is broadcastable: bias[0, c, 0, 0, 0]
        float val = mean_val + bias[c_idx];

        // 3. Softmax (over channels C)
        // We need to compute exp(val - max_val) for this element
        // However, Softmax requires global synchronization or a two-pass approach.
        // Since we are fusing, we can't easily do a full softmax across C in a single kernel
        // without shared memory reduction for every (n, h, w) slice.
        // Given the constraints and the goal of a single kernel launch, 
        // we implement a "local" softmax approximation or assume the user wants element-wise ops.
        // BUT, strictly following PyTorch semantics for softmax(dim=1):
        // We must compute max and sum across the C dimension.
        // This is impossible in a single simple kernel without shared memory.
        // To keep it efficient and correct, we launch a separate kernel or use a reduction.
        // However, for this specific "inline" task, we often assume a simplified version 
        // or implement a reduction within the kernel.
        // Let's implement a block reduction for softmax.
        
        // Actually, implementing a full softmax reduction inside this specific kernel
        // is complex because threads are mapped to spatial outputs.
        // A better approach for a single fused kernel is to map threads to (N, H, W)
        // and have each thread loop over C.
        
        // RE-DESIGNING THE KERNEL MAPPING:
        // Map threads to (N, H, W) locations.
        // Each thread computes the full C vector for that location.
        // This ensures softmax correctness and allows fusion.
    }
}

// Optimized Kernel: Each thread handles one (n, h, w) location and processes all channels.
__global__ void fused_ops_optimized_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int N, int C, int D, int H, int W, 
    float scaling_factor) {

    // Map thread to (n, h, w)
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int n_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (w_idx < W && h_idx < H && n_idx < N) {
        
        // 1. Compute Mean Pooling and Bias Add for all channels
        // We need a local array or register storage for the C values to do Softmax.
        // If C is small (e.g., 64), registers are fine.
        // If C is large, we might spill to local memory (which is L1/L2).
        // Given C=64 in the test config, this is efficient.

        float vals[64]; // Assuming C <= 64 for register allocation, or dynamic local memory
        // Note: Dynamic local memory would require kernel launch configuration.
        // We will rely on the compiler optimizing this for C=64.
        // For general C, we might need to loop, but let's stick to the test config assumption or local arrays.
        // Since we don't know C at compile time in the CUDA source string, 
        // we can't declare float vals[C].
        // We will compute in two passes over the input data to save memory.

        // Pass 1: Compute Mean + Bias and find Max
        float max_val = -INFINITY;
        
        // Iterate over channels
        for (int c = 0; c < C; ++c) {
            // Calculate input pointer for (n, c, d, h, w)
            // Sum over D
            long long input_base = (long long)n_idx * C * D * H * W + 
                                   (long long)c * D * H * W + 
                                   (long long)h_idx * W + 
                                   w_idx;
            
            float sum_d = 0.0f;
            for (int d = 0; d < D; ++d) {
                sum_d += input[input_base + (long long)d * H * W];
            }
            float mean_val = sum_d / D;
            float val = mean_val + bias[c];
            
            // Store temporarily? No, we need it for softmax denominator later.
            // We can't store it easily without dynamic allocation.
            // Strategy: Recompute. It's memory bound, but keeps the kernel simple and correct.
            // Or better: Use local memory if needed. 
            // Given D=32, C=64, recomputing is 32 loads * 2 passes = 64 loads per element.
            // Storing would be 32 loads + 1 write + 1 read + 1 write = 34 ops.
            // Recomputing is likely faster than local memory spilling for large C.
            
            if (val > max_val) {
                max_val = val;
            }
        }

        // Pass 2: Compute Exp, Sum, Tanh, Scale and Write Output
        float sum_exp = 0.0f;
        
        // We need to compute exp(val - max) and sum them.
        // We have to loop again to compute the sum.
        for (int c = 0; c < C; ++c) {
             long long input_base = (long long)n_idx * C * D * H * W + 
                                   (long long)c * D * H * W + 
                                   (long long)h_idx * W + 
                                   w_idx;
            
            float sum_d = 0.0f;
            for (int d = 0; d < D; ++d) {
                sum_d += input[input_base + (long long)d * H * W];
            }
            float mean_val = sum_d / D;
            float val = mean_val + bias[c];
            
            sum_exp += expf(val - max_val);
        }

        // Pass 3: Finalize and Write
        for (int c = 0; c < C; ++c) {
            long long input_base = (long long)n_idx * C * D * H * W + 
                                   (long long)c * D * H * W + 
                                   (long long)h_idx * W + 
                                   w_idx;
            
            float sum_d = 0.0f;
            for (int d = 0; d < D; ++d) {
                sum_d += input[input_base + (long long)d * H * W];
            }
            float mean_val = sum_d / D;
            float val = mean_val + bias[c];
            
            float softmax_val = expf(val - max_val) / sum_exp;
            float tanh_val = tanhf(softmax_val);
            float final_val = tanh_val * scaling_factor;

            // Write to output: (N, C, 1, H, W)
            // Output layout: output[n, c, 0, h, w]
            long long output_idx = (long long)n_idx * C * H * W + 
                                   (long long)c * H * W + 
                                   (long long)h_idx * W + 
                                   w_idx;
            output[output_idx] = final_val;
        }
    }
}

// Helper function to launch the kernel
torch::Tensor fused_ops_cuda(
    torch::Tensor input, 
    torch::Tensor bias, 
    float scaling_factor) {
    
    // Input dims: (N, C, D, H, W)
    // Output dims: (N, C, 1, H, W)
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::zeros({N, C, 1, H, W}, input.options());

    // Grid and Block dimensions
    // Threads mapped to (W, H, N)
    dim3 blockDim(16, 16, 1); // 256 threads per block
    dim3 gridDim(
        (W + blockDim.x - 1) / blockDim.x,
        (H + blockDim.y - 1) / blockDim.y,
        (N + blockDim.z - 1) / blockDim.z
    );

    fused_ops_optimized_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        scaling_factor
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs:
    1. Transposed 3D convolution (using PyTorch's optimized implementation)
    2. Fused Mean pooling, Bias addition, Softmax, Tanh, and Scaling (using custom CUDA kernel)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        # Apply the fused CUDA kernel
        x = self.fused_ops.fused_ops_cuda(x, self.bias, self.scaling_factor)
        return x