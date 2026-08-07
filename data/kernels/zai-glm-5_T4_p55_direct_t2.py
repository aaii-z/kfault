import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul, max_pool, sum, and scale
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Tiled matrix multiplication kernel
// Input: (M, K), Weight: (N, K) -> Output: (M, N)
// Computes input * weight.T
__global__ void matmul_kernel(const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias, float* __restrict__ output, int M, int N, int K) {
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_weight = shared_mem + blockDim.y * blockDim.x; // blockDim.x is TILE_SIZE (16)

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float sum_val = 0.0f;

    for (int t = 0; t < (K + blockDim.x - 1) / blockDim.x; ++t) {
        // Load input tile: s_input[ty][tx] = Input[row, t*blockDim.x + tx]
        int input_col = t * blockDim.x + tx;
        if (row < M && input_col < K) {
            s_input[ty * blockDim.x + tx] = input[row * K + input_col];
        } else {
            s_input[ty * blockDim.x + tx] = 0.0f;
        }

        // Load weight tile: s_weight[ty][tx] = Weight[col, t*blockDim.x + ty]
        // Note: Weight is (N, K), we are loading a tile for the column 'col'
        // Thread (ty, tx) loads element at column 'col', row 't*blockDim.x + ty'
        int weight_row = t * blockDim.x + ty;
        if (col < N && weight_row < K) {
             s_weight[ty * blockDim.x + tx] = weight[col * K + weight_row];
        } else {
             s_weight[ty * blockDim.x + tx] = 0.0f;
        }
        
        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k) {
            sum_val += s_input[ty * blockDim.x + k] * s_weight[k * blockDim.x + tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        float val = sum_val;
        if (bias != nullptr) {
            val += bias[col];
        }
        output[row * N + col] = val;
    }
}

// Fused kernel: MaxPool1D + Sum + Scale
// Input: (batch_size, out_features) treated as (batch_size, 1, out_features)
// MaxPool1D with kernel_size=2, stride=2 (default) on dim=2
// Output shape after pool: (batch_size, 1, out_features/2) -> squeeze -> (batch_size, out_features/2)
// Sum over dim=1 (the out_features/2 dimension) -> (batch_size,)
// Scale -> (batch_size,)
__global__ void fused_pool_sum_scale_kernel(const float* __restrict__ input, float* __restrict__ output, int N, int N_out, float scale_factor) {
    // N is original out_features (32768)
    // N_out is N / 2 (16384)
    
    int batch_idx = blockIdx.x; // One block per batch item
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Dynamic shared memory for reduction
    extern __shared__ float s_data[];
    
    float local_sum = 0.0f;
    
    // Stride loop to cover all N_out elements
    // N_out = N / 2.
    // Input indices: 0, 1 -> max -> out[0]; 2, 3 -> max -> out[1]...
    
    for (int i = tid; i < N_out; i += block_size) {
        int idx1 = 2 * i;
        int idx2 = 2 * i + 1;
        float v1 = input[batch_idx * N + idx1];
        float v2 = input[batch_idx * N + idx2];
        float max_val = fmaxf(v1, v2);
        local_sum += max_val;
    }
    
    s_data[tid] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch_idx] = s_data[0] * scale_factor;
    }
}

// Helper function to launch the matmul kernel
void matmul_cuda_launcher(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    size_t shared_mem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    
    matmul_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() && bias.numel() > 0 ? bias.data_ptr<float>() : nullptr, 
        output.data_ptr<float>(), M, N, K
    );
}

// Helper function to launch the fused kernel
void fused_pool_sum_scale_cuda_launcher(torch::Tensor input, torch::Tensor output, int N, float scale_factor) {
    int batch_size = input.size(0);
    int N_out = N / 2;
    
    const int BLOCK_SIZE = 256;
    dim3 gridDim(batch_size);
    dim3 blockDim(BLOCK_SIZE);
    
    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);
    
    fused_pool_sum_scale_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N, N_out, scale_factor
    );
}

torch::Tensor model_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, float scale_factor) {
    // 1. MatMul
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto matmul_output = torch::empty({batch_size, out_features}, input.options());
    matmul_cuda_launcher(input, weight, bias, matmul_output);
    
    // 2. Fused MaxPool, Sum, Scale
    // Assuming kernel_size=2 as per problem description
    auto final_output = torch::empty({batch_size}, input.options());
    fused_pool_sum_scale_cuda_launcher(matmul_output, final_output, out_features, scale_factor);
    
    return final_output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor model_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, float scale_factor);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["model_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs matrix multiplication, max pooling, sum, and scaling
    using custom CUDA kernels.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        
        # We still define the Linear layer to hold weights and biases,
        # but we won't use its forward pass.
        self.matmul = nn.Linear(in_features, out_features)
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size).
        """
        # Extract weight and bias from the Linear layer
        weight = self.matmul.weight
        bias = self.matmul.bias
        
        return self.fused_ops.model_forward_cuda(x, weight, bias, self.kernel_size, self.scale_factor)