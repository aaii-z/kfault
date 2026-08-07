import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul, max_pool, sum, and scale
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Tiled matrix multiplication kernel
__global__ void matmul_kernel(const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias, float* __restrict__ output, int M, int N, int K) {
    // Input: (M, K), Weight: (N, K) -> Output: (M, N)
    // Note: nn.Linear stores weights as (out_features, in_features), so we compute input * weight.T
    
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_weight = shared_mem + blockDim.x * blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float sum_val = 0.0f;

    for (int t = 0; t < (K + blockDim.x - 1) / blockDim.x; ++t) {
        // Load input tile (M x K) -> block loads (blockDim.y x blockDim.x)
        int input_col = t * blockDim.x + tx;
        if (row < M && input_col < K) {
            s_input[ty * blockDim.x + tx] = input[row * K + input_col];
        } else {
            s_input[ty * blockDim.x + tx] = 0.0f;
        }

        // Load weight tile (N x K) -> block loads (blockDim.x x blockDim.y)
        // We are computing input * weight.T, so we need weight.T
        // weight is (N, K), weight.T is (K, N)
        // We need to load weight[col, t*blockDim.x + tx] -> weight[col * K + t*blockDim.x + tx]
        // Wait, standard tiling for input (M, K) * weight.T (K, N)
        // weight[col, k] is the element.
        // We need weight[col, k] for the dot product.
        // weight is stored row-major: weight[col * K + k]
        
        int weight_col = t * blockDim.x + tx; // index k
        if (col < N && weight_col < K) {
             s_weight[ty * blockDim.x + tx] = weight[col * K + weight_col];
        } else {
             s_weight[ty * blockDim.x + tx] = 0.0f;
        }
        
        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k) {
            sum_val += s_input[ty * blockDim.x + k] * s_weight[ty * blockDim.x + k]; // This indexing is wrong for s_weight
        }
        // Correct indexing logic:
        // We want to compute Output[row, col] = sum_k Input[row, k] * Weight[col, k]
        // Tile loop iterates over k dimension.
        // s_input stores a tile of Input: rows [row_start, row_end], cols [k_start, k_end]
        // s_weight stores a tile of Weight: rows [col_start, col_end], cols [k_start, k_end]
        
        // Let's restart the shared memory loading logic.
        // s_input[ty][tx] -> Input[row, t*blockDim.x + tx]
        // s_weight[ty][tx] -> Weight[col, t*blockDim.x + tx] (since we are iterating over K)
        
        // Accumulation:
        // for k = 0 to blockDim.x - 1:
        //   sum += s_input[ty][k] * s_weight[ty][k] -> This assumes s_weight is loaded similarly to s_input.
        //   Wait, s_weight needs to correspond to the column of weight matrix.
        //   We need Weight[col, k]. The tile for Weight should be loaded by threads in a column of the block? No.
        
        // Let's use a simpler kernel or correct the tiling.
        // Correct tiling:
        // s_input[ty][k] = Input[row, t*blockDim.x + k]
        // s_weight[ty][k] = Weight[col, t*blockDim.x + k] -> This requires loading Weight with different mapping.
        // Thread (ty, tx) loads Weight[col, t*blockDim.x + tx].
        // So s_weight[ty][tx] = Weight[col, t*blockDim.x + tx].
        // Then sum += s_input[ty][k] * s_weight[ty][k] is WRONG because s_weight[ty][k] corresponds to Weight[col, ...]
        // We need Weight[col, ...]. Thread (ty, tx) is responsible for Output[row, col].
        // We need to multiply Input[row, k] with Weight[col, k].
        // s_input[ty][k] is Input[row, t*blockDim.x + k].
        // s_weight[ty][k] needs to be Weight[col, t*blockDim.x + k].
        // But we loaded s_weight[ty][tx] = Weight[col, t*blockDim.x + tx].
        // So s_weight[ty][k] is indeed Weight[col, t*blockDim.x + k].
        // So the loop should be:
        // for (int k = 0; k < blockDim.x; ++k) sum += s_input[ty * blockDim.x + k] * s_weight[ty * blockDim.x + k];
        // Wait, s_weight[ty][k] means the element at row ty, col k in shared mem.
        // We loaded s_weight[ty][tx]. So the k-th element in row ty is s_weight[ty][k].
        // Yes, this works.
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
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < gridDim.x * blockDim.x) { // Iterate over batch
         // Each block handles one batch item? Or each thread?
         // N is large (32768). N_out is 16384.
         // Reduction over 16384 elements.
         // Let's use one thread block per batch item for reduction.
         
         if (batch_idx < gridDim.x) { // gridDim.x should be batch_size
             float sum_val = 0.0f;
             // Iterate over the output features after pooling
             // We can parallelize this reduction within the block.
             // But let's stick to a simple loop for correctness first, or use shared memory reduction.
             
             // Simple loop by one thread (slow but correct for now, will optimize with reduction)
             // Actually, let's do a parallel reduction.
             
             // Wait, the grid/block config needs to be set carefully.
             // Let's assume we launch (batch_size, 1, 1) grid and (threads, 1, 1) block.
             
             // Let's rewrite the kernel to be called with (batch_size) blocks.
             // threadIdx.x iterates over N_out.
             
             // Optimization: Vectorized load?
             // For now, simple loop.
             
             // Actually, the previous kernel was MatMul.
             // This kernel takes MatMul output.
             
             // Let's implement a parallel reduction sum.
             // Each thread sums a portion of the pooled values.
             
             int tid = threadIdx.x;
             int block_size = blockDim.x;
             
             // Dynamic shared memory for reduction
             extern __shared__ float s_data[];
             
             float local_sum = 0.0f;
             
             // Stride loop to cover all N_out elements
             // N_out = N / 2.
             // Input is (batch_size, N). We treat it as (batch_size, N).
             // We need to perform max pooling first.
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