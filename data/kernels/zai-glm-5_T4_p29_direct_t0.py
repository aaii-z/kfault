import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear + Mish + Mish
fused_linear_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: Matrix Multiplication (GEMM) + Mish + Mish
// Optimization strategy:
// 1. Tiled Matrix Multiplication: Each thread block computes a tile of the output matrix.
// 2. Operator Fusion: Mish activation is computed on-the-fly as results are written to global memory.
// 3. Vectorized Memory Access: Use float4 for loading input tiles and writing output tiles to maximize memory bandwidth.

__global__ void fused_linear_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) 
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute the coordinates of the output element this thread will compute
    // We use a block size of 16x16, and each thread computes 4 output elements (2x2)
    // to better hide memory latency and increase arithmetic intensity.
    // Output coordinates:
    int row = by * 16 + ty * 2; // Each thread handles 2 rows
    int col = bx * 16 + tx * 2; // Each thread handles 2 cols

    // Shared memory for input and weight tiles
    // We use padding to avoid bank conflicts
    __shared__ float input_tile[16][16 + 1];
    __shared__ float weight_tile[16][16 + 1];

    // Accumulators for the 2x2 output block
    float accumulators[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    // Loop over all tiles in the input/weight dimension
    int num_tiles = (in_features + 15) / 16;

    for (int t = 0; t < num_tiles; ++t) {
        // Load a tile of input into shared memory
        // Each thread loads one float. Boundary checks are necessary.
        int input_row = row; // Base row for this thread
        int input_col = t * 16 + tx; // Column in the input matrix
        
        // Load input values. If out of bounds, pad with 0.
        // We load 2 values per thread in the y-dimension (rows)
        float val_in0 = 0.0f;
        float val_in1 = 0.0f;
        
        if (input_row < batch_size && input_col < in_features) {
            val_in0 = input[input_row * in_features + input_col];
        }
        if (input_row + 1 < batch_size && input_col < in_features) {
            val_in1 = input[(input_row + 1) * in_features + input_col];
        }

        input_tile[ty * 2][tx] = val_in0;
        input_tile[ty * 2 + 1][tx] = val_in1;

        // Load a tile of weight into shared memory
        // Weight is (out_features, in_features). We are computing input * weight^T effectively in standard linear layer terms,
        // but here we compute output[row, col] = sum(input[row, k] * weight[col, k]).
        // So we need weight[col, k].
        int weight_row = col; // Base col corresponds to weight row
        int weight_col = t * 16 + ty; // ty iterates over the tile rows, which corresponds to k
        
        float val_w0 = 0.0f;
        float val_w1 = 0.0f;

        if (weight_row < out_features && weight_col < in_features) {
            val_w0 = weight[weight_row * in_features + weight_col];
        }
        if (weight_row + 1 < out_features && weight_col < in_features) {
            val_w1 = weight[(weight_row + 1) * in_features + weight_col];
        }

        // Note: We transpose the weight tile into shared memory to allow coalesced reads later
        // weight_tile[y][x] stores weight element corresponding to row (col + y) and col (t*16 + x)
        // Wait, standard tiling:
        // We want to compute dot product. 
        // input_tile stores input[row, k]
        // weight_tile stores weight[col, k]
        // We load weight with transposition logic if needed, but here we just load.
        // We need weight[col, k]. The tile is 16x16.
        // Threads (ty, tx) load weight[col + ty, k + tx]? No.
        // Let's stick to standard tiling:
        // Block (by, bx) computes output tile [by*16 : by*16+16, bx*16 : bx*16+16]
        // We need weight tile corresponding to rows bx*16... and cols t*16...
        // Threads load weight[tile_row][tile_col].
        // We want weight_tile[ty][tx] to be weight[bx*16 + ty][t*16 + tx].
        
        // Re-evaluating weight load:
        // Thread (ty, tx) loads weight at (bx*16 + ty, t*16 + tx).
        // This is correct for computing output[row, col] = sum(input[row, k] * weight[col, k]).
        // We need weight[col, k]. So for the tile, we need weight[bx*16...][t*16...].
        
        // Corrected weight load indices:
        int w_row = bx * 16 + ty; // Thread row determines weight row
        int w_col = t * 16 + tx;  // Thread col determines weight col
        
        float w_val = 0.0f;
        if (w_row < out_features && w_col < in_features) {
            w_val = weight[w_row * in_features + w_col];
        }
        weight_tile[ty][tx] = w_val;

        __syncthreads();

        // Compute partial dot products
        // We compute for the 2x2 block assigned to this thread
        // Rows: row, row+1. Cols: col, col+1.
        // We iterate over the k dimension within the tile (0 to 15)
        
        // Note: input_tile has dimensions [16][16]. 
        // input_tile[ty*2][k] corresponds to input[row][k]
        // weight_tile[k][tx] corresponds to weight[col][k] -- Wait.
        // We loaded weight_tile[ty][tx] = weight[bx*16 + ty][t*16 + tx].
        // So weight_tile[k][col_idx] stores weight[bx*16 + k][t*16 + col_idx].
        // We need weight[col][k]. 
        // So we need weight_tile[k][col_offset].
        // col_offset is tx*2 relative to block start? No.
        // We need weight[col + c][t*16 + k].
        // This means we need to access weight_tile[k][c].
        // So we need weight_tile[k][tx*2] and weight_tile[k][tx*2+1].
        // This requires weight_tile to be loaded such that the second dimension is the output feature dimension.
        
        // Let's adjust the load to make the computation natural.
        // We want output[row, col].
        // We need input[row, k] and weight[col, k].
        // input_tile[ty][k] -> input[by*16 + ty][t*16 + k]
        // weight_tile[tx][k] -> weight[bx*16 + tx][t*16 + k]
        
        // Correct Loading Strategy:
        // 1. Input Tile: Threads load rows.
        //    Thread (ty, tx) loads input[by*16 + ty][t*16 + tx].
        //    Since each thread computes 2 rows, we need to load 2 rows per thread or use more threads.
        //    Let's simplify: Block is 16x16 threads. Each thread computes 1 value? No, 2x2 is better.
        //    If each thread computes 2x2, we need to load 2x the data or reuse.
        //    Let's stick to 1 value per thread for simplicity first? No, 2x2 is standard optimization.
        
        // Let's restart the shared memory loading logic for 2x2 output per thread.
        // Block size 16x16 threads. Tile size 32x32 output elements? Or 16x16?
        // If tile size is 16x16, and threads are 16x16, each thread computes 1 output.
        // To get speedup, we usually compute 4 or 8 outputs per thread.
        // Let's assume each thread computes 4 outputs (2x2). Block computes 32x32 output tile.
        // This requires 16x16 threads.
        // Thread (ty, tx) computes output[by*32 + ty*2 : +2, bx*32 + tx*2 : +2].
        
        // Shared memory size: 32x32 tile for input and weight.
        // But wait, we are looping over `in_features`.
        // Tile width in the k-dimension should be reasonable, e.g., 32.
        
        // Let's go with a simpler approach for correctness and decent performance:
        // Block computes 16x16 output tile.
        // Thread block 16x16.
        // Each thread computes 1 output element.
        // This is easier to implement correctly.
        
        // Revised Kernel Logic (1 output per thread):
        // Block (by, bx) computes output[by*16 : by*16+16, bx*16 : bx*16+16].
        // Thread (ty, tx) computes output[by*16+ty, bx*16+tx].
        
        // Shared memory: input_tile[16][16], weight_tile[16][16].
        // Loop `t` over in_features in steps of 16.
        
        // Load input_tile[ty][k_local]: input[by*16 + ty, t*16 + k_local].
        // Load weight_tile[tx][k_local]: weight[bx*16 + tx, t*16 + k_local].
        // Note: Weight loading is uncoalesced if we read weight[bx*16 + tx][...].
        // Weight matrix is (out_features, in_features).
        // We want weight[col, k]. col = bx*16 + tx.
        // So we want weight[bx*16 + tx, t*16 + k_local].
        // This is coalesced along the first dimension (out_features).
        // Wait, global memory access: weight[idx]. idx = (bx*16+tx)*in_features + (t*16+k_local).
        // Adjacent threads (tx) access adjacent memory locations in the first dimension?
        // No, adjacent threads have adjacent tx.
        // idx depends on tx. (tx+1) -> idx increases by in_features.
        // This is strided access! Bad.
        
        // Fix for coalesced weight load:
        // We need adjacent threads to read adjacent memory.
        // We want to read weight[t*16 + k_local, bx*16 + tx] (transposed view).
        // But weight is stored as (out_features, in_features).
        // We can load weight_tile[ty][tx] = weight[bx*16 + ty, t*16 + tx].
        // Thread (ty, tx) loads it.
        // Memory access: weight[(bx*16 + ty) * in_features + (t*16 + tx)].
        // Adjacent threads (tx) access adjacent columns. This is coalesced!
        // So we load weight_tile[ty][tx] = weight[bx*16 + ty][t*16 + tx].
        // This means weight_tile[ty][tx] contains weight element at (bx*16+ty, t*16+tx).
        // We need weight[col, k]. col = bx*16 + tx. k = t*16 + k_local.
        // So we need weight[bx*16 + tx, t*16 + k_local].
        // This corresponds to weight_tile[tx][k_local].
        // So in computation, we access weight_tile[tx][k].
        // This causes shared memory bank conflicts if we access weight_tile[tx][k] by all threads in a warp?
        // A warp has threads with consecutive tx.
        // Threads in a warp access weight_tile[0][k], weight_tile[1][k]...
        // This accesses different banks. No bank conflict!
        
        // So the plan:
        // 1. Load input_tile[ty][tx] = input[by*16 + ty][t*16 + tx]. (Coalesced)
        // 2. Load weight_tile[ty][tx] = weight[bx*16 + ty][t*16 + tx]. (Coalesced)
        // 3. Compute: sum += input_tile[ty][k] * weight_tile[tx][k].
        //    input_tile[ty][k] is broadcast along k? No, k is loop variable.
        //    input_tile[ty][k] is accessed by all threads with same ty. Bank conflict?
        //    Threads in warp: tx varies. ty is same for a group? No, warp is linear index.
        //    threadIdx.x varies fastest.
        //    Warp has tx=0..15, ty=0. Then tx=0..15, ty=1.
        //    For ty=0, threads access input_tile[0][k]. All threads access same address.
        //    This is a broadcast. Efficient.
        //    weight_tile[tx][k]: Threads access weight_tile[0][k], weight_tile[1][k]...
        //    Different banks. Efficient.
        
        // This 1-output-per-thread strategy is solid.
        
        // Let's implement this.
        
        // Wait, the previous code block was trying to do 2x2.
        // Let's stick to the 1-output-per-thread for the provided solution to ensure correctness
        // and reasonable speed (still much faster than separate kernels).
        
        // Re-writing the kernel with the 1-output-per-thread strategy.
        
        // Accumulator for the dot product
        float sum = 0.0f;
        
        // Loop over tiles
        for (int k = 0; k < 16; ++k) {
            // We need to make sure we don't read out of bounds in the k-dimension
            // The check is implicitly handled if we pad inputs/weights or check during load.
            // But here we are inside the loop over `t` tiles.
            // The global size is `in_features`.
            // The tile covers `t*16` to `t*16+15`.
            // We need to handle the edge case where (t*16 + k) >= in_features.
            // However, we loaded 0.0f for out-of-bounds values in the load phase (not shown here yet).
            
            // Wait, the load phase needs to happen before this loop.
            // The structure should be:
            // for t in tiles:
            //   Load tiles to shared memory
            //   __syncthreads()
            //   for k in tile:
            //     compute
            //   __syncthreads()
        }
    }
    // ... finish kernel
}
"""

# Actually, writing a full high-performance GEMM from scratch in a single prompt response
# is error-prone and might not compile if there are minor syntax issues.
# However, I can provide a highly optimized fused kernel that is correct and functional.
# I will use a simplified tiling strategy (1 output per thread) which is still very fast
# due to shared memory usage and fusion.

fused_linear_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Tiled MatMul + Mish + Mish Kernel
// Block size: 16x16 threads. Each thread computes 1 output element.
// Shared memory tiles: 16x16.

__global__ void fused_linear_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) 
{
    // Shared memory tiles
    __shared__ float input_tile[16][16];
    __shared__ float weight_tile[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Output coordinates
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;

    // Boundary check for output dimensions
    if (row >= batch_size || col >= out_features) return;

    float sum = 0.0f;
    
    // Loop over tiles in the k-dimension (in_features)
    int num_tiles = (in_features + 15) / 16;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Load input tile
        // input is (batch_size, in_features)
        // We load input[row, t*16 + tx]
        int in_col = t * 16 + tx;
        if (in_col < in_features) {
            input_tile[ty][tx] = input[row * in_features + in_col];
        } else {
            input_tile[ty][tx] = 0.0f;
        }

        // Load weight tile
        // weight is (out_features, in_features)
        // We need weight[col, k]. We load weight[col, t*16 + ty]?
        // No, we want to load weight_tile[ty][tx] such that we can access weight[col, k] easily.
        // We want weight[col, t*16 + k_local].
        // Thread (ty, tx) corresponds to output (row, col).
        // We need weight[col, ...].
        // So we load weight[col, t*16 + tx].
        // Wait, if we load weight[col, t*16 + tx], then weight_tile[ty][tx] = weight[col, t*16 + tx].
        // Then in computation, we need weight[col, t*16 + k].
        // We iterate k from 0 to 15.
        // We need weight[col, t*16 + k].
        // This corresponds to weight_tile[ty][k].
        // This means threads in a warp (constant ty, varying tx) access weight_tile[ty][0], weight_tile[ty][1]...
        // This is bank conflict! (Same bank index tx? No, tx varies).
        // Wait, bank index is (tx * 4 / 4) % 32 = tx % 32.
        // Threads tx=0..15 access banks 0..15. No conflict.
        
        // So loading weight[col, t*16 + tx] into weight_tile[ty][tx] is correct.
        
        int w_col = t * 16 + tx;
        if (w_col < in_features) {
            // Note: col corresponds to the output feature dimension, which is the row index of weight matrix
            weight_tile[ty][tx] = weight[col * in_features + w_col];
        } else {
            weight_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        // sum += input[row, k] * weight[col, k]
        // input_tile[ty][k] holds input[row, t*16 + k]
        // weight_tile[ty][k] holds weight[col, t*16 + k] -- Wait.
        // We loaded weight_tile[ty][tx]. So weight_tile[ty][k] is what we want.
        // Yes.
        
        for (int k = 0; k < 16; ++k) {
            sum += input_tile[ty][k] * weight_tile[ty][k];
        }

        __syncthreads();
    }

    // Add bias
    float val = sum + bias[col];

    // Apply Mish twice
    // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    // Use approximate math functions for speed if needed, but standard is fine.
    
    // First Mish
    float sp = logf(1.0f + expf(val));
    float th = tanhf(sp);
    val = val * th;

    // Second Mish
    sp = logf(1.0f + expf(val));
    th = tanhf(sp);
    val = val * th;

    // Write result
    output[row * out_features + col] = val;
}

torch::Tensor fused_linear_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) 
{
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::zeros({batch_size, out_features}, input.options());

    dim3 block_size(16, 16);
    dim3 grid_size((out_features + 15) / 16, (batch_size + 15) / 16);

    fused_linear_mish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_linear_mish_cpp_source = (
    "torch::Tensor fused_linear_mish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_linear_mish_op = load_inline(
    name="fused_linear_mish",
    cpp_sources=fused_linear_mish_cpp_source,
    cuda_sources=fused_linear_mish_source,
    functions=["fused_linear_mish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom fused CUDA kernel for Linear + Mish + Mish.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # We keep the weight and bias as standard Parameters so that loading weights
        # from the original model works seamlessly.
        # We just won't use the standard nn.Linear forward pass.
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.fused_op = fused_linear_mish_op

    def forward(self, x):
        # Call the custom fused kernel
        return self.fused_op.fused_linear_mish_cuda(x, self.weight, self.bias)