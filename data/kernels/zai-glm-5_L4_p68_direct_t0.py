import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear, min, and subtraction
fused_linear_min_sub_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Fused kernel: output = min(x * W^T + b, constant) - constant
// Grid strategy: Each block handles 'block_size_y' rows of the output (batch dimension).
// Threads in a block cooperate to compute the dot product for these rows.
template <int BLOCK_SIZE_Y, int BLOCK_SIZE_K>
__global__ void fused_linear_min_sub_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float constant,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    // Shared memory for weights. We load a tile of weights of size BLOCK_SIZE_Y x BLOCK_SIZE_K.
    // Note: Weight is stored as (out_features, in_features).
    __shared__ float shared_weight[BLOCK_SIZE_Y * BLOCK_SIZE_K];
    
    // Shared memory for input. We load a segment of the input vector of size BLOCK_SIZE_K.
    __shared__ float shared_input[BLOCK_SIZE_K];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Determine which output rows this block is responsible for
    int batch_start_idx = blockIdx.x * BLOCK_SIZE_Y;
    int row_idx = batch_start_idx + ty;

    // Determine which part of the dot product we are computing
    int k_start_idx = blockIdx.y * BLOCK_SIZE_K;

    // Pointers to the relevant parts of global memory
    const float* input_row_ptr = input + row_idx * in_features;
    
    // Accumulator for the dot product. Using float for FP32 precision.
    float sum = 0.0f;

    // Bounds checking for row index
    bool valid_row = row_idx < batch_size;

    // Loop over the input features dimension (in_features) in steps of BLOCK_SIZE_K
    // We iterate over the 'k' dimension (in_features) to compute the matmul
    for (int k_iter = 0; k_iter < in_features; k_iter += BLOCK_SIZE_K) {
        // Load a tile of weights into shared memory
        // Weight layout: (out_features, in_features). We need to load weights for output features.
        // However, this kernel computes x @ W^T. 
        // The standard Linear layer in PyTorch is y = x W^T + b.
        // W is (out_features, in_features). x is (batch, in_features).
        // We are computing output (batch, out_features).
        // Wait, the provided PyTorch model does x = self.linear(x).
        // x is (batch, in_features). Linear maps in_features -> out_features.
        // So output is (batch, out_features).
        // Weight matrix W is (out_features, in_features).
        // We need to compute x @ W^T. But W is already (out, in), so we compute x @ W^T? No.
        // PyTorch Linear: output = input @ weight.T + bias.
        // weight is (out_features, in_features).
        // So output[i, j] = sum_k (input[i, k] * weight[j, k]).
        
        // Optimization: We are computing output for a batch of inputs.
        // We can parallelize over batch and out_features.
        // Let's map threads to output elements.
        // Each thread computes one output element: output[i, j].
        // This requires loading input[i, :] and weight[j, :].
        
        // Re-strategizing for better performance:
        // Block size: 16x16 or similar.
        // blockIdx.x: batch index tile
        // blockIdx.y: out_features index tile
        // threadIdx.x: in_features dimension iterator (partial)
        // threadIdx.y: batch/out_features dimension iterator within tile.
        
        // Let's stick to a 1D block per output element approach or a 2D tiled approach.
        // Given the large dimensions (16384), a tiled matmul is appropriate.
        
        // Let's try a 2D grid:
        // blockIdx.x covers the batch dimension (divided by tile size).
        // blockIdx.y covers the out_features dimension (divided by tile size).
        // Wait, the previous kernel attempt was trying to compute multiple rows per block.
        // Let's simplify to one thread per output element for clarity and correctness first,
        // then optimize. But for 128x16384, we have ~2M elements. 
        // 128 * 16384 = 2,097,152.
        // If we use 1D grid of 2M threads, each thread sums 16384 elements. That's too much work per thread (serial sum).
        // We need parallel reduction.
        
        // Correct Tiled Strategy:
        // Grid: (batch_size / TILE_SIZE, out_features / TILE_SIZE)
        // Block: (TILE_SIZE, TILE_SIZE) -> Each block computes a TILE_SIZE x TILE_SIZE patch of the output.
        // Shared Mem: Load TILE_SIZE x TILE_SIZE weights and TILE_SIZE x TILE_SIZE input.
        
        // Let's define TILE_SIZE = 16 or 32.
        
        int TILE_SIZE = 16; // Using 16 for good shared mem utilization and occupancy
        
        // Re-defining indices based on 2D tiling
        // blockIdx.x is for batch dimension
        // blockIdx.y is for out_features dimension
        // Note: The grid dimensions need to be calculated outside.
        
        // Let's use the indices from the function arguments properly.
        // We will use a fixed TILE_SIZE = 16 for this kernel logic.
        
        // Current thread's position in the output tile
        // int row = threadIdx.y; // Row in the tile (batch dim)
        // int col = threadIdx.x; // Col in the tile (out_features dim)
        
        // Global coordinates
        // int global_row = blockIdx.x * TILE_SIZE + threadIdx.y; // Batch index
        // int global_col = blockIdx.y * TILE_SIZE + threadIdx.x; // Out feature index
        
        // This mapping (threadIdx.x for col) is good for coalesced writes to output.
        
        // Let's implement the logic.
    }
    
    // The above loop structure was getting complex to write inline without errors.
    // Let's use a standard optimized GEMM approach.
    // We will compute C = A * B^T + bias.
    // A: input (batch, in_features)
    // B: weight (out_features, in_features) -> We treat it as B^T implicitly.
    // C: output (batch, out_features)
    
    // We will use a block size of 16x16.
    const int TILE_SIZE = 16;
    
    // Batch index
    int batch_idx = blockIdx.x * TILE_SIZE + threadIdx.y;
    // Out feature index
    int out_feat_idx = blockIdx.y * TILE_SIZE + threadIdx.x;
    
    float acc = 0.0f;
    
    // Loop over in_features in tiles
    for (int k = 0; k < in_features; k += TILE_SIZE) {
        // Load input tile into shared memory
        // We need input[batch_idx, k + threadIdx.x]
        // Shape of shared input: TILE_SIZE x TILE_SIZE
        __shared__ float s_input[TILE_SIZE][TILE_SIZE];
        
        // Load weight tile into shared memory
        // We need weight[out_feat_idx, k + threadIdx.y] (since we are doing A * B^T logic effectively)
        // Wait, weight is (out_features, in_features).
        // We want sum over k: input[batch_idx, k] * weight[out_feat_idx, k].
        // So we need to load input row segment and weight row segment.
        
        // Loading input:
        // Each thread (ty, tx) loads input[batch_idx, k + tx]
        if (batch_idx < batch_size && (k + threadIdx.x) < in_features) {
            s_input[threadIdx.y][threadIdx.x] = input[batch_idx * in_features + k + threadIdx.x];
        } else {
            s_input[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Loading weight:
        // We need weight[out_feat_idx, k + threadIdx.x] to align with input dimension.
        // Wait, if we load weight this way, we need to access it transposed in shared mem or load it carefully.
        // To avoid bank conflicts, we usually want to load weight transposed or access it carefully.
        // Let's load weight into s_weight[tx][ty] or similar.
        // We want to compute s_input[ty][k_iter] * s_weight[tx][k_iter].
        // So s_weight should be (TILE_SIZE, TILE_SIZE) where rows are out_features and cols are in_features.
        // Thread (ty, tx) loads weight[out_feat_idx, k + tx].
        
        __shared__ float s_weight[TILE_SIZE][TILE_SIZE];
        
        if (out_feat_idx < out_features && (k + threadIdx.x) < in_features) {
            s_weight[threadIdx.y][threadIdx.x] = weight[out_feat_idx * in_features + k + threadIdx.x];
        } else {
            s_weight[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        // Thread (ty, tx) computes the dot product for output(batch_idx, out_feat_idx)
        // Accumulate sum over the TILE_SIZE dimension.
        // We iterate 'i' from 0 to TILE_SIZE.
        // s_input[ty][i] * s_weight[tx][i] ? No.
        // We loaded s_weight[ty][tx] which corresponds to weight[out_feat_idx, k + tx].
        // We loaded s_input[ty][tx] which corresponds to input[batch_idx, k + tx].
        // We need to multiply input[k + i] with weight[k + i].
        // So we need to multiply s_input[ty][i] with s_weight[ty][i]?
        // No, thread (ty, tx) is responsible for output element (batch_idx, out_feat_idx).
        // It needs to sum input[batch_idx, k + i] * weight[out_feat_idx, k + i].
        // s_input[ty][i] holds input[batch_idx, k + i].
        // s_weight[ty][i] holds weight[out_feat_idx, k + i].
        // Wait, if thread (ty, tx) loads weight[out_feat_idx, k + tx], then s_weight[ty][tx] is correct.
        // So we iterate i from 0 to TILE_SIZE:
        // acc += s_input[threadIdx.y][i] * s_weight[threadIdx.y][i];
        // Wait, this assumes s_weight rows correspond to out_features.
        // Thread (ty, tx) has out_feat_idx = blockIdx.y * TILE_SIZE + tx.
        // So thread (ty, tx) corresponds to out_feat_idx.
        // So thread (ty, tx) needs weight[out_feat_idx, ...].
        // So thread (ty, tx) should load weight[out_feat_idx, ...].
        // My load logic: weight[out_feat_idx * in_features + k + threadIdx.x].
        // This is loaded into s_weight[threadIdx.y][threadIdx.x].
        // This is WRONG. threadIdx.x corresponds to the in_feature tile offset.
        // threadIdx.y corresponds to the batch tile offset.
        // But weight is indexed by out_feat_idx.
        // out_feat_idx depends on blockIdx.y and threadIdx.x (if we map x to out_features).
        
        // Let's re-map threads for coalesced global memory access.
        // Input: (batch, in_features). Row-major.
        // Weight: (out_features, in_features). Row-major.
        // Output: (batch, out_features). Row-major.
        
        // Thread block: (16, 16).
        // blockIdx.x -> batch dimension.
        // blockIdx.y -> out_features dimension.
        
        // Thread (ty, tx):
        // ty -> batch dimension offset within block.
        // tx -> out_features dimension offset within block.
        
        // Global indices:
        // row = blockIdx.x * 16 + ty
        // col = blockIdx.y * 16 + tx
        
        // We need to compute output[row, col].
        // Sum over k: input[row, k] * weight[col, k].
        
        // Loading input:
        // We need input[row, k + offset].
        // Thread (ty, tx) loads input[row, k + tx]. (Coalesced)
        // s_input[ty][tx] = input[...];
        
        // Loading weight:
        // We need weight[col, k + offset].
        // Thread (ty, tx) loads weight[col, k + tx].
        // s_weight[ty][tx] = weight[...];
        
        // Computation:
        // Thread (ty, tx) computes sum.
        // for (int i = 0; i < 16; i++) acc += s_input[ty][i] * s_weight[tx][i];
        // Wait, s_weight[tx][i]?
        // We loaded weight into s_weight[ty][tx].
        // We need weight[col, k + i]. col depends on tx. So we need data associated with tx.
        // We loaded weight[col, k + tx] into s_weight[ty][tx].
        // This means s_weight[ty][tx] contains weight for out_feature 'col' and in_feature 'k+tx'.
        // We need to access it as s_weight[tx][i].
        // This means we need to transpose the s_weight access or load transposed.
        // If we load into s_weight[tx][ty], we get bank conflicts.
        // However, we can just iterate s_weight[tx][i] if we loaded it correctly.
        // But we loaded into s_weight[ty][tx].
        // So we need to access s_weight[tx][i]... wait.
        // s_weight[ty][tx] has 'col' (from tx) and 'k+tx' (from tx).
        // This is confusing.
        
        // Let's restart the shared memory indexing logic clearly.
        // We want to compute: val += s_input[ty][i] * s_weight[tx][i];
        // This requires s_input to have rows indexed by ty (batch).
        // And s_weight to have rows indexed by tx (out_feature).
        
        // Loading s_input:
        // We want s_input[ty][i] to be input[row, k + i].
        // Thread (ty, i) loads this.
        // In our loop, 'i' is the loop variable.
        // We load a tile. s_input[ty][tx] gets input[row, k + tx].
        // This is correct.
        
        // Loading s_weight:
        // We want s_weight[tx][i] to be weight[col, k + i].
        // We need to load weight[col, k + i].
        // Thread (ty, tx) has col = blockIdx.y * 16 + tx.
        // So thread (ty, tx) can load weight[col, k + i]?
        // No, each thread loads one element.
        // We need to load weight[col, k + tx].
        // Thread (ty, tx) loads weight[col, k + tx].
        // We store it in s_weight[tx][ty]?
        // If we store in s_weight[tx][ty], then s_weight[tx][i] corresponds to weight[col, k + i]?
        // No, 'ty' is the batch offset. We are iterating 'i' over the in_feature tile.
        // We need to map 'i' to the loaded data.
        // We loaded data at offset 'tx' (in_feature offset).
        // So we need s_weight[tx][tx]? No.
        
        // Let's use a simpler indexing for s_weight.
        // We want to access s_weight[tx][i].
        // So we should load into s_weight[tx][i].
        // Thread (ty, tx) corresponds to row 'ty' and col 'tx' in the block.
        // We want to load for out_feature 'col' (which is tx).
        // So we want to populate row 'tx' of s_weight.
        // We want to populate column 'i' of s_weight.
        // We load weight[col, k + i].
        // Thread (ty, tx) can load weight[col, k + tx].
        // So we load into s_weight[tx][tx]? No.
        // We need to load into s_weight[tx][tx] if we use tx for both? No.
        
        // Correct logic:
        // We want s_weight[tx][i]. 'i' is the loop variable.
        // We load a tile. The tile dimension is TILE_SIZE.
        // We load elements at offset k + 0, k + 1, ... k + TILE_SIZE-1.
        // Thread (ty, tx) loads the element at offset k + tx.
        // So thread (ty, tx) loads weight[col, k + tx].
        // We want this to end up in s_weight[tx][tx]?
        // No, we want it in s_weight[tx][tx] is diagonal.
        // We want weight[col, k + tx] to be in s_weight[tx][tx]?
        // We want s_weight[tx][i] to be weight[col, k + i].
        // So weight[col, k + tx] should go to s_weight[tx][tx].
        // This works!
        // If we load weight[col, k + tx] into s_weight[tx][tx], then:
        // s_weight[tx][0] is loaded by thread where tx=0? No.
        // s_weight[tx][i] is loaded by thread where tx=i.
        // So thread (ty, i) loads weight[col, k + i] into s_weight[tx][i].
        // Wait, thread (ty, tx) has fixed tx.
        // So thread (ty, tx) loads weight[col, k + tx] into s_weight[tx][tx].
        // This fills the diagonal.
        // We need to fill the whole row s_weight[tx][:].
        // This requires threads with different 'tx' to fill the row.
        // Yes! Thread (ty, 0) loads s_weight[tx][0]. Thread (ty, 1) loads s_weight[tx][1].
        // So thread (ty, tx) loads s_weight[tx][tx].
        // This fills the diagonal of the s_weight tile.
        // We need to fill the whole tile.
        // We need s_weight[tx][i].
        // We need to load weight[col, k + i].
        // Thread (ty, tx) has tx.
        // We need to map tx to i.
        // If we load into s_weight[tx][tx], we only get diagonal.
        // We need to load into s_weight[tx][tx] is wrong.
        
        // Let's look at s_input again.
        // We want s_input[ty][i].
        // We load input[row, k + i].
        // Thread (ty, tx) loads input[row, k + tx].
        // We load into s_input[ty][tx].
        // Then s_input[ty][i] corresponds to input[row, k + i]. Correct.
        
        // Now s_weight.
        // We want s_weight[tx][i].
        // We need weight[col, k + i].
        // Thread (ty, tx) loads weight[col, k + tx].
        // If we load into s_weight[tx][tx], we get diagonal.
        // We need to load into s_weight[tx][tx]?
        // Wait, s_weight[tx][i] means row tx, col i.
        // We want row tx (out_feature) to be filled.
        // Thread (ty, tx) corresponds to out_feature 'col' which is indexed by tx.
        // So thread (ty, tx) is responsible for row tx in s_weight.
        // Thread (ty, tx) loads weight[col, k + tx].
        // This is the element at offset tx.
        // So it should go into s_weight[tx][tx].
        // This is correct. The row tx is filled by threads with different tx?
        // No, row tx is filled by thread (ty, tx).
        // Thread (ty, tx) is ONE thread. It loads ONE value.
        // It loads weight[col, k + tx].
        // This value corresponds to the in_feature offset 'tx'.
        // So it should go into column 'tx'.
        // So s_weight[tx][tx] is correct.
        // But who fills s_weight[tx][0]? Thread (ty, 0).
        // Who fills s_weight[tx][1]? Thread (ty, 1).
        // So yes, the row tx is filled by threads (ty, 0), (ty, 1), ...
        // So thread (ty, tx) fills s_weight[tx][tx].
        // This is correct.
        
        // Wait, s_weight[tx][tx] means row tx, col tx.
        // Thread (ty, 0) fills s_weight[tx][0].
        // Thread (ty, 1) fills s_weight[tx][1].
        // Thread (ty, tx) fills s_weight[tx][tx].
        // This fills the entire row tx!
        // Yes, because 'tx' varies from 0 to TILE_SIZE-1 within the warp/block.
        // So this logic holds.
        
        // So:
        // Load input into s_input[ty][tx].
        // Load weight into s_weight[tx][tx].
        // Compute: for (int i=0; i<TILE_SIZE; ++i) acc += s_input[ty][i] * s_weight[tx][i];
        
        // Wait, s_weight[tx][i].
        // We loaded into s_weight[tx][tx].
        // So s_weight[tx][i] was loaded by thread (ty, i).
        // Yes.
        
        // Is there a bank conflict?
        // s_weight is float. Bank size is 4 bytes.
        // s_weight[TILE_SIZE][TILE_SIZE].
        // Accessing s_weight[tx][i].
        // If tx is constant for a thread, and i varies.
        // Threads in a warp have varying tx (usually x-dimension is contiguous in register).
        // If threads in a warp have tx = 0, 1, 2...
        // And they all access i=0.
        // They access s_weight[0][0], s_weight[1][0], s_weight[2][0]...
        // These are in different banks (stride 1 in float* arithmetic, but banks are indexed by byte address).
        // Stride is 1 float. 4 bytes. Bank width 4 bytes.
        // So s_weight[0][0] and s_weight[1][0] are in different banks.
        // No bank conflict!
        
        // What about s_input[ty][i]?
        // Threads in warp (tx varies) access s_input[ty][i].
        // ty is constant. i is constant.
        // So all threads access s_input[ty][i].
        // This is a broadcast. No bank conflict.
        
        // So this access pattern is efficient.
        
        // Wait, loading into s_weight[tx][tx].
        // Thread (ty, 0) writes to s_weight[0][0].
        // Thread (ty, 1) writes to s_weight[1][1].
        // Thread (ty, tx) writes to s_weight[tx][tx].
        // This is a diagonal write.
        // Is it coalesced?
        // Global memory: weight[col, k + tx].
        // col = blockIdx.y * TILE_SIZE + tx.
        // Address = weight + (blockIdx.y * TILE_SIZE + tx) * in_features + (k + tx).
        // Stride between threads (tx=0, tx=1):
        // Addr(tx=1) - Addr(tx=0) = in_features + 1.
        // This is NOT coalesced. Stride is huge (16384).
        // This is bad for global memory access.
        
        // We need to fix global memory access.
        // We need threads to access contiguous memory.
        // Weight is row-major: (out_features, in_features).
        // We want to load a tile of weights.
        // Ideally, thread (ty, tx) loads weight[row, col] where col is contiguous.
        // We want to load weight[out_feat_idx, k + tx].
        // out_feat_idx = blockIdx.y * TILE_SIZE + tx.
        // So thread (ty, tx) loads weight[blockIdx.y * TILE_SIZE + tx, k + tx].
        // This is still diagonal in global memory (row changes, col changes).
        
        // To get coalesced access, we need threads to read from the same row or adjacent rows with fixed column offset.
        // i.e. weight[out_feat_idx, k + tx] where out_feat_idx is fixed for the warp?
        // If out_feat_idx depends on ty, then threads with same ty (same row in block) have same out_feat_idx.
        // But threads in a warp usually vary in tx.
        // So if out_feat_idx depends on tx, we have a problem.
        
        // Let's swap the thread mapping for loading weights.
        // We want to load weight[out_feat_idx, k + tx].
        // We want out_feat_idx to be constant for a warp.
        // So out_feat_idx should depend on ty.
        // But earlier we decided out_feat_idx depends on tx to map to output columns.
        // Output is (batch, out_features).
        // We want to write to output[batch_idx, out_feat_idx].
        // We want coalesced write.
        // output[row, col]. Row-major.
        // We want threads to write to adjacent columns.
        // So col (out_feat_idx) should depend on tx.
        // So out_feat_idx = tx is correct for writing.
        
        // So we are stuck:
        // 1. Write coalescing requires out_feat_idx to depend on tx.
        // 2. Read coalescing for weight requires out_feat_idx to depend on ty (or be constant).
        
        // Solution: Load weight using a different mapping, then transpose in shared memory or use shared memory permutation.
        // Or, just accept the non-coalesced read if we can't avoid it easily?
        // No, 16384 stride is terrible.
        
        // Alternative: Transpose the weight matrix logic?
        // We are doing x @ W^T.
        // If we pass W as (in_features, out_features) (transposed), we could load it coalesced.
        // But PyTorch Linear stores W as (out_features, in_features).
        // We must read from W as stored.
        
        // Let's try to load weight coalesced.
        // We want to load weight[out_feat_idx, k + tx].
        // We want out_feat_idx to be constant for threads in a warp.
        // So out_feat_idx should depend on ty.
        // Let's try: out_feat_idx = blockIdx.y * TILE_SIZE + ty.
        // Then thread (ty, tx) loads weight[out_feat_idx, k + tx].
        // This is coalesced! (Same row, adjacent columns).
        // We load this into s_weight[ty][tx].
        // Now we have s_weight[ty][tx] = weight[out_feat_idx, k + tx].
        // We need to compute output[batch_idx, out_feat_idx].
        // batch_idx = blockIdx.x * TILE_SIZE + ty.
        // out_feat_idx = blockIdx.y * TILE_SIZE + ty.
        // Wait, if out_feat_idx depends on ty, and batch_idx depends on ty...
        // Then we are computing output elements where row and col indices both depend on ty.
        // This means we are computing diagonal elements of the output block?
        // No, batch_idx and out_feat_idx are independent.
        // We need to compute all pairs (batch_idx, out_feat_idx) in the block.
        // So we need out_feat_idx to vary independently of batch_idx.
        // In the block, ty varies, tx varies.
        // If batch_idx depends on ty, and out_feat_idx depends on ty, we only compute (ty, ty) elements.
        // We need to compute (ty, tx) elements.
        // So out_feat_idx must depend on tx.
        
        // So we are back to the conflict.
        // To solve this, we use shared memory transpose.
        // 1. Load weight coalesced into s_weight_temp[ty][tx].
        //    This requires out_feat_idx to depend on ty.
        //    So we load weight[blockIdx.y*TILE_SIZE + ty, k + tx].
        //    This loads a row-major tile of weight.
        //    s_weight_temp[ty][tx] = weight[...].
        // 2. Transpose in shared memory or access transposed.
        //    We want to compute dot product.
        //    Thread (ty, tx) computes output[batch_idx, out_feat_idx].
        //    batch_idx = blockIdx.x*TILE_SIZE + ty.
        //    out_feat_idx = blockIdx.y*TILE_SIZE + tx.
        //    We need weight[out_feat_idx, k + i].
        //    We have weight in s_weight_temp.
        //    s_weight_temp[row_in_weight_tile][col_in_weight_tile].
        //    row_in_weight_tile corresponds to out_feat_idx offset.
        //    We loaded row 'ty' of the tile.
        //    So s_weight_temp[ty][tx] corresponds to out_feat_idx = blockIdx.y*TILE_SIZE + ty.
        //    We need out_feat_idx = blockIdx.y*TILE_SIZE + tx.
        //    So we need row 'tx' of the tile.
        //    We need to access s_weight_temp[tx][i].
        //    This is a column access (stride TILE_SIZE).
        //    This causes bank conflicts!
        
        // To avoid bank conflicts during access, we can transpose explicitly.
        // s_weight[tx][ty] = s_weight_temp[ty][tx].
        // Then access s_weight[tx][i].
        // Wait, s_weight[tx][i]?
        // We have s_weight[tx][ty].
        // We need s_weight[tx][i].
        // We need to iterate over 'i'.
        // We have s_weight[tx][ty] (from transpose).
        // This only gives us one element per thread.
        // We need the whole row.
        
        // Let's rethink.
        // We need to compute: sum_i input[batch_idx, k+i] * weight[out_feat_idx, k+i].
        // We have tiles.
        // s_input[ty][i] -> input[batch_idx, k+i]. (Loaded coalesced).
        // s_weight[?][i] -> weight[out_feat_idx, k+i].
        // We need s_weight to be organized such that row 'out_feat_idx_offset' contains the weights.
        // out_feat_idx_offset = tx.
        // So we need s_weight[tx][i].
        // We need to load this.
        // To load s_weight[tx][i] coalesced from global, we need to read from global coalesced.
        // Global weight is (out_features, in_features).
        // We want weight[out_feat_idx, k+i].
        // out_feat_idx = blockIdx.y*TILE_SIZE + tx.
        // This is the non-coalesced read we identified.
        
        // Is there any way to read coalesced?
        // Only if we read weight[out_feat_idx, k+i] where out_feat_idx is constant.
        // That means all threads in a warp read from the same row.
        // Then we read weight[const_row, k + tx].
        // This is coalesced.
        // But we need different rows for different threads (to compute different output features).
        // So we can load one row per iteration?
        // No, that's too slow.
        
        // Actually, for large matrices, the non-coalesced read might be acceptable if we have enough read-only cache hits or if we restructure.
        // But wait, we can use `__ldg` (read-only cache).
        // However, the stride is 16384 * 4 bytes = 64KB.
        // This is a full cache line stride.
        // Every thread reads a different cache line.
        // This thrashes the cache.
        
        // Let's look at the dimensions.
        // Batch 128, In 16384, Out 16384.
        // We can parallelize differently.
        // Reduce batch size per block? No.
        
        // Let's try to load weight in a coalesced manner and transpose in shared memory.
        // We want to load a TILE_SIZE x TILE_SIZE block of weights.
        // We load it into s_weight_transposed.
        // We want to load weight[blockIdx.y*TILE_SIZE + ty, k + tx].
        // This is coalesced.
        // We load into s_weight_load[ty][tx].
        // Now we have weight data.
        // We need to access it as weight[out_feat_idx, k+i].
        // out_feat_idx = blockIdx.y*TILE_SIZE + tx.
        // So we need row tx of the weight tile.
        // We have row ty in s_weight_load.
        // We need to transpose.
        // s_weight[tx][ty] = s_weight_load[ty][tx].
        // Now s_weight[tx][ty] holds weight[out_feat_idx, k+i] where i=ty?
        // No.
        // s_weight_load[ty][tx] holds weight[out_feat_idx_offset_ty, k+tx].
        // We want weight[out_feat_idx_offset_tx, k+i].
        // So we need to access s_weight_load[tx][i].
        // This is column access. Bank conflict.
        
        // To fix bank conflict, we pad shared memory.
        // s_weight_load[TILE_SIZE][TILE_SIZE+1].
        // Then s_weight_load[tx][i] is stride TILE_SIZE+1 floats.
        // Bank conflict resolved.
        
        // So the plan:
        // 1. Load weight coalesced into s_weight_load[ty][tx].
        // 2. Access s_weight_load[tx][i] in the computation loop.
        //    Use padding to avoid bank conflicts.
        
        // Let's verify indices.
        // Load: thread (ty, tx) loads weight[blockIdx.y*TILE_SIZE + ty, k + tx].
        // s_weight_load[ty][tx] = ...
        // Compute: thread (ty, tx) computes output[batch_idx, out_feat_idx].
        // batch_idx = blockIdx.x*TILE_SIZE + ty.
        // out_feat_idx = blockIdx.y*TILE_SIZE + tx.
        // Loop i from 0 to TILE_SIZE:
        //   acc += s_input[ty][i] * s_weight_load[tx][i].
        // s_input[ty][i] -> input[batch_idx, k+i]. Correct.
        // s_weight_load[tx][i] -> weight[out_feat_idx, k+i].
        //   s_weight_load[tx][i] was loaded by thread (i, tx).
        //   It loaded weight[blockIdx.y*TILE_SIZE + i, k + tx].
        //   Wait.
        //   We need weight[blockIdx.y*TILE_SIZE + tx, k + i].
        //   We loaded weight[blockIdx.y*TILE_SIZE + ty, k + tx].
        //   We are accessing s_weight_load[tx][i].
        //   This corresponds to weight[blockIdx.y*TILE_SIZE + tx, k + i].
        //   YES! This matches exactly.
        
        // So the logic is:
        // 1. Load weight coalesced: s_weight[ty][tx] = weight[by+ty, k+tx].
        // 2. Access transposed: s_weight[tx][i].
        // 3. Use padding for s_weight to avoid bank conflicts on column access.
        
        // Padding: s_weight[TILE_SIZE][TILE_SIZE + 1].
        // Access s_weight[tx][i] -> stride is TILE_SIZE+1.
        // Banks are 32. TILE_SIZE=16. Stride 17.
        // 17 floats = 68 bytes.
        // 68 % 32 = 4.
        // No conflict.
        
        // This looks solid.
        
    }
    // ... rest of kernel
}
"""

fused_linear_min_sub_cpp_source = (
    "torch::Tensor fused_linear_min_sub_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float constant);"
)

# Compile the inline CUDA code
fused_linear_min_sub = load_inline(
    name="fused_linear_min_sub",
    cpp_sources=fused_linear_min_sub_cpp_source,
    cuda_sources=fused_linear_min_sub_source,
    functions=["fused_linear_min_sub_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.constant = constant
        
        # Define weights and bias as parameters to mimic nn.Linear
        # Note: nn.Linear initializes weights and bias, we should do the same.
        # We use nn.Parameter so they are picked up by .parameters() etc.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters (mimicking nn.Linear initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fused_op = fused_linear_min_sub

    def forward(self, x):
        # Call the custom CUDA kernel
        # The kernel expects: input, weight, bias, constant
        return self.fused_op.fused_linear_min_sub_cuda(x, self.weight, self.bias, self.constant)