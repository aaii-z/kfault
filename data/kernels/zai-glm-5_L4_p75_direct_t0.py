import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM + GroupNorm + Min + Bias
custom_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Kernel for fused GEMM + GroupNorm + Min + Bias
// Grid: (batch_size, num_groups)
// Block: 256 threads (8 warps)
__global__ void fused_gemm_gn_min_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    const float* __restrict__ bias_add,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups,
    int group_size) {

    // 1. GEMM
    // Each block computes one row of the GEMM output (out_features elements)
    // We use a 1D block to compute the dot product for one output element?
    // No, the block is responsible for the whole row? No, that's too much work.
    // Let's map block.y to batch index, and block.x to output feature index?
    // If out_features is 8192, we need 8192 blocks per batch item? That's too many blocks.
    // Better approach: Each block computes one output row.
    // Threads in the block cooperate to compute the dot product.
    // However, we also need to do GroupNorm.
    // GroupNorm requires statistics over (N, C/G) or (C/G).
    // If we compute GEMM row by row, we don't have the full column data for GroupNorm.
    // Wait, the Min operation reduces the row to a single scalar.
    // So the output is (batch_size, 1, 1, 1) after squeeze? No, bias is (1, out_features, 1, 1).
    // Wait, bias shape is (1, out_features, 1, 1).
    // x = x + self.bias. x is (batch_size, 1). bias is (1, out_features, 1, 1).
    // Broadcasting: (batch_size, 1) + (1, out_features, 1, 1) -> (batch_size, out_features, 1, 1).
    // Wait, torch.min(x, dim=1) on (batch_size, out_features) results in (batch_size, 1).
    // Then adding bias (1, out_features, 1, 1) broadcasts to (batch_size, out_features, 1, 1).
    // This implies the output shape is (batch_size, out_features, 1, 1).
    // This means the Min operation is NOT reducing to a scalar per batch.
    // It reduces to (batch_size, 1). Then bias adds back the 'out_features' dimension?
    // Let's check broadcasting rules.
    // x shape: (batch_size, 1)
    // bias shape: (1, out_features, 1, 1)
    // Result shape: (batch_size, out_features, 1, 1)
    // This means for each batch item, we have a vector of size 1.
    // We add a bias vector of size out_features.
    // The result is a matrix where each row is the scalar repeated out_features times, plus bias.
    // Wait, x is (batch_size, 1). bias is (1, out_features, 1, 1).
    // x + bias -> (batch_size, out_features, 1, 1).
    // x is broadcast to (batch_size, out_features, 1, 1) by repeating the scalar out_features times.
    // So the output is: min_val repeated out_features times + bias_vector.
    // This is equivalent to: output[b, f, 0, 0] = min_val[b] + bias[0, f, 0, 0].
    
    // So the computation is:
    // 1. GEMM: (batch_size, in_features) x (in_features, out_features) -> (batch_size, out_features)
    // 2. GroupNorm: Normalize (batch_size, out_features). Output (batch_size, out_features).
    // 3. Min: Reduce dim 1. Output (batch_size, 1).
    // 4. Bias: Broadcast add. Output (batch_size, out_features, 1, 1).

    // Optimization Strategy:
    // We need to compute the full GEMM and GroupNorm before we can find the Min.
    // We can fuse GEMM and GroupNorm.
    // Then a separate kernel for Min and Bias? Or fuse everything?
    // Fusing everything is hard because Min requires the full result of GN.
    // However, we can do "Online Softmax" style Min?
    // Min is an associative operator. We can compute Min locally and update.
    // But we need the Min of the *entire* row (out_features elements).
    // If we compute GEMM+GN in blocks, we can maintain a running Min.
    // But we need the final Min to produce the output.
    // And the output is (batch_size, out_features).
    // So we need to write the final result.
    // If we compute GEMM+GN block-by-block, we can update a running Min.
    // Once the row is finished, we have the global Min.
    // Then we need to write the output: min_val + bias.
    // But we need to write it for *all* out_features.
    // This means we need to either:
    // a) Store the GEMM+GN output to global memory, then reduce.
    // b) Recompute or keep values in registers/shared memory?
    // Given out_features=8192, we can't keep a row in shared memory for all batches.
    // But we can process one batch item at a time? No, we want high throughput.
    // We can use a "split-K" approach for GEMM, but for Min?
    // Actually, we can fuse GEMM and GroupNorm.
    // Let's look at the GroupNorm. It needs Mean and Var of the group.
    // Group size = 16. Num groups = 512.
    // We can compute GEMM output for a group, then normalize it immediately.
    // This keeps data in registers/shared memory.
    // Then we update the running Min.
    // Once all groups are processed, we have the global Min for that row.
    // Then we need to write the output.
    // Output is (batch_size, out_features).
    // We need to write: min_val + bias[f].
    // We have min_val. We have bias.
    // We can write this output.
    // But wait, we discarded the GEMM+GN values?
    // No, the output depends on the Min, not the individual GEMM+GN values.
    // The output is min_val + bias.
    // So we DON'T need to store the GEMM+GN output!
    // We just need the Min of the GEMM+GN output.
    // This is a massive optimization. We don't write the large intermediate tensor.
    
    // Plan:
    // Each block processes one batch item (one row).
    // Block size = 256 threads.
    // We iterate over the output features (8192) in chunks.
    // For each chunk:
    // 1. Compute GEMM part (dot product).
    // 2. Compute Group Normalization.
    //    - Need Mean/Var of the group.
    //    - Group size is 16.
    //    - We can compute the whole group (16 elements) within the block?
    //    - 8192 / 16 = 512 groups.
    //    - 256 threads. 256 / 16 = 16 groups handled in parallel?
    //    - Or 256 threads cooperate on one group? No, 16 is small.
    //    - Let's assign one thread per output element?
    //    - 256 threads -> 256 output elements at a time.
    //    - 256 / 16 = 16 groups.
    //    - We need Mean/Var for these 16 groups.
    //    - We can compute Mean/Var for 16 elements (one group) using shared memory reduction within a warp?
    //    - Group size 16. A warp is 32 threads.
    //    - We can map 2 groups to 1 warp? Or 1 group to 1 half-warp?
    //    - Let's try: 1 warp handles 2 groups (32 threads, 2x16 elements).
    //    - 256 threads = 8 warps.
    //    - 8 warps * 2 groups/warp = 16 groups.
    //    - We process 16 groups (256 elements) at a time.
    //    - We need to iterate 8192 / 256 = 32 times.
    //    - In each iteration:
    //      a. Compute GEMM for 256 elements.
    //         - Input is (1, in_features). Weight is (out_features, in_features).
    //         - We need to access input and weight.
    //         - Input vector is shared. Weight columns are different.
    //         - We can load input into shared memory?
    //         - in_features = 8192. Too large for shared mem?
    //         - Shared mem is 48KB. 8192 floats = 32KB. Yes, it fits!
    //         - We can load the whole input vector to shared memory.
    //         - This saves global memory access for the input.
    //         - Weight matrix is huge. We stream it.
    //      b. Compute GroupNorm for these 256 elements.
    //         - We have 16 groups.
    //         - Compute sum and sum-of-squares for each group.
    //         - Use warp reduction.
    //         - Normalize.
    //      c. Update running Min.
    //         - Find min among the 256 normalized values.
    //         - Update the block-wide running Min.
    //         - Use shared memory reduction.
    // 3. After the loop, we have the global Min for the row.
    // 4. Write output.
    //    - Output is (batch_size, out_features).
    //    - We need to write min_val + bias[f] for all f.
    //    - We can do this in parallel. 256 threads write 256 values.
    //    - We need to iterate again? Or just write?
    //    - We just need to write. We don't need to recompute.
    //    - We have min_val in registers (or shared mem).
    //    - We have bias pointer.
    //    - We can write directly.

    // Mapping:
    // blockIdx.x: batch index (0 to batch_size-1)
    // We launch batch_size blocks.
    // Block size: 256.
    
    // Shared Memory allocation:
    // 1. Input vector: in_features floats. (32KB)
    // 2. Partial sums for Min reduction.
    // 3. Partial sums for GroupNorm (Mean/Var).
    // Total shared memory: 32KB + small amount. Fits in 48KB.

    extern __shared__ float s_data[];
    float* s_input = s_data; 
    // We need space for reduction. 
    // Let's put reduction buffer after input.
    // s_input is in_features.
    // Reduction buffer: we need to reduce Min.
    // We can reuse space or use separate small space.
    // Let's just use a small separate buffer or end of s_input.
    // Actually, we need to compute Group Norm stats.
    // We have 16 groups in flight. We need sum and sum_sq for each.
    // 16 * 2 = 32 floats.
    // We can put them in shared memory.
    // s_gn_stats: 32 floats.
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    // Pointers for this batch item
    const float* input_ptr = input + batch_idx * in_features;
    // Weight is (out_features, in_features). Row-major.
    // We need column 'f' of weight to compute output 'f'.
    // Weight[f, :] is the row f.
    // Dot product: input . weight_row_f.
    
    // Load input to shared memory
    // Cooperative load
    for (int i = tid; i < in_features; i += block_dim) {
        s_input[i] = input_ptr[i];
    }
    __syncthreads();

    // Initialize Min
    float thread_min = FLT_MAX;

    // Iterate over output features in chunks of 256
    // We process 'out_features' total.
    // Chunk size = 256.
    int chunk_size = 256;
    int num_chunks = (out_features + chunk_size - 1) / chunk_size;

    // We need to handle Group Norm.
    // We have 16 groups per chunk.
    // We need shared memory for Group Norm stats.
    // 16 groups * 2 (sum, sum_sq) = 32 floats.
    // We can use the space after s_input.
    // But s_input is 8192 floats.
    // We can just declare a small buffer in shared mem.
    // extern __shared__ float s_data[]; -> s_input uses it.
    // We need to dynamically allocate enough.
    // Size = in_features * sizeof(float) + 256 * sizeof(float) (for reduction/gn).
    // Let's assume we have enough.
    float* s_gn_stats = s_input + in_features; 
    // s_gn_stats[0..15] = sum, s_gn_stats[16..31] = sum_sq.
    
    // Optimization: 1 warp handles 2 groups.
    // Warp size 32.
    // Lane 0-15 handles group 0. Lane 16-31 handles group 1.
    // We have 8 warps. Handles 16 groups.
    // This matches our chunk of 256 elements (16 groups * 16 elements).
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Loop over chunks
    for (int c = 0; c < num_chunks; ++c) {
        int f_base = c * chunk_size;
        int f = f_base + tid;
        
        // 1. Compute GEMM
        float val = 0.0f;
        if (f < out_features) {
            // Compute dot product for output element f
            // Weight row: weight + f * in_features
            const float* weight_row = weight + f * in_features;
            // Manual dot product loop
            // Unrolling? Vectorizing?
            // in_features = 8192.
            // We can try to use intrinsic or just loop.
            // Since input is in shared memory, access is fast.
            // Weight is global.
            for (int k = 0; k < in_features; ++k) {
                val += s_input[k] * weight_row[k];
            }
        }
        
        // 2. Group Normalization
        // We need to normalize 'val'.
        // Group index for 'f': f / group_size.
        // We are processing 16 groups at once.
        // Group indices relative to this chunk: 0 to 15.
        // Global group index: (f / group_size).
        
        // We need to compute Mean and Var for each group.
        // But wait, GroupNorm is defined over the group.
        // We have computed ONE element per group (since we process 256 elements, 16 groups, 16 elements/group).
        // Wait, group_size = 16.
        // In one chunk of 256, we have 16 groups.
        // Each thread holds one element of a group.
        // We need to compute Mean and Var of the group.
        // We have all 16 elements of the group distributed in a warp (or half-warp).
        // We need to reduce them to get Sum and SumSq.
        
        // Let's organize the reduction.
        // We have 16 groups.
        // Warp 'w' (0-7) handles groups 2*w and 2*w+1.
        // Threads in warp: 0-31.
        // Threads 0-15: group 2*w.
        // Threads 16-31: group 2*w+1.
        
        // We need to compute sum and sum_sq for the group.
        // We can use warp shuffle.
        // But we need to separate the two groups in the warp.
        // Mask for group 2*w: 0xFFFF (lanes 0-15).
        // Mask for group 2*w+1: 0xFFFF0000 (lanes 16-31).
        // Actually, simpler: just use __shfl_down_sync.
        
        // Wait, we only have ONE value per thread.
        // We need to sum the 16 values of the group.
        // The 16 values are in lanes 0-15 (or 16-31).
        // We can do a reduction within the active lanes.
        
        // However, we need to handle the case where f >= out_features.
        // If f >= out_features, val is 0, but shouldn't contribute to stats?
        // Actually, if f >= out_features, we are padding.
        // We should handle boundaries.
        // But out_features is 8192, multiple of 256. So no boundary issues in loop.
        
        // Compute sum and sum_sq for the group.
        // We need to identify which group we belong to.
        // tid is the global thread index in block.
        // local_f = tid % 16. (Element index within group).
        // local_group = tid / 16. (Group index within chunk, 0-15).
        
        // Wait, if we use the warp approach:
        // Warp 0 (tid 0-31) handles groups 0 and 1.
        // tid 0-15 -> group 0.
        // tid 16-31 -> group 1.
        // This mapping is consistent.
        
        // We need to compute sum of val for the group.
        // We can use warp reduction.
        // For group 0 (lanes 0-15):
        // sum = reduce(val, lanes 0-15).
        // We can use __shfl_down_sync(0xFFFF, val, ...).
        
        float sum = val;
        float sum_sq = val * val;
        
        // Identify group mask
        unsigned int mask = (lane_id < 16) ? 0xFFFF : 0xFFFF0000;
        
        // Warp reduction
        // We need to reduce 16 values.
        // Step 1: reduce 16 to 8.
        // Step 2: reduce 8 to 4.
        // ...
        // We can use __shfl_down_sync.
        // Note: we must ensure we don't mix values from different groups.
        // The mask handles this.
        
        // Reduction for sum
        #pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        // Reduction for sum_sq
        #pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(mask, sum_sq, offset);
        }
        
        // Now lane 0 (and 16) have the sum and sum_sq for their respective groups.
        // Broadcast to all lanes in the group?
        // Or just compute mean/var in lane 0 and broadcast.
        // We need to normalize 'val' in all threads.
        // So we need mean and var in all threads.
        
        // Broadcast sum and sum_sq from lane 0 (and 16).
        // Lane 0 has sum for group 0. Lane 16 has sum for group 1.
        // Source lane for group 0 is 0. Source lane for group 1 is 16.
        // We can use __shfl_sync.
        // For group 0: source is 0.
        // For group 1: source is 16.
        // Source lane index = (lane_id >= 16) ? 16 : 0;
        int src_lane = (lane_id >= 16) ? 16 : 0;
        
        float group_sum = __shfl_sync(mask, sum, src_lane);
        float group_sum_sq = __shfl_sync(mask, sum_sq, src_lane);
        
        // Compute mean and var
        float mean = group_sum / group_size;
        float var = group_sum_sq / group_size - mean * mean;
        
        // Normalize
        // Using epsilon for numerical stability
        float norm_val = (val - mean) * rsqrtf(var + 1e-5f);
        
        // Scale and shift (GN learnable parameters)
        // gn_weight and gn_bias are (out_features).
        // We need to access them.
        int global_group_idx = (f_base + tid) / group_size; // This is messy.
        // Better: global feature index f.
        // We already computed f.
        // But we need to be careful with f.
        // f is valid only if f < out_features.
        // We computed val only if f < out_features.
        // If f >= out_features, val is 0, sum is 0, etc.
        // But we should not write output for f >= out_features.
        
        // We need gn_weight[f] and gn_bias[f].
        // We can fetch them.
        // But wait, we need to apply them *after* normalization.
        // norm_val = norm_val * gn_weight[f] + gn_bias[f].
        
        if (f < out_features) {
            float w = gn_weight[f];
            float b = gn_bias[f];
            norm_val = norm_val * w + b;
            
            // Update Min
            // We need the Min of the normalized values.
            thread_min = fminf(thread_min, norm_val);
        }
    }
    
    // After loop, we have thread_min.
    // We need to reduce across the block to find the global Min for this batch item.
    // Block reduction.
    // Use shared memory.
    float* s_min = s_gn_stats; // Reuse space
    s_min[tid] = thread_min;
    __syncthreads();
    
    // Reduce in shared memory
    // Assume block size is power of 2 (256).
    for (int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
        }
        __syncthreads();
    }
    
    // s_min[0] has the global Min for this batch item.
    float global_min = s_min[0];
    
    // 3. Write output
    // Output shape: (batch_size, out_features, 1, 1).
    // We need to write global_min + bias[f].
    // We can parallelize this.
    // Each thread writes one element.
    // Iterate over out_features again.
    // We can reuse the loop structure or just write.
    // Since we are in the same kernel, we can just write.
    // We need to write to output tensor.
    // Output is 4D. We write to output[batch_idx, f, 0, 0].
    
    // Optimization: We can write directly.
    // We don't need to loop if we have enough threads?
    // We have 256 threads. out_features = 8192.
    // We need to write 8192 values.
    // We can iterate.
    
    for (int f = tid; f < out_features; f += block_dim) {
        // Bias is (1, out_features, 1, 1).
        // bias_add is a pointer to the tensor data.
        // bias_add[0, f, 0, 0] -> bias_add[f].
        float b = bias_add[f];
        float out_val = global_min + b;
        
        // Output tensor: (batch_size, out_features, 1, 1).
        // output[batch_idx, f, 0, 0] = out_val.
        // Output pointer arithmetic:
        // output + (batch_idx * out_features + f) * 1 * 1.
        output[batch_idx * out_features + f] = out_val;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor bias_add,
    int num_groups) {

    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0); // weight is (out_features, in_features)
    int group_size = out_features / num_groups;

    auto output = torch::zeros({batch_size, out_features, 1, 1}, input.options());

    // Shared memory size:
    // in_features floats for input.
    // 256 floats for reduction/stats.
    size_t shared_mem_size = in_features * sizeof(float) + 256 * sizeof(float);

    // Launch kernel
    int block_size = 256;
    int num_blocks = batch_size; // One block per batch item

    fused_gemm_gn_min_bias_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        bias_add.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        num_groups,
        group_size
    );

    return output;
}
"""

custom_kernel_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor bias_add,
    int num_groups);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=custom_kernel_cpp_source,
    cuda_sources=custom_kernel_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        # We need to maintain the parameters to pass to the kernel
        # We create a Linear layer just to hold the weight, but we won't use its forward pass
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Initialize the custom op
        self.fused_op = fused_op

    def forward(self, x):
        # We extract the weights from the modules
        # Note: nn.Linear stores weight as (out_features, in_features), which matches our kernel
        # nn.GroupNorm stores weight and bias as (num_channels).
        
        # Call the custom CUDA kernel
        # We need to squeeze the bias to match the expected shape in kernel or handle in kernel
        # bias shape is (1, out_features, 1, 1). Kernel expects flat or specific access.
        # The kernel expects bias_add to be accessible via index f.
        # bias_add[f] works if it's contiguous.
        
        return self.fused_op.fused_op_cuda(
            x, 
            self.gemm.weight, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.bias, 
            self.group_norm.num_groups
        )