import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM + GroupNorm + Min + Bias
custom_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Kernel for fused GEMM + GroupNorm + Min + Bias
// Grid: (batch_size)
// Block: 256 threads
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

    extern __shared__ float s_data[];
    float* s_input = s_data; 
    float* s_min = s_data + in_features; // Reuse shared memory for reduction
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    // Pointers for this batch item
    const float* input_ptr = input + batch_idx * in_features;
    
    // 1. Load input to shared memory
    for (int i = tid; i < in_features; i += block_dim) {
        s_input[i] = input_ptr[i];
    }
    __syncthreads();

    // 2. Compute GEMM, GroupNorm, and find Min
    float thread_min = FLT_MAX;
    
    // Iterate over output features
    // Each thread processes multiple output features if out_features > block_dim
    for (int f = tid; f < out_features; f += block_dim) {
        // Compute GEMM dot product
        float val = 0.0f;
        const float* weight_row = weight + f * in_features;
        for (int k = 0; k < in_features; ++k) {
            val += s_input[k] * weight_row[k];
        }
        
        // Compute Group Normalization
        // Group Norm is computed over (N, C/G). 
        // Since we are processing one item at a time (N=1 for this block), 
        // we compute stats over the group.
        // However, we are processing elements in a strided manner (f += block_dim).
        // The threads in a block do NOT hold all elements of a group simultaneously.
        // Therefore, we cannot compute Group Norm here efficiently without collaboration.
        // BUT, wait. The previous logic failed because of shape mismatch.
        // Let's look at the logic again.
        // If we cannot compute GN here, we might need a different approach.
        // However, usually Group Norm is applied per group.
        // If we process one element, we need mean/var of its group.
        // We can precompute mean/var? No, depends on input.
        
        // Correct approach for GN in this setup:
        // We need the sum of elements in the group.
        // We can use atomic adds or a separate pass?
        // Or, we can restructure the loop.
        // If we iterate over groups?
        // We have 512 groups. Block has 256 threads.
        // We can assign 2 groups to each thread? No.
        // We can assign threads to groups.
        // If we process one group at a time.
        // We load the 16 elements of the group.
        // Compute sum/var.
        // Normalize.
        // Update Min.
        // This works if we iterate over groups.
        
        // Let's switch to iterating over groups.
        // This requires changing the loop structure.
        // But wait, the current loop iterates over output features.
        // If we stick to this loop, we can't compute GN correctly in one pass without atomics.
        // Is there a way?
        // Maybe the "Min" operation allows us to skip precise GN? No.
        
        // Let's try to implement the Group-Iterative approach inside the kernel.
        // It's cleaner.
    }
    
    // Refined Logic: Iterate over Groups
    // num_groups = 512. block_dim = 256.
    // Each thread can handle a chunk of groups?
    // Or we iterate `g` from 0 to num_groups.
    // For each group `g`:
    //   Threads in block cooperate to load the `group_size` elements.
    //   Compute GEMM for these elements.
    //   Compute GN stats (sum, sum_sq).
    //   Normalize.
    //   Update Min.
    
    // This requires shared memory for the elements of the current group.
    // group_size = 16. Very small.
    // We can put `group_size` floats in shared mem (or registers).
    // Actually, we can just compute them and reduce.
    
    // Let's restart the kernel logic for the main loop.
    thread_min = FLT_MAX; // Reset
    
    // Shared memory for Group Norm stats
    // We need sum and sum_sq for the current group being processed.
    // Since block processes one group at a time (or multiple?), we need shared mem for reduction.
    // s_min is already allocated. We can use it for sum and sum_sq.
    
    __shared__ float s_sum;
    __shared__ float s_sum_sq;
    
    for (int g = 0; g < num_groups; ++g) {
        // Determine which features belong to this group
        int f_start = g * group_size;
        int f_end = f_start + group_size;
        
        // We need to compute GEMM for features [f_start, f_end).
        // We have 256 threads. Group size is 16.
        // We have more threads than work for this group!
        // We can parallelize the GEMM dot product?
        // No, dot product is over `in_features` (8192).
        // We can parallelize the dot product calculation.
        // Thread t computes partial sum for feature f.
        
        // Strategy:
        // Threads collaborate to compute the `group_size` output elements.
        // Since `group_size` (16) is small, we can assign a warp (or few threads) to each output element?
        // Or better: All threads cooperate to compute one output element, then move to next?
        // That would be too slow (serialization).
        
        // Best Strategy for small group_size and large in_features:
        // Parallelize over output features within the group?
        // We have 16 output features. We have 256 threads.
        // Assign 16 threads to compute the 16 dot products.
        // The other 240 threads are idle? Waste of resources.
        // Better: Assign all threads to compute partial sums for the 16 outputs.
        // Reduction tree.
        
        // Shared memory for partial results.
        // We need 16 outputs.
        // s_partial[16][256]? Too large.
        // s_partial[256][16]? 4096 floats = 16KB. Manageable.
        // Actually, we can use shuffle instructions (warp level).
        
        // Let's use Warp-level parallelism.
        // We have 8 warps (256 threads).
        // We need to compute 16 outputs.
        // Each warp can compute 2 outputs?
        // 8 warps * 2 outputs = 16 outputs. Perfect.
        
        // Warp w computes outputs f_start + 2*w and f_start + 2*w + 1.
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        
        if (warp_id < 8) { // We only need 8 warps
            int f0 = f_start + 2 * warp_id;
            int f1 = f_start + 2 * warp_id + 1;
            
            // Compute partial dot product for f0 and f1
            // Sum over in_features.
            // We split in_features across the 32 lanes of the warp.
            // Each lane sums a chunk of size in_features / 32.
            // in_features = 8192. Chunk = 256.
            
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            
            // Input is in s_input (shared mem).
            // Weight is in global mem.
            // Weight rows: f0 and f1.
            const float* w0 = weight + f0 * in_features;
            const float* w1 = weight + f1 * in_features;
            
            // Loop over the chunk assigned to this lane
            // Chunk start: lane_id * chunk_size
            // Chunk size: (in_features + 31) / 32
            int chunk_size = (in_features + 31) / 32;
            int start_idx = lane_id * chunk_size;
            int end_idx = min(start_idx + chunk_size, in_features);
            
            for (int k = start_idx; k < end_idx; ++k) {
                float in_val = s_input[k];
                sum0 += in_val * w0[k];
                sum1 += in_val * w1[k];
            }
            
            // Warp reduction for sum0 and sum1
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum0 += __shfl_down_sync(0xFFFFFFFF, sum0, offset);
                sum1 += __shfl_down_sync(0xFFFFFFFF, sum1, offset);
            }
            
            // Lane 0 has the result.
            float val0 = 0.0f;
            float val1 = 0.0f;
            if (lane_id == 0) {
                val0 = sum0;
                val1 = sum1;
            }
            
            // Now we have the GEMM output for the group in val0, val1 (held by lane 0 of each warp).
            // We need to compute Group Norm.
            // Group Norm needs Mean and Var of all 16 values in the group.
            // We have values distributed in lane 0 of 8 warps.
            // We need to gather them and compute stats.
            
            // We can use shared memory to share these 16 values.
            // Or we can use another warp shuffle if we organize threads.
            // Since only 8 threads (lane 0 of 8 warps) have data, we can write to shared mem.
            __shared__ float s_group_vals[16]; // 16 elements
            
            if (lane_id == 0) {
                s_group_vals[2 * warp_id] = val0;
                s_group_vals[2 * warp_id + 1] = val1;
            }
            __syncthreads(); // Wait for all warps to write
            
            // Now compute Mean and Var.
            // We can use the first warp (or all threads) to compute this.
            // Let's use the first warp (tid 0-31).
            if (tid < 32) {
                float val = 0.0f;
                if (tid < 16) {
                    val = s_group_vals[tid];
                }
                
                // Compute sum
                float sum = val;
                #pragma unroll
                for (int offset = 8; offset > 0; offset /= 2) {
                    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
                }
                
                // Compute sum_sq
                float sum_sq = val * val;
                #pragma unroll
                for (int offset = 8; offset > 0; offset /= 2) {
                    sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
                }
                
                // Lane 0 has sum and sum_sq
                if (tid == 0) {
                    float mean = sum / group_size;
                    float var = sum_sq / group_size - mean * mean;
                    s_sum = mean;
                    s_sum_sq = var;
                }
            }
            __syncthreads(); // Wait for stats
            
            float mean = s_sum;
            float var = s_sum_sq;
            
            // Normalize
            // We need to apply GN to the values.
            // And update Min.
            // The values are in s_group_vals.
            // Threads can read, normalize, and update Min.
            
            // We have 256 threads. 16 values.
            // All threads can participate in Min update?
            // Or just the first 16?
            // Let's use first 16 threads to normalize and update Min.
            // Then broadcast Min?
            // Actually, we can just update a shared Min.
            
            if (tid < 16) {
                float val = s_group_vals[tid];
                float norm_val = (val - mean) * rsqrtf(var + 1e-5f);
                
                // Scale and shift
                int f = f_start + tid;
                norm_val = norm_val * gn_weight[f] + gn_bias[f];
                
                // Update Min
                // We need a shared Min variable.
                // We can use atomicMin? Or just normal Min.
                // Since we are inside a loop over groups, we need to persist the Min.
                // s_min[0] can store the running Min.
                // But multiple threads writing to s_min[0] is a race.
                // We can use atomicMin on a shared variable? No, atomicMin works on global/shared memory.
                // atomicMin is slow.
                // Better: compute Min among the 16 values, then update the running Min.
                
                s_group_vals[tid] = norm_val; // Store normalized value back
            }
            __syncthreads();
            
            // Reduce Min within the group
            // Use first warp.
            if (tid < 32) {
                 float m = (tid < 16) ? s_group_vals[tid] : FLT_MAX;
                 #pragma unroll
                 for (int offset = 8; offset > 0; offset /= 2) {
                     m = fminf(m, __shfl_down_sync(0xFFFFFFFF, m, offset));
                 }
                 if (tid == 0) {
                     // Update running Min
                     // We need a persistent Min for the block.
                     // s_min[0] is used for reduction at the end.
                     // We can use s_min[0] as running Min.
                     // But careful: s_min is used for final reduction.
                     // We can use s_min[0] here.
                     // Initial value set to FLT_MAX before loop.
                     // We need to read s_min[0], update, write back.
                     // Race condition?
                     // Only tid 0 does this. Safe.
                     s_min[0] = fminf(s_min[0], m);
                 }
            }
        }
        // Implicit sync at end of loop iteration? 
        // No, we need __syncthreads() because we write to s_min[0] and read it next iter.
        // But only tid 0 writes. And next iter tid 0 reads.
        // However, we need to ensure all threads are done with the group before moving to next?
        // Yes, because we reuse s_group_vals.
        __syncthreads();
    }
    
    // 3. Finalize Min and Write Output
    // s_min[0] contains the global Min for this batch item.
    float global_min = s_min[0];
    
    // Write output
    // Output shape: (batch_size, out_features, 1, 1).
    // We need to write global_min + bias[f].
    // Parallelize over out_features.
    
    for (int f = tid; f < out_features; f += block_dim) {
        float b = bias_add[f];
        float out_val = global_min + b;
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

    // Output shape: (batch_size, out_features, 1, 1)
    auto output = torch::zeros({batch_size, out_features, 1, 1}, input.options());

    // Shared memory size:
    // in_features floats for input.
    // 16 floats for group values.
    // 2 floats for sum/var.
    // 1 float for min.
    // Total: in_features + 19 floats.
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
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(
            x, 
            self.gemm.weight, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.bias.view(-1), # Flatten bias to (out_features,) for easy indexing
            self.group_norm.num_groups
        )