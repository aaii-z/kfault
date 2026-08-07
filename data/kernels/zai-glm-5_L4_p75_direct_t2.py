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

    // Shared memory layout:
    // 1. Input vector (in_features)
    // 2. Group values (group_size) - reused for GN stats and Min reduction
    // 3. Group Norm stats (mean, var)
    // 4. Running Min
    extern __shared__ float s_data[];
    float* s_input = s_data; 
    float* s_group_vals = s_data + in_features;
    float* s_gn_stats = s_group_vals + group_size; // [mean, var]
    float* s_running_min = s_gn_stats + 2; 
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    // Pointers for this batch item
    const float* input_ptr = input + batch_idx * in_features;
    
    // 1. Load input to shared memory
    for (int i = tid; i < in_features; i += block_dim) {
        s_input[i] = input_ptr[i];
    }
    
    // Initialize running min
    if (tid == 0) {
        s_running_min[0] = FLT_MAX;
    }
    __syncthreads();

    // 2. Iterate over Groups
    // We have 512 groups. block_dim = 256.
    // We process groups in a strided loop to maximize parallelism.
    // Each thread processes one output feature (one element of a group).
    // We need to synchronize per group.
    
    // To handle synchronization efficiently with strided processing:
    // We can process `block_dim` groups at a time?
    // No, group_size is 16. block_dim is 256.
    // If we process one group at a time, we use 16 threads. 240 threads idle.
    // If we process multiple groups at once?
    // We can process `block_dim / group_size` groups in parallel?
    // 256 / 16 = 16 groups in parallel.
    // This fits perfectly.
    
    int groups_per_step = block_dim / group_size; // 16 groups
    int total_steps = (num_groups + groups_per_step - 1) / groups_per_step;
    
    for (int step = 0; step < total_steps; ++step) {
        int g_start = step * groups_per_step;
        int g_end = min(g_start + groups_per_step, num_groups);
        
        // Each thread is responsible for one element in one of the groups.
        // Thread t calculates feature f = g * group_size + offset.
        // t = g * 16 + offset.
        // g = t / 16. offset = t % 16.
        
        float thread_val = FLT_MAX; // GEMM result for this thread
        int feature_idx = -1;
        int group_idx = -1;
        
        // Calculate which group and offset this thread handles
        int local_group_id = tid / group_size; // 0 to 15
        int local_offset = tid % group_size;   // 0 to 15
        
        int current_group = g_start + local_group_id;
        
        if (current_group < g_end) {
            feature_idx = current_group * group_size + local_offset;
            group_idx = current_group;
            
            // Compute GEMM dot product for this feature
            float sum = 0.0f;
            const float* weight_row = weight + feature_idx * in_features;
            
            // Parallelize dot product?
            // No, if we parallelize dot product, we need reduction.
            // Here we have 256 threads. We are computing 16 groups * 16 features = 256 features.
            // So each thread computes ONE output feature.
            // This means each thread computes a FULL dot product of size 8192.
            // This is heavy but maximizes parallelism.
            
            // Optimization: Loop unrolling or vectorization?
            // Just a simple loop for now.
            for (int k = 0; k < in_features; ++k) {
                sum += s_input[k] * weight_row[k];
            }
            thread_val = sum;
        }
        
        // Now we have computed the GEMM results for 16 groups in parallel.
        // Results are in registers (thread_val).
        
        // We need to compute Group Norm for each group.
        // This requires Mean and Var of the 16 values in each group.
        // We can use Warp Shuffle (Warp Reduction) for this.
        // Threads 0-15 form the first group? No.
        // Threads handling group 0 are: 0, 1, ..., 15. (Indices 0-15)
        // Threads handling group 1 are: 16, ..., 31. (Indices 16-31)
        // This is a contiguous mapping. Perfect for Warp Shuffles.
        
        // We have 16 groups. Each group uses 16 threads.
        // We can use `__shfl_down_sync` with a mask.
        // Mask for group 0: threads 0-15. Mask = (1 << 16) - 1 = 0xFFFF.
        // Mask for group 1: threads 16-31. Mask = 0xFFFF << 16.
        // ...
        // Mask for group 15: threads 240-255.
        
        // However, warp shuffle works on 32 threads.
        // Group 0 (threads 0-15) is in Warp 0.
        // Group 1 (threads 16-31) is in Warp 0.
        // We need to separate them.
        // We can use `__shfl_sync` with full mask but careful indexing.
        // Or just use Shared Memory reduction which is easier for non-power-of-2 or segmented reductions.
        // Actually, 16 is power of 2.
        
        // Let's use Shared Memory for reduction to be safe and clear.
        // We write thread_val to shared memory.
        s_group_vals[tid] = thread_val;
        __syncthreads();
        
        // Now compute Mean/Var for each group.
        // We can assign one thread per group to compute stats?
        // Or the first thread of each group (local_offset == 0).
        
        if (local_offset == 0 && current_group < g_end) {
            // This thread is the "leader" for the group.
            // It reads the 16 values from shared memory.
            // Note: values are at indices [local_group_id * 16, ... + 15]
            // Wait, s_group_vals is indexed by tid.
            // tid = local_group_id * 16 + local_offset.
            // So values for this group are at s_group_vals[local_group_id * 16 ...].
            
            float sum = 0.0f;
            float sum_sq = 0.0f;
            int base_idx = local_group_id * group_size;
            
            for (int i = 0; i < group_size; ++i) {
                float val = s_group_vals[base_idx + i];
                sum += val;
                sum_sq += val * val;
            }
            
            float mean = sum / group_size;
            float var = sum_sq / group_size - mean * mean;
            
            // Store stats to shared memory
            // We need to store stats for this group.
            // We can reuse s_gn_stats. Since we process groups in parallel, we need an array.
            // s_gn_stats[local_group_id * 2] = mean;
            // s_gn_stats[local_group_id * 2 + 1] = var;
            // But we declared s_gn_stats as size 2.
            // We need to allocate enough space.
            // Let's adjust shared memory allocation.
            // For now, let's just use the first 2 slots and iterate? No, that serializes.
            // Let's assume we allocated enough: s_gn_stats[groups_per_step * 2].
            // Re-declare pointer offset logic.
            // Actually, we can just compute and keep in registers? No, other threads need it.
            // We can broadcast using shfl?
            // Leader thread broadcasts to the other 15 threads in the group.
            
            // Broadcast mean and var
            // Leader has lane_id = local_offset = 0.
            // Members have local_offset 1-15.
            // We can use __shfl_sync.
            
            // But first, let's write to a small shared array for simplicity.
            // We need `groups_per_step * 2` floats.
            // We allocated `group_size + 2` floats for s_group_vals area.
            // We have `groups_per_step = 16`. We need 32 floats.
            // We have plenty of shared memory (input is 8192 floats).
            // We can just allocate a dedicated section.
            
            // For this code, we will use a simple approach:
            // Leader writes to a shared array `s_group_stats[local_group_id]`.
            // We need to declare it.
            // We can just overlay it on top of `s_input` if we are careful?
            // No, input is needed.
            // We can overlay on `s_group_vals`? No, needed for normalization.
            // We can just declare a new pointer in the shared memory space.
            // We have `s_data` of size `in_features + group_size + 2 + 1`.
            // We need `groups_per_step * 2`.
            // Let's adjust the shared mem size calculation in the host code.
            // For now, let's assume we have space.
            
            // Actually, simpler:
            // Leader computes mean/var.
            // Leader normalizes its own value.
            // Leader writes normalized value to s_group_vals[base_idx].
            // Leader broadcasts mean/var to other threads?
            // Or other threads read from shared mem.
            
            // Store mean/var to shared mem for the group
            // We need a specific array for this.
            // Let's use the space after s_running_min.
            // s_running_min is at s_gn_stats + 2.
            // Let's use s_gn_stats[0] and s_gn_stats[1] for the first group?
            // No, we need parallel storage.
            // Let's use the first `groups_per_step * 2` elements of `s_group_vals` after we are done with GEMM?
            // No, we need `s_group_vals` to read the GEMM results.
            
            // Correct Shared Mem Layout:
            // s_input (in_features)
            // s_group_vals (block_dim) -> Used to hold GEMM results, then Normalized results.
            // s_gn_stats (groups_per_step * 2) -> Mean and Var for each group.
            // s_running_min (1)
            
            // Let's assume we have a pointer `s_gn_stats_parallel` pointing to free space.
            // We will just use `s_group_vals` space cleverly?
            // No, let's just use a fixed offset.
            // We will adjust the kernel launch shared memory size.
            
            // For now, let's use a small struct/array in shared mem.
            __shared__ float s_gn_par_stats[32]; // 16 groups * 2 stats.
            
            s_gn_par_stats[local_group_id * 2] = mean;
            s_gn_par_stats[local_group_id * 2 + 1] = var;
        }
        __syncthreads();
        
        // 3. Normalize and Update Min
        // All threads read stats and normalize their value.
        if (current_group < g_end) {
            float mean = s_gn_par_stats[local_group_id * 2];
            float var = s_gn_par_stats[local_group_id * 2 + 1];
            
            float val = s_group_vals[tid]; // Read GEMM result
            float norm_val = (val - mean) * rsqrtf(var + 1e-5f);
            
            // Scale and shift
            norm_val = norm_val * gn_weight[feature_idx] + gn_bias[feature_idx];
            
            // Update Min
            // We need to find Min among the 16 values of this group.
            // Then update the global Min.
            
            // First, reduce Min within the group.
            // We can write norm_val to s_group_vals and reduce.
            s_group_vals[tid] = norm_val;
        }
        __syncthreads();
        
        // Reduce Min within group
        // Leader thread (local_offset == 0) finds min of its group.
        if (local_offset == 0 && current_group < g_end) {
            float group_min = FLT_MAX;
            int base_idx = local_group_id * group_size;
            for (int i = 0; i < group_size; ++i) {
                group_min = fminf(group_min, s_group_vals[base_idx + i]);
            }
            
            // Update global Min
            // Use atomicMin on shared memory?
            // Or just sequentialize? Only 16 groups.
            // Atomic is better.
            atomicMin((int*)&s_running_min[0], __float_as_int(group_min));
        }
        __syncthreads();
    }
    
    // 3. Finalize Min and Write Output
    // s_running_min[0] contains the global Min for this batch item.
    float global_min = s_running_min[0];
    
    // Write output
    // Output shape: (batch_size, 1, out_features, 1).
    // We need to write global_min + bias[f].
    // Parallelize over out_features.
    
    for (int f = tid; f < out_features; f += block_dim) {
        float b = bias_add[f];
        float out_val = global_min + b;
        // Output index: (batch_idx, 0, f, 0)
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

    // Output shape: (batch_size, 1, out_features, 1)
    auto output = torch::zeros({batch_size, 1, out_features, 1}, input.options());

    // Shared memory size calculation:
    // 1. Input: in_features * sizeof(float)
    // 2. Group Vals: block_dim * sizeof(float) (256 * 4 = 1KB)
    // 3. GN Stats: 32 * sizeof(float) (16 groups * 2 stats * 4 = 128B)
    // 4. Running Min: 1 * sizeof(float)
    // Total dynamic shared memory needed: in_features + 256 + 32 + 1 floats.
    // Note: s_gn_par_stats is declared as __shared__ float s_gn_par_stats[32]; inside kernel.
    // This is static shared memory. We need to account for it or make it dynamic.
    // Easier to just request enough dynamic shared memory.
    // We will use dynamic shared memory for everything to be safe.
    
    // Let's adjust the kernel to use a raw pointer for all shared data.
    // Re-declare inside kernel:
    // float* s_input = s_data;
    // float* s_group_vals = s_input + in_features;
    // float* s_gn_par_stats = s_group_vals + 256; // block_dim
    // float* s_running_min = s_gn_par_stats + 32;
    
    size_t shared_mem_size = (in_features + 256 + 32 + 1) * sizeof(float);

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