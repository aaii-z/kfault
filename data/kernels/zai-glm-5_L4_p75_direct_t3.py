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
    // 2. Group values (block_dim) - reused for GN stats and Min reduction
    // 3. GN Stats (groups_per_step * 2)
    // 4. Running Min (1)
    
    extern __shared__ float s_data[];
    float* s_input = s_data; 
    float* s_group_vals = s_data + in_features;
    float* s_gn_par_stats = s_group_vals + blockDim.x; 
    float* s_running_min = s_gn_par_stats + 32; // Max 16 groups * 2 stats
    
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
    // group_size is 16. 
    // We process `block_dim / group_size` groups in parallel.
    // 256 / 16 = 16 groups in parallel.
    
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
            
            // Simple loop for dot product
            for (int k = 0; k < in_features; ++k) {
                sum += s_input[k] * weight_row[k];
            }
            thread_val = sum;
        }
        
        // Write GEMM result to shared memory
        s_group_vals[tid] = thread_val;
        __syncthreads();
        
        // Compute Mean/Var for each group
        // Leader thread (local_offset == 0) computes stats
        if (local_offset == 0 && current_group < g_end) {
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
            // Write normalized value back to shared memory
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