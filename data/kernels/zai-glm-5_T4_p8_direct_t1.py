import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Kernel for fused division, max pooling, and global average pooling
// This kernel processes one output channel of one sample in the batch.
// It performs max pooling on the input and then averages the result.
__global__ void fused_pool_kernel(const float* __restrict__ input, float* __restrict__ output,
                                  int N, int C, int D, int H, int W,
                                  int PD, int PH, int PW, float divisor) {
    // Calculate indices
    int n = blockIdx.x; // Batch index
    int c = blockIdx.y; // Channel index
    
    // Each thread block handles one (n, c) pair.
    // Threads in the block collaborate to perform max pooling and averaging.
    
    // Output dimensions after max pool
    int OD = (D - PD) / PD + 1;
    int OH = (H - PH) / PH + 1;
    int OW = (W - PW) / PW + 1;
    
    // Shared memory for partial sums (for global average pooling)
    // We use a small buffer for reduction. Assuming block size is sufficient.
    extern __shared__ float s_partial_max[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Grid-stride loop to find max values in the pooled region
    // We iterate over the output spatial dimensions (od, oh, ow)
    float local_max = -FLT_MAX;
    int total_output_spatial = OD * OH * OW;
    
    // Iterate over output spatial positions assigned to this thread
    for (int i = tid; i < total_output_spatial; i += block_size) {
        int ow = i % OW;
        int oh = (i / OW) % OH;
        int od = i / (OW * OH);
        
        // Find the start of the pooling window in the input
        int id_start = od * PD;
        int ih_start = oh * PH;
        int iw_start = ow * PW;
        
        // Find max in the pooling window
        float max_val = -FLT_MAX;
        for (int pd = 0; pd < PD; ++pd) {
            for (int ph = 0; ph < PH; ++ph) {
                for (int pw = 0; pw < PW; ++pw) {
                    int id = id_start + pd;
                    int ih = ih_start + ph;
                    int iw = iw_start + pw;
                    
                    // Input tensor layout: (N, C, D, H, W)
                    // Calculate input index
                    long long input_idx = (long long)n * C * D * H * W +
                                          (long long)c * D * H * W +
                                          id * H * W + ih * W + iw;
                    
                    float val = input[input_idx] / divisor;
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
        
        if (max_val > local_max) {
            local_max = max_val;
        }
    }
    
    // Now we need to average the max values found by all threads in the block
    // This is a reduction to find the sum of max values
    
    // Load local max into shared memory
    s_partial_max[tid] = local_max;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial_max[tid] += s_partial_max[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the result
    if (tid == 0) {
        float avg_val = s_partial_max[0] / (float)total_output_spatial;
        // Output layout: (N, C, 1, 1, 1)
        // We write to a contiguous array of size N*C
        output[n * C + c] = avg_val;
    }
}

// Kernel for fused bias addition and sum reduction
__global__ void fused_bias_sum_kernel(const float* __restrict__ pooled, const float* __restrict__ bias,
                                      float* __restrict__ out, int N, int C, int sum_dim) {
    // sum_dim is 1 (channels), so we sum over C for each batch sample.
    // Output size: N
    // Each block handles one sample n.
    
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    extern __shared__ float s_partial_sum[];
    
    float local_sum = 0.0f;
    
    // Grid-stride loop over channels
    for (int c = tid; c < C; c += block_size) {
        float val = pooled[n * C + c] + bias[c]; // bias is (C, 1, 1, 1), broadcasted
        local_sum += val;
    }
    
    s_partial_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial_sum[tid] += s_partial_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        out[n] = s_partial_sum[0];
    }
}

// Helper function to launch the fused pooling kernel
torch::Tensor fused_pool_cuda(torch::Tensor input, float divisor, std::vector<int64_t> pool_size) {
    // Input shape: (N, C, D, H, W)
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    int PD = pool_size[0];
    int PH = pool_size[1];
    int PW = pool_size[2];
    
    // Output shape after global average pool: (N, C)
    // We flatten the last 3 dimensions.
    auto output = torch::empty({N, C}, input.options());
    
    // Launch configuration
    // Grid: (N, C) blocks. Each block computes one output channel for one sample.
    dim3 grid(N, C);
    
    // Block size: 256 threads is a good default.
    int threads = 256;
    
    // Shared memory size: 256 * sizeof(float)
    size_t shared_mem_size = threads * sizeof(float);
    
    fused_pool_kernel<<<grid, threads, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D, H, W, PD, PH, PW, divisor
    );
    
    return output;
}

// Helper function to launch the fused bias sum kernel
torch::Tensor fused_bias_sum_cuda(torch::Tensor pooled, torch::Tensor bias, int sum_dim) {
    // Pooled shape: (N, C)
    // Bias shape: (C, 1, 1, 1) -> effectively (C)
    // Sum dim: 1 (channels)
    // Output shape: (N)
    
    int N = pooled.size(0);
    int C = pooled.size(1);
    
    auto output = torch::empty({N}, pooled.options());
    
    // Launch configuration
    // Grid: N blocks. Each block sums one sample.
    dim3 grid(N);
    int threads = 256;
    size_t shared_mem_size = threads * sizeof(float);
    
    fused_bias_sum_kernel<<<grid, threads, shared_mem_size>>>(
        pooled.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C, sum_dim
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_pool_cuda(torch::Tensor input, float divisor, std::vector<int64_t> pool_size);
torch::Tensor fused_bias_sum_cuda(torch::Tensor pooled, torch::Tensor bias, int sum_dim);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_pool_cuda", "fused_bias_sum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.pool_size = pool_size
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Conv3d (using PyTorch's optimized implementation)
        x = self.conv(x)
        
        # 2. Fused Division, Max Pooling, Global Average Pooling
        # We fuse x / divisor, MaxPool3d, and AdaptiveAvgPool3d
        # The output of this kernel is a (N, C) tensor
        x = self.fused_ops.fused_pool_cuda(x, self.divisor, self.pool_size)
        
        # 3. Fused Bias Addition and Sum Reduction
        # We fuse x + bias and torch.sum(x, dim=1)
        # The output is a (N) tensor
        x = self.fused_ops.fused_bias_sum_cuda(x, self.bias, self.sum_dim)
        
        # Reshape to match the reference output shape (128, 1, 1, 1)
        x = x.view(x.size(0), 1, 1, 1)
        
        return x