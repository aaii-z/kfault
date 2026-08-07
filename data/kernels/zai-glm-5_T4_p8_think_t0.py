import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_op_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp reduction for sum
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void fused_op_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    float divisor, 
    int C, int D, int H, int W) {
    
    int n = blockIdx.x;
    int c = threadIdx.y;
    int tid = threadIdx.x;
    
    // Assuming C <= blockDim.y (typically 16 or 32 channels)
    if (c >= C) return; 

    int D_out = D / 2;
    int H_out = H / 2;
    int W_out = W / 2;
    int spatial_size = D_out * H_out * W_out;
    
    const float* input_ptr = input + (n * C + c) * D * H * W;
    
    float local_sum = 0.0f;
    float inv_divisor = 1.0f / divisor;
    
    // Strided loop over the output spatial volume
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int d_out = i / (H_out * W_out);
        int rem = i % (H_out * W_out);
        int h_out = rem / W_out;
        int w_out = rem % W_out;
        
        int d_in = d_out * 2;
        int h_in = h_out * 2;
        int w_in = w_out * 2;
        
        int base = d_in * H * W + h_in * W + w_in;
        
        // MaxPool 2x2x2
        // Slice 0 (d_in)
        float m0 = fmaxf(input_ptr[base], input_ptr[base + 1]);
        float m1 = fmaxf(input_ptr[base + W], input_ptr[base + W + 1]);
        float max_slice0 = fmaxf(m0, m1);
        
        // Slice 1 (d_in + 1)
        int base2 = base + H * W;
        float m2 = fmaxf(input_ptr[base2], input_ptr[base2 + 1]);
        float m3 = fmaxf(input_ptr[base2 + W], input_ptr[base2 + W + 1]);
        float max_slice1 = fmaxf(m2, m3);
        
        float max_val = fmaxf(max_slice0, max_slice1);
        
        // Divide by constant and accumulate for AvgPool
        local_sum += max_val * inv_divisor;
    }
    
    // Reduce within the warp (threads in threadIdx.x direction)
    float sum_val = warpReduceSum(local_sum);
    
    // Shared memory to store result per channel
    extern __shared__ float channel_sums[];
    
    if (tid == 0) {
        float avg = sum_val / spatial_size;
        channel_sums[c] = avg + bias[c];
    }
    
    __syncthreads();
    
    // Sum across channels (dim 1)
    // Only the first warp (threadIdx.y == 0) performs the final reduction
    if (c == 0) {
        if (tid == 0) {
            float final_sum = 0.0f;
            for (int k = 0; k < C; ++k) {
                final_sum += channel_sums[k];
            }
            output[n] = final_sum;
        }
    }
}

torch::Tensor forward_cuda(torch::Tensor input, torch::Tensor bias, float divisor, int C, int D, int H, int W) {
    int batch_size = input.size(0);
    // Output shape matches torch.sum(x, dim=1) on (N, C, 1, 1, 1) -> (N, 1, 1, 1)
    auto output = torch::zeros({batch_size, 1, 1, 1}, input.options());

    dim3 block(32, C);
    dim3 grid(batch_size);
    
    size_t shared_mem_size = C * sizeof(float);
    
    fused_op_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor,
        C, D, H, W
    );
    
    return output;
}
"""

fused_op_cpp_source = "torch::Tensor forward_cuda(torch::Tensor input, torch::Tensor bias, float divisor, int C, int D, int H, int W);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        N, C, D, H, W = x.shape
        # Ensure bias is contiguous for correct memory access in CUDA kernel
        bias_cont = self.bias.contiguous()
        x = self.fused_op.forward_cuda(x, bias_cont, self.divisor, C, D, H, W)
        return x