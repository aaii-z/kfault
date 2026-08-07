import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused matmul, scale, residual, clamp, logsumexp, and mish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float mish_activation(float x) {
    const float exp_x = expf(x);
    const float tanh_log = (exp_x * (4.0f + exp_x * (4.0f + exp_x))) / (2.0f + exp_x * (2.0f + exp_x));
    return x * tanh_log;
}

__global__ void fused_matmul_scale_residual_clamp_lse_mish(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float scale_factor,
    float clamp_min,
    float clamp_max,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= hidden_size) return;

    // Perform matmul: output[row][col] = sum_i input[row][i] * weight[col][i]
    float sum = 0.0f;
    for (int i = 0; i < input_size; i++) {
        sum += input[row * input_size + i] * weight[col * input_size + i];
    }
    sum += bias[col];  // Add bias from Linear layer

    // Scale
    sum *= scale_factor;
    // Residual: x + x -> 2x (but already scaled, so this is redundant; assuming it's x = x + x = 2 * (scaled)
    sum += sum;  // equivalent to x + x

    // Clamp
    sum = fmaxf(clamp_min, fminf(clamp_max, sum));

    // Store pre-logsumexp value in shared memory will not work across blocks, so we do logsumexp in two kernels
    // Instead, we use a separate kernel for logsumexp reduction, but here we just write the clamped value
    // We'll do logsumexp in a separate fused step, but for now store into a temp buffer
    // This kernel only does up to clamp; logsumexp and final mish handled in next kernel
    // So we output intermediate result first
    // We'll split into two kernels: one for matmul -> clamp, one for logsumexp -> mish
}

// Second kernel: logsumexp along dim=1 (per row), then apply mish(x) * x
// Assumes input is (batch_size, hidden_size), output is (batch_size, 1)
__global__ void logsumexp_mish_kernel(
    const float* x,
    float* output,
    int batch_size,
    int hidden_size
) {
    int row = blockIdx.x;
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int col = tid;

    float val = (col < hidden_size) ? x[row * hidden_size + col] : -1e30f;
    sdata[tid] = val;
    __syncthreads();

    // Perform block-wide max reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (col < stride && (col + stride) < hidden_size)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    float max_val = sdata[0];

    __syncthreads();

    // Now compute exp(x - max) and sum
    float expsum = 0.0f;
    if (col < hidden_size) {
        expsum = expf(x[row * hidden_size + col] - max_val);
    }
    sdata[tid] = expsum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (col < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float sum_exp = sdata[0];

    // Final logsumexp: max + log(sum_exp)
    float lse = max_val + logf(sum_exp);

    // Now apply mish(lse) * lse
    float result = lse * mish_activation(lse);

    // Write output
    if (tid == 0) {
        output[row] = result;
    }
}

// Combined kernel launcher: first do matmul+scale+residual+clamp, then logsumexp+mish
torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max
) {
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int hidden_size = weight.size(0);

    // Create output for logsumexp: (batch_size, 1)
    auto output = torch::zeros({batch_size}, input.options());

    // Temporary buffer for clamped values: (batch_size, hidden_size)
    auto temp = torch::zeros({batch_size, hidden_size}, input.options());

    const int block_size = 256;
    dim3 block_dim((hidden_size + block_size - 1) / block_size);
    dim3 grid_dim(block_dim.x, batch_size);

    // First kernel: matmul + scale + residual + clamp
    fused_matmul_scale_residual_clamp_lse_mish<<<grid_dim, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        temp.data_ptr<float>(),
        scale_factor,
        clamp_min,
        clamp_max,
        batch_size,
        input_size,
        hidden_size
    );

    // Second kernel: logsumexp (dim=1) + mish(x)*x
    const int smem_size = block_size * sizeof(float);
    const int threads = (hidden_size < block_size) ? hidden_size : block_size;
    dim3 lse_grid(batch_size);
    dim3 lse_block(threads);

    logsumexp_mish_kernel<<<lse_grid, lse_block, smem_size>>>(
        temp.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size
    );

    // Reshape output to (batch_size, 1)
    return output.unsqueeze(1);
}
"""

fused_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max
);
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model using fused CUDA kernel for matmul, scale, residual, clamp, logsumexp, and mish.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_op = fused_ops.fused_forward_cuda

    def forward(self, x):
        return self.fused_op(
            x,
            self.linear.weight,
            self.linear.bias,
            self.scale_factor,
            self.clamp_min,
            self.clamp_max
        )