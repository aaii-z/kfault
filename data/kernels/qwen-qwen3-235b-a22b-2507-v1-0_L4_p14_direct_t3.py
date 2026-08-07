import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for high-performance fused matmul, divide, sum, and scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Efficient fused kernel: compute matmul(x, weight.T), divide by 2, sum over hidden dim, scale
// Each block handles one row; each thread in block helps compute the large inner product
// We use block-level reduction to sum over hidden_size

__global__ void fused_matmul_divide_sum_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float scaling_factor,
    int batch_size,
    int input_size,
    int hidden_size) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    __shared__ float sdata[256]; // assuming block size is 256

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    float sum = 0.0f;

    // Use grid-stride loop over hidden dimension for better load balancing
    for (int h = blockIdx.y * block_size + tid; h < hidden_size; h += gridDim.y * block_size) {
        float val = 0.0f;
        // Compute dot product for weight[h, :] with input[row, :]
        for (int k = 0; k < input_size; ++k) {
            val += input[row * input_size + k] * weight[h * input_size + k];
        }
        sum += val * 0.5f;  // Apply division by 2
    }

    // Store per-thread partial sum
    sdata[tid] = sum;
    __syncthreads();

    // Block-level reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this row
    if (tid == 0) {
        output[row] = sdata[0] * scaling_factor;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    float scaling_factor) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    TORCH_CHECK(x.dtype() == torch::kFloat, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat, "weight must be float32");

    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = weight.size(0);

    auto output = torch::empty({batch_size}, x.options());

    if (batch_size == 0 || hidden_size == 0) {
        return output.unsqueeze(1);
    }

    const int block_size = 256;
    dim3 grid(batch_size, (hidden_size + block_size - 1) / block_size); // 2D grid
    dim3 block(block_size);

    fused_matmul_divide_sum_scale_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        input_size,
        hidden_size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output.unsqueeze(1);
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor);
"""

# Compile the inline CUDA kernel with optimization flags
fused_op = load_inline(
    name="fused_op",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matmul, divide, sum, and scale.
    Uses 2D grid for parallelism across batch and hidden dimensions, with block reduction.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(x, self.weight, self.scaling_factor)