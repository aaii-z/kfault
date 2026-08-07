import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_sigmoid_sum_kernel(const float* input, float* output, int batch_size, int hidden_size) {
    int i = blockIdx.x;
    if (i >= batch_size) return;
    
    float sum = 0.0f;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    for (int j = tid; j < hidden_size; j += num_threads) {
        float z = input[i * hidden_size + j];
        float s = 1.0f / (1.0f + expf(-z));
        sum += s;
    }
    
    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[i] = sdata[0];
    }
}

torch::Tensor fused_sigmoid_sum_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    int batch_size = input.size(0);
    int hidden_size = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());
    
    const int block_size = 256;
    const int grid_size = batch_size;
    
    fused_sigmoid_sum_kernel<<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size
    );
    
    return output;
}
"""

fused_sigmoid_sum_cpp = "torch::Tensor fused_sigmoid_sum_cuda(torch::Tensor input);"

fused_sigmoid_sum = load_inline(
    name="fused_sigmoid_sum",
    cpp_sources=fused_sigmoid_sum_cpp,
    cuda_sources=fused_sigmoid_sum_source,
    functions=["fused_sigmoid_sum_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses sigmoid and sum operations into a single CUDA kernel.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.fused_sigmoid_sum = fused_sigmoid_sum

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = self.linear(x)
        x = self.fused_sigmoid_sum.fused_sigmoid_sum_cuda(x)
        return x.unsqueeze(-1)