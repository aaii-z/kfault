import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Add+LayerNorm+GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr float GELU_COEF_A = 0.7978845608028654f;
constexpr float GELU_COEF_B = 0.044715f;

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void add_layernorm_gelu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float sum_weight,
    int num_elements,
    int C,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int vector_idx = idx / C;
    int channel_idx = idx % C;

    float x = input[idx] + sum_weight;
    
    __shared__ float s_mean;
    __shared__ float s_variance;

    float mean = blockReduceSum(x);
    if (threadIdx.x == 0) s_mean = mean / C;
    __syncthreads();
    mean = s_mean;

    float variance = (x - mean) * (x - mean);
    variance = blockReduceSum(variance);
    if (threadIdx.x == 0) s_variance = variance / C + eps;
    __syncthreads();
    variance = s_variance;

    float normalized = (x - mean) * rsqrtf(variance);
    float scaled = normalized * weight[channel_idx] + bias[channel_idx];
    
    // GELU approximation
    float gelu = 0.5f * scaled * (1.0f + tanhf(GELU_COEF_A * (scaled + GELU_COEF_B * scaled * scaled * scaled)));
    output[idx] = gelu;
}

torch::Tensor fused_add_layernorm_gelu_cuda(
    torch::Tensor input,
    float sum_weight,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    int num_elements = input.numel();
    int C = weight.size(0);
    auto output = torch::empty_like(input);

    dim3 block_dim(256);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);

    add_layernorm_gelu_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        sum_weight,
        num_elements,
        C,
        eps
    );

    return output;
}
"""

fused_kernel_cpp = "torch::Tensor fused_add_layernorm_gelu_cuda(torch::Tensor input, float sum_weight, torch::Tensor weight, torch::Tensor bias, float eps);"

# Load the custom CUDA extension
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=['fused_add_layernorm_gelu_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        
    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_ops.fused_add_layernorm_gelu_cuda(x, self.sum_weight.item(), self.norm.weight, self.norm.bias, self.norm.eps)
        x = self.avg_pool(x)
        return x