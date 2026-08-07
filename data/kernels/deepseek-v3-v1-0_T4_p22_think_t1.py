import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float tmp = __shfl_down_sync(0xffffffff, val, offset);
        if (tmp > val) val = tmp;
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduceMax(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__device__ float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ float mish_activation(float x) {
    return x * tanhf(softplus(x));
}

__global__ void fused_linear_scale_clamp_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size,
    float scale_factor,
    float clamp_min,
    float clamp_max) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[row * input_size + i] * weight[col * input_size + i];
        }
        if (bias) {
            sum += bias[col];
        }
        
        sum = sum * scale_factor * 2.0f;
        
        if (sum < clamp_min) sum = clamp_min;
        if (sum > clamp_max) sum = clamp_max;
        
        output[row * hidden_size + col] = sum;
    }
}

__global__ void logsumexp_mish_reduce_kernel(
    const float* input,
    float* output,
    int batch_size,
    int hidden_size) {
    
    int tid = threadIdx.x;
    int row = blockIdx.x;

    if (row >= batch_size) return;

    const float* row_data = input + row * hidden_size;

    float max_val = -INFINITY;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        if (row_data[i] > max_val) max_val = row_data[i];
    }
    max_val = blockReduceMax(max_val);
    __syncthreads();

    float exp_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        exp_sum += expf(row_data[i] - max_val);
    }
    exp_sum = blockReduceSum(exp_sum);
    __syncthreads();

    if (tid == 0) {
        float logsumexp = logf(exp_sum) + max_val;
        float mish_val = mish_activation(logsumexp);
        output[row] = logsumexp * mish_val;
    }
}

torch::Tensor fused_linear_scale_clamp_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);
    
    auto output = torch::empty({batch_size, hidden_size}, input.options());
    
    dim3 blocks((hidden_size + TILE_DIM - 1) / TILE_DIM,
                (batch_size + TILE_DIM - 1) / TILE_DIM);
    dim3 threads(TILE_DIM, TILE_DIM);
    
    fused_linear_scale_clamp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        scale_factor,
        clamp_min,
        clamp_max);
    
    return output;
}

torch::Tensor logsumexp_mish_reduce_cuda(torch::Tensor input) {
    CHECK_INPUT(input);
    
    int batch_size = input.size(0);
    int hidden_size = input.size(1);
    
    auto output = torch::empty({batch_size, 1}, input.options());
    
    int threads = 256;
    logsumexp_mish_reduce_kernel<<<batch_size, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size);
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_linear_scale_clamp_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max);

torch::Tensor logsumexp_mish_reduce_cuda(torch::Tensor input);
"""

# Load the custom CUDA operations
custom_ops = load_inline(
    name='custom_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        'fused_linear_scale_clamp_cuda',
        'logsumexp_mish_reduce_cuda'
    ],
    verbose=True,
    extra_cflags=['-O2'],
    extra_ldflags=['-lcublas']
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = custom_ops.fused_linear_scale_clamp_cuda(
            x, self.weight, self.bias,
            self.scale_factor, self.clamp_min, self.clamp_max
        )
        x = custom_ops.logsumexp_mish_reduce_cuda(x)
        return x