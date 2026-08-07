import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Gemm + LogSumExp + 2x LeakyReLU + 2x GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

__device__ inline float leaky_relu(float x, float negative_slope) {
    return fmaxf(x, negative_slope * x);
}

__global__ void fused_gemm_lse_lrelu_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features)
{
    int row = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float shared_data[];
    float* s_input = shared_data;
    float* s_weight = &shared_data[blockDim.x];

    float sum = 0.0f;

    // Load input vector for this row into shared memory
    for (int k = tid; k < in_features; k += blockDim.x) {
        s_input[k] = input[row * in_features + k];
    }
    __syncthreads();

    // Compute output for each neuron in out_features
    for (int j = blockIdx.x; j < out_features; j += gridDim.x) {
        float local_sum = 0.0f;
        for (int k = 0; k < in_features; k += blockDim.x) {
            __syncthreads();
            if (k + tid < in_features) {
                s_weight[tid] = weight[j * in_features + (k + tid)];
            } else {
                s_weight[tid] = 0.0f;
            }
            __syncthreads();

            for (int kk = 0; kk < blockDim.x && (k + kk) < in_features; ++kk) {
                local_sum += s_input[k + kk] * s_weight[kk];
            }
        }
        local_sum += bias[j];

        // Store result in global memory
        output[row * out_features + j] = local_sum;
    }

    __syncthreads();

    // Now perform LogSumExp along dim=1 (across out_features) for this row
    if (threadIdx.x == 0) {
        float max_val = -INFINITY;
        for (int j = 0; j < out_features; ++j) {
            float val = output[row * out_features + j];
            max_val = fmaxf(max_val, val);
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < out_features; ++j) {
            float val = output[row * out_features + j];
            sum_exp += expf(val - max_val);
        }

        float lse = max_val + logf(sum_exp);

        // Apply two LeakyReLU and two GELU activations
        float activated = lse;
        activated = leaky_relu(activated, 0.01f);
        activated = leaky_relu(activated, 0.01f);
        activated = gelu(activated);
        activated = gelu(activated);

        output[row] = activated;
    }

    // Zero out other positions since output should be (batch_size, 1)
    for (int j = threadIdx.x + 1; j < out_features; j += blockDim.x) {
        output[row * out_features + j] = 0.0f;
    }
}

torch::Tensor fused_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    dim3 block(block_size);
    dim3 grid(1, batch_size);

    // Use shared memory for input and weight tiles
    int shared_mem_size = 2 * block_size * sizeof(float);

    fused_gemm_lse_lrelu_gelu_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    // Now extract only the first column since LogSumExp collapses to (batch_size, 1)
    return output.narrow(1, 0, 1);
}
"""

fused_cpp_source = """
torch::Tensor fused_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Gemm + LogSumExp + LeakyReLU + GELU operations in a single CUDA kernel.
    Output shape is (batch_size, 1) to match the original model.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fused_kernel = fused_op.fused_forward

    def forward(self, x):
        return self.fused_kernel(x, self.linear.weight, self.linear.bias)