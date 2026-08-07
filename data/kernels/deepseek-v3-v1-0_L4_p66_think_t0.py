import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + dropout + softmax
fused_linear_dropout_softmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

const int BLOCK_SIZE = 256;

__global__ void fused_linear_dropout_softmax_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float dropout_p,
    bool training,
    curandState* states
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= batch_size || col >= out_features) return;

    extern __shared__ float shared_data[];
    float* row_buffer = shared_data;
    float* max_val = &shared_data[out_features];
    float* sum_exp = &shared_data[out_features + 1];

    // Linear transformation
    float val = 0.0f;
    for (int k = 0; k < in_features; k++) {
        val += input[row * in_features + k] * weight[col * in_features + k];
    }
    val += bias[col];

    // Dropout
    if (training) {
        curandState local_state = states[row * out_features + col];
        float random = curand_uniform(&local_state);
        states[row * out_features + col] = local_state;
        if (random < dropout_p) {
            val = 0.0f;
        } else {
            val /= (1.0f - dropout_p);
        }
    }

    row_buffer[col] = val;
    __syncthreads();

    // Online softmax
    if (col == 0) {
        max_val[0] = -INFINITY;
        for (int i = 0; i < out_features; i++) {
            if (row_buffer[i] > max_val[0]) {
                max_val[0] = row_buffer[i];
            }
        }
    }
    __syncthreads();

    if (col == 0) {
        sum_exp[0] = 0.0f;
        for (int i = 0; i < out_features; i++) {
            sum_exp[0] += expf(row_buffer[i] - max_val[0]);
        }
    }
    __syncthreads();

    output[row * out_features + col] = expf(row_buffer[col] - max_val[0]) / sum_exp[0];
}

void fused_linear_dropout_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float dropout_p,
    bool training
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    curandState* states;
    if (training) {
        cudaMalloc(&states, batch_size * out_features * sizeof(curandState));
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234);
        curandGenerateUniform(gen, (float*)states, batch_size * out_features);
        curandDestroyGenerator(gen);
    }

    dim3 blocks(batch_size);
    dim3 threads(out_features);
    int shared_mem_size = (out_features + 2) * sizeof(float);

    fused_linear_dropout_softmax_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        dropout_p,
        training,
        states
    );

    if (training) {
        cudaFree(states);
    }
}
"""

fused_linear_dropout_softmax_cpp_source = """
void fused_linear_dropout_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float dropout_p,
    bool training
);
"""

# Compile the inline CUDA code
fused_linear_dropout_softmax = load_inline(
    name="fused_linear_dropout_softmax",
    cpp_sources=fused_linear_dropout_softmax_cpp_source,
    cuda_sources=fused_linear_dropout_softmax_source,
    functions=["fused_linear_dropout_softmax_cuda"],
    verbose=True,
    extra_cflags=["-lcublas", "-lcurand"],
    extra_ldflags=["-lcublas", "-lcurand"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        output = torch.empty(x.size(0), self.out_features, device=x.device)
        fused_linear_dropout_softmax.fused_linear_dropout_softmax_cuda(
            x, self.weight, self.bias, output, self.dropout_p, self.training
        )
        return output