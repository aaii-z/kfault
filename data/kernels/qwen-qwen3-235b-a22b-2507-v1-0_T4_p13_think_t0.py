import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_kernel(const float* input, const float* bias, float* output,
                            int B, int C, int D, int H, int W, float scaling_factor) {
    int b = blockIdx.x / (H * W);
    int hw_idx = blockIdx.x % (H * W);
    int h = hw_idx / W;
    int w = hw_idx % W;
    int c = threadIdx.x;
    if (c >= C) return;

    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
        int idx = b * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
        sum += input[idx];
    }
    float mean_val = sum / D;
    float x = mean_val + bias[c];
    float exp_x = expf(x);

    extern __shared__ float shared_data[];
    shared_data[c] = exp_x;
    __syncthreads();

    for (int stride = C / 2; stride > 0; stride >>= 1) {
        if (c < stride) {
            shared_data[c] += shared_data[c + stride];
        }
        __syncthreads();
    }

    float sum_exp = (c == 0) ? shared_data[0] : 0.0f;
    __syncthreads();
    if (c == 0) shared_data[0] = sum_exp;
    __syncthreads();
    sum_exp = shared_data[0];

    float softmax_val = exp_x / sum_exp;
    float result = tanhf(softmax_val) * scaling_factor;
    int out_idx = b * (C * H * W) + c * (H * W) + h * W + w;
    output[out_idx] = result;
}

torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor) {
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    TORCH_CHECK(input.dim() == 5, "Input must be 5D tensor");
    TORCH_CHECK(bias.numel() == C, "Bias size mismatch");

    auto output = torch::empty({B, C, 1, H, W}, input.options());
    int blocks = B * H * W;
    int threads = C;
    size_t shared_mem = C * sizeof(float);

    fused_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W, scaling_factor
    );
    return output;
}
"""

fused_op_cpp = "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3 --use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_op_cuda(x, self.bias.squeeze(), self.scaling_factor)
        return x