import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_ops_kernel(
    const float* conv_output,
    const float* bias,
    float* output,
    int B, int C, int D, int H, int W,
    float scaling_factor
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    int c = threadIdx.x;

    if (b >= B || h >= H || w >= W || c >= C) return;

    // Compute mean over depth dimension
    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
        int idx = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        sum += conv_output[idx];
    }
    float mean = sum / D;

    // Add bias
    mean += bias[c];

    // Store intermediate values for softmax
    extern __shared__ float shared[];
    shared[c] = mean;
    __syncthreads();

    // Compute softmax over channels
    float max_val = -INFINITY;
    for (int i = 0; i < C; ++i) {
        if (shared[i] > max_val) {
            max_val = shared[i];
        }
    }

    float exp_sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        exp_sum += expf(shared[i] - max_val);
    }

    float softmax_val = expf(mean - max_val) / exp_sum;

    // Apply tanh and scaling
    float tanh_val = tanhf(softmax_val);
    float result = tanh_val * scaling_factor;

    // Write output (depth dimension reduced to 1)
    int out_idx = b * C * 1 * H * W + c * 1 * H * W + 0 * H * W + h * W + w;
    output[out_idx] = result;
}

torch::Tensor fused_ops_cuda(
    torch::Tensor conv_output,
    torch::Tensor bias,
    float scaling_factor
) {
    CHECK_INPUT(conv_output);
    CHECK_INPUT(bias);

    int B = conv_output.size(0);
    int C = conv_output.size(1);
    int D = conv_output.size(2);
    int H = conv_output.size(3);
    int W = conv_output.size(4);

    auto output = torch::empty({B, C, 1, H, W}, conv_output.options());

    dim3 grid(B, H, W);
    dim3 block(C);
    size_t shared_mem = C * sizeof(float);

    fused_ops_kernel<<<grid, block, shared_mem>>>(
        conv_output.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W,
        scaling_factor
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor conv_output, torch::Tensor bias, float scaling_factor);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding
        )
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        # Squeeze bias to match expected shape [C]
        bias_1d = self.bias.squeeze()
        x = fused_ops.fused_ops_cuda(x, bias_1d, self.scaling_factor)
        return x