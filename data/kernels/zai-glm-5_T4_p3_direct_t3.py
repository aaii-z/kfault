import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LayerNorm, AvgPool3d, and GELU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for fused LayerNorm + AvgPool3d + GELU
// LayerNorm is applied per channel (C) over spatial dimensions (D, H, W)
__global__ void fused_ln_pool_gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, int D, int H, int W,
    int pool_D, int pool_H, int pool_W,
    float epsilon) {

    // Output dimensions
    int out_D = (D + pool_D - 1) / pool_D;
    int out_H = (H + pool_H - 1) / pool_H;
    int out_W = (W + pool_W - 1) / pool_W;

    // Calculate total threads needed for output
    int total_elements = N * C * out_D * out_H * out_W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // Decompose linear index into output coordinates (n, c, od, oh, ow)
        int tmp = idx;
        int ow = tmp % out_W;
        tmp /= out_W;
        int oh = tmp % out_H;
        tmp /= out_H;
        int od = tmp % out_D;
        tmp /= out_D;
        int c = tmp % C;
        int n = tmp / C;

        // Calculate input pooling window boundaries
        int id_start = od * pool_D;
        int id_end = min(id_start + pool_D, D);
        int ih_start = oh * pool_H;
        int ih_end = min(ih_start + pool_H, H);
        int iw_start = ow * pool_W;
        int iw_end = min(iw_start + pool_W, W);

        // 1. Compute Mean and Variance for LayerNorm across (D, H, W) for this specific (n, c)
        // LayerNorm is applied *before* pooling. The normalization statistics are computed
        // over the full spatial dimensions D, H, W.
        
        float ln_sum = 0.0f;
        int input_spatial_size = D * H * W;
        
        // Input pointer for this (n, c)
        // Input layout: (N, C, D, H, W)
        const float* __restrict__ input_nc = input + (n * C + c) * input_spatial_size;

        for (int i = 0; i < input_spatial_size; ++i) {
            ln_sum += input_nc[i];
        }
        
        float ln_mean = ln_sum / input_spatial_size;
        
        float ln_var_sum = 0.0f;
        for (int i = 0; i < input_spatial_size; ++i) {
            float diff = input_nc[i] - ln_mean;
            ln_var_sum += diff * diff;
        }
        float ln_var = ln_var_sum / input_spatial_size;
        
        float ln_inv_std = rsqrtf(ln_var + epsilon);
        float ln_gamma = gamma[c];
        float ln_beta = beta[c];

        // 2. Apply LayerNorm, Pooling, and GELU
        // We compute the average of the normalized values in the pooling window.
        
        float pool_sum = 0.0f;
        int pool_count = 0;
        
        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    // Input index
                    int in_idx = id * H * W + ih * W + iw;
                    float val = input_nc[in_idx];
                    
                    // LayerNorm
                    val = (val - ln_mean) * ln_inv_std;
                    val = val * ln_gamma + ln_beta;
                    
                    // Accumulate for AvgPool
                    pool_sum += val;
                    pool_count++;
                }
            }
        }
        
        float pooled_val = pool_sum / pool_count;
        
        // GELU approximation (tanh approximation)
        float x = pooled_val;
        float gelu = x * 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        
        output[idx] = gelu;
    }
}

torch::Tensor fused_ln_pool_gelu_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int pool_D, int pool_H, int pool_W,
    float epsilon) {

    auto N = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    int out_D = (D + pool_D - 1) / pool_D;
    int out_H = (H + pool_H - 1) / pool_H;
    int out_W = (W + pool_W - 1) / pool_W;

    auto output = torch::zeros({N, C, out_D, out_H, out_W}, input.options());

    int total_elements = output.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_ln_pool_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N, C, D, H, W,
        pool_D, pool_H, pool_W,
        epsilon
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ln_pool_gelu_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int pool_D, int pool_H, int pool_W, float epsilon);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ln_pool_gelu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by a sum, 
    and a fused kernel for LayerNorm, Average Pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        
        # We need to access the weights of LayerNorm directly for the custom kernel
        # norm_shape is expected to be (out_channels,) based on the problem description
        self.norm = nn.LayerNorm(norm_shape)
        
        self.pool_kernel_size = pool_kernel_size
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Transposed Convolution (using PyTorch's optimized implementation)
        x = self.conv_transpose(x)
        
        # 2. Sum (Element-wise addition)
        # Broadcasting sum_weight (scalar) across the tensor
        x = x + self.sum_weight
        
        # 3. Fused LayerNorm + AvgPool + GELU
        # Extract parameters for the custom kernel
        gamma = self.norm.weight
        beta = self.norm.bias
        epsilon = self.norm.eps
        
        # Call the custom CUDA kernel
        x = self.fused_ops.fused_ln_pool_gelu_cuda(
            x, gamma, beta, 
            self.pool_kernel_size[0], 
            self.pool_kernel_size[1], 
            self.pool_kernel_size[2], 
            epsilon
        )
        
        return x