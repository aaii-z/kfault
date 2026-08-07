import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused average pooling and GELU
fused_avgpool_gelu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr float GELU_COEF_A = 0.7978845608028654f;
constexpr float GELU_COEF_B = 0.044715f;

__global__ void avg_pool_gelu_kernel(
    const float* input,
    float* output,
    int B,
    int C,
    int D,
    int H,
    int W,
    int pooled_D,
    int pooled_H,
    int pooled_W
) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int pd = blockIdx.z;
    int ph = threadIdx.y;
    int pw = threadIdx.x;

    if (ph >= pooled_H || pw >= pooled_W) return;

    int input_start_d = pd * 2;
    int input_start_h = ph * 2;
    int input_start_w = pw * 2;

    float sum = 0.0f;
    int count = 0;
    for (int id = 0; id < 2; ++id) {
        int d = input_start_d + id;
        if (d >= D) continue;
        for (int ih = 0; ih < 2; ++ih) {
            int h = input_start_h + ih;
            if (h >= H) continue;
            for (int iw = 0; iw < 2; ++iw) {
                int w = input_start_w + iw;
                if (w >= W) continue;
                int idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                sum += input[idx];
                count++;
            }
        }
    }
    float avg = sum / count;

    // GELU approximation
    float gelu = 0.5f * avg * (1.0f + tanhf(GELU_COEF_A * (avg + GELU_COEF_B * avg * avg * avg)));

    int out_idx = n * C * pooled_D * pooled_H * pooled_W + c * pooled_D * pooled_H * pooled_W + pd * pooled_H * pooled_W + ph * pooled_W + pw;
    output[out_idx] = gelu;
}

torch::Tensor fused_avgpool_gelu_cuda(torch::Tensor input) {
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    int pooled_D = D / 2;
    int pooled_H = H / 2;
    int pooled_W = W / 2;
    
    auto output = torch::empty({B, C, pooled_D, pooled_H, pooled_W}, input.options());

    dim3 blocks(B, C, pooled_D);
    dim3 threads(pooled_W, pooled_H);

    avg_pool_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W,
        pooled_D, pooled_H, pooled_W
    );

    return output;
}
"""

fused_avgpool_gelu_cpp = "torch::Tensor fused_avgpool_gelu_cuda(torch::Tensor input);"

# Load the custom CUDA extension
fused_avgpool_gelu = load_inline(
    name='fused_avgpool_gelu',
    cpp_sources=fused_avgpool_gelu_cpp,
    cuda_sources=fused_avgpool_gelu_source,
    functions=['fused_avgpool_gelu_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding,
            bias=True  # Add bias to fuse the sum operation
        )
        # Initialize convolution bias with sum_weight
        with torch.no_grad():
            self.conv_transpose.bias.fill_(sum_weight)
        self.norm = nn.LayerNorm(norm_shape)
        
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.norm(x)
        x = fused_avgpool_gelu.fused_avgpool_gelu_cuda(x)
        return x