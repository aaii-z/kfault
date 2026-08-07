import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LayerNorm-GELU-Scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr int kBlockSize = 256;

__global__ void fused_layernorm_gelu_scale_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float scaling_factor,
    float eps,
    int num_elements,
    int num_channels
) {
    extern __shared__ float sdata[];
    int outer_index = blockIdx.x;
    int tid = threadIdx.x;

    if (outer_index >= num_elements) return;

    int offset = outer_index * num_channels;
    const float* x = input + offset;
    float* y = output + offset;

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < num_channels; i += blockDim.x) {
        sum += x[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float mean = sdata[0] / num_channels;
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < num_channels; i += blockDim.x) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    sdata[tid] = var_sum;
    __syncthreads();

    for (int s = blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float variance = sdata[0] / num_channels;
    __syncthreads();

    // Normalize, apply affine transformation, GELU, and scale
    if (tid < num_channels) {
        float normalized = (x[tid] - mean) * rsqrtf(variance + eps);
        float affine = normalized * weight[tid] + bias[tid];
        float gelu = 0.5f * affine * (1.0f + erff(affine * 0.7071067811865475f));
        y[tid] = gelu * scaling_factor;
    }
}

torch::Tensor fused_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float eps
) {
    auto sizes = input.sizes();
    int num_channels = sizes[4];
    int num_elements = input.numel() / num_channels;
    
    auto output = torch::empty_like(input);
    
    dim3 grid(num_elements);
    dim3 block(kBlockSize);
    size_t shared_mem_size = kBlockSize * sizeof(float);
    
    fused_layernorm_gelu_scale_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        eps,
        num_elements,
        num_channels
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float eps
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_layernorm_gelu_scale_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        self.eps = eps
        self.scaling_factor = scaling_factor
        
        # Initialize layer norm parameters
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        x = self.conv_transpose(x)
        # Reshape for channel-wise operations
        original_shape = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = fused_ops.fused_layernorm_gelu_scale_cuda(
            x, self.weight, self.bias, 
            self.scaling_factor, self.eps
        )
        x = x.permute(0, 4, 1, 2, 3)
        return x.view(original_shape)