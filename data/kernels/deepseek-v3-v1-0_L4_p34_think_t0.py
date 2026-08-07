import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LayerNorm-GELU-Scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

constexpr int kBlockSize = 256;

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
    int outer_index = blockIdx.x;
    int inner_index = threadIdx.x;
    
    if (inner_index >= num_channels) return;
    
    int offset = outer_index * num_channels;
    const float* x = input + offset;
    float* y = output + offset;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = inner_index; i < num_channels; i += blockDim.x) {
        sum += x[i];
    }
    sum = blockReduceSum(sum);
    __syncthreads();
    float mean = sum / num_channels;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = inner_index; i < num_channels; i += blockDim.x) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = blockReduceSum(var_sum);
    __syncthreads();
    float variance = var_sum / num_channels;
    
    // Normalize and apply affine transformation
    float normalized = (x[inner_index] - mean) * rsqrtf(variance + eps);
    float affine = normalized * weight[inner_index] + bias[inner_index];
    
    // GELU approximation
    float gelu = 0.5f * affine * (1.0f + tanhf(0.7978845608028654f * 
                 (affine + 0.044715f * affine * affine * affine)));
    
    // Scale and write output
    y[inner_index] = gelu * scaling_factor;
}

torch::Tensor fused_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float eps
) {
    auto sizes = input.sizes();
    int num_channels = sizes[1];
    int num_elements = input.numel() / num_channels;
    
    auto output = torch::empty_like(input);
    
    dim3 grid(num_elements);
    dim3 block(kBlockSize);
    
    fused_layernorm_gelu_scale_kernel<<<grid, block>>>(
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