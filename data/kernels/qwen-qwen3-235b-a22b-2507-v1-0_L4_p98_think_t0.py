import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused GELU, scaling, and max reduction
fused_gelu_scale_max_source = """
#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>

__global__ void fused_gelu_scale_max_kernel(const float* input, float* output, float scale_factor, int batch_size, int pooled_size) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;

    const float* x = input + sample_idx * pooled_size;
    float max_val = -FLT_MAX;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    for (int i = tid; i < pooled_size; i += num_threads) {
        float val = x[i];
        float sqrt_2_over_pi = 0.7978845608028654f;
        float coeff = 0.044715f;
        float val_cube = val * val * val;
        float inner = sqrt_2_over_pi * (val + coeff * val_cube);
        float tanh_inner = tanhf(inner);
        float gelu_val = 0.5f * val * (1.0f + tanh_inner);
        float scaled_val = scale_factor * gelu_val;
        
        if (scaled_val > max_val) {
            max_val = scaled_val;
        }
    }
    
    __shared__ float sdata[256];
    sdata[tid] = max_val;
    __syncthreads();
    
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[sample_idx] = sdata[0];
    }
}

torch::Tensor fused_gelu_scale_max_cuda(torch::Tensor input, float scale_factor) {
    int batch_size = input.size(0);
    int pooled_size = input.size(1);
    auto output = torch::empty(batch_size, input.options());
    
    const int block_size = 256;
    fused_gelu_scale_max_kernel<<<batch_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scale_factor,
        batch_size,
        pooled_size
    );
    
    return output;
}
"""

fused_gelu_scale_max_cpp = "torch::Tensor fused_gelu_scale_max_cuda(torch::Tensor input, float scale_factor);"

fused_gelu_scale_max = load_inline(
    name="fused_gelu_scale_max",
    cpp_sources=fused_gelu_scale_max_cpp,
    cuda_sources=fused_gelu_scale_max_source,
    functions=["fused_gelu_scale_max_cuda"],
    verbose=True,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GELU, scaling, and max reduction CUDA kernel.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor
        self.fused_gelu_scale_max = fused_gelu_scale_max

    def forward(self, x):
        x = self.matmul(x)
        x = self.avg_pool(x.unsqueeze(1)).squeeze(1)
        x = self.fused_gelu_scale_max.fused_gelu_scale_max_cuda(x, self.scale_factor)
        return x