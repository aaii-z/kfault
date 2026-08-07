import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + BatchNorm + Scale
fused_conv_bn_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_bn_scale_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    const float* __restrict__ bn_weight, 
    const float* __restrict__ bn_bias, 
    const float* __restrict__ bn_mean, 
    const float* __restrict__ bn_var, 
    float* __restrict__ output,
    int batch_size, int in_channels, int in_h, int in_w, 
    int out_channels, int kernel_size, 
    float scale_factor, float eps) {

    int out_h = in_h - kernel_size + 1;
    int out_w = in_w - kernel_size + 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int ow = blockIdx.x * blockDim.x + tx;
    int oh = blockIdx.y * blockDim.y + ty;
    
    int n_oc = blockIdx.z;
    int n = n_oc / out_channels;
    int oc = n_oc % out_channels;

    if (n >= batch_size || oc >= out_channels || oh >= out_h || ow >= out_w) return;

    extern __shared__ float shared_mem[];
    
    float* s_weights = shared_mem;
    int weight_size = in_channels * kernel_size * kernel_size;
    
    int tile_h = blockDim.y + kernel_size - 1;
    int tile_w = blockDim.x + kernel_size - 1;
    float* s_input = &shared_mem[weight_size];

    int total_threads = blockDim.x * blockDim.y;
    int tid = ty * blockDim.x + tx;
    
    // Load weights for this output channel into shared memory
    for (int i = tid; i < weight_size; i += total_threads) {
        s_weights[i] = weight[oc * weight_size + i];
    }
    
    float val = 0.0f;
    
    int ih_start = blockIdx.y * blockDim.y;
    int iw_start = blockIdx.x * blockDim.x;

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Load input tile into shared memory
        int tile_size = tile_h * tile_w;
        for (int i = tid; i < tile_size; i += total_threads) {
            int r = i / tile_w;
            int c = i % tile_w;
            
            int ih = ih_start + r;
            int iw = iw_start + c;
            
            float v = 0.0f;
            if (ih < in_h && iw < in_w) {
                v = input[n * in_channels * in_h * in_w + ic * in_h * in_w + ih * in_w + iw];
            }
            s_input[r * tile_w + c] = v;
        }
        
        __syncthreads();
        
        // Compute convolution for this input channel
        float* w_ptr = &s_weights[ic * kernel_size * kernel_size];
        
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                val += w_ptr[kh * kernel_size + kw] * s_input[(ty + kh) * tile_w + (tx + kw)];
            }
        }
        
        __syncthreads();
    }
    
    // Add bias
    val += bias[oc];
    
    // Batch Normalization (using running stats for inference)
    float mean = bn_mean[oc];
    float var = bn_var[oc];
    float gamma = bn_weight[oc];
    float beta = bn_bias[oc];
    
    val = gamma * (val - mean) / sqrtf(var + eps) + beta;
    
    // Scale
    val *= scale_factor;
    
    // Write output
    output[n * out_channels * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = val;
}

torch::Tensor fused_conv_bn_scale_cuda(
    torch::Tensor input, 
    torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var,
    float scale_factor, float eps) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_h = in_h - kernel_size + 1;
    int out_w = in_w - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());
    
    const int block_dim = 16;
    dim3 block(block_dim, block_dim);
    
    dim3 grid((out_w + block.x - 1) / block.x, 
              (out_h + block.y - 1) / block.y, 
              batch_size * out_channels);
              
    int weight_mem = in_channels * kernel_size * kernel_size;
    int tile_mem = (block_dim + kernel_size - 1) * (block_dim + kernel_size - 1);
    size_t shared_mem_size = (weight_mem + tile_mem) * sizeof(float);
    
    fused_conv_bn_scale_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_h, in_w,
        out_channels, kernel_size,
        scale_factor, eps
    );
    
    return output;
}
"""

fused_conv_bn_scale_cpp_source = """
torch::Tensor fused_conv_bn_scale_cuda(
    torch::Tensor input, 
    torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var,
    float scale_factor, float eps);
"""

fused_op = load_inline(
    name="fused_conv_bn_scale",
    cpp_sources=fused_conv_bn_scale_cpp_source,
    cuda_sources=fused_conv_bn_scale_source,
    functions=["fused_conv_bn_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_conv_bn_scale_cuda(
            x, 
            self.conv.weight, 
            self.conv.bias,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.scaling_factor,
            self.bn.eps
        )