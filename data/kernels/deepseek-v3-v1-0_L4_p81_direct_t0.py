import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ float swish(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void fused_ops_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int batch_size, 
    int in_features, 
    int out_features) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k += 4) {
            if (k + 3 < in_features) {
                float4 input_val = *reinterpret_cast<const float4*>(&input[row * in_features + k]);
                float4 weight_val = *reinterpret_cast<const float4*>(&weight[col * in_features + k]);
                
                sum += input_val.x * weight_val.x;
                sum += input_val.y * weight_val.y;
                sum += input_val.z * weight_val.z;
                sum += input_val.w * weight_val.w;
            } else {
                for (int i = 0; i < 4 && (k + i) < in_features; i++) {
                    sum += input[row * in_features + k + i] * weight[col * in_features + k + i];
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Swish: x * sigmoid(x)
        float swish_val = sum * (1.0f / (1.0f + expf(-sum)));
        
        // Divide by 2.0
        swish_val *= 0.5f;
        
        // Clamp between -1 and 1
        swish_val = fminf(fmaxf(swish_val, -1.0f), 1.0f);
        
        // Tanh activation
        float tanh_val = tanhf(swish_val);
        
        // Final clamp between -1 and 1
        output[row * out_features + col] = fminf(fmaxf(tanh_val, -1.0f), 1.0f);
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );
    
    fused_ops_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
        
        # Load custom CUDA operation
        self.fused_ops = fused_ops

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.fused_ops.fused_ops_cuda(x, self.weight, self.bias if self.has_bias else torch.Tensor())