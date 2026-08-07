import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* input, float* output, int size, float add_value, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sp = log1pf(expf(x));
        float mish = x * tanhf(sp);
        float added = mish + add_value;
        float clamped = added < -1.0f ? -1.0f : (added > 1.0f ? 1.0f : added);
        output[idx] = clamped * scale;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, float add_value, float scale) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size, 
        add_value, 
        scale
    );
    
    return output;
}
"""

fused_op_cpp = "torch::Tensor fused_op_cuda(torch::Tensor input, float add_value, float scale);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_op_cuda(x, self.add_value, self.scale)
        return x

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128  
kernel_size  = 3
stride       = 2  
padding      = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]