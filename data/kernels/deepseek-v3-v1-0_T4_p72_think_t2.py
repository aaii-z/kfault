import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused average pooling
fused_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_avg_pool_kernel(const float* input, float* output, 
                                     int batch_size, int channels, 
                                     int input_depth, int input_height, int input_width) {
    int output_depth = (input_depth + 1) / 2;
    output_depth = (output_depth + 1) / 2;
    int output_height = (input_height + 1) / 2;
    output_height = (output_height + 1) / 2;
    int output_width = (input_width + 1) / 2;
    output_width = (output_width + 1) / 2;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * channels * output_depth * output_height * output_width;
    if (idx >= total_output) return;

    int n = idx / (channels * output_depth * output_height * output_width);
    int c = (idx % (channels * output_depth * output_height * output_width)) / (output_depth * output_height * output_width);
    int od = (idx % (output_depth * output_height * output_width)) / (output_height * output_width);
    int oh = (idx % (output_height * output_width)) / output_width;
    int ow = idx % output_width;

    int id1 = od * 2;
    int id2 = min(id1 + 1, input_depth - 1);
    int ih1 = oh * 2;
    int ih2 = min(ih1 + 1, input_height - 1);
    int iw1 = ow * 2;
    int iw2 = min(iw1 + 1, input_width - 1);

    float sum = 0.0f;
    int count = 0;
    for (int d = id1; d <= id2; d++) {
        for (int h = ih1; h <= ih2; h++) {
            for (int w = iw1; w <= iw2; w++) {
                int input_idx = ((n * channels + c) * input_depth + d) * input_height * input_width + h * input_width + w;
                sum += input[input_idx];
                count++;
            }
        }
    }
    output[idx] = sum / count;
}

torch::Tensor fused_avg_pool_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int output_depth = (input_depth + 1) / 2;
    output_depth = (output_depth + 1) / 2;
    int output_height = (input_height + 1) / 2;
    output_height = (output_height + 1) / 2;
    int output_width = (input_width + 1) / 2;
    output_width = (output_width + 1) / 2;

    auto output = torch::zeros({batch_size, channels, output_depth, output_height, output_width}, input.options());

    int total_elements = batch_size * channels * output_depth * output_height * output_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_avg_pool_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels, input_depth, input_height, input_width
    );

    return output;
}
"""

fused_avg_pool_cpp_source = "torch::Tensor fused_avg_pool_cuda(torch::Tensor input);"

fused_avg_pool = load_inline(
    name="fused_avg_pool",
    cpp_sources=fused_avg_pool_cpp_source,
    cuda_sources=fused_avg_pool_source,
    functions=["fused_avg_pool_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.fused_avg_pool = fused_avg_pool

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.fused_avg_pool.fused_avg_pool_cuda(x)
        return x