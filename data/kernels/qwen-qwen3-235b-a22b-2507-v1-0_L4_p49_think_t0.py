import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_sigmoid_kernel(const float* input, float* output, int batch, int channels, int D, int H, int W) {
    int idx = blockIdx.x;
    int total_spatial = batch * D * H * W;
    if (idx >= total_spatial) return;

    int spatial_idx = idx;
    int b = spatial_idx / (D * H * W);
    spatial_idx %= (D * H * W);
    int d = spatial_idx / (H * W);
    spatial_idx %= (H * W);
    int h = spatial_idx / W;
    int w = spatial_idx % W;

    int base = b * (channels * D * H * W) + d * (H * W) + h * W + w;

    extern __shared__ float s_data[];

    int c = threadIdx.x;
    float val = -INFINITY;
    if (c < channels) {
        int input_idx = base + c * (D * H * W);
        val = input[input_idx];
    }
    s_data[c] = val;
    __syncthreads();

    float max_val = -INFINITY;
    for (int i = 0; i < channels; i++) {
        max_val = fmaxf(max_val, s_data[i]);
    }
    __syncthreads();

    float exp_val = 0.0f;
    if (c < channels) {
        exp_val = expf(s_data[c] - max_val);
        s_data[c] = exp_val;
    }
    __syncthreads();

    float sum_val = 0.0f;
    for (int i = 0; i < channels; i++) {
        sum_val += s_data[i];
    }
    __syncthreads();

    if (c < channels) {
        float softmax_val = s_data[c] / sum_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-softmax_val));
        int output_idx = base + c * (D * H * W);
        output[output_idx] = sigmoid_val;
    }
}

torch::Tensor softmax_sigmoid_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    int batch = input.size(0);
    int channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::empty_like(input);
    int total_spatial = batch * D * H * W;
    int block_size = channels;
    int num_blocks = total_spatial;

    softmax_sigmoid_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, D, H, W
    );

    return output;
}
"""

softmax_sigmoid_cpp = "torch::Tensor softmax_sigmoid_cuda(torch::Tensor input);"

softmax_sigmoid = load_inline(
    name="softmax_sigmoid",
    cpp_sources=softmax_sigmoid_cpp,
    cuda_sources=softmax_sigmoid_source,
    functions=["softmax_sigmoid_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax_sigmoid = softmax_sigmoid

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        x = self.softmax_sigmoid.softmax_sigmoid_cuda(x)
        return x