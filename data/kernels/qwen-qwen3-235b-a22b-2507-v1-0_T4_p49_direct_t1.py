import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fusing Softmax (dim=1) and Sigmoid
softmax_sigmoid_fusion_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

__global__ void softmax_sigmoid_fused_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width
) {
    int spatial_idx = blockIdx.x;
    int tid = threadIdx.x;
    int total_spatial = depth * height * width;
    if (spatial_idx >= total_spatial) return;

    int total_threads = blockDim.x;
    int batch_idx = blockIdx.y;
    int batch_spatial_size = channels * total_spatial;
    int spatial_offset = batch_idx * batch_spatial_size + spatial_idx;

    // Shared memory for max and sum reduction
    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];

    // Step 1: Find maximum value across channels for numerical stability
    float max_val = -FLT_MAX;
    for (int c = tid; c < channels; c += total_threads) {
        int idx = spatial_offset + c * total_spatial;
        float val = input[idx];
        max_val = fmaxf(max_val, val);
    }

    // Reduce max_val across threads
    shared_max[tid] = max_val;
    __syncthreads();
    for (int stride_val = total_threads / 2; stride_val > 0; stride_val >>= 1) {
        if (tid < stride_val) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride_val]);
        }
        __syncthreads();
    }
    max_val = shared_max[0];

    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int c = tid; c < channels; c += total_threads) {
        int idx = spatial_offset + c * total_spatial;
        float val = input[idx];
        float exp_val = expf(val - max_val);
        local_sum += exp_val;
        output[idx] = exp_val;
    }

    // Reduce sum across threads
    shared_sum[tid] = local_sum;
    __syncthreads();
    for (int stride_val = total_threads / 2; stride_val > 0; stride_val >>= 1) {
        if (tid < stride_val) {
            shared_sum[tid] += shared_sum[tid + stride_val];
        }
        __syncthreads();
    }
    float total_sum = shared_sum[0];

    // Final softmax and sigmoid
    float inv_total_sum = 1.0f / (total_sum + 1e-8f);
    for (int c = tid; c < channels; c += total_threads) {
        int idx = spatial_offset + c * total_spatial;
        float softmax_val = output[idx] * inv_total_sum;
        float sigmoid_val = 1.0f / (1.0f + expf(-softmax_val));
        output[idx] = sigmoid_val;
    }
}

torch::Tensor softmax_sigmoid_fused_cuda(torch::Tensor input, int64_t dim) {
    auto input_contig = input.contiguous();
    int64_t batch_size = input.size(0);
    int64_t channels = input.size(1);
    int64_t depth = input.size(2);
    int64_t height = input.size(3);
    int64_t width = input.size(4);

    auto output = torch::empty_like(input_contig);

    const int total_spatial = depth * height * width;
    const int block_size = 256;
    dim3 grid(total_spatial, batch_size);  // each block handles one spatial location per batch
    dim3 block(block_size);

    softmax_sigmoid_fused_kernel<<<grid, block>>>(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, depth, height, width
    );

    return output;
}
"""

softmax_sigmoid_fusion_cpp = """
torch::Tensor softmax_sigmoid_fused_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the fused kernel
softmax_sigmoid_fused = load_inline(
    name="softmax_sigmoid_fused",
    cpp_sources=softmax_sigmoid_fusion_cpp,
    cuda_sources=softmax_sigmoid_fusion_source,
    functions=["softmax_sigmoid_fused_cuda"],
    verbose=False,
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel to fuse Softmax and Sigmoid operations.
    ConvTranspose3d remains as-is due to complexity, but Softmax(dim=1) + Sigmoid are fused.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        self.softmax_sigmoid_fused = softmax_sigmoid_fused

    def forward(self, x):
        x = self.conv_transpose(x)
        # Apply fused softmax (dim=1) + sigmoid
        x = self.softmax_sigmoid_fused.softmax_sigmoid_fused_cuda(x, 1)
        return x