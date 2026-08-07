import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + Softmax + Sigmoid
fused_conv_transpose_softmax_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Efficient fused kernel for ConvTranspose3d + Softmax (along dim=1) + Sigmoid
__global__ void fused_conv_transpose_softmax_sigmoid_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    // Input dimensions
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions
    int out_channels, int out_depth, int out_height, int out_width,
    // Conv params
    int kernel_size, int stride, int padding, int output_padding,
    // Weight dimensions
    int weight_depth, int weight_height, int weight_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int d = (idx / (out_width * out_height)) % out_depth;
    int c = (idx / (out_width * out_height * out_depth)) % out_channels;
    int b = idx / (out_width * out_height * out_depth * out_channels);

    // ConvTranspose3d: iterate over input channels and kernel
    float sum = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        for (int kz = 0; kz < kernel_size; ++kz) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    // Compute input position
                    int i_d = d - (kz - padding) * stride;
                    int i_h = h - (ky - padding) * stride;
                    int i_w = w - (kx - padding) * stride;

                    // Check bounds
                    if (i_d >= 0 && i_d < in_depth && i_h >= 0 && i_h < in_height && i_w >= 0 && i_w < in_width) {
                        int input_idx = b * in_channels * in_depth * in_height * in_width +
                                        ci * in_depth * in_height * in_width +
                                        i_d * in_height * in_width +
                                        i_h * in_width +
                                        i_w;
                        int weight_idx = c * in_channels * weight_depth * weight_height * weight_width +
                                         ci * weight_depth * weight_height * weight_width +
                                         kz * weight_height * weight_width +
                                         ky * weight_width +
                                         kx;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    if (bias) sum += bias[c];

    output[idx] = sum;

    // Softmax is applied across channel dimension (dim=1), so we need to compute per spatial position
    // We'll do softmax in a separate step due to data dependency, but fuse the exp and normalization
    // We use two-pass softmax: first compute max for numerical stability
    __shared__ float shared_data[32]; // assuming max 32 channels for shared memory
    float max_val = -FLT_MAX;
    for (int cc = 0; cc < out_channels; ++cc) {
        int temp_idx = b * out_channels * out_depth * out_height * out_width +
                       cc * out_depth * out_height * out_width +
                       d * out_height * out_width +
                       h * out_width +
                       w;
        float val = output[temp_idx];
        max_val = fmaxf(max_val, val);
    }

    float exp_sum = 0.0f;
    float current_val = output[idx];
    current_val = expf(current_val - max_val);
    output[idx] = current_val;

    // Block-level reduction to compute sum
    int tid = threadIdx.x;
    shared_data[tid] = current_val;
    __syncthreads();

    for (int stride_val = blockDim.x / 2; stride_val > 0; stride_val >>= 1) {
        if (tid < stride_val) {
            shared_data[tid] += shared_data[tid + stride_val];
        }
        __syncthreads();
    }

    float sum_exp = shared_data[0];

    // Final reduction across blocks not implemented for simplicity; using atomic for now
    // In practice, use proper multi-block reduction
    float* block_sum = &output[batch_size * out_channels * out_depth * out_height * out_width + blockIdx.x];
    if (tid == 0) *block_sum = sum_exp;
    __syncthreads();

    // Re-read sum (simplified; in practice use global reduction)
    // Approximate by assuming single block per spatial position
    float final_sum = *block_sum;
    float softmax_out = current_val / (final_sum + 1e-8f);

    // Sigmoid
    softmax_out = 1.0f / (1.0f + expf(-softmax_out));

    output[idx] = softmax_out;
}
"""

fused_conv_transpose_softmax_sigmoid_cpp = """
torch::Tensor fused_conv_transpose_softmax_sigmoid_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding, int output_padding,
    int D, int H, int W);
"""

# Note: The above kernel is a simplified version. For correctness and performance, we split the operations
# and use PyTorch's built-in transposed conv, then fuse softmax+sigmoid in CUDA.

# Let's instead do: use native ConvTranspose3d, then fuse Softmax + Sigmoid in CUDA
# Because fusing transposed conv is very complex and error-prone

softmax_sigmoid_fusion_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

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
    int n_elements = batch_size * total_spatial;

    // Each block handles one spatial location (d, h, w) across all channels for a given batch
    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;
    float inv_sum_exp = 0.0f;

    // Step 1: find max across channels for numerical stability
    for (int c = tid; c < channels; c += total_threads) {
        int idx = batch_idx * channels * total_spatial + c * total_spatial + spatial_idx;
        float val = input[idx];
        max_val = fmaxf(max_val, val);
    }

    // Reduce max_val across threads
    __shared__ float shared_max[256];
    shared_max[tid] = max_val;
    for (int stride_val = total_threads / 2; stride_val > 0; stride_val >>= 1) {
        __syncthreads();
        if (tid < stride_val) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride_val]);
        }
    }
    __syncthreads();
    max_val = shared_max[0];

    // Step 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int c = tid; c < channels; c += total_threads) {
        int idx = batch_idx * channels * total_spatial + c * total_spatial + spatial_idx;
        float val = input[idx];
        float exp_val = expf(val - max_val);
        local_sum += exp_val;
        // Store exp_val temporarily in output
        output[idx] = exp_val;
    }

    // Reduce sum
    __shared__ float shared_sum[256];
    shared_sum[tid] = local_sum;
    for (int stride_val = total_threads / 2; stride_val > 0; stride_val >>= 1) {
        __syncthreads();
        if (tid < stride_val) {
            shared_sum[tid] += shared_sum[tid + stride_val];
        }
    }
    __syncthreads();
    float total_sum = shared_sum[0];

    // Step 3: compute softmax and then sigmoid
    float inv_total_sum = 1.0f / (total_sum + 1e-8f);
    for (int c = tid; c < channels; c += total_threads) {
        int idx = batch_idx * channels * total_spatial + c * total_spatial + spatial_idx;
        float softmax_val = output[idx] * inv_total_sum;
        // Sigmoid
        float sigmoid_val = 1.0f / (1.0f + expf(-softmax_val));
        output[idx] = sigmoid_val;
    }
}
"""

softmax_sigmoid_fusion_cpp = """
torch::Tensor softmax_sigmoid_fused_cuda(
    torch::Tensor input,
    int64_t dim);
"""

# Compile the fused softmax + sigmoid kernel
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