import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + InstanceNorm2d + divide by constant
fused_conv_instancenorm_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function to calculate 2D convolution indices
__device__ int get_img_idx(int n, int c, int h, int w, int H, int W) {
    return n * (c * H * W) + c * (H * W) + h * W + w;
}

__global__ void fused_conv_instancenorm_divide_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int padding,
    int stride,
    float divide_by
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (output_width * output_height * out_channels);

    // Convolution computation
    float conv_val = 0.0f;
    int k_half = kernel_size / 2;
    
    for (int ki = 0; ki < kernel_size; ++ki) {
        for (int kj = 0; kj < kernel_size; ++kj) {
            int i_h = h * stride + ki - padding;
            int i_w = w * stride + kj - padding;
            
            if (i_h >= 0 && i_h < input_height && i_w >= 0 && i_w < input_width) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    int in_idx = get_img_idx(n, ic, i_h, i_w, input_height, input_width);
                    int weight_idx = c * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + ki * kernel_size + kj;
                    conv_val += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }
    conv_val += bias[c];

    // InstanceNorm: compute mean and variance over HxW for each (n, c)
    float mean = 0.0f;
    float var = 0.0f;
    
    int spatial_size = output_height * output_width;
    int base_out_idx = n * out_channels * spatial_size + c * spatial_size;
    
    // Use shared memory to reduce global memory reads
    extern __shared__ float shared_data[];
    float* shared_buf = shared_data;

    // Each thread loads one element
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        int hw = i;
        int w_loc = hw % output_width;
        int h_loc = hw / output_width;

        // Recompute conv for this spatial location
        float val = 0.0f;
        for (int ki = 0; ki < kernel_size; ++ki) {
            for (int kj = 0; kj < kernel_size; ++kj) {
                int i_h = h_loc * stride + ki - padding;
                int i_w = w_loc * stride + kj - padding;
                
                if (i_h >= 0 && i_h < input_height && i_w >= 0 && i_w < input_width) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        int in_idx = get_img_idx(n, ic, i_h, i_w, input_height, input_width);
                        int weight_idx = c * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + ki * kernel_size + kj;
                        val += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
        val += bias[c];
        shared_buf[i] = val;
    }
    __syncthreads();

    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < spatial_size; ++i) {
        sum += shared_buf[i];
    }
    mean = sum / spatial_size;

    // Compute variance
    float sum_sq = 0.0f;
    for (int i = 0; i < spatial_size; ++i) {
        float diff = shared_buf[i] - mean;
        sum_sq += diff * diff;
    }
    var = sum_sq / spatial_size;

    // Normalize and divide
    float eps = 1e-5f;
    float normalized = (conv_val - mean) / sqrt(var + eps);
    output[idx] = normalized / divide_by;
}

torch::Tensor fused_conv_instancenorm_divide_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t padding,
    int64_t stride,
    float divide_by
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);

    auto output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    auto output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    int total_elements = batch_size * out_channels * output_height * output_width;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    size_t shared_mem_size = output_height * output_width * sizeof(float);

    fused_conv_instancenorm_divide_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        padding,
        stride,
        divide_by
    );

    return output;
}
"""

fused_conv_instancenorm_divide_cpp_source = """
torch::Tensor fused_conv_instancenorm_divide_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t padding,
    int64_t stride,
    float divide_by
);
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_conv_instancenorm_divide",
    cpp_sources=fused_conv_instancenorm_divide_cpp_source,
    cuda_sources=fused_conv_instancenorm_divide_source,
    functions=["fused_conv_instancenorm_divide_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with fused Conv2d + InstanceNorm2d + divide-by-constant in CUDA.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2
        self.divide_by = divide_by

        # Create learnable parameters
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Use the fused CUDA kernel
        return fused_ops.fused_conv_instancenorm_divide_cuda(
            x,
            self.weight,
            self.bias,
            self.out_channels,
            self.kernel_size,
            self.padding,
            self.stride,
            self.divide_by
        )