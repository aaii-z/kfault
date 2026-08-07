import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution + GELU + global average pooling
fused_conv_gelu_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Approximate GELU function
__device__ __forceinline__ float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float approx_coef = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + approx_coef * x * x)));
}

__global__ void fused_conv_gelu_pool_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    // Calculate output spatial dimensions
    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int spatial_size = output_height * output_width;

    // Each block handles one output channel for all batches
    int c = blockIdx.x;
    int b = blockIdx.y;

    if (c >= out_channels || b >= batch_size) return;

    extern __shared__ float shared_mem[];
    float* channel_sums = shared_mem;

    // Initialize shared memory
    if (threadIdx.x == 0) {
        channel_sums[0] = 0.0f;
    }
    __syncthreads();

    // Perform convolution, GELU, and pooling in one pass
    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            for (int ic = 0; ic < in_channels; ++ic) {
                int h_in = -padding + kh * dilation;
                int w_in = -padding + kw * dilation;
                
                for (int h = threadIdx.x; h < output_height; h += blockDim.x) {
                    for (int w = threadIdx.y; w < output_width; w += blockDim.y) {
                        int h_actual = h * stride + h_in;
                        int w_actual = w * stride + w_in;
                        
                        if (h_actual >= 0 && h_actual < input_height && w_actual >= 0 && w_actual < input_width) {
                            int input_idx = ((b * in_channels + ic) * input_height + h_actual) * input_width + w_actual;
                            int weight_idx = ((c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                            
                            float val = input[input_idx] * weight[weight_idx];
                            atomicAdd(&sum, val);
                        }
                    }
                }
            }
        }
    }

    // Add bias and apply GELU
    if (bias != nullptr) {
        sum += bias[c];
    }
    float activated = gelu_approx(sum);

    // Atomic add to shared memory for reduction
    atomicAdd(&channel_sums[0], activated);
    __syncthreads();

    // Write final averaged result
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        output[b * out_channels + c] = channel_sums[0] / spatial_size;
    }
}

torch::Tensor fused_conv_gelu_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels}, input.options());

    dim3 block(16, 16);
    dim3 grid(out_channels, batch_size);
    size_t shared_mem_size = sizeof(float);

    fused_conv_gelu_pool_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        groups
    );

    return output;
}
"""

fused_conv_gelu_pool_cpp_source = """
torch::Tensor fused_conv_gelu_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
);
"""

# Compile the inline CUDA code
fused_conv_gelu_pool = load_inline(
    name="fused_conv_gelu_pool",
    cpp_sources=fused_conv_gelu_pool_cpp_source,
    cuda_sources=fused_conv_gelu_pool_source,
    functions=["fused_conv_gelu_pool_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudnn", "-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize weights and bias similarly to nn.Conv2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Set convolution parameters
        self.stride = 1
        self.padding = kernel_size // 2
        self.dilation = 1
        self.groups = 1

    def forward(self, x):
        return fused_conv_gelu_pool.fused_conv_gelu_pool_cuda(
            x, 
            self.weight, 
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )