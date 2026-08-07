import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv3D + Softmax
fused_conv_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function to calculate index in 5D tensor
__device__ int get_index(int n, int c, int d, int h, int w, int channels, int depth, int height, int width) {
    return n * (channels * depth * height * width) + c * (depth * height * width) + d * (height * width) + h * width + w;
}

__global__ void fused_conv3d_softmax(const float* input, const float* weight, const float* bias,
                                     float* output, int batch_size, int in_channels, int out_channels,
                                     int input_depth, int input_height, int input_width,
                                     int output_depth, int output_height, int output_width,
                                     int kernel_size, int pad, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (idx >= total_elements) return;

    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int d = (idx / (output_width * output_height)) % output_depth;
    int c = (idx / (output_width * output_height * output_depth)) % out_channels;
    int n = idx / (output_width * output_height * output_depth * out_channels);

    // Convolution computation
    float conv_val = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_d = d * stride - pad + kd;
                    int in_h = h * stride - pad + kh;
                    int in_w = w * stride - pad + kw;

                    if (in_d >= 0 && in_d < input_depth && in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                        int in_idx = get_index(n, ci, in_d, in_h, in_w, in_channels, input_depth, input_height, input_width);
                        int w_idx = c * (in_channels * kernel_size * kernel_size * kernel_size) + 
                                    ci * (kernel_size * kernel_size * kernel_size) + 
                                    kd * (kernel_size * kernel_size) + kh * kernel_size + kw;
                        conv_val += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    conv_val += bias[c];

    // Store pre-softmax values in temporary storage (reuse output for now)
    int out_idx = get_index(n, c, d, h, w, out_channels, output_depth, output_height, output_width);
    output[out_idx] = conv_val;

    // Synchronize to ensure all outputs are written
    __syncthreads();

    // Softmax across channel dimension for each spatial location
    float max_val = -FLT_MAX;
    for (int cc = 0; cc < out_channels; ++cc) {
        float val = output[get_index(n, cc, d, h, w, out_channels, output_depth, output_height, output_width)];
        max_val = fmaxf(max_val, val);
    }

    float sum = 0.0f;
    for (int cc = 0; cc < out_channels; ++cc) {
        float val = output[get_index(n, cc, d, h, w, out_channels, output_depth, output_height, output_width)];
        val = expf(val - max_val);
        output[get_index(n, cc, d, h, w, out_channels, output_depth, output_height, output_width)] = val;
        sum += val;
    }

    float inv_sum = 1.0f / sum;
    for (int cc = 0; cc < out_channels; ++cc) {
        int temp_idx = get_index(n, cc, d, h, w, out_channels, output_depth, output_height, output_width);
        output[temp_idx] *= inv_sum;
    }
}
"""

fused_conv_softmax_cpp = """
torch::Tensor fused_conv3d_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int pad,
    int stride);
"""

fused_conv_softmax = load_inline(
    name="fused_conv_softmax",
    cpp_sources=fused_conv_softmax_cpp,
    cuda_sources=fused_conv_softmax_source,
    functions=["fused_conv3d_softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Conv3D + Softmax in CUDA, followed by two MaxPool3d layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        
        # Create the convolution parameters manually
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)
        
        # Pooling layers (use built-in for simplicity and efficiency)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)
        
        # Store kernel config
        self.pad = kernel_size // 2
        self.stride = 1

    def forward(self, x):
        batch_size, _, input_depth, input_height, input_width = x.shape
        
        # Calculate output dimensions for Conv3D
        output_depth = (input_depth + 2 * self.pad - self.kernel_size) // self.stride + 1
        output_height = (input_height + 2 * self.pad - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.pad - self.kernel_size) // self.stride + 1
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=x.dtype)
        
        # Launch fused CUDA kernel
        grid_size = (batch_size * self.out_channels * output_depth * output_height * output_width + 255) // 256
        block_size = 256
        
        fused_conv_softmax.fused_conv3d_softmax_cuda(
            x, self.weight, self.bias,
            input_depth, input_height, input_width,
            output_depth, output_height, output_width,
            self.kernel_size, self.pad, self.stride
        )
        
        # Perform max pooling operations using built-in functions
        x = self.pool1(output)
        x = self.pool2(x)
        
        return x