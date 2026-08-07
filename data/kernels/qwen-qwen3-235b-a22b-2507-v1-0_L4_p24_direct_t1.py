import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 3D convolution, min reduction, and softmax
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void check_errors(const char* msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s: %s\\n", msg, cudaGetErrorString(err));
  }
}

// Fused forward pass: Conv3D -> Min(dim=2) -> Softmax(dim=1)
std::vector<torch::Tensor> fused_conv_min_softmax_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int out_channels, int kernel_size, int pad,
    int batch_size, int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    auto options = torch::TensorOptions()
                      .dtype(input.dtype())
                      .device(input.device())
                      .requires_grad(false);
    
    // Output after min reduction: [batch_size, out_channels, out_height, out_width]
    auto min_output = torch::zeros({batch_size, out_channels, out_height, out_width}, options);
    
    // Final output after softmax
    auto softmax_output = torch::zeros({batch_size, out_channels, out_height, out_width}, options);
    
    // Temporary buffer for convolution output: [batch_size, out_channels, out_depth, out_height, out_width]
    auto conv_buffer = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, options);
    
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();
    float* conv_ptr = conv_buffer.data_ptr<float>();
    float* min_out_ptr = min_output.data_ptr<float>();
    float* softmax_out_ptr = softmax_output.data_ptr<float>();
    
    const int k_half = kernel_size / 2;
    const int stride = 1;
    const int threads_per_block = 256;
    
    // Total number of elements in conv output
    int total_conv = batch_size * out_channels * out_depth * out_height * out_width;
    int blocks_conv = (total_conv + threads_per_block - 1) / threads_per_block;
    
    // Conv3D kernel
    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Lambda to compute linear index
    auto get_conv_idx = [&](int b, int c, int d, int h, int w) {
        return b * (out_channels * out_depth * out_height * out_width) +
               c * (out_depth * out_height * out_width) +
               d * (out_height * out_width) +
               h * out_width + w;
    };
    
    auto get_input_idx = [&](int b, int c, int d, int h, int w) {
        return b * (in_channels * in_depth * in_height * in_width) +
               c * (in_depth * in_height * in_width) +
               d * (in_height * in_width) +
               h * in_width + w;
    };
    
    // Conv kernel
    dim3 block(threads_per_block);
    dim3 grid(blocks_conv);
    
    auto conv_kernel = []__global__(
        const float* input,
        const float* weight,
        const float* bias,
        float* output,
        int batch_size,
        int in_channels,
        int in_depth,
        int in_height,
        int in_width,
        int out_channels,
        int out_depth,
        int out_height,
        int out_width,
        int kernel_size,
        int pad
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch_size * out_channels * out_depth * out_height * out_width;
        if (idx >= total) return;
        
        int w = idx % out_width;
        int h = (idx / out_width) % out_height;
        int d = (idx / (out_width * out_height)) % out_depth;
        int c = (idx / (out_width * out_height * out_depth)) % out_channels;
        int b = idx / (out_width * out_height * out_depth * out_channels);
        
        float val = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int id = -pad + d + kd;
                        int ih = -pad + h + kh;
                        int iw = -pad + w + kw;
                        
                        if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                          ic * (in_depth * in_height * in_width) +
                                          id * (in_height * in_width) +
                                          ih * in_width + iw;
                            int weight_idx = c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                           ic * (kernel_size * kernel_size * kernel_size) +
                                           kd * (kernel_size * kernel_size) +
                                           kh * kernel_size + kw;
                            val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        val += bias[c];
        output[idx] = val;
    };
    
    conv_kernel<<<grid, block, 0, stream>>>(
        input_ptr, weight_ptr, bias_ptr, conv_ptr,
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, out_depth, out_height, out_width,
        kernel_size, pad
    );
    
    // Min reduction along depth (dim=2): [B, C, D, H, W] -> [B, C, H, W]
    int total_min = batch_size * out_channels * out_height * out_width;
    int blocks_min = (total_min + threads_per_block - 1) / threads_per_block;
    dim3 grid_min(blocks_min);
    
    auto min_kernel = []__global__(
        const float* conv_output,
        float* min_output,
        int batch_size,
        int out_channels,
        int out_depth,
        int out_height,
        int out_width
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * out_channels * out_height * out_width) return;
        
        int w = idx % out_width;
        int h = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % out_channels;
        int b = idx / (out_width * out_height * out_channels);
        
        float min_val = FLT_MAX;
        for (int d = 0; d < out_depth; ++d) {
            int conv_idx = b * (out_channels * out_depth * out_height * out_width) +
                          c * (out_depth * out_height * out_width) +
                          d * (out_height * out_width) +
                          h * out_width + w;
            min_val = fminf(min_val, conv_output[conv_idx]);
        }
        min_output[idx] = min_val;
    };
    
    min_kernel<<<grid_min, block, 0, stream>>>(
        conv_ptr, min_out_ptr,
        batch_size, out_channels, out_depth, out_height, out_width
    );
    
    // Softmax along channel dimension (dim=1): [B, C, H, W]
    int spatial_size = out_height * out_width;
    int blocks_softmax = (spatial_size + threads_per_block - 1) / threads_per_block;
    dim3 grid_softmax(blocks_softmax);
    
    auto softmax_kernel = []__global__(
        const float* input,
        float* output,
        int batch_size,
        int out_channels,
        int out_height,
        int out_width
    ) {
        int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (spatial_idx >= out_height * out_width) return;
        
        int h = spatial_idx / out_width;
        int w = spatial_idx % out_width;
        
        // First pass: find max over channels and batches
        float max_val = -FLT_MAX;
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < out_channels; ++c) {
                int idx = b * (out_channels * out_height * out_width) +
                         c * (out_height * out_width) +
                         h * out_width + w;
                max_val = fmaxf(max_val, input[idx]);
            }
        }
        
        // Second pass: compute exp(x - max) and sum
        float sum_val = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < out_channels; ++c) {
                int idx = b * (out_channels * out_height * out_width) +
                         c * (out_height * out_width) +
                         h * out_width + w;
                float exp_val = expf(input[idx] - max_val);
                output[idx] = exp_val;
                sum_val += exp_val;
            }
        }
        
        // Third pass: normalize
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < out_channels; ++c) {
                int idx = b * (out_channels * out_height * out_width) +
                         c * (out_height * out_width) +
                         h * out_width + w;
                output[idx] /= sum_val;
            }
        }
    };
    
    softmax_kernel<<<grid_softmax, block, 0, stream>>>(
        min_out_ptr, softmax_out_ptr,
        batch_size, out_channels, out_height, out_width
    );
    
    return {min_output, softmax_output};
}
"""

cpp_source = """
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> fused_conv_min_softmax_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int out_channels, int kernel_size, int pad,
    int batch_size, int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width);
"""

# Compile the fused CUDA kernel
fused_ops = load_inline(
    name="fused_conv_min_softmax",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_min_softmax_forward"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dim = dim
        self.pad = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=self.pad)
        
    def forward(self, x):
        batch_size, _, in_depth, in_height, in_width = x.shape
        # With padding = kernel_size // 2 and stride=1, output spatial dims are same as input
        out_depth = in_depth
        out_height = in_height
        out_width = in_width
        
        result = fused_ops.fused_conv_min_softmax_forward(
            x, self.conv.weight, self.conv.bias,
            self.in_channels, self.out_channels, self.kernel_size, self.pad,
            batch_size, in_depth, in_height, in_width,
            out_depth, out_height, out_width
        )
        return result[1]  # Return softmax output