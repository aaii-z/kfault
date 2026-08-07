import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 3D convolution, min reduction, and softmax
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function to calculate 3D index
__device__ int get_index(int b, int c, int d, int h, int w, 
                        int channels, int depth, int height, int width) {
    return b * (channels * depth * height * width) + 
           c * (depth * height * width) + 
           d * (height * width) + 
           h * width + w;
}

__global__ void fused_conv_min_softmax(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    // Input dimensions
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions after conv
    int out_channels, int out_depth, int out_height, int out_width,
    // Conv parameters
    int kernel_size, int pad, int stride,
    // Min reduction dimension (2 = depth)
    int min_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Decode index
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % out_channels;
    int b = idx / (out_width * out_height * out_channels);
    
    // Perform convolution (with implicit padding)
    float conv_val = 0.0f;
    int k_half = kernel_size / 2;
    
    for (int dc = 0; dc < in_channels; ++dc) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int d_in = -pad + stride * 0 + kd;
                    int h_in = -pad + stride * h + kh;
                    int w_in = -pad + stride * w + kw;
                    
                    if (d_in >= 0 && d_in < in_depth && 
                        h_in >= 0 && h_in < in_height && 
                        w_in >= 0 && w_in < in_width) {
                        int in_idx = get_index(b, dc, d_in, h_in, w_in, 
                                              in_channels, in_depth, in_height, in_width);
                        int weight_idx = c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                        dc * (kernel_size * kernel_size * kernel_size) +
                                        kd * (kernel_size * kernel_size) +
                                        kh * kernel_size + kw;
                        conv_val += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    conv_val += bias[c];
    
    // Store conv result in shared memory for min reduction across depth
    extern __shared__ float shared_data[];
    shared_data[threadIdx.x] = conv_val;
    __syncthreads();
    
    // Now perform min reduction along depth dimension
    // Each thread block handles one (b, c, h, w) location, reducing over depth
    float min_val = FLT_MAX;
    for (int d = 0; d < out_depth; ++d) {
        float val = 0.0f;
        for (int dc = 0; dc < in_channels; ++dc) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int d_in = -pad + stride * d + kd;
                        int h_in = -pad + stride * h + kh;
                        int w_in = -pad + stride * w + kw;
                        
                        if (d_in >= 0 && d_in < in_depth && 
                            h_in >= 0 && h_in < in_height && 
                            w_in >= 0 && w_in < in_width) {
                            int in_idx = get_index(b, dc, d_in, h_in, w_in, 
                                                  in_channels, in_depth, in_height, in_width);
                            int weight_idx = c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                            dc * (kernel_size * kernel_size * kernel_size) +
                                            kd * (kernel_size * kernel_size) +
                                            kh * kernel_size + kw;
                            val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        val += bias[c];
        min_val = fminf(min_val, val);
    }
    
    // Now we have min-reduced value, apply softmax across channels
    // First, we need to compute max across channels for numerical stability
    float max_val = -FLT_MAX;
    if (threadIdx.x < out_channels) {
        max_val = min_val; // current channel's min value
    }
    __syncthreads();
    
    // Use block-wide reduction to find max
    for (int stride_val = blockDim.x / 2; stride_val > 0; stride_val >>= 1) {
        if (threadIdx.x < stride_val && (threadIdx.x + stride_val) < out_channels) {
            float other = shared_data[threadIdx.x + stride_val];
            max_val = fmaxf(max_val, other);
        }
        __syncthreads();
    }
    
    // Broadcast max_val to all threads
    float global_max = shared_data[0];
    __syncthreads();
    max_val = global_max;
    
    // Compute exp(x - max) for softmax
    float exp_val = expf(min_val - max_val);
    shared_data[threadIdx.x] = exp_val;
    __syncthreads();
    
    // Sum all exp values
    float sum_val = 0.0f;
    for (int i = 0; i < blockDim.x; ++i) {
        sum_val += shared_data[i];
    }
    
    // Final softmax output
    output[idx] = exp_val / sum_val;
}
"""

cpp_source = """
torch::Tensor fused_conv_min_softmax_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int out_channels, int kernel_size, int pad,
    int batch_size, int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int min_dim);
"""

# Note: The above kernel has a flaw in shared memory usage for softmax reduction.
# Let's simplify and do proper fusion: Conv -> Min -> Softmax in three steps but fused kernel.

# Rewriting with correct logic: We'll do separate kernels but fused in one call
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Launch configuration
#define THREADS_PER_BLOCK 256

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
    
    // Get raw pointers
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();
    float* min_out_ptr = min_output.data_ptr<float>();
    float* softmax_out_ptr = softmax_output.data_ptr<float>();
    
    // Temporary storage for convolution outputs
    auto conv_buffer = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, options);
    float* conv_ptr = conv_buffer.data_ptr<float>();
    
    const int k_half = kernel_size / 2;
    const int stride = 1;
    
    // 1. Perform Conv3D
    int total_conv_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int blocks_conv = (total_conv_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Lambda for getting linear index
    auto get_idx = [&](int b, int c, int d, int h, int w) {
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
    
    // Conv3D kernel
    auto conv_kernel = []__device__(const float* input, const float* weight, const float* bias,
                                    float* output,
                                    int b, int c, int d, int h, int w,
                                    int in_channels, int in_depth, int in_height, int in_width,
                                    int kernel_size, int pad, int stride) {
        float val = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int id = -pad + stride * d + kd;
                        int ih = -pad + stride * h + kh;
                        int iw = -pad + stride * w + kw;
                        
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
        return val;
    };
    
    // Launch conv kernel
    auto conv_kernel_wrapper = []__global__(const float* input, const float* weight, const float* bias,
                                            float* output,
                                            int batch_size, int out_channels, int out_depth, int out_height, int out_width,
                                            int in_channels, int in_depth, int in_height, in_width,
                                            int kernel_size, int pad, int stride) {
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
                        int id = -pad + stride * d + kd;
                        int ih = -pad + stride * h + kh;
                        int iw = -pad + stride * w + kw;
                        
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
    
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(blocks_conv);
    
    conv_kernel_wrapper<<<grid, block, 0, stream>>>(
        input_ptr, weight_ptr, bias_ptr, conv_ptr,
        batch_size, out_channels, out_depth, out_height, out_width,
        in_channels, in_depth, in_height, in_width,
        kernel_size, pad, stride
    );
    
    // 2. Min reduction along depth (dim=2)
    auto min_kernel = []__global__(const float* conv_output, float* min_output,
                                   int batch_size, int out_channels, int out_depth, int out_height, int out_width) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch_size * out_channels * out_height * out_width;
        if (idx >= total) return;
        
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
    
    int total_min_elements = batch_size * out_channels * out_height * out_width;
    int blocks_min = (total_min_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid_min(blocks_min);
    
    min_kernel<<<grid_min, block, 0, stream>>>(
        conv_ptr, min_out_ptr,
        batch_size, out_channels, out_depth, out_height, out_width
    );
    
    // 3. Softmax along channel dimension (dim=1)
    auto softmax_kernel = []__global__(const float* input, float* output,
                                       int batch_size, int out_channels, int out_height, int out_width) {
        int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int spatial_size = out_height * out_width;
        if (spatial_idx >= spatial_size) return;
        
        int h = spatial_idx / out_width;
        int w = spatial_idx % out_width;
        
        // Find max in channel dimension for numerical stability
        float max_val = -FLT_MAX;
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < out_channels; ++c) {
                int idx = b * (out_channels * out_height * out_width) +
                         c * (out_height * out_width) +
                         h * out_width + w;
                max_val = fmaxf(max_val, input[idx]);
            }
        }
        
        // Compute exp and sum
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
        
        // Normalize
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < out_channels; ++c) {
                int idx = b * (out_channels * out_height * out_width) +
                         c * (out_height * out_width) +
                         h * out_width + w;
                output[idx] /= sum_val;
            }
        }
    };
    
    int spatial_size = out_height * out_width;
    int blocks_softmax = (spatial_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid_softmax(blocks_softmax);
    
    softmax_kernel<<<grid_softmax, block, 0, stream>>>(
        min_out_ptr, softmax_out_ptr,
        batch_size, out_channels, out_height, out_width
    );
    
    return {min_output, softmax_output};
}
"""

cpp_source = """
std::vector<torch::Tensor> fused_conv_min_softmax_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int out_channels, int kernel_size, int pad,
    int batch_size, int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width);
"""

# Compile the CUDA kernel
fused_ops = load_inline(
    name="fused_conv_min_softmax",
    cpp_sources=cpp_source,
    cuda_sources=fused_op_source,
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
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        batch_size, _, in_depth, in_height, in_width = x.shape
        out_depth = in_depth + 2 * self.pad - self.kernel_size + 1
        out_height = in_height + 2 * self.pad - self.kernel_size + 1
        out_width = in_width + 2 * self.pad - self.kernel_size + 1
        
        result = fused_ops.fused_conv_min_softmax_forward(
            x, self.conv.weight, self.conv.bias,
            self.in_channels, self.out_channels, self.kernel_size, self.pad,
            batch_size, in_depth, in_height, in_width,
            out_depth, out_height, out_width
        )
        return result[1]  # Return softmax output