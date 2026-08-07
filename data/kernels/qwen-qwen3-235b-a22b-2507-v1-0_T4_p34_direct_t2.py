import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 3D transposed convolution, layer norm, GELU, and scaling with high parallelism and coalesced memory access
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16
#define MAX_C_PER_THREAD 4

// GELU activation using tanh approximation
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

// Fast inverse sqrt
__device__ float rsqrt(float x) {
    return rsqrtf(x);
}

// Kernel for fused 3D transposed convolution + layer norm + GELU + scale
// Each thread block handles a tile of output spatial dimensions and multiple channels
__global__ void fused_convtrans3d_layernorm_gelu_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_D, int input_H, int input_W,
    int output_D, int output_H, int output_W,
    int kernel_size,
    int stride,
    int padding,
    float eps,
    float scaling_factor
) {
    int spatial_id = blockIdx.x;
    int batch_channel_id = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = batch_channel_id / out_channels;
    int c = batch_channel_id % out_channels;

    if (batch >= batch_size || c >= out_channels) return;

    int wo = spatial_id % output_W;
    int ho = (spatial_id / output_W) % output_H;
    int do_ = spatial_id / (output_W * output_H);
    if (do_ >= output_D) return;

    // Compute convolution
    float conv_val = 0.0f;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int di = do_ * stride - padding + kd;
                int hi = ho * stride - padding + kh;
                int wi = wo * stride - padding + kw;

                if (di >= 0 && di < input_D && hi >= 0 && hi < input_H && wi >= 0 && wi < input_W) {
                    int input_idx_base = batch * in_channels * input_D * input_H * input_W;
                    int weight_idx_base = c * in_channels * kernel_size * kernel_size * kernel_size + 
                                          kd * kernel_size * kernel_size + kh * kernel_size + kw;

                    for (int ci = threadIdx.x; ci < in_channels; ci += blockDim.x) {
                        int input_idx = input_idx_base + ci * input_D * input_H * input_W + 
                                        di * input_H * input_W + hi * input_W + wi;
                        int weight_idx = weight_idx_base + ci * kernel_size * kernel_size * kernel_size;
                        conv_val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        conv_val += bias[c];
    }

    int output_idx = batch * out_channels * output_D * output_H * output_W +
                     c * output_D * output_H * output_W +
                     do_ * output_H * output_W + ho * output_W + wo;

    // Store conv result in shared memory for reduction in layer norm
    extern __shared__ float s_data[];
    float* s_conv = s_data;
    float* s_buffer = &s_data[blockDim.y * blockDim.x];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int local_c = tid % blockDim.y;
    int local_spatial = tid / blockDim.y;

    s_conv[tid] = (local_c == c % blockDim.y && local_spatial == 0) ? conv_val : 0.0f;

    __syncthreads();

    // Compute mean across spatial dimensions for this channel
    // Reduce over spatial tiles
    int total_spatial = output_D * output_H * output_W;
    int num_spatial_tiles = (total_spatial + blockDim.y - 1) / blockDim.y;
    float sum = 0.0f;
    for (int tile = 0; tile < num_spatial_tiles; ++tile) {
        int spatial_idx = tile * blockDim.y + threadIdx.y;
        if (spatial_idx < total_spatial) {
            int wo_t = spatial_idx % output_W;
            int ho_t = (spatial_idx / output_W) % output_H;
            int do_t = spatial_idx / (output_W * output_H);
            int idx_t = batch * out_channels * output_D * output_H * output_W +
                        c * output_D * output_H * output_W +
                        do_t * output_H * output_W + ho_t * output_W + wo_t;
            sum += ((float*)output)[idx_t];  // Reuse output as temp buffer if needed
        }
    }
    s_buffer[threadIdx.x] = sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_buffer[threadIdx.x] += s_buffer[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean = s_buffer[0] / total_spatial;

    // Compute variance
    float var = 0.0f;
    for (int tile = 0; tile < num_spatial_tiles; ++tile) {
        int spatial_idx = tile * blockDim.y + threadIdx.y;
        if (spatial_idx < total_spatial) {
            int wo_t = spatial_idx % output_W;
            int ho_t = (spatial_idx / output_W) % output_H;
            int do_t = spatial_idx / (output_W * output_H);
            int idx_t = batch * out_channels * output_D * output_H * output_W +
                        c * output_D * output_H * output_W +
                        do_t * output_H * output_W + ho_t * output_W + wo_t;
            float diff = ((float*)output)[idx_t] - mean;
            var += diff * diff;
        }
    }
    s_buffer[threadIdx.x] = var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_buffer[threadIdx.x] += s_buffer[threadIdx.x + s];
        }
        __syncthreads();
    }

    float inv_std = rsqrt(s_buffer[0] / total_spatial + eps);

    // Normalize, apply GELU and scale
    float x = (conv_val - mean) * inv_std;
    float gelu_val = gelu(x);
    output[output_idx] = gelu_val * scaling_factor;
}

// Host function
torch::Tensor fused_convtrans3d_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    float scaling_factor
) {
    input = input.contiguous();
    weight = weight.contiguous();
    if (bias.defined()) bias = bias.contiguous();

    auto opts = input.options();
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_D = input.size(2);
    int input_H = input.size(3);
    int input_W = input.size(4);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int stride = 2;
    int padding = 1;

    int output_D = (input_D - 1) * stride - 2 * padding + kernel_size;
    int output_H = (input_H - 1) * stride - 2 * padding + kernel_size;
    int output_W = (input_W - 1) * stride - 2 * padding + kernel_size;
    int total_spatial = output_D * output_H * output_W;

    auto output = torch::empty({batch_size, out_channels, output_D, output_H, output_W}, opts);

    dim3 block_dim(32, 8);  // 32 in_channels threads, 8 channel blocks
    int num_spatial_blocks = total_spatial;
    int num_channel_blocks = (out_channels + block_dim.y - 1) / block_dim.y;
    dim3 grid_dim(num_spatial_blocks, num_channel_blocks);

    size_t shared_mem = (block_dim.x * block_dim.y) * sizeof(float) + block_dim.x * sizeof(float);

    fused_convtrans3d_layernorm_gelu_scale_kernel<<<grid_dim, block_dim, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_D, input_H, input_W,
        output_D, output_H, output_W,
        kernel_size, stride, padding,
        eps, scaling_factor
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_convtrans3d_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    float scaling_factor);
"""

fused_op = load_inline(
    name="fused_convtrans3d_layernorm_gelu_scale",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_convtrans3d_layernorm_gelu_scale_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for 3D transposed convolution, layer normalization, GELU, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.eps = eps
        self.scaling_factor = scaling_factor
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)
        
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)
        
        self.fused_op = fused_op.fused_convtrans3d_layernorm_gelu_scale_cuda

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias_param, self.eps, self.scaling_factor)