import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused ConvTranspose3d + Mean Pooling + Bias Add + Softmax + Tanh + Scaling
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Utility function to calculate 5D tensor index: (b, c, d, h, w)
__device__ int get_index(int b, int c, int d, int h, int w, int C, int D, int H, int W) {
    return ((b * C + c) * D + d) * H * W + h * W + w;
}

__global__ void fused_conv_transpose_mean_softmax_tanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int IN_C, int OUT_C, int D, int H, int W,
    int K, int stride, int padding,
    float scaling_factor,
    int OUT_H, int OUT_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * OUT_H * OUT_W * OUT_C;
    
    if (idx >= total_elements) return;

    // Decode index: each thread handles one (b, h, w, c) combination
    int w = idx % OUT_W;
    int h = (idx / OUT_W) % OUT_H;
    int c = (idx / (OUT_H * OUT_W)) % OUT_C;
    int b = idx / (OUT_C * OUT_H * OUT_W);

    // Step 1: Perform conv_transpose3d + mean over depth
    float val = 0.0f;
    for (int ki = 0; ki < K; ++ki) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int i_h = h * stride - padding + kh;
                int i_w = w * stride - padding + kw;
                if (i_h >= 0 && i_h < H && i_w >= 0 && i_w < W) {
                    for (int ic = 0; ic < IN_C; ++ic) {
                        float w_val = weight[c * IN_C * K * K * K + ic * K * K * K + ki * K * K + kh * K + kw];
                        float in_val = 0.0f;
                        // Average over depth
                        for (int id = 0; id < D; ++id) {
                            in_val += input[b * IN_C * D * H * W + ic * D * H * W + id * H * W + i_h * W + i_w];
                        }
                        in_val /= D;
                        val += w_val * in_val;
                    }
                }
            }
        }
    }

    // Add bias
    val += bias[c];

    // Shared memory for softmax per spatial location: OUT_C floats per spatial site
    extern __shared__ float shared_data[];
    int spatial_idx = h * OUT_W + w;  // Local spatial index within batch
    int offset = spatial_idx * OUT_C;
    shared_data[threadIdx.x] = val;
    __syncthreads();

    // First pass: find max across channels for softmax stability
    float max_val = -FLT_MAX;
    for (int cc = 0; cc < OUT_C; ++cc) {
        float temp_val = shared_data[cc];
        max_val = fmaxf(max_val, temp_val);
    }
    __syncthreads();

    // Second pass: compute exp(val - max) and sum
    float exp_val = __expf(val - max_val);
    float exp_sum = 0.0f;
    for (int cc = 0; cc < OUT_C; ++cc) {
        float temp_val = shared_data[cc];
        exp_sum += __expf(temp_val - max_val);
    }
    __syncthreads();

    // Softmax: exp_val / (exp_sum + eps)
    float softmax_out = exp_val / (exp_sum + 1e-10f);

    // Tanh and scale
    float result = tanhf(softmax_out) * scaling_factor;

    // Write output
    output[idx] = result;
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    int stride,
    int padding,
    int K,
    int OUT_H,
    int OUT_W) {
    
    int B = input.size(0);
    int IN_C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int OUT_C = weight.size(0);

    auto output = torch::empty({B, OUT_C, OUT_H, OUT_W}, input.options());

    const int total_elements = B * OUT_C * OUT_H * OUT_W;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    // Shared memory for softmax: OUT_C floats per spatial location
    size_t shared_mem_size = OUT_C * sizeof(float);

    fused_conv_transpose_mean_softmax_tanh_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, IN_C, OUT_C, D, H, W,
        K, stride, padding,
        scaling_factor,
        OUT_H, OUT_W
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    int stride,
    int padding,
    int K,
    int OUT_H,
    int OUT_W);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scaling_factor = scaling_factor

        # Initialize transposed convolution weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

        # Output spatial dimensions computed from input config
        # Assuming input spatial dimensions are fixed (from problem context)
        self.OUT_H = 128
        self.OUT_W = 128

        # Bind the fused CUDA function
        self.fused_op = fused_op.fused_op_cuda

    def forward(self, x):
        return self.fused_op(
            x,
            self.weight,
            self.bias,
            self.scaling_factor,
            self.stride,
            self.padding,
            self.kernel_size,
            self.OUT_H,
            self.OUT_W
        )