import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 3D transposed convolution + bias add + residual operations
fused_convtranspose_bias_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Shared memory for tiling the input if needed (not used in direct implementation below)
// We use a direct approach: each thread computes one output element

__global__ void fused_convtranspose3d_bias_residual_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int T, int H, int W, int I, int O, int K, int S, int P, int OP,
    int Ti, int Hi, int Wi) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * O * Ti * Hi * Wi;
    
    if (idx >= total_elements) return;

    // Decode index: (b, o, ti, hi, wi)
    int wi = idx % Wi;
    int hi = (idx / Wi) % Hi;
    int ti = (idx / (Wi * Hi)) % Ti;
    int o = (idx / (Wi * Hi * Ti)) % O;
    int b = idx / (O * Ti * Hi * Wi);

    float sum = 0.0f;

    // Compute input indices range based on transposed convolution mapping
    int t_start = ti * S - P;
    int h_start = hi * S - P;
    int w_start = wi * S - P;

    for (int kt = 0; kt < K; ++kt) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int it = t_start + kt;
                int ih = h_start + kh;
                int iw = w_start + kw;

                // Check bounds
                if (it >= 0 && it < T && ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    // Weight index: (o, i, kt, kh, kw) -> (O, I, K, K, K)
                    // We store weight as (O, I, K, K, K)
                    int weight_idx = o * (I * K * K * K) + kt * (I * K * K) + kh * (I * K) + kw * I;
                    // Input index: (b, i, it, ih, iw) -> (B, I, T, H, W)
                    int input_idx_base = b * (I * T * H * W) + it * (I * H * W) + ih * (I * W) + iw * I;

                    for (int i = 0; i < I; ++i) {
                        float input_val = input[input_idx_base + i];
                        float weight_val = weight[weight_idx + i];
                        sum += input_val * weight_val;
                    }
                }
            }
        }
    }

    // Add bias (broadcasted per channel)
    float result = sum + bias[o];

    // First residual: result = result + original (where original = result after bias)
    float original = result;
    result += original;

    // Multiply by original
    result = result * original;

    // Second residual add
    result += original;

    output[idx] = result;
}

torch::Tensor fused_convtranspose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding,
    torch::IntArrayRef dilation,
    int64_t groups) {
    
    // Ensure input is contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    const int B = input.size(0);
    const int I = input.size(1);
    const int T = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);

    const int O = weight.size(0);  // (O, I, K, K, K)
    const int K = weight.size(2);

    const int S = stride[0];
    const int P = padding[0];
    const int OP = output_padding[0];

    // Compute output spatial dimensions
    int Ti = (T - 1) * S - 2 * P + K + OP;
    int Hi = (H - 1) * S - 2 * P + K + OP;
    int Wi = (W - 1) * S - 2 * P + K + OP;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto output = torch::empty({B, O, Ti, Hi, Wi}, options);

    const int total_elements = B * O * Ti * Hi * Wi;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    fused_convtranspose3d_bias_residual_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, T, H, W, I, O, K, S, P, OP, Ti, Hi, Wi
    );

    // Synchronize to catch any kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_convtranspose_cpp_source = """
torch::Tensor fused_convtranspose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding,
    torch::IntArrayRef dilation,
    int64_t groups);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_convtranspose3d",
    cpp_sources=fused_convtranspose_cpp_source,
    cuda_sources=fused_convtranspose_bias_residual_source,
    functions=["fused_convtranspose3d_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr"],
)

class ModelNew(nn.Module):
    """
    Optimized model using fused CUDA kernel for transposed convolution, bias add, 
    and multiple residual operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = 1

        # Initialize transposed convolution weights: (out_channels, in_channels, k, k, k)
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape).cuda())

        # Bias parameter: (out_channels, 1, 1, 1) -> (out_channels,)
        self.bias = nn.Parameter(torch.randn(bias_shape).cuda().view(-1))

        # Store configuration as lists for passing to CUDA kernel
        self.stride_list = [stride] * 3
        self.padding_list = [padding] * 3
        self.output_padding_list = [output_padding] * 3
        self.dilation_list = [1] * 3

        # Register fused CUDA operator
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_convtranspose3d_cuda(
            x,
            self.weight,
            self.bias,
            self.stride_list,
            self.padding_list,
            self.output_padding_list,
            self.dilation_list,
            self.groups
        )