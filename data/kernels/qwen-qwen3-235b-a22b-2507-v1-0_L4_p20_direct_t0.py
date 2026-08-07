import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 3D transposed convolution + bias add + residual operations
fused_convtranspose_bias_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Efficiently compute transposed convolution with grouped bias and multiple residual operations
__global__ void fused_convtranspose3d_bias_residual_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int T, int H, int W, int I, int O, int K, int S, int P, int OP,
    int Ti, int Hi, int Wi) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * Ti * Hi * Wi * O;
    
    if (idx >= total_elements) return;

    int wi = idx % Wi;
    int hi = (idx / Wi) % Hi;
    int ti = (idx / (Wi * Hi)) % Ti;
    int b = idx / (Ti * Hi * Wi);
    int o = (idx / (Ti * Hi * Wi)) % O;

    // Compute input indices based on stride, padding, output_padding
    int t_start = ti * S - P;
    int h_start = hi * S - P;
    int w_start = wi * S - P;

    float sum = 0.0f;

    for (int kt = 0; kt < K; ++kt) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ti_loc = t_start + kt;
                int hi_loc = h_start + kh;
                int wi_loc = w_start + kw;

                if (ti_loc >= 0 && ti_loc < T && 
                    hi_loc >= 0 && hi_loc < H && 
                    wi_loc >= 0 && wi_loc < W) {

                    int weight_idx = o * I * K * K * K + kt * K * K * I + kh * K * I + kw * I;
                    int input_idx = b * I * T * H * W + ti_loc * H * W * I + hi_loc * W * I + wi_loc * I;

                    for (int i = 0; i < I; ++i) {
                        sum += input[input_idx + i] * weight[weight_idx + i];
                    }
                }
            }
        }
    }

    // Add bias (broadcasted per channel)
    float result = sum + bias[o];

    // Store intermediate result (equivalent to first residual add)
    float original = result;
    result += original;  // First residual: result = result + original

    // Multiply by original (nonlinear step)
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
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    
    const int B = input.size(0);
    const int I = input.size(1);
    const int T = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);

    const int O = weight.size(1);
    const int K = weight.size(2);

    const int S = stride[0];
    const int P = padding[0];
    const int OP = output_padding[0];

    // Compute output dimensions
    int Ti = (T - 1) * S - 2 * P + K + OP;
    int Hi = (H - 1) * S - 2 * P + K + OP;
    int Wi = (W - 1) * S - 2 * P + K + OP;

    auto output = torch::empty({B, O, Ti, Hi, Wi}, options);

    const int total_threads = B * Ti * Hi * Wi * O;
    const int block_size = 256;
    const int grid_size = (total_threads + block_size - 1) / block_size;

    dim3 block(block_size);
    dim3 grid(grid_size);

    fused_convtranspose3d_bias_residual_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, T, H, W, I, O, K, S, P, OP, Ti, Hi, Wi
    );

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

        # Initialize transposed convolution weights
        weight_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape).cuda().permute(1, 0, 2, 3, 4))  # (O, I, K, K, K)

        # Bias parameter
        self.bias = nn.Parameter(torch.randn(bias_shape).cuda().view(-1))  # (O,)

        # Store configuration
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