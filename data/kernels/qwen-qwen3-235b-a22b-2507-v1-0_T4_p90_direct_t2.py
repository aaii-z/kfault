import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused 3D convolution, LeakyReLU, add, clamp, and GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Efficient GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_leakyrelu_add_clamp_gelu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* sum_tensor,
    float* output,
    int B, int T, int H, int W,
    int C_out, int C_in,
    int kT, int kH, int kW,
    int pad_t, int pad_h, int pad_w,
    int stride_t, int stride_h, int stride_w,
    int out_T, int out_H, int out_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_T * out_H * out_W * C_out;
    
    if (idx >= total_elements) return;

    int w = idx % out_W;
    int h = (idx / out_W) % out_H;
    int t = (idx / (out_W * out_H)) % out_T;
    int c = (idx / (out_T * out_H * out_W)) % C_out;
    int b = idx / (out_T * out_H * out_W * C_out);

    // Conv3d computation
    float val = 0.0f;
    for (int ci = 0; ci < C_in; ci++) {
        for (int kt = 0; kt < kT; kt++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int i_t = t * stride_t + kt - pad_t;
                    int i_h = h * stride_h + kh - pad_h;
                    int i_w = w * stride_w + kw - pad_w;
                    
                    if (i_t >= 0 && i_t < T && i_h >= 0 && i_h < H && i_w >= 0 && i_w < W) {
                        int input_idx = b * C_in * T * H * W + ci * T * H * W + i_t * H * W + i_h * W + i_w;
                        int weight_idx = c * C_in * kT * kH * kW + ci * kT * kH * kW + kt * kH * kW + kh * kW;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    val += bias[c];

    // LeakyReLU: max(0.2 * x, x)
    val = fmaxf(0.2f * val, val);

    // Add sum_tensor (broadcasted: C_out,1,1,1)
    val += sum_tensor[c];

    // Clamp to [-1.0, 1.0]
    val = fmaxf(-1.0f, fminf(1.0f, val));

    // GELU activation
    val = gelu(val);

    output[idx] = val;
}

torch::Tensor fused_conv_leakyrelu_add_clamp_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor sum_tensor
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(sum_tensor);

    const auto B = input.size(0);
    const auto C_in = input.size(1);
    const auto T = input.size(2);
    const auto H = input.size(3);
    const auto W = input.size(4);

    const auto C_out = weight.size(0);
    const auto kT = weight.size(2);
    const auto kH = weight.size(3);
    const auto kW = weight.size(4);

    const int pad_t = (kT - 1) / 2;
    const int pad_h = (kH - 1) / 2;
    const int pad_w = (kW - 1) / 2;
    const int stride_t = 1;
    const int stride_h = 1;
    const int stride_w = 1;

    const int out_T = (T + 2 * pad_t - kT) / stride_t + 1;
    const int out_H = (H + 2 * pad_h - kH) / stride_h + 1;
    const int out_W = (W + 2 * pad_w - kW) / stride_w + 1;

    auto output = torch::empty({B, C_out, out_T, out_H, out_W}, input.options());

    const int total_elements = B * C_out * out_T * out_H * out_W;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    fused_conv_leakyrelu_add_clamp_gelu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        B, T, H, W, C_out, C_in,
        kT, kH, kW,
        pad_t, pad_h, pad_w,
        stride_t, stride_h, stride_w,
        out_T, out_H, out_W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_leakyrelu_add_clamp_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor sum_tensor
);
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_conv_leakyrelu_add_clamp_gelu",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_leakyrelu_add_clamp_gelu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv3d + LeakyReLU + Add + Clamp + GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        
        # Create convolution parameters manually
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels, kernel_size, kernel_size, kernel_size
        ))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=0.2, nonlinearity='leaky_relu')
        nn.init.zeros_(self.bias)

        # Bind the fused CUDA function
        self.fused_op = fused_ops.fused_conv_leakyrelu_add_clamp_gelu_cuda

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias, self.sum_tensor)