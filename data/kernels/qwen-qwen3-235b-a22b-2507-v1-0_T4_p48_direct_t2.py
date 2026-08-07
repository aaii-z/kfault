import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv3D + Scale + Tanh + Scale + Sigmoid
fused_conv_tanh_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function to calculate 3D index
__device__ int get_index(int d, int h, int w, int D, int H, int W) {
    return d * H * W + h * W + w;
}

__global__ void fused_conv_tanh_sigmoid_kernel(
    const float* input,
    const float* weight,
    const float* bias_vec,
    const float* scaling_factor,
    float* output,
    int B, int C_out, int C_in, int D, int H, int W,
    int kD, int kH, int kW,
    int pad_d, int pad_h, int pad_w,
    int D_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C_out * D_out * H_out * W_out;
    
    if (idx >= total_elements) return;

    int w_idx = idx % W_out;
    int h_idx = (idx / W_out) % H_out;
    int d_idx = (idx / (W_out * H_out)) % D_out;
    int c_out_idx = (idx / (W_out * H_out * D_out)) % C_out;
    int b_idx = idx / (W_out * H_out * D_out * C_out);

    // Convolution computation
    float conv_val = 0.0f;
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kd = 0; kd < kD; kd++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int in_d = d_idx + kd - pad_d;
                    int in_h = h_idx + kh - pad_h;
                    int in_w = w_idx + kw - pad_w;

                    if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                        int in_idx = b_idx * C_in * D * H * W +
                                     c_in * D * H * W +
                                     in_d * H * W +
                                     in_h * W +
                                     in_w;
                        int weight_idx = c_out_idx * C_in * kD * kH * kW +
                                         c_in * kD * kH * kW +
                                         kd * kH * kW +
                                         kh * kW +
                                         kw;
                        conv_val += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Apply scaling factor (per channel): conv_val *= scaling_factor[c_out_idx]
    conv_val *= scaling_factor[c_out_idx];

    // Apply tanh
    float tanh_val = tanhf(conv_val);

    // Multiply by bias (per channel)
    float biased_val = tanh_val * bias_vec[c_out_idx];

    // Apply sigmoid with clamping to avoid overflow
    float clamped_val = fmaxf(-50.0f, fminf(50.0f, biased_val));
    float sigmoid_val = 1.0f / (1.0f + expf(-clamped_val));

    // Write output
    output[idx] = sigmoid_val;
}

torch::Tensor fused_conv_tanh_sigmoid_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_vec,
    torch::Tensor scaling_factor,
    int D_out, int H_out, int W_out
) {
    auto B = input.size(0);
    auto C_in = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    auto C_out = weight.size(0);
    auto kD = weight.size(2);
    auto kH = weight.size(3);
    auto kW = weight.size(4);

    int pad_d = kD / 2;
    int pad_h = kH / 2;
    int pad_w = kW / 2;

    auto output = torch::empty({B, C_out, D_out, H_out, W_out}, input.options());

    int total_elements = B * C_out * D_out * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    fused_conv_tanh_sigmoid_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_vec.data_ptr<float>(),
        scaling_factor.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_out, C_in, D, H, W,
        kD, kH, kW,
        pad_d, pad_h, pad_w,
        D_out, H_out, W_out
    );

    return output;
}
"""

fused_conv_tanh_sigmoid_cpp_source = """
torch::Tensor fused_conv_tanh_sigmoid_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_vec,
    torch::Tensor scaling_factor,
    int D_out, int H_out, int W_out
);
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_conv_tanh_sigmoid",
    cpp_sources=fused_conv_tanh_sigmoid_cpp_source,
    cuda_sources=fused_conv_tanh_sigmoid_source,
    functions=["fused_conv_tanh_sigmoid_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv3D + Scale + Tanh + Scale + Sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        # Conv3D parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # assuming square kernel and symmetric padding

        # Initialize weights and biases for convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.scaling_factor = nn.Parameter(torch.ones(bias_shape))  # Match original scaling factor initialization
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store bias shape for output size calculation
        self.bias_shape = bias_shape

        # Register the fused CUDA function
        self.fused_op = fused_ops.fused_conv_tanh_sigmoid_cuda

    def forward(self, x):
        B, C, D, H, W = x.shape

        # Output spatial dimensions
        D_out = D + 2 * self.padding - self.kernel_size + 1
        H_out = H + 2 * self.padding - self.kernel_size + 1
        W_out = W + 2 * self.padding - self.kernel_size + 1

        # Launch fused kernel
        output = self.fused_op(
            x,
            self.weight,
            self.bias.view(-1),  # flatten bias
            self.scaling_factor.view(-1),  # flatten scaling factor
            D_out, H_out, W_out
        )

        return output