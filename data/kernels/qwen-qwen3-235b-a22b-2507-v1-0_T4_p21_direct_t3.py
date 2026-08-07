import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d + Bias + Scale + Sigmoid + GroupNorm
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(0)

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void fused_conv_bias_scale_sigmoid_gn_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* scale,
    const float* gamma,
    const float* beta,
    float* output,
    int B, int C_in, int C_out, int H, int W, int H_out, int W_out, int K,
    int groups, int C_per_group, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C_out * H_out * W_out;
    
    if (idx >= total_elements) return;

    int w_idx = idx % W_out;
    int h_idx = (idx / W_out) % H_out;
    int c_idx = (idx / (W_out * H_out)) % C_out;
    int b_idx = idx / (W_out * H_out * C_out);

    int group_id = c_idx / C_per_group;

    // Convolution computation
    float conv_val = 0.0f;
    int p = K / 2;  // padding assumed to be K//2 for square kernel
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            int h_im = h_idx + i - p;
            int w_im = w_idx + j - p;
            if (h_im >= 0 && h_im < H && w_im >= 0 && w_im < W) {
                for (int c_in = 0; c_in < C_in; c_in++) {
                    int input_offset = ((b_idx * C_in + c_in) * H + h_im) * W + w_im;
                    int weight_offset = (((c_idx * C_in) + c_in) * K + i) * K + j;
                    conv_val += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    // Add bias and scale
    conv_val += bias[c_idx];
    conv_val *= scale[c_idx];
    float x = 1.0f / (1.0f + expf(-conv_val)); // sigmoid

    // Compute mean and variance for the group
    float mean = 0.0f, var = 0.0f;
    int group_start = group_id * C_per_group;
    int group_end = group_start + C_per_group;
    int spatial_size = H_out * W_out;

    // Use block-level reduction for mean
    float sum = 0.0f;
    for (int c = group_start; c < group_end; c++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                float conv_val_local = 0.0f;
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < K; j++) {
                        int h_im = h + i - p;
                        int w_im = w + j - p;
                        if (h_im >= 0 && h_im < H && w_im >= 0 && w_im < W) {
                            for (int c_in = 0; c_in < C_in; c_in++) {
                                int input_idx = ((b_idx * C_in + c_in) * H + h_im) * W + w_im;
                                int weight_idx = (((c * C_in) + c_in) * K + i) * K + j;
                                conv_val_local += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                conv_val_local += bias[c];
                conv_val_local *= scale[c];
                float val = 1.0f / (1.0f + expf(-conv_val_local));
                sum += val;
            }
        }
    }
    mean = sum / (C_per_group * spatial_size);

    // Variance reduction
    float var_sum = 0.0f;
    for (int c = group_start; c < group_end; c++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                float conv_val_local = 0.0f;
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < K; j++) {
                        int h_im = h + i - p;
                        int w_im = w + j - p;
                        if (h_im >= 0 && h_im < H && w_im >= 0 && w_im < W) {
                            for (int c_in = 0; c_in < C_in; c_in++) {
                                int input_idx = ((b_idx * C_in + c_in) * H + h_im) * W + w_im;
                                int weight_idx = (((c * C_in) + c_in) * K + i) * K + j;
                                conv_val_local += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                conv_val_local += bias[c];
                conv_val_local *= scale[c];
                float val = 1.0f / (1.0f + expf(-conv_val_local));
                float diff = val - mean;
                var_sum += diff * diff;
            }
        }
    }
    var = var_sum / (C_per_group * spatial_size);

    float inv_std = rsqrtf(var + eps);
    float gn_val = (x - mean) * inv_std * gamma[group_id] + beta[group_id];

    output[idx] = gn_val;
}

torch::Tensor fused_conv_bias_scale_sigmoid_gn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(scale);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);

    const auto B = input.size(0);
    const auto C_in = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const auto C_out = weight.size(0);
    const auto K = weight.size(2);
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int C_per_group = C_out / groups;

    auto output = torch::empty({B, C_out, H_out, W_out}, input.options());

    const int total_elements = B * C_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_bias_scale_sigmoid_gn_kernel<<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, C_out, H, W, H_out, W_out, K,
        groups, C_per_group, eps
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_bias_scale_sigmoid_gn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float eps
);
"""

# Compile the inline CUDA module
fused_ops = load_inline(
    name="fused_conv_bias_scale_sigmoid_gn",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_bias_scale_sigmoid_gn_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv2d + Bias + Scale + Sigmoid + GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.bias_shape = bias_shape
        self.scale_shape = scale_shape

        # Conv weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        # GroupNorm parameters (gamma and beta)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.eps = self.group_norm.eps

        # Fused CUDA kernel
        self.fused_kernel = fused_ops.fused_conv_bias_scale_sigmoid_gn_cuda

    def forward(self, x):
        if self.training:
            x = torch.nn.functional.conv2d(x, self.weight)
            x = x + self.bias
            x = x * self.scale
            x = torch.sigmoid(x)
            x = self.group_norm(x)
            return x
        else:
            return self.fused_kernel(
                x,
                self.weight,
                self.bias.view(-1),
                self.scale.view(-1),
                self.group_norm.weight,
                self.group_norm.bias,
                self.num_groups,
                self.eps
            )