import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv3D + GroupNorm + min + clamp + dropout
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Helper function to get linear index from 3D coordinates
__device__ int get_index(int d, int h, int w, int D, int H, int W) {
    return d * H * W + h * W + w;
}

// CUDA kernel for fused Conv3D + GroupNorm + min + clamp + dropout
__global__ void fused_conv_gn_min_clamp_dropout_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* gamma,
    const float* beta,
    float* output,
    int B, int C, int D, int H, int W,
    int out_channels, int kernel_size,
    int groups,
    float min_val,
    float max_val,
    float dropout_p,
    uint64_t seed) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * (D - kernel_size + 1) * (H - kernel_size + 1) * (W - kernel_size + 1);
    
    if (idx >= total_elements) return;

    // Initialize CURAND for dropout
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Decode index
    int temp = idx;
    int w = temp % (W - kernel_size + 1); temp /= (W - kernel_size + 1);
    int h = temp % (H - kernel_size + 1); temp /= (H - kernel_size + 1);
    int d = temp % (D - kernel_size + 1); temp /= (D - kernel_size + 1);
    int c = temp % out_channels;
    int b = temp / out_channels;

    int C_per_group = C / groups;
    int group_id = c / (out_channels / groups);
    int c_g = c % (out_channels / groups);

    // Convolution (3x3x3)
    float conv_val = 0.0f;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_d = d + kd;
                int in_h = h + kh;
                int in_w = w + kw;
                for (int ic = 0; ic < C_per_group; ++ic) {
                    int weight_idx = c * C_per_group * kernel_size * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size * kernel_size +
                                     kd * kernel_size * kernel_size + kh * kernel_size + kw;
                    int input_idx = b * C * D * H * W +
                                    (group_id * C_per_group + ic) * D * H * W +
                                    in_d * H * W + in_h * W + in_w;
                    conv_val += weight[weight_idx] * input[input_idx];
                }
            }
        }
    }
    conv_val += bias[c];

    // Group Normalization
    int group_size = (D - kernel_size + 1) * (H - kernel_size + 1) * (W - kernel_size + 1) * (out_channels / groups);
    int group_offset = b * groups * group_size + group_id * group_size;
    int inner_idx = c_g * (D - kernel_size + 1) * (H - kernel_size + 1) * (W - kernel_size + 1) +
                    d * (H - kernel_size + 1) * (W - kernel_size + 1) + h * (W - kernel_size + 1) + w;

    float mean = 0.0f, var = 0.0f;
    int spatial_size = (D - kernel_size + 1) * (H - kernel_size + 1) * (W - kernel_size + 1);
    int feat_per_group = out_channels / groups;
    for (int i = 0; i < feat_per_group; ++i) {
        for (int j = 0; j < spatial_size; ++j) {
            float x = input[b * out_channels * spatial_size + (group_id * feat_per_group + i) * spatial_size + j];
            mean += x;
            var += x * x;
        }
    }
    mean /= (feat_per_group * spatial_size);
    var = var / (feat_per_group * spatial_size) - mean * mean;
    float eps = 1e-5f;
    float norm_val = (conv_val - mean) / sqrtf(var + eps);
    norm_val = gamma[c] * norm_val + beta[c];

    // Apply min and clamp
    float clamped_val = fminf(norm_val, min_val);
    clamped_val = fmaxf(clamped_val, min_val);
    clamped_val = fminf(clamped_val, max_val);

    // Dropout
    float rand_val = curand_uniform(&state);
    float drop_val = (rand_val > dropout_p) ? clamped_val / (1.0f - dropout_p) : 0.0f;

    output[idx] = drop_val;
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float min_val,
    float max_val,
    float dropout_p,
    uint64_t seed) {

    auto B = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto kernel_size = weight.size(2);
    auto out_channels = weight.size(0);

    auto output_size = (D - kernel_size + 1);
    auto H_out = (H - kernel_size + 1);
    auto W_out = (W - kernel_size + 1);
    auto total_elements = B * out_channels * output_size * H_out * W_out;

    auto output = torch::zeros({B, out_channels, output_size, H_out, W_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_gn_min_clamp_dropout_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W,
        out_channels, kernel_size,
        groups,
        min_val,
        max_val,
        dropout_p,
        seed
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float min_val,
    float max_val,
    float dropout_p,
    uint64_t seed);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv3D + GroupNorm + min + clamp + dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        self.seed = 123456

        # Define parameters manually to access them in CUDA kernel
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.norm = nn.GroupNorm(groups, out_channels)
        # Initialize gamma and beta for GroupNorm
        self.gamma = nn.Parameter(self.norm.weight)
        self.beta = nn.Parameter(self.norm.bias)

    def forward(self, x):
        return fused_op.fused_op_cuda(
            x,
            self.weight,
            self.bias,
            self.gamma,
            self.beta,
            self.groups,
            self.min_value,
            self.max_value,
            self.dropout_p,
            self.seed
        )