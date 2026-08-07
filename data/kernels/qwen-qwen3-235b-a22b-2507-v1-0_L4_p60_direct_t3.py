import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + Swish + GroupNorm + HardSwish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utility function to get linear index in 5D tensor: (n, c, d, h, w)
__device__ int get_index(int n, int c, int d, int h, int w, 
                         int size_c, int size_d, int size_h, int size_w) {
    return ((n * size_c + c) * size_d + d) * size_h * size_w + h * size_w + w;
}

// Fused kernel: each thread block computes one group normalization group per batch
// Each thread handles one output element
__global__ void fused_convtranspose_swish_gn_hardswish(
    const float* input,
    const float* weight,
    const float* bias,
    const float* gn_weight,
    const float* gn_bias,
    float* output,
    // Dimensions
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int out_depth, int out_height, int out_width,
    // Conv params
    int kernel_size, int stride, int padding,
    // Group norm params
    int groups, float eps
) {
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decode index
    int temp = idx;
    int w = temp % out_width; temp /= out_width;
    int h = temp % out_height; temp /= out_height;
    int d = temp % out_depth; temp /= out_depth;
    int c = temp % out_channels; temp /= out_channels;
    int n = temp;

    // Transposed convolution: sum over input channels and kernel
    float conv_val = 0.0f;
    int k_radius = kernel_size / 2;
    for (int kd = 0; kd < kernel_size; ++kd) {
        int id = d * stride - padding + kd;
        if (id < 0 || id >= in_depth) continue;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = h * stride - padding + kh;
            if (ih < 0 || ih >= in_height) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw = w * stride - padding + kw;
                if (iw < 0 || iw >= in_width) continue;
                for (int ic = 0; ic < in_channels; ++ic) {
                    int in_idx = get_index(n, ic, id, ih, iw, in_channels, in_depth, in_height, in_width);
                    int w_idx = ((ic * out_channels + c) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                    conv_val += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    // Add bias
    conv_val += bias[c];

    // Swish: x * sigmoid(x), clamped to avoid overflow
    float sigmoid_val = 1.0f / (1.0f + expf(-fmaxf(-15.0f, fminf(15.0f, conv_val))));
    float swish_val = conv_val * sigmoid_val;

    // Group Normalization: compute mean and variance over group
    int group_size = out_channels / groups;
    int group_id = c / group_size;
    int c_start = group_id * group_size;

    // First, we need to compute mean and variance over the group for this batch element
    float mean = 0.0f, var = 0.0f;
    int group_volume = group_size * out_depth * out_height * out_width;

    // Use shared memory for reduction
    extern __shared__ float shared_mem[];
    float* shared = shared_mem;
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    // Each thread computes one element in the group
    int local_idx = threadIdx.x;
    int total_group_elements = group_volume;
    int num_threads = blockDim.x;

    // Loop over all elements in the group to compute sum and sum of squares
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = c_local * out_depth * out_height * out_width + get_index(0,0,d,h,w, group_size, out_depth, out_height, out_width);
         i < group_volume;
         i += num_threads) {
        int c_g = i / (out_depth * out_height * out_width);
        int rest = i % (out_depth * out_height * out_width);
        int d_g = rest / (out_height * out_width);
        int h_g = (rest / out_width) % out_height;
        int w_g = rest % out_width;

        int global_c = c_start + c_g;
        int in_idx_g = get_index(n, global_c, d_g, h_g, w_g, out_channels, out_depth, out_height, out_width);
        float val = 0.0f;

        // Recompute swish value for this element
        float conv_val_g = 0.0f;
        for (int kd = 0; kd < kernel_size; ++kd) {
            int id_g = d_g * stride - padding + kd;
            if (id_g < 0 || id_g >= in_depth) continue;
            for (int kh = 0; kh < kernel_size; ++kh) {
                int ih_g = h_g * stride - padding + kh;
                if (ih_g < 0 || ih_g >= in_height) continue;
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int iw_g = w_g * stride - padding + kw;
                    if (iw_g < 0 || iw_g >= in_width) continue;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        int in_idx_val = get_index(n, ic, id_g, ih_g, iw_g, in_channels, in_depth, in_height, in_width);
                        int w_idx = ((ic * out_channels + global_c) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                        conv_val_g += input[in_idx_val] * weight[w_idx];
                    }
                }
            }
        }
        conv_val_g += bias[global_c];
        float sigmoid_val_g = 1.0f / (1.0f + expf(-fmaxf(-15.0f, fminf(15.0f, conv_val_g))));
        val = conv_val_g * sigmoid_val_g;

        sum += val;
        sum_sq += val * val;
    }

    mean_shared[local_idx] = sum;
    var_shared[local_idx] = sum_sq;
    __syncthreads();

    // Reduce within block
    for (int stride_val = num_threads / 2; stride_val > 0; stride_val >>= 1) {
        if (local_idx < stride_val) {
            mean_shared[local_idx] += mean_shared[local_idx + stride_val];
            var_shared[local_idx] += var_shared[local_idx + stride_val];
        }
        __syncthreads();
    }

    if (local_idx == 0) {
        float total_sum = mean_shared[0];
        float total_sum_sq = var_shared[0];
        mean = total_sum / total_group_elements;
        var = total_sum_sq / total_group_elements - mean * mean;
    }

    // Broadcast mean and variance to all threads in block
    __shared__ float mean_shared_val, var_shared_val;
    if (local_idx == 0) {
        var_shared_val = fmaxf(var + eps, 1e-12f);
        mean_shared_val = mean;
    }
    __syncthreads();

    // Normalize
    float x_hat = (swish_val - mean_shared_val) * rsqrtf(var_shared_val);
    float gn_val = x_hat * gn_weight[c] + gn_bias[c];

    // HardSwish: x * relu6(x + 3) / 6
    float hs = fmaxf(0.0f, fminf(6.0f, gn_val + 3.0f));
    output[idx] = gn_val * hs / 6.0f;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int in_channels, int out_channels, int kernel_size, int stride, int padding,
    int groups, float eps
);
"""

fused_kernel_source += """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int in_channels, int out_channels, int kernel_size, int stride, int padding,
    int groups, float eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(gn_weight);
    CHECK_INPUT(gn_bias);

    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_depth = input_sizes[2];
    int in_height = input_sizes[3];
    int in_width = input_sizes[4];

    // Output spatial dimensions for transposed conv
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;

    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, 
                               input.options().dtype(torch::kFloat)).contiguous();

    const int threads_per_block = 256;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = threads_per_block * 2 * sizeof(float);  // for sum and sum_sq

    fused_convtranspose_swish_gn_hardswish<<<num_blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, out_depth, out_height, out_width,
        kernel_size, stride, padding,
        groups, eps
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_convtranspose_swish_gn_hardswish",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for:
    ConvTranspose3d + Swish + GroupNorm + HardSwish
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.eps = eps

        # Transposed convolution parameters: note weight shape is (in_channels, out_channels, k, k, k)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # GroupNorm parameters
        self.group_norm_weight = nn.Parameter(torch.ones(out_channels))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_channels))

        # Fused CUDA op
        self.fused_op = fused_ops.fused_op_cuda

    def forward(self, x):
        bias = self.bias if self.bias is not None else torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        return self.fused_op(
            x,
            self.weight,
            bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.groups,
            self.eps
        )