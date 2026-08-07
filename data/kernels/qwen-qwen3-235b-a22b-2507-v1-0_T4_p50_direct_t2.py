import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + Scale + AvgPool3d + Bias + Scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_avgpool_kernel(
    const float* input,
    const float* weight,
    const float* bias_vec,
    float* output,
    float scale1,
    float scale2,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int final_depth,
    int final_height,
    int final_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * final_depth * final_height * final_width;
    
    if (idx >= total_elements) return;

    // Decode output index (after avg_pool, so size is halved)
    int temp = idx;
    int w_out = temp % final_width; temp /= final_width;
    int h_out = temp % final_height; temp /= final_height;
    int d_out = temp % final_depth; temp /= final_depth;
    int c_out = temp % out_channels;
    int batch = temp / out_channels;

    // ConvTranspose3d with kernel=3, stride=2, padding=1
    // Output size: (in_depth - 1)*stride - 2*padding + kernel = 2*in_depth
    int conv_d_start = d_out * 2;
    int conv_h_start = h_out * 2;
    int conv_w_start = w_out * 2;

    float sum = 0.0f;

    // 3x3x3 kernel
    for (int kd = 0; kd < 3; kd++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int d_offset = conv_d_start + kd - 1; // because padding = 1
                int h_offset = conv_h_start + kh - 1;
                int w_offset = conv_w_start + kw - 1;

                if (d_offset >= 0 && d_offset < out_depth &&
                    h_offset >= 0 && h_offset < out_height &&
                    w_offset >= 0 && w_offset < out_width) {

                    int d_in = d_offset / 2;
                    int h_in = h_offset / 2;
                    int w_in = w_offset / 2;

                    // Check if this input location contributes (stride=2)
                    if (d_offset % 2 == 0 && h_offset % 2 == 0 && w_offset % 2 == 0 &&
                        d_in < in_depth && h_in < in_height && w_in < in_width) {

                        for (int c_in = 0; c_in < in_channels; c_in++) {
                            int input_idx = batch * in_channels * in_depth * in_height * in_width +
                                            c_in * in_depth * in_height * in_width +
                                            d_in * in_height * in_width +
                                            h_in * in_width +
                                            w_in;

                            int weight_idx = c_out * in_channels * 3 * 3 * 3 +
                                             c_in * 3 * 3 * 3 +
                                             kd * 3 * 3 +
                                             kh * 3 +
                                             kw;

                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }

    // Apply scale1
    sum = sum * scale1;

    // 2x2x2 average pooling: we are already at the pooled location
    // Each output point comes from a 2x2x2 block in the conv_transpose output
    // Since we are computing one output point per 2x2x2 block, we just divide by 8
    float pooled = sum / 8.0f;

    // Add bias (broadcasted over spatial dimensions)
    pooled += bias_vec[c_out];

    // Apply scale2
    pooled *= scale2;

    output[idx] = pooled;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_transpose_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_vec,
    float scale1,
    float scale2,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width);
"""

fused_cpp_impl = """
torch::Tensor fused_conv_transpose_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_vec,
    float scale1,
    float scale2,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width) {

    // Compute output dimensions after conv_transpose
    int out_depth = 2 * in_depth;
    int out_height = 2 * in_height;
    int out_width = 2 * in_width;

    // After avg_pool3d(kernel=2), output is halved
    int final_depth = out_depth / 2;
    int final_height = out_height / 2;
    int final_width = out_width / 2;

    auto output = torch::zeros({batch_size, out_channels, final_depth, final_height, final_width},
                               input.options());

    const int total_elements = output.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_transpose_avgpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_vec.data_ptr<float>(),
        output.data_ptr<float>(),
        scale1,
        scale2,
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        final_depth,
        final_height,
        final_width
    );

    return output;
}
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_transpose",
    cpp_sources=[fused_cpp_source, fused_cpp_impl],
    cuda_sources=[fused_kernel_source],
    functions=["fused_conv_transpose_op"],
    verbose=False,
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for ConvTranspose3d + Scale + AvgPool3d + Bias + Scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        assert kernel_size == 3 and stride == 2 and padding == 1, "Fused kernel only supports kernel=3, stride=2, padding=1"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.fused_op = fused_op

    def forward(self, x):
        batch_size, in_channels, depth, height, width = x.shape
        return self.fused_op.fused_conv_transpose_op(
            x,
            self.weight,
            self.bias.view(-1),
            self.scale1.item(),
            self.scale2.item(),
            batch_size,
            in_channels,
            self.out_channels,
            depth,
            height,
            width
        )