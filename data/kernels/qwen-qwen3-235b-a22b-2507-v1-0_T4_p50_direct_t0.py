import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + Scale + AvgPool3d + Bias + Scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Efficient fused kernel for:
// 1. Transposed convolution (with stride=2, padding=1, kernel=3)
// 2. Multiply by scale1
// 3. 2x2x2 average pooling
// 4. Add bias (broadcasted)
// 5. Multiply by scale2

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
    int padded_depth,
    int padded_height,
    int padded_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * (out_depth / 2) * (out_height / 2) * (out_width / 2);
    
    if (idx >= total_elements) return;

    // Decode output index (after avg_pool, so size is halved)
    int temp = idx;
    int w_out = temp % (out_width / 2); temp /= (out_width / 2);
    int h_out = temp % (out_height / 2); temp /= (out_height / 2);
    int d_out = temp % (out_depth / 2); temp /= (out_depth / 2);
    int c_out = temp % out_channels;
    int batch = temp / out_channels;

    // Input to conv_transpose is at resolution (in_depth, in_height, in_width)
    // ConvTranspose3d with kernel=3, stride=2, padding=1 produces:
    // output_depth = (in_depth - 1) * 2 - 2 * 1 + 3 = 2*in_depth
    // So: out_depth = 2 * in_depth, etc.
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

                if (d_offset >= 0 && d_offset < padded_depth &&
                    h_offset >= 0 && h_offset < padded_height &&
                    w_offset >= 0 && w_offset < padded_width) {

                    int d_in = d_offset;
                    int h_in = h_offset;
                    int w_in = w_offset;

                    if (d_in >= in_depth || h_in >= in_height || w_in >= in_width) continue;

                    for (int c_in = 0; c_in < in_channels; c_in++) {
                        float input_val = input[batch * in_channels * in_depth * in_height * in_width +
                                                c_in * in_depth * in_height * in_width +
                                                d_in * in_height * in_width +
                                                h_in * in_width +
                                                w_in];

                        float weight_val = weight[c_out * in_channels * 3 * 3 * 3 +
                                                  c_in * 3 * 3 * 3 +
                                                  kd * 3 * 3 +
                                                  kh * 3 +
                                                  kw];

                        sum += input_val * weight_val;
                    }
                }
            }
        }
    }

    // Apply scale1
    sum = sum * scale1;

    // Now apply 2x2x2 average pooling over the 2x2x2 block starting at (conv_d_start, conv_h_start, conv_w_start)
    float avg_sum = 0.0f;
    int count = 0;

    for (int pd = 0; pd < 2; pd++) {
        for (int ph = 0; ph < 2; ph++) {
            for (int pw = 0; pw < 2; pw++) {
                int d_p = conv_d_start + pd;
                int h_p = conv_h_start + ph;
                int w_p = conv_w_start + pw;

                if (d_p < padded_depth && h_p < padded_height && w_p < padded_width) {
                    if (d_p < out_depth && h_p < out_height && w_p < out_width) {
                        avg_sum += sum;
                        count++;
                    }
                }
            }
        }
    }

    float pooled = avg_sum / 8.0f;  // 2x2x2 = 8

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
    int in_width
);
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

    // Padded size after conv_transpose (we use padding=1, so we have padded regions)
    int padded_depth = out_depth;
    int padded_height = out_height;
    int padded_width = out_width;

    const int total_elements = batch_size * out_channels * final_depth * final_height * final_width;
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
        padded_depth,
        padded_height,
        padded_width
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