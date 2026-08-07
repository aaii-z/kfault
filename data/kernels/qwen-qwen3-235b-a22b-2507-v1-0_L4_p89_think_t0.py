import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    """
    Optimized model that fuses softmax, subtract, swish, and max operations into a single CUDA kernel.
    """
    _fused_op = None

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

        if ModelNew._fused_op is None:
            cuda_source = r'''
            #include <cuda_runtime.h>
            #include <cmath>
            #include <torch/extension.h>

            __global__ void fused_softmax_subtract_swish_max_kernel(
                const float* x, 
                const float* s, 
                float* out, 
                int batch, int C, int D, int H, int W)
            {
                int total_spatial = batch * D * H * W;
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= total_spatial) return;

                int w = idx % W;
                int hw = idx / W;
                int h = hw % H;
                int dhw = hw / H;
                int d = dhw % D;
                int bdhw = dhw / D;
                int b = bdhw;

                float max_x = -1e20f;
                for (int c = 0; c < C; c++) {
                    float val = x[b * C * D * H * W + c * D * H * W + d * H * W + h * W + w];
                    if (val > max_x) max_x = val;
                }

                float sum_exp = 0.0f;
                float exp_vals[32];

                for (int c = 0; c < C; c++) {
                    float val = x[b * C * D * H * W + c * D * H * W + d * H * W + h * W + w];
                    float exp_val = expf(val - max_x);
                    exp_vals[c] = exp_val;
                    sum_exp += exp_val;
                }

                if (sum_exp == 0.0f) {
                    sum_exp = 1.0f;
                }

                float max_swish = -1e20f;
                for (int c = 0; c < C; c++) {
                    float y = exp_vals[c] / sum_exp;
                    float z = y - s[c];
                    float sigmoid_z = 1.0f / (1.0f + expf(-z));
                    float swish = z * sigmoid_z;
                    if (swish > max_swish) max_swish = swish;
                }

                out[b * D * H * W + d * H * W + h * W + w] = max_swish;
            }

            torch::Tensor fused_softmax_subtract_swish_max_cuda(
                torch::Tensor x, 
                torch::Tensor s)
            {
                TORCH_CHECK(x.dim() == 5, "Input must be 5D tensor (batch, channels, depth, height, width)");
                TORCH_CHECK(s.dim() == 1, "Subtract parameter must be 1D");
                
                int batch = x.size(0);
                int C = x.size(1);
                int D = x.size(2);
                int H = x.size(3);
                int W = x.size(4);
                int total_spatial = batch * D * H * W;

                TORCH_CHECK(s.size(0) == C, "Subtract parameter length must match number of channels");

                auto out = torch::empty({batch, D, H, W}, x.options());
                x = x.contiguous();

                const int block_size = 256;
                const int num_blocks = (total_spatial + block_size - 1) / block_size;

                fused_softmax_subtract_swish_max_kernel<<<num_blocks, block_size>>>(
                    x.data_ptr<float>(), 
                    s.data_ptr<float>(), 
                    out.data_ptr<float>(), 
                    batch, C, D, H, W);

                return out;
            }
            '''

            cpp_source = "torch::Tensor fused_softmax_subtract_swish_max_cuda(torch::Tensor x, torch::Tensor s);"
            
            module = load_inline(
                name="fused_softmax_subtract_swish_max",
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                functions=["fused_softmax_subtract_swish_max_cuda"],
                verbose=False,
                extra_cuda_cflags=["-O3"]
            )
            ModelNew._fused_op = module

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = ModelNew._fused_op.fused_softmax_subtract_swish_max_cuda(x, self.subtract)
        return x