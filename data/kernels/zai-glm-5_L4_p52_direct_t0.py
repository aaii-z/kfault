import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused activation and batch normalization
fused_act_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for fused Softplus + Tanh + Multiply + BatchNorm
__global__ void fused_act_bn_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ gamma, 
    const float* __restrict__ beta,
    const float* __restrict__ running_mean, 
    const float* __restrict__ running_var,
    float* __restrict__ out,
    float eps, int N, int C, int H, int W) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    
    if (index < total_elements) {
        // Calculate indices
        int w_idx = index % W;
        int h_idx = (index / W) % H;
        int c_idx = (index / (W * H)) % C;
        // int n_idx = index / (W * H * C); // Not needed for weight indexing

        // Load input
        float val = x[index];
        
        // Fused Activation: x * tanh(softplus(x))
        // softplus(x) = log(1 + exp(x))
        // Optimization: log(1 + exp(x)) approx log(exp(x)) for large x, but log1p(exp(x)) is safer.
        // However, for performance, we can use the standard approximation or direct calculation.
        // Let's use direct calculation for accuracy within FP32 range.
        float sp = log1pf(expf(val)); 
        float th = tanhf(sp);
        float act_val = val * th;

        // Batch Normalization
        // out = (act_val - mean) / sqrt(var + eps) * gamma + beta
        float mean = running_mean[c_idx];
        float var = running_var[c_idx];
        float scale = gamma[c_idx];
        float shift = beta[c_idx];

        float norm_val = (act_val - mean) * rsqrtf(var + eps);
        out[index] = norm_val * scale + shift;
    }
}

torch::Tensor fused_act_bn_cuda(
    torch::Tensor x, 
    torch::Tensor gamma, 
    torch::Tensor beta,
    torch::Tensor running_mean, 
    torch::Tensor running_var,
    float eps) {
    
    auto output = torch::empty_like(x);
    int total_elements = x.numel();
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_act_bn_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        eps, N, C, H, W
    );

    return output;
}
"""

fused_act_bn_cpp_source = (
    "torch::Tensor fused_act_bn_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor running_mean, torch::Tensor running_var, float eps);"
)

# Compile the inline CUDA code
fused_act_bn_op = load_inline(
    name="fused_act_bn",
    cpp_sources=fused_act_bn_cpp_source,
    cuda_sources=fused_act_bn_source,
    functions=["fused_act_bn_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for fused activation and batch normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.fused_act_bn = fused_act_bn_op

    def forward(self, x):
        # Standard Convolution
        x = self.conv(x)
        
        # Fused Path: Activation + BatchNorm
        # During training, BatchNorm layers update running statistics. 
        # To maintain correctness of the running stats while using the fused kernel for the forward pass,
        # we need to trigger the stats update logic. 
        # We perform a manual update if training is enabled.
        
        if self.training:
            # Calculate mean and variance of the activation output for stats update
            # We detach x to avoid affecting the gradient computation graph for the main path
            with torch.no_grad():
                # Activation function applied for stats calculation
                # Note: This adds some overhead, but is necessary for correct BatchNorm behavior
                act_x = torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x)
                
                # Update running mean/var using the BatchNorm's internal logic
                # We use the standard BN forward on the activated values just to update the buffers.
                # This is a common trick to keep buffers in sync.
                self.bn(act_x)
        
        # Use the fused kernel for the actual forward computation
        # We use the running_mean and running_var buffers directly.
        x = self.fused_act_bn.fused_act_bn_cuda(
            x, 
            self.bn.weight, 
            self.bn.bias, 
            self.bn.running_mean, 
            self.bn.running_var, 
            self.bn.eps
        )
        
        return x