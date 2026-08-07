import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution and double Mish
fused_conv_double_mish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ float mish_activation(float x) {
    return x * tanh(log1pf(expf(x)));
}

__global__ void double_mish_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float first_mish = mish_activation(x);
        output[idx] = mish_activation(first_mish);
    }
}

torch::Tensor fused_conv_double_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    auto output = torch::empty_like(input);
    
    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);
    
    // Set tensor descriptors
    auto dims = input.sizes();
    int n = dims[0], c = dims[1], h = dims[2], w = dims[3];
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    
    // Set filter descriptor
    auto weight_dims = weight.sizes();
    int out_c = weight_dims[0], in_c = weight_dims[1], k_h = weight_dims[2], k_w = weight_dims[3];
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_c, in_c, k_h, k_w);
    
    // Set convolution descriptor
    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, dilation, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    // Find convolution algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(cudnn, input_desc, filter_desc, conv_desc, output_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    
    // Perform convolution
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size);
    auto workspace = torch::empty({(int64_t)workspace_size}, torch::kByte);
    
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, input_desc, input.data_ptr<float>(),
                          filter_desc, weight.data_ptr<float>(), conv_desc, algo,
                          workspace.data_ptr<uint8_t>(), workspace_size, &beta,
                          output_desc, output.data_ptr<float>());
    
    // Add bias
    cudnnAddTensor(cudnn, &alpha, bias.data_ptr<float>(), &alpha, output_desc, output.data_ptr<float>());
    
    // Apply double Mish activation
    int num_elements = output.numel();
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    double_mish_kernel<<<num_blocks, block_size>>>(output.data_ptr<float>(), output.data_ptr<float>(), num_elements);
    
    // Cleanup
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);
    
    return output;
}
"""

fused_conv_double_mish_cpp_source = """
torch::Tensor fused_conv_double_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups);
"""

# Compile the inline CUDA code
fused_conv_double_mish = load_inline(
    name="fused_conv_double_mish",
    cpp_sources=fused_conv_double_mish_cpp_source,
    cuda_sources=fused_conv_double_mish_source,
    functions=["fused_conv_double_mish_cuda"],
    verbose=True,
    extra_cflags=["-lcudnn"],
    extra_ldflags=["-lcudnn"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2
        self.dilation = 1
        self.groups = 1
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
    def forward(self, x):
        return fused_conv_double_mish.fused_conv_double_mish_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )