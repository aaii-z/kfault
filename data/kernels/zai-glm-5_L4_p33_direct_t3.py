import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, scaling, and batch normalization
fused_gemm_scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

// Kernel for scaling and computing sum and sum of squares for Batch Normalization statistics
__global__ void scale_bn_stats_kernel(const float* __restrict__ x, const float* __restrict__ bias, 
                                     const float* __restrict__ scale, 
                                     float* __restrict__ sum_out, float* __restrict__ sum_sq_out, 
                                     int batch_size, int out_features) {
    // Use shared memory to reduce global memory access for the partial sums
    extern __shared__ float shared_data[]; 
    float* s_sum = shared_data;
    float* s_sum_sq = shared_data + blockDim.x;

    int feature_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Initialize shared memory
    s_sum[tid] = 0.0f;
    s_sum_sq[tid] = 0.0f;
    __syncthreads();

    // Pointers to the specific column (feature) for this block
    const float* x_col = x + feature_idx;
    float scale_val = scale[feature_idx];
    float bias_val = bias[feature_idx];

    // Grid-stride loop to process all rows for this feature
    for (int i = tid; i < batch_size; i += blockDim.x) {
        float val = x_col[i * out_features] + bias_val; // Add bias
        val = val * scale_val; // Scale
        // Accumulate in shared memory
        s_sum[tid] += val;
        s_sum_sq[tid] += val * val;
    }
    __syncthreads();

    // Reduce shared memory values to a single value per block
    // Only thread 0 does the final accumulation to global memory
    if (tid == 0) {
        float block_sum = 0.0f;
        float block_sum_sq = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            block_sum += s_sum[i];
            block_sum_sq += s_sum_sq[i];
        }
        // Atomic add to global memory
        atomicAdd(&sum_out[feature_idx], block_sum);
        atomicAdd(&sum_sq_out[feature_idx], block_sum_sq);
    }
}

// Kernel for the final Batch Normalization forward pass
__global__ void bn_forward_kernel(const float* __restrict__ x, const float* __restrict__ bias, 
                                  const float* __restrict__ scale, 
                                  const float* __restrict__ mean, const float* __restrict__ inv_std, 
                                  const float* __restrict__ gamma, const float* __restrict__ beta, 
                                  float* __restrict__ out, int batch_size, int out_features) {
    int feature_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    float s = scale[feature_idx];
    float b = bias[feature_idx];
    float g = gamma[feature_idx];
    float bn_b = beta[feature_idx];
    float m = mean[feature_idx];
    float i_std = inv_std[feature_idx];

    // Process rows in a grid-stride loop
    for (int i = tid; i < batch_size; i += blockDim.x) {
        int idx = i * out_features + feature_idx;
        float val = x[idx];
        // Apply bias (from Linear), scaling (from the model), then BN logic
        val = val + b; // Bias
        val = val * s; // Scale
        val = (val - m) * i_std; // Normalize
        val = val * g + bn_b; // Affine
        out[idx] = val;
    }
}

// Helper to handle cublas
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}

torch::Tensor fused_gemm_scale_bn_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                                       torch::Tensor scale, torch::Tensor running_mean, 
                                       torch::Tensor running_var, torch::Tensor gamma, 
                                       torch::Tensor beta, bool training, float eps, float momentum) {
    // 1. GEMM
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(0);
    
    auto gemm_out = torch::empty({batch_size, out_features}, x.options());
    
    // Perform GEMM: out = x * W^T
    float alpha = 1.0f;
    float beta_val = 0.0f; 
    
    cublasHandle_t handle = get_cublas_handle();
    
    // cuBLAS assumes column-major storage.
    // We want to compute C = x * W^T (Row-Major).
    // This is equivalent to C^T = W * x^T (Col-Major).
    // A = W (Out, In) row-major. Treated as Col-Major (In, Out) with ld=Out. op(A)=T -> W (Out, In).
    // B = x (Batch, In) row-major. Treated as Col-Major (In, Batch) with ld=Batch. op(B)=N -> x^T (In, Batch).
    // C = gemm_out (Batch, Out) row-major. Treated as Col-Major (Out, Batch) with ld=Batch.
    
    cublasSgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N, 
                out_features, batch_size, in_features, 
                &alpha, 
                weight.data_ptr<float>(), out_features, 
                x.data_ptr<float>(), batch_size,         
                &beta_val, 
                gemm_out.data_ptr<float>(), batch_size); 

    // 2. Fused Bias + Scale + BN
    auto out = torch::empty_like(gemm_out);
    
    const int block_size = 256;
    const int num_blocks = out_features;
    
    // Handle optional bias
    const bool has_bias = bias.defined() && bias.numel() > 0;
    auto bias_tensor = has_bias ? bias : torch::zeros({1}, x.options()); 

    if (training) {
        // Compute stats
        auto sum = torch::zeros(out_features, x.options());
        auto sum_sq = torch::zeros(out_features, x.options());
        
        size_t shared_mem_size = 2 * block_size * sizeof(float);
        scale_bn_stats_kernel<<<num_blocks, block_size, shared_mem_size>>>(
            gemm_out.data_ptr<float>(), bias_tensor.data_ptr<float>(), scale.data_ptr<float>(),
            sum.data_ptr<float>(), sum_sq.data_ptr<float>(),
            batch_size, out_features
        );
        
        // Compute mean and var
        // Note: These are small tensors (out_features size), CPU reduction is fine or small kernel.
        // Using Torch ops for simplicity and stability on CPU.
        auto mean_cpu = sum.to(torch::kCPU) / batch_size;
        auto var_cpu = (sum_sq.to(torch::kCPU) / batch_size) - (mean_cpu * mean_cpu);
        
        auto mean = mean_cpu.to(x.options());
        auto inv_std = 1.0 / torch::sqrt(var_cpu.to(x.options()) + eps);
        
        // Update running stats
        // running_mean = (1 - momentum) * running_mean + momentum * mean
        // PyTorch BN uses exponential moving average.
        // Note: PyTorch updates running stats on CPU tensors usually, but here they might be CUDA.
        // We perform update on CPU to avoid extra kernels/complexity, then copy back if needed (or if buffer is CPU).
        // The buffers provided by PyTorch are on the same device as the model parameters (CUDA).
        torch::lerp_out(running_mean, running_mean.to(torch::kCPU), mean_cpu, momentum);
        torch::lerp_out(running_var, running_var.to(torch::kCPU), var_cpu, momentum);
        // The buffers are CPU tensors in standard PyTorch BN? No, they are on device.
        // lerp_out works in-place. If we passed CPU tensors, we need to copy back.
        // Actually, running_mean/var in the arguments are CUDA tensors.
        // Let's do the update on CUDA to avoid sync.
        // Or just use the CPU values we already have.
        // Since we brought mean/var to CPU for calculation, we can update the CPU copies of running stats if we had them.
        // But we have CUDA buffers.
        // Let's just do:
        // running_mean.copy_(mean_cpu); // This is wrong, should be lerp
        // running_var.copy_(var_cpu);   // This is wrong.
        
        // Correct approach for running stats update without sync overhead (or minimal):
        // running_mean = running_mean * (1 - momentum) + mean * momentum
        // We can use torch::lerp_out on CUDA tensors directly.
        torch::lerp_out(running_mean, running_mean, mean, momentum);
        torch::lerp_out(running_var, running_var, var_cpu.to(x.options()), momentum); // var_cpu is correct unbiased var? 
        // PyTorch BN uses biased var for running_var update by default.
        
        // Launch forward kernel
        bn_forward_kernel<<<num_blocks, block_size>>>(
            gemm_out.data_ptr<float>(), bias_tensor.data_ptr<float>(), scale.data_ptr<float>(),
            mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            out.data_ptr<float>(), batch_size, out_features
        );
        
    } else {
        // Inference mode
        auto mean = running_mean;
        auto inv_std = 1.0 / torch::sqrt(running_var + eps);
        
        bn_forward_kernel<<<num_blocks, block_size>>>(
            gemm_out.data_ptr<float>(), bias_tensor.data_ptr<float>(), scale.data_ptr<float>(),
            mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            out.data_ptr<float>(), batch_size, out_features
        );
    }
    
    return out;
}
"""

fused_gemm_scale_bn_cpp_source = (
    "torch::Tensor fused_gemm_scale_bn_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor scale, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor gamma, "
    "torch::Tensor beta, bool training, float eps, float momentum);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_gemm_scale_bn_cpp_source,
    cuda_sources=fused_gemm_scale_bn_source,
    functions=["fused_gemm_scale_bn_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Parameters for GEMM (Linear)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Parameter for Scaling
        self.scale = nn.Parameter(torch.Tensor(scale_shape))
        
        # Parameters for BatchNorm
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        
        self.eps = eps
        self.momentum = momentum
        
        # Initialize weights (simple initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize scale
        nn.init.ones_(self.scale)

        self.fused_op = fused_ops.fused_gemm_scale_bn_cuda

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias, self.scale, 
                             self.running_mean, self.running_var, self.gamma, self.beta, 
                             self.training, self.eps, self.momentum)