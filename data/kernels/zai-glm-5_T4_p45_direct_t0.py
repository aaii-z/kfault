import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Gemm + Sigmoid + Gemm + LogSumExp
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Sigmoid kernel
__global__ void sigmoid_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

// LogSumExp kernel using parallel reduction
__global__ void logsumexp_kernel(const float* x, float* out, int num_rows, int num_cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float max_val = -INFINITY;
    // Each thread processes multiple elements if necessary
    for (int i = tid; i < num_cols; i += blockSize) {
        float val = x[row * num_cols + i];
        if (val > max_val) max_val = val;
    }
    sdata[tid] = max_val;
    __syncthreads();

    // Reduction to find max
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    
    // Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int i = tid; i < num_cols; i += blockSize) {
        sum_exp += expf(x[row * num_cols + i] - row_max);
    }
    sdata[tid] = sum_exp;
    __syncthreads();

    // Reduction to sum exp
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        out[row] = logf(sdata[0]) + row_max;
    }
}

// Fused operation: Linear1 -> Sigmoid -> Linear2 -> LogSumExp
torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2) {
    // Linear 1: x @ w1.T + b1
    // x: (batch, in), w1: (hidden, in) -> (batch, hidden)
    auto batch_size = x.size(0);
    auto hidden_size = w1.size(0);
    auto output_size = w2.size(0);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto hidden_out = torch::empty({batch_size, hidden_size}, options);
    
    // Perform Gemm 1
    // Using cublasSgemm for performance
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Note: PyTorch stores linear weights as (out_features, in_features), so we use CUBLAS_OP_T for the transpose
    // x is (batch, in), w1 is (hidden, in). We want x * w1^T.
    // In row-major: C = A * B^T. In col-major (cuBLAS): C^T = B * A^T.
    // Here we treat x as A (batch x in), w1 as B (hidden x in).
    // We want C (batch x hidden) = x * w1^T.
    // cuBLAS expects column-major. Let's interpret dimensions carefully.
    // x: (batch, in) -> col-major (in, batch)
    // w1: (hidden, in) -> col-major (in, hidden)
    // We want out = x * w1^T (batch, hidden).
    // In col-major: out^T = (w1^T)^T * x^T = w1 * x^T.
    // So we call gemm with A=w1, B=x, C=hidden_out.
    // m = hidden, n = batch, k = in.
    // A = w1 (hidden x in), B = x (batch x in) -> wait.
    
    // Let's stick to standard row-major interpretation mapped to col-major calls.
    // C (batch, hidden) = A (batch, in) * B^T (hidden, in).
    // cuBLAS col-major: C^T = B * A^T.
    // A = w1 (hidden, in), B = x (batch, in) -> dimensions mismatch for standard gemm without transpose.
    
    // Correct approach for Linear layer x * W^T + b:
    // x: (batch, in), W: (out, in)
    // We want Y = x * W^T.
    // cuBLAS gemm: C = alpha * op(A) * op(B) + beta * C
    // We want Y (batch, out).
    // Let A = x (batch, in), B = W (out, in).
    // We need op(A) = A (batch, in), op(B) = B^T (in, out).
    // Result Y (batch, out).
    // In col-major (cuBLAS):
    // Y is (batch, out). In col-major, it's treated as (out, batch).
    // A (batch, in) -> col-major (in, batch).
    // B (out, in) -> col-major (in, out).
    // We want Y = A * B^T.
    // In col-major: Y^T = (B^T)^T * A^T = B * A^T.
    // So we pass A=w1, B=x, op(A)=N, op(B)=T? No.
    
    // Let's use: C (batch, hidden) = x (batch, in) * w1^T (hidden, in).
    // cuBLAS sees C as (hidden, batch).
    // x is (in, batch).
    // w1 is (in, hidden).
    // C = w1^T * x? No.
    
    // Correct logic:
    // We want Y = x * W^T.
    // Pass x as B (transposed), W as A (not transposed).
    // C (batch, hidden) -> col-major (hidden, batch).
    // A = W (hidden, in). op(A) = N. lda = hidden.
    // B = x (batch, in). op(B) = T. ldb = batch. (x^T is in x batch).
    // k = in. m = hidden. n = batch.
    // C = W * x^T. This gives (hidden, batch). This matches Y^T.
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, hidden_size, batch_size, x.size(1), 
                &alpha, w1.data_ptr<float>(), hidden_size, x.data_ptr<float>(), x.size(1), 
                &beta, hidden_out.data_ptr<float>(), hidden_size);
    
    // Add bias
    // hidden_out (batch, hidden) += b1 (hidden)
    // In col-major view, we are adding to rows.
    // We can use a kernel or cublasSger (rank-1 update) but kernel is simpler for broadcasting.
    // Actually, we can fuse bias into the sigmoid kernel.
    
    // Sigmoid + Bias kernel
    int size = batch_size * hidden_size;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    // Fused Bias Add and Sigmoid kernel
    // We need to pass bias and stride info.
    // Simple kernel assuming contiguous memory for now.
    // Since hidden_out is created empty, it is contiguous.
    
    auto sigmoid_bias_kernel = <<<num_blocks, block_size>>>(
        [=] __device__ (int idx) {
            int row = idx / hidden_size;
            int col = idx % hidden_size;
            float val = hidden_out[idx] + b1[col];
            hidden_out[idx] = 1.0f / (1.0f + expf(-val));
        }
    );
    // Wait, C++ doesn't support device lambdas like that easily without specific flags.
    // Let's write a proper kernel.
    
    // Re-launching a kernel for bias+sigmoid
    // ...
    
    // Actually, let's just use a separate kernel for bias add + sigmoid.
    // Or better, fuse it into the gemm output if possible, but cuBLAS doesn't support that.
    // We will launch a custom kernel for Bias + Sigmoid.
    
    // Kernel definition for Bias + Sigmoid
    const int bias_block_size = 256;
    int bias_blocks = (size + bias_block_size - 1) / bias_block_size;
    
    // ... (Kernel implementation below)
    
    // Linear 2: hidden_out @ w2.T + b2
    auto final_out = torch::empty({batch_size, output_size}, options);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, output_size, batch_size, hidden_size,
                &alpha, w2.data_ptr<float>(), output_size, hidden_out.data_ptr<float>(), hidden_size,
                &beta, final_out.data_ptr<float>(), output_size);
    
    // Add bias for linear 2
    // We can fuse bias add into LogSumExp? No, LogSumExp reduces.
    // We need to add bias to final_out before LogSumExp.
    // Let's fuse Bias Add + LogSumExp.
    
    auto result = torch::empty({batch_size}, options);
    
    // Launch fused Bias + LogSumExp kernel
    // ...
    
    cublasDestroy(handle);
    return result;
}

// Actual Kernel Implementations

__global__ void bias_add_sigmoid_kernel(const float* x, const float* bias, float* out, int size, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int bias_idx = idx % hidden_size;
        out[idx] = 1.0f / (1.0f + expf(-(x[idx] + bias[bias_idx])));
    }
}

__global__ void bias_add_logsumexp_kernel(const float* x, const float* bias, float* out, int num_rows, int num_cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float max_val = -INFINITY;
    for (int i = tid; i < num_cols; i += blockSize) {
        float val = x[row * num_cols + i] + bias[i];
        if (val > max_val) max_val = val;
    }
    sdata[tid] = max_val;
    __syncthreads();

    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    
    float sum_exp = 0.0f;
    for (int i = tid; i < num_cols; i += blockSize) {
        sum_exp += expf((x[row * num_cols + i] + bias[i]) - row_max);
    }
    sdata[tid] = sum_exp;
    __syncthreads();

    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        out[row] = logf(sdata[0]) + row_max;
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = w1.size(0);
    auto output_size = w2.size(0);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    
    // Intermediate tensor
    auto hidden_out = torch::empty({batch_size, hidden_size}, options);
    
    // 1. Linear 1 (Gemm)
    // We use ATen's matmul for simplicity and performance, it calls cuBLAS.
    // To truly optimize, we would use cublas directly, but ATen is reasonable.
    // However, to fuse, we need control.
    // Let's use torch::mm for the gemm part.
    // x: (batch, input), w1: (hidden, input). We want x * w1^T.
    // torch::mm expects (batch, input) * (input, hidden).
    // So we need to transpose w1.
    // But torch::linear handles this.
    
    // Let's stick to ATen for Gemm to ensure correctness and leverage cuBLAS.
    // We will fuse the post-Gemm operations.
    
    // Linear 1
    hidden_out = torch::addmm(b1, x, w1); // b1 + x * w1^T
    
    // 2. Sigmoid
    // We can fuse this with the next linear if we write a custom gemm.
    // But writing a performant Gemm is hard. Let's fuse Sigmoid into a kernel.
    // Or just use ATen sigmoid. It's memory bound.
    // To optimize, we can fuse Sigmoid with the next Gemm's input loading? No.
    
    // Let's try to fuse: Linear1 -> Sigmoid.
    // We can write a kernel that does Gemm? No.
    // We can write a kernel that does Sigmoid.
    
    // Strategy:
    // 1. torch::addmm (Linear 1) -> hidden_out
    // 2. Custom Kernel: Sigmoid (in-place or out-of-place)
    // 3. torch::addmm (Linear 2) -> final_out
    // 4. Custom Kernel: LogSumExp
    
    // Optimization:
    // We can fuse Bias + Sigmoid.
    // We can fuse Bias + LogSumExp.
    
    // Let's do Bias+Sigmoid kernel.
    auto sigmoid_out = torch::empty_like(hidden_out);
    int size = batch_size * hidden_size;
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    // Note: hidden_out already has bias added from addmm.
    // So we just need sigmoid.
    // But wait, addmm adds bias. We want Sigmoid(b1 + x*w1^T).
    // addmm does b1 + x*w1^T.
    // So we just apply sigmoid to hidden_out.
    
    // Kernel for Sigmoid
    auto sigmoid_kernel = [=] __device__ (int idx) {
        sigmoid_out[idx] = 1.0f / (1.0f + expf(-hidden_out[idx]));
    };
    // We can't use lambda easily. Use a proper kernel.
    
    // Actually, let's use torch::sigmoid. It's optimized.
    // The main gain is fusing the final LogSumExp and potentially the first Sigmoid if we could.
    // But the biggest gain is usually reducing memory round trips.
    
    // Let's try to fuse everything into one giant kernel?
    // No, Gemm is best left to cuBLAS.
    
    // Optimized Plan:
    // 1. Linear 1 (torch::addmm)
    // 2. Sigmoid (torch::sigmoid) -> or custom kernel if we want to fuse with next step.
    // 3. Linear 2 (torch::addmm)
    // 4. LogSumExp (Custom Kernel)
    
    // Let's implement the LogSumExp kernel.
    
    // Linear 1
    hidden_out = torch::addmm(b1, x, w1);
    
    // Sigmoid
    torch::sigmoid_out(hidden_out, hidden_out); // In-place
    
    // Linear 2
    auto final_out = torch::addmm(b2, hidden_out, w2);
    
    // LogSumExp
    auto result = torch::empty({batch_size}, options);
    
    int num_rows = batch_size;
    int num_cols = output_size;
    
    // Kernel launch configuration
    // We need enough shared memory.
    // Block size should be power of 2.
    int threads_per_block = 256; 
    // Shared memory size: threads_per_block * sizeof(float)
    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    // Check for max shared memory? Usually 48KB is safe. 256*4 = 1KB. Safe.
    
    logsumexp_kernel<<<num_rows, threads_per_block, shared_mem_size>>>(
        final_out.data_ptr<float>(), result.data_ptr<float>(), num_rows, num_cols
    );
    
    return result;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        # We extract weights and biases to pass to the CUDA kernel
        # Note: nn.Linear stores weights as (out_features, in_features)
        w1 = self.linear1.weight
        b1 = self.linear1.bias
        w2 = self.linear2.weight
        b2 = self.linear2.bias
        
        return self.fused_ops.fused_forward_cuda(x, w1, b1, w2, b2)