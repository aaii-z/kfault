__global__ void fused_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-x));
        float swish = x * sigmoid_val;
        float divided = swish / 2.0f;
        float clamped1 = fmaxf(fminf(divided, 1.0f), -1.0f);
        float tanh_val = tanhf(clamped1);
        float clamped2 = fmaxf(fminf(tanh_val, 1.0f), -1.0f);
        output[idx] = clamped2;
    }
}
