__global__ void fused_bias_tanh_kernel(const float* x, const float* bias, float* out, int size, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int c = (idx / spatial_size) % channels;
    float val = x[idx] - bias[c];
    out[idx] = tanhf(val);
}

torch::Tensor fused_bias_tanh_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
    TORCH_CHECK(bias.device().is_cuda(), "bias must be on CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be 4-dimensional");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3-dimensional");
    TORCH_CHECK(bias.size(0) == x.size(1) && bias.size(1) == 1 && bias.size(2) == 1,
                "bias shape mismatch: expected (C,1,1) but got ", bias.sizes());

    auto bias_contig = bias.contiguous();
    int batch = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    int spatial_size = height * width;
    int size = batch * channels * height * width;
    auto out = torch::empty_like(x);
