__global__ void spatial_sum_kernel(const float* input, float* output, int B, int C, int H, int W) {
    int c = blockIdx.x;
    int b = blockIdx.y;
    if (c >= C || b >= B) return;

    float sum = 0.0f;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int index = ((b * C + c) * H + h) * W + w;
            sum += input[index];
        }
    }
    output[b * C + c] = sum;
}
