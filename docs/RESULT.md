# kfault - First Collection Results (2026-07-09)

## Setup

- **Model:** DeepSeek V3.2 (`deepseek.v3-v1:0`) on AWS Bedrock, us-west-2.
- **Two arms, same weights:** `think` = `reasoning_effort: high` (model thinks
  before answering) vs `direct` = `reasoning_effort: none` (answers directly).
  The ONLY difference between arms is the reasoning switch. This is a
  controlled experiment: any difference in outcomes is caused by reasoning.
- **Tasks:** 20 KernelBench Level-2 problems, chosen randomly with seed 42
  (problems 1, 4, 5, 12, 14, 15, 18, 26, 28, 29, 30, 32, 36, 54, 55, 58, 65,
  70, 72, 76). One greedy sample (temperature 0) per problem per arm = 40 runs.
- **Pipeline:** official KernelBench one-shot prompt; classifier stages
  syntactic -> runtime -> functional -> silent -> pass; silent battery =
  fresh random draws, non-contiguous inputs, self-consistency, extreme (x100)
  inputs; fp16 as a separate robustness flag. Weight sync via KernelBench-style
  seeding (validated 6/6 on hand-crafted kernels). GPU: Modal T4,
  compute-sanitizer unavailable there (Stage 2b self-disables).
- **Timing:** kernel vs PyTorch reference on the standard input, CUDA events,
  median of 10 runs after 3 warm-ups.

## Headline results

| arm | syntactic | runtime | functional | silent | pass | pass rate |
|--------|-----------|---------|------------|--------|------|-----------|
| think | 5 | 0 | 4 | 1 | 10 | 50% |
| direct | 13 | 2 | 5 | 0 | 0 | 0% |

- **False-pass rate (RQ2), think arm:** 1 of 11 standard-passing kernels
  (9%) is a silent failure. The direct arm produced zero standard-passers,
  so its false-pass rate is undefined at this sample size.
- **Reasoning strongly reduces visible failures:** 50% vs 0% pass rate with
  identical weights.
- **Speed reality check:** of the 10 passing think kernels, only 2 beat
  PyTorch by more than 1.1x. Median speedup: **1.02x** (no real gain).
  One "pass" (p32) is 2.7x slower than PyTorch.
- **fp16 fragility:** all 10 passing kernels crash in half precision
  (they hardcode `float`). Reported as a robustness flag, not counted
  as silent.
- Both arms repeatedly forget `import math` in the Python wrapper
  (7 of 40 runs fail only because of this).

## Full results (40 runs)

Speedup = PyTorch time / kernel time on the standard input; "-" = kernel
never ran. A speedup for a `functional` kernel measures the wrong
computation, so treat those numbers as context only.

| # | problem | arm | category | detail | speedup |
|---|---------|-----|----------|--------|---------|
| 1 | Conv2D ReLU BiasAdd | direct | syntactic | build error | - |
| 1 | Conv2D ReLU BiasAdd | think | pass | survived battery (fp16_crash) | 1.06x |
| 4 | Conv2d Mish Mish | direct | functional | shape mismatch | 17.96x |
| 4 | Conv2d Mish Mish | think | syntactic | build error | - |
| 5 | ConvTranspose2d Subtract Tanh | direct | syntactic | build error | - |
| 5 | ConvTranspose2d Subtract Tanh | think | functional | mismatch on standard input | 1.18x |
| 12 | Gemm Multiply LeakyReLU | direct | runtime | illegal memory access | - |
| 12 | Gemm Multiply LeakyReLU | think | pass | survived battery (fp16_crash) | 1.01x |
| 14 | Gemm Divide Sum Scaling | direct | functional | mismatch on standard input | 3.72x |
| 14 | Gemm Divide Sum Scaling | think | **silent** | **non_contig** | **0.01x** |
| 15 | ConvTranspose3d BatchNorm Subtract | direct | syntactic | build error | - |
| 15 | ConvTranspose3d BatchNorm Subtract | think | pass | survived battery (fp16_crash) | 1.03x |
| 18 | Matmul Sum Max AvgPool LogSumExp | direct | functional | mismatch on standard input | 0.12x |
| 18 | Matmul Sum Max AvgPool LogSumExp | think | functional | mismatch on standard input | 0.02x |
| 26 | ConvTranspose3d Add HardSwish | direct | syntactic | build error | - |
| 26 | ConvTranspose3d Add HardSwish | think | syntactic | wrong __init__ arity | - |
| 28 | BMM InstanceNorm Sum ResidualAdd | direct | syntactic | name 'math' is not defined | - |
| 28 | BMM InstanceNorm Sum ResidualAdd | think | syntactic | name 'sqrt' is not defined | - |
| 29 | Matmul Mish Mish | direct | syntactic | name 'math' is not defined | - |
| 29 | Matmul Mish Mish | think | syntactic | name 'math' is not defined | - |
| 30 | Gemm GroupNorm Hardtanh | direct | syntactic | name 'math' is not defined | - |
| 30 | Gemm GroupNorm Hardtanh | think | pass | survived battery (fp16_crash) | 0.96x |
| 32 | Conv2d Scaling Min | direct | syntactic | build error | - |
| 32 | Conv2d Scaling Min | think | pass | survived battery (fp16_crash) | 0.37x |
| 36 | ConvTranspose2d Min Sum GELU Add | direct | functional | mismatch on standard input | 0.04x |
| 36 | ConvTranspose2d Min Sum GELU Add | think | pass | survived battery (fp16_crash) | 1.00x |
| 54 | Conv2d Multiply LeakyReLU GELU | direct | syntactic | name 'math' is not defined | - |
| 54 | Conv2d Multiply LeakyReLU GELU | think | pass | survived battery (fp16_crash) | 1.23x |
| 55 | Matmul MaxPool Sum Scale | direct | syntactic | build error | - |
| 55 | Matmul MaxPool Sum Scale | think | pass | survived battery (fp16_crash) | 1.00x |
| 58 | ConvTranspose3d LogSumExp HardSwish | direct | syntactic | build error | - |
| 58 | ConvTranspose3d LogSumExp HardSwish | think | pass | survived battery (fp16_crash) | 1.34x |
| 65 | Conv2d AvgPool Sigmoid Sum | direct | syntactic | build error | - |
| 65 | Conv2d AvgPool Sigmoid Sum | think | functional | mismatch on standard input | 1.05x |
| 70 | Gemm Sigmoid Scaling ResidualAdd | direct | functional | mismatch on standard input | 1.00x |
| 70 | Gemm Sigmoid Scaling ResidualAdd | think | pass | survived battery (fp16_crash) | 1.02x |
| 72 | ConvTranspose3d BatchNorm AvgPool | direct | syntactic | build error | - |
| 72 | ConvTranspose3d BatchNorm AvgPool | think | syntactic | name 'sqrt' is not defined | - |
| 76 | Gemm Add ReLU | direct | runtime | illegal memory access | - |
| 76 | Gemm Add ReLU | think | functional | mismatch on standard input | 0.84x |

## Worked example 1: the silent failure (p14, think arm)

**Task:** `14_Gemm_Divide_Sum_Scaling` - a linear layer, divide by 2, sum
over the hidden dimension, multiply by a scaling factor.

**What the model did:** it fused everything into one CUDA kernel. One thread
block per batch row; the row of `x` is loaded into shared memory; each thread
computes dot products against some weight rows; a block-level reduction sums
them:

```cuda
__global__ void fused_operation_kernel(const float* x, const float* weight,
                                       float* output, int batch_size,
                                       int input_size, int hidden_size,
                                       float scaling_factor) {
    int i = blockIdx.x;
    ...
    for (int k = threadIdx.x; k < input_size; k += blockDim.x) {
        x_shared[k] = x[i * input_size + k];   // <-- THE BUG
    }
    ...
}

torch::Tensor fused_operation_cuda(torch::Tensor x, torch::Tensor weight,
                                   float scaling_factor) {
    ...
    fused_operation_kernel<<<batch_size, block_size, shared_memory_size>>>(
        x.data_ptr<float>(), ...);            // no x.contiguous() anywhere
```

**Why it passes the standard test:** `x[i * input_size + k]` walks the raw
memory buffer assuming the tensor is stored row by row with no gaps
("contiguous"). KernelBench's standard input IS contiguous, and the math is
otherwise correct, so the output matches PyTorch and the normal `allclose`
check says PASS.

**Why it is actually wrong:** a tensor with the same values can be stored
differently in memory (a transposed view, a sliced view, a broadcast). Our
battery feeds the kernel a non-contiguous tensor with identical values. The
formula `i * input_size + k` now points at the wrong memory locations, the
kernel reads garbage, and the output is wrong. PyTorch handles strides
automatically, so the reference stays correct and the mismatch is exposed.
One line, `x = x.contiguous();`, would have fixed it.

**Bonus finding:** this "correct-looking" kernel is also **83x slower** than
PyTorch (2550 ms vs 31 ms). A naive one-block-per-row matmul cannot compete
with cuBLAS. So the kernel is silently wrong AND useless for speed - and the
standard benchmark check would have reported it as a success.

**Why the model made this mistake:** reasoning was spent on the hard part
(fusing four ops, shared-memory reduction) and it looks impressive. But the
model never considered that input tensors may have non-standard memory
layout - a classic "assumption bug" that no amount of self-checking against
the visible test would ever reveal. That is exactly the failure class this
project measures: reasoning removes the failures the test can see, not the
ones it cannot.

## Worked example 2 (short): the most common visible failure

7 of 40 runs died with `name 'math' is not defined` (or `sqrt`). Example
(p28, direct arm): the model wrote a custom weight-init in the Python wrapper:

```python
nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))   # uses math...
```

...but never wrote `import math` (it did remember `#include <math.h>` on the
CUDA side). One missing line kills the whole file. Both arms make this exact
mistake, which suggests it comes from the shared base model's habits, not
from reasoning.

## Caveats (honest limits of this data)

- One greedy sample per (problem, arm): the 9% false-pass rate has a very
  wide confidence interval. Multi-sample runs are needed for real statistics.
- The direct arm has zero standard-passers, so the silent-share comparison
  between arms (the relocation hypothesis) cannot be computed yet. Options:
  more samples, or medium/low reasoning arms (which also give a dose-response
  curve).
- compute-sanitizer does not run on Modal T4, so Stage 2b covers only
  crash-visible runtime failures here.
- Token usage and generation-time fields are recorded for future runs, but
  are missing for these 40 (they were generated before the field was added).
- `classify_seconds` includes GPU container start-up; use the `timing` field
  (pure CUDA events) for any performance claims.

## Where the data lives

Modal volume `kfault-results`, one JSON per (level, problem, arm, sample).
Each file stores: prompt, raw LLM output (including the full reasoning
trace), extracted kernel code, category + failure detail, robustness flags,
timing (ref_ms / kernel_ms / speedup), seeds and settings, timestamps, and
(for future runs) token usage and generation latency.
