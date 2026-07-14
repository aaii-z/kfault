# kfault — Experiment & Pipeline, Explained

**A Systematic Taxonomy of Correctness Failures in LLM-Generated GPU Kernels**
Author: Ali Ansari · Compute: Modal.com (serverless GPUs)

---

## 1. The idea in one paragraph

When you ask an LLM to write a CUDA kernel, "did it work?" is usually answered
by a single `torch.allclose` check against PyTorch on one standard input. Our
thesis is that this check is **too weak**, and that stronger (reasoning) models
don't actually *fix* correctness — they **relocate** failures from *visible*
ones (won't compile, crashes, obviously wrong) to *silent* ones: kernels that
**pass the standard check but are wrong under stress**. If that's true, reported
KernelBench pass rates **overcount** real correctness.

To measure this we built a pipeline that sorts every generated kernel into a
4-stage taxonomy and, crucially, adds a **Stage 4 "silent-failure detector"**
that stress-tests kernels which already passed the standard check.

---

## 2. The taxonomy (first matching stage wins)

```
generate kernel
      │
      ▼
┌─────────────┐  fails to compile / import         →  SYNTACTIC   (visible)
│  Stage 1    │───────────────────────────────────────────────────────►
└─────┬───────┘
      ▼
┌─────────────┐  crashes on the standard input     →  RUNTIME     (visible)
│  Stage 2    │  (or compute-sanitizer flags it)
└─────┬───────┘───────────────────────────────────────────────────────►
      ▼
┌─────────────┐  wrong output on the standard input →  FUNCTIONAL  (visible)
│  Stage 3    │  (basic allclose vs PyTorch)
└─────┬───────┘───────────────────────────────────────────────────────►
      ▼
┌─────────────┐  passes standard check, but WRONG   →  SILENT      (★ hidden)
│  Stage 4    │  under the stress battery
└─────┬───────┘───────────────────────────────────────────────────────►
      ▼
    PASS  (survives the whole battery)
```

- **Visible** = a normal user would notice immediately.
- **Silent** = passes the usual acceptance test; only stress reveals the bug.
  This is the star of the project.

---

## 3. The pipeline (`pipeline.py`)

The CPU `diagnose` function generates and caches code; a restricted GPU worker
classifies and benchmarks it:

1. **Prompt** — the *official* KernelBench one-shot CUDA/fp32 prompt, so our
   generations are directly comparable to published pass rates.
2. **Generate** — call DeepSeek V3.2 through AWS Bedrock. The two arms use the
   same model with `reasoning_effort` set to `high` or `none`.
3. **Stage 1 — syntactic:** `exec` the code; `load_inline` compiles the CUDA.
   A build/import error ⇒ `syntactic`.
4. **Stage 2 — runtime:** run once on the standard input and
   `torch.cuda.synchronize()` to surface async faults. Optionally re-run under
   **`compute-sanitizer --tool memcheck`** to catch illegal/out-of-bounds
   memory that doesn't crash. A crash or a real memory violation ⇒ `runtime`.
5. **Stage 3 — functional:** `allclose` vs PyTorch (fp32, TF32 disabled for a
   fair comparison). Wrong shape or values ⇒ `functional`.
6. **Stage 4 — silent:** the fixed stress **battery** (below). Any *core*
   failure ⇒ `silent`; otherwise ⇒ `pass`.

Raw generations and classifications are cached separately in a Modal Volume.
Changing GPU, seed, or sanitizer settings reuses the paid generation while
creating a correctly keyed classification. A single command sweeps many
problems in parallel and prints the category table + false-pass rate.

### 3.1 The silent-failure battery (Stage 4)

Deliberately **small and fixed** (the plan warns against sprawl). Split into two
tiers:

**CORE** — same dtype (fp32); a correct kernel *must* pass these, so failing one
means `silent`:

| probe             | what it catches                                   |
|-------------------|---------------------------------------------------|
| `fresh_seed`      | 3 new random draws — luck / hardcoded output      |
| `non_contig`      | non-contiguous inputs — stride assumptions        |
| `nondeterministic`| same input twice must agree — races / uninit / missing `__syncthreads` |
| `zeros`           | zero floating inputs — hidden value assumptions   |
| `small_magnitude` | near-zero values — precision / branch errors      |
| `extreme`         | ×100 magnitude inputs — overflow / instability    |
| `custom_*`        | problem-defined empty/odd/boundary shapes         |

**ROBUSTNESS** — reported separately, **not** counted as silent (KernelBench is
fp32-only, so counting these would inflate the number):

| probe    | what it catches          |
|----------|--------------------------|
| `fp16`   | half-precision / dtype assumptions |

`nondeterministic` is the key *reference-free* metamorphic check: it compares
two runs of the kernel against **each other**, so it flags unstable output even
where the kernel happens to match PyTorch on the first draw.

### 3.2 Validation harness (`validate.py`) — why you can trust the labels

Before trusting any number, we feed the classifier **hand-crafted kernels of
known category** and assert it labels each correctly. Current status:
**10/10**, including weights, input mutation, structured outputs, nested
inputs, and custom boundary shapes. Each case runs in its
own container because a real `runtime` (illegal memory) corrupts the CUDA
context for the rest of the process.

---

## 4. The experiment we ran

**Setup:** KernelBench **Level 2**, problems **1–10**, each generated by both
**R1** and **V3** (20 kernels), all on a T4, single-shot, temperature 0.

**Category distribution**

```
model    syntactic    runtime  functional     silent       pass
v3               5          1           4          0          0
r1               2          1           6          1          0
```

**False-pass rate (RQ2)** — of kernels that pass the standard check, how many
are actually silent?

```
v3: no standard-passing kernels
r1: 1/1  = 100%   (the single R1 kernel that passed the standard check was silent)
```

**What it shows:** the pipeline runs end-to-end and the Stage-4 novelty *fired*
— R1's problem-9 kernel passed the standard `allclose` yet the battery caught
it. Directionally, R1 shifts failures away from *visible* (2 syntactic vs V3's
5) toward *later* stages (6 functional + 1 silent), consistent with the
relocation hypothesis.

**What it does NOT yet show:** these are tiny counts. "100% false-pass" is
`1/1`; **zero passes** for either model is expected for single-shot temp-0
generation on KernelBench's hardest level, but it means the silent/pass
denominator is far too small for a real statistic. This run is a *proof of
concept*, not a result.

### The real silent case (R1, problem 9 — Matmul → Subtract → Multiply → ReLU)

R1 produced a shared-memory **tiled fused GEMM** that computes
`ReLU((X·Wᵀ + b − s) · m)`. It passed compile, run, and the standard `allclose`,
but Stage 4 flagged it (`extreme_crash` under the ×100 probe). It is a genuine
"passes the basic check, breaks under stress" case — exactly the phenomenon the
thesis targets.

---

## 5. Five silent-failure examples

Each shows the **buggy** kernel, what it **outputs**, the **correct** version,
and **which battery probe** catches it. Snippets are trimmed to the essential
line(s).

### Example 1 — Stride assumption → caught by `non_contig`

A kernel that indexes raw storage linearly is correct only when the input is
contiguous (which KernelBench inputs always are).

```cuda
// BUGGY: reads storage in linear order, ignores strides
__global__ void scale2x(const float* x, float* out, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * 2.0f;      // x[i] = i-th element of STORAGE
}
torch::Tensor scale(torch::Tensor x){
    auto out = torch::empty_like(x);
    scale2x<<<(x.numel()+255)/256,256>>>(x.data_ptr<float>(),
                                          out.data_ptr<float>(), x.numel());
    return out;                            // note: no x.contiguous()
}
```

Logical input `[[1,2],[3,4]]` stored contiguously → **correct** `[[2,4],[6,8]]`.
Pass a **transposed** (non-contiguous) view of the same values — storage is
`[1,3,2,4]` — the kernel reads in the wrong order and returns `[[2,6],[4,8]]`.

```
standard (contiguous):  out = [[2,4],[6,8]]   ✓ matches PyTorch
non-contiguous view:    out = [[2,6],[4,8]]   ✗ scrambled  → SILENT
correct output:              [[2,4],[6,8]]
```

**Fix:** make it layout-safe.
```cuda
x = x.contiguous();   // before launch (or index using x.strides())
```

---

### Example 2 — Numerical instability → caught by `extreme`

A softmax without the standard max-subtraction is correct for typical inputs but
overflows on large ones.

```cuda
// BUGGY: exp() with no max subtraction
float sum = 0.f;
for (int j=0;j<C;j++) sum += expf(x[row*C+j]);
for (int j=0;j<C;j++) out[row*C+j] = expf(x[row*C+j]) / sum;
```

```
standard input  ~N(0,1):     out = valid probabilities        ✓ matches PyTorch
extreme input   (×100):      expf(300)=inf → inf/inf = NaN     ✗ all NaN → SILENT
correct output  (×100):      valid probabilities (PyTorch subtracts the row max)
```

**Fix:** subtract the row max first (numerically stable softmax).
```cuda
float m = -INFINITY; for(int j=0;j<C;j++) m = fmaxf(m, x[row*C+j]);
for(int j=0;j<C;j++) sum += expf(x[row*C+j]-m);
out[row*C+j] = expf(x[row*C+j]-m) / sum;
```

---

### Example 3 — Missing `__syncthreads()` (race) → caught by `nondeterministic`

A shared-memory reduction without barriers can read half-updated shared memory.
The result changes run-to-run.

```cuda
// BUGGY: no __syncthreads between load and reduction steps
__shared__ float s[256];
s[tid] = x[i];
for (int stride=128; stride>0; stride>>=1)
    if (tid < stride) s[tid] += s[tid+stride];   // races: no barrier
if (tid==0) out[blockIdx.x] = s[0];
```

```
run 1 on input X:  block sum = 12.31
run 2 on input X:  block sum = 12.44     ✗ same input, different answer → SILENT
correct output:    block sum = 12.375    (stable, matches PyTorch every time)
```

Our `nondeterministic` probe runs the kernel **twice on the same input** and
compares the two outputs — no reference needed.

**Fix:** synchronize.
```cuda
s[tid] = x[i]; __syncthreads();
for (int stride=128; stride>0; stride>>=1){
    if (tid < stride) s[tid] += s[tid+stride];
    __syncthreads();
}
```

The same probe also catches **uninitialized output**: a kernel that allocates
`torch::empty_like(out)` and only writes the covered indices leaves the tail as
garbage that varies each run.

---

### Example 4 — Input-independent / hardcoded output → caught by `fresh_seed`

The sneakiest class: the kernel barely depends on the input, so it matches on
the *specific* standard tensor but not on a fresh draw. (This is essentially
what happened in the real R1 p9 case: a large `subtract_value` sends everything
below zero, ReLU zeros both outputs, and they agree on the standard draw.)

```cuda
// BUGGY: forgets the matmul contribution; output collapses to ReLU(b - s)*m,
// which is ~all zeros whenever (bias - subtract) < 0
out[row*C+col] = fmaxf((bias[col] - subtract) * mul, 0.f);   // X·Wᵀ term missing!
```

```
standard draw:   ref ≈ 0 (ReLU zeros it), buggy = 0        ✓ allclose passes
fresh draw:      ref has non-zero entries, buggy still 0   ✗ mismatch → SILENT
correct output:  ReLU((X·Wᵀ + b − s)·m)   (depends on the input, as it should)
```

Our `fresh_seed` probe draws **3 new random inputs of the same shape**; an
output that doesn't track the input is exposed.

**Fix:** actually accumulate the `X·Wᵀ` dot product before the fused epilogue.

---

### Example 5 — Dtype assumption → reported as `fp16` (ROBUSTNESS, *not* silent)

A kernel hardcoded to `float` breaks in half precision. We report this but
**deliberately do not count it as silent**, because KernelBench is fp32-only and
counting fp16 breakage would inflate the false-pass number.

```cuda
// BUGGY under fp16: pointer typed float*, but tensor is half
elementwise<<<...>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
```

```
fp32 input:   correct                                   ✓ (this is what we grade on)
fp16 input:   data_ptr<float>() on a half tensor throws  → robustness flag "fp16"
```

This example exists to explain the **core vs robustness** split: the false-pass
rate only counts core failures, keeping the headline number defensible.

---

## 6. How the examples map to the pipeline

| Example | Silent class            | Battery probe        | Tier       |
|---------|-------------------------|----------------------|------------|
| 1       | stride assumption       | `non_contig`         | core       |
| 2       | numerical instability   | `extreme`            | core       |
| 3       | race / missing sync     | `nondeterministic`   | core       |
| 4       | input-independent output| `fresh_seed`         | core       |
| 5       | dtype assumption        | `fp16`               | robustness |

Every silent label the pipeline emits comes from one of these probes, and every
probe is exercised by the validation harness.

---

## 7. Known limitations & next steps

1. **Input shapes are problem-specific.** The generic battery covers fresh
   draws, strides, zero/small/extreme floating values, and determinism. Problems
   should define `get_stress_inputs()` for valid empty, odd, broadcasting, and
   boundary shapes that cannot be inferred safely.
2. **compute-sanitizer may be unavailable on Modal's T4** ("Device not
   supported"); the
   stage self-disables and degrades gracefully. Try a bare-metal GPU type
   (L4/A10G/A100) in the RQ4 multi-GPU pass to actually exercise it.
3. **Scale.** Sweep all of Level 2 (100 problems) + Level 3 so the silent/pass
   denominator supports statistics (RQ2/RQ3).
4. **Multiple samples per problem** (temperature > 0, k attempts) — single-shot
   temp-0 yields very few standard-passing kernels.
5. **Adversarial native code.** Generated CUDA/C++ still runs as native code.
   Modal isolation limits its network, credentials, resources, and lifetime,
   but the benchmark assumes optimization attempts rather than deliberately
   hostile extensions.

---

## 8. How to reproduce

```bash
# validate the classifier first (10 regression cases)
uv run modal run validate.py                          # expect 10/10

# run the taxonomy over problems (cached; re-runs are free)
uv run modal run pipeline.py --level 2 --problems 1-10
```
