# kfault — Silent Failures in LLM-Generated GPU Kernels
### Presentation deck (one section = one slide)

---

## Slide 1 — The claim

> Reasoning models don't **fix** correctness bugs in CUDA kernels —
> they **relocate** them from *visible* to *silent*.

A "silent" kernel **passes the usual `allclose` check but is wrong under stress.**
So reported KernelBench pass rates **overcount** real correctness.

---

## Slide 2 — The 4-stage taxonomy

```
compile? ─no→ SYNTACTIC     ┐
   │yes                     │
run?     ─crash→ RUNTIME    │  VISIBLE  (user notices)
   │ok                      │
correct? ─no→ FUNCTIONAL    ┘
   │yes (passes allclose)
stress?  ─fail→ SILENT   ★  HIDDEN  (only stress reveals it)
   │survives
        PASS
```

The **silent** stage is the novelty. Everything else is standard.

---

## Slide 3 — The pipeline (all on one Modal GPU)

```
KernelBench task ─► LLM (R1 vs V3) ─► Stage 1 compile
                                   ─► Stage 2 run (+compute-sanitizer)
                                   ─► Stage 3 allclose vs PyTorch
                                   ─► Stage 4 STRESS BATTERY ─► silent / pass
```

**Stress battery** (fixed, small):

- `non_contig` — non-contiguous inputs (stride bugs)
- `extreme` — ×100 inputs (overflow / instability)
- `nondeterministic` — same input twice must agree (races / uninit)
- `fresh_seed` — new random draws (hardcoded output)
- `fp16` — dtype (reported separately, **not** counted as silent)

**Validated:** 5/5 on hand-crafted kernels of known category.

---

## Slide 4 — First results (Level 2, problems 1–10)

```
model    syntactic  runtime  functional  silent  pass
V3            5        1          4         0      0
R1            2        1          6         1      0
```

- R1 shifts failures *away* from visible (2 vs 5 syntactic) → later stages.
- **The silent detector fired:** R1's problem-9 kernel passed the standard
  check but the battery caught it.
- ⚠️ Proof of concept, not a statistic — counts are tiny (need full Level 2/3).

---

## Slide 5 — What "silent" looks like: 5 examples

| # | Bug                         | Passes standard because… | Caught by          |
|---|-----------------------------|--------------------------|--------------------|
| 1 | Ignores strides             | test inputs are contiguous | `non_contig`     |
| 2 | Softmax w/o max-subtract    | small values don't overflow | `extreme`       |
| 3 | Missing `__syncthreads()`   | race happens to align once | `nondeterministic`|
| 4 | Output ignores input        | this input → ReLU zeros both | `fresh_seed`   |
| 5 | Hardcoded `float`           | (fp32-only test)          | `fp16` (robust)   |

---

## Slide 6 — One example in full (stride bug)

```cuda
// BUGGY: reads storage linearly, ignores strides
if (i < n) out[i] = x[i] * 2.0f;      // no x.contiguous()
```

```
contiguous input:      [[2,4],[6,8]]   ✓ matches PyTorch  → "looks correct"
non-contiguous view:   [[2,6],[4,8]]   ✗ scrambled        → SILENT
correct:               [[2,4],[6,8]]
```

**Fix:** `x = x.contiguous();` before the launch.
The standard `allclose` never tries a non-contiguous input — so it misses this.

---

## Slide 7 — Numerical-instability example (softmax)

```cuda
// BUGGY: exp with no max subtraction
sum += expf(x[j]);   out[j] = expf(x[j]) / sum;
```

```
input ~N(0,1):   valid probabilities   ✓
input ×100:      expf(300)=inf → NaN    ✗  → SILENT
correct (×100):  valid probabilities (PyTorch subtracts the row max)
```

Same code, "correct" on the test, silently broken in production.

---

## Slide 8 — Why it matters

- A single `allclose` is the industry-standard acceptance test — and it is
  **blind to a whole class of bugs**.
- Reasoning models make kernels that *look* more correct while hiding harder,
  stress-only failures → **pass rates overstate trustworthiness**.
- **Deliverable:** a validated 4-category taxonomy + a concrete **false-pass
  rate** showing how much reported correctness is overcounted.

---

## Slide 9 — Honesty slide (limitations / next)

- **Weight sync:** reference vs generated models aren't seeded to share weights
  yet → Level-2 `functional` counts not final (fix: seed like KernelBench).
- **compute-sanitizer** doesn't run on Modal's T4 (self-disables); use L4/A100.
- **Scale up:** full Level 2 + 3, multiple samples per problem, for real stats.

---

## Slide 10 — Takeaway

> **"Passes the test" ≠ "correct."**
> kfault measures the gap — and shows reasoning models widen the *silent* part
> of it.
