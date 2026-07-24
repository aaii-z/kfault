# kfault — Supervisor Update (2026-07-10)

**Project:** A Systematic Taxonomy of Correctness Failures in LLM-Generated GPU Kernels
**Author:** Ali Ansari (100526858)
**Status:** First full data collection done (40 runs, 2026-07-09). Pipeline validated end-to-end.

---

## 1. The claim (one sentence)

> Reasoning models do not **fix** correctness bugs in CUDA kernels — they
> **relocate** them from *visible* failures (compile error, crash, wrong
> output) to *silent* failures (passes the standard test, wrong under stress).
> So reported KernelBench pass rates **overcount** real correctness.

**Research questions:**
- **RQ1** — How do failure types distribute with reasoning on vs off?
- **RQ2** — What fraction of "correct" kernels are actually silent failures? (the false-pass rate)
- **RQ3** — Does reasoning *increase the share* of silent failures while reducing visible ones? (relocation hypothesis)
- **RQ4 (bonus)** — Are some kernels correct on one GPU but wrong on another?

---

## 2. What I did (the experiment)

- **Model:** DeepSeek V3.2 (`deepseek.v3-v1:0`) on **AWS Bedrock** (us-west-2).
- **Controlled contrast — two arms, same weights, same prompt:**
  - `think` = `reasoning_effort: high` (model reasons before answering)
  - `direct` = `reasoning_effort: none` (answers directly)
  The ONLY difference is the reasoning switch, so any difference in outcomes
  is **caused by reasoning** — not by comparing two different models.
- **Tasks:** 20 KernelBench Level-2 problems, sampled randomly with seed 42.
  One greedy sample (temperature 0) per problem per arm = **40 runs**.
- **Classifier:** every generated kernel goes through a 4-stage taxonomy:

  ```
  compile? ─no→   SYNTACTIC   ┐
     │yes                     │
  run?     ─crash→ RUNTIME    │  VISIBLE (the user notices)
     │ok                      │
  correct? ─no→   FUNCTIONAL  ┘
     │yes (passes allclose)
  stress?  ─fail→ SILENT  ★     HIDDEN (only stress reveals it)
     │survives
           PASS
  ```

- **Stress battery** (fixed, applied only to kernels that pass the standard
  `allclose`): fresh random draws, non-contiguous inputs, self-consistency
  (same input twice must agree), extreme (×100) inputs. **fp16** is reported
  as a separate robustness flag, not counted as silent.
- **Timing:** kernel vs PyTorch reference, CUDA events, median of 10 runs
  after 3 warm-ups.
- **Infrastructure:** everything runs serverless on **Modal** (T4 GPU
  containers); results cached as one JSON per (problem, arm) on a Modal
  volume, including the full reasoning trace.

---

## 3. Headline results

| arm    | syntactic | runtime | functional | silent | pass | pass rate |
|--------|-----------|---------|------------|--------|------|-----------|
| think  | 5         | 0       | 4          | 1      | 10   | **50%**   |
| direct | 13        | 2       | 5          | 0      | 0    | **0%**    |

- **Reasoning strongly reduces visible failures:** 50% vs 0% pass rate with
  identical weights. (RQ1 answered clearly.)
- **The silent detector works on real data (RQ2):** 1 of 11 standard-passing
  kernels (**9%**) is silently wrong — the standard benchmark check would have
  called it a success.
- **Speed reality check:** of the 10 passing think kernels, only 2 beat
  PyTorch by more than 1.1×. Median speedup **1.02×** — "correct" does not
  mean "fast". One pass is 2.7× *slower* than PyTorch.
- **fp16 fragility:** all 10 passing kernels crash in half precision (they
  hardcode `float`).
- **Shared model habit:** both arms repeatedly forget `import math` in the
  Python wrapper — 7 of 40 runs die only because of this one missing line.

### The star exhibit: the silent failure (problem 14, think arm)

The model fused Gemm + divide + sum + scaling into one impressive CUDA
kernel — but indexed the input as if it were contiguous in memory
(`x[i * input_size + k]`, no `x.contiguous()`). The standard KernelBench
input IS contiguous, so the kernel **passes `allclose`**. Feed it a
non-contiguous view of the *same values* and it reads the wrong memory and
returns garbage. One line would have fixed it. Bonus: this "passing" kernel
is also **83× slower** than PyTorch (2550 ms vs 31 ms).

This is exactly the failure class the project measures: **reasoning removes
the failures the test can see, not the ones it cannot.**

---

## 4. How I got here (process and thinking)

1. **Built the taxonomy + battery first, then validated it** on hand-crafted
   kernels with known bug types (stride bug, softmax overflow, missing
   `__syncthreads()`, hardcoded output, fp16) — every probe catches the bug
   class it was designed for.
2. **Chose a within-model design.** Early prototype compared two different
   models (R1 vs V3 via OpenRouter). Problem: any difference could come from
   different weights, not reasoning. Switch to DeepSeek V3.2 on Bedrock,
   where one flag toggles reasoning on the *same* weights → a clean causal
   contrast.
3. **Found and fixed a validity bug (weight-sync gap).** The generated model
   and the PyTorch reference were not guaranteed identical weights, which
   could mislabel kernels as `functional`. Fixed with KernelBench-style
   seeding; validated 6/6 on hand-crafted kernels (2026-07-09). All labels
   in this report are post-fix.
4. **Engineering constraints handled:** compute-sanitizer does not run on
   Modal T4 (the stage self-disables; L4/A100 planned); Modal batches capped
   at 10 containers; code extractor fixed for reasoning outputs that split
   the answer into segments.
5. **Kept the honesty rules:** speedups of `functional` kernels measure the
   wrong computation (context only); `classify_seconds` includes container
   start-up, so performance claims use pure CUDA-event timing only.

---

## 5. Honest limits of this data

- **n = 1 sample per (problem, arm):** the 9% false-pass rate has a very wide
  confidence interval. Multi-sample runs are needed for real statistics.
- **RQ3 (relocation) cannot be tested yet:** the direct arm has zero
  standard-passers, so its silent share is undefined. Options: more samples,
  or add medium/low reasoning arms (also gives a dose-response curve).
- compute-sanitizer coverage missing on T4 (crash-visible failures only).
- Token usage / generation-latency fields exist but are empty for these 40
  runs (added after they were generated).

## 6. Next steps

1. Scale up: more samples per problem + full Level 2 (and Level 3) for real
   statistics on RQ2/RQ3.
2. Add medium/low reasoning arms → dose-response curve of reasoning vs
   failure profile.
3. Run on L4/A100 to enable compute-sanitizer, and re-run across GPU types
   for the RQ4 hardware-divergence bonus.

---

## 7. Figure guide (files in `figures-light/`, dark theme in `figures-dark/`)

| file | suggested slide |
|------|-----------------|
| `kfault-00-headline.png` | Title / headline numbers (50% vs 0%, 9% false-pass, 1.02× median speedup) |
| `kfault-01-taxonomy.png` | The 4-stage taxonomy + stress battery |
| `kfault-02-outcomes.png` | Outcome counts per arm (RQ1) |
| `kfault-03-survival.png` | Survival through the classifier stages |
| `kfault-04-speedup.png` | Speedup distribution — "pass ≠ fast" |
| `kfault-05-worked-example.png` | The p14 silent stride bug, step by step |
| `kfault-06-patterns.png` | Recurring failure patterns (`import math`, fp16, …) |
| `kfault-07-table-full-results.png` | Full 40-run results table (backup slide) |
| `kfault-08-limitations.png` | Caveats + next steps |

**Deeper material if asked:** full per-run table and worked examples in
`docs/RESULT.md`; project plan and RQs in `plan.md`; raw JSONs (with full
reasoning traces) on the Modal volume `kfault-results`.
