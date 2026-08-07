# Cross-model comparison — KernelBench Level 2

Three models, same stratified batches, same classifier, same GPU (L4), same
prompt, greedy sampling, 3 repair turns. Both arms are the same weights with
`reasoning_effort` set to `high` (think) or `none` (direct).

Run 2026-08-03/04. Per-model detail in `runs/*/STATUS.md`.

All three models cover all 100 problems.

## False-pass rate (RQ2), silent / (silent + pass)

| | GLM-5 | DeepSeek | Qwen3-235B |
|---|---|---|---|
| think one-shot | 10/69 = 14% | 8/49 = 16% | 5/72 = **7%** |
| **think repaired** | **19/94 = 20%** | **18/94 = 19%** | **9/92 = 10%** |
| direct one-shot | 11/49 = 22% | 1/3 = 33% ⚠ | 11/13 = 85% ⚠ |
| direct repaired | 18/65 = 28% | 8/14 = 57% ⚠ | 14/22 = 64% ⚠ |

⚠ = denominator too small to report as a bare percentage.

## Pass counts (out of 100)

| | GLM-5 | DeepSeek | Qwen |
|---|---|---|---|
| think one-shot | 59 | 41 | 67 |
| think repaired | 75 | 76 | **83** |
| direct one-shot | 38 | 2 | 2 |
| direct repaired | 47 | 6 | 8 |

## Finding 1 — two models converge, the third does not

GLM-5 and DeepSeek land in almost exactly the same place after repair:

| | GLM-5 think | DeepSeek think |
|---|---|---|
| pass | 75 | 76 |
| silent | 19 | 18 |
| false-pass | 20% | 19% |

Different labs, different training data, different one-shot ability (59 vs
41) — same endpoint. **About one in five kernels that look correct are not.**

Qwen3-235B breaks it: **10%**, about half — while passing *more* kernels than
either (83 vs 75 and 76). It is both the most capable and the most trustworthy
of the three.

So ~20% is not a universal constant of reasoning. Two models agree; a third
sits well below. The interesting question for the paper is what differs, not
what is shared.

## Finding 2 — repair relocates failures in every model, but by different amounts

| | GLM-5 | DeepSeek | Qwen |
|---|---|---|---|
| repaired to pass | 25 | 39 | 22 |
| **visible failure became SILENT** | **16** | **17** | **7** |
| still visibly failing | 40 | 90 | 86 |
| false-pass one-shot to repaired | 14 to 20% (+6) | 16 to 19% (+3) | 7 to 10% (+3) |

The direction is consistent across all three: **repair raises the false-pass
rate**. Every model ends with more silent failures than it started with, and a
standard KernelBench harness scores all of them as successes.

What differs is the level, not the direction — 20% / 19% / 10%.

An earlier version of this file, written when Qwen had only 60 problems,
claimed Qwen showed *no* relocation (9% flat). The full 100 problems corrected
that to 7 to 10%. Treat the effect as general and its magnitude as
model-dependent.

**Confounder that must be addressed before calling Qwen "safer":** its repair
loop is also less *effective* — 22 rescued against 25 and 39, with 86 chains
still failing. A repair loop that changes less has less chance to hide
anything. "Safer model" and "weaker repair loop" predict the same low hidden
count, and the totals cannot separate them. Separating them needs the per-turn
traces.

## Finding 3 — GLM-5's non-reasoning arm is the outlier

Reasoning off, the models are nothing alike:

| direct arm, after repair | pass |
|---|---|
| GLM-5 | 47/100 |
| DeepSeek | 6/100 |
| Qwen3-235B | 8/100 |

Two of three models essentially cannot write correct Level-2 CUDA kernels
without reasoning. GLM-5 can. This makes GLM-5 the only model in the set where
the *relocation* question (RQ1/RQ3) can be studied with both arms; the others
contribute think-arm replications for RQ2 only.

## Threats to validity

- **n = 3 models**, and they disagree on magnitude. Report the spread, not an average.
- **Qwen batches 07-10 ran on a different Modal workspace** (credit exhausted on
  the first). Same image, pinned packages and GPU, but note the change.
- **Only GLM-5 has two usable arms.** DeepSeek and Qwen direct-arm false-pass
  rests on denominators of 3-22 and is not reportable.
- **Qwen's low hidden-failure count is confounded** with its weak repair loop.
- **compute-sanitizer is unavailable on Modal**, so stage 2b never ran for any
  model. Silent memory errors that do not crash are undetected.
- **Greedy, one sample per problem.** No within-model variance estimate.
- **Level 2 only.** Level 3 is planned but not started.
