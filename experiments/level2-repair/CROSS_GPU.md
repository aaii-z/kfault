# RQ4 — cross-GPU divergence, KernelBench Level 2

> Are some kernels correct on one GPU but silently wrong on another?

**Answer: divergence is real but rare, and concentrated in one model.** Three
models, each 100 problems, every kernel classified on both an L4 and a T4:

| | GLM-5 | DeepSeek v3 | Qwen3-235B |
|---|---|---|---|
| kernels on both GPUs | 367 | 496 | 441 |
| identical label | 361 (98.4%) | 490 (98.8%) | 429 (97.3%) |
| harness artifact | 6 | 6 | 9 |
| **genuine divergence** | **0** | **0** | **3** |

**Qwen is the outlier.** Two of three models show zero cross-GPU divergence
across 863 kernels between them. Only Qwen's kernels behave differently on
Turing than on Ada, and only in 3 of 441 cases (0.7%).

Testing one model would have given the wrong answer either way: GLM-5 or
DeepSeek alone supports a clean null, Qwen alone suggests divergence is a
general phenomenon. It is neither -- it tracks the model.

Same generations, same classifier schema. Only the device differs.

| | L4 | T4 |
|---|---|---|
| architecture | Ada Lovelace | Turing |
| compute capability | sm_89 | sm_75 |
| memory | 24 GB | 16 GB |
| run date | 2026-08-03 | 2026-08-04 |
| forward budget | 5 s | 20 s |
| container timeout | 900 s | 2700 s |

**The two runs use different time limits on purpose.** A T4 is roughly 3.8x
slower (measured: L2 p41 direct took 3.54 s per forward on L4, 13.55 s on T4).
A fixed 5 s budget therefore fails kernels for running on slower hardware
rather than for being wrong. Scaling the budget keeps the *criterion* equal
even though the *number* differs. `KFAULT_CLASSIFIER_TIMEOUT_S` had to rise
with it, since 20 s x 90 forward calls exceeds a 900 s container.

## How to compare two GPU runs correctly

This took three attempts to get right. Two rules, both learned the hard way:

**1. Join on `generation_artifact_key`, never on final turn.** Once a repair
chain diverges, later turns contain *different generated kernels*. Comparing
final labels then compares different code and attributes the difference to the
device. The first version of this analysis reported "195/200 identical, 1
genuine divergence (p56)" on that basis; it was measuring repair-chain drift,
not hardware.

**2. Filter on `schema` and run date.** 21 records from 2026-07-29 use
classifier schema 3, and 4 more records carry `cfg=T4` while having actually
run on an L4 (the pre-`.env()` labelling bug, left uncorrected because they had
no `runtime_environment` to prove the device). Both sets sit on `gpu=T4` keys.
Including them compared *classifier versions* across problems 1, 2, 3, 4, 14,
which produced a second false "genuine divergence" at p14 direct.

With both rules applied, GLM-5's genuine-divergence count goes to zero. The
same corrected method applied to Qwen finds 3, so the rules do not simply
suppress everything.

### GLM-5's 6 artifacts

**Forward budget (3)** — p3 direct, p56 direct, p59 direct. All three ran
between 5.4 s and 6.0 s per forward on L4, tripping its 5 s budget, while T4's
scaled 20 s budget let them run to a `functional` verdict. This is the L4 run
being *stricter*, not the GPUs disagreeing.

**T4 memory, 16 GB (3)** — p100 think and p19 think died with `CUDA out of
memory` where L4 passes and fails respectively; p4 direct had battery probes
skipped on T4, so its `pass` there is not comparable to L4's `silent`.

Qwen's 9 artifacts follow the same two causes: five forward-budget (p18, p94,
p14 and others, 5.6-9.2 s per forward on L4) and four T4 memory (p83, p100).

## The 3 genuine divergences (all Qwen3-235B, all `direct` arm)

All three follow exactly one pattern:

```
p88 direct t0   L4: functional  output: value mismatch
                T4: runtime     CUDA error: an illegal memory access

p13 direct t2   L4: functional  output: shape mismatch (16,64,128,128) vs ...
                T4: runtime     CUDA error: an illegal memory access

p64 direct t0   L4: functional  output: shape mismatch (1024,) vs (1024, 1)
                T4: runtime     CUDA error: an illegal memory access
```

**The same out-of-bounds access lands on valid memory on the L4 and faults on
the T4.** On Ada the kernel reads whatever happens to be there and returns a
wrong number; on Turing the same read crosses an allocation boundary and traps.

Note the direction: the bug is *more* visible on the older, smaller GPU. A
`runtime` crash is loud; a `functional` wrong answer is quieter. Testing only
on an L4 would classify all three as ordinary wrong-output bugs and never learn
they are memory errors.

Why only Qwen: the mechanism needs a latent out-of-bounds access to exist in
the first place. This is not evidence that Qwen is architecture-sensitive in
general -- it is evidence that its `direct` arm writes more kernels with memory
bugs, and memory bugs are what can diverge.

**None of the three is a `pass` on either device**, so no kernel was scored
correct on one GPU while silently wrong on the other. The strongest form of the
RQ4 hypothesis is still unsupported; what the data shows is that the *category*
of a failure can change with the architecture.

## What this means

The false-pass rate is **not** an artifact of the measuring device: both models
reproduce their L4 false-pass rate on T4, and no kernel passed one GPU while
silently failing the other.

But failure *classification* is architecture-dependent for a small fraction of
kernels (0% of GLM-5, 0.7% of Qwen). A single-GPU study will mislabel some
memory errors as ordinary wrong-answer bugs.

## Aggregate results also reproduce

| arm | stage | syntactic | runtime | functional | silent | pass | (L4 pass) |
|-----|-------|-----------|---------|------------|--------|------|-----------|
| direct | one-shot | 16 | 9 | 26 | 10 | 39 | (38) |
| direct | repaired | 6 | 5 | 25 | 16 | 48 | (47) |
| think | one-shot | 17 | 1 | 11 | 10 | 58 | (59) |
| think | repaired | 0 | 0 | 5 | 19 | 74 | (75) |

False-pass on T4: think 20% repaired (L4: 20%), direct 25% (L4: 28%).
Repair outcomes on T4: 25 rescued, 16 hidden, 41 still failing, against
25 / 16 / 40 on L4.

**The relocation finding reproduces on a second architecture.** That is a
stronger claim than the cross-model replication alone, because the kernels are
byte-identical rather than merely comparable.

(Table above is GLM-5. Qwen's T4 aggregates likewise track its L4 run.)

## Threats to validity

- **compute-sanitizer is unavailable on both devices.** Races, missing
  `__syncthreads()` and unsynchronised warp behaviour are the failure modes
  most likely to differ across architectures, and are exactly what cannot be
  detected here. **This remains the biggest hole in the method**: the 3
  divergences found were caught only because they changed an output or crashed.
  A race that silently agrees on both devices is invisible to this study.
- **3 kernels had battery probes skipped on T4** because of 16 GB memory
  pressure, so they were tested less thoroughly there.
- **Three models, one architecture pair.** The 3-vs-0 split rests on a single
  L4/T4 comparison; a third architecture (A10G, Ampere) would test whether
  Qwen's divergences reproduce or are specific to Turing.
- **Greedy, single sample.** A divergence appearing only at temperature > 0
  would not be seen.
- **Two architectures.** A10G (Ampere) would add a third point cheaply, since
  generations are cached and GPU is not part of the generation key.
