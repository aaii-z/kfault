# Status — Level-2 repair experiment, Qwen3-235B

Model `qwen.qwen3-235b-a22b-2507-v1:0` on Bedrock. Third model, run
2026-08-04, to test whether the GLM-5 / DeepSeek agreement at ~20% false-pass
is a property of reasoning or a two-model coincidence.

Settings: GPU L4 · repair turns 3 · samples 1 · same stratified batches as the
other models.

**Complete: all 10 batches, 100 problems, on both L4 and T4.** Batches 07-10
ran in a different Modal workspace from 01-06. They failed once first because
the `bedrock` Modal Secret had not been migrated: Volumes and Secrets are
separate per-workspace objects and both must be recreated. No data was
affected -- the failure happened before any container started.

| batch | date (UTC) | state | pass think/direct | silent think/direct | notes |
|-------|-----------|-------|-------------------|---------------------|-------|
| 01 | 2026-08-04 | done | 9/1 | 1/2 | screening batch. think 9/10 cleared the go/no-go. |
| 02 | 2026-08-04 | done | 8/1 | 0/0 | zero silent failures in either arm. |
| 03 | 2026-08-04 | done | 8/2 | 2/2 | |
| 04 | 2026-08-04 | done | 7/0 | 0/0 | direct passed nothing. |
| 05 | 2026-08-04 | done | 10/2 | 0/1 | think 10/10 pass, 0 silent. |
| 06 | 2026-08-04 | done | 8/0 | 2/0 | direct: 6 syntactic, 4 functional, 0 pass. |
| 07 | 2026-08-04 | done | 9/0 | 0/2 | first batch on the new workspace. |
| 08 | 2026-08-04 | done | 8/0 | 2/2 | direct: 6 syntactic, 0 pass. |
| 09 | 2026-08-04 | done | 6/2 | 2/3 | think's worst batch: 25% batch-level false-pass. |
| 10 | 2026-08-04 | done | — | — | see full-set totals below. |

## Result (100 problems, 200 chains, 505 attempts)

`uv run modal run analyze.py --level 2 --model-id qwen.qwen3-235b-a22b-2507-v1:0 --gpu L4`

| arm | stage | syntactic | runtime | functional | silent | pass |
|-----|-------|-----------|---------|------------|--------|------|
| direct | one-shot | 52 | 11 | 24 | 11 | 2 |
| direct | repaired | 37 | 5 | 36 | 14 | 8 |
| think | one-shot | 12 | 2 | 12 | 5 | 67 |
| think | repaired | 1 | 1 | 6 | 9 | 83 |

False-pass (RQ2):

| | one-shot | repaired |
|---|---|---|
| think | 5/72 = **7%** | 9/92 = **10%** |
| direct | 11/13 = 85% ⚠ | 14/22 = 64% ⚠ |

⚠ small denominator; report with the n attached, not as a bare percentage.

Repair outcomes over 200 chains:

```
 85  passed or silent on turn 0, no repair attempted
 22  repaired to pass
  7  visible failure repaired into a SILENT failure
 86  still visibly failing after repair
```

## Findings

**1. False-pass is ~10%, about half the other two models.** GLM-5 and DeepSeek
both landed at 19-20% after repair. Qwen sits at 10% while passing *more*
kernels than either (83 vs 75 and 76). The divergence held as the sample grew
from 60 to 100 problems, so it is not an artifact of which batches ran first.

**2. Relocation happens here too, but weakly.** CORRECTION: at n=60 this file
reported the rate as flat (9% to 9%) and concluded relocation was absent. The
full 100 problems show it rising 7% to 10%, with 7 chains converting a visible
failure into a silent one. So the effect is present in all three models --
GLM-5 +6 points, DeepSeek +3, Qwen +3 -- but the *level* differs sharply
(20% / 19% / 10%). The earlier "no relocation in Qwen" claim was an artifact
of the smaller sample and should not be used.

Confounder that still stands: Qwen's repair is less *effective* too -- 22
rescued against 25 (GLM-5) and 39 (DeepSeek), with 86 chains still failing. A
repair loop that changes less has less opportunity to hide anything. "Qwen is
safer" and "Qwen's repair does little" predict the same low hidden count, and
the totals cannot separate them. That needs the per-turn traces.

## Caveats

- The `direct` arm is unmeasurable for RQ2 (denominators 13 and 22), the same
  limitation as DeepSeek. Note that 11 of its 13 one-shot passes were silent.
- Batches 07-10 ran in a different Modal workspace from 01-06. Same image,
  same pinned packages, same L4 -- but it is an environment change worth noting.

## T4 replication (2026-08-04/05)

All 100 problems re-run on a T4 (Turing, sm_75) against the L4 (Ada, sm_89)
run above, for RQ4. Full write-up in [[../../CROSS_GPU.md]].

Settings scaled to the slower device: `KFAULT_FORWARD_LIMIT_S=20` and
`KFAULT_CLASSIFIER_TIMEOUT_S=2700`. A fixed 5 s budget fails kernels for being
on slower hardware rather than for being wrong.

| arm | stage | syntactic | runtime | functional | silent | pass | error |
|-----|-------|-----------|---------|------------|--------|------|-------|
| direct | one-shot | 52 | 9 | 23 | 12 | 2 | 2 |
| direct | repaired | 35 | 5 | 37 | 15 | 6 | 2 |
| think | one-shot | 12 | 1 | 13 | 5 | 66 | 1 |
| think | repaired | 1 | 1 | 5 | 9 | 82 | 2 |

false-pass repaired: think 9/91 = **10%**, direct 15/21 = 71% (n too small)
repair outcomes: 20 rescued, **7 hidden**, 84 still failing (200 chains)

Compare L4: think 83 pass / 9 silent / 10%, and 22 rescued / 7 hidden / 86
still failing. Qwen's lower false-pass rate reproduces on a second
architecture, and so does its lower hidden count.

### This is the only model of the three that diverges across GPUs

Cross-GPU comparison (same code, same schema, joined on
`generation_artifact_key`): 441 kernels ran on both devices, 429 (97.3%)
identical, 9 artifacts, and **3 genuine divergences** -- p88 direct t0,
p13 direct t2, p64 direct t0. All three are the same mechanism:

```
L4 (Ada):     functional  wrong value, or shape mismatch
T4 (Turing):  runtime     CUDA error: an illegal memory access
```

An out-of-bounds read that lands on valid memory on Ada and faults on Turing.
GLM-5 (0 of 367) and DeepSeek (0 of 496) show none.

This is **not** evidence that Qwen is architecture-sensitive in general. The
mechanism needs a latent memory bug to exist in the first place, and Qwen's
`direct` arm writes more of them. Note the direction too: the bug is *louder*
on the older GPU, so testing only on an L4 would file all three as ordinary
wrong-answer bugs.
