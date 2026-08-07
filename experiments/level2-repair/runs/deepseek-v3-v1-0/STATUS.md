# Status — Level-2 repair experiment, DeepSeek V3.2

Model `deepseek.v3-v1:0` on Bedrock. Second model, run 2026-08-03 to test
whether the GLM-5 findings are a property of reasoning or of that one model.

Settings: GPU L4 · repair turns 3 · samples 1 · same 10 stratified batches as
[[../zai-glm-5/STATUS.md]].

| batch | date (UTC) | state | pass think/direct | silent think/direct | notes |
|-------|-----------|-------|-------------------|---------------------|-------|
| 01 | 2026-08-03 | done | 9/0 | 1/1 | screening batch. direct passed 0/10 with 7 `functional`. |
| 02 | 2026-08-03 | done | 9/0 | 1/2 | direct 0/10 again, 7 `functional`. Confirms 01 was not noise. |
| 03 | 2026-08-03 | done | 6/1 | 3/1 | one think `empty`, one direct `empty`. |
| 04 | 2026-08-03 | done | 6/0 | 2/1 | direct 8 `functional`. |
| 05 | 2026-08-03 | done | 8/0 | 1/0 | direct 0 pass, 0 silent: nothing to measure. |
| 06 | 2026-08-03 | done | 6/1 | 3/2 | |
| 07 | 2026-08-03 | done | 10/0 | 0/0 | think's best batch: 10/10 pass, 0 silent. |
| 08 | 2026-08-03 | done | 6/1 | 3/0 | |
| 09 | 2026-08-03 | done | 7/2 | 3/1 | direct's best batch, still only 2 passes. |
| 10 | 2026-08-03 | done | 9/1 | 1/0 | |

T4 replication, per batch (think/direct pass, then think/direct silent):

| batch | pass think/direct | silent think/direct | notes |
|-------|-------------------|---------------------|-------|
| 01 | 9/0 | 1/1 | |
| 02 | 9/0 | 1/2 | every cell identical to L4 |
| 03 | 6/0 | 3/1 | 2 direct `error` rows |
| 04 | 6/0 | 2/1 | every cell identical to L4 |
| 05 | 7/0 | 1/0 | one think `error` |
| 06 | 6/1 | 3/2 | |
| 07 | 10/0 | 0/0 | think 10/10 pass, 0 silent |
| 08 | 5/1 | 3/0 | |
| 09 | 6/2 | 3/1 | |
| 10 | 8/1 | 1/0 | |

All 10 batches completed, 200 chains, 557 attempts, no error rows.

## Full-set result (all 100 problems)

`uv run modal run analyze.py --level 2 --model-id deepseek.v3-v1:0 --gpu L4`

| arm | stage | syntactic | runtime | functional | silent | pass |
|-----|-------|-----------|---------|------------|--------|------|
| direct | one-shot | 31 | 30 | 36 | 1 | 2 |
| direct | repaired | 10 | 13 | 63 | 8 | 6 |
| think | one-shot | 15 | 16 | 20 | 8 | 41 |
| think | repaired | 1 | 1 | 4 | 18 | 76 |

False-pass (RQ2):

| | one-shot | repaired |
|---|---|---|
| think | 8/49 = 16% | 18/94 = **19%** |
| direct | 1/3 = 33% | 8/14 = 57% |

Repair outcomes over 200 chains:

```
 52  passed or silent on turn 0, no repair attempted
  2  failed on turn 0 but no repair recorded
 39  repaired to pass
 17  visible failure repaired into a SILENT failure
 90  still visibly failing after repair
```

## What this model can and cannot answer

**Usable — the `think` arm.** 76/100 pass after repair with a 94-kernel
denominator, directly comparable to GLM-5.

**Not usable — the `direct` arm's false-pass rate.** Across all 100 problems
the direct arm passed 2 one-shot and 6 after repair. The one-shot rate (1/3)
and repaired rate (8/14) rest on denominators too small to report as a
measurement. This was predicted by the batch-01 screening (0/10 passes) and
held for all 10 batches.

The direct arm is still evidence for RQ1: without reasoning this model
produces overwhelmingly `functional` failures (63/100 after repair), i.e. it
writes kernels that build and run but compute the wrong answer. That is a
capability-floor observation, not a false-pass measurement. Do not report
"57% false-pass for direct" without the n=14 caveat attached.

## T4 replication (2026-08-06/07)

All 100 problems re-run on a T4 (Turing, sm_75) against the L4 (Ada, sm_89)
run above, for RQ4. Full write-up in [[../../CROSS_GPU.md]].

Settings scaled to the slower device: `KFAULT_FORWARD_LIMIT_S=20` and
`KFAULT_CLASSIFIER_TIMEOUT_S=2700`. A fixed 5 s budget fails kernels for being
on slower hardware rather than for being wrong.

| arm | stage | syntactic | runtime | functional | silent | pass | error |
|-----|-------|-----------|---------|------------|--------|------|-------|
| direct | one-shot | 30 | 28 | 36 | 1 | 2 | 0 |
| direct | repaired | 10 | 14 | 59 | 8 | 5 | 1 |
| think | one-shot | 14 | 15 | 19 | 9 | 41 | 1 |
| think | repaired | 1 | 1 | 5 | 18 | 72 | 2 |

false-pass repaired: think 18/90 = **20%**, direct 8/13 = 62% (n too small)
repair outcomes: 34 rescued, **16 hidden**, 87 still failing (196 chains)

Compare L4: think 76 pass / 18 silent / 19%, and 39 / 17 / 90. Close on every
axis. **The relocation finding reproduces on a second architecture.**

Cross-GPU comparison (same code, same schema, joined on
`generation_artifact_key`): **496 kernels ran on both devices, 490 (98.8%)
got an identical label, 6 artifacts, 0 genuine divergences.** DeepSeek behaves
identically on Turing and Ada.

## Cross-model comparison

See [[../../CROSS_MODEL.md]]. Headline: the two models' `think` arms land in
almost exactly the same place after repair (GLM-5 75 pass / 19 silent / 20%;
DeepSeek 76 pass / 18 silent / 19%), while their `direct` arms are far apart.
