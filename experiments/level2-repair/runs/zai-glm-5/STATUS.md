# Status — Level-2 repair experiment

Fill one row per finished batch. Keep it honest: record failed and partial
runs too, and note anything that differed from the protocol in README.md.

Settings: GPU L4 · repair turns 3 · samples 1 · model `zai.glm-5`

| batch | problems | date (UTC) | GPU | state | pass think/direct | silent think/direct | notes |
|-------|----------|-----------|-----|-------|-------------------|---------------------|-------|
| 01 | 1,11,21,31,41,51,61,71,81,91 | 2026-07-30 | T4 | overwritten | 8/4 | 2/3 | first run. p91 direct: 900s timeout, no data. false-pass think 20%, direct 43%. every pass flags fp16_crash — probe was a harness artifact (fixed 2026-08-03). **Records no longer exist**: the 2026-08-03 re-run overwrote them (see issue 4). Counts here come from the run log, which is the only surviving copy. |
| 01 | 1,11,21,31,41,51,61,71,81,91 | 2026-08-03 | L4 | done | 8/4 | 2/3 | re-run with the fp16 + forward-limit fixes, `FORCE=1`. Counts matched the earlier T4 log exactly — suggestive of reproducibility, but NOT a controlled check: the two runs differ in both GPU and classifier version, and the T4 records were overwritten so the comparison cannot be redone from data. false-pass think 20%, direct 43%. No `fp16_crash` anywhere: the probe fix works. **20/20 complete**: p91 direct, a data hole across two earlier runs, resolved to `functional` after the 900s timeout fix (chain below). |
| 02 | 2,12,22,32,42,52,62,72,82,92 | 2026-07-30 | L4 | overwritten | 6/3 | 1/2 | first run. GPU was unknown from the log (header was echoed outside the tee; fixed 2026-08-03), later recovered as L4 from `runtime_environment.gpu_name`. Hit `ConflictError: APP_STATE_STOPPED`; p42/p52 both arms died on Deadline exceeded. false-pass think 14%, direct 40%. Records overwritten by the re-run below. |
| 02 | 2,12,22,32,42,52,62,72,82,92 | 2026-08-03 | L4 | done | 8/5 | 2/2 | re-run with the 2026-08-03 fixes, `FORCE=1`. **20/20 complete, zero error rows**: p42 and p52 both arms, previously lost to Deadline exceeded, now carry real labels. false-pass think 20%, direct 29%. No `fp16_crash`. |
| 03 | 3,13,23,33,43,53,63,73,83,93 | 2026-08-03 | L4 | done | 7/5 | 2/2 | first batch run with every 2026-08-03 fix in place and the GPU correctly recorded (`cfg=L4`). 20/20, no errors. false-pass think 22%, direct 29%. p3 direct tripped the new `forward_time_limit` at turn 0 (6.0s per forward) and repair rescued it to `functional` — the slow-kernel path works end to end. |
| 04 | 4,14,24,34,44,54,64,74,84,94 | 2026-08-03 | L4 | done | 7/6 | 2/3 | 20/20, no errors. false-pass think 22%, direct 33%. |
| 05 | 5,15,25,35,45,55,65,75,85,95 | 2026-08-03 | L4 | done | 9/6 | 0/2 | 20/20, no errors. think's best batch: 9 pass, 0 silent. false-pass think 0%, direct 25%. |
| 06 | 6,16,26,36,46,56,66,76,86,96 | 2026-08-03 | L4 | done | 5/3 | 5/0 | 20/20, no errors. **The one batch where the pattern reverses**: false-pass think 50%, direct 0%. But direct's 0% rests on only 3 passing kernels (vs 1 syntactic, 3 runtime, 3 functional), so the rate is not meaningful — the denominator is too small. think passed 10 but 5 were silent. Do not read this batch as evidence for direct. p66 and p96 direct: `CUDA error: invalid argument` (labelled `runtime`, real kernel faults, not harness holes). |
| 07 | 7,17,27,37,47,57,67,77,87,97 | 2026-08-03 | L4 | done | 7/5 | 1/1 | 20/20, no errors. false-pass think 12%, direct 17%. |
| 08 | 8,18,28,38,48,58,68,78,88,98 | 2026-08-03 | L4 | done | 7/6 | 3/2 | 20/20, no errors. false-pass think 30%, direct 25%. |
| 09 | 9,19,29,39,49,59,69,79,89,99 | 2026-08-03 | L4 | done | 8/1 | 1/2 | 20/20, no errors. direct's worst batch: 1 pass, 3 syntactic, 4 functional. One think `empty` (generation produced no code block). false-pass think 11%, direct 67% (on 3 passes — small denominator). |
| 10 | 10,20,30,40,50,60,70,80,90,100 | 2026-08-03 | L4 | done | 9/6 | 1/1 | 20/20, no errors. false-pass think 10%, direct 14%. |

state: `todo` · `running` · `done` · `partial` · `failed` · `overwritten`

Batches 01 and 02 have two rows each: the original run and the 2026-08-03
re-run after the classifier fixes. **Report the 2026-08-03 rows.** The earlier
rows are kept for the record only — their underlying records were overwritten
(issue 4), so those counts survive solely in `logs/`.

## Full-set result (all 100 problems, 2026-08-03)

`uv run modal run analyze.py --level 2 --gpu L4` — 200 chains, 380 attempts.

| arm | stage | syntactic | runtime | functional | silent | pass | empty |
|-----|-------|-----------|---------|------------|--------|------|-------|
| direct | one-shot | 16 | 9 | 26 | 11 | 38 | 0 |
| direct | repaired | 6 | 5 | 24 | 18 | 47 | 0 |
| think | one-shot | 17 | 1 | 11 | 10 | 59 | 2 |
| think | repaired | 0 | 0 | 5 | 19 | 75 | 1 |

False-pass (RQ2), silent / (silent + pass):

| | one-shot | repaired |
|---|---|---|
| think | 10/69 = 14% | 19/94 = **20%** |
| direct | 11/49 = 22% | 18/65 = **28%** |

Repair outcomes over 200 chains:

```
118  passed or silent on turn 0, no repair attempted
  1  failed on turn 0 but no repair recorded
 25  repaired to pass
 16  visible failure repaired into a SILENT failure
 40  still visibly failing after repair
```

Three findings:

1. **Reasoning helps a lot.** think 59 vs direct 38 one-shot; 75 vs 47 after
   repair. Same weights, only reasoning effort differs.
2. **Repair fixes visible failures.** think's `syntactic` went 17 to 0 and
   `runtime` 1 to 0; direct's `syntactic` 16 to 6.
3. **Repair also hides failures.** False-pass rose in both arms (think 14 to
   20%, direct 22 to 28%) and silent counts rose (think 10 to 19, direct 11 to
   18). 16 chains turned a visible failure into a silent one, against 25
   genuine rescues. A standard KernelBench harness would score all 16 as
   successes. This is only measurable because stage 4 is withheld from the
   repair loop.

## T4 replication (2026-08-04)

All 100 problems re-run on a T4 (Turing, sm_75) against the original L4 (Ada,
sm_89), to answer RQ4. Full write-up in [[../../CROSS_GPU.md]].

Settings differ deliberately: `KFAULT_FORWARD_LIMIT_S=20` and
`KFAULT_CLASSIFIER_TIMEOUT_S=2700`, scaled to T4's ~3.8x slower forward pass.
A fixed 5 s budget fails kernels for being on slower hardware, not for being
wrong.

| arm | stage | syntactic | runtime | functional | silent | pass |
|-----|-------|-----------|---------|------------|--------|------|
| direct | one-shot | 16 | 9 | 26 | 10 | 39 |
| direct | repaired | 6 | 5 | 25 | 16 | 48 |
| think | one-shot | 17 | 1 | 11 | 10 | 58 |
| think | repaired | 0 | 0 | 5 | 19 | 74 |

false-pass repaired: think 19/93 = 20%, direct 16/64 = 25%
repair outcomes: 25 rescued, 16 hidden, 41 still failing

Compare L4: think 75 pass / 19 silent / 20%, and 25 / 16 / 40. The relocation
result reproduces on a second architecture.

Of 367 kernels classified on both GPUs, 361 (98.4%) got an identical label.
All 6 differences are harness artifacts. **Zero genuine architectural
divergences, and zero kernels that pass on L4 while silently failing on T4.**

Comparing *final turns* instead of matched generations gives a wrong answer
(it compares different code once repair chains diverge); see CROSS_GPU.md.

## Known issues found in batch 01

1. **`fp16` probe fires on every kernel.** ~~100% of passing kernels flag
   `fp16_crash`.~~ **Fixed 2026-08-03.** `kernel_model.half()` cannot work with
   a `load_inline` kernel compiled against `data_ptr<float>()`, so the probe had
   no variance and measured the harness, not the kernel. The probe is kept but
   a *refusal* (dtype error before any thread runs) is now recorded as
   `skipped_dtype_unsupported` in `battery_diagnostics` instead of a
   `fp16_crash` robustness flag. Only a kernel that actually runs in half and
   returns wrong numbers still earns a flag. See `_is_dtype_unsupported` in
   `kfault/battery.py`.
2. **p91 direct never completes.** ~~Timed out at 900s, then again at 1800s.~~
   **Fixed 2026-08-03.** Same failure recurred as batch 02 p42/p52. Raising the
   container timeout was never the fix: the battery makes ~20 forward calls and
   `benchmark()` another 70, so 1800s is really a ~20s per-forward budget, and
   blowing it returned *no row at all*. Stage 2a now times one **warm** forward
   (the first call pays lazy module load, so it is not a fair sample) and, past
   `FORWARD_TIME_LIMIT_S` (default 5.0s, env `KFAULT_FORWARD_LIMIT_S`), returns
   `runtime` / stage `forward_time_limit` immediately. Every kernel now yields a
   labelled row. The measured time is kept in `extra["forward_seconds"]`.
   *Assumption to revisit:* these are filed under `runtime` rather than a new
   `timeout` category, so the headline table keeps 5 categories. The stage name
   is distinct, so they can be re-bucketed in analysis without a re-run.
3. **compute-sanitizer is `unavailable` on Modal** (confirmed on both T4 and
   L4). Stage 2b contributes nothing on this platform.
4. **`classification_config["gpu"]` is always `T4`, whatever hardware ran.**
   Found 2026-08-03, OPEN. Of 142 level-2 records, 72 really ran on a Tesla T4
   and 69 on an NVIDIA L4 — but every one of them records `gpu: T4`.

   Cause: `KFAULT_GPU` is read from the local shell. The
   `@app.function(gpu=GPU)` decorator is evaluated locally, so the *hardware*
   is correct, but `classification_config` is built inside the container, where
   Modal does not forward local env vars, so `GPU` falls back to its `"T4"`
   default. The image sets no `.env()`.

   Consequences:
   - README's claim that T4 and L4 runs "land in different cache entries; they
     never overwrite each other" is **false** — they share one entry. The
     2026-08-03 `FORCE=1` re-runs therefore overwrote the original batch 01/02
     records rather than sitting beside them.
   - `analyze.py --gpu L4` matches nothing; `--gpu T4` returns a mix of both
     devices. Any GPU-stratified result is currently invalid.

   Not lost: `runtime_environment.gpu_name` records the true device on every
   record, so the data is recoverable without re-running anything. Analysis
   should filter on that field, not on `classification_config["gpu"]`.

   Fix (not yet applied, changes the cache key): add
   `.env({"KFAULT_GPU": GPU})` to the image in `pipeline.py`.

## L2 p91 direct: the hang, and what it cost (2026-08-03)

The one kernel that hung. Worth keeping because it sets the cost of a hang and
justifies the repair-feedback change.

| turn | category | stage | secs |
|------|----------|-------|------|
| 0 | runtime | container_timeout | 903 |
| 1 | runtime | container_timeout | 936 |
| 2 | runtime | container_timeout | 952 |
| 3 | functional | standard_output_comparison | 103 |

Total 49.7 min for one problem-arm. It had been a data hole in both earlier
runs; the `FunctionTimeoutError` handler turned it into real labels.

Two lessons:

1. The model recovered on its own by turn 3 — but only after three timeouts,
   because the feedback it received was the harness's own message
   (`Task's current input ... hit its timeout of 900s`), which says nothing
   about the kernel. `build_feedback` now writes a real description for
   `container_timeout` and `forward_time_limit` instead. 47 of those 50 minutes
   were spent on feedback the model could not act on.
2. A hang costs up to 4 x `CLASSIFIER_TIMEOUT_S` (~60 min) because `runtime` is
   repairable. That is acceptable while hangs stay rare (1 in 142 records). If
   batches 03-10 show more, reconsider making `container_timeout` terminal.

## Cache impact of the 2026-08-03 fixes

**None. No re-run needed.** `CLASSIFIER_CACHE_VERSION` stays at 4 and the
default forward budget is omitted from the classification cache key, so every
existing record stays valid. Reasoning:

- The fp16 fix cannot change a category. `classify` decides the category from
  `core` alone; `robust` only ever reached `extra["robustness_flags"]` and the
  detail string, and no analysis code reads that field. Cached records still
  carry the old `fp16_crash` text, but it never entered a number.
- The forward-time limit can only fire on a kernel slower than the budget. Any
  kernel with a cached record already completed ~90 forwards inside 1800s, and
  its per-forward cost is readable offline from
  `extra["timing"]["kernel"]["median_ms"]` — no GPU required. The kernels that
  actually blew the timeout (01 p91, 02 p42/p52) have no record to invalidate
  and need a run either way.

Only if `KFAULT_FORWARD_LIMIT_S` is set to a non-default value does
`forward_time_limit` enter the cache key, splitting those runs into their own
entries.

**One consistency check worth running** before writing up batches 01-02
alongside 03-10: list any cached record with kernel `median_ms > 5000`. Those
are the only ones a fresh run would relabel `runtime`, and they can be re-run
selectively with `--force`.

