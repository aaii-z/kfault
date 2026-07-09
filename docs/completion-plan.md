# kfault — Completion Plan (as of 2026-07-09)

Roadmap from current state to (a) dissertation submission end of August 2026
and (b) a published conference/workshop paper or poster. Complements `plan.md`
(the original 16-week task list) — this is the *remaining* critical path.

## Where we are (~week 8 of 16)

**Done (Phases 1–2):** Modal harness; generation via OpenRouter with the
official KernelBench one-shot prompt; full 4-stage classifier in `pipeline.py`
(syntactic / runtime / functional / silent / pass) incl. the Stage-4 silent
battery; result caching in the `kfault-results` volume; T5 validation harness
(`validate.py`, 5/5 known-category kernels).

**Not done:** real data collection (only ~10-problem pilots), all analysis
(T9–T12), all writing.

**Blockers that gate the data:**

1. **Weight-sync gap** — `classify()` built `Model` / `ModelNew` without
   seeding or weight copy, so problems with learnable params get garbage
   `functional`/`silent` labels.
2. **Single sample per problem** — `temperature=0`, one generation per
   (problem, model): the RQ2 false-pass rate has no real denominator and
   R1-vs-V3 tests lack power.
3. **compute-sanitizer self-disables on Modal T4** — Stage 2b degrades
   silently; run diagnostics on L4/A100 or document the limitation.

## Step 1 — Fix the pipeline (week of Jul 9, ~2–3 days) — DONE 2026-07-09

- [x] Weight sync: mirror KernelBench — `torch.manual_seed(seed)` before
      `get_init_inputs()`, before constructing **each** model, and before every
      `get_inputs()` draw (incl. the battery's fresh draws and the fp16
      reference rebuild).
- [x] Extend `validate.py` with a learnable-parameter case (`nn.Linear` +
      correct CUDA kernel) that must classify `pass` — fails as `functional`
      without the seed fix, so the fix is itself validated. **6/6 validated
      2026-07-09.** (Bonus: this case exposed a second latent bug — the fp16
      battery check mutated `kernel_model` in place via `.half()` and the
      `.float()` restore both could be skipped on exception AND loses precision
      permanently; fixed with a `state_dict` snapshot + `finally` restore.)
- [x] Multi-sample generation: `sample` index in the cache key, sample 0 greedy
      (temp 0, comparable to published pass rates), samples 1..k−1 at
      `--temperature` (default 0.7); run with `--samples k`.
- [x] Re-classification without API spend: `--force` reuses the cached
      `raw_response` and only re-runs the classifier (`--regen` for a true
      re-generation) — so the post-fix rerun of existing pilots costs GPU
      minutes, not API budget.
- [x] Sanitizer / GPU: selectable via `KFAULT_GPU` env var (default T4).
      Remaining choice at collection time: run on L4 (sanitizer works) or keep
      T4 and state in the paper that Stage 2b covers only crash-visible
      runtime failures.

**First action of Step 2:** re-run the existing Level-2 pilot with `--force`
(free re-classification) — the pre-fix cached labels are untrustworthy.

## Step 2 — Full data collection (mid-Jul → end of Jul)

Scope decided 2026-07-09: **30 random Level-2 problems** (seed 42), 2 models,
**3 samples** (sample 0 greedy + 2 at temp 0.7) = 180 runs, est. $10-20.
Problem list: 1,4,5,12,14,15,18,26,28,29,30,32,36,54,55,58,65,70,72,76,78,
82,85,87,89,90,94,95,96,98.
**ON HOLD:** user may switch the LLM provider from OpenRouter to AWS Bedrock
before starting. CPU/GPU split is done and smoke-tested (GPU no longer idles
during generation).

- [ ] Run the 30-problem x 3-sample collection (command ready, awaiting go +
      provider decision).
- [ ] RQ4 bonus: re-run the battery under a second GPU type
      (`KFAULT_GPU=L4/A100`) — hardware-divergent correctness is a headline
      figure if it appears.

## Step 3 — Analysis (late Jul → mid-Aug)

- [ ] RQ1/RQ3: category × model cross-tabs; chi-square / Fisher on R1-vs-V3;
      specifically test whether the **silent share of standard-passers** is
      higher for R1 (the relocation claim).
- [ ] RQ2: false-pass rate **with confidence interval**, framed against
      KernelBench's reported pass rates.
- [ ] T9 root-cause labelling on the silent subset: LLM-labelled, but
      hand-label a ~30-kernel subset first and report agreement.

## Step 4 — Dissertation (now → end of Aug)

- [ ] Intro / lit review / methodology drafted **now, in parallel** with
      collection (methodology ≈ `explain.md` formalised).
- [ ] Results + discussion in the final two weeks — lead with relocation +
      the false-pass number. Submit end of August.

## Step 5 — Publication (Sept →)

Sequencing per `docs/venues.md`:

- [ ] **NeurIPS 2026 workshops** (Deep Learning for Code / ML for Systems /
      Reliable ML): accepted-workshop list lands mid-summer, contribution
      deadlines late Aug–Sept. **Watch CFPs from now** — condensing the
      dissertation into 4–8 pages may coincide with submission week.
- [ ] **EuroMLSys 2027** (deadline ~Jan–Feb 2027): the polished 6-page
      version; use autumn to add the o3 / GPT-4o secondary pair if budget
      allows (broadens the claim beyond one model family).
- [ ] **UK poster** (Turing / CDT symposium, autumn): cheap first outing;
      `present.md` is already ~80% of the poster content.

**Critical path:** Step 1 — every downstream number depends on the weight-sync
fix.
