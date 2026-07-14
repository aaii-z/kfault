# kfault — Project Plan

> Historical planning document: current implementation and reproduction
> details live in `README.md` and `docs/completion-plan.md`. Unchecked prototype
> items below do not describe the current code.

**Title:** A Systematic Taxonomy of Correctness Failures in LLM-Generated GPU Kernels
**Author:** Ali Ansari (100526858)
**Timeline:** ~16 weeks (mid-May → end of August 2026)
**Compute:** **Modal.com** (serverless cloud GPUs) — everything runs in a Modal
container, GPU type selected per-run via `gpu="..."`. No local/HPC setup.

## Headline thesis (the sharp, publishable claim)

> Reasoning models do not fix correctness — they **relocate** failures from
> *visible* (syntactic / runtime) to *silent* (passes a basic check, wrong under
> stress). Reported KernelBench pass rates therefore **overcount** correctness.

The descriptive 4-stage taxonomy (syntactic → runtime → functional → silent) is
the spine; the **silent-failure stage is the star**. Everything below serves that.

## Core research questions

- **RQ1** — How do failure types distribute across reasoning vs non-reasoning models?
- **RQ2** — What fraction of "correct" kernels (pass the standard `allclose`) are
  actually **silent failures** under stress? (the false-pass / validity number)
- **RQ3** — Do reasoning models show a *higher share of silent failures* even as they
  reduce visible failures? (the relocation hypothesis)
- **RQ4 (easy bonus on Modal)** — Are some kernels correct on one GPU but silently
  wrong on another? (just re-run with `gpu="L4"`, `"A10G"`, etc.)

---

## Scope decisions (lock these to stay feasible)

- **Models:** primary pair DeepSeek **R1 (reasoning)** vs **V3 (non-reasoning)** —
  shared base = cleanest contrast. Secondary pair (o3 / GPT-4o) only if budget allows.
- **Tasks:** KernelBench **Level 2 + Level 3** (highest failure rates). Level 1 excluded.
- **Silent-failure battery is FIXED and small** (see T4) — do not let it sprawl.
- **Dropped / bonus only:** reasoning-trace analysis = dropped. Multi-GPU divergence
  (RQ4) = cheap bonus on Modal, not a dependency.

---

## Task list

### Phase 1 — Modal harness (Weeks 1–3)

- [ ] **T1. Modal environment.** Confirm the Modal image in `demo.py` (CUDA 12.4 +
      torch + ninja) compiles & runs inline CUDA on a T4. Add `compute-sanitizer`
      (ships in the CUDA devel image) and verify it runs inside the container.
      *Deliverable: working Modal app, 2–3 reference kernels pass.*
- [ ] **T2. Generation pipeline.** Generalise `demo.py`: parametrise model + provider
      (DeepSeek R1/V3 via OpenRouter), Level 2/3 task selection, loop over all tasks,
      store raw response + extracted kernel + metadata as one JSON per run (in `results/`
      or a Modal volume). *Deliverable: automated generation across model × task.*

### Phase 2 — The diagnostic pipeline (Weeks 3–5) ← core engineering

- [ ] **T3. Stages 1–3 (visible failures)**, all inside the Modal function:
      - Stage 1: compile with `nvcc` → **syntactic failure** on fail (log error).
      - Stage 2: run under `compute-sanitizer` → **runtime failure** (races, illegal mem).
      - Stage 3: `allclose` vs PyTorch on standard input → **functional failure**.
      *Deliverable: kernel → {syntactic, runtime, functional, pass} labeller.*
- [ ] **T4. Stage 4 — silent-failure detector (THE NOVELTY).** For kernels that pass
      Stage 3, re-test with a **fixed battery**:
      - input perturbations: non-contiguous, fp16/bf16, extreme values (large / NaN / inf),
        edge shapes (size 1, prime, non-power-of-2);
      - 2–3 **metamorphic checks**: linear ops scale with input scaling; reductions are
        permutation-invariant; (optional) idempotence where applicable.
      Label **silent failure** if any check breaks. *Deliverable: pass → silent/pass label.*
- [ ] **T5. Pipeline validation.** Hand-craft kernels with known failure types (one per
      category) and confirm correct classification. *Deliverable: validated classifier.*

### Phase 3 — Data collection (Weeks 5–8)

- [ ] **T6. Run primary pair.** All Level 2 + 3 tasks × {R1, V3} through the full
      pipeline on Modal. Record category + diagnostics per kernel. *Deliverable: dataset.*
- [ ] **T7. (Conditional) Secondary pair.** o3 / GPT-4o if API budget + time allow.
- [ ] **T8. (Bonus) Multi-GPU pass.** Re-run the silent-failure battery with a different
      `gpu=` value (e.g. L4, A10G, A100) to detect **hardware-divergent correctness** (RQ4).
      Trivial on Modal — one parameter change.

### Phase 4 — Analysis (Weeks 7–12)

- [ ] **T9. Root-cause labelling.** On the **silent-failure subset**, assign root cause
      (thread indexing / missing sync / fp precision / shape assumption). LLM-based
      classifier *validated against a hand-labelled subset*.
- [ ] **T10. Core cross-tabs + stats.** Failure category × model type × task level;
      significance tests on R1 vs V3 distributions (RQ1, RQ3).
- [ ] **T11. Validity number.** False-pass rate (RQ2): % of Stage-3-passing kernels
      caught by Stage 4. Frame against KernelBench / CUDA Agent reported pass rates.
- [ ] **T12. Per-kernel-type profiles.** GEMM / reduction / softmax / layernorm profiles.

### Phase 5 — Writing & submission (Weeks 8–16)

- [ ] **T13. Draft intro / lit review / methodology** (in parallel with Phase 3–4).
- [ ] **T14. Results + discussion** — lead with relocation finding + validity number.
- [ ] **T15. Presentation + proofread + submit.**

---

## Success criteria

- Validated 4-category taxonomy with **statistically significant** R1-vs-V3 differences (RQ1/RQ3).
- A concrete **false-pass rate** showing reported correctness is overcounted (RQ2).
- Actionable finding: "reasoning models avoid visible errors but introduce a specific
  class of silent failures" — novel and practically significant.

## Risks

- **API cost** → prioritise cheap R1/V3; secondary pair optional.
- **Modal credits / GPU quota** → keep runs on cheap T4 by default; reserve bigger GPUs
  only for the RQ4 multi-GPU bonus; cache results so reruns don't re-spend.
- **Too few failures** → mitigated by Level 2/3 focus.
- **Silent-failure scope creep** → mitigated by the FIXED battery in T4.
- **Novelty pre-emption** (KernelBench Fig 3, robust-kbench partially taxonomise) →
  mitigated by leading with the relocation hypothesis + validity number, not a bare taxonomy.

## Immediate next step

Extend `demo.py` into a minimal **silent-failure probe** (T4 prototype): take a kernel
that passes standard `allclose`, re-run it through the fixed battery + metamorphic checks
inside the same Modal function, emit a `visible / functional / silent / pass` label.
Run on a few Level 2 problems → first "% false pass" datapoint.
