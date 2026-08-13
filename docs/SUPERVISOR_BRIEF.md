# Supervisor brief — status, findings, novelty

**Submission: Thursday 20 August 2026 — 13 days.** Report 70%, presentation 20%.

This is what exists, what it shows, what is genuinely new, and what is missing
against the proposal.

---

## 0. RAISE FIRST: the AI-usage policy

The module brief contains two requirements that need reconciling, and only the
supervisor or module organiser can do it:

- A **"Statement of AI usage"** is a *required deliverable* (1 page, max 200
  words), which implies disclosure is expected.
- The plagiarism section states: *"The use of chatGPT (or other Artificial
  Intelligence applications/websites) to produce any part of your submissions
  is strictly prohibited. This includes but is not limited to: **coding,
  testing/debugging or documenting code**; writing or paraphrasing passages of
  text; or the generation of bibliographies."*

This project used an AI assistant substantially for pipeline code, debugging,
analysis scripts and documentation. That falls squarely inside the prohibited
list as written, while the required statement page suggests some disclosed use
is anticipated.

**Ask directly in this meeting**: what use is permitted, and what exactly should
the Statement of AI usage say? The brief itself says to *"ask your module
organiser if you are unsure about anything"* — this is that situation, and it
is much better resolved 13 days out than after submission.

---

## 1. Experiments completed

| model | Level 2, L4 | Level 2, T4 |
|---|---|---|
| `zai.glm-5` | 100 problems | 100 problems |
| `deepseek.v3-v1:0` | 100 problems | 100 problems |
| `qwen.qwen3-235b-a22b-2507-v1:0` | 100 problems | 100 problems |

Each problem runs **two arms** (`reasoning_effort: high` vs `none`, identical
weights) and **up to 4 turns** (one-shot + 3 repair turns).

**2,926 classified kernel attempts. 1,196 repair chains. No missing cells.**

Dataset exported as `data/attempts.csv`, `data/chains.csv`, and 2,829 kernel
source files — every figure regenerates from those with `uv run python figures.py`.

### Deviation from the proposal, worth flagging early

The proposal's primary pair was **DeepSeek R1 vs V3** — two different
checkpoints. What was actually run toggles `reasoning_effort` on **identical
weights**. Any difference is therefore caused by reasoning and nothing else.
This is a *stronger* controlled comparison than proposed, not a substitution of
convenience, and should be presented that way.

Hardware is Modal cloud (L4 Ada + T4 Turing), not the CMP lab PCs. Same image,
pinned CUDA 12.4 and torch 2.6.0 across every run.

---

## 2. What the data shows

### Finding A — reasoning helps a lot (confirms known work)

| pass rate /100, after repair | think | direct |
|---|---|---|
| GLM-5 | 75 | 47 |
| DeepSeek V3.2 | 76 | 6 |
| Qwen3-235B | 83 | 8 |

Two of three models essentially cannot write correct Level-2 CUDA without
reasoning. **This is confirmation, not a finding** — the literature already
reports it (proposal §3.3).

### Finding B — repair relocates failures  ← THE CONTRIBUTION

| | one-shot | after 3 repair turns |
|---|---|---|
| GLM-5 false-pass | 14% | **20%** |
| DeepSeek false-pass | 16% | **19%** |
| Qwen false-pass | 7% | **10%** |

Repair raises the pass rate **and** the share of passes that are secretly
wrong. Every model moves the same direction.

Mechanism, counted per chain:

| | GLM-5 | DeepSeek | Qwen |
|---|---|---|---|
| repaired to pass | 25 | 39 | 22 |
| **visible failure became SILENT** | **16** | **17** | **7** |
| still visibly failing | 40 | 90 | 86 |

**40 chains total** turned a loud failure into a quiet one. A standard
KernelBench harness scores every one of them as a repair success.

Typical shape (DeepSeek, L2 p22 direct):

```
turn 0: runtime   name 'math' is not defined     <- trivial, loud
turn 1: silent    non_contig                     <- "fixed", now silently wrong
```

This is only measurable because stage 4 is **withheld** from the repair
feedback. Feeding silent failures back would let the model fit the held-out
test and erase the effect.

### Finding C — cross-GPU divergence is rare and model-specific

| genuine divergences | kernels compared |
|---|---|
| GLM-5: **0** | 367 |
| DeepSeek: **0** | 496 |
| Qwen: **3** | 441 |

No kernel passed on one GPU and silently failed on the other. Qwen's three are
one mechanism: an out-of-bounds read that lands on valid memory on Ada and
faults on Turing — *louder* on the older GPU.

Scope this as the proposal did: a cheap bonus (§4.5), not a contribution.

### Finding D — passing kernels are barely faster

Median speedup vs PyTorch: **1.07x–1.19x**. Qwen is the most *correct* and the
*slowest*.

One task, `80_Gemm_Max_Subtract_GELU`, is **degenerate**: its output is
identically zero regardless of input. GLM-5's reasoning arm proved this in a
comment and returned a zero tensor (2670x "speedup"); its direct arm failed all
four turns on both GPUs. Excluded from statistics, but worth reporting — it is
a flaw in KernelBench that affects anyone citing its speedups.

---

## 3. Novelty — honest assessment

| claim | novel? |
|---|---|
| **Repair improves benchmark scores while reducing trustworthiness** | **Yes.** Nothing in the reviewed literature reports it. Counterintuitive, practically important (every agentic CUDA system uses repair loops), replicated across 3 models and 2 architectures. |
| **~10–20% of KernelBench passes are false passes** | **Probably.** Depends on prior work — see risk below. |
| Reasoning reduces syntactic/runtime errors | **No.** Established (Ouyang 2025; Zhu 2026). Report as confirmation. |
| Cross-GPU divergence | **Thin.** 3 cases in ~1,300 kernels, one model. A paragraph. |
| KernelBench task 80 is degenerate | Minor but quotable. |

### The one literature risk to resolve

`robust-kbench` (Lange et al. 2025) — *"Towards robust agentic CUDA kernel
benchmarking, **verification**, and optimization"*. If it already stress-tests
generated kernels, the novelty of the false-pass *idea* shrinks to the
*measurement* and the *cross-model comparison*. **Read it before the write-up
commits to a novelty claim.**

### Suggested framing

Not "a taxonomy" — taxonomies are hard to publish. Lead with:

> **Iterative repair improves CUDA benchmark scores while making the resulting
> kernels less trustworthy, and standard benchmarks cannot see the difference.**

That has a victim (anyone running repair loops), evidence (40 chains, 3 models,
2 GPUs), and a mechanism visible in the source code.

---

## 4. Gaps against the proposal — state these before an examiner does

| gap | severity |
|---|---|
| **No statistical tests.** §4.7 makes "statistically meaningful differences" the success criterion. One greedy sample per problem = no confidence intervals. 20% vs 10% is currently unfalsifiable. | **High** |
| **compute-sanitizer never ran.** §4.3 Stage 2. Reports `Device not supported` on both Modal GPUs. Races and non-crashing out-of-bounds reads are undetectable — awkward for a study about *silent* failures. | **High** |
| **Level 3 not run.** §4.2 includes it. Every claim reads "on Level-2 problems". | Medium |

### One methodological dependency to disclose

**73 of 86 silent failures (85%) are caught by the non-contiguous-input probe
alone.** KernelBench tests 5 random *contiguous* inputs, so it never probes
this.

The probe is discriminating rather than a blanket penalty — 266 passing kernels
also use raw pointers without a `.contiguous()` guard and still pass it, and 26
passing kernels call `.contiguous()` explicitly, so the fix is one line and some
models already apply it.

But a reviewer can fairly say these kernels satisfy KernelBench's stated
contract. **Recommend reporting the rate two ways:**

| definition | silent | GLM-5 think false-pass |
|---|---|---|
| strict (within KernelBench's contract) | 13 | ~4% |
| extended (drop-in-replacement semantics) | 86 | 20% |

Both numbers make the paper harder to attack, not easier.

---

## 4b. What the marking scheme rewards

Report is 70%, weighted:

| element | weight | your position |
|---|---|---|
| Preparation and design | **30%** | Strong — pipeline design, controlled arms, cache/provenance, stratified batches |
| Implementation and evaluation | **30%** | Strong — 2,926 kernels, 3 models, 2 GPUs, exported dataset |
| Introduction | 10% | To write |
| Literature review | 10% | Largely done in the proposal; needs `robust-kbench` |
| Conclusion | 10% | To write |
| Quality of written report | 10% | Structure, numbered figures, correct references |

**60% of the report mark sits on the experimental work, which is finished.**

Note that "Implementation and evaluation" explicitly includes *"Comparison
against other studies and/or benchmarks, **highlighting limitations and scope
for improvement**"*. The gaps in §4 are therefore **marks to be earned, not
weaknesses to conceal** — the sanitizer gap, the non-contiguous dependency and
the missing statistical test should each be stated plainly and analysed.

### The binding constraint is length, not content

**Core material, including table of contents and appendices: max 40 pages.**

With 2,926 attempts, 3 models, 2 GPUs and 5 figures, the risk is now
over-stuffing. Suggested budget:

| section | pages |
|---|---|
| Introduction | 3 |
| Literature review | 5 |
| Design (pipeline, taxonomy, battery, arms) | 8 |
| Implementation (Modal harness, caching, validation) | 6 |
| Evaluation (findings A–D, figures 1–5) | 12 |
| Limitations and threats to validity | 3 |
| Conclusion and further work | 3 |

Bibliography is unlimited and sits outside the 40 pages.

## 4c. Presentation (20%, marked separately)

20 minutes including questions, then a viva of up to 20 minutes. Slides are
marked 40% on *"use of diagrams, illustrations, graphs as opposed to text"* —
the five figures in `docs/figures/paper/` are directly reusable, and the
p80 degenerate-task kernel is a strong live demo: readable source, a clear
argument in the comments, and a result the audience will not expect.

## 5. Recommendation for the remaining two weeks

**Write. Do not run more experiments.** 13 days, and 60% of the report mark is
already earned by work that is finished.

There is already more than enough for the dissertation, and the binding
constraint is now the write-up and the 40-page limit, not the data. The one exception is
analysis-only work with no new runs:

1. **Dual strict/extended reporting** (hours, no compute) — closes the biggest
   methodological hole.
2. **Read `robust-kbench`** (hours) — decides how the novelty claim is worded.

Only if the supervisor judges §4.7 to be blocking:

3. **Multi-sample on one model**, `--samples 5 --temperature 0.7` — gives real
   confidence intervals. Roughly 1-2 days including compute.

**Level 3 is not worth starting** at this point; it would consume the writing
window for a generality claim that a stated limitation covers adequately.

---

## 6. Decisions needed from this meeting

0. **What does the Statement of AI usage need to say, and what use was
   permitted?** (See §0 — resolve this first.)
1. Is **repair relocation** the headline claim of the dissertation? (Recommended: yes.)
2. Accept **Level 2 only** with a stated limitation, or spend part of the window on Level 3?
3. Is the missing **statistical test** (§4.7) blocking? If yes, multi-sample starts today.
4. How should the **compute-sanitizer gap** be presented — as a platform
   limitation, or does it need fixing on other hardware before submission?
5. Report false-pass under **both** the strict and extended definitions?
