# kfault — architecture

How the system is built and why. This describes what exists, not what was
planned; the plan is `papers/100526858-proposal.pdf`, and the results are in
`experiments/level2-repair/`.

---

## 1. What the system measures

KernelBench asks a model to replace a PyTorch module with a custom CUDA kernel,
then scores the result pass or fail against `allclose` on a handful of random
inputs. That check is weaker than it looks: a kernel can produce the right
numbers on the standard input and the wrong ones on a transposed tensor, an
extreme value, or a second call.

kfault classifies every generated kernel into one of five categories, and adds
a **repair loop** so the same kernel can be measured before and after automated
correction.

```
syntactic   does not build
runtime     crashes, trips the sanitizer, or is too slow to evaluate
functional  runs, wrong answer on the standard input
silent      passes the standard check, wrong under the stress battery
pass        correct on the standard input and survives the battery
```

**First matching stage wins.** `silent` is the category of interest: those
kernels are scored as successes by KernelBench.

---

## 2. Experimental design

**Two arms, identical weights.** `reasoning_effort: high` (think) versus `none`
(direct), same model, same prompt, same temperature 0. The only difference is
the reasoning switch, so any difference in outcome is attributable to it.

Not every model honours the off switch. Those that do are recorded in
`kfault/config.py`; models that reason regardless were rejected, since their
`direct` arm would silently be a second `think` arm.

**Repair loop, 3 turns.** After a failure the model is shown its own code and
the actual error, and asked again. Only `syntactic`, `runtime`, `functional`
and `empty` are fed back.

**`silent` is deliberately withheld.** Showing the model the stress battery
would let it fit the held-out test, and the false-pass rate — the study's main
measurement — would become meaningless.

---

## 3. The classifier

`kfault/classifier.py`, staged so the first failure wins:

| stage | check | on failure |
|---|---|---|
| 1 | `compile()` then `exec()`; `load_inline` builds here | `syntactic` |
| 2a | one forward pass on the standard input, then `cuda.synchronize()` | `runtime` |
| 2a' | one **warm** forward under `FORWARD_TIME_LIMIT_S` | `runtime` |
| 2b | `compute-sanitizer` memcheck | `runtime` |
| 3 | output comparison against the reference | `functional` |
| 4 | the stress battery | `silent` |

**Weight synchronisation** mirrors KernelBench: `torch.manual_seed(seed)` before
`get_init_inputs()`, before constructing each model, and before every input
draw. Without it, problems with learnable parameters give the two models
different weights and every kernel looks wrong.

### Stage 2a' — the per-forward time limit

A classification makes roughly 90 forward calls (battery probes plus
benchmarking), so the container timeout is really a per-forward budget. A
kernel needing tens of seconds per call cannot finish and previously produced
*no row at all*.

The check times a **second, warm** call: the first pays lazy module loading and
CUDA context setup, so it is not a fair sample. Past the budget the kernel is
labelled `runtime` / `forward_time_limit` immediately.

The budget must be scaled with the hardware. A T4 is roughly 3.8x slower than
an L4, so a fixed limit fails kernels for being on a slower device rather than
for being wrong. `KFAULT_CLASSIFIER_TIMEOUT_S` has to rise with it — the config
refuses an inconsistent pair rather than producing data that looks fine and is
not.

### Stage 2b — unavailable on this platform

`compute-sanitizer` reports `Device not supported` on both Modal GPUs, so races
and non-crashing out-of-bounds reads are undetected. This is the largest gap in
the method and is stated in every results document.

---

## 4. The stress battery

`kfault/battery.py`. Each probe presents inputs the standard check never uses,
and compares against the *reference model on the same inputs* — so a probe can
only fail if the two disagree.

| probe | catches |
|---|---|
| `fresh_seed_1..3` | hardcoded or lucky outputs |
| `non_contig` | assuming dense row-major layout |
| `nondeterministic` | races, reused internal buffers |
| `zeros`, `small_magnitude`, `extreme` | numerical instability |
| `fp16` | dtype assumptions — a **robustness flag**, not a silent failure |

Two rules keep the headline number honest:

**A harness fault is never blamed on the kernel.** An OOM during the probes
returns `skipped_out_of_memory`, not a silent failure. On a 16 GB card this
inflated the silent count before it was fixed.

**Refusing a dtype is not failing it.** A kernel compiled against
`data_ptr<float>()` cannot accept half input; it raises before a thread runs.
That is the declared contract being enforced, so it is recorded as
`skipped_dtype_unsupported`. Before this fix, 100% of passing kernels carried an
`fp16_crash` flag — a probe with no variance measures the harness, not the
kernel. Note that models also write their own `TORCH_CHECK` guards, whose
messages never mention "half"; both forms are handled.

---

## 5. Code layout

```
kfault/          pure logic - no Modal, no GPU, no network
  classifier.py    the five stages
  battery.py       the stress probes
  comparison.py    output comparison, tree mapping, cloning
  benchmark.py     interleaved CUDA-event timing
  repair.py        repair feedback; which categories are repairable
  prompts.py       prompt construction, response parsing
  generation.py    model calls and the generation cache key
  analysis.py      chains, distributions, false-pass, cross-GPU joins
  sanitizer.py     compute-sanitizer driver and output parsing
  cache.py         stable keys, atomic writes
  config.py        every tunable, and the guards between them

pipeline.py      Modal app: generate -> classify -> repair
validate.py      known-answer validation of the classifier
analyze.py       tables from the cache
export.py        cache -> data/*.csv + kernel sources
figures.py       data/*.csv -> docs/figures/paper/
```

**The split is the point.** Everything in `kfault/` is pure, so the classifier,
battery, repair feedback and analysis are all testable without a GPU or a
network call — 86 tests run in under a second. Only the root scripts touch
Modal.

`pipeline.py` is the exception and does too much: image definition, cache
lookup, attempt orchestration and CLI in one file. It is the one module that
does not respect the separation the rest of the codebase keeps.

---

## 6. Caching

Two independent keys, which is what makes re-labelling cheap:

- **Generation** — model, reasoning effort, problem, sample, temperature,
  prompt hash. GPU is *not* part of it, so the same kernel can be classified on
  different hardware without regenerating.
- **Classification** — generation artefact, seed, sanitizer flag, GPU, CUDA
  image, pinned packages, schema version, and the forward budget when it is not
  the default.

`--force` re-classifies while reusing the stored model output; `--regen`
requests genuinely new generations.

**A field that decides an outcome must be in the key.** The forward budget
changes `runtime` versus a full battery run, so two budgets must not share an
entry. It is omitted when default so existing entries stay valid.

**Every `KFAULT_*` setting must be forwarded into the image.** Modal does not
pass local environment variables to containers. `classification_config` is built
container-side, so a value set only in the local shell is silently ignored — the
cause of 70 records recording the wrong GPU. `runtime_environment.gpu_name`,
written on the device itself, is the ground truth that analysis reads.

---

## 7. Validation

`validate.py` runs 11 hand-written kernels of known category through the real
classifier: correct, correct-with-weights, mutating, structured output, nested
input, custom stress inputs, plus one each of `syntactic`, `runtime`,
`functional`, `silent`, and a correct-but-too-slow kernel.

It asserts the category **and, where it matters, the failing stage** — `runtime`
is reachable several ways, so a slow kernel could otherwise pass for the wrong
reason.

Run it before trusting any collection.

---

## 8. Reproducing

```bash
uv run modal run validate.py                       # 11/11 before trusting labels
./experiments/level2-repair/run_batch.sh 1         # one batch
uv run modal run analyze.py --level 2 --gpu L4     # tables
uv run modal run export.py --level 2 --with-code   # cache -> data/
uv run python figures.py                           # data/ -> figures
```

Model and device are environment-selected:

```bash
KFAULT_MODEL=... KFAULT_GPU=T4 \
KFAULT_FORWARD_LIMIT_S=20 KFAULT_CLASSIFIER_TIMEOUT_S=2700 \
  ./experiments/level2-repair/run_batch.sh 1
```

No figure or table contains a hand-copied number: everything derives from
`data/attempts.csv` and `data/chains.csv`, which derive from the cache.

---

## 9. Known limitations

- **`compute-sanitizer` is unavailable** on both GPUs used. Races and silent
  out-of-bounds reads are undetectable.
- **One greedy sample per problem**, so results carry no confidence intervals.
- **85% of silent failures come from the `non_contig` probe.** KernelBench tests
  contiguous inputs only, so a kernel that assumes dense layout satisfies its
  stated contract while silently narrowing the module it replaces. Report the
  rate under both readings.
- **Level 2 only.**
- **`pipeline.py` cohesion**, as above.
