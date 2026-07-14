# kfault implementation and research completion guide

Updated: 14 July 2026

This is the current operational guide for the repository. It explains the
hardening work already completed, how to reproduce the pipeline, how to collect
the final dataset, and how to turn that dataset into a defensible dissertation
and paper. `docs/RESULT.md` is a historical first collection and must not be
mixed with results produced by the current schema-v3 classifier without
reclassification.

## 1. What the system does

For each KernelBench problem, the pipeline:

1. Loads the pinned PyTorch reference problem.
2. Sends the official-style one-shot prompt to DeepSeek V3.2 through AWS
   Bedrock.
3. Uses the same model in two arms: `think` sets `reasoning_effort=high`, while
   `direct` sets it to `none`.
4. Extracts Python containing an embedded CUDA C++ extension and a Python
   `ModelNew` wrapper, matching the KernelBench CUDA backend format.
5. Classifies the result as `syntactic`, `runtime`, `functional`, `silent`, or
   `pass`.
6. Benchmarks only kernels that pass both the standard comparison and the full
   core stress battery.
7. Saves generation and classification artifacts independently in the
   `kfault-results` Modal Volume.

The Python output is intentional: CUDA C++ performs the computation, while
Python compiles it with `load_inline`, exposes it to PyTorch, and defines the
`ModelNew` interface required by KernelBench.

## 2. Hardening work completed

### Correctness and comparison

- Reference and generated models receive independent recursive tensor clones.
  An in-place kernel can no longer alter the reference inputs.
- Initialization inputs are also cloned for each model.
- Nested tuple, list, dictionary, named-tuple, scalar, integer, floating, and
  complex outputs are compared recursively with useful mismatch paths.
- Nested input containers are moved to CUDA recursively.
- Correctness and the complete stress battery run before benchmarking.
- Incorrect or silent kernels are no longer given misleading speed figures.
- The first determinism output is cloned before the second invocation, so a
  reused internal output buffer cannot hide nondeterminism.

Relevant files: `kfault/comparison.py`, `kfault/classifier.py`.

### Stress battery

Core FP32 probes now include:

- Three fresh seeded draws.
- Non-contiguous inputs.
- Repeated-input determinism.
- Zero-valued floating inputs.
- Small-magnitude floating inputs.
- Inputs scaled by 100.
- Optional problem-defined `get_stress_inputs()` cases.

Only floating tensors are converted to FP16 or scaled. Integer indices, masks,
and labels retain their original dtype and values. A reference failure on a
synthetic probe is recorded as a skipped/invalid probe instead of blaming the
generated kernel. FP16 remains a robustness flag rather than an FP32 failure.

Problem-specific shape testing uses this optional reference-problem function:

```python
def get_stress_inputs():
    return {
        "empty_batch": [torch.randn(0, 16)],
        "odd_shape": [torch.randn(7, 13)],
        "single_element": [torch.randn(1, 1)],
    }
```

This is the safe way to add empty, odd, broadcasting, and boundary shapes;
their validity cannot be inferred for every architecture by a generic tool.

Relevant file: `kfault/battery.py`.

### Sanitizer and failure diagnostics

Compute Sanitizer now reports one explicit status:

- `clean`
- `violation`
- `kernel_error`
- `timeout`
- `unavailable`
- `tool_error`

Exit codes, error counts, and bounded log excerpts are retained. Only confirmed
violations and kernel errors are attributed to the generated kernel. Tool
unavailability or timeout becomes a visible validation warning; it is not
silently converted into a memory-clean result.

Classifier failures also include a structured stage, error type, and message.
Stages distinguish Python parsing, generated-code execution/build, model
lookup, model initialization, standard forward execution, sanitizer errors,
standard output comparison, and the stress battery.

Relevant files: `kfault/sanitizer.py`, `kfault/classifier.py`.

### Benchmarking

The benchmark now:

- Runs only after correctness validation.
- Uses fresh cloned inputs for every call.
- Interleaves reference and kernel measurements in deterministic randomized
  order to reduce order/thermal bias.
- Uses 5 warm-ups and 30 measured iterations.
- Reports median, p10, p90, and population standard deviation.
- Retains `ref_ms`, `kernel_ms`, and `speedup` fields for compatibility.

Relevant file: `kfault/benchmark.py`.

### Security and reproducibility

Generated Python and native CUDA code execute in a single-use Modal worker
with:

- All outbound networking blocked.
- Access to other Modal resources restricted.
- No Bedrock secret attached.
- CPU, memory, and lifetime limits.
- Minimal included source.

The CUDA container is pinned by digest. Torch, Ninja, OpenAI, Datasets, and
Modal versions are pinned, as is the exact KernelBench dataset revision. GPU,
Torch, CUDA, and compute-capability metadata are saved with results.

This limits blast radius but does not make arbitrary native C++ harmless. The
research assumes optimization attempts, not deliberately hostile extensions.

Relevant files: `pipeline.py`, `pyproject.toml`, `uv.lock`.

### Caching and API-cost protection

Generation and classification use separate cache keys:

- Generation key: model, reasoning effort, problem, sample, temperature,
  dataset revision, prompt hash, and client version.
- Classification key: exact generation artifact, seed, sanitizer setting,
  GPU, CUDA image, packages, and classifier schema.

Therefore, changing GPU, seed, or sanitizer settings reuses the already-paid
generation. `--force` reclassifies; `--regen` deliberately pays for a new model
response. JSON files are written atomically. Matching legacy responses are
imported only when model, arm, sample, temperature, and exact prompt agree.

Relevant files: `kfault/cache.py`, `pipeline.py`.

### Parsing, validation, and project cleanup

- Truncated `<reasoning>` output is treated as empty.
- Completed reasoning blocks are removed before code-fence parsing.
- Fence tags such as `python3`, `py`, and `c++` are handled.
- The preferred fenced block is the one defining `ModelNew`.
- The GPU validation suite now contains 10 category/regression cases,
  including weights, mutating inputs, structured outputs, integer indices,
  custom shapes, and nested inputs.
- There are 13 fast local tests for parsing, cache keys, atomic cache writes,
  and sanitizer parsing.
- `demo.py` is now only a compatibility entrypoint; `pipeline.py` is the
  maintained implementation.
- The project name, environment example, Just commands, lockfile, README, and
  historical-document warnings were corrected.

Relevant files: `kfault/prompts.py`, `validate.py`, `tests/`, `demo.py`,
`justfile`, `README.md`.

## 3. Clean setup

Requirements:

- Python 3.12 or newer.
- `uv`.
- `just` (optional; every command is also shown without it).
- A Modal account and authenticated Modal CLI.
- An AWS Bedrock bearer token authorized for the configured model.

From the repository root:

```bash
uv sync --locked
uv run modal token new
uv run modal secret create bedrock \
  AWS_BEARER_TOKEN_BEDROCK=your_real_token
```

Never commit the real token. `env.example` documents the required variable;
the actual value belongs in the Modal secret named `bedrock`.

Confirm local integrity:

```bash
just test
just lint
```

Equivalent commands:

```bash
uv run python -m unittest discover -s tests -v
uvx ruff==0.12.4 check .
uvx ruff==0.12.4 format --check .
uv lock --check
```

Expected local result: 13 tests pass and all lint/format/lock checks pass.

## 4. GPU validation before data collection

Run the full known-category suite before trusting research labels:

```bash
KFAULT_GPU=L4 uv run modal run validate.py
```

Expected result:

```text
10/10 cases classified correctly
classifier validated
```

L4 or A100 is recommended because Compute Sanitizer may be unavailable on T4.
If budget requires T4, run the suite there too, but record the sanitizer status
and do not describe `unavailable` as a clean memory check.

If validation is not 10/10, stop. Do not collect or analyze the main dataset
until the failure is understood and the full suite passes again.

## 5. Smoke test the complete pipeline

Start with one problem and one sample per arm:

```bash
KFAULT_GPU=L4 uv run modal run pipeline.py \
  --level 2 \
  --problems 1 \
  --samples 1
```

Check that:

- Both `think` and `direct` complete.
- A category and detail are printed for each.
- Sanitizer status is visible.
- A `pass` result contains timing statistics.
- The `kfault-results` Volume contains `generations/` and `classifications/`.

Inspect the Volume:

```bash
uv run modal volume ls kfault-results /
```

## 6. Cache and rerun commands

Normal rerun; returns an exact cached classification when available:

```bash
KFAULT_GPU=L4 uv run modal run pipeline.py --level 2 --problems 1
```

Reclassify the same generation after classifier changes, without another LLM
request:

```bash
KFAULT_GPU=L4 uv run modal run pipeline.py \
  --level 2 --problems 1 --force
```

Request a genuinely new generation and then classify it:

```bash
KFAULT_GPU=L4 uv run modal run pipeline.py \
  --level 2 --problems 1 --force --regen
```

`--regen` incurs model cost. Use it only intentionally. Changing
`KFAULT_GPU`, seed, or sanitizer configuration should create another
classification while preserving and reusing the generation.

## 7. Final data-collection design

The planned primary collection is:

- KernelBench Level 2.
- 30 preselected problems.
- Two reasoning arms of the same model.
- Three samples per problem/arm.
- Sample 0 at temperature 0.
- Samples 1 and 2 at temperature 0.7.
- Total: 30 × 2 × 3 = 180 generated kernels.

Problem IDs:

```text
1,4,5,12,14,15,18,26,28,29,30,32,36,54,55,58,65,70,72,76,
78,82,85,87,89,90,94,95,96,98
```

Run in batches rather than launching all work blindly. First collect five
problems:

```bash
KFAULT_GPU=L4 uv run modal run pipeline.py \
  --level 2 \
  --problems "1,4,5,12,14" \
  --samples 3 \
  --temperature 0.7
```

Review failures, sanitizer coverage, cost, and runtime. If healthy, run the
remaining fixed list:

```bash
KFAULT_GPU=L4 uv run modal run pipeline.py \
  --level 2 \
  --problems "15,18,26,28,29,30,32,36,54,55,58,65,70,72,76,78,82,85,87,89,90,94,95,96,98" \
  --samples 3 \
  --temperature 0.7
```

Do not change the problem list after seeing outcomes. Record the git commit,
date, GPU, command, model ID, Bedrock region, and any interrupted/retried jobs.
The JSON metadata records most execution settings, but the research log should
also preserve collection decisions and anomalies.

## 8. Optional second-GPU study

RQ4 should reuse generations and change only classification hardware:

```bash
KFAULT_GPU=A100 uv run modal run pipeline.py \
  --level 2 \
  --problems "1,4,5,12,14,15,18,26,28,29,30,32,36,54,55,58,65,70,72,76,78,82,85,87,89,90,94,95,96,98" \
  --samples 3 \
  --temperature 0.7
```

Do not add `--regen`. The split cache should reuse the exact Bedrock responses
and create A100-specific classifications. Confirm matching
`generation_artifact_key` values before comparing GPUs.

## 9. Export and preserve results

List and download the complete Modal Volume:

```bash
uv run modal volume ls kfault-results /
uv run modal volume get kfault-results / exported-results
```

The downloaded structure should include:

```text
exported-results/
├── generations/
└── classifications/
```

Make a read-only dated backup before analysis. Preserve raw JSON; analysis
should write new tables rather than modifying source artifacts. Record a
SHA-256 manifest if results will support a publication.

Each classification includes:

- Generation and classification keys/configuration.
- Prompt, raw response, and extracted code.
- Category, detail, structured failure stage, and diagnostics.
- Sanitizer status and excerpt.
- Robustness flags and per-probe battery diagnostics.
- Timing distribution for passing kernels.
- Runtime environment and timestamps.

## 10. Analysis plan

Build one analysis row per classification artifact. Deduplicate by
`classification_key`. Verify that the expected primary dataset contains 180
unique combinations of problem, arm, and sample before computing headlines.

A minimal artifact audit using only the standard library:

```python
import json
from collections import Counter
from pathlib import Path

records = [
    json.loads(path.read_text())
    for path in Path("exported-results/classifications").glob("*.json")
]
by_key = {record["classification_key"]: record for record in records}
records = list(by_key.values())

combinations = {
    (record["problem_id"], record["arm"], record["sample"])
    for record in records
    if record["level"] == 2
}
print("unique classifications:", len(records))
print("primary combinations:", len(combinations), "expected: 180")
print("categories:", Counter(record["category"] for record in records))
```

Before final analysis, restrict this audit to the frozen problem list and
primary generation/classification configuration. Multiple GPU classifications
or schema versions are deliberately separate artifacts and otherwise inflate
the apparent row count.

### RQ1: failure-category distribution

Report counts and percentages for each arm across:

- `empty`
- `syntactic`
- `runtime`
- `functional`
- `silent`
- `pass`
- infrastructure `error` separately

Do not silently remove empty generations or infrastructure errors. Give both
the intended denominator and the successfully classified denominator.

### RQ2: false-pass rate

The predeclared false-pass rate is:

```text
silent / (silent + pass)
```

This denominator contains kernels that passed the standard functional check.
Report the numerator, denominator, percentage, and a binomial confidence
interval. Do not describe FP16-only flags as silent FP32 failures.

### RQ3: reasoning and failure relocation

Compare the two arms on:

- Visible failure probability: syntactic/runtime/functional.
- Standard-pass probability: silent/pass.
- Silent share among standard-passers.
- Final robust-pass probability.

The arms share problems, so observations are clustered by problem; three
samples from one problem are not three independent tasks. Use problem-level
paired summaries or a problem-clustered bootstrap for confidence intervals.
Fisher's exact test can be reported as a small-sample exploratory comparison,
but it should not be the only inferential analysis. A mixed-effects logistic
model with problem as a random effect is appropriate if sample size and
supervision permit it.

Treat sample 0 as the greedy primary comparison to published KernelBench-style
generation. Treat temperature-0.7 samples as the repeated-sampling analysis.

### RQ4: hardware sensitivity

Join classifications on `generation_artifact_key` and compare category changes
across GPU types. Report a transition table such as T4/L4 → A100, and distinguish
true output changes from sanitizer availability changes.

### Performance

Use timing only where `category == "pass"` and timing is present. Report:

- Median speedup and confidence interval.
- Fraction above 1.0× and 1.1×.
- p10/p90 runtime spread.
- Slow passing kernels as well as fast ones.

Never quote the speed of a functional or silent kernel as a successful
optimization.

### Root-cause analysis

Create a fixed codebook before labeling, for example:

- Contiguity/stride assumption.
- Shape or launch-grid assumption.
- Out-of-bounds access.
- Race/synchronization.
- Uninitialized output.
- Numeric precision/overflow.
- Hardcoded/input-independent result.
- Wrapper/import/build mistake.
- Weight/state mismatch.
- Unsupported dtype.

Blind the annotator to reasoning arm where practical. Double-label a subset,
resolve disagreements, and report agreement (for example Cohen's kappa). Keep
automated LLM labels separate from human-confirmed labels.

## 11. Required tables and figures

At minimum produce:

1. Category counts and percentages by arm.
2. False-pass rate with confidence interval by arm.
3. Standard-pass → silent/pass survival diagram.
4. Root-cause distribution for silent failures.
5. Passing-kernel speedup distribution.
6. Problem-paired arm comparison.
7. GPU transition matrix if RQ4 is completed.
8. A worked CUDA example showing why the standard test passed and the stress
   probe failed.

Always show raw denominators. With small denominators, emphasize uncertainty
rather than a dramatic percentage.

## 12. Dissertation/paper structure

Suggested structure:

1. **Introduction:** why standard kernel pass rates can overstate correctness.
2. **Background:** KernelBench, generated CUDA extensions, reasoning models,
   GPU correctness hazards.
3. **Research questions:** RQ1–RQ4 with predeclared metrics.
4. **Method:** fixed tasks, paired arms, prompt, pinned environment, classifier,
   stress battery, validation, cache, and statistical plan.
5. **Results:** category distribution, false-pass rate, root causes, speed, and
   optional hardware transitions.
6. **Discussion:** what reasoning fixes, what it misses, and implications for
   benchmark validity.
7. **Threats to validity:** sample size, one model/provider, fixed battery,
   problem-specific shapes, sanitizer coverage, GPU dependence, and generated
   native-code risk.
8. **Related work.**
9. **Conclusion and future work.**

Avoid claiming that reasoning *causes* every observed difference without
qualification. This is a controlled same-model setting, but generation-service
behavior, finite sampling, and problem clustering still matter.

## 13. Completion checklist

### Before collection

- [ ] Local 13/13 tests pass.
- [ ] Ruff lint and formatting pass.
- [ ] `uv lock --check` passes.
- [ ] Modal GPU validation is 10/10 on the chosen primary GPU.
- [ ] Compute Sanitizer status is confirmed and documented.
- [ ] Problem list, samples, temperature, seed, GPU, and metrics are frozen.
- [ ] Cost limit and batch strategy are recorded.

### Collection

- [ ] Five-problem pilot reviewed.
- [ ] All 180 primary jobs accounted for.
- [ ] Failed infrastructure jobs retried without `--regen` unless generation
      itself failed.
- [ ] No post-outcome changes to the task list or primary metric.
- [ ] Optional second-GPU classifications reuse generation artifact keys.

### Analysis

- [ ] Exported artifacts backed up and checksummed.
- [ ] Classification keys deduplicated.
- [ ] Missing/error outcomes reported.
- [ ] Confidence intervals use problem-aware resampling where appropriate.
- [ ] Root-cause codebook frozen before full labeling.
- [ ] Figures and tables include denominators.
- [ ] Historical pre-schema-v3 results kept separate or reclassified.

### Writing and release

- [ ] Method records exact git commit and environment.
- [ ] Claims match the actual sample size and uncertainty.
- [ ] Threats to validity include sanitizer and shape coverage.
- [ ] No credentials or private reasoning traces are released accidentally.
- [ ] Release package contains code, commands, problem IDs, aggregate tables,
      and a privacy-reviewed artifact subset.
- [ ] Final `just test`, `just lint`, and `just validate` pass.

## 14. Suggested completion schedule

### 14–18 July: freeze and validate

- Run all local checks and the 10-case L4 validation.
- Freeze problem IDs, model ID, arms, temperatures, seed, primary GPU, and
  statistical definitions in the research log.
- Run and inspect the five-problem pilot.
- Finish the method and background chapters while compute runs.

### 19–31 July: collect and audit

- Complete the remaining primary collection in monitored batches.
- Retry infrastructure failures without unnecessary regeneration.
- Export a first backup and verify every expected combination.
- Start blinded root-cause codebook development on a pilot subset.

### 1–9 August: analyze

- Freeze the exported primary dataset.
- Produce category, false-pass, paired-arm, and performance tables.
- Compute uncertainty intervals and problem-clustered bootstrap results.
- Complete double-labeling and agreement analysis for root causes.
- Run the optional second-GPU classification only if primary analysis is safe.

### 10–18 August: write results and discussion

- Insert final figures and tables into the dissertation.
- Write results without interpretation first, then discussion and threats.
- Check that every headline number can be traced to artifact keys and code.

### 19–28 August: review and release

- Obtain supervisor/peer review.
- Correct claims, captions, limitations, and reproducibility instructions.
- Privacy-review any released prompts/responses and remove credentials.
- Tag the final git commit and archive code, aggregate data, and checksums.
- Leave several days before submission for formatting and upload failures.

Protect the primary path: if optional RQ4 work threatens the dissertation
deadline, omit it and report it as future work.

## 15. Troubleshooting

### Compute Sanitizer unavailable

Use L4 or A100. Keep the recorded `unavailable` status; do not rewrite it as
zero errors. If T4 is retained, state that Stage 2b coverage was unavailable.

### A rerun appears to call Bedrock again

Check model ID, arm, sample, temperature, dataset revision, and prompt hash.
Any generation-setting change correctly creates another generation key. Do not
use `--regen` for ordinary reclassification.

### A classifier change returns an old label

Increment `CLASSIFIER_CACHE_VERSION`, then run with `--force`. Generation and
classification are separate, so this should not spend another model request.

### A synthetic probe crashes the reference

Inspect `battery_diagnostics`. Reference failures should be marked skipped. If
the shape is problem-specific, move it into a valid `get_stress_inputs()` case.

### Modal job fails during native compilation

Inspect the structured failure stage and message. Distinguish Python parsing,
extension build, model initialization, forward runtime, sanitizer, and Modal
infrastructure failures before assigning a research category.

## 16. Current verification status

The hardening change was locally verified with:

- 13/13 local tests.
- Ruff lint passing.
- Ruff formatting passing.
- Python compilation passing.
- `uv lock --check` passing.
- `git diff --check` passing.

The 10-case Modal GPU suite must still be executed with the researcher's Modal
credentials before final data collection.
