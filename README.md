# kfault

A failure taxonomy for LLM-generated CUDA kernels.

KernelBench scores a generated kernel pass or fail against `allclose` on a few
random inputs. `kfault` adds a stress battery on top, so kernels that pass that
check but are wrong on non-contiguous, extreme or repeated inputs are counted
separately as **silent failures**, and adds a repair loop so the same kernel can
be measured before and after automated correction.

| category | meaning |
|---|---|
| `syntactic` | does not build |
| `runtime` | crashes, trips the sanitizer, or is too slow to evaluate |
| `functional` | runs, wrong answer on the standard input |
| `silent` | passes the standard check, wrong under the stress battery |
| `pass` | correct and survives the battery |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how it works and why,
and [`experiments/level2-repair/`](experiments/level2-repair/) for results.

## Setup

Python 3.12, [uv](https://docs.astral.sh/uv/), a Modal account and an AWS
Bedrock bearer token.

```bash
uv sync
uv run modal token new
uv run modal secret create bedrock AWS_BEARER_TOKEN_BEDROCK=your_token
```

The CUDA image, Python packages and KernelBench dataset revision are pinned in
`kfault/config.py`. Generated code runs in a single-use GPU container with no
network and no Modal-resource access.

## Run

```bash
just test                     # 86 local tests, no GPU
just validate                 # 11 known-answer kernels through the classifier
just run                      # Level 2, problems 1-10, both arms
```

A full experiment batch, with the model and device selected by environment:

```bash
KFAULT_MODEL=zai.glm-5 KFAULT_GPU=L4 \
  ./experiments/level2-repair/run_batch.sh 1
```

On a slower device both time limits must be raised together, or the config
refuses the pair:

```bash
KFAULT_GPU=T4 KFAULT_FORWARD_LIMIT_S=20 KFAULT_CLASSIFIER_TIMEOUT_S=2700 \
  ./experiments/level2-repair/run_batch.sh 1
```

## Results

```bash
uv run modal run analyze.py --level 2 --gpu L4    # tables
uv run modal run export.py --level 2 --with-code  # cache -> data/
uv run python figures.py                          # data/ -> docs/figures/paper/
```

Generations and classifications use independent cache keys, so `--force`
re-classifies a cached response without a new model request, while `--regen`
always requests one.

No figure or table contains a hand-copied number; everything derives from
`data/attempts.csv` and `data/chains.csv`.

## Layout

```
kfault/       pure logic: classifier, battery, repair, analysis, caching
pipeline.py   Modal app: generate, classify, repair
validate.py   known-answer validation of the classifier
analyze.py    tables from the cache
export.py     cache to CSV, JSONL and kernel sources
figures.py    figures from the CSVs
tests/        86 tests, no GPU or network
data/         exported dataset
docs/         architecture, analysis, figures
experiments/  per-model results and batch records
```

Problem files may define `get_stress_inputs()`, returning a mapping from case
name to argument list, to add empty, odd, broadcasting or boundary shapes that
cannot be inferred safely by a generic harness.
