# kfault

`kfault` generates CUDA C++ extensions in KernelBench's Python `ModelNew`
format, classifies visible and stress-only failures, and benchmarks kernels on
Modal GPUs.

For the complete implementation record, runbook, data-collection protocol,
analysis plan, and dissertation checklist, see
[`docs/IMPLEMENTATION_AND_RESEARCH_GUIDE.md`](docs/IMPLEMENTATION_AND_RESEARCH_GUIDE.md).

## Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), a Modal account,
and an AWS Bedrock bearer token.

```bash
uv sync
uv run modal token new
uv run modal secret create bedrock \
  AWS_BEARER_TOKEN_BEDROCK=your_token
```

The Modal image, Python packages, and KernelBench dataset revision are pinned
in `pipeline.py`. Generated code runs in a single-use GPU container with no
network or Modal-resource access.

## Verify and run

```bash
just test       # local parser/cache/sanitizer tests
just validate   # GPU classifier regression suite (10 cases)
just run        # Level 2, problems 1-10, both reasoning arms
```

For a sanitizer-capable GPU:

```bash
KFAULT_GPU=L4 uv run modal run pipeline.py --problems 1-10
```

Generation artifacts and classification artifacts use separate keys in the
`kfault-results` Modal volume. `--force` reclassifies without another model
request; `--regen` always requests a new generation.

Problem files may optionally define `get_stress_inputs()`, returning a mapping
from case name to argument list, to test valid empty, odd, broadcasting, or
boundary shapes that cannot be inferred safely by a generic harness.
