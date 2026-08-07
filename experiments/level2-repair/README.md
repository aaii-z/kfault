# Level-2 repair experiment

KernelBench Level 2, all 100 problems, split into **10 batches of 10** so the
run fits the 10-container limit and can be spread over several sessions.

## Protocol

| | |
|---|---|
| level | 2 |
| problems | 1-100, all |
| arms | `think` (reasoning high) and `direct` (reasoning none), same weights |
| samples | 1 (greedy, temperature 0) |
| repair turns | 3 |
| GPU | L4 (compute-sanitizer works; 24 GB avoids battery OOM) |
| model | `zai.glm-5` on Bedrock |

Per batch: 10 problems x 2 arms x up to 4 turns = 20-80 attempts.

## Why the batches are stratified, not sequential

Batch `k` holds problems `k, k+10, k+20, ... k+90` — not `1-10`, `11-20`.

KernelBench Level 2 ids cluster by operator family (many `Conv2d_*` together,
many `Gemm_*` together). Sequential blocks would make each batch a biased
sample of one family. Stratified batches mean **every batch is a
representative sample of the level**, so a partially finished experiment is
still analysable and reportable.

## Running

```bash
./run_batch.sh 1                 # batch 01, L4, 3 repair turns
KFAULT_GPU=T4 ./run_batch.sh 1   # cheaper, but no sanitizer
REPAIR_TURNS=0 ./run_batch.sh 1  # one-shot baseline only
```

Record each finished batch in `STATUS.md`.

## Order does not matter

Batches are independent. Run them in any order, on any day. Two properties
make this safe:

- **Generations are cached by content**, not by run, so re-running a finished
  batch reuses them and issues no new model requests.
- **`gpu` is part of the classification cache key.** A batch run on L4 and a
  batch run on T4 land in different cache entries; they never overwrite each
  other. But mixing GPUs across batches means mixing environments in one
  table — pick one GPU and stay on it. Record any exception in `STATUS.md`.

## Turn 0 is the baseline

Turn 0 of a repair run uses the **same cache key** as a one-shot run. So this
experiment produces both arms of the one-shot vs repair comparison in a single
pass — no separate baseline run is needed, and the 27 already-paid pilot
problems are reused for free.

## Layout

```
batches/batch01.txt ... batch10.txt   problem list + names, one spec line
logs/                                 tee'd run output, timestamped
run_batch.sh                          runner
STATUS.md                             progress and cost tracking
```
