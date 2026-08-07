#!/usr/bin/env bash
# Run one batch of the Level-2 repair experiment.
#
#   ./run_batch.sh 3           # batch 03, default settings
#   KFAULT_GPU=T4 ./run_batch.sh 3
#   REPAIR_TURNS=0 ./run_batch.sh 3    # one-shot baseline only
#   FORCE=1 ./run_batch.sh 3           # re-label from cached generations
#
# FORCE=1 re-classifies without regenerating: the cached model output is
# reused, so no new model requests are issued. Use it after a classifier fix.
#
# Each batch is 10 problems x 2 arms, matching the 10-container limit.
# Run batches ONE AT A TIME -- two at once exceeds the limit and they queue.
set -euo pipefail

BATCH=$(printf "%02d" "${1:?usage: run_batch.sh <1-10>}")
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

: "${KFAULT_GPU:=L4}"
: "${REPAIR_TURNS:=3}"
: "${LEVEL:=2}"
: "${FORCE:=0}"
: "${KFAULT_MODEL:=zai.glm-5}"
export KFAULT_GPU KFAULT_MODEL

EXTRA=()
if [ "$FORCE" != "0" ]; then
    EXTRA+=(--force)
fi

FILE="$HERE/batches/batch$BATCH.txt"
PROBLEMS=$(grep -v '^#' "$FILE" | grep -v '^$' | head -1)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
# One directory per model. Batch definitions stay shared -- every model must
# run the SAME stratified problem sets or the comparison is not valid -- but
# results are separated, because two models write batches with the same number
# and "the newest batch01 log" was otherwise ambiguous.
SLUG=$(printf '%s' "$KFAULT_MODEL" | tr -c 'A-Za-z0-9' '-' | sed 's/-\{2,\}/-/g;s/-$//')
LOGDIR="$HERE/runs/$SLUG/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/batch${BATCH}_${SLUG}_${STAMP}.log"

# The header must go INTO the log, not just to the terminal: batch 01/02 logs
# recorded no GPU, so which device produced them could not be recovered later.
HEADER="batch $BATCH | problems $PROBLEMS | gpu $KFAULT_GPU"
HEADER="$HEADER | model $KFAULT_MODEL"
HEADER="$HEADER | repair_turns $REPAIR_TURNS | force $FORCE"
HEADER="$HEADER | started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "$HEADER" | tee "$LOG"
echo "log: $LOG"

cd "$ROOT"
uv run modal run pipeline.py \
    --level "$LEVEL" \
    --problems "$PROBLEMS" \
    --repair-turns "$REPAIR_TURNS" \
    "${EXTRA[@]}" \
    2>&1 | tee -a "$LOG"

echo "done: batch $BATCH -> $LOG"
echo "now record the outcome in $HERE/runs/$SLUG/STATUS.md"
