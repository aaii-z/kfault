#!/usr/bin/env bash
# Run several batches back to back, strictly one at a time.
#
#   ./run_remaining.sh 4 10       # batches 04..10
#   KFAULT_GPU=T4 ./run_remaining.sh 5 6
#
# Why sequential: each batch is 10 problems x 2 arms, which already fills the
# 10-container limit. Two batches at once do not go faster, they just queue,
# and their logs interleave.
#
# A failing batch does NOT stop the rest. Every batch writes its own log and
# its outcome is summarised at the end, so one bad run cannot silently take
# down an overnight sweep.
set -uo pipefail

FIRST="${1:?usage: run_remaining.sh <first> <last>}"
LAST="${2:?usage: run_remaining.sh <first> <last>}"
HERE=$(cd "$(dirname "$0")" && pwd)

declare -a OUTCOMES=()

for batch in $(seq "$FIRST" "$LAST"); do
    printf '\n===== batch %02d starting (%s) =====\n' \
        "$batch" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if "$HERE/run_batch.sh" "$batch"; then
        OUTCOMES+=("batch $(printf '%02d' "$batch"): ok")
    else
        code=$?
        OUTCOMES+=("batch $(printf '%02d' "$batch"): FAILED (exit $code)")
        echo "batch $batch failed with exit $code; continuing with the rest"
    fi
done

echo
echo "===== sweep finished ====="
for line in "${OUTCOMES[@]}"; do
    echo "  $line"
done
echo "now record each outcome in $HERE/STATUS.md"
