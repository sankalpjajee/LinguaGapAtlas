#!/usr/bin/env bash
# Run every model in models.tsv on both task groups: the multiple-choice pairs
# (lga_mc) and MultiLoKo (lga_multiloko). Results go to results/<group>/<model>/.
#
# Usage: ./run.sh [models.tsv] [extra lm_eval arguments, e.g. --limit 20]
#   ONLY=2 ./run.sh                 run only the 2nd model (SLURM arrays: ONLY=$SLURM_ARRAY_TASK_ID)
#   BACKEND=vllm ./run.sh           use vLLM instead of transformers (faster; pip install vllm)
#   TASK_GROUPS=lga_mc ./run.sh     run one group only
#   BATCH_SIZE=16 ./run.sh          fixed multiple-choice batch size instead of auto
#   GEN_BATCH_SIZE=8 ./run.sh       MultiLoKo batch size (default 16). Generation never uses
#                                   auto: lm-eval's probe tries batch 64 at the model's full
#                                   context (8k tokens for Aya), and on a busy GPU it can crash
#                                   with a cuBLAS allocation error instead of backing off.
set -euo pipefail
cd "$(dirname "$0")"
[ -f tasks/lga_mc.yaml ] || { echo "tasks/ is missing: run python prepare.py first" >&2; exit 1; }
models=${1:-models.tsv}
shift || true
backend=${BACKEND:-hf}
task_groups=${TASK_GROUPS:-lga_mc lga_multiloko}

n=0
# tabs -> "|" so that empty columns survive `read` (tab is IFS whitespace and would collapse)
grep -v '^#' "$models" | tail -n +2 | tr '\t' '|' | while IFS='|' read -r model label extra; do
  [ -z "$model" ] && continue
  n=$((n + 1))
  if [ -n "${ONLY:-}" ] && [ "$n" != "$ONLY" ]; then continue; fi
  for group in $task_groups; do
    [ -f "tasks/$group.yaml" ] || { echo "skipping $group: not prepared"; continue; }
    if [ "$group" = lga_multiloko ]; then batch=${GEN_BATCH_SIZE:-16}; else batch=${BATCH_SIZE:-auto}; fi
    echo "=== $model ($label): $group"
    lm_eval --model "$backend" \
      --model_args "pretrained=$model,dtype=bfloat16${extra:+,$extra}" \
      --include_path tasks --tasks "$group" \
      --batch_size "$batch" --log_samples \
      --output_path "results/$group" "$@" < /dev/null
  done
done
