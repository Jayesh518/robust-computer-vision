#!/bin/bash
# Phase 8: train and evaluate every setting in configs/sweep.yaml for every seed. Resumable, seed-major (after the first pass every
# setting exists for seed 0 and can be compared before seeds 1 and 2 finish). The explicit baseline replicate runs first so that
# it can be checked against the earlier A1 runs before the rest is trusted.
#
# Disk guard: after a run's metrics JSON and .done marker exist, its evaluation prediction files (up to ~110 MB per set) and training
# folder are deleted; the checkpoint (models/ft/<name>.pt, ~5 MB for yolo11n) and the per-epoch curve (results/curves/<name>.csv) are kept.
# The run stops if less than MIN_FREE_GB is free.
#
#   SEEDS="0 1 2" bash scripts/run_sweep.sh        ONLY="lr_1.2e-4 n_2k" bash scripts/run_sweep.sh
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1 2}"
MIN_FREE_GB="${MIN_FREE_GB:-15}"
PY=.venv/Scripts/python
EVAL_SETS=$($PY scripts/sweep_plan.py --eval-sets)
mkdir -p runs results/curves models/ft
# the replicate goes first; the other settings keep the config order
FIRST=$($PY scripts/sweep_plan.py | awk -F'\t' '$1=="lr_1.2e-4"')
REST=$($PY scripts/sweep_plan.py | awk -F'\t' '$1!="lr_1.2e-4"')
PLAN="$FIRST
$REST"

for seed in $SEEDS; do
  while IFS=$'\t' read -r setting group level targs; do
    [ -z "$setting" ] && continue
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $setting "* ]]; then continue; fi
    name="sw_${setting}_s${seed}"
    free_gb=$(df -k /c | awk 'NR==2{printf "%d", $4/1048576}')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then echo "STOP: only ${free_gb} GB free"; exit 2; fi
    manifest=$(echo "$targs" | grep -o -- '--train-list [^ ]*' | awk '{print $2}')
    if [ -n "$manifest" ] && [ ! -f "$manifest" ]; then echo "SKIP $name (data not ready: $manifest)"; continue; fi
    if [ ! -f "models/ft/$name.pt" ]; then
      echo "=== TRAIN $name $(date +%H:%M:%S)"
      t0=$(date +%s)
      $PY scripts/train.py $targs --seed $seed --name "$name" > "runs/log_train_$name.txt" 2>&1 \
        || { echo "TRAIN FAILED $name (see runs/log_train_$name.txt)"; exit 1; }
      echo $(( $(date +%s) - t0 )) > "results/curves/$name.train_seconds"
    fi
    if [ ! -f "results/ft_$name.done" ]; then
      echo "=== EVAL $name $(date +%H:%M:%S)"
      $PY scripts/run_baseline.py --weights "models/ft/$name.pt" --name "ft_$name" --sets $EVAL_SETS --n-boot 100 2>&1 \
        | grep --line-buffered -E "^(clean|low|rain|occ|exdark|dawn|bdd)|Traceback|Error"
      [ "${PIPESTATUS[0]}" = "0" ] && touch "results/ft_$name.done" || { echo "EVAL FAILED $name"; exit 1; }
    fi
    # cleanup only after the .done marker exists
    if [ -f "results/ft_$name.done" ]; then
      [ -f "runs/train/$name/results.csv" ] && cp "runs/train/$name/results.csv" "results/curves/$name.csv"
      rm -rf "runs/train/$name" runs/ft_${name}_* 2>/dev/null
    fi
    echo "=== DONE $name $(date +%H:%M:%S)"
  done <<< "$PLAN"
done
echo "SWEEP ALL DONE $(date +%H:%M:%S)"
