#!/bin/bash
# Phase 5, second batch (added after D0 showed that even a near-zero learning rate lowers the scores). Runs after run_phase5.sh.
#   D3  near-zero learning rate AND no augmentation: if this matches B0, the loss comes from batch-norm statistics adapting to mosaic images
cd "$(dirname "$0")/.."
until grep -q "PHASE5 ALL DONE" /tmp/phase5.log; do sleep 30; done
SETS="clean_bench1k low_light_s3 rain_s3 exdark_all dawn_fog bdd_clear_day"
run() { name=$1; ep=$2; shift 2
  if [ ! -f "models/ft/$name.pt" ]; then echo "=== TRAIN $name $(date +%H:%M:%S)"
    .venv/Scripts/python scripts/train.py --variant B1 --seed 0 --epochs $ep --name "$name" "$@" > "runs/log_train_$name.txt" 2>&1 || { echo "TRAIN FAILED $name"; exit 1; }; fi
  if [ ! -f "results/ft_$name.done" ]; then echo "=== EVAL $name $(date +%H:%M:%S)"
    .venv/Scripts/python scripts/run_baseline.py --weights "models/ft/$name.pt" --name "ft_$name" --sets $SETS --n-boot 100 2>&1 | grep --line-buffered -E "^(clean|low|rain|exdark|dawn|bdd)|Traceback|Error"
    [ "${PIPESTATUS[0]}" = "0" ] && touch "results/ft_$name.done" || { echo "EVAL FAILED $name"; exit 1; }; fi
  echo "=== DONE $name $(date +%H:%M:%S)"; }
run D3_null_plain 3 --optimizer SGD --lr0 0.000001 --lrf 1.0 --warmup-epochs 0 --plain-aug
echo "PHASE5B ALL DONE $(date +%H:%M:%S)"
