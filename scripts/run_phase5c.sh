#!/bin/bash
# Phase 5c: test the batch-norm explanation for the loss that appears without learning. Seed 0, B1 data.
#   D5  near-zero learning rate + ALL batch-norm layers locked: if batch-norm drift is the whole procedural loss this should match B0
#   D4  the original clean-control recipe + all batch-norm layers locked, 10 epochs
cd "$(dirname "$0")/.."
SETS="clean_bench1k low_light_s3 rain_s3 exdark_all dawn_fog bdd_clear_day"
run() { name=$1; ep=$2; shift 2
  if [ ! -f "models/ft/$name.pt" ]; then echo "=== TRAIN $name $(date +%H:%M:%S)"
    .venv/Scripts/python scripts/train.py --variant B1 --seed 0 --epochs $ep --name "$name" "$@" > "runs/log_train_$name.txt" 2>&1 || { echo "TRAIN FAILED $name"; exit 1; }; fi
  if [ ! -f "results/ft_$name.done" ]; then echo "=== EVAL $name $(date +%H:%M:%S)"
    .venv/Scripts/python scripts/run_baseline.py --weights "models/ft/$name.pt" --name "ft_$name" --sets $SETS --n-boot 100 2>&1 | grep --line-buffered -E "^(clean|low|rain|exdark|dawn|bdd)|Traceback|Error"
    [ "${PIPESTATUS[0]}" = "0" ] && touch "results/ft_$name.done" || { echo "EVAL FAILED $name"; exit 1; }; fi
  echo "=== DONE $name $(date +%H:%M:%S)"; }
run D5_null_lockbn 3 --optimizer SGD --lr0 0.000001 --lrf 1.0 --warmup-epochs 0 --lock-bn
run D4_lockbn 10 --lock-bn
echo "PHASE5C ALL DONE $(date +%H:%M:%S)"
