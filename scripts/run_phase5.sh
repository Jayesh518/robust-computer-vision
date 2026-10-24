#!/bin/bash
# Phase 5 diagnosis: B1 data, seed 0, ONE change per experiment against the original B1 recipe. Resumable.
#   D0  near-zero learning rate: checks that the train/save/evaluate path itself does not change the model (expect ~B0)
#   D1  AdamW lr 2e-5 (about 1/6 of the automatic 1.2e-4)
#   D2  backbone frozen (first 10 layers), automatic optimizer
cd "$(dirname "$0")/.."
SETS="clean_bench1k low_light_s3 rain_s3 exdark_all dawn_fog bdd_clear_day"
run() {  # name, epochs, extra train args...
  name=$1; ep=$2; shift 2
  if [ ! -f "models/ft/$name.pt" ]; then
    echo "=== TRAIN $name $(date +%H:%M:%S)"
    .venv/Scripts/python scripts/train.py --variant B1 --seed 0 --epochs $ep --name "$name" "$@" > "runs/log_train_$name.txt" 2>&1 || { echo "TRAIN FAILED $name"; exit 1; }
  fi
  if [ ! -f "results/ft_$name.done" ]; then
    echo "=== EVAL $name $(date +%H:%M:%S)"
    .venv/Scripts/python scripts/run_baseline.py --weights "models/ft/$name.pt" --name "ft_$name" --sets $SETS --n-boot 100 2>&1 \
      | grep --line-buffered -E "^(clean|low|rain|exdark|dawn|bdd)|Traceback|Error"
    [ "${PIPESTATUS[0]}" = "0" ] && touch "results/ft_$name.done" || { echo "EVAL FAILED $name"; exit 1; }
  fi
  echo "=== DONE $name $(date +%H:%M:%S)"
}
run D0_null_lr 3 --optimizer SGD --lr0 0.000001 --lrf 1.0 --warmup-epochs 0
run D1_lr2e-5 10 --optimizer AdamW --lr0 0.00002
run D2_freeze_backbone 10 --freeze 10
echo "PHASE5 ALL DONE $(date +%H:%M:%S)"
