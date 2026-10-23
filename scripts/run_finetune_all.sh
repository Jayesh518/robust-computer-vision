#!/bin/bash
# Train and evaluate every variant for every seed, resumable: a finished checkpoint or a finished evaluation is skipped.
# Order is seed-major, so after the first pass B1/A1/A2 exist for seed 0 and can be compared before seeds 1 and 2 finish.
# Override with e.g.  SEEDS="0" EPOCHS=10 bash scripts/run_finetune_all.sh
# Evaluation uses the 23 "core" sets (the 12 dropped are the mild severities, clean_full and BDD fog, which carry no conclusions).
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-10}"
BASE="${BASE:-yolo11n}"
CORE="clean_bench1k low_light_s2 low_light_s3 fog_s3 rain_s3 snow_s3 motion_blur_s2 motion_blur_s3 gaussian_noise_s3 occ_clean occ_f25 occ_f50 occ_f75 exdark_all dawn_fog dawn_rain dawn_snow bdd_clear_day bdd_night_clear bdd_rainy_day bdd_snowy_day bdd_rainy_night bdd_snowy_night"
for seed in $SEEDS; do
  for v in B1 A1 A2; do
    name="${v}_${BASE}_s${seed}"
    if [ ! -f "models/ft/$name.pt" ]; then
      echo "=== TRAIN $name $(date +%H:%M:%S)"
      .venv/Scripts/python scripts/train.py --variant $v --seed $seed --epochs $EPOCHS --weights models/$BASE.pt --name "$name" > "runs/log_train_$name.txt" 2>&1 \
        || { echo "TRAIN FAILED $name (see runs/log_train_$name.txt)"; exit 1; }
    fi
    if [ ! -f "results/ft_$name.done" ]; then
      echo "=== EVAL $name $(date +%H:%M:%S)"
      .venv/Scripts/python scripts/run_baseline.py --weights "models/ft/$name.pt" --name "ft_$name" --sets $CORE --n-boot 100 2>&1 \
        | grep --line-buffered -E "^(clean|low|fog|rain|snow|motion|gaussian|occ|exdark|dawn|bdd)|Traceback|Error"
      [ "${PIPESTATUS[0]}" = "0" ] && touch "results/ft_$name.done" || { echo "EVAL FAILED $name"; exit 1; }
    fi
    echo "=== DONE $name $(date +%H:%M:%S)"
  done
done
echo "FINETUNE ALL DONE $(date +%H:%M:%S)"
