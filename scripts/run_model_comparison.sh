#!/bin/bash
# Zero-shot baseline (B0) for several model sizes/families on the SAME sets: metrics, bootstrap CIs, found rate, paired drops.
# Weights download automatically on first use. Run from the project root. Results: results/b0_<model>.json
cd "$(dirname "$0")/.."
for m in yolo11n yolo11s yolo26n; do
  echo "=== $m $(date +%H:%M:%S)"
  .venv/Scripts/python scripts/run_baseline.py --weights models/$m.pt --name b0_$m 2>&1 \
    | grep --line-buffered -E "^(clean|low|fog|rain|snow|motion|gaussian|occ|exdark|dawn|bdd)|already done|^wrote|Traceback|Error"
done
echo "COMPARISON DONE $(date +%H:%M:%S)"
