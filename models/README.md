# Model card and model files

Weights are **not committed** (see `.gitignore`). **No fine-tuned weights are released**: in the experiments none was an improvement over the pretrained model on real photographs (see `docs/report.md`). The fine-tuned checkpoints (`models/ft/`, 66 files of 5 to 20 MB) exist only on the author's machine and can be regenerated with the scripts in `scripts/`.

## The model this project recommends
| | |
|---|---|
| Model | YOLO11s (Ultralytics), **pretrained, used unmodified** (`models/yolo11s.pt`, downloaded automatically by Ultralytics on first use) |
| Task | Object detection; this project keeps only nine classes: person (primary), cat, dog, horse, sheep, cow (secondary), car, bus, truck (context) |
| Training data | COCO train2017 (by Ultralytics, not by this project) |
| Size and speed | 9.5 M parameters; 16.4 ms per frame, batch 1, forward pass only, RTX 3050 Laptop GPU (about 18 ms per frame including post-processing in the video test) |
| Licence | AGPL-3.0 (code and weights) |

### Measured behaviour (person mAP50-95, 95 % bootstrap intervals in the README)
Clean COCO sample 0.594; ExDark (real low light) 0.406; DAWN fog (real, 117 person boxes) 0.486; BDD100K clear day 0.263, clear night 0.191. Under strong synthetic degradation it falls to 0.21 (low light), 0.34 (rain) and about 0.19 (motion blur).

### Intended use
Research and demonstration: studying how a detector degrades; the camera-only perception component of a low-speed, controlled-environment rover (Project 3 of this portfolio) **after validation on that rover's own camera**. Optional dark-channel dehazing in front of it if fog is expected (helped on real fog, at 3 to 9 times the detector's CPU time).

### Not for
- Safety-critical decisions on its own: a missed person is possible, especially small, partly hidden or low-contrast people (heads in water, people in the dark). It must be one layer among several (range sensing, manual emergency stop, a human operator).
- Any statement about whether a person is alive, conscious or unhurt.
- Unattended or high-speed operation, or operation near people without a human controlling the stop.

### Known limits (details: `docs/report.md`, section 9)
Evaluated on public datasets and synthetic degradations, not on a rover camera; real fog evidence is one small set; no real rain or snow set with enough people; thermal imaging and debris detection are out of scope.

## Other pretrained models evaluated (zero-shot only)
| File | Notes |
|---|---|
| `yolo11n.pt` | 2.6 M parameters, 16.1 ms; the model used for all fine-tuning experiments (fits 4 GB) |
| `yolo26n.pt` | 2.6 M parameters, 22.0 ms on the test GPU; no advantage over yolo11n. yolo26s was not evaluated |

## Fine-tuned checkpoints (local only, not released)
Named by experiment and seed: `B1/A1/A2_yolo11n_s{0,1,2}` (clean control, condition-aware augmentation, plus BDD100K adverse data), `B1L/A1L_*` (batch-norm locked), `D*` (phase 5 diagnosis runs), `sw_<setting>_s<seed>` (the 45-run sweep). Training data: COCO train subsets (and BDD100K for A2 only); ExDark and DAWN were never trained on. Config and arguments for each: `configs/sweep.yaml`, `scripts/train.py`, `PHASES.md`. If weights are ever released, only those trained on COCO and BDD100K may be, with the BDD100K notice included (policy in `data/README.md`).
