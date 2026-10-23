# Project phases and status

Standard shape: see the workspace `docs/phases.md`. Status date: 2026-10-24.

| Phase | Content | Status |
|---|---|---|
| 0. Scope and protocol | Nine classes (person primary, five animals secondary, three vehicles as context); synthetic degradations (low light, fog, rain, snow, motion blur, noise) and a concrete-slab occlusion test; metrics fixed in advance; no alive/dead claim | **done** |
| 1. Setup, data and licences | COCO, ExDark, DAWN, BDD100K acquired; licence review recorded in `data/README.md`; RTTS skipped; BDD100K md5 mismatch recorded as unverified | **done** |
| 2. Pipeline and validation | Label mapping, deterministic degradations and occlusion, Ultralytics evaluation, bootstrap intervals, found-rate; cross-checked against pycocotools; 117 tests; CI | **done** |
| 3. Baseline (B0) | yolo11n, yolo11s, yolo26n zero-shot on 35 sets each, with intervals and latency | **done** |
| 4. Intervention (B1, A1, A2) | Clean fine-tune control, condition-aware augmentation, plus BDD100K adverse data; yolo11n, 10 epochs, 3 seeds each | **done** (first recipe) |
| 5. Diagnosis and iteration | Explain why the clean fine-tune scores below zero-shot and why real-photo transfer is weak: lower learning rate, frozen layers, more training data, yolo11s; then re-run the affected variants | **next** |
| 6. Extension (E1) | Enhancement preprocessing (low-light, dehaze, derain) before the zero-shot model, reporting accuracy gain against added latency | planned |
| 7. Report and release | Report, failure gallery from the author's own photographs, model card, weights trained only on COCO and BDD data, two-page summary | planned |

## Gate for phase 5 (what must be true to close it)
- B1 is no longer below zero-shot on clean COCO, or the reason it is below is shown by an experiment.
- The real-photo comparison (ExDark, DAWN, BDD100K) is re-run on the best recipe with 3 seeds.
- Anything still unexplained is listed in the README.
