# Project phases and status

Standard shape: see the workspace `docs/phases.md`. Status date: 2026-10-24.

| Phase | Content | Status |
|---|---|---|
| 0. Scope and protocol | Nine classes (person primary, five animals secondary, three vehicles as context); synthetic degradations (low light, fog, rain, snow, motion blur, noise) and a concrete-slab occlusion test; metrics fixed in advance; no alive/dead claim | **done** |
| 1. Setup, data and licences | COCO, ExDark, DAWN, BDD100K acquired; licence review recorded in `data/README.md`; RTTS skipped; BDD100K md5 mismatch recorded as unverified | **done** |
| 2. Pipeline and validation | Label mapping, deterministic degradations and occlusion, Ultralytics evaluation, bootstrap intervals, found-rate; cross-checked against pycocotools; 117 tests; CI | **done** |
| 3. Baseline (B0) | yolo11n, yolo11s, yolo26n zero-shot on 35 sets each, with intervals and latency | **done** |
| 4. Intervention (B1, A1, A2) | Clean fine-tune control, condition-aware augmentation, plus BDD100K adverse data; yolo11n, 10 epochs, 3 seeds each | **done** (first recipe) |
| 5. Diagnosis and iteration | Explain why the clean fine-tune scores below zero-shot: no-learning control (D0), no-augmentation control (D3), lower learning rate (D1), frozen backbone (D2) | **done, gate not fully met**: about half the loss comes from the training path itself, none of the recipes removed the rest; causes still unexplained are listed in the README |
| 6. Extension (E1) | Enhancement preprocessing (low-light, dehaze, derain) before the zero-shot model, reporting accuracy gain against added latency | **next** |
| 7. Report and release | Report, failure gallery from the author's own photographs, model card, weights trained only on COCO and BDD data, two-page summary | planned |

## Gate for phase 5 (what must be true to close it)
- B1 is no longer below zero-shot on clean COCO, or the reason it is below is shown by an experiment.
- The real-photo comparison (ExDark, DAWN, BDD100K) is re-run on the best recipe with 3 seeds.
- Anything still unexplained is listed in the README.

## Phase 5 plan (written 2026-10-24, before any phase 5 result)
Open problems: (a) the clean control B1 scores below zero-shot B0 on clean COCO (0.506 vs 0.532) and on several real sets; (b) the augmentation gain barely transfers to real photographs. The earlier runs used Ultralytics' automatic optimizer (AdamW, lr 0.000119), so a too-high learning rate is not the obvious cause.

Experiments, all on the B1 data, seed 0, 10 epochs unless stated, one change each (test sets: clean sample, low light s3, rain s3, ExDark, DAWN fog, BDD clear day; all n=1):
- **D0 null control** (3 epochs, learning rate about zero): the train/save/evaluate path alone should leave the model close to B0. If it does not, the cause is the pipeline, not the learning.
- **D1 lower learning rate** (AdamW 2e-5).
- **D2 frozen backbone** (first 10 layers).

Decision rules, fixed now: a recipe counts as fixing the control if its clean-sample person mAP50-95 is within 0.01 of B0 (0.532) **and** its ExDark score is not below B0 (0.376). If none does, the remaining untested causes are more training data (the training subset is 8 % of COCO train) and the 9-class label restriction; they are then listed as unexplained rather than guessed. Only a recipe that passes is re-run with the three variants over 3 seeds.

### Adjustment logged 2026-10-24 (after D0, before D1/D2 results)
D0 (learning rate about zero, 3 epochs) scored clean 0.521, ExDark 0.363, rain s3 0.188 against zero-shot 0.532, 0.376, 0.218, so about 0.01 to 0.03 of the earlier "fine-tuning loss" appears **without any learning**. Consequences: (1) the pre-registered pass rule (within 0.01 of B0 on clean) is unreachable if the path itself costs that much, so recipes are compared against D0 as the floor instead of against B0 alone, and this change is reported; (2) a new test **D3** (near-zero learning rate plus no augmentation) checks whether batch-norm statistics adapting to mosaic-augmented images explains it. D1 and D2 continue unchanged.

### Phase 5 outcome (2026-10-24)
No recipe met the pre-registered pass rule, so no recipe was re-run over three seeds. Closed with open items: exact cause of the no-learning loss (batch-norm statistics suspected, untested), effect of more training data, effect of the 9-class label restriction. Details and table in the README.
