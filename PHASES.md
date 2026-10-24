# Project phases and status

Standard shape: see the workspace `docs/phases.md`. Status date: 2026-10-24.

| Phase | Content | Status |
|---|---|---|
| 0. Scope and protocol | Nine classes (person primary, five animals secondary, three vehicles as context); synthetic degradations (low light, fog, rain, snow, motion blur, noise) and a concrete-slab occlusion test; metrics fixed in advance; no alive/dead claim | **done** |
| 1. Setup, data and licences | COCO, ExDark, DAWN, BDD100K acquired; licence review recorded in `data/README.md`; RTTS skipped; BDD100K md5 mismatch recorded as unverified | **done** |
| 2. Pipeline and validation | Label mapping, deterministic degradations and occlusion, Ultralytics evaluation, bootstrap intervals, found-rate; cross-checked against pycocotools; 117 tests; CI | **done** |
| 3. Baseline (B0) | yolo11n, yolo11s, yolo26n zero-shot on 35 sets each, with intervals and latency | **done** |
| 4. Intervention (B1, A1, A2) | Clean fine-tune control, condition-aware augmentation, plus BDD100K adverse data; yolo11n, 10 epochs, 3 seeds each | **done** (first recipe) |
| 5. Diagnosis and iteration | Explain why the clean fine-tune scores below zero-shot: no-learning (D0), no-augmentation (D3), lower learning rate (D1), frozen backbone (D2), batch-norm lock (D4, D5) | **done, closed (5a-5d)**; gate met in part: the procedural loss (about 0.01) is explained by batch-norm statistics drift and removed by locking them (D5); the loss from learning on the clean subset is not removed by any recipe tried; more data and the 9-class label restriction are untested |
| 6. Extension (E1) | Classical enhancement preprocessing (CLAHE and adaptive gamma for low light, dark-channel-prior dehazing for fog, non-local-means for noise) before the zero-shot model, accuracy gain against added latency | **done**: dehazing helps on fog, the low-light methods do not (see README) |
| 7. Report and release | Report, failure gallery from the author's own photographs, model card, weights trained only on COCO and BDD data, two-page summary | **next** |

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

### Phase 5c plan (written 2026-10-24, before any result; reopens phase 5 to test the batch-norm explanation)
Source check (ultralytics 8.4.173): the trainer calls `model.train()` every epoch, which re-estimates every BatchNorm layer's running statistics on the training batches (momentum 0.03) even at a near-zero learning rate. `freeze=10` only locks the backbone's statistics, so D2 was a partial test. A callback on `on_train_epoch_start` would be undone (it fires before `model.train()`), so the lock is applied on `on_train_batch_start`.
- **D5** (learning rate about zero + all BatchNorm locked, 3 epochs). Prediction: if drift is the whole no-learning loss, D5 matches B0. **Confirmed** if its clean and ExDark scores are within 0.005 of B0 (0.532, 0.376).
- **D4** (original clean-control recipe + all BatchNorm locked, 10 epochs). **Fixes the control** if clean is within 0.01 of B0 and ExDark is not below B0.
- If D5 is confirmed and D4 fixes the control, B1/A1/A2 are re-run with the lock over 3 seeds (about 7 hours). If D5 is not confirmed, the batch-norm explanation is rejected and the open items stay open.

### Phase 5c outcome (2026-10-24)
D5 (learning rate about zero + batch-norm locked) matched zero-shot (clean 0.529 vs 0.532, ExDark 0.381 vs 0.376, the latter just inside the 0.005 tolerance): batch-norm drift explains the loss without learning. D4 (original recipe + lock) did not fix the control (clean 0.515, ExDark 0.370, low light s3 0.127), so by the rule written beforehand the variants were not re-run. Still open: effect of more training data, effect of the 9-class label restriction, and (answered later in 5d) whether a batch-norm-locked A1 beats zero-shot on real photographs.

### Phase 5d plan (written 2026-10-24, before any result)
Question: once the batch-norm side effect is removed, does condition-aware augmentation beat the **zero-shot** model on real photographs? Runs: B1L (clean control) and A1L (condition-aware augmentation), both with every BatchNorm layer locked, 3 seeds each, 23 core sets, yolo11n, 10 epochs.
Decision rule, fixed now. Let `d` = mean over the 3 seeds of (A1L minus zero-shot) in person mAP50-95 on the real sets.
- **Transfer supported** if `d >= +0.01` on ExDark **and** on DAWN fog **and** on BDD clear night, and A1L minus B1L is positive on ExDark in all 3 seeds.
- **Transfer not supported** if `d` is below +0.01 on ExDark or on DAWN fog; reported as a negative result.
- Anything in between (mixed sets) is reported as mixed, with the sets named.
The other findings (synthetic gains, low-light loss) are reported regardless of the outcome. A2 is not re-run: it is confounded by extra data and its BDD gain is in-domain.

### Phase 5d outcome (2026-10-24)
B1L and A1L (batch-norm locked), 3 seeds each, 23 sets. Rule applied as written: A1L minus zero-shot was -0.014 (ExDark), -0.019 (DAWN fog), -0.010 (BDD clear night), all below the required +0.010, and A1L did not beat B1L on ExDark in all seeds (-0.023, +0.001, +0.007). **Transfer not supported**: a negative result for this recipe. Locking batch-norm did not improve B1 and made A1's synthetic gains smaller and less stable. Phase 5 is closed; next is phase 6.

## Phase 6 plan (E1), written 2026-10-24 before any enhanced image or score exists
**Question.** Does classical preprocessing in front of the unchanged zero-shot yolo11n recover accuracy lost to low light, fog or noise, and at what latency cost? This is a different route from phase 5, where training on degradations did not transfer to real photographs.

**Methods** (src/rcv/enhance.py; deterministic, no learned weights, so no new licence): CLAHE and adaptive gamma for low light; dark-channel-prior dehazing for fog; non-local-means denoising for noise. **Not tested:** rain, snow and motion blur, because no credible classical single-image method is offered for them here (derain and deblur normally need learned models; a weak method would only produce an uninformative result). Learned enhancers (for example Zero-DCE) are left for later.

**Sets and controls.** Each source set is evaluated three ways on identical images: the original, a re-encoded identity copy ("none", controls for the extra JPEG generation), and each enhanced version. Sources: low light s2 and s3 and ExDark and BDD clear night (CLAHE, gamma); fog s3 and DAWN fog (dehazing); noise s3 (denoising); and the clean 1000-image sample with all four methods (harm on images that need no enhancement).

**Measures.** Person mAP50-95 with the **paired** bootstrap difference (enhanced minus original, same images, 95 % interval); enhancement time per image on the CPU (median and 95th percentile at native image size) added to the detector's 16.1 ms forward pass.

**Decision rule, fixed now.**
- A method **helps on a condition** if the paired gain is at least +0.01 **and** the interval's lower bound is above 0, on the real set (ExDark for low light, DAWN fog for fog) **and** on the synthetic set. Helping only on the synthetic set is reported as synthetic-only. Noise has no real set, so it can only be synthetic-only.
- **Harm check:** a method with a clean-sample gain below -0.01 whose interval excludes 0 is reported as harmful when applied unconditionally (it would need a gate).
- **Control check:** the re-encode control must be within 0.005 of the original; otherwise comparisons are made against the control instead.
- DAWN fog has only 117 person boxes; a wide interval there counts as inconclusive, not as a null result.
- Latency is reported for every method whatever the accuracy outcome; no threshold is set on it.

### Phase 6 outcome (2026-10-24)
Rule applied as written. Dark-channel dehazing **helps on fog**: synthetic +0.084 [+0.075, +0.091], real DAWN fog +0.039 [+0.007, +0.071] (117 person boxes, lower bound only just above 0); about 51 to 151 ms per image on the CPU against the detector's 16 ms. CLAHE and adaptive gamma **do not help on low light** (ExDark -0.001 and -0.030; synthetic -0.02 and -0.06; the synthetic noise is amplified by brightening). Non-local-means is synthetic-only and harmful on clean images (-0.042). Re-encode control moved some sets by up to 0.008, so gains against the control are also reported. Not tested: rain, snow, motion blur, learned enhancers.
