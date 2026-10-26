# Robust Computer Vision Under Difficult Conditions: how a pretrained YOLO detector degrades, and what recovers accuracy

Jayesh Chaudhari. Personal project, public data only. Status: complete (2026-10-26). Code, tables and figures: this repository. Short version: [summary.md](summary.md).

## 1. Question and answer in brief

**Question.** How much does a pretrained YOLO detector lose on **people** (primary class), and on animals and vehicles, in low light, fog, rain, snow, motion blur, noise and partial occlusion, and which inexpensive intervention (fine-tuning, condition-aware augmentation, or enhancement preprocessing) recovers the most accuracy?

**Answer.**
1. The loss is large under my synthetic degradations (person mAP50-95 down by 0.38 for strong low light, 0.39 for motion blur, 0.31 for rain, against a clean score of 0.53) but on **real** photographs only night shows a clear drop (BDD100K: 0.198 in daytime, 0.149 at night, a drop of 0.049 [0.031, 0.070]).
2. **Fine-tuning learned my synthetic degradations (+0.07 to +0.17) but did not beat the pretrained model on real photographs.** Over a 45-run sweep that changed one variable at a time, only 2 of 60 comparisons with the setting's own zero-shot model came out ahead, both on the smallest real set (117 person boxes).
3. **Classical enhancement helps in one case:** dark-channel dehazing in front of the unchanged detector raised accuracy on real fog (+0.039 [+0.007, +0.071]); low-light enhancement did not help.
4. The best detector among everything tested is the **pretrained yolo11s**, with dehazing in front of it for fog. No fine-tuned weights are released, because none is an improvement.

Every claim below carries its limits; section 9 lists what this study cannot say.

## 2. Scope and what is deliberately not claimed

Single-camera RGB. Classes: person (primary); cat, dog, horse, sheep, cow (secondary); car, bus, truck (context). The motivation is finding people in poor conditions (disaster response), but:

- A 2D camera cannot tell whether a person is under, in front of or behind something. The most this work supports is "person detected, possibly partly hidden", for a human to verify.
- **No claim is made about whether anyone is alive, conscious or unhurt.** Thermal imaging, simulated thermal sensing and any life-signs inference are later stages and not part of this study.
- The occlusion test is a proxy: a synthetic slab hides a share of a person's **box**, not of the body, and a slab is not rubble.

## 3. Data and licences

| Data | Use | Terms (details and the disputed claims: `data/README.md`) |
|---|---|---|
| COCO val2017 | clean reference; source of the synthetic degradation and occlusion tests | annotations CC BY 4.0; images follow the Flickr terms and each photo's own licence (secondary-sourced: COCO's own terms page could not be retrieved) |
| COCO train2017 | fine-tuning subsets (2k to 30k images) | as above |
| ExDark | real low light, **test-only** | used on the author's own judgement for personal non-commercial research; no written confirmation from the owners (an email draft exists, unsent) |
| DAWN | real fog, rain, snow, **test-only** | CC BY-NC 3.0 (a third-party summary said CC BY 4.0; the repository record was checked and says BY-NC) |
| BDD100K | real night, rain, snow, clear-day reference; adverse-condition training images (variant A2) | educational, research and non-profit use; **the downloaded files failed the checksum published in the project's docs** (a legacy mirror was used because the official hosts were unreachable), so authenticity is unverified; structure and archive checks pass |
| RTTS | not used | no licence stated |

BDD100K's `rider` class is mapped to `person` (a decision taken before seeing results). No dataset images appear anywhere in this repository.

## 4. Methods

**Evaluation.** Ultralytics validation at 640 px, confidence 0.001, NMS IoU 0.6, on the nine project classes (COCO label space, so the pretrained head is kept and results are comparable). Person AP is computed per image and bootstrapped (images resampled, 95 % intervals); the AP computation reproduces pycocotools to 1e-9 and was cross-checked against pycocotools on the same detections (worst difference 0.024 across classes, explained by multi-label NMS and crowd regions). Intervals for differences are **paired** wherever the same images appear in both conditions.

**Synthetic degradations** (`src/rcv/corrupt.py`): low light, fog, rain, snow, motion blur, Gaussian noise, three severities each, deterministic from a hash seed. **Occlusion** (`src/rcv/occlude.py`): a slab hides 25, 50 or 75 % of a person's box. The severities are my own settings, so the ranking *between* conditions says nothing about real weather. Strict mAP50-95 on occlusion is inflated downward because a detector boxes what it can *see*; a lenient **found rate** (a detection of a person with IoU at least 0.1 and confidence at least 0.25) is reported alongside.

**Models.** Pretrained yolo11n (2.6 M parameters), yolo11s (9.5 M), yolo26n (2.6 M). Batch-1 forward pass on an RTX 3050 Laptop GPU: 16.1, 16.4 and 22.0 ms.

**Training variants** (yolo11n, 10 epochs, labels in COCO-80 space): B1 clean control (9,831 COCO images); A1 adds a condition-aware augmentation that degrades each image with probability 0.5 using the same generator as the synthetic tests; A2 adds 3,130 BDD100K adverse-condition images. Checkpoints were chosen on a held-out 500-image COCO train split; the test sets were never trained on.

**Pre-registration.** For the diagnosis phases, the enhancement experiment and the final sweep, hypotheses and decision rules were committed before the runs (git history shows the order). A difference counted as **real** if its mean over three seeds was at least 0.01 in absolute value and had the same sign in every seed. This is a conservative screen, not a significance test; with 15 settings and 8 sets a few chance flags are expected.

## 5. Results: how much does the pretrained detector lose?

Person mAP50-95, yolo11n zero-shot (intervals are 95 % bootstrap):

| Condition | Evidence | Score | Drop |
|---|---|---|---|
| Clean COCO sample (1,000 images) | clean | 0.532 [0.509, 0.551] | |
| Low light, strongest | synthetic | 0.150 | -0.382 |
| Motion blur, strongest | synthetic | 0.147 | -0.386 |
| Rain, strongest | synthetic | 0.218 | -0.314 |
| Noise / fog / snow, strongest | synthetic | 0.335 / 0.393 / 0.408 | -0.197 / -0.139 / -0.124 |
| BDD100K clear day | real | 0.198 [0.184, 0.213] | reference |
| BDD100K clear night | real | 0.149 | -0.049 [-0.070, -0.031] |
| BDD100K rainy / snowy day | real | no detectable drop | +0.001 [-0.029, +0.029] / -0.026 [-0.056, +0.004] |
| ExDark low light | real | 0.376 [0.366, 0.386] | no clean reference |
| DAWN fog | real | 0.399 [0.309, 0.492] (117 person boxes) | no clean reference |

Observations: synthetic rain overstates what real rain does in BDD100K (possible reasons, not separated: harsher synthetic rain, a low daytime baseline of 0.198 because of small distant pedestrians, per-video weather tags). A bigger model raises every score (yolo11s: 0.594 clean) but does not remove the synthetic low-light and blur losses.

**Occlusion.** At 50 % of the box hidden, strict person mAP50-95 falls from 0.721 to 0.064, but a person is still found in 0.761 [0.727, 0.796] of cases (0.950 unoccluded); at 75 % hidden, 0.431 [0.393, 0.472], mostly with loose overlap.

## 6. Results: does fine-tuning recover accuracy?

Three seeds per variant (mean ± standard deviation):

- **A1 against the clean control B1:** +0.125 ± 0.004 on strongest synthetic low light, +0.152 ± 0.006 on rain, +0.105 ± 0.007 on motion blur. These are **in-distribution** gains (training and test degradations share a generator).
- **On real photographs** the A1 gain over B1 was small: ExDark +0.008 ± 0.005, DAWN fog +0.026 ± 0.007, BDD100K about zero.
- **A2** (adding BDD100K images) gained +0.029 ± 0.012 on BDD night over A1, but the BDD test images come from the same source as its training images, and A2 has more data, so this is in-domain and confounded.
- **The clean control scored below zero-shot** (clean 0.506 against 0.532).

### Diagnosis (Phase 5)

Why was the clean control worse than the model it started from? Controlled runs (seed 0):

- A run with essentially **no learning** (learning rate near zero) already lost about 0.01 on clean and on ExDark, with or without augmentation. The cause: in training mode every BatchNorm layer re-estimates its running statistics on the small training batches. Locking all of them restored zero-shot scores (clean 0.529, ExDark 0.381; the ExDark gap of +0.005 was just inside the tolerance set beforehand).
- A lower learning rate, a frozen backbone and the batch-norm lock **did not remove the rest** of the loss, which comes from learning on the clean subset.
- With the side effect removed, three seeds of condition-aware augmentation (A1L) were **below zero-shot on all three real sets the rule named** (ExDark -0.014, DAWN fog -0.019, BDD night -0.010; needed +0.010). **Transfer was not supported**: a negative result for this recipe.

### Fine-tuning sweep (Phase 8)

45 runs: 15 settings x 3 seeds, one variable changed at a time from a fixed baseline (the explicit baseline replicate reproduced the earlier A1 runs within seed spread). Full tables and figures are in the README.

| Variable | What it did |
|---|---|
| Learning rate | 1e-3 hurt every set; 3e-4 hurt clean and several real sets; 2e-5 and 6e-5 were within noise of the baseline on clean and real sets and learned the synthetic degradations less (about -0.03). Cosine decay made no reportable difference. |
| Epochs | 5 under-trained (-0.022 on low light, -0.024 on rain). 30 added +0.012 and +0.015 on those synthetic sets, nothing on real photographs, at three times the training time. |
| Training images | The most reliable lever: the gap to zero-shot on clean images shrank from -0.038 (2k) to -0.019 (30k), at 6 to 68 minutes of training. It did not close the gap. |
| Frozen layers | Freezing 22 layers kept real-photo accuracy within 0.002 of zero-shot but removed the synthetic gain (low light +0.001 over zero-shot). |
| Starting model | yolo11s beat yolo11n by a wide margin at the same recipe, but **fine-tuned yolo11s ended below its own zero-shot model on every real set**, and took five times longer to train (121 against 24 minutes). yolo26n behaved like yolo11n. |

**Bottom line:** against each setting's own zero-shot model, only 2 of 60 setting-by-real-set comparisons came out ahead (30k images and yolo26n, both on DAWN fog), and many were reliably below. Hypotheses: more data narrows the clean loss (supported in direction, not closed); freezing protects real-photo accuracy at the cost of the synthetic gain (supported), a lower learning rate alone does not; a larger model is better than yolo11n at the same recipe (supported except on the 50 %-hidden set); more epochs help only synthetic sets (supported).

## 7. Results: enhancement preprocessing (Phase 6)

Classical, deterministic methods applied before the unchanged zero-shot yolo11n; paired gains in person mAP50-95 with 95 % intervals:

| Method | Synthetic | Real | Cost (CPU, per image) | Verdict |
|---|---|---|---|---|
| Dark-channel dehazing | fog +0.084 [+0.075, +0.091] | DAWN fog +0.039 [+0.007, +0.071] | 51 ms (synthetic), 151 ms (DAWN photographs) | **helps** (thin real set) |
| CLAHE | low light -0.021 | ExDark -0.001 [-0.003, +0.002]; BDD night +0.005 | 2 to 5 ms | does not help |
| Adaptive gamma | low light -0.056 | ExDark -0.030 | about 1 ms | hurts |
| Non-local-means denoising | noise +0.029 | none available | about 550 ms | synthetic-only; harmful on clean images (-0.042) |

Brightening amplified the noise in the synthetic low-light images (measured about doubling), which is the likely reason low-light enhancement hurt. A re-encode control moved some sets by up to 0.008, so gains against the control are also tabulated in the README; no verdict changed. Rain, snow, motion blur and learned enhancers were not tested.

## 8. An informal test on a flood-rescue video

A 44-second third-party phone video of a flood rescue (not redistributed; not part of the evaluation) was run through the pretrained yolo11s at 640 px and 1280 px. There is no ground truth, so this is qualitative. Frames with no person detected at confidence 0.35 totalled 5.5 s (12 %) at 640 px, in short gaps mostly within the first 4 seconds, where the only person is a small head far out in the water; people standing or wading were boxed with scores of about 0.5 to 0.9 in the frames inspected. Two heads at the end were found at 1280 px (confidence about 0.4) but missed at 640 px, and 1280 px flickered more. This was reviewed by watching the annotated video, not scored. It is consistent with the measured weak spots (small, partly hidden, low-contrast people) and with the rule below: **the detector must never be the only safety layer.**

## 9. Limitations and threats to validity

- **Synthetic degradations are simplified models.** Ranking between conditions depends on my severity choices; training and test degradations share one generator, so synthetic gains from augmentation are optimistic.
- **Real-photo evidence is thin where it matters most:** DAWN fog has 117 person boxes (28 in rain), BDD100K has 13 foggy images, there is no real-photo animal evidence outside ExDark low light, and no real rain or snow set with enough persons.
- **Each dataset labels only its own classes,** so a correctly detected but unlabelled person counts as a false positive and lowers every real-set score.
- **One model family and recipe.** yolo11n for training; yolo11s and yolo26n starting points were not tuned (their learning rate was left at the baseline); yolo26s was not evaluated; the 80-class label space, optimizer type, image size and augmentation strength were not varied. Known confounds in the sweep: data size at a fixed number of epochs also changes the number of gradient steps; epochs 5 and 30 change how many epochs use mosaic augmentation; freezing also freezes BatchNorm statistics.
- **Latency was measured on one laptop GPU** (forward pass only); a rover's embedded computer may rank models differently.
- **Data provenance:** BDD100K failed its published checksum; ExDark is used without written permission; both are listed in `data/README.md`.
- **The "real effect" rule is a screen, not a hypothesis test.**

## 10. Safety, ethics and licences

No alive/dead or life-signs claim is made. Fine-tuned weights are not released: none is an improvement over the pretrained model, and weights trained on BDD100K or tested against ExDark and DAWN carry terms I would rather not extend. Ultralytics code and weights are AGPL-3.0. Personal project on personal time and hardware, with public data only; no employer code, data or hardware is used. No dataset images are published here, and the flood video is third-party footage that stays local.

## 11. Reproducing this work

```
pip install -r requirements.txt            # see requirements-lock.txt for tested versions
python scripts/download_data.py coco dawn exdark bdd100k   # read data/README.md for each licence first
python scripts/prepare_coco.py && python scripts/make_synthetic.py && python scripts/prepare_real_sets.py && python scripts/prepare_bdd.py
bash scripts/run_model_comparison.sh        # zero-shot baselines, three models
python scripts/make_results_tables.py --update-readme
python scripts/prepare_train_data.py && python scripts/prepare_sweep_data.py
bash scripts/run_finetune_all.sh            # B1, A1, A2 x 3 seeds
bash scripts/run_phase5.sh; bash scripts/run_phase5b.sh; bash scripts/run_phase5c.sh; bash scripts/run_lockbn_all.sh
python scripts/make_enhanced_sets.py && python scripts/run_enhancement.py
bash scripts/run_sweep.sh                   # the 45-run sweep (about 30 hours on a 4 GB laptop GPU)
python scripts/make_sweep_tables.py --update-readme
pytest                                      # 143 tests
```

Per-run result files are in `results/` (JSON), per-epoch curves in `results/curves/`, figures in `results/figures/`. The pre-registered plans and their outcomes are in `PHASES.md`, in the order they were written.
