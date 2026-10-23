# Robust Computer Vision Under Difficult Conditions

**Status: In progress** (tooling stage; no experimental results yet)

How much does a pretrained YOLO detector degrade on **people** (primary) and animals (secondary) in low light, fog, rain, snow, motion blur, noise and partial occlusion, and which cheap intervention (fine-tuning, condition-aware augmentation, enhancement preprocessing) recovers the most accuracy per millisecond of latency?

The motivation is disaster response: finding people in poor conditions. Part 1 of 3 in a connected portfolio: computer vision, then ROS 2 navigation ([ros2-slam-navigation](https://github.com/jayesh518/ros2-slam-navigation)), then a physical rover ([autonomous-rescue-rover](https://github.com/jayesh518/autonomous-rescue-rover)).

## Scope (staged on purpose)
**This project: single-camera RGB.** Thermal sensing, simulated thermal and any life-signs inference are later stages, not claims of this repository.

| Role | Classes (all COCO classes) | How reported |
|---|---|---|
| Primary | `person` | headline results, always first |
| Secondary | `cat, dog, horse, sheep, cow` | second table |
| Context | `car, bus, truck` | third table; lets an alert say "person partly hidden near a vehicle" |

- **Conditions:** low light, fog, rain, snow, motion blur, gaussian noise, and **partial occlusion** (a synthetic concrete-like slab hiding 25 / 50 / 75 % of a person's box).
- **Model:** `yolo11n` (and `yolo11s` if VRAM allows), 640 px, on an RTX 3050 Laptop GPU (4 GB).

## What this can and cannot say
- A 2D camera cannot distinguish "under" from "in front of" or "behind". The most this project can support is wording like *"person detected, partly hidden, overlapping a vehicle"*, always for a human to verify.
- The occlusion test is a **proxy** for debris: it hides a share of the **box**, not of the body, and a slab is not rubble.
- **No alive/dead or life-signs claim is made or planned.** A thermal camera reads surface heat only; it cannot see through concrete or rubble, and clothing, ambient temperature and the slow cooling of a body make temperature unreliable. A wrong "dead" label could cost a life. If thermal work is added later, the only output considered is "heat signature consistent with a person: present / absent / uncertain", as a hint for a human operator.

## What is real and what is synthetic
Every reported number will be labelled accordingly.

| Evidence | Classes | Type |
|---|---|---|
| ExDark (low light) | person, cat, dog, car, bus | real photographs |
| RTTS (haze) | person, car, bus | real photographs |
| DAWN (fog, rain, snow) | person, car, bus, truck | real photographs |
| BDD100K (night, rain, snow tags) | person, car, bus, truck | real photographs |
| COCO val, degraded by `src/rcv/corrupt.py` / occluded by `src/rcv/occlude.py` | all 9 | **synthetic** (simplified models, not real weather or rubble) |

**Limitation:** no public dataset provides real fog/rain/snow images with animal boxes, so horse, cow and sheep (and cats and dogs outside low light) are evaluated under synthetic degradations only. Synthetic results are a controlled stress test, not a substitute for real conditions.

## Experiment variants (planned)
| ID | Description |
|---|---|
| B0 | Pretrained, zero-shot (baseline) |
| B1 | Fine-tuned on a clean subset only (control for the effect of fine-tuning itself) |
| A1 | Fine-tuned with condition-aware augmentation |
| A2 | A1 plus BDD100K adverse-condition training splits (person and vehicles only: BDD100K has no animals). ExDark and DAWN are test-only. |
| E1 | Enhancement preprocessing (low-light, dehaze, derain) + B0, measuring accuracy gain vs added latency |

Metrics: mAP50-95, mAP50, precision, recall (per class, per condition, per severity, per occlusion level), batch-1 latency, FPS, model size. Trained variants use 3 seeds (mean +/- std) and bootstrap CIs on differences. Negative results will be reported.

## Later stages (not part of this repository's claims yet)
1. Simulated thermal camera and humans in Gazebo (Project 2), as a controlled test.
2. RGB vs thermal vs fusion on real paired datasets (e.g. LLVIP, FLIR ADAS, HIT-UAV; licenses unverified).
3. Debris/slab detection needs its own dataset and labels; not in scope here.

## Setup
```
python -m venv .venv
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.7.1 torchvision==0.22.1
.venv\Scripts\python -m pip install -r requirements.txt
# requirements-lock.txt lists the exact versions that passed the tests (use it to reproduce an environment exactly)
.venv\Scripts\python -m pytest -q            # unit tests (CPU)
.venv\Scripts\python scripts\smoke_test.py   # GPU + pipeline check, writes results\smoke.json (not an accuracy result)
```

## Structure
```
configs/    classes.yaml (roles, label spaces), conditions.yaml (degradations, occlusion), datasets.yaml (sources, label maps)
src/rcv/    labels.py, corrupt.py, occlude.py, stats.py, eval.py
scripts/    smoke_test.py
tests/      label mapping, degradation determinism/monotonicity, occlusion exactness, statistics
data/       not committed; see data/README.md for sources and licenses
models/     not committed; see models/README.md
results/    small, publishable summaries (JSON/CSV); smoke.json is an environment check only
docs/       report.md (written once results exist)
```

### B0 on REAL adverse-condition photographs (test-only sets, 2026-10-23)
Source: `results/baseline_b0_yolo11n_real.json`. Same zero-shot `yolo11n`, same protocol. These are real photographs, not synthetic.

| Set (condition) | Images | Person boxes | Person mAP50-95 | Person mAP50 | Person recall | Other classes (mAP50-95) |
|---|---|---|---|---|---|---|
| ExDark (low light) | 4,906 | 7,460 | **0.375** | 0.743 | 0.681 | cat 0.340 (909 boxes), dog 0.357 (1,018), car 0.380, bus 0.574 |
| DAWN fog | 299 | 117 | **0.398** | 0.642 | 0.615 | vehicles 0.316 |
| DAWN snow | 204 | 158 | **0.452** | 0.742 | 0.633 | vehicles 0.317 |
| DAWN rain | 200 | **28** | 0.554 | 0.774 | 0.853 | vehicles 0.298 |

How to read this, and what it does NOT show:
- **There is no clean reference for these sets, so no "drop" can be computed.** The numbers cannot be compared with the COCO clean value (0.532): different image sources, object sizes and annotation rules. In particular ExDark and DAWN annotate only their own class lists, so an unannotated person that the detector correctly finds counts as a false positive, which lowers every score. A fair within-dataset reference (clear daytime versus night, rain, snow, fog) needs BDD100K, which is being downloaded.
- **DAWN rain has only 28 person boxes.** Its 0.554 and 0.853 recall are not meaningful; do not quote them. DAWN is a vehicle dataset and its person counts are small everywhere (117 and 158 for fog and snow are also thin).
- ExDark is the first **real** evidence for the secondary class: cat and dog reach about 0.35 mAP50-95 in real low light. That is a measurement of the baseline on this set, not a degradation figure.
- Synthetic low light at severity 2 happens to score about the same (0.378) as real ExDark (0.375). That is a coincidence of magnitude between two different image sets and does not show that the synthetic model is realistic.

## Evaluation protocol, and how it was checked
All numbers come from Ultralytics `val` (confidence 0.001, NMS IoU 0.6, 640 px, multi-label NMS, no special handling of COCO "crowd" regions). `scripts/crosscheck_coco_eval.py` re-scores the same detections with the official `pycocotools` on the original COCO annotations. On the 1000-image sample the per-class AP50-95 agrees to within 0.024 (person 0.014, sheep 0.024, cow 0.020, all others 0.009 or less); the pycocotools value is higher for the crowd-heavy classes because it ignores detections that match crowd regions. See `results/crosscheck_coco.json`.

Two things to know when reading any number here:
- **NMS mode matters.** Scoring `predict` output (one class per box) instead of `val` output (multi-label) lowers AP for confusable classes by up to 0.07 (dog 0.703 -> 0.631; cat and truck about 0.03). The first run of the cross-check compared the wrong detections and failed because of this (`results/crosscheck_coco_attempt1_predict_reference.json` is kept as the record). All results here use the `val` protocol.
- **Small classes have wide uncertainty.** Cat, dog, horse, sheep and cow have few instances in the 1000-image sample, so their per-class AP is noisy. Bootstrap intervals will be reported with the results.

## Label spaces (a classic source of silent bugs)
Three numberings exist and are kept apart in `configs/classes.yaml`: the project id (0..8, used by fine-tuned models), the COCO 80-class index (what a COCO-pretrained model outputs) and the category id inside COCO's annotation JSON (person=1, car=3, cat=17, ...). `src/rcv/labels.py` converts between them and is unit-tested.

## Results
### B0: zero-shot `yolo11n`, synthetic degradations (first baseline, 2026-10-23)
Source: `results/baseline_b0_yolo11n.json` (regenerate with `scripts/make_synthetic.py` then `scripts/run_baseline.py`). **All condition results are SYNTHETIC** (degraded COCO val images, simplified models in `src/rcv/corrupt.py`), measured on a fixed 1000-image sample, with the **same images** used for the clean reference. One deterministic run: no confidence intervals yet.

Person (primary class), mAP50-95. Clean reference: **0.532**.

| Condition | severity 1 | severity 2 | severity 3 |
|---|---|---|---|
| low light | 0.475 | 0.378 | 0.147 |
| fog | 0.507 | 0.469 | 0.392 |
| rain | 0.504 | 0.460 | 0.215 |
| snow | 0.514 | 0.477 | 0.407 |
| motion blur | 0.434 | 0.274 | 0.143 |
| gaussian noise | 0.502 | 0.442 | 0.333 |

Reading this correctly:
- Severities are **my own parameter choices** (`configs/conditions.yaml`). "Rain s3" versus "fog s3" says nothing about real rain versus real fog; only the trend within a condition is meaningful. The first real-condition tests (ExDark, DAWN, BDD100K) are still to come.
- Animals (secondary) fall faster than people at strong severities (for example rain s3: animals 0.049, person 0.215), but animals have far fewer instances in the sample, so treat that as a lead, not a finding, until bootstrap intervals exist.

### Occlusion (concrete-like slab hides part of each person's box; 500 images, every person at least 40 px)
| Hidden fraction | mAP50-95 | mAP50 |
|---|---|---|
| 0 (reference) | 0.722 | 0.924 |
| 25 % | 0.373 | 0.805 |
| 50 % | 0.063 | 0.286 |
| 75 % | 0.047 | 0.080 |

**Do not read this as "the detector fails at 50 % occlusion".** The ground truth is the person's **full** box, but a detector draws a box around what it can **see**. Once half the box is hidden, even a perfect detection of the visible half has IoU of about 0.5 with the full box, so mAP50-95 collapses even though the person was found (mAP50 at 25 % stays at 0.805 while mAP50-95 halves). For the rescue question "was a person found at all?", a lenient overlap threshold (for example IoU 0.1 to 0.3) is the right measure. **That metric is the next thing to add**; until it exists the occlusion numbers above only show localisation degrading.

## Licenses and attribution
- **Ultralytics YOLO code and weights are AGPL-3.0.** Fine for this research; relevant if the code is reused commercially.
- Each dataset has its own terms, some research-only. They are listed in `data/README.md` and must be verified before use. No data or weights are redistributed from this repository.
- This is a personal project, built on personal time and hardware with public data only. It uses no employer code, data or hardware.
