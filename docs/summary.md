# Robust Computer Vision Under Difficult Conditions: two-page summary

Jayesh Chaudhari · personal project, public data only · full report: [report.md](report.md) · tables and figures: [README](../README.md)

## What was asked
How much does a pretrained YOLO object detector lose on **people** (and animals and vehicles) in low light, fog, rain, snow, motion blur, noise and partial occlusion, and what cheaply recovers accuracy: fine-tuning, condition-aware augmentation, or enhancement preprocessing? The motivation is finding people in poor conditions (disaster response). The work is single-camera RGB and makes **no claim about whether a person is alive or unhurt**.

## How it was tested
- **Models:** pretrained yolo11n, yolo11s, yolo26n (COCO weights), zero-shot, then fine-tuned yolo11n variants.
- **Evidence kinds, always labelled:** *synthetic* (COCO images degraded by my own code, a controlled stress test) and *real* (ExDark low light, DAWN fog, BDD100K night, rain, snow). Real sets were never trained on.
- **Measure:** person mAP50-95 with 95 % bootstrap intervals; paired differences where the same images appear twice; a lenient "found rate" for occlusion. The scoring was cross-checked against pycocotools; 143 automated tests.
- **Honest process:** hypotheses and decision rules for the later experiments were committed to git before the runs. Three seeds per fine-tuning setting; a difference counted as real only if at least 0.01 and the same sign in all three seeds.

## What was found
| Question | Answer |
|---|---|
| How much does the pretrained model lose? | Large under synthetic stress (person mAP50-95 0.53 clean → 0.15 low light, 0.15 motion blur, 0.22 rain, strongest settings). On **real** photographs only night shows a clear drop: BDD100K 0.198 by day, 0.149 at night (-0.049 [-0.070, -0.031]); daytime rain and snow show none. |
| Does occlusion kill detection? | A person with half the box hidden is still found 76 % of the time (95 % when unhidden), although the strict score collapses; at 75 % hidden, 43 %. |
| Does fine-tuning recover accuracy? | It learns the synthetic degradations (+0.07 to +0.17) but **not** real photographs: of 60 comparisons with the matching zero-shot model, only 2 came out ahead (both on the smallest real set), many were reliably below. A negative result for this recipe. |
| Why was the clean fine-tune worse than the model it started from? | About 0.01 of it is a training-procedure side effect: batch-norm statistics drift in training mode, even at a near-zero learning rate. Locking them removed it. The rest comes from learning on the clean subset; no learning rate, freezing or lock removed it. |
| What does each fine-tuning variable do? | Too high a learning rate hurts everywhere; more data is the most reliable lever (gap to zero-shot -0.038 at 2k images to -0.019 at 30k); freezing the backbone preserves real-photo accuracy but removes the synthetic gain; more epochs help only the synthetic sets; a bigger starting model is better but, fine-tuned, still ends below its own zero-shot. |
| Does preprocessing help? | **Dehazing helps on fog** (+0.039 [+0.007, +0.071] on real fog, +0.084 synthetic) but costs 3 to 9 times the detector's time on a CPU; low-light enhancement did not help (brightening amplified noise); denoising hurt clean images. |

## What it means
- The best detector among everything tested for real photographs is the **pretrained yolo11s** (+0.06 over yolo11n on clean images, the same 16 ms on the test GPU), with dehazing in front of it if fog is expected. **No fine-tuned weights are released**, because none is an improvement.
- Real-world robustness cannot be inferred from synthetic degradations: synthetic rain overstated real rain, and augmentation gains did not carry over.
- An informal test of yolo11s on a 44-second flood-rescue video (not redistributed, no ground truth): most people were boxed for most of the clip (judged by watching it, not scored); heads in the water were missed for a few seconds, mostly at the start, and 12 % of frames had no detection at all. Qualitative only.
- For a rover or a rescue aid the detector must be **one layer among several** (range sensing, a manual emergency stop, a human operator), never the only one.

## What it cannot say
One model family and recipe; synthetic degradations that are my own simplified models; a very small real fog set (117 person boxes) and no real rain or snow set with enough people; each dataset labels only its own classes (so correct detections of unlabelled people count against the score); latency from one laptop GPU only; BDD100K files failed their published checksum and ExDark is used without written permission (both recorded). yolo26s and learned enhancers were not tested. Details in report section 9.

## Reproduce
Everything is scripted: `bash scripts/run_sweep.sh`, `python scripts/make_sweep_tables.py`, and the other commands listed in report section 11. A live webcam demo is in `demo/` (local, nothing saved).
