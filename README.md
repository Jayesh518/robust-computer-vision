# Robust Computer Vision Under Difficult Conditions

**Status: Planned**

Investigates how a pretrained YOLO-family detector performs in difficult visual conditions (low light, motion blur, occlusion, clutter) and whether fine-tuning improves the accuracy/speed trade-off.

Part 1 of 3 in a connected portfolio: computer vision, then ROS 2 navigation ([ros2-slam-navigation](https://github.com/jayesh518/ros2-slam-navigation)), then a physical rover ([autonomous-rescue-rover](https://github.com/jayesh518/autonomous-rescue-rover)). Portfolio site: [engineering-portfolio](https://github.com/jayesh518/engineering-portfolio).

## Planned scope
- Baseline: pretrained YOLO detector (weights source and license documented).
- Evaluate on images/videos per visual condition.
- Fine-tune on a curated, license-checked dataset; compare against baseline on a held-out test set.
- Optional: tracking and ONNX/TensorRT deployment.

## Metrics (reported only once measured)
Precision, recall, mAP50-95, per-class results, latency, FPS, model size, per-condition breakdown.

## Structure
```
src/        inference and evaluation code
data/       not committed; see data/README.md for how to obtain
models/     not committed; checkpoints and download notes
docs/       report.md and figures
```

## Results
None yet. This section will be filled from real experiment logs.

## Notes
Uses pretrained weights; no model is trained from scratch. Datasets and weights are not redistributed.
