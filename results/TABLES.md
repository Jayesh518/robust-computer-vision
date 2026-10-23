Models evaluated zero-shot: yolo11n, yolo11s, yolo26n. Intervals are 95 % bootstrap CIs (images resampled, 200 draws). Evidence labels: **synthetic** = degraded COCO val images; **real** = genuine photographs; **clean** = reference.

### Model comparison on clean images (COCO val, 1000-image sample; mAP50-95)

| Model | Person (primary) | Animals (secondary, macro) | Vehicles (context, macro) | Latency batch 1 |
|---|---|---|---|---|
| yolo11n | 0.532 [0.509, 0.551] | 0.612 [0.575, 0.660] | 0.408 [0.363, 0.475] | 16.1 ms median (2.6 M params) |
| yolo11s | 0.594 [0.573, 0.611] | 0.678 [0.645, 0.722] | 0.517 [0.465, 0.577] | 16.4 ms median (9.5 M params) |
| yolo26n | 0.528 [0.507, 0.547] | 0.637 [0.600, 0.682] | 0.440 [0.388, 0.496] | 22.0 ms median (2.6 M params) |

### Synthetic degradation, strongest severity (s3): person mAP50-95 and drop from clean

Severities are my own settings (`configs/conditions.yaml`): compare a condition with itself across models, not one condition against another.

| Condition (synthetic) | yolo11n: s3 (drop) | yolo11s: s3 (drop) | yolo26n: s3 (drop) |
|---|---|---|---|
| low light | 0.150 (-0.382) | 0.207 (-0.386) | 0.155 (-0.373) |
| fog | 0.393 (-0.139) | 0.496 (-0.098) | 0.411 (-0.116) |
| rain | 0.218 (-0.314) | 0.341 (-0.252) | 0.232 (-0.295) |
| snow | 0.408 (-0.124) | 0.498 (-0.096) | 0.414 (-0.113) |
| motion blur | 0.147 (-0.386) | 0.188 (-0.406) | 0.147 (-0.381) |
| gaussian noise | 0.335 (-0.197) | 0.422 (-0.172) | 0.357 (-0.171) |

### yolo11n: person mAP50-95 by condition and severity (synthetic), with CIs

| Condition | clean | s1 | s2 | s3 |
|---|---|---|---|---|
| low light | 0.532 [0.509, 0.551] | 0.475 [0.455, 0.494] | 0.379 [0.359, 0.398] | 0.150 [0.136, 0.166] |
| fog | 0.532 [0.509, 0.551] | 0.507 [0.485, 0.526] | 0.469 [0.448, 0.487] | 0.393 [0.375, 0.413] |
| rain | 0.532 [0.509, 0.551] | 0.504 [0.482, 0.523] | 0.460 [0.442, 0.479] | 0.218 [0.201, 0.234] |
| snow | 0.532 [0.509, 0.551] | 0.514 [0.493, 0.533] | 0.477 [0.457, 0.495] | 0.408 [0.389, 0.429] |
| motion blur | 0.532 [0.509, 0.551] | 0.434 [0.413, 0.454] | 0.276 [0.256, 0.294] | 0.147 [0.133, 0.162] |
| gaussian noise | 0.532 [0.509, 0.551] | 0.502 [0.480, 0.520] | 0.442 [0.423, 0.460] | 0.335 [0.315, 0.352] |

### Occlusion (synthetic slab hides part of each person's box; 500 images, all persons at least 40 px)

The strict score uses the person's FULL box as ground truth, but a detector boxes what it can SEE, so it falls even when the person is found. The **found rate** (a detection of a person at confidence 0.25 or more overlapping the full box with the stated IoU) is the lenient measure for "was a person detected there at all?".

| Model | Hidden | mAP50-95 | mAP50 | found IoU>=0.1 | found IoU>=0.3 | found IoU>=0.5 |
|---|---|---|---|---|---|---|
| yolo11n | 0 % | 0.721 [0.699, 0.747] | 0.924 | 0.950 [0.935, 0.965] | 0.913 [0.894, 0.933] | 0.885 [0.863, 0.909] |
| yolo11n | 25 % | 0.372 [0.353, 0.395] | 0.805 | 0.882 [0.860, 0.909] | 0.817 [0.788, 0.850] | 0.751 [0.718, 0.787] |
| yolo11n | 50 % | 0.064 [0.057, 0.073] | 0.286 | 0.761 [0.727, 0.796] | 0.642 [0.604, 0.688] | 0.345 [0.313, 0.382] |
| yolo11n | 75 % | 0.047 [0.042, 0.060] | 0.080 | 0.431 [0.393, 0.472] | 0.116 [0.092, 0.141] | 0.078 [0.059, 0.100] |
| yolo11s | 0 % | 0.772 [0.755, 0.794] | 0.950 | 0.960 [0.945, 0.973] | 0.943 [0.923, 0.960] | 0.919 [0.899, 0.940] |
| yolo11s | 25 % | 0.388 [0.370, 0.412] | 0.814 | 0.901 [0.879, 0.924] | 0.846 [0.818, 0.875] | 0.777 [0.745, 0.811] |
| yolo11s | 50 % | 0.060 [0.053, 0.070] | 0.304 | 0.809 [0.781, 0.837] | 0.687 [0.651, 0.725] | 0.382 [0.348, 0.417] |
| yolo11s | 75 % | 0.039 [0.033, 0.047] | 0.066 | 0.516 [0.478, 0.556] | 0.112 [0.092, 0.134] | 0.068 [0.052, 0.087] |
| yolo26n | 0 % | 0.729 [0.705, 0.754] | 0.927 | 0.951 [0.936, 0.967] | 0.914 [0.892, 0.935] | 0.885 [0.862, 0.911] |
| yolo26n | 25 % | 0.374 [0.358, 0.396] | 0.807 | 0.888 [0.865, 0.913] | 0.817 [0.784, 0.851] | 0.744 [0.711, 0.782] |
| yolo26n | 50 % | 0.079 [0.072, 0.089] | 0.313 | 0.759 [0.729, 0.793] | 0.647 [0.612, 0.690] | 0.362 [0.328, 0.398] |
| yolo26n | 75 % | 0.068 [0.055, 0.080] | 0.099 | 0.441 [0.401, 0.482] | 0.123 [0.102, 0.146] | 0.091 [0.072, 0.112] |

### REAL photographs (test-only): person mAP50-95, with person-box counts

BDD100K rows give the drop against the BDD **clear-daytime** reference (different images, so the interval comes from independent resampling). ExDark and DAWN have no clean reference, so no drop can be computed for them. Sets with few person boxes carry wide intervals: read the interval, not the point.

| Set (condition) | Person boxes | yolo11n | yolo11s | yolo26n | Drop vs reference (yolo11n) |
|---|---|---|---|---|---|
| ExDark (low light) | 7460 | 0.376 [0.366, 0.386] | 0.406 [0.397, 0.417] | 0.376 [0.366, 0.386] | - |
| DAWN fog | 117 | 0.399 [0.309, 0.492] | 0.486 [0.399, 0.563] | 0.388 [0.302, 0.481] | - |
| DAWN rain | 28 | 0.554 [0.426, 0.688] | 0.544 [0.409, 0.675] | 0.517 [0.373, 0.653] | - |
| DAWN snow | 158 | 0.453 [0.393, 0.528] | 0.515 [0.454, 0.588] | 0.468 [0.406, 0.545] | - |
| BDD clear day (reference) | 2624 | 0.198 [0.184, 0.213] | 0.263 [0.246, 0.278] | 0.198 [0.184, 0.215] | - |
| BDD clear night | 2328 | 0.149 [0.136, 0.164] | 0.191 [0.179, 0.206] | 0.142 [0.130, 0.156] | +0.049 [+0.031, +0.070] |
| BDD rainy day | 565 | 0.197 [0.173, 0.225] | 0.236 [0.213, 0.267] | 0.192 [0.167, 0.221] | +0.001 [-0.029, +0.029] |
| BDD snowy day | 708 | 0.223 [0.195, 0.254] | 0.296 [0.265, 0.330] | 0.220 [0.193, 0.251] | -0.026 [-0.056, +0.004] |
| BDD rainy night (vs clear night) | 235 | 0.109 [0.078, 0.154] | 0.127 [0.094, 0.173] | 0.107 [0.079, 0.152] | +0.040 [-0.014, +0.073] |
| BDD snowy night (vs clear night) | 288 | 0.157 [0.113, 0.208] | 0.206 [0.163, 0.258] | 0.160 [0.114, 0.213] | -0.009 [-0.063, +0.035] |
| BDD foggy (13 images) | 5 | 0.146 [0.006, 0.315] | 0.087 [0.035, 0.132] | 0.171 [0.045, 0.303] | - |

### REAL low light (ExDark): the other classes, macro mAP50-95 (cat and dog are the only real-photo animal evidence)

| Model | Animals: cat, dog | Vehicles: car, bus |
|---|---|---|
| yolo11n | 0.350 [0.334, 0.369] | 0.477 [0.464, 0.492] |
| yolo11s | 0.402 [0.387, 0.421] | 0.520 [0.506, 0.533] |
| yolo26n | 0.361 [0.347, 0.379] | 0.491 [0.478, 0.504] |

