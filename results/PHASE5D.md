Person mAP50-95, mean ± sample standard deviation over 3 seeds (yolo11n, 10 epochs). B1L and A1L are B1 and A1 with every BatchNorm layer locked.

| Run | clean COCO | low light s3 (synth.) | rain s3 (synth.) | motion blur s3 (synth.) | ExDark (real) | DAWN fog (real) | DAWN snow (real) | BDD clear day (real) | BDD clear night (real) |
|---|---|---|---|---|---|---|---|---|---|
| B0 zero-shot | 0.532 | 0.150 | 0.218 | 0.147 | 0.376 | 0.399 | 0.453 | 0.198 | 0.149 |
| B1 original | 0.506 ± 0.006 | 0.128 ± 0.003 | 0.211 ± 0.002 | 0.117 ± 0.006 | 0.362 ± 0.007 | 0.375 ± 0.003 | 0.432 ± 0.014 | 0.193 ± 0.003 | 0.140 ± 0.003 |
| A1 original | 0.504 ± 0.003 | 0.253 ± 0.006 | 0.363 ± 0.008 | 0.221 ± 0.011 | 0.369 ± 0.008 | 0.402 ± 0.005 | 0.429 ± 0.004 | 0.192 ± 0.007 | 0.139 ± 0.004 |
| B1L locked | 0.510 ± 0.009 | 0.121 ± 0.009 | 0.230 ± 0.012 | 0.113 ± 0.016 | 0.367 ± 0.003 | 0.370 ± 0.010 | 0.441 ± 0.003 | 0.196 ± 0.002 | 0.143 ± 0.002 |
| A1L locked | 0.498 ± 0.016 | 0.224 ± 0.041 | 0.337 ± 0.049 | 0.201 ± 0.040 | 0.362 ± 0.014 | 0.380 ± 0.020 | 0.429 ± 0.013 | 0.190 ± 0.009 | 0.139 ± 0.005 |

**Pre-registered rule, applied as written** (transfer supported if A1L minus zero-shot is at least +0.01 on all three of ExDark, DAWN fog and BDD clear night, and A1L beats B1L on ExDark in all 3 seeds):

| Set | A1L minus zero-shot, per seed | mean | needed |
|---|---|---|---|
| ExDark (real) | -0.029, -0.011, -0.002 | -0.014 | at least +0.010 |
| DAWN fog (real) | -0.041, -0.013, -0.002 | -0.019 | at least +0.010 |
| BDD clear night (real) | -0.016, -0.008, -0.006 | -0.010 | at least +0.010 |
| ExDark, A1L minus B1L | -0.023, +0.001, +0.007 | -0.005 | positive in all 3 seeds |

**Verdict: transfer NOT supported.**

