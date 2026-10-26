Baseline: A1 recipe on yolo11n (explicit AdamW lr 0.000119, momentum 0.9, 10 epochs, 9,831 images, nothing frozen). Seeds finished so far per setting: 3 to 3 of 3. Cells are person mAP50-95, mean ± sample std over the seeds present. **Effect** columns are the mean paired difference to the baseline: **REAL** = at least 0.01 in absolute value and the same sign in all 3 seeds; otherwise noise (or pending until 3 seeds exist). Zero-shot yolo11n for reference: clean COCO 0.532, low light s3 (synth.) 0.150, rain s3 (synth.) 0.218, 50 % hidden (synth.) 0.064, ExDark (real) 0.376, DAWN fog (real) 0.399, BDD day (real) 0.198, BDD night (real) 0.149.

#### Learning rate (AdamW)

| Level | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) | training time | train-val mAP50-95 (last epoch) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2e-5 | 0.514 ± 0.001 (n=3) | 0.222 ± 0.011 (n=3) | 0.338 ± 0.012 (n=3) | 0.086 ± 0.012 (n=3) | 0.365 ± 0.004 (n=3) | 0.393 ± 0.002 (n=3) | 0.195 ± 0.004 (n=3) | 0.144 ± 0.005 (n=3) | 24 min | 0.479 |
| 6e-5 | 0.509 ± 0.004 (n=3) | 0.230 ± 0.018 (n=3) | 0.341 ± 0.019 (n=3) | 0.088 ± 0.009 (n=3) | 0.364 ± 0.010 (n=3) | 0.392 ± 0.005 (n=3) | 0.192 ± 0.006 (n=3) | 0.143 ± 0.010 (n=3) | 24 min | 0.476 |
| 1.2e-4 (baseline) | 0.506 ± 0.002 (n=3) | 0.257 ± 0.001 (n=3) | 0.370 ± 0.001 (n=3) | 0.099 ± 0.009 (n=3) | 0.368 ± 0.006 (n=3) | 0.399 ± 0.021 (n=3) | 0.192 ± 0.002 (n=3) | 0.138 ± 0.003 (n=3) | 24 min | 0.458 |
| 3e-4 | 0.490 ± 0.003 (n=3) | 0.253 ± 0.006 (n=3) | 0.366 ± 0.006 (n=3) | 0.110 ± 0.010 (n=3) | 0.350 ± 0.006 (n=3) | 0.386 ± 0.009 (n=3) | 0.182 ± 0.003 (n=3) | 0.125 ± 0.008 (n=3) | 24 min | 0.432 |
| 1e-3 | 0.454 ± 0.001 (n=3) | 0.237 ± 0.004 (n=3) | 0.346 ± 0.003 (n=3) | 0.070 ± 0.007 (n=3) | 0.326 ± 0.006 (n=3) | 0.345 ± 0.004 (n=3) | 0.164 ± 0.000 (n=3) | 0.097 ± 0.004 (n=3) | 24 min | 0.348 |

Effect against the baseline:

| Level | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) |
|---|---|---|---|---|---|---|---|---|
| 2e-5 | +0.007 noise | -0.034 **REAL** | -0.032 **REAL** | -0.014 noise | -0.002 noise | -0.006 noise | +0.002 noise | +0.006 noise |
| 6e-5 | +0.002 noise | -0.026 **REAL** | -0.029 **REAL** | -0.011 **REAL** | -0.004 noise | -0.007 noise | -0.000 noise | +0.005 noise |
| 1.2e-4 (baseline) | baseline | baseline | baseline | baseline | baseline | baseline | baseline | baseline |
| 3e-4 | -0.016 **REAL** | -0.004 noise | -0.004 noise | +0.011 noise | -0.018 **REAL** | -0.014 noise | -0.010 **REAL** | -0.014 **REAL** |
| 1e-3 | -0.053 **REAL** | -0.020 **REAL** | -0.024 **REAL** | -0.029 **REAL** | -0.042 **REAL** | -0.054 **REAL** | -0.028 **REAL** | -0.041 **REAL** |

![Learning rate (AdamW)](results/figures/sweep_lr.png)

#### Learning-rate schedule

| Level | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) | training time | train-val mAP50-95 (last epoch) |
|---|---|---|---|---|---|---|---|---|---|---|
| linear (baseline) | 0.506 ± 0.002 (n=3) | 0.257 ± 0.001 (n=3) | 0.370 ± 0.001 (n=3) | 0.099 ± 0.009 (n=3) | 0.368 ± 0.006 (n=3) | 0.399 ± 0.021 (n=3) | 0.192 ± 0.002 (n=3) | 0.138 ± 0.003 (n=3) | 24 min | 0.458 |
| cosine | 0.508 ± 0.002 (n=3) | 0.259 ± 0.001 (n=3) | 0.370 ± 0.001 (n=3) | 0.121 ± 0.009 (n=3) | 0.365 ± 0.001 (n=3) | 0.400 ± 0.012 (n=3) | 0.191 ± 0.002 (n=3) | 0.139 ± 0.001 (n=3) | 24 min | 0.458 |

Effect against the baseline:

| Level | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) |
|---|---|---|---|---|---|---|---|---|
| linear (baseline) | baseline | baseline | baseline | baseline | baseline | baseline | baseline | baseline |
| cosine | +0.001 noise | +0.002 noise | -0.000 noise | +0.022 **REAL** | -0.003 noise | +0.001 noise | -0.001 noise | +0.000 noise |

![Learning-rate schedule](results/figures/sweep_schedule.png)

#### Epochs

| Level | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) | training time | train-val mAP50-95 (last epoch) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.502 ± 0.004 (n=3) | 0.235 ± 0.019 (n=3) | 0.346 ± 0.019 (n=3) | 0.090 ± 0.014 (n=3) | 0.358 ± 0.012 (n=3) | 0.391 ± 0.005 (n=3) | 0.188 ± 0.005 (n=3) | 0.136 ± 0.007 (n=3) | 12 min | 0.457 |
| 10 (baseline) | 0.506 ± 0.002 (n=3) | 0.257 ± 0.001 (n=3) | 0.370 ± 0.001 (n=3) | 0.099 ± 0.009 (n=3) | 0.368 ± 0.006 (n=3) | 0.399 ± 0.021 (n=3) | 0.192 ± 0.002 (n=3) | 0.138 ± 0.003 (n=3) | 24 min | 0.458 |
| 30 | 0.501 ± 0.005 (n=3) | 0.269 ± 0.006 (n=3) | 0.385 ± 0.008 (n=3) | 0.107 ± 0.013 (n=3) | 0.361 ± 0.002 (n=3) | 0.381 ± 0.009 (n=3) | 0.192 ± 0.000 (n=3) | 0.134 ± 0.003 (n=3) | 68 min | 0.456 |

Effect against the baseline:

| Level | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) |
|---|---|---|---|---|---|---|---|---|
| 5 | -0.005 noise | -0.022 **REAL** | -0.024 **REAL** | -0.009 noise | -0.010 noise | -0.008 noise | -0.004 noise | -0.002 noise |
| 10 (baseline) | baseline | baseline | baseline | baseline | baseline | baseline | baseline | baseline |
| 30 | -0.005 noise | +0.012 **REAL** | +0.015 **REAL** | +0.008 noise | -0.007 noise | -0.018 **REAL** | +0.000 noise | -0.004 noise |

![Epochs](results/figures/sweep_epochs.png)

#### Training images

| Level | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) | training time | train-val mAP50-95 (last epoch) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2k | 0.494 ± 0.003 (n=3) | 0.217 ± 0.011 (n=3) | 0.326 ± 0.005 (n=3) | 0.101 ± 0.031 (n=3) | 0.351 ± 0.005 (n=3) | 0.385 ± 0.024 (n=3) | 0.183 ± 0.003 (n=3) | 0.131 ± 0.003 (n=3) | 6 min | 0.459 |
| 5k | 0.497 ± 0.008 (n=3) | 0.235 ± 0.017 (n=3) | 0.343 ± 0.011 (n=3) | 0.098 ± 0.002 (n=3) | 0.359 ± 0.006 (n=3) | 0.378 ± 0.015 (n=3) | 0.187 ± 0.005 (n=3) | 0.134 ± 0.007 (n=3) | 13 min | 0.459 |
| 9.8k (baseline) | 0.506 ± 0.002 (n=3) | 0.257 ± 0.001 (n=3) | 0.370 ± 0.001 (n=3) | 0.099 ± 0.009 (n=3) | 0.368 ± 0.006 (n=3) | 0.399 ± 0.021 (n=3) | 0.192 ± 0.002 (n=3) | 0.138 ± 0.003 (n=3) | 24 min | 0.458 |
| 30k | 0.513 ± 0.010 (n=3) | 0.280 ± 0.017 (n=3) | 0.392 ± 0.018 (n=3) | 0.102 ± 0.009 (n=3) | 0.369 ± 0.009 (n=3) | 0.418 ± 0.004 (n=3) | 0.195 ± 0.007 (n=3) | 0.143 ± 0.011 (n=3) | 68 min | 0.462 |

Effect against the baseline:

| Level | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) |
|---|---|---|---|---|---|---|---|---|
| 2k | -0.013 **REAL** | -0.040 **REAL** | -0.044 **REAL** | +0.002 noise | -0.016 **REAL** | -0.014 noise | -0.009 noise | -0.007 noise |
| 5k | -0.009 noise | -0.022 **REAL** | -0.027 **REAL** | -0.001 noise | -0.009 noise | -0.022 noise | -0.006 noise | -0.005 noise |
| 9.8k (baseline) | baseline | baseline | baseline | baseline | baseline | baseline | baseline | baseline |
| 30k | +0.007 noise | +0.023 **REAL** | +0.022 **REAL** | +0.003 noise | +0.002 noise | +0.019 **REAL** | +0.002 noise | +0.004 noise |

![Training images](results/figures/sweep_data.png)

#### Frozen layers

| Level | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) | training time | train-val mAP50-95 (last epoch) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 (baseline) | 0.506 ± 0.002 (n=3) | 0.257 ± 0.001 (n=3) | 0.370 ± 0.001 (n=3) | 0.099 ± 0.009 (n=3) | 0.368 ± 0.006 (n=3) | 0.399 ± 0.021 (n=3) | 0.192 ± 0.002 (n=3) | 0.138 ± 0.003 (n=3) | 24 min | 0.458 |
| 10 | 0.513 ± 0.004 (n=3) | 0.167 ± 0.008 (n=3) | 0.301 ± 0.007 (n=3) | 0.075 ± 0.005 (n=3) | 0.369 ± 0.003 (n=3) | 0.401 ± 0.010 (n=3) | 0.193 ± 0.003 (n=3) | 0.142 ± 0.001 (n=3) | 23 min | 0.496 |
| 22 | 0.518 ± 0.003 (n=3) | 0.151 ± 0.001 (n=3) | 0.248 ± 0.007 (n=3) | 0.077 ± 0.009 (n=3) | 0.378 ± 0.004 (n=3) | 0.402 ± 0.002 (n=3) | 0.198 ± 0.001 (n=3) | 0.150 ± 0.001 (n=3) | 22 min | 0.512 |

Effect against the baseline:

| Level | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) |
|---|---|---|---|---|---|---|---|---|
| 0 (baseline) | baseline | baseline | baseline | baseline | baseline | baseline | baseline | baseline |
| 10 | +0.007 noise | -0.090 **REAL** | -0.069 **REAL** | -0.024 **REAL** | +0.002 noise | +0.001 noise | +0.001 noise | +0.003 noise |
| 22 | +0.012 **REAL** | -0.106 **REAL** | -0.122 **REAL** | -0.022 **REAL** | +0.010 **REAL** | +0.002 noise | +0.005 noise | +0.011 **REAL** |

![Frozen layers](results/figures/sweep_freeze.png)

#### Starting model

| Level | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) | training time | train-val mAP50-95 (last epoch) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11n (baseline) | 0.506 ± 0.002 (n=3) | 0.257 ± 0.001 (n=3) | 0.370 ± 0.001 (n=3) | 0.099 ± 0.009 (n=3) | 0.368 ± 0.006 (n=3) | 0.399 ± 0.021 (n=3) | 0.192 ± 0.002 (n=3) | 0.138 ± 0.003 (n=3) | 24 min | 0.458 |
| yolo11s | 0.556 ± 0.009 (n=3) | 0.287 ± 0.012 (n=3) | 0.438 ± 0.011 (n=3) | 0.080 ± 0.027 (n=3) | 0.381 ± 0.005 (n=3) | 0.427 ± 0.012 (n=3) | 0.242 ± 0.007 (n=3) | 0.166 ± 0.009 (n=3) | 121 min | 0.538 |
| yolo26n | 0.515 ± 0.002 (n=3) | 0.251 ± 0.004 (n=3) | 0.370 ± 0.003 (n=3) | 0.118 ± 0.011 (n=3) | 0.367 ± 0.001 (n=3) | 0.405 ± 0.007 (n=3) | 0.190 ± 0.004 (n=3) | 0.138 ± 0.000 (n=3) | 28 min | 0.478 |

Effect against the baseline:

| Level | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) |
|---|---|---|---|---|---|---|---|---|
| yolo11n (baseline) | baseline | baseline | baseline | baseline | baseline | baseline | baseline | baseline |
| yolo11s | +0.049 **REAL** | +0.031 **REAL** | +0.068 **REAL** | -0.019 **REAL** | +0.013 **REAL** | +0.028 noise | +0.050 **REAL** | +0.027 **REAL** |
| yolo26n | +0.009 noise | -0.006 noise | -0.000 noise | +0.019 **REAL** | -0.000 noise | +0.006 noise | -0.003 noise | -0.000 noise |

Each starting model against **its own** zero-shot score (person mAP50-95, mean over seeds minus zero-shot):

| Model | clean COCO | low light s3 (synth.) | rain s3 (synth.) | 50 % hidden (synth.) | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) |
|---|---|---|---|---|---|---|---|---|
| yolo11n | -0.026 | +0.107 | +0.152 | +0.036 | -0.008 | +0.000 | -0.005 | -0.010 |
| yolo11s | -0.038 | +0.080 | +0.096 | +0.020 | -0.026 | -0.059 | -0.021 | -0.025 |
| yolo26n | -0.013 | +0.096 | +0.137 | +0.039 | -0.009 | +0.017 | -0.008 | -0.004 |

![Starting model](results/figures/sweep_model.png)

#### Every setting against its own zero-shot model

Mean over seeds of (fine-tuned minus the zero-shot score of the same starting model). **beats** = at least +0.01 and positive in all 3 seeds; **below** = at least -0.01 and negative in all 3 seeds; no tag = within noise. On the synthetic sets every setting beats zero-shot, which is expected because training and test degradations come from the same generator; the real sets are the honest test.

| Setting | clean COCO | ExDark (real) | DAWN fog (real) | BDD day (real) | BDD night (real) | low light s3 (synth.) | rain s3 (synth.) |
|---|---|---|---|---|---|---|---|
| lr_2e-5 | -0.019 below | -0.010 below | -0.006 | -0.003 | -0.004 | +0.072 **beats** | +0.120 **beats** |
| lr_6e-5 | -0.024 below | -0.012 below | -0.007 | -0.006 | -0.005 | +0.080 **beats** | +0.123 **beats** |
| lr_1.2e-4 | -0.026 below | -0.008 | +0.000 | -0.005 | -0.010 below | +0.107 **beats** | +0.152 **beats** |
| lr_3e-4 | -0.042 below | -0.026 below | -0.013 below | -0.016 below | -0.024 below | +0.103 **beats** | +0.148 **beats** |
| lr_1e-3 | -0.078 below | -0.050 below | -0.054 below | -0.033 below | -0.052 below | +0.087 **beats** | +0.128 **beats** |
| cos_1.2e-4 | -0.024 below | -0.011 below | +0.001 | -0.007 | -0.010 below | +0.108 **beats** | +0.152 **beats** |
| ep_5 | -0.030 below | -0.018 below | -0.008 | -0.010 | -0.012 below | +0.084 **beats** | +0.128 **beats** |
| ep_30 | -0.031 below | -0.015 below | -0.018 below | -0.005 | -0.015 below | +0.119 **beats** | +0.167 **beats** |
| n_2k | -0.038 below | -0.024 below | -0.014 | -0.015 below | -0.017 below | +0.067 **beats** | +0.109 **beats** |
| n_5k | -0.035 below | -0.017 below | -0.022 below | -0.011 below | -0.015 below | +0.085 **beats** | +0.125 **beats** |
| n_30k | -0.019 below | -0.006 | +0.019 **beats** | -0.003 | -0.006 | +0.129 **beats** | +0.174 **beats** |
| frz_10 | -0.019 below | -0.006 | +0.001 | -0.004 | -0.007 | +0.017 **beats** | +0.083 **beats** |
| frz_22 | -0.014 below | +0.002 | +0.002 | -0.000 | +0.001 | +0.001 | +0.030 **beats** |
| m_11s | -0.038 below | -0.026 below | -0.059 below | -0.021 below | -0.025 below | +0.080 **beats** | +0.096 **beats** |
| m_26n | -0.013 below | -0.009 | +0.017 **beats** | -0.008 | -0.004 | +0.096 **beats** | +0.137 **beats** |

