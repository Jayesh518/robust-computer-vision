Person mAP50-95 with 95 % bootstrap intervals; every run is one seed (n=1), so differences below about 0.02 are within noise (seed-to-seed standard deviation in the 3-seed runs was 0.003 to 0.024 depending on the set).

| Run | What it is | clean COCO | low light s3 (synth.) | rain s3 (synth.) | ExDark (real) | DAWN fog (real) | BDD clear day (real) |
|---|---|---|---|---|---|---|---|
| B0 zero-shot | pretrained, nothing trained | 0.532 [0.509, 0.551] | 0.150 [0.136, 0.166] | 0.218 [0.201, 0.234] | 0.376 [0.366, 0.386] | 0.399 [0.309, 0.492] | 0.198 [0.184, 0.213] |
| D0 lr ~0, 3 epochs | training path with essentially no learning | 0.520 [0.500, 0.539] | 0.156 [0.142, 0.170] | 0.191 [0.176, 0.210] | 0.363 [0.353, 0.373] | 0.394 [0.309, 0.466] | 0.198 [0.185, 0.208] |
| D3 lr ~0, no augmentation | same, and no mosaic/colour/geometry augmentation | 0.522 [0.501, 0.540] | 0.171 [0.159, 0.187] | 0.203 [0.187, 0.221] | 0.363 [0.353, 0.373] | 0.394 [0.313, 0.472] | 0.196 [0.182, 0.206] |
| D1 lr 2e-5 | AdamW at 1/6 of the automatic rate | 0.516 [0.496, 0.536] | 0.129 [0.116, 0.146] | 0.236 [0.222, 0.253] | 0.376 [0.365, 0.386] | 0.374 [0.284, 0.459] | 0.196 [0.182, 0.207] |
| D2 frozen backbone | first 10 layers frozen | 0.516 [0.495, 0.536] | 0.130 [0.114, 0.144] | 0.219 [0.205, 0.235] | 0.369 [0.359, 0.379] | 0.387 [0.297, 0.464] | 0.196 [0.183, 0.209] |
| B1 original (seed 0) | original clean-control recipe | 0.510 [0.491, 0.530] | 0.128 [0.113, 0.143] | 0.212 [0.198, 0.230] | 0.368 [0.358, 0.377] | 0.379 [0.295, 0.458] | 0.193 [0.180, 0.205] |
| A1 original (seed 0) | original recipe + condition-aware augmentation | 0.501 [0.481, 0.522] | 0.255 [0.239, 0.276] | 0.370 [0.351, 0.389] | 0.371 [0.362, 0.380] | 0.399 [0.306, 0.480] | 0.192 [0.178, 0.204] |
