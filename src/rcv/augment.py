"""Condition-aware training augmentation (variant A1/A2): randomly degrade training images with the SAME degradation
models that build the synthetic test benchmark (src/rcv/corrupt.py).

Passed to Ultralytics as `augmentations=[ConditionAwareDegradation(...)]`. Ultralytics serialises custom Albumentations
transforms with `A.to_dict` and rebuilds them in the data-loader workers, so this is a proper registered
`A.ImageOnlyTransform` subclass (a lambda would not survive the round trip).

CAVEAT that must travel with every A1/A2 result: the degradations used for training and for the synthetic test sets come
from the same generator, so A1/A2 gains on the SYNTHETIC sets are in-distribution (optimistic). Only the REAL sets
(ExDark, DAWN, BDD100K) show whether the gain transfers to genuine photographs.
"""
from __future__ import annotations

import albumentations as A

from . import corrupt


class ConditionAwareDegradation(A.ImageOnlyTransform):
    """With probability `p`, degrade the image with a random condition at a random severity (uniform choice).

    Randomness comes from albumentations' own generator, so it follows the training seed and differs per sample.
    """

    def __init__(self, conditions=corrupt.CONDITIONS, severities=(1, 2, 3), p: float = 0.5):
        super().__init__(p=p)
        self.conditions = tuple(conditions)
        self.severities = tuple(int(s) for s in severities)
        unknown = [c for c in self.conditions if c not in corrupt.CONDITIONS]
        if unknown:
            raise ValueError(f"unknown conditions {unknown}; choose from {corrupt.CONDITIONS}")

    def get_params(self) -> dict:
        return {"condition": self.py_random.choice(self.conditions),
                "severity": self.py_random.choice(self.severities),
                "seed": self.py_random.randrange(2 ** 32)}

    def apply(self, img, condition="fog", severity=1, seed=0, **params):
        return corrupt.apply(img, condition, severity, seed=seed)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("conditions", "severities")
