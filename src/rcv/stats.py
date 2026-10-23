"""Small statistics helpers: mean/std over seeds and bootstrap confidence intervals."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1; 0.0 when there is a single value)."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        raise ValueError("mean_std needs at least one value")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean. Returns (mean, low, high). Deterministic for a given seed."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    return float(a.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI of mean(a - b) for paired per-image scores. A CI that excludes 0 means a clear difference."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_diff_ci needs arrays of equal shape")
    return bootstrap_ci(a - b, n_boot=n_boot, alpha=alpha, seed=seed)
