"""Classical, deterministic image enhancement used as preprocessing before a detector (phase 6, experiment E1).

No learned weights and no randomness: the same input always gives the same output, so enhanced sets are reproducible and
carry no extra licence. All functions take and return a BGR uint8 image of the same size.

  clahe     contrast-limited adaptive histogram equalisation on the L channel (low light)
  gamma     one global gamma, chosen per image so the mean luminance reaches a target (low light)
  dcp       dark-channel-prior dehazing, He et al. 2009, with a box-filter guided filter (haze, fog)
  nlm       non-local-means colour denoising (sensor noise)
  none      identity (used with a re-encode to control for the extra JPEG generation)

There is no credible classical single-image method for rain, snow or motion blur in this set, so none is offered:
those conditions are reported as not tested rather than tested with a method known to be weak.
"""
from __future__ import annotations

import cv2
import numpy as np


def none(img: np.ndarray) -> np.ndarray:
    return img.copy()


def clahe(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma(img: np.ndarray, target: float = 0.45, lo: float = 0.25) -> np.ndarray:
    """Brighten with out = in ** g where g makes the mean luminance reach `target` (0..1). Bright images are left alone (g <= 1 only)."""
    lum = float(np.clip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0, 1e-3, 0.999))
    g = float(np.clip(np.log(target) / np.log(lum), lo, 1.0))
    if g >= 0.999:
        return img.copy()
    table = (np.arange(256) / 255.0) ** g * 255.0
    return cv2.LUT(img, np.clip(table, 0, 255).astype(np.uint8))


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    k = (2 * radius + 1, 2 * radius + 1)
    mean = lambda x: cv2.boxFilter(x, cv2.CV_32F, k)
    mg, ms = mean(guide), mean(src)
    cov = mean(guide * src) - mg * ms
    var = mean(guide * guide) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return mean(a) * guide + mean(b)


def dcp(img: np.ndarray, patch: int = 15, omega: float = 0.95, t0: float = 0.1, radius: int = 40, eps: float = 1e-3) -> np.ndarray:
    """Dark channel prior dehazing. I = J t + A (1 - t); A from the brightest dark-channel pixels, t from the dark channel."""
    f = img.astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(f.min(axis=2), kernel)
    n = max(1, int(dark.size * 0.001))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    flat = f.reshape(-1, 3)[idx]
    A = flat[np.argmax(flat.sum(axis=1))]
    A = np.maximum(A, 1e-3)
    t = 1.0 - omega * cv2.erode((f / A).min(axis=2), kernel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = np.clip(_guided_filter(gray, t.astype(np.float32), radius, eps), t0, 1.0)
    J = (f - A) / t[..., None] + A
    return np.clip(J * 255.0, 0, 255).astype(np.uint8)


def nlm(img: np.ndarray, h: float = 10.0) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


METHODS = {"none": none, "clahe": clahe, "gamma": gamma, "dcp": dcp, "nlm": nlm}

# which methods are tried on which source set (source name -> methods); "none" is always added as the re-encode control
PLAN = {
    "clean_bench1k": ["clahe", "gamma", "dcp", "nlm"],        # harm check: enhancement applied to images that need none
    "low_light_s2": ["clahe", "gamma"],
    "low_light_s3": ["clahe", "gamma"],
    "exdark_all": ["clahe", "gamma"],
    "bdd_night_clear": ["clahe", "gamma"],
    "fog_s3": ["dcp"],
    "dawn_fog": ["dcp"],
    "gaussian_noise_s3": ["nlm"],
}
