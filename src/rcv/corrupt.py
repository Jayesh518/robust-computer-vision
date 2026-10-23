"""Deterministic synthetic degradations for the robustness benchmark.

All functions take an RGB or BGR uint8 image (H, W, 3) and return a uint8 image of the same shape.
Parameters per condition and severity (1 mild .. 3 strong) live in configs/conditions.yaml.

The same (image key, condition, severity) always produces the same output: the random generator is seeded from a hash
of these three values plus a base seed from the config. This makes the benchmark exactly regenerable.

Models (documented so results can be interpreted):
  low_light       exposure scale in linear light, then Poisson (shot) noise and Gaussian read noise, back to sRGB.
                  The image is NOT re-brightened: it stays dark and noisy, like an under-exposed photo.
  fog             atmospheric scattering  I = J * t + A * (1 - t),  t = exp(-beta * depth).
                  depth is a smooth random field blended with a vertical gradient (far at the top).
  rain            random slanted streaks added on top of a slightly blurred, slightly darkened image.
  snow            random small bright flakes (screen blend) plus a mild haze.
  motion_blur     convolution with a straight line kernel of random angle.
  gaussian_noise  additive N(0, sigma^2) on [0, 1] pixel values.

These are simplified physical models, not photographs of real weather. Results on them are reported as "synthetic".
"""
from __future__ import annotations

import zlib
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "conditions.yaml"
CONDITIONS = ("low_light", "fog", "rain", "snow", "motion_blur", "gaussian_noise")


@lru_cache(maxsize=None)
def load_conditions(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def derive_seed(base: int, key: str, condition: str, severity: int) -> int:
    """Stable 32-bit seed from the base seed and the (image key, condition, severity) triple."""
    return zlib.crc32(f"{base}|{key}|{condition}|{severity}".encode("utf-8")) & 0xFFFFFFFF


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def _low_light(x, rng, exposure, read_noise, shot_gain):
    lin = _srgb_to_linear(x) * exposure
    photons = rng.poisson(lin * shot_gain).astype(np.float32) / shot_gain
    noisy = photons + rng.normal(0.0, read_noise, lin.shape).astype(np.float32)
    return _linear_to_srgb(noisy)


def _fog(x, rng, beta, airlight):
    h, w = x.shape[:2]
    field = rng.random((max(2, h // 48), max(2, w // 48))).astype(np.float32)
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)
    gradient = np.linspace(1.0, 0.25, h, dtype=np.float32)[:, None]  # top = far, bottom = near
    depth = np.clip(0.6 * gradient + 0.4 * field, 0.0, 1.0)
    t = np.exp(-beta * depth)[..., None]
    return x * t + airlight * (1.0 - t)


def _rain(x, rng, density, length, thickness, alpha):
    h, w = x.shape[:2]
    n = int(density * h * w)
    layer = np.zeros((h, w), np.uint8)
    angle = np.deg2rad(rng.uniform(-12.0, 12.0))  # from vertical
    dx, dy = length * np.sin(angle), length * np.cos(angle)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    for x0, y0 in zip(xs, ys):
        cv2.line(layer, (int(x0), int(y0)), (int(x0 + dx), int(y0 + dy)), 255, int(thickness))
    layer = cv2.GaussianBlur(layer, (3, 3), 0).astype(np.float32) / 255.0
    base = cv2.GaussianBlur(x, (0, 0), 0.8)
    return base * 0.92 + alpha * layer[..., None]


def _snow(x, rng, density, radius, haze):
    h, w = x.shape[:2]
    n = int(density * h * w)
    layer = np.zeros((h, w), np.uint8)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    radii = radius * rng.uniform(0.6, 1.4, n)
    for x0, y0, r in zip(xs, ys, radii):
        cv2.circle(layer, (int(x0), int(y0)), max(1, int(round(r))), 255, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.8).astype(np.float32) / 255.0
    out = x * (1.0 - haze) + haze
    return 1.0 - (1.0 - out) * (1.0 - 0.9 * layer[..., None])


def _motion_blur(x, rng, length):
    size = max(3, int(length) | 1)  # odd kernel size
    c = size // 2
    angle = np.deg2rad(rng.uniform(0.0, 180.0))
    dx, dy = np.cos(angle) * c, np.sin(angle) * c
    kernel = np.zeros((size, size), np.float32)
    cv2.line(kernel, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    kernel /= kernel.sum()
    return cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT)


def _gaussian_noise(x, rng, sigma):
    return x + rng.normal(0.0, sigma, x.shape).astype(np.float32)


_FUNCS = {
    "low_light": _low_light,
    "fog": _fog,
    "rain": _rain,
    "snow": _snow,
    "motion_blur": _motion_blur,
    "gaussian_noise": _gaussian_noise,
}


def apply(img: np.ndarray, condition: str, severity: int, key: str = "", seed: Optional[int] = None,
          conditions: Optional[dict] = None) -> np.ndarray:
    """Degrade `img` (uint8, H x W x 3) with `condition` at `severity` (1..3).

    `key` identifies the image (e.g. its file name) so that every image gets its own, but reproducible, randomness.
    Pass `seed` to override the derived seed.
    """
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be a uint8 array of shape (H, W, 3)")
    cfg = conditions or load_conditions()
    if condition not in cfg["conditions"]:
        raise ValueError(f"unknown condition {condition!r}; choose from {tuple(cfg['conditions'])}")
    severities = cfg["conditions"][condition]["severities"]
    if severity not in severities:
        raise ValueError(f"severity must be one of {sorted(severities)}, got {severity!r}")
    s = derive_seed(cfg["seed_base"], key, condition, severity) if seed is None else seed
    rng = np.random.default_rng(s)
    x = img.astype(np.float32) / 255.0
    y = _FUNCS[condition](x, rng, **severities[severity])
    return np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)
