"""Synthetic occlusion: a concrete-like slab hides an exact fraction of an object's bounding box.

Purpose: a measurable PROXY for "a person partly hidden by debris or a vehicle". It lets the report plot detection recall
against occlusion level (25 %, 50 %, 75 % of the box hidden). It is NOT a model of rubble, and the covered fraction is a
fraction of the BOX area, not of the body: part of a box is background even for a fully visible person.

The ground-truth box stays the full box, because the person is still there.

Determinism: the slab texture and the roughness of its edge come from a generator seeded from (key, direction, fraction).
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .corrupt import derive_seed

DIRECTIONS = ("bottom", "top", "left", "right")


def concrete_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Procedural grey 'concrete': multi-scale noise, fine speckle and a few dark cracks. uint8 (h, w, 3)."""
    base = np.full((h, w), 128.0, np.float32)
    for scale, amp in ((3, 22.0), (8, 14.0), (24, 8.0)):
        gh, gw = max(2, h // scale), max(2, w // scale)
        layer = rng.normal(0.0, 1.0, (gh, gw)).astype(np.float32)
        base += amp * cv2.resize(layer, (w, h), interpolation=cv2.INTER_CUBIC)
    base += rng.normal(0.0, 6.0, (h, w)).astype(np.float32)  # fine grain
    crack = np.zeros((h, w), np.uint8)
    for _ in range(max(1, (h + w) // 120)):
        p = rng.random(2) * (w, h)
        for _ in range(8):
            q = p + rng.normal(0.0, max(h, w) * 0.05, 2)
            cv2.line(crack, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), 255, 1)
            p = q
    base -= 45.0 * (cv2.GaussianBlur(crack, (3, 3), 0).astype(np.float32) / 255.0)
    base = np.clip(base, 0, 255).astype(np.uint8)
    return np.dstack([base, base, np.clip(base.astype(np.int16) - 3, 0, 255).astype(np.uint8)])


def _slab_mask(bh: int, bw: int, fraction: float, direction: str, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask over a (bh, bw) box: True where the slab covers. Covered area / box area ~= fraction."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    vertical = direction in ("bottom", "top")
    n_along = bw if vertical else bh                      # length of the slab edge
    noise = rng.normal(0.0, 1.0, max(3, n_along // 10)).astype(np.float32)
    noise = cv2.resize(noise.reshape(1, -1), (n_along, 1), interpolation=cv2.INTER_CUBIC).ravel()
    noise = noise / (np.abs(noise).max() + 1e-6) * jitter
    uu, vv = np.meshgrid(np.linspace(0, 1, bw, dtype=np.float32), np.linspace(0, 1, bh, dtype=np.float32))
    # depth = how far into the box, measured from the side the slab enters
    depth = {"bottom": 1.0 - vv, "top": vv, "left": uu, "right": 1.0 - uu}[direction]
    edge = noise[None, :] if vertical else noise[:, None]
    lo, hi = -1.0, 2.0
    for _ in range(40):                                   # bisection on slab depth L (covered fraction is monotonic in L)
        mid = (lo + hi) / 2
        covered = float(((depth <= mid + edge)).mean())
        lo, hi = (mid, hi) if covered < fraction else (lo, mid)
    return depth <= (lo + hi) / 2 + edge


def occlude_box(img: np.ndarray, box_xyxy: Sequence[float], fraction: float, direction: str = "bottom",
                key: str = "", seed_base: int = 0, jitter: float = 0.06) -> tuple[np.ndarray, float]:
    """Cover `fraction` of `box_xyxy` (pixels, x0 y0 x1 y1) with a concrete-like slab.

    Returns (new image, actual covered fraction of the clipped box). The input image is not modified.
    """
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be a uint8 array of shape (H, W, 3)")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    H, W = img.shape[:2]
    x0, y0 = max(0, int(round(box_xyxy[0]))), max(0, int(round(box_xyxy[1])))
    x1, y1 = min(W, int(round(box_xyxy[2]))), min(H, int(round(box_xyxy[3])))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("box is empty after clipping to the image")
    out = img.copy()
    if fraction == 0.0:
        return out, 0.0
    bh, bw = y1 - y0, x1 - x0
    rng = np.random.default_rng(derive_seed(seed_base, key, f"occlusion-{direction}", int(round(fraction * 100))))
    mask = _slab_mask(bh, bw, fraction, direction, jitter, rng)
    tex = concrete_texture(bh, bw, rng)
    region = out[y0:y1, x0:x1]
    region[mask] = tex[mask]
    return out, float(mask.mean())
