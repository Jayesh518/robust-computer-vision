import cv2
import numpy as np
import pytest

from rcv import enhance


def _scene(seed=0, size=96):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(40, 220, (size, size, 3), dtype=np.uint8), (0, 0), 6)
    return base


@pytest.mark.parametrize("name", list(enhance.METHODS))
def test_shape_dtype_and_determinism(name):
    img = _scene()
    a, b = enhance.METHODS[name](img), enhance.METHODS[name](img)
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_none_returns_a_copy_not_the_same_array():
    img = _scene()
    out = enhance.none(img)
    assert np.array_equal(out, img) and out is not img


def test_gamma_brightens_dark_and_leaves_bright_alone():
    dark = (_scene() * 0.15).astype(np.uint8)
    bright = np.clip(_scene().astype(int) + 60, 0, 255).astype(np.uint8)
    assert enhance.gamma(dark).mean() > dark.mean() + 20
    assert np.array_equal(enhance.gamma(bright), bright)


def test_clahe_raises_contrast_of_a_flat_image():
    flat = np.clip(_scene() * 0.2 + 100, 0, 255).astype(np.uint8)
    assert cv2.cvtColor(enhance.clahe(flat), cv2.COLOR_BGR2GRAY).std() > cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY).std()


def test_dcp_restores_contrast_of_a_hazy_image_without_nans():
    clear = _scene(size=128)
    t = 0.45
    hazy = np.clip(clear.astype(np.float32) * t + 235 * (1 - t), 0, 255).astype(np.uint8)
    out = enhance.dcp(hazy)
    gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).std()
    assert gray(out) > gray(hazy) * 1.3
    assert np.isfinite(out).all()


def test_nlm_reduces_noise_on_a_flat_image():
    rng = np.random.default_rng(1)
    flat = np.full((64, 64, 3), 128, np.uint8)
    noisy = np.clip(flat + rng.normal(0, 18, flat.shape), 0, 255).astype(np.uint8)
    assert enhance.nlm(noisy).std() < noisy.std() * 0.6


def test_plan_only_names_known_methods():
    for src, methods in enhance.PLAN.items():
        assert all(m in enhance.METHODS and m != "none" for m in methods), src
