import numpy as np
import pytest

from rcv import corrupt


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(0)
    base = rng.integers(40, 215, (96, 128, 3), dtype=np.uint8)
    base[:, :64] = 200  # a flat bright region helps the noise tests
    return base


@pytest.mark.parametrize("condition", corrupt.CONDITIONS)
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_shape_dtype_preserved(img, condition, severity):
    out = corrupt.apply(img, condition, severity, key="a.jpg")
    assert out.shape == img.shape and out.dtype == np.uint8


@pytest.mark.parametrize("condition", corrupt.CONDITIONS)
def test_deterministic_for_same_key(img, condition):
    a = corrupt.apply(img, condition, 2, key="img_001.jpg")
    b = corrupt.apply(img, condition, 2, key="img_001.jpg")
    assert np.array_equal(a, b)


@pytest.mark.parametrize("condition", ["low_light", "rain", "snow", "gaussian_noise", "fog"])
def test_different_keys_give_different_randomness(img, condition):
    a = corrupt.apply(img, condition, 2, key="img_001.jpg")
    b = corrupt.apply(img, condition, 2, key="img_002.jpg")
    assert not np.array_equal(a, b)


def test_input_not_modified(img):
    before = img.copy()
    corrupt.apply(img, "rain", 3, key="x")
    assert np.array_equal(img, before)


def test_low_light_gets_darker_with_severity(img):
    means = [corrupt.apply(img, "low_light", s, key="k").mean() for s in (1, 2, 3)]
    assert means[0] > means[1] > means[2]
    assert means[0] < img.mean()  # even the mildest level is darker than the original


def test_fog_reduces_contrast_with_severity(img):
    stds = [corrupt.apply(img, "fog", s, key="k").std() for s in (1, 2, 3)]
    assert stds[0] > stds[1] > stds[2]
    assert stds[0] < img.std()


def test_motion_blur_removes_detail_with_severity(img):
    def hf(x):  # high-frequency energy
        return np.abs(np.diff(x.astype(float), axis=1)).mean()
    energy = [hf(corrupt.apply(img, "motion_blur", s, key="k")) for s in (1, 2, 3)]
    assert energy[0] > energy[1] > energy[2]


def test_gaussian_noise_adds_variance_on_flat_region(img):
    flat = img[:, :64]
    flat_img = np.full((96, 128, 3), 128, np.uint8)
    stds = [corrupt.apply(flat_img, "gaussian_noise", s, key="k").std() for s in (1, 2, 3)]
    assert stds[0] < stds[1] < stds[2]
    assert stds[0] > 0 and flat.size > 0


def test_rain_and_snow_add_bright_pixels(img):
    dark = np.full((96, 128, 3), 30, np.uint8)
    for cond in ("rain", "snow"):
        out = corrupt.apply(dark, cond, 3, key="k")
        assert out.max() > dark.max() + 30


def test_seed_override_is_respected(img):
    a = corrupt.apply(img, "snow", 2, key="k1", seed=123)
    b = corrupt.apply(img, "snow", 2, key="k2", seed=123)
    assert np.array_equal(a, b)


def test_bad_arguments(img):
    with pytest.raises(ValueError):
        corrupt.apply(img, "sandstorm", 1)
    with pytest.raises(ValueError):
        corrupt.apply(img, "fog", 9)
    with pytest.raises(ValueError):
        corrupt.apply(img.astype(np.float32), "fog", 1)
    with pytest.raises(ValueError):
        corrupt.apply(img[:, :, 0], "fog", 1)


def test_derive_seed_is_stable_and_sensitive():
    s = corrupt.derive_seed(1, "a.jpg", "fog", 2)
    assert s == corrupt.derive_seed(1, "a.jpg", "fog", 2)
    assert s != corrupt.derive_seed(1, "a.jpg", "fog", 3)
    assert s != corrupt.derive_seed(2, "a.jpg", "fog", 2)
    assert 0 <= s < 2**32
