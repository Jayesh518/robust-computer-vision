import numpy as np
import pytest

from rcv import occlude


@pytest.fixture(scope="module")
def img():
    return np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)


BOX = (60, 40, 220, 200)  # 160 x 160


@pytest.mark.parametrize("direction", occlude.DIRECTIONS)
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_covered_fraction_matches_target(img, direction, fraction):
    _, actual = occlude.occlude_box(img, BOX, fraction, direction, key="p1.jpg")
    assert actual == pytest.approx(fraction, abs=0.02)


@pytest.mark.parametrize("direction", occlude.DIRECTIONS)
def test_only_pixels_inside_the_box_change(img, direction):
    out, _ = occlude.occlude_box(img, BOX, 0.5, direction, key="p1.jpg")
    x0, y0, x1, y1 = BOX
    outside = np.ones(img.shape[:2], bool)
    outside[y0:y1, x0:x1] = False
    assert np.array_equal(out[outside], img[outside])
    assert not np.array_equal(out[y0:y1, x0:x1], img[y0:y1, x0:x1])


def test_changed_pixels_agree_with_reported_fraction(img):
    out, actual = occlude.occlude_box(img, BOX, 0.5, "bottom", key="k")
    x0, y0, x1, y1 = BOX
    changed = (out[y0:y1, x0:x1] != img[y0:y1, x0:x1]).any(axis=2).mean()
    assert changed == pytest.approx(actual, abs=0.03)  # a few texture pixels may equal the original by chance


def test_slab_enters_from_the_requested_side(img):
    x0, y0, x1, y1 = BOX
    out, _ = occlude.occlude_box(img, BOX, 0.4, "bottom", key="k")
    diff = (out[y0:y1, x0:x1] != img[y0:y1, x0:x1]).any(axis=2)
    assert diff[-10:].mean() > 0.9 and diff[:10].mean() < 0.05    # bottom rows covered, top rows untouched
    out, _ = occlude.occlude_box(img, BOX, 0.4, "left", key="k")
    diff = (out[y0:y1, x0:x1] != img[y0:y1, x0:x1]).any(axis=2)
    assert diff[:, :10].mean() > 0.9 and diff[:, -10:].mean() < 0.05


def test_deterministic_and_key_sensitive(img):
    a, _ = occlude.occlude_box(img, BOX, 0.5, "bottom", key="a.jpg")
    b, _ = occlude.occlude_box(img, BOX, 0.5, "bottom", key="a.jpg")
    c, _ = occlude.occlude_box(img, BOX, 0.5, "bottom", key="b.jpg")
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_zero_and_full_fraction(img):
    out, f = occlude.occlude_box(img, BOX, 0.0)
    assert f == 0.0 and np.array_equal(out, img)
    _, f = occlude.occlude_box(img, BOX, 1.0)
    assert f > 0.97


def test_input_not_modified(img):
    before = img.copy()
    occlude.occlude_box(img, BOX, 0.75, "top", key="k")
    assert np.array_equal(img, before)


def test_box_is_clipped_to_the_image(img):
    out, f = occlude.occlude_box(img, (-50, -50, 100, 100), 0.5, "right", key="k")
    assert f == pytest.approx(0.5, abs=0.02) and out.shape == img.shape


def test_bad_arguments(img):
    with pytest.raises(ValueError):
        occlude.occlude_box(img, BOX, 1.5)
    with pytest.raises(ValueError):
        occlude.occlude_box(img, BOX, 0.5, "diagonal")
    with pytest.raises(ValueError):
        occlude.occlude_box(img, (500, 500, 600, 600), 0.5)
    with pytest.raises(ValueError):
        occlude.occlude_box(img.astype(np.float32), BOX, 0.5)


def test_texture_is_grey_and_varied():
    tex = occlude.concrete_texture(64, 64, np.random.default_rng(1))
    assert tex.shape == (64, 64, 3) and tex.dtype == np.uint8
    assert 80 < tex.mean() < 180 and tex.std() > 8
