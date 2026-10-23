import numpy as np
import pytest

A = pytest.importorskip("albumentations")
from rcv import corrupt  # noqa: E402
from rcv.augment import ConditionAwareDegradation  # noqa: E402


@pytest.fixture(scope="module")
def img():
    return np.random.default_rng(0).integers(30, 220, (96, 128, 3), dtype=np.uint8)


def _run(transform, image, seed=0):
    return A.Compose([transform], seed=seed)(image=image)["image"]


def test_p_one_always_changes_and_keeps_shape_and_dtype(img):
    out = _run(ConditionAwareDegradation(p=1.0), img)
    assert out.shape == img.shape and out.dtype == np.uint8 and not np.array_equal(out, img)


def test_p_zero_is_the_identity(img):
    assert np.array_equal(_run(ConditionAwareDegradation(p=0.0), img), img)


def test_same_seed_same_result_different_seed_different_result(img):
    t = ConditionAwareDegradation(p=1.0)
    assert np.array_equal(_run(t, img, seed=5), _run(ConditionAwareDegradation(p=1.0), img, seed=5))
    assert not np.array_equal(_run(t, img, seed=5), _run(ConditionAwareDegradation(p=1.0), img, seed=6))


def test_every_condition_and_severity_gets_used():
    t = ConditionAwareDegradation(p=1.0)
    t.set_random_seed(1) if hasattr(t, "set_random_seed") else None
    seen = set()
    for _ in range(400):
        params = t.get_params()
        seen.add((params["condition"], params["severity"]))
    assert {c for c, _ in seen} == set(corrupt.CONDITIONS) and {s for _, s in seen} == {1, 2, 3}


def test_survives_the_serialisation_roundtrip_ultralytics_uses(img):
    t = ConditionAwareDegradation(conditions=("fog", "rain"), severities=(2, 3), p=0.7)
    t2 = A.from_dict(A.to_dict(t))
    assert isinstance(t2, ConditionAwareDegradation)
    assert t2.conditions == ("fog", "rain") and t2.severities == (2, 3) and t2.p == pytest.approx(0.7)
    assert _run(ConditionAwareDegradation(p=1.0), img).shape == img.shape


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        ConditionAwareDegradation(conditions=("sandstorm",))


def test_bounding_boxes_pass_through_untouched(img):
    comp = A.Compose([ConditionAwareDegradation(p=1.0)], bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]), seed=0)
    out = comp(image=img, bboxes=[(0.5, 0.5, 0.4, 0.4)], labels=[1])
    assert out["bboxes"][0] == pytest.approx((0.5, 0.5, 0.4, 0.4)) and list(out["labels"]) == [1]
