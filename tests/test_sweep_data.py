import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepare_sweep_data as psd  # noqa: E402


def _names(n, prefix="a"):
    return [f"{prefix}{i:06d}.jpg" for i in range(n)]


def test_subsets_are_nested_and_sized():
    base, extra = _names(6000), _names(3000, "b")
    m = psd.build_manifests(base, extra)
    s2, s5, s9, s30 = (set(m[k]) for k in ("coco_train_2k", "coco_train_5k", "coco_train_9.8k", "coco_train_30k"))
    assert (len(s2), len(s5)) == (2000, 5000)
    assert s2 < s5 < s9 < s30
    assert len(m["coco_train_30k"]) == 9000 and len(set(m["coco_train_30k"])) == 9000


def test_manifests_are_deterministic_and_independent_of_input_order():
    base, extra = _names(6000), _names(3000, "b")
    a = psd.build_manifests(base, extra)
    b = psd.build_manifests(list(reversed(base)), list(reversed(extra)))
    assert a == b


def test_small_subsets_are_not_the_first_n_by_name():
    base = _names(6000)
    m = psd.build_manifests(base, [])
    assert set(m["coco_train_2k"]) != set(sorted(base)[:2000])


def test_base_manifest_is_exactly_the_existing_set():
    base = _names(500)
    assert psd.build_manifests(base, _names(10, "b"))["coco_train_9.8k"] == sorted(base)
