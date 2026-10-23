import numpy as np
import pytest

from rcv import stats


def test_mean_std_known_values():
    m, s = stats.mean_std([1.0, 2.0, 3.0])
    assert m == pytest.approx(2.0) and s == pytest.approx(1.0)  # sample std (ddof=1)


def test_mean_std_single_value_has_zero_std():
    assert stats.mean_std([0.5]) == (0.5, 0.0)


def test_empty_inputs_rejected():
    for fn in (stats.mean_std, stats.bootstrap_ci):
        with pytest.raises(ValueError):
            fn([])


def test_bootstrap_ci_deterministic_and_brackets_mean():
    vals = np.random.default_rng(1).normal(0.4, 0.1, 200)
    a = stats.bootstrap_ci(vals, seed=7)
    b = stats.bootstrap_ci(vals, seed=7)
    assert a == b
    mean, lo, hi = a
    assert lo < mean < hi and hi - lo < 0.05


def test_paired_diff_ci_detects_a_real_difference():
    rng = np.random.default_rng(2)
    base = rng.uniform(0.2, 0.8, 300)
    better = base + 0.05 + rng.normal(0, 0.01, 300)
    mean, lo, hi = stats.paired_diff_ci(better, base, seed=0)
    assert lo > 0 and mean == pytest.approx(0.05, abs=0.005)


def test_paired_diff_ci_no_difference_includes_zero():
    rng = np.random.default_rng(3)
    a = rng.uniform(0.2, 0.8, 300)
    b = a + rng.normal(0, 0.02, 300)
    _, lo, hi = stats.paired_diff_ci(a, b, seed=0)
    assert lo < 0 < hi


def test_paired_diff_shape_mismatch():
    with pytest.raises(ValueError):
        stats.paired_diff_ci([1, 2, 3], [1, 2])
