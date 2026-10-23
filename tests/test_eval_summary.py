import pytest

from rcv.eval import role_summary


def _m(v):
    return {"map50_95": v, "map50": v + 0.1, "precision": v + 0.2, "recall": v + 0.3}


def test_roles_are_grouped_and_macro_averaged():
    per_class = {"person": _m(0.5), "dog": _m(0.2), "cow": _m(0.4), "car": _m(0.6), "bus": _m(0.8)}
    s = role_summary(per_class)
    assert s["primary"]["classes"] == ["person"] and s["primary"]["map50_95"] == pytest.approx(0.5)
    assert s["secondary"]["classes"] == ["cow", "dog"] and s["secondary"]["map50_95"] == pytest.approx(0.3)
    assert s["context"]["classes"] == ["bus", "car"] and s["context"]["map50"] == pytest.approx(0.8)


def test_role_with_no_classes_is_reported_as_missing_not_zero():
    s = role_summary({"person": _m(0.5)})
    assert s["secondary"]["map50_95"] is None and s["secondary"]["classes"] == []
    assert s["context"]["recall"] is None


def test_unknown_class_names_are_ignored():
    s = role_summary({"person": _m(0.5), "chair": _m(0.9)})
    assert s["primary"]["map50_95"] == pytest.approx(0.5)
    assert all("chair" not in v["classes"] for v in s.values())
