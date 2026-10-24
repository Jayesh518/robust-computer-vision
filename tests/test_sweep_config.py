import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sweep_plan  # noqa: E402


def _train_parser() -> argparse.ArgumentParser:
    """The subset of scripts/train.py's arguments a setting may use (kept in sync by reading the real file below)."""
    ap = argparse.ArgumentParser()
    for flag, kw in (("--variant", {}), ("--epochs", {"type": int}), ("--optimizer", {}), ("--lr0", {"type": float}),
                     ("--momentum", {"type": float}), ("--freeze", {"type": int}), ("--weights", {}), ("--train-list", {}),
                     ("--cos-lr", {"action": "store_true"})):
        ap.add_argument(flag, **kw)
    return ap


def test_every_flag_used_exists_in_train_py():
    src = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    cfg = sweep_plan.load()
    used = {tok for s in cfg["settings"] for tok in shlex.split(sweep_plan.train_args(cfg, s)) if tok.startswith("--")}
    for flag in used:
        assert f'"{flag}"' in src, f"train.py has no {flag}"


def test_every_setting_parses_and_overrides_the_base():
    cfg = sweep_plan.load()
    base = _train_parser().parse_args(shlex.split(f"--variant {cfg['variant']} {cfg['base_args']}"))
    assert (base.optimizer, base.epochs, base.lr0, base.momentum) == ("AdamW", 10, 0.000119, 0.9)
    for s in cfg["settings"]:
        a = _train_parser().parse_args(shlex.split(sweep_plan.train_args(cfg, s)))
        assert a.variant == "A1"
    lr = {s["name"]: s for s in cfg["settings"]}
    assert _train_parser().parse_args(shlex.split(sweep_plan.train_args(cfg, lr["lr_3e-4"]))).lr0 == 0.0003
    assert _train_parser().parse_args(shlex.split(sweep_plan.train_args(cfg, lr["ep_30"]))).epochs == 30


def test_names_unique_and_expected_grid_size():
    cfg = sweep_plan.load()
    assert len(cfg["settings"]) == 15
    assert len({s["name"] for s in cfg["settings"]}) == 15
    assert set(cfg["eval_sets"]) >= {"clean_bench1k", "exdark_all", "dawn_fog", "bdd_night_clear"}


def test_manifest_placeholder_is_resolved():
    cfg = sweep_plan.load()
    for s in cfg["settings"]:
        assert "{" not in sweep_plan.train_args(cfg, s)
