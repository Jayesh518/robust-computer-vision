"""Turn configs/sweep.yaml into command lines. Used by run_sweep.sh and the tests; has no side effects.

    python scripts/sweep_plan.py                 # one line per setting: name<TAB>group<TAB>level<TAB>train.py arguments
    python scripts/sweep_plan.py --eval-sets     # the evaluation set names, space separated
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sweep.yaml"


def load(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = [s["name"] for s in cfg["settings"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate setting names in sweep.yaml")
    return cfg


def train_args(cfg: dict, setting: dict) -> str:
    """Base arguments, then the setting's own; later arguments win in argparse, so a setting overrides the base."""
    extra = setting["args"].replace("{manifests}", cfg["manifests"])
    return f"--variant {cfg['variant']} {cfg['base_args']} {extra}".strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-sets", action="store_true")
    args = ap.parse_args()
    cfg = load()
    if args.eval_sets:
        print(" ".join(cfg["eval_sets"]))
        return
    for s in cfg["settings"]:
        print(f"{s['name']}\t{s['group']}\t{s['level']}\t{train_args(cfg, s)}")


if __name__ == "__main__":
    main()
