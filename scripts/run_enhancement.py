"""Phase 6 (E1): evaluate the unchanged zero-shot model on original, re-encoded (control) and enhanced versions of the same images.

Per source set: the original, the identity re-encode ("none") and each planned enhancement are evaluated with the same code as the
baselines (Ultralytics val + bootstrap CIs). The effect of an enhancement is the PAIRED difference in person mAP50-95
(enhanced minus original, same images, 95 % bootstrap interval). Writes results/e1_<model>.json. Resumable per source set.

    .venv\\Scripts\\python scripts\\run_enhancement.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rcv import enhance, labels  # noqa: E402
from rcv.eval import evaluate, role_summary, write_json  # noqa: E402
import run_baseline as rb  # noqa: E402

ENH = ROOT / "data" / "processed" / "enhanced"


def gain_vs(cur: list, ref: list, n_boot: int, seed: int) -> dict | None:
    d = rb.drop_ci(cur, ref, True, n_boot, seed)          # drop = reference minus current, so the gain is its negative
    if d is None:
        return None
    return {"gain": -d["drop"], "lo": -d["hi"], "hi": -d["lo"], "paired": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo11n.pt"))
    ap.add_argument("--name", default="e1_yolo11n")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", nargs="*", help="subset of enhance.PLAN sources")
    args = ap.parse_args()

    sets = rb.all_sets()
    coco_ids = sorted(labels.to_space(i, "coco80") for i in range(len(labels.class_names())))
    out_path = ROOT / "results" / f"{args.name}.json"
    result = {"note": "E1: zero-shot model on original, re-encoded control (none) and classically enhanced images; gains are paired.",
              "weights": args.weights, "n_boot": args.n_boot, "sources": {}}
    if out_path.exists():
        result["sources"] = json.loads(out_path.read_text(encoding="utf-8")).get("sources", {})
    lat = json.loads((ROOT / "results" / "enhancement_latency.json").read_text(encoding="utf-8"))["methods"]

    def run(label: str, yaml_path: Path) -> tuple[dict, list]:
        r = evaluate(args.weights, str(yaml_path), imgsz=640, batch=16, classes=coco_ids, project=str(ROOT / "runs"),
                     name=f"{args.name}_{label}", save_json=True)
        r["roles"] = role_summary(r["per_class"])
        entries = rb.add_ci(r, yaml_path, set(coco_ids), args.n_boot, args.seed)
        return r, entries["person"]

    t0 = time.time()
    for source, methods in enhance.PLAN.items():
        if args.sources and source not in args.sources:
            continue
        done = result["sources"].get(source)
        if done and all(m in done["methods"] for m in methods):
            print(f"{source}: already done", flush=True)
            continue
        orig, e_orig = run(source, sets[source])
        none, e_none = run(f"{source}__none", ENH / f"{source}__none" / "coco80" / "coco80.yaml")
        rec = {"evidence": rb.evidence_for(source), "original": orig, "none": none,
               "none_vs_original": gain_vs(e_none, e_orig, args.n_boot, args.seed), "methods": {}}
        for m in methods:
            r, e = run(f"{source}__{m}", ENH / f"{source}__{m}" / "coco80" / "coco80.yaml")
            rec["methods"][m] = {"result": r, "gain_vs_original": gain_vs(e, e_orig, args.n_boot, args.seed),
                                 "gain_vs_none": gain_vs(e, e_none, args.n_boot, args.seed), "latency": lat[f"{source}__{m}"]}
            g = rec["methods"][m]["gain_vs_original"]
            print(f"{source:18s} {m:6s} person mAP50-95 original {orig['ci']['per_class']['person']['ap']:.3f} -> "
                  f"{r['ci']['per_class']['person']['ap']:.3f}  gain {g['gain']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        c = rec["none_vs_original"]
        print(f"{source:18s} none   re-encode control gain {c['gain']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]", flush=True)
        result["sources"][source] = rec
        write_json(out_path, result)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
