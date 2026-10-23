"""Evaluate one model on all prepared sets: metrics, bootstrap confidence intervals, lenient "found" rate, paired drops.

Used for B0 (zero-shot pretrained weights) and, with --weights, for any fine-tuned checkpoint.

Per set it records
  * Ultralytics validation metrics (overall, per class, role summary: person = headline)
  * object counts per class (so thin sets are visible next to their scores)
  * bootstrap 95 % CIs for AP50-95 per class and per role (images resampled; matching by pycocotools, src/rcv/ci.py)
  * person "found" rate at IoU 0.1 / 0.3 / 0.5 (confidence >= 0.25) with CIs: the lenient measure for occlusion
  * for synthetic sets and BDD sets: the person mAP50-95 drop against the reference set, with a CI
    (paired image resampling when both sets share the same images, independent otherwise)
Results are written to results/<name>.json after EVERY set, so an interrupted run loses nothing.

Evidence labels: "synthetic" (degraded COCO val), "real" (ExDark, DAWN, BDD100K photographs), "clean" (reference).

Run from the project root:
    .venv\\Scripts\\python scripts\\run_baseline.py --weights models/yolo11n.pt --name baseline_b0_yolo11n
    .venv\\Scripts\\python scripts\\run_baseline.py --sets clean_bench1k fog_s3
Always run from this file (never `python -`): the data-loader workers cannot start from a stdin script on Windows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcv import ci, corrupt, data, labels  # noqa: E402
from rcv.eval import environment_info, evaluate, role_summary, write_json  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
PAIRED_CLEAN = {"bench": "clean_bench1k", "occ": "occ_clean"}
BDD_SETS = ("bdd_clear_day", "bdd_night_clear", "bdd_rainy_day", "bdd_snowy_day", "bdd_rainy_night", "bdd_snowy_night", "bdd_foggy_all")
REAL_SETS = ("exdark_all", "dawn_fog", "dawn_rain", "dawn_snow") + BDD_SETS     # real photographs
BDD_REFERENCE = {"bdd_night_clear": "bdd_clear_day", "bdd_rainy_day": "bdd_clear_day", "bdd_snowy_day": "bdd_clear_day",
                 "bdd_rainy_night": "bdd_night_clear", "bdd_snowy_night": "bdd_night_clear"}   # not paired: different images
FOUND_IOUS = (0.1, 0.3, 0.5)
FOUND_CONF = 0.25


def all_sets() -> dict[str, Path]:
    sets = {"clean_full": PROCESSED / "coco_val_clean" / "coco80" / "coco80.yaml",
            "clean_bench1k": PROCESSED / "coco_val_bench1k" / "coco80" / "coco80.yaml"}
    syn = PROCESSED / "coco_val_synthetic"
    cfg = corrupt.load_conditions()
    for cond in cfg["conditions"]:
        for sev in sorted(cfg["conditions"][cond]["severities"]):
            sets[f"{cond}_s{sev}"] = syn / f"{cond}_s{sev}" / "coco80" / "coco80.yaml"
    sets["occ_clean"] = syn / "occ_clean" / "coco80" / "coco80.yaml"
    for f in cfg["occlusion"]["fractions"]:
        sets[f"occ_f{int(f * 100)}"] = syn / f"occ_f{int(f * 100)}" / "coco80" / "coco80.yaml"
    for name in REAL_SETS:   # real-condition, test-only sets (scripts/prepare_real_sets.py, scripts/prepare_bdd.py)
        sets[name] = PROCESSED / "real" / name / "coco80" / "coco80.yaml"
    return sets


def reference_for(name: str) -> tuple[str | None, bool]:
    """(reference set name, paired?) - paired means the reference holds the very same images."""
    if name in ("clean_full", "clean_bench1k", "occ_clean", "bdd_clear_day", "bdd_foggy_all", "exdark_all") or name.startswith("dawn_"):
        return None, False
    if name in BDD_REFERENCE:
        return BDD_REFERENCE[name], False
    return (PAIRED_CLEAN["occ"] if name.startswith("occ_") else PAIRED_CLEAN["bench"]), True


def evidence_for(name: str) -> str:
    return "real" if name in REAL_SETS else ("clean" if name.startswith("clean_") or name == "occ_clean" else "synthetic")


def add_ci(r: dict, yaml_path: Path, coco_ids: set[int], n_boot: int, seed: int) -> dict:
    """Bootstrap CIs, object counts and found rates for one evaluated set. Returns {class: per-image entries} for reuse."""
    names80 = data.space_names("coco80")
    gt, stem_to_id = ci.gt_from_yolo(yaml_path.parent, names80)
    preds = json.loads((Path(r["save_dir"]) / "predictions.json").read_text(encoding="utf-8"))
    dets = ci.detections_from_ultralytics(preds, stem_to_id, coco_ids)
    counts = {}
    for a in gt["annotations"]:
        counts[a["category_id"]] = counts.get(a["category_id"], 0) + 1
    project_names = labels.class_names()
    idx_of = {n: labels.to_space(i, "coco80") for i, n in enumerate(project_names)}
    entries = {n: ci.per_image_entries(gt, dets, idx) for n, idx in idx_of.items() if counts.get(idx, 0) > 0}
    report = ci.bootstrap_report(entries, labels.class_roles(), n_boot=n_boot, seed=seed)
    person = idx_of["person"]
    found = {}
    if counts.get(person, 0) > 0:
        for thr in FOUND_IOUS:
            flags = ci.found_flags(gt, dets, person, conf=FOUND_CONF, iou_thr=thr)
            found[f"iou{thr}"] = ci.found_rate_ci(flags, seed=seed)
    r["ci"] = {"n_boot": n_boot, "n_images": len(gt["images"]), "per_class": report["per_class"], "roles": report["roles"],
               "found_person": {"conf": FOUND_CONF, **found}}
    r["objects_per_class"] = {n: counts.get(idx, 0) for n, idx in idx_of.items()}
    return entries


def drop_ci(cur: list, ref: list, paired: bool, n_boot: int, seed: int) -> dict | None:
    """Person AP50-95 drop (reference minus this set) with a CI."""
    rng = np.random.default_rng(seed)
    if paired:
        if len(cur) != len(ref):
            return None
        draws = rng.integers(0, len(cur), size=(n_boot, len(cur)))
        diffs = np.array([ci.ap_from_entries([ref[i] for i in d])[0] - ci.ap_from_entries([cur[i] for i in d])[0] for d in draws])
    else:
        dc = rng.integers(0, len(cur), size=(n_boot, len(cur)))
        dr = rng.integers(0, len(ref), size=(n_boot, len(ref)))
        diffs = np.array([ci.ap_from_entries([ref[i] for i in b])[0] - ci.ap_from_entries([cur[i] for i in a])[0] for a, b in zip(dc, dr)])
    diffs = diffs[~np.isnan(diffs)]
    point = ci.ap_from_entries(ref)[0] - ci.ap_from_entries(cur)[0]
    return {"drop": float(point), "lo": float(np.quantile(diffs, 0.025)), "hi": float(np.quantile(diffs, 0.975)), "paired": paired}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo11n.pt"))
    ap.add_argument("--name", default="baseline_b0_yolo11n")
    ap.add_argument("--sets", nargs="*", help="subset of set names (default: all that exist)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ci", action="store_true", help="skip bootstrap and found-rate (faster)")
    args = ap.parse_args()

    sets = all_sets()
    todo = args.sets or [s for s in sets if sets[s].exists()]
    missing = [s for s in todo if not sets[s].exists()]
    if missing:
        sys.exit(f"missing prepared sets (run prepare_coco.py / make_synthetic.py / prepare_real_sets.py / prepare_bdd.py): {missing}")
    coco_ids = sorted(labels.to_space(i, "coco80") for i in range(len(labels.class_names())))

    out_path = ROOT / "results" / f"{args.name}.json"
    result = {"note": "Evidence labels per set: synthetic = degraded COCO val, real = ExDark/DAWN/BDD100K photographs, clean = reference.",
              "environment": environment_info(), "weights": args.weights, "imgsz": args.imgsz, "n_boot": args.n_boot, "sets": {}}
    if out_path.exists():  # resume
        result["sets"] = json.loads(out_path.read_text(encoding="utf-8")).get("sets", {})

    # evaluation order: references first, so their per-image records are available for the paired drops
    order = sorted(todo, key=lambda n: 0 if n in ("clean_bench1k", "occ_clean", "bdd_clear_day", "bdd_night_clear") else 1)
    entries_cache: dict[str, list] = {}
    t0 = time.time()
    for name in order:
        done = result["sets"].get(name)
        if done and (args.no_ci or "ci" in done) and not (name in entries_cache):
            if not args.no_ci and ("ci" in done) and reference_for(name)[0] is None and name not in ("clean_bench1k", "occ_clean", "bdd_clear_day", "bdd_night_clear"):
                print(f"{name}: already done, skipped", flush=True)
                continue
        r = evaluate(args.weights, str(sets[name]), imgsz=args.imgsz, batch=args.batch, classes=coco_ids,
                     project=str(ROOT / "runs"), name=f"{args.name}_{name}", save_json=not args.no_ci)
        r["roles"] = role_summary(r["per_class"])
        r["evidence"] = evidence_for(name)
        if not args.no_ci:
            entries = add_ci(r, sets[name], set(coco_ids), args.n_boot, args.seed)
            if "person" in entries:
                if name in ("clean_bench1k", "occ_clean", "bdd_clear_day", "bdd_night_clear"):
                    entries_cache[name] = entries["person"]
                ref, paired = reference_for(name)
                if ref and ref in entries_cache and "person" in entries:
                    d = drop_ci(entries["person"], entries_cache[ref], paired, args.n_boot, args.seed)
                    if d:
                        r["person_drop_vs_" + ref] = d
        result["sets"][name] = r
        write_json(out_path, result)
        p = r["roles"]["primary"]
        lo_hi = ""
        if "ci" in r and "person" in r["ci"]["per_class"]:
            c = r["ci"]["per_class"]["person"]
            lo_hi = f" [{c['lo']:.3f}, {c['hi']:.3f}]"
        print(f"{name:18s} person mAP50-95 {p['map50_95']:.3f}{lo_hi}  mAP50 {p['map50']:.3f}  R {p['recall']:.3f}  "
              f"| all-class {r['map50_95']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
