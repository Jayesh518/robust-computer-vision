"""Fine-tune a COCO-pretrained YOLO on the project data. One call = one variant x one seed.

Variants (everything else is IDENTICAL between them, so differences come only from what is listed):
    B1  control: fine-tune on the clean COCO subset with Ultralytics' default augmentation.
        Separates "fine-tuning on this label set" from "learning robustness".
    A1  B1 + condition-aware augmentation: each training image is degraded with probability 0.5 using the same
        degradation models as the synthetic benchmark (src/rcv/augment.py).
    A2  A1 + the BDD100K adverse-condition subset (fog, rain, snow, night) added to the training data.
        CONFOUND to keep in mind: A2 also has more and different data than A1, not only adverse conditions.

Labels stay in the COCO-80 index space, so the pretrained head is kept (no re-initialised classifier) and the results
are directly comparable with the zero-shot baseline B0. Only the 9 project classes are labelled; the other 71 never
appear in the training labels.

Test data are never touched: COCO val2017, ExDark, DAWN and BDD100K val are test-only. Checkpoints are chosen on the
held-out COCO train-val subset. Fine-tuned weights go to models/ft/<name>.pt (git-ignored).

Run from the project root (never via `python -`, the data-loader workers cannot start from stdin on Windows):
    .venv\\Scripts\\python scripts\\train.py --variant A1 --seed 0 --epochs 10
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcv import data  # noqa: E402
from rcv.augment import ConditionAwareDegradation  # noqa: E402

TRAIN = ROOT / "data" / "processed" / "train"


def make_yaml(variant: str) -> Path:
    train_dirs = [TRAIN / "coco_train_subset" / "coco80" / "images"]
    if variant == "A2":
        train_dirs.append(TRAIN / "bdd_adverse_train" / "coco80" / "images")
    for d in train_dirs + [TRAIN / "coco_trainval_subset" / "coco80" / "images"]:
        if not d.is_dir():
            sys.exit(f"missing training data: {d} (run scripts/prepare_train_data.py)")
    cfg = {"path": str(TRAIN), "train": [str(d) for d in train_dirs],
           "val": str(TRAIN / "coco_trainval_subset" / "coco80" / "images"), "names": data.space_names("coco80")}
    out = TRAIN / f"train_{'A2' if variant == 'A2' else 'coco'}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["B1", "A1", "A2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo11n.pt"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="debug: fraction of the training images to use")
    ap.add_argument("--p-degrade", type=float, default=0.5)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    model_tag = Path(args.weights).stem
    name = args.name or f"{args.variant}_{model_tag}_s{args.seed}"
    kwargs = dict(data=str(make_yaml(args.variant)), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                  seed=args.seed, deterministic=True, workers=args.workers, amp=True, plots=False, val=True,
                  warmup_epochs=1, close_mosaic=3, project=str(ROOT / "runs" / "train"), name=name, exist_ok=True,
                  fraction=args.fraction, device=0)
    if args.variant in ("A1", "A2"):
        kwargs["augmentations"] = [ConditionAwareDegradation(p=args.p_degrade)]
    print(f"== training {name}: {kwargs['data']} | augmentation: {'condition-aware p=%.2f' % args.p_degrade if 'augmentations' in kwargs else 'default only'}", flush=True)

    model = YOLO(args.weights)
    model.train(**kwargs)

    best = ROOT / "runs" / "train" / name / "weights" / "best.pt"
    dest = ROOT / "models" / "ft" / f"{name}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)
    print("saved", dest, flush=True)


if __name__ == "__main__":
    main()
