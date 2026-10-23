"""Build the fine-tuning data (never committed): a COCO train2017 subset, a small held-out train-validation split, and a
BDD100K adverse-condition subset (used only by variant A2).

COCO train subset  (data/processed/train/coco_train_subset)
    up to PER_ANIMAL images for each animal class (cat, dog, horse, sheep, cow), plus N_OTHER random further images that
    contain a project class, so persons and vehicles are well covered. ALL project-class boxes in those images are labelled.
COCO train-val     (data/processed/train/coco_trainval_subset)   500 other train2017 images, only to pick the best epoch
BDD adverse        (data/processed/train/bdd_adverse_train)
    foggy (all), rainy (cap), snowy (cap), night (not rainy/snowy/foggy; cap), from the BDD100K TRAIN split.
    rider is mapped to person (configs/datasets.yaml, decision recorded). BDD has no animal labels.

Test data are never used here: COCO val2017, ExDark, DAWN and BDD100K val stay test-only.
Only the needed COCO images are downloaded, one file at a time from images.cocodataset.org.

Run from the project root:   .venv\\Scripts\\python scripts\\prepare_train_data.py
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcv import data, labels  # noqa: E402

COCO = ROOT / "data" / "coco"
BDD = ROOT / "data" / "bdd100k"
OUT = ROOT / "data" / "processed" / "train"
SEED = 20261025
PER_ANIMAL, N_OTHER, N_TRAINVAL = 1200, 4000, 500
BDD_CAPS = {"foggy": 10_000, "rainy": 1000, "snowy": 1000, "night": 1000}       # foggy: keep all that exist
ANIMALS = ("cat", "dog", "horse", "sheep", "cow")


def fetch(fname: str, dest_dir: Path, tries: int = 4) -> bool:
    dest = dest_dir / fname
    if dest.exists():
        return True
    for i in range(tries):
        try:
            with urllib.request.urlopen(f"http://images.cocodataset.org/train2017/{fname}", timeout=40) as r:
                blob = r.read()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(blob)
            with Image.open(tmp) as im:
                im.verify()                      # reject truncated downloads
            tmp.replace(dest)
            return True
        except Exception:
            time.sleep(1.5 * (i + 1))
    return False


def select_coco(coco: dict) -> tuple[list, list]:
    selected = data.select_images(coco, "coco80")           # [(img, rows in coco80 space)]
    names = labels.class_names()
    idx80 = {n: labels.to_space(i, "coco80") for i, n in enumerate(names)}
    rng = np.random.default_rng(SEED)
    chosen: set[int] = set()
    for cls in ANIMALS:
        pool = [i for i, (_, rows) in enumerate(selected) if any(r[0] == idx80[cls] for r in rows)]
        rng.shuffle(pool)
        chosen.update(pool[:PER_ANIMAL])
    rest = [i for i in range(len(selected)) if i not in chosen]
    rng.shuffle(rest)
    chosen.update(rest[:N_OTHER])
    val_pool = [i for i in range(len(selected)) if i not in chosen]
    rng.shuffle(val_pool)
    val = val_pool[:N_TRAINVAL]
    return [selected[i] for i in sorted(chosen)], [selected[i] for i in sorted(val)]


def class_counts(selected: list) -> dict:
    names = labels.class_names()
    inv = {labels.to_space(i, "coco80"): n for i, n in enumerate(names)}
    out = {n: 0 for n in names}
    for _, rows in selected:
        for r in rows:
            out[inv[r[0]]] += 1
    return out


def bdd_adverse() -> list:
    zl = zipfile.ZipFile(BDD / "bdd100k_labels.zip")
    zi = zipfile.ZipFile(BDD / "bdd100k_images_100k.zip")
    members = set(zi.namelist())
    groups: dict[str, list] = {k: [] for k in BDD_CAPS}
    for n in sorted(x for x in zl.namelist() if x.startswith("100k/train/") and x.endswith(".json")):
        d = json.loads(zl.read(n))
        a = d.get("attributes", {})
        w, t = a.get("weather"), a.get("timeofday")
        g = "foggy" if w == "foggy" else "rainy" if w == "rainy" else "snowy" if w == "snowy" else "night" if t == "night" else None
        if g is None:
            continue
        boxes = [(o["category"], o["box2d"]["x1"], o["box2d"]["y1"], o["box2d"]["x2"], o["box2d"]["y2"])
                 for fr in d.get("frames", []) for o in fr.get("objects", []) if o.get("box2d")]
        rows = data.named_boxes_to_rows("bdd100k_train", boxes, 1280, 720, "coco80")
        member = f"100k/train/{d['name']}.jpg"
        if rows and member in members:
            groups[g].append((d["name"], member, rows))
    rng = np.random.default_rng(SEED)
    picked = []
    for g, items in groups.items():
        rng.shuffle(items)
        picked += items[:BDD_CAPS[g]]
        print(f"BDD train group {g}: {len(items)} usable images, using {min(len(items), BDD_CAPS[g])}", flush=True)
    raw = BDD / "raw"
    selected = []
    for name, member, rows in picked:
        if not (raw / member).exists():
            zi.extract(member, raw)
        selected.append(({"file_name": f"bdd_{name}.jpg", "src": str(raw / member), "width": 1280, "height": 720}, rows))
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-coco", action="store_true")
    ap.add_argument("--skip-bdd", action="store_true")
    args = ap.parse_args()
    summary = {}

    if not args.skip_coco:
        ann = COCO / "annotations" / "instances_train2017.json"
        if not ann.exists():
            with zipfile.ZipFile(COCO / "annotations_trainval2017.zip") as z:
                z.extract("annotations/instances_train2017.json", COCO)
        coco = data.load_coco(ann)
        train, val = select_coco(coco)
        images_dir = COCO / "train2017"
        images_dir.mkdir(parents=True, exist_ok=True)
        todo = [img["file_name"] for img, _ in train + val]
        t0, failed = time.time(), []
        with cf.ThreadPoolExecutor(args.workers) as ex:
            for i, (fn, ok) in enumerate(zip(todo, ex.map(lambda f: fetch(f, images_dir), todo)), 1):
                if not ok:
                    failed.append(fn)
                if i % 1000 == 0:
                    print(f"downloaded {i}/{len(todo)} ({time.time() - t0:.0f}s), failed so far {len(failed)}", flush=True)
        bad = set(failed)
        train = [(i, r) for i, r in train if i["file_name"] not in bad]
        val = [(i, r) for i, r in val if i["file_name"] not in bad]
        data.write_set(train, images_dir, OUT / "coco_train_subset" / "coco80", "coco80", "coco_train_subset")
        data.write_set(val, images_dir, OUT / "coco_trainval_subset" / "coco80", "coco80", "coco_trainval_subset")
        summary["coco_train_subset"] = {"images": len(train), "objects": class_counts(train)}
        summary["coco_trainval_subset"] = {"images": len(val), "objects": class_counts(val)}
        summary["coco_download_failed"] = failed

    if not args.skip_bdd:
        sel = bdd_adverse()
        data.write_set(sel, ROOT, OUT / "bdd_adverse_train" / "coco80", "coco80", "bdd_adverse_train")
        summary["bdd_adverse_train"] = {"images": len(sel), "objects": class_counts(sel)}

    OUT.mkdir(parents=True, exist_ok=True)
    prev = json.loads((OUT / "summary.json").read_text(encoding="utf-8")) if (OUT / "summary.json").exists() else {}
    prev.update(summary)
    (OUT / "summary.json").write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
