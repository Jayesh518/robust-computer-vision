"""Phase 8 data: a 30k-image COCO train superset and fixed, NESTED subset manifests for the data-size sweep.

Nothing here changes the existing training data. The existing 9,831-image set is re-derived with the same seeded selection and must match
the files on disk exactly (checked). Extra images (random COCO train2017 images that contain a project class and are not in the existing
set or the 500-image train-validation split) are added up to 30,000 images in total and written next to the others
(data/processed/train/coco_train_extra). Manifests (data/processed/train/manifests/*.txt, one absolute image path per line):

    coco_train_2k  within  coco_train_5k  within  coco_train_9.8k (= the existing set)  within  coco_train_30k

2k and 5k come from one seeded shuffle of the existing set, so they are identical for every seed and every setting.
The test sets are never touched; the train-validation split is the existing 500 images.

    .venv\\Scripts\\python scripts\\prepare_sweep_data.py            (downloads about 20k images, resumable)
    .venv\\Scripts\\python scripts\\prepare_sweep_data.py --dry-run  (counts and checks only, no download)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rcv import data  # noqa: E402
import prepare_train_data as ptd  # noqa: E402

TRAIN = ptd.OUT
MANIFESTS = TRAIN / "manifests"
SHUFFLE_SEED = 20261026
TARGET_TOTAL = 30_000
SMALL_SIZES = {"coco_train_2k": 2_000, "coco_train_5k": 5_000}


def nested_order(names: list[str], seed: int) -> list[str]:
    """Deterministic shuffle of the sorted names; prefixes of the result are nested random subsets."""
    ordered = sorted(names)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered))
    return [ordered[i] for i in perm]


def build_manifests(base_names: list[str], extra_names: list[str], seed: int = SHUFFLE_SEED) -> dict[str, list[str]]:
    """File-name lists for the four training sizes. 2k within 5k within base (the existing set) within base + extra."""
    order = nested_order(base_names, seed)
    out = {k: order[:n] for k, n in SMALL_SIZES.items()}
    out["coco_train_9.8k"] = sorted(base_names)
    out["coco_train_30k"] = sorted(base_names) + sorted(extra_names)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ann = ptd.COCO / "annotations" / "instances_train2017.json"
    coco = data.load_coco(ann)
    train, val = ptd.select_coco(coco)                                     # same seeded selection as the existing data
    base_names = [img["file_name"] for img, _ in train]
    on_disk = sorted(p.name for p in (TRAIN / "coco_train_subset" / "coco80" / "images").iterdir())
    # a handful of images may have failed to download originally; compare against what exists
    expected = sorted(n for n in base_names if n in set(on_disk))
    if expected != on_disk:
        sys.exit("the re-derived selection does not match the files in coco_train_subset: stop and investigate")
    base = [(i, r) for i, r in train if i["file_name"] in set(on_disk)]
    print(f"existing training set verified: {len(base)} images (selection matches disk)")

    taken = {i["file_name"] for i, _ in train} | {i["file_name"] for i, _ in val}
    pool = [(i, r) for i, r in data.select_images(coco, "coco80") if i["file_name"] not in taken]
    rng = np.random.default_rng(SHUFFLE_SEED + 1)
    rng.shuffle(pool)
    need = TARGET_TOTAL - len(base)
    extra = sorted(pool[:need], key=lambda t: t[0]["file_name"])
    print(f"extra images to use: {len(extra)} (pool of {len(pool)} unused images with a project class)")
    if args.dry_run:
        return

    images_dir = ptd.COCO / "train2017"
    todo = [i["file_name"] for i, _ in extra]
    failed, t0 = [], time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for k, (fn, ok) in enumerate(zip(todo, ex.map(lambda f: ptd.fetch(f, images_dir), todo)), 1):
            if not ok:
                failed.append(fn)
            if k % 2000 == 0:
                print(f"downloaded {k}/{len(todo)} ({time.time() - t0:.0f}s), failed so far {len(failed)}", flush=True)
    bad = set(failed)
    extra = [(i, r) for i, r in extra if i["file_name"] not in bad]
    data.write_set(extra, images_dir, TRAIN / "coco_train_extra" / "coco80", "coco80", "coco_train_extra")

    man = build_manifests([i["file_name"] for i, _ in base], [i["file_name"] for i, _ in extra])
    where = {n: TRAIN / "coco_train_subset" / "coco80" / "images" for n in man["coco_train_9.8k"]}
    where.update({i["file_name"]: TRAIN / "coco_train_extra" / "coco80" / "images" for i, _ in extra})
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    for name, names in man.items():
        (MANIFESTS / f"{name}.txt").write_text("\n".join(str(where[n] / n) for n in names) + "\n", encoding="utf-8")
    summary = {"sizes": {k: len(v) for k, v in man.items()}, "download_failed": failed,
               "extra_objects": ptd.class_counts(extra), "shuffle_seed": SHUFFLE_SEED}
    (MANIFESTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["sizes"]), "failed:", len(failed))


if __name__ == "__main__":
    main()
