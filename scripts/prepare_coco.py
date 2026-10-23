"""Unpack COCO val2017 and build the clean evaluation sets.

Produces, under data/processed/ (never committed):
  coco_val_clean/{coco80,proj}/      all val images that contain at least one project-class object
  coco_val_bench1k/{coco80,proj}/    a fixed random sample of 1000 of them (seed in configs): the images every
                                     degraded version is built from, so clean-vs-degraded comparisons are paired

Run from the project root:   .venv\\Scripts\\python scripts\\prepare_coco.py
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcv import data  # noqa: E402

BENCH_N = 1000
BENCH_SEED = 20261023


def extract(zip_path: Path, dest: Path, only_prefix: str | None = None, only_file: str | None = None) -> None:
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist()
                   if (only_prefix is None or m.startswith(only_prefix)) and (only_file is None or m.endswith(only_file))]
        z.extractall(dest, members)
    print(f"extracted {len(members)} files from {zip_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default=str(ROOT / "data" / "coco"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()
    coco_dir, out = Path(args.coco), Path(args.out)

    if not (coco_dir / "val2017").is_dir():
        extract(coco_dir / "val2017.zip", coco_dir, only_prefix="val2017/")
    ann = coco_dir / "annotations" / "instances_val2017.json"
    if not ann.exists():
        extract(coco_dir / "annotations_trainval2017.zip", coco_dir, only_file="instances_val2017.json")
    coco = data.load_coco(ann)
    print("val2017 images:", len(coco["images"]), "| annotations:", len(coco["annotations"]))

    summary = {}
    for space in ("coco80", "proj"):
        selected = data.select_images(coco, space)
        bench = data.sample_images(selected, BENCH_N, BENCH_SEED)
        data.write_set(selected, coco_dir / "val2017", out / "coco_val_clean" / space, space, "COCO val2017 clean")
        data.write_set(bench, coco_dir / "val2017", out / "coco_val_bench1k" / space, space, "COCO val2017 bench1k")
        summary[space] = {"clean_images": len(selected), "bench_images": len(bench),
                          "clean_objects": sum(len(r) for _, r in selected), "bench_objects": sum(len(r) for _, r in bench)}
    print(summary)


if __name__ == "__main__":
    main()
