"""Build the synthetic benchmark from the clean bench1k set.

For every condition x severity in configs/conditions.yaml it writes a copy of the bench1k images, degraded by
src/rcv/corrupt.py, with the SAME labels as the clean images (the objects are still there). It also builds the
occlusion sets: a concrete-like slab (src/rcv/occlude.py) hides 25/50/75 % of every person's box.

Output (never committed):  data/processed/coco_val_synthetic/<name>/<space>/{images,labels,<space>.yaml}
   names: <condition>_s<severity>   and   occ_<direction-mix>_f<percent>   plus   occ_clean  (the unoccluded reference)

Run from the project root:   .venv\\Scripts\\python scripts\\make_synthetic.py [--only fog rain] [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcv import corrupt, data, labels, occlude  # noqa: E402

OCC_N = 500
OCC_SEED = 20261024
JPEG_QUALITY = 95


def write_labels(out_space: Path, fname: str, rows) -> None:
    (out_space / "labels").mkdir(parents=True, exist_ok=True)
    (out_space / "labels" / (Path(fname).stem + ".txt")).write_text(
        "\n".join(labels.yolo_line(r) for r in rows) + "\n", encoding="utf-8")


def build(name: str, selected_by_space: dict, images_dir: Path, out_root: Path, transform) -> int:
    """Create one synthetic set (both label spaces). `transform(img, info, rows_proj) -> image`."""
    n = 0
    for (img_info, rows_proj), (_, rows_c80) in zip(selected_by_space["proj"], selected_by_space["coco80"]):
        src = cv2.imread(str(images_dir / img_info["file_name"]))
        if src is None:
            raise FileNotFoundError(images_dir / img_info["file_name"])
        out = transform(src, img_info, rows_proj)
        for space, rows in (("proj", rows_proj), ("coco80", rows_c80)):
            sp = out_root / name / space
            (sp / "images").mkdir(parents=True, exist_ok=True)
            dst = sp / "images" / img_info["file_name"]
            if space == "proj":
                cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                first = dst
            elif not dst.exists():
                try:
                    import os
                    os.link(first, dst)
                except OSError:
                    import shutil
                    shutil.copy2(first, dst)
            write_labels(sp, img_info["file_name"], rows)
        n += 1
    for space in ("proj", "coco80"):
        data.write_dataset_yaml(out_root / name / space, space, name, n)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default=str(ROOT / "data" / "coco"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "coco_val_synthetic"))
    ap.add_argument("--only", nargs="*", help="condition names and/or 'occlusion' to build (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N images")
    args = ap.parse_args()
    coco_dir, out_root = Path(args.coco), Path(args.out)
    images_dir = coco_dir / "val2017"
    coco = data.load_coco(coco_dir / "annotations" / "instances_val2017.json")
    cfg = corrupt.load_conditions()

    # the same bench1k sample as scripts/prepare_coco.py
    sel = {s: data.sample_images(data.select_images(coco, s), 1000, 20261023) for s in ("proj", "coco80")}
    if args.limit:
        sel = {s: v[: args.limit] for s, v in sel.items()}
    want = set(args.only) if args.only else set(cfg["conditions"]) | {"occlusion"}
    manifest = {}
    t0 = time.time()

    for cond in cfg["conditions"]:
        if cond not in want:
            continue
        for sev in sorted(cfg["conditions"][cond]["severities"]):
            name = f"{cond}_s{sev}"
            n = build(name, sel, images_dir, out_root,
                      lambda im, info, rows, c=cond, s=sev: corrupt.apply(im, c, s, key=info["file_name"]))
            manifest[name] = n
            print(f"{name}: {n} images ({time.time() - t0:.0f}s)", flush=True)

    if "occlusion" in want:
        occ = cfg["occlusion"]
        cands = {s: data.occlusion_candidates(data.select_images(coco, s), 40.0) for s in ("proj", "coco80")}
        occ_sel = {s: data.sample_images(cands[s], OCC_N, OCC_SEED) for s in ("proj", "coco80")}
        if args.limit:
            occ_sel = {s: v[: args.limit] for s, v in occ_sel.items()}
        manifest["occ_candidates"] = len(cands["proj"])
        n = build("occ_clean", occ_sel, images_dir, out_root, lambda im, info, rows: im)
        manifest["occ_clean"] = n
        print(f"occ_clean: {n} images", flush=True)
        for frac in occ["fractions"]:
            def occlude_all(im, info, rows, f=frac):
                out = im
                for i, r in enumerate(rows):
                    if r[0] != 0:  # person only
                        continue
                    direction = occ["directions"][int(np.random.default_rng(
                        corrupt.derive_seed(OCC_SEED, info["file_name"], f"dir{i}", int(f * 100))).integers(0, 4))]
                    box = data.yolo_row_to_xyxy(r, info["width"], info["height"])
                    out, _ = occlude.occlude_box(out, box, f, direction, key=f"{info['file_name']}#{i}",
                                                 seed_base=OCC_SEED, jitter=occ["edge_jitter"])
                return out
            name = f"occ_f{int(frac * 100)}"
            n = build(name, occ_sel, images_dir, out_root, occlude_all)
            manifest[name] = n
            print(f"{name}: {n} images ({time.time() - t0:.0f}s)", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("done", manifest)


if __name__ == "__main__":
    main()
