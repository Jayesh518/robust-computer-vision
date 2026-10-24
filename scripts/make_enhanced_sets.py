"""Build the phase 6 (E1) enhanced evaluation sets from the existing prepared sets.

For every source set in enhance.PLAN this writes <source>__none (re-encoded, not enhanced: the JPEG control) and
<source>__<method> for each planned method, under data/processed/enhanced/. Image names are kept, labels are copied unchanged
(enhancement does not move or resize anything), and a dataset yaml is written. Resumable: finished sets are skipped.
It also measures the enhancement time per image (CPU, native size, 100 sample images) into results/enhancement_latency.json.

    .venv\\Scripts\\python scripts\\make_enhanced_sets.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rcv import enhance  # noqa: E402
import run_baseline as rb  # noqa: E402

OUT = ROOT / "data" / "processed" / "enhanced"
JPEG_QUALITY = 95


def build(source: str, method: str, src_yaml: Path) -> Path:
    dest = OUT / f"{source}__{method}" / "coco80"
    yaml_path = dest / "coco80.yaml"
    n_src = len(list((src_yaml.parent / "images").iterdir()))
    if yaml_path.exists() and len(list((dest / "images").iterdir())) == n_src:
        return yaml_path
    (dest / "images").mkdir(parents=True, exist_ok=True)
    fn = enhance.METHODS[method]
    for p in sorted((src_yaml.parent / "images").iterdir()):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"unreadable image: {p}")
        cv2.imwrite(str(dest / "images" / (p.stem + ".jpg")), fn(img), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if (dest / "labels").exists():
        shutil.rmtree(dest / "labels")
    shutil.copytree(src_yaml.parent / "labels", dest / "labels")
    cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    cfg["path"] = str(dest.resolve())
    cfg["description"] = f"{source} enhanced with {method}"
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return yaml_path


def time_method(src_yaml: Path, method: str, n: int = 100) -> dict:
    files = sorted((src_yaml.parent / "images").iterdir())[:: max(1, len(list((src_yaml.parent / "images").iterdir())) // n)][:n]
    imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in files]
    fn = enhance.METHODS[method]
    fn(imgs[0])  # warm-up
    t = []
    for im in imgs:
        t0 = time.perf_counter()
        fn(im)
        t.append((time.perf_counter() - t0) * 1000)
    h, w = int(np.median([i.shape[0] for i in imgs])), int(np.median([i.shape[1] for i in imgs]))
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.quantile(t, 0.95)), "n": len(t), "median_size_hw": [h, w]}


def main() -> None:
    sets = rb.all_sets()
    latency: dict = {}
    for source, methods in enhance.PLAN.items():
        if not sets[source].exists():
            sys.exit(f"missing source set {source}")
        for m in ["none"] + methods:
            t0 = time.time()
            build(source, m, sets[source])
            print(f"{source}__{m:6s} ready ({time.time() - t0:.0f}s)", flush=True)
            if m != "none":
                latency[f"{source}__{m}"] = time_method(sets[source], m)
    out = ROOT / "results" / "enhancement_latency.json"
    out.write_text(json.dumps({"note": "CPU, single call per image at the stored image size, 100 sampled images; OpenCV default threading",
                               "opencv": cv2.__version__, "methods": latency}, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
