"""Environment smoke test: is the GPU stack working, and does the degradation + detection path run end to end?

This is NOT an accuracy experiment. It uses two sample images that ship with Ultralytics and only counts detections,
so it must never be quoted as a result. It records environment details and batch-1 latency for later reference.

Usage (from the project root):
    .venv\\Scripts\\python scripts\\smoke_test.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from rcv import corrupt, labels  # noqa: E402
from rcv.eval import environment_info, measure_latency, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo11n.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "smoke.json"))
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    import ultralytics
    from ultralytics import YOLO

    Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)  # downloads the official pretrained weights on first use
    assets = Path(ultralytics.__file__).parent / "assets"
    images = [assets / "bus.jpg", assets / "zidane.jpg"]
    names80 = model.names
    keep = set(labels.coco80_to_project())

    def count(img_bgr) -> dict:
        r = model.predict(img_bgr, conf=args.conf, verbose=False, device=0)[0]
        out: dict[str, int] = {}
        for c in r.boxes.cls.tolist():
            if int(c) in keep:
                out[names80[int(c)]] = out.get(names80[int(c)], 0) + 1
        return out

    rows = []
    for path in images:
        img = cv2.imread(str(path))
        row = {"image": path.name, "clean": count(img)}
        for cond in corrupt.CONDITIONS:
            row[f"{cond}_sev2"] = count(corrupt.apply(img, cond, 2, key=path.name))
        rows.append(row)
        print(path.name, row)

    latency = measure_latency(args.weights)
    payload = {
        "note": "Smoke test only: detection counts on 2 sample images. Not an accuracy result.",
        "environment": environment_info(),
        "detections_6_classes": rows,
        "latency_batch1": latency,
    }
    print("latency:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in latency.items()})
    print("wrote", write_json(args.out, payload))


if __name__ == "__main__":
    main()
