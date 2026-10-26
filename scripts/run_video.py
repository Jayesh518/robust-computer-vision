"""Run a detector over a video and write (a) an annotated copy and (b) per-frame statistics. Everything stays local (outputs/ is git-ignored).

There is NO ground truth for a casual video, so this reports behaviour (how many persons per frame, how stable the count is, speed), not accuracy.
A person found in the water is reported as "person detected", nothing more: no claim about whether anyone is alive, conscious or unhurt.

    .venv\\Scripts\\python scripts\\run_video.py test1.mp4 --weights models/yolo11s.pt --imgsz 640 --conf 0.35
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

from rcv import labels  # noqa: E402

BGR = {"primary": (52, 104, 235), "secondary": (214, 120, 42), "context": (122, 175, 27)}      # person, animals, vehicles


def summarise(person_confs: list[list[float]], conf: float) -> dict:
    """Behaviour statistics from the per-frame person confidences (no ground truth involved)."""
    counts = np.array([sum(c >= conf for c in f) for f in person_confs])
    flicker = float(np.abs(np.diff(counts)).mean()) if len(counts) > 1 else 0.0
    allc = [c for f in person_confs for c in f if c >= conf]
    return {"frames": int(len(counts)), "conf": conf, "mean_persons_per_frame": float(counts.mean()) if len(counts) else 0.0,
            "max_persons_in_a_frame": int(counts.max()) if len(counts) else 0,
            "frames_with_no_person": int((counts == 0).sum()), "fraction_frames_with_a_person": float((counts > 0).mean()) if len(counts) else 0.0,
            "count_flicker": flicker,                        # mean absolute change of the person count between consecutive frames
            "mean_person_confidence": float(np.mean(allc)) if allc else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo11s.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out-width", type=int, default=540, help="width of the annotated copy (height follows)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    to_proj = labels.coco80_to_project()
    names, roles = labels.class_names(), labels.class_roles()
    model = YOLO(args.weights)
    tag = args.tag or f"{Path(args.weights).stem}_{args.imgsz}"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps, W, H = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(3)), int(cap.get(4))
    cap.release()
    ow = args.out_width
    oh = int(round(H * ow / W))
    writer = cv2.VideoWriter(str(out_dir / f"{Path(args.video).stem}_{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    person_confs, times = [], []
    t_all = time.time()
    # very low conf here so the stored confidences allow several thresholds; drawing uses args.conf
    for r in model.predict(args.video, stream=True, imgsz=args.imgsz, conf=0.05, device=args.device, classes=sorted(to_proj), verbose=False, half=True):
        frame = cv2.resize(r.orig_img, (ow, oh))
        sx, sy = ow / W, oh / H
        confs = []
        for c, p, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
            name = names[to_proj[int(c)]]
            if roles[name] == "primary":
                confs.append(float(p))
            if p < args.conf:
                continue
            x1, y1, x2, y2 = (int(v) for v in (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy))
            col = BGR[roles[name]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"{name} {p:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        person_confs.append(confs)
        times.append(sum(r.speed.values()))
        writer.write(frame)
    writer.release()
    stats = {"video": Path(args.video).name, "weights": Path(args.weights).name, "imgsz": args.imgsz, "device": args.device,
             "per_threshold": {str(t): summarise(person_confs, t) for t in (0.25, 0.35, 0.5)},
             "ms_per_frame_median": float(np.median(times)), "wall_seconds": time.time() - t_all}
    (out_dir / f"{Path(args.video).stem}_{tag}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / f"{Path(args.video).stem}_{tag}_persons.json").write_text(json.dumps(person_confs), encoding="utf-8")
    s = stats["per_threshold"][str(args.conf)] if str(args.conf) in stats["per_threshold"] else stats["per_threshold"]["0.35"]
    print(f"{tag}: {s['frames']} frames | persons/frame {s['mean_persons_per_frame']:.2f} (max {s['max_persons_in_a_frame']}) | "
          f"frames with no person {s['frames_with_no_person']} | count flicker {s['count_flicker']:.2f} | {stats['ms_per_frame_median']:.0f} ms/frame (median) | wall {stats['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
