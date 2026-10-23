"""Independent check of the evaluation pipeline against the official COCO evaluation code.

Pipeline under test:  COCO json -> our label conversion (rcv.labels / rcv.data) -> YOLO txt labels -> Ultralytics val.
Reference:            detections scored by pycocotools COCOeval on the ORIGINAL COCO annotation file.

Two references are computed, because they answer different questions:
  same_detections   the detections Ultralytics val itself produced (saved with save_json), re-scored by pycocotools.
                    Differences here come ONLY from label conversion and scoring, which is what this check is for.
                    This decides pass/fail.
  predict_single    detections from model.predict (one class per box). Ultralytics val uses multi-label NMS (a box may
                    be reported for several classes), which raises AP for confusable classes. Reported to show how
                    much the NMS mode alone changes the numbers; it does not decide pass/fail.
The first run of this check scored only predict_single and showed gaps up to 0.07 (dog); the cause was the NMS mode, not
the labels, which is why same_detections was added (results/crosscheck_coco_attempt1_predict_reference.json is kept).

Small differences (about 0.01 to 0.02) are expected: pycocotools ignores detections that match crowd regions and caps
100 detections per image, Ultralytics counts crowd matches as false positives and caps 300.

Run from the project root:   .venv\\Scripts\\python scripts\\crosscheck_coco_eval.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from rcv import labels  # noqa: E402
from rcv.eval import evaluate, write_json  # noqa: E402

TOLERANCE = 0.03


def score(gt: COCO, detections: list[dict], img_ids: list[int], json_ids: list[int], names: list[str]) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(detections)
    out = {}
    for name, jid in zip(names, json_ids):
        with contextlib.redirect_stdout(io.StringIO()):
            e = COCOeval(gt, dt, "bbox")
            e.params.imgIds, e.params.catIds = img_ids, [jid]
            e.evaluate(); e.accumulate(); e.summarize()
        out[name] = float(e.stats[0])
    return out


def main() -> None:
    weights = str(ROOT / "models" / "yolo11n.pt")
    ann = ROOT / "data" / "coco" / "annotations" / "instances_val2017.json"
    set_dir = ROOT / "data" / "processed" / "coco_val_bench1k" / "coco80"
    images_dir, yaml_path = set_dir / "images", set_dir / "coco80.yaml"
    names = labels.class_names()
    json_ids = [c["coco_json_id"] for c in labels._load("classes.yaml")["classes"]]            # official COCO category ids
    idx80_to_json = {labels.to_space(i, "coco80"): json_ids[i] for i in range(len(names))}
    classes = sorted(idx80_to_json)
    files = sorted(images_dir.glob("*.jpg"))
    img_ids = [int(p.stem) for p in files]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann))

    # 1) our pipeline, and the detections it produced
    ours = evaluate(weights, str(yaml_path), classes=classes, project=str(ROOT / "runs"), name="crosscheck", save_json=True)
    pred_file = Path(ours["save_dir"]) / "predictions.json"
    same = []
    for d in json.loads(pred_file.read_text(encoding="utf-8")):
        idx80 = int(d["category_id"]) - 1        # non-COCO dataset yamls write category_id = class index + 1
        if idx80 in idx80_to_json:
            same.append({"image_id": int(d["image_id"]), "category_id": idx80_to_json[idx80], "bbox": d["bbox"], "score": d["score"]})

    # 2) detections from predict (single label per box)
    model = YOLO(weights)
    single = []
    CHUNK = 8  # small explicit batches: handing predict() 1000 paths at once exhausts a 4 GB GPU
    for start in range(0, len(files), CHUNK):
        chunk = files[start:start + CHUNK]
        for res, path in zip(model.predict([str(p) for p in chunk], conf=0.001, iou=0.6, imgsz=640, half=True,
                                           device=0, batch=CHUNK, verbose=False, classes=classes), chunk):
            for (x0, y0, x1, y1), c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                single.append({"image_id": int(path.stem), "category_id": idx80_to_json[int(c)],
                               "bbox": [x0, y0, x1 - x0, y1 - y0], "score": s})

    ref_same = score(gt, same, img_ids, json_ids, names)
    ref_single = score(gt, single, img_ids, json_ids, names)
    report = {}
    for n in names:
        mine = ours["per_class"].get(n, {}).get("map50_95")
        report[n] = {"ultralytics_on_converted_labels": mine,
                     "pycocotools_same_detections": ref_same[n], "diff_same_detections": abs(mine - ref_same[n]),
                     "pycocotools_predict_single_label": ref_single[n], "diff_predict_single_label": abs(mine - ref_single[n])}
    worst = max(v["diff_same_detections"] for v in report.values())
    out = {"set": "coco_val_bench1k", "n_images": len(files), "n_detections_same": len(same), "n_detections_single": len(single),
           "tolerance": TOLERANCE, "per_class": report, "worst_abs_diff_same_detections": worst,
           "worst_abs_diff_predict_single_label": max(v["diff_predict_single_label"] for v in report.values()),
           "passed": bool(worst <= TOLERANCE)}
    write_json(ROOT / "results" / "crosscheck_coco.json", out)
    print(f"{'class':8s} {'ultralytics':>11s} {'pycoco(same)':>13s} {'diff':>6s} | {'pycoco(single)':>14s} {'diff':>6s}")
    for n, v in report.items():
        print(f"{n:8s} {v['ultralytics_on_converted_labels']:11.3f} {v['pycocotools_same_detections']:13.3f} {v['diff_same_detections']:6.3f} | "
              f"{v['pycocotools_predict_single_label']:14.3f} {v['diff_predict_single_label']:6.3f}")
    print(f"detections: same={len(same)} single={len(single)} | worst diff (same detections) {worst:.4f} -> "
          f"{'PASSED' if out['passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()
