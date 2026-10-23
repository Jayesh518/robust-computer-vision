"""Bootstrap confidence intervals for detection AP, and a lenient "was it found?" rate.

Why: small classes (animals) and small sets (DAWN, BDD rain) give noisy AP. A single number hides that.

How the AP bootstrap works: the matching of detections to ground truth is done ONCE per class with pycocotools
(`COCOeval.evaluate`). Each image then has its own matching record. A bootstrap resample draws images with replacement,
concatenates their records and recomputes AP exactly as `COCOeval.accumulate` does (101-point interpolated precision,
IoU 0.50:0.95, area 'all', at most 100 detections per image). `ap_from_entries` over ALL images reproduces COCOeval's AP
(tested), so the interval is centred on the same number that is reported elsewhere.

The same image indices are used for every class in a resample, so role averages (secondary = animals, context =
vehicles) are resampled consistently.

Found rate: a ground-truth box counts as FOUND if some detection of the same class with confidence >= `conf` overlaps it
with IoU >= `iou_thr`. A lenient threshold (0.1 to 0.3) answers "was a person detected there at all?", which is the
question that matters when the ground-truth box is the FULL person and part of the person is hidden.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
MAX_DETS = 100


# ---------------------------------------------------------------- ground truth and detections in COCO form
def norm_stem(stem: str) -> str:
    """Key under which an image is looked up. Ultralytics writes a numeric file name such as '000000123456' as the integer
    123456 in predictions.json, so numeric names are normalised (leading zeros dropped); other names are kept as they are."""
    return str(int(stem)) if stem.isdigit() else stem


def gt_from_yolo(set_dir: str | Path, class_names: dict[int, str]) -> tuple[dict, dict[str, int]]:
    """COCO-format ground truth built from <set_dir>/images and <set_dir>/labels. Returns (coco dict, stem -> image id)."""
    from PIL import Image

    set_dir = Path(set_dir)
    images, anns, stem_to_id = [], [], {}
    for i, img in enumerate(sorted((set_dir / "images").iterdir()), start=1):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        with Image.open(img) as im:
            w, h = im.size
        stem_to_id[norm_stem(img.stem)] = i
        images.append({"id": i, "file_name": img.name, "width": w, "height": h})
        lab = set_dir / "labels" / (img.stem + ".txt")
        if not lab.exists():
            continue
        for line in lab.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *(float(v) for v in p[1:5])
            x0, y0, ww, hh = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": c, "bbox": [x0, y0, ww, hh],
                         "area": ww * hh, "iscrowd": 0})
    cats = [{"id": c, "name": n} for c, n in class_names.items()]
    return {"images": images, "annotations": anns, "categories": cats}, stem_to_id


def detections_from_ultralytics(pred_json: list[dict], stem_to_id: dict[str, int], keep_classes: set[int]) -> list[dict]:
    """Ultralytics val `predictions.json` (category_id = class index + 1 for non-COCO yamls) -> COCO detections."""
    out = []
    for d in pred_json:
        iid = stem_to_id.get(norm_stem(str(d["image_id"])))
        cls = int(d["category_id"]) - 1
        if iid is None or cls not in keep_classes:
            continue
        out.append({"image_id": iid, "category_id": cls, "bbox": d["bbox"], "score": d["score"]})
    return out


# ---------------------------------------------------------------- per-image matching records and AP
def per_image_entries(gt_dict: dict, detections: list[dict], category_id: int) -> list[Optional[dict]]:
    """pycocotools matching record for each image (in the order of gt_dict['images']) for one class."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        dets = [d for d in detections if d["category_id"] == category_id]
        dt = gt.loadRes(dets) if dets else COCO()
        e = COCOeval(gt, dt, "bbox")
        e.params.catIds = [category_id]
        e.params.imgIds = [im["id"] for im in gt_dict["images"]]
        e.evaluate()
    n = len(e.params.imgIds)
    return list(e.evalImgs[:n])           # first n records = area range 'all'


def ap_from_entries(entries: Sequence[Optional[dict]]) -> tuple[float, float]:
    """(AP50-95, AP50) from per-image records, mirroring COCOeval.accumulate. Returns (nan, nan) when there is no GT."""
    es = [e for e in entries if e is not None]
    if not es:
        return float("nan"), float("nan")
    scores = np.concatenate([e["dtScores"][:MAX_DETS] for e in es])
    order = np.argsort(-scores, kind="mergesort")
    dtm = np.concatenate([e["dtMatches"][:, :MAX_DETS] for e in es], axis=1)[:, order]
    dtig = np.concatenate([e["dtIgnore"][:, :MAX_DETS] for e in es], axis=1)[:, order]
    gtig = np.concatenate([e["gtIgnore"] for e in es])
    npig = int(np.count_nonzero(gtig == 0))
    if npig == 0:
        return float("nan"), float("nan")
    tps = np.logical_and(dtm, np.logical_not(dtig))
    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtig))
    tp_sum = np.cumsum(tps, axis=1).astype(float)
    fp_sum = np.cumsum(fps, axis=1).astype(float)
    aps = []
    for tp, fp in zip(tp_sum, fp_sum):
        if tp.size == 0:
            aps.append(0.0)
            continue
        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]        # monotone precision envelope (same as the COCOeval loop)
        idx = np.searchsorted(rc, REC_THRS, side="left")
        q = np.zeros(len(REC_THRS))
        ok = idx < len(pr)
        q[ok] = pr[idx[ok]]
        aps.append(float(q.mean()))
    return float(np.mean(aps)), aps[0]


# ---------------------------------------------------------------- bootstrap
def bootstrap_ap(entries_by_class: dict[str, list], n_boot: int = 200, alpha: float = 0.05, seed: int = 0) -> dict:
    """Point estimate and percentile CI of AP50-95 per class, resampling IMAGES (same draw for every class)."""
    classes = list(entries_by_class)
    n_img = len(next(iter(entries_by_class.values())))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_img, size=(n_boot, n_img))
    out = {}
    for c in classes:
        ents = entries_by_class[c]
        point = ap_from_entries(ents)[0]
        vals = np.array([ap_from_entries([ents[i] for i in d])[0] for d in draws])
        vals = vals[~np.isnan(vals)]
        out[c] = {"ap": point, "lo": float(np.quantile(vals, alpha / 2)) if vals.size else float("nan"),
                  "hi": float(np.quantile(vals, 1 - alpha / 2)) if vals.size else float("nan"), "n_boot_valid": int(vals.size)}
    return out


def bootstrap_role(entries_by_class: dict[str, list], members: Sequence[str], n_boot: int = 200, alpha: float = 0.05,
                   seed: int = 0) -> dict:
    """Macro-mean AP50-95 over `members` (only those with ground truth) with a CI from consistent image resampling."""
    members = [m for m in members if m in entries_by_class and not np.isnan(ap_from_entries(entries_by_class[m])[0])]
    if not members:
        return {"classes": [], "ap": None, "lo": None, "hi": None}
    n_img = len(entries_by_class[members[0]])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_img, size=(n_boot, n_img))
    point = float(np.mean([ap_from_entries(entries_by_class[m])[0] for m in members]))
    vals = []
    for d in draws:
        per = [ap_from_entries([entries_by_class[m][i] for i in d])[0] for m in members]
        per = [p for p in per if not np.isnan(p)]
        if per:
            vals.append(np.mean(per))
    vals = np.array(vals)
    return {"classes": sorted(members), "ap": point, "lo": float(np.quantile(vals, alpha / 2)),
            "hi": float(np.quantile(vals, 1 - alpha / 2))}


def bootstrap_report(entries_by_class: dict[str, list], roles: dict[str, str], n_boot: int = 200, alpha: float = 0.05,
                     seed: int = 0) -> dict:
    """Per-class and per-role (macro mean) AP50-95 with CIs, resampling images once and sharing the draws.

    `roles` maps class name -> role. Classes with no ground truth are skipped. Same draws as bootstrap_ap/bootstrap_role
    for the same seed and image count, so the numbers agree with them.
    """
    classes = [c for c in entries_by_class if not np.isnan(ap_from_entries(entries_by_class[c])[0])]
    if not classes:
        return {"per_class": {}, "roles": {}}
    n_img = len(entries_by_class[classes[0]])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_img, size=(n_boot, n_img))
    point = {c: ap_from_entries(entries_by_class[c])[0] for c in classes}
    samples = {c: np.array([ap_from_entries([entries_by_class[c][i] for i in d])[0] for d in draws]) for c in classes}
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    per_class = {}
    for c in classes:
        v = samples[c][~np.isnan(samples[c])]
        per_class[c] = {"ap": point[c], "lo": float(np.quantile(v, lo_q)), "hi": float(np.quantile(v, hi_q)),
                        "n_boot_valid": int(v.size)}
    out_roles = {}
    for role in ("primary", "secondary", "context"):
        members = [c for c in classes if roles.get(c) == role]
        if not members:
            out_roles[role] = {"classes": [], "ap": None, "lo": None, "hi": None}
            continue
        stack = np.vstack([samples[c] for c in members])
        with np.errstate(all="ignore"):
            macro = np.nanmean(stack, axis=0)
        macro = macro[~np.isnan(macro)]
        out_roles[role] = {"classes": sorted(members), "ap": float(np.mean([point[c] for c in members])),
                           "lo": float(np.quantile(macro, lo_q)), "hi": float(np.quantile(macro, hi_q))}
    return {"per_class": per_class, "roles": out_roles}


# ---------------------------------------------------------------- lenient found rate
def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4), all xywh."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def found_flags(gt_dict: dict, detections: list[dict], category_id: int, conf: float, iou_thr: float) -> list[np.ndarray]:
    """Per image: boolean array, one entry per ground-truth box of the class (True = found)."""
    gts: dict[int, list] = {}
    for a in gt_dict["annotations"]:
        if a["category_id"] == category_id:
            gts.setdefault(a["image_id"], []).append(a["bbox"])
    dts: dict[int, list] = {}
    for d in detections:
        if d["category_id"] == category_id and d["score"] >= conf:
            dts.setdefault(d["image_id"], []).append(d["bbox"])
    flags = []
    for im in gt_dict["images"]:
        g = np.array(gts.get(im["id"], []), dtype=float).reshape(-1, 4)
        d = np.array(dts.get(im["id"], []), dtype=float).reshape(-1, 4)
        if len(g) == 0:
            flags.append(np.zeros(0, bool))
        elif len(d) == 0:
            flags.append(np.zeros(len(g), bool))
        else:
            flags.append((_iou(g, d) >= iou_thr).any(axis=1))
    return flags


def found_rate_ci(flags: list[np.ndarray], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Found rate over all GT boxes, with an image-level bootstrap CI."""
    total = sum(len(f) for f in flags)
    if total == 0:
        return {"rate": None, "lo": None, "hi": None, "n_gt": 0}
    found = np.array([int(f.sum()) for f in flags])
    count = np.array([len(f) for f in flags])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(flags), size=(n_boot, len(flags)))
    num, den = found[draws].sum(axis=1), count[draws].sum(axis=1)
    ok = den > 0
    rates = num[ok] / den[ok]
    return {"rate": float(found.sum() / total), "lo": float(np.quantile(rates, alpha / 2)),
            "hi": float(np.quantile(rates, 1 - alpha / 2)), "n_gt": int(total)}
