"""Build YOLO-format evaluation sets from COCO annotations, restricted to the project classes.

Ultralytics finds a label file by swapping '/images/' for '/labels/' in the image path, so each prepared set has the
layout   <out>/images/*.jpg   <out>/labels/*.txt   <out>/<space>.yaml.
Images are hard-linked from the source folder when possible (no extra disk use), otherwise copied.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

import yaml

from . import labels

# Standard 80-class COCO names, in the order a COCO-pretrained YOLO outputs them.
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def space_names(space: str) -> dict[int, str]:
    """Class index -> name for the requested label space."""
    if space == "proj":
        return dict(enumerate(labels.class_names()))
    if space == "coco80":
        return dict(enumerate(COCO80))
    raise ValueError(f"space must be one of {labels.SPACES}, got {space!r}")


def load_coco(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select_images(coco: dict, space: str = "proj", min_side_px: float = 0.0) -> list[tuple[dict, list[tuple]]]:
    """Images that contain at least one target-class object, with their YOLO rows in `space`.

    Images with no target object are skipped (they cannot produce a target-class result). Rows for crowd regions are
    dropped by `labels.coco_annotations_to_yolo`. `min_side_px` drops objects whose box is smaller than that on its
    shorter side (used by the occlusion subset, where tiny people cannot be meaningfully occluded).
    """
    by_image: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    out = []
    for img in sorted(coco["images"], key=lambda i: i["id"]):
        anns = by_image.get(img["id"], [])
        rows = labels.coco_annotations_to_yolo(anns, img["width"], img["height"], space)
        if rows:
            out.append((img, rows))
    return out


def sample_images(selected: list[tuple[dict, list[tuple]]], n: int, seed: int) -> list[tuple[dict, list[tuple]]]:
    """Deterministic sample of up to n images (sorted by id afterwards, so order never depends on the sampler)."""
    import numpy as np

    if n >= len(selected):
        return list(selected)
    idx = np.random.default_rng(seed).choice(len(selected), size=n, replace=False)
    return [selected[i] for i in sorted(int(i) for i in idx)]


def occlusion_candidates(selected: list[tuple[dict, list[tuple]]], min_side_px: float = 40.0,
                         person_class: int = 0) -> list[tuple[dict, list[tuple]]]:
    """Images with at least one person where EVERY person box has its shorter side >= min_side_px.

    Used for the occlusion test: all persons in these images are occluded, so no small, un-occluded person is left in
    the ground truth to blur the result.
    """
    out = []
    for img, rows in selected:
        persons = [r for r in rows if r[0] == person_class]
        if not persons:
            continue
        if all(min(r[3] * img["width"], r[4] * img["height"]) >= min_side_px for r in persons):
            out.append((img, rows))
    return out


def yolo_row_to_xyxy(row: tuple, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """YOLO (cls, cx, cy, w, h) normalised -> pixel (x0, y0, x1, y1)."""
    _, cx, cy, w, h = row
    return ((cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h)


def pixel_xyxy_to_yolo(x0: float, y0: float, x1: float, y1: float, img_w: int, img_h: int):
    """Pixel corner box -> YOLO (cx, cy, w, h) normalised and clipped; None if empty after clipping."""
    return labels.coco_bbox_to_yolo([x0, y0, x1 - x0, y1 - y0], img_w, img_h)


def parse_voc_xml(path: str | Path) -> dict:
    """Pascal VOC annotation (used by DAWN): {'width', 'height', 'objects': [(name, x0, y0, x1, y1), ...]}."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    size = root.find("size")
    objects = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        objects.append((o.findtext("name", "").strip(), float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                        float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
    return {"width": int(size.findtext("width")), "height": int(size.findtext("height")), "objects": objects}


def parse_exdark_annotation(path: str | Path) -> list[tuple[str, float, float, float, float]]:
    """ExDark ground-truth file: header line '% bbGt version=3', then 'Class l t w h <7 unused columns>' per object.

    Returns [(class_name, x0, y0, x1, y1), ...] in pixels (the file stores left, top, width, height).
    """
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        p = line.split()
        l, t, w, h = (float(v) for v in p[1:5])
        out.append((p[0], l, t, l + w, t + h))
    return out


def named_boxes_to_rows(dataset: str, boxes: Iterable[tuple], img_w: int, img_h: int, space: str = "proj") -> list[tuple]:
    """[(name, x0, y0, x1, y1)] in pixels -> YOLO rows in `space`, keeping only classes mapped for `dataset`.

    Uses the exact-match name map in configs/datasets.yaml, so an unmapped name (for example 'bike') is dropped.
    """
    rows = []
    for name, x0, y0, x1, y1 in boxes:
        cls = labels.map_name(dataset, name, space)
        if cls is None:
            continue
        box = pixel_xyxy_to_yolo(x0, y0, x1, y1, img_w, img_h)
        if box is not None:
            rows.append((cls, *box))
    return rows


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_set(selected: Iterable[tuple[dict, list[tuple]]], images_dir: str | Path, out_dir: str | Path,
              space: str = "proj", name: str = "set") -> Path:
    """Write images (linked) and YOLO labels to out_dir and return the path of the dataset yaml."""
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for img, rows in selected:
        fname = img["file_name"]
        src = Path(img["src"]) if "src" in img else images_dir / fname   # "src": full source path when files sit in subfolders
        _link_or_copy(src, out_dir / "images" / fname)
        (out_dir / "labels" / (Path(fname).stem + ".txt")).write_text(
            "\n".join(labels.yolo_line(r) for r in rows) + "\n", encoding="utf-8")
        n += 1
    return write_dataset_yaml(out_dir, space, name, n)


def write_dataset_yaml(out_dir: str | Path, space: str, name: str = "set", n_images: Optional[int] = None) -> Path:
    out_dir = Path(out_dir)
    cfg = {"path": str(out_dir.resolve()), "train": "images", "val": "images", "names": space_names(space)}
    if n_images is not None:
        cfg["n_images"] = n_images          # informational; ignored by Ultralytics
    cfg["description"] = f"{name} ({space} label space)"
    p = out_dir / f"{space}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p
