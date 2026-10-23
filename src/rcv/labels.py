"""Label space and annotation conversion.

The project uses 9 classes (configs/classes.yaml): person (primary), five animals (secondary) and three vehicles (context).
All are COCO classes, so a COCO-pretrained YOLO already knows them.

Two label "spaces" are supported because the two kinds of model speak different class indices:
  proj   ids 0..5, used by fine-tuned project-class models
  coco80  the standard 80-class indices, used by a COCO-pretrained model (so the zero-shot baseline is evaluated fairly)
Other datasets are mapped into the same space with the name maps in configs/datasets.yaml.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
SPACES = ("proj", "coco80")


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _classes() -> list[dict]:
    return sorted(_load("classes.yaml")["classes"], key=lambda c: c["id"])


def class_names() -> list[str]:
    """Project class names ordered by project id."""
    return [c["name"] for c in _classes()]


def class_roles() -> dict[str, str]:
    """Class name -> role ('primary' person, 'secondary' animals, 'context' vehicles). Used to group reported results."""
    return {c["name"]: c["role"] for c in _classes()}


def _check_space(space: str) -> None:
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}")


def to_space(project_id: int, space: str = "proj") -> int:
    """Convert a project class id to the class index of the requested label space."""
    _check_space(space)
    c = _classes()[project_id]
    return c["id"] if space == "proj" else c["coco_index"]


def coco_json_map(space: str = "proj") -> dict[int, int]:
    """COCO annotation-file category_id -> class index in the requested space (only the project classes)."""
    _check_space(space)
    return {c["coco_json_id"]: (c["id"] if space == "proj" else c["coco_index"]) for c in _classes()}


def coco80_to_project() -> dict[int, int]:
    """Class index of a COCO-pretrained model's prediction -> project id (to read zero-shot predictions)."""
    return {c["coco_index"]: c["id"] for c in _classes()}


def map_name(dataset: str, raw_name: str, space: str = "proj") -> Optional[int]:
    """Map a dataset's own class name to a class index in `space`, or None if the class is not used.

    Matching is exact and case-sensitive on purpose: a class is only used if the config maps it explicitly (BDD100K "rider" is
    mapped to "person" in configs/datasets.yaml by an explicit, documented decision, never by guessing).
    """
    label_map = _load("datasets.yaml")["datasets"][dataset]["label_map"]
    if not isinstance(label_map, dict):
        raise ValueError(f"dataset {dataset!r} has no name map (label_map={label_map!r})")
    target = label_map.get(raw_name)
    if target is None:
        return None
    return to_space(class_names().index(target), space)


def coco_bbox_to_yolo(bbox: Iterable[float], img_w: int, img_h: int) -> Optional[tuple[float, float, float, float]]:
    """COCO [x, y, w, h] in pixels -> YOLO (cx, cy, w, h) normalised to [0, 1], clipped to the image.

    Returns None for boxes that are empty after clipping.
    """
    x, y, w, h = [float(v) for v in bbox]
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(img_w), x + w), min(float(img_h), y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return ((x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h, (x1 - x0) / img_w, (y1 - y0) / img_h)


def coco_annotations_to_yolo(
    annotations: Iterable[dict], img_w: int, img_h: int, space: str = "proj"
) -> list[tuple[int, float, float, float, float]]:
    """Convert the COCO-JSON instance annotations of one image to YOLO rows (class, cx, cy, w, h).

    Keeps only the project classes and skips crowd regions (iscrowd=1), which are not single objects.
    `category_id` is the ORIGINAL COCO json id (person=1, cat=17, ...), not the 0..79 index.
    """
    mapping = coco_json_map(space)
    rows = []
    for a in annotations:
        if a.get("iscrowd", 0):
            continue
        cls = mapping.get(a["category_id"])
        if cls is None:
            continue
        box = coco_bbox_to_yolo(a["bbox"], img_w, img_h)
        if box is not None:
            rows.append((cls, *box))
    return rows


def yolo_line(row: tuple[int, float, float, float, float]) -> str:
    """One YOLO label-file line."""
    cls, cx, cy, w, h = row
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
