"""Build the REAL-condition test sets from ExDark (low light) and DAWN (fog, rain, snow).

Both are TEST-ONLY sets in this project (never trained on; see data/README.md, release policy).

Outputs (never committed):  data/processed/real/<name>/<space>/{images,labels,<space>.yaml}
    exdark_all   every ExDark image with at least one project-class object (People, Cat, Dog, Car, Bus)
    dawn_fog / dawn_rain / dawn_snow   DAWN images with at least one project-class object (person, car, bus, truck)
Sandstorm images are not used. A summary with object counts per class is written next to the sets.

Caveat recorded in the output: both datasets annotate only THEIR OWN class list. ExDark labels 12 classes, DAWN labels
vehicles and a few people, so objects of a project class that the authors did not annotate count as false positives.

Run from the project root:   .venv\\Scripts\\python scripts\\prepare_real_sets.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcv import data, labels  # noqa: E402

OUT = ROOT / "data" / "processed" / "real"
NAMES = labels.class_names()


def count_classes(selected) -> dict:
    c = collections.Counter(NAMES[r[0]] for _, rows in selected for r in rows)   # rows are in 'proj' space here
    return {n: c.get(n, 0) for n in NAMES}


def dawn_sets() -> dict[str, list]:
    raw = ROOT / "data" / "dawn" / "raw"
    sets = {}
    for cond in ("Fog", "Rain", "Snow"):
        selected = []
        for jpg in sorted((raw / cond).glob("*.jpg")):
            xml = raw / cond / f"{cond}_PASCAL_VOC" / f"{jpg.stem}.xml"
            if not xml.exists():
                continue
            ann = data.parse_voc_xml(xml)
            rows = data.named_boxes_to_rows("dawn", ann["objects"], ann["width"], ann["height"], "proj")
            if rows:
                info = {"file_name": f"{cond.lower()}_{jpg.name}", "src": str(jpg), "width": ann["width"], "height": ann["height"]}
                selected.append((info, rows))
        sets[f"dawn_{cond.lower()}"] = selected
    return sets


def exdark_set() -> tuple[list, dict]:
    imgs = ROOT / "data" / "exdark" / "raw" / "ExDark"
    anns = ROOT / "data" / "exdark" / "raw" / "ExDark_Annno"
    selected, rotated, missing = [], 0, 0
    for cls_dir in sorted(p for p in imgs.iterdir() if p.is_dir()):
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            ann = anns / cls_dir.name / (img_path.name + ".txt")
            if not ann.exists():
                missing += 1
                continue
            with Image.open(img_path) as im:
                w, h = im.size
                if im.getexif().get(274, 1) in (5, 6, 7, 8):     # EXIF orientation that swaps width and height
                    rotated += 1
            rows = data.named_boxes_to_rows("exdark", data.parse_exdark_annotation(ann), w, h, "proj")
            if rows:
                selected.append(({"file_name": f"{cls_dir.name}_{img_path.name}", "src": str(img_path), "width": w, "height": h}, rows))
    return selected, {"images_with_swapping_exif_orientation": rotated, "images_without_annotation_file": missing}


def main() -> None:
    summary = {}
    all_sets = dawn_sets()
    ex, ex_notes = exdark_set()
    all_sets["exdark_all"] = ex
    for name, selected in all_sets.items():
        for space in ("proj", "coco80"):
            rows_space = [(i, [(labels.to_space(r[0], space), *r[1:]) for r in rows]) for i, rows in selected]
            data.write_set(rows_space, ROOT, OUT / name / space, space, name)
        summary[name] = {"images": len(selected), "objects_per_class": count_classes(selected)}
        print(name, summary[name], flush=True)
    summary["exdark_notes"] = ex_notes
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("exdark notes:", ex_notes)


if __name__ == "__main__":
    main()
