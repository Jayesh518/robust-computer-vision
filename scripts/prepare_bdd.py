"""Build the BDD100K validation sets by weather and time of day (real photographs, test-only).

Reads data/bdd100k/bdd100k_labels.zip (one JSON per image, with weather / timeofday attributes) and
data/bdd100k/bdd100k_images_100k.zip, extracts only the val images that are needed and writes, never committed:
    data/processed/real/<name>/<space>/{images,labels,<space>.yaml}

Sets (val split; clear daytime is the within-dataset reference for every other set):
    bdd_clear_day    weather clear,  timeofday daytime       (reference)
    bdd_night_clear  weather clear,  timeofday night         (isolates the lighting effect)
    bdd_rainy_day / bdd_snowy_day    daytime rain / snow     (isolates the weather effect)
    bdd_rainy_night / bdd_snowy_night                         (weather and darkness together)
    bdd_foggy_all    weather foggy (only 13 val images: far too few for conclusions)
Classes: person (rider mapped to person, see configs/datasets.yaml), car, bus, truck. Other BDD classes are ignored, so
an object of a project class that BDD does not annotate counts as a false positive.

Run from the project root:   .venv\\Scripts\\python scripts\\prepare_bdd.py
"""
from __future__ import annotations

import collections
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcv import data, labels  # noqa: E402

BDD = ROOT / "data" / "bdd100k"
OUT = ROOT / "data" / "processed" / "real"
W, H = 1280, 720
NAMES = labels.class_names()

SETS = {
    "bdd_clear_day": lambda a: a.get("weather") == "clear" and a.get("timeofday") == "daytime",
    "bdd_night_clear": lambda a: a.get("weather") == "clear" and a.get("timeofday") == "night",
    "bdd_rainy_day": lambda a: a.get("weather") == "rainy" and a.get("timeofday") == "daytime",
    "bdd_snowy_day": lambda a: a.get("weather") == "snowy" and a.get("timeofday") == "daytime",
    "bdd_rainy_night": lambda a: a.get("weather") == "rainy" and a.get("timeofday") == "night",
    "bdd_snowy_night": lambda a: a.get("weather") == "snowy" and a.get("timeofday") == "night",
    "bdd_foggy_all": lambda a: a.get("weather") == "foggy",
}


def image_rows(label_json: dict, dataset: str = "bdd100k_val") -> list[tuple]:
    boxes = []
    for fr in label_json.get("frames", []):
        for o in fr.get("objects", []):
            b = o.get("box2d")
            if b:
                boxes.append((o["category"], b["x1"], b["y1"], b["x2"], b["y2"]))
    return data.named_boxes_to_rows(dataset, boxes, W, H, "proj")


def main() -> None:
    zl = zipfile.ZipFile(BDD / "bdd100k_labels.zip")
    zi = zipfile.ZipFile(BDD / "bdd100k_images_100k.zip")
    members = set(zi.namelist())
    chosen: dict[str, list] = {k: [] for k in SETS}
    for n in sorted(x for x in zl.namelist() if x.startswith("100k/val/") and x.endswith(".json")):
        d = json.loads(zl.read(n))
        rows = image_rows(d)
        if not rows:
            continue
        img_member = f"100k/val/{d['name']}.jpg"
        if img_member not in members:
            continue
        for set_name, pred in SETS.items():
            if pred(d.get("attributes", {})):
                chosen[set_name].append((d["name"], img_member, rows))

    need = sorted({m for v in chosen.values() for _, m, _ in v})
    raw = BDD / "raw"
    for m in need:
        if not (raw / m).exists():
            zi.extract(m, raw)
    with Image.open(raw / need[0]) as im:
        assert im.size == (W, H), f"unexpected BDD image size {im.size}"

    summary = {}
    for set_name, items in chosen.items():
        selected = [({"file_name": f"{name}.jpg", "src": str(raw / member), "width": W, "height": H}, rows) for name, member, rows in items]
        for space in ("proj", "coco80"):
            rs = [(i, [(labels.to_space(r[0], space), *r[1:]) for r in rows]) for i, rows in selected]
            data.write_set(rs, ROOT, OUT / set_name / space, space, set_name)
        c = collections.Counter(NAMES[r[0]] for _, rows in selected for r in rows)
        summary[set_name] = {"images": len(selected), "objects_per_class": {n: c.get(n, 0) for n in NAMES if c.get(n, 0)}}
        print(set_name, summary[set_name], flush=True)
    (OUT / "summary_bdd.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("extracted images:", len(need))


if __name__ == "__main__":
    main()
