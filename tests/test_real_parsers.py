import pytest

from rcv import data

VOC = """<annotation><filename>a.jpg</filename><size><width>200</width><height>100</height><depth>3</depth></size>
<object><name>car</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>70</ymax></bndbox></object>
<object><name>person</name><bndbox><xmin>150</xmin><ymin>10</ymin><xmax>190</xmax><ymax>90</ymax></bndbox></object>
<object><name>train</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
</annotation>"""

EXDARK = """% bbGt version=3
People 73 273 34 78 0 0 0 0 0 0 0
Cat 10 20 100 50 0 0 0 0 0 0 0
Bicycle 5 5 20 20 0 0 0 0 0 0 0
"""


def test_parse_voc_xml(tmp_path):
    f = tmp_path / "a.xml"
    f.write_text(VOC)
    d = data.parse_voc_xml(f)
    assert (d["width"], d["height"]) == (200, 100)
    assert d["objects"][0] == ("car", 10.0, 20.0, 110.0, 70.0)
    assert [o[0] for o in d["objects"]] == ["car", "person", "train"]


def test_dawn_rows_keep_mapped_classes_only_and_normalise(tmp_path):
    f = tmp_path / "a.xml"
    f.write_text(VOC)
    d = data.parse_voc_xml(f)
    rows = data.named_boxes_to_rows("dawn", d["objects"], d["width"], d["height"], "proj")
    assert [r[0] for r in rows] == [6, 0]                         # car=6, person=0; 'train' is not a project class
    cx, cy, w, h = rows[0][1:]
    assert (cx, cy, w, h) == pytest.approx((0.3, 0.45, 0.5, 0.5))   # (10..110 of 200, 20..70 of 100)
    rows80 = data.named_boxes_to_rows("dawn", d["objects"], d["width"], d["height"], "coco80")
    assert [r[0] for r in rows80] == [2, 0]                       # car=2, person=0 in COCO-80 indices


def test_parse_exdark_skips_header_and_converts_ltwh_to_corners(tmp_path):
    f = tmp_path / "x.jpg.txt"
    f.write_text(EXDARK)
    boxes = data.parse_exdark_annotation(f)
    assert boxes[0] == ("People", 73.0, 273.0, 107.0, 351.0)
    assert [b[0] for b in boxes] == ["People", "Cat", "Bicycle"]


def test_exdark_rows_use_exact_class_names(tmp_path):
    f = tmp_path / "x.jpg.txt"
    f.write_text(EXDARK + "people 1 1 5 5 0 0 0 0 0 0 0\n")     # wrong case must be dropped, not guessed
    rows = data.named_boxes_to_rows("exdark", data.parse_exdark_annotation(f), 400, 400, "proj")
    assert [r[0] for r in rows] == [0, 1]                         # People -> person(0), Cat -> cat(1); Bicycle, 'people' dropped


def test_empty_after_clipping_is_dropped():
    rows = data.named_boxes_to_rows("dawn", [("car", 300, 300, 400, 400)], 200, 100, "proj")
    assert rows == []


def test_write_set_accepts_source_paths_in_subfolders(tmp_path):
    import cv2
    import numpy as np
    src = tmp_path / "raw" / "Cat"
    src.mkdir(parents=True)
    cv2.imwrite(str(src / "img.jpg"), np.zeros((50, 50, 3), np.uint8))
    img = {"file_name": "Cat_img.jpg", "src": str(src / "img.jpg"), "width": 50, "height": 50}
    out = tmp_path / "out"
    data.write_set([(img, [(1, 0.5, 0.5, 0.2, 0.2)])], tmp_path, out, "proj", "t")
    assert (out / "images" / "Cat_img.jpg").exists() and (out / "labels" / "Cat_img.txt").exists()
