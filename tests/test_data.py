import cv2
import numpy as np
import pytest
import yaml

from rcv import data, labels


def test_coco80_list_is_80_names_and_matches_the_configured_indices():
    assert len(data.COCO80) == 80 and len(set(data.COCO80)) == 80
    # every project class must sit at its configured COCO index, otherwise the zero-shot baseline is scored wrongly
    for c in labels._load("classes.yaml")["classes"]:
        assert data.COCO80[c["coco_index"]] == c["name"], c


def test_coco80_list_matches_the_names_inside_the_real_pretrained_model():
    """Guards against a typo in the hand-written list (one name was once missing). Skipped when the weights or torch are absent."""
    from pathlib import Path
    weights = Path(__file__).resolve().parents[1] / "models" / "yolo11n.pt"
    if not weights.exists():
        pytest.skip("models/yolo11n.pt not downloaded")
    ultralytics = pytest.importorskip("ultralytics")
    names = ultralytics.YOLO(str(weights)).names
    assert [names[i] for i in range(80)] == data.COCO80


def test_space_names():
    assert data.space_names("proj")[0] == "person" and len(data.space_names("proj")) == 9
    assert data.space_names("coco80")[16] == "dog" and len(data.space_names("coco80")) == 80
    with pytest.raises(ValueError):
        data.space_names("coco91")


@pytest.fixture()
def tiny_coco(tmp_path):
    src = tmp_path / "src_images"
    src.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        cv2.imwrite(str(src / name), np.full((100, 200, 3), 128, np.uint8))
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100},
                   {"id": 2, "file_name": "b.jpg", "width": 200, "height": 100},
                   {"id": 3, "file_name": "c.jpg", "width": 200, "height": 100}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [20, 10, 40, 80], "iscrowd": 0},    # person
            {"image_id": 1, "category_id": 3, "bbox": [100, 50, 60, 30], "iscrowd": 0},   # car
            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "iscrowd": 0},      # bicycle: not a project class
            {"image_id": 2, "category_id": 2, "bbox": [0, 0, 10, 10], "iscrowd": 0},      # only a bicycle -> image skipped
            {"image_id": 3, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 1},      # only a crowd -> image skipped
        ],
    }
    return coco, src


def test_select_images_keeps_only_images_with_target_objects(tiny_coco):
    coco, _ = tiny_coco
    sel = data.select_images(coco, "proj")
    assert [img["id"] for img, _ in sel] == [1]
    assert sorted(r[0] for r in sel[0][1]) == [0, 6]  # person, car
    sel80 = data.select_images(coco, "coco80")
    assert sorted(r[0] for r in sel80[0][1]) == [0, 2]


def test_write_set_layout_labels_and_yaml(tiny_coco, tmp_path):
    coco, src = tiny_coco
    out = tmp_path / "out"
    ypath = data.write_set(data.select_images(coco, "proj"), src, out, "proj", "tiny")
    assert (out / "images" / "a.jpg").exists() and not (out / "images" / "b.jpg").exists()
    lines = (out / "labels" / "a.txt").read_text().strip().splitlines()
    assert len(lines) == 2
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0" and float(cx) == pytest.approx(0.2) and float(w) == pytest.approx(0.2) and float(h) == pytest.approx(0.8)
    cfg = yaml.safe_load(ypath.read_text())
    assert cfg["names"][0] == "person" and cfg["val"] == "images" and cfg["n_images"] == 1


def _fake_selected(n):
    return [({"id": i, "file_name": f"{i}.jpg", "width": 200, "height": 100}, [(0, 0.5, 0.5, 0.2, 0.8)]) for i in range(n)]


def test_sample_images_is_deterministic_sorted_and_bounded():
    sel = _fake_selected(50)
    a = data.sample_images(sel, 10, seed=1)
    b = data.sample_images(sel, 10, seed=1)
    c = data.sample_images(sel, 10, seed=2)
    assert [i["id"] for i, _ in a] == [i["id"] for i, _ in b]
    assert [i["id"] for i, _ in a] != [i["id"] for i, _ in c]
    ids = [i["id"] for i, _ in a]
    assert ids == sorted(ids) and len(set(ids)) == 10
    assert len(data.sample_images(sel, 500, seed=1)) == 50  # asking for more than exists returns everything


def test_occlusion_candidates_need_persons_and_all_of_them_large_enough():
    big = (0, 0.5, 0.5, 0.2, 0.8)      # 40 x 80 px in a 200 x 100 image
    small = (0, 0.5, 0.5, 0.05, 0.3)   # 10 x 30 px
    car = (6, 0.5, 0.5, 0.01, 0.01)
    im = {"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}
    sel = [(dict(im, id=1), [big]), (dict(im, id=2), [big, small]), (dict(im, id=3), [car]), (dict(im, id=4), [big, car])]
    assert [i["id"] for i, _ in data.occlusion_candidates(sel, min_side_px=40)] == [1, 4]


def test_yolo_row_to_xyxy_roundtrip_with_coco_conversion():
    box = labels.coco_bbox_to_yolo([20, 10, 40, 80], 200, 100)
    x0, y0, x1, y1 = data.yolo_row_to_xyxy((0, *box), 200, 100)
    assert (x0, y0, x1, y1) == pytest.approx((20, 10, 60, 90))


def test_write_set_is_idempotent(tiny_coco, tmp_path):
    coco, src = tiny_coco
    out = tmp_path / "out"
    sel = data.select_images(coco, "coco80")
    data.write_set(sel, src, out, "coco80", "tiny")
    data.write_set(sel, src, out, "coco80", "tiny")  # second run must not fail on existing links
    assert len(list((out / "images").iterdir())) == 1
