import contextlib
import io

import numpy as np
import pytest

from rcv import ci


def _random_problem(seed=0, n_img=40, n_cls=1):
    rng = np.random.default_rng(seed)
    images = [{"id": i + 1, "file_name": f"{i + 1}.jpg", "width": 640, "height": 480} for i in range(n_img)]
    anns, dets = [], []
    for im in images:
        for _ in range(int(rng.integers(0, 6))):
            c = int(rng.integers(0, n_cls))
            x, y = rng.uniform(0, 500), rng.uniform(0, 350)
            w, h = rng.uniform(20, 140), rng.uniform(20, 130)
            anns.append({"id": len(anns) + 1, "image_id": im["id"], "category_id": c, "bbox": [x, y, w, h],
                         "area": w * h, "iscrowd": 0})
            if rng.random() < 0.8:                        # a noisy true detection
                j = rng.normal(0, 6, 4)
                dets.append({"image_id": im["id"], "category_id": c, "score": float(rng.uniform(0.2, 1.0)),
                             "bbox": [x + j[0], y + j[1], max(5, w + j[2]), max(5, h + j[3])]})
        for _ in range(int(rng.integers(0, 4))):          # false positives
            dets.append({"image_id": im["id"], "category_id": int(rng.integers(0, n_cls)), "score": float(rng.uniform(0.01, 0.9)),
                         "bbox": [rng.uniform(0, 500), rng.uniform(0, 350), rng.uniform(20, 100), rng.uniform(20, 100)]})
    return {"images": images, "annotations": anns, "categories": [{"id": c, "name": str(c)} for c in range(n_cls)]}, dets


def test_ap_reproduces_pycocotools_exactly():
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    for seed in range(4):
        gt_dict, dets = _random_problem(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = gt_dict
            gt.createIndex()
            e = COCOeval(gt, gt.loadRes(dets), "bbox")
            e.params.catIds = [0]
            e.evaluate(); e.accumulate(); e.summarize()
        entries = ci.per_image_entries(gt_dict, dets, 0)
        ap, ap50 = ci.ap_from_entries(entries)
        assert ap == pytest.approx(e.stats[0], abs=1e-9)
        assert ap50 == pytest.approx(e.stats[1], abs=1e-9)


def test_ap_is_nan_without_ground_truth():
    pytest.importorskip("pycocotools")
    gt_dict, dets = _random_problem(1, n_cls=2)
    gt_dict["annotations"] = [a for a in gt_dict["annotations"] if a["category_id"] != 1]
    entries = ci.per_image_entries(gt_dict, dets, 1)
    assert np.isnan(ci.ap_from_entries(entries)[0])


def test_bootstrap_is_deterministic_and_brackets_the_point_estimate():
    pytest.importorskip("pycocotools")
    gt_dict, dets = _random_problem(2, n_img=60)
    ents = {"a": ci.per_image_entries(gt_dict, dets, 0)}
    r1 = ci.bootstrap_ap(ents, n_boot=100, seed=5)["a"]
    r2 = ci.bootstrap_ap(ents, n_boot=100, seed=5)["a"]
    assert r1 == r2
    assert r1["lo"] <= r1["ap"] <= r1["hi"] and r1["hi"] - r1["lo"] > 0


def test_fewer_images_give_a_wider_interval():
    pytest.importorskip("pycocotools")
    big_gt, big_dets = _random_problem(3, n_img=200)
    small_gt = {"images": big_gt["images"][:20], "annotations": [a for a in big_gt["annotations"] if a["image_id"] <= 20],
                "categories": big_gt["categories"]}
    small_dets = [d for d in big_dets if d["image_id"] <= 20]
    wide = ci.bootstrap_ap({"a": ci.per_image_entries(small_gt, small_dets, 0)}, n_boot=150, seed=1)["a"]
    narrow = ci.bootstrap_ap({"a": ci.per_image_entries(big_gt, big_dets, 0)}, n_boot=150, seed=1)["a"]
    assert (wide["hi"] - wide["lo"]) > (narrow["hi"] - narrow["lo"])


def test_role_bootstrap_ignores_classes_without_ground_truth():
    pytest.importorskip("pycocotools")
    gt_dict, dets = _random_problem(4, n_img=50, n_cls=2)
    gt_dict["annotations"] = [a for a in gt_dict["annotations"] if a["category_id"] != 1]
    ents = {"present": ci.per_image_entries(gt_dict, dets, 0), "absent": ci.per_image_entries(gt_dict, dets, 1)}
    r = ci.bootstrap_role(ents, ["present", "absent"], n_boot=50, seed=0)
    assert r["classes"] == ["present"] and r["lo"] <= r["ap"] <= r["hi"]
    assert ci.bootstrap_role(ents, ["absent"], n_boot=10)["ap"] is None


def test_found_flags_use_iou_and_confidence():
    gt = {"images": [{"id": 1}, {"id": 2}], "annotations": [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 100, 100]},
        {"image_id": 1, "category_id": 0, "bbox": [300, 300, 50, 50]},
        {"image_id": 2, "category_id": 0, "bbox": [0, 0, 100, 100]}]}
    dets = [{"image_id": 1, "category_id": 0, "score": 0.9, "bbox": [0, 0, 100, 50]},       # IoU 0.5 with the first box
            {"image_id": 2, "category_id": 0, "score": 0.1, "bbox": [0, 0, 100, 100]},      # perfect but below the confidence bar
            {"image_id": 2, "category_id": 5, "score": 0.9, "bbox": [0, 0, 100, 100]}]      # wrong class
    f = ci.found_flags(gt, dets, 0, conf=0.25, iou_thr=0.3)
    assert [x.tolist() for x in f] == [[True, False], [False]]
    f_strict = ci.found_flags(gt, dets, 0, conf=0.25, iou_thr=0.6)
    assert [x.tolist() for x in f_strict] == [[False, False], [False]]
    f_low_conf = ci.found_flags(gt, dets, 0, conf=0.05, iou_thr=0.3)
    assert [x.tolist() for x in f_low_conf] == [[True, False], [True]]


def test_found_rate_ci_values_and_empty_case():
    flags = [np.array([True, True]), np.array([False]), np.array([True]), np.zeros(0, bool)]
    r = ci.found_rate_ci(flags, n_boot=500, seed=0)
    assert r["rate"] == pytest.approx(0.75) and r["n_gt"] == 4 and r["lo"] <= 0.75 <= r["hi"]
    assert ci.found_rate_ci([np.zeros(0, bool)])["rate"] is None


def test_gt_from_yolo_and_detection_mapping(tmp_path):
    import cv2
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a1.jpg"), np.zeros((100, 200, 3), np.uint8))
    (tmp_path / "labels" / "a1.txt").write_text("2 0.5 0.5 0.5 0.5\n")
    gt, stem_to_id = ci.gt_from_yolo(tmp_path, {0: "person", 2: "car"})
    assert stem_to_id == {"a1": 1}
    a = gt["annotations"][0]
    assert a["category_id"] == 2 and a["bbox"] == pytest.approx([50, 25, 100, 50]) and a["area"] == pytest.approx(5000)
    preds = [{"image_id": "a1", "category_id": 3, "bbox": [1, 2, 3, 4], "score": 0.5},      # class index 2 (+1 offset)
             {"image_id": "a1", "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.5},      # class index 0, not kept
             {"image_id": "zz", "category_id": 3, "bbox": [1, 2, 3, 4], "score": 0.5}]      # unknown image
    out = ci.detections_from_ultralytics(preds, stem_to_id, keep_classes={2})
    assert out == [{"image_id": 1, "category_id": 2, "bbox": [1, 2, 3, 4], "score": 0.5}]


def test_report_agrees_with_the_separate_bootstrap_functions():
    pytest.importorskip("pycocotools")
    gt_dict, dets = _random_problem(6, n_img=50, n_cls=2)
    ents = {"person": ci.per_image_entries(gt_dict, dets, 0), "dog": ci.per_image_entries(gt_dict, dets, 1)}
    roles = {"person": "primary", "dog": "secondary"}
    rep = ci.bootstrap_report(ents, roles, n_boot=60, seed=3)
    single = ci.bootstrap_ap(ents, n_boot=60, seed=3)
    for c in ents:
        assert rep["per_class"][c]["ap"] == pytest.approx(single[c]["ap"])
        assert rep["per_class"][c]["lo"] == pytest.approx(single[c]["lo"])
        assert rep["per_class"][c]["hi"] == pytest.approx(single[c]["hi"])
    role = ci.bootstrap_role(ents, ["dog"], n_boot=60, seed=3)
    assert rep["roles"]["secondary"]["lo"] == pytest.approx(role["lo"]) and rep["roles"]["secondary"]["hi"] == pytest.approx(role["hi"])
    assert rep["roles"]["context"]["ap"] is None


def test_numeric_zero_padded_file_names_match_ultralytics_integer_image_ids(tmp_path):
    """Regression: COCO file '000000123456.jpg' is written by Ultralytics as image_id 123456; every detection used to be dropped."""
    import cv2
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "000000000123.jpg"), np.zeros((100, 200, 3), np.uint8))
    (tmp_path / "labels" / "000000000123.txt").write_text("0 0.5 0.5 0.5 0.5" + chr(10))
    gt, stem_to_id = ci.gt_from_yolo(tmp_path, {0: "person"})
    preds = [{"image_id": 123, "category_id": 1, "bbox": [50, 25, 100, 50], "score": 0.9}]
    dets = ci.detections_from_ultralytics(preds, stem_to_id, keep_classes={0})
    assert len(dets) == 1 and dets[0]["image_id"] == 1
    assert ci.norm_stem("000000000123") == "123" and ci.norm_stem("Cat_img") == "Cat_img" and ci.norm_stem("0") == "0"
