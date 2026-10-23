import pytest

from rcv import labels


def test_class_order_and_count():
    assert labels.class_names() == ["person", "cat", "dog", "horse", "sheep", "cow", "car", "bus", "truck"]


def test_person_is_the_only_primary_class_and_roles_are_complete():
    roles = labels.class_roles()
    assert [n for n, r in roles.items() if r == "primary"] == ["person"]
    assert {n for n, r in roles.items() if r == "secondary"} == {"cat", "dog", "horse", "sheep", "cow"}
    assert {n for n, r in roles.items() if r == "context"} == {"car", "bus", "truck"}
    assert set(roles.values()) == {"primary", "secondary", "context"}


def test_coco_json_ids_are_the_original_coco_numbering():
    # COCO annotation files use person=1, car=3, bus=6, truck=8, cat=17, dog=18, horse=19, sheep=20, cow=21
    assert labels.coco_json_map("proj") == {1: 0, 17: 1, 18: 2, 19: 3, 20: 4, 21: 5, 3: 6, 6: 7, 8: 8}


def test_coco80_space_uses_model_indices():
    # A COCO-pretrained YOLO outputs person=0, car=2, bus=5, truck=7, cat=15, dog=16, horse=17, sheep=18, cow=19
    assert labels.coco_json_map("coco80") == {1: 0, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 3: 2, 6: 5, 8: 7}
    assert labels.coco80_to_project() == {0: 0, 15: 1, 16: 2, 17: 3, 18: 4, 19: 5, 2: 6, 5: 7, 7: 8}


def test_no_two_classes_share_an_index_in_any_numbering():
    for space in labels.SPACES:
        values = list(labels.coco_json_map(space).values())
        assert len(values) == len(set(values)) == len(labels.class_names())


def test_to_space_roundtrip():
    for pid in range(len(labels.class_names())):
        assert labels.coco80_to_project()[labels.to_space(pid, "coco80")] == pid


def test_bad_space_rejected():
    with pytest.raises(ValueError):
        labels.coco_json_map("coco91")
    with pytest.raises(ValueError):
        labels.coco_json_map("proj6")  # the old name must not silently work


def test_bbox_conversion_center_and_size():
    cx, cy, w, h = labels.coco_bbox_to_yolo([10, 20, 30, 40], 100, 200)
    assert (cx, cy, w, h) == pytest.approx((0.25, 0.2, 0.3, 0.2))


def test_bbox_is_clipped_to_image_and_empty_is_dropped():
    box = labels.coco_bbox_to_yolo([-10, -10, 30, 30], 100, 100)
    assert box == pytest.approx((0.1, 0.1, 0.2, 0.2))  # clipped to [0,20] x [0,20]
    assert labels.coco_bbox_to_yolo([200, 200, 10, 10], 100, 100) is None
    assert labels.coco_bbox_to_yolo([10, 10, 0, 5], 100, 100) is None


def test_annotations_filter_classes_and_crowds():
    anns = [
        {"category_id": 1, "bbox": [0, 0, 50, 50], "iscrowd": 0},    # person -> kept
        {"category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 1},    # crowd -> dropped
        {"category_id": 3, "bbox": [0, 0, 10, 10], "iscrowd": 0},    # car (json id 3) -> kept as context
        {"category_id": 2, "bbox": [0, 0, 10, 10], "iscrowd": 0},    # bicycle (json id 2) -> not a project class
        {"category_id": 18, "bbox": [50, 50, 50, 50], "iscrowd": 0}, # dog -> kept
    ]
    rows = labels.coco_annotations_to_yolo(anns, 100, 100, "proj")
    assert [r[0] for r in rows] == [0, 6, 2]
    rows80 = labels.coco_annotations_to_yolo(anns, 100, 100, "coco80")
    assert [r[0] for r in rows80] == [0, 2, 16]


def test_dataset_name_maps_are_exact():
    assert labels.map_name("exdark", "People") == 0
    assert labels.map_name("exdark", "Dog") == 2
    assert labels.map_name("exdark", "Car") == 6
    assert labels.map_name("exdark", "Bicycle") is None         # not a project class
    assert labels.map_name("exdark", "people") is None          # case-sensitive on purpose
    assert labels.map_name("bdd100k_val", "rider") == 0         # explicit decision: a rider is a human (see configs/datasets.yaml)
    assert labels.map_name("bdd100k_val", "bike") is None       # bicycles and motorcycles are not project classes
    assert labels.map_name("bdd100k_val", "Rider") is None      # still exact and case-sensitive
    assert labels.map_name("bdd100k_train", "rider") == 0
    assert labels.map_name("bdd100k_val", "person") == 0
    assert labels.map_name("bdd100k_val", "truck") == 8
    assert labels.map_name("exdark", "Cat", space="coco80") == 15
    assert labels.map_name("dawn", "car", space="coco80") == 2


def test_every_dataset_map_targets_a_known_class_and_matches_its_available_list():
    cfg = labels._load("datasets.yaml")["datasets"]
    names = set(labels.class_names())
    for ds, d in cfg.items():
        if isinstance(d["label_map"], dict):
            targets = set(d["label_map"].values())
            assert targets <= names, f"{ds} maps to unknown classes {targets - names}"
            assert targets == set(d["classes_available"]), f"{ds}: classes_available disagrees with label_map"


def test_yolo_line_format():
    assert labels.yolo_line((2, 0.5, 0.5, 0.25, 0.125)) == "2 0.500000 0.500000 0.250000 0.125000"
