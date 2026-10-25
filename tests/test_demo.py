import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

import server  # noqa: E402


def test_keeps_only_project_classes_above_threshold_and_maps_names_and_roles():
    # coco80 ids: 0 person, 2 car, 15 cat, 24 backpack (not a project class)
    boxes = server.select_project_boxes([0, 2, 15, 24, 0], [0.9, 0.8, 0.7, 0.99, 0.2],
                                        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 1, 1], [1, 1, 2, 2]], min_conf=0.35)
    assert [b["label"] for b in boxes] == ["person", "car", "cat"]
    assert [b["role"] for b in boxes] == ["primary", "context", "secondary"]
    assert boxes[0]["x2"] == 3.0 and boxes[0]["conf"] == 0.9


def test_empty_input_gives_empty_list():
    assert server.select_project_boxes([], [], [], 0.3) == []


def test_enhancer_names_are_exactly_the_offered_options():
    assert set(server.ENHANCERS) == {"none", "clahe", "gamma", "dehaze"}
    assert server.ENHANCERS["none"] is None


def test_server_only_listens_on_localhost_and_never_writes_frames():
    src = (Path(server.__file__)).read_text(encoding="utf-8")
    assert '("127.0.0.1"' in src
    assert "imwrite" not in src and "write_bytes" not in src
