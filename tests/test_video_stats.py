import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_video  # noqa: E402


def test_counts_flicker_and_empty_frames():
    frames = [[0.9, 0.8], [0.9], [], [0.6, 0.55, 0.2]]
    s = run_video.summarise(frames, conf=0.5)
    assert s["frames"] == 4
    assert s["max_persons_in_a_frame"] == 2
    assert s["frames_with_no_person"] == 1
    assert s["fraction_frames_with_a_person"] == 0.75
    assert s["mean_persons_per_frame"] == (2 + 1 + 0 + 2) / 4
    assert s["count_flicker"] == (1 + 1 + 2) / 3          # |2-1|, |1-0|, |0-2|


def test_threshold_changes_the_count():
    frames = [[0.9, 0.3], [0.4]]
    assert run_video.summarise(frames, 0.25)["mean_persons_per_frame"] == 1.5
    assert run_video.summarise(frames, 0.5)["mean_persons_per_frame"] == 0.5


def test_empty_video():
    s = run_video.summarise([], 0.35)
    assert s["frames"] == 0 and s["mean_persons_per_frame"] == 0.0
