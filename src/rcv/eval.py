"""Evaluation helpers: Ultralytics validation wrapper and batch-1 latency measurement.

torch / ultralytics are imported lazily so the rest of the package (and CI) works without a GPU stack.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Optional


def environment_info() -> dict:
    """Versions and hardware, recorded next to every result so numbers can be reproduced."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = int(torch.cuda.get_device_properties(0).total_memory / 2**20)
    except ImportError:
        info["torch"] = None
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None
    return info


def measure_latency(weights: str, imgsz: int = 640, runs: int = 100, warmup: int = 20,
                    device: Optional[str | int] = 0, half: bool = True) -> dict:
    """Batch-1 inference latency in milliseconds on a fixed random image.

    Measures the model forward pass only (no image decoding, no NMS). Warm-up runs are excluded, and the GPU is
    synchronised before reading the clock, otherwise asynchronous CUDA calls make the numbers meaningless.
    """
    import numpy as np
    import torch
    from ultralytics import YOLO

    model = YOLO(weights)
    use_cuda = torch.cuda.is_available() and device not in ("cpu", None)
    dev = f"cuda:{device}" if use_cuda and isinstance(device, int) else ("cuda" if use_cuda else "cpu")
    net = model.model.to(dev).eval()
    if use_cuda and half:
        net = net.half()
    x = torch.from_numpy(np.random.default_rng(0).random((1, 3, imgsz, imgsz), dtype=np.float32)).to(dev)
    if use_cuda and half:
        x = x.half()
    times = []
    with torch.inference_mode():
        for i in range(warmup + runs):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            net(x)
            if use_cuda:
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {
        "weights": str(weights), "imgsz": imgsz, "device": dev, "half": bool(use_cuda and half), "runs": runs,
        "mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
        "fps": float(1000.0 / t.mean()),
        "params_m": float(sum(p.numel() for p in net.parameters()) / 1e6),
        "size_mb": float(Path(weights).stat().st_size / 2**20) if Path(weights).exists() else None,
    }


def evaluate(weights: str, data_yaml: str, imgsz: int = 640, batch: int = 8, device: Optional[str | int] = 0,
             split: str = "val", conf: float = 0.001, iou: float = 0.6, half: bool = True,
             classes: Optional[list[int]] = None, project: Optional[str] = None, name: Optional[str] = None,
             save_json: bool = False) -> dict:
    """Run Ultralytics validation and return a plain dict of metrics (overall and per class).

    `classes` restricts evaluation to those class indices in the dataset's label space (e.g. the 6 project classes
    when a COCO-pretrained model with 80 outputs is evaluated on a COCO-space label set).
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    res = model.val(data=data_yaml, imgsz=imgsz, batch=batch, device=device, split=split, conf=conf, iou=iou,
                    half=half, classes=classes, project=project, name=name, plots=False, verbose=False, save_json=save_json)
    box = res.box
    per_class = {}
    for i, cls_idx in enumerate(box.ap_class_index):
        per_class[str(res.names[int(cls_idx)])] = {
            "precision": float(box.p[i]), "recall": float(box.r[i]),
            "map50": float(box.ap50[i]), "map50_95": float(box.maps[int(cls_idx)]),
        }
    return {
        "weights": str(weights), "data": str(data_yaml), "split": split, "imgsz": imgsz,
        "precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50), "map50_95": float(box.map),
        "per_class": per_class, "save_dir": str(res.save_dir),
    }


def role_summary(per_class: dict[str, dict]) -> dict:
    """Group per-class metrics by role (person = primary, animals = secondary, vehicles = context).

    Each role gets the macro mean (unweighted over the classes that have ground truth in that set) of mAP50-95, mAP50,
    precision and recall, plus which classes were averaged. The primary role is the headline result.
    """
    from . import labels

    roles = labels.class_roles()
    out: dict[str, dict] = {}
    for role in ("primary", "secondary", "context"):
        members = [n for n in per_class if roles.get(n) == role]
        if not members:
            out[role] = {"classes": [], "map50_95": None, "map50": None, "precision": None, "recall": None}
            continue
        out[role] = {"classes": sorted(members)}
        for k in ("map50_95", "map50", "precision", "recall"):
            out[role][k] = float(sum(per_class[n][k] for n in members) / len(members))
    return out


def write_json(path: str | Path, payload: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
