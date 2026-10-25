"""Live webcam demo: a small local server. The browser opens the webcam and posts JPEG frames to /detect; the server runs the detector and
returns the boxes; the browser draws them. Nothing is saved, and the server listens on this computer only (127.0.0.1).

    .venv\\Scripts\\python demo\\server.py                      # CPU (safe while a training job uses the GPU)
    .venv\\Scripts\\python demo\\server.py --device 0           # GPU, when nothing else is training
then open http://127.0.0.1:8765

Models offered: every .pt in models/ (pretrained) and models/ft/ (the fine-tuned checkpoints from the experiments). Only the nine project
classes are returned. Optional preprocessing (CLAHE, adaptive gamma, dark-channel dehazing) comes from src/rcv/enhance.py.
This is a research demo, not a safety system. It says nothing about whether a person is alive, conscious or unhurt.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcv import enhance, labels  # noqa: E402

MAX_BODY = 4 * 1024 * 1024          # a webcam JPEG is a few tens of KB; refuse anything much bigger
ENHANCERS = {"none": None, "clahe": enhance.clahe, "gamma": enhance.gamma, "dehaze": enhance.dcp}


def select_project_boxes(coco80_ids, confs, xyxy, min_conf: float) -> list[dict]:
    """Keep detections of the nine project classes above min_conf; return plain dicts (pixels of the frame that was sent)."""
    to_proj = labels.coco80_to_project()
    names, roles = labels.class_names(), labels.class_roles()
    out = []
    for c, p, b in zip(coco80_ids, confs, xyxy):
        c = int(c)
        if c not in to_proj or float(p) < min_conf:
            continue
        name = names[to_proj[c]]
        x1, y1, x2, y2 = (float(v) for v in b)
        out.append({"label": name, "role": roles[name], "conf": round(float(p), 3),
                    "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)})
    return out


def list_models() -> dict[str, Path]:
    found = {}
    for folder in (ROOT / "models", ROOT / "models" / "ft"):
        for p in sorted(folder.glob("*.pt")) if folder.exists() else []:
            found[p.stem if folder.name == "models" else f"ft/{p.stem}"] = p
    return found


class Detector:
    """Holds one loaded model at a time (a second model would double the memory)."""

    def __init__(self, device: str, imgsz: int):
        self.device, self.imgsz = device, imgsz
        self.lock = threading.Lock()
        self.name, self.model = None, None
        self.models = list_models()

    def load(self, name: str):
        if name not in self.models:
            raise KeyError(name)
        if name != self.name:
            from ultralytics import YOLO

            self.model, self.name = YOLO(str(self.models[name])), name

    def detect(self, frame: np.ndarray, model: str, conf: float, enh: str) -> dict:
        fn = ENHANCERS.get(enh)
        with self.lock:
            self.load(model)
            t0 = time.perf_counter()
            img = fn(frame) if fn else frame
            t1 = time.perf_counter()
            r = self.model.predict(img, imgsz=self.imgsz, conf=min(conf, 0.5), device=self.device, verbose=False, half=False)[0]
            t2 = time.perf_counter()
        b = r.boxes
        boxes = select_project_boxes(b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xyxy.cpu().numpy(), conf)
        return {"boxes": boxes, "width": frame.shape[1], "height": frame.shape[0],
                "enhance_ms": round((t1 - t0) * 1000, 1), "detect_ms": round((t2 - t1) * 1000, 1), "model": self.name}


def make_handler(det: Detector, html: bytes):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, html, "text/html; charset=utf-8")
            elif path == "/models":
                self._send(200, json.dumps({"models": list(det.models), "enhancements": list(ENHANCERS), "device": det.device}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/detect":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > MAX_BODY:
                return self._send(413, b"bad size", "text/plain")
            q = parse_qs(u.query)
            model = q.get("model", ["yolo11n"])[0]
            enh = q.get("enh", ["none"])[0]
            try:
                conf = min(max(float(q.get("conf", ["0.35"])[0]), 0.05), 0.95)
            except ValueError:
                return self._send(400, b"bad conf", "text/plain")
            frame = cv2.imdecode(np.frombuffer(self.rfile.read(n), np.uint8), cv2.IMREAD_COLOR)
            if frame is None or model not in det.models or enh not in ENHANCERS:
                return self._send(400, b"bad request", "text/plain")
            self._send(200, json.dumps(det.detect(frame, model, conf, enh)).encode(), "application/json")

        def log_message(self, *a):          # no per-request logging: keeps the console readable and records nothing about the camera
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cpu", help="cpu, or a GPU index such as 0")
    ap.add_argument("--imgsz", type=int, default=480, help="inference size; 640 is what the experiments used, 480 is faster on a CPU")
    args = ap.parse_args()
    det = Detector(args.device, args.imgsz)
    if not det.models:
        sys.exit("no .pt files found in models/")
    html = (Path(__file__).parent / "index.html").read_bytes()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(det, html))      # this computer only
    print(f"demo on http://127.0.0.1:{args.port}  (device {args.device}, imgsz {args.imgsz}, {len(det.models)} models). Ctrl+C to stop.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
