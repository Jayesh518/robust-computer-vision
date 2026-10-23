"""Batch-1 forward-pass latency of each model on the GPU (results/latency_models.json).

Measures the network forward pass only (no image decoding, no NMS or post-processing), at 640 px with half precision,
after warm-up and with the GPU synchronised before every clock read. Run it while the GPU is otherwise IDLE: any other
job on the GPU inflates the numbers. A laptop GPU is noisy, so the median and the 95th percentile are both reported.

    .venv\\Scripts\\python scripts\\measure_latency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcv.eval import environment_info, measure_latency, write_json  # noqa: E402

MODELS = ["yolo11n", "yolo11s", "yolo26n"]


def main() -> None:
    out = {}
    for m in MODELS:
        r = measure_latency(str(ROOT / "models" / f"{m}.pt"), runs=200, warmup=40)
        out[m] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
        print(f"{m}: median {r['p50_ms']:.1f} ms | p95 {r['p95_ms']:.1f} ms | {r['params_m']:.1f} M params | {r['size_mb']:.1f} MB", flush=True)
    out["environment"] = environment_info()
    write_json(ROOT / "results" / "latency_models.json", out)
    print("wrote results/latency_models.json")


if __name__ == "__main__":
    main()
