"""Resumable downloads of the datasets used by this project (official sources only). Never commits anything.

    .venv\\Scripts\\python scripts\\download_data.py dawn exdark

Each download restarts itself if the connection stalls (below 50 KB/s for 30 s) and resumes from where it stopped.
Check data/README.md for the license terms BEFORE running; they differ per dataset.

  coco    val2017 images + 2017 annotations     (http://images.cocodataset.org)
  dawn    Fog / Rain / Snow zips                (Mendeley Data, CC BY-NC 3.0; the Sand set is not used)
  exdark  images (1.5 GB) + ground truth (4 MB) (Google Drive links from the official GitHub repository)
  bdd100k 100K images (5 GB) + 2018 labels (181 MB), from the legacy Berkeley mirror http://dl.yf.io/bdd100k/
          (the documented hosts bdd-data.berkeley.edu and dl.cv.ethz.ch were unreachable on 2026-10-23, see
          github.com/bdd100k/bdd100k/issues/369). The mirror is plain HTTP, so the md5 of each file is computed and
          compared with the md5 published in BDD100K's docs; the result is printed and must be recorded in data/README.md.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DAWN_BASE = "https://data.mendeley.com/public-files/datasets/766ygrbt8y/files"
DAWN = {  # file -> (Mendeley file id, size in bytes from the Mendeley API)
    "Fog.zip": ("c286e791-88c9-4bec-b43d-2ed276dd11ba", 30_571_774),
    "Rain.zip": ("7b37087a-db6b-4ce9-8ec4-ee64f963f073", 29_881_690),
    "Snow.zip": ("d5abcff0-484f-49c9-b9f7-bf36c3b325a2", 43_899_038),
}
COCO = {
    "val2017.zip": ("http://images.cocodataset.org/zips/val2017.zip", 815_585_330),
    "annotations_trainval2017.zip": ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", 252_907_541),
}
EXDARK = {  # Google Drive file ids from the official repository's Dataset/README.md and Groundtruth/README.md
    "ExDark_images.zip": "1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx",
    "ExDark_Annno.zip": "1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i",
}


def curl(url: str, dest: Path, expected: int | None, attempts: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, attempts + 1):
        subprocess.run(["curl", "-L", "-C", "-", "-sS", "--connect-timeout", "20", "--speed-limit", "50000",
                        "--speed-time", "30", "-o", str(dest), url])
        size = dest.stat().st_size if dest.exists() else 0
        if expected is None or size >= expected:
            break
        print(f"  {dest.name}: {size}/{expected} bytes, retry {i}", flush=True)
    size = dest.stat().st_size if dest.exists() else 0
    if expected is not None and size != expected:
        raise RuntimeError(f"{dest.name}: got {size} bytes, expected {expected}")
    if dest.suffix == ".zip" and not zipfile.is_zipfile(dest):
        raise RuntimeError(f"{dest.name} is not a valid zip")
    print(f"ok {dest.name} ({size / 2**20:.1f} MB)", flush=True)


BDD_BASE = "http://dl.yf.io/bdd100k"
# file -> md5 published in https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst ("100K Images", "Labels").
# The published sizes (5.3 GB / 107 MB) may describe a different packaging than the mirror's files; a mismatch is reported, not hidden.
BDD = {"bdd100k_labels.zip": "e21be3e7d6a07ee439faf61e769667e4", "bdd100k_images_100k.zip": "5a0359c86a0b8713adab1eee9a3041cb"}


def head_size(url: str) -> int:
    out = subprocess.run(["curl", "-sIL", "--max-time", "40", url], capture_output=True, text=True).stdout
    sizes = [int(l.split(":", 1)[1]) for l in out.splitlines() if l.lower().startswith("content-length:")]
    if not sizes:
        raise RuntimeError(f"could not read the size of {url}")
    return sizes[-1]


def md5_of(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_bdd100k() -> None:
    for name, documented in BDD.items():
        url = f"{BDD_BASE}/{name}"
        dest = DATA / "bdd100k" / name
        curl(url, dest, head_size(url))
        got = md5_of(dest)
        status = "MATCHES the documented md5" if got == documented else "DOES NOT match the documented md5 (packaging may differ; integrity not independently verified)"
        print(f"md5 {name}: {got} -> {status}", flush=True)
        report = DATA / "bdd100k" / f"{name}.md5-check.txt"
        report.write_text(chr(10).join([got, f"documented: {documented}", status]) + chr(10), encoding="utf-8")


def get_coco() -> None:
    for name, (url, size) in COCO.items():
        curl(url, DATA / "coco" / name, size)


def get_dawn() -> None:
    for name, (fid, size) in DAWN.items():
        curl(f"{DAWN_BASE}/{fid}/file_downloaded", DATA / "dawn" / name, size)


def get_exdark() -> None:
    import gdown  # pip install gdown

    out = DATA / "exdark"
    out.mkdir(parents=True, exist_ok=True)
    for name, fid in EXDARK.items():
        dest = out / name
        if dest.exists() and zipfile.is_zipfile(dest):
            print(f"already have {name}")
            continue
        gdown.download(id=fid, output=str(dest), quiet=False, resume=True)
        if not zipfile.is_zipfile(dest):
            raise RuntimeError(f"{name} is not a valid zip (Google Drive may have shown a quota or confirmation page)")
        print(f"ok {name} ({dest.stat().st_size / 2**20:.1f} MB)", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["coco"]
    for w in which:
        {"coco": get_coco, "dawn": get_dawn, "exdark": get_exdark, "bdd100k": get_bdd100k}[w]()
    print("DOWNLOADS DONE")
