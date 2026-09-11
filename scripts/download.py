#!/usr/bin/env python
"""Download XiChen checkpoints / demo data from Zenodo (pure stdlib).

Fetches the two release archives into this repository's root and verifies
their SHA-256 checksums, then unpacks them into ``ckpts/`` and ``data/``.

The Zenodo record DOI and the file checksums below are PLACEHOLDERS — they
are filled in when the archives are uploaded (see README § Download).

Usage:
    python scripts/download.py                 # both archives
    python scripts/download.py --ckpts-only
    python scripts/download.py --data-only
    python scripts/download.py --keep-archives
"""
import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- filled in at Zenodo upload time -----------------------------------------
ZENODO_RECORD = "XXXXXXX"  # e.g. "17210930"
FILES = {
    "xichen_1p0deg_ckpts_v1.tar.gz": {
        "url": f"https://zenodo.org/records/{ZENODO_RECORD}/files/xichen_1p0deg_ckpts_v1.tar.gz",
        "sha256": None,  # "<sha256 of the ckpts tarball>"
        "dest": REPO_ROOT,
    },
    "xichen_1p0deg_data_v1.tar.gz": {
        "url": f"https://zenodo.org/records/{ZENODO_RECORD}/files/xichen_1p0deg_data_v1.tar.gz",
        "sha256": None,  # "<sha256 of the data tarball>"
        "dest": REPO_ROOT,
    },
}
# ------------------------------------------------------------------------------


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fetch(name: str, info: dict, keep_archive: bool) -> None:
    if info["sha256"] is None or "XXXXXXX" in info["url"]:
        sys.exit(
            f"error: {name} still has placeholder URL/checksum — the Zenodo "
            "record has not been published yet. Download manually instead "
            "(see README § Download weights and demo data)."
        )
    archive = REPO_ROOT / name
    if not archive.exists():
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(info["url"], archive)
    else:
        print(f"{name} already downloaded, reusing")

    digest = sha256_of(archive)
    if digest != info["sha256"]:
        archive.unlink()
        sys.exit(f"error: SHA-256 mismatch for {name}: {digest} (deleted)")

    print(f"unpacking {name} ...")
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(info["dest"], filter="data")
        except TypeError:
            # Python < 3.12 lacks tarfile 'data' extraction filter; fall back
            # to unfiltered extraction (README floor is Python >= 3.10).
            print("warning: Python <3.12, extracting without tar path-filtering")
            tf.extractall(info["dest"])
    if not keep_archive:
        archive.unlink()
    print(f"done: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--ckpts-only", action="store_true")
    group.add_argument("--data-only", action="store_true")
    ap.add_argument("--keep-archives", action="store_true",
                    help="do not delete the .tar.gz after unpacking")
    args = ap.parse_args()

    todo = list(FILES)
    if args.ckpts_only:
        todo = [n for n in todo if "ckpts" in n]
    if args.data_only:
        todo = [n for n in todo if "data" in n]
    for name in todo:
        fetch(name, FILES[name], args.keep_archives)


if __name__ == "__main__":
    main()
