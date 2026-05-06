"""
One-shot downloader for the HAM10000 dataset from Harvard Dataverse.

Fetches:
  - HAM10000_images_part_1.zip
  - HAM10000_images_part_2.zip
  - HAM10000_metadata.csv  (or HAM10000_metadata.tab as fallback)

Extracts the two zips so the final layout is:
  Project/HAM10000/
    HAM10000_metadata.csv
    HAM10000_images_part_1/   (~5,000 .jpg)
    HAM10000_images_part_2/   (~5,000 .jpg)

Idempotent: re-running skips files that already exist with non-zero size.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

DOI = "doi:10.7910/DVN/DBW86T"
DATAVERSE = "https://dataverse.harvard.edu"
META_URL = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={DOI}"
DOWNLOAD_TPL = f"{DATAVERSE}/api/access/datafile/{{file_id}}"

WANTED_NAMES = [
    "HAM10000_images_part_1.zip",
    "HAM10000_images_part_2.zip",
    "HAM10000_metadata.csv",
    "HAM10000_metadata.tab",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST_ROOT = PROJECT_ROOT / "HAM10000"


def fetch_file_index() -> dict[str, dict]:
    """Return {filename: {id, size}} for files listed in the dataset's latest version."""
    print(f"Fetching dataset metadata from Harvard Dataverse...")
    req = urllib.request.Request(META_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    files = payload.get("data", {}).get("latestVersion", {}).get("files", [])
    index: dict[str, dict] = {}
    for f in files:
        df = f.get("dataFile", {})
        name = df.get("filename") or f.get("label")
        file_id = df.get("id")
        size = df.get("filesize")
        if name and file_id is not None:
            index[name] = {"id": file_id, "size": size}
    return index


def download(file_id: int, dest: Path, expected_size: int | None) -> None:
    """Stream-download a single file. Skip if already complete."""
    if dest.exists():
        local_size = dest.stat().st_size
        if expected_size is None or local_size == expected_size:
            print(f"  [skip] {dest.name} already present ({local_size:,} bytes)")
            return
        print(f"  [redo] {dest.name} size mismatch (local {local_size:,} vs expected {expected_size:,})")

    url = DOWNLOAD_TPL.format(file_id=file_id)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {dest.name} <- file id {file_id}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        total = 0
        chunk = 1 << 20  # 1 MiB
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            total += len(buf)
            if expected_size:
                pct = total / expected_size * 100
                sys.stdout.write(f"\r        {total/1e6:7.1f} MB / {expected_size/1e6:7.1f} MB  ({pct:5.1f}%)")
            else:
                sys.stdout.write(f"\r        {total/1e6:7.1f} MB")
            sys.stdout.flush()
    sys.stdout.write("\n")
    tmp.replace(dest)


def extract_zip(zip_path: Path, target_root: Path) -> None:
    """Extract zip into target_root; the zip is expected to contain a top-level folder
    matching the dataset's image part name (e.g. HAM10000_images_part_1/...).
    """
    expected_dir = target_root / zip_path.stem  # e.g. HAM10000/HAM10000_images_part_1
    if expected_dir.exists() and any(expected_dir.iterdir()):
        existing = sum(1 for _ in expected_dir.glob("*.jpg"))
        print(f"  [skip] {zip_path.name} already extracted ({existing} jpgs in {expected_dir.name})")
        return
    print(f"  [unzip] {zip_path.name} -> {target_root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_root)


def main() -> int:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    index = fetch_file_index()
    print(f"Dataset has {len(index)} files. Looking for: {WANTED_NAMES}")
    available = [n for n in WANTED_NAMES if n in index]
    if not available:
        print("ERROR: none of the expected filenames were found in the dataset listing.")
        print("Available names:", sorted(index)[:20], "...")
        return 1

    # Prefer .csv metadata over .tab if both present
    metadata_name = "HAM10000_metadata.csv" if "HAM10000_metadata.csv" in index else "HAM10000_metadata.tab"
    targets = [
        ("HAM10000_images_part_1.zip", DEST_ROOT / "HAM10000_images_part_1.zip"),
        ("HAM10000_images_part_2.zip", DEST_ROOT / "HAM10000_images_part_2.zip"),
        (metadata_name, DEST_ROOT / "HAM10000_metadata.csv"),
    ]

    print("\nDownloading:")
    for name, dest in targets:
        if name not in index:
            print(f"  [warn] '{name}' not in dataset listing - skipping")
            continue
        download(index[name]["id"], dest, index[name].get("size"))

    print("\nExtracting:")
    for name in ("HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"):
        z = DEST_ROOT / name
        if z.exists():
            extract_zip(z, DEST_ROOT)
        else:
            print(f"  [warn] {name} not found; cannot extract")

    print("\nFinal counts:")
    for sub in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        d = DEST_ROOT / sub
        n = sum(1 for _ in d.glob("*.jpg")) if d.exists() else 0
        print(f"  {sub}: {n} jpgs")
    meta = DEST_ROOT / "HAM10000_metadata.csv"
    print(f"  HAM10000_metadata.csv: {'present' if meta.exists() else 'MISSING'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
