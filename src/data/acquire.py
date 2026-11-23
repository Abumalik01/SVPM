"""Phase B, step 1: acquire CVEfixes from its official Zenodo record.

What this script does (run it on YOUR machine; it needs internet access to
zenodo.org):
  1. Reads the record metadata from the Zenodo API (version, licence, files).
  2. Downloads the files into data/raw/ (resumable) and verifies checksums.
  3. Converts the compressed SQL dump into an SQLite database.
  4. Writes reports/provenance.json describing exactly what was obtained.

It never claims success unless each step finished. Usage:
    python -m src.data.acquire --check        # only show record info
    python -m src.data.acquire --download     # download + verify
    python -m src.data.acquire --build-db     # dump -> SQLite
    python -m src.data.acquire --all
Add --latest to use the newest Zenodo version instead of the pinned one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

from src.data import settings as S


# --------------------------------------------------------------------------
# Zenodo metadata
# --------------------------------------------------------------------------
def fetch_record(record_id: str) -> dict:
    resp = requests.get(f"{S.ZENODO_API}/{record_id}", timeout=60)
    resp.raise_for_status()
    return resp.json()


def resolve_latest_record_id() -> str:
    """Ask Zenodo for the newest version of the CVEfixes concept record."""
    resp = requests.get(f"{S.ZENODO_API}/{S.ZENODO_CONCEPT_RECORD}/versions/latest",
                        timeout=60, allow_redirects=True)
    resp.raise_for_status()
    return str(resp.json()["id"])


def parse_record(rec: dict) -> dict:
    """Extract the facts we record. Defensive: Zenodo's JSON layout has
    changed over time, so missing keys become None instead of guesses."""
    meta = rec.get("metadata", {})
    lic = meta.get("license")
    if isinstance(lic, dict):
        lic = lic.get("id")
    files = []
    for f in rec.get("files", []):
        files.append({
            "name": f.get("key") or f.get("filename"),
            "size_bytes": f.get("size") or f.get("filesize"),
            "checksum": f.get("checksum"),  # e.g. "md5:abcd..."
            "url": (f.get("links") or {}).get("self")
                   or (f.get("links") or {}).get("download"),
        })
    return {
        "record_id": str(rec.get("id")),
        "doi": rec.get("doi") or meta.get("doi"),
        "version": meta.get("version"),
        "publication_date": meta.get("publication_date"),
        "title": meta.get("title"),
        "licence": lic,
        "files": files,
    }


# --------------------------------------------------------------------------
# Download with resume + checksum
# --------------------------------------------------------------------------
def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:  # already complete
            return
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        with open(dest, mode) as out:
            for block in r.iter_content(chunk_size=1 << 20):
                out.write(block)


def verify_checksum(path: Path, checksum: str | None) -> bool | None:
    """True/False if a checksum was provided, None if Zenodo gave none."""
    if not checksum:
        return None
    algo, _, expected = checksum.partition(":")
    if not expected:  # bare md5
        algo, expected = "md5", checksum
    if algo.lower() != "md5":
        return None
    return md5_of(path) == expected.lower()


# --------------------------------------------------------------------------
# SQL dump -> SQLite
# --------------------------------------------------------------------------
def find_dump(raw_dir: Path) -> Path | None:
    for pattern in ("**/*.sql.gz", "**/*.sql"):
        hits = sorted(raw_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def extract_zips(raw_dir: Path) -> None:
    for z in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw_dir)


def build_sqlite(dump: Path, db_path: Path) -> None:
    """Stream the SQL dump into SQLite. Uses the sqlite3 command line tool if
    present (fastest); otherwise falls back to a pure-Python statement loop."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        raise FileExistsError(f"{db_path} already exists; delete it to rebuild.")
    if shutil.which("sqlite3") and dump.suffix == ".gz" and shutil.which("gzip"):
        cmd = f'gzip -dc "{dump}" | sqlite3 "{db_path}"'
        subprocess.run(cmd, shell=True, check=True)
        return
    opener = gzip.open if dump.suffix == ".gz" else open
    con = sqlite3.connect(db_path)
    try:
        buf = ""
        with opener(dump, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                buf += line
                if sqlite3.complete_statement(buf):
                    con.execute(buf)
                    buf = ""
        con.commit()
    finally:
        con.close()


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def write_provenance(info: dict, steps: dict) -> Path:
    S.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "https://doi.org/10.5281/zenodo.4476563",
        "record": info,
        "steps_completed": steps,
        "python": sys.version.split()[0],
        "note": "Values come from the Zenodo API and local verification only.",
    }
    S.PROVENANCE_PATH.write_text(json.dumps(payload, indent=2))
    return S.PROVENANCE_PATH


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--build-db", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--latest", action="store_true")
    a = ap.parse_args()
    if not (a.check or a.download or a.build_db or a.all):
        ap.print_help()
        return 0

    steps = {"metadata": False, "downloaded": False, "checksum_verified": None,
             "database_built": False}
    rid = resolve_latest_record_id() if a.latest else S.PINNED_RECORD_ID
    info = parse_record(fetch_record(rid))
    steps["metadata"] = True
    print(json.dumps(info, indent=2))

    if a.download or a.all:
        results = []
        for f in info["files"]:
            if not f["url"]:
                print(f"No download URL for {f['name']}; skipping.")
                continue
            dest = S.RAW_DIR / f["name"]
            print(f"Downloading {f['name']} ({f['size_bytes']} bytes) ...")
            download_file(f["url"], dest)
            ok = verify_checksum(dest, f["checksum"])
            f["sha256_local"] = sha256_of(dest)
            results.append(ok)
            print(f"  checksum verified: {ok}")
        steps["downloaded"] = bool(results)
        steps["checksum_verified"] = (all(r is True for r in results)
                                      if results else None)

    if a.build_db or a.all:
        extract_zips(S.RAW_DIR)
        dump = find_dump(S.RAW_DIR)
        if dump is None:
            print("No .sql/.sql.gz dump found under data/raw/. Check the download.")
        else:
            print(f"Building SQLite database from {dump.name} ...")
            build_sqlite(dump, S.DB_PATH)
            steps["database_built"] = S.DB_PATH.exists()

    print("Provenance written to", write_provenance(info, steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
