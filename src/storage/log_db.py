"""Prediction log (SQLite, standard library). Stores feature values and results only.
Uploaded source code is NEVER stored (approved decision U5)."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

from src.ml.features import FEATURE_NAMES

SCHEMA = f"""CREATE TABLE IF NOT EXISTS prediction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_utc TEXT NOT NULL, input_mode TEXT NOT NULL,
    function_name TEXT, {', '.join(f + ' INTEGER' for f in FEATURE_NAMES)},
    predicted_label INTEGER NOT NULL, model_score REAL NOT NULL, threshold REAL NOT NULL, model_version TEXT NOT NULL)"""


def _migrate_if_old(con: sqlite3.Connection) -> None:
    """Early versions required every feature column; keep old rows by renaming that table."""
    info = con.execute("PRAGMA table_info(prediction_log)").fetchall()
    if info and any(r[1] == "top_nesting_level" and r[3] == 1 for r in info):
        con.execute("ALTER TABLE prediction_log RENAME TO prediction_log_old")
        con.commit()


def connect(path: Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    _migrate_if_old(con)
    con.execute(SCHEMA)
    return con


def log_prediction(con: sqlite3.Connection, mode: str, function_name: str | None, features: dict, result: dict) -> None:
    cols = ["timestamp_utc", "input_mode", "function_name", *FEATURE_NAMES,
            "predicted_label", "model_score", "threshold", "model_version"]
    con.execute(
        f"INSERT INTO prediction_log ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        (dt.datetime.now(dt.timezone.utc).isoformat(), mode, function_name, *[(None if features.get(f) is None else int(features[f])) for f in FEATURE_NAMES],
         result["predicted_label"], result["model_score"], result["threshold"], result["model_version"]))
    con.commit()


def recent(con: sqlite3.Connection, n: int = 50) -> list[dict]:
    cur = con.execute("SELECT * FROM prediction_log ORDER BY id DESC LIMIT ?", (int(n),))
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]
