"""Feature extraction for uploaded C/C++ source files (prototype, approved D17: complete files only).

It uses the SAME Lizard routine as dataset building (src.data.build_universe.lizard_functions), applied to the
whole uploaded file, so training and prediction use one process. Only the five approved features are produced.
"""
from __future__ import annotations

from pathlib import Path

from src.data.build_universe import lizard_functions

ALLOWED_EXTENSIONS = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh"}
MAX_BYTES = 1_000_000
MAX_FUNCTIONS = 2000


class ExtractionError(ValueError):
    """A problem with the uploaded file that the user can understand and fix."""


def decode_source(filename: str, raw: bytes) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ExtractionError(f"Unsupported file type '{ext or 'none'}'. Upload a complete C or C++ source file "
                              f"({', '.join(sorted(ALLOWED_EXTENSIONS))}).")
    if len(raw) == 0:
        raise ExtractionError("The file is empty.")
    if len(raw) > MAX_BYTES:
        raise ExtractionError(f"The file is larger than {MAX_BYTES // 1000} KB. Upload a smaller source file.")
    if b"\x00" in raw:
        raise ExtractionError("The file looks like a binary file, not source code.")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def extract_functions(filename: str, source: str) -> list[dict]:
    """Return one dict per function: name, start_line, end_line and the five features."""
    try:
        funcs = lizard_functions(filename, source)
    except Exception as exc:                                  # noqa: BLE001 - report as a user-level problem
        raise ExtractionError("The file could not be analysed. It may not be valid C or C++ source.") from exc
    if not funcs:
        raise ExtractionError("No functions were found in this file.")
    if len(funcs) > MAX_FUNCTIONS:
        raise ExtractionError(f"The file has more than {MAX_FUNCTIONS} functions; upload a smaller file.")
    return funcs
