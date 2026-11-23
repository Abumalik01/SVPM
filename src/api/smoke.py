"""One-command check that the running API works. Start the server first, then run:

    python -m src.api.smoke                     (default http://127.0.0.1:8000)
    python -m src.api.smoke --url http://127.0.0.1:8000

It sends a metrics request, a source-file upload and a bad upload, and prints PASS/FAIL for each.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import requests

SAMPLE_C = b"""int add(int a, int b) { if (a > 0 && b > 0) { return a + b; } return 0; }
void copy(char *d, const char *s) { while ((*d++ = *s++)); }
int total(int n) { int s = 0; for (int i = 0; i < n; i++) { if (i % 2) { s += i; } else { s -= i; } } return s; }
"""


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    return ok


BIG_FILE = ("\n".join(f"int f{i}(int a, int b) {{ int s = 0; for (int k = 0; k < a; k++) {{ if (k % 3 == b) {{ s += k; }} else {{ s -= 1; }} }} return s; }}"
                     for i in range(200))).encode()
SECONDS_LIMIT = 5.0          # my reading of "a few seconds" in Chapter Three, for a decision by the supervisor


def timing(api: str) -> bool:
    """Measure response times (NFR: a prediction within a few seconds). Prints median and slowest."""
    single, upload = [], []
    for _ in range(20):
        t0 = time.perf_counter()
        requests.post(f"{api}/predict/metrics", json={"nloc": 30, "complexity": 4, "token_count": 200, "n_params": 2, "top_nesting_level": 0}, timeout=30)
        single.append(time.perf_counter() - t0)
    for _ in range(3):
        t0 = time.perf_counter()
        requests.post(f"{api}/predict/source", files={"file": ("big.c", BIG_FILE, "text/plain")}, timeout=120)
        upload.append(time.perf_counter() - t0)
    ok = max(single) < SECONDS_LIMIT and max(upload) < SECONDS_LIMIT
    print(f"{'PASS' if ok else 'FAIL'}  timing: one function median {statistics.median(single):.3f}s, slowest {max(single):.3f}s | "
          f"file with 200 functions median {statistics.median(upload):.3f}s, slowest {max(upload):.3f}s (limit {SECONDS_LIMIT:.0f}s)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--timing", action="store_true", help="also measure response times")
    a = ap.parse_args()
    results = []
    try:
        h = requests.get(f"{a.url}/health", timeout=15)
    except requests.RequestException as exc:
        print(f"FAIL  server not reachable at {a.url}: {exc}\nStart it with: python -m uvicorn src.api.main:app --port 8000")
        return 1
    info = (h.json() or {}).get("active_model") if h.ok else None
    results.append(check("GET /health", h.ok and info is not None, f"active model: {info and info.get('version')}"))
    if info is None:
        print("No active model. Run: python -m src.ml.final_fit --promote")
        return 1
    m = requests.post(f"{a.url}/predict/metrics", json={"nloc": 30, "complexity": 4, "token_count": 200, "n_params": 2,
                                                         "top_nesting_level": 0}, timeout=30)
    results.append(check("POST /predict/metrics", m.ok and "model_score" in m.text, f"HTTP {m.status_code}"))
    u = requests.post(f"{a.url}/predict/source", files={"file": ("sample.c", SAMPLE_C, "text/plain")}, timeout=60)
    n = len(u.json().get("results", [])) if u.ok else 0
    results.append(check("POST /predict/source (3 functions expected)", u.ok and n == 3, f"HTTP {u.status_code}, functions: {n}"))
    b = requests.post(f"{a.url}/predict/source", files={"file": ("notes.txt", b"hello", "text/plain")}, timeout=30)
    results.append(check("bad upload is refused with a clear message", b.status_code == 400, f"HTTP {b.status_code}"))
    if a.timing:
        results.append(timing(a.url))
    print("ALL PASSED" if all(results) else "SOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
