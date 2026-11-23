# Phase B: CVEfixes acquisition and audit

Setup (Python 3.10+):
    python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python -m pytest -q tests                              # 11 tests should pass

Acquire (needs internet access to zenodo.org; about 12.7 GB download for v1.0.8, more when unpacked):
    python -m src.data.acquire --check        # shows version, licence, files
    python -m src.data.acquire --all          # download, verify, build data/interim/CVEfixes.db

Audit:
    python -m src.data.audit                  # writes reports/audit_report.json and audit_summary.md

Then send me reports/audit_summary.md and reports/provenance.json.
