"""Central settings for Phase B (dataset acquisition and audit).

Everything that is a choice (not a fact about the data) lives here, so it can
be reviewed and recorded in the decision log.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- Acquisition -----------------------------------------------------------
# Concept record of CVEfixes on Zenodo (always resolves to the newest version).
ZENODO_CONCEPT_RECORD = "4476563"
# Version pinned for reproducibility (Zenodo page: v1.0.8, published 28 July 2024).
# Confirm this is the version you want; use --latest to check for a newer one.
PINNED_RECORD_ID = "4476564"
ZENODO_API = "https://zenodo.org/api/records"

RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
DB_PATH = INTERIM_DIR / "CVEfixes.db"
REPORT_DIR = ROOT / "reports"
PROVENANCE_PATH = REPORT_DIR / "provenance.json"

# --- Audit -----------------------------------------------------------------
# Tables and columns expected from the authors' collection code
# (secureIT-project/CVEfixes, Code/collect_commits.py and collect_projects.py).
EXPECTED_SCHEMA = {
    "fixes": ["cve_id", "hash", "repo_url"],
    "commits": ["hash", "repo_url", "author", "author_date", "author_timezone",
                "committer", "committer_date", "committer_timezone", "msg",
                "merge", "parents", "num_lines_added", "num_lines_deleted",
                "dmm_unit_complexity", "dmm_unit_interfacing", "dmm_unit_size"],
    "file_change": ["file_change_id", "hash", "filename", "old_path", "new_path",
                    "change_type", "diff", "diff_parsed", "num_lines_added",
                    "num_lines_deleted", "code_after", "code_before", "nloc",
                    "complexity", "token_count", "programming_language"],
    "method_change": ["method_change_id", "file_change_id", "name", "signature",
                      "parameters", "start_line", "end_line", "code", "nloc",
                      "complexity", "token_count", "top_nesting_level",
                      "before_change"],
    "repository": ["repo_url", "repo_name", "description", "date_created",
                   "date_last_push", "homepage", "repo_language", "owner",
                   "forks_count", "stars_count"],
    "cve": ["cve_id"],
    "cwe": ["cwe_id", "cwe_name"],
    "cwe_classification": ["cve_id", "cwe_id"],
}

NUMERIC_FEATURES = ["nloc", "complexity", "token_count", "top_nesting_level",
                    "n_params", "code_len"]

# Languages the student said they are interested in; always reported.
LANGUAGES_OF_INTEREST = ["C", "C++", "Java", "Python"]

# Expected file extensions, used only to cross-check the guessed language.
EXT_EXPECTED = {
    "C": {".c", ".h"},
    "C++": {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h"},
    "Java": {".java"},
    "Python": {".py"},
}


@dataclass
class RecommendationRules:
    """ADVISORY thresholds (engineering assumptions, not published standards).
    Change them here and re-run; the report records the values used."""
    min_projects: int = 30               # enough groups for grouped cross-validation
    min_minority_rows: int = 1500        # rows of the rarer class in the subset
    max_top_project_share: float = 0.35  # one project must not dominate
    max_exact_pair_share: float = 0.50   # before/after pairs with identical metrics


@dataclass
class AuditConfig:
    db_path: Path = DB_PATH
    out_dir: Path = REPORT_DIR
    top_languages: int = 15
    rules: RecommendationRules = field(default_factory=RecommendationRules)
