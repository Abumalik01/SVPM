"""Project paths for the prototype (kept separate from src/data/settings.py so that file stays untouched)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOG_DB = ROOT / "logs" / "predictions.db"
REPORT_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"
