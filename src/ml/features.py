"""Phase D: the five approved method-level Lizard features and safe data loading.
load_development() returns ONLY the development partition: the sealed test data
is never returned by this module."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml import splits

FEATURE_NAMES = ["nloc", "complexity", "token_count", "top_nesting_level", "n_params"]   # original approved five
FEATURE_SETS = {"v1_five": list(FEATURE_NAMES),
                "v2_four": ["nloc", "complexity", "token_count", "n_params"]}   # chosen by the pre-registered rule (S2)


def check_features(df: pd.DataFrame, names=None) -> None:
    names = list(names or FEATURE_NAMES)
    missing = [c for c in names + ["label", "repo_url", "fold"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    x = df[names].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError("features contain missing or non-finite values")
    if not set(df.label.unique()) <= {0, 1}:
        raise ValueError("label must be 0 or 1")


def development_only(df_assigned: pd.DataFrame, names=None) -> pd.DataFrame:
    dev = splits.get_development(df_assigned)
    assert set(dev.partition.unique()) == {"dev"}
    check_features(dev, names)
    return dev.reset_index(drop=True)


def load_development(csv_path, assignment_path):
    """Return (primary_dev, secondary_dev). Both use the identical project assignment."""
    primary = pd.read_csv(csv_path)
    assignment = pd.read_csv(assignment_path)
    secondary = primary[primary.pair_complete].copy()
    p = development_only(splits.apply_assignment(primary, assignment))
    s = development_only(splits.apply_assignment(secondary, assignment))
    return p, s


def load_development_v2(csv_path, assignment_path, names=None):
    """Function dataset (target v2): returns ONLY the development partition (sealed test never returned)."""
    df = pd.read_csv(csv_path)
    return development_only(splits.apply_assignment(df, pd.read_csv(assignment_path)), names)
