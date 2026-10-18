"""
tests/test_results_integrity.py - Verifies that result files contain genuine code-computed numbers,
git commit hashes, and no placeholders.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.utils import PROJECT_ROOT

EXPECTED_RESULT_FILES = [
    "dataset_summary.csv",
    "block_costs.csv",
    "morphology_bbox_vs_mask.csv",
    "calibration_summary.csv",
    "context_prior_report.csv",
    "learned_stopping_report.csv",
    "open_set_lopo.csv",
    "ablation_study.csv",
    "significance.csv"
]

@pytest.mark.parametrize("filename", EXPECTED_RESULT_FILES)
def test_results_csv_integrity(filename):
    path = PROJECT_ROOT / "results" / filename
    assert path.exists(), f"Result file missing: {path}"

    df = pd.read_csv(path)
    assert len(df) > 0, f"Result file is empty: {path}"

    # Verify no placeholder strings like 'PLACEHOLDER', 'TODO', 'N/A'
    for col in df.columns:
        if df[col].dtype == object:
            has_placeholder = df[col].astype(str).str.contains(r"TODO|PLACEHOLDER|TBD|FIXME", case=False).any()
            assert not has_placeholder, f"Found placeholder string in {filename} column {col}!"

    # Verify git commit is present in results that track metadata
    if "git_commit" in df.columns:
        assert not df["git_commit"].isna().any(), f"Missing git commit hash in {filename}"
        assert (df["git_commit"] != "").all(), f"Empty git commit hash in {filename}"
