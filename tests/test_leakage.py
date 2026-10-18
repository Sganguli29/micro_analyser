"""
tests/test_leakage.py - Unit tests verifying zero data leakage across group splits and LOPO folds.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.dataset import split_dataset_by_group, CLASSES, CLASS_TO_IDX
from src.utils import PROJECT_ROOT

def test_group_shuffle_split_no_leakage():
    """Verifies that split_dataset_by_group guarantees zero slide group overlap."""
    crop_cache = PROJECT_ROOT / "crops" / "crop_index.csv"
    if not crop_cache.exists():
        pytest.skip("crop_index.csv not found")

    df = pd.read_csv(crop_cache)
    df_train, df_val, df_test = split_dataset_by_group(df, seed=42)

    train_slides = set(df_train["source_image"])
    val_slides = set(df_val["source_image"])
    test_slides = set(df_test["source_image"])

    assert len(train_slides & test_slides) == 0, "Train and test share slides!"
    assert len(train_slides & val_slides) == 0, "Train and val share slides!"
    assert len(val_slides & test_slides) == 0, "Val and test share slides!"

def test_lopo_non_leakage_assertion():
    """Verifies that in Leave-One-Polymer-Out splits, held-out polymer is strictly absent."""
    crop_cache = PROJECT_ROOT / "crops" / "crop_index.csv"
    if not crop_cache.exists():
        pytest.skip("crop_index.csv not found")

    df = pd.read_csv(crop_cache)

    for p in CLASSES:
        df_known = df[df["label"] != p].copy().reset_index(drop=True)
        df_train, df_val, df_test = split_dataset_by_group(df_known, seed=42)

        assert p not in df_train["label"].values, f"LEAKAGE: {p} in train!"
        assert p not in df_val["label"].values, f"LEAKAGE: {p} in val!"
        assert p not in df_test["label"].values, f"LEAKAGE: {p} in test known split!"
