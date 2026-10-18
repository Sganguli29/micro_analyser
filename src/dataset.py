"""
src/dataset.py - Dataset loading, group-based splitting, and leakage verification.
"""

from pathlib import Path
from typing import Tuple, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["ABS", "Nylon", "PE", "PET", "PS", "PVC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def load_crop_index(crop_index_path: Optional[Path] = None) -> pd.DataFrame:
    """Loads crop_index.csv and validates required columns."""
    if crop_index_path is None:
        crop_index_path = PROJECT_ROOT / "crops" / "crop_index.csv"
    
    if not crop_index_path.exists():
        raise FileNotFoundError(f"Crop index not found at {crop_index_path}. Run Phase 1 preparation first.")
    
    df = pd.read_csv(crop_index_path)
    required_cols = {"crop", "label", "source_image", "w", "h"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Crop index missing required columns: {missing}")
    return df

def split_dataset_by_group(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.25,
    val_size: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into Train, Val, and Test splits using GroupShuffleSplit on source_image.
    
    Guarantees:
    - Zero slide/group overlap between train, val, and test splits.
    - Test size is test_size of total (default 25%).
    - Val size is val_size of (1 - test_size) (default 20% of 75% = 15% of total).
    - Train size is 80% of 75% = 60% of total.
    """
    groups = df["source_image"].values
    y = df["label"].map(CLASS_TO_IDX).values

    # Split 1: (Train+Val) and Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(df, y, groups=groups))

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Split 2: Train and Val
    groups_train_val = df_train_val["source_image"].values
    y_train_val = df_train_val["label"].map(CLASS_TO_IDX).values

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss_val.split(df_train_val, y_train_val, groups=groups_train_val))

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # Non-negotiable leakage check
    train_groups = set(df_train["source_image"])
    val_groups = set(df_val["source_image"])
    test_groups = set(df_test["source_image"])

    assert len(train_groups & test_groups) == 0, f"DATA LEAKAGE: {len(train_groups & test_groups)} shared groups between train and test!"
    assert len(train_groups & val_groups) == 0, f"DATA LEAKAGE: {len(train_groups & val_groups)} shared groups between train and val!"
    assert len(val_groups & test_groups) == 0, f"DATA LEAKAGE: {len(val_groups & test_groups)} shared groups between val and test!"

    return df_train, df_val, df_test

def load_crop_image(crop_rel_path: str, project_root: Optional[Path] = None) -> np.ndarray:
    """
    Loads a crop image as float32 RGB in range [0, 1].
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    full_path = project_root / crop_rel_path
    if not full_path.exists():
        raise FileNotFoundError(f"Crop image file not found: {full_path}")
    
    bgr = cv2.imread(str(full_path))
    if bgr is None:
        raise ValueError(f"Failed to read image at: {full_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb
