"""
src/morphology_audit.py - Rigorous audit comparing bounding-box vs. segmented mask morphology.

Computes 9 shape descriptors:
- area, perimeter, circularity, eccentricity, solidity, extent, aspect ratio, equivalent diameter, Feret diameter
both from:
1. Bounding-box crop (unsegmented mask, all-True over crop)
2. Segmented mask (foreground particle via Otsu & morphological cleanup)

Reports per-feature numerical differences and classification performance using each representation.
Outputs: results/morphology_bbox_vs_mask.csv.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.dataset import load_crop_index, load_crop_image, split_dataset_by_group, CLASSES, CLASS_TO_IDX
from src.features import extract_shape
from src.segmentation import segment_particle
from src.utils import get_git_commit_hash, PROJECT_ROOT

SHAPE_FEATURE_NAMES = [
    "shape_area",
    "shape_perimeter",
    "shape_circularity",
    "shape_eccentricity",
    "shape_solidity",
    "shape_extent",
    "shape_aspect_ratio",
    "shape_equivalent_diameter",
    "shape_feret_diameter"
]

def audit_morphology(
    seed: int = 42,
    max_samples: int = None,
    output_csv: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes shape descriptors under bounding-box vs. segmented mask,
    evaluates feature divergences, and compares downstream Shape-only classification accuracy.
    """
    if output_csv is None:
        output_csv = PROJECT_ROOT / "results" / "morphology_bbox_vs_mask.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df_crops = load_crop_index()
    if max_samples and max_samples < len(df_crops):
        df_crops = df_crops.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    df_train, df_val, df_test = split_dataset_by_group(df_crops, seed=seed)

    print(f"Auditing morphology on {len(df_crops)} crops across Train ({len(df_train)}), Val ({len(df_val)}), Test ({len(df_test)})...")

    # Extract both bbox and mask shape features
    def extract_split_shapes(split_df: pd.DataFrame, desc: str):
        bbox_rows = []
        mask_rows = []
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=desc):
            rgb = load_crop_image(row["crop"])
            H, W = rgb.shape[:2]
            
            # Condition 1: Bounding-box crop mask (all-True)
            bbox_mask = np.ones((H, W), dtype=bool)
            feats_bbox = extract_shape(rgb, bbox_mask)
            bbox_rows.append(feats_bbox)

            # Condition 2: Segmented mask
            seg_mask = segment_particle(rgb)
            feats_mask = extract_shape(rgb, seg_mask)
            mask_rows.append(feats_mask)

        return pd.DataFrame(bbox_rows), pd.DataFrame(mask_rows)

    df_bbox_train, df_mask_train = extract_split_shapes(df_train, "Extracting Train Shapes")
    df_bbox_val, df_mask_val = extract_split_shapes(df_val, "Extracting Val Shapes")
    df_bbox_test, df_mask_test = extract_split_shapes(df_test, "Extracting Test Shapes")

    y_train = df_train["label"].map(CLASS_TO_IDX).values
    y_val = df_val["label"].map(CLASS_TO_IDX).values
    y_test = df_test["label"].map(CLASS_TO_IDX).values

    # Train shape-only model on Bounding Box features
    rf_bbox = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        method="isotonic", cv=3
    )
    pipe_bbox = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", rf_bbox)
    ])
    pipe_bbox.fit(df_bbox_train[SHAPE_FEATURE_NAMES].values, y_train)
    probs_bbox_test = pipe_bbox.predict_proba(df_bbox_test[SHAPE_FEATURE_NAMES].values)
    preds_bbox_test = np.argmax(probs_bbox_test, axis=1)

    acc_bbox = accuracy_score(y_test, preds_bbox_test)
    f1_bbox = f1_score(y_test, preds_bbox_test, average="macro")
    nll_bbox = log_loss(y_test, probs_bbox_test, labels=list(range(len(CLASSES))))

    # Train shape-only model on Segmented Mask features
    rf_mask = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        method="isotonic", cv=3
    )
    pipe_mask = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", rf_mask)
    ])
    pipe_mask.fit(df_mask_train[SHAPE_FEATURE_NAMES].values, y_train)
    probs_mask_test = pipe_mask.predict_proba(df_mask_test[SHAPE_FEATURE_NAMES].values)
    preds_mask_test = np.argmax(probs_mask_test, axis=1)

    acc_mask = accuracy_score(y_test, preds_mask_test)
    f1_mask = f1_score(y_test, preds_mask_test, average="macro")
    nll_mask = log_loss(y_test, probs_mask_test, labels=list(range(len(CLASSES))))

    print("\n--- CLASSIFICATION PERFORMANCE: BBOX VS MASK ---")
    print(f"Bounding Box Crop Shape: Test Acc={acc_bbox:.4f}, Macro-F1={f1_bbox:.4f}, NLL={nll_bbox:.4f}")
    print(f"Segmented Mask Shape:    Test Acc={acc_mask:.4f}, Macro-F1={f1_mask:.4f}, NLL={nll_mask:.4f}")

    # Compute per-feature difference statistics on test set
    records = []
    
    # Metadata rows
    records.append({
        "descriptor": "SYSTEM_METADATA",
        "mean_bbox": np.nan, "mean_mask": np.nan, "mean_abs_diff": np.nan, "median_abs_diff": np.nan,
        "spearman_corr": np.nan, "model_accuracy": np.nan, "model_macro_f1": np.nan, "model_nll": np.nan,
        "git_commit": get_git_commit_hash(), "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })
    records.append({
        "descriptor": "MODEL_ACCURACY_BBOX_CROP",
        "mean_bbox": round(acc_bbox, 4), "mean_mask": np.nan, "mean_abs_diff": np.nan, "median_abs_diff": np.nan,
        "spearman_corr": np.nan, "model_accuracy": round(acc_bbox, 4), "model_macro_f1": round(f1_bbox, 4), "model_nll": round(nll_bbox, 4),
        "git_commit": get_git_commit_hash(), "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })
    records.append({
        "descriptor": "MODEL_ACCURACY_SEGMENTED_MASK",
        "mean_bbox": np.nan, "mean_mask": round(acc_mask, 4), "mean_abs_diff": np.nan, "median_abs_diff": np.nan,
        "spearman_corr": np.nan, "model_accuracy": round(acc_mask, 4), "model_macro_f1": round(f1_mask, 4), "model_nll": round(nll_mask, 4),
        "git_commit": get_git_commit_hash(), "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })

    for feat in SHAPE_FEATURE_NAMES:
        vals_bbox = df_bbox_test[feat].values
        vals_mask = df_mask_test[feat].values
        abs_diff = np.abs(vals_bbox - vals_mask)
        
        # Correlation
        corr = float(pd.Series(vals_bbox).corr(pd.Series(vals_mask), method="spearman"))
        
        records.append({
            "descriptor": feat,
            "mean_bbox": round(float(np.mean(vals_bbox)), 4),
            "mean_mask": round(float(np.mean(vals_mask)), 4),
            "mean_abs_diff": round(float(np.mean(abs_diff)), 4),
            "median_abs_diff": round(float(np.median(abs_diff)), 4),
            "spearman_corr": round(corr, 4) if not np.isnan(corr) else 0.0,
            "model_accuracy": np.nan,
            "model_macro_f1": np.nan,
            "model_nll": np.nan,
            "git_commit": get_git_commit_hash(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)
    print(f"Morphology audit written to: {output_csv}")

    results_dict = {
        "acc_bbox": acc_bbox, "f1_bbox": f1_bbox, "nll_bbox": nll_bbox,
        "acc_mask": acc_mask, "f1_mask": f1_mask, "nll_mask": nll_mask
    }
    return df_out, results_dict
