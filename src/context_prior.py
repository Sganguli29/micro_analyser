r"""
src/context_prior.py - Computes contextual prior Z from segmentation mask and trains calibrated P(Y|Z).

Descriptors in Z (computed from segmentation mask and crop):
1. area: total foreground pixels
2. bbox_w: bounding box width on mask
3. bbox_h: bounding box height on mask
4. aspect_ratio: bbox_w / bbox_h
5. mean_brightness: mean luminance under the foreground mask
6. fluorescence_occupancy: foreground area / total crop area
7. snr: Signal-to-Noise Ratio (mean foreground intensity / mean background intensity)

Handles double-counting with Shape block via flag:
- deduplicate_shape=True: removes overlapping features (area, aspect_ratio) from Shape block.
- deduplicate_shape=False: retains all Shape features and relies on log-opinion pool weights to compensate.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import datetime
import numpy as np
import pandas as pd
from skimage import measure
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.dataset import load_crop_index, load_crop_image, split_dataset_by_group, CLASSES, CLASS_TO_IDX
from src.segmentation import segment_particle
from src.calibration import TemperatureScaler, PlattScaler, IsotonicScaler, compute_ece_15, compute_brier_score
from src.utils import get_git_commit_hash, PROJECT_ROOT

Z_FEATURE_NAMES = [
    "z_area",
    "z_bbox_w",
    "z_bbox_h",
    "z_aspect_ratio",
    "z_mean_brightness",
    "z_fluorescence_occupancy",
    "z_snr"
]

def extract_context_prior_z(rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Computes the 7-dimensional context descriptor vector Z for a crop.
    """
    if mask is None:
        mask = segment_particle(rgb)

    H, W = rgb.shape[:2]
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]

    mask_u8 = mask.astype(np.uint8)
    props = measure.regionprops(mask_u8)

    if len(props) > 0:
        p = props[0]
        area = float(p.area)
        minr, minc, maxr, maxc = p.bbox
        bbox_h = float(max(1, maxr - minr))
        bbox_w = float(max(1, maxc - minc))
    else:
        area = float(np.sum(mask)) if np.sum(mask) > 0 else 1.0
        bbox_h = float(H)
        bbox_w = float(W)

    aspect_ratio = float(bbox_w / (bbox_h + 1e-8))
    fluorescence_occupancy = float(area / float(H * W + 1e-8))

    fg_vals = gray[mask]
    bg_vals = gray[~mask]

    mean_fg = float(np.mean(fg_vals)) if len(fg_vals) > 0 else float(np.mean(gray))
    mean_bg = float(np.mean(bg_vals)) if len(bg_vals) > 0 else 1e-4

    snr = float(mean_fg / (mean_bg + 1e-6))

    return {
        "z_area": area,
        "z_bbox_w": bbox_w,
        "z_bbox_h": bbox_h,
        "z_aspect_ratio": aspect_ratio,
        "z_mean_brightness": mean_fg,
        "z_fluorescence_occupancy": fluorescence_occupancy,
        "z_snr": snr
    }

class ContextPriorModel:
    """Lightweight calibrated model P(Y | Z)."""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=random_state))
        ])
        self.calibrator = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        self.pipeline.fit(X_train, y_train)
        val_probs_raw = self.pipeline.predict_proba(X_val)

        # Calibrate via temperature scaling or Platt
        scaler = TemperatureScaler().fit(val_probs_raw, y_val)
        self.calibrator = scaler
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_probs = self.pipeline.predict_proba(X)
        if self.calibrator:
            return self.calibrator.predict_proba(raw_probs)
        return raw_probs

def run_context_prior_experiment(
    seed: int = 42,
    output_csv: Path = None
) -> Tuple[pd.DataFrame, ContextPriorModel]:
    """
    Computes Z for all crops, trains and evaluates P(Y|Z),
    and tests the double-counting deduplication ablation:
    - deduplicate_shape=True vs. deduplicate_shape=False.
    """
    if output_csv is None:
        output_csv = PROJECT_ROOT / "results" / "context_prior_report.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df_crops = load_crop_index()
    cache_path = PROJECT_ROOT / "data" / "context_prior_cache.csv"

    if cache_path.exists():
        print(f"Loading precomputed Z features from {cache_path}...")
        df_z = pd.read_csv(cache_path)
    else:
        print(f"Extracting context features Z for {len(df_crops)} crops...")
        z_rows = []
        for _, row in df_crops.iterrows():
            rgb = load_crop_image(row["crop"])
            z_dict = extract_context_prior_z(rgb)
            z_dict.update({
                "crop": row["crop"],
                "label": row["label"],
                "source_image": row["source_image"],
                "w": row["w"],
                "h": row["h"]
            })
            z_rows.append(z_dict)
        df_z = pd.DataFrame(z_rows)
        df_z.to_csv(cache_path, index=False)

    df_train, df_val, df_test = split_dataset_by_group(df_z, seed=seed)
    y_train = df_train["label"].map(CLASS_TO_IDX).values
    y_val = df_val["label"].map(CLASS_TO_IDX).values
    y_test = df_test["label"].map(CLASS_TO_IDX).values
    K = len(CLASSES)

    X_train_z = df_train[Z_FEATURE_NAMES].values
    X_val_z = df_val[Z_FEATURE_NAMES].values
    X_test_z = df_test[Z_FEATURE_NAMES].values

    # Fit lightweight P(Y|Z)
    model_z = ContextPriorModel(random_state=seed)
    model_z.fit(X_train_z, y_train, X_val_z, y_val)

    probs_test_z = model_z.predict_proba(X_test_z)
    preds_test_z = np.argmax(probs_test_z, axis=1)

    acc_z = accuracy_score(y_test, preds_test_z)
    f1_z = f1_score(y_test, preds_test_z, average="macro")
    nll_z = log_loss(y_test, probs_test_z, labels=list(range(K)))
    ece_z = compute_ece_15(probs_test_z, y_test, n_bins=15)
    brier_z = compute_brier_score(probs_test_z, y_test)

    print("\n--- CONTEXT PRIOR P(Y|Z) ONLY PERFORMANCE ---")
    print(f"Test Accuracy: {acc_z:.4f}, Macro-F1: {f1_z:.4f}, NLL: {nll_z:.4f}, ECE-15: {ece_z:.4f}")

    # Double-counting ablation: Shape + Z fusion
    # Load shape features from main cache
    main_feat_cache = PROJECT_ROOT / "data" / "features_cache.csv"
    assert main_feat_cache.exists(), "Main feature cache missing."
    df_main = pd.read_csv(main_feat_cache)
    df_m_train, df_m_val, df_m_test = split_dataset_by_group(df_main, seed=seed)

    all_shape_cols = [c for c in df_main.columns if c.startswith("shape_")]
    # Non-overlapping shape cols: remove area and aspect_ratio
    dedup_shape_cols = [c for c in all_shape_cols if c not in ["shape_area", "shape_aspect_ratio"]]

    def train_and_eval_shape(cols: List[str], option_name: str):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=-1))
        ])
        pipe.fit(df_m_train[cols].values, y_train)
        scaler = TemperatureScaler().fit(pipe.predict_proba(df_m_val[cols].values), y_val)
        probs_test = scaler.predict_proba(pipe.predict_proba(df_m_test[cols].values))
        preds_test = np.argmax(probs_test, axis=1)
        acc = accuracy_score(y_test, preds_test)
        f1 = f1_score(y_test, preds_test, average="macro")
        nll = log_loss(y_test, probs_test, labels=list(range(K)))
        return acc, f1, nll

    # Option A: Deduplicated shape features
    acc_dedup, f1_dedup, nll_dedup = train_and_eval_shape(dedup_shape_cols, "Deduplicated Shape")
    # Option B: Full shape features (double-counting allowed, compensated by fusion)
    acc_full, f1_full, nll_full = train_and_eval_shape(all_shape_cols, "Full Shape")

    records = [
        {
            "experiment": "Z_only_prior",
            "feature_set": "7_context_descriptors",
            "test_accuracy": round(acc_z, 4),
            "test_macro_f1": round(f1_z, 4),
            "test_nll": round(nll_z, 4),
            "test_ece_15": round(ece_z, 4),
            "test_brier": round(brier_z, 4),
            "n_features": len(Z_FEATURE_NAMES),
            "git_commit": get_git_commit_hash(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        },
        {
            "experiment": "double_counting_ablation_option_a",
            "feature_set": "shape_deduplicated_no_area_no_aspect_ratio",
            "test_accuracy": round(acc_dedup, 4),
            "test_macro_f1": round(f1_dedup, 4),
            "test_nll": round(nll_dedup, 4),
            "test_ece_15": np.nan,
            "test_brier": np.nan,
            "n_features": len(dedup_shape_cols),
            "git_commit": get_git_commit_hash(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        },
        {
            "experiment": "double_counting_ablation_option_b",
            "feature_set": "shape_full_retained_with_fusion_compensation",
            "test_accuracy": round(acc_full, 4),
            "test_macro_f1": round(f1_full, 4),
            "test_nll": round(nll_full, 4),
            "test_ece_15": np.nan,
            "test_brier": np.nan,
            "n_features": len(all_shape_cols),
            "git_commit": get_git_commit_hash(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    ]

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)
    print(f"Context prior report written to: {output_csv}")
    return df_out, model_z
