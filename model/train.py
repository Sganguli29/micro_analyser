"""
model/train.py - Phases 3 & 5: Model Training, Fusion, Evaluation, and Serve Bundle Export

Handles:
- Feature extraction over all crops with precise runtime profiling (empirical block costs)
- Strict GroupShuffleSplit on source_image with zero train/val/test leakage assertion
- Calibrated Random Forest experts (SimpleImputer -> StandardScaler -> CalibratedClassifierCV(isotonic, cv=3))
- Log-opinion pool weight optimization (Nelder-Mead on validation NLL)
- Simulation pool generation from training split (50 per class)
- Full policy benchmarking (Adaptive vs Fixed-Best, Fixed-Cheapest, Random, All-Blocks)
- Macro-F1 & Tau sweep frontier analysis
- First-block choice distribution analysis
- Exporting results/ CSVs + manifest.json and serve_bundle/ assets for the backend
"""

import os
import sys
import time
import json
import hashlib
import warnings
import joblib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats, optimize
from scipy.special import softmax, logsumexp
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss

warnings.filterwarnings("ignore")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.features import BLOCK_EXTRACTORS, extract_all_blocks
from engine import AdaptiveInformationEngine

RANDOM_SEED = 42
CLASSES = ["ABS", "Nylon", "PE", "PET", "PS", "PVC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class FastSampleExpert:
    """Lightweight wrapper providing instantaneous predict_proba during batch evaluations."""
    def __init__(self, proba_row):
        self.proba = proba_row.reshape(1, -1)
    def predict_proba(self, X):
        return self.proba

def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error across n_bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    
    for i in range(n_bins):
        bin_mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (bin_count / N) * np.abs(bin_acc - bin_conf)
            
    return float(ece)

def extract_and_profile_features(crop_index_df: pd.DataFrame, project_root: Path, cache_path: Path):
    """
    Extracts features for all crops, measures extraction time per block,
    computes normalized costs, and saves to CSV cache.
    """
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}...")
        feat_df = pd.read_csv(cache_path)
        cost_path = cache_path.parent / "block_costs.json"
        if cost_path.exists():
            with open(cost_path, "r") as f:
                block_costs = json.load(f)
            return feat_df, block_costs

    print(f"Extracting features for {len(crop_index_df)} crops and profiling runtime...")
    
    timing_records = {b: [] for b in BLOCK_EXTRACTORS}
    feature_rows = []
    
    # Warmup
    dummy_rgb = np.random.rand(64, 64, 3).astype(np.float32)
    dummy_mask = np.ones((64, 64), dtype=bool)
    for b, fn in BLOCK_EXTRACTORS.items():
        fn(dummy_rgb, dummy_mask)

    for idx, row in crop_index_df.iterrows():
        crop_rel = row["crop"]
        crop_path = project_root / crop_rel
        
        img_bgr = cv2.imread(str(crop_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)
        
        row_dict = {
            "crop": crop_rel,
            "label": row["label"],
            "source_image": row["source_image"],
            "w": row["w"],
            "h": row["h"]
        }
        
        for b, fn in BLOCK_EXTRACTORS.items():
            t0 = time.perf_counter()
            feats = fn(img_rgb, mask)
            dt = time.perf_counter() - t0
            timing_records[b].append(dt)
            row_dict.update(feats)
            
        feature_rows.append(row_dict)

    feat_df = pd.DataFrame(feature_rows)
    feat_df.to_csv(cache_path, index=False)
    print(f"Saved feature cache ({feat_df.shape[0]} rows, {feat_df.shape[1]} cols) to {cache_path}")

    # Compute empirical median execution time per block
    raw_medians = {b: float(np.median(timing_records[b])) for b in BLOCK_EXTRACTORS}
    min_median = min(raw_medians.values())
    block_costs = {b: round(float(raw_medians[b] / min_median), 2) for b in BLOCK_EXTRACTORS}
    
    print("\n--- EMPIRICAL BLOCK COSTS (Normalized to 1.0 min) ---")
    for b in BLOCK_EXTRACTORS:
        print(f"  {b}: median raw {raw_medians[b]*1000.0:.3f} ms -> normalized cost = {block_costs[b]}")

    cost_path = cache_path.parent / "block_costs.json"
    with open(cost_path, "w") as f:
        json.dump(block_costs, f, indent=2)

    return feat_df, block_costs

def run_training_pipeline():
    np.random.seed(RANDOM_SEED)

    crops_dir = PROJECT_ROOT / "crops"
    crop_index_path = crops_dir / "crop_index.csv"
    assert crop_index_path.exists(), f"Missing crop_index.csv at {crop_index_path}. Run Phase 1 first."

    crop_df = pd.read_csv(crop_index_path)
    data_dir = PROJECT_ROOT / "data"
    cache_path = data_dir / "features_cache.csv"
    
    feat_df, block_costs = extract_and_profile_features(crop_df, PROJECT_ROOT, cache_path)
    
    # Hash feature table
    feat_bytes = pd.read_csv(cache_path).to_csv(index=False).encode('utf-8')
    feat_hash = hashlib.sha256(feat_bytes).hexdigest()
    print(f"Feature table SHA256: {feat_hash}")

    # Determine feature columns per block
    sample_rgb = np.ones((32, 32, 3), dtype=np.float32)
    sample_mask = np.ones((32, 32), dtype=bool)
    block_columns = {}
    for b, fn in BLOCK_EXTRACTORS.items():
        sample_out = fn(sample_rgb, sample_mask)
        cols = list(sample_out.keys())
        block_columns[b] = cols
        print(f"Block '{b}': {len(cols)} features ({cols[:3]}...)")

    # PHASE 3: GroupShuffleSplit on source_image
    print("\n--- SPLITTING DATASET (GroupShuffleSplit by source_image) ---")
    groups = feat_df["source_image"].values
    y = feat_df["label"].map(CLASS_TO_IDX).values
    
    # 1. Split into (Train+Val) and Test (75% / 25%)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(gss_test.split(feat_df, y, groups=groups))
    
    df_train_val = feat_df.iloc[train_val_idx].reset_index(drop=True)
    df_test = feat_df.iloc[test_idx].reset_index(drop=True)
    groups_train_val = df_train_val["source_image"].values
    y_train_val = df_train_val["label"].map(CLASS_TO_IDX).values

    # 2. Split (Train+Val) into Train and Val (80% / 20% of 75% -> 60% / 15% of total)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss_val.split(df_train_val, y_train_val, groups=groups_train_val))
    
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # CRITICAL LEAKAGE ASSERTION: Zero group overlap between train, val, and test
    train_groups = set(df_train["source_image"])
    val_groups = set(df_val["source_image"])
    test_groups = set(df_test["source_image"])

    leak_train_test = train_groups & test_groups
    leak_train_val = train_groups & val_groups
    leak_val_test = val_groups & test_groups

    assert len(leak_train_test) == 0, f"DATA LEAKAGE DETECTED: {len(leak_train_test)} shared groups between train and test! Aborting."
    assert len(leak_train_val) == 0, f"DATA LEAKAGE DETECTED: {len(leak_train_val)} shared groups between train and val! Aborting."
    assert len(leak_val_test) == 0, f"DATA LEAKAGE DETECTED: {len(leak_val_test)} shared groups between val and test! Aborting."

    print(f"Zero group overlap verified across all splits!")
    print(f"  Train set: {len(df_train)} crops across {len(train_groups)} source images")
    print(f"  Val set:   {len(df_val)} crops across {len(val_groups)} source images")
    print(f"  Test set:  {len(df_test)} crops across {len(test_groups)} source images")

    y_train = df_train["label"].map(CLASS_TO_IDX).values
    y_val = df_val["label"].map(CLASS_TO_IDX).values
    y_test = df_test["label"].map(CLASS_TO_IDX).values

    # Empirical Prior Distribution from Training set
    train_class_counts = np.bincount(y_train, minlength=len(CLASSES))
    prior = train_class_counts / float(len(y_train))
    log_prior = np.log(prior + 1e-9)
    print(f"Empirical Train Prior: {dict(zip(CLASSES, np.round(prior, 4)))}")

    # PHASE 3: Train Calibrated Expert per Block
    print("\n--- TRAINING CALIBRATED EXPERTS PER BLOCK ---")
    experts = {}
    block_report_rows = []
    
    val_probs_by_block = {}
    test_probs_by_block = {}

    for b, cols in block_columns.items():
        print(f"Fitting expert for '{b}'...")
        X_train_b = df_train[cols].values
        X_val_b = df_val[cols].values
        X_test_b = df_test[cols].values

        base_rf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("calibrated_rf", CalibratedClassifierCV(base_rf, method="isotonic", cv=3))
        ])

        clf.fit(X_train_b, y_train)
        experts[b] = clf

        val_p = clf.predict_proba(X_val_b)
        test_p = clf.predict_proba(X_test_b)

        val_probs_by_block[b] = val_p
        test_probs_by_block[b] = test_p

        val_acc = accuracy_score(y_val, np.argmax(val_p, axis=1))
        test_acc = accuracy_score(y_test, np.argmax(test_p, axis=1))
        val_nll = log_loss(y_val, val_p, labels=list(range(len(CLASSES))))
        test_nll = log_loss(y_test, test_p, labels=list(range(len(CLASSES))))
        val_ece = compute_ece(val_p, y_val, n_bins=10)
        test_ece = compute_ece(test_p, y_test, n_bins=10)

        block_report_rows.append({
            "block": b,
            "n_features": len(cols),
            "cost": block_costs[b],
            "val_acc": round(val_acc, 4),
            "test_acc": round(test_acc, 4),
            "val_nll": round(val_nll, 4),
            "test_nll": round(test_nll, 4),
            "val_ece": round(val_ece, 4),
            "test_ece": round(test_ece, 4)
        })

    df_block_report = pd.DataFrame(block_report_rows)
    print("\n--- BLOCK REPORT TABLE ---")
    print(df_block_report.to_string(index=False))

    # PHASE 3: Weighted Log-Opinion Pool Fusion Optimization
    print("\n--- FITTING LOG-OPINION POOL WEIGHTS ---")
    val_lrs = {}
    for b in BLOCK_EXTRACTORS:
        p_clipped = np.clip(val_probs_by_block[b], 1e-6, 1.0)
        val_lrs[b] = np.log(p_clipped) - log_prior[None, :]

    def nll_objective(theta):
        weights = np.exp(theta)
        fused_log_p = log_prior.copy()[None, :]
        for i, b in enumerate(BLOCK_EXTRACTORS):
            fused_log_p = fused_log_p + weights[i] * val_lrs[b]
        fused_p = softmax(fused_log_p, axis=1)
        return log_loss(y_val, fused_p, labels=list(range(len(CLASSES))))

    init_theta = np.zeros(len(BLOCK_EXTRACTORS))
    res = optimize.minimize(nll_objective, init_theta, method="Nelder-Mead")
    opt_weights = np.exp(res.x)
    block_weights = {b: round(float(opt_weights[i]), 4) for i, b in enumerate(BLOCK_EXTRACTORS)}
    
    print("Fitted Block Weights (w_b):")
    for b, w in block_weights.items():
        print(f"  {b}: {w}")

    # Evaluate All-Blocks fused performance on Test set
    test_fused_log_p = log_prior.copy()[None, :]
    for b in BLOCK_EXTRACTORS:
        p_clipped = np.clip(test_probs_by_block[b], 1e-6, 1.0)
        lr_b = np.log(p_clipped) - log_prior[None, :]
        test_fused_log_p = test_fused_log_p + block_weights[b] * lr_b
    test_fused_p = softmax(test_fused_log_p, axis=1)
    all_blocks_test_acc = accuracy_score(y_test, np.argmax(test_fused_p, axis=1))
    all_blocks_test_macro_f1 = f1_score(y_test, np.argmax(test_fused_p, axis=1), average="macro")
    print(f"All-Blocks Fusion Test Accuracy: {all_blocks_test_acc:.4f}, Macro-F1: {all_blocks_test_macro_f1:.4f}")

    # PHASE 4: Build Training Simulation Pool (50 per class strictly from TRAIN split)
    print("\n--- BUILDING TRAINING SIMULATION POOL ---")
    pool_indices = []
    for c_idx in range(len(CLASSES)):
        c_train_indices = np.where(y_train == c_idx)[0]
        assert len(c_train_indices) >= 50, f"Class {CLASSES[c_idx]} has only {len(c_train_indices)} training samples (need 50)!"
        sampled_c = np.random.choice(c_train_indices, size=50, replace=False)
        pool_indices.extend(sampled_c)

    pool_indices = np.array(pool_indices)
    pool_cls = y_train[pool_indices]
    pool_counts = np.bincount(pool_cls, minlength=len(CLASSES)).astype(float)
    print(f"Simulation pool size M={len(pool_indices)} ({pool_counts} samples/class)")

    pool_lr = {}
    for b, cols in block_columns.items():
        X_pool_b = df_train.iloc[pool_indices][cols].values
        p_pool_raw = experts[b].predict_proba(X_pool_b)
        p_pool = np.clip(p_pool_raw, 1e-6, 1.0)
        p_pool /= p_pool.sum(axis=1, keepdims=True)
        pool_lr[b] = block_weights[b] * (np.log(p_pool) - log_prior[None, :])

    # Establish baseline orderings
    fixed_best_order = list(df_block_report.sort_values(by="val_acc", ascending=False)["block"])
    cheapest_order = list(df_block_report.sort_values(by="cost", ascending=True)["block"])
    print(f"Fixed Best-First Order: {fixed_best_order}")
    print(f"Fixed Cheapest Order:   {cheapest_order}")

    # Full Engine for inference
    engine = AdaptiveInformationEngine(
        classes=CLASSES,
        log_prior=log_prior,
        experts=experts,
        block_weights=block_weights,
        block_costs=block_costs,
        pool_lr=pool_lr,
        pool_cls=pool_cls,
        pool_counts=pool_counts,
        block_feature_names=block_columns,
        fixed_order=fixed_best_order,
        cheapest_order=cheapest_order
    )

    # PHASE 5: Comprehensive Policy Evaluation on Test Split
    print("\n--- PHASE 5: COMPREHENSIVE POLICY BENCHMARK ON TEST SPLIT ---")
    
    # Prepare fast mock engine for batch evaluation
    N_test = len(df_test)
    test_dummy_features = {b: np.zeros((1, len(block_columns[b]))) for b in BLOCK_EXTRACTORS}

    def evaluate_policy_fast(pol_name, tau_val=0.90, lambda_val=0.01):
        preds = []
        costs = []
        n_blocks = []
        first_blocks = []
        
        for i in range(N_test):
            true_lbl = CLASSES[y_test[i]]
            # Fast sample experts
            sample_experts = {b: FastSampleExpert(test_probs_by_block[b][i]) for b in BLOCK_EXTRACTORS}
            sample_engine = AdaptiveInformationEngine(
                classes=CLASSES,
                log_prior=log_prior,
                experts=sample_experts,
                block_weights=block_weights,
                block_costs=block_costs,
                pool_lr=pool_lr,
                pool_cls=pool_cls,
                pool_counts=pool_counts,
                block_feature_names=block_columns,
                fixed_order=fixed_best_order,
                cheapest_order=cheapest_order
            )

            res = sample_engine.run(
                test_dummy_features,
                policy=pol_name,
                tau=tau_val,
                lambda_=lambda_val,
                true_label=true_lbl,
                random_seed=RANDOM_SEED + i
            )
            preds.append(CLASS_TO_IDX[res["final"]["predicted_class"]])
            costs.append(res["final"]["total_cost"])
            n_blocks.append(len(res["final"]["blocks_used"]))
            if len(res["steps"]) > 0:
                first_blocks.append(res["steps"][0]["chosen_block"])

        preds = np.array(preds)
        costs = np.array(costs)
        n_blocks = np.array(n_blocks)

        macro_f1 = f1_score(y_test, preds, average="macro")
        acc = accuracy_score(y_test, preds)
        mean_cost = float(np.mean(costs))
        mean_b = float(np.mean(n_blocks))
        
        return {
            "policy": pol_name,
            "tau": tau_val,
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(acc, 4),
            "mean_cost": round(mean_cost, 2),
            "mean_blocks": round(mean_b, 2),
            "preds": preds,
            "costs": costs,
            "n_blocks": n_blocks,
            "first_blocks": first_blocks
        }

    # Main comparison at default tau=0.90
    policies_to_run = ["adaptive", "fixed_best_first", "fixed_cheapest", "random_order", "all_blocks"]
    policy_results = {}
    comp_table_rows = []

    for pol in policies_to_run:
        res = evaluate_policy_fast(pol, tau_val=0.90, lambda_val=0.01)
        policy_results[pol] = res
        all_cost = engine.all_blocks_cost
        cost_saving = round((1.0 - res["mean_cost"] / all_cost) * 100.0, 1)
        comp_table_rows.append({
            "Policy": pol,
            "Macro-F1": res["macro_f1"],
            "Accuracy": res["accuracy"],
            "Mean Cost": res["mean_cost"],
            "Mean Blocks": res["mean_blocks"],
            "Cost Saving (%)": cost_saving
        })

    df_comp = pd.DataFrame(comp_table_rows)
    print("\n--- POLICY COMPARISON (Headline Metrics at tau=0.90) ---")
    print(df_comp.to_string(index=False))

    # First block selection distribution for adaptive
    first_choice_counts = pd.Series(policy_results["adaptive"]["first_blocks"]).value_counts()
    print("\n--- ADAPTIVE FIRST-BLOCK SELECTION DISTRIBUTION ---")
    for b_name, count in first_choice_counts.items():
        pct = (count / N_test) * 100.0
        print(f"  {b_name}: {count}/{N_test} ({pct:.1f}%)")

    # Per-Class Table for Adaptive
    per_class_rows = []
    ad_preds = policy_results["adaptive"]["preds"]
    ad_n_blocks = policy_results["adaptive"]["n_blocks"]
    ad_costs = policy_results["adaptive"]["costs"]

    for c_idx, c_name in enumerate(CLASSES):
        c_mask = (y_test == c_idx)
        c_total = np.sum(c_mask)
        c_acc = accuracy_score(y_test[c_mask], ad_preds[c_mask]) if c_total > 0 else 0.0
        c_f1 = f1_score(y_test == c_idx, ad_preds == c_idx, average="binary") if c_total > 0 else 0.0
        c_mb = float(np.mean(ad_n_blocks[c_mask])) if c_total > 0 else 0.0
        c_mcost = float(np.mean(ad_costs[c_mask])) if c_total > 0 else 0.0
        per_class_rows.append({
            "Polymer": c_name,
            "Test Samples": c_total,
            "Accuracy": round(c_acc, 4),
            "F1-Score": round(c_f1, 4),
            "Mean Blocks": round(c_mb, 2),
            "Mean Cost": round(c_mcost, 2)
        })

    df_per_class = pd.DataFrame(per_class_rows)
    print("\n--- ADAPTIVE PER-CLASS PERFORMANCE TABLE ---")
    print(df_per_class.to_string(index=False))

    # Tau Sweep for Cost-vs-Accuracy and Cost-vs-F1 frontier
    print("\n--- SWEEPING TAU OVER [0.40, 0.55, 0.70, 0.80, 0.90, 0.95, 0.99] ---")
    tau_vals = [0.40, 0.55, 0.70, 0.80, 0.90, 0.95, 0.99]
    tau_sweep_rows = []
    
    sweep_policies = ["adaptive", "fixed_best_first", "fixed_cheapest"]
    sweep_data = {pol: {"costs": [], "macro_f1": [], "acc": []} for pol in sweep_policies}

    for tau_v in tau_vals:
        for pol in sweep_policies:
            res_tau = evaluate_policy_fast(pol, tau_val=tau_v)
            tau_sweep_rows.append({
                "policy": pol,
                "tau": tau_v,
                "macro_f1": res_tau["macro_f1"],
                "accuracy": res_tau["accuracy"],
                "mean_cost": res_tau["mean_cost"],
                "mean_blocks": res_tau["mean_blocks"]
            })
            sweep_data[pol]["costs"].append(res_tau["mean_cost"])
            sweep_data[pol]["macro_f1"].append(res_tau["macro_f1"])
            sweep_data[pol]["acc"].append(res_tau["accuracy"])

    df_tau_sweep = pd.DataFrame(tau_sweep_rows)

    # Save Results & Plots
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_block_report.to_csv(results_dir / "block_report.csv", index=False)
    df_comp.to_csv(results_dir / "policy_comparison.csv", index=False)
    df_per_class.to_csv(results_dir / "per_class_metrics.csv", index=False)
    df_tau_sweep.to_csv(results_dir / "tau_sweep.csv", index=False)

    # Plot Cost vs Macro-F1 Frontier
    plt.figure(figsize=(8, 5))
    colors = {"adaptive": "#10b981", "fixed_best_first": "#3b82f6", "fixed_cheapest": "#f59e0b"}
    markers = {"adaptive": "o", "fixed_best_first": "s", "fixed_cheapest": "^"}

    for pol in sweep_policies:
        plt.plot(
            sweep_data[pol]["costs"],
            sweep_data[pol]["macro_f1"],
            marker=markers[pol],
            linewidth=2,
            label=pol.replace("_", " ").title(),
            color=colors[pol]
        )
    
    plt.scatter([engine.all_blocks_cost], [all_blocks_test_macro_f1], color="#ef4444", marker="*", s=150, zorder=5, label="All Blocks (Ceiling)")
    plt.xlabel("Mean Cost Paid", fontsize=12)
    plt.ylabel("Macro-F1 Score", fontsize=12)
    plt.title("Cost vs Macro-F1 Trade-off Frontier", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(results_dir / "frontier_macro_f1.png", dpi=300)
    plt.close()

    # Save Manifest JSON
    manifest = {
        "dataset": "sombsuk/dataset_microplastic (Rermborirak et al., 2025)",
        "license": "CC BY-NC-ND 4.0",
        "random_seed": RANDOM_SEED,
        "classes": CLASSES,
        "feature_table_sha256": feat_hash,
        "split_counts": {
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test)
        },
        "block_costs": block_costs,
        "block_weights": block_weights,
        "default_tau": 0.90,
        "default_lambda": 0.01,
        "first_block_distribution": first_choice_counts.to_dict()
    }
    with open(results_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Export serve_bundle/
    serve_dir = PROJECT_ROOT / "serve_bundle"
    serve_crops_dir = serve_dir / "demo_crops"
    serve_crops_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save experts
    joblib.dump(experts, serve_dir / "experts.joblib")

    # 2. Save simulation pool
    np.savez_compressed(
        serve_dir / "simulation_pool.npz",
        pool_cls=pool_cls,
        pool_counts=pool_counts,
        **{f"lr_{b}": pool_lr[b] for b in BLOCK_EXTRACTORS}
    )

    # 3. Save metadata
    meta = {
        "classes": CLASSES,
        "log_prior": log_prior.tolist(),
        "prior": prior.tolist(),
        "block_costs": block_costs,
        "block_weights": block_weights,
        "block_feature_names": block_columns,
        "block_descriptions": {
            "shape": "Morphological descriptors (area, perimeter, circularity, eccentricity, aspect ratio, solidity)",
            "intensity": "Greyscale statistics, percentiles (5, 25, 50, 75, 95), and distribution skewness",
            "colour": "Per-channel RGB mean/std, channel ratio fractions, and HSV colour space statistics",
            "texture": "GLCM co-occurrence matrix (32 levels, d=[1,3], 4 angles) properties and uniform LBP histogram",
            "frequency": "2D Fourier transform radial power spectrum in 6 concentric frequency bands + high/low ratio"
        },
        "fixed_order": fixed_best_order,
        "cheapest_order": cheapest_order,
        "block_report": df_block_report.to_dict(orient="records"),
        "default_tau": 0.90,
        "default_lambda": 0.01,
        "all_blocks_cost": engine.all_blocks_cost
    }
    with open(serve_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 4. Select ~60 demo test crops (10 per class strictly from test split)
    print("\nSelecting 60 demo crops strictly from TEST split...")
    demo_records = []
    demo_id = 0
    for c_name in CLASSES:
        c_test_df = df_test[df_test["label"] == c_name]
        n_select = min(10, len(c_test_df))
        sampled_rows = c_test_df.sample(n=n_select, random_state=RANDOM_SEED)
        
        for _, row in sampled_rows.iterrows():
            src_crop_rel = row["crop"]
            src_full = PROJECT_ROOT / src_crop_rel
            dest_filename = f"demo_{demo_id:03d}_{c_name}_{Path(src_crop_rel).name}"
            dest_full = serve_crops_dir / dest_filename
            
            img_data = cv2.imread(str(src_full))
            if img_data is not None:
                cv2.imwrite(str(dest_full), img_data)
                demo_records.append({
                    "demo_id": demo_id,
                    "file": f"demo_crops/{dest_filename}",
                    "true_label": c_name,
                    "source_image": row["source_image"],
                    "w": row["w"],
                    "h": row["h"]
                })
                demo_id += 1

    df_demo = pd.DataFrame(demo_records)
    df_demo.to_csv(serve_dir / "demo_index.csv", index=False)
    print(f"Exported {len(df_demo)} demo crops to {serve_dir / 'demo_index.csv'}")

    print("\nPhase 3 & 5 Model Training & Evaluation successfully finished!")
    return df_block_report, df_comp, first_choice_counts

if __name__ == "__main__":
    run_training_pipeline()
