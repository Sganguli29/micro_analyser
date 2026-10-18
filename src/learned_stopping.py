r"""
src/learned_stopping.py - Learned stopping policies via simulated acquisition trajectories.

At each acquisition step t, extracts state feature vector:
1. Full posterior probabilities P(y) [K=6 features]
2. Max posterior probability [1 feature]
3. Shannon entropy H(P) [1 feature]
4. Margin p_{(1)} - p_{(2)} [1 feature]
5. Number of blocks acquired [1 feature]
6. Cumulative cost paid so far in ms [1 feature]
7. Number of blocks remaining [1 feature]
8. Context descriptors Z [7 features]
Total: 19 state features.

Labels:
- Label (a): STOP = 1 if final prediction after acquiring all remaining blocks equals current prediction.
- Label (b): STOP = 1 if expected accuracy gain from continuing < \lambda * remaining_cost.

Trains Logistic Regression and Random Forest stopping classifiers, tunes probability threshold on validation,
and reports feature importances.
Outputs: results/learned_stopping_report.csv.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from src.dataset import split_dataset_by_group, CLASSES, CLASS_TO_IDX
from src.math_utils import compute_entropy, compute_max_posterior, compute_margin, fuse_log_opinion_pool, compute_monte_carlo_eig, fit_log_opinion_pool_weights
from src.features import BLOCK_EXTRACTORS
from src.context_prior import Z_FEATURE_NAMES
from src.cost_model import load_block_costs_ms
from src.utils import get_git_commit_hash, PROJECT_ROOT

STATE_FEATURE_NAMES = (
    [f"post_{c}" for c in CLASSES] +
    ["max_post", "entropy", "margin", "n_blocks_acquired", "cumulative_cost_ms", "n_blocks_remaining"] +
    Z_FEATURE_NAMES
)

def simulate_acquisition_trajectories(
    split_df: pd.DataFrame,
    expert_probs_by_block: Dict[str, np.ndarray],
    block_weights: Dict[str, float],
    block_costs_ms: Dict[str, float],
    pool_lr: Dict[str, np.ndarray],
    pool_cls: np.ndarray,
    pool_counts: np.ndarray,
    log_prior: np.ndarray,
    label_option: str = "a",
    lambda_cost: float = 0.005,
    include_step_zero: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    r"""
    Simulates step-by-step acquisition trajectories for particles in split_df.
    
    Returns:
        (X_states, y_stops, trajectory_df)
    """
    all_blocks = list(block_costs_ms.keys())
    state_rows = []
    stop_labels = []
    meta_rows = []
    N = len(split_df)

    for i in range(N):
        row = split_df.iloc[i]
        true_y = CLASS_TO_IDX[row["label"]]
        z_vals = [float(row[col]) for col in Z_FEATURE_NAMES]

        # Final prediction under exhaustive evaluation of all 5 blocks
        sample_probs = {b: expert_probs_by_block[b][i] for b in all_blocks}
        final_post = fuse_log_opinion_pool(log_prior, sample_probs, block_weights)
        final_pred = int(np.argmax(final_post))

        remaining_blocks = list(all_blocks)
        blocks_acquired = []
        cumulative_cost = 0.0

        for step in range(len(all_blocks)):
            # Form current posterior from acquired blocks
            cur_expert_dict = {b: (sample_probs[b] if b in blocks_acquired else None) for b in all_blocks}
            current_P = fuse_log_opinion_pool(log_prior, cur_expert_dict, block_weights)
            current_pred = int(np.argmax(current_P))
            max_p = compute_max_posterior(current_P)
            ent = compute_entropy(current_P)
            mrg = compute_margin(current_P)
            rem_cost = float(sum(block_costs_ms[b] for b in remaining_blocks))

            # Assemble state vector
            state_vec = list(current_P) + [
                max_p, ent, mrg, float(len(blocks_acquired)), cumulative_cost, float(len(remaining_blocks))
            ] + z_vals

            # Label definition
            if label_option == "a":
                stop = 1 if (current_pred == final_pred) else 0
            else:
                gain = float((final_pred == true_y) - (current_pred == true_y))
                stop = 1 if (gain < lambda_cost * rem_cost) else 0

            # Record decision step
            if include_step_zero or len(blocks_acquired) >= 1:
                state_rows.append(state_vec)
                stop_labels.append(stop)
                meta_rows.append({
                    "sample_idx": i,
                    "crop": row["crop"],
                    "step": step,
                    "n_acquired": len(blocks_acquired),
                    "current_pred": current_pred,
                    "final_pred": final_pred,
                    "true_y": true_y,
                    "stop_label": stop
                })

            if len(remaining_blocks) == 0:
                break

            # Choose next block via EIG / Cost
            scores, _ = compute_monte_carlo_eig(
                log_post=np.log(current_P + 1e-9),
                candidate_blocks=remaining_blocks,
                block_costs=block_costs_ms,
                pool_lr=pool_lr,
                pool_cls=pool_cls,
                pool_counts=pool_counts
            )
            chosen_b = max(remaining_blocks, key=lambda b: scores[b])
            blocks_acquired.append(chosen_b)
            remaining_blocks.remove(chosen_b)
            cumulative_cost += block_costs_ms[chosen_b]

    X_states = np.array(state_rows, dtype=np.float32)
    y_stops = np.array(stop_labels, dtype=np.int64)
    trajectory_df = pd.DataFrame(meta_rows)
    return X_states, y_stops, trajectory_df

def train_learned_stopping_policy(
    label_option: str = "a",
    seed: int = 42,
    output_csv: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains Logistic Regression and Random Forest learned stopping classifiers,
    tunes probability threshold on validation, and reports feature importances.
    """
    if output_csv is None:
        output_csv = PROJECT_ROOT / "results" / "learned_stopping_report.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    block_costs_ms = load_block_costs_ms()
    feat_cache = PROJECT_ROOT / "data" / "features_cache.csv"
    z_cache = PROJECT_ROOT / "data" / "context_prior_cache.csv"

    assert feat_cache.exists() and z_cache.exists(), "Feature caches missing. Run earlier modules."
    df_feat = pd.read_csv(feat_cache)
    df_z = pd.read_csv(z_cache)

    # Merge on crop to guarantee matching alignment
    df_merged = df_feat.merge(df_z[["crop"] + Z_FEATURE_NAMES], on="crop")
    df_train, df_val, df_test = split_dataset_by_group(df_merged, seed=seed)

    y_train = df_train["label"].map(CLASS_TO_IDX).values
    y_val = df_val["label"].map(CLASS_TO_IDX).values
    y_test = df_test["label"].map(CLASS_TO_IDX).values
    K = len(CLASSES)

    # Prior distribution from training set
    prior_p = np.bincount(y_train, minlength=K).astype(float) / len(y_train)
    log_prior = np.log(prior_p + 1e-9)

    # Extract block column names
    sample_rgb = np.ones((32, 32, 3), dtype=np.float32)
    sample_mask = np.ones((32, 32), dtype=bool)
    block_columns = {b: [c for c in fn(sample_rgb, sample_mask).keys() if c in df_feat.columns] for b, fn in BLOCK_EXTRACTORS.items()}

    print("Fitting block expert classifiers on train split...")
    train_expert_probs = {}
    val_expert_probs = {}
    test_expert_probs = {}
    trained_rf_models = {}

    for b, cols in block_columns.items():
        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=4, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
        # Out-of-fold predictions on train to prevent artificial overconfidence
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        train_expert_probs[b] = cross_val_predict(rf, df_train[cols].values, y_train, cv=cv, method="predict_proba", n_jobs=-1)
        
        # Fit on full train for val and test inference
        rf.fit(df_train[cols].values, y_train)
        val_expert_probs[b] = rf.predict_proba(df_val[cols].values)
        test_expert_probs[b] = rf.predict_proba(df_test[cols].values)
        trained_rf_models[b] = rf

    # Fit log-opinion pool weights on validation split
    block_weights = fit_log_opinion_pool_weights(val_expert_probs, y_val, log_prior)

    # Prepare Monte Carlo simulation pool from train set
    pool_indices = []
    for c_idx in range(K):
        c_train_idx = np.where(y_train == c_idx)[0]
        sampled = np.random.RandomState(seed).choice(c_train_idx, size=min(50, len(c_train_idx)), replace=False)
        pool_indices.extend(sampled)
    pool_indices = np.array(pool_indices)
    pool_cls = y_train[pool_indices]
    pool_counts = np.bincount(pool_cls, minlength=K).astype(float)
    pool_lr = {}
    for b in BLOCK_EXTRACTORS:
        p_pool = np.clip(train_expert_probs[b][pool_indices], 1e-6, 1.0)
        p_pool /= p_pool.sum(axis=1, keepdims=True)
        pool_lr[b] = block_weights[b] * (np.log(p_pool) - log_prior[None, :])

    print(f"Simulating trajectories for learned stopping under Label ({label_option})...")
    X_train_s, y_train_s, _ = simulate_acquisition_trajectories(
        df_train, train_expert_probs, block_weights, block_costs_ms, pool_lr, pool_cls, pool_counts, log_prior,
        label_option=label_option, include_step_zero=True
    )
    X_val_s, y_val_s, _ = simulate_acquisition_trajectories(
        df_val, val_expert_probs, block_weights, block_costs_ms, pool_lr, pool_cls, pool_counts, log_prior,
        label_option=label_option, include_step_zero=True
    )
    X_test_s, y_test_s, _ = simulate_acquisition_trajectories(
        df_test, test_expert_probs, block_weights, block_costs_ms, pool_lr, pool_cls, pool_counts, log_prior,
        label_option=label_option, include_step_zero=True
    )

    print(f"Trajectory samples: Train={len(y_train_s)}, Val={len(y_val_s)}, Test={len(y_test_s)} (Base STOP rate: {100*np.mean(y_train_s):.1f}%)")

    # 1. Logistic Regression
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
    ])
    pipe_lr.fit(X_train_s, y_train_s)
    val_probs_lr = pipe_lr.predict_proba(X_val_s)[:, 1]
    test_probs_lr = pipe_lr.predict_proba(X_test_s)[:, 1]

    # 2. Random Forest
    rf_stop = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed, n_jobs=-1)
    rf_stop.fit(X_train_s, y_train_s)
    val_probs_rf = rf_stop.predict_proba(X_val_s)[:, 1]
    test_probs_rf = rf_stop.predict_proba(X_test_s)[:, 1]

    # Tune threshold on validation split
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thresh_lr = 0.5
    best_f1_lr = 0.0
    best_thresh_rf = 0.5
    best_f1_rf = 0.0

    for t in thresholds:
        f1_v_lr = f1_score(y_val_s, (val_probs_lr >= t).astype(int), zero_division=0)
        if f1_v_lr > best_f1_lr:
            best_f1_lr = f1_v_lr
            best_thresh_lr = float(t)

        f1_v_rf = f1_score(y_val_s, (val_probs_rf >= t).astype(int), zero_division=0)
        if f1_v_rf > best_f1_rf:
            best_f1_rf = f1_v_rf
            best_thresh_rf = float(t)

    # Evaluate on Test split
    test_preds_lr = (test_probs_lr >= best_thresh_lr).astype(int)
    test_preds_rf = (test_probs_rf >= best_thresh_rf).astype(int)

    acc_lr = accuracy_score(y_test_s, test_preds_lr)
    f1_lr = f1_score(y_test_s, test_preds_lr)
    auc_lr = roc_auc_score(y_test_s, test_probs_lr)

    acc_rf = accuracy_score(y_test_s, test_preds_rf)
    f1_rf = f1_score(y_test_s, test_preds_rf)
    auc_rf = roc_auc_score(y_test_s, test_probs_rf)

    print("\n--- LEARNED STOPPING CLASSIFIER TEST PERFORMANCE ---")
    print(f"Logistic Regression: Accuracy={acc_lr:.4f}, F1={f1_lr:.4f}, ROC-AUC={auc_lr:.4f} (tuned threshold={best_thresh_lr:.2f})")
    print(f"Random Forest:       Accuracy={acc_rf:.4f}, F1={f1_rf:.4f}, ROC-AUC={auc_rf:.4f} (tuned threshold={best_thresh_rf:.2f})")

    # Feature Importances from Random Forest
    rf_importances = rf_stop.feature_importances_
    sorted_idx = np.argsort(rf_importances)[::-1]
    print("\nTop 8 Feature Importances for Stopping Decisions:")
    for i in range(min(8, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  {STATE_FEATURE_NAMES[idx]}: {rf_importances[idx]:.4f}")

    records = [
        {
            "model": "logistic_regression",
            "label_option": label_option,
            "optimal_threshold": round(best_thresh_lr, 2),
            "test_accuracy": round(acc_lr, 4),
            "test_f1": round(f1_lr, 4),
            "test_auc": round(auc_lr, 4),
            "test_precision": round(float(precision_score(y_test_s, test_preds_lr, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y_test_s, test_preds_lr, zero_division=0)), 4),
            "top_feature": STATE_FEATURE_NAMES[sorted_idx[0]],
            "git_commit": get_git_commit_hash(),
            "timestamp": datetime.datetime.now().isoformat()
        },
        {
            "model": "random_forest",
            "label_option": label_option,
            "optimal_threshold": round(best_thresh_rf, 2),
            "test_accuracy": round(acc_rf, 4),
            "test_f1": round(f1_rf, 4),
            "test_auc": round(auc_rf, 4),
            "test_precision": round(float(precision_score(y_test_s, test_preds_rf, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y_test_s, test_preds_rf, zero_division=0)), 4),
            "top_feature": STATE_FEATURE_NAMES[sorted_idx[0]],
            "git_commit": get_git_commit_hash(),
            "timestamp": datetime.datetime.now().isoformat()
        }
    ]

    report_df = pd.DataFrame(records)
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        # Avoid exact duplicates
        combined_df = pd.concat([existing_df, report_df], ignore_index=True).drop_duplicates(subset=["model", "label_option"], keep="last")
        combined_df.to_csv(output_csv, index=False)
    else:
        report_df.to_csv(output_csv, index=False)

    print(f"Saved learned stopping report to {output_csv}")
    models_dict = {
        "logistic_regression": pipe_lr,
        "random_forest": rf_stop,
        "best_thresh_lr": best_thresh_lr,
        "best_thresh_rf": best_thresh_rf,
        "feature_names": STATE_FEATURE_NAMES,
        "rf_importances": {STATE_FEATURE_NAMES[i]: float(rf_importances[i]) for i in sorted_idx}
    }
    return report_df, models_dict
