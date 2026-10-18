r"""
experiments/run_diagnostics.py - Experimental diagnostics, confusion matrices, and error inspections.

Produces:
1. Confusion matrices per configuration (figures/confusion_<config>.png & .pdf at 300 dpi).
2. Histogram of stopping steps across configurations (figures/stopping_step_histogram.png & .pdf).
3. First-block choice heatmap per polymer (figures/first_block_heatmap.png & .pdf).
4. Per-class performance error table (results/per_class_error_table.csv).
5. 20 worst misclassified particles with annotated visual crops and trajectory logs
   (saved to results/diagnostics/worst_particles/ and results/diagnostics/worst_misclassifications.csv).
"""

from pathlib import Path
import sys
import argparse
import datetime
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import split_dataset_by_group, CLASSES, CLASS_TO_IDX
from src.features import BLOCK_EXTRACTORS
from src.context_prior import Z_FEATURE_NAMES, ContextPriorModel
from src.cost_model import load_block_costs_ms
from src.math_utils import fit_log_opinion_pool_weights, fuse_log_opinion_pool
from src.policies import ContextAwareEIGCostPolicy, EIGCostPolicy
from src.learned_stopping import train_learned_stopping_policy
from src.pipeline import AdaptivePipeline
from src.diagnostics import (
    plot_confusion_matrices, plot_stopping_histogram,
    plot_first_block_heatmap, save_worst_misclassifications
)
from src.utils import get_git_commit_hash

def run_diagnostics_suite(
    seed: int = 42
):
    print(f"\n========================================================")
    print(f"RUNNING EXPERIMENTAL DIAGNOSTICS & ERROR AUDIT (SEED = {seed})")
    print(f"========================================================")

    block_costs_ms = load_block_costs_ms()
    feat_cache = PROJECT_ROOT / "data" / "features_cache.csv"
    z_cache = PROJECT_ROOT / "data" / "context_prior_cache.csv"
    df_feat = pd.read_csv(feat_cache)
    df_z = pd.read_csv(z_cache)
    df_merged = df_feat.merge(df_z[["crop"] + Z_FEATURE_NAMES], on="crop")

    sample_rgb = np.ones((32, 32, 3), dtype=np.float32)
    sample_mask = np.ones((32, 32), dtype=bool)
    block_columns = {b: [c for c in fn(sample_rgb, sample_mask).keys() if c in df_feat.columns] for b, fn in BLOCK_EXTRACTORS.items()}
    all_blocks = list(block_costs_ms.keys())
    K = len(CLASSES)

    df_train, df_val, df_test = split_dataset_by_group(df_merged, seed=seed)
    y_train = df_train["label"].map(CLASS_TO_IDX).values
    y_val = df_val["label"].map(CLASS_TO_IDX).values
    y_test = df_test["label"].map(CLASS_TO_IDX).values

    prior_p = np.bincount(y_train, minlength=K).astype(float) / len(y_train)
    log_prior = np.log(prior_p + 1e-9)

    # Train expert models
    rf_models = {}
    val_probs = {}
    test_probs = {}
    for b, cols in block_columns.items():
        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=4, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
        rf.fit(df_train[cols].values, y_train)
        rf_models[b] = rf
        val_probs[b] = rf.predict_proba(df_val[cols].values)
        test_probs[b] = rf.predict_proba(df_test[cols].values)

    block_weights = fit_log_opinion_pool_weights(val_probs, y_val, log_prior)

    # Monte Carlo pool
    pool_indices = []
    for c_idx in range(K):
        c_idx_train = np.where(y_train == c_idx)[0]
        sampled = np.random.RandomState(seed).choice(c_idx_train, size=min(50, len(c_idx_train)), replace=False)
        pool_indices.extend(sampled)
    pool_indices = np.array(pool_indices)
    pool_cls = y_train[pool_indices]
    pool_counts = np.bincount(pool_cls, minlength=K).astype(float)
    pool_lr = {}
    for b in BLOCK_EXTRACTORS:
        p_p = np.clip(rf_models[b].predict_proba(df_train.iloc[pool_indices][block_columns[b]].values), 1e-6, 1.0)
        p_p /= p_p.sum(axis=1, keepdims=True)
        pool_lr[b] = block_weights[b] * (np.log(p_p) - log_prior[None, :])

    # Context model
    X_train_z = df_train[Z_FEATURE_NAMES].values
    X_val_z = df_val[Z_FEATURE_NAMES].values
    X_test_z = df_test[Z_FEATURE_NAMES].values
    context_model = ContextPriorModel(random_state=seed)
    context_model.fit(X_train_z, y_train, X_val_z, y_val)

    # Learned stopping model
    _, stop_bundle = train_learned_stopping_policy(label_option="a", seed=seed)
    learned_stop_clf = stop_bundle["random_forest"]
    learned_thresh = stop_bundle["best_thresh_rf"]

    # OOD thresholds
    val_fused = fuse_log_opinion_pool(log_prior, val_probs, block_weights)
    tau_conf = float(np.percentile(np.max(val_fused, axis=1), 5.0))
    tau_ent = float(np.percentile([-np.sum(p * np.log(np.clip(p, 1e-9, 1.0))) for p in val_fused], 95.0))

    # Pipelines
    policy_global = EIGCostPolicy(pool_lr, pool_cls, pool_counts)
    policy_context = ContextAwareEIGCostPolicy(pool_lr, pool_cls, pool_counts)

    pipe_cfg1 = AdaptivePipeline(policy_global, block_costs_ms, block_weights, log_prior, use_context_prior=False, fixed_tau=0.85)
    pipe_cfg2 = AdaptivePipeline(policy_context, block_costs_ms, block_weights, log_prior, context_prior_model=context_model, use_context_prior=True, fixed_tau=0.85)
    pipe_cfg3 = AdaptivePipeline(policy_context, block_costs_ms, block_weights, log_prior, context_prior_model=context_model, use_context_prior=True, stopping_method="learned", learned_stopping_model=learned_stop_clf, learned_stopping_threshold=learned_thresh)
    pipe_cfg4 = AdaptivePipeline(policy_context, block_costs_ms, block_weights, log_prior, context_prior_model=context_model, use_context_prior=True, stopping_method="learned", learned_stopping_model=learned_stop_clf, learned_stopping_threshold=learned_thresh, use_ood=True, ood_tau_conf=tau_conf, ood_tau_ent=tau_ent)

    pipes = {
        "config_1_global_adaptive": pipe_cfg1,
        "config_2_context_aware": pipe_cfg2,
        "config_3_learned_stopping": pipe_cfg3,
        "config_4_full_ood": pipe_cfg4
    }

    eval_results = {}
    cms = {}
    n_blocks_dict = {}

    for name, p in pipes.items():
        res = p.evaluate_batch(test_probs, X_test_z, y_test)
        eval_results[name] = res
        n_blocks_dict[name] = res["n_blocks"]
        valid_idx = (res["preds"] >= 0)
        cm = confusion_matrix(y_test[valid_idx], res["preds"][valid_idx], labels=list(range(K)))
        cms[name] = cm

    # 1. Plot Confusion Matrices
    print("Generating publication confusion matrices...")
    plot_confusion_matrices(cms)

    # 2. Plot Stopping Step Histogram
    print("Generating stopping step histogram...")
    plot_stopping_histogram(n_blocks_dict)

    # 3. Plot First-Block Heatmap (on Config 3: Adaptive Context)
    print("Generating first-block selection heatmap...")
    plot_first_block_heatmap(y_test, eval_results["config_3_learned_stopping"]["first_blocks"], all_blocks)

    # 4. Generate Per-Class Error Table (Config 3)
    c3_res = eval_results["config_3_learned_stopping"]
    c3_preds = c3_res["preds"]
    c3_costs = c3_res["costs"]
    c3_blocks = c3_res["n_blocks"]

    per_class_rows = []
    for c_idx, c_name in enumerate(CLASSES):
        mask = (y_test == c_idx)
        tot = int(np.sum(mask))
        acc = float(np.mean(c3_preds[mask] == c_idx)) if tot > 0 else 0.0
        f1 = float(f1_score(y_test == c_idx, c3_preds == c_idx, average="binary")) if tot > 0 else 0.0
        mb = float(np.mean(c3_blocks[mask])) if tot > 0 else 0.0
        mc = float(np.mean(c3_costs[mask])) if tot > 0 else 0.0
        per_class_rows.append({
            "polymer": c_name,
            "test_samples": tot,
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "mean_blocks": round(mb, 2),
            "mean_cost_ms": round(mc, 3),
            "git_commit": get_git_commit_hash(),
            "timestamp": datetime.datetime.now().isoformat()
        })

    df_per_class = pd.DataFrame(per_class_rows)
    out_table = PROJECT_ROOT / "results" / "per_class_error_table.csv"
    df_per_class.to_csv(out_table, index=False)
    print(f"Per-class error table written to {out_table}")

    # 5. Extract Detailed Trajectories and Save 20 Worst Misclassifications
    print("Extracting individual particle trajectories for error inspection...")
    all_trajectories = []
    all_confidences = []
    
    for i in range(len(y_test)):
        sample_probs = {b: test_probs[b][i] for b in all_blocks}
        single_res = pipe_cfg3.run_particle(
            particle_probs_by_block=sample_probs,
            context_z=X_test_z[i],
            particle_idx=i,
            true_label=int(y_test[i])
        )
        all_trajectories.append(single_res["trajectory"])
        all_confidences.append(single_res["confidence"])

    save_worst_misclassifications(
        df_test=df_test,
        y_true=y_test,
        preds=c3_preds,
        confidences=np.array(all_confidences),
        trajectories=all_trajectories,
        n_worst=20
    )

    print("\nDiagnostics suite completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Run Diagnostics and Publication Figures")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    run_diagnostics_suite(seed=args.seed)

if __name__ == "__main__":
    main()
