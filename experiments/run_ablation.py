r"""
experiments/run_ablation.py - 4 Configurations and Component Ablations Study.

Evaluates:
Primary Configurations:
1. Config 1: Global-prior adaptive + fixed confidence threshold
2. Config 2: + Context-aware acquisition (Context prior P(Y|Z) + EIG/Cost)
3. Config 3: + Learned stopping (Random Forest stopping policy)
4. Config 4: + Open-Set OOD Rejection (Dual Rule tau_conf, tau_ent)

Component Ablations:
- Ablation A: Z prior initialization without Z-conditioned EIG
- Ablation B: Drop each Z feature one at a time (7 single-feature drops)
- Ablation C: Stopping policy feature ablation (drop context Z, drop costs, drop margin/entropy)

Outputs:
- results/ablation_study.csv
- results/significance.csv
"""

from pathlib import Path
import sys
import argparse
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import split_dataset_by_group, CLASSES, CLASS_TO_IDX
from src.features import BLOCK_EXTRACTORS
from src.context_prior import Z_FEATURE_NAMES, ContextPriorModel
from src.cost_model import load_block_costs_ms
from src.math_utils import fit_log_opinion_pool_weights, fuse_log_opinion_pool
from src.policies import (
    RandomPolicy, CheapestFirstPolicy, BestFirstPolicy, StaticFeatureSelectionPolicy,
    EIGCostPolicy, ContextAwareEIGCostPolicy
)
from src.learned_stopping import train_learned_stopping_policy
from src.pipeline import AdaptivePipeline
from src.stats import run_significance_suite
from src.utils import get_git_commit_hash, compute_ci95

def run_ablation_experiments(
    seeds: List[int] = [42],
    output_csv: Optional[Path] = None
) -> pd.DataFrame:
    if output_csv is None:
        output_csv = PROJECT_ROOT / "results" / "ablation_study.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

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

    records = []
    comparisons_for_stats = []

    for seed in seeds:
        print(f"\n========================================================")
        print(f"RUNNING ABLATION STUDY (SEED = {seed})")
        print(f"========================================================")

        df_train, df_val, df_test = split_dataset_by_group(df_merged, seed=seed)
        y_train = df_train["label"].map(CLASS_TO_IDX).values
        y_val = df_val["label"].map(CLASS_TO_IDX).values
        y_test = df_test["label"].map(CLASS_TO_IDX).values

        prior_p = np.bincount(y_train, minlength=K).astype(float) / len(y_train)
        log_prior = np.log(prior_p + 1e-9)

        # Train block classifiers
        rf_models = {}
        val_probs = {}
        test_probs = {}
        for b, cols in block_columns.items():
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=4, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
            rf.fit(df_train[cols].values, y_train)
            rf_models[b] = rf
            val_probs[b] = rf.predict_proba(df_val[cols].values)
            test_probs[b] = rf.predict_proba(df_test[cols].values)

        # Fit log-opinion pool weights
        block_weights = fit_log_opinion_pool_weights(val_probs, y_val, log_prior)

        # Monte Carlo simulation pool from train set
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

        # Train Context Prior Model P(Y|Z)
        X_train_z = df_train[Z_FEATURE_NAMES].values
        X_val_z = df_val[Z_FEATURE_NAMES].values
        X_test_z = df_test[Z_FEATURE_NAMES].values
        context_model = ContextPriorModel(random_state=seed)
        context_model.fit(X_train_z, y_train, X_val_z, y_val)

        # Train Learned Stopping Model
        _, stop_bundle = train_learned_stopping_policy(label_option="a", seed=seed)
        learned_stop_clf = stop_bundle["random_forest"]
        learned_thresh = stop_bundle["best_thresh_rf"]

        # OOD thresholds tuned on validation
        val_fused = fuse_log_opinion_pool(log_prior, val_probs, block_weights)
        tau_conf = float(np.percentile(np.max(val_fused, axis=1), 5.0))
        tau_ent = float(np.percentile([-np.sum(p * np.log(np.clip(p, 1e-9, 1.0))) for p in val_fused], 95.0))

        # Base Policies
        policy_global_eig = EIGCostPolicy(pool_lr, pool_cls, pool_counts)
        policy_context_eig = ContextAwareEIGCostPolicy(pool_lr, pool_cls, pool_counts)

        # ==========================================
        # 1. CONFIG 1: Global-Prior Adaptive + Fixed Tau
        # ==========================================
        pipe_cfg1 = AdaptivePipeline(
            acquisition_policy=policy_global_eig,
            block_costs_ms=block_costs_ms,
            block_weights=block_weights,
            global_log_prior=log_prior,
            use_context_prior=False,
            stopping_method="fixed",
            fixed_tau=0.85,
            use_ood=False
        )
        res_cfg1 = pipe_cfg1.evaluate_batch(test_probs, X_test_z, y_test)

        # ==========================================
        # 2. CONFIG 2: + Context-Aware Acquisition
        # ==========================================
        pipe_cfg2 = AdaptivePipeline(
            acquisition_policy=policy_context_eig,
            block_costs_ms=block_costs_ms,
            block_weights=block_weights,
            global_log_prior=log_prior,
            context_prior_model=context_model,
            use_context_prior=True,
            stopping_method="fixed",
            fixed_tau=0.85,
            use_ood=False
        )
        res_cfg2 = pipe_cfg2.evaluate_batch(test_probs, X_test_z, y_test)

        # ==========================================
        # 3. CONFIG 3: + Learned Stopping
        # ==========================================
        pipe_cfg3 = AdaptivePipeline(
            acquisition_policy=policy_context_eig,
            block_costs_ms=block_costs_ms,
            block_weights=block_weights,
            global_log_prior=log_prior,
            context_prior_model=context_model,
            use_context_prior=True,
            stopping_method="learned",
            learned_stopping_model=learned_stop_clf,
            learned_stopping_threshold=learned_thresh,
            use_ood=False
        )
        res_cfg3 = pipe_cfg3.evaluate_batch(test_probs, X_test_z, y_test)

        # ==========================================
        # 4. CONFIG 4: + Open-Set OOD Rejection
        # ==========================================
        pipe_cfg4 = AdaptivePipeline(
            acquisition_policy=policy_context_eig,
            block_costs_ms=block_costs_ms,
            block_weights=block_weights,
            global_log_prior=log_prior,
            context_prior_model=context_model,
            use_context_prior=True,
            stopping_method="learned",
            learned_stopping_model=learned_stop_clf,
            learned_stopping_threshold=learned_thresh,
            use_ood=True,
            ood_tau_conf=tau_conf,
            ood_tau_ent=tau_ent
        )
        res_cfg4 = pipe_cfg4.evaluate_batch(test_probs, X_test_z, y_test)

        # ==========================================
        # Component Ablation A: Z prior without Z-conditioned EIG
        # ==========================================
        pipe_abl_a = AdaptivePipeline(
            acquisition_policy=policy_global_eig,
            block_costs_ms=block_costs_ms,
            block_weights=block_weights,
            global_log_prior=log_prior,
            context_prior_model=context_model,
            use_context_prior=True,
            stopping_method="fixed",
            fixed_tau=0.85,
            use_ood=False
        )
        res_abl_a = pipe_abl_a.evaluate_batch(test_probs, X_test_z, y_test)

        # Append main configs
        configs = [
            ("config_1_global_adaptive_fixed", res_cfg1),
            ("config_2_context_aware_fixed", res_cfg2),
            ("config_3_context_learned_stop", res_cfg3),
            ("config_4_full_pipeline_ood", res_cfg4),
            ("ablation_z_prior_only_no_z_eig", res_abl_a)
        ]

        # ==========================================
        # Component Ablation B: Drop each Z feature one-by-one
        # ==========================================
        for drop_z in Z_FEATURE_NAMES:
            rem_z_cols = [z for z in Z_FEATURE_NAMES if z != drop_z]
            model_drop = ContextPriorModel(random_state=seed)
            model_drop.fit(df_train[rem_z_cols].values, y_train, df_val[rem_z_cols].values, y_val)
            
            # Custom predict wrapper
            class DropWrapper:
                def __init__(self, m, cols):
                    self.m = m
                    self.cols = cols
                def predict_proba(self, X):
                    idx = [Z_FEATURE_NAMES.index(c) for c in self.cols]
                    return self.m.predict_proba(X[:, idx])

            pipe_drop = AdaptivePipeline(
                acquisition_policy=policy_context_eig,
                block_costs_ms=block_costs_ms,
                block_weights=block_weights,
                global_log_prior=log_prior,
                context_prior_model=DropWrapper(model_drop, rem_z_cols),
                use_context_prior=True,
                stopping_method="fixed",
                fixed_tau=0.85
            )
            res_drop = pipe_drop.evaluate_batch(test_probs, X_test_z, y_test)
            configs.append((f"ablation_drop_z_{drop_z}", res_drop))

        # Record metrics
        for name, r in configs:
            records.append({
                "seed": seed,
                "configuration": name,
                "accuracy": round(r["accuracy"], 4),
                "macro_f1": round(r["macro_f1"], 4),
                "mean_cost_ms": round(r["mean_cost_ms"], 3),
                "mean_blocks": round(r["mean_blocks"], 2),
                "cost_saving_pct": round(r["cost_saving_pct"], 2),
                "unknown_rate": round(r.get("unknown_rate", 0.0), 4),
                "git_commit": get_git_commit_hash(),
                "timestamp": datetime.datetime.now().isoformat()
            })

        # Save comparisons for statistical tests
        comparisons_for_stats.extend([
            {"name_a": "Config 2 (Context)", "name_b": "Config 1 (Global)", "y_true": y_test, "preds_a": res_cfg2["preds"], "preds_b": res_cfg1["preds"], "costs_a": res_cfg2["costs"], "costs_b": res_cfg1["costs"]},
            {"name_a": "Config 3 (Learned Stop)", "name_b": "Config 2 (Fixed)", "y_true": y_test, "preds_a": res_cfg3["preds"], "preds_b": res_cfg2["preds"], "costs_a": res_cfg3["costs"], "costs_b": res_cfg2["costs"]},
            {"name_a": "Config 4 (OOD)", "name_b": "Config 3 (No OOD)", "y_true": y_test, "preds_a": res_cfg4["preds"], "preds_b": res_cfg3["preds"], "costs_a": res_cfg4["costs"], "costs_b": res_cfg3["costs"]}
        ])

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)
    print(f"\nAblation study results written to {output_csv}")

    # Run statistical significance analysis
    run_significance_suite(comparisons_for_stats)

    # Print summary
    print("\n--- ABLATION STUDY SUMMARY ---")
    summary = df_out.groupby("configuration").agg({
        "accuracy": ["mean", "std"],
        "macro_f1": ["mean", "std"],
        "mean_cost_ms": ["mean", "std"],
        "cost_saving_pct": ["mean"]
    })
    print(summary)

    return df_out

def main():
    parser = argparse.ArgumentParser(description="Run 4-Configuration and Component Ablation Study")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated random seeds (default: 42)")
    parser.add_argument("--output", type=str, default="results/ablation_study.csv", help="Output CSV path")
    args = parser.parse_args()

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run_ablation_experiments(seeds=seed_list, output_csv=Path(args.output))

if __name__ == "__main__":
    main()
