r"""
experiments/run_tradeoff.py - Accuracy vs. Mean Cost Pareto Frontier Analysis.

Evaluates:
1. Random Order
2. Cheapest-First
3. Best-First
4. Static Feature Selection
5. Adaptive EIG/Cost (Global Prior)
6. Adaptive EIG/Cost (Context Prior)
7. Learned Stopping Policy (Threshold Sweep)
8. Theoretical Oracle Upper Bound

Generates:
- figures/pareto.png (300 dpi) & figures/pareto.pdf
- results/pareto_frontier.csv
"""

from pathlib import Path
import sys
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import split_dataset_by_group, CLASSES, CLASS_TO_IDX
from src.features import BLOCK_EXTRACTORS
from src.context_prior import Z_FEATURE_NAMES, ContextPriorModel
from src.cost_model import load_block_costs_ms
from src.math_utils import fit_log_opinion_pool_weights
from src.policies import (
    RandomPolicy, CheapestFirstPolicy, BestFirstPolicy, StaticFeatureSelectionPolicy,
    EIGCostPolicy, ContextAwareEIGCostPolicy, OraclePolicy
)
from src.learned_stopping import train_learned_stopping_policy
from src.pipeline import AdaptivePipeline
from src.utils import save_figure, get_git_commit_hash

def run_tradeoff_sweep(
    seed: int = 42,
    output_csv: Optional[Path] = None,
    output_fig: Optional[Path] = None
) -> pd.DataFrame:
    if output_csv is None:
        output_csv = PROJECT_ROOT / "results" / "pareto_frontier.csv"
    if output_fig is None:
        output_fig = PROJECT_ROOT / "figures" / "pareto"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_fig.parent.mkdir(parents=True, exist_ok=True)

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

    # Train expert classifiers
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

    # Fixed threshold sweep values
    tau_vals = [0.40, 0.55, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
    learned_thresh_vals = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    records = []
    plot_series = {}

    # 1. Fixed Policies
    fixed_policies = {
        "random": RandomPolicy(seed=seed),
        "cheapest_first": CheapestFirstPolicy(),
        "best_first": BestFirstPolicy(),
        "static_selection": StaticFeatureSelectionPolicy(),
        "adaptive_global": EIGCostPolicy(pool_lr, pool_cls, pool_counts),
        "adaptive_context": ContextAwareEIGCostPolicy(pool_lr, pool_cls, pool_counts),
    }

    for pol_name, pol_obj in fixed_policies.items():
        print(f"Sweeping tau for policy: {pol_name}...")
        costs = []
        accs = []
        use_ctx = (pol_name == "adaptive_context")
        
        for tau in tau_vals:
            pipe = AdaptivePipeline(
                acquisition_policy=pol_obj,
                block_costs_ms=block_costs_ms,
                block_weights=block_weights,
                global_log_prior=log_prior,
                context_prior_model=context_model if use_ctx else None,
                use_context_prior=use_ctx,
                stopping_method="fixed",
                fixed_tau=tau
            )
            res = pipe.evaluate_batch(test_probs, X_test_z, y_test)
            records.append({
                "policy": pol_name,
                "stopping_type": "fixed_tau",
                "threshold": tau,
                "accuracy": round(res["accuracy"], 4),
                "macro_f1": round(res["macro_f1"], 4),
                "mean_cost_ms": round(res["mean_cost_ms"], 3),
                "mean_blocks": round(res["mean_blocks"], 2),
                "cost_saving_pct": round(res["cost_saving_pct"], 2),
                "seed": seed,
                "git_commit": get_git_commit_hash(),
                "timestamp": datetime.datetime.now().isoformat()
            })
            costs.append(res["mean_cost_ms"])
            accs.append(res["accuracy"])

        plot_series[pol_name] = (costs, accs)

    # 2. Learned Stopping Threshold Sweep
    print("Sweeping threshold for Learned Stopping Policy...")
    l_costs = []
    l_accs = []
    for l_th in learned_thresh_vals:
        pipe_l = AdaptivePipeline(
            acquisition_policy=ContextAwareEIGCostPolicy(pool_lr, pool_cls, pool_counts),
            block_costs_ms=block_costs_ms,
            block_weights=block_weights,
            global_log_prior=log_prior,
            context_prior_model=context_model,
            use_context_prior=True,
            stopping_method="learned",
            learned_stopping_model=learned_stop_clf,
            learned_stopping_threshold=l_th
        )
        res_l = pipe_l.evaluate_batch(test_probs, X_test_z, y_test)
        records.append({
            "policy": "learned_stopping",
            "stopping_type": "learned_prob",
            "threshold": l_th,
            "accuracy": round(res_l["accuracy"], 4),
            "macro_f1": round(res_l["macro_f1"], 4),
            "mean_cost_ms": round(res_l["mean_cost_ms"], 3),
            "mean_blocks": round(res_l["mean_blocks"], 2),
            "cost_saving_pct": round(res_l["cost_saving_pct"], 2),
            "seed": seed,
            "git_commit": get_git_commit_hash(),
            "timestamp": datetime.datetime.now().isoformat()
        })
        l_costs.append(res_l["mean_cost_ms"])
        l_accs.append(res_l["accuracy"])
    plot_series["learned_stopping"] = (l_costs, l_accs)

    # 3. Oracle Upper Bound Policy
    print("Computing Oracle upper bound...")
    oracle_pol = OraclePolicy(test_probs, block_weights, log_prior)
    oracle_costs = []
    oracle_accs = []
    for i in range(len(y_test)):
        sub = oracle_pol.get_optimal_block_subset(i, int(y_test[i]), block_costs_ms)
        c_sub = sum(block_costs_ms[b] for b in sub)
        oracle_costs.append(c_sub)
    mean_oracle_cost = float(np.mean(oracle_costs))
    oracle_acc = 1.0 # By definition of finding the correct label if possible
    records.append({
        "policy": "oracle_upper_bound",
        "stopping_type": "oracle",
        "threshold": 0.0,
        "accuracy": oracle_acc,
        "macro_f1": 1.0,
        "mean_cost_ms": round(mean_oracle_cost, 3),
        "mean_blocks": round(mean_oracle_cost / (sum(block_costs_ms.values()) / 5.0), 2),
        "cost_saving_pct": round((1.0 - mean_oracle_cost / sum(block_costs_ms.values())) * 100.0, 2),
        "seed": seed,
        "git_commit": get_git_commit_hash(),
        "timestamp": datetime.datetime.now().isoformat()
    })
    plot_series["oracle_upper_bound"] = ([mean_oracle_cost], [oracle_acc])

    # Save CSV
    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)
    print(f"Pareto results written to {output_csv}")

    # Plot Pareto Figure (300 DPI)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    style_map = {
        "random": ("Random", "gray", "x", "--"),
        "cheapest_first": ("Cheapest First", "darkorange", "^", "-."),
        "best_first": ("Best First", "brown", "v", "-."),
        "static_selection": ("Static Selection", "purple", "s", ":"),
        "adaptive_global": ("Adaptive (Global Prior)", "steelblue", "D", "-"),
        "adaptive_context": ("Adaptive (Context Prior)", "green", "o", "-"),
        "learned_stopping": ("Ours: Adaptive + Learned Stopping", "crimson", "*", "-"),
        "oracle_upper_bound": ("Oracle (Upper Bound)", "gold", "P", "")
    }

    for pol, (costs, accs) in plot_series.items():
        label, color, marker, ls = style_map.get(pol, (pol, "black", "o", "-"))
        # Sort by cost for smooth curve
        sort_order = np.argsort(costs)
        sc = np.array(costs)[sort_order]
        sa = np.array(accs)[sort_order]
        
        if pol == "oracle_upper_bound":
            ax.scatter(sc, sa, color=color, marker=marker, s=150, zorder=5, label=label, edgecolors="black")
        else:
            ax.plot(sc, sa, color=color, marker=marker, linestyle=ls, linewidth=2, markersize=7, label=label)

    exhaustive_cost = float(sum(block_costs_ms.values()))
    ax.axvline(x=exhaustive_cost, color="gray", linestyle=":", label=f"Exhaustive Cost ({exhaustive_cost:.1f} ms)")
    
    ax.set_xlabel("Mean Wall-Clock Extraction Cost per Particle (ms)", fontsize=11)
    ax.set_ylabel("Classification Accuracy", fontsize=11)
    ax.set_title("Accuracy vs. Feature Extraction Cost Pareto Frontier", fontsize=12, pad=12)
    ax.set_ylim(0.70, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()

    save_figure(fig, output_fig)
    plt.close(fig)
    print(f"Saved Pareto frontier plot to {output_fig}.png and .pdf")

    return df_out

def main():
    parser = argparse.ArgumentParser(description="Generate Accuracy vs. Cost Pareto Curves")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-csv", type=str, default="results/pareto_frontier.csv", help="Output CSV path")
    parser.add_argument("--output-fig", type=str, default="figures/pareto", help="Output figure base path")
    args = parser.parse_args()

    run_tradeoff_sweep(seed=args.seed, output_csv=Path(args.output_csv), output_fig=Path(args.output_fig))

if __name__ == "__main__":
    main()
