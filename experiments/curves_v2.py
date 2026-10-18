"""experiments/curves_v2.py - Accuracy-cost curves for all acquisition orders (tau sweep) and
learned-stopping feature importances, seeds 0-4. Output: results/v2_analysis/policy_curves.csv, stop_importance.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from src.dataset import split_dataset_by_group, CLASSES
from src.adaptive_v2 import BLOCKS, Z_COLS, block_columns, Trajectories, summarize
from run_v2 import setup, load_costs, TAU_GRID, train_stopper
from analysis_v2 import recost

OUT = ROOT / "results" / "v2_analysis"
NAMES = [f"p_{c}" for c in CLASSES] + ["max_post", "entropy", "margin", "n_acquired", "cum_cost", "n_remaining"] + Z_COLS


def main():
    feats = pd.read_csv(ROOT / "data/features_cache_v2.csv")
    df = feats.merge(pd.read_csv(ROOT / "data/context_prior_cache.csv")[["crop"] + Z_COLS], on="crop")
    cols = block_columns(feats)
    costs, seg_ms, z_ms = load_costs(ROOT / "data/block_costs_ms_v2.json")
    rows, imp = [], []
    for seed in range(5):
        tr, va, te = split_dataset_by_group(df, seed=seed)
        S = setup(df, CLASSES, seed, cols, costs, seg_ms, z_ms, tr, va, [te])
        y = te.label.map({c: i for i, c in enumerate(CLASSES)}).values
        T = S["tests"][0]; Z = te[Z_COLS].values; FC = S["full_cost"]; env = S["env"]["glob"]
        val_acc = {b: float(np.mean(S["val"][b].argmax(1) == S["y_va"])) for b in BLOCKS}
        best = sorted(BLOCKS, key=lambda b: -val_acc[b]); cheap = sorted(BLOCKS, key=lambda b: env["costs"][b])
        rs = np.random.RandomState(seed); rnd = [list(rs.permutation(BLOCKS)) for _ in range(len(te))]
        pols = {"EIG/Cost": None, "Cheapest-first": lambda i: cheap, "Best-first": lambda i: best, "Random order": lambda i: rnd[i]}
        for name, order in pols.items():
            tt = Trajectories(T, Z, order=order, **env)
            for tau in TAU_GRID:
                rows.append(dict(seed=seed, policy=name, tau=tau, **summarize(tt.results(("fixed", tau)), y, FC)))
        clf = train_stopper(S, seed, Trajectories(S["oof"], S["Z_tr"], **S["env"]["ctx"]))["random_forest"]
        for n, v in zip(NAMES, clf.feature_importances_):
            imp.append(dict(seed=seed, feature=n, importance=float(v)))
        pd.DataFrame(rows).to_csv(OUT / "policy_curves.csv", index=False)
        pd.DataFrame(imp).to_csv(OUT / "stop_importance.csv", index=False)
        print("seed", seed, flush=True)


if __name__ == "__main__":
    main()
