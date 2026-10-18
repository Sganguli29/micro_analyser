"""experiments/lopo_pipeline_q_v2.py - In-pipeline LOPO detection vs. false-unknown rate as the per-step
threshold percentile q varies (seeds 0-1). Output: results/v2_analysis/lopo_pipeline_q.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from src.dataset import split_dataset_by_group, CLASSES
from src.adaptive_v2 import Z_COLS, block_columns, Trajectories, tune_threshold, fit_step_thresholds, apply_ood
from run_v2 import setup, load_costs, train_stopper, allblock_acc, THR_GRID, DELTA

feats = pd.read_csv(ROOT / "data/features_cache_v2.csv")
df = feats.merge(pd.read_csv(ROOT / "data/context_prior_cache.csv")[["crop"] + Z_COLS], on="crop")
cols = block_columns(feats); costs, seg, zc = load_costs(ROOT / "data/block_costs_ms_v2.json")
rows = []
for seed in (0, 1):
    for held in CLASSES:
        known = [c for c in CLASSES if c != held]
        dk = df[df.label != held].reset_index(drop=True); dood = df[df.label == held].reset_index(drop=True)
        tr, va, te = split_dataset_by_group(dk, seed=seed)
        S = setup(df, known, seed, cols, costs, seg, zc, tr, va, [te, dood])
        env = S["env"]["ctx"]
        clf = train_stopper(S, seed, Trajectories(S["oof"], S["Z_tr"], **env))["random_forest"]
        tv = Trajectories(S["val"], S["Z_va"], **env)
        thr, _ = tune_threshold(lambda g: ("learned", clf, g), THR_GRID, tv, S["y_va"],
                                allblock_acc(S, S["val"], S["y_va"], "ctx") - DELTA, S["full_cost"])
        rv = tv.results(("learned", clf, thr))
        tk = Trajectories(S["tests"][0], te[Z_COLS].values, **env); to = Trajectories(S["tests"][1], dood[Z_COLS].values, **env)
        variants = {"learned stop": (rv, tk.results(("learned", clf, thr)), to.results(("learned", clf, thr))),
                    "all blocks": (tv.results(("none",)), tk.results(("none",)), to.results(("none",)))}
        for vname, (r_v, r_k, r_o) in variants.items():
            for q in (1, 2.5, 5, 10, 20, 30):
                th = fit_step_thresholds(r_v, q=q)
                for m in ("dual", "energy"):
                    rows.append(dict(seed=seed, held_out=held, acquisition=vname, q=q, method=m,
                                     detect=float(apply_ood(r_o, th, m).mean()), false_unknown=float(apply_ood(r_k, th, m).mean())))
        print(seed, held, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results/v2_analysis/lopo_pipeline_q.csv", index=False)
