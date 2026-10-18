"""experiments/component_ablation_v2.py - Ablation of EIG/Cost pipeline components (class prior, tuned tau).

Variants: main (out-of-fold pool, 50 per class, fitted weights, isotonic experts); in-sample pool predictions;
pool size 10/25/100 per class; unit pool weights (naive Bayes); uncalibrated random-forest experts.
Output: results/v2_analysis/component_ablation.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from src.dataset import split_dataset_by_group, CLASSES
from src.adaptive_v2 import (BLOCKS, Z_COLS, block_columns, fit_experts, Fusion, fit_weights, EIG, Trajectories,
                             summarize, tune_threshold)
from run_v2 import load_costs, TAU_GRID


def run_variant(probs_va, probs_te, pool_probs, y_tr, y_va, y_te, Zva, Zte, lp, w, costs_g, FC, seed, per_class):
    env = dict(fusion=Fusion(lp, w), eig=EIG(pool_probs, y_tr, lp, w, 6, seed, per_class=per_class),
               costs=costs_g, use_context=False, upfront_cost=0.0)
    tv = Trajectories(probs_va, Zva, **env); tt = Trajectories(probs_te, Zte, **env)
    f = env["fusion"]
    acc_full = float(np.mean([f.logits({b: probs_va[b][i] for b in BLOCKS}, BLOCKS).argmax() == y_va[i] for i in range(len(y_va))]))
    tau, _ = tune_threshold(lambda g: ("fixed", g), TAU_GRID, tv, y_va, acc_full - 0.01, FC)
    out = summarize(tt.results(("fixed", tau)), y_te, FC)
    out["tau"] = tau
    out["allblock_acc"] = float(np.mean([f.logits({b: probs_te[b][i] for b in BLOCKS}, BLOCKS).argmax() == y_te[i] for i in range(len(y_te))]))
    return out


def main():
    feats = pd.read_csv(ROOT / "data/features_cache_v2.csv")
    df = feats.merge(pd.read_csv(ROOT / "data/context_prior_cache.csv")[["crop"] + Z_COLS], on="crop")
    cols = block_columns(feats)
    costs, seg_ms, z_ms = load_costs(ROOT / "data/block_costs_ms_v2.json")
    costs_g = dict(costs); costs_g["shape"] += seg_ms
    FC = sum(costs.values()) + seg_ms
    rows = []
    for seed in range(5):
        tr, va, te = split_dataset_by_group(df, seed=seed)
        cm = {c: i for i, c in enumerate(CLASSES)}
        y_tr, y_va, y_te = (d.label.map(cm).values for d in (tr, va, te))
        oof, val, (tst, trn) = fit_experts(tr, va, [te, tr], y_tr, y_va, cols, seed, 6)
        lp = np.log(np.bincount(y_tr, minlength=6) / len(y_tr))
        w = fit_weights(val, y_va, lp, BLOCKS)
        Zva, Zte = va[Z_COLS].values, te[Z_COLS].values
        common = (y_tr, y_va, y_te, Zva, Zte, lp)
        variants = {
            "main (OOF pool, 50/class, fitted w, isotonic)": (val, tst, oof, w, 50),
            "in-sample pool predictions": (val, tst, trn, w, 50),
            "pool 10 per class": (val, tst, oof, w, 10),
            "pool 25 per class": (val, tst, oof, w, 25),
            "pool 100 per class": (val, tst, oof, w, 100),
            "unit weights (naive Bayes)": (val, tst, oof, {b: 1.0 for b in BLOCKS}, 50),
        }
        # uncalibrated experts
        gkf = GroupKFold(n_splits=3)
        raw_oof, raw_val, raw_te = {}, {}, {}
        for b in BLOCKS:
            X = tr[cols[b]].fillna(tr[cols[b]].median()).values
            mk = lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced_subsample",
                                                random_state=seed, n_jobs=2)
            o = np.zeros((len(tr), 6))
            for a, bb in gkf.split(X, y_tr, tr.source_image.values):
                o[bb] = mk().fit(X[a], y_tr[a]).predict_proba(X[bb])
            m = mk().fit(X, y_tr)
            raw_oof[b] = o
            raw_val[b] = m.predict_proba(va[cols[b]].fillna(tr[cols[b]].median()).values)
            raw_te[b] = m.predict_proba(te[cols[b]].fillna(tr[cols[b]].median()).values)
        w_raw = fit_weights(raw_val, y_va, lp, BLOCKS)
        variants["uncalibrated experts"] = (raw_val, raw_te, raw_oof, w_raw, 50)
        for name, (pv, pt, pool, ww, pc) in variants.items():
            r = run_variant(pv, pt, pool, *common, ww, costs_g, FC, seed, pc)
            rows.append(dict(seed=seed, variant=name, **r))
            print(seed, name, round(r["accuracy"], 4), round(r["mean_cost_ms"], 3), flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results/v2_analysis/component_ablation.csv", index=False)


if __name__ == "__main__":
    main()
