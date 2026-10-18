"""experiments/size_aware_costs_v2.py - Size-dependent extraction costs.

Fits per-block power laws cost = a * pixels^k on results/v2_analysis/cost_vs_size.csv, then, per seed,
charges every particle its size-predicted cost and compares acquisition that selects blocks with the
median costs (as in the main experiments) with size-aware EIG/Cost that selects with the particle's own
predicted costs. Also records the initial EIG and EIG/Cost scores of each block (seed 0).
Output: results/v2_analysis/size_aware.csv, initial_eig.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import softmax
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from src.dataset import split_dataset_by_group, CLASSES
from src.adaptive_v2 import BLOCKS, Z_COLS, block_columns, summarize
from src.math_utils import compute_entropy
from run_v2 import setup, load_costs, TAU_GRID, allblock_acc

OUT = ROOT / "results" / "v2_analysis"


def fit_laws():
    c = pd.read_csv(OUT / "cost_vs_size.csv")
    laws = {}
    for b in BLOCKS + ["segmentation"]:
        k, m = np.polyfit(np.log(c.pixels), np.log(c[b]), 1)
        laws[b] = (np.exp(m), k)
    return laws


def paths(probs, pix, env, laws, select_size_aware):
    """Returns per-particle arrays of predictions, confidences and charged cumulative costs (N, 5)."""
    N = len(pix)
    P = np.zeros((N, 5, 6)); C = np.zeros((N, 5))
    f, eig = env["fusion"], env["eig"]
    for i in range(N):
        ci = {b: laws[b][0] * pix[i] ** laws[b][1] for b in BLOCKS}
        seg = laws["segmentation"][0] * pix[i] ** laws["segmentation"][1]
        ci_charge = dict(ci); ci_charge["shape"] += seg
        sel = ci_charge if select_size_aware else env["costs"]
        row = {b: probs[b][i] for b in BLOCKS}
        acq, rem, cum = [], list(BLOCKS), 0.0
        logit = f.logits(row, acq)
        for k in range(5):
            sc = eig.scores(logit, rem, sel)
            b = max(rem, key=lambda q: sc[q])
            rem.remove(b); acq.append(b); cum += ci_charge[b]
            logit = f.logits(row, acq)
            P[i, k] = softmax(logit); C[i, k] = cum
    return P, C


def at_tau(P, C, tau):
    fire = P.max(-1) >= tau; fire[:, 4] = True
    k = fire.argmax(1); r = np.arange(len(P))
    return P[r, k].argmax(1), C[r, k], k + 1


def main():
    laws = fit_laws()
    feats = pd.read_csv(ROOT / "data/features_cache_v2.csv")
    df = feats.merge(pd.read_csv(ROOT / "data/context_prior_cache.csv")[["crop"] + Z_COLS], on="crop")
    idx = pd.read_csv(ROOT / "crops/crop_index.csv")[["crop", "w", "h"]]
    df = df.merge(idx, on="crop")
    cols = block_columns(feats)
    costs, seg_ms, z_ms = load_costs(ROOT / "data/block_costs_ms_v2.json")
    rows, init = [], []
    for seed in range(5):
        tr, va, te = split_dataset_by_group(df, seed=seed)
        S = setup(df, CLASSES, seed, cols, costs, seg_ms, z_ms, tr, va, [te])
        cm = {c: i for i, c in enumerate(CLASSES)}
        y_va, y_te = S["y_va"], te.label.map(cm).values
        env = S["env"]["glob"]
        pix_va = (va.w * va.h).values.astype(float); pix_te = (te.w * te.h).values.astype(float)
        target = allblock_acc(S, S["val"], y_va, "glob") - 0.01
        for aware in (False, True):
            Pv, Cv = paths(S["val"], pix_va, env, laws, aware)
            Pt, Ct = paths(S["tests"][0], pix_te, env, laws, aware)
            full_te = np.mean([sum(laws[b][0] * p ** laws[b][1] for b in BLOCKS + ["segmentation"]) for p in pix_te])
            best = None
            for tau in TAU_GRID:
                pr, cc, _ = at_tau(Pv, Cv, tau)
                if np.mean(pr == y_va) >= target and (best is None or cc.mean() < best[1]):
                    best = (tau, cc.mean())
            tau = best[0] if best else 0.99
            for lab, t in (("tau085", 0.85), ("tuned", tau)):
                pr, cc, nb = at_tau(Pt, Ct, t)
                rows.append(dict(seed=seed, selection="size-aware" if aware else "median costs", stop=lab, tau=t,
                                 accuracy=float(np.mean(pr == y_te)), mean_cost_ms=float(cc.mean()),
                                 full_cost_ms=float(full_te), saving_pct=float(100 * (1 - cc.mean() / full_te)),
                                 mean_blocks=float(nb.mean())))
            print(seed, aware, rows[-1], flush=True)
        if seed == 0:
            lp = S["lp"]; P0 = softmax(lp)
            sc_c = env["eig"].scores(lp.copy(), list(BLOCKS), env["costs"])
            sc_1 = env["eig"].scores(lp.copy(), list(BLOCKS), {b: 1.0 for b in BLOCKS})
            for b in BLOCKS:
                init.append(dict(block=b, cost_ms=env["costs"][b], eig_nats=sc_1[b], eig_per_ms=sc_c[b],
                                 prior_entropy=compute_entropy(P0)))
            pd.DataFrame(init).to_csv(OUT / "initial_eig.csv", index=False)
        pd.DataFrame(rows).to_csv(OUT / "size_aware.csv", index=False)
    print({b: (round(a, 5), round(k, 3)) for b, (a, k) in laws.items()})


if __name__ == "__main__":
    main()
