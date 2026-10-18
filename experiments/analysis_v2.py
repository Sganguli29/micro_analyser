r"""
experiments/analysis_v2.py - Additional analyses for the journal version (uses the v2 pipeline).

Per seed (default 0-4):
  * acquisition-order baselines under measured costs (EIG/Cost, cost-unaware EIG, cheapest-first,
    best-first, random) at tau = 0.85 and at a validation-tuned tau
  * per-particle records (prediction, blocks acquired, acquisition order, cost) for configs 1T and 3
  * calibration of block experts (isotonic vs. uncalibrated forest) and of the fused posterior
  * sensitivity of the tuned operating point to the tolerance delta
  * per-step open-set thresholds
Seed 0 only: LOPO ROC curves (all-block posterior) and example acquisition trajectories.
Outputs go to results/v2_analysis/.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import softmax, logsumexp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from src.dataset import split_dataset_by_group, CLASSES
from src.adaptive_v2 import (BLOCKS, Z_COLS, block_columns, Trajectories, summarize, tune_threshold,
                             fit_step_thresholds, Fusion, EIG)
from src.math_utils import compute_entropy
from run_v2 import setup, allblock_acc, train_stopper, load_costs, TAU_GRID, THR_GRID

OUT = ROOT / "results" / "v2_analysis"


def ece15(P, y, bins=15):
    conf = P.max(1); pred = P.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0; rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
            rows.append((lo, hi, int(m.sum()), float(conf[m].mean()), float(acc[m].mean())))
    return e, rows


def nll(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def brier(P, y):
    Y = np.eye(P.shape[1])[y]
    return float(np.mean(np.sum((P - Y) ** 2, 1)))


def recost(traj, costs_charge, seg_for_shape, upfront):
    """Recompute cumulative costs along each particle's order with the charged (measured) costs."""
    for i, order in enumerate(traj.order):
        cum = upfront
        for k, b in enumerate(order):
            cum += costs_charge[b] + (seg_for_shape if b == "shape" else 0.0)
            traj.cost[i, k] = cum
    return traj


def per_seed(df, seed, cols, costs, seg_ms, z_ms):
    tr, va, te = split_dataset_by_group(df, seed=seed)
    S = setup(df, CLASSES, seed, cols, costs, seg_ms, z_ms, tr, va, [te])
    cmap = {c: i for i, c in enumerate(CLASSES)}
    y_te = te.label.map(cmap).values; Z_te = te[Z_COLS].values; y_va = S["y_va"]
    T = S["tests"][0]; FC = S["full_cost"]
    target = allblock_acc(S, S["val"], y_va, "glob") - 0.01
    envg = S["env"]["glob"]
    rows = []

    # ---- acquisition-order baselines (class prior)
    val_acc = {b: float(np.mean(S["val"][b].argmax(1) == y_va)) for b in BLOCKS}
    best_first = sorted(BLOCKS, key=lambda b: -val_acc[b])
    cheapest = sorted(BLOCKS, key=lambda b: envg["costs"][b])
    rs = np.random.RandomState(seed)
    rnd_te = [list(rs.permutation(BLOCKS)) for _ in range(len(te))]
    rnd_va = [list(rs.permutation(BLOCKS)) for _ in range(len(va))]
    uni = {b: 1.0 for b in BLOCKS}
    policies = {
        "eig_cost": (None, None, None),
        "eig_uniform_cost": (None, None, uni),
        "cheapest_first": (lambda i: cheapest, lambda i: cheapest, None),
        "best_first": (lambda i: best_first, lambda i: best_first, None),
        "random_order": (lambda i: rnd_te[i], lambda i: rnd_va[i], None),
    }
    for name, (ord_te, ord_va, sel_costs) in policies.items():
        env = dict(envg)
        if sel_costs is not None:
            env["costs"] = sel_costs
        tt = Trajectories(T, Z_te, order=ord_te, **env)
        tv = Trajectories(S["val"], S["Z_va"], order=ord_va, **env)
        if sel_costs is not None:
            recost(tt, costs, seg_ms, 0.0); recost(tv, costs, seg_ms, 0.0)
        tau, _ = tune_threshold(lambda g: ("fixed", g), TAU_GRID, tv, y_va, target, FC)
        for label, stop in (("tau085", ("fixed", 0.85)), ("tuned", ("fixed", tau))):
            s = summarize(tt.results(stop), y_te, FC)
            rows.append(dict(seed=seed, policy=name, stop=label, tau=stop[1], **s))
        if name == "eig_cost":
            tt_eig = tt
    first_counts = pd.Series([o[0] for o in tt_eig.order]).value_counts().to_dict()
    pos = np.zeros((5, 5))
    for o in tt_eig.order:
        for k, b in enumerate(o):
            pos[k, BLOCKS.index(b)] += 1

    # ---- configs 1T and 3: per-particle records
    tv_g = Trajectories(S["val"], S["Z_va"], **envg)
    tau1, _ = tune_threshold(lambda g: ("fixed", g), TAU_GRID, tv_g, y_va, target, FC)
    envc = S["env"]["ctx"]
    tr_traj = Trajectories(S["oof"], S["Z_tr"], **envc)
    clf = train_stopper(S, seed, tr_traj)["random_forest"]
    tv_c = Trajectories(S["val"], S["Z_va"], **envc); tt_c = Trajectories(T, Z_te, **envc)
    target_c = allblock_acc(S, S["val"], y_va, "ctx") - 0.01
    thr, _ = tune_threshold(lambda g: ("learned", clf, g), THR_GRID, tv_c, y_va, target_c, FC)
    r1 = tt_eig.results(("fixed", tau1)); r3 = tt_c.results(("learned", clf, thr))
    recs = []
    for i in range(len(te)):
        recs.append(dict(seed=seed, y=int(y_te[i]), pred_1T=r1[i]["pred"], nb_1T=r1[i]["n_blocks"],
                         cost_1T=r1[i]["cost"], pred_3=r3[i]["pred"], nb_3=r3[i]["n_blocks"], cost_3=r3[i]["cost"],
                         order=">".join(tt_eig.order[i]), conf_3=float(r3[i]["P"].max())))

    # ---- delta sensitivity
    sens = []
    for d in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
        tg = allblock_acc(S, S["val"], y_va, "glob") - d
        t1, _ = tune_threshold(lambda g: ("fixed", g), TAU_GRID, tv_g, y_va, tg, FC)
        t3, _ = tune_threshold(lambda g: ("learned", clf, g), THR_GRID, tv_c, y_va, tg, FC)
        for nm, tt_, st in (("tuned_tau", tt_eig, ("fixed", t1)), ("learned_rf", tt_c, ("learned", clf, t3))):
            sens.append(dict(seed=seed, delta=d, policy=nm, param=st[-1], **summarize(tt_.results(st), y_te, FC)))

    # ---- calibration
    cal = []
    names = BLOCKS + ["context"]
    for b in BLOCKS:
        rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced_subsample",
                                    random_state=seed, n_jobs=2).fit(tr[cols[b]].fillna(tr[cols[b]].median()).values, S["y_tr"])
        Praw = rf.predict_proba(te[cols[b]].fillna(tr[cols[b]].median()).values)
        for lab, P in (("uncalibrated", Praw), ("isotonic", T[b])):
            cal.append(dict(seed=seed, model=b, calibration=lab, ece=ece15(P, y_te)[0], nll=nll(P, y_te),
                            brier=brier(P, y_te), acc=float(np.mean(P.argmax(1) == y_te))))
    f = envg["fusion"]
    Lall = np.array([f.logits({b: T[b][i] for b in BLOCKS}, BLOCKS) for i in range(len(te))])
    Pall = softmax(Lall, 1)
    P3 = np.array([r["P"] for r in r3])
    rel = {}
    for lab, P in (("fused_all_blocks", Pall), ("fused_at_stop_cfg3", P3)):
        e, bins = ece15(P, y_te)
        cal.append(dict(seed=seed, model=lab, calibration="pool", ece=e, nll=nll(P, y_te), brier=brier(P, y_te),
                        acc=float(np.mean(P.argmax(1) == y_te))))
        rel[lab] = (P.max(1), (P.argmax(1) == y_te).astype(int))
    rel["texture_isotonic"] = (T["texture"].max(1), (T["texture"].argmax(1) == y_te).astype(int))
    rel["shape_isotonic"] = (T["shape"].max(1), (T["shape"].argmax(1) == y_te).astype(int))

    # ---- per-step thresholds at the operating point
    th = fit_step_thresholds(tv_c.results(("learned", clf, thr)))
    thr_rows = [dict(seed=seed, n_blocks=k, tau_conf=th[k][0], tau_ent=th[k][1], tau_energy=th[k][2],
                     pooled=bool(np.allclose(th[k], th["pooled"])))
                for k in range(1, 6)]
    thr_rows.append(dict(seed=seed, n_blocks="all-block (v1)", tau_conf=fit_step_thresholds(tv_c.results(("none",)))[5][0],
                         tau_ent=fit_step_thresholds(tv_c.results(("none",)))[5][1]))

    # ---- example trajectories (seed 0)
    ex = None
    if seed == 0:
        nb = np.array([r["n_blocks"] for r in r3]); pr = np.array([r["pred"] for r in r3])
        easy = int(np.where((nb == 1) & (pr == y_te))[0][0])
        hard = int(np.where((nb >= 4) & (pr == y_te))[0][0])
        ex = {}
        for tag, i in (("easy", easy), ("hard", hard)):
            ex[tag] = dict(crop=te.crop.iloc[i], label=te.label.iloc[i], order=tt_c.order[i],
                           P=tt_c.P[i].tolist(), cost=tt_c.cost[i].tolist(), stop_at=int(nb[i]),
                           stop_prob=tt_c.stop_probs(clf)[i].tolist(), threshold=thr)
    return rows, recs, sens, cal, thr_rows, rel, ex, first_counts, pos


def lopo_roc(df, seed, cols, costs, seg_ms, z_ms):
    out = {}
    for held in CLASSES:
        known = [c for c in CLASSES if c != held]
        dk = df[df.label != held].reset_index(drop=True); dood = df[df.label == held].reset_index(drop=True)
        tr, va, te = split_dataset_by_group(dk, seed=seed)
        S = setup(df, known, seed, cols, costs, seg_ms, z_ms, tr, va, [te, dood])
        f = S["env"]["ctx"]["fusion"]; names = BLOCKS + ["context"]
        L = np.vstack([np.array([f.logits({b: S["tests"][j][b][i] for b in names}, names)
                                 for i in range(len(x))]) for j, x in ((0, te), (1, dood))])
        P = softmax(L, 1)
        y = np.r_[np.zeros(len(te)), np.ones(len(dood))]
        ent = np.array([compute_entropy(p) for p in P])
        sc = {"energy": -logsumexp(L, 1), "dual_rule": (1 - P.max(1)) + ent / np.log(len(known)), "msp": 1 - P.max(1)}
        out[held] = {m: dict(zip(("fpr", "tpr"), [a.tolist() for a in roc_curve(y, s)[:2]]), auc=roc_auc_score(y, s))
                     for m, s in sc.items()}
        print("roc", held, {m: round(v["auc"], 3) for m, v in out[held].items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(ROOT / "data/features_cache_v2.csv")
    df = feats.merge(pd.read_csv(ROOT / "data/context_prior_cache.csv")[["crop"] + Z_COLS], on="crop")
    cols = block_columns(feats)
    costs, seg_ms, z_ms = load_costs(ROOT / "data/block_costs_ms_v2.json")
    A = {k: [] for k in ("pol", "rec", "sens", "cal", "thr")}
    rels, firsts, poss = {}, [], []
    for sd in [int(s) for s in a.seeds.split(",")]:
        pol, rec, sens, cal, thr, rel, ex, fc, pos = per_seed(df, sd, cols, costs, seg_ms, z_ms)
        A["pol"] += pol; A["rec"] += rec; A["sens"] += sens; A["cal"] += cal; A["thr"] += thr
        for k, (c, ok) in rel.items():
            rels.setdefault(k, [[], []]); rels[k][0] += list(c); rels[k][1] += list(ok)
        firsts.append(dict(seed=sd, **fc)); poss.append(pos)
        if ex:
            json.dump(ex, open(OUT / "example_trajectories.json", "w"), indent=1)
        for k, fn in (("pol", "policies.csv"), ("rec", "per_particle.csv"), ("sens", "delta_sensitivity.csv"),
                      ("cal", "calibration.csv"), ("thr", "step_thresholds.csv")):
            pd.DataFrame(A[k]).to_csv(OUT / fn, index=False)
        print("seed", sd, "done", flush=True)
    np.savez_compressed(OUT / "reliability.npz", **{f"{k}__conf": np.array(v[0]) for k, v in rels.items()},
                        **{f"{k}__ok": np.array(v[1]) for k, v in rels.items()})
    pd.DataFrame(firsts).to_csv(OUT / "first_block_counts.csv", index=False)
    np.save(OUT / "order_position_counts.npy", np.sum(poss, 0))
    json.dump(lopo_roc(df, 0, cols, costs, seg_ms, z_ms), open(OUT / "lopo_roc_seed0.json", "w"))


if __name__ == "__main__":
    main()
