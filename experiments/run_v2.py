r"""
experiments/run_v2.py - Corrected multi-seed evaluation of the adaptive pipeline (see src/adaptive_v2.py).

Per seed:
  * closed-set configurations 1-4 (+ tuned-threshold controls and component ablations)
  * accuracy-cost trade-off curves (fixed tau sweep, learned-stop threshold sweep)
  * learned-stopping classifier metrics on held-out on-policy trajectories
  * leave-one-polymer-out (LOPO): full-block detector benchmark and in-pipeline open-set evaluation

Usage:
  python experiments/run_v2.py --features data/features_cache_v2.csv --seeds 0,1,2,3,4 --out results/v2
"""
import argparse, json, sys, time, datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dataset import split_dataset_by_group, CLASSES
from src.adaptive_v2 import (BLOCKS, Z_COLS, block_columns, fit_experts, Fusion, fit_weights, EIG,
                             Trajectories, summarize, stopping_training_data, fit_stopping_models,
                             tune_threshold, fit_step_thresholds, apply_ood, ood_scores, log_ratio,
                             ContextExpert)
from src.math_utils import compute_entropy
from src.utils import get_git_commit_hash

TAU_GRID = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
THR_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
DELTA = 0.01  # tolerated validation accuracy loss vs. all-block fusion


def load_costs(path):
    c = json.load(open(path))
    return c["block_costs_ms"], c["segmentation_ms"], c["context_z_ms"]


def setup(df, classes, seed, cols, costs, seg_ms, z_ms, tr=None, va=None, te_list=None):
    K = len(classes)
    cmap = {c: i for i, c in enumerate(classes)}
    y_tr = tr.label.map(cmap).values; y_va = va.label.map(cmap).values
    oof, val, tests = fit_experts(tr, va, te_list, y_tr, y_va, cols, seed, K)
    prior = np.bincount(y_tr, minlength=K) / len(y_tr)
    lp = np.log(prior + 1e-9)
    w_glob = fit_weights(val, y_va, lp, BLOCKS)
    w_ctx = fit_weights(val, y_va, lp, BLOCKS + ["context"])
    costs_glob = dict(costs); costs_glob["shape"] = costs["shape"] + seg_ms   # mask paid when Shape acquired
    full_cost = sum(costs.values()) + seg_ms
    env = {
        "glob": dict(fusion=Fusion(lp, w_glob), eig=EIG(oof, y_tr, lp, w_glob, K, seed), costs=costs_glob,
                     use_context=False, upfront_cost=0.0),
        "ctx": dict(fusion=Fusion(lp, w_ctx), eig=EIG(oof, y_tr, lp, w_ctx, K, seed), costs=costs,
                    use_context=True, upfront_cost=seg_ms + z_ms),
    }
    return dict(K=K, y_tr=y_tr, y_va=y_va, oof=oof, val=val, tests=tests, lp=lp, w_glob=w_glob,
                w_ctx=w_ctx, env=env, full_cost=full_cost, Z_tr=tr[Z_COLS].values, Z_va=va[Z_COLS].values)


def allblock_acc(S, probs, y, key):
    f = S["env"][key]["fusion"]
    names = BLOCKS + (["context"] if key == "ctx" else [])
    lg = np.array([f.logits({b: probs[b][i] for b in names}, names) for i in range(len(y))])
    return float(np.mean(lg.argmax(1) == y))


def train_stopper(S, seed, traj_tr):
    X, L = stopping_training_data(traj_tr)
    return fit_stopping_models(X, L, seed)


# ------------------------------------------------------------- OOD detectors (full-block benchmark)
class OpenMaxFixed:
    """OpenMax-style EVT rejection on fused log-evidence vectors.
    MAV per class from correctly classified validation particles; Weibull fitted to the largest
    distances to the MAV; unknown score = Weibull CDF of the distance to the *predicted* class's MAV.
    (v1 averaged CDFs over all classes in probability space, which inverted the score.)"""
    def __init__(self, tail=20):
        self.tail = tail

    def fit(self, AV, y, pred):
        self.mav, self.wb = {}, {}
        for k in np.unique(y):
            m = (y == k) & (pred == k)
            if m.sum() < 5:
                m = y == k
            mav = AV[m].mean(0)
            d = np.sort(np.linalg.norm(AV[m] - mav, axis=1))[-self.tail:]
            self.mav[k] = mav
            self.wb[k] = stats.weibull_min.fit(d, floc=0)
        return self

    def score(self, AV, pred):
        out = []
        for a, p in zip(AV, pred):
            c, loc, sc = self.wb[p]
            out.append(stats.weibull_min.cdf(np.linalg.norm(a - self.mav[p]), c, loc=loc, scale=sc))
        return np.array(out)


class Mahalanobis:
    def fit(self, AV, y):
        self.mu = {k: AV[y == k].mean(0) for k in np.unique(y)}
        D = np.vstack([AV[y == k] - self.mu[k] for k in self.mu])
        self.icov = np.linalg.pinv(np.cov(D, rowvar=False) + 1e-4 * np.eye(AV.shape[1]))
        return self

    def score(self, AV):
        return np.array([min((a - m) @ self.icov @ (a - m) for m in self.mu.values()) for a in AV])


def fpr_at_95(y_ood, s):
    tau = np.percentile(s[y_ood == 1], 5.0)      # 95% of unknowns flagged
    return float(np.mean(s[y_ood == 0] >= tau))


def lopo_fold(df, held, seed, cols, costs, seg_ms, z_ms, do_pipeline):
    known = [c for c in CLASSES if c != held]
    dk = df[df.label != held].reset_index(drop=True)
    dood = df[df.label == held].reset_index(drop=True)
    tr, va, te = split_dataset_by_group(dk, seed=seed)
    assert held not in set(tr.label) and held not in set(va.label)
    S = setup(df, known, seed, cols, costs, seg_ms, z_ms, tr, va, [te, dood])
    cmap = {c: i for i, c in enumerate(known)}
    y_te = te.label.map(cmap).values
    names = BLOCKS + ["context"]
    f = S["env"]["ctx"]["fusion"]

    def full(probs, n):
        return np.array([f.logits({b: probs[b][i] for b in names}, names) for i in range(n)])
    LV = full(S["val"], len(va)); LK = full(S["tests"][0], len(te)); LO = full(S["tests"][1], len(dood))
    PV, PK, PO = softmax(LV, 1), softmax(LK, 1), softmax(LO, 1)
    yood = np.r_[np.zeros(len(te)), np.ones(len(dood))].astype(int)
    L = np.vstack([LK, LO]); P = np.vstack([PK, PO]); pred = P.argmax(1)
    ent = lambda Q: np.array([compute_entropy(q) for q in Q])
    from scipy.special import logsumexp
    om = OpenMaxFixed().fit(LV, S["y_va"], PV.argmax(1))
    mh = Mahalanobis().fit(LV, S["y_va"])
    Kk = len(known)
    sc = {
        "msp": (1 - P.max(1), 1 - PV.max(1)),
        "entropy": (ent(P), ent(PV)),
        "energy": (-logsumexp(L, 1), -logsumexp(LV, 1)),
        "mahalanobis": (mh.score(L), mh.score(LV)),
        "openmax": (om.score(L, pred), om.score(LV, PV.argmax(1))),
        "dual_rule": ((1 - P.max(1)) + ent(P) / np.log(Kk), (1 - PV.max(1)) + ent(PV) / np.log(Kk)),
    }
    tc, tent = np.percentile(PV.max(1), 5), np.percentile(ent(PV), 95)
    dual_rej = (P.max(1) < tc) | (ent(P) > tent)
    rows = []
    closed_acc = float(np.mean(PK.argmax(1) == y_te))
    closed_f1 = float(f1_score(y_te, PK.argmax(1), average="macro"))
    for m, (s, sv) in sc.items():
        if m == "dual_rule":
            rej = dual_rej
        else:
            rej = s >= np.percentile(sv, 95)          # threshold from known validation particles
        rows.append(dict(seed=seed, held_out=held, method=m, closed_acc=closed_acc, closed_f1=closed_f1,
                         auroc=roc_auc_score(yood, s), aupr=average_precision_score(yood, s),
                         fpr95=fpr_at_95(yood, s), unknown_detect=float(rej[yood == 1].mean()),
                         false_unknown=float(rej[yood == 0].mean())))
    prow = []
    if do_pipeline:
        env = S["env"]["ctx"]
        tr_traj = Trajectories(S["oof"], S["Z_tr"], **env)
        va_traj = Trajectories(S["val"], S["Z_va"], **env)
        clf = train_stopper(S, seed, tr_traj)["random_forest"]
        target = allblock_acc(S, S["val"], S["y_va"], "ctx") - DELTA
        thr, _ = tune_threshold(lambda g: ("learned", clf, g), THR_GRID, va_traj, S["y_va"], target, S["full_cost"])
        th = fit_step_thresholds(va_traj.results(("learned", clf, thr)))
        rk = Trajectories(S["tests"][0], te[Z_COLS].values, **env).results(("learned", clf, thr))
        ro = Trajectories(S["tests"][1], dood[Z_COLS].values, **env).results(("learned", clf, thr))
        for method in ("dual", "energy"):
            rejk = apply_ood(rk, th, method); rejo = apply_ood(ro, th, method)
            predk = np.array([r["pred"] for r in rk])
            acc_acc = float(np.mean(predk[~rejk] == y_te[~rejk])) if (~rejk).any() else 0.0
            costs_all = np.array([r["cost"] for r in rk + ro])
            prow.append(dict(seed=seed, held_out=held, method=method, threshold=thr,
                             unknown_detect=float(rejo.mean()), false_unknown=float(rejk.mean()),
                             accepted_acc=acc_acc, mean_cost_ms=float(costs_all.mean()),
                             mean_blocks=float(np.mean([r["n_blocks"] for r in rk + ro]))))
    return rows, prow


# ------------------------------------------------------------- closed-set per seed
def closed_set(df, seed, cols, costs, seg_ms, z_ms, out):
    tr, va, te = split_dataset_by_group(df, seed=seed)
    S = setup(df, CLASSES, seed, cols, costs, seg_ms, z_ms, tr, va, [te])
    cmap = {c: i for i, c in enumerate(CLASSES)}
    y_te = te.label.map(cmap).values; Z_te = te[Z_COLS].values
    T = S["tests"][0]; FC = S["full_cost"]; y_va = S["y_va"]
    rec, preds, curves = [], {}, []
    acc_full_va = {k: allblock_acc(S, S["val"], y_va, k) for k in ("glob", "ctx")}
    acc_full_te = {k: allblock_acc(S, T, y_te, k) for k in ("glob", "ctx")}
    TV = {k: Trajectories(S["val"], S["Z_va"], **S["env"][k]) for k in ("glob", "ctx")}
    TT = {k: Trajectories(T, Z_te, **S["env"][k]) for k in ("glob", "ctx")}

    def add(name, res, extra=None, reject=None):
        s = summarize(res, y_te, FC, reject)
        s.update(seed=seed, configuration=name, **(extra or {}))
        rec.append(s)
        p = np.array([r["pred"] for r in res])
        if reject is not None:
            p = np.where(reject, -1, p)
        preds[name] = (p, np.array([r["cost"] for r in res]))
        return res

    add("1_global_fixed085", TT["glob"].results(("fixed", 0.85)))
    add("2_context_fixed085", TT["ctx"].results(("fixed", 0.85)))
    for key, name in (("glob", "1T_global_tuned_tau"), ("ctx", "2T_context_tuned_tau")):
        tau, _ = tune_threshold(lambda g: ("fixed", g), TAU_GRID, TV[key], y_va, acc_full_va[key] - DELTA, FC)
        add(name, TT[key].results(("fixed", tau)), {"param": tau})
        for g in TAU_GRID:
            curves.append(dict(seed=seed, policy=f"fixed_tau_{key}", param=g,
                               **summarize(TT[key].results(("fixed", g)), y_te, FC)))
    # learned stopping (on-policy, OOF training trajectories)
    tr_traj = Trajectories(S["oof"], S["Z_tr"], **S["env"]["ctx"])
    models = train_stopper(S, seed, tr_traj)
    Xt, Lt = stopping_training_data(TT["ctx"])
    stop_rows = []
    feat_names = [f"p_{c}" for c in CLASSES] + ["max_post", "entropy", "margin", "n_acquired",
                                               "cum_cost", "n_remaining"] + Z_COLS
    for mname, clf in models.items():
        thr, _ = tune_threshold(lambda g: ("learned", clf, g), THR_GRID, TV["ctx"], y_va, acc_full_va["ctx"] - DELTA, FC)
        pt = clf.predict_proba(Xt)[:, 1]
        yhat = (pt >= thr).astype(int)
        row = dict(seed=seed, model=mname, threshold=thr, accuracy=accuracy_score(Lt, yhat),
                   f1=f1_score(Lt, yhat), auc=roc_auc_score(Lt, pt), recall=float(yhat[Lt == 1].mean()),
                   specificity=float(1 - yhat[Lt == 0].mean()), stop_label_rate=float(Lt.mean()))
        if mname == "random_forest":
            row["top_feature"] = feat_names[int(np.argmax(clf.feature_importances_))]
        else:
            row["top_feature"] = feat_names[int(np.argmax(np.abs(clf.named_steps["clf"].coef_[0])))]
        stop_rows.append(row)
        tag = "3_context_learned_rf" if mname == "random_forest" else "3b_context_learned_lr"
        res3 = add(tag, TT["ctx"].results(("learned", clf, thr)), {"param": thr})
        if mname == "random_forest":
            res3_rf, clf_rf, thr_rf = res3, clf, thr
        for g in THR_GRID:
            curves.append(dict(seed=seed, policy=f"learned_{mname}", param=g,
                               **summarize(TT["ctx"].results(("learned", clf, g)), y_te, FC)))
    # open-set on the closed test set (all known): false unknowns at the operating point
    th = fit_step_thresholds(TV["ctx"].results(("learned", clf_rf, thr_rf)))
    add("4_full_dual_perstep", res3_rf, {"param": thr_rf}, reject=apply_ood(res3_rf, th, "dual"))
    add("4E_full_energy_perstep", res3_rf, {"param": thr_rf}, reject=apply_ood(res3_rf, th, "energy"))
    th5 = fit_step_thresholds(TV["ctx"].results(("none",)))
    th_v1 = {k: th5[5] for k in th5}
    add("4v1_dual_allblock_thresholds", res3_rf, {"param": thr_rf}, reject=apply_ood(res3_rf, th_v1, "dual"))
    # ablations (fixed tau 0.85): unit context weight; drop each Z descriptor
    w1 = dict(S["w_ctx"]); w1["context"] = 1.0
    env1 = dict(S["env"]["ctx"]); env1["fusion"] = Fusion(S["lp"], w1)
    env1["eig"] = EIG(S["oof"], S["y_tr"], S["lp"], w1, S["K"], seed)
    add("abl_context_unit_weight", Trajectories(T, Z_te, **env1).results(("fixed", 0.85)), {"w_context": 1.0})
    for zc in Z_COLS:
        keep = [c for c in Z_COLS if c != zc]
        m = ContextExpert(seed).fit(tr[keep].values, S["y_tr"], va[keep].values, y_va)
        val2 = dict(S["val"]); val2["context"] = m.predict_proba(va[keep].values)
        T2 = dict(T); T2["context"] = m.predict_proba(te[keep].values)
        w2 = fit_weights(val2, y_va, S["lp"], BLOCKS + ["context"])
        env2 = dict(S["env"]["ctx"]); env2["fusion"] = Fusion(S["lp"], w2)
        env2["eig"] = EIG(S["oof"], S["y_tr"], S["lp"], w2, S["K"], seed)
        add(f"abl_drop_{zc}", Trajectories(T2, Z_te, **env2).results(("fixed", 0.85)),
            {"context_only_acc": float(np.mean(T2["context"].argmax(1) == y_te)), "w_context": w2["context"]})
    pz = T["context"]
    rec.append(dict(seed=seed, configuration="ref_context_only", accuracy=float(np.mean(pz.argmax(1) == y_te)),
                    macro_f1=float(f1_score(y_te, pz.argmax(1), average="macro")),
                    mean_cost_ms=seg_ms + z_ms, mean_blocks=0.0,
                    cost_saving_pct=100 * (1 - (seg_ms + z_ms) / FC), rejected_pct=0.0))
    for k in ("glob", "ctx"):
        rec.append(dict(seed=seed, configuration=f"ref_all_blocks_{k}", accuracy=acc_full_te[k],
                        mean_cost_ms=FC + (z_ms if k == "ctx" else 0.0), mean_blocks=5.0,
                        cost_saving_pct=100 * (1 - (FC + (z_ms if k == "ctx" else 0.0)) / FC), rejected_pct=0.0))
    blk = [dict(seed=seed, block=b, test_acc=float(np.mean(T[b].argmax(1) == y_te)),
                w_glob=S["w_glob"].get(b), w_ctx=S["w_ctx"].get(b)) for b in BLOCKS + ["context"]]
    np.savez_compressed(out / f"preds_seed{seed}.npz", y=y_te,
                        **{f"{k}__pred": v[0] for k, v in preds.items()},
                        **{f"{k}__cost": v[1] for k, v in preds.items()})
    fb = {"seed": seed,
          "cfg1": pd.Series([r["first"] for r in TT["glob"].results(("none",))]).value_counts().to_dict(),
          "cfg2": pd.Series([r["first"] for r in TT["ctx"].results(("none",))]).value_counts().to_dict(),
          "cfg3_nblocks": pd.Series([r["n_blocks"] for r in res3_rf]).value_counts().sort_index().to_dict()}
    return rec, curves, stop_rows, blk, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_cache_v2.csv")
    ap.add_argument("--costs", default="data/block_costs_ms_v2.json")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--lopo-pipeline-seeds", default="0,1,2")
    ap.add_argument("--out", default="results/v2")
    ap.add_argument("--skip-lopo", action="store_true")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(ROOT / a.features)
    z = pd.read_csv(ROOT / "data" / "context_prior_cache.csv")[["crop"] + Z_COLS]
    df = feats.merge(z, on="crop")
    cols = block_columns(feats)
    costs, seg_ms, z_ms = load_costs(ROOT / a.costs)
    seeds = [int(s) for s in a.seeds.split(",")]
    pseeds = [int(s) for s in a.lopo_pipeline_seeds.split(",")] if a.lopo_pipeline_seeds else []
    meta = dict(git_commit=get_git_commit_hash(), timestamp=datetime.datetime.now().isoformat(),
                features=a.features, seeds=seeds, delta=DELTA, costs=costs, seg_ms=seg_ms, z_ms=z_ms,
                n_features={b: len(c) for b, c in cols.items()})
    R, C, ST, BL, FB, LO, LP = [], [], [], [], [], [], []
    for sd in seeds:
        t0 = time.time()
        r, c, st, bl, fb = closed_set(df, sd, cols, costs, seg_ms, z_ms, out)
        R += r; C += c; ST += st; BL += bl; FB.append(fb)
        pd.DataFrame(R).to_csv(out / "configs.csv", index=False)
        pd.DataFrame(C).to_csv(out / "tradeoff.csv", index=False)
        pd.DataFrame(ST).to_csv(out / "stopping.csv", index=False)
        pd.DataFrame(BL).to_csv(out / "blocks.csv", index=False)
        json.dump(FB, open(out / "first_blocks.json", "w"), indent=1, default=str)
        print(f"seed {sd} closed-set done in {time.time() - t0:.0f}s", flush=True)
        if a.skip_lopo:
            continue
        for held in CLASSES:
            t1 = time.time()
            rows, prow = lopo_fold(df, held, sd, cols, costs, seg_ms, z_ms, sd in pseeds)
            LO += rows; LP += prow
            pd.DataFrame(LO).to_csv(out / "lopo_detectors.csv", index=False)
            pd.DataFrame(LP).to_csv(out / "lopo_pipeline.csv", index=False)
            print(f"  seed {sd} LOPO {held} done in {time.time() - t1:.0f}s", flush=True)
    json.dump(meta, open(out / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
