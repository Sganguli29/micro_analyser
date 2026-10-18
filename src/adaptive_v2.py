r"""
src/adaptive_v2.py - Corrected adaptive acquisition pipeline (v2).

Fixes relative to src/pipeline.py (v1):
1. Consistent context fusion. The context prior P(Y|Z) enters the log-opinion pool as one more
   weighted expert measured against the *class prior*:
       log P(y|Z,A) = log P0(y) + w_Z [log P(y|Z) - log P0(y)] + sum_b w_b [log P_b(y|x_b) - log P0(y)]
   v1 instead replaced P0 by P(y|Z) everywhere, so every block's evidence was measured against
   a quantity the block expert never saw. w_Z is fitted jointly with the block weights on validation NLL.
2. Out-of-fold simulation pool. The Monte Carlo EIG pool and the stopping-policy training trajectories
   use out-of-fold (GroupKFold) predictions on the training split; v1 used in-sample random-forest
   predictions, which are near-certain and distort EIG and the stopping states.
3. On-policy learned stopping. Trajectories are simulated with the same initial belief and acquisition
   policy that the deployed pipeline uses, and the STOP threshold is chosen on validation as the
   cheapest threshold whose pipeline accuracy stays within `delta` of all-block accuracy.
4. Operating-point OOD thresholds. Open-set thresholds are calibrated on validation runs of the
   deployed pipeline separately for each number of acquired blocks, so they match the posteriors they
   are applied to (v1 calibrated on 5-block posteriors and applied them after 1 block).
5. Segmentation cost is accounted for: the Otsu mask (needed by Z and by Shape) is paid once per particle.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy.special import softmax, logsumexp
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

from src.math_utils import compute_entropy, compute_margin, fit_log_opinion_pool_weights
from src.calibration import TemperatureScaler

BLOCKS = ["shape", "intensity", "colour", "texture", "frequency"]
PREFIX = {"shape": "shape_", "intensity": "intensity_", "colour": "colour_",
          "texture": "texture_", "frequency": "freq_"}
Z_COLS = ["z_area", "z_bbox_w", "z_bbox_h", "z_aspect_ratio",
          "z_mean_brightness", "z_fluorescence_occupancy", "z_snr"]


def block_columns(df) -> Dict[str, List[str]]:
    return {b: [c for c in df.columns if c.startswith(PREFIX[b])] for b in BLOCKS}


# ----------------------------------------------------------------- experts
def make_block_expert(seed: int):
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                class_weight="balanced_subsample", random_state=seed, n_jobs=2)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", CalibratedClassifierCV(rf, method="isotonic", cv=3))])


class ContextExpert:
    """P(Y|Z): standardised multinomial logistic regression + temperature scaling."""
    def __init__(self, seed: int):
        self.pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler()),
                              ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=seed))])
        self.temp = None

    def fit(self, X, y, Xv=None, yv=None):
        self.pipe.fit(np.log1p(np.abs(X)) * np.sign(X), y)
        if Xv is not None:
            self.temp = TemperatureScaler().fit(self._raw(Xv), yv)
        return self

    def _raw(self, X):
        return self.pipe.predict_proba(np.log1p(np.abs(X)) * np.sign(X))

    def predict_proba(self, X):
        p = self._raw(X)
        return self.temp.predict_proba(p) if self.temp is not None else p


def fit_experts(tr, va, te_list, y_tr, y_va, cols, seed, K):
    """Fits block experts + context expert. Returns probs dicts for val, each test frame and OOF train."""
    groups = tr["source_image"].values
    gkf = GroupKFold(n_splits=3)
    names = BLOCKS + ["context"]
    oof = {b: np.zeros((len(tr), K)) for b in names}
    val = {}
    tests = [dict() for _ in te_list]
    for b in names:
        Xtr = tr[Z_COLS].values if b == "context" else tr[cols[b]].values
        for f_tr, f_te in gkf.split(Xtr, y_tr, groups):
            if b == "context":
                m = ContextExpert(seed).fit(Xtr[f_tr], y_tr[f_tr])
            else:
                m = make_block_expert(seed).fit(Xtr[f_tr], y_tr[f_tr])
            oof[b][f_te] = m.predict_proba(Xtr[f_te])
        if b == "context":
            m = ContextExpert(seed).fit(Xtr, y_tr, va[Z_COLS].values, y_va)
            val[b] = m.predict_proba(va[Z_COLS].values)
            for t, te in zip(tests, te_list):
                t[b] = m.predict_proba(te[Z_COLS].values)
        else:
            m = make_block_expert(seed).fit(Xtr, y_tr)
            val[b] = m.predict_proba(va[cols[b]].values)
            for t, te in zip(tests, te_list):
                t[b] = m.predict_proba(te[cols[b]].values)
    return oof, val, tests


# ----------------------------------------------------------------- fusion
def log_ratio(p, log_prior):
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    return np.log(p) - log_prior


class Fusion:
    def __init__(self, log_prior, weights):
        self.lp = log_prior
        self.w = weights

    def logits(self, probs_row: Dict[str, np.ndarray], names) -> np.ndarray:
        z = self.lp.copy()
        for b in names:
            z = z + self.w[b] * log_ratio(probs_row[b], self.lp)
        return z


def fit_weights(val_probs, y_val, log_prior, names):
    """Non-negative pool weights minimising validation NLL. Bounded L-BFGS-B from several starts
    (including the Nelder-Mead solution used in v1); the lowest-NLL solution is kept."""
    from scipy.optimize import minimize
    K = len(log_prior)
    LR = np.stack([log_ratio(val_probs[b], log_prior) for b in names])          # [B,N,K]
    def nll(w):
        z = log_prior[None, :] + np.tensordot(w, LR, axes=1)
        z = z - logsumexp(z, axis=1, keepdims=True)
        return -float(np.mean(z[np.arange(len(y_val)), y_val]))
    nm = fit_log_opinion_pool_weights({b: val_probs[b] for b in names}, y_val, log_prior, block_names=names)
    starts = [np.array([nm[b] for b in names]), np.full(len(names), 0.5), np.full(len(names), 1.0)]
    best = None
    for x0 in starts:
        r = minimize(nll, x0, method="L-BFGS-B", bounds=[(0.0, 3.0)] * len(names))
        if best is None or r.fun < best.fun:
            best = r
    return {b: round(float(best.x[i]), 4) for i, b in enumerate(names)}


# ----------------------------------------------------------------- EIG
class EIG:
    def __init__(self, oof_probs, y_tr, log_prior, weights, K, seed, per_class=50):
        rs = np.random.RandomState(seed)
        idx = []
        for c in range(K):
            ci = np.where(y_tr == c)[0]
            idx.extend(rs.choice(ci, size=min(per_class, len(ci)), replace=False))
        idx = np.array(idx)
        self.cls = y_tr[idx]
        self.counts = np.bincount(self.cls, minlength=K).astype(float)
        self.lr = {b: weights[b] * log_ratio(oof_probs[b][idx], log_prior) for b in BLOCKS}

    def scores(self, logit, remaining, costs):
        P = softmax(logit)
        H0 = compute_entropy(P)
        w = P[self.cls] / np.maximum(self.counts[self.cls], 1.0)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        out = {}
        for b in remaining:
            newP = softmax(logit[None, :] + self.lr[b], axis=1)
            H1 = -np.sum(newP * np.log(newP + 1e-12), axis=1)
            out[b] = max(0.0, H0 - float(np.dot(w, H1))) / max(costs[b], 1e-6)
        return out


# ----------------------------------------------------------------- pipeline
def state_vector(P, n_acq, cum_cost, n_rem, z):
    return np.concatenate([P, [P.max(), compute_entropy(P), compute_margin(P), n_acq, cum_cost, n_rem], z])


class Trajectories:
    """Full EIG/Cost acquisition paths for a batch of particles.

    The acquisition order never depends on the stopping rule, so each particle's path is computed once
    (all five blocks) and any stopping rule is applied afterwards by choosing the first step at which it
    fires. Index k = 0..4 means k+1 blocks acquired. Arrays: P, logit [N,5,K]; cost [N,5]; S [N,5,D]."""
    def __init__(self, probs, Z, *, fusion, eig, costs, use_context, upfront_cost, order=None):
        """order: None for EIG/Cost; otherwise a function i -> list of the five blocks (fixed or random orders)."""
        N = len(Z)
        K = len(fusion.lp)
        self.P = np.zeros((N, 5, K)); self.logit = np.zeros((N, 5, K))
        self.cost = np.zeros((N, 5)); self.first = []; self.order = []
        S = []
        for i in range(N):
            row = {b: probs[b][i] for b in probs}
            acquired = ["context"] if use_context else []
            remaining = list(BLOCKS)
            cum = upfront_cost
            logit = fusion.logits(row, acquired)
            si = []
            fixed = order(i) if order is not None else None
            for k in range(5):
                if fixed is None:
                    sc = eig.scores(logit, remaining, costs)
                    b = max(remaining, key=lambda q: sc[q])
                else:
                    b = fixed[k]
                remaining.remove(b); acquired.append(b); cum += costs[b]
                if k == 0:
                    self.first.append(b)
                logit = fusion.logits(row, acquired)
                P = softmax(logit)
                self.P[i, k] = P; self.logit[i, k] = logit; self.cost[i, k] = cum
                si.append(state_vector(P, k + 1, cum, len(remaining), Z[i]))
            S.append(si)
            self.order.append([a for a in acquired if a != "context"])
        self.S = np.array(S)
        self.N = N
        self._prob_cache = {}

    def stop_probs(self, clf):
        key = id(clf)
        if key not in self._prob_cache:
            D = self.S.shape[-1]
            self._prob_cache[key] = clf.predict_proba(self.S.reshape(-1, D))[:, 1].reshape(self.N, 5)
        return self._prob_cache[key]

    def stop_index(self, stop):
        if stop[0] == "none":
            return np.full(self.N, 4)
        if stop[0] == "fixed":
            fire = self.P.max(-1) >= stop[1]
        elif stop[0] == "learned":
            fire = self.stop_probs(stop[1]) >= stop[2]
        else:
            raise ValueError(stop)
        fire[:, 4] = True
        return fire.argmax(1)

    def results(self, stop):
        idx = self.stop_index(stop)
        r = np.arange(self.N)
        return [dict(P=self.P[i, k], logit=self.logit[i, k], pred=int(self.P[i, k].argmax()),
                     cost=float(self.cost[i, k]), n_blocks=int(k + 1), first=self.first[i])
                for i, k in zip(r, idx)]


def summarize(res, y, full_cost, reject=None):
    from sklearn.metrics import f1_score
    pred = np.array([r["pred"] for r in res])
    cost = np.array([r["cost"] for r in res])
    nb = np.array([r["n_blocks"] for r in res])
    keep = np.ones(len(y), bool) if reject is None else ~reject
    acc = float(np.mean(pred[keep] == y[keep])) if keep.any() else 0.0
    f1 = float(f1_score(y[keep], pred[keep], average="macro")) if keep.any() else 0.0
    return {"accuracy": acc, "macro_f1": f1, "mean_cost_ms": float(cost.mean()),
            "mean_blocks": float(nb.mean()), "cost_saving_pct": float(100 * (1 - cost.mean() / full_cost)),
            "rejected_pct": 0.0 if reject is None else float(100 * reject.mean())}


# ----------------------------------------------------------------- learned stopping
def stopping_training_data(traj: Trajectories):
    """On-policy states after 1..4 blocks. Label (a): STOP=1 if current prediction equals the
    prediction after all five blocks."""
    pred = traj.P.argmax(-1)
    X = traj.S[:, :4].reshape(-1, traj.S.shape[-1])
    L = (pred[:, :4] == pred[:, 4:5]).reshape(-1).astype(int)
    return X, L


def fit_stopping_models(X, L, seed):
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed,
                                n_jobs=2, class_weight="balanced")
    lr = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, C=1.0))])
    rf.fit(X, L)
    lr.fit(X, L)
    return {"random_forest": rf, "logistic_regression": lr}


def tune_threshold(make_stop, grid, traj_va, y_va, target_acc, full_cost):
    """Cheapest stopping setting on validation whose accuracy >= target_acc."""
    curve = []
    for g in grid:
        s = summarize(traj_va.results(make_stop(g)), y_va, full_cost)
        curve.append((g, s["accuracy"], s["mean_cost_ms"]))
    ok = [c for c in curve if c[1] >= target_acc]
    best = min(ok, key=lambda c: c[2]) if ok else max(curve, key=lambda c: c[1])
    return best[0], curve


# ----------------------------------------------------------------- OOD
def energy(logit):
    return -float(logsumexp(logit))


def ood_scores(r):
    P = r["P"]
    return {"conf": float(P.max()), "ent": compute_entropy(P), "energy": energy(r["logit"])}


def fit_step_thresholds(res_val, q=5.0, min_n=30):
    """Per-number-of-blocks thresholds: conf at q-th pct, entropy and energy at (100-q)-th pct."""
    sc = [ood_scores(r) for r in res_val]
    nb = np.array([r["n_blocks"] for r in res_val])
    conf = np.array([s["conf"] for s in sc]); ent = np.array([s["ent"] for s in sc])
    en = np.array([s["energy"] for s in sc])
    pooled = (np.percentile(conf, q), np.percentile(ent, 100 - q), np.percentile(en, 100 - q))
    th = {"pooled": pooled}
    for k in range(1, 6):
        m = nb == k
        th[k] = (np.percentile(conf[m], q), np.percentile(ent[m], 100 - q), np.percentile(en[m], 100 - q)) \
            if m.sum() >= min_n else pooled
    return th


def apply_ood(res, th, method="dual", per_step=True):
    rej = []
    for r in res:
        s = ood_scores(r)
        c, e, en = th[r["n_blocks"]] if per_step else th["pooled"]
        if method == "dual":
            rej.append(s["conf"] < c or s["ent"] > e)
        else:
            rej.append(s["energy"] > en)
    return np.array(rej)
