r"""
src/open_set.py - Open-set polymer identification and out-of-distribution (OOD) detection.

Implements Leave-One-Polymer-Out (LOPO) cross-validation across all 6 polymers:
For each polymer p \in {ABS, Nylon, PE, PET, PS, PVC}:
1. Train all block classifiers and log-opinion pool weights on the remaining 5 known polymers.
2. Tune rejection thresholds (\tau_conf, \tau_ent) on validation data containing ZERO samples of p.
3. Assert strictly that p is completely absent from train and val sets (no test-set leakage).
4. Evaluate on known test set + all held-out samples of p.

Baselines implemented:
- Dual Rule (\tau_conf, \tau_ent): UNKNOWN if max_post < \tau_conf or entropy > \tau_ent
- Max Softmax Probability (MSP): 1 - max_k P(y=k|x)
- Entropy: H(P)
- Energy Score: -log \sum_k exp(f_k(x))
- Mahalanobis Distance: \min_k (z - \mu_k)^T \Sigma^{-1} (z - \mu_k)
- OpenMax: Extreme value Weibull tail calibration on class mean activation vectors

Metrics: AUROC, AUPR, FPR@95%TPR, Unknown Detection Rate, False Unknown Rate, Closed-set Accuracy, Macro-F1.
Outputs: results/open_set_lopo.csv.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import datetime
import numpy as np
import pandas as pd
from scipy.special import softmax, logsumexp
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from src.dataset import split_dataset_by_group, CLASSES
from src.features import BLOCK_EXTRACTORS
from src.math_utils import compute_entropy, compute_max_posterior, fuse_log_opinion_pool, fit_log_opinion_pool_weights
from src.cost_model import load_block_costs_ms
from src.utils import get_git_commit_hash, PROJECT_ROOT

def compute_fpr_at_95_tpr(y_true_ood: np.ndarray, ood_scores: np.ndarray) -> float:
    r"""
    Computes FPR of in-distribution classifier when TPR is 95%.
    Here y_true_ood = 1 for OOD (unknown), 0 for ID (known).
    In-distribution score = -ood_scores (higher = more in-distribution).
    We find threshold tau such that 95% of ID samples have score >= tau.
    Then FPR is the fraction of OOD samples with score >= tau.
    """
    id_scores = -ood_scores[y_true_ood == 0]
    ood_test_scores = -ood_scores[y_true_ood == 1]
    
    if len(id_scores) == 0 or len(ood_test_scores) == 0:
        return 0.0
    
    # 95th percentile threshold on ID (at which 95% of ID samples have score >= tau)
    tau = np.percentile(id_scores, 5.0)
    fpr = float(np.mean(ood_test_scores >= tau))
    return fpr

class MahalanobisDetector:
    """Computes minimum Mahalanobis distance to known class centroids in feature space."""
    def __init__(self, eps: float = 1e-4):
        self.eps = eps
        self.means = {}
        self.inv_cov = None

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int):
        self.means = {}
        diffs = []
        for k in range(num_classes):
            idx = np.where(y == k)[0]
            if len(idx) > 0:
                mean_k = np.mean(X[idx], axis=0)
                self.means[k] = mean_k
                diffs.append(X[idx] - mean_k)
            else:
                self.means[k] = np.zeros(X.shape[1])
        
        all_diffs = np.vstack(diffs) if len(diffs) > 0 else np.zeros_like(X)
        cov = np.cov(all_diffs, rowvar=False) + self.eps * np.eye(X.shape[1])
        self.inv_cov = np.linalg.pinv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        scores = []
        for i in range(len(X)):
            x = X[i]
            dists = []
            for k, mu in self.means.items():
                delta = x - mu
                d = np.dot(np.dot(delta, self.inv_cov), delta)
                dists.append(d)
            scores.append(np.min(dists))
        return np.array(scores, dtype=np.float64)

class OpenMaxDetector:
    """Extreme Value Theory (EVT) Weibull calibrated OpenMax detector."""
    def __init__(self, tail_size: int = 20, alpha: int = 3):
        self.tail_size = tail_size
        self.alpha = alpha
        self.mavs = {}
        self.weibull_models = {}

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int):
        self.mavs = {}
        self.weibull_models = {}
        for k in range(num_classes):
            idx = np.where(y == k)[0]
            if len(idx) < 5:
                continue
            mav = np.mean(X[idx], axis=0)
            self.mavs[k] = mav
            # Euclidean distances to MAV
            dists = np.linalg.norm(X[idx] - mav, axis=1)
            sorted_dists = np.sort(dists)
            tail = sorted_dists[-min(self.tail_size, len(sorted_dists)):]
            try:
                # Fit Weibull on extreme tail
                shape, loc, scale = stats.weibull_min.fit(tail, floc=0)
                self.weibull_models[k] = (shape, loc, scale)
            except Exception:
                self.weibull_models[k] = None
        return self

    def predict_unknown_prob(self, X: np.ndarray) -> np.ndarray:
        unknown_probs = []
        for i in range(len(X)):
            x = X[i]
            w_scores = []
            for k, mav in self.mavs.items():
                d = np.linalg.norm(x - mav)
                if k in self.weibull_models and self.weibull_models[k] is not None:
                    shape, loc, scale = self.weibull_models[k]
                    # Probability of being an outlier under Weibull CDF
                    w = stats.weibull_min.cdf(d, shape, loc=loc, scale=scale)
                    w_scores.append(w)
                else:
                    w_scores.append(0.5)
            # OpenMax unknown score is the average or max tail probability
            unknown_probs.append(float(np.mean(w_scores)) if len(w_scores) > 0 else 0.5)
        return np.array(unknown_probs, dtype=np.float64)

def run_lopo_open_set_evaluation(
    seed: int = 42,
    output_csv: Optional[Path] = None
) -> pd.DataFrame:
    r"""
    Executes Leave-One-Polymer-Out (LOPO) open-set evaluation across all 6 polymers.
    
    Hard non-leakage guarantee:
    - Held-out polymer p is absent from training and validation splits.
    - Asserted via code assertions in every fold.
    """
    if output_csv is None:
        output_csv = PROJECT_ROOT / "results" / "open_set_lopo.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    feat_cache = PROJECT_ROOT / "data" / "features_cache.csv"
    assert feat_cache.exists(), "Feature cache missing. Run earlier modules."
    df_feat = pd.read_csv(feat_cache)

    sample_rgb = np.ones((32, 32, 3), dtype=np.float32)
    sample_mask = np.ones((32, 32), dtype=bool)
    block_columns = {b: [c for c in fn(sample_rgb, sample_mask).keys() if c in df_feat.columns] for b, fn in BLOCK_EXTRACTORS.items()}

    results = []

    print(f"\n========================================================")
    print(f"RUNNING LEAVE-ONE-POLYMER-OUT (LOPO) OPEN-SET EVALUATION")
    print(f"========================================================")

    for held_out_polymer in CLASSES:
        print(f"\nEvaluating LOPO Fold: Held-out Polymer = {held_out_polymer}")
        
        # 1. Split into known polymers and held-out OOD polymer
        df_known = df_feat[df_feat["label"] != held_out_polymer].copy().reset_index(drop=True)
        df_ood_test = df_feat[df_feat["label"] == held_out_polymer].copy().reset_index(drop=True)

        known_classes = [c for c in CLASSES if c != held_out_polymer]
        known_map = {c: i for i, c in enumerate(known_classes)}
        K_known = len(known_classes)

        # 2. Split known polymers by slide group
        df_train, df_val, df_test_known = split_dataset_by_group(df_known, seed=seed)

        # MANDATORY RESEARCH INTEGRITY ASSERTIONS
        assert held_out_polymer not in df_train["label"].values, f"LEAKAGE: {held_out_polymer} in train!"
        assert held_out_polymer not in df_val["label"].values, f"LEAKAGE: {held_out_polymer} in val!"
        print(f"  Integrity Check Passed: '{held_out_polymer}' absent from train ({len(df_train)}) and val ({len(df_val)}).")
        print(f"  Test pool: {len(df_test_known)} known test samples + {len(df_ood_test)} held-out {held_out_polymer} samples.")

        y_train = df_train["label"].map(known_map).values
        y_val = df_val["label"].map(known_map).values
        y_test_known = df_test_known["label"].map(known_map).values

        # Prior distribution over the 5 known classes
        prior_p = np.bincount(y_train, minlength=K_known).astype(float) / len(y_train)
        log_prior = np.log(prior_p + 1e-9)

        # Train 5 block expert classifiers on known train set
        rf_models = {}
        val_probs_dict = {}
        test_known_probs_dict = {}
        ood_test_probs_dict = {}

        for b, cols in block_columns.items():
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=4, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
            rf.fit(df_train[cols].values, y_train)
            rf_models[b] = rf
            val_probs_dict[b] = rf.predict_proba(df_val[cols].values)
            test_known_probs_dict[b] = rf.predict_proba(df_test_known[cols].values)
            ood_test_probs_dict[b] = rf.predict_proba(df_ood_test[cols].values)

        # Fit log-opinion pool weights on validation split
        block_weights = fit_log_opinion_pool_weights(val_probs_dict, y_val, log_prior)

        # Compute fused posteriors
        val_fused = fuse_log_opinion_pool(log_prior, val_probs_dict, block_weights)
        test_known_fused = fuse_log_opinion_pool(log_prior, test_known_probs_dict, block_weights)
        ood_test_fused = fuse_log_opinion_pool(log_prior, ood_test_probs_dict, block_weights)

        # Closed-set metrics on known test set
        closed_preds = np.argmax(test_known_fused, axis=1)
        closed_acc = accuracy_score(y_test_known, closed_preds)
        closed_f1 = f1_score(y_test_known, closed_preds, average="macro")

        # Combine test set for open-set evaluation: known = 0, OOD = 1
        all_test_fused = np.vstack([test_known_fused, ood_test_fused])
        y_ood_true = np.concatenate([np.zeros(len(test_known_fused), dtype=int), np.ones(len(ood_test_fused), dtype=int)])

        # Compute unnormalized log opinion pool (logits) for energy score
        val_logits = np.repeat(log_prior[None, :], len(df_val), axis=0).copy()
        for b, p_b in val_probs_dict.items():
            val_logits = val_logits + block_weights[b] * (np.log(np.clip(p_b, 1e-6, 1.0)) - log_prior[None, :])
        
        all_logits = np.repeat(log_prior[None, :], len(all_test_fused), axis=0).copy()
        for b in BLOCK_EXTRACTORS:
            p_comb = np.vstack([test_known_probs_dict[b], ood_test_probs_dict[b]])
            all_logits = all_logits + block_weights[b] * (np.log(np.clip(p_comb, 1e-6, 1.0)) - log_prior[None, :])

        # Baseline 1: Max Softmax Probability (MSP)
        scores_msp = 1.0 - np.max(all_test_fused, axis=1)

        # Baseline 2: Entropy
        scores_entropy = np.array([compute_entropy(p) for p in all_test_fused])

        # Baseline 3: Energy Score (-logsumexp of logits)
        scores_energy = -logsumexp(all_logits, axis=-1)

        # Baseline 4: Mahalanobis Distance on fused probability space
        # Fit on validation probabilities
        maha = MahalanobisDetector()
        maha.fit(val_fused, y_val, num_classes=K_known)
        scores_maha = maha.score(all_test_fused)

        # Baseline 5: OpenMax
        openmax = OpenMaxDetector(tail_size=25)
        openmax.fit(val_fused, y_val, num_classes=K_known)
        scores_openmax = openmax.predict_unknown_prob(all_test_fused)

        # Baseline 6: Dual Rule (\tau_conf, \tau_ent)
        val_max_post = np.max(val_fused, axis=1)
        val_ent = np.array([compute_entropy(p) for p in val_fused])
        
        # Operational threshold tuned on validation to maintain 95% TPR on ID (max 5% false unknown)
        tau_conf = float(np.percentile(val_max_post, 5.0))
        tau_ent = float(np.percentile(val_ent, 95.0))
        
        # Binary unknown decision rule
        test_max_post = np.max(all_test_fused, axis=1)
        test_ent = scores_entropy
        is_rejected_dual = (test_max_post < tau_conf) | (test_ent > tau_ent)

        # Continuous composite score for Dual Rule: (1 - conf) + normalized entropy
        scores_dual = (1.0 - test_max_post) + (test_ent / np.log(K_known))

        # Operational rejection rates
        unknown_det_rate = float(np.mean(is_rejected_dual[y_ood_true == 1])) # TPR for OOD
        false_unknown_rate = float(np.mean(is_rejected_dual[y_ood_true == 0])) # FPR for ID

        methods = {
            "msp": scores_msp,
            "entropy": scores_entropy,
            "energy": scores_energy,
            "mahalanobis": scores_maha,
            "openmax": scores_openmax,
            "dual_rule": scores_dual
        }

        for method_name, sc in methods.items():
            auroc = roc_auc_score(y_ood_true, sc)
            aupr = average_precision_score(y_ood_true, sc)
            fpr95 = compute_fpr_at_95_tpr(y_ood_true, sc)

            row = {
                "held_out_polymer": held_out_polymer,
                "method": method_name,
                "closed_set_accuracy": round(closed_acc, 4),
                "closed_set_macro_f1": round(closed_f1, 4),
                "auroc": round(auroc, 4),
                "aupr": round(aupr, 4),
                "fpr_at_95_tpr": round(fpr95, 4),
                "unknown_detection_rate": round(unknown_det_rate if method_name == "dual_rule" else float(np.mean(sc[y_ood_true == 1] >= np.percentile(sc[y_ood_true == 0], 95.0))), 4),
                "false_unknown_rate": round(false_unknown_rate if method_name == "dual_rule" else float(np.mean(sc[y_ood_true == 0] >= np.percentile(sc[y_ood_true == 0], 95.0))), 4),
                "tau_conf": round(tau_conf, 4) if method_name == "dual_rule" else None,
                "tau_ent": round(tau_ent, 4) if method_name == "dual_rule" else None,
                "seed": seed,
                "git_commit": get_git_commit_hash(),
                "timestamp": datetime.datetime.now().isoformat()
            }
            results.append(row)
            print(f"  [{method_name:12s}] AUROC={auroc:.4f}, AUPR={aupr:.4f}, FPR@95%TPR={fpr95:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"\nLOPO Open-Set Results successfully written to {output_csv}")

    # Summary table across folds
    summary = df_results.groupby("method").agg({
        "auroc": ["mean", "std"],
        "aupr": ["mean", "std"],
        "fpr_at_95_tpr": ["mean", "std"],
        "closed_set_accuracy": ["mean"]
    })
    print("\n--- LOPO OPEN-SET BENCHMARK SUMMARY (Mean ± Std over 6 Folds) ---")
    print(summary)
    return df_results
