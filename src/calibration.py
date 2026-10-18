r"""
src/calibration.py - Rigorous classifier calibration: Temperature scaling, Platt (Sigmoid), and Isotonic regression.

Supports:
- Uncalibrated base Random Forest
- Temperature Scaling: learns scalar T > 0 on validation logits
- Platt Scaling: Multiclass logistic calibration on logits
- Isotonic Regression: Monotonic piecewise constant calibration per class

Reports:
- 15-bin Expected Calibration Error (ECE)
- Multiclass Brier score
- Negative Log-Likelihood (NLL)
before and after calibration, selecting the best method by validation NLL.
Saves reliability diagrams to figures/reliability_<block>.png and figures/reliability_<block>.pdf.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from src.dataset import split_dataset_by_group, CLASSES, CLASS_TO_IDX
from src.features import BLOCK_EXTRACTORS
from src.utils import get_git_commit_hash, save_figure, PROJECT_ROOT

def compute_ece_15(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    r"""
    Computes Expected Calibration Error (ECE) across n_bins (default 15).
    Formula:
        ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} | acc(B_m) - conf(B_m) |
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        bin_mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        bin_count = int(np.sum(bin_mask))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[bin_mask]))
            bin_conf = float(np.mean(confidences[bin_mask]))
            ece += (bin_count / N) * np.abs(bin_acc - bin_conf)

    return float(ece)

def compute_brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    r"""
    Computes multiclass Brier score:
    Formula:
        Brier = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik}^*)^2
    """
    N, K = probs.shape
    one_hot = np.zeros((N, K), dtype=np.float64)
    one_hot[np.arange(N), y_true] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

class TemperatureScaler:
    """Post-hoc temperature scaling: p_T = softmax(log(p) / T)."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray):
        logits = np.log(np.clip(val_probs, 1e-7, 1.0))
        K = val_probs.shape[1]

        def loss_fn(t_arr):
            t = t_arr[0]
            scaled_p = softmax(logits / t, axis=1)
            return log_loss(y_val, scaled_p, labels=list(range(K)))

        res = minimize(loss_fn, [1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(probs, 1e-7, 1.0))
        return softmax(logits / self.temperature, axis=1)

class PlattScaler:
    """Platt scaling via multiclass logistic regression on log-probabilities."""
    def __init__(self):
        self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray):
        logits = np.log(np.clip(val_probs, 1e-7, 1.0))
        self.model.fit(logits, y_val)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(probs, 1e-7, 1.0))
        return self.model.predict_proba(logits)

class IsotonicScaler:
    """Monotonic piecewise constant calibration per class with normalization."""
    def __init__(self, num_classes: int = 6):
        self.num_classes = num_classes
        self.models = [IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1.0) for _ in range(num_classes)]

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray):
        for k in range(self.num_classes):
            target_binary = (y_val == k).astype(float)
            self.models[k].fit(val_probs[:, k], target_binary)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        cal_p = np.column_stack([self.models[k].predict(probs[:, k]) for k in range(self.num_classes)])
        cal_p = np.clip(cal_p, 1e-6, 1.0)
        cal_p /= np.sum(cal_p, axis=1, keepdims=True)
        return cal_p

class CalibratedExpert:
    """Combines a fitted feature pipeline with a fitted probability calibrator."""
    def __init__(self, base_pipeline: Pipeline, calibrator: Any, method_name: str):
        self.base_pipeline = base_pipeline
        self.calibrator = calibrator
        self.method_name = method_name

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_probs = self.base_pipeline.predict_proba(X)
        if self.calibrator is None:
            return raw_probs
        return self.calibrator.predict_proba(raw_probs)

def plot_reliability_diagram(
    probs_uncal: np.ndarray,
    probs_cal: np.ndarray,
    y_true: np.ndarray,
    block_name: str,
    method_name: str,
    output_path: Path,
    n_bins: int = 15
):
    """
    Plots and saves 15-bin reliability diagram and confidence histogram.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for probs, label, color in [
        (probs_uncal, "Uncalibrated", "#ef4444"),
        (probs_cal, f"Calibrated ({method_name})", "#10b981")
    ]:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == y_true).astype(float)

        accs = []
        confs = []
        counts = []
        for i in range(n_bins):
            mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
            cnt = int(np.sum(mask))
            counts.append(cnt)
            if cnt > 0:
                accs.append(float(np.mean(accuracies[mask])))
                confs.append(float(np.mean(confidences[mask])))
            else:
                accs.append(np.nan)
                confs.append(bin_centers[i])

        ax1.plot(confs, accs, marker="o", linewidth=2, label=label, color=color)
        if label.startswith("Calibrated"):
            ax2.bar(bin_centers, counts, width=1.0 / n_bins, alpha=0.6, color=color, edgecolor="black")

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax1.set_ylabel("Observed Accuracy", fontsize=11)
    ax1.set_title(f"Reliability Diagram: Block '{block_name}'", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.set_xlabel("Confidence Bin", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_figure(fig, output_path, dpi=300)
    plt.close(fig)

def evaluate_and_calibrate_experts(
    feature_df: pd.DataFrame,
    seed: int = 42,
    figures_dir: Path = None,
    output_csv: Path = None
) -> Tuple[pd.DataFrame, Dict[str, CalibratedExpert]]:
    """
    Fits base RF models, tests Temperature Scaling, Platt, and Isotonic,
    selects the best method by validation NLL, and computes 15-bin ECE, Brier, NLL.
    """
    if figures_dir is None:
        figures_dir = PROJECT_ROOT / "figures"
    if output_csv is None:
        output_csv = PROJECT_ROOT / "results" / "calibration_summary.csv"

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df_train, df_val, df_test = split_dataset_by_group(feature_df, seed=seed)
    y_train = df_train["label"].map(CLASS_TO_IDX).values
    y_val = df_val["label"].map(CLASS_TO_IDX).values
    y_test = df_test["label"].map(CLASS_TO_IDX).values
    K = len(CLASSES)

    block_columns = {}
    sample_rgb = np.ones((32, 32, 3), dtype=np.float32)
    sample_mask = np.ones((32, 32), dtype=bool)
    for b, fn in BLOCK_EXTRACTORS.items():
        sample_out = fn(sample_rgb, sample_mask)
        block_columns[b] = [c for c in sample_out.keys() if c in feature_df.columns]

    records = []
    best_calibrated_experts = {}

    print(f"Calibrating experts across {len(block_columns)} blocks (Evaluating Platt, Temperature, Isotonic)...")

    for b, cols in block_columns.items():
        X_train = df_train[cols].values
        X_val = df_val[cols].values
        X_test = df_test[cols].values

        # Base Random Forest Pipeline
        base_rf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1
        )
        base_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("rf", base_rf)
        ])
        base_pipe.fit(X_train, y_train)

        # Raw probabilities
        val_p_uncal = base_pipe.predict_proba(X_val)
        test_p_uncal = base_pipe.predict_proba(X_test)

        uncal_val_nll = log_loss(y_val, val_p_uncal, labels=list(range(K)))
        uncal_test_nll = log_loss(y_test, test_p_uncal, labels=list(range(K)))
        uncal_val_ece = compute_ece_15(val_p_uncal, y_val, n_bins=15)
        uncal_test_ece = compute_ece_15(test_p_uncal, y_test, n_bins=15)
        uncal_val_brier = compute_brier_score(val_p_uncal, y_val)
        uncal_test_brier = compute_brier_score(test_p_uncal, y_test)

        # 1. Temperature Scaling
        temp_scaler = TemperatureScaler().fit(val_p_uncal, y_val)
        val_p_temp = temp_scaler.predict_proba(val_p_uncal)
        test_p_temp = temp_scaler.predict_proba(test_p_uncal)
        temp_val_nll = log_loss(y_val, val_p_temp, labels=list(range(K)))

        # 2. Platt (Sigmoid) Scaling
        platt_scaler = PlattScaler().fit(val_p_uncal, y_val)
        val_p_platt = platt_scaler.predict_proba(val_p_uncal)
        test_p_platt = platt_scaler.predict_proba(test_p_uncal)
        platt_val_nll = log_loss(y_val, val_p_platt, labels=list(range(K)))

        # 3. Isotonic Regression
        iso_scaler = IsotonicScaler(num_classes=K).fit(val_p_uncal, y_val)
        val_p_iso = iso_scaler.predict_proba(val_p_uncal)
        test_p_iso = iso_scaler.predict_proba(test_p_uncal)
        iso_val_nll = log_loss(y_val, val_p_iso, labels=list(range(K)))

        # Select best method by validation NLL
        method_nlls = {
            "isotonic": iso_val_nll,
            "platt": platt_val_nll,
            "temperature": temp_val_nll
        }
        best_method = min(method_nlls, key=method_nlls.get)

        if best_method == "isotonic":
            best_calibrator = iso_scaler
            val_p_best, test_p_best = val_p_iso, test_p_iso
        elif best_method == "platt":
            best_calibrator = platt_scaler
            val_p_best, test_p_best = val_p_platt, test_p_platt
        else:
            best_calibrator = temp_scaler
            val_p_best, test_p_best = val_p_temp, test_p_temp

        best_expert = CalibratedExpert(base_pipe, best_calibrator, best_method)
        best_calibrated_experts[b] = best_expert

        cal_test_nll = log_loss(y_test, test_p_best, labels=list(range(K)))
        cal_val_ece = compute_ece_15(val_p_best, y_val, n_bins=15)
        cal_test_ece = compute_ece_15(test_p_best, y_test, n_bins=15)
        cal_test_brier = compute_brier_score(test_p_best, y_test)

        # Plot reliability diagram
        fig_path = figures_dir / f"reliability_{b}.png"
        plot_reliability_diagram(test_p_uncal, test_p_best, y_test, b, best_method, fig_path, n_bins=15)

        records.append({
            "block": b,
            "best_method": best_method,
            "uncal_test_nll": round(uncal_test_nll, 4),
            "cal_test_nll": round(cal_test_nll, 4),
            "uncal_test_ece_15": round(uncal_test_ece, 4),
            "cal_test_ece_15": round(cal_test_ece, 4),
            "uncal_test_brier": round(uncal_test_brier, 4),
            "cal_test_brier": round(cal_test_brier, 4),
            "nll_improvement": round(uncal_test_nll - cal_test_nll, 4),
            "ece_improvement": round(uncal_test_ece - cal_test_ece, 4),
            "git_commit": get_git_commit_hash(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

    df_report = pd.DataFrame(records)
    df_report.to_csv(output_csv, index=False)
    print(f"Calibration summary saved to: {output_csv}")
    return df_report, best_calibrated_experts
