r"""
src/stats.py - Rigorous statistical hypothesis testing: McNemar test and Paired Bootstrap.

Non-negotiable research integrity:
- McNemar test with continuity correction between model predictions on identical test samples.
- Paired bootstrap (10,000 resamples) for accuracy, macro-F1, and wall-clock cost differences.
- Computes empirical 95% bootstrap confidence intervals and exact p-values.
- Outputs results/significance.csv.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple, Callable, Optional
import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

from src.utils import get_git_commit_hash, PROJECT_ROOT

def compute_mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray
) -> Tuple[float, float, int, int]:
    r"""
    Performs McNemar's test for paired binary classification outcomes.
    
    Formula:
        b = \sum [ y_A == y_{true} \land y_B \neq y_{true} ]
        c = \sum [ y_A \neq y_{true} \land y_B == y_{true} ]
        \chi^2 = \frac{(|b - c| - 1)^2}{b + c}
        
    Returns:
        (chi2_stat, p_value, b, c)
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    b = int(np.sum(correct_a & ~correct_b)) # A correct, B incorrect
    c = int(np.sum(~correct_a & correct_b)) # A incorrect, B correct

    total_discordant = b + c
    if total_discordant == 0:
        return 0.0, 1.0, b, c

    if total_discordant < 25:
        # Exact binomial test
        p_val = float(2.0 * stats.binom.cdf(min(b, c), total_discordant, 0.5))
        chi2_stat = float((b - c) ** 2 / total_discordant)
    else:
        # Edwards continuity correction
        chi2_stat = float(((abs(b - c) - 1.0) ** 2) / total_discordant)
        p_val = float(stats.chi2.sf(chi2_stat, df=1))

    return chi2_stat, p_val, b, c

def compute_paired_bootstrap(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    costs_a: Optional[np.ndarray] = None,
    costs_b: Optional[np.ndarray] = None,
    n_resamples: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    r"""
    Computes paired bootstrap resampling (10,000 resamples) between system A and system B.
    
    Returns 95% empirical bootstrap confidence intervals and empirical p-values for:
    - \Delta Accuracy = Acc(A) - Acc(B)
    - \Delta Macro-F1 = F1(A) - F1(B)
    - \Delta Cost (ms) = Cost(A) - Cost(B) (if costs provided)
    """
    N = len(y_true)
    rng = np.random.RandomState(seed)

    diff_acc = []
    diff_f1 = []
    diff_cost = []

    for _ in range(n_resamples):
        boot_idx = rng.randint(0, N, size=N)
        y_b = y_true[boot_idx]
        pa_b = preds_a[boot_idx]
        pb_b = preds_b[boot_idx]

        acc_a = np.mean(pa_b == y_b)
        acc_b = np.mean(pb_b == y_b)
        diff_acc.append(acc_a - acc_b)

        f1_a = f1_score(y_b, pa_b, average="macro", zero_division=0)
        f1_b = f1_score(y_b, pb_b, average="macro", zero_division=0)
        diff_f1.append(f1_a - f1_b)

        if costs_a is not None and costs_b is not None:
            ca_b = np.mean(costs_a[boot_idx])
            cb_b = np.mean(costs_b[boot_idx])
            diff_cost.append(ca_b - cb_b)

    diff_acc = np.array(diff_acc)
    diff_f1 = np.array(diff_f1)

    # 95% CI
    ci_acc = (float(np.percentile(diff_acc, 2.5)), float(np.percentile(diff_acc, 97.5)))
    ci_f1 = (float(np.percentile(diff_f1, 2.5)), float(np.percentile(diff_f1, 97.5)))

    # Two-sided bootstrap p-value: 2 * min(P(diff <= 0), P(diff >= 0))
    p_acc = float(2.0 * min(np.mean(diff_acc <= 0), np.mean(diff_acc >= 0)))
    p_f1 = float(2.0 * min(np.mean(diff_f1 <= 0), np.mean(diff_f1 >= 0)))

    res = {
        "delta_acc_mean": float(np.mean(diff_acc)),
        "delta_acc_ci95_low": ci_acc[0],
        "delta_acc_ci95_high": ci_acc[1],
        "p_val_acc_boot": min(1.0, p_acc),
        "delta_f1_mean": float(np.mean(diff_f1)),
        "delta_f1_ci95_low": ci_f1[0],
        "delta_f1_ci95_high": ci_f1[1],
        "p_val_f1_boot": min(1.0, p_f1),
    }

    if costs_a is not None and costs_b is not None:
        diff_cost = np.array(diff_cost)
        ci_cost = (float(np.percentile(diff_cost, 2.5)), float(np.percentile(diff_cost, 97.5)))
        p_cost = float(2.0 * min(np.mean(diff_cost <= 0), np.mean(diff_cost >= 0)))
        res.update({
            "delta_cost_mean_ms": float(np.mean(diff_cost)),
            "delta_cost_ci95_low": ci_cost[0],
            "delta_cost_ci95_high": ci_cost[1],
            "p_val_cost_boot": min(1.0, p_cost)
        })

    return res

def run_significance_suite(
    comparisons: List[Dict[str, Any]],
    output_csv: Optional[Path] = None
) -> pd.DataFrame:
    r"""
    Executes McNemar test and 10,000 paired bootstrap resamples for all paired comparisons.
    
    Args:
        comparisons: List of dicts with keys:
            - name_a: str
            - name_b: str
            - y_true: np.ndarray
            - preds_a: np.ndarray
            - preds_b: np.ndarray
            - costs_a: Optional[np.ndarray]
            - costs_b: Optional[np.ndarray]
            
    Returns:
        DataFrame saved to results/significance.csv.
    """
    if output_csv is None:
        output_csv = PROJECT_ROOT / "results" / "significance.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"\n========================================================")
    print(f"RUNNING STATISTICAL SIGNIFICANCE TESTS (10,000 BOOTSTRAPS)")
    print(f"========================================================")

    for comp in comparisons:
        na = comp["name_a"]
        nb = comp["name_b"]
        y_true = comp["y_true"]
        pa = comp["preds_a"]
        pb = comp["preds_b"]
        ca = comp.get("costs_a")
        cb = comp.get("costs_b")

        chi2, p_mcnemar, b, c = compute_mcnemar_test(y_true, pa, pb)
        boot = compute_paired_bootstrap(y_true, pa, pb, ca, cb, n_resamples=10000, seed=42)

        row = {
            "model_a": na,
            "model_b": nb,
            "mcnemar_chi2": round(chi2, 4),
            "mcnemar_p_value": round(p_mcnemar, 6),
            "mcnemar_b": b,
            "mcnemar_c": c,
            "delta_acc_mean": round(boot["delta_acc_mean"], 4),
            "delta_acc_ci95": f"[{boot['delta_acc_ci95_low']:.4f}, {boot['delta_acc_ci95_high']:.4f}]",
            "p_val_acc_boot": round(boot["p_val_acc_boot"], 6),
            "delta_f1_mean": round(boot["delta_f1_mean"], 4),
            "delta_f1_ci95": f"[{boot['delta_f1_ci95_low']:.4f}, {boot['delta_f1_ci95_high']:.4f}]",
            "p_val_f1_boot": round(boot["p_val_f1_boot"], 6),
            "delta_cost_mean_ms": round(boot.get("delta_cost_mean_ms", 0.0), 3),
            "delta_cost_ci95": f"[{boot.get('delta_cost_ci95_low', 0.0):.3f}, {boot.get('delta_cost_ci95_high', 0.0):.3f}]",
            "p_val_cost_boot": round(boot.get("p_val_cost_boot", 1.0), 6),
            "git_commit": get_git_commit_hash(),
            "timestamp": datetime.datetime.now().isoformat()
        }
        rows.append(row)
        print(f"  {na} vs {nb}:")
        print(f"    McNemar p={p_mcnemar:.5f} (b={b}, c={c})")
        print(f"    Delta Acc: {boot['delta_acc_mean']:+.4f} (95% CI: [{boot['delta_acc_ci95_low']:.4f}, {boot['delta_acc_ci95_high']:.4f}], p={boot['p_val_acc_boot']:.5f})")
        if "delta_cost_mean_ms" in boot:
            print(f"    Delta Cost: {boot['delta_cost_mean_ms']:+.3f} ms (p={boot['p_val_cost_boot']:.5f})")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"\nSignificance tests successfully written to {output_csv}")
    return df_out
