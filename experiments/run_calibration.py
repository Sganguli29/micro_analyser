"""
experiments/run_calibration.py - CLI entry point to calibrate experts and generate reliability diagrams.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import evaluate_and_calibrate_experts

def main():
    parser = argparse.ArgumentParser(description="Calibrate per-block experts and plot reliability diagrams.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cache", type=str, default=str(PROJECT_ROOT / "data" / "features_cache.csv"), help="Feature cache CSV.")
    parser.add_argument("--figures-dir", type=str, default=str(PROJECT_ROOT / "figures"), help="Directory to save reliability diagrams.")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "calibration_summary.csv"), help="Output summary CSV.")
    args = parser.parse_args()

    feat_df = pd.read_csv(args.cache)
    df_report, experts = evaluate_and_calibrate_experts(
        feature_df=feat_df,
        seed=args.seed,
        figures_dir=Path(args.figures_dir),
        output_csv=Path(args.output)
    )

    print("\n--- EXPERT CALIBRATION REPORT (ECE 15-bins, Brier, NLL) ---")
    print(df_report[["block", "best_method", "uncal_test_nll", "cal_test_nll", "uncal_test_ece_15", "cal_test_ece_15", "uncal_test_brier", "cal_test_brier"]].to_string(index=False))

if __name__ == "__main__":
    main()
