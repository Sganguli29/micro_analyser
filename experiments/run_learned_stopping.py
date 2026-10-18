"""
experiments/run_learned_stopping.py - CLI entry point to train and evaluate learned stopping policies.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.learned_stopping import train_learned_stopping_policy

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate learned stopping policies on trajectory data.")
    parser.add_argument("--label-option", type=str, default="a", choices=["a", "b"], help="Label definition: 'a' (accuracy agreement) or 'b' (cost-weighted).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "learned_stopping_report.csv"), help="Output CSV path.")
    args = parser.parse_args()

    df_out, models = train_learned_stopping_policy(
        label_option=args.label_option,
        seed=args.seed,
        output_csv=Path(args.output)
    )

    print("\n--- LEARNED STOPPING EVALUATION SUMMARY ---")
    print(df_out[["model", "label_option", "optimal_threshold", "test_accuracy", "test_f1", "test_auc", "test_precision", "test_recall"]].to_string(index=False))

if __name__ == "__main__":
    main()
