"""
experiments/run_context_prior.py - CLI entry point for the Context Prior P(Y|Z) experiment.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.context_prior import run_context_prior_experiment

def main():
    parser = argparse.ArgumentParser(description="Evaluate Context Prior P(Y|Z) and double-counting ablation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "context_prior_report.csv"), help="Output CSV path.")
    args = parser.parse_args()

    df_out, _ = run_context_prior_experiment(
        seed=args.seed,
        output_csv=Path(args.output)
    )

    print("\n--- CONTEXT PRIOR REPORT ---")
    print(df_out[["experiment", "feature_set", "test_accuracy", "test_macro_f1", "test_nll"]].to_string(index=False))

if __name__ == "__main__":
    main()
