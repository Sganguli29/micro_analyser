"""
experiments/run_open_set.py - Entry point for LOPO Open-Set and OOD polymer classification.
"""

import argparse
from pathlib import Path
import sys

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.open_set import run_lopo_open_set_evaluation

def main():
    parser = argparse.ArgumentParser(description="Leave-One-Polymer-Out (LOPO) Open-Set Evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="results/open_set_lopo.csv", help="Output CSV path")
    args = parser.parse_args()

    run_lopo_open_set_evaluation(seed=args.seed, output_csv=Path(args.output))

if __name__ == "__main__":
    main()
