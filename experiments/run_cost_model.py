"""
experiments/run_cost_model.py - CLI entry point to measure real wall-clock block extraction times.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cost_model import measure_block_costs

def main():
    parser = argparse.ArgumentParser(description="Measure wall-clock feature extraction costs in ms.")
    parser.add_argument("--sample-size", type=int, default=120, help="Number of particles to profile.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per particle.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for particle selection.")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "block_costs.csv"), help="Output CSV path.")
    args = parser.parse_args()

    df_costs, costs_dict = measure_block_costs(
        sample_size=args.sample_size,
        repeats=args.repeats,
        seed=args.seed,
        output_csv=Path(args.output)
    )

    print("\n--- MEASURED REAL WALL-CLOCK EXTRACTION COSTS (ms) ---")
    print(df_costs[["block_name", "median_cost_ms", "mean_cost_ms", "std_cost_ms", "min_cost_ms", "max_cost_ms"]].to_string(index=False))

if __name__ == "__main__":
    main()
