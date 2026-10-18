"""
experiments/run_morphology.py - CLI entry point for the morphology audit.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.morphology_audit import audit_morphology

def main():
    parser = argparse.ArgumentParser(description="Audit morphology descriptors: Bounding-box vs Segmented Mask.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for rapid testing.")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "morphology_bbox_vs_mask.csv"), help="Output CSV path.")
    args = parser.parse_args()

    df_out, res = audit_morphology(
        seed=args.seed,
        max_samples=args.max_samples,
        output_csv=Path(args.output)
    )

    print("\n--- MORPHOLOGY AUDIT SUMMARY ---")
    print(df_out[df_out["descriptor"].str.startswith("shape_")][["descriptor", "mean_bbox", "mean_mask", "mean_abs_diff", "spearman_corr"]].to_string(index=False))

if __name__ == "__main__":
    main()
