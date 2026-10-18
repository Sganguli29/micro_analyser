"""
src/dataset_report.py - Dataset summary generation and resolution statistics.
"""

from pathlib import Path
from typing import Dict, Any
import argparse
import datetime
import pandas as pd
import numpy as np

from src.dataset import load_crop_index, split_dataset_by_group, CLASSES
from src.utils import get_git_commit_hash, PROJECT_ROOT

def generate_dataset_summary(
    crop_index_path: Path = None,
    seed: int = 42,
    output_csv: Path = None
) -> pd.DataFrame:
    """
    Computes dataset statistics:
    - Total particles and per-class distribution
    - Train / Val / Test split counts and group counts
    - Resolution statistics (min, max, mean, median for width, height, and area)
    
    Saves results to results/dataset_summary.csv.
    """
    if output_csv is None:
        output_csv = PROJECT_ROOT / "results" / "dataset_summary.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = load_crop_index(crop_index_path)
    df_train, df_val, df_test = split_dataset_by_group(df, seed=seed)

    total_particles = len(df)
    unique_slides = df["source_image"].nunique()

    # Per-class counts
    class_counts = df["label"].value_counts().to_dict()

    # Resolution statistics
    w = df["w"].values
    h = df["h"].values
    area = w * h

    records = [
        {"metric_category": "metadata", "metric_name": "git_commit", "value": get_git_commit_hash(), "unit": "hash"},
        {"metric_category": "metadata", "metric_name": "timestamp_utc", "value": datetime.datetime.now(datetime.timezone.utc).isoformat(), "unit": "iso8601"},
        {"metric_category": "metadata", "metric_name": "split_seed", "value": seed, "unit": "integer"},
        {"metric_category": "dataset_totals", "metric_name": "total_particles", "value": total_particles, "unit": "count"},
        {"metric_category": "dataset_totals", "metric_name": "total_unique_source_slides", "value": unique_slides, "unit": "count"},
    ]

    # Per-class counts
    for c in CLASSES:
        cnt = class_counts.get(c, 0)
        pct = round(100.0 * cnt / total_particles, 2)
        records.append({
            "metric_category": "class_distribution",
            "metric_name": f"count_{c}",
            "value": cnt,
            "unit": f"count ({pct}%)"
        })

    # Split sizes
    splits = [("train", df_train), ("val", df_val), ("test", df_test)]
    for name, split_df in splits:
        n_crops = len(split_df)
        n_slides = split_df["source_image"].nunique()
        pct = round(100.0 * n_crops / total_particles, 2)
        records.extend([
            {"metric_category": "split_sizes", "metric_name": f"{name}_crop_count", "value": n_crops, "unit": f"count ({pct}%)"},
            {"metric_category": "split_sizes", "metric_name": f"{name}_source_slide_count", "value": n_slides, "unit": "count"},
        ])

    # Resolution stats
    res_metrics = [
        ("width_min", np.min(w), "pixels"),
        ("width_max", np.max(w), "pixels"),
        ("width_mean", round(float(np.mean(w)), 2), "pixels"),
        ("width_median", round(float(np.median(w)), 2), "pixels"),
        ("width_std", round(float(np.std(w)), 2), "pixels"),
        ("height_min", np.min(h), "pixels"),
        ("height_max", np.max(h), "pixels"),
        ("height_mean", round(float(np.mean(h)), 2), "pixels"),
        ("height_median", round(float(np.median(h)), 2), "pixels"),
        ("height_std", round(float(np.std(h)), 2), "pixels"),
        ("area_min", np.min(area), "pixels^2"),
        ("area_max", np.max(area), "pixels^2"),
        ("area_mean", round(float(np.mean(area)), 2), "pixels^2"),
        ("area_median", round(float(np.median(area)), 2), "pixels^2"),
        ("area_std", round(float(np.std(area)), 2), "pixels^2"),
    ]
    for m_name, val, unit in res_metrics:
        records.append({
            "metric_category": "resolution_statistics",
            "metric_name": m_name,
            "value": val,
            "unit": unit
        })

    df_summary = pd.DataFrame(records)
    df_summary.to_csv(output_csv, index=False)
    print(f"Dataset summary successfully written to: {output_csv}")
    return df_summary

def main():
    parser = argparse.ArgumentParser(description="Generate dataset summary report.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group split.")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "dataset_summary.csv"), help="Output CSV path.")
    args = parser.parse_args()

    summary_df = generate_dataset_summary(seed=args.seed, output_csv=Path(args.output))
    print("\n--- DATASET SUMMARY HIGHLIGHTS ---")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
