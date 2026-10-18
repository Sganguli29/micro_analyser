"""
src/cost_model.py - Real wall-clock extraction time profiling per feature block in milliseconds.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset import load_crop_index, load_crop_image, split_dataset_by_group
from src.features import BLOCK_EXTRACTORS, extract_all_blocks
from src.segmentation import segment_particle
from src.utils import get_system_hardware_info, PROJECT_ROOT

def measure_block_costs(
    sample_size: int = 150,
    repeats: int = 5,
    seed: int = 42,
    output_csv: Path = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Measures wall-clock extraction time in milliseconds per feature block per particle.
    
    - Excludes warm-up run.
    - Computes median over >= 3 repeats per crop (default 5 repeats).
    - Records hardware, OS, and library versions.
    - Outputs results/block_costs.csv with mean, median, std, min, max, and full-feature cost in ms.
    """
    if output_csv is None:
        output_csv = PROJECT_ROOT / "results" / "block_costs.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df_crops = load_crop_index()
    # Sample balanced particles across classes
    sample_df = df_crops.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(5, sample_size // 6)), random_state=seed)
    ).reset_index(drop=True)

    print(f"Profiling feature extraction costs on {len(sample_df)} particles ({repeats} repeats each)...")

    # 1. Warm-up
    print("Executing warm-up passes (excluded from timing)...")
    dummy_rgb = np.random.rand(64, 64, 3).astype(np.float32)
    dummy_mask = segment_particle(dummy_rgb)
    for _ in range(5):
        for b_name, fn in BLOCK_EXTRACTORS.items():
            fn(dummy_rgb, dummy_mask)
        extract_all_blocks(dummy_rgb, dummy_mask)

    # 2. Timing loop
    # block_timings_ms[block] = list of median timings per particle
    block_timings_ms = {b: [] for b in BLOCK_EXTRACTORS}
    all_blocks_timings_ms = []

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Profiling blocks"):
        rgb = load_crop_image(row["crop"])
        mask = segment_particle(rgb)

        # Measure each block individually
        for b_name, fn in BLOCK_EXTRACTORS.items():
            durations = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn(rgb, mask)
                t1 = time.perf_counter()
                durations.append((t1 - t0) * 1000.0)  # Convert to milliseconds
            block_timings_ms[b_name].append(float(np.median(durations)))

        # Measure all blocks together
        all_durations = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            extract_all_blocks(rgb, mask)
            t1 = time.perf_counter()
            all_durations.append((t1 - t0) * 1000.0)
        all_blocks_timings_ms.append(float(np.median(all_durations)))

    # 3. Compute statistics
    hw_info = get_system_hardware_info()
    records = []

    block_medians_dict = {}

    for b_name in BLOCK_EXTRACTORS:
        times = np.array(block_timings_ms[b_name])
        med_val = round(float(np.median(times)), 4)
        block_medians_dict[b_name] = med_val
        records.append({
            "block_name": b_name,
            "median_cost_ms": med_val,
            "mean_cost_ms": round(float(np.mean(times)), 4),
            "std_cost_ms": round(float(np.std(times, ddof=1)), 4),
            "min_cost_ms": round(float(np.min(times)), 4),
            "max_cost_ms": round(float(np.max(times)), 4),
            "n_particles_profiled": len(times),
            "repeats_per_particle": repeats,
            "hardware_cpu": hw_info["processor"],
            "hardware_ram_gb": hw_info["ram_total_gb"],
            "os": hw_info["os"],
            "python_version": hw_info["python_version"],
            "git_commit": hw_info["git_commit"]
        })

    # Full feature cost row
    all_times = np.array(all_blocks_timings_ms)
    full_cost_median = round(float(np.median(all_times)), 4)
    records.append({
        "block_name": "full_feature_set",
        "median_cost_ms": full_cost_median,
        "mean_cost_ms": round(float(np.mean(all_times)), 4),
        "std_cost_ms": round(float(np.std(all_times, ddof=1)), 4),
        "min_cost_ms": round(float(np.min(all_times)), 4),
        "max_cost_ms": round(float(np.max(all_times)), 4),
        "n_particles_profiled": len(all_times),
        "repeats_per_particle": repeats,
        "hardware_cpu": hw_info["processor"],
        "hardware_ram_gb": hw_info["ram_total_gb"],
        "os": hw_info["os"],
        "python_version": hw_info["python_version"],
        "git_commit": hw_info["git_commit"]
    })

    df_costs = pd.DataFrame(records)
    df_costs.to_csv(output_csv, index=False)
    print(f"Block costs in milliseconds saved to: {output_csv}")

    # Also save block_costs_ms.json for direct import across modules
    json_path = PROJECT_ROOT / "data" / "block_costs_ms.json"
    with open(json_path, "w") as f:
        json.dump({
            "block_costs_ms": block_medians_dict,
            "full_cost_ms": full_cost_median,
            "hardware": hw_info
        }, f, indent=2)

    return df_costs, block_medians_dict

def load_block_costs_ms() -> Dict[str, float]:
    """Loads measured median wall-clock extraction costs in milliseconds."""
    json_path = PROJECT_ROOT / "data" / "block_costs_ms.json"
    csv_path = PROJECT_ROOT / "results" / "block_costs.csv"
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
            return data["block_costs_ms"]
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        return dict(zip(df[df["block_name"] != "full_feature_set"]["block_name"],
                        df[df["block_name"] != "full_feature_set"]["median_cost_ms"]))
    else:
        # Run cost measurement if not already generated
        _, costs = measure_block_costs(sample_size=60, repeats=3)
        return costs
