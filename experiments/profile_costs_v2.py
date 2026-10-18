"""Profiles per-particle wall-clock cost (ms) of segmentation, context descriptors Z and each block
as used in v2 (Shape on the Otsu mask; other blocks on the full crop). Median of 5 repeats per particle,
warm-up excluded. Writes data/block_costs_ms_v2.json and results/v2/block_costs_v2.csv."""
import sys, time, json, platform, os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.dataset import load_crop_index, load_crop_image
from src.segmentation import segment_particle
from src.context_prior import extract_context_prior_z
from src.features import extract_shape, extract_intensity, extract_colour, extract_texture, extract_frequency

def timeit(fn, reps=5):
    d = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); d.append((time.perf_counter() - t) * 1000)
    return float(np.median(d))

idx = load_crop_index()
sample = idx.groupby("label", group_keys=False).apply(lambda g: g.sample(20, random_state=0)).reset_index(drop=True)
x = np.random.RandomState(0).rand(64, 64, 3).astype(np.float32); m0 = segment_particle(x)
for _ in range(5):
    extract_context_prior_z(x, m0); [f(x, m0) for f in (extract_shape, extract_intensity, extract_colour, extract_texture, extract_frequency)]
T = {k: [] for k in ["segmentation", "context_z", "shape", "intensity", "colour", "texture", "frequency"]}
for _, r in sample.iterrows():
    rgb = load_crop_image(r.crop); m = segment_particle(rgb); full = np.ones(m.shape, bool)
    T["segmentation"].append(timeit(lambda: segment_particle(rgb)))
    T["context_z"].append(timeit(lambda: extract_context_prior_z(rgb, m)))
    T["shape"].append(timeit(lambda: extract_shape(rgb, m)))
    for k, f in (("intensity", extract_intensity), ("colour", extract_colour), ("texture", extract_texture), ("frequency", extract_frequency)):
        T[k].append(timeit(lambda: f(rgb, full)))
rows = [dict(component=k, median_ms=np.median(v), mean_ms=np.mean(v), sd_ms=np.std(v, ddof=1), n=len(v)) for k, v in T.items()]
out = ROOT / "results" / "v2"; out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).round(4).to_csv(out / "block_costs_v2.csv", index=False)
med = {k: round(float(np.median(v)), 4) for k, v in T.items()}
json.dump({"block_costs_ms": {b: med[b] for b in ["shape", "intensity", "colour", "texture", "frequency"]},
           "segmentation_ms": med["segmentation"], "context_z_ms": med["context_z"],
           "hardware": {"os": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count(),
                        "python": platform.python_version()}},
          open(ROOT / "data" / "block_costs_ms_v2.json", "w"), indent=1)
print(pd.DataFrame(rows).round(3))
