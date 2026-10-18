"""experiments/overhead_v2.py - Decision overhead, cost vs. particle size, and block x polymer accuracy.

Outputs (results/v2_analysis/):
  overhead.csv          per-decision compute time of EIG scoring, fusion and stopping classifiers (ms)
  cost_vs_size.csv      per-particle extraction time per block with crop size (120 particles)
  block_class_acc.csv   single-block test accuracy per polymer (seeds 0-4)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from src.dataset import split_dataset_by_group, CLASSES, load_crop_index, load_crop_image
from src.adaptive_v2 import BLOCKS, Z_COLS, block_columns, Trajectories, state_vector
from src.segmentation import segment_particle
from src.features import extract_shape, extract_intensity, extract_colour, extract_texture, extract_frequency
from run_v2 import setup, train_stopper, load_costs

OUT = ROOT / "results" / "v2_analysis"


def tmed(fn, reps=200):
    fn()
    d = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); d.append((time.perf_counter() - t) * 1000)
    return float(np.median(d))


def main(cost_only=False):
    feats = pd.read_csv(ROOT / "data/features_cache_v2.csv")
    df = feats.merge(pd.read_csv(ROOT / "data/context_prior_cache.csv")[["crop"] + Z_COLS], on="crop")
    cols = block_columns(feats)
    costs, seg_ms, z_ms = load_costs(ROOT / "data/block_costs_ms_v2.json")
    acc_rows, ov = [], []
    for seed in ([] if cost_only else range(5)):
        tr, va, te = split_dataset_by_group(df, seed=seed)
        S = setup(df, CLASSES, seed, cols, costs, seg_ms, z_ms, tr, va, [te])
        cmap = {c: i for i, c in enumerate(CLASSES)}
        y = te.label.map(cmap).values
        for b in BLOCKS + ["context"]:
            pred = S["tests"][0][b].argmax(1)
            for k, c in enumerate(CLASSES):
                acc_rows.append(dict(seed=seed, block=b, polymer=c, acc=float(np.mean(pred[y == k] == k))))
        if seed == 0:
            env = S["env"]["ctx"]
            row = {b: S["tests"][0][b][0] for b in BLOCKS + ["context"]}
            logit = env["fusion"].logits(row, ["context"])
            ov.append(dict(component="EIG/Cost scoring, 5 candidate blocks", ms=tmed(lambda: env["eig"].scores(logit, list(BLOCKS), costs))))
            ov.append(dict(component="EIG/Cost scoring, 2 candidate blocks", ms=tmed(lambda: env["eig"].scores(logit, BLOCKS[:2], costs))))
            ov.append(dict(component="Fusion update (log-opinion pool)", ms=tmed(lambda: env["fusion"].logits(row, BLOCKS + ["context"]))))
            tr_traj = Trajectories(S["oof"], S["Z_tr"], **env)
            models = train_stopper(S, seed, tr_traj)
            s = state_vector(np.ones(6) / 6, 1, 1.0, 4, te[Z_COLS].values[0])[None, :]
            rf = models["random_forest"]
            ov.append(dict(component="Stopping RF (300 trees), n_jobs=2", ms=tmed(lambda: rf.predict_proba(s), 50)))
            rf.set_params(n_jobs=1)
            ov.append(dict(component="Stopping RF (300 trees), n_jobs=1", ms=tmed(lambda: rf.predict_proba(s), 50)))
            lr = models["logistic_regression"]
            ov.append(dict(component="Stopping logistic regression", ms=tmed(lambda: lr.predict_proba(s))))
            ov.append(dict(component="Confidence threshold test", ms=tmed(lambda: float(np.max(np.ones(6) / 6)) >= 0.7)))
            pd.DataFrame(ov).to_csv(OUT / "overhead.csv", index=False)
            print(pd.DataFrame(ov), flush=True)
        print("seed", seed, flush=True)
    if not cost_only:
        pd.DataFrame(acc_rows).to_csv(OUT / "block_class_acc.csv", index=False)
    # cost vs size
    idx = load_crop_index()
    sample = idx.groupby("label").sample(20, random_state=0).reset_index(drop=True)
    rows = []
    for _, r in sample.iterrows():
        rgb = load_crop_image(r.crop); m = segment_particle(rgb); full = np.ones(m.shape, bool)
        d = dict(crop=r.crop, label=r.label, pixels=int(rgb.shape[0] * rgb.shape[1]))
        d["segmentation"] = tmed(lambda: segment_particle(rgb), 5)
        d["shape"] = tmed(lambda: extract_shape(rgb, m), 5)
        for k, f in (("intensity", extract_intensity), ("colour", extract_colour), ("texture", extract_texture), ("frequency", extract_frequency)):
            d[k] = tmed(lambda: f(rgb, full), 5)
        rows.append(d)
    pd.DataFrame(rows).to_csv(OUT / "cost_vs_size.csv", index=False)
    print("done")


if __name__ == "__main__":
    main(cost_only="--cost-only" in sys.argv)
