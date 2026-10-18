"""experiments/lopo_assign_v2.py - Which known polymer is an unseen polymer assigned to (all-block posterior, seeds 0-2)?
Output: results/v2_analysis/lopo_assignment.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import softmax
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from src.dataset import split_dataset_by_group, CLASSES
from src.adaptive_v2 import BLOCKS, Z_COLS, block_columns
from run_v2 import setup, load_costs

feats = pd.read_csv(ROOT / "data/features_cache_v2.csv")
df = feats.merge(pd.read_csv(ROOT / "data/context_prior_cache.csv")[["crop"] + Z_COLS], on="crop")
cols = block_columns(feats); costs, seg, zc = load_costs(ROOT / "data/block_costs_ms_v2.json")
rows = []
for seed in range(3):
    for held in CLASSES:
        known = [c for c in CLASSES if c != held]
        dk = df[df.label != held].reset_index(drop=True); dood = df[df.label == held].reset_index(drop=True)
        tr, va, te = split_dataset_by_group(dk, seed=seed)
        S = setup(df, known, seed, cols, costs, seg, zc, tr, va, [te, dood])
        f = S["env"]["ctx"]["fusion"]; names = BLOCKS + ["context"]
        P = softmax(np.array([f.logits({b: S["tests"][1][b][i] for b in names}, names) for i in range(len(dood))]), 1)
        for k, c in enumerate(known):
            rows.append(dict(seed=seed, held_out=held, assigned=c, frac=float(np.mean(P.argmax(1) == k)),
                             mean_conf=float(P.max(1)[P.argmax(1) == k].mean()) if (P.argmax(1) == k).any() else np.nan))
        print(seed, held, flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/v2_analysis/lopo_assignment.csv", index=False)
