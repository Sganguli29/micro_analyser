r"""
src/diagnostics.py - Experimental diagnostics, error analysis, and publication figures.

Produces:
1. Confusion matrices per configuration (figures/confusion_<config>.png & .pdf at 300 dpi).
2. Histogram of stopping steps (figures/stopping_step_histogram.png & .pdf).
3. First-block selection heatmap per polymer (figures/first_block_heatmap.png & .pdf).
4. Per-class error table (results/per_class_error_table.csv).
5. 20 worst misclassified particle images with their step-by-step acquisition trajectories
   (saved to results/diagnostics/worst_particles/ and results/diagnostics/worst_misclassifications.csv).
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from src.dataset import CLASSES, CLASS_TO_IDX, load_crop_image
from src.utils import save_figure, PROJECT_ROOT

def plot_confusion_matrices(
    conf_matrices: Dict[str, np.ndarray],
    figures_dir: Optional[Path] = None
):
    """Generates and saves publication-quality 300 dpi confusion matrix figures."""
    if figures_dir is None:
        figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for cfg_name, cm in conf_matrices.items():
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        # Row-normalized percentages
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0) * 100.0
        
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            cbar_kws={"label": "Percentage (%)"},
            ax=ax
        )
        ax.set_title(f"Confusion Matrix: {cfg_name.replace('_', ' ').title()}", fontsize=12, pad=10)
        ax.set_xlabel("Predicted Polymer", fontsize=11)
        ax.set_ylabel("True Polymer", fontsize=11)
        plt.tight_layout()

        out_base = figures_dir / f"confusion_{cfg_name}"
        save_figure(fig, out_base)
        plt.close(fig)

def plot_stopping_histogram(
    n_blocks_by_config: Dict[str, np.ndarray],
    figures_dir: Optional[Path] = None
):
    """Plots histogram of stopping steps across configurations."""
    if figures_dir is None:
        figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    bins = np.arange(0.5, 6.5, 1)

    for cfg_name, nb in n_blocks_by_config.items():
        counts, _ = np.histogram(nb, bins=bins)
        pcts = counts / len(nb) * 100.0
        steps = np.arange(1, 6)
        ax.plot(steps, pcts, marker="o", linewidth=2, label=cfg_name.replace("_", " ").title())

    ax.set_xlabel("Stopping Step (Number of Blocks Acquired)", fontsize=11)
    ax.set_ylabel("Particles Stopped (%)", fontsize=11)
    ax.set_title("Distribution of Acquisition Stopping Steps", fontsize=12, pad=10)
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()

    out_base = figures_dir / "stopping_step_histogram"
    save_figure(fig, out_base)
    plt.close(fig)

def plot_first_block_heatmap(
    y_true: np.ndarray,
    first_blocks: List[str],
    all_blocks: List[str],
    figures_dir: Optional[Path] = None
):
    r"""
    Heatmap of which block is chosen first, conditioned on true polymer.
    Demonstrates that adaptive policies tailor initial acquisition to particle context.
    """
    if figures_dir is None:
        figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.zeros((len(CLASSES), len(all_blocks)), dtype=float)
    fb_array = np.array(first_blocks)

    for c_idx, c_name in enumerate(CLASSES):
        c_mask = (y_true == c_idx)
        total_c = np.sum(c_mask)
        if total_c > 0:
            for b_idx, b_name in enumerate(all_blocks):
                matrix[c_idx, b_idx] = np.mean(fb_array[c_mask] == b_name) * 100.0

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=[b.title() for b in all_blocks],
        yticklabels=CLASSES,
        cbar_kws={"label": "Initial Block Choice (%)"},
        ax=ax
    )
    ax.set_title("Initial Feature Block Choice by True Polymer", fontsize=12, pad=10)
    ax.set_xlabel("First Acquired Block", fontsize=11)
    ax.set_ylabel("True Polymer", fontsize=11)
    plt.tight_layout()

    out_base = figures_dir / "first_block_heatmap"
    save_figure(fig, out_base)
    plt.close(fig)

def save_worst_misclassifications(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    preds: np.ndarray,
    confidences: np.ndarray,
    trajectories: List[List[Dict[str, Any]]],
    output_dir: Optional[Path] = None,
    n_worst: int = 20
) -> pd.DataFrame:
    """
    Identifies and saves the 20 worst misclassified particles with high confidence,
    annotating their crop images and trajectory logs.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "diagnostics" / "worst_particles"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Errors
    err_mask = (preds != y_true) & (preds >= 0)
    err_indices = np.where(err_mask)[0]

    # Sort descending by confidence (most confident errors)
    sorted_err = err_indices[np.argsort(confidences[err_indices])[::-1]]
    top_worst = sorted_err[:min(n_worst, len(sorted_err))]

    records = []

    for rank, idx in enumerate(top_worst):
        row = df_test.iloc[idx]
        crop_rel = row["crop"]
        true_c = CLASSES[y_true[idx]]
        pred_c = CLASSES[preds[idx]]
        conf = float(confidences[idx])
        traj = trajectories[idx]

        # Load image
        img_rgb = load_crop_image(crop_rel)
        img_bgr = cv2.cvtColor((img_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Annotate
        h, w = img_bgr.shape[:2]
        canvas = np.zeros((h + 60, max(w, 200), 3), dtype=np.uint8)
        canvas[:h, :w] = img_bgr
        cv2.putText(canvas, f"True: {true_c}", (10, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(canvas, f"Pred: {pred_c} ({conf*100:.1f}%)", (10, h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out_img_name = f"rank{rank+1:02d}_{true_c}_as_{pred_c}_{Path(crop_rel).stem}.png"
        cv2.imwrite(str(output_dir / out_img_name), canvas)

        records.append({
            "rank": rank + 1,
            "crop": crop_rel,
            "true_polymer": true_c,
            "predicted_polymer": pred_c,
            "confidence": round(conf, 4),
            "n_blocks": len(traj),
            "trajectory": json.dumps(traj),
            "image_path": str(output_dir / out_img_name)
        })

    df_worst = pd.DataFrame(records)
    csv_path = PROJECT_ROOT / "results" / "diagnostics" / "worst_misclassifications.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_worst.to_csv(csv_path, index=False)
    print(f"Saved {len(records)} worst misclassified images to {output_dir}")
    return df_worst
