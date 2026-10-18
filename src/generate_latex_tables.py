r"""
src/generate_latex_tables.py - Generates results/tables.tex directly from results/*.csv.

Research Integrity:
- Never hardcodes or invents any table entry.
- Reads every number directly from computed CSV files in results/.
- Formats into clean LaTeX booktabs tables ready for publication.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import PROJECT_ROOT

def generate_latex_tables(output_tex: Path = None):
    if output_tex is None:
        output_tex = PROJECT_ROOT / "results" / "tables.tex"
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    results_dir = PROJECT_ROOT / "results"
    tex_sections = []

    tex_sections.append(r"% Auto-generated publication tables from microplastics_demo experimental results")
    tex_sections.append(r"% Generated directly from code-computed CSV files in results/ (Zero manual transcription)")
    tex_sections.append(r"\usepackage{booktabs}")
    tex_sections.append(r"\usepackage{siunitx}")
    tex_sections.append(r"\usepackage{multirow}")
    tex_sections.append("")

    # Table 1: Dataset Summary
    p_ds = results_dir / "dataset_summary.csv"
    if p_ds.exists():
        df_ds = pd.read_csv(p_ds)
        tex_sections.append(r"% Table 1: Dataset Partition Summary")
        tex_sections.append(r"\begin{table}[ht]")
        tex_sections.append(r"\centering")
        tex_sections.append(r"\caption{Microplastic polymer dataset summary across group-stratified splits.}")
        tex_sections.append(r"\label{tab:dataset_summary}")
        tex_sections.append(df_ds.to_latex(index=False, float_format="%.2f"))
        tex_sections.append(r"\end{table}")
        tex_sections.append("")

    # Table 2: Measured Block Costs
    p_cost = results_dir / "block_costs.csv"
    if p_cost.exists():
        df_cost = pd.read_csv(p_cost)
        tex_sections.append(r"% Table 2: Wall-Clock Extraction Costs (ms)")
        tex_sections.append(r"\begin{table}[ht]")
        tex_sections.append(r"\centering")
        tex_sections.append(r"\caption{Empirical wall-clock feature extraction time per particle (milliseconds).}")
        tex_sections.append(r"\label{tab:block_costs}")
        cols_show = [c for c in ["block", "median_ms", "mean_ms", "std_ms", "relative_cost_pct"] if c in df_cost.columns]
        tex_sections.append(df_cost[cols_show].to_latex(index=False, float_format="%.2f"))
        tex_sections.append(r"\end{table}")
        tex_sections.append("")

    # Table 3: Calibration & Reliability
    p_cal = results_dir / "calibration_summary.csv"
    if p_cal.exists():
        df_cal = pd.read_csv(p_cal)
        tex_sections.append(r"% Table 3: Calibration Performance per Feature Block")
        tex_sections.append(r"\begin{table}[ht]")
        tex_sections.append(r"\centering")
        tex_sections.append(r"\caption{Expected Calibration Error (15 bins), Brier score, and NLL before and after calibration.}")
        tex_sections.append(r"\label{tab:calibration}")
        cols_show = [c for c in ["block", "calibrator", "test_acc", "ece_raw", "ece_cal", "brier_raw", "brier_cal", "nll_raw", "nll_cal"] if c in df_cal.columns]
        tex_sections.append(df_cal[cols_show].to_latex(index=False, float_format="%.4f"))
        tex_sections.append(r"\end{table}")
        tex_sections.append("")

    # Table 4: Morphology Audit (BBox vs Mask)
    p_morph = results_dir / "morphology_bbox_vs_mask.csv"
    if p_morph.exists():
        df_morph = pd.read_csv(p_morph)
        tex_sections.append(r"% Table 4: Morphology Audit: Bounding Box Crop vs Segmented Mask")
        tex_sections.append(r"\begin{table}[ht]")
        tex_sections.append(r"\centering")
        tex_sections.append(r"\caption{Comparison of shape descriptors and classification accuracy computed from BBox crops vs. Otsu segmented masks.}")
        tex_sections.append(r"\label{tab:morphology_audit}")
        cols_show = [c for c in ["feature", "bbox_mean", "mask_mean", "mean_abs_diff", "paired_p_val"] if c in df_morph.columns]
        tex_sections.append(df_morph[cols_show].to_latex(index=False, float_format="%.4f"))
        tex_sections.append(r"\end{table}")
        tex_sections.append("")

    # Table 5: Open-Set LOPO Benchmark
    p_lopo = results_dir / "open_set_lopo.csv"
    if p_lopo.exists():
        df_lopo = pd.read_csv(p_lopo)
        tex_sections.append(r"% Table 5: Open-Set Leave-One-Polymer-Out (LOPO) Benchmark")
        tex_sections.append(r"\begin{table}[ht]")
        tex_sections.append(r"\centering")
        tex_sections.append(r"\caption{Leave-One-Polymer-Out open-set polymer detection across 6 folds.}")
        tex_sections.append(r"\label{tab:open_set_lopo}")
        agg_lopo = df_lopo.groupby("method").agg({
            "auroc": "mean",
            "aupr": "mean",
            "fpr_at_95_tpr": "mean",
            "unknown_detection_rate": "mean",
            "false_unknown_rate": "mean",
            "closed_set_accuracy": "mean"
        }).reset_index()
        tex_sections.append(agg_lopo.to_latex(index=False, float_format="%.4f"))
        tex_sections.append(r"\end{table}")
        tex_sections.append("")

    # Table 6: 4-Configuration Ablation Study
    p_abl = results_dir / "ablation_study.csv"
    if p_abl.exists():
        df_abl = pd.read_csv(p_abl)
        tex_sections.append(r"% Table 6: System Configuration and Component Ablation Study")
        tex_sections.append(r"\begin{table}[ht]")
        tex_sections.append(r"\centering")
        tex_sections.append(r"\caption{Ablation analysis of primary configurations and context prior components.}")
        tex_sections.append(r"\label{tab:ablation_study}")
        cols_show = [c for c in ["configuration", "accuracy", "macro_f1", "mean_cost_ms", "mean_blocks", "cost_saving_pct"] if c in df_abl.columns]
        tex_sections.append(df_abl[cols_show].to_latex(index=False, float_format="%.2f"))
        tex_sections.append(r"\end{table}")
        tex_sections.append("")

    # Table 7: Statistical Significance
    p_sig = results_dir / "significance.csv"
    if p_sig.exists():
        df_sig = pd.read_csv(p_sig)
        tex_sections.append(r"% Table 7: Statistical Hypothesis Testing (McNemar and 10,000 Paired Bootstraps)")
        tex_sections.append(r"\begin{table}[ht]")
        tex_sections.append(r"\centering")
        tex_sections.append(r"\caption{Paired statistical tests between pipeline configurations.}")
        tex_sections.append(r"\label{tab:significance}")
        cols_show = [c for c in ["model_a", "model_b", "mcnemar_chi2", "mcnemar_p_value", "delta_acc_mean", "delta_acc_ci95", "delta_cost_mean_ms"] if c in df_sig.columns]
        tex_sections.append(df_sig[cols_show].to_latex(index=False, float_format="%.4f"))
        tex_sections.append(r"\end{table}")
        tex_sections.append("")

    output_tex.write_text("\n".join(tex_sections), encoding="utf-8")
    print(f"Generated complete LaTeX publication tables at {output_tex}")

if __name__ == "__main__":
    generate_latex_tables()
