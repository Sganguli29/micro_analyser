#!/usr/bin/env bash
# run_all.sh - Complete end-to-end reproduction of all paper experiments from raw data.
set -e

echo "=========================================================================="
echo "Microplastic Polymer Classification - Adaptive Multimodal Pipeline Suite"
echo "=========================================================================="

echo "[1/12] Generating Dataset Summary Report..."
python experiments/dataset_report.py

echo "[2/12] Profiling Wall-Clock Extraction Costs (milliseconds)..."
python experiments/run_cost_model.py

echo "[3/12] Conducting Morphology Audit (BBox vs Otsu Mask)..."
python experiments/run_morphology.py

echo "[4/12] Evaluating Calibration & Reliability Diagrams (15 bins)..."
python experiments/run_calibration.py

echo "[5/12] Extracting Context Prior Z & Double-Counting Deduplication..."
python experiments/run_context_prior.py

echo "[6/12] Simulating Trajectories & Optimizing Learned Stopping Policies..."
python experiments/run_learned_stopping.py --label-option a
python experiments/run_learned_stopping.py --label-option b

echo "[7/12] Executing Leave-One-Polymer-Out (LOPO) Open-Set Benchmark (6 Folds)..."
python experiments/run_open_set.py --seed 42

echo "[8/12] Running 4-Configuration and Component Ablation Study with Bootstraps..."
python experiments/run_ablation.py --seeds 42

echo "[9/12] Evaluating Accuracy vs Cost Pareto Frontier Sweep..."
python experiments/run_tradeoff.py --seed 42

echo "[10/12] Generating Publication Figures, Diagnostics & Error Audit..."
python experiments/run_diagnostics.py --seed 42

echo "[11/12] Compiling LaTeX Publication Tables (results/tables.tex)..."
python src/generate_latex_tables.py

echo "[12/12] Running Verification Test Suite (Non-Leakage & Math Integrity)..."
python -m pytest tests/

echo "=========================================================================="
echo "All experiments completed successfully! All outputs saved to results/ & figures/"
echo "=========================================================================="
