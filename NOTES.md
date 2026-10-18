# Codebase Audit and Research Notes

**Project:** Adaptive Information Selection for Microplastic Polymer Classification  
**Date:** October 2026  
**Repository Commit:** `35396bcf13bb497c4c27c8af6b86e6e638199f03`  
**Python Environment:** Python 3.14.5 (scikit-learn 1.9.0, scikit-image 0.26.0, scipy 1.18.1, numpy 2.5.2, pandas 3.0.5)

---

## 1. Executive Summary & Architecture Overview

The codebase implements a complete demonstration of adaptive feature block selection for classifying microplastic polymer types from Nile Red fluorescence microscopy images across 6 classes:
$$\mathcal{Y} = \{\text{ABS}, \text{Nylon}, \text{PE}, \text{PET}, \text{PS}, \text{PVC}\}$$

The existing architecture consists of:
1. **Data Ingestion (`data/download.py`, `data/prepare.py`)**: Downloads 32 zip archives from the sombsuk/dataset_microplastic repository, extracts YOLO annotations, pads bounding boxes by 8%, filters crops $< 24\,\text{px}$, caps class size at 1,000 crops/class, and exports `crops/crop_index.csv` (5,680 particles).
2. **Feature Extraction (`model/features.py`)**: 5 feature blocks totaling 62 features (`shape`: 8, `intensity`: 10, `colour`: 15, `texture`: 22, `frequency`: 7).
3. **Training & Fusion (`model/train.py`)**:
   - `GroupShuffleSplit` on `source_image` to avoid spatial and illumination leakage across crops from the same slide.
   - Calibrated Random Forest expert per block using `CalibratedClassifierCV(method="isotonic", cv=3)`.
   - Weighted Log-Opinion Pool fusion with weights $w_b$ fitted via Nelder-Mead on validation NLL.
   - Training simulation pool ($M=300$, 50 particles/class) to approximate Expected Information Gain (EIG).
4. **Adaptive Engine (`engine.py`)**: Standalone Bayesian sequential decision engine evaluating $\text{EIG}(b \mid \text{current belief}) / \text{Cost}(b)$ and executing early stopping based on fixed confidence threshold $\tau$ or minimum score $\lambda$.
5. **Serving & UI (`backend/main.py`, `frontend/index.html`)**: FastAPI server with modular models (`serve_bundle/experts/*.joblib`) and an interactive HTML/JS timeline visualization.
6. **Tests (`tests/test_engine.py`)**: 5 unit tests validating posterior sums, EIG non-negativity, block count bounds, feature lengths, and split leakage isolation.

---

## 2. Detailed Component Review

### 2.1 Data Pipeline (`data/prepare.py`)
- **Dataset Source**: Somboon Sukpancharoen et al. (Rermborirak et al., 2025, *J. Hazardous Materials Advances* 19, 100787).
- **Images and Annotations**: 2,566 matched pairs of `.jpg` fluorescence microscope slides and YOLO `.txt` annotations in `data/extracted/`.
- **Bounding Box Preprocessing**:
  - Center coordinates $(c_x, c_y)$ and dimensions $(b_w, b_h)$ are converted to pixel coordinates.
  - An 8% margin is added to each dimension ($pad_w = 0.08 \cdot b_w$, $pad_h = 0.08 \cdot b_h$) to preserve fluorescence halos.
  - Crops with $w < 24$ or $h < 24$ are filtered out (16,422 small boxes skipped).
- **Class Balancing**:
  - Class distribution is capped at 1,000 randomly sampled crops per class (`random.seed(42)`).
  - Total extracted crops: **5,680** (ABS: 1000, Nylon: 1000, PE: 1000, PET: 680, PS: 1000, PVC: 1000).
- **Index**: Saved to `crops/crop_index.csv` with columns: `crop`, `label`, `source_image`, `w`, `h`.

### 2.2 Feature Extractors (`model/features.py`)
Extractors implement signature `fn(rgb: float array HxWx3 in [0,1], mask: bool HxW) -> dict[str, float]`.
1. **`shape` (8 features)**:
   - Extracted from `mask` via `skimage.measure.regionprops`.
   - Descriptors: `shape_area`, `shape_perimeter`, `shape_eccentricity`, `shape_solidity`, `shape_extent`, `shape_circularity` ($4\pi A / P^2$), `shape_aspect_ratio` ($W / H$), `shape_equivalent_diameter` ($\sqrt{4A/\pi}$).
2. **`intensity` (10 features)**:
   - Greyscale luminance: $Y = 0.2989 R + 0.5870 G + 0.1140 B$.
   - Descriptors: `mean`, `std`, `min`, `max`, percentiles (5, 25, 50, 75, 95), `skewness` (scipy.stats).
3. **`colour` (15 features)**:
   - RGB channel means, standard deviations, and relative channel fractions: $r_{frac} = R / (R+G+B+\epsilon)$.
   - HSV colour space (via `skimage.color.rgb2hsv`): `mean` and `std` for Hue, Saturation, and Value.
4. **`texture` (22 features)**:
   - 32-level GLCM across offsets: distances $d \in \{1, 3\}$, angles $\theta \in \{0, \pi/4, \pi/2, 3\pi/4\}$.
   - 6 GLCM properties: `contrast`, `dissimilarity`, `homogeneity`, `energy`, `correlation`, `ASM` (each computed as mean and std across offsets $\rightarrow 12$ features).
   - Uniform Local Binary Pattern (`skimage.feature.local_binary_pattern`, $P=8, R=1$): 10-bin normalized histogram.
5. **`frequency` (7 features)**:
   - 2D Fast Fourier Transform (`np.fft.fft2`), shifted to center.
   - Radial coordinates normalized to $[0, 1]$; power spectrum integrated across 6 concentric frequency bands as fractions of total spectral energy.
   - High-to-low power ratio: $(\text{band}_4 + \text{band}_5) / (\text{band}_0 + \text{band}_1 + \epsilon)$.

### 2.3 Feature Cache & Empirical Cost Profiling (`data/features_cache.csv`, `data/block_costs.json`)
- All 5,680 crops are extracted and cached in `data/features_cache.csv` (SHA-256: `56e8cce82b95718879c911f16a9530ebffa08745f71579aed46c1cf10b499d91`).
- **Existing Cost Model**: In `model/train.py`, runtime was measured per block, and the median runtime across crops was normalized to the cheapest block ($1.0$):
  - `frequency`: $1.00$
  - `colour`: $2.14$
  - `intensity`: $2.84$
  - `texture`: $4.89$
  - `shape`: $5.88$
  - Sum of normalized costs: $16.75$

### 2.4 Data Split & Calibrated Classification (`model/train.py`)
- **GroupShuffleSplit**: Grouped by `source_image`.
  - Train: 3,406 crops (1,192 groups / slides).
  - Validation: 877 crops (299 groups / slides).
  - Test: 1,397 crops (498 groups / slides).
  - Explicit assertions enforce zero group overlap between all splits:
    $$\mathcal{G}_{train} \cap \mathcal{G}_{val} = \emptyset, \quad \mathcal{G}_{train} \cap \mathcal{G}_{test} = \emptyset, \quad \mathcal{G}_{val} \cap \mathcal{G}_{test} = \emptyset$$
- **Classifiers**:
  - Per-block pipeline: `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()` $\rightarrow$ `CalibratedClassifierCV(RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced_subsample', random_state=42), method='isotonic', cv=3)`.
  - Calibration metric: 10-bin Expected Calibration Error (ECE) evaluated on validation and test.

### 2.5 Log-Opinion Pool Fusion
- **Formulation**:
  $$\log P(y \mid \mathcal{A}) = \log P(y) + \sum_{b \in \mathcal{A}} w_b \cdot \left[ \log P_b(y \mid x_b) - \log P(y) \right] - \log Z$$
  where $P(y)$ is the empirical training prior, and $Z$ is the normalization constant ($\text{logsumexp}$).
- **Optimization**:
  - Optimization variable: $\theta_b = \log w_b$.
  - Objective: Minimize validation cross-entropy / negative log-likelihood (NLL) using Nelder-Mead simplex optimization.
  - Optimal weights: `shape`: 0.2001, `intensity`: 0.2106, `colour`: 0.3544, `texture`: 0.4867, `frequency`: 0.5902.
  - *Theoretical assumption*: Assumes conditional independence of features given class; the weights $w_b < 1$ penalize correlated redundancy between blocks (e.g., texture and frequency).

### 2.6 Adaptive Engine (`engine.py`)
- **Simulation Pool**:
  - Sampled from training set: 50 particles per class ($M = 300$).
  - For each sample $i$ in the pool and block $b$, the precomputed log-ratio is:
    $$\Delta \ell_{b, i} = w_b \cdot (\log P_b(y \mid x_{b, i}) - \log P(y))$$
- **Belief-Weighted Expected Information Gain**:
  At any step with current log-posterior $\log P$, the current entropy is:
  $$H_0 = -\sum_{k=1}^K P(y=k) \ln P(y=k)$$
  Future belief under simulated particle $i$:
  $$\tilde{P}_i = \text{softmax}(\log P + \Delta \ell_{b, i}), \quad H_{1, i} = -\sum_{k=1}^K \tilde{P}_{i, k} \ln \tilde{P}_{i, k}$$
  Weights re-weighted by current belief:
  $$w_i = \frac{P(\text{class}_i)}{\text{count}(\text{class}_i)}, \quad \tilde{w}_i = \frac{w_i}{\sum_j w_j}$$
  $$\text{EIG}(b \mid P) = \max\left(0, H_0 - \sum_{i=1}^M \tilde{w}_i H_{1, i}\right)$$
  $$\text{Score}(b) = \frac{\text{EIG}(b \mid P)}{\text{Cost}(b)}$$
- **Stopping Rules**:
  1. $\max_k P(y=k) \ge \tau$ (default $\tau = 0.90$).
  2. Best score $< \lambda$ (default $\lambda = 0.01$) after at least 1 block evaluated.
  3. All blocks exhausted.

---

## 3. Discrepancies, Gaps, and Inconsistencies vs. Research Requirements

| Requirement | Current Status in Repo | Required Implementation Plan |
|---|---|---|
| **Cost Metrics (Module 2)** | Costs are normalized multipliers (1.0 to 5.88) without units. | Must measure real wall-clock time in **milliseconds** per particle over $\ge 3$ repeats excluding warm-up. Must log hardware (CPU, RAM, versions) and write `results/block_costs.csv`. All downstream costs must be in **ms**. |
| **Mask vs Bounding Box (Module 6)** | `extract_all_blocks` defaults to `mask = np.ones((H, W), dtype=bool)`, meaning the whole rectangular crop is treated as the mask. Feret diameter is missing. | Implement real segmentation (e.g., Otsu / fluorescence thresholding on foreground) vs. bounding-box crop. Add Feret diameter. Compare differences and accuracies in `morphology_audit.py` and output `results/morphology_bbox_vs_mask.csv`. Confirm main pipeline uses segmented mask. |
| **Calibration Scope (Module 3)** | Only isotonic calibration (`method="isotonic"`) with 10 bins. | Implement Uncalibrated, Temperature Scaling, Platt (Sigmoid), and Isotonic. Evaluate 15-bin ECE, Brier score, and NLL before/after. Select best per block on validation NLL. Save reliability diagrams to `figures/reliability_<block>.png`. |
| **Context Prior $Z$ (Module 5)** | Absent. Prior is static empirical prior $P(y)$. | Compute $Z$ from segmentation mask only (area, bbox w, bbox h, aspect ratio, mean brightness, fluorescence occupancy, SNR). Train calibrated lightweight classifier $P(Y \mid Z)$. Implement double-counting flag: (a) remove overlapping features from Shape, (b) keep both with fusion weights. |
| **Mathematical Implementations (Module 4)** | Present in `engine.py` and `train.py`, but spread across scripts without explicit standalone docstrings and Monte Carlo predictive formulation. | Formalize in `src/information.py` / `src/fusion.py`: explicit log-opinion pool, entropy $H$, margin (top-1 minus top-2), Monte Carlo EIG over predictive distribution, and $Z$-conditioned EIG. |
| **Learned Stopping (Module 7)** | Fixed threshold $\tau \ge 0.90$ only. | Implement `src/learned_stopping.py`. Generate trajectory training data from train/val splits. Extract state vectors (full posterior, max posterior, entropy, margin, blocks acquired, cumulative cost, remaining blocks, $Z$). Implement Label (a) (accuracy agreement with exhaustive) and Label (b) (cost-weighted gain $< \lambda \cdot \text{remaining cost}$). Train Logistic Regression and Random Forest; tune threshold on validation. Report feature importances. |
| **Open-Set / OOD / LOPO (Module 8)** | Absent. System is strictly closed-set 6-class. | Implement `src/open_set.py`. Dual-threshold rule ($\tau_{conf}, \tau_{ent}$). Leave-One-Polymer-Out (LOPO) across all 6 classes with strict assertion that held-out polymer is absent in train and val. Compare against MSP, Entropy, Energy score, Mahalanobis distance, and OpenMax. Report AUROC, AUPR, FPR@95%TPR, unknown detection rate, false unknown rate, closed-set acc, macro F1. |
| **Unified Policies (Module 9)** | Partial in `engine.py`. | Unify under standard interface in `src/policies.py`: random, cheapest-first, best-first, static at matched budget, EIG/Cost (global), EIG(b\|Z)/Cost (context), oracle (cheapest correct subset). |
| **Unified Pipeline (Module 10)** | Split between `engine.py` and `backend/main.py`. | Unified execution pipeline in `src/pipeline.py`: compute $Z \rightarrow$ init $P(Y \mid Z) \rightarrow$ sequential loop $\rightarrow$ stopping $\rightarrow$ OOD test $\rightarrow$ class or UNKNOWN. |
| **Statistical Rigor & Multi-seed** | Single seed (`seed=42`). | All experiments must take `--seed` and run with seeds 0–9. Report mean $\pm$ std and 95% CI. |
| **Significance Testing (Module 13)** | Absent. | McNemar test and paired bootstrap (10,000 resamples) between configurations and baselines in `src/stats.py`. Output to `results/significance.csv`. |
| **Diagnostics & Visuals (Module 14)** | Only 1 trade-off PNG. | Confusion matrices, stopping step histogram, first-block heatmap per true polymer, per-class error table, 20 worst misclassified particles with trajectories in `src/diagnostics.py`. |
| **LaTeX & Reproduction Script** | Absent. | Generate `results/tables.tex` directly from CSVs. Save figures as 300 dpi `.png` and `.pdf`. Create `run_all.sh`. |

---

## 4. Reusability Strategy & Next Steps

1. **Reuse Existing Data**: `crops/crop_index.csv` and the 5,680 cropped images in `crops/` are fully extracted and validated.
2. **Reuse Feature Logic**: Core GLCM, LBP, FFT, colour, and intensity formulas in `model/features.py` are sound and tested. We will adapt them into `src/features.py` while adding Feret diameter and genuine mask segmentation.
3. **Preserve Baseline Integrity**: The existing `engine.py` will serve as the benchmark for Configuration 1 (global-prior adaptive + fixed threshold), ensuring direct comparability.
4. **Modularity Plan**: Build modular packages under `src/` and testable experiment CLIs under `experiments/`.
