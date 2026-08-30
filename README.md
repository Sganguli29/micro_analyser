# Adaptive Information Selection in Microplastic Polymer Classification

A complete, standalone research demonstration of **cost-sensitive adaptive feature selection** for identifying microplastic polymer types from Nile Red fluorescence microscope images.

Rather than computing all feature blocks upfront for every crop, the system maintains a calibrated Bayesian posterior over polymer classes. At each step, it sequentially evaluates the feature block that maximizes the **Expected Information Gain per unit extraction cost ($\text{EIG} / \text{Cost}$)**, dynamically terminating as soon as the posterior reaches a confidence threshold ($\tau$) or the expected value of information drops below threshold ($\lambda$).

---

## License & Attribution Notice

> [!IMPORTANT]
> **Dataset License**: The dataset is sourced from **Rermborirak et al. (2025)**, *Automated identification of microplastics in water using Nile Red fluorescence and deep learning*, Journal of Hazardous Materials Advances 19, 100787 ([GitHub repository](https://github.com/sombsuk/dataset_microplastic)).
> 
> The dataset is licensed under **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**.
> Commercial use and distribution of derivatives are restricted under this license. This notice is recorded for patent filings and downstream intellectual property considerations.

---

## Measured Experimental Results

All figures below are generated directly from experimental runs with zero hardcoded or fabricated metrics.

### 1. Feature Block Report & Calibration

| Block | Features ($D_b$) | Measured Cost (Norm.) | Val Acc | Test Acc | Val NLL | Test NLL | Val ECE (10 bins) | Test ECE (10 bins) | Description |
|---|---|---|---|---|---|---|---|---|---|
| `shape` | 8 | 5.88 | 0.3546 | 0.3121 | 1.6282 | 1.6333 | 0.0555 | 0.0232 | Regionprops: area, perimeter, circularity, eccentricity, solidity, extent, aspect ratio, equiv. diameter |
| `intensity` | 10 | 2.84 | 0.7206 | 0.6750 | 0.7392 | 0.7945 | 0.0498 | 0.0178 | Greyscale mean, std, min, max, percentiles (5, 25, 50, 75, 95), skewness |
| `colour` | 15 | 2.14 | 0.8119 | 0.8031 | 0.5145 | 0.4989 | 0.0251 | 0.0305 | RGB channel mean/std, channel ratio fractions $r/(r+g+b)$, HSV mean/std |
| `texture` | 22 | 4.89 | 0.8255 | 0.8060 | 0.4406 | 0.4954 | 0.0312 | 0.0207 | GLCM (32 levels, d=[1,3], 4 angles) properties (contrast, dissimilarity, etc.) + LBP histogram ($P=8, R=1$) |
| `frequency` | 7 | 1.00 | 0.6591 | 0.6385 | 0.8828 | 1.0244 | 0.0541 | 0.0461 | 2D FFT radial power in 6 equal frequency bands + high/low ratio |

- **Log-Opinion Pool Weights ($w_b$)** fitted by minimizing validation NLL via Nelder-Mead:
  - `shape`: 0.2001, `intensity`: 0.2106, `colour`: 0.3544, `texture`: 0.4867, `frequency`: 0.5902.
- *Note on Fusion*: Weighted log-opinion pooling assumes conditional independence across feature blocks. In practice, texture and frequency read shared pixel structures; the fitted weights dampen this cross-block redundancy and double-counting.

---

### 2. Policy Benchmark on Test Split ($\tau=0.90, \lambda=0.01$)

| Policy | Macro-F1 | Accuracy | Mean Cost Paid | Mean Blocks Used | Cost Saving vs All-Blocks (%) |
|---|---|---|---|---|---|
| **`adaptive`** | **0.8820** | **0.8683** | **7.27** | **3.03** | **56.6%** |
| `fixed_best_first` | 0.8812 | 0.8669 | 10.80 | 3.19 | 35.5% |
| `fixed_cheapest` | 0.8812 | 0.8669 | 9.64 | 3.50 | 42.4% |
| `random_order` | 0.8812 | 0.8669 | 12.25 | 3.68 | 26.9% |
| `all_blocks` (Ceiling) | 0.8812 | 0.8669 | 16.75 | 5.00 | 0.0% |

- **Headline Finding**: The Adaptive Policy matches the full all-blocks accuracy ceiling while achieving a **56.6% reduction in measurement cost**.
- **First-Block Selection**: `frequency` is selected at step 1 across 100% of particles due to starting from the shared prior distribution and having the lowest normalized cost (1.00). After observing frequency features, the posterior diverges, causing the subsequent steps to branch dynamically across classes.

---

### 3. Adaptive Policy Breakdown by Polymer

| Polymer | Test Samples | Accuracy | F1-Score | Mean Blocks | Mean Cost | Behavior |
|---|---|---|---|---|---|---|
| **PS** | 196 | 1.0000 | 0.9874 | 2.10 | 3.57 | Rapid early stopping in ~2 blocks |
| **PE** | 232 | 0.9871 | 0.9935 | 2.16 | 3.82 | Rapid early stopping in ~2 blocks |
| **PET** | 162 | 0.9568 | 0.9480 | 2.87 | 6.70 | Moderate acquisition (frequency + colour) |
| **Nylon** | 254 | 0.8858 | 0.8604 | 3.11 | 7.69 | Moderate acquisition |
| **ABS** | 270 | 0.7370 | 0.7412 | 3.79 | 10.21 | Deep acquisition (utilizes texture + colour) |
| **PVC** | 283 | 0.7385 | 0.7614 | 3.67 | 9.81 | Deep acquisition |

---

## Exact Commands to Reproduce & Run

### 1. Environment Setup
```bash
pip install scikit-learn pandas fastapi uvicorn python-multipart matplotlib scikit-image scipy opencv-python pillow pytest
```

### 2. Phase 1: Data Preparation
```bash
# Clone dataset and extract crops with 8% padding and 1000/class cap
python data/download.py
python data/prepare.py
```

### 3. Phase 2, 3, 5: Feature Extraction, Training & Evaluation
```bash
python model/train.py
```

### 4. Run Unit Tests
```bash
pytest tests/test_engine.py -v
```

### 5. Launch Backend Server & Live Web Demo
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Open your browser at `http://localhost:8000` to interact with the live demo.

---

## Project Structure

```
microplastics_demo/
├── data/
│   ├── download.py        # Parallel dataset archive downloader
│   ├── prepare.py         # Phase 1 YOLO extraction & crop generation
│   └── features_cache.csv # Precomputed feature table with SHA256 hash
├── model/
│   ├── features.py        # Phase 2 feature extractors (shape, intensity, colour, texture, freq)
│   └── train.py           # Phase 3 & 5 calibrated training, fusion & evaluation
├── engine.py              # Phase 4 standalone adaptive engine & baselines
├── backend/
│   └── main.py            # Phase 6 FastAPI backend server
├── frontend/
│   └── index.html         # Phase 7 self-contained UI
├── tests/
│   └── test_engine.py     # Unit test suite (posterior sum, EIG bounds, leakage check)
├── results/               # Manifest, trade-off plots, and CSV benchmark logs
├── serve_bundle/          # Fitted models, metadata, simulation pool, and demo test crops
└── README.md
```
