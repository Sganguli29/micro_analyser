"""
model/features.py - Phase 2: Feature Block Extractors

Implements 5 modular extractors:
- shape: regionprops on mask
- intensity: greyscale statistics & percentiles & skew
- colour: RGB channel stats, fractions, HSV stats
- texture: GLCM (32 levels, d=[1,3], 4 angles) props (mean/std) + LBP uniform (P=8, R=1)
- frequency: 2D FFT radial power in 6 equal bands + high/low ratio

Signature: fn(rgb: float array HxWx3 in [0,1], mask: bool HxW) -> dict[str, float]
"""

import time
import numpy as np
import pandas as pd
from scipy import stats
from skimage import measure, color, feature
from pathlib import Path

def extract_shape(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Extracts shape features from binary mask using skimage.measure.regionprops.
    Returns: area, perimeter, eccentricity, solidity, extent, circularity, aspect_ratio, equivalent_diameter
    """
    H, W = mask.shape
    mask_u8 = mask.astype(np.uint8)
    props = measure.regionprops(mask_u8)
    
    if len(props) == 0:
        area = float(H * W)
        perimeter = float(2 * (H + W))
        eccentricity = 0.0
        solidity = 1.0
        extent = 1.0
        eq_diam = np.sqrt(4.0 * area / np.pi)
    else:
        p = props[0]
        area = float(p.area)
        perimeter = float(p.perimeter) if p.perimeter > 0 else float(2 * (H + W))
        eccentricity = float(p.eccentricity)
        solidity = float(p.solidity)
        extent = float(p.extent)
        eq_diam = float(p.equivalent_diameter_area)

    circularity = float(4.0 * np.pi * area / (perimeter ** 2 + 1e-8))
    aspect_ratio = float(W) / float(H + 1e-8)

    return {
        "shape_area": area,
        "shape_perimeter": perimeter,
        "shape_eccentricity": eccentricity,
        "shape_solidity": solidity,
        "shape_extent": extent,
        "shape_circularity": circularity,
        "shape_aspect_ratio": aspect_ratio,
        "shape_equivalent_diameter": eq_diam
    }

def extract_intensity(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Extracts greyscale intensity stats under the mask.
    Returns: mean, std, min, max, p5, p25, p50, p75, p95, skew
    """
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    vals = gray[mask]
    if len(vals) == 0:
        vals = gray.ravel()

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    min_val = float(np.min(vals))
    max_val = float(np.max(vals))
    
    p5 = float(np.percentile(vals, 5))
    p25 = float(np.percentile(vals, 25))
    p50 = float(np.percentile(vals, 50))
    p75 = float(np.percentile(vals, 75))
    p95 = float(np.percentile(vals, 95))
    
    skew_val = float(stats.skew(vals)) if std_val > 1e-7 else 0.0

    return {
        "intensity_mean": mean_val,
        "intensity_std": std_val,
        "intensity_min": min_val,
        "intensity_max": max_val,
        "intensity_p5": p5,
        "intensity_p25": p25,
        "intensity_p50": p50,
        "intensity_p75": p75,
        "intensity_p95": p95,
        "intensity_skew": skew_val
    }

def extract_colour(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Extracts RGB channel means, stds, fractions, and HSV stats.
    """
    r = rgb[:, :, 0][mask]
    g = rgb[:, :, 1][mask]
    b = rgb[:, :, 2][mask]

    r_mean = float(np.mean(r))
    r_std = float(np.std(r))
    g_mean = float(np.mean(g))
    g_std = float(np.std(g))
    b_mean = float(np.mean(b))
    b_std = float(np.std(b))

    total_rgb = r_mean + g_mean + b_mean + 1e-8
    r_frac = float(r_mean / total_rgb)
    g_frac = float(g_mean / total_rgb)
    b_frac = float(b_mean / total_rgb)

    hsv = color.rgb2hsv(rgb)
    h_vals = hsv[:, :, 0][mask]
    s_vals = hsv[:, :, 1][mask]
    v_vals = hsv[:, :, 2][mask]

    return {
        "colour_r_mean": r_mean,
        "colour_r_std": r_std,
        "colour_g_mean": g_mean,
        "colour_g_std": g_std,
        "colour_b_mean": b_mean,
        "colour_b_std": b_std,
        "colour_r_frac": r_frac,
        "colour_g_frac": g_frac,
        "colour_b_frac": b_frac,
        "colour_h_mean": float(np.mean(h_vals)),
        "colour_h_std": float(np.std(h_vals)),
        "colour_s_mean": float(np.mean(s_vals)),
        "colour_s_std": float(np.std(s_vals)),
        "colour_v_mean": float(np.mean(v_vals)),
        "colour_v_std": float(np.std(v_vals))
    }

def extract_texture(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Extracts GLCM properties across 8 offsets (32 levels, d=[1,3], 4 angles) + LBP uniform histogram (P=8, R=1, 10 bins).
    """
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    # Quantize to 32 levels
    gray32 = np.clip((gray * 31.999).astype(np.uint8), 0, 31)

    distances = [1, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = feature.graycomatrix(gray32, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)

    res = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        prop_mat = feature.graycoprops(glcm, prop)
        res[f"texture_glcm_{prop}_mean"] = float(np.mean(prop_mat))
        res[f"texture_glcm_{prop}_std"] = float(np.std(prop_mat))

    # LBP (P=8, R=1, uniform -> 10 bins)
    gray256 = np.clip((gray * 255.0).astype(np.uint8), 0, 255)
    lbp = feature.local_binary_pattern(gray256, P=8, R=1, method="uniform")
    lbp_vals = lbp[mask]
    n_bins = 10
    lbp_hist, _ = np.histogram(lbp_vals, bins=n_bins, range=(0, n_bins), density=True)
    for i in range(n_bins):
        res[f"texture_lbp_bin_{i}"] = float(lbp_hist[i])

    return res

def extract_frequency(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Extracts 2D FFT radial power in 6 equal bands + high/low ratio.
    """
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    H, W = gray.shape

    F = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(F) ** 2
    total_power = float(np.sum(power)) + 1e-9

    # Radial coordinates normalized to [0, 1]
    y_coords, x_coords = np.indices((H, W))
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r_norm = np.sqrt(((y_coords - cy) / max(cy, 1.0)) ** 2 + ((x_coords - cx) / max(cx, 1.0)) ** 2)
    # Clip r_norm to max 1.0
    r_norm = np.clip(r_norm, 0.0, 1.0)

    res = {}
    band_powers = []
    num_bands = 6
    for b in range(num_bands):
        r_low = b / float(num_bands)
        r_high = (b + 1) / float(num_bands)
        if b == num_bands - 1:
            band_mask = (r_norm >= r_low) & (r_norm <= r_high)
        else:
            band_mask = (r_norm >= r_low) & (r_norm < r_high)
        b_pwr = float(np.sum(power[band_mask])) / total_power
        band_powers.append(b_pwr)
        res[f"freq_band_{b}"] = b_pwr

    # High / low ratio (bands 4+5 vs bands 0+1)
    high_pwr = band_powers[4] + band_powers[5]
    low_pwr = band_powers[0] + band_powers[1] + 1e-8
    res["freq_high_low_ratio"] = float(high_pwr / low_pwr)

    return res

BLOCK_EXTRACTORS = {
    "shape": extract_shape,
    "intensity": extract_intensity,
    "colour": extract_colour,
    "texture": extract_texture,
    "frequency": extract_frequency
}

def extract_all_blocks(rgb: np.ndarray, mask: np.ndarray = None) -> dict[str, dict[str, float]]:
    """
    Extracts all 5 blocks for an image crop.
    Mask defaults to all-True over crop if None.
    """
    if mask is None:
        mask = np.ones((rgb.shape[0], rgb.shape[1]), dtype=bool)
    
    out = {}
    for block_name, extractor in BLOCK_EXTRACTORS.items():
        out[block_name] = extractor(rgb, mask)
    return out
