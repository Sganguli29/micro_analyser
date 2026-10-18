"""
src/features.py - Feature block extractors with Feret diameter and mask segmentation support.

Implements the 5 modular extractors:
- shape: regionprops on mask (area, perimeter, circularity, eccentricity, solidity, extent, aspect_ratio, equivalent_diameter, feret_diameter)
- intensity: greyscale statistics, percentiles, skewness under the mask
- colour: RGB stats, relative channel fractions, HSV stats under the mask
- texture: GLCM (32 levels, d=[1,3], 4 angles) properties (mean/std) + uniform LBP histogram (P=8, R=1, 10 bins)
- frequency: 2D FFT radial power in 6 equal bands + high/low ratio

Signature: fn(rgb: float array HxWx3 in [0, 1], mask: bool array HxW) -> dict[str, float]
"""

import numpy as np
from scipy import stats
from skimage import measure, color, feature
from src.segmentation import segment_particle

def extract_shape(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Extracts 9 shape descriptors from binary mask using skimage.measure.regionprops:
    area, perimeter, circularity, eccentricity, solidity, extent, aspect_ratio, equivalent_diameter, feret_diameter.
    """
    H, W = mask.shape
    mask_u8 = mask.astype(np.uint8)
    props = measure.regionprops(mask_u8)

    if len(props) == 0:
        area = float(np.sum(mask)) if np.sum(mask) > 0 else float(H * W)
        perimeter = float(2 * (H + W))
        eccentricity = 0.0
        solidity = 1.0
        extent = 1.0
        eq_diam = float(np.sqrt(4.0 * area / np.pi))
        feret_diam = float(np.sqrt(H**2 + W**2))
    else:
        p = props[0]
        area = float(p.area)
        perimeter = float(p.perimeter) if p.perimeter > 0 else float(2 * (H + W))
        eccentricity = float(p.eccentricity)
        solidity = float(p.solidity)
        extent = float(p.extent)
        eq_diam = float(p.equivalent_diameter_area)
        # Feret diameter (maximum distance between parallel tangents)
        try:
            feret_diam = float(p.feret_diameter_max)
        except Exception:
            feret_diam = float(np.sqrt(H**2 + W**2))

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
        "shape_equivalent_diameter": eq_diam,
        "shape_feret_diameter": feret_diam
    }

def extract_intensity(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Extracts 10 greyscale intensity stats under the mask:
    mean, std, min, max, p5, p25, p50, p75, p95, skewness.
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
    Extracts 15 colour stats:
    RGB channel means, stds, channel ratios, and HSV stats.
    """
    r = rgb[:, :, 0][mask]
    g = rgb[:, :, 1][mask]
    b = rgb[:, :, 2][mask]

    if len(r) == 0:
        r = rgb[:, :, 0].ravel()
        g = rgb[:, :, 1].ravel()
        b = rgb[:, :, 2].ravel()

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

    if len(h_vals) == 0:
        h_vals = hsv[:, :, 0].ravel()
        s_vals = hsv[:, :, 1].ravel()
        v_vals = hsv[:, :, 2].ravel()

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
    Extracts 22 texture features:
    GLCM properties (contrast, dissimilarity, homogeneity, energy, correlation, ASM mean/std across 8 offsets)
    and uniform LBP histogram (P=8, R=1, 10 bins).
    """
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    gray32 = np.clip((gray * 31.999).astype(np.uint8), 0, 31)

    distances = [1, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = feature.graycomatrix(gray32, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)

    res = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        prop_mat = feature.graycoprops(glcm, prop)
        res[f"texture_glcm_{prop}_mean"] = float(np.mean(prop_mat))
        res[f"texture_glcm_{prop}_std"] = float(np.std(prop_mat))

    gray256 = np.clip((gray * 255.0).astype(np.uint8), 0, 255)
    lbp = feature.local_binary_pattern(gray256, P=8, R=1, method="uniform")
    lbp_vals = lbp[mask]
    if len(lbp_vals) == 0:
        lbp_vals = lbp.ravel()

    n_bins = 10
    lbp_hist, _ = np.histogram(lbp_vals, bins=n_bins, range=(0, n_bins), density=True)
    for i in range(n_bins):
        res[f"texture_lbp_bin_{i}"] = float(lbp_hist[i])

    return res

def extract_frequency(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Extracts 7 frequency features:
    2D FFT radial power in 6 equal bands + high/low spectral energy ratio.
    """
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    H, W = gray.shape

    F = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(F) ** 2
    total_power = float(np.sum(power)) + 1e-9

    y_coords, x_coords = np.indices((H, W))
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r_norm = np.sqrt(((y_coords - cy) / max(cy, 1.0)) ** 2 + ((x_coords - cx) / max(cx, 1.0)) ** 2)
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
    If mask is None, automatically computes the segmented foreground particle mask.
    """
    if mask is None:
        mask = segment_particle(rgb)

    out = {}
    for block_name, extractor in BLOCK_EXTRACTORS.items():
        out[block_name] = extractor(rgb, mask)
    return out
