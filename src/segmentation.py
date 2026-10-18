"""
src/segmentation.py - Robust foreground segmentation for Nile Red fluorescence microplastic crops.
"""

import numpy as np
import cv2
from skimage import filters, morphology, measure

def segment_particle(rgb: np.ndarray) -> np.ndarray:
    """
    Extracts binary foreground mask for the microplastic particle from RGB crop.
    
    Fluorescence crops have dark (low-intensity) background and bright particle emissions.
    Uses Otsu's optimal thresholding on luminance followed by morphological cleanup
    and largest connected component selection.
    
    Args:
        rgb: Float array HxWx3 in [0, 1] or uint8 in [0, 255].
        
    Returns:
        mask: Boolean 2D array of shape (H, W) where True indicates the particle.
    """
    if rgb.dtype != np.uint8:
        # Scale to uint8 [0, 255] for robust histogram computation
        rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb

    # Standard ITU-R BT.601 luminance
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    
    # Handle degenerate/blank images
    if np.max(gray) == np.min(gray):
        return np.ones((gray.shape[0], gray.shape[1]), dtype=bool)

    try:
        # Otsu thresholding
        thresh = filters.threshold_otsu(gray)
        binary = gray > thresh
    except Exception:
        binary = gray > np.mean(gray)

    # If binary mask is empty, fallback to top 20% brightest pixels
    if not np.any(binary):
        binary = gray > np.percentile(gray, 80)

    # Morphological closing of small gaps and holes
    binary_u8 = binary.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel)

    # Keep largest connected component (crop contains 1 primary particle)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels > 1:
        # Index 0 is background; find largest component among 1..num_labels-1
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_idx)
    else:
        mask = closed.astype(bool)

    # Final sanity check: if mask collapsed to empty, return central region
    if not np.any(mask):
        H, W = gray.shape
        mask = np.zeros((H, W), dtype=bool)
        mask[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = True

    return mask
