"""
src/utils.py - Core utilities for research integrity, metadata logging, and statistics.
"""

import os
import sys
import platform
import subprocess
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_git_commit_hash() -> str:
    """Returns the current git commit hash, or 'unknown' if not in git."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return out
    except Exception:
        return "unknown"

def get_system_hardware_info() -> Dict[str, Any]:
    """Captures CPU, RAM, OS, Python and key library versions."""
    info = {
        "os": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash()
    }
    
    # Memory info
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / (1024 ** 3), 2)
        info["ram_available_gb"] = round(vm.available / (1024 ** 3), 2)
    except ImportError:
        # Fallback for Windows without psutil
        try:
            out = subprocess.check_output(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"]).decode().split()
            bytes_val = int(out[-1])
            info["ram_total_gb"] = round(bytes_val / (1024 ** 3), 2)
        except Exception:
            info["ram_total_gb"] = "N/A"

    # CPU cores
    info["cpu_count_logical"] = os.cpu_count()

    # Library versions
    for pkg in ["numpy", "scipy", "scikit-learn", "skimage", "pandas", "cv2", "joblib"]:
        try:
            mod = __import__(pkg)
            info[f"version_{pkg}"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"version_{pkg}"] = "not_installed"

    return info

def compute_ci95(values: List[float] | np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes mean, std, and 95% confidence interval using Student's t distribution.
    Returns (mean, std, ci_low, ci_high).
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if n == 1:
        return mean, 0.0, mean, mean
    std = float(np.std(arr, ddof=1))
    # Student's t critical value for df = n - 1
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    margin = t_crit * (std / np.sqrt(n))
    return mean, std, mean - margin, mean + margin

def save_figure(fig, fig_path: Path, dpi: int = 300):
    """Saves figure in both PNG (at specified dpi) and PDF formats."""
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = fig_path.with_suffix(".png")
    pdf_path = fig_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure: {png_path} and {pdf_path}")
