"""
backend/main.py - Phase 6: FastAPI Backend for Microplastic Information Selection Demo

Endpoints:
- GET /
  Serves frontend/index.html
- GET /api/meta
  Returns classes, blocks metadata (costs, weights, descriptions), block report, defaults
- GET /api/demo
  Returns list of test demo particles from demo_index.csv
- POST /api/classify
  Classifies uploaded image or demo_id using Adaptive or Baseline policies
  Returns complete transparent step-by-step trace with candidate scores for all blocks
"""

import io
import json
import time
import joblib
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import features and engine
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.features import BLOCK_EXTRACTORS, extract_all_blocks
from engine import AdaptiveInformationEngine

app = FastAPI(
    title="Microplastic Polymer Adaptive Information Selection Demo",
    description="Backend API for cost-sensitive dynamic feature acquisition in microplastic classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bundle state
BUNDLE = {
    "engine": None,
    "metadata": None,
    "demo_df": None,
    "demo_crops_dir": None,
    "block_feature_names": None
}

def load_bundle():
    serve_dir = PROJECT_ROOT / "serve_bundle"
    meta_path = serve_dir / "metadata.json"
    experts_path = serve_dir / "experts.joblib"
    experts_dir = serve_dir / "experts"
    pool_path = serve_dir / "simulation_pool.npz"
    demo_index_path = serve_dir / "demo_index.csv"

    if not meta_path.exists() or not pool_path.exists():
        print("Warning: serve_bundle metadata or pool missing.")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    if experts_dir.exists() and any(experts_dir.glob("*.joblib")):
        experts = {b: joblib.load(experts_dir / f"{b}.joblib") for b in meta["block_costs"]}
    elif experts_path.exists():
        experts = joblib.load(experts_path)
    else:
        print("Warning: No expert models found.")
        return

    pool_data = np.load(pool_path)

    pool_cls = pool_data["pool_cls"]
    pool_counts = pool_data["pool_counts"]
    pool_lr = {b: pool_data[f"lr_{b}"] for b in meta["block_costs"].keys()}

    engine = AdaptiveInformationEngine(
        classes=meta["classes"],
        log_prior=np.array(meta["log_prior"]),
        experts=experts,
        block_weights=meta["block_weights"],
        block_costs=meta["block_costs"],
        pool_lr=pool_lr,
        pool_cls=pool_cls,
        pool_counts=pool_counts,
        block_feature_names=meta["block_feature_names"],
        fixed_order=meta.get("fixed_order"),
        cheapest_order=meta.get("cheapest_order")
    )

    demo_df = pd.read_csv(demo_index_path) if demo_index_path.exists() else pd.DataFrame()

    BUNDLE["engine"] = engine
    BUNDLE["metadata"] = meta
    BUNDLE["demo_df"] = demo_df
    BUNDLE["demo_crops_dir"] = serve_dir / "demo_crops"
    BUNDLE["block_feature_names"] = meta["block_feature_names"]
    print("Successfully loaded serve_bundle into memory!")

@app.on_event("startup")
def startup_event():
    load_bundle()

# Mount static demo images and frontend assets
demo_crops_path = PROJECT_ROOT / "serve_bundle" / "demo_crops"
demo_crops_path.mkdir(parents=True, exist_ok=True)
app.mount("/demo_crops", StaticFiles(directory=str(demo_crops_path)), name="demo_crops")

crops_path = PROJECT_ROOT / "crops"
if crops_path.exists():
    app.mount("/crops", StaticFiles(directory=str(crops_path)), name="crops")

results_path = PROJECT_ROOT / "results"
if results_path.exists():
    app.mount("/results", StaticFiles(directory=str(results_path)), name="results")

@app.get("/")
def get_index():
    index_file = PROJECT_ROOT / "frontend" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Frontend index.html not yet created."}

@app.get("/api/meta")
def get_meta():
    if BUNDLE["metadata"] is None:
        load_bundle()
    if BUNDLE["metadata"] is None:
        raise HTTPException(status_code=503, detail="Bundle not loaded. Model training required.")
    return BUNDLE["metadata"]

@app.get("/api/demo")
def get_demo_particles():
    if BUNDLE["demo_df"] is None:
        load_bundle()
    if BUNDLE["demo_df"] is None or BUNDLE["demo_df"].empty:
        return []
    records = BUNDLE["demo_df"].to_dict(orient="records")
    return records

from fastapi import Request

@app.post("/api/classify")
async def classify(request: Request):
    if BUNDLE["engine"] is None:
        load_bundle()
    if BUNDLE["engine"] is None:
        raise HTTPException(status_code=503, detail="Engine not loaded. Please train model first.")

    engine = BUNDLE["engine"]
    block_feature_names = BUNDLE["block_feature_names"]

    content_type = request.headers.get("content-type", "")
    req_demo_id = None
    req_policy = "adaptive"
    req_tau = 0.90
    req_lambda = 0.01
    img_rgb = None
    true_label = None

    if "application/json" in content_type:
        body = await request.json()
        req_demo_id = body.get("demo_id")
        req_policy = body.get("policy", "adaptive")
        req_tau = float(body.get("tau", 0.90))
        req_lambda = float(body.get("lambda_", 0.01))
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        req_demo_id = int(form.get("demo_id")) if form.get("demo_id") is not None else None
        req_policy = form.get("policy", "adaptive")
        req_tau = float(form.get("tau", 0.90))
        req_lambda = float(form.get("lambda_", 0.01))
        
        file = form.get("file")
        if file is not None and hasattr(file, "read"):
            contents = await file.read()
            if len(contents) > 0:
                np_arr = np.frombuffer(contents, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise HTTPException(status_code=400, detail="Invalid image file provided.")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if img_rgb is None and req_demo_id is not None:
        demo_df = BUNDLE["demo_df"]
        match = demo_df[demo_df["demo_id"] == req_demo_id]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"Demo ID {req_demo_id} not found.")
        row = match.iloc[0]
        true_label = row["true_label"]
        rel_file = row["file"]
        img_file_path = PROJECT_ROOT / "serve_bundle" / rel_file
        if not img_file_path.exists():
            img_file_path = PROJECT_ROOT / rel_file
        img_bgr = cv2.imread(str(img_file_path))
        if img_bgr is None:
            raise HTTPException(status_code=500, detail=f"Failed to read image at {img_file_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if img_rgb is None:
        raise HTTPException(status_code=400, detail="Either a valid 'demo_id' or uploaded 'file' must be provided.")

    H, W = img_rgb.shape[:2]
    if H < 24 or W < 24:
        raise HTTPException(status_code=400, detail=f"Image dimension ({W}x{H}) is smaller than 24px minimum required for texture extraction.")

    # Extract all 5 blocks
    mask = np.ones((H, W), dtype=bool)
    raw_blocks = extract_all_blocks(img_rgb, mask)

    # Format feature vectors matching exact column ordering
    features_dict = {}
    for b, col_list in block_feature_names.items():
        extracted = raw_blocks[b]
        vec = np.array([extracted[c] for c in col_list], dtype=np.float64)
        assert len(vec) == len(col_list), f"Feature length mismatch for block {b}: expected {len(col_list)}, got {len(vec)}"
        features_dict[b] = vec

    # Run policy engine
    trace = engine.run(
        features_dict,
        policy=req_policy,
        tau=req_tau,
        lambda_=req_lambda,
        true_label=true_label
    )

    return trace
