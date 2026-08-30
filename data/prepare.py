"""
data/prepare.py - Phase 1: Data Preparation Pipeline

Extracts Nile Red fluorescence microplastic dataset, parses YOLO annotations,
pads bounding boxes by 8%, filters out crops < 24px, balances classes (cap at 1000/class, seed 42),
and generates crops/<POLYMER>/<source_stem>__<box_index>.png and crops/crop_index.csv.
"""

import os
import sys
import glob
import zipfile
import random
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

CLASS_MAP = {
    0: "ABS",
    1: "Nylon",
    2: "PE",
    3: "PET",
    4: "PS",
    5: "PVC"
}

CAP_PER_CLASS = 1000
MIN_DIM = 24
PADDING_RATIO = 0.08
RANDOM_SEED = 42

def extract_all_zips(raw_dir: Path, extract_dir: Path):
    """Extracts all zip files found in raw_dir into extract_dir."""
    print(f"Scanning for zip files in {raw_dir}...")
    zip_files = list(raw_dir.glob("*.zip")) + list(raw_dir.glob("**/*.zip"))
    # deduplicate
    zip_files = list(dict.fromkeys(zip_files))
    print(f"Found {len(zip_files)} zip file(s): {[z.name for z in zip_files]}")
    
    for zpath in zip_files:
        print(f"Extracting {zpath.name}...")
        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                zf.extractall(extract_dir)
        except Exception as e:
            print(f"Warning: Failed to extract {zpath.name}: {e}")

def find_images_and_annotations(extract_dir: Path):
    """
    Finds all images (.jpg, .jpeg, .png, .tif, .bmp) and YOLO annotation files (.txt).
    Builds maps by file stem.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images_by_stem = {}
    annots_by_stem = {}

    for path in extract_dir.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        stem = path.stem
        if ext in image_extensions:
            if stem not in images_by_stem:
                images_by_stem[stem] = path
        elif ext == ".txt":
            if stem.lower() in {"classes", "notes", "readme"}:
                continue
            if stem not in annots_by_stem:
                annots_by_stem[stem] = path

    print(f"Discovered {len(images_by_stem)} images and {len(annots_by_stem)} annotation text files.")
    return images_by_stem, annots_by_stem

def parse_yolo_file(txt_path: Path):
    """
    Parses a YOLO annotation file.
    Returns list of dicts: [{'class_id': int, 'cx': float, 'cy': float, 'w': float, 'h': float}, ...]
    """
    boxes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cid = int(float(parts[0]))
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    boxes.append({
                        "class_id": cid,
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h
                    })
                except ValueError:
                    continue
    return boxes

def prepare_dataset(project_root: Path):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    data_dir = project_root / "data"
    repo_dir = data_dir / "dataset_microplastic"
    extract_dir = data_dir / "extracted"
    crops_dir = project_root / "crops"
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract all zips
    extract_all_zips(repo_dir, extract_dir)

    # 2. Discover images and annotations
    images_by_stem, annots_by_stem = find_images_and_annotations(extract_dir)

    # Report unmatched
    unmatched_images = set(images_by_stem.keys()) - set(annots_by_stem.keys())
    print(f"Report: {len(unmatched_images)} images had no matching label file.")
    if len(unmatched_images) > 0 and len(unmatched_images) <= 10:
        print(f"Unmatched image stems: {list(unmatched_images)}")

    matched_stems = sorted(list(set(images_by_stem.keys()) & set(annots_by_stem.keys())))
    print(f"Found {len(matched_stems)} matched image-annotation pairs.")

    # 3. Collect all candidate crops
    candidates_by_class = {cid: [] for cid in CLASS_MAP.keys()}
    total_boxes_found = 0
    skipped_small_boxes = 0

    print("Parsing annotations and checking bounding boxes...")
    for stem in matched_stems:
        img_path = images_by_stem[stem]
        annot_path = annots_by_stem[stem]

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

        boxes = parse_yolo_file(annot_path)
        for box_idx, b in enumerate(boxes):
            cid = b["class_id"]
            if cid not in CLASS_MAP:
                continue
            total_boxes_found += 1

            # Bounding box in pixels
            cx = b["cx"] * img_w
            cy = b["cy"] * img_h
            bw = b["w"] * img_w
            bh = b["h"] * img_h

            # Pad 8%
            pad_w = PADDING_RATIO * bw
            pad_h = PADDING_RATIO * bh

            xmin = max(0, int(round(cx - bw / 2.0 - pad_w)))
            xmax = min(img_w, int(round(cx + bw / 2.0 + pad_w)))
            ymin = max(0, int(round(cy - bh / 2.0 - pad_h)))
            ymax = min(img_h, int(round(cy + bh / 2.0 + pad_h)))

            crop_w = xmax - xmin
            crop_h = ymax - ymin

            if crop_w < MIN_DIM or crop_h < MIN_DIM:
                skipped_small_boxes += 1
                continue

            candidates_by_class[cid].append({
                "img_path": img_path,
                "source_image": stem,
                "box_idx": box_idx,
                "class_id": cid,
                "label": CLASS_MAP[cid],
                "bbox": (xmin, ymin, xmax, ymax),
                "w": crop_w,
                "h": crop_h
            })

    print(f"Total raw boxes found: {total_boxes_found}")
    print(f"Skipped boxes < {MIN_DIM}px: {skipped_small_boxes}")
    print("Raw candidate counts per class:")
    for cid, cname in CLASS_MAP.items():
        print(f"  {cname} (class {cid}): {len(candidates_by_class[cid])} candidates")

    # 4. Apply cap of 1000 per class (sampled randomly)
    selected_crops = []
    for cid, cname in CLASS_MAP.items():
        cands = candidates_by_class[cid]
        if len(cands) > CAP_PER_CLASS:
            selected = random.sample(cands, CAP_PER_CLASS)
            print(f"Capping {cname}: sampled {CAP_PER_CLASS} from {len(cands)}")
        else:
            selected = cands
            print(f"Keeping all {len(selected)} crops for {cname}")
        selected_crops.extend(selected)

    # 5. Extract and save crops
    print(f"Extracting and saving {len(selected_crops)} crops...")
    
    # Create polymer subfolders
    for cname in CLASS_MAP.values():
        (crops_dir / cname).mkdir(parents=True, exist_ok=True)

    # Group selected crops by image path for efficient loading
    by_img = {}
    for crop_info in selected_crops:
        by_img.setdefault(crop_info["img_path"], []).append(crop_info)

    crop_records = []
    for img_path, crops_for_img in by_img.items():
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: Failed to load {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for cinfo in crops_for_img:
            xmin, ymin, xmax, ymax = cinfo["bbox"]
            crop_rgb = img_rgb[ymin:ymax, xmin:xmax]
            
            label = cinfo["label"]
            source_stem = cinfo["source_image"]
            box_idx = cinfo["box_idx"]
            crop_filename = f"{source_stem}__{box_idx}.png"
            rel_crop_path = f"crops/{label}/{crop_filename}"
            full_crop_path = crops_dir / label / crop_filename

            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(full_crop_path), crop_bgr)

            crop_records.append({
                "crop": rel_crop_path.replace("\\", "/"),
                "label": label,
                "source_image": source_stem,
                "w": cinfo["w"],
                "h": cinfo["h"]
            })

    # 6. Write crop_index.csv
    df_index = pd.DataFrame(crop_records)
    crop_index_path = crops_dir / "crop_index.csv"
    df_index.to_csv(crop_index_path, index=False)
    print(f"Successfully saved {len(df_index)} crop entries to {crop_index_path}")

    # 7. Print final counts and verify constraints
    counts = df_index["label"].value_counts().to_dict()
    print("\n--- PHASE 1 FINAL PER-CLASS CROP COUNTS ---")
    for cid in range(len(CLASS_MAP)):
        cname = CLASS_MAP[cid]
        cnt = counts.get(cname, 0)
        print(f"  {cname}: {cnt} crops")

    for cid, cname in CLASS_MAP.items():
        cnt = counts.get(cname, 0)
        if cnt < 100:
            print(f"\n[ALERT] Class {cname} has fewer than 100 crops ({cnt})!")
            return False, counts

    print("\nPhase 1 Data Preparation complete and verified!")
    return True, counts

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    prepare_dataset(project_root)
