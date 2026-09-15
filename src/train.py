"""Stage 3: building segmentation with an SVM classifier.

Logic ported from ``code/code/03_ml_model.py`` of the notebook workflow:
  1. Train an SVM on 3 HLP images (pixel features + red mask labels).
  2. Apply the trained SVM to all dataset images.
  3. Post-process with morphology + contour filtering.
No Google-Drive mounting / pip-installs; functions take explicit paths.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import RunPaths

logger = logging.getLogger("skyscrapper.train")

warnings.filterwarnings("ignore")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

TRAIN_IMAGES_COUNT = 3
SVM_KERNEL = "rbf"
SVM_C = 10.0
SVM_GAMMA = "scale"

RED_LOWER1 = np.array([0, 70, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 70, 50])
RED_UPPER2 = np.array([180, 255, 255])

MORPH_KERNEL = 5
MORPH_ITERATIONS = 2
MIN_AREA = 200
MAX_AREA_RATIO = 0.5
MIN_SOLIDITY = 0.4
MAX_AXIS_RATIO = 8.0

CONTOURS_FILE = "building_contours.csv"
METRICS_FILE = "segmentation_metrics.json"
SVM_MODEL_FILE = "svm_model.pkl"


# ---------------------------------------------------------------------------
# HLP red-mask extraction
# ---------------------------------------------------------------------------
def extract_red_mask(hlp_img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(hlp_img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return red_mask


# ---------------------------------------------------------------------------
# Pixel-level features for the SVM
# ---------------------------------------------------------------------------
def extract_pixel_features(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    r = img_rgb[:, :, 0].astype(np.float32).flatten()
    g = img_rgb[:, :, 1].astype(np.float32).flatten()
    b = img_rgb[:, :, 2].astype(np.float32).flatten()
    hue = hsv[:, :, 0].astype(np.float32).flatten()
    sat = hsv[:, :, 1].astype(np.float32).flatten()
    val = hsv[:, :, 2].astype(np.float32).flatten()

    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sx ** 2 + sy ** 2).flatten()

    mean_local = cv2.blur(gray, (5, 5))
    sq_local = cv2.blur(gray ** 2, (5, 5))
    local_var = (sq_local - mean_local ** 2).flatten()
    local_var = np.clip(local_var, 0, None)

    ys, xs = np.mgrid[0:h, 0:w]
    norm_x = (xs.astype(np.float32).flatten() / w)
    norm_y = (ys.astype(np.float32).flatten() / h)

    features = np.column_stack([
        r, g, b, hue, sat, val,
        sobel_mag, local_var,
        norm_x, norm_y,
    ])
    return features


# ---------------------------------------------------------------------------
# Build training data from HLP images
# ---------------------------------------------------------------------------
def find_hlp_images(hlp_path: str) -> Dict[str, str]:
    images: Dict[str, str] = {}
    if not os.path.isdir(hlp_path):
        return images
    for f in os.listdir(hlp_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            base_name = os.path.splitext(f)[0].rstrip(".")
            images[base_name] = os.path.join(hlp_path, f)
    return images


def build_training_data(hlp_path: str) -> Tuple[np.ndarray, np.ndarray]:
    hlp_images = find_hlp_images(hlp_path)
    if not hlp_images:
        raise FileNotFoundError(f"No HLP images found at: {hlp_path}")

    sorted_names = sorted(hlp_images.keys())[:TRAIN_IMAGES_COUNT]
    logger.info("Using %d HLP images for SVM training: %s", len(sorted_names), sorted_names)

    all_features = []
    all_labels = []

    for name in sorted_names:
        img_path = hlp_images[name]
        img = cv2.imread(img_path)
        if img is None:
            logger.warning("  Cannot read: %s", img_path)
            continue

        h_orig, w_orig = img.shape[:2]
        max_dim = max(h_orig, w_orig)
        if max_dim > 512:
            scale = 512.0 / max_dim
            img = cv2.resize(img, None, fx=scale, fy=scale)

        features = extract_pixel_features(img)
        gt_mask = extract_red_mask(img)
        labels = (gt_mask > 0).astype(np.uint8).flatten()

        all_features.append(features)
        all_labels.append(labels)

        building_pct = labels.mean() * 100
        logger.info("  %s: %d pixels, %.1f%% building", name, len(labels), building_pct)

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    logger.info("Training data: %d samples, %d features", X.shape[0], X.shape[1])
    logger.info("Class distribution: building=%d (%.1f%%), background=%d (%.1f%%)",
                int(np.sum(y == 1)), float(np.mean(y == 1) * 100),
                int(np.sum(y == 0)), float(np.mean(y == 0) * 100))
    return X, y


def train_svm(X: np.ndarray, y: np.ndarray) -> Tuple[SVC, StandardScaler]:
    logger.info("Training SVM (kernel=%s, C=%.1f)...", SVM_KERNEL, SVM_C)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, probability=False, cache_size=800)
    svm.fit(X_scaled, y)

    train_acc = accuracy_score(y, svm.predict(X_scaled))
    logger.info("Training accuracy: %.4f", train_acc)
    return svm, scaler


def segment_with_svm(img: np.ndarray, svm: SVC, scaler: StandardScaler) -> np.ndarray:
    h_orig, w_orig = img.shape[:2]
    features = extract_pixel_features(img)
    features_scaled = scaler.transform(features)
    predictions = svm.predict(features_scaled)
    mask = (predictions.reshape(h_orig, w_orig) * 255).astype(np.uint8)

    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def find_buildings(mask: np.ndarray, img_area: int) -> List[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    buildings: List[dict] = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue
        if area > img_area * MAX_AREA_RATIO:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < MIN_SOLIDITY:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        axis_ratio = max(bw, bh) / (min(bw, bh) + 1)
        if axis_ratio > MAX_AXIS_RATIO:
            continue

        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + bw // 2, y + bh // 2

        buildings.append({
            "building_id": f"B{i + 1:03d}",
            "contour_index": i,
            "area_px": int(area),
            "centroid_x": cx,
            "centroid_y": cy,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_width": bw,
            "bbox_height": bh,
            "solidity": round(float(solidity), 3),
            "axis_ratio": round(float(axis_ratio), 3),
        })
    return buildings


def process_all_images(svm: SVC, scaler: StandardScaler, run: RunPaths) -> Tuple[pd.DataFrame, dict]:
    dataset_path = run.inputs_dataset
    Path(run.out_masks).mkdir(parents=True, exist_ok=True)
    Path(run.out_predictions).mkdir(parents=True, exist_ok=True)
    Path(run.out_models).mkdir(parents=True, exist_ok=True)

    image_files = []
    for root, _dirs, files in os.walk(dataset_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, f))

    if not image_files:
        raise FileNotFoundError(f"No images found in {dataset_path}")

    logger.info("Found %d images to process with SVM", len(image_files))

    all_buildings = []
    total_buildings = 0
    total_pixels = 0
    building_pixels = 0
    images_processed = 0
    images_failed = 0

    for idx, img_path in enumerate(image_files, 1):
        img_name = os.path.basename(img_path)
        if idx % 10 == 0 or idx == len(image_files):
            logger.info("  Processing image %d / %d: %s", idx, len(image_files), img_name)

        img = cv2.imread(img_path)
        if img is None:
            images_failed += 1
            logger.warning("  Failed to read: %s", img_name)
            continue

        images_processed += 1
        h, w = img.shape[:2]
        img_area = h * w

        mask = segment_with_svm(img, svm, scaler)

        mask_path = os.path.join(run.out_masks, f"{os.path.splitext(img_name)[0]}_mask.png")
        cv2.imwrite(mask_path, mask)

        total_pixels += h * w
        building_pixels += int(np.sum(mask > 0))

        buildings = find_buildings(mask, img_area)
        for b in buildings:
            b["image_name"] = img_name
            all_buildings.append(b)
        total_buildings += len(buildings)

    contours_df = pd.DataFrame(all_buildings)

    metrics = {
        "total_images": len(image_files),
        "images_processed": images_processed,
        "images_failed": images_failed,
        "total_buildings_detected": total_buildings,
        "avg_buildings_per_image": round(total_buildings / max(images_processed, 1), 2),
        "total_pixels": int(total_pixels),
        "building_pixels": int(building_pixels),
        "building_area_ratio": round(building_pixels / max(total_pixels, 1), 4),
        "model_type": "SVM",
        "parameters": {
            "kernel": SVM_KERNEL,
            "C": SVM_C,
            "gamma": SVM_GAMMA,
            "features": ["r", "g", "b", "hue", "sat", "val", "sobel_mag", "local_var", "norm_x", "norm_y"],
            "num_features": 10,
            "train_images_count": TRAIN_IMAGES_COUNT,
            "min_area": MIN_AREA,
            "max_area_ratio": MAX_AREA_RATIO,
            "min_solidity": MIN_SOLIDITY,
            "max_axis_ratio": MAX_AXIS_RATIO,
        },
    }
    return contours_df, metrics


def save_outputs(contours_df: pd.DataFrame, metrics: dict, svm: SVC,
                 scaler: StandardScaler, run: RunPaths) -> Dict[str, str]:
    contours_path = os.path.join(run.out_predictions, CONTOURS_FILE)
    contours_df.to_csv(contours_path, index=False)
    logger.info("Building contours saved to %s (%d buildings)", contours_path, len(contours_df))

    metrics_path = os.path.join(run.out_models, METRICS_FILE)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    model_path = os.path.join(run.out_models, SVM_MODEL_FILE)
    joblib.dump({"svm": svm, "scaler": scaler}, model_path)
    logger.info("SVM model saved to %s", model_path)

    return {"contours": contours_path, "metrics": metrics_path, "model": model_path}


def print_summary(contours_df: pd.DataFrame, metrics: dict):
    logger.info("=" * 50)
    logger.info("  SEGMENTATION RESULTS (SVM)")
    logger.info("=" * 50)
    logger.info("  Images processed   : %d / %d", metrics["images_processed"], metrics["total_images"])
    logger.info("  Buildings detected : %d", metrics["total_buildings_detected"])
    logger.info("  Avg per image      : %.1f", metrics["avg_buildings_per_image"])
    logger.info("  Building area ratio: %.2f%%", metrics["building_area_ratio"] * 100)
    logger.info("  Model              : SVM (%s)", SVM_KERNEL)
    logger.info("=" * 50)
    if not contours_df.empty:
        logger.info("Sample buildings:")
        print(contours_df.head(10).to_string())


def stage_train_and_segment(run: RunPaths) -> Dict[str, str]:
    """Run stage 3; returns paths of produced contours/metrics/model files."""
    logger.info("--- Step 1: Build Training Data ---")
    X, y = build_training_data(run.inputs_hlp)

    logger.info("--- Step 2: Train SVM ---")
    svm, scaler = train_svm(X, y)

    logger.info("--- Step 3: Segment Dataset Images ---")
    contours_df, metrics = process_all_images(svm, scaler, run)

    logger.info("--- Step 4: Save Outputs ---")
    paths = save_outputs(contours_df, metrics, svm, scaler, run)
    print_summary(contours_df, metrics)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit("This module is not meant to be run directly; import and call stage_* functions.")
