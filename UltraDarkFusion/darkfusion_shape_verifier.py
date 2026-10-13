#!/usr/bin/env python3
"""
SAM3 Shape-Based Class Verification

Uses SAM3 foreground masks to verify if an object matches the expected
shape profile for its labeled class. Protects valid objects with unusual
appearance but correct shape (e.g., person in weird pose).
"""

import numpy as np
import math
import cv2
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from collections import defaultdict


SHAPE_METRICS_VERSION = "sam3-normalized-contours-v2"
# Absolute rasterization tolerances prevent a resolution change alone from
# becoming a shape mismatch. Raw pixel dimensions never contribute a vote.
SHAPE_SCORE_METRICS = {
    "area_fraction": 0.05, "occupancy": 0.05, "aspect_ratio": 0.12,
    "compactness": 0.05, "solidity": 0.05, "vertical_extent": 0.05,
    "horizontal_extent": 0.05, "largest_component_fraction": 0.05,
}
SHAPE_MATCH_THRESHOLD = 0.75


def shape_annotation_crop(image, bounds):
    """Crop normalized xyxy annotation bounds, preserving OpenCV BGR colors."""
    if not isinstance(image, np.ndarray) or image.ndim < 2 or image.size == 0:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bounds]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    height, width = image.shape[:2]
    left = int(math.floor(max(0.0, min(1.0, x1)) * width))
    top = int(math.floor(max(0.0, min(1.0, y1)) * height))
    right = int(math.ceil(max(0.0, min(1.0, x2)) * width))
    bottom = int(math.ceil(max(0.0, min(1.0, y2)) * height))
    if right - left < 3 or bottom - top < 3:
        return None
    return np.ascontiguousarray(image[top:bottom, left:right])


class ShapeProfiler:
    """Build and score shape profiles for object classes."""
    
    def __init__(self):
        """Initialize shape profiler."""
        self.class_profiles = {}  # class_id → shape stats
        self.samples_per_class = defaultdict(list)
    
    def analyze_mask_shape(self, mask):
        """Measure actual contours; scoring uses dimensionless shape features."""
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
            return None
        try:
            if not np.isfinite(mask).all():
                return None
            binary = np.ascontiguousarray(mask > 0, dtype=np.uint8)
            rows, cols = np.where(binary)
            if len(rows) < 9:
                return None
            height, width = binary.shape
            bbox_h = int(rows.max() - rows.min() + 1)
            bbox_w = int(cols.max() - cols.min() + 1)
            if min(bbox_h, bbox_w) < 3:
                return None
            area = float(binary.sum())
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                return None
            contour_area, perimeter = 0.0, 0.0
            external_contours, external_areas = [], []
            for contour, relations in zip(contours, hierarchy[0]):
                current_area = float(cv2.contourArea(contour))
                perimeter += float(cv2.arcLength(contour, True))
                if relations[3] < 0:
                    contour_area += current_area
                    external_contours.append(contour)
                    external_areas.append(current_area)
                else:
                    contour_area -= current_area
            if contour_area <= 0 or perimeter <= 0 or not external_contours:
                return None
            hull = cv2.convexHull(np.concatenate(external_contours, axis=0))
            hull_area = float(cv2.contourArea(hull))
            if hull_area <= 0:
                return None
            return {
                # Retain raw geometry for diagnostics, not as votes in the score.
                "area": area, "bbox_h": bbox_h, "bbox_w": bbox_w,
                "bbox_area": bbox_h * bbox_w,
                "area_fraction": area / (height * width),
                "occupancy": area / (bbox_h * bbox_w),
                "aspect_ratio": bbox_h / bbox_w,
                "compactness": min(1.0, 4.0 * math.pi * contour_area / (perimeter * perimeter)),
                "solidity": min(1.0, contour_area / hull_area),
                "vertical_extent": bbox_h / height,
                "horizontal_extent": bbox_w / width,
                "largest_component_fraction": max(external_areas) / sum(external_areas),
            }
        except (TypeError, ValueError, cv2.error):
            return None
    
    def add_shape_sample(self, class_id, shape_metrics):
        """Add a shape sample to class profile.
        
        Args:
            class_id: Integer class ID
            shape_metrics: Dict from analyze_mask_shape()
        """
        if shape_metrics and isinstance(class_id, int):
            self.samples_per_class[class_id].append(shape_metrics)
    
    def build_profiles(self, min_samples=5):
        """Build statistical profiles from collected samples.
        
        Args:
            min_samples: Minimum samples per class to build profile
        """
        self.class_profiles = {}
        
        for class_id, samples in self.samples_per_class.items():
            if len(samples) < min_samples:
                continue
            
            # Compute statistics per metric
            profile = {}
            for metric_name in SHAPE_SCORE_METRICS:
                if not all(metric_name in sample for sample in samples):
                    continue
                values = [s[metric_name] for s in samples]
                values = np.array(values, dtype=np.float32)
                if not np.isfinite(values).all():
                    continue
                
                profile[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }
            
            if len(profile) == len(SHAPE_SCORE_METRICS):
                self.class_profiles[class_id] = profile
    
    def score_shape_match(self, class_id, shape_metrics, threshold_std=2.0):
        """Score how well a shape matches class profile.
        
        Args:
            class_id: Integer class ID
            shape_metrics: Dict from analyze_mask_shape()
            threshold_std: How many std deviations to allow (2.0 = 95% CI)
        
        Returns:
            float: Score 0-1 (1.0 = perfect match, 0.0 = outlier)
                   None if class has no profile
        """
        if class_id not in self.class_profiles or not shape_metrics:
            return None
        
        profile = self.class_profiles[class_id]
        anomaly_count = 0
        total_metrics = 0
        
        for metric_name, tolerance_floor in SHAPE_SCORE_METRICS.items():
            if metric_name not in profile:
                continue
            value = shape_metrics.get(metric_name)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            
            total_metrics += 1
            stats = profile[metric_name]
            mean = stats["mean"]
            std = stats["std"]
            
            tolerance = max(tolerance_floor, float(threshold_std) * std)
            if metric_name == "aspect_ratio":
                tolerance = max(tolerance, abs(mean) * 0.10)
            if abs(value - mean) > tolerance:
                anomaly_count += 1
        
        if total_metrics == 0:
            return None
        
        # Score: 1.0 = all metrics within threshold, 0.0 = all outliers
        match_score = max(0.0, 1.0 - (anomaly_count / total_metrics))
        return match_score
    
    def get_class_profile_summary(self, class_id):
        """Get human-readable summary of class profile."""
        if class_id not in self.class_profiles:
            return None
        
        profile = self.class_profiles[class_id]
        summary = {
            "class_id": class_id,
            "metrics": {}
        }
        
        for metric_name, stats in profile.items():
            summary["metrics"][metric_name] = {
                "mean": f"{stats['mean']:.3f}",
                "std": f"{stats['std']:.3f}",
                "range": f"[{stats['min']:.3f}, {stats['max']:.3f}]",
            }
        
        return summary


def shape_verification_score(class_id, mask, profiler, weight=0.5):
    """Combined score for shape-based class verification.
    
    Args:
        class_id: Integer class ID
        mask: Binary SAM3 mask
        profiler: ShapeProfiler instance
        weight: How much to weight shape score (0-1)
    
    Returns:
        dict with:
            - shape_score: 0-1 match to class profile
            - class_matches: bool, True if shape fits class
            - confidence: weighted confidence
            - details: explanation
    """
    if profiler is None or mask is None:
        return {
            "shape_score": None,
            "class_matches": None,
            "confidence": None,
            "details": "No profiler or mask"
        }
    
    shape_metrics = profiler.analyze_mask_shape(mask)
    if shape_metrics is None:
        return {
            "shape_score": None,
            "class_matches": None,
            "confidence": None,
            "details": "Could not analyze mask shape"
        }
    
    shape_score = profiler.score_shape_match(class_id, shape_metrics)
    if shape_score is None:
        return {
            "shape_score": None,
            "class_matches": None,
            "confidence": None,
            "details": f"No profile for class {class_id}"
        }
    
    # A profile match is supporting evidence, not a calibrated probability.
    class_matches = shape_score >= SHAPE_MATCH_THRESHOLD
    
    return {
        "shape_score": shape_score,
        "class_matches": class_matches,
        "confidence": shape_score if class_matches else 1.0 - shape_score,
        "details": f"Shape profile agreement {shape_score:.2f} - {'consistent' if class_matches else 'different'}"
    }


class ShapeReferenceCache:
    """Versioned trusted-reference metrics; cache failure never blocks a scan."""

    def __init__(self, path, model_identity):
        self.path = Path(path)
        self.model_identity = model_identity
        self.connection = None
        self.disabled = not bool(model_identity)

    def _open(self):
        if self.disabled:
            return None
        if self.connection is not None:
            return self.connection
        connection = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self.path), timeout=0.25)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS shape_metrics (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            self.connection = connection
            return connection
        except (OSError, sqlite3.Error):
            if connection is not None:
                connection.close()
            self.disabled = True
            return None

    def key(self, image_path, record, image_bytes=None):
        image_path = os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(image_path))))
        if image_bytes is None:
            image_bytes = Path(image_path).read_bytes()
        identity = {
            "version": SHAPE_METRICS_VERSION, "opencv": cv2.__version__,
            "model": self.model_identity, "image": image_path,
            "content": hashlib.sha256(image_bytes).hexdigest(),
            "bounds": list(record["bounds"]), "class_id": int(record["class_id"]),
            "label": " ".join(str(record.get("line", "")).split()),
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _valid(metrics):
        if not isinstance(metrics, dict):
            return False
        try:
            return all(name in metrics and math.isfinite(float(metrics[name]))
                       and float(metrics[name]) > 0
                       and (name == "aspect_ratio" or float(metrics[name]) <= 1)
                       for name in SHAPE_SCORE_METRICS)
        except (TypeError, ValueError):
            return False

    def get(self, key):
        connection = self._open()
        if connection is None:
            return None
        try:
            row = connection.execute("SELECT payload FROM shape_metrics WHERE cache_key = ?", (key,)).fetchone()
            metrics = json.loads(row[0]) if row else None
            return metrics if self._valid(metrics) else None
        except (sqlite3.Error, TypeError, ValueError):
            return None

    def put(self, key, metrics):
        if not self._valid(metrics):
            return
        connection = self._open()
        if connection is None:
            return
        try:
            connection.execute("INSERT OR REPLACE INTO shape_metrics(cache_key, payload) VALUES (?, ?)",
                               (key, json.dumps(metrics, allow_nan=False)))
        except (sqlite3.Error, TypeError, ValueError):
            self.disabled = True

    def close(self):
        if self.connection is not None:
            try:
                self.connection.commit()
            except sqlite3.Error:
                pass
            finally:
                try:
                    self.connection.close()
                except sqlite3.Error:
                    pass
                self.connection = None
