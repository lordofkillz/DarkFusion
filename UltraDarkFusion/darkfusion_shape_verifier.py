#!/usr/bin/env python3
"""
SAM3 Shape-Based Class Verification

Uses SAM3 foreground masks to verify if an object matches the expected
shape profile for its labeled class. Protects valid objects with unusual
appearance but correct shape (e.g., person in weird pose).
"""

import numpy as np
from collections import defaultdict


class ShapeProfiler:
    """Build and score shape profiles for object classes."""
    
    def __init__(self):
        """Initialize shape profiler."""
        self.class_profiles = {}  # class_id → shape stats
        self.samples_per_class = defaultdict(list)
    
    def analyze_mask_shape(self, mask):
        """Analyze shape characteristics of a binary mask.
        
        Args:
            mask: Binary numpy array (H, W)
        
        Returns:
            dict with shape metrics or None if mask invalid
        """
        if mask is None or not isinstance(mask, np.ndarray):
            return None
        
        if mask.size == 0 or mask.sum() == 0:
            return None
        
        try:
            # Basic metrics
            area = float(mask.sum())
            h, w = mask.shape
            
            # Bounding box of foreground
            rows, cols = np.where(mask > 0)
            if len(rows) == 0:
                return None
            
            bbox_h = int(rows.max() - rows.min() + 1)
            bbox_w = int(cols.max() - cols.min() + 1)
            bbox_area = bbox_h * bbox_w
            
            # Shape descriptors
            occupancy = area / max(1, bbox_area)  # How much of bbox is filled
            aspect_ratio = bbox_h / max(1, bbox_w)  # Height/Width
            compactness = (4 * np.pi * area) / max(1, (np.sqrt(area) * 4) ** 2)  # Circularity
            
            # Solidity: ratio of area to convex hull (rough measure)
            # For simplicity, use occupancy as proxy
            solidity = occupancy
            
            # Vertical extent: how tall vs wide
            vertical_extent = bbox_h / max(1, h)
            horizontal_extent = bbox_w / max(1, w)
            
            return {
                "area": area,
                "bbox_h": bbox_h,
                "bbox_w": bbox_w,
                "bbox_area": bbox_area,
                "occupancy": occupancy,
                "aspect_ratio": aspect_ratio,
                "compactness": compactness,
                "solidity": solidity,
                "vertical_extent": vertical_extent,
                "horizontal_extent": horizontal_extent,
            }
        except Exception as e:
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
            for metric_name in samples[0].keys():
                values = [s[metric_name] for s in samples]
                values = np.array(values, dtype=np.float32)
                
                profile[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }
            
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
        
        for metric_name, value in shape_metrics.items():
            if metric_name not in profile:
                continue
            
            total_metrics += 1
            stats = profile[metric_name]
            mean = stats["mean"]
            std = stats["std"]
            
            if std < 0.001:  # No variance in this metric
                # Check if value matches mean
                if abs(value - mean) < 0.01:
                    continue
                else:
                    anomaly_count += 1
            else:
                # Z-score: how many std deviations from mean
                z = abs(value - mean) / (std + 1e-6)
                if z > threshold_std:
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
    
    # Hard threshold: if shape way off, likely wrong class
    class_matches = shape_score > 0.4  # Below 0.4 = significantly different
    
    return {
        "shape_score": shape_score,
        "class_matches": class_matches,
        "confidence": shape_score if class_matches else 1.0 - shape_score,
        "details": f"Shape match {shape_score:.2f} - {'valid' if class_matches else 'anomalous'}"
    }
