"""Candidate-only SAM3 foreground evidence for Dataset Analysis.

The verifier changes review priority only.  It never writes, deletes, or
suppresses a dataset annotation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


EVIDENCE_VERSION = "darkfusion-sam3-foreground-v1"
SAM3_STRIDE = 14


def _normalized_bounds(bounds):
    values = list(bounds or [])[:4]
    if len(values) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def analyze_foreground_mask(mask, pixel_box):
    """Return conservative foreground geometry for one annotation box."""
    array = np.squeeze(np.asarray(mask))
    if array.ndim != 2:
        return None
    binary = array > 0
    height, width = binary.shape
    x1, y1, x2, y2 = (int(round(float(value))) for value in pixel_box)
    x1, x2 = max(0, min(width - 1, x1)), max(1, min(width, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(1, min(height, y2))
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
    roi = binary[y1:y2, x1:x2].astype(np.uint8, copy=False)
    box_area, mask_area, inside_area = roi.size, int(binary.sum()), int(roi.sum())
    if not box_area or not mask_area or not inside_area:
        return None

    occupancy = inside_area / float(box_area)
    containment = inside_area / float(mask_area)
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
    component_areas = [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, int(component_count))
        if int(stats[index, cv2.CC_STAT_AREA]) >= max(2, int(inside_area * .01))
    ]
    largest_fraction = max(component_areas) / float(inside_area) if component_areas else 0.0
    border_sides = sum((bool(np.any(roi[0, :])), bool(np.any(roi[-1, :])),
                        bool(np.any(roi[:, 0])), bool(np.any(roi[:, -1]))))
    coherence = max(0.0, min(1.0, .36 * min(1.0, occupancy / .16) + .32 * containment + .32 * largest_fraction))
    if occupancy < .035 or containment < .35 or largest_fraction < .45 or coherence < .43:
        status = "weak_foreground_separation"
    elif coherence >= .74 and occupancy >= .08 and containment >= .62:
        status = "coherent_foreground"
    else:
        status = "ambiguous_foreground"
    return {
        "status": status, "coherence": round(coherence, 6),
        "foreground_risk": round(1.0 - coherence, 6), "occupancy": round(occupancy, 6),
        "containment": round(containment, 6), "largest_component_fraction": round(largest_fraction, 6),
        "component_count": len(component_areas), "border_sides": int(border_sides),
        "mask_pixels": mask_area, "inside_mask_pixels": inside_area,
    }


def combine_foreground_evidence(base_evidence, foreground_result):
    """Adjust review priority without turning a segmentation mask into a verdict."""
    combined = dict(base_evidence or {})
    try:
        priority = float(combined.get("priority", 0.0) or 0.0)
    except (TypeError, ValueError):
        priority = 0.0
    if not isinstance(foreground_result, dict) or foreground_result.get("error"):
        combined["foreground_adjustment"] = 0.0
        return combined
    try:
        risk = max(0.0, min(1.0, float(foreground_result.get("foreground_risk", 0.0) or 0.0)))
        coherence = max(0.0, min(1.0, float(foreground_result.get("coherence", 0.0) or 0.0)))
    except (TypeError, ValueError):
        combined["foreground_adjustment"] = 0.0
        return combined
    status = str(foreground_result.get("status", ""))
    adjustment = (
        .10 + .15 * risk if status == "weak_foreground_separation" else
        .06 * ((risk - .55) / .45) if status == "ambiguous_foreground" and risk > .55 else
        -.07 * coherence if status == "coherent_foreground" else 0.0
    )
    original_priority = priority
    priority = max(0.0, min(1.0, priority + adjustment))
    combined.update(priority=round(priority, 6), foreground_adjustment=round(priority - original_priority, 6), foreground_risk=round(risk, 6))
    combined["strength"] = "high" if priority >= .70 else "moderate" if priority >= .55 else "review"
    return combined


class ForegroundBackgroundVerifier:
    """Run cached SAM3 box prompts only for Dataset Analysis candidates."""

    def __init__(self, cache_path, *, model_path, device="cuda", imgsz=640, status=None, cancelled=None):
        self.cache_path = os.fspath(cache_path)
        self.model_path = os.path.abspath(os.fspath(model_path))
        self.requested_device = str(device or "cuda")
        self.imgsz = max(19 * SAM3_STRIDE, int(round(max(256, int(imgsz or 640)) / SAM3_STRIDE)) * SAM3_STRIDE)
        self.status, self.cancelled = status or (lambda _message: None), cancelled or (lambda: False)
        self.backend, self.device, self.model, self._torch, self._connection = "sam3", "", None, None, None

    def _prepare_cache(self):
        if self._connection is None:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.cache_path)
            self._connection.execute("CREATE TABLE IF NOT EXISTS foreground_scores (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            self._connection.commit()

    def prepare(self):
        if self.model is not None:
            return self
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"SAM3 checkpoint not found: {self.model_path}")
        import torch
        from ultralytics import SAM
        self._torch = torch
        self.device = "cuda:0" if self.requested_device.lower() != "cpu" and torch.cuda.is_available() else "cpu"
        self.status(f"Loading SAM3 foreground verifier on {self.device}...")
        self.model = SAM(self.model_path)
        return self

    def _cache_key(self, record):
        image_file, bounds = os.path.abspath(os.fspath(record.get("image_file", ""))), _normalized_bounds(record.get("bounds"))
        if bounds is None:
            raise ValueError("invalid annotation bounds")
        image_stat, model_stat = os.stat(image_file), os.stat(self.model_path)
        identity = {"version": EVIDENCE_VERSION, "image": os.path.normcase(image_file), "image_size": image_stat.st_size, "image_mtime_ns": image_stat.st_mtime_ns, "bounds": [round(value, 7) for value in bounds], "model_size": model_stat.st_size, "model_mtime_ns": model_stat.st_mtime_ns, "imgsz": self.imgsz}
        return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    @staticmethod
    def _pixel_box(bounds, width, height):
        x1, y1, x2, y2 = bounds
        return [max(0, min(width - 1, int(round(x1 * width)))), max(0, min(height - 1, int(round(y1 * height)))), max(1, min(width, int(round(x2 * width)))), max(1, min(height, int(round(y2 * height))))]

    @staticmethod
    def _extract_masks(results, width, height):
        masks = []
        for result in list(results or []):
            data = getattr(getattr(result, "masks", None), "data", None)
            if data is None:
                continue
            if hasattr(data, "detach"):
                data = data.detach().cpu().numpy()
            for mask in np.asarray(data):
                mask = np.squeeze(mask)
                if mask.ndim == 2:
                    masks.append(cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST) > .5 if mask.shape != (height, width) else mask > .5)
        return masks

    def _predict(self, image_rgb, boxes):
        self.prepare()
        kwargs = {"source": image_rgb, "bboxes": [list(box) for box in boxes], "imgsz": self.imgsz, "device": self.device, "verbose": False}
        if self.device.startswith("cuda"):
            kwargs["quantize"] = 16
        try:
            return self.model.predict(**kwargs)
        except (TypeError, ValueError):
            if "quantize" not in kwargs:
                raise
            kwargs.pop("quantize")
            kwargs["half"] = True
            try:
                return self.model.predict(**kwargs)
            except (TypeError, ValueError):
                kwargs.pop("half")
                return self.model.predict(**kwargs)

    def _predict_masks(self, image_rgb, boxes):
        height, width = image_rgb.shape[:2]
        masks = self._extract_masks(self._predict(image_rgb, boxes), width, height)
        if len(masks) == len(boxes):
            return masks
        exact = []
        for box in boxes:
            one = self._extract_masks(self._predict(image_rgb, [box]), width, height)
            exact.append(one[0] if len(one) == 1 else None)
        return exact

    def score_records(self, records):
        records = list(records or [])
        self._prepare_cache()
        if records and not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"SAM3 checkpoint not found: {self.model_path}")
        results, pending, writes = [None] * len(records), OrderedDict(), []
        for index, record in enumerate(records):
            if self.cancelled():
                raise RuntimeError("Foreground verification cancelled")
            try:
                key = self._cache_key(record)
                row = self._connection.execute("SELECT payload FROM foreground_scores WHERE cache_key = ?", (key,)).fetchone()
                cached = json.loads(row[0]) if row else None
            except (OSError, ValueError, json.JSONDecodeError) as error:
                results[index] = {"error": str(error), "cached": False}
                continue
            if isinstance(cached, dict):
                results[index] = dict(cached, cached=True)
            else:
                pending.setdefault(os.path.abspath(os.fspath(record.get("image_file", ""))), []).append((index, record, key))
        for image_number, (image_file, items) in enumerate(pending.items(), 1):
            if self.cancelled():
                raise RuntimeError("Foreground verification cancelled")
            self.status(f"SAM3 foreground verification: image {image_number:,}/{len(pending):,}...")
            image_bgr = cv2.imread(image_file, cv2.IMREAD_COLOR)
            if image_bgr is None:
                for index, _record, _key in items:
                    results[index] = {"error": "unavailable image", "cached": False}
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            valid = []
            for index, record, key in items:
                bounds = _normalized_bounds(record.get("bounds"))
                if bounds is None:
                    results[index] = {"error": "invalid annotation bounds", "cached": False}
                    continue
                box = self._pixel_box(bounds, width, height)
                if box[2] <= box[0] or box[3] <= box[1]:
                    results[index] = {"error": "empty annotation bounds", "cached": False}
                    continue
                valid.append((index, record, key, box))
            masks = self._predict_masks(image_rgb, [item[3] for item in valid]) if valid else []
            for (index, _record, key, box), mask in zip(valid, masks):
                metrics = analyze_foreground_mask(mask, box) if mask is not None else None
                if metrics is None:
                    results[index] = {"error": "SAM3 returned no usable mask", "cached": False}
                    continue
                payload = dict(metrics, backend=self.backend, cached=False)
                results[index] = payload
                writes.append((key, json.dumps({key: value for key, value in payload.items() if key != "cached"}, separators=(",", ":"))))
        if writes:
            self._connection.executemany("INSERT OR REPLACE INTO foreground_scores (cache_key, payload) VALUES (?, ?)", writes)
            self._connection.commit()
        return results

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self.model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
