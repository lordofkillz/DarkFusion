"""Dataset geometry estimates for a starting YOLO training resolution.

These are detail-retention heuristics, not predictions of model accuracy. The
longest image side is resized to imgsz, as in Ultralytics' detection data loader.
Keep one set of measurements, rather than copying every label per candidate.
"""

import math
import re
from collections import Counter

import numpy as np


AUTO_SIZES = (320, 416, 512, 640, 768, 832, 960, 1024, 1280, 1536, 1920, 2048)


def aligned_size(value, stride=32):
    """Round up like Ultralytics check_imgsz, within the Trainer's limits."""
    stride = max(32, int(stride))
    low = math.ceil(160 / stride) * stride
    high = math.floor(2048 / stride) * stride
    if high < low:
        raise ValueError("Model stride exceeds the supported training image sizes.")
    return min(high, max(low, math.ceil(float(value) / stride - 1e-9) * stride))


def candidate_sizes(text, current_size=640, stride=32):
    automatic = str(text or "auto").strip().lower() in {"", "auto"}
    if automatic:
        values = AUTO_SIZES
    else:
        tokens = re.split(r"[,;\s]+", str(text).strip())
        if not tokens or any(not token.isdecimal() or int(token) <= 0 for token in tokens):
            raise ValueError("Use Auto or positive image sizes separated by commas (for example 512,640,1024).")
        values = [int(token) for token in tokens]
    return sorted({aligned_size(value, stride) for value in (*values, current_size)})


def _pct(values):
    return float(np.mean(values) * 100) if len(values) else 0.0


class TrainingSizeAnalysis:
    def __init__(self):
        self.images = []
        self.native_sides = []
        self.relative_sides = []
        self.classes = []
        self.kinds = []
        self.pose_spacing = []
        self.class_counts = Counter()
        self.annotation_counts = Counter()
        self.visible_keypoints = 0
        self.invalid_labels = 0

    def add_image(self, width, height, annotations):
        if width <= 0 or height <= 0:
            return
        longest = max(width, height)
        self.images.append((width, height))
        for annotation in annotations:
            class_id, nw, nh, kind, visible, _span_x, _span_y, *extra = annotation
            if (class_id < 0 or not math.isfinite(nw) or not math.isfinite(nh)
                    or not 0 < nw <= 1 or not 0 < nh <= 1):
                self.invalid_labels += 1
                continue
            side = min(nw * width, nh * height)
            self.native_sides.append(side)
            self.relative_sides.append(side / longest)
            self.classes.append(class_id)
            self.kinds.append(kind)
            # Nearest distinct visible keypoints in source pixels. Unlike an
            # axis-aligned span, vertical/horizontal skeletons are not zero-size.
            spacing = float(extra[0]) if extra else 0.0
            self.pose_spacing.append(spacing / longest if spacing > 0 else 0.0)
            self.class_counts[class_id] += 1
            self.annotation_counts[kind] += 1
            if kind == "pose":
                self.visible_keypoints += int(visible)

    def recommend(self, sizes="auto", current_size=640, strides=(8, 16, 32), task="detect", mask_ratio=4):
        min_stride = max(1, min(strides))
        max_stride = max(32, max(strides))
        mask_ratio = max(1, int(mask_ratio))
        current_size = aligned_size(current_size, max_stride)
        candidates = candidate_sizes(sizes, current_size, max_stride)
        native = np.asarray(self.native_sides, dtype=float)
        relative = np.asarray(self.relative_sides, dtype=float)
        class_ids = np.asarray(self.classes, dtype=int)
        segment = np.asarray(self.kinds) == "seg"
        pose_spacing = np.asarray(self.pose_spacing, dtype=float)
        images = np.asarray(self.images, dtype=float).reshape(-1, 2)
        longest = images.max(axis=1) if len(images) else np.array([])
        class_order = np.argsort(class_ids, kind="stable")
        class_chunks = np.split(class_order, np.flatnonzero(np.diff(class_ids[class_order])) + 1)
        groups = {int(class_ids[index[0]]): index for index in class_chunks if len(index)}
        target = float(max(2 * min_stride, 6 * mask_ratio if task == "segment" else 0))
        # Upsampling cannot restore missing source detail. Aim for the smaller
        # of the heuristic target and the object's actual native pixels.
        available_target = np.minimum(native, target)
        required = available_target / relative if len(native) else np.array([])
        warnings = []
        if len(native) and str(sizes or "auto").strip().lower() in {"", "auto"}:
            # Include a data-derived candidate, so the guess is not restricted
            # to an old short list ending at 832.
            required_size = max(
                float(np.quantile(required, .90, method="higher")),
                float(np.quantile(np.minimum(native, min_stride) / relative, .98, method="higher")),
                float(np.quantile(np.minimum(native, 4) / relative, .995, method="higher")),
                max(float(np.quantile(required[index], .50, method="higher")) for index in groups.values()),
            )
            if task == "pose" and np.any(pose_spacing > 0):
                valid = pose_spacing > 0
                # Retain at least one stride or the native keypoint separation.
                native_spacing = pose_spacing[valid] * (native[valid] / relative[valid])
                pose_required = np.minimum(native_spacing, min_stride) / pose_spacing[valid]
                required_size = max(required_size, float(np.quantile(pose_required, .90, method="higher")))
            candidates = sorted(set(candidates + [aligned_size(max(320, required_size), max_stride)]))

        rows = []
        for size in candidates:
            sides = relative * size
            fraction = np.minimum(sides / available_target, 1) if len(native) else np.array([])
            coverage = _pct(sides + 1e-9 >= available_target)
            class_coverage = [_pct(sides[index] + 1e-9 >= available_target[index]) for index in groups.values()]
            class_fraction = [float(np.mean(fraction[index])) for index in groups.values()]
            detail = 100 * (.5 * float(np.mean(fraction)) + .5 * float(np.mean(class_fraction))) if len(native) else 0.0
            meets = bool(len(native) and coverage >= 90 - 1e-9
                         and _pct(sides + 1e-9 >= np.minimum(native, min_stride)) >= 98 - 1e-9
                         and _pct(sides + 1e-9 >= np.minimum(native, 4)) >= 99.5 - 1e-9
                         and min(class_coverage) >= 50 - 1e-9)
            pose = pose_spacing > 0
            pose_under = _pct(pose_spacing[pose] * size < min_stride)
            if task == "pose" and np.any(pose):
                native_spacing = pose_spacing[pose] * (native[pose] / relative[pose])
                pose_fraction = np.minimum(pose_spacing[pose] * size / np.minimum(native_spacing, min_stride), 1)
                detail = min(detail, 100 * float(np.mean(pose_fraction)))
                meets = meets and _pct(pose_fraction >= 1 - 1e-9) >= 90 - 1e-9
            masks = sides[segment] / mask_ratio
            # Approximate square-input padding before augmentation/rect batches.
            padding = float(np.mean(1 - images.prod(axis=1) / longest ** 2)) if len(images) else 0.0
            rows.append({
                "imgsz": size, "labels": len(native),
                "tiny4_pct": _pct(sides < 4),
                "under_stride_pct": _pct(sides < min_stride),
                "under_2stride_pct": _pct(sides < 2 * min_stride),
                "p10_min_side": float(np.percentile(sides, 10)) if len(sides) else 0.0,
                "p25_min_side": float(np.percentile(sides, 25)) if len(sides) else 0.0,
                "median_min_side": float(np.median(sides)) if len(sides) else 0.0,
                "mask_p10_min_side": float(np.percentile(masks, 10)) if len(masks) else 0.0,
                "mask_under_4_pct": _pct(masks < 4), "mask_under_8_pct": _pct(masks < 8),
                "pose_spread_under_stride": int(np.count_nonzero(pose_spacing[pose] * size < min_stride)),
                "pose_spread_under_stride_pct": pose_under,
                "avg_padding_waste": padding,
                "risky_class_count": sum(_pct(sides[index] < 2 * min_stride) > 50 for index in groups.values()),
                "target_coverage_pct": coverage,
                "worst_class_coverage_pct": min(class_coverage) if class_coverage else 0.0,
                "score": detail, "meets_detail_target": meets,
                "upscaled_images_pct": _pct(longest < size),
                "pixel_cost_vs_640": (size / 640) ** 2,
            })
        if not len(native) or task == "classify":
            chosen = next(row for row in rows if row["imgsz"] == current_size)
            warnings.append("Insufficient object-size evidence; kept the current image size."
                            if not len(native) else "Classification uses a different crop pipeline; kept the current image size.")
        else:
            passing = [row for row in rows if row["meets_detail_target"]]
            if passing:
                chosen = passing[0]
            else:
                best = max(row["score"] for row in rows)
                # Continuous retention cannot collapse all difficult sizes to
                # zero. A relative tolerance still distinguishes candidates
                # when even the best available resolution retains very little.
                chosen = next(row for row in rows if row["score"] >= best * .995)
                warnings.append("No candidate meets the detail target. Compare larger inputs or image tiles on validation data.")
        source_limited = _pct(native < target)
        if source_limited:
            warnings.append(f"{source_limited:.1f}% of objects have fewer than {target:g} source pixels on their short side; enlarging them adds no source detail.")
        if len(native) and (len(native) < 30 or any(len(index) < 10 for index in groups.values())):
            warnings.append("Limited examples for one or more classes; treat the size estimate as low confidence.")
        if task == "segment":
            warnings.append("Mask analysis uses polygon bounds; thin structures and boundary detail still need visual validation.")
        class_rows = []
        for cid, index in groups.items():
            sides = relative[index] * chosen["imgsz"]
            class_rows.append({"class_id": cid, "labels": len(index),
                               "tiny4_pct": _pct(sides < 4),
                               "under_stride_pct": _pct(sides < min_stride),
                               "under_2stride_pct": _pct(sides < 2 * min_stride),
                               "median_min_side": float(np.median(sides))})
        class_rows.sort(key=lambda row: (-row["under_2stride_pct"], -row["under_stride_pct"], row["median_min_side"]))
        note = (
            f"Starting estimate: {chosen['imgsz']}px, the smallest candidate meeting the detail target. "
            if chosen["meets_detail_target"] and len(native) and task != "classify"
            else f"Starting estimate: {chosen['imgsz']}px. "
        )
        note += (f"Target: retain {target:g}px on the short side (or available native detail) for at least 90% of objects "
                 "and 50% within each class; also retain one stride for 98% and 4px for 99.5%. "
                 f"Input pixel cost is {chosen['pixel_cost_vs_640']:.2f}x 640; actual speed and VRAM require measurement. "
                 "Confirm accuracy on validation data; this estimate is before augmentation.")
        return {"candidate_rows": rows, "recommended_row": chosen, "class_risk_rows": class_rows,
                "size_note": note, "warnings": warnings,
                "native_summary": {"image_count": len(images), "label_count": len(native),
                                   "median_long_side": float(np.median(longest)) if len(longest) else 0.0,
                                   "p10_object_short_side": float(np.percentile(native, 10)) if len(native) else 0.0,
                                   "source_limited_pct": source_limited, "target_pixels": target}}
