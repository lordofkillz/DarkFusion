import math

import numpy as np


def prediction_size_limits(img_width, img_height, min_size_px=0.0, max_percent=1.0):
    try:
        img_width = max(1.0, float(img_width))
        img_height = max(1.0, float(img_height))
    except Exception:
        img_width = 1.0
        img_height = 1.0

    try:
        min_size_px = max(0.0, float(min_size_px or 0.0))
    except Exception:
        min_size_px = 0.0

    try:
        max_percent = float(max_percent or 1.0)
    except Exception:
        max_percent = 1.0

    if max_percent <= 0.0:
        max_percent = 1.0

    max_percent = min(max_percent, 1.0)
    return (
        min_size_px,
        max(min_size_px, max_percent * img_width),
        min_size_px,
        max(min_size_px, max_percent * img_height),
    )


def _finite_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def points_xyxy(points, img_width, img_height, normalized=False):
    try:
        pts = np.asarray(points, dtype=np.float32)
        if pts.size < 4:
            return None
        pts = pts.reshape(-1, 2)
    except Exception:
        return None

    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return None

    if normalized:
        pts = pts.copy()
        pts[:, 0] *= float(img_width)
        pts[:, 1] *= float(img_height)

    finite = np.isfinite(pts[:, :2]).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] < 2:
        return None

    xs = pts[:, 0]
    ys = pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def prediction_xyxy_from_value(prediction, img_width, img_height):
    if prediction is None:
        return None

    if isinstance(prediction, dict):
        xyxy = prediction.get("xyxy")
        if xyxy is not None and len(xyxy) >= 4:
            values = [_finite_float(v) for v in xyxy[:4]]
            if all(v is not None for v in values):
                return tuple(values)

        points = prediction.get("points")
        if points is not None:
            xyxy = points_xyxy(points, img_width, img_height, normalized=False)
            if xyxy is not None:
                return xyxy

        bbox = prediction.get("bbox")
        if bbox is not None and len(bbox) >= 4:
            values = [_finite_float(v) for v in bbox[:4]]
            if all(v is not None for v in values):
                return tuple(values)

    segmentation = getattr(prediction, "segmentation", None)
    if segmentation:
        xyxy = points_xyxy(segmentation, img_width, img_height, normalized=True)
        if xyxy is not None:
            return xyxy

    obb = getattr(prediction, "obb", None)
    if obb:
        xyxy = points_xyxy(obb, img_width, img_height, normalized=True)
        if xyxy is not None:
            return xyxy

    to_xyxy = getattr(prediction, "to_xyxy", None)
    if callable(to_xyxy):
        try:
            xyxy = to_xyxy(img_width, img_height)
            values = [_finite_float(v) for v in xyxy[:4]]
            if all(v is not None for v in values):
                return tuple(values)
        except Exception:
            pass

    if all(hasattr(prediction, attr) for attr in ("x_center", "y_center", "width", "height")):
        try:
            xc = float(prediction.x_center)
            yc = float(prediction.y_center)
            w = float(prediction.width)
            h = float(prediction.height)
            return (
                (xc - w / 2.0) * float(img_width),
                (yc - h / 2.0) * float(img_height),
                (xc + w / 2.0) * float(img_width),
                (yc + h / 2.0) * float(img_height),
            )
        except Exception:
            pass

    if isinstance(prediction, (list, tuple)) and len(prediction) >= 4:
        values = [_finite_float(v) for v in prediction[:4]]
        if all(v is not None for v in values):
            return tuple(values)

    return None


def prediction_size_allowed_xyxy(xyxy, img_width, img_height, min_size_px=0.0, max_percent=1.0):
    if xyxy is None or len(xyxy) < 4:
        return False

    values = [_finite_float(v) for v in xyxy[:4]]
    if not all(v is not None for v in values):
        return False

    try:
        img_width = max(1.0, float(img_width))
        img_height = max(1.0, float(img_height))
    except Exception:
        return False

    x1, y1, x2, y2 = values
    left = max(0.0, min(img_width, min(x1, x2)))
    top = max(0.0, min(img_height, min(y1, y2)))
    right = max(0.0, min(img_width, max(x1, x2)))
    bottom = max(0.0, min(img_height, max(y1, y2)))

    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    if width <= 0.0 or height <= 0.0:
        return False

    min_w, max_w, min_h, max_h = prediction_size_limits(
        img_width,
        img_height,
        min_size_px=min_size_px,
        max_percent=max_percent,
    )
    return min_w <= width <= max_w and min_h <= height <= max_h


def prediction_size_allowed(prediction, img_width, img_height, min_size_px=0.0, max_percent=1.0):
    xyxy = prediction_xyxy_from_value(prediction, img_width, img_height)
    return prediction_size_allowed_xyxy(
        xyxy,
        img_width,
        img_height,
        min_size_px=min_size_px,
        max_percent=max_percent,
    )


def filter_predictions_to_size_limits(
    predictions,
    img_width,
    img_height,
    min_size_px=0.0,
    max_percent=1.0,
):
    if not predictions:
        return [], 0

    kept = []
    skipped = 0
    for prediction in predictions:
        if prediction_size_allowed(
            prediction,
            img_width,
            img_height,
            min_size_px=min_size_px,
            max_percent=max_percent,
        ):
            kept.append(prediction)
        else:
            skipped += 1

    return kept, skipped


def filter_overlay_to_size_limits(
    overlay,
    img_width=None,
    img_height=None,
    min_size_px=0.0,
    max_percent=1.0,
):
    if not isinstance(overlay, dict):
        return overlay, 0

    frame_shape = overlay.get("frame_shape") or (img_height, img_width)
    try:
        frame_h = int(img_height or frame_shape[0])
        frame_w = int(img_width or frame_shape[1])
    except Exception:
        return overlay, 0

    boxes = list(overlay.get("boxes") or [])
    polygons = list(overlay.get("polygons") or [])
    keypoints = list(overlay.get("keypoints") or [])

    kept_boxes = []
    kept_box_sources = set()
    skipped = 0

    for index, box in enumerate(boxes):
        source_id = index
        if isinstance(box, dict):
            source_id = box.get("source_index", index)

        if prediction_size_allowed(
            box,
            frame_w,
            frame_h,
            min_size_px=min_size_px,
            max_percent=max_percent,
        ):
            kept_boxes.append(box)
            kept_box_sources.add(source_id)
        else:
            skipped += 1

    kept_polygons = []
    has_box_sources = bool(boxes)
    for polygon in polygons:
        source_id = polygon.get("source_box_index") if isinstance(polygon, dict) else None
        if source_id is not None and has_box_sources:
            if source_id not in kept_box_sources:
                skipped += 1
                continue

            polygon_xyxy = prediction_xyxy_from_value(polygon, frame_w, frame_h)
            if polygon_xyxy is not None and not prediction_size_allowed_xyxy(
                polygon_xyxy,
                frame_w,
                frame_h,
                min_size_px=min_size_px,
                max_percent=max_percent,
            ):
                skipped += 1
                continue

            kept_polygons.append(polygon)
            continue

        if prediction_size_allowed(
            polygon,
            frame_w,
            frame_h,
            min_size_px=min_size_px,
            max_percent=max_percent,
        ):
            kept_polygons.append(polygon)
        else:
            skipped += 1

    kept_keypoints = []
    for keypoint_item in keypoints:
        source_id = keypoint_item.get("source_box_index") if isinstance(keypoint_item, dict) else None
        if source_id is not None and has_box_sources:
            if source_id in kept_box_sources:
                kept_keypoints.append(keypoint_item)
            else:
                skipped += 1
            continue

        if prediction_size_allowed(
            keypoint_item,
            frame_w,
            frame_h,
            min_size_px=min_size_px,
            max_percent=max_percent,
        ):
            kept_keypoints.append(keypoint_item)
        else:
            skipped += 1

    filtered = dict(overlay)
    filtered["boxes"] = kept_boxes
    filtered["polygons"] = kept_polygons
    filtered["keypoints"] = kept_keypoints
    if skipped:
        filtered["filtered_size_count"] = int(filtered.get("filtered_size_count", 0) or 0) + skipped

    return filtered, skipped
