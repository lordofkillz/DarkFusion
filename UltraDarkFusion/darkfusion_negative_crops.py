"""Safe crop planning for validation false-positive negative examples."""

from __future__ import annotations

import math
import os

import cv2


NEGATIVE_FOLDER_NAMES = ("blanks", "negative_crops", "negatives", "negative")


def dataset_root_for_image(image_path, label_path="", dataset_hint=""):
    """Resolve the dataset root shared by normal and validation review."""
    image_path = os.path.abspath(str(image_path or ""))
    label_path = os.path.abspath(str(label_path or image_path))
    parts = list(os.path.normpath(image_path).split(os.sep))
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        root = os.sep.join(parts[:index])
        if root and os.path.splitdrive(image_path)[0] and root.endswith(":"):
            root += os.sep
        if root:
            return os.path.abspath(root)

    hint = os.path.abspath(str(dataset_hint or "")) if dataset_hint else ""
    if hint and os.path.isdir(hint):
        try:
            if os.path.commonpath([hint, image_path]) == hint:
                return hint
        except ValueError:
            pass
    try:
        common = os.path.commonpath([os.path.dirname(image_path), os.path.dirname(label_path)])
    except ValueError:
        common = os.path.dirname(image_path)
    return os.path.abspath(common or os.path.dirname(image_path))


def resolve_negative_folder(dataset_dir, create=False):
    """Reuse an existing negative folder; otherwise choose the legacy ``blanks`` path."""
    dataset_dir = os.path.abspath(str(dataset_dir or ""))
    if os.path.basename(dataset_dir).lower() in NEGATIVE_FOLDER_NAMES:
        target = dataset_dir
    else:
        target = ""
        for name in NEGATIVE_FOLDER_NAMES:
            candidate = os.path.join(dataset_dir, name)
            if os.path.isdir(candidate):
                target = candidate
                break
        if not target:
            target = os.path.join(dataset_dir, NEGATIVE_FOLDER_NAMES[0])
    if create:
        os.makedirs(target, exist_ok=True)
    return os.path.abspath(target)


def pad_crop_for_training(image, minimum_size=128, stride=32):
    """Center a native-resolution crop on a reflected, stride-aligned square canvas."""
    if image is None or getattr(image, "size", 0) == 0:
        return None, (0, 0, 0, 0)
    height, width = image.shape[:2]
    stride = max(1, int(stride or 1))
    side = max(width, height, int(minimum_size or 0), stride)
    side = int(math.ceil(side / float(stride)) * stride)
    left = (side - width) // 2
    right = side - width - left
    top = (side - height) // 2
    bottom = side - height - top
    if not any((left, top, right, bottom)):
        return image.copy(), (0, 0, 0, 0)
    # Reflection avoids teaching the model a synthetic solid border. Very tiny
    # crops cannot support reflection, so replicate their edge pixels instead.
    border_type = cv2.BORDER_REFLECT_101 if width > 1 and height > 1 else cv2.BORDER_REPLICATE
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, border_type)
    return padded, (left, top, right, bottom)


def object_bounds(value):
    """Return normalized xyxy bounds from a review object, or None."""
    if not isinstance(value, dict):
        return None
    bounds = list(value.get("bbox", []) or [])
    if len(bounds) < 4:
        points = [
            point for point in list(value.get("points", []) or [])
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if not points:
            return None
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        bounds = [min(xs), min(ys), max(xs), max(ys)]
    try:
        x1, y1, x2, y2 = (float(item) for item in bounds[:4])
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _pixel_bounds(value, width, height):
    bounds = object_bounds(value)
    if bounds is None:
        return None
    return (
        bounds[0] * width,
        bounds[1] * height,
        bounds[2] * width,
        bounds[3] * height,
    )


def _intersects(left, right):
    return (
        left[0] < right[2]
        and left[2] > right[0]
        and left[1] < right[3]
        and left[3] > right[1]
    )


def _centered_rect(cx, cy, crop_width, crop_height, image_width, image_height):
    crop_width = min(float(image_width), max(1.0, float(crop_width)))
    crop_height = min(float(image_height), max(1.0, float(crop_height)))
    x1 = min(max(0.0, cx - crop_width / 2.0), image_width - crop_width)
    y1 = min(max(0.0, cy - crop_height / 2.0), image_height - crop_height)
    return (x1, y1, x1 + crop_width, y1 + crop_height)


def plan_negative_crop(
    image_width,
    image_height,
    prediction,
    ground_truth=(),
    *,
    aspect_ratio=1.0,
    context_scale=2.0,
    minimum_context=96,
    safety_margin=4,
):
    """Plan a centered crop containing one false detection and no known labels.

    The returned rectangle uses integer, exclusive-end pixel coordinates.  A
    failed plan includes a human-readable ``reason`` instead of a rectangle.
    """
    width = int(image_width or 0)
    height = int(image_height or 0)
    bad = _pixel_bounds(prediction, width, height)
    if width <= 0 or height <= 0 or bad is None:
        return {"rect": None, "reason": "The image or prediction bounds are invalid."}

    margin = max(0.0, float(safety_margin))
    protected = []
    for item in ground_truth or ():
        bounds = _pixel_bounds(item, width, height)
        if bounds is None:
            continue
        protected.append((
            max(0.0, bounds[0] - margin),
            max(0.0, bounds[1] - margin),
            min(float(width), bounds[2] + margin),
            min(float(height), bounds[3] + margin),
        ))

    if any(_intersects(bad, bounds) for bounds in protected):
        return {
            "rect": None,
            "reason": (
                "The rejected prediction overlaps a saved ground-truth object, so it cannot "
                "be exported as an empty-label crop safely."
            ),
        }

    bad_width = max(1.0, bad[2] - bad[0])
    bad_height = max(1.0, bad[3] - bad[1])
    cx = (bad[0] + bad[2]) / 2.0
    cy = (bad[1] + bad[3]) / 2.0
    aspect = max(0.1, min(10.0, float(aspect_ratio or 1.0)))

    # The smallest crop fully contains the false detection with a small border.
    border = max(2.0, min(12.0, 0.08 * max(bad_width, bad_height)))
    minimum_width = bad_width + border * 2.0
    minimum_height = bad_height + border * 2.0
    if minimum_width / minimum_height < aspect:
        minimum_width = minimum_height * aspect
    else:
        minimum_height = minimum_width / aspect

    desired_width = max(minimum_width, bad_width * max(1.0, context_scale), float(minimum_context))
    desired_height = max(minimum_height, bad_height * max(1.0, context_scale), float(minimum_context) / aspect)
    if desired_width / desired_height < aspect:
        desired_width = desired_height * aspect
    else:
        desired_height = desired_width / aspect

    fit_scale = min(1.0, width / desired_width, height / desired_height)
    desired_width *= fit_scale
    desired_height *= fit_scale
    minimum_fit = min(1.0, width / minimum_width, height / minimum_height)
    minimum_width *= minimum_fit
    minimum_height *= minimum_fit

    # Prefer the most context that fits, shrinking only when a valid label is near.
    for step in range(81):
        fraction = 1.0 - (step / 80.0)
        crop_width = minimum_width + (desired_width - minimum_width) * fraction
        crop_height = minimum_height + (desired_height - minimum_height) * fraction
        candidate = _centered_rect(cx, cy, crop_width, crop_height, width, height)
        if not (
            candidate[0] <= bad[0]
            and candidate[1] <= bad[1]
            and candidate[2] >= bad[2]
            and candidate[3] >= bad[3]
        ):
            continue
        if any(_intersects(candidate, bounds) for bounds in protected):
            continue
        x1 = max(0, int(math.floor(candidate[0])))
        y1 = max(0, int(math.floor(candidate[1])))
        x2 = min(width, int(math.ceil(candidate[2])))
        y2 = min(height, int(math.ceil(candidate[3])))
        integer_rect = (x1, y1, x2, y2)
        if any(_intersects(integer_rect, bounds) for bounds in protected):
            continue
        return {
            "rect": integer_rect,
            "reason": "",
            "prediction_rect": tuple(int(round(value)) for value in bad),
            "protected_count": len(protected),
        }

    return {
        "rect": None,
        "reason": (
            "No centered crop can contain the entire rejected prediction without also "
            "including a saved ground-truth object."
        ),
    }
