"""Explicit dataset augmentation using the same Real-ESRGAN crop backend.

Callers supply existing annotation regions or SAM masks. This module never
detects objects, edits labels, reads viewer selections, or writes source files.
"""
from concurrent.futures import CancelledError
import math

import cv2
import numpy as np

from darkfusion_object_clarity import RealESRGANCrops


def prepare_object_context(image, bounds):
    """Match the annotation viewer's padded, bounded input crop using BGR arrays."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bounds
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Clarity augmentation bounds are outside the source image")
    size = math.ceil(max(64, x2 - x1 + 32, y2 - y1 + 32))
    left = max(0, min(width - size, math.floor((x1 + x2 - size) / 2)))
    top = max(0, min(height - size, math.floor((y1 + y2 - size) / 2)))
    right, bottom = min(width, left + size), min(height, top + size)
    context = np.ascontiguousarray(image[top:bottom, left:right])
    scale = min(1., 256. / max(context.shape[:2]))
    if scale < 1.:
        context = cv2.resize(context, (max(8, round(context.shape[1] * scale)),
                                      max(8, round(context.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    sx, sy = context.shape[1] * 4 / (right - left), context.shape[0] * 4 / (bottom - top)
    crop = (round((x1 - left) * sx), round((y1 - top) * sy),
            max(1, round((x2 - left) * sx)), max(1, round((y2 - top) * sy)))
    return context, (crop[0], crop[1], max(crop[0] + 1, crop[2]), max(crop[1] + 1, crop[3]))


class ObjectClarityAugmenter:
    """Reuse one model for a batch; blend restored crops inward from each mask."""
    def __init__(self, model_folder, factory=RealESRGANCrops):
        self.model_folder, self.factory = model_folder, factory
        self.model = None

    def __call__(self, image, regions, cancel=None):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Object clarity augmentation requires an 8-bit BGR image")
        result = image.copy()

        def check_cancel():
            if cancel is not None and cancel.is_set():
                raise CancelledError("Object clarity augmentation canceled")

        for bounds, supplied_mask in regions:
            check_cancel()
            x1, y1, x2, y2 = map(int, bounds)
            if x2 <= x1 or y2 <= y1:
                continue
            mask = np.ones((y2 - y1, x2 - x1), np.uint8) if supplied_mask is None else np.asarray(supplied_mask)
            if mask.shape != (y2 - y1, x2 - x1):
                raise ValueError("Object clarity mask does not match its annotation bounds")
            if not np.isfinite(mask).all():
                continue
            mask = (mask > 0).astype(np.uint8)
            if not mask.any():
                continue
            context, crop = prepare_object_context(image, (x1, y1, x2, y2))
            if self.model is None:
                self.model = self.factory(self.model_folder)
            check_cancel()
            restored = self.model(context)
            check_cancel()
            cx1, cy1, cx2, cy2 = crop
            restored = cv2.resize(restored[cy1:cy2, cx1:cx2], (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
            # Feather only inside the annotation, preserving holes/background.
            distance = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
            alpha = np.minimum(distance, 1.)[:, :, None]
            source = image[y1:y2, x1:x2].astype(np.float32)
            blended = np.rint(source + (restored.astype(np.float32) - source) * alpha).clip(0, 255).astype(np.uint8)
            target = result[y1:y2, x1:x2]
            target[mask > 0] = blended[mask > 0]
        check_cancel()
        return result
