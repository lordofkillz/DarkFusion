"""Shared annotation geometry and YOLO label parsing and serialization."""

import logging
from PyQt5.QtCore import QRectF
from shapely.geometry import Polygon, box as shapely_box


class BoundingBox:
    """
    Represents a bounding box for object detection.
    """

    EXPECTED_KEYPOINT_COUNT = None

    @staticmethod

    def _clip(val, minv=0.0, maxv=1.0):
        return max(minv, min(maxv, float(val)))

    def __init__(self, class_id, x_center, y_center, width, height, confidence=None,
                 segmentation=None, keypoints=None, obb=None):
        self.class_id = int(class_id)

        self.x_center = self._clip(x_center)
        self.y_center = self._clip(y_center)
        self.width = self._clip(width)
        self.height = self._clip(height)

        self.confidence = float(confidence) if confidence is not None else None
        self.segmentation = segmentation.copy() if segmentation is not None else []
        self.keypoints = keypoints.copy() if keypoints is not None else []
        self.obb = obb.copy() if obb is not None else []

    @classmethod

    def from_rect(cls, rect, img_width, img_height, class_id, confidence=None):
        x_center = (rect.x() + rect.width() / 2) / img_width
        y_center = (rect.y() + rect.height() / 2) / img_height
        width = rect.width() / img_width
        height = rect.height() / img_height
        return cls(class_id, x_center, y_center, width, height, confidence)

    def to_rect(self, img_width, img_height):
        x = self.x_center * img_width - self.width * img_width / 2
        y = self.y_center * img_height - self.height * img_height / 2
        width = self.width * img_width
        height = self.height * img_height
        return QRectF(x, y, width, height)

    @staticmethod

    def from_str(label_str, preferred_polygon_type=None):
        if not label_str.strip():
            logging.warning("Empty label string encountered.")
            return None

        parts = label_str.strip().split()

        if len(parts) < 2:
            logging.warning(f"Incomplete label string: {label_str}")
            return None

        try:
            class_id = int(float(parts[0]))
            floats = list(map(float, parts[1:]))
        except (ValueError, IndexError) as e:
            logging.warning(f"Failed to parse floats from label string '{label_str}': {e}")
            return None

        def is_normalized_bbox(vals):
            if len(vals) < 4:
                return False

            x_center, y_center, width, height = vals[:4]

            return (
                0.0 <= x_center <= 1.0 and
                0.0 <= y_center <= 1.0 and
                0.0 < width <= 1.0 and
                0.0 < height <= 1.0
            )

        def is_pose_keypoint_line(vals):
            expected_kpts = getattr(BoundingBox, "EXPECTED_KEYPOINT_COUNT", None)

            if expected_kpts is not None and expected_kpts > 0:
                expected_float_count = 4 + (expected_kpts * 3)

                if len(vals) != expected_float_count:
                    return False
            elif len(vals) <= 4 or (len(vals) - 4) % 3 != 0:
                return False

            if not is_normalized_bbox(vals):
                return False

            keypoint_values = vals[4:]

            for i in range(0, len(keypoint_values), 3):
                try:
                    x = float(keypoint_values[i])
                    y = float(keypoint_values[i + 1])
                    visibility_value = float(keypoint_values[i + 2])
                except Exception:
                    return False

                if not visibility_value.is_integer():
                    return False

                v = int(visibility_value)
                if v not in (0, 1, 2):
                    return False

                if v in (1, 2):
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                        return False

            return True

        polygon_preference = str(preferred_polygon_type or "").strip().lower()

        if len(floats) == 8 and polygon_preference in {"seg", "segment", "segmentation"}:
            return BoundingBox(class_id, 0, 0, 0, 0, segmentation=floats)

        if len(floats) == 8:
            return BoundingBox(class_id, 0, 0, 0, 0, obb=floats)

        if len(floats) == 4:
            x_center, y_center, width, height = floats
            return BoundingBox(class_id, x_center, y_center, width, height)

        if len(floats) == 5:
            x_center, y_center, width, height, confidence = floats
            return BoundingBox(class_id, x_center, y_center, width, height, confidence=confidence)

        if is_pose_keypoint_line(floats):
            x_center, y_center, width, height = floats[:4]
            keypoints = [
                (floats[i], floats[i + 1], int(float(floats[i + 2])))
                for i in range(4, len(floats), 3)
            ]
            return BoundingBox(class_id, x_center, y_center, width, height, keypoints=keypoints)

        if len(floats) >= 6 and len(floats) % 2 == 0:
            return BoundingBox(class_id, 0, 0, 0, 0, segmentation=floats)

        logging.warning(f"Unrecognized label format: {label_str}")
        return None

    def to_str(self, remove_confidence=True):
        class_id_str = f"{int(self.class_id)}"

        if self.obb:
            obb_str = " ".join(f"{coord:.6f}" for coord in self.obb)
            return f"{class_id_str} {obb_str}"

        if self.segmentation:
            seg_str = " ".join(f"{coord:.6f}" for coord in self.segmentation)
            return f"{class_id_str} {seg_str}"

        # Preserve tiny image-pixel boxes through YOLO save/reload cycles.
        # Six decimal places can round a 4px box below its configured minimum.
        bbox_str = f"{class_id_str} {self.x_center:.16f} {self.y_center:.16f} {self.width:.16f} {self.height:.16f}"

        if self.keypoints:
            kpt_str = " ".join(f"{x:.6f} {y:.6f} {v:d}" for x, y, v in self.keypoints)
            return f"{bbox_str} {kpt_str}"

        if self.confidence is not None and not remove_confidence:
            bbox_str += f" {self.confidence:.6f}"

        return bbox_str

    def to_dict(self):
        return {
            "class_id": self.class_id,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "segmentation": self.segmentation,
            "keypoints": self.keypoints,
            "obb": self.obb,
        }

    @property

    def confidence_safe(self):
        try:
            return float(self.confidence)
        except (TypeError, ValueError):
            return 0.0

    def num_visible_keypoints(self):
        return sum(
            1 for kp in self.keypoints
            if isinstance(kp, (list, tuple)) and len(kp) == 3 and kp[2] == 2
        )

    def iou(self, other):
        if not isinstance(other, BoundingBox):
            return 0.0

        x1_min = self.x_center - self.width / 2
        y1_min = self.y_center - self.height / 2
        x1_max = self.x_center + self.width / 2
        y1_max = self.y_center + self.height / 2

        x2_min = other.x_center - other.width / 2
        y2_min = other.y_center - other.height / 2
        x2_max = other.x_center + other.width / 2
        y2_max = other.y_center + other.height / 2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - inter_area

        if union <= 0:
            return 0.0

        return inter_area / union

    def to_polygon(self, image_width, image_height):
        if self.segmentation:
            points = [
                (self.segmentation[i] * image_width, self.segmentation[i + 1] * image_height)
                for i in range(0, len(self.segmentation), 2)
            ]
            return Polygon(points)

        if self.obb:
            points = [
                (self.obb[i] * image_width, self.obb[i + 1] * image_height)
                for i in range(0, 8, 2)
            ]
            return Polygon(points)

        rect = self.to_rect(image_width, image_height)
        return shapely_box(rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())

    def to_xyxy(self, img_width, img_height):
        x1 = (self.x_center - self.width / 2) * img_width
        y1 = (self.y_center - self.height / 2) * img_height
        x2 = (self.x_center + self.width / 2) * img_width
        y2 = (self.y_center + self.height / 2) * img_height
        return x1, y1, x2, y2


def class_name_index(class_name, class_names):
    """Match exact names first, then unique case-insensitive names; never guess."""
    names = [str(name).strip() for name in class_names]
    name = str(class_name).strip()
    if name in names:
        return names.index(name)
    matches = [index for index, value in enumerate(names) if value.casefold() == name.casefold()]
    return matches[0] if len(matches) == 1 else None


class _LabelRectEntry(tuple):
    """Keep a persisted row ID without changing the legacy bbox tuple format."""

    def __new__(cls, values, true_line_index=None):
        entry = super().__new__(cls, values)
        entry.true_line_index = true_line_index
        return entry


DEFAULT_POSE_KEYPOINT_LABELS = (
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle",
)

DEFAULT_POSE_SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

DEFAULT_POSE_FLIP_IDX = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


class InvalidAnnotationError(Exception): ...
