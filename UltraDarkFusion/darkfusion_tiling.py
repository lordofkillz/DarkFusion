"""Dataset tiling worker and annotation transforms, independent of the editor."""

import cv2
import json
import logging
import numpy as np
import os
import shutil
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from darkfusion_annotations import BoundingBox
from darkfusion_dataset_operations import (
    DATASET_METADATA_FILES, dataset_metadata_path, resolve_dataset_metadata_path,
)
from darkfusion_image_loading import (
    DATASET_IMAGE_SUFFIXES, OUTPUT_IMAGE_SUFFIXES, normalize_image_extension, reflect_pad_pil,
)
from darkfusion_ui_controls import blocked_signals


logger = logging.getLogger("UltraDarkFusionLogger")


class TileWorker(QThread):
    update_progress = pyqtSignal(int)
    detailed_progress = pyqtSignal(int, int, int)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        directory,
        tile_width,
        tile_height,
        overlap_ratio=0.20,
        image_files=None,
        output_directory=None,
        image_format=".jpg",
        keep_empty_tiles=False,
        pad_edges=True,
        min_area_ratio=0.25,
        annotation_mode=None,
        object_centered_tiles=True,
        tile_layout_mode=None,
    ):
        super().__init__()
        self.directory = os.path.abspath(directory).replace("\\", "/")
        self.tile_width = int(tile_width)
        self.tile_height = int(tile_height)
        self.overlap_ratio = max(0.0, min(0.9, float(overlap_ratio)))
        self.min_area_ratio = max(0.0, min(1.0, float(min_area_ratio)))
        if tile_layout_mode is None:
            tile_layout_mode = "grid_objects" if object_centered_tiles else "grid"
        tile_layout_mode = str(tile_layout_mode or "grid_objects").strip().lower()
        if tile_layout_mode not in {"grid_objects", "objects", "grid"}:
            tile_layout_mode = "grid_objects"
        self.tile_layout_mode = tile_layout_mode
        self.object_centered_tiles = tile_layout_mode in {"grid_objects", "objects"}
        self.image_files = [
            os.path.abspath(path).replace("\\", "/")
            for path in (image_files or [])
            if path and os.path.isfile(path)
        ]
        visible_percent = int(round(self.min_area_ratio * 100.0))
        tile_folder_name = (
            f"tiles_{self.tile_width}x{self.tile_height}"
            f"_ov{int(round(self.overlap_ratio * 100))}"
            f"_vis{visible_percent}"
        )
        if self.tile_layout_mode == "grid_objects":
            tile_folder_name += "_grid_objects"
        elif self.tile_layout_mode == "objects":
            tile_folder_name += "_objects_only"
        else:
            tile_folder_name += "_grid_only"
        self.output_directory = (
            os.path.abspath(output_directory).replace("\\", "/")
            if output_directory
            else os.path.join(self.directory, tile_folder_name).replace("\\", "/")
        )
        image_format = str(image_format or ".jpg").strip().lower()
        self.image_format = "source" if image_format in {"source", "same", "original"} else normalize_image_extension(image_format)
        self.keep_empty_tiles = bool(keep_empty_tiles)
        self.pad_edges = bool(pad_edges)
        self.annotation_mode = str(annotation_mode or "").strip().lower()
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def polygon_label_parse_preference(self):
        if self.annotation_mode in {"seg", "segment", "segmentation"}:
            return "segmentation"
        if self.annotation_mode == "obb":
            return "obb"
        return None

    def run(self):
        try:
            self._run_tiling()
        except Exception as e:
            self.failed.emit(str(e))

    def _run_tiling(self):
        image_files = list(self.image_files)
        if not image_files:
            image_files = [
                os.path.join(self.directory, f).replace("\\", "/")
                for f in os.listdir(self.directory)
                if f.lower().endswith(DATASET_IMAGE_SUFFIXES)
            ]

        total = len(image_files)
        if total <= 0:
            self.update_progress.emit(100)
            self.detailed_progress.emit(0, 0, 0)
            self.completed.emit({
                "processed": 0,
                "total": 0,
                "tiles": 0,
                "failed": [],
                "canceled": False,
                "output_directory": self.output_directory,
            })
            return

        tiled_dir = self.output_directory
        os.makedirs(tiled_dir, exist_ok=True)
        self.copy_dataset_metadata(tiled_dir)

        stride_x = int(self.tile_width * (1.0 - self.overlap_ratio))
        stride_y = int(self.tile_height * (1.0 - self.overlap_ratio))
        stride_x = max(1, stride_x)
        stride_y = max(1, stride_y)

        generated_tiles = 0
        failures = []
        processed = 0
        self.detailed_progress.emit(0, total, 0)

        for i, image_path in enumerate(image_files):
            if self._cancel_requested:
                break

            filename = os.path.basename(image_path)
            label_path = os.path.splitext(image_path)[0] + ".txt"

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    img_w, img_h = image.size
                    labels = self.read_labels(label_path)

                    tile_windows = self.tile_windows_for_image(
                        img_w,
                        img_h,
                        labels,
                        stride_x,
                        stride_y,
                    )

                    tile_id = 0

                    for x1, y1 in tile_windows:
                        if self._cancel_requested:
                            break
                        y2 = min(y1 + self.tile_height, img_h)

                        x2 = min(x1 + self.tile_width, img_w)

                        tile_w = x2 - x1
                        tile_h = y2 - y1

                        if tile_w <= 0 or tile_h <= 0:
                            continue

                        tile_img = image.crop((x1, y1, x2, y2))

                        pad_right = self.tile_width - tile_w
                        pad_bottom = self.tile_height - tile_h

                        if (pad_right > 0 or pad_bottom > 0) and not self.pad_edges:
                            continue

                        if pad_right > 0 or pad_bottom > 0:
                            tile_img = reflect_pad_pil(tile_img, pad_right, pad_bottom)

                        tile_labels = []

                        for bbox in labels:
                            adjusted = self.adjust_annotation_for_tile(
                                bbox=bbox,
                                tile_x1=x1,
                                tile_y1=y1,
                                tile_w=tile_w,
                                tile_h=tile_h,
                                out_w=self.tile_width,
                                out_h=self.tile_height,
                                orig_w=img_w,
                                orig_h=img_h,
                                min_area_ratio=self.min_area_ratio,
                            )

                            if adjusted is not None:
                                tile_labels.append(adjusted.to_str())

                        if not tile_labels and not self.keep_empty_tiles:
                            continue

                        tile_base = os.path.splitext(filename)[0] + f"_tile_{tile_id}"
                        tile_ext = self.tile_extension_for_image(image_path)
                        tile_image_path = os.path.join(tiled_dir, f"{tile_base}{tile_ext}")
                        tile_label_path = os.path.join(tiled_dir, f"{tile_base}.txt")

                        self.save_tile_image(tile_img, tile_image_path)

                        with open(tile_label_path, "w", encoding="utf-8") as labelf:
                            if tile_labels:
                                labelf.write("\n".join(tile_labels) + "\n")

                        tile_id += 1
                        generated_tiles += 1
                        if generated_tiles % 25 == 0:
                            self.detailed_progress.emit(processed, total, generated_tiles)

            except Exception as e:
                logger.error(f"Failed tiling image {image_path}: {e}")
                failures.append(f"{image_path}: {e}")

            if self._cancel_requested:
                break

            processed = i + 1
            self.update_progress.emit(processed * 100 // total)
            self.detailed_progress.emit(processed, total, generated_tiles)

        self.completed.emit({
            "processed": processed,
            "total": total,
            "tiles": generated_tiles,
            "failed": failures,
            "canceled": self._cancel_requested,
            "output_directory": self.output_directory,
        })

    def copy_dataset_metadata(self, target_directory):
        for file_name in DATASET_METADATA_FILES:
            source = resolve_dataset_metadata_path(self.directory, file_name, migrate=True)
            if os.path.isfile(source):
                try:
                    target = dataset_metadata_path(target_directory, file_name, create_parent=True)
                    shutil.copy2(source, target)
                except Exception as e:
                    logger.warning(f"Failed copying tile metadata {source}: {e}")

    def tile_extension_for_image(self, image_path):
        if self.image_format == "source":
            extension = os.path.splitext(image_path)[1].lower()
            return extension if extension in OUTPUT_IMAGE_SUFFIXES else ".jpg"
        return self.image_format

    def save_tile_image(self, tile_img, output_path):
        extension = os.path.splitext(output_path)[1].lower()
        save_kwargs = {}
        save_format = None
        if extension in {".jpg", ".jpeg"}:
            save_format = "JPEG"
            save_kwargs["quality"] = 95
            save_kwargs["subsampling"] = 0
        elif extension == ".webp":
            save_format = "WEBP"
            save_kwargs["quality"] = 95
        elif extension == ".png":
            save_format = "PNG"

        tile_img.save(output_path, format=save_format, **save_kwargs)

    def _tile_positions(self, image_size, tile_size, stride):
        if image_size <= tile_size:
            return [0]

        positions = list(range(0, image_size - tile_size + 1, stride))
        last = image_size - tile_size

        if not positions or positions[-1] != last:
            positions.append(last)

        return sorted(set(max(0, p) for p in positions))

    def _centered_tile_origin(self, center, tile_size, image_size):
        if image_size <= tile_size:
            return 0

        max_origin = max(0, image_size - tile_size)
        origin = int(round(float(center) - (tile_size / 2.0)))
        return max(0, min(max_origin, origin))

    def annotation_pixel_bounds(self, bbox, orig_w, orig_h):
        ann_type = self.annotation_type(bbox)

        try:
            if ann_type in ("bbox", "bbox_kpt"):
                x1 = (bbox.x_center - bbox.width / 2.0) * orig_w
                y1 = (bbox.y_center - bbox.height / 2.0) * orig_h
                x2 = (bbox.x_center + bbox.width / 2.0) * orig_w
                y2 = (bbox.y_center + bbox.height / 2.0) * orig_h
            elif ann_type == "seg":
                pts = np.asarray(bbox.segmentation, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= orig_w
                pts[:, 1] *= orig_h
                x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
                x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            elif ann_type == "obb":
                pts = np.asarray(bbox.obb[:8], dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= orig_w
                pts[:, 1] *= orig_h
                x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
                x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            else:
                return None
        except Exception:
            return None

        x1 = max(0.0, min(float(orig_w), float(x1)))
        y1 = max(0.0, min(float(orig_h), float(y1)))
        x2 = max(0.0, min(float(orig_w), float(x2)))
        y2 = max(0.0, min(float(orig_h), float(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def tile_windows_for_image(self, img_w, img_h, labels, stride_x, stride_y):
        x_positions = self._tile_positions(img_w, self.tile_width, stride_x)
        y_positions = self._tile_positions(img_h, self.tile_height, stride_y)
        windows = []
        seen = set()

        def add_window(x, y):
            x = int(round(float(x)))
            y = int(round(float(y)))
            if img_w > self.tile_width:
                x = max(0, min(img_w - self.tile_width, x))
            else:
                x = 0
            if img_h > self.tile_height:
                y = max(0, min(img_h - self.tile_height, y))
            else:
                y = 0

            key = (x, y)
            if key in seen:
                return
            seen.add(key)
            windows.append(key)

        if self.tile_layout_mode in {"grid_objects", "grid"}:
            for y in y_positions:
                for x in x_positions:
                    add_window(x, y)

        if self.tile_layout_mode in {"grid_objects", "objects"}:
            for bbox in labels or []:
                bounds = self.annotation_pixel_bounds(bbox, img_w, img_h)
                if bounds is None:
                    continue

                x1, y1, x2, y2 = bounds
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                add_window(
                    self._centered_tile_origin(center_x, self.tile_width, img_w),
                    self._centered_tile_origin(center_y, self.tile_height, img_h),
                )

        return sorted(windows, key=lambda point: (point[1], point[0]))

    def _load_expected_keypoint_count(self):
        candidates = [
            resolve_dataset_metadata_path(self.directory, "points.json", migrate=True),
            resolve_dataset_metadata_path(self.directory, "pose.json", migrate=True),
        ]

        for candidate in candidates:
            if not os.path.exists(candidate):
                continue

            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            if not isinstance(data, dict):
                continue

            keypoints = data.get("keypoints")
            if isinstance(keypoints, list) and keypoints:
                return len(keypoints)

            kpt_shape = data.get("kpt_shape")
            if isinstance(kpt_shape, list) and kpt_shape:
                try:
                    return max(0, int(kpt_shape[0]))
                except Exception:
                    pass

        return 0

    def read_labels(self, label_path):
        labels = []
        BoundingBox.EXPECTED_KEYPOINT_COUNT = self._load_expected_keypoint_count() or None
        polygon_preference = self.polygon_label_parse_preference()

        try:
            with open(label_path, "r", encoding="utf-8") as file:
                for line in file.readlines():
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        bb = BoundingBox.from_str(
                            line,
                            preferred_polygon_type=polygon_preference,
                        )
                    except Exception as e:
                        logger.warning(f"Failed parsing label line in {label_path}: {e}")
                        continue

                    if bb is not None:
                        labels.append(bb)

        except FileNotFoundError:
            logger.warning(f"No label file found for {label_path}")

        except Exception as e:
            logger.error(f"Failed reading labels {label_path}: {e}")

        return labels

    def annotation_type(self, bbox):
        if getattr(bbox, "segmentation", None):
            return "seg"
        if getattr(bbox, "obb", None):
            return "obb"
        if getattr(bbox, "keypoints", None):
            return "bbox_kpt"
        return "bbox"

    def adjust_annotation_for_tile(
        self,
        bbox,
        tile_x1,
        tile_y1,
        tile_w,
        tile_h,
        out_w,
        out_h,
        orig_w,
        orig_h,
        min_area_ratio=0.25,
    ):
        ann_type = self.annotation_type(bbox)

        if ann_type == "seg":
            return self.adjust_segmentation_for_tile(
                bbox, tile_x1, tile_y1, tile_w, tile_h, out_w, out_h, orig_w, orig_h, min_area_ratio
            )

        if ann_type == "obb":
            return self.adjust_obb_for_tile(
                bbox, tile_x1, tile_y1, tile_w, tile_h, out_w, out_h, orig_w, orig_h, min_area_ratio
            )

        return self.adjust_bbox_or_keypoints_for_tile(
            bbox, tile_x1, tile_y1, tile_w, tile_h, out_w, out_h, orig_w, orig_h, min_area_ratio
        )

    def adjust_bbox_or_keypoints_for_tile(
        self,
        bbox,
        tile_x1,
        tile_y1,
        tile_w,
        tile_h,
        out_w,
        out_h,
        orig_w,
        orig_h,
        min_area_ratio=0.25,
    ):
        abs_x1 = (bbox.x_center - bbox.width / 2.0) * orig_w
        abs_y1 = (bbox.y_center - bbox.height / 2.0) * orig_h
        abs_x2 = (bbox.x_center + bbox.width / 2.0) * orig_w
        abs_y2 = (bbox.y_center + bbox.height / 2.0) * orig_h

        inter_x1 = max(abs_x1, tile_x1)
        inter_y1 = max(abs_y1, tile_y1)
        inter_x2 = min(abs_x2, tile_x1 + tile_w)
        inter_y2 = min(abs_y2, tile_y1 + tile_h)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return None

        orig_area = max(0.0, abs_x2 - abs_x1) * max(0.0, abs_y2 - abs_y1)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

        if orig_area <= 0.0 or (inter_area / orig_area) < min_area_ratio:
            return None

        tile_box_x1 = inter_x1 - tile_x1
        tile_box_y1 = inter_y1 - tile_y1
        tile_box_x2 = inter_x2 - tile_x1
        tile_box_y2 = inter_y2 - tile_y1

        norm_x1 = self._clip01(tile_box_x1 / out_w)
        norm_y1 = self._clip01(tile_box_y1 / out_h)
        norm_x2 = self._clip01(tile_box_x2 / out_w)
        norm_y2 = self._clip01(tile_box_y2 / out_h)

        new_x_center = (norm_x1 + norm_x2) / 2.0
        new_y_center = (norm_y1 + norm_y2) / 2.0
        new_width = norm_x2 - norm_x1
        new_height = norm_y2 - norm_y1

        if new_width <= 0.001 or new_height <= 0.001:
            return None

        new_keypoints = []

        if getattr(bbox, "keypoints", None):
            for kp in bbox.keypoints:
                try:
                    kx, ky, kv = kp
                    kx = float(kx)
                    ky = float(ky)
                    kv = int(float(kv))
                except Exception:
                    new_keypoints.append((0.0, 0.0, 0))
                    continue

                if kv == 0:
                    new_keypoints.append((0.0, 0.0, 0))
                    continue

                abs_kx = kx * orig_w
                abs_ky = ky * orig_h

                if not (tile_x1 <= abs_kx <= tile_x1 + tile_w and tile_y1 <= abs_ky <= tile_y1 + tile_h):
                    new_keypoints.append((0.0, 0.0, 0))
                    continue

                new_kx = self._clip01((abs_kx - tile_x1) / out_w)
                new_ky = self._clip01((abs_ky - tile_y1) / out_h)

                new_keypoints.append((new_kx, new_ky, kv))

            if not any(kp[2] in (1, 2) for kp in new_keypoints):
                return None

        return BoundingBox(
            bbox.class_id,
            new_x_center,
            new_y_center,
            new_width,
            new_height,
            confidence=bbox.confidence,
            segmentation=[],
            keypoints=new_keypoints,
            obb=[],
        )

    def adjust_segmentation_for_tile(
        self,
        bbox,
        tile_x1,
        tile_y1,
        tile_w,
        tile_h,
        out_w,
        out_h,
        orig_w,
        orig_h,
        min_area_ratio=0.25,
    ):
        seg = getattr(bbox, "segmentation", None)

        if not seg or len(seg) < 6 or len(seg) % 2 != 0:
            return None

        points = []

        for i in range(0, len(seg), 2):
            x = float(seg[i]) * orig_w
            y = float(seg[i + 1]) * orig_h
            points.append((x, y))

        orig_area = abs(self.polygon_area(points))
        if orig_area <= 0:
            return None

        clipped = self.clip_polygon_to_rect(
            points,
            tile_x1,
            tile_y1,
            tile_x1 + tile_w,
            tile_y1 + tile_h,
        )

        if len(clipped) < 3:
            return None

        clipped_area = abs(self.polygon_area(clipped))
        if clipped_area <= 0 or (clipped_area / orig_area) < min_area_ratio:
            return None

        normalized = []

        for x, y in clipped:
            nx = self._clip01((x - tile_x1) / out_w)
            ny = self._clip01((y - tile_y1) / out_h)
            normalized.extend([nx, ny])

        if len(normalized) < 6:
            return None

        return BoundingBox(
            bbox.class_id,
            0,
            0,
            0,
            0,
            confidence=bbox.confidence,
            segmentation=normalized,
            keypoints=[],
            obb=[],
        )

    def adjust_obb_for_tile(
        self,
        bbox,
        tile_x1,
        tile_y1,
        tile_w,
        tile_h,
        out_w,
        out_h,
        orig_w,
        orig_h,
        min_area_ratio=0.25,
    ):
        obb = getattr(bbox, "obb", None)

        if not obb or len(obb) < 8:
            return None

        points = []

        for i in range(0, 8, 2):
            x = float(obb[i]) * orig_w
            y = float(obb[i + 1]) * orig_h
            points.append((x, y))

        orig_area = abs(self.polygon_area(points))
        if orig_area <= 0:
            return None

        clipped = self.clip_polygon_to_rect(
            points,
            tile_x1,
            tile_y1,
            tile_x1 + tile_w,
            tile_y1 + tile_h,
        )

        if len(clipped) < 3:
            return None

        clipped_area = abs(self.polygon_area(clipped))

        if clipped_area <= 0 or (clipped_area / orig_area) < min_area_ratio:
            return None

        try:
            pts = np.array(
                [[x - tile_x1, y - tile_y1] for x, y in clipped],
                dtype=np.float32,
            )

            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)

            normalized = []

            for x, y in box:
                nx = self._clip01(float(x) / out_w)
                ny = self._clip01(float(y) / out_h)
                normalized.extend([nx, ny])

            if len(normalized) != 8:
                return None

            return BoundingBox(
                bbox.class_id,
                0,
                0,
                0,
                0,
                confidence=bbox.confidence,
                segmentation=[],
                keypoints=[],
                obb=normalized,
            )

        except Exception as e:
            logger.warning(f"Failed adjusting OBB tile: {e}")
            return None

    def clip_polygon_to_rect(self, polygon, xmin, ymin, xmax, ymax):

        def clip_edge(points, inside_fn, intersect_fn):
            if not points:
                return []

            output = []
            prev = points[-1]
            prev_inside = inside_fn(prev)

            for curr in points:
                curr_inside = inside_fn(curr)

                if curr_inside:
                    if not prev_inside:
                        output.append(intersect_fn(prev, curr))
                    output.append(curr)
                elif prev_inside:
                    output.append(intersect_fn(prev, curr))

                prev = curr
                prev_inside = curr_inside

            return output

        def intersect_x(p1, p2, x):
            x1, y1 = p1
            x2, y2 = p2

            if x2 == x1:
                return x, y1

            t = (x - x1) / (x2 - x1)
            y = y1 + t * (y2 - y1)
            return x, y

        def intersect_y(p1, p2, y):
            x1, y1 = p1
            x2, y2 = p2

            if y2 == y1:
                return x1, y

            t = (y - y1) / (y2 - y1)
            x = x1 + t * (x2 - x1)
            return x, y

        points = list(polygon)

        points = clip_edge(
            points,
            lambda p: p[0] >= xmin,
            lambda p1, p2: intersect_x(p1, p2, xmin),
        )

        points = clip_edge(
            points,
            lambda p: p[0] <= xmax,
            lambda p1, p2: intersect_x(p1, p2, xmax),
        )

        points = clip_edge(
            points,
            lambda p: p[1] >= ymin,
            lambda p1, p2: intersect_y(p1, p2, ymin),
        )

        points = clip_edge(
            points,
            lambda p: p[1] <= ymax,
            lambda p1, p2: intersect_y(p1, p2, ymax),
        )

        return points

    def polygon_area(self, points):
        if len(points) < 3:
            return 0.0

        area = 0.0

        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            area += x1 * y2 - x2 * y1

        return area / 2.0

    def _clip01(self, value):
        return max(0.0, min(1.0, float(value)))


def ensure_tile_images_controls(self):
    group = getattr(self, "groupBox", None)
    if group is None:
        return

    layout = group.layout()

    spin_style = ""
    spin_source = getattr(self, "w_img", None)
    if isinstance(spin_source, QtWidgets.QSpinBox):
        spin_style = spin_source.styleSheet()

    checkbox_style = ""
    checkbox_source = getattr(self, "crop_images_checkbox", None)
    if isinstance(checkbox_source, QtWidgets.QCheckBox):
        checkbox_style = checkbox_source.styleSheet()

    combo_style = ""
    combo_source = getattr(self, "select_format", None)
    if isinstance(combo_source, QtWidgets.QComboBox):
        combo_style = combo_source.styleSheet()

    overlap = getattr(self, "tile_overlap_spinbox", None)
    if not isinstance(overlap, QtWidgets.QSpinBox):
        overlap = QtWidgets.QSpinBox(group)
        overlap.setObjectName("tile_overlap_spinbox")
        setattr(self, "tile_overlap_spinbox", overlap)
        if isinstance(layout, QtWidgets.QGridLayout):
            layout.addWidget(overlap, 1, 0, 1, 1)
    overlap.setRange(0, 90)
    overlap.setSingleStep(5)
    overlap.setSuffix("%")
    overlap.setValue(max(0, min(90, int(overlap.value() or 20))))
    overlap.setToolTip("Tile overlap percent. Higher overlap and object-centered extra tiles reduce cut-off objects at tile borders.")
    if spin_style:
        overlap.setStyleSheet(spin_style)

    min_visible = getattr(self, "tile_min_visible_spinbox", None)
    if not isinstance(min_visible, QtWidgets.QSpinBox):
        min_visible = QtWidgets.QSpinBox(group)
        min_visible.setObjectName("tile_min_visible_spinbox")
        setattr(self, "tile_min_visible_spinbox", min_visible)
        if isinstance(layout, QtWidgets.QGridLayout):
            layout.addWidget(min_visible, 1, 1, 1, 1)
    min_visible.setRange(1, 100)
    min_visible.setSingleStep(5)
    min_visible.setSuffix("%")
    min_visible.setValue(max(1, min(100, int(min_visible.value() or 25))))
    min_visible.setToolTip("Minimum visible object area required to keep a label in a tile. Pose labels with no remaining keypoints are dropped.")
    if spin_style:
        min_visible.setStyleSheet(spin_style)

    tile_format = getattr(self, "tile_format_combo", None)
    if not isinstance(tile_format, QtWidgets.QComboBox):
        tile_format = QtWidgets.QComboBox(group)
        tile_format.setObjectName("tile_format_combo")
        setattr(self, "tile_format_combo", tile_format)
        if isinstance(layout, QtWidgets.QGridLayout):
            layout.addWidget(tile_format, 1, 2, 1, 1)
    current = tile_format.currentData() or tile_format.currentText() or ".jpg"
    with blocked_signals(tile_format):
        tile_format.clear()
        for text, data in (
            ("Source", "source"),
            (".jpg", ".jpg"),
            (".png", ".png"),
            (".webp", ".webp"),
        ):
            tile_format.addItem(text, data)
        index = tile_format.findData(current)
        if index < 0:
            index = tile_format.findText(str(current))
        tile_format.setCurrentIndex(index if index >= 0 else tile_format.findData(".jpg"))
    tile_format.setToolTip("Output format for generated tiles.")
    if combo_style:
        tile_format.setStyleSheet(combo_style)

    layout_mode = getattr(self, "tile_layout_combo", None)
    if not isinstance(layout_mode, QtWidgets.QComboBox):
        layout_mode = QtWidgets.QComboBox(group)
        layout_mode.setObjectName("tile_layout_combo")
        setattr(self, "tile_layout_combo", layout_mode)
        if isinstance(layout, QtWidgets.QGridLayout):
            layout.addWidget(layout_mode, 1, 3, 1, 1)
    current_layout = layout_mode.currentData() or "grid_objects"
    with blocked_signals(layout_mode):
        layout_mode.clear()
        layout_mode.addItem("Grid + Objects", "grid_objects")
        layout_mode.addItem("Objects Only (Centered)", "objects")
        layout_mode.addItem("Grid Only", "grid")
        index = layout_mode.findData(str(current_layout))
        layout_mode.setCurrentIndex(index if index >= 0 else 0)
    layout_mode.setToolTip(
        "Grid + Objects keeps the normal overlapping grid and adds one centered tile per annotation. "
            "Objects Only creates centered detection crops without the grid."
    )
    if combo_style:
        layout_mode.setStyleSheet(combo_style)

    for column, (object_name, text, tooltip, default_checked) in enumerate((
        (
            "tile_keep_empty_checkbox",
            "Keep Empty",
            "Keep background-only tiles. Turn off to avoid flooding training with empty crops.",
            False,
        ),
        (
            "tile_pad_edges_checkbox",
            "Pad Edges",
            "Reflect-pad edge tiles so every output keeps the requested tile size.",
            True,
        ),
    )):
        checkbox = getattr(self, object_name, None)
        if not isinstance(checkbox, QtWidgets.QCheckBox):
            checkbox = QtWidgets.QCheckBox(group)
            checkbox.setObjectName(object_name)
            setattr(self, object_name, checkbox)
            checkbox.setChecked(default_checked)
            if isinstance(layout, QtWidgets.QGridLayout):
                layout.addWidget(checkbox, 2, column, 1, 1)
        checkbox.setText(text)
        checkbox.setToolTip(tooltip)
        checkbox.setLayoutDirection(Qt.LeftToRight)
        if checkbox_style:
            checkbox.setStyleSheet(checkbox_style)


def _compact_tile_images_panel(self):
    if getattr(self, "_tile_images_panel_compacted", False):
        return

    group = getattr(self, "groupBox", None)
    layout = self._prepare_group_grid(group)
    if layout is None:
        return

    self.ensure_tile_images_controls()
    # Designer captions no longer belong to the rebuilt grid. Leaving
    # them unmanaged can paint stale labels over the new controls.
    for caption in group.findChildren(QtWidgets.QLabel, options=Qt.FindDirectChildrenOnly):
        caption.hide()

    names = (
        "crop_img_button",
        "split_data",
        "w_img",
        "h_img",
        "tile_overlap_spinbox",
        "tile_min_visible_spinbox",
        "tile_format_combo",
        "tile_layout_combo",
        "tile_keep_empty_checkbox",
        "tile_pad_edges_checkbox",
    )
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)

    parameters = QtWidgets.QGridLayout()
    parameters.setObjectName("tile_dimensions_grid")
    parameters.setHorizontalSpacing(8)
    parameters.setVerticalSpacing(8)
    for row, pairs in enumerate((("Tile W", "w_img", "Tile H", "h_img"),
                                 ("Overlap", "tile_overlap_spinbox", "Visible", "tile_min_visible_spinbox"))):
        for column, (caption, name) in enumerate(((pairs[0], pairs[1]), (pairs[2], pairs[3]))):
            widgets[name].setMinimumWidth(72)
            parameters.addWidget(self._workflow_label(caption, group, 52), row, column * 2)
            parameters.addWidget(widgets[name], row, column * 2 + 1)
    parameters.setColumnStretch(1, 1)
    parameters.setColumnStretch(3, 1)
    mode_row = self._row_layout()
    mode_row.addWidget(self._workflow_label("Layout", group, 52))
    mode_row.addWidget(widgets["tile_layout_combo"], 1)
    format_row = self._row_layout()
    format_row.addWidget(self._workflow_label("Format", group, 52))
    format_row.addWidget(widgets["tile_format_combo"])
    format_row.addStretch(1)
    options = self._row_layout()
    options.addWidget(widgets["tile_keep_empty_checkbox"])
    options.addWidget(widgets["tile_pad_edges_checkbox"])
    options.addStretch(1)
    actions = self._row_layout()
    actions.addWidget(widgets["crop_img_button"], 1)
    actions.addWidget(widgets["split_data"], 1)
    for row, child_layout in enumerate((parameters, mode_row, format_row, options, actions)):
        layout.addLayout(child_layout, row, 0)

    self._tile_images_panel_compacted = True


def collect_current_tile_images(self, dataset_dir=None):
    dataset_dir = dataset_dir or self.current_dataset_directory()
    if not dataset_dir:
        return []
    dataset_dir = self.normalize_path(dataset_dir)

    try:
        image_files = self.collect_current_dataset_conversion_files()
    except Exception:
        image_files = []

    if not image_files:
        image_files = self.get_image_files(dataset_dir, extensions=DATASET_IMAGE_SUFFIXES)

    clean_files = []
    seen = set()

    def is_generated_tile_output(image_path):
        try:
            relative = os.path.relpath(image_path, dataset_dir).replace("\\", "/")
        except Exception:
            return False
        parts = [part.lower() for part in relative.split("/") if part]
        return any(part.startswith("tiles_") for part in parts[:-1])

    for image_file in image_files:
        image_file = self.normalize_path(image_file)
        if image_file in seen or not os.path.isfile(image_file):
            continue
        if self.is_placeholder_file(image_file):
            continue
        if is_generated_tile_output(image_file):
            continue
        try:
            common_dir = self.normalize_path(os.path.commonpath([dataset_dir, image_file]))
            if common_dir != dataset_dir:
                continue
        except Exception:
            continue
        clean_files.append(image_file)
        seen.add(image_file)

    return sorted(clean_files)


def tile_overlap_percent(self):
    widget = getattr(self, "tile_overlap_spinbox", None)
    if widget is not None and hasattr(widget, "value"):
        return max(0, min(90, int(widget.value())))
    return 20


def tile_min_visible_percent(self):
    widget = getattr(self, "tile_min_visible_spinbox", None)
    if widget is not None and hasattr(widget, "value"):
        return max(1, min(100, int(widget.value())))
    return 25


def selected_tile_image_format(self):
    combo = getattr(self, "tile_format_combo", None)
    if isinstance(combo, QtWidgets.QComboBox):
        value = str(combo.currentData() or combo.currentText() or ".jpg").strip().lower()
        if value in {"source", "same", "original"}:
            return "source"
        return normalize_image_extension(value)
    return ".jpg"


def selected_tile_layout_mode(self):
    combo = getattr(self, "tile_layout_combo", None)
    if isinstance(combo, QtWidgets.QComboBox):
        value = str(combo.currentData() or "grid_objects").strip().lower()
        if value in {"grid_objects", "objects", "grid"}:
            return value
    return "grid_objects"


def tile_keep_empty_enabled(self):
    checkbox = getattr(self, "tile_keep_empty_checkbox", None)
    return bool(checkbox.isChecked()) if checkbox is not None else False


def tile_pad_edges_enabled(self):
    checkbox = getattr(self, "tile_pad_edges_checkbox", None)
    return bool(checkbox.isChecked()) if checkbox is not None else True


def crop_images(self):
    active_worker = getattr(self, "_tile_worker", None)
    if active_worker is not None:
        try:
            if active_worker.isRunning():
                active_worker.cancel()
                self.statusBar().showMessage("Canceling image tiling...")
                return
        except RuntimeError:
            self._tile_worker = None

    dataset_dir = self.current_dataset_directory()
    image_files = self.collect_current_tile_images(dataset_dir)
    if not dataset_dir or not image_files:
        QMessageBox.warning(self, "Tile Images", "Open an image dataset before tiling.")
        return

    width = int(self.w_img.value())
    height = int(self.h_img.value())
    overlap_percent = self.tile_overlap_percent()
    min_visible_percent = self.tile_min_visible_percent()
    tile_format = self.selected_tile_image_format()
    tile_layout_mode = self.selected_tile_layout_mode()
    layout_suffix = {
        "grid_objects": "grid_objects",
        "objects": "objects_only",
        "grid": "grid_only",
    }[tile_layout_mode]
    output_directory = os.path.join(
        dataset_dir,
        f"tiles_{width}x{height}_ov{overlap_percent}_vis{min_visible_percent}_{layout_suffix}",
    )

    worker = TileWorker(
        dataset_dir,
        width,
        height,
        overlap_ratio=overlap_percent / 100.0,
        image_files=image_files,
        output_directory=output_directory,
        image_format=tile_format,
        keep_empty_tiles=self.tile_keep_empty_enabled(),
        pad_edges=self.tile_pad_edges_enabled(),
        min_area_ratio=min_visible_percent / 100.0,
        annotation_mode=self.current_mode(),
        tile_layout_mode=tile_layout_mode,
    )
    self._tile_worker = worker
    worker.detailed_progress.connect(self.on_tile_progress)
    worker.completed.connect(self.on_tile_completed)
    worker.failed.connect(self.on_tile_failed)
    worker.finished.connect(lambda worker=worker: self.on_tile_worker_finished(worker))

    button = getattr(self, "crop_img_button", None)
    if isinstance(button, QtWidgets.QPushButton):
        self._tile_button_idle_text = button.text()
        button.setText("Cancel Tiling")
        button.setEnabled(True)

    self.set_label_progress(0, max(1, len(image_files)), value=0, progress_format="Tiling 0/{0}".format(len(image_files)))
    self.statusBar().showMessage(f"Tiling {len(image_files)} current-dataset images to {output_directory}")
    worker.start()


def on_tile_progress(self, processed, total, tile_count):
    processed = max(0, int(processed))
    total = max(0, int(total))
    tile_count = max(0, int(tile_count))
    self.set_label_progress(
        0,
        max(1, total),
        value=min(processed, max(1, total)),
        progress_format=f"Tiling {processed}/{total} | {tile_count} tiles",
    )
    self.statusBar().showMessage(
        f"Tiling images: {processed}/{total} processed, {tile_count} tiles created"
    )


def on_tile_completed(self, summary):
    summary = dict(summary or {})
    processed = int(summary.get("processed", 0))
    total = int(summary.get("total", 0))
    tile_count = int(summary.get("tiles", 0))
    failures = list(summary.get("failed", []) or [])
    canceled = bool(summary.get("canceled", False))
    output_directory = summary.get("output_directory", "")

    if canceled:
        self.reset_label_progress(0)
        self.statusBar().showMessage(
            f"Tiling canceled after {processed}/{total} images and {tile_count} tiles.",
            5000,
        )
        return

    self.reset_label_progress(100 if not failures else 0)
    if failures:
        preview = "\n".join(failures[:5])
        more = f"\n...and {len(failures) - 5} more." if len(failures) > 5 else ""
        QMessageBox.warning(
            self,
            "Tile Images",
            (
                f"Tiling finished with {len(failures)} failed image(s).\n"
                    f"Created {tile_count} tiles from {processed}/{total} images.\n\n"
                    f"{preview}{more}"
            ),
        )
    else:
        QMessageBox.information(
            self,
            "Tiling Complete",
            (
                f"Created {tile_count} tiles from {processed} image(s).\n\n"
                    f"Output:\n{output_directory}"
            ),
        )
    self.statusBar().showMessage(
        f"Tiling complete: {tile_count} tiles created in {output_directory}",
        7000,
    )


def on_tile_failed(self, message):
    self.reset_label_progress(0)
    self.statusBar().showMessage("Image tiling failed.", 5000)
    QMessageBox.critical(self, "Tile Images", f"Image tiling failed:\n{message}")


def on_tile_worker_finished(self, worker):
    button = getattr(self, "crop_img_button", None)
    if isinstance(button, QtWidgets.QPushButton):
        button.setText(getattr(self, "_tile_button_idle_text", "Tile Dataset"))
        button.setEnabled(True)
    self._cleanup_worker_reference("_tile_worker", worker)
