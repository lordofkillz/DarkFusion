"""Shared SAM loading, startup warmup, prompt handling, and annotation snapping.

Callbacks operate on the existing main-window owner; importing this module does
not construct predictors or start warmup jobs.
"""

import cv2
import logging
import math
import numpy as np
import os
import psutil
import threading
import time
import torch
from darkfusion_gpu_memory import adopt_saved_training, model_allowed as gpu_model_allowed
from darkfusion_inference import (
    SAM3_DEFAULT_IMGSZ, normalize_sam_imgsz, sam_predict_with_image_cache,
    sam_predictor_session,
)
from threading import Thread
from ultralytics import SAM


logger = logging.getLogger("UltraDarkFusionLogger")


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ensure_sam_model_loaded(self):
    """
        Load the SAM3 model once and reuse it.
        Expected model path: Sam/sam3.pt
        """
    if not gpu_model_allowed(self, "sam_prompt"):
        return False
    if getattr(self, "sam_model", None) is not None:
        return True
    lock = getattr(self, "_sam_model_load_lock", None)
    if lock is None:
        lock = threading.RLock()
        self._sam_model_load_lock = lock

    with lock:
        if getattr(self, "sam_model", None) is not None:
            return True

        try:
            sam_path = self._sam_model_path()
            if not os.path.isabs(sam_path):
                sam_path = os.path.join(os.getcwd(), sam_path)

            if not os.path.exists(sam_path):
                logger.warning(f"SAM3 model not found: {sam_path}")
                self.sam_model = None
                return False

            sam_stat = os.stat(sam_path)
            self.sam_model = SAM(sam_path)
            self._sam_loaded_checkpoint_identity = (
                os.path.normcase(os.path.realpath(os.path.abspath(sam_path))),
                sam_stat.st_size, sam_stat.st_mtime_ns,
            )
            logger.info(
                "Loaded SAM3 model: %s (imgsz=%s, fp16=%s)",
                sam_path,
                self._sam_imgsz(),
                self._sam_half_enabled(),
            )
            return True

        except Exception as e:
            logger.error(f"Failed loading SAM3 model: {e}")
            self.sam_model = None
            return False


def ensure_sam_semantic_predictor_loaded(self, imgsz=None):
    """Lazily create the SAM3 text-prompt predictor used by polygon Snap only."""
    if not gpu_model_allowed(self, "sam_semantic"):
        return None
    sam_path = os.path.abspath(self._sam_model_path())
    predictor = getattr(self, "sam_semantic_snap_predictor", None)
    if predictor is not None and getattr(self, "_sam_semantic_snap_model_path", None) == sam_path:
        return predictor
    lock = getattr(self, "_sam_model_load_lock", None)
    if lock is None:
        lock = threading.RLock()
        self._sam_model_load_lock = lock

    with lock:
        sam_path = self._sam_model_path()
        if not os.path.isabs(sam_path):
            sam_path = os.path.join(os.getcwd(), sam_path)
        sam_path = os.path.abspath(sam_path)

        predictor = getattr(self, "sam_semantic_snap_predictor", None)
        if (
            predictor is not None
            and getattr(self, "_sam_semantic_snap_model_path", None) == sam_path
        ):
            return predictor

        if not os.path.exists(sam_path):
            logger.warning("SAM3 semantic Snap model not found: %s", sam_path)
            return None

        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            semantic_imgsz = int(math.ceil(self._sam_imgsz(imgsz) / 14.0) * 14)
            overrides = {
                "conf": 0.15,
                "iou": 0.70,
                "task": "segment",
                "mode": "predict",
                "model": sam_path,
                "imgsz": semantic_imgsz,
                "device": self._sam_device(),
                "save": False,
                "verbose": False,
            }
            if self._sam_half_enabled() and str(overrides["device"]).startswith("cuda"):
                overrides["quantize"] = 16

            predictor = SAM3SemanticPredictor(overrides=overrides)
            # Ultralytics 8.4.113's SAM3SemanticPredictor thresholds mask
            # logits with model.mask_threshold, but SAM3SemanticModel does not
            # define that otherwise-standard SAM attribute.  Initialize the
            # model here and supply the normal SAM logit threshold without
            # patching the installed package.
            predictor.setup_model()
            if not hasattr(predictor.model, "mask_threshold"):
                predictor.model.mask_threshold = 0.0
                logger.info(
                    "Applied SAM3 semantic mask-threshold compatibility fix."
                )
            self.sam_semantic_snap_predictor = predictor
            self._sam_semantic_snap_model_path = sam_path
            logger.info(
                "Initialized SAM3 semantic predictor for polygon Snap (imgsz=%s).",
                overrides["imgsz"],
            )
            return predictor
        except Exception as e:
            logger.warning("Could not initialize semantic polygon Snap: %s", e)
            self.sam_semantic_snap_predictor = None
            self._sam_semantic_snap_model_path = None
            return None


def start_sam3_startup_warmup(self, requested=False, prompt_only=False):
    """Preload and prime the reusable SAM3 runtimes without blocking the UI."""
    if getattr(self, "_gpu_memory", None) is not None:
        adopt_saved_training(self)
    if not gpu_model_allowed(self, "sam_prompt"):
        return False
    if not requested and not bool(getattr(self, "settings", {}).get("sam3StartupWarmup", True)):
        logger.info("SAM3 startup warm-up is disabled.")
        return False
    if getattr(self, "_sam3_startup_warmup_active", False):
        return False
    worker = getattr(self, "_sam3_startup_warmup_thread", None)
    if worker is not None and worker.is_alive():
        return False

    sam_path = self._sam_model_path()
    if not os.path.isabs(sam_path):
        sam_path = os.path.join(os.getcwd(), sam_path)
    if not os.path.isfile(sam_path):
        logger.warning("Skipping SAM3 startup warm-up; model not found: %s", sam_path)
        return False

    prompt_allowed, semantic_allowed, capacity_note = self._sam3_startup_warmup_capacity()
    if not prompt_allowed and not requested:
        logger.warning("Skipping SAM3 startup warm-up: %s", capacity_note)
        return False
    semantic_allowed = semantic_allowed and not prompt_only
    self._sam3_startup_warm_semantic = semantic_allowed
    if not semantic_allowed:
        logger.info(
            "Startup will warm prompt SAM3 only; semantic Snap remains lazy (%s).",
            capacity_note,
        )

    self._sam3_startup_warmup_active = True
    self._sam3_startup_warmup_thread = Thread(
        target=self._run_sam3_startup_warmup,
        name="SAM3StartupWarmup",
        daemon=True,
    )
    self._sam3_startup_warmup_thread.start()
    logger.info("SAM3 startup warm-up started in the background.")
    return True


def _sam3_startup_warmup_capacity(self):
    """Avoid eagerly allocating both 3.45 GB SAM3 runtimes on small systems."""
    gib = float(1024 ** 3)
    try:
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gib, total_gib = free_bytes / gib, total_bytes / gib
            prompt_allowed = free_gib >= 5.0
            semantic_allowed = total_gib >= 12.0 and free_gib >= 11.0
            return (
                prompt_allowed,
                semantic_allowed,
                f"GPU memory {free_gib:.1f} GB free / {total_gib:.1f} GB total",
            )

        available_gib = psutil.virtual_memory().available / gib
        return (
            available_gib >= 9.0,
            available_gib >= 18.0,
            f"system memory {available_gib:.1f} GB available",
        )
    except Exception as e:
        logger.debug("Could not measure SAM3 warm-up capacity: %s", e)
        return True, False, "memory capacity could not be measured"


def _run_sam3_startup_warmup(self):
    prompt_ready = False
    prompt_primed = False
    semantic_ready = False
    semantic_primed = False
    try:
        prompt_ready = self.ensure_sam_model_loaded()
        if prompt_ready:
            # Prime the configured encoder shape used by Snap, so its first
            # real prompt does not initialize kernels for a different size.
            # Real images still require their own encoder pass when opened.
            dummy = np.zeros((96, 96, 3), dtype=np.uint8)
            cv2.rectangle(dummy, (24, 24), (72, 72), (255, 255, 255), -1)
            prompt_primed = self.get_sam_mask(
                dummy, [20, 20, 76, 76], imgsz=self._sam_imgsz()
            ) is not None

        # Polygon Snap has a separate text-prompt predictor. Prime its full
        # prediction path too, rather than deferring kernel setup to a draw.
        if getattr(self, "_sam3_startup_warm_semantic", True):
            predictor = self.ensure_sam_semantic_predictor_loaded()
            semantic_ready = predictor is not None
            if semantic_ready and gpu_model_allowed(self, "sam_semantic"):
                try:
                    with sam_predictor_session(predictor), torch.inference_mode():
                        try:
                            dummy = np.zeros((96, 96, 3), dtype=np.uint8)
                            cv2.rectangle(dummy, (24, 24), (72, 72), (255, 255, 255), -1)
                            sam_predict_with_image_cache(
                                predictor, dummy, text=["visual"],
                                bboxes=[[20, 20, 76, 76]], labels=[1],
                            )
                            semantic_primed = True
                        finally:
                            predictor.reset_image()
                except Exception as e:
                    logger.warning("SAM3 semantic prediction warm-up did not complete: %s", e)
        logger.info(
            "SAM3 startup warm-up finished (prompt_loaded=%s, prompt_primed=%s, semantic_loaded=%s, semantic_primed=%s).",
            prompt_ready,
            prompt_primed,
            semantic_ready,
            semantic_primed,
        )
    except Exception as e:
        logger.warning("SAM3 startup warm-up did not complete: %s", e, exc_info=True)
        self._safe_empty_cache()
    finally:
        self._sam3_startup_warmup_active = False
        if not getattr(self, "_sam3_propagation_shutting_down", False):
            finished = getattr(self, "sam3_startup_warmup_finished", None)
            if finished is not None:
                finished.emit()


def mask_to_obb_points(self, mask, img_width, img_height):
    """
        Convert binary mask to 4 normalized OBB corner points using cv2.minAreaRect().
        Returns [(x_norm, y_norm), ...] or [].
        """
    if mask is None:
        return []

    try:
        mask = (mask > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) < 10:
            return []

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)

        # Keep SAM-derived OBBs in the same corner order used by _obb_line_from_mask:
        # top-left, top-right, bottom-right, bottom-left.
        box = sorted(box, key=lambda p: (p[1], p[0]))
        top_two = sorted(box[:2], key=lambda p: p[0])
        bottom_two = sorted(box[2:], key=lambda p: p[0], reverse=True)
        box = np.array([
            top_two[0],
            top_two[1],
            bottom_two[0],
            bottom_two[1],
        ])

        points = []

        for x, y in box:
            nx = max(0.0, min(1.0, float(x) / float(img_width)))
            ny = max(0.0, min(1.0, float(y) / float(img_height)))
            points.append((nx, ny))

        if len(points) != 4:
            return []

        return points

    except Exception as e:
        logger.warning(f"Failed converting SAM3 mask to OBB: {e}")
        return []


def pad_bbox_xyxy(self, bbox_xyxy, img_w, img_h, pad_ratio=0.08):
    """
        Add padding around a bbox prompt before sending it to SAM3.
        bbox_xyxy = [x1, y1, x2, y2]
        """
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    if x2 <= x1 or y2 <= y1:
        return [x1, y1, x2, y2]

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w - 1, x2 + pad_x)
    y2 = min(img_h - 1, y2 + pad_y)

    return [x1, y1, x2, y2]


def validate_sam_mask_for_bbox(
    self,
    mask,
    original_bbox_xyxy,
    class_name="",
    min_area=20,
    max_area_multiplier=3.0,
):
    if mask is None:
        return False

    try:
        mask_area = int(np.count_nonzero(mask))
    except Exception:
        return False

    if mask_area < min_area:
        return False

    x1, y1, x2, y2 = [int(v) for v in original_bbox_xyxy]
    bbox_area = max(1, abs(x2 - x1) * abs(y2 - y1))

    class_name = (class_name or "").lower()

    if class_name in {"person", "human", "player"}:
        max_area_multiplier = 2.5
    elif class_name in {"skateboard", "surfboard", "snowboard", "gun", "rifle", "knife", "tool"}:
        max_area_multiplier = 1.8
    elif class_name in {"car", "truck", "bus", "train"}:
        max_area_multiplier = 3.5

    if mask_area > bbox_area * max_area_multiplier:
        logger.warning(
            f"SAM3 mask rejected: too large for class '{class_name}' "
                f"(mask_area={mask_area}, bbox_area={bbox_area}, max={max_area_multiplier}x)"
        )
        return False

    # Make sure most of the mask is still inside/near the original bbox.
    crop = mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return False

    inside_area = int(np.count_nonzero(crop))
    inside_ratio = inside_area / max(1, mask_area)

    if class_name in {"person", "human", "player"}:
        min_inside_ratio = 0.45
    elif class_name in {"skateboard", "surfboard", "snowboard", "gun", "rifle", "knife", "tool"}:
        min_inside_ratio = 0.60
    else:
        min_inside_ratio = 0.50

    if inside_ratio < min_inside_ratio:
        logger.warning(
            f"SAM3 mask rejected: too much outside bbox for class '{class_name}' "
                f"(inside_ratio={inside_ratio:.2f}, required={min_inside_ratio:.2f})"
        )
        return False

    return True


def polygon_guide_mask(self, normalized_points, img_width, img_height):
    """Rasterize normalized polygon points into an image-sized binary guide."""
    if not normalized_points or len(normalized_points) < 3:
        return None

    try:
        img_width = int(img_width)
        img_height = int(img_height)
        if img_width <= 1 or img_height <= 1:
            return None

        pixel_points = []
        for x, y in normalized_points:
            px = int(round(float(x) * img_width))
            py = int(round(float(y) * img_height))
            pixel_points.append(
                [
                    max(0, min(px, img_width - 1)),
                    max(0, min(py, img_height - 1)),
                ]
            )

        contour = np.asarray(pixel_points, dtype=np.int32)
        if contour.shape[0] < 3 or abs(cv2.contourArea(contour)) < 2.0:
            return None

        guide_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(guide_mask, [contour], 1)
        return guide_mask if np.count_nonzero(guide_mask) >= 3 else None
    except Exception as e:
        logger.warning("Failed creating polygon guide mask: %s", e)
        return None


def select_sam_mask_for_polygon(self, mask, guide_mask, min_overlap_ratio=0.03):
    """
        Select the SAM component that best agrees with a hand-drawn polygon.

        The guide is deliberately used after the bbox-prompt prediction: this
        prevents a larger nearby object from winning merely because it produced
        the largest contour.  Excessive connected spill is clipped to a modest
        band around the rough polygon, while ordinary boundary refinement is
        retained.
        """
    if mask is None or guide_mask is None:
        return None

    try:
        candidate_mask = (np.asarray(mask) > 0).astype(np.uint8)
        guide = (np.asarray(guide_mask) > 0).astype(np.uint8)
        if candidate_mask.ndim > 2:
            candidate_mask = candidate_mask.squeeze()
        if guide.ndim > 2:
            guide = guide.squeeze()
        if candidate_mask.ndim != 2 or guide.ndim != 2:
            return None
        if candidate_mask.shape != guide.shape:
            candidate_mask = cv2.resize(
                candidate_mask,
                (guide.shape[1], guide.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        guide_area = int(np.count_nonzero(guide))
        if guide_area < 3 or np.count_nonzero(candidate_mask) < 3:
            return None

        guide_points = cv2.findNonZero(guide)
        gx, gy, gw, gh = cv2.boundingRect(guide_points)
        moments = cv2.moments(guide, binaryImage=True)
        guide_cx = moments["m10"] / max(moments["m00"], 1.0)
        guide_cy = moments["m01"] / max(moments["m00"], 1.0)
        guide_diag = max(1.0, math.hypot(gw, gh))

        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidate_mask,
            connectivity=8,
        )
        best_label = None
        best_score = -1.0
        best_overlap = 0

        for label in range(1, component_count):
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            if component_area < 3:
                continue

            component = labels == label
            overlap = int(np.count_nonzero(component & (guide > 0)))
            if overlap <= 0:
                continue

            guide_coverage = overlap / max(1, guide_area)
            component_precision = overlap / max(1, component_area)
            union = guide_area + component_area - overlap
            iou = overlap / max(1, union)
            cx, cy = centroids[label]
            center_distance = math.hypot(cx - guide_cx, cy - guide_cy) / guide_diag
            center_agreement = max(0.0, 1.0 - center_distance)
            centroid_inside = 1.0 if guide[
                max(0, min(int(round(cy)), guide.shape[0] - 1)),
                max(0, min(int(round(cx)), guide.shape[1] - 1)),
            ] else 0.0

            score = (
                0.48 * guide_coverage
                + 0.22 * component_precision
                + 0.15 * iou
                + 0.10 * center_agreement
                + 0.05 * centroid_inside
            )
            if score > best_score:
                best_score = score
                best_label = label
                best_overlap = overlap

        required_overlap = max(3, int(math.ceil(guide_area * float(min_overlap_ratio))))
        if best_label is None or best_overlap < required_overlap:
            logger.debug(
                "Polygon-guided SAM mask rejected: overlap=%s required=%s.",
                best_overlap,
                required_overlap,
            )
            return None

        selected = (labels == best_label).astype(np.uint8)

        # Keep useful boundary expansion, but cut off a connected mask that
        # travels far beyond the shape the user actually traced.
        margin = max(4, min(96, int(round(max(gw, gh) * 0.12))))
        guide_contours, _ = cv2.findContours(
            (guide * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        allowed = guide.copy()
        if guide_contours:
            cv2.polylines(
                allowed,
                guide_contours,
                True,
                1,
                thickness=(margin * 2) + 1,
                lineType=cv2.LINE_8,
            )

        trimmed = selected & allowed
        if np.count_nonzero(trimmed) >= max(10, required_overlap):
            selected = trimmed.astype(np.uint8)

        return selected if np.count_nonzero(selected) >= 3 else None
    except Exception as e:
        logger.warning("Failed selecting polygon-guided SAM mask: %s", e)
        return None


def _annotation_snap_image(self, image, bbox_xyxy):
    """Let interactive SAM3 Snap see active clarity without changing inputs elsewhere."""
    clarity = getattr(self, "_object_clarity_controller", None)
    if (clarity is None or image is not getattr(self, "processed_image", None)
            or not getattr(self, "annotation_scene_active", False)
            or getattr(self, "train_view_active", False)):
        return image
    try:
        return clarity.image_for_snap(image, bbox_xyxy)
    except Exception as error:
        logger.warning("Clarity pixels unavailable for Snap; using normal input: %s", error)
        return image


def sam_snap_class_polygon_to_mask(
    self,
    image_np,
    bbox_xyxy,
    class_name,
    guide_mask,
    imgsz=1036,
):
    """
        Use the current dropdown class as a SAM3 concept prompt for polygon Snap.

        This helper is intentionally called only by SegmentationDrawer's Snap
        finalize path.  The rough polygon selects the intended instance from the
        semantic candidates and constrains distant mask spill.
        """
    prompt = self.sam_text_prompt_for_class(class_name)
    if not prompt or image_np is None or guide_mask is None:
        return None
    if not isinstance(image_np, np.ndarray) or image_np.size == 0:
        return None

    snap_started = time.perf_counter()
    predictor = self.ensure_sam_semantic_predictor_loaded(imgsz=imgsz)
    if predictor is None:
        return None

    try:
        img_h, img_w = image_np.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        image_for_sam = image_np
        if getattr(self, "_object_clarity_controller", None) is not None:
            image_for_sam = self._annotation_snap_image(image_np, bbox_xyxy)
        if image_for_sam.ndim == 2:
            image_for_sam = cv2.cvtColor(image_for_sam, cv2.COLOR_GRAY2BGR)
        elif image_for_sam.ndim == 3 and image_for_sam.shape[2] == 4:
            image_for_sam = cv2.cvtColor(image_for_sam, cv2.COLOR_BGRA2BGR)
        elif image_for_sam.ndim != 3 or image_for_sam.shape[2] != 3:
            return None
        image_for_sam = np.ascontiguousarray(image_for_sam)

        with sam_predictor_session(predictor), torch.inference_mode():
            results, image_reused = sam_predict_with_image_cache(
                predictor, image_for_sam,
                text=[prompt],
                bboxes=[[x1, y1, x2, y2]],
                labels=[1],
            )

        if not results:
            return None

        guide = (np.asarray(guide_mask) > 0).astype(np.uint8)
        guide_area = int(np.count_nonzero(guide))
        if guide_area < 3:
            return None

        best_mask = None
        best_score = -1.0
        for result in results:
            result_masks = getattr(getattr(result, "masks", None), "data", None)
            if result_masks is None:
                continue

            for candidate in result_masks:
                if hasattr(candidate, "detach"):
                    candidate = candidate.detach().cpu().numpy()
                candidate = np.asarray(candidate)
                if candidate.ndim > 2:
                    candidate = candidate.squeeze()
                candidate = (candidate > 0.5).astype(np.uint8)
                if candidate.shape != (img_h, img_w):
                    candidate = cv2.resize(
                        candidate,
                        (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                selected = self.select_sam_mask_for_polygon(candidate, guide)
                if selected is None:
                    continue

                selected_area = int(np.count_nonzero(selected))
                overlap = int(np.count_nonzero((selected > 0) & (guide > 0)))
                union = guide_area + selected_area - overlap
                guide_coverage = overlap / max(1, guide_area)
                iou = overlap / max(1, union)
                precision = overlap / max(1, selected_area)
                score = 0.55 * guide_coverage + 0.30 * iou + 0.15 * precision
                if score > best_score:
                    best_score = score
                    best_mask = selected

        if best_mask is not None:
            logger.debug(
                "Polygon Snap used dropdown concept '%s' (match=%.3f, image_reused=%s, elapsed_ms=%.1f).",
                prompt,
                best_score,
                image_reused,
                (time.perf_counter() - snap_started) * 1000.0,
            )
        return best_mask
    except Exception as e:
        logger.warning(
            "SAM3 dropdown-guided polygon Snap failed for '%s': %s",
            prompt,
            e,
        )
        return None


def sam_snap_bbox_to_mask(self, image_np, bbox_xyxy, imgsz=1036):
    """
        Run SAM3 on one drawn bbox prompt and return a binary mask or None.

        bbox_xyxy = [x1, y1, x2, y2] in image pixels.
        """
    if image_np is None:
        return None

    if not isinstance(image_np, np.ndarray) or image_np.size == 0:
        return None

    if not self.ensure_sam_model_loaded():
        return None

    try:
        img_h, img_w = image_np.shape[:2]

        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid SAM3 bbox prompt: {bbox_xyxy}")
            return None

        image_for_sam = image_np

        if getattr(self, "_object_clarity_controller", None) is not None:
            image_for_sam = self._annotation_snap_image(image_np, bbox_xyxy)

        # Ultralytics expects OpenCV BGR arrays and converts to RGB itself.
        # Preserve that contract, including the polygon snap fallback.
        if len(image_for_sam.shape) == 2:
            image_for_sam = cv2.cvtColor(image_for_sam, cv2.COLOR_GRAY2BGR)
        elif len(image_for_sam.shape) == 3 and image_for_sam.shape[2] == 4:
            image_for_sam = cv2.cvtColor(image_for_sam, cv2.COLOR_BGRA2BGR)

        with torch.inference_mode():
            results = self.sam_model.predict(
                source=image_for_sam,
                bboxes=[[x1, y1, x2, y2]],
                **self._sam_predict_kwargs(imgsz=imgsz),
            )

        if not results:
            return None

        result = results[0]

        if not hasattr(result, "masks") or result.masks is None:
            return None

        masks = result.masks.data

        if masks is None or len(masks) == 0:
            return None

        mask = masks[0]

        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()

        mask = np.asarray(mask)

        if mask.ndim > 2:
            mask = mask.squeeze()

        mask = (mask > 0.5).astype(np.uint8)

        if mask.shape[:2] != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        if np.count_nonzero(mask) <= 0:
            return None

        return mask

    except Exception as e:
        logger.warning(f"SAM3 bbox snap failed: {e}")
        return None


def mask_to_bbox_rect(self, mask):
    """
        Convert binary mask to x, y, w, h pixel bbox.
        """
    if mask is None:
        return None

    try:
        mask = (mask > 0).astype(np.uint8)
        points = cv2.findNonZero(mask)

        if points is None:
            return None

        x, y, w, h = cv2.boundingRect(points)

        if w <= 1 or h <= 1:
            return None

        return int(x), int(y), int(w), int(h)

    except Exception as e:
        logger.warning(f"Failed converting SAM3 mask to bbox: {e}")
        return None


def mask_to_polygon_points(
    self,
    mask,
    img_width,
    img_height,
    epsilon_ratio=0.0008,
    max_points=300,
    preserve_thin=False,
):
    """
        Convert binary mask to normalized Ultralytics YOLO segmentation points.

        Output:
            [(x_norm, y_norm), ...]
        Later save path should flatten to:

            class x1 y1 x2 y2 x3 y3 ...
        """
    if mask is None:
        return []

    try:
        mask = np.asarray(mask)

        if mask.ndim > 2:
            mask = mask.squeeze()

        if mask.size == 0:
            return []

        mask = (mask > 0).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if not preserve_thin:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return []

        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) < 10:
            return []

        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.0, epsilon_ratio * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if approx is None or len(approx) < 3:
            return []

        pts = approx.reshape(-1, 2)

        cleaned = []
        last = None

        for x, y in pts:
            x = int(max(0, min(int(x), img_width - 1)))
            y = int(max(0, min(int(y), img_height - 1)))

            if last is not None and x == last[0] and y == last[1]:
                continue

            cleaned.append((x, y))
            last = (x, y)

        if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned.pop()

        if len(cleaned) < 3:
            return []

        if len(cleaned) > max_points:
            step = max(1, int(math.ceil(len(cleaned) / max_points)))
            cleaned = cleaned[::step]

        points = []

        for x, y in cleaned:
            nx = max(0.0, min(1.0, float(x) / float(img_width)))
            ny = max(0.0, min(1.0, float(y) / float(img_height)))
            points.append((nx, ny))

        return points

    except Exception as e:
        logger.warning(f"Failed converting SAM3 mask to polygon: {e}")
        return []


def _sam_device(self):
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sam_half_enabled(self):
    settings = getattr(self, "settings", {}) or {}
    return torch.cuda.is_available() and bool(
        getattr(self, "_sam_force_fp16", False) or
        getattr(self, "sam_use_fp16", settings.get("sam3Fp16", True))
    )


def _sam_imgsz(self, requested=None):
    settings = getattr(self, "settings", {}) or {}
    configured = getattr(self, "sam_imgsz", settings.get("sam3ImgSize", SAM3_DEFAULT_IMGSZ))
    configured = normalize_sam_imgsz(configured)

    if requested is None:
        return configured

    requested = normalize_sam_imgsz(requested)
    return min(requested, configured)


def _sam_predict_kwargs(self, imgsz=None, device=None):
    device = device or self._sam_device()
    kwargs = {
        "imgsz": self._sam_imgsz(imgsz),
        "verbose": False,
        "device": device,
    }
    if self._sam_half_enabled() and str(device).startswith("cuda"):
        kwargs["quantize"] = 16
    return kwargs


def _sam_model_path(self):
    return getattr(self, "sam_model_path", "Sam/sam3.pt")


def _get_sam_model(self):
    if self.ensure_sam_model_loaded():
        return getattr(self, "sam_model", None)
    return None


def _clean_mask(self, mask, kernel_size=5):
    if mask is None:
        return None
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return mask_u8.astype(bool)


def _normalize_sam_mask_to_image(self, mask, img_width, img_height):
    if mask is None or img_width <= 0 or img_height <= 0:
        return None

    try:
        mask = np.asarray(mask)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            return None

        mask = mask > 0
        if mask.shape[:2] != (img_height, img_width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (img_width, img_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if not mask.any():
            return None

        return mask
    except Exception as e:
        logging.warning("Failed normalizing SAM3 mask to image size: %s", e)
        return None


def _mask_bbox(self, mask):
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


@staticmethod
def _bbox_area_xyxy(bbox):
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@staticmethod
def _bbox_intersection_area_xyxy(first, second):
    if first is None or second is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in first]
    bx1, by1, bx2, by2 = [float(v) for v in second]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


@staticmethod
def _bbox_center_distance_xyxy(first, second):
    if first is None or second is None:
        return float("inf")
    ax1, ay1, ax2, ay2 = [float(v) for v in first]
    bx1, by1, bx2, by2 = [float(v) for v in second]
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def _keypoint_bbox_xyxy_from_box_obj(
    self,
    box_obj,
    img_width,
    img_height,
    pad_ratio=0.18,
    min_pad_px=8,
):
    keypoints = getattr(box_obj, "keypoints", None) or []
    points = []

    for keypoint in keypoints:
        if not isinstance(keypoint, (list, tuple)) or len(keypoint) < 3:
            continue
        try:
            x_norm = float(keypoint[0])
            y_norm = float(keypoint[1])
            visibility = int(float(keypoint[2]))
        except Exception:
            continue
        if visibility <= 0:
            continue
        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            continue
        points.append((x_norm * img_width, y_norm * img_height))

    if len(points) < 2:
        return None

    pts = np.asarray(points, dtype=np.float32)
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = max(float(min_pad_px), width * float(pad_ratio))
    pad_y = max(float(min_pad_px), height * float(pad_ratio) * 1.15)

    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(img_width - 1), x2 + pad_x)
    y2 = min(float(img_height - 1), y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _choose_keypoint_adjusted_bbox(self, sam_bbox, keypoint_bbox, img_width, img_height):
    if keypoint_bbox is None:
        return sam_bbox
    if sam_bbox is None:
        return keypoint_bbox

    sam_area = self._bbox_area_xyxy(sam_bbox)
    keypoint_area = self._bbox_area_xyxy(keypoint_bbox)
    if sam_area <= 0 or keypoint_area <= 0:
        return keypoint_bbox

    keypoint_diag = math.sqrt(max(1.0, keypoint_area))
    center_distance = self._bbox_center_distance_xyxy(sam_bbox, keypoint_bbox)
    keypoint_coverage = self._bbox_intersection_area_xyxy(sam_bbox, keypoint_bbox) / max(1.0, keypoint_area)

    # Under sight overlays SAM can snap to the foreground weapon instead of the person.
    # In that case, keep the pose row intact and rebuild the bbox from visible joints.
    if sam_area > keypoint_area * 2.75:
        return keypoint_bbox
    if center_distance > keypoint_diag * 0.85:
        return keypoint_bbox
    if keypoint_coverage < 0.55:
        return keypoint_bbox

    ux1 = max(0, min(int(round(min(sam_bbox[0], keypoint_bbox[0]))), img_width - 1))
    uy1 = max(0, min(int(round(min(sam_bbox[1], keypoint_bbox[1]))), img_height - 1))
    ux2 = max(0, min(int(round(max(sam_bbox[2], keypoint_bbox[2]))), img_width - 1))
    uy2 = max(0, min(int(round(max(sam_bbox[3], keypoint_bbox[3]))), img_height - 1))

    union_bbox = (ux1, uy1, ux2, uy2)
    if self._bbox_area_xyxy(union_bbox) > keypoint_area * 2.75:
        return keypoint_bbox
    return union_bbox


def polygon_to_mask(self, segmentation, img_width, img_height):
    if len(segmentation) < 6 or len(segmentation) % 2 != 0:
        logging.warning("Invalid segmentation length. Must be even and >= 6.")
        return np.zeros((img_height, img_width), dtype=bool)

    points = np.array(segmentation, dtype=np.float32).reshape(-1, 2)
    points[:, 0] *= img_width
    points[:, 1] *= img_height
    points = np.round(points).astype(np.int32)

    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def polygon_to_mask_shapely(self, segmentation, img_width, img_height):
    return self.polygon_to_mask(segmentation, img_width, img_height)


def obb_to_mask(self, obb, img_width, img_height):
    if len(obb) != 8:
        return np.zeros((img_height, img_width), dtype=bool)
    points = np.array(obb, dtype=np.float32).reshape(-1, 2)
    points[:, 0] *= img_width
    points[:, 1] *= img_height
    points = np.round(points).astype(np.int32)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def get_polygon_from_mask(self, mask, simplify=True, epsilon_factor=0.001):
    try:
        if mask is None or not isinstance(mask, np.ndarray):
            return []
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []

        contour = max(contours, key=cv2.contourArea)
        if simplify:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
        return contour.reshape(-1, 2).tolist()
    except Exception as e:
        logging.error("get_polygon_from_mask failed: %s", e)
        return []


def _contour_from_mask(self, mask):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _seg_line_from_mask(self, class_id, mask, img_width, img_height):
    contour = self._contour_from_mask(mask)
    if contour is None:
        return None
    seg_points = contour.reshape(-1, 2)
    # Four polygon points serialize to the same token count as a YOLO OBB.
    # Add one harmless collinear midpoint so the saved segmentation remains
    # unambiguous even when it is later opened outside Segmentation mode.
    if len(seg_points) == 4:
        midpoint = np.rint(
            (seg_points[0].astype(np.float32) + seg_points[1].astype(np.float32)) / 2.0
        ).astype(seg_points.dtype)
        seg_points = np.insert(seg_points, 1, midpoint, axis=0)
    return f"{class_id} " + " ".join(
        f"{x / img_width:.6f} {y / img_height:.6f}" for x, y in seg_points
    )


def _obb_line_from_mask(self, class_id, mask, img_width, img_height):

    contour = self._contour_from_mask(mask)
    if contour is None:
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    # enforce consistent clockwise ordering starting from top-left
    box = sorted(box, key=lambda p: (p[1], p[0]))  # sort by y then x
    top_two = sorted(box[:2], key=lambda p: p[0])
    bottom_two = sorted(box[2:], key=lambda p: p[0], reverse=True)

    ordered = np.array([
        top_two[0],
        top_two[1],
        bottom_two[0],
        bottom_two[1],
    ])

    ordered[:, 0] = np.clip(ordered[:, 0], 0, img_width)
    ordered[:, 1] = np.clip(ordered[:, 1], 0, img_height)

    flat = ordered.flatten()

    return f"{class_id} " + " ".join(
        f"{v / img_width:.6f}" if i % 2 == 0 else f"{v / img_height:.6f}"
        for i, v in enumerate(flat)
    )


def get_sam_mask(self, image, bbox, model_path=None, device=None, imgsz=1036):
    model = self._get_sam_model()
    if model is None:
        return None

    image = self.ensure_rgb(image)
    if image is None:
        return None

    x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    h, w = image.shape[:2]

    x_min = max(0, min(x_min, w - 1))
    y_min = max(0, min(y_min, h - 1))
    x_max = max(0, min(x_max, w - 1))
    y_max = max(0, min(y_max, h - 1))

    if x_max <= x_min or y_max <= y_min:
        return None

    results = None
    try:
        self._safe_sync_cuda()
        with torch.inference_mode():
            results = model.predict(
                source=image,
                bboxes=[[x_min, y_min, x_max, y_max]],
                **self._sam_predict_kwargs(imgsz=imgsz, device=device),
            )
    except torch.cuda.OutOfMemoryError as e:
        logging.critical("CUDA out of memory during SAM3 inference: %s", e, exc_info=True)
        self._safe_empty_cache()
        return None
    except torch.cuda.CudaError as e:
        logging.critical("CUDA crash during SAM3 inference: %s", e, exc_info=True)
        self._safe_empty_cache()
        return None
    except RuntimeError as e:
        message = str(e).lower()
        if "cuda" in message or "out of memory" in message:
            logging.critical("Runtime CUDA failure during SAM3 inference: %s", e, exc_info=True)
            self._safe_empty_cache()
            return None
        logging.error("SAM3 model.predict failed: %s", e, exc_info=True)
        return None
    except Exception as e:
        logging.error("SAM3 model.predict failed: %s", e, exc_info=True)
        return None

    try:
        if not results:
            return None
        result = results[0]
        if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
            return None

        best_mask = None
        best_score = -1

        for candidate in result.masks.data:
            if hasattr(candidate, "detach"):
                candidate = candidate.detach().float().cpu().numpy()

            candidate = self._normalize_sam_mask_to_image(candidate, w, h)
            if candidate is None:
                continue

            cleaned = self._clean_mask(candidate, kernel_size=5)
            if cleaned is None or not cleaned.any():
                continue

            area = int(cleaned.sum())
            if area > best_score:
                best_score = area
                best_mask = cleaned

        if best_mask is None or not best_mask.any():
            return None

        return best_mask
    except Exception as e:
        logging.error("Failed parsing SAM3 output: %s", e, exc_info=True)
        return None
    finally:
        try:
            del results
        except Exception:
            pass
        self._safe_empty_cache()


def generate_segmentation_mask(self, image, bbox):
    return self.get_sam_mask(image, bbox)


def preprocess_image_for_segmentation(self, image):
    if image is None:
        logger.warning("preprocess_image_for_segmentation called with None image")
        return None

    if not isinstance(image, np.ndarray):
        logger.warning("preprocess_image_for_segmentation requires a numpy ndarray image")
        return None

    if image.size == 0:
        logger.warning("preprocess_image_for_segmentation called with an empty image")
        return None

    try:
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            channels = image.shape[2]
            if channels == 1:
                gray = image[:, :, 0]
            elif channels == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                logger.warning(
                    "Unsupported channel count for segmentation preprocessing: %s",
                    channels
                )
                return None
        else:
            logger.warning(
                "Unsupported image dimensions for segmentation preprocessing: %s",
                image.shape
            )
            return None

        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresholded = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    except Exception as e:
        logger.error("Failed to preprocess image for segmentation: %s", e, exc_info=True)
        return None


def _bbox_to_mask(self, bbox, img_width, img_height):
    if bbox is None:
        return None
    x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    x_min = max(0, min(x_min, img_width - 1))
    y_min = max(0, min(y_min, img_height - 1))
    x_max = max(0, min(x_max, img_width))
    y_max = max(0, min(y_max, img_height))
    if x_max <= x_min or y_max <= y_min:
        return None
    mask = np.zeros((img_height, img_width), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def _process_label_to_mask(self, box_obj, image, img_width, img_height, use_sam=True):
    label_type = self._label_type(box_obj)
    original_bbox = self._label_seed_bbox(box_obj, img_width, img_height)

    if label_type == "segmentation":
        mask = self.polygon_to_mask(box_obj.segmentation, img_width, img_height)
    elif label_type == "obb":
        mask = self.obb_to_mask(box_obj.obb, img_width, img_height)
    else:
        if original_bbox is None:
            return None, None, label_type
        mask = self.get_sam_mask(image, original_bbox) if use_sam else self._bbox_to_mask(
            original_bbox,
            img_width,
            img_height,
        )

    if mask is None or not mask.any():
        return None, original_bbox, label_type
    return mask, original_bbox, label_type


def _update_box_obj_from_mask(self, box_obj, mask, label_type, img_width, img_height):
    polygon = self.get_polygon_from_mask(mask)
    if not polygon:
        return None

    bbox = self._mask_bbox(mask)
    if bbox is None:
        return None

    if label_type == "segmentation":
        box_obj.segmentation = self._normalize_polygon(polygon, img_width, img_height)
    elif label_type == "obb":
        contour = self._contour_from_mask(mask)
        if contour is not None:
            rect = cv2.minAreaRect(contour)
            pts = cv2.boxPoints(rect)
            pts[:, 0] = np.clip(pts[:, 0], 0, img_width)
            pts[:, 1] = np.clip(pts[:, 1], 0, img_height)
            box_obj.obb = [coord for pt in pts.tolist() for coord in (pt[0] / img_width, pt[1] / img_height)]
    elif label_type == "keypoints":
        keypoint_bbox = self._keypoint_bbox_xyxy_from_box_obj(box_obj, img_width, img_height)
        bbox = self._choose_keypoint_adjusted_bbox(bbox, keypoint_bbox, img_width, img_height)
        box_obj.x_center, box_obj.y_center, box_obj.width, box_obj.height = self._xyxy_to_yolo_xywh(
            bbox, img_width, img_height
        )
    else:
        box_obj.x_center, box_obj.y_center, box_obj.width, box_obj.height = self._xyxy_to_yolo_xywh(
            bbox, img_width, img_height
        )

    return bbox


def _segment_to_box(self, polygon_points, img_width, img_height):
    if polygon_points is None or len(polygon_points) < 3:
        return None
    pts = np.asarray(polygon_points, dtype=np.float32)
    try:
        if pts.ndim != 2 or pts.shape[1] < 2:
            return None
        x = np.clip(pts[:, 0], 0, img_width)
        y = np.clip(pts[:, 1], 0, img_height)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return None
        x1, y1, x2, y2 = x.min(), y.min(), x.max(), y.max()
        if x2 <= x1 or y2 <= y1:
            return None
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None
