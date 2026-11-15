import logging
import inspect
import hashlib
import tempfile
import time
import cv2
import numpy as np

from sahi.predict import get_sliced_prediction
from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
)
from sahi.prediction import ObjectPrediction
from sahi.slicing import slice_image, get_slice_bboxes
from sahi.utils.cv import read_image
import os
from PIL import Image, UnidentifiedImageError
from sahi import AutoDetectionModel
from prediction_size_filter import prediction_size_allowed_xyxy, polygon_touches_roi_rect
from darkfusion_inference import load_inference_model

logger = logging.getLogger(__name__)

class SahiPredictWrapper:
    def __init__(
        self,
        model_type,
        model_path,
        confidence_threshold,
        device,
        fp16=False,
        inference_backend="auto",
        onnx_provider="auto",
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=False,
        perform_standard_pred=True,
        show_preview=False,
        min_size_px=0.0,
        max_percent=1.0,
        ignore_teammates=False,
        teammate_threshold=0.90,
        roi_enabled=False,
        roi_bounds=None,
        roi_reference_size=None,
        onnx_runtime_options=None,
        model=None,
        prediction_kwargs=None,
        cancelled=None,
    ):
        self.fp16 = bool(fp16)
        self.inference_backend = str(inference_backend or "auto").strip().lower()
        self.onnx_provider = str(onnx_provider or "auto").strip().lower()
        self.onnx_runtime_options = None if onnx_runtime_options is None else dict(onnx_runtime_options)
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self._uses_darkfusion_onnx = False
        self.onnx_runtime_model = None
        self.prediction_kwargs = dict(prediction_kwargs or {})
        self.cancelled = cancelled
        if model is not None:
            from types import SimpleNamespace
            self._uses_darkfusion_onnx = getattr(model, '_darkfusion_inference_backend', '') == 'onnxruntime'
            self.onnx_runtime_model = model if self._uses_darkfusion_onnx else None
            self.detection_model = SimpleNamespace(model=model, device=device)
        else:
            self.detection_model = self._create_detection_model(
                model_type=model_type,
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=device,
            )
        native_model = getattr(self.detection_model, "model", None)
        self.task = str(getattr(self.onnx_runtime_model if self._uses_darkfusion_onnx else native_model, "task", "detect") or "detect")
        if self.task not in {"detect", "segment", "pose", "obb"}:
            raise ValueError("SAHI auto-label requires an object-annotation model; classification/ReID models cannot write object labels.")
        # SAHI's generic object adapter can discard pose/OBB data. Keep its
        # slicing/merging, but read the native Ultralytics results for these.
        self.geometry_model = native_model if model is not None or self.task in {"pose", "obb"} else None
        self.postprocess_type = postprocess_type
        self.postprocess_match_metric = postprocess_match_metric
        self.postprocess_match_threshold = float(postprocess_match_threshold)
        self.postprocess_class_agnostic = bool(postprocess_class_agnostic)
        self.perform_standard_pred = bool(perform_standard_pred)
        self.show_preview = bool(show_preview)
        self.min_size_px = max(0.0, float(min_size_px or 0.0))
        self.max_percent = max(0.0, min(1.0, float(max_percent or 1.0)))
        if self.max_percent <= 0.0:
            self.max_percent = 1.0
        self.size_filtered_count = 0
        self.ignore_teammates = bool(ignore_teammates)
        self.teammate_threshold = max(0.50, min(0.999, float(teammate_threshold or 0.90)))
        self.teammate_filtered_count = 0
        self._teammate_classifier = None
        self.roi_enabled = bool(roi_enabled)
        if (isinstance(roi_bounds, (list, tuple)) and len(roi_bounds) == 4 and
                all(isinstance(value, (int, float)) for value in roi_bounds)):
            self.roi_bounds = [tuple(roi_bounds)]
        else:
            self.roi_bounds = [
                tuple(bounds) for bounds in (roi_bounds or [])
                if isinstance(bounds, (list, tuple)) and len(bounds) == 4
            ]
        self.roi_reference_size = (
            tuple(roi_reference_size)
            if isinstance(roi_reference_size, (list, tuple)) and len(roi_reference_size) == 2
            else None
        )

    def _rois_for_image(self, image_width, image_height):
        """Return all active ROIs scaled into one image's pixel space."""
        if not self.roi_enabled or not self.roi_bounds:
            return []
        scaled = []
        for source_bounds in self.roi_bounds:
            try:
                x, y, width, height = (float(value) for value in source_bounds)
                ref_width, ref_height = self.roi_reference_size or (image_width, image_height)
                if ref_width <= 0 or ref_height <= 0:
                    continue
                x *= image_width / float(ref_width)
                width *= image_width / float(ref_width)
                y *= image_height / float(ref_height)
                height *= image_height / float(ref_height)
                x = max(0, min(int(round(x)), int(image_width) - 1))
                y = max(0, min(int(round(y)), int(image_height) - 1))
                width = max(1, min(int(round(width)), int(image_width) - x))
                height = max(1, min(int(round(height)), int(image_height) - y))
                scaled.append((x, y, width, height))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
        return scaled

    def _roi_for_image(self, image_width, image_height):
        """Compatibility helper returning the first scaled ROI, if any."""
        scaled = self._rois_for_image(image_width, image_height)
        return scaled[0] if scaled else None

    @staticmethod
    def _roi_prediction_allowed(bbox, selected_rois, inference_roi, image_width, image_height):
        if not selected_rois:
            return True
        if (isinstance(selected_rois, (list, tuple)) and len(selected_rois) == 4 and
                all(isinstance(value, (int, float)) for value in selected_rois)):
            selected_rois = [selected_rois]
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
            crop_x, crop_y = (0, 0) if inference_roi is None else inference_roi[:2]
            left = min(x1, x2) + crop_x
            top = min(y1, y2) + crop_y
            right = max(x1, x2) + crop_x
            bottom = max(y1, y2) + crop_y
            return any(
                not (
                    right < select_x or left > select_x + select_width or
                    bottom < select_y or top > select_y + select_height
                )
                for select_x, select_y, select_width, select_height in selected_rois
            )
        except (TypeError, ValueError):
            return False

    def _create_detection_model(self, model_type, model_path, confidence_threshold, device):
        suffix = os.path.splitext(str(model_path or ""))[1].lower()
        use_darkfusion_onnx = (
            suffix == ".onnx"
            and self.inference_backend in {"auto", "onnx", "onnxruntime", "ort"}
        )

        if use_darkfusion_onnx:
            runtime_model = load_inference_model(
                model_path,
                backend=self.inference_backend,
                provider=self.onnx_provider,
                fp16=self.fp16,
                runtime_options=self.onnx_runtime_options,
            )
            if runtime_model.task not in {"detect", "segment", "pose", "obb"}:
                raise ValueError("SAHI auto-label requires a detection, segmentation, pose or OBB model; this model does not predict object annotations.")
            logger.info(
                "SAHI is using DarkFusion ONNX Runtime (%s): %s",
                runtime_model.provider,
                model_path,
            )
            # SAHI's Ultralytics adapter expects a torch-like ``boxes.data``
            # tensor. DarkFusion deliberately returns ArrayView objects for
            # provider-safe ONNX Runtime inference, so use SAHI only for
            # slicing/merging and consume its ONNX results directly.
            self._uses_darkfusion_onnx = True
            self.onnx_runtime_model = runtime_model
            return None

        detection_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )

        if suffix == ".engine":
            from darkfusion_tensorrt import enable_engine_cuda_graph
            enable_engine_cuda_graph(getattr(detection_model, "model", None), model_path)

        # SAHI does not expose Ultralytics' half argument. Setting the model
        # override before its first prediction gives CUDA .pt models the same
        # shared FP16 choice as DarkFusion's other auto-label workflows.
        if self.fp16 and suffix == ".pt" and str(device).lower().startswith("cuda"):
            runtime_model = getattr(detection_model, "model", None)
            overrides = getattr(runtime_model, "overrides", None)
            if isinstance(overrides, dict):
                overrides["half"] = True
                logger.info("SAHI FP16 enabled for CUDA PyTorch model: %s", model_path)

        return detection_model

    @staticmethod
    def _xyxy_iou(first, second):
        left = max(float(first[0]), float(second[0]))
        top = max(float(first[1]), float(second[1]))
        right = min(float(first[2]), float(second[2]))
        bottom = min(float(first[3]), float(second[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
        second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
        return intersection / max(1e-9, first_area + second_area - intersection)

    @staticmethod
    def _deduplicate_tile_predictions(predictions):
        # Overlapping slices/full-frame passes can produce identical geometry.
        # Collapse those before SAHI's spatial-index merge, retaining the best
        # confidence and keeping distinct masks/skeletons/oriented boxes apart.
        unique = {}
        for prediction in predictions:
            key = (int(prediction['class_id']), tuple(prediction['bbox']),
                   repr(prediction.get('segmentation')), repr(prediction.get('obb')),
                   repr(prediction.get('keypoints')))
            previous = unique.get(key)
            if previous is None or prediction['confidence'] > previous['confidence']:
                unique[key] = prediction
        return list(unique.values())

    def _merge_onnx_tile_predictions(self, predictions):
        """Apply the selected native SAHI postprocessor to ONNX tile results."""
        predictions = self._deduplicate_tile_predictions(predictions)
        if not predictions:
            return []
        if len(predictions) == 1:
            # SAHI's tensor adapter can squeeze a single prediction to a
            # scalar during class-wise NMS. There is nothing to suppress.
            return predictions
        if all(prediction.get("obb") is not None for prediction in predictions):
            from darkfusion_onnx_runtime import numpy_rotated_nms
            rotated = []
            for prediction in predictions:
                if prediction.get("xywhr") is not None:
                    rotated.append(prediction["xywhr"])
                else:
                    (cx, cy), (width, height), angle = cv2.minAreaRect(np.asarray(prediction["obb"], dtype=np.float32))
                    rotated.append([cx, cy, width, height, np.deg2rad(angle)])
            keep = numpy_rotated_nms(rotated, [prediction["confidence"] for prediction in predictions],
                [prediction["class_id"] for prediction in predictions], self.postprocess_match_threshold,
                300, agnostic=self.postprocess_class_agnostic)
            return [predictions[index] for index in keep]
        processors = {
            "GREEDYNMM": GreedyNMMPostprocess,
            "NMM": NMMPostprocess,
            "NMS": NMSPostprocess,
            "LSNMS": LSNMSPostprocess,
        }
        processor_class = processors.get(str(self.postprocess_type or "GREEDYNMM").upper(), GreedyNMMPostprocess)
        # Merging axis-aligned boxes cannot reconstruct an oriented box or a
        # skeleton. For these tasks keep the winning original geometry intact.
        if any(prediction.get("obb") is not None or prediction.get("keypoints") is not None for prediction in predictions):
            processor_class = NMSPostprocess
        processor = processor_class(
            match_threshold=max(0.05, min(0.95, float(self.postprocess_match_threshold))),
            match_metric=str(self.postprocess_match_metric or "IOS").upper(),
            class_agnostic=bool(self.postprocess_class_agnostic),
        )
        object_predictions = [
            ObjectPrediction(
                bbox=list(prediction["bbox"]),
                score=float(prediction["confidence"]),
                category_id=int(prediction["class_id"]),
                category_name=str(prediction["class_id"]),
                segmentation=prediction.get("segmentation") or None,
                full_shape=prediction.get("full_shape"),
            )
            for prediction in predictions
        ]
        for item, prediction in zip(object_predictions, predictions):
            item._darkfusion_geometry = {key: prediction[key] for key in ("obb", "keypoints") if key in prediction}
        merged = [
            {
                "bbox": tuple(float(value) for value in prediction.bbox.to_voc_bbox()),
                "confidence": float(prediction.score.value),
                "class_id": int(prediction.category.id),
                "segmentation": prediction.mask.segmentation if prediction.mask else None,
                **getattr(prediction, "_darkfusion_geometry", {}),
            }
            for prediction in processor(object_predictions)
        ]
        # Some SAHI versions retain both the winner and an identical merged
        # box when equal-score candidates visit the spatial index in a
        # different order. Do not send that duplicate to display/tracking.
        return self._deduplicate_tile_predictions(merged)

    def _decode_tile_predictions(self, result, offset_x, offset_y, image_shape):
        def array(value):
            if value is None:
                return np.empty((0, 6))
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            return np.asarray(value.numpy() if hasattr(value, "numpy") else value)
        predictions = []
        boxes = getattr(result, "boxes", None)
        rows = array(getattr(boxes, "data", None))
        obb = getattr(result, "obb", None)
        if obb is not None:
            if getattr(getattr(obb, 'data', None), 'is_cuda', False) and hasattr(obb, 'cpu'):
                obb = obb.cpu()
            rows = np.column_stack((array(obb.xyxy), array(obb.conf), array(obb.cls)))
            obb_corners, obb_rotated = array(obb.xyxyxyxy), array(obb.xywhr)
        masks = getattr(result, "masks", None)
        polygons = masks.xy if masks is not None else []
        keypoints = getattr(result, "keypoints", None)
        pose_points = array(keypoints.data) if keypoints is not None else []
        for index, row in enumerate(rows):
            if len(row) < 6:
                continue
            x1, y1, x2, y2, confidence, class_id = (float(value) for value in row[:6])
            if x2 <= x1 or y2 <= y1:
                continue
            prediction = {
                "bbox": (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                "confidence": confidence,
                "class_id": int(class_id),
            }
            if obb is not None:
                points = np.array(obb_corners[index], dtype=np.float32, copy=True)
                points += [offset_x, offset_y]
                prediction["obb"] = points.tolist()
                rotated = np.array(obb_rotated[index], dtype=np.float32, copy=True)
                rotated[:2] += [offset_x, offset_y]
                prediction["xywhr"] = rotated.tolist()
            elif index < len(polygons) and len(polygons[index]) >= 3:
                points = np.array(polygons[index], dtype=np.float32, copy=True)
                points += [offset_x, offset_y]
                prediction["segmentation"] = [points.reshape(-1).tolist()]
                prediction["full_shape"] = list(image_shape[:2])
            if index < len(pose_points):
                points = np.array(pose_points[index], dtype=np.float32, copy=True)
                visible = points[:, 2] > .01 if points.shape[1] > 2 else np.any(points[:, :2] != 0, axis=1)
                points[visible, :2] += [offset_x, offset_y]
                points[~visible, :2] = 0
                prediction["keypoints"] = points.tolist()
            predictions.append(prediction)
        return predictions

    def _darkfusion_onnx_sliced_predictions(
        self, image_rgb, slice_height, slice_width, overlap_height_ratio,
        overlap_width_ratio, *, inference_inputs=None, input_is_bgr=False,
        batch_size=1, minimum_batch=1,
    ):
        model = self._direct_model()
        if model is None:
            raise RuntimeError("The selected SAHI inference model was not initialized.")
        started = time.perf_counter()
        if inference_inputs is None:
            slices = slice_image(
                image=image_rgb, slice_height=max(1, int(slice_height)),
                slice_width=max(1, int(slice_width)),
                overlap_height_ratio=float(overlap_height_ratio),
                overlap_width_ratio=float(overlap_width_ratio),
                auto_slice_resolution=False, verbose=False,
            )
            inference_inputs = [(tile.image, tuple(int(value) for value in tile.starting_pixel[:2]))
                                for tile in slices.sliced_image_list]
        else:
            inference_inputs = list(inference_inputs)
        if self.perform_standard_pred and not (
                getattr(self, 'prediction_kwargs', None) and len(inference_inputs) == 1
                and inference_inputs[0][0].shape == image_rgb.shape):
            inference_inputs.append((image_rgb, (0, 0)))
        prepared_ms = (time.perf_counter() - started) * 1000
        native_kwargs = {} if self._uses_darkfusion_onnx else {'device': self.detection_model.device, 'verbose': False}
        native_kwargs.update(getattr(self, 'prediction_kwargs', {}))
        native_kwargs.setdefault('conf', self.confidence_threshold)
        native_kwargs.setdefault('iou', self.postprocess_match_threshold)
        native_kwargs.setdefault('max_det', 300)
        predictions, calls, inferred_ms = [], 0, 0.
        batch_size = max(1, int(batch_size))
        index = 0
        while index < len(inference_inputs):
            if getattr(self, 'cancelled', None) is not None and self.cancelled():
                raise RuntimeError('SAHI inference stopped.')
            group = [inference_inputs[index]]
            index += 1
            # Keep the full-image pass separate from differently shaped tiles,
            # retaining native letterbox behavior and fixed-engine contracts.
            while (index < len(inference_inputs) and len(group) < batch_size
                   and inference_inputs[index][0].shape == group[0][0].shape):
                group.append(inference_inputs[index])
                index += 1
            inputs = [image if input_is_bgr else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                      for image, _offset in group]
            if len(inputs) < minimum_batch:
                inputs.extend([inputs[-1]] * (minimum_batch - len(inputs)))
            call_kwargs = dict(native_kwargs)
            if len(inputs) > 1:
                call_kwargs['batch'] = len(inputs)
            before_inference = time.perf_counter()
            result_list = list(model.predict(inputs if len(inputs) > 1 else inputs[0], **call_kwargs) or [])
            if len(result_list) != len(inputs):
                raise RuntimeError(f'SAHI model returned {len(result_list)} results for {len(inputs)} tile inputs.')
            if getattr(self, 'cancelled', None) is not None and self.cancelled():
                raise RuntimeError('SAHI inference stopped.')
            for result, (_image, (offset_x, offset_y)) in zip(result_list, group):
                predictions.extend(self._decode_tile_predictions(result, offset_x, offset_y, image_rgb.shape))
            inferred_ms += (time.perf_counter() - before_inference) * 1000
            calls += 1
        before_merge = time.perf_counter()
        merged = self._merge_onnx_tile_predictions(predictions)
        self.last_sliced_profile = dict(sahi_tiles=len(inference_inputs), sahi_calls=calls,
            sahi_batch=batch_size, sahi_prepare_ms=prepared_ms, sahi_model_ms=inferred_ms,
            sahi_merge_ms=(time.perf_counter()-before_merge)*1000)
        return merged

    def _direct_model(self):
        return self.onnx_runtime_model if self._uses_darkfusion_onnx else getattr(self, "geometry_model", None)

    def _onnx_class_name(self, class_id):
        names = getattr(self._direct_model(), "names", {}) or {}
        if isinstance(names, dict):
            return str(names.get(class_id, names.get(str(class_id), str(class_id)))).strip()
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id]).strip()
        return str(class_id)

    def _onnx_teammate_rejected_indices(self, image_rgb, predictions, class_names):
        """Apply the normal teammate filter to DarkFusion ONNX tile detections."""
        if not self.ignore_teammates or not predictions:
            return set()
        try:
            from darkfusion_teammate_review import TeammateMarkerClassifier

            if self._teammate_classifier is None:
                provider_names = " ".join(
                    str(provider) for provider in getattr(self._direct_model(), "providers", [])
                ).lower()
                self._teammate_classifier = TeammateMarkerClassifier(
                    "cuda" if "cuda" in provider_names or str(getattr(getattr(self, "detection_model", None), "device", "")).startswith("cuda") else "cpu"
                )
            image = Image.fromarray(image_rgb).convert("RGB")
            image_h, image_w = image_rgb.shape[:2]
            bounds = []
            indices = []
            for index, prediction in enumerate(predictions):
                class_id = int(prediction["class_id"])
                category_name = self._onnx_class_name(class_id).lower()
                if not any(token in category_name for token in ("person", "player", "enemy", "character")):
                    continue
                x1, y1, x2, y2 = (float(value) for value in prediction["bbox"])
                bounds.append([
                    max(0.0, min(1.0, x1 / max(1, image_w))),
                    max(0.0, min(1.0, y1 / max(1, image_h))),
                    max(0.0, min(1.0, x2 / max(1, image_w))),
                    max(0.0, min(1.0, y2 / max(1, image_h))),
                ])
                indices.append(index)
            if not bounds:
                return set()
            local_rejected = self._teammate_classifier.friendly_indices(
                [image], [bounds], threshold=self.teammate_threshold, batch_size=96, ocr=True,
            )[0]
            rejected = {
                indices[local_index]
                for local_index in local_rejected
                if 0 <= int(local_index) < len(indices)
            }
            if self._teammate_classifier.last_ocr_error:
                logger.warning(
                    "SAHI teammate OCR unavailable; visual marker filtering remains active: %s",
                    self._teammate_classifier.last_ocr_error,
                )
            self.teammate_filtered_count += len(rejected)
            return rejected
        except Exception as error:
            logger.warning("SAHI teammate filter unavailable; keeping predictions: %s", error)
            return set()

    def _teammate_rejected_indices(self, image_rgb, predictions):
        if not self.ignore_teammates or not predictions:
            return set()
        try:
            from darkfusion_teammate_review import TeammateMarkerClassifier

            if self._teammate_classifier is None:
                self._teammate_classifier = TeammateMarkerClassifier(
                    "cuda" if str(getattr(self.detection_model, "device", "")).lower() != "cpu" else "cpu"
                )
            image = Image.fromarray(image_rgb).convert("RGB")
            image_h, image_w = image_rgb.shape[:2]
            bounds = []
            indices = []
            for index, obj in enumerate(predictions):
                category_name = str(obj.category.name).strip().lower()
                if category_name and not any(
                    token in category_name for token in ("person", "player", "enemy", "character")
                ):
                    continue
                x1, y1, x2, y2 = [float(value) for value in obj.bbox.to_voc_bbox()]
                bounds.append([
                    max(0.0, min(1.0, x1 / max(1, image_w))),
                    max(0.0, min(1.0, y1 / max(1, image_h))),
                    max(0.0, min(1.0, x2 / max(1, image_w))),
                    max(0.0, min(1.0, y2 / max(1, image_h))),
                ])
                indices.append(index)
            local_rejected = self._teammate_classifier.friendly_indices(
                [image],
                [bounds],
                threshold=self.teammate_threshold,
                batch_size=96,
                ocr=True,
            )[0]
            rejected = {
                indices[local_index]
                for local_index in local_rejected
                if 0 <= int(local_index) < len(indices)
            }
            if self._teammate_classifier.last_ocr_error:
                logger.warning(
                    "SAHI teammate OCR unavailable; visual marker filtering remains active: %s",
                    self._teammate_classifier.last_ocr_error,
                )
            self.teammate_filtered_count += len(rejected)
            return rejected
        except Exception as error:
            logger.warning("SAHI teammate filter unavailable; keeping predictions: %s", error)
            return set()

    @staticmethod
    def get_unique_color(class_name):
        digest = hashlib.sha1((class_name or "class").encode("utf-8")).digest()
        seed = int.from_bytes(digest[:3], "big")
        return seed % 255, (seed >> 8) % 255, (seed >> 16) % 255

    def read_class_names(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as file:
            class_names = [line.strip() for line in file.readlines()]
        return class_names

    def _supported_sliced_kwargs(self, kwargs):
        signature = inspect.signature(get_sliced_prediction)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in signature.parameters}

    def _write_yolo_lines(self, txt_file_path, new_lines, overwrite=True):
        if not overwrite and os.path.exists(txt_file_path):
            with open(txt_file_path, "r", encoding="utf-8") as f:
                existing_lines = [line.strip() for line in f if line.strip()]
        else:
            existing_lines = []

        merged_lines = list(existing_lines)
        seen = set(existing_lines)
        for line in new_lines:
            if line not in seen:
                merged_lines.append(line)
                seen.add(line)

        output_directory = os.path.dirname(os.path.abspath(txt_file_path))
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".darkfusion-sahi-",
            suffix=".tmp",
            dir=output_directory,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as f:
                descriptor = None
                for line in merged_lines:
                    f.write(line + "\n")
            os.replace(temporary_path, txt_file_path)
            temporary_path = None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_path is not None:
                try:
                    os.remove(temporary_path)
                except FileNotFoundError:
                    pass

    def process_image(
        self,
        image_path,
        slice_height,
        slice_width,
        overlap_height_ratio,
        overlap_width_ratio,
        class_names,
        desired_classes=None,
        overwrite=True,
    ):
        # A damaged image is safe to skip. Once the image is readable, failures
        # must reach SahiWorker so the UI does not report a broken inference run
        # as a successful run that happened to produce zero labels.
        try:
            image_rgb = read_image(image_path)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            logger.warning("Cannot open image %s: %s", image_path, error)
            return 0
        except Exception as error:
            logger.warning("Cannot read image %s: %s", image_path, error)
            return 0

        if image_rgb is None or not hasattr(image_rgb, "shape") or len(image_rgb.shape) < 2:
            logger.warning("Cannot open image %s: image decoder returned no pixels", image_path)
            return 0

        class_names = [name.strip() for name in (class_names or []) if name.strip()]
        desired_classes = [name.strip() for name in (desired_classes or class_names) if name.strip()]
        class_to_id = {name.lower(): idx for idx, name in enumerate(class_names)}
        desired_lookup = {name.lower() for name in desired_classes}
        excluded_names = [
            name for name in class_names
            if desired_lookup and name.lower() not in desired_lookup
        ]

        image_h, image_w = image_rgb.shape[:2]
        selected_rois = self._rois_for_image(image_w, image_h)
        inference_roi = (0, 0, image_w, image_h)
        roi_x, roi_y = 0, 0
        inference_image = image_rgb
        if inference_image.size == 0:
            logger.warning("ROI is empty for %s; skipping image", image_path)
            return 0

        direct_results = self._uses_darkfusion_onnx or getattr(self, "geometry_model", None) is not None
        if direct_results:
            try:
                predictions = self._darkfusion_onnx_sliced_predictions(
                    inference_image,
                    slice_height,
                    slice_width,
                    overlap_height_ratio,
                    overlap_width_ratio,
                )
            except Exception as error:
                raise RuntimeError(f"DarkFusion SAHI inference failed for {image_path}: {error}") from error
            teammate_rejected = self._onnx_teammate_rejected_indices(
                inference_image, predictions, class_names
            )
        else:
            sliced_kwargs = self._supported_sliced_kwargs({
                "image": inference_image,
                "detection_model": self.detection_model,
                "slice_height": slice_height,
                "slice_width": slice_width,
                "overlap_height_ratio": overlap_height_ratio,
                "overlap_width_ratio": overlap_width_ratio,
                "perform_standard_pred": self.perform_standard_pred,
                "postprocess_type": self.postprocess_type,
                "postprocess_match_metric": self.postprocess_match_metric,
                "postprocess_class_agnostic": self.postprocess_class_agnostic,
                "exclude_classes_by_name": excluded_names or None,
                "verbose": 0,
            })
            try:
                result = get_sliced_prediction(**sliced_kwargs)
            except Exception as error:
                raise RuntimeError(f"SAHI inference failed for {image_path}: {error}") from error
            predictions = list(result.object_prediction_list)
            teammate_rejected = self._teammate_rejected_indices(inference_image, predictions)

        txt_file_path = os.path.splitext(image_path)[0] + '.txt'
        yolo_lines = []
        skipped_size_count = 0
        for prediction_index, obj in enumerate(predictions):
            if prediction_index in teammate_rejected:
                continue
            if direct_results:
                class_id = int(obj["class_id"])
                category_name = self._onnx_class_name(class_id)
                bbox = obj["bbox"]
                polygons = obj.get("segmentation") or []
                oriented_points = obj.get("obb")
                keypoints = obj.get("keypoints")
            else:
                category_name = str(obj.category.name).strip()
                class_id = int(obj.category.id)
                bbox = obj.bbox.to_voc_bbox()
                polygons = obj.mask.segmentation if getattr(obj, "mask", None) else []
                oriented_points = None
                keypoints = None
            category_key = category_name.lower()

            if desired_lookup and category_key not in desired_lookup:
                continue

            if not self._roi_prediction_allowed(
                bbox, selected_rois, inference_roi, image_w, image_h
            ):
                continue
            geometry = [np.asarray(oriented_points).reshape(-1, 2)] if oriented_points is not None else [
                np.asarray(polygon).reshape(-1, 2) for polygon in polygons if len(polygon) >= 6]
            if selected_rois and geometry and not any(
                    polygon_touches_roi_rect(points + [roi_x, roi_y], roi)
                    for points in geometry for roi in selected_rois):
                continue

            raw_x1 = float(bbox[0]) + roi_x
            raw_y1 = float(bbox[1]) + roi_y
            raw_x2 = float(bbox[2]) + roi_x
            raw_y2 = float(bbox[3]) + roi_y
            bbox = (
                max(0.0, min(float(image_w), min(raw_x1, raw_x2))),
                max(0.0, min(float(image_h), min(raw_y1, raw_y2))),
                max(0.0, min(float(image_w), max(raw_x1, raw_x2))),
                max(0.0, min(float(image_h), max(raw_y1, raw_y2))),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                skipped_size_count += 1
                continue

            if not prediction_size_allowed_xyxy(
                bbox,
                image_w,
                image_h,
                min_size_px=self.min_size_px,
                max_percent=self.max_percent,
            ):
                skipped_size_count += 1
                continue

            xc = (bbox[0] + bbox[2]) / 2 / image_w
            yc = (bbox[1] + bbox[3]) / 2 / image_h
            w = (bbox[2] - bbox[0]) / image_w
            h = (bbox[3] - bbox[1]) / image_h

            # The dropdown owns dataset IDs. Never assign an unknown model
            # class to an unrelated project class by falling back to its ID.
            class_id = class_to_id.get(category_key)
            if class_id is None:
                continue
            if oriented_points is not None or polygons:
                if oriented_points is not None:
                    points = np.asarray(oriented_points, dtype=np.float32).reshape(-1, 2)
                else:
                    contours = [np.asarray(polygon, dtype=np.float32).reshape(-1, 2) for polygon in polygons if len(polygon) >= 6]
                    if not contours:
                        continue
                    points = max(contours, key=cv2.contourArea)
                points = points.copy()
                points[:, 0] = np.clip((points[:, 0] + roi_x) / image_w, 0., 1.)
                points[:, 1] = np.clip((points[:, 1] + roi_y) / image_h, 0., 1.)
                yolo_lines.append(str(class_id) + " " + " ".join(f"{value:.6f}" for value in points.reshape(-1)))
            else:
                line = f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                for point in keypoints or []:
                    visible = (len(point) < 3 or point[2] > .01) and any(value != 0 for value in point[:2])
                    px = np.clip((point[0] + roi_x) / image_w, 0., 1.) if visible else 0.
                    py = np.clip((point[1] + roi_y) / image_h, 0., 1.) if visible else 0.
                    line += f" {px:.6f} {py:.6f} {2 if visible else 0}"
                yolo_lines.append(line)

        self._write_yolo_lines(txt_file_path, yolo_lines, overwrite=overwrite)
        self.size_filtered_count += skipped_size_count

        return len(yolo_lines)

    def process_folder(
        self,
        folder_path,
        class_names_file,
        slice_height,
        slice_width,
        overlap_height_ratio,
        overlap_width_ratio,
        desired_classes=None,
        overwrite=True,
        progress_callback=None,
        class_names=None,
    ):
        class_names = [
            str(name).strip()
            for name in (class_names or [])
            if str(name).strip()
        ]
        if not class_names and class_names_file:
            class_names = self.read_class_names(class_names_file)
        desired_classes = desired_classes or class_names
        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if not os.path.isdir(folder_path):
            logger.info(f"Image directory not found: {folder_path}")
            return {"images": 0, "images_with_detections": 0, "labels": 0}

        image_files = [
            image_file for image_file in sorted(os.listdir(folder_path))
            if any(image_file.lower().endswith(ext) for ext in allowed_extensions)
        ]

        total_labels = 0
        images_with_detections = 0
        self.size_filtered_count = 0
        self.teammate_filtered_count = 0

        last_image_path = ""
        last_detected_image_path = ""

        for index, image_file in enumerate(image_files, start=1):
            image_path = os.path.join(folder_path, image_file)
            labels_written = self.process_image(
                image_path=image_path,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                class_names=class_names,
                desired_classes=desired_classes,
                overwrite=overwrite,
            )

            total_labels += labels_written
            last_image_path = image_path
            if labels_written:
                images_with_detections += 1
                last_detected_image_path = image_path

            if progress_callback:
                progress_callback(index, len(image_files), image_path, labels_written)

        return {
            "images": len(image_files),
            "images_with_detections": images_with_detections,
            "labels": total_labels,
            "skipped_size": int(self.size_filtered_count),
            "skipped_teammates": int(self.teammate_filtered_count),
            "last_image_path": last_image_path,
            "last_detected_image_path": last_detected_image_path,
        }


class SahiVideoModel:
    """Slice live pixels with the existing model; return shared result geometry."""

    def __init__(self, model, options, prediction_kwargs, cancelled=None):
        self.model = model
        self.names = model.names
        self.task = str(getattr(model, 'task', 'detect'))
        self.options = dict(options or {})
        self.prediction_kwargs = dict(prediction_kwargs or {})
        self.device = self.prediction_kwargs.get('device')
        if self.device is None or self.device == '':
            self.device = str(getattr(model, 'device', None) or 'cpu')
        self.tracker = None
        self._slice_plan_key = None
        self._slice_boxes = ()
        self._batch_limit = None
        self.last_profile = {}
        self._darkfusion_inference_backend = getattr(model, '_darkfusion_inference_backend', 'ultralytics')
        from ultralytics.utils import DEFAULT_CFG_DICT
        self._native_quantize_keyword = self._darkfusion_inference_backend != 'onnxruntime' and 'quantize' in DEFAULT_CFG_DICT
        self.predictor = SahiPredictWrapper(
            'ultralytics', '', self.prediction_kwargs.get('conf', .25), self.device,
            model=model, prediction_kwargs=self.prediction_kwargs, cancelled=cancelled,
            postprocess_type=self.options.get('postprocess_type', 'GREEDYNMM'),
            postprocess_match_metric=self.options.get('postprocess_match_metric', 'IOS'),
            postprocess_match_threshold=self.options.get('postprocess_match_threshold', .5),
            postprocess_class_agnostic=self.options.get('postprocess_class_agnostic', False),
            perform_standard_pred=self.options.get('perform_standard_pred', True),
        )

    def _tile_inputs(self, frame):
        height, width = frame.shape[:2]
        key = (height, width, int(self.options.get('slice_height', 640)),
               int(self.options.get('slice_width', 640)),
               float(self.options.get('overlap_height_ratio', .2)),
               float(self.options.get('overlap_width_ratio', .2)))
        if key != self._slice_plan_key:
            self._slice_boxes = tuple(tuple(box) for box in get_slice_bboxes(
                image_height=height, image_width=width, slice_height=max(1, key[2]),
                slice_width=max(1, key[3]), overlap_height_ratio=key[4],
                overlap_width_ratio=key[5], auto_slice_resolution=False))
            self._slice_plan_key = key
        return [(frame[y1:y2, x1:x2], (x1, y1)) for x1, y1, x2, y2 in self._slice_boxes]

    def _tile_batch_policy(self):
        """Return maximum/minimum tile batch sizes supported by this backend."""
        if self._batch_limit is not None:
            return self._batch_limit, 1
        model = self.model
        if self._darkfusion_inference_backend == 'onnxruntime':
            shape = getattr(model, 'input_shape', ()) or ()
            if shape and isinstance(shape[0], (int, np.integer)) and shape[0] > 0:
                # The shared ORT model pads/trims its fixed batch itself.
                return int(shape[0]), 1
            providers = str(getattr(model, 'providers', None) or getattr(model, 'provider', '')).lower()
            # RTX/TensorRT EP optimization profiles can further restrict a
            # symbolic graph's batch. Keep their existing compiled contract.
            if 'tensorrt' in providers:
                return 1, 1
            return (4, 1) if any(name in providers for name in ('cuda', 'dml')) else (1, 1)
        auto_backend = getattr(getattr(model, 'predictor', None), 'model', None)
        if getattr(auto_backend, 'format', '') == 'engine':
            backend = getattr(auto_backend, 'backend', None)
            try:
                if not backend.dynamic:
                    fixed = int(backend.bindings['images'].shape[0])
                    return max(1, fixed), max(1, fixed)
                engine = backend.model
                if hasattr(engine, 'get_tensor_profile_shape'):
                    minimum, _optimal, maximum = engine.get_tensor_profile_shape('images', 0)
                else:
                    minimum, _optimal, maximum = engine.get_profile_shape(0, engine.get_binding_index('images'))
                low, high = max(1, int(minimum[0])), max(1, int(maximum[0]))
                return max(low, min(4, high)), low
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                return 1, 1
        import torch
        if str(self.device).lower() != 'cpu' and isinstance(getattr(model, 'model', None), torch.nn.Module):
            return 4, 1
        return 1, 1

    def predict(self, source, **kwargs):
        from darkfusion_onnx_runtime import OrtResult, OrtBoxes, OrtMasks, OrtKeypoints, OrtOBB
        frame = np.asarray(source)
        if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError('SAHI video requires a nonempty BGR frame.')
        # Per-tile tracking would treat adjacent tiles as different video frames.
        # Track only the merged full-frame result in track(), below.
        self.predictor.prediction_kwargs = dict(self.prediction_kwargs, **kwargs)
        if self._native_quantize_keyword and 'half' in self.predictor.prediction_kwargs:
            half = self.predictor.prediction_kwargs.pop('half')
            # Match Ultralytics' native mapping once without emitting a
            # deprecation warning for every slice of every video frame.
            self.predictor.prediction_kwargs.setdefault('quantize', 16 if half else None)
        self.predictor.prediction_kwargs['mode'] = 'predict'
        self.predictor.prediction_kwargs.pop('_prepared_video', None)
        before_plan = time.perf_counter()
        inputs = self._tile_inputs(frame)
        plan_ms = (time.perf_counter() - before_plan) * 1000
        batch_size, minimum_batch = self._tile_batch_policy()
        def sliced(batch):
            return self.predictor._darkfusion_onnx_sliced_predictions(
                frame, self.options.get('slice_height', 640), self.options.get('slice_width', 640),
                self.options.get('overlap_height_ratio', .2), self.options.get('overlap_width_ratio', .2),
                inference_inputs=inputs, input_is_bgr=True, batch_size=batch, minimum_batch=minimum_batch)
        try:
            predictions = sliced(batch_size)
        except RuntimeError as error:
            if batch_size <= 1 or minimum_batch > 1 or 'out of memory' not in str(error).lower():
                raise
            error.__traceback__ = None
            self._batch_limit = 1
            logger.warning('SAHI tile batch exceeded available memory; retaining single-tile inference.')
            from darkfusion_gpu_memory import _trim_cuda
            _trim_cuda()
            predictions = sliced(1)
        self.last_profile = dict(self.predictor.last_sliced_profile)
        self.last_profile['sahi_prepare_ms'] += plan_ms
        predictions = sorted(predictions, key=lambda item: item['confidence'], reverse=True)
        predictions = predictions[:int(kwargs.get('max_det', self.prediction_kwargs.get('max_det', 300)))]
        shape = frame.shape[:2]
        result = OrtResult(frame, '', self.names, {})
        rows = [list(item['bbox']) + [item['confidence'], item['class_id']] for item in predictions]
        result.boxes = OrtBoxes(rows, shape)
        if self.task == 'segment':
            polygons = []
            for item in predictions:
                contours = [np.asarray(points, dtype=np.float32).reshape(-1, 2)
                            for points in item.get('segmentation', []) if len(points) >= 6]
                polygons.append(max(contours, key=cv2.contourArea) if contours else np.empty((0, 2), np.float32))
            # Overlays already consume polygons. Avoid allocating N full-frame
            # raster masks just to convert them back into the same polygons.
            result.masks = OrtMasks(np.empty((len(polygons), 0, 0), np.uint8), shape)
            result.masks._xy = polygons
        elif self.task == 'pose' and predictions:
            result.keypoints = OrtKeypoints([item['keypoints'] for item in predictions], shape)
        elif self.task == 'obb':
            result.obb = OrtOBB([item['xywhr'] + [item['confidence'], item['class_id']]
                                for item in predictions], shape)
        return [result]

    __call__ = predict

    def track(self, source, persist=False, tracker='bytetrack.yaml', **kwargs):
        from darkfusion_onnx_tracking import ProjectTrackerAdapter, tracker_config_key
        results = self.predict(source, **kwargs)
        key = tracker_config_key(tracker, self.device)
        if not persist or self.tracker is None or self.tracker.config_key != key:
            self.tracker = ProjectTrackerAdapter(tracker, self.device)
        return [self.tracker.apply(result) for result in results]
