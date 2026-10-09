import logging
import inspect
import hashlib
import tempfile

from sahi.predict import get_sliced_prediction
from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
)
from sahi.prediction import ObjectPrediction
from sahi.slicing import slice_image
from sahi.utils.cv import read_image
import os
from PIL import Image, UnidentifiedImageError
from sahi import AutoDetectionModel
from prediction_size_filter import prediction_size_allowed_xyxy

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
    ):
        self.fp16 = bool(fp16)
        self.inference_backend = str(inference_backend or "auto").strip().lower()
        self.onnx_provider = str(onnx_provider or "auto").strip().lower()
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self._uses_darkfusion_onnx = False
        self.onnx_runtime_model = None
        self.detection_model = self._create_detection_model(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )
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
            from darkfusion_onnx_runtime import DarkFusionOnnxModel

            cache_key = hashlib.sha1(os.path.abspath(model_path).encode("utf-8")).hexdigest()[:12]
            cache_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                ".darkfusion_cache",
                "sahi_onnx",
                cache_key,
            )
            runtime_model = DarkFusionOnnxModel(
                model_path,
                providers=self.onnx_provider,
                strict_provider=True,
                include_cpu_fallback=True,
                fp16=self.fp16,
                cache_dir=cache_dir,
            )
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

    def _merge_onnx_tile_predictions(self, predictions):
        """Apply the selected native SAHI postprocessor to ONNX tile results."""
        if not predictions:
            return []
        processors = {
            "GREEDYNMM": GreedyNMMPostprocess,
            "NMM": NMMPostprocess,
            "NMS": NMSPostprocess,
            "LSNMS": LSNMSPostprocess,
        }
        processor_class = processors.get(str(self.postprocess_type or "GREEDYNMM").upper(), GreedyNMMPostprocess)
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
            )
            for prediction in predictions
        ]
        return [
            {
                "bbox": tuple(float(value) for value in prediction.bbox.to_voc_bbox()),
                "confidence": float(prediction.score.value),
                "class_id": int(prediction.category.id),
            }
            for prediction in processor(object_predictions)
        ]

    def _darkfusion_onnx_sliced_predictions(
        self,
        image_rgb,
        slice_height,
        slice_width,
        overlap_height_ratio,
        overlap_width_ratio,
    ):
        if self.onnx_runtime_model is None:
            raise RuntimeError("DarkFusion ONNX Runtime was not initialized for SAHI.")
        slices = slice_image(
            image=image_rgb,
            slice_height=max(1, int(slice_height)),
            slice_width=max(1, int(slice_width)),
            overlap_height_ratio=float(overlap_height_ratio),
            overlap_width_ratio=float(overlap_width_ratio),
            auto_slice_resolution=False,
            verbose=False,
        )
        predictions = []
        inference_inputs = [
            (tile.image, tuple(int(value) for value in tile.starting_pixel[:2]))
            for tile in slices.sliced_image_list
        ]
        if self.perform_standard_pred:
            inference_inputs.append((image_rgb, (0, 0)))

        for inference_image, (offset_x, offset_y) in inference_inputs:
            result_list = self.onnx_runtime_model.predict(
                inference_image,
                conf=self.confidence_threshold,
                iou=self.postprocess_match_threshold,
                max_det=300,
            )
            if not result_list:
                continue
            boxes = getattr(result_list[0], "boxes", None)
            rows = getattr(getattr(boxes, "data", None), "numpy", lambda: [])()
            for row in rows:
                if len(row) < 6:
                    continue
                x1, y1, x2, y2, confidence, class_id = (float(value) for value in row[:6])
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append({
                    "bbox": (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                    "confidence": confidence,
                    "class_id": int(class_id),
                })
        return self._merge_onnx_tile_predictions(predictions)

    def _onnx_teammate_rejected_indices(self, image_rgb, predictions, class_names):
        """Apply the normal teammate filter to DarkFusion ONNX tile detections."""
        if not self.ignore_teammates or not predictions:
            return set()
        try:
            from darkfusion_teammate_review import TeammateMarkerClassifier

            if self._teammate_classifier is None:
                provider_names = " ".join(
                    str(provider) for provider in getattr(self.onnx_runtime_model, "providers", [])
                ).lower()
                self._teammate_classifier = TeammateMarkerClassifier(
                    "cuda" if "cuda" in provider_names else "cpu"
                )
            image = Image.fromarray(image_rgb).convert("RGB")
            image_h, image_w = image_rgb.shape[:2]
            bounds = []
            indices = []
            for index, prediction in enumerate(predictions):
                class_id = int(prediction["class_id"])
                category_name = (
                    class_names[class_id].strip().lower()
                    if 0 <= class_id < len(class_names) else str(class_id)
                )
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

        if self._uses_darkfusion_onnx:
            try:
                predictions = self._darkfusion_onnx_sliced_predictions(
                    inference_image,
                    slice_height,
                    slice_width,
                    overlap_height_ratio,
                    overlap_width_ratio,
                )
            except Exception as error:
                raise RuntimeError(f"DarkFusion ONNX SAHI inference failed for {image_path}: {error}") from error
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
            if self._uses_darkfusion_onnx:
                class_id = int(obj["class_id"])
                category_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
                bbox = obj["bbox"]
            else:
                category_name = str(obj.category.name).strip()
                class_id = int(obj.category.id)
                bbox = obj.bbox.to_voc_bbox()
            category_key = category_name.lower()

            if desired_lookup and category_key not in desired_lookup:
                continue

            if not self._roi_prediction_allowed(
                bbox, selected_rois, inference_roi, image_w, image_h
            ):
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

            class_id = class_to_id.get(category_key, class_id)
            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

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
