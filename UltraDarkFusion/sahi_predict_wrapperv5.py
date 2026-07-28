import logging
import inspect
import hashlib

from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
import os
from PIL import UnidentifiedImageError
from sahi import AutoDetectionModel
from prediction_size_filter import prediction_size_allowed_xyxy

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SahiPredictWrapper:
    def __init__(
        self,
        model_type,
        model_path,
        confidence_threshold,
        device,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=False,
        perform_standard_pred=True,
        show_preview=False,
        min_size_px=0.0,
        max_percent=1.0,
    ):
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device
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

        with open(txt_file_path, "w", encoding="utf-8") as f:
            for line in merged_lines:
                f.write(line + "\n")

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
        try:
            image_rgb = read_image(image_path)
            class_names = [name.strip() for name in (class_names or []) if name.strip()]
            desired_classes = [name.strip() for name in (desired_classes or class_names) if name.strip()]
            class_to_id = {name.lower(): idx for idx, name in enumerate(class_names)}
            desired_lookup = {name.lower() for name in desired_classes}
            excluded_names = [
                name for name in class_names
                if desired_lookup and name.lower() not in desired_lookup
            ]

            sliced_kwargs = self._supported_sliced_kwargs({
                "image": image_rgb,
                "detection_model": self.detection_model,
                "slice_height": slice_height,
                "slice_width": slice_width,
                "overlap_height_ratio": overlap_height_ratio,
                "overlap_width_ratio": overlap_width_ratio,
                "perform_standard_pred": self.perform_standard_pred,
                "postprocess_type": self.postprocess_type,
                "postprocess_match_metric": self.postprocess_match_metric,
                "postprocess_match_threshold": self.postprocess_match_threshold,
                "postprocess_class_agnostic": self.postprocess_class_agnostic,
                "exclude_classes_by_name": excluded_names or None,
                "verbose": 0,
            })

            result = get_sliced_prediction(**sliced_kwargs)

            txt_file_path = os.path.splitext(image_path)[0] + '.txt'
            yolo_lines = []
            skipped_size_count = 0

            for obj in result.object_prediction_list:
                category_name = str(obj.category.name).strip()
                category_key = category_name.lower()

                if desired_lookup and category_key not in desired_lookup:
                    continue

                bbox = obj.bbox.to_voc_bbox()
                image_h, image_w = image_rgb.shape[:2]

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

                class_id = class_to_id.get(category_key, int(obj.category.id))
                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            self._write_yolo_lines(txt_file_path, yolo_lines, overwrite=overwrite)
            self.size_filtered_count += skipped_size_count

            return len(yolo_lines)

        except UnidentifiedImageError:
            logger.info(f"Cannot open image: {image_path}")
            return 0
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return 0

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
            "last_image_path": last_image_path,
            "last_detected_image_path": last_detected_image_path,
        }
