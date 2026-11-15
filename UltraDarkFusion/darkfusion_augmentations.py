"""Dataset augmentation controls, effect composition, previews, and batch jobs.

The existing callbacks use the main window's shared image, annotation, model,
and GPU services. Importing this module does not start a job or load weights.
"""

import concurrent.futures
import contextlib
import cv2
import json
import logging
import math
import numpy as np
import os
import psutil
import random
import shutil
import threading
import time
import torch
import uuid
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import QColorDialog, QComboBox, QMessageBox
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from darkfusion_annotation_style import get_color
from darkfusion_annotations import BoundingBox
from darkfusion_dataset_operations import (
    read_json_file, resolve_dataset_classes_path, resolve_dataset_metadata_path,
    write_label_text_atomic,
)
from darkfusion_environment import APP_DIR
from darkfusion_extraction import ExtractionWriteError, save_extraction_outputs
from darkfusion_gpu_memory import (
    gpu_work, model_allowed as gpu_model_allowed, release_worker_models, sam_batch_resources,
)
from darkfusion_image_loading import (
    DATASET_IMAGE_SUFFIXES, normalize_augmentation_output_format, normalize_image_extension,
    save_cv_image,
)
from darkfusion_image_viewers import AugmentationPreviewViewer, NestedLabelSuggestionViewer
from darkfusion_inference import SAM3_DEFAULT_IMGSZ, normalize_sam_imgsz, sam_predictor_session
from darkfusion_sam import DEVICE
from darkfusion_ui_controls import blocked_signals
from darkfusion_verified_images import VerifiedImagesTracker
from datetime import datetime
from perlin_noise import PerlinNoise
from sahi.slicing import get_slice_bboxes as get_sahi_slice_bboxes
from ultralytics import SAM


logger = logging.getLogger("UltraDarkFusionLogger")


try:
    import sip
except ModuleNotFoundError:
    from PyQt5 import sip


class ImageProcessingThread(QThread):
    progressSignal = pyqtSignal(int)

    def __init__(self, image_files, yolo_annotation_files, save_folder_path, limit_percentage, image_format):
        super().__init__()
        self.image_files = image_files
        self.yolo_annotation_files = yolo_annotation_files
        self.save_folder_path = save_folder_path
        self.limit_percentage = limit_percentage
        self.image_format = normalize_augmentation_output_format(image_format)
        self.image_extension = self.image_format if self.image_format == "source" else normalize_image_extension(self.image_format)
        self.counter = 0
        self.rejected_count = 0
        self.failure_message = ""
        self.output_failures = []
        self.stable_diffusion_pipeline = None
        self.sam_model = None
        self.sam_imgsz = normalize_sam_imgsz(SAM3_DEFAULT_IMGSZ)
        self.sam_half = torch.cuda.is_available()
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True
        self.requestInterruption()

    def should_cancel(self):
        return self._cancel_requested or self.isInterruptionRequested()

    def load_sam_model(self):
        try:
            sam_path = os.path.join("Sam", "sam3.pt")
            model = SAM(sam_path)
            logger.info(
                "Loaded SAM3 for negative-image generation at imgsz=%s, fp16=%s.",
                self.sam_imgsz,
                self.sam_half,
            )
            return model
        except Exception as e:
            logger.error(f"Error loading SAM3 model: {e}")
            return None

    def load_stable_diffusion_pipeline(self):
        try:
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline

            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                trust_remote_code=False,
                torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32
            )
            pipe.to(DEVICE)
            # Diffusers already selects SDPA on modern PyTorch. Replacing it
            # with sliced attention adds serial work and disables that kernel.
            if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                pipe.enable_attention_slicing()
            return pipe
        except Exception as e:
            logger.error(f"Error loading inpaint model: {e}")
            self.failure_message = f"Inpaint model could not be loaded: {e}"
            return None

    def should_reject_inpaint(self, roi_img):
        if roi_img is None or roi_img.size == 0:
            return True

        if self.is_solid_color(roi_img):
            return True

        if self.has_large_black_region(roi_img, threshold=18, min_area_ratio=0.03):
            return True

        if self.has_low_texture_region(roi_img):
            return True

        return False

    def has_low_texture_region(self, img, std_threshold=8.0):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray)) < std_threshold

    @release_worker_models("sam_model", "stable_diffusion_pipeline")
    def run(self):
        if self.should_cancel():
            return

        if self.sam_model is None:
            self.sam_model = self.load_sam_model()

        if self.should_cancel():
            return

        if self.stable_diffusion_pipeline is None:
            self.stable_diffusion_pipeline = self.load_stable_diffusion_pipeline()

        if self.should_cancel():
            return

        if self.stable_diffusion_pipeline is None:
            logger.error("Inpaint pipeline could not be loaded. Skipping negative-image generation.")
            self.failure_message = self.failure_message or "Inpaint model could not be loaded."
            return

        total_images = len(self.image_files)
        if total_images == 0:
            self.progressSignal.emit(100)
            return

        try:
            os.makedirs(self.save_folder_path, exist_ok=True)
        except OSError as error:
            self.failure_message = f"Could not create negative-image output folder: {error}"
            logger.error(self.failure_message)
            return

        for index, (image_path, yolo_annotation_path) in enumerate(
            zip(self.image_files, self.yolo_annotation_files)
        ):
            if self.should_cancel():
                break

            try:
                self.process_single_image(
                    image_path=image_path,
                    yolo_annotation_path=yolo_annotation_path,
                    save_folder_path=self.save_folder_path,
                    limit_percentage=self.limit_percentage,
                    image_format=self.image_format,
                )
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                self.output_failures.append(f"{image_path}: {e}")

            if self.should_cancel():
                break

            self.progressSignal.emit(int(((index + 1) / total_images) * 100))

    def process_single_image(self, image_path, yolo_annotation_path, save_folder_path, limit_percentage, image_format):
        if self.should_cancel():
            return

        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Error reading image: {image_path}")
            return

        if not os.path.exists(yolo_annotation_path):
            logger.warning(f"Annotation file missing: {yolo_annotation_path}")
            return

        img_height, img_width = img.shape[:2]
        boxes = self.read_yolo_boxes(yolo_annotation_path, img_width, img_height)

        if not boxes:
            logger.warning(f"No valid boxes found for: {image_path}")
            return

        mask = self.build_mask_with_sam(image_path, img.shape, boxes)

        if self.should_cancel() or mask is None or not np.any(mask):
            logger.warning(f"No valid mask regions found for: {image_path}")
            return

        mask = self.refine_mask(mask)

        inpainted_img, roi = self.fill_cropped_area(img, mask)
        if self.should_cancel() or inpainted_img is None or roi is None:
            logger.error(f"Failed to inpaint image: {image_path}")
            if not self.should_cancel():
                self.output_failures.append(f"Could not inpaint {image_path}.")
            return

        x1, y1, x2, y2 = roi
        edited_roi = inpainted_img[y1:y2 + 1, x1:x2 + 1]

        if self.should_reject_inpaint(edited_roi):
            logger.warning(f"Rejected bad inpaint result: {image_path}")
            self.rejected_count += 1
            return

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_format = normalize_augmentation_output_format(image_format)
        if output_format == "source":
            output_format = normalize_image_extension(os.path.splitext(image_path)[1], default=".jpg").lstrip(".")
        else:
            output_format = normalize_image_extension(output_format, default=".jpg").lstrip(".")

        negative_filename = f"{base_filename}_no_labels_negative.{output_format}"

        self.save_image_with_limit(
            inpainted_img,
            save_folder_path,
            negative_filename,
            limit_percentage,
            image_format
        )

    def read_yolo_boxes(self, yolo_annotation_path, img_width, img_height):
        boxes = []

        with open(yolo_annotation_path, "r", encoding="utf-8") as f:
            yolo_data = [line.strip() for line in f if line.strip()]

        for data in yolo_data:
            parts = data.split()
            if len(parts) < 5:
                continue

            try:
                int(float(parts[0]))
                values = [float(x) for x in parts[1:]]
                x_center, y_center, w, h = values[:4]
            except Exception:
                continue

            value_count = len(values)
            bbox_like = (
                value_count in (4, 5)
                and 0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 < w <= 1.0
                and 0.0 < h <= 1.0
            )
            pose_like = (
                value_count > 4
                and (value_count - 4) % 3 == 0
                and all(
                    float(v).is_integer() and int(float(v)) in (0, 1, 2)
                    for v in values[6::3]
                )
                and 0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 < w <= 1.0
                and 0.0 < h <= 1.0
            )
            polygon_like = (
                not pose_like
                and value_count >= 6
                and value_count % 2 == 0
                and all(0.0 <= float(v) <= 1.0 for v in values)
            )

            if not (bbox_like or pose_like or polygon_like):
                continue

            if polygon_like:
                xs = values[0::2]
                ys = values[1::2]
                x_min = max(0, min(img_width - 1, int(min(xs) * img_width)))
                x_max = max(0, min(img_width - 1, int(max(xs) * img_width)))
                y_min = max(0, min(img_height - 1, int(min(ys) * img_height)))
                y_max = max(0, min(img_height - 1, int(max(ys) * img_height)))
            else:
                x_min = max(0, min(img_width - 1, int((x_center - w / 2) * img_width)))
                x_max = max(0, min(img_width - 1, int((x_center + w / 2) * img_width)))
                y_min = max(0, min(img_height - 1, int((y_center - h / 2) * img_height)))
                y_max = max(0, min(img_height - 1, int((y_center + h / 2) * img_height)))

            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])

        return boxes

    def build_mask_with_sam(self, image_path, image_shape, boxes):
        img_height, img_width = image_shape[:2]
        combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        if self.sam_model is None:
            for x_min, y_min, x_max, y_max in boxes:
                combined_mask[y_min:y_max, x_min:x_max] = 255
            return combined_mask

        for box in boxes:
            if self.should_cancel():
                return None

            results = None
            try:
                predict_kwargs = {
                    "source": image_path,
                    "bboxes": [box],
                    "imgsz": self.sam_imgsz,
                    "verbose": False,
                    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
                }
                if self.sam_half:
                    predict_kwargs["quantize"] = 16

                with torch.inference_mode():
                    results = self.sam_model.predict(
                        **predict_kwargs
                    )

                if self.should_cancel():
                    return None

                if not results:
                    combined_mask[box[1]:box[3], box[0]:box[2]] = 255
                    continue

                r = results[0]

                if getattr(r, "masks", None) is not None and r.masks.data is not None:
                    mask_tensor = r.masks.data[0]
                    mask = mask_tensor.detach().cpu().numpy()
                    mask = (mask > 0).astype(np.uint8) * 255
                    combined_mask = np.maximum(combined_mask, mask)
                else:
                    combined_mask[box[1]:box[3], box[0]:box[2]] = 255

            except Exception as e:
                logger.warning(f"SAM3 failed for box {box}; using the box region as fallback mask. {e}")
                combined_mask[box[1]:box[3], box[0]:box[2]] = 255
            finally:
                try:
                    del results
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        return combined_mask

    def refine_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = (mask > 10).astype(np.uint8) * 255
        return mask

    def get_mask_roi(self, mask, pad=64):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        x1 = max(0, xs.min() - pad)
        y1 = max(0, ys.min() - pad)
        x2 = xs.max() + pad
        y2 = ys.max() + pad
        return x1, y1, x2, y2

    def fill_cropped_area(self, original_img, mask):
        roi = self.get_mask_roi(mask, pad=64)
        if roi is None:
            return None, None

        h, w = original_img.shape[:2]
        x1, y1, x2, y2 = roi
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        cropped_img = original_img[y1:y2 + 1, x1:x2 + 1]
        cropped_mask = mask[y1:y2 + 1, x1:x2 + 1]

        prompt = (
            "empty background, no people, no person, no human, no body, no face, "
            "clean realistic room background, seamless wall and floor continuation, "
            "matching original scene, realistic indoor environment, photorealistic"
        )
        negative_prompt = (
            "person, people, human, man, woman, child, face, head, body, arms, legs, "
            "hands, zombie, soldier, character, gun, weapon, blood, corpse, skull, "
            "text, watermark, logo, object, duplicate object, shadow person, silhouette, "
            "black box, black square, rectangle, censor bar, blur block, artifact, distortion"
        )

        try:
            mask_image = Image.fromarray(cropped_mask).convert("RGB")
            image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

            with torch.inference_mode():
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if DEVICE.type == "cuda"
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    result = self.stable_diffusion_pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=image.resize((512, 512)),
                        mask_image=mask_image.resize((512, 512)),
                        num_inference_steps=28,
                        guidance_scale=9.0
                    )

            if self.should_cancel():
                return None, None

            inpainted_crop = cv2.cvtColor(np.array(result.images[0]), cv2.COLOR_RGB2BGR)
            inpainted_crop = cv2.resize(
                inpainted_crop,
                (cropped_img.shape[1], cropped_img.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )

            output = original_img.copy()
            output[y1:y2 + 1, x1:x2 + 1] = inpainted_crop
            return output, (x1, y1, x2, y2)

        except Exception as e:
            logger.error(f"Error in inpainting: {e}")
            return None, None

    def save_image_with_limit(self, img, save_folder_path, filename, limit, image_format):
        if self.should_cancel():
            return

        save_negative_image = random.choices([True, False], weights=[limit, 100 - limit], k=1)[0]

        if save_negative_image and (self.is_solid_color(img) or self.has_large_black_region(img)):
            self.rejected_count += 1
            return

        if save_negative_image:
            negative_filename = os.path.join(save_folder_path, filename)
            txt_path = os.path.splitext(negative_filename)[0] + ".txt"
            if os.path.isfile(negative_filename) and os.path.isfile(txt_path):
                logger.info("Negative already exists; skipping duplicate: %s", negative_filename)
                return
            def write_empty_label(path, _labels, mode="w"):
                write_label_text_atomic(path, "")
                return True

            save_extraction_outputs(
                negative_filename, img, save_cv_image, label_path=txt_path,
                labels=[], label_writer=write_empty_label,
            )

            self.counter += 1

    def is_solid_color(self, img):
        return np.all(img == img[0, 0])

    def has_large_black_region(self, img, threshold=10, min_area_ratio=0.08):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        black = (gray < threshold).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(black, 8)
        total_area = img.shape[0] * img.shape[1]

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area / total_area >= min_area_ratio:
                return True

        return False


def _relocate_extra_label_to_augmentation_dialog(self):
    """Move the legacy nested-label controls into an Augmentation dialog."""
    if getattr(self, "_extra_label_dialog_ready", False):
        return

    group = getattr(self, "groupBox_14", None)
    effects_frame = getattr(self, "frame_15", None)
    if not isinstance(group, QtWidgets.QGroupBox) or not isinstance(effects_frame, QtWidgets.QFrame):
        return

    old_parent = group.parentWidget()
    old_layout = old_parent.layout() if old_parent is not None else None
    self._take_layout_widget(old_layout, group)

    dialog = QtWidgets.QDialog(self)
    dialog.setObjectName("extra_label_dialog")
    dialog.setWindowTitle("Extra Label / SAM3 Nested Labels")
    dialog.setModal(False)
    dialog.setMinimumWidth(620)
    dialog.setSizeGripEnabled(True)

    dialog_layout = QtWidgets.QVBoxLayout(dialog)
    dialog_layout.setContentsMargins(14, 14, 14, 14)
    dialog_layout.setSpacing(10)

    description = QtWidgets.QLabel(
        "Configure the nested label created inside a parent annotation. "
            "These values also provide fallback geometry for SAM3 nested-label generation.",
        dialog,
    )
    description.setObjectName("extra_label_dialog_description")
    description.setWordWrap(True)
    dialog_layout.addWidget(description)

    sam_group = QtWidgets.QGroupBox("SAM3 Nested Labels", dialog)
    sam_group.setObjectName("sam3_auto_heads_group")
    sam_layout = QtWidgets.QFormLayout(sam_group)
    sam_layout.setContentsMargins(12, 16, 12, 12)
    sam_layout.setSpacing(8)

    self.auto_head_source_class_combo = QtWidgets.QComboBox(sam_group)
    self.auto_head_source_class_combo.setObjectName("auto_head_source_class_combo")
    self.auto_head_source_class_combo.setToolTip("Only annotations of this parent class are searched for heads.")
    sam_layout.addRow("Parent class", self.auto_head_source_class_combo)

    self.auto_head_target_class_combo = QtWidgets.QComboBox(sam_group)
    self.auto_head_target_class_combo.setObjectName("auto_head_target_class_combo")
    self.auto_head_target_class_combo.setToolTip("Class assigned to accepted SAM3 nested labels.")
    self.auto_head_target_class_combo.currentIndexChanged.connect(self._sync_auto_head_target_class_id)
    sam_layout.addRow("Output class", self.auto_head_target_class_combo)

    self.nested_label_mode_combo = QtWidgets.QComboBox(sam_group)
    self.nested_label_mode_combo.setObjectName("nested_label_mode_combo")
    self.nested_label_mode_combo.addItem("Hybrid: Prompt + Guide", "hybrid")
    self.nested_label_mode_combo.addItem("Geometry Only (SAM3 Off)", "geometry")
    self.nested_label_mode_combo.setToolTip(
        "Hybrid matches SAM3 prompt masks to the guide. Geometry Only uses the percentage box directly."
    )
    self.nested_label_mode_combo.currentIndexChanged.connect(self.update_nested_label_mode_controls)
    sam_layout.addRow("Mode", self.nested_label_mode_combo)

    self.auto_head_prompt_edit = QtWidgets.QLineEdit("human head", sam_group)
    self.auto_head_prompt_edit.setObjectName("auto_head_prompt_edit")
    self.auto_head_prompt_edit.setToolTip("Short text concept sent to SAM3 once per image.")
    sam_layout.addRow("SAM3 prompt", self.auto_head_prompt_edit)

    self.auto_head_min_confidence = QtWidgets.QDoubleSpinBox(sam_group)
    self.auto_head_min_confidence.setObjectName("auto_head_min_confidence")
    self.auto_head_min_confidence.setRange(0.05, 0.95)
    self.auto_head_min_confidence.setSingleStep(0.05)
    self.auto_head_min_confidence.setDecimals(2)
    self.auto_head_min_confidence.setValue(0.35)
    self.auto_head_min_confidence.setToolTip("Reject SAM3 candidates below this confidence.")
    sam_layout.addRow("Min confidence", self.auto_head_min_confidence)

    self.dynamic_guide_scan_checkbox = QtWidgets.QCheckBox(
        "Scan this geometry box across the parent (SAHI)", group
    )
    self.dynamic_guide_scan_checkbox.setObjectName("dynamic_guide_scan_checkbox")
    self.dynamic_guide_scan_checkbox.setChecked(False)
    self.dynamic_guide_scan_checkbox.setToolTip(
        "Move a window of the configured W/H size across each parent label. X/Y is the preferred starting position; SAM3 runs only once."
    )
    self.dynamic_guide_scan_help = QtWidgets.QLabel(
        "Off: use this box at Center X / Top Y.  On: keep its Width / Height and slide it across the parent to find the SAM3 prompt.",
        group,
    )
    self.dynamic_guide_scan_help.setObjectName("dynamic_guide_scan_help")
    self.dynamic_guide_scan_help.setWordWrap(True)
    self.dynamic_guide_scan_help.setStyleSheet("color: #9aa4b2;")
    geometry_layout = group.layout()
    if isinstance(geometry_layout, QtWidgets.QGridLayout):
        geometry_layout.addWidget(self.dynamic_guide_scan_checkbox, 6, 0, 1, 2)
        geometry_layout.addWidget(self.dynamic_guide_scan_help, 7, 0, 1, 2)

    self.guide_match_threshold = QtWidgets.QDoubleSpinBox(sam_group)
    self.guide_match_threshold.setObjectName("guide_match_threshold")
    self.guide_match_threshold.setRange(0.10, 0.90)
    self.guide_match_threshold.setSingleStep(0.05)
    self.guide_match_threshold.setDecimals(2)
    self.guide_match_threshold.setValue(0.35)
    self.guide_match_threshold.setToolTip(
        "Minimum fraction of a SAM3 candidate inside a scan window. For example, 0.35 means 35 percent."
    )
    sam_layout.addRow("Mask overlap (0-1)", self.guide_match_threshold)

    self.auto_head_geometry_fallback = QtWidgets.QCheckBox(
        "Use geometry fallback when SAM3 misses", sam_group
    )
    self.auto_head_geometry_fallback.setObjectName("auto_head_geometry_fallback")
    self.auto_head_geometry_fallback.setChecked(False)
    self.auto_head_geometry_fallback.setToolTip(
        "Adds the percentage-based nested box when SAM3 finds no safe prompted object. Leave off for cleaner labels."
    )
    sam_layout.addRow(self.auto_head_geometry_fallback)

    self.auto_head_preview_button = QtWidgets.QPushButton("Review Suggestions", sam_group)
    self.auto_head_preview_button.setObjectName("auto_head_preview_button")
    self.auto_head_preview_button.setMinimumHeight(30)
    self.auto_head_preview_button.setToolTip(
        "Open the large Previous/Next suggestion viewer. Labels are not changed until Apply Current or Apply All Safe."
    )
    self.auto_head_preview_button.clicked.connect(self.open_nested_label_suggestion_review)
    sam_layout.addRow(self.auto_head_preview_button)

    self.auto_head_run_button = QtWidgets.QPushButton("Generate Selected Labels", sam_group)
    self.auto_head_run_button.setObjectName("auto_head_run_button")
    self.auto_head_run_button.setMinimumHeight(30)
    self.auto_head_run_button.setToolTip(
        "Find the prompted concept inside existing parent labels and append accepted output-class boxes."
    )
    self.auto_head_run_button.clicked.connect(self.run_sam3_auto_heads)
    self.auto_head_run_button.hide()

    self.auto_head_status_label = QtWidgets.QLabel("Ready", sam_group)
    self.auto_head_status_label.setObjectName("auto_head_status_label")
    self.auto_head_status_label.setWordWrap(True)
    sam_layout.addRow(self.auto_head_status_label)
    dialog_layout.addWidget(sam_group)

    self.auto_head_preview_label = QtWidgets.QLabel("Preview Current Image", dialog)
    self.auto_head_preview_label.setObjectName("auto_head_preview_label")
    self.auto_head_preview_label.setAlignment(Qt.AlignCenter)
    self.auto_head_preview_label.setMinimumSize(560, 315)
    self.auto_head_preview_label.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
    )
    self.auto_head_preview_label.setStyleSheet(
        "QLabel#auto_head_preview_label {"
            "background-color: #0d1117; color: #8b949e; "
            "border: 1px solid #30363d; border-radius: 4px;"
            "}"
    )
    self.auto_head_preview_label.setToolTip(
        "Orange: parent labels. Green: proposed output labels. Cyan: existing output labels. Magenta: live geometry fallback guide. Click to enlarge."
    )
    self.auto_head_preview_label.setCursor(Qt.PointingHandCursor)
    self.auto_head_preview_label.installEventFilter(self)
    self.auto_head_preview_label.hide()

    self._auto_head_geometry_preview_timer = QtCore.QTimer(self)
    self._auto_head_geometry_preview_timer.setSingleShot(True)
    self._auto_head_geometry_preview_timer.timeout.connect(self._render_cached_auto_head_preview)
    for control in (
        getattr(self, "head_width", None),
        getattr(self, "head_height", None),
        getattr(self, "head_horizontal", None),
        getattr(self, "head_verticle", None),
    ):
        if control is not None and hasattr(control, "valueChanged"):
            control.valueChanged.connect(self.queue_auto_head_geometry_preview_update)
    self.auto_head_geometry_fallback.stateChanged.connect(
        self.queue_auto_head_geometry_preview_update
    )
    self.dynamic_guide_scan_checkbox.stateChanged.connect(
        self.queue_auto_head_geometry_preview_update
    )
    self.guide_match_threshold.valueChanged.connect(
        self.queue_auto_head_geometry_preview_update
    )
    self.update_nested_label_mode_controls()

    group.setTitle("Nested Label Settings")
    self.label_8.setText("Width (% parent)")
    self.label_38.setText("Height (% parent)")
    self.label_39.setText("Center X (%)")
    self.label_40.setText("Top Y (%)")
    self.head_width.setToolTip("Width of the extra-label box as a percentage of the parent label.")
    self.head_height.setToolTip("Height of the extra-label box as a percentage of the parent label.")
    self.head_horizontal.setToolTip("Horizontal center of the extra-label box: 0 is the parent's left edge and 100 is its right edge.")
    self.head_verticle.setToolTip("Top edge of the extra-label box: 0 is the parent's top and 100 is its bottom.")
    group.setParent(dialog)
    group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    dialog_layout.addWidget(group)

    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, parent=dialog)
    buttons.rejected.connect(dialog.hide)
    dialog_layout.addWidget(buttons)

    open_button = getattr(self, "extra_label_dialog_button", None)
    if not isinstance(open_button, QtWidgets.QPushButton):
        open_button = QtWidgets.QPushButton("Extra Label...", effects_frame)
        open_button.setObjectName("extra_label_dialog_button")
        setattr(self, "extra_label_dialog_button", open_button)
    open_button.setToolTip("Open nested-label geometry and SAM3 nested-label settings.")
    open_button.setMinimumHeight(28)
    open_button.clicked.connect(self.show_extra_label_dialog)

    self.extra_label_dialog = dialog
    self._extra_label_dialog_ready = True


def show_extra_label_dialog(self):
    dialog = getattr(self, "extra_label_dialog", None)
    if not isinstance(dialog, QtWidgets.QDialog):
        self._relocate_extra_label_to_augmentation_dialog()
        dialog = getattr(self, "extra_label_dialog", None)
    if not isinstance(dialog, QtWidgets.QDialog):
        return
    self.refresh_auto_head_class_selectors()
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def _sync_auto_head_target_class_id(self, _index=None):
    combo = getattr(self, "auto_head_target_class_combo", None)
    class_field = getattr(self, "class_id", None)
    if not isinstance(combo, QtWidgets.QComboBox) or not isinstance(class_field, QtWidgets.QLineEdit):
        return
    class_id = combo.currentData()
    if class_id is not None:
        class_field.setText(str(int(class_id)))
    target_name = self._selected_auto_head_target_name()
    run_button = getattr(self, "auto_head_run_button", None)
    if isinstance(run_button, QtWidgets.QPushButton):
        run_button.setText(f"Generate {target_name} Labels" if target_name else "Generate Selected Labels")
    if getattr(self, "_auto_head_preview_context", None):
        self._render_cached_auto_head_preview()


def _selected_auto_head_target_name(self):
    combo = getattr(self, "auto_head_target_class_combo", None)
    if not isinstance(combo, QtWidgets.QComboBox) or combo.currentIndex() < 0:
        return ""
    text = str(combo.currentText()).strip()
    return text.split(":", 1)[-1].strip() if ":" in text else text


def selected_nested_label_mode(self):
    combo = getattr(self, "nested_label_mode_combo", None)
    if not isinstance(combo, QtWidgets.QComboBox):
        return "hybrid"
    return str(combo.currentData() or "hybrid")


def update_nested_label_mode_controls(self, *_args):
    uses_sam = self.selected_nested_label_mode() != "geometry"
    for control_name in (
        "auto_head_prompt_edit",
        "auto_head_min_confidence",
        "dynamic_guide_scan_checkbox",
        "dynamic_guide_scan_help",
        "guide_match_threshold",
        "auto_head_geometry_fallback",
    ):
        control = getattr(self, control_name, None)
        if control is not None:
            control.setEnabled(uses_sam)
    self.queue_auto_head_geometry_preview_update()


def refresh_auto_head_class_selectors(self):
    source_combo = getattr(self, "auto_head_source_class_combo", None)
    target_combo = getattr(self, "auto_head_target_class_combo", None)
    if not isinstance(source_combo, QtWidgets.QComboBox) or not isinstance(target_combo, QtWidgets.QComboBox):
        return

    classes = self.load_classes(create_if_missing=False) or []
    source_name = str(getattr(self, "settings", {}).get("autoHeadSourceClass", "person"))
    target_id_text = str(self.class_id.text() if hasattr(self, "class_id") else "").strip()
    try:
        target_id = int(target_id_text)
    except (TypeError, ValueError):
        target_id = None

    with blocked_signals(source_combo), blocked_signals(target_combo):
        source_combo.clear()
        target_combo.clear()
        for class_id, class_name in enumerate(classes):
            text = f"{class_id}: {class_name}"
            source_combo.addItem(text, class_id)
            target_combo.addItem(text, class_id)

        source_index = next(
            (i for i, name in enumerate(classes) if str(name).casefold() == source_name.casefold()),
            next((i for i, name in enumerate(classes) if str(name).casefold() in {"person", "people", "human"}), 0),
        ) if classes else -1
        if target_id is None or not (0 <= target_id < len(classes)):
            target_id = next(
                (i for i, name in enumerate(classes) if str(name).casefold() in {"head", "human head"}),
                0,
            ) if classes else -1
        source_combo.setCurrentIndex(source_index)
        target_combo.setCurrentIndex(target_id)

    self._sync_auto_head_target_class_id()


def ensure_yolo_augmentation_controls(self):
    frame = getattr(self, "frame_15", None)
    if frame is None:
        return

    layout = frame.layout()
    checkbox_style = ""
    style_source = getattr(self, "smoke_checkbox", None)
    if isinstance(style_source, QtWidgets.QCheckBox):
        checkbox_style = style_source.styleSheet()

    controls = (
        (
            "brightness_contrast_checkbox",
            "Light",
            "Randomly adjust brightness and contrast for YOLO lighting generalization.",
        ),
        (
            "hsv_jitter_checkbox",
            "HSV",
            "Randomly shift hue, saturation, and value without changing labels.",
        ),
        (
            "gaussian_noise_checkbox",
            "Noise",
            "Add light sensor/compression noise without changing labels.",
        ),
        (
            "jpeg_compression_checkbox",
            "JPEG",
            "Round-trip through JPEG compression to simulate stream and upload artifacts.",
        ),
    )

    for index, (object_name, text, tooltip) in enumerate(controls):
        widget = getattr(self, object_name, None)
        if not isinstance(widget, QtWidgets.QCheckBox):
            widget = QtWidgets.QCheckBox(frame)
            widget.setObjectName(object_name)
            widget.setLayoutDirection(Qt.LeftToRight)
            widget.setTristate(False)
            setattr(self, object_name, widget)
            if isinstance(layout, QtWidgets.QGridLayout):
                layout.addWidget(widget, 8 + index // 2, index % 2, 1, 1)

        if checkbox_style:
            widget.setStyleSheet(checkbox_style)
        widget.setText(text)
        widget.setToolTip(tooltip)


def _compact_effects_panel(self):
    frame = getattr(self, "frame_15", None)
    if frame is None or getattr(self, "_effects_panel_compacted", False):
        return

    layout = frame.layout()
    if not isinstance(layout, QtWidgets.QGridLayout):
        return
    # Populate any missing dynamic effects before arranging the panel.
    self.ensure_yolo_augmentation_controls()
    frame.setProperty("collapsibleSectionTitle", "OpenCV Augmentation")

    frame.setMinimumWidth(0)
    frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    clarity = getattr(self, "object_clarity_augmentation_checkbox", None)
    if not isinstance(clarity, QtWidgets.QCheckBox):
        clarity = QtWidgets.QCheckBox("Object Clarity", frame)
        clarity.setObjectName("object_clarity_augmentation_checkbox")
        clarity.setChecked(bool(getattr(self, "settings", {}).get("augmentationObjectClarity", False)))
        clarity.setToolTip("Create Real-ESRGAN variants inside existing annotation areas. Restored detail is estimated; image size and labels are preserved.")
        clarity.toggled.connect(lambda checked: self._object_clarity_augmentation_changed(checked, sam=False))
        self.object_clarity_augmentation_checkbox = clarity
    names = (
        "crop_button",
        "sight_picture",
        "smoke_checkbox",
        "glass_checkbox",
        "flash_checkbox",
        "motion_blur",
        "brightness_contrast_checkbox",
        "hsv_jitter_checkbox",
        "gaussian_noise_checkbox",
        "jpeg_compression_checkbox",
        "flip_checkbox",
        "make_copy",
        "extra_label_dialog_button",
        "object_clarity_augmentation_checkbox",
    )
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)

    layout.setContentsMargins(10, 8, 10, 8)
    layout.setHorizontalSpacing(8)
    layout.setVerticalSpacing(6)
    for col in range(3):
        layout.setColumnStretch(col, 1)

    widgets["crop_button"].setText("Run Augmentation")
    widgets["make_copy"].setText("Copy")

    widgets["crop_button"].setMinimumHeight(28)
    widgets["crop_button"].setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    preview = getattr(self, "augmentation_preview_label", None)
    if not isinstance(preview, QtWidgets.QLabel):
        preview = QtWidgets.QLabel(frame)
        preview.setObjectName("augmentation_preview_label")
        self.augmentation_preview_label = preview

    preview.setAlignment(Qt.AlignCenter)
    preview.setMinimumSize(180, 112)
    preview.setMaximumSize(QtWidgets.QWIDGETSIZE_MAX, 160)
    preview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    preview.setScaledContents(False)
    preview.setText("Preview")
    preview.setCursor(Qt.PointingHandCursor)
    preview.setToolTip("Preview of the selected augmentation. Click to open the zoomable image viewer.")
    preview.installEventFilter(self)
    preview.setStyleSheet(
        "QLabel#augmentation_preview_label {"
            "background-color: #0d1117;"
            "color: #8b949e;"
            "border: 1px solid #30363d;"
            "border-radius: 4px;"
            "}"
    )
    self._take_layout_widget(layout, preview)

    effects = ("brightness_contrast_checkbox", "hsv_jitter_checkbox", "gaussian_noise_checkbox",
               "jpeg_compression_checkbox", "flip_checkbox", "motion_blur", "smoke_checkbox",
               "flash_checkbox", "glass_checkbox", "sight_picture", "make_copy")
    for index, name in enumerate(effects):
        layout.addWidget(widgets[name], index // 3, index % 3)
    layout.addWidget(widgets["extra_label_dialog_button"], 3, 2)
    layout.addWidget(clarity, 4, 0, 1, 3)
    layout.addWidget(preview, 5, 0, 1, 3)
    layout.addWidget(widgets["crop_button"], 6, 0, 1, 3)

    self._effects_panel_compacted = True
    self.setup_augmentation_preview_timer()
    self.queue_augmentation_preview_update()


def _compact_shared_augmentation_percent_panel(self):
    if getattr(self, "_shared_augmentation_percent_compacted", False):
        return

    tab = getattr(self, "tab_21", None)
    root_layout = tab.layout() if tab is not None else None
    if not isinstance(root_layout, QtWidgets.QGridLayout):
        return

    label = getattr(self, "augmentation_percent_label", None)
    spin = getattr(self, "augmentation_percent_spinbox", None)
    if label is None or spin is None:
        return

    output_combo = self._ensure_augmentation_output_format_combo(tab)
    if output_combo is None:
        return

    frame = getattr(self, "augmentation_percent_scope_frame", None)
    if not isinstance(frame, QtWidgets.QFrame):
        frame = QtWidgets.QFrame(tab)
        frame.setObjectName("augmentation_percent_scope_frame")
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        frame.setFrameShadow(QtWidgets.QFrame.Raised)
        frame.setStyleSheet("QFrame#augmentation_percent_scope_frame { background-color: rgb(127, 127, 127); }")
        setattr(self, "augmentation_percent_scope_frame", frame)
    frame.setProperty("collapsibleSectionTitle", "Augmentation Settings")

    row = frame.layout()
    if not isinstance(row, QtWidgets.QGridLayout):
        row = QtWidgets.QGridLayout(frame)
        row.setObjectName("augmentation_percent_scope_layout")
    while row.count():
        row.takeAt(0)

    for widget in (label, spin):
        parent = widget.parentWidget()
        parent_layout = parent.layout() if parent is not None else None
        self._take_layout_widget(parent_layout, widget)
        self._compact_control(widget)

    output_label = getattr(self, "augmentation_output_format_label", None)
    if not isinstance(output_label, QtWidgets.QLabel):
        output_label = self._workflow_label("Output", frame, 58)
        output_label.setObjectName("augmentation_output_format_label")
        setattr(self, "augmentation_output_format_label", output_label)

    for widget in (output_label, output_combo):
        parent = widget.parentWidget()
        parent_layout = parent.layout() if parent is not None else None
        self._take_layout_widget(parent_layout, widget)
        self._compact_control(widget)

    label.setText("Augment % (OpenCV + SAM3)")
    label.setToolTip(
        "Percent used for normal augmentation, SAM3 Copy/Paste, Clean BG, Shadow / Highlight, Object Detail, Object Clarity, and Negatives. SAM3 Auto-Fit, Seg, and OBB run on all images."
    )
    label.setMinimumWidth(0)
    label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

    spin.setRange(1, 100)
    spin.setSuffix("%")
    spin.setMaximumWidth(120)
    spin.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    spin.setToolTip(
        "Percent used for normal augmentation, SAM3 Copy/Paste, Clean BG, Shadow / Highlight, Object Detail, Object Clarity, and Negatives. SAM3 Auto-Fit, Seg, and OBB run on all images."
    )
    output_label.setToolTip("Output image format for generated augmentation images.")
    output_combo.setMaximumWidth(110)
    output_combo.setToolTip(
        "Output format for normal augmentation, SAM3 Copy/Paste, Clean BG, Shadow / Highlight, Object Detail, Object Clarity, and Negatives. Source keeps each image extension."
    )

    row.setContentsMargins(10, 6, 10, 6)
    row.setSpacing(8)
    row.addWidget(label, 0, 0)
    row.addWidget(spin, 0, 1)
    row.addWidget(output_label, 1, 0)
    row.addWidget(output_combo, 1, 1)
    row.setColumnStretch(0, 1)

    self._shared_augmentation_percent_compacted = True


def _ensure_augmentation_output_format_combo(self, parent=None, selected_value=None):
    combo = getattr(self, "image_format_combo", None)
    if not isinstance(combo, QtWidgets.QComboBox):
        combo = QtWidgets.QComboBox(parent or getattr(self, "tab_21", None))
        combo.setObjectName("image_format_combo")
        setattr(self, "image_format_combo", combo)

    selected = normalize_augmentation_output_format(selected_value, default="source")
    with blocked_signals(combo):
        combo.clear()
        for text, data in (
            ("Source", "source"),
            (".jpg", ".jpg"),
            (".png", ".png"),
            (".webp", ".webp"),
        ):
            combo.addItem(text, data)

        index = combo.findData(selected)
        if index < 0:
            index = combo.findText(selected)
        combo.setCurrentIndex(index if index >= 0 else combo.findData("source"))

    return combo


def _reorder_augment_convert_panels(self):
    if getattr(self, "_augment_convert_panels_reordered", False):
        return

    tab = getattr(self, "tab_21", None)
    layout = tab.layout() if tab is not None else None
    if not isinstance(layout, QtWidgets.QGridLayout):
        return

    ordered_widgets = [
        getattr(self, "augmentation_percent_scope_frame", None),
        getattr(self, "frame_15", None),
        getattr(self, "groupBox_17", None),
    ]
    ordered_widgets = [widget for widget in ordered_widgets if isinstance(widget, QtWidgets.QWidget)]
    if not ordered_widgets:
        return

    for widget in ordered_widgets:
        self._take_layout_widget(layout, widget)
        widget.setMinimumWidth(0)
        widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    for row, widget in enumerate(ordered_widgets):
        layout.addWidget(widget, row, 0, 1, 1)
        layout.setRowStretch(row, 0)

    layout.setColumnStretch(0, 1)
    self._remove_legacy_create_blanks_panel(layout)
    self._augment_convert_panels_reordered = True


def _remove_legacy_create_blanks_panel(self, root_layout=None):
    if getattr(self, "_legacy_create_blanks_panel_removed", False):
        return

    for widget_name in ("process_image_button", "limit_blanks"):
        widget = getattr(self, widget_name, None)
        if isinstance(widget, QtWidgets.QWidget):
            parent = widget.parentWidget()
            parent_layout = parent.layout() if parent is not None else None
            self._take_layout_widget(parent_layout, widget)
            widget.setParent(None)
            widget.deleteLater()
        setattr(self, widget_name, None)

    group = getattr(self, "groupBox_5", None)
    if isinstance(group, QtWidgets.QWidget):
        parent = group.parentWidget()
        layout = root_layout or (parent.layout() if parent is not None else None)
        self._take_layout_widget(layout, group)
        group.setParent(None)
        group.deleteLater()
        setattr(self, "groupBox_5", None)

    self._legacy_create_blanks_panel_removed = True


def _compact_sam3_tools_panel(self):
    frame = getattr(self, "frame_16", None)
    if frame is None or getattr(self, "_sam3_panel_compacted", False):
        return

    layout = frame.layout()
    if not isinstance(layout, QtWidgets.QGridLayout):
        return

    frame.setMinimumWidth(0)
    frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    negative_checkbox = getattr(self, "create_negatives_checkbox", None)
    if not isinstance(negative_checkbox, QtWidgets.QCheckBox):
        negative_checkbox = QtWidgets.QCheckBox(frame)
        negative_checkbox.setObjectName("create_negatives_checkbox")
        setattr(self, "create_negatives_checkbox", negative_checkbox)

    object_detail_checkbox = getattr(self, "object_detail_checkbox", None)
    if not isinstance(object_detail_checkbox, QtWidgets.QCheckBox):
        object_detail_checkbox = QtWidgets.QCheckBox(frame)
        object_detail_checkbox.setObjectName("object_detail_checkbox")
        setattr(self, "object_detail_checkbox", object_detail_checkbox)

    clarity = getattr(self, "sam_object_clarity_checkbox", None)
    if not isinstance(clarity, QtWidgets.QCheckBox):
        clarity = QtWidgets.QCheckBox("Object Clarity", frame)
        clarity.setObjectName("sam_object_clarity_checkbox")
        clarity.setChecked(bool(getattr(self, "settings", {}).get("samObjectClarity", False)))
        clarity.setToolTip("Create Real-ESRGAN variants inside labeled SAM3 masks, preserving mask holes, background, image size and label geometry.")
        clarity.toggled.connect(lambda checked: self._object_clarity_augmentation_changed(checked, sam=True))
        self.sam_object_clarity_checkbox = clarity

    names = (
        "screen_update",
        "overwrite_var",
        "obb_checkbox",
        "segmentation_checkbox",
        "noise_remove_checkbox",
        "shadow_var",
        "copy_paste_checkbox",
        "object_detail_checkbox",
        "sam_object_clarity_checkbox",
        "create_negatives_checkbox",
        "process_folder_btn",
    )
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)

    layout.setContentsMargins(10, 8, 10, 8)
    layout.setHorizontalSpacing(8)
    layout.setVerticalSpacing(6)
    for col in range(2):
        layout.setColumnStretch(col, 1)

    widgets["screen_update"].setText("Preview")
    widgets["overwrite_var"].setText("Auto-Fit")
    widgets["obb_checkbox"].setText("OBB")
    widgets["segmentation_checkbox"].setText("Seg")
    widgets["noise_remove_checkbox"].setText("Clean BG")
    widgets["shadow_var"].setText("Shadow / Highlight")
    widgets["shadow_var"].setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
    widgets["shadow_var"].setToolTip(
        "Create one shadow, filled highlight, textured highlight, or outline variant per selected image, using existing SAM3 masks."
    )
    widgets["copy_paste_checkbox"].setText("Copy/Paste")
    widgets["object_detail_checkbox"].setText("Object Detail")
    widgets["object_detail_checkbox"].setToolTip(
        "Create labeled-object variants with subtle natural-edge sharpening and local contrast; pixels outside each SAM3 mask stay unchanged."
    )
    widgets["create_negatives_checkbox"].setText("Negatives")
    widgets["create_negatives_checkbox"].setToolTip(
        "Create object-removed empty-label negative images from the current dataset."
    )
    widgets["process_folder_btn"].setText("Run SAM3")
    widgets["process_folder_btn"].setMinimumHeight(28)
    widgets["process_folder_btn"].setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    compact_order = (
        "screen_update",
        "overwrite_var",
        "obb_checkbox",
        "segmentation_checkbox",
        "noise_remove_checkbox",
        "copy_paste_checkbox",
        "object_detail_checkbox",
        "create_negatives_checkbox",
        "sam_object_clarity_checkbox",
    )
    for index, name in enumerate(compact_order):
        layout.addWidget(widgets[name], index // 2, index % 2)

    appearance_row = self._row_layout()
    appearance_row.setObjectName("sam_appearance_controls")
    appearance_row.addWidget(widgets["shadow_var"])
    self.appearance_style_combo = QtWidgets.QComboBox(frame)
    self.appearance_style_combo.setObjectName("appearance_style_combo")
    for title, value in (
        ("Shadows", "shadow"),
        ("Solid Highlight", "solid"),
        ("Textured Highlight", "textured"),
        ("Outline", "outline"),
        ("Mixed", "mixed"),
    ):
        self.appearance_style_combo.addItem(title, value)
    self.appearance_style_combo.setMinimumWidth(0)
    self.appearance_style_combo.setMinimumContentsLength(4)
    self.appearance_style_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    self.appearance_style_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    self.appearance_style_combo.setToolTip(
        "Mixed chooses one appearance per image, not an extra copy for every style. Shadows do not use the highlight color."
    )
    appearance_row.addWidget(self.appearance_style_combo, 1)
    self.appearance_color_button = QtWidgets.QPushButton(frame)
    self.appearance_color_button.setObjectName("appearance_color_button")
    self.appearance_color_button.setFixedSize(40, 28)
    self.appearance_color_button.setAccessibleName("Highlight color")
    appearance_row.addWidget(self.appearance_color_button)
    layout.addLayout(appearance_row, 5, 0, 1, 2)
    layout.addWidget(widgets["process_folder_btn"], 6, 0, 1, 2)

    self._restore_appearance_augmentation_controls()
    self.appearance_style_combo.currentIndexChanged.connect(self._on_appearance_style_changed)
    self.appearance_color_button.clicked.connect(self.pick_appearance_augmentation_color)
    widgets["shadow_var"].toggled.connect(self._sync_appearance_augmentation_controls)
    self._sam3_panel_compacted = True


def _appearance_augmentation_config(self):
    settings = getattr(self, "settings", {}) or {}
    combo = getattr(self, "appearance_style_combo", None)
    style = combo.currentData() if isinstance(combo, QtWidgets.QComboBox) else settings.get("augmentationAppearanceStyle", "shadow")
    if style not in ("shadow", "solid", "textured", "outline", "mixed"):
        style = "shadow"
    button = getattr(self, "appearance_color_button", None)
    value = button.property("appearanceColor") if isinstance(button, QtWidgets.QPushButton) else None
    color = QColor(str(value or settings.get("augmentationAppearanceColor", "#ff3030")))
    if not color.isValid():
        color = QColor("#ff3030")
    return style, color.name()


def _restore_appearance_augmentation_controls(self):
    combo = getattr(self, "appearance_style_combo", None)
    if not isinstance(combo, QtWidgets.QComboBox):
        return
    settings = getattr(self, "settings", {}) or {}
    index = combo.findData(settings.get("augmentationAppearanceStyle", "shadow"))
    with blocked_signals(combo):
        combo.setCurrentIndex(index if index >= 0 else 0)
    color = QColor(str(settings.get("augmentationAppearanceColor", "#ff3030")))
    if not color.isValid():
        color = QColor("#ff3030")
    self.appearance_color_button.setProperty("appearanceColor", color.name())
    self._sync_appearance_augmentation_controls()


def _sync_appearance_augmentation_controls(self, *_args):
    combo = getattr(self, "appearance_style_combo", None)
    if not isinstance(combo, QtWidgets.QComboBox):
        return
    enabled = self.shadow_var.isChecked()
    style, color = self._appearance_augmentation_config()
    self.shadow_var.ensurePolished()
    self.shadow_var.setMinimumWidth(self.shadow_var.sizeHint().width())
    popup_width = max(combo.fontMetrics().horizontalAdvance(combo.itemText(index)) for index in range(combo.count())) + 40
    combo.view().setMinimumWidth(popup_width)
    combo.setEnabled(enabled)
    button = self.appearance_color_button
    button.setEnabled(enabled and style != "shadow")
    button.setStyleSheet(
        f"QPushButton {{ background:{color}; border:1px solid #888888; border-radius:3px; padding:0px; }}"
            "QPushButton:disabled { background:#494949; border-color:#666666; }"
    )
    button.setToolTip(
        f"Pick highlight / outline color ({color}). Red is the default. This does not change annotation class colors."
    )


def _on_appearance_style_changed(self, *_args):
    self._sync_appearance_augmentation_controls()
    if not getattr(self, "_restoring_settings", False):
        self.settings["augmentationAppearanceStyle"] = self._appearance_augmentation_config()[0]
        self.queue_settings_save()


def pick_appearance_augmentation_color(self):
    current = QColor(self._appearance_augmentation_config()[1])
    color = QColorDialog.getColor(current, self, "Pick Highlight / Outline Color")
    if not color.isValid() or color.name() == current.name():
        return
    self.appearance_color_button.setProperty("appearanceColor", color.name())
    self.settings["augmentationAppearanceColor"] = color.name()
    self._sync_appearance_augmentation_controls()
    self.queue_settings_save()


def on_augmentation_percent_changed(self, value):
    self.augmentation_percent = max(1, min(100, int(value)))
    if hasattr(self, "settings"):
        self.settings["augmentationPercent"] = self.augmentation_percent
        self.saveSettings()


def on_check_button_click(self):
    self.overwrite = self.overwrite_var.isChecked()


def on_shadow_checkbox_click(self):
    self.shadow = self.shadow_var.isChecked()


def on_obb_checkbox_checked(self, state):
    self.is_obb_generation_enabled = state == QtCore.Qt.Checked
    logging.info("OBB generation %s.", "enabled" if self.is_obb_generation_enabled else "disabled")


def on_noise_remove_checked(self, state):
    self.is_noise_remove_enabled = state == QtCore.Qt.Checked
    logging.info("Noise reduction %s.", "enabled" if self.is_noise_remove_enabled else "disabled")


def on_copy_paste_checkbox_click(self, state: int):
    self.copy_paste_enabled = state == QtCore.Qt.Checked
    logging.info("Copy-paste augmentation %s.", "enabled" if self.copy_paste_enabled else "disabled")


def on_object_detail_checkbox_click(self, state: int):
    self.object_detail_enabled = state == QtCore.Qt.Checked
    logging.info("Object-detail augmentation %s.", "enabled" if self.object_detail_enabled else "disabled")


def on_create_negatives_checkbox_click(self, state: int):
    self.create_negatives_enabled = state == QtCore.Qt.Checked
    logging.info("Negative-image augmentation %s.", "enabled" if self.create_negatives_enabled else "disabled")


def on_segmentation_checkbox_checked(self, state):
    self.is_segmentation_enabled = state == QtCore.Qt.Checked
    logging.info("Segmentation mode %s.", "enabled" if self.is_segmentation_enabled else "disabled")


def generate_unique_name(self):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    return f"{current_time}_{unique_id}"


def process_directory(self):
    self.image_files = self._scan_images(self.batch_directory)


def _scan_images(self, directory):
    if not directory or not os.path.isdir(directory):
        return []
    return sorted(self.get_image_files(directory))


def ensure_rgb(self, image):
    """Legacy name: normalize gray/BGRA to OpenCV BGR, not RGB.

        SAM3's Ultralytics wrapper performs the BGR-to-RGB conversion itself.
        Keep this behavior for the existing mask and augmentation callers.
        """
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _read_image_cv(self, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        logging.error("Cannot read image: %s", image_path)
        return None
    return self.ensure_rgb(image)


def _safe_empty_cache(self):
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            logging.warning("torch.cuda.empty_cache() failed: %s", e)


def _safe_sync_cuda(self):
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as e:
            logging.warning("torch.cuda.synchronize() failed: %s", e)


def _resources_tight_for_sam_batch(self, min_ram_mb=768, min_vram_mb=768):
    try:
        mem = psutil.virtual_memory()
        if mem.available < min_ram_mb * 1024 * 1024:
            logging.warning(
                "Skipping SAM3 batch work because system RAM is low: %.0f MB free",
                mem.available / (1024 * 1024),
            )
            return True
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            if free_bytes < min_vram_mb * 1024 * 1024:
                logging.warning(
                    "Skipping SAM3 batch work because CUDA VRAM is low: %.0f MB free",
                    free_bytes / (1024 * 1024),
                )
                self._safe_empty_cache()
                return True
        except Exception:
            pass

    return False


def _prepare_dir(self, name):
    path = os.path.join(self.image_directory, name)
    os.makedirs(path, exist_ok=True)
    self.copy_classes_txt(path)
    return path


def draw_mask_and_bbox(self, image, mask, class_id=None, num_classes=1, original_bbox=None, obb_points=None, prioritize="largest"):
    refined_mask = self._clean_mask(mask, kernel_size=3)
    if refined_mask is None or not refined_mask.any():
        logging.warning("No foreground pixels in mask.")
        return image, (0, 0, 0, 0)

    qt_color = get_color(class_id, num_classes)
    bgr_color = (qt_color.blue(), qt_color.green(), qt_color.red())

    overlay = image.copy()
    alpha = 0.4
    overlay[refined_mask > 0] = (
        overlay[refined_mask > 0] * (1 - alpha) + np.array(bgr_color, dtype=np.uint8) * alpha
    ).astype(np.uint8)
    image_with_boxes = overlay

    bbox = self._mask_bbox(refined_mask)
    if bbox is None:
        return image, (0, 0, 0, 0)

    x_min, y_min, x_max, y_max = bbox

    if getattr(self, "overwrite", False) and not self.is_obb_generation_enabled:
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), bgr_color, 2)

    if original_bbox and not getattr(self, "overwrite", False):
        ox1, oy1, ox2, oy2 = original_bbox
        cv2.rectangle(image_with_boxes, (ox1, oy1), (ox2, oy2), bgr_color, 2)

    if self.is_obb_generation_enabled:
        pts = None
        if obb_points and len(obb_points) == 8:
            img_height, img_width = image.shape[:2]
            pts = np.array([
                [int(obb_points[0] * img_width), int(obb_points[1] * img_height)],
                [int(obb_points[2] * img_width), int(obb_points[3] * img_height)],
                [int(obb_points[4] * img_width), int(obb_points[5] * img_height)],
                [int(obb_points[6] * img_width), int(obb_points[7] * img_height)],
            ], dtype=np.int32)
        else:
            contour = self._contour_from_mask(refined_mask)
            if contour is not None:
                rect = cv2.minAreaRect(contour)
                pts = cv2.boxPoints(rect).astype(np.int32)

        if pts is not None:
            cv2.polylines(image_with_boxes, [pts], isClosed=True, color=bgr_color, thickness=2)

    return image_with_boxes, (x_min, y_min, x_max, y_max)


def show_image(self, img, source_path=None):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img_rgb.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(img_rgb.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap.fromImage(qImg)
    scene = QtWidgets.QGraphicsScene(self)
    scene.setProperty("darkfusion_source_image", os.path.abspath(source_path) if source_path else "")
    scene.addPixmap(pixmap)
    old_scene = self.screen_view.scene()
    if old_scene is not None and old_scene is not scene:
        self.screen_view.setScene(None)
        try:
            old_scene.clear()
            old_scene.deleteLater()
        except RuntimeError:
            pass

    self.screen_view.setScene(scene)
    # This scene contains rendered pixels, not editable annotation items.
    # Reuse the editor's read-only gate so navigation cannot serialize an
    # empty augmentation preview over the source labels.
    self.annotation_scene_active = False
    self.screen_view.setSceneRect(QtCore.QRectF(pixmap.rect()))
    self.screen_view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


def _note_sam3_output_failure(self, path, error):
    failures = getattr(self, "_sam3_output_failures", None)
    if failures is None:
        failures = self._sam3_output_failures = []
    failures.append(f"{path}: {error}")
    logger.error("SAM3 output was not saved: %s: %s", path, error)


def _save_sam3_output_pair(self, image_path, image, label_path, labels, source_image_path=None):
    """Share checked pair saving across SAM3 conversions and augmentations."""
    try:
        if source_image_path is not None:
            image = source_image_path
            image_writer = lambda target, source: bool(shutil.copy2(source, target))
        else:
            image_writer = save_cv_image
        return save_extraction_outputs(
            image_path, image, image_writer, label_path=label_path, labels=labels,
            label_writer=lambda path, values, mode="w": self._write_label_lines(path, values),
            preserve_existing_labels=False,
        )
    except Exception as error:
        self._note_sam3_output_failure(image_path, error)
        return False


def _save_sam3_label_updates(self, label_path, labels):
    if self._write_label_lines(label_path, labels):
        return True
    self._note_sam3_output_failure(label_path, "The annotation file could not be replaced.")
    return False


def _show_sam3_batch_result(self, message):
    failures = getattr(self, "_sam3_output_failures", [])
    if failures:
        QMessageBox.warning(self, "SAM3 Output Errors",
            f"{message}\n{len(failures)} output operation(s) failed. "
                "Check the log and any recovery paths below.\n" + "\n".join(failures[:5]))
    else:
        QMessageBox.information(self, "Information", message)


def apply_noise_reduction(self, image, masks_for_bboxes, image_file_path):
    if image is None or not isinstance(image, np.ndarray):
        logging.error("Invalid image passed to apply_noise_reduction.")
        return image

    original_image = image.copy()
    modified = False

    for mask, (x_min, y_min, x_max, y_max) in masks_for_bboxes:
        try:
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)
            if x_max <= x_min or y_max <= y_min:
                continue

            roi = image[y_min:y_max, x_min:x_max]
            mask_crop = (mask[y_min:y_max, x_min:x_max] > 0).astype(np.uint8) * 255
            if mask_crop.shape[:2] != roi.shape[:2]:
                continue

            inverted_mask = cv2.bitwise_not(mask_crop)
            blurred_roi = cv2.GaussianBlur(roi, (3, 3), 0)
            fg = cv2.bitwise_and(roi, roi, mask=mask_crop)
            bg = cv2.bitwise_and(blurred_roi, blurred_roi, mask=inverted_mask)
            image[y_min:y_max, x_min:x_max] = cv2.add(fg, bg)
            modified = True
        except Exception as e:
            logging.error("Noise reduction failed on region %s: %s", (x_min, y_min, x_max, y_max), e, exc_info=True)

    ext = self.augmentation_output_extension_for_image(image_file_path)
    if modified and not np.array_equal(image, original_image):
        yolo_file_path = self.get_label_file_safe(image_file_path)
        label_lines = self._valid_annotation_lines_from_file(yolo_file_path)
        if not label_lines:
            logging.warning("Skipping noise-reduced image with no valid annotations: %s", image_file_path)
            return image

        nr_folder = os.path.join(os.path.dirname(image_file_path), "noise_reduced")
        os.makedirs(nr_folder, exist_ok=True)
        nr_image_name = f"nr_{self.generate_unique_name()}{ext}"
        nr_image_path = os.path.join(nr_folder, nr_image_name)
        nr_txt_path = os.path.join(nr_folder, os.path.splitext(nr_image_name)[0] + ".txt")
        if self._save_sam3_output_pair(nr_image_path, image, nr_txt_path, label_lines):
            logging.info("Saved noise-reduced image and labels: %s", nr_image_path)

    return image


def _render_object_appearance(self, image, masks_for_bboxes, style, color, rng=None):
    """Change appearance only; preserve visible mask support and all geometry."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
        return image
    rng = rng or random
    if style == "mixed":
        # One coherent appearance for the image, not one new file per mode.
        style = rng.choices(("shadow", "solid", "textured", "outline"), weights=(4, 2, 2, 2), k=1)[0]
    if style not in ("shadow", "solid", "textured", "outline"):
        style = "shadow"

    height, width = image.shape[:2]
    union_mask = np.zeros((height, width), np.uint8)
    object_widths = []
    for entry in masks_for_bboxes or ():
        try:
            mask, _bounds = entry
            raw_mask = np.asarray(mask)
            if not np.isfinite(raw_mask).all():
                continue
            mask = self._normalize_sam_mask_to_image(raw_mask, width, height)
            if mask is None:
                continue
            mask_u8 = (mask > 0).astype(np.uint8)
            _x, _y, mask_width, mask_height = cv2.boundingRect(mask_u8)
            object_widths.append(min(mask_width, mask_height))
            union_mask |= mask_u8
        except (TypeError, ValueError, cv2.error):
            logging.warning("Skipping an invalid SAM3 appearance mask.")
    if not object_widths:
        return image

    # Crop pixel work to mask bounds. Combining masks avoids double-darkening
    # overlaps and includes the final row/column (legacy mask bboxes are inclusive).
    x, y, crop_width, crop_height = cv2.boundingRect(union_mask)
    support = union_mask[y:y + crop_height, x:x + crop_width]
    region = image[y:y + crop_height, x:x + crop_width]
    source = region.astype(np.float32)
    mask_alpha = support.astype(np.float32)
    padded = np.pad(mask_alpha, 1)
    alpha = cv2.GaussianBlur(padded, (3, 3), 0.6)[1:-1, 1:-1] * mask_alpha

    qcolor = QColor(str(color))
    if not qcolor.isValid():
        qcolor = QColor("#ff3030")
    tint = np.array((qcolor.blue(), qcolor.green(), qcolor.red()), np.float32)

    if style == "shadow":
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)[:, :, None]
        retained_color = rng.uniform(0.35, 0.95)
        target = gray + (source - gray) * retained_color
        darkness = rng.uniform(0.35, 0.75)
        if rng.random() < 0.5:
            # A soft directional shadow, not an opaque shape over the box.
            angle = rng.uniform(0.0, np.pi * 2.0)
            yy, xx = np.ogrid[-1.0:1.0:complex(0, crop_height), -1.0:1.0:complex(0, crop_width)]
            field = xx * np.cos(angle) + yy * np.sin(angle)
            shade = np.clip((field - rng.uniform(-0.3, 0.3)) * 1.8 + 0.5, 0.0, 1.0)
            target *= (1.0 - (1.0 - darkness) * shade[:, :, None])
        else:
            target *= darkness
    elif style == "solid":
        # A filled scan/threat silhouette: no armor texture in the interior.
        target = tint * rng.uniform(0.85, 1.0)
    elif style == "textured":
        lightness = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        shaded_tint = tint * (0.35 + 0.65 * lightness[:, :, None])
        strength = rng.uniform(0.70, 0.92)
        target = source * (1.0 - strength) + shaded_tint * strength
    else:
        # Inward-only, scale-aware outline: no halo outside the visible
        # object, no larger labels, and no inferred hidden body parts.
        thickness = max(1, min(3, int(round(float(np.percentile(object_widths, 25)) * 0.012))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness * 2 + 1, thickness * 2 + 1))
        inner = cv2.erode(support, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        boundary = (support - inner).astype(np.float32)
        alpha = cv2.GaussianBlur(np.pad(boundary, 1), (3, 3), 0.5)[1:-1, 1:-1]
        alpha *= mask_alpha * rng.uniform(0.75, 1.0)
        target = tint

    result = image.copy()
    blended = np.clip(source + (target - source) * alpha[:, :, None], 0, 255).astype(np.uint8)
    result[y:y + crop_height, x:x + crop_width] = np.where(support[:, :, None] > 0, blended, region)
    return result


def create_shadow_image(self, image, masks_for_bboxes, image_file_path):
    """Keep both legacy callers and checked image/label saving in one path."""
    style, color = self._appearance_augmentation_config()
    appearance_image = self._render_object_appearance(image, masks_for_bboxes, style, color)
    if image is None or np.array_equal(appearance_image, image):
        return image
    label_lines = self._valid_annotation_lines_from_file(self.get_label_file_safe(image_file_path))
    if not label_lines:
        logging.warning("Skipping appearance image with no valid annotations: %s", image_file_path)
        return image

    # Retain the existing output location and extension/percentage workflow.
    folder = os.path.join(os.path.dirname(image_file_path), "shadow")
    os.makedirs(folder, exist_ok=True)
    ext = self.augmentation_output_extension_for_image(image_file_path)
    name = f"shadow_{self.generate_unique_name()}"
    target_image = os.path.join(folder, name + ext)
    target_label = os.path.join(folder, name + ".txt")
    if self._save_sam3_output_pair(target_image, appearance_image, target_label, label_lines):
        logging.info("Saved %s appearance image and unchanged labels: %s", style, target_image)
        return appearance_image
    return image


def _object_clarity_augmentation_changed(self, checked, sam=False):
    key = "samObjectClarity" if sam else "augmentationObjectClarity"
    self.settings[key] = bool(checked)
    self.queue_settings_save(delay_ms=180)
    if not sam:
        self.queue_augmentation_preview_update()


def _object_clarity_augmentation_selected(self, sam=False):
    name = "sam_object_clarity_checkbox" if sam else "object_clarity_augmentation_checkbox"
    widget = getattr(self, name, None)
    if isinstance(widget, QtWidgets.QCheckBox):
        return widget.isChecked()
    key = "samObjectClarity" if sam else "augmentationObjectClarity"
    return bool(getattr(self, "settings", {}).get(key, False))


def _object_clarity_annotation_regions(self, image, labels, polygon_parse_preference=None):
    height, width = image.shape[:2]
    preference = polygon_parse_preference or self._polygon_label_parse_preference()
    records = []
    for line in labels:
        annotation = BoundingBox.from_str(line, preferred_polygon_type=preference)
        if annotation is None:
            continue
        polygon = annotation.to_polygon(width, height)
        if polygon.is_empty or not polygon.is_valid:
            continue
        left, top, right, bottom = polygon.bounds
        x1, y1 = max(0, math.floor(left)), max(0, math.floor(top))
        x2, y2 = min(width, math.ceil(right)), min(height, math.ceil(bottom))
        if x1 >= x2 or y1 >= y2:
            continue
        mask = None
        if annotation.segmentation or annotation.obb:
            mask = np.zeros((y2-y1, x2-x1), np.uint8)
            vertices = np.rint(np.asarray(polygon.exterior.coords) - (x1, y1)).astype(np.int32)
            cv2.fillPoly(mask, [vertices], 1)
            for hole in polygon.interiors:
                vertices = np.rint(np.asarray(hole.coords) - (x1, y1)).astype(np.int32)
                cv2.fillPoly(mask, [vertices], 0)
        records.append(((x1, y1, x2, y2), mask))
    return records


def _render_object_clarity_augmentation(self, image, records):
    """Run only the new neural effect on a worker; keep cancellation/UI live."""
    if not gpu_model_allowed(self, "clarity_augmentation"):
        return None
    if getattr(self, "_clarity_augmentation_shutting_down", False):
        return None
    if not records:
        return image.copy()
    if getattr(self, "_clarity_augmentation_busy", False):
        return None
    from darkfusion_clarity_augmentation import ObjectClarityAugmenter
    executor = getattr(self, "_clarity_augmentation_executor", None)
    if executor is None:
        executor = self._clarity_augmentation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clarity-augmentation")
        self._clarity_augmentation_renderer = ObjectClarityAugmenter(os.path.join(APP_DIR, "models", "object_clarity"))
    cancel = self._clarity_augmentation_cancel = threading.Event()
    self._clarity_augmentation_busy = True
    job = getattr(self, "_augmentation_job", None)
    sam_active = bool(getattr(self, "_sam3_batch_active", False))
    controls = []
    for name, active in (("crop_button", job is not None), ("process_folder_btn", sam_active)):
        widget = getattr(self, name, None)
        if isinstance(widget, QtWidgets.QWidget) and not active:
            controls.append((widget, widget.isEnabled()))
            widget.setEnabled(False)
    future = executor.submit(self._clarity_augmentation_renderer, image.copy(),
                             [(bounds, mask.copy() if mask is not None else None) for bounds, mask in records], cancel)
    loop, timer = QtCore.QEventLoop(), QTimer()

    def poll():
        if ((job is not None and job.get("cancel_requested"))
                or (sam_active and self._sam3_should_stop())
                or getattr(self, "_clarity_augmentation_shutting_down", False)):
            cancel.set()
        if future.done() or getattr(self, "_clarity_augmentation_shutting_down", False):
            loop.quit()

    timer.timeout.connect(poll)
    timer.start(15)
    try:
        if not future.done():
            loop.exec_()
        if not future.done():
            cancel.set()
            return None
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            return None
    finally:
        timer.stop()
        self._clarity_augmentation_busy = False
        for widget, enabled in controls:
            if not sip.isdeleted(widget):
                widget.setEnabled(enabled)
        if getattr(self, "_clarity_augmentation_preview_pending", False):
            self._clarity_augmentation_preview_pending = False
            if not getattr(self, "_clarity_augmentation_shutting_down", False):
                self.queue_augmentation_preview_update()


def create_object_clarity_image(self, image, masks_for_bboxes, image_file_path):
    """Save one original-sized, mask-only neural clarity variant with labels."""
    if image is None or image.size == 0:
        return image
    label_file = self.get_label_file_safe(image_file_path)
    labels = self._valid_annotation_lines_from_file(label_file)
    if not labels:
        return image
    height, width = image.shape[:2]
    records = []
    for mask, _bounds in masks_for_bboxes:
        normalized = self._normalize_sam_mask_to_image(mask, width, height)
        if normalized is None or not normalized.any():
            continue
        x, y, w, h = cv2.boundingRect((normalized > 0).astype(np.uint8))
        records.append(((x, y, x+w, y+h), normalized[y:y+h, x:x+w]))
    if not records:
        return image
    try:
        result = self._render_object_clarity_augmentation(image, records)
    except Exception as error:
        self._note_sam3_output_failure(image_file_path, error)
        return image
    if result is None or np.array_equal(result, image):
        return image
    folder = os.path.join(os.path.dirname(image_file_path), "object_clarity")
    os.makedirs(folder, exist_ok=True)
    stem = "clarity_" + self.generate_unique_name()
    target = os.path.join(folder, stem + self.augmentation_output_extension_for_image(image_file_path))
    if self._save_sam3_output_pair(target, result, os.path.join(folder, stem + ".txt"), labels):
        return result
    return image


def create_object_detail_image(self, image, masks_for_bboxes, image_file_path):
    """Enhance existing object detail without drawing a synthetic outline."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        logging.error("Invalid image passed to create_object_detail_image.")
        return image

    img_height, img_width = image.shape[:2]
    union_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for mask, _bbox in masks_for_bboxes:
        normalized = self._normalize_sam_mask_to_image(mask, img_width, img_height)
        if normalized is not None:
            union_mask[normalized > 0] = 255

    object_pixels = int(cv2.countNonZero(union_mask))
    if object_pixels < 64:
        logging.info("Skipping object-detail image with no usable object mask: %s", image_file_path)
        return image

    # Keep all blending inward from the true mask boundary. The final multiply
    # makes it impossible for Gaussian feathering to alter background pixels.
    object_scale = max(1.0, np.sqrt(float(object_pixels)))
    inset_px = max(1, min(4, int(round(object_scale / 160.0))))
    kernel_size = inset_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inner_mask = cv2.erode(union_mask, kernel, iterations=1)
    if cv2.countNonZero(inner_mask) < 32:
        inner_mask = union_mask.copy()

    inner_alpha = cv2.GaussianBlur(inner_mask.astype(np.float32) / 255.0, (0, 0), 0.8)
    inner_alpha *= (union_mask > 0).astype(np.float32)

    mode = random.choices(
        ("natural edges", "local contrast", "mixed"),
        weights=(0.45, 0.35, 0.20),
        k=1,
    )[0]
    source = image.astype(np.float32)
    enhanced = source.copy()

    if mode in ("local contrast", "mixed"):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clip_limit = random.uniform(1.2, 1.8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        contrast_lab = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
        contrast_image = cv2.cvtColor(contrast_lab, cv2.COLOR_LAB2BGR).astype(np.float32)
        contrast_strength = random.uniform(0.10, 0.22)
        enhanced += (contrast_image - source) * contrast_strength

    if mode in ("natural edges", "mixed"):
        sigma = random.uniform(0.8, 1.35)
        blurred = cv2.GaussianBlur(image, (0, 0), sigma).astype(np.float32)
        unsharp = np.clip(source + (source - blurred) * random.uniform(0.20, 0.42), 0, 255)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        gradient /= max(24.0, float(np.percentile(gradient[union_mask > 0], 90)))
        gradient = np.clip(gradient, 0.0, 1.0)

        boundary = cv2.subtract(union_mask, inner_mask).astype(np.float32) / 255.0
        boundary = cv2.GaussianBlur(boundary, (0, 0), 0.7)
        edge_alpha = np.maximum(gradient * 0.35, boundary * gradient)
        edge_alpha *= (union_mask > 0).astype(np.float32)
        enhanced += (unsharp - source) * edge_alpha[:, :, None]

    enhanced = np.clip(enhanced, 0, 255)
    detail_image = np.clip(
        source + (enhanced - source) * inner_alpha[:, :, None],
        0,
        255,
    ).astype(np.uint8)

    if np.array_equal(detail_image, image):
        return image

    yolo_file_path = self.get_label_file_safe(image_file_path)
    label_lines = self._valid_annotation_lines_from_file(yolo_file_path)
    if not label_lines:
        logging.warning("Skipping object-detail image with no valid annotations: %s", image_file_path)
        return image

    detail_folder = os.path.join(os.path.dirname(image_file_path), "object_detail")
    os.makedirs(detail_folder, exist_ok=True)
    ext = self.augmentation_output_extension_for_image(image_file_path)
    detail_image_name = f"detail_{self.generate_unique_name()}{ext}"
    detail_image_path = os.path.join(detail_folder, detail_image_name)
    detail_txt_path = os.path.join(detail_folder, os.path.splitext(detail_image_name)[0] + ".txt")
    if self._save_sam3_output_pair(detail_image_path, detail_image, detail_txt_path, label_lines):
        logging.info("Saved %s object-detail image and unchanged labels: %s", mode, detail_image_path)
        return detail_image
    return image


def validate_segmentation_quality(self, mask, original_bbox, min_solidity=0.45, min_contiguity=0.6):
    """Check if segmentation is reliable enough for copy-paste.
        
        Returns (is_valid, reason) where is_valid is bool and reason is diagnostic string.
        
        Checks:
        - Mask solidity: % of bbox filled by mask (should be 40-80%)
        - Contiguity: largest component vs total pixels (fragmented = bad)
        - Coverage consistency: bbox from mask vs original bbox
        """
    if mask is None or not mask.any():
        return False, "No mask pixels"
    
    try:
        # Calculate mask solidity (% of bbox filled)
        x_min, y_min, x_max, y_max = [int(v) for v in original_bbox]
        bbox_area = max(1, (x_max - x_min) * (y_max - y_min))
        mask_pixels = int(mask.sum())
        solidity = float(mask_pixels) / float(bbox_area)
        
        # Too empty or too full both indicate problems
        if solidity < min_solidity:
            return False, f"Solidity {solidity:.2f} < {min_solidity}"
        if solidity > 0.95:  # Nearly perfect fill suggests false positive
            return False, f"Solidity {solidity:.2f} too high (likely false positive)"
        
        # Check contiguity (is mask one blob or fragmented?)
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        num_labels, labels = cv2.connectedComponents(mask_uint8)
        if num_labels < 2:
            return False, "No connected components found"
        
        # Get size of largest component
        component_sizes = np.bincount(labels.flat)
        largest_component = component_sizes[1:].max() if len(component_sizes) > 1 else 0
        contiguity_ratio = float(largest_component) / float(mask_pixels) if mask_pixels > 0 else 0.0
        
        # Fragmented masks are problematic
        if contiguity_ratio < min_contiguity:
            return False, f"Contiguity {contiguity_ratio:.2f} < {min_contiguity} ({num_labels} components)"
        
        # Check if mask bbox is reasonable compared to original
        mask_coords = np.where(mask > 0)
        if len(mask_coords[0]) == 0:
            return False, "No mask coordinates"
        
        mask_y_min, mask_y_max = mask_coords[0].min(), mask_coords[0].max()
        mask_x_min, mask_x_max = mask_coords[1].min(), mask_coords[1].max()
        mask_bbox_area = max(1, (mask_x_max - mask_x_min) * (mask_y_max - mask_y_min))
        
        # Mask bbox should be reasonably close to original (within 50-200%)
        bbox_ratio = float(mask_bbox_area) / float(bbox_area)
        if bbox_ratio < 0.5 or bbox_ratio > 2.0:
            return False, f"Bbox ratio {bbox_ratio:.2f} out of range"
        
        return True, f"Valid (solidity={solidity:.2f}, contiguity={contiguity_ratio:.2f})"
    
    except Exception as e:
        return False, f"Validation error: {e}"


def extract_segmented_object(self, image, mask, x_min, y_min, x_max, y_max):
    if image is None or mask is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None

    img_h, img_w = image.shape[:2]
    mask = self._normalize_sam_mask_to_image(mask, img_w, img_h)
    if mask is None:
        return None

    x_min = max(0, min(int(x_min), img_w - 1))
    y_min = max(0, min(int(y_min), img_h - 1))
    x_max = max(0, min(int(x_max), img_w))
    y_max = max(0, min(int(y_max), img_h))
    if x_max <= x_min or y_max <= y_min:
        return None

    cropped_object = image[y_min:y_max, x_min:x_max]
    mask_cropped = (mask[y_min:y_max, x_min:x_max] > 0)
    if cropped_object.size == 0 or mask_cropped.size == 0 or not mask_cropped.any():
        return None
    if mask_cropped.shape[:2] != cropped_object.shape[:2]:
        return None

    if cropped_object.ndim == 2:
        cropped_object = cv2.cvtColor(cropped_object, cv2.COLOR_GRAY2BGR)
    elif cropped_object.ndim == 3 and cropped_object.shape[2] == 4:
        cropped_object = cv2.cvtColor(cropped_object, cv2.COLOR_BGRA2BGR)

    segmented_object = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2BGRA)
    segmented_object[:, :, 3] = (mask_cropped.astype(np.uint8) * 255)
    return segmented_object


def overlay_object(self, target_image, segment, x, y):
    if target_image is None or segment is None:
        return target_image
    if target_image.ndim != 3 or target_image.shape[2] not in (3, 4):
        return target_image
    if segment.ndim != 3 or segment.shape[2] not in (3, 4):
        return target_image

    x = int(x)
    y = int(y)
    if x < 0 or y < 0 or x >= target_image.shape[1] or y >= target_image.shape[0]:
        return target_image

    if target_image.shape[2] == 4:
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGRA2BGR)

    if segment.shape[2] == 4:
        alpha = segment[:, :, 3].astype(np.float32) / 255.0
        segment = cv2.cvtColor(segment, cv2.COLOR_BGRA2BGR)
    else:
        alpha = np.any(segment > 0, axis=2).astype(np.float32)

    obj_h, obj_w = segment.shape[:2]
    y_end, x_end = y + obj_h, x + obj_w
    if y_end > target_image.shape[0] or x_end > target_image.shape[1]:
        obj_h = min(obj_h, target_image.shape[0] - y)
        obj_w = min(obj_w, target_image.shape[1] - x)
        if obj_h <= 0 or obj_w <= 0:
            return target_image
        segment = segment[:obj_h, :obj_w]
        alpha = alpha[:obj_h, :obj_w]
        y_end, x_end = y + obj_h, x + obj_w

    region = target_image[y:y_end, x:x_end]
    alpha_3 = np.repeat(alpha[:, :, None], 3, axis=2)
    blended = (alpha_3 * segment + (1.0 - alpha_3) * region).astype(np.uint8)
    target_image[y:y_end, x:x_end] = blended
    return target_image


def check_overlap(self, new_bbox, existing_bboxes, min_spacing=10):
    x_min1, y_min1, x_max1, y_max1 = new_bbox
    for x_min2, y_min2, x_max2, y_max2 in existing_bboxes:
        x_min2 -= min_spacing
        y_min2 -= min_spacing
        x_max2 += min_spacing
        y_max2 += min_spacing
        if not (x_max1 < x_min2 or x_min1 > x_max2 or y_max1 < y_min2 or y_min1 > y_max2):
            return True
    return False


def _transform_pasted_point(
    self,
    norm_x,
    norm_y,
    source_bbox,
    source_width,
    source_height,
    target_bbox,
    target_width,
    target_height,
    hide_outside=False,
):
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
        return None

    src_x1, src_y1, src_x2, src_y2 = [float(v) for v in source_bbox]
    dst_x1, dst_y1, dst_x2, dst_y2 = [float(v) for v in target_bbox]
    src_w = max(1e-6, src_x2 - src_x1)
    src_h = max(1e-6, src_y2 - src_y1)
    dst_w = max(1e-6, dst_x2 - dst_x1)
    dst_h = max(1e-6, dst_y2 - dst_y1)

    src_x = float(norm_x) * float(source_width)
    src_y = float(norm_y) * float(source_height)
    if not np.isfinite([src_x, src_y]).all():
        return None

    rel_x = (src_x - src_x1) / src_w
    rel_y = (src_y - src_y1) / src_h
    dst_x = dst_x1 + (rel_x * dst_w)
    dst_y = dst_y1 + (rel_y * dst_h)

    if hide_outside and (
        rel_x < 0.0
        or rel_x > 1.0
        or rel_y < 0.0
        or rel_y > 1.0
        or dst_x < 0.0
        or dst_x > target_width
        or dst_y < 0.0
        or dst_y > target_height
    ):
        return None

    dst_x = float(np.clip(dst_x, 0.0, float(target_width)))
    dst_y = float(np.clip(dst_y, 0.0, float(target_height)))
    return dst_x / float(target_width), dst_y / float(target_height)


def _transform_pasted_label(
    self,
    box_obj,
    source_bbox,
    source_width,
    source_height,
    target_bbox,
    target_width,
    target_height,
):
    x_center, y_center, box_w, box_h = self._xyxy_to_yolo_xywh(
        target_bbox,
        target_width,
        target_height,
    )
    label_type = self._label_type(box_obj)

    if label_type == "keypoints":
        pasted_keypoints = []
        for keypoint in getattr(box_obj, "keypoints", []) or []:
            if len(keypoint) < 3:
                pasted_keypoints.append((0.0, 0.0, 0))
                continue

            kx, ky, visibility = keypoint[:3]
            visibility = int(float(visibility))
            if visibility not in (1, 2):
                pasted_keypoints.append((0.0, 0.0, 0))
                continue

            pasted_point = self._transform_pasted_point(
                kx,
                ky,
                source_bbox,
                source_width,
                source_height,
                target_bbox,
                target_width,
                target_height,
                hide_outside=True,
            )
            if pasted_point is None:
                pasted_keypoints.append((0.0, 0.0, 0))
            else:
                pasted_keypoints.append((pasted_point[0], pasted_point[1], visibility))

        return BoundingBox(
            box_obj.class_id,
            x_center,
            y_center,
            box_w,
            box_h,
            keypoints=pasted_keypoints,
        )

    if label_type == "segmentation":
        transformed_segmentation = []
        segmentation = getattr(box_obj, "segmentation", []) or []
        for index in range(0, len(segmentation) - 1, 2):
            pasted_point = self._transform_pasted_point(
                segmentation[index],
                segmentation[index + 1],
                source_bbox,
                source_width,
                source_height,
                target_bbox,
                target_width,
                target_height,
            )
            if pasted_point is not None:
                transformed_segmentation.extend(pasted_point)

        if len(transformed_segmentation) >= 6:
            return BoundingBox(
                box_obj.class_id,
                x_center,
                y_center,
                box_w,
                box_h,
                segmentation=transformed_segmentation,
            )

    if label_type == "obb":
        transformed_obb = []
        obb = getattr(box_obj, "obb", []) or []
        for index in range(0, len(obb) - 1, 2):
            pasted_point = self._transform_pasted_point(
                obb[index],
                obb[index + 1],
                source_bbox,
                source_width,
                source_height,
                target_bbox,
                target_width,
                target_height,
            )
            if pasted_point is not None:
                transformed_obb.extend(pasted_point)

        if len(transformed_obb) == 8:
            return BoundingBox(
                box_obj.class_id,
                x_center,
                y_center,
                box_w,
                box_h,
                obb=transformed_obb,
            )

    return BoundingBox(box_obj.class_id, x_center, y_center, box_w, box_h)


def apply_copy_paste_augmentation(self, progress_offset=0, target_image_files=None, progress_total=None):
    dataset_dir = self.current_dataset_directory()
    if not dataset_dir:
        QMessageBox.critical(self, "Error", "Please select a valid image directory first.")
        return

    self.image_directory = dataset_dir
    source_image_files = list(getattr(self, "image_files", []) or [])
    if not source_image_files:
        source_image_files = self._scan_images(self.image_directory)

    source_image_files = [
        self.normalize_path(path)
        for path in source_image_files
        if path and os.path.isfile(path) and not self.is_placeholder_file(path)
    ]
    target_image_files = (
        list(target_image_files)
        if target_image_files is not None
        else list(source_image_files)
    )
    target_image_files = [
        self.normalize_path(path)
        for path in target_image_files
        if path and os.path.isfile(path) and not self.is_placeholder_file(path)
    ]

    if not source_image_files or not target_image_files:
        logging.info("Copy-paste augmentation skipped because no images are loaded.")
        return

    copy_paste_dir = os.path.join(self.image_directory, "copy_and_paste")
    os.makedirs(copy_paste_dir, exist_ok=True)

    min_size_threshold = 10
    class_to_boxes = {}
    polygon_parse_preference = self._polygon_label_parse_preference()

    for source_index, src_img_path in enumerate(source_image_files):
        if source_index % 32 == 0:
            QtWidgets.QApplication.processEvents()
        if self._sam3_should_stop():
            return

        src_txt_path = self.get_label_file_safe(src_img_path)
        if not src_txt_path:
            continue
        for ann in self._read_label_lines(src_txt_path):
            box = BoundingBox.from_str(
                ann,
                preferred_polygon_type=polygon_parse_preference,
            )
            if not box:
                continue
            if self._label_type(box) not in ("bbox", "keypoints", "segmentation", "obb"):
                continue
            class_to_boxes.setdefault(box.class_id, []).append((src_img_path, box, ann))

    if not class_to_boxes:
        logging.info("Copy-paste augmentation skipped because no bbox/keypoint/segmentation/OBB labels were found.")
        return

    segment_cache = OrderedDict()
    max_segment_cache = 48
    skipped_candidates = 0

    for idx, image_file in enumerate(target_image_files):
        try:
            QtWidgets.QApplication.processEvents()
            if self._sam3_should_stop():
                break

            image = self._read_image_cv(image_file)
            if image is None:
                continue

            tgt_h, tgt_w = image.shape[:2]
            if tgt_h < min_size_threshold * 2 or tgt_w < min_size_threshold * 2:
                logging.info("Skipping tiny copy-paste target image: %s", image_file)
                continue

            annotation_file = self.get_label_file_safe(image_file)
            annotations = self._read_label_lines(annotation_file)
            existing_classes = set()
            existing_bboxes = []

            for ann in annotations:
                bbox = BoundingBox.from_str(
                    ann,
                    preferred_polygon_type=polygon_parse_preference,
                )
                if bbox and self._label_type(bbox) in ("bbox", "keypoints", "segmentation", "obb"):
                    xyxy = self._label_seed_bbox(bbox, tgt_w, tgt_h)
                    if xyxy is not None:
                        existing_bboxes.append(xyxy)
                        existing_classes.add(bbox.class_id)

            objects_to_add = max(0, np.random.randint(3, 6) - len(existing_bboxes))
            modified = False

            for _ in range(objects_to_add):
                QtWidgets.QApplication.processEvents()
                if self._sam3_should_stop():
                    break

                try:
                    if existing_classes:
                        chosen_class = np.random.choice(list(existing_classes))
                    else:
                        chosen_class = np.random.choice(list(class_to_boxes.keys()))

                    candidate_boxes = class_to_boxes.get(chosen_class, [])
                    if not candidate_boxes:
                        continue

                    src_img_path, bbox, source_label = candidate_boxes[np.random.randint(len(candidate_boxes))]
                    src_img = self._read_image_cv(src_img_path)
                    if src_img is None:
                        continue

                    src_h, src_w = src_img.shape[:2]
                    xyxy = self._label_seed_bbox(bbox, src_w, src_h)
                    if xyxy is None:
                        continue

                    x_min, y_min, x_max, y_max = xyxy
                    cache_key = (src_img_path, source_label)

                    cached_segment = segment_cache.get(cache_key)
                    if cached_segment is not None:
                        segment = cached_segment.copy()
                        segment_cache.move_to_end(cache_key)
                    else:
                        if self._resources_tight_for_sam_batch():
                            self.stop_batch = True
                            break

                        mask = self.generate_segmentation_mask(src_img, xyxy)
                        QtWidgets.QApplication.processEvents()
                        if self._sam3_should_stop():
                            break
                        if mask is None:
                            continue

                        segment = self.extract_segmented_object(src_img, mask, x_min, y_min, x_max, y_max)
                        if segment is None or segment.size == 0:
                            continue

                        # Validate segmentation quality before using
                        is_valid, quality_reason = self.validate_segmentation_quality(mask, xyxy)
                        if not is_valid:
                            logging.info("Skipping copy-paste candidate (quality check failed: %s): %s", quality_reason, src_img_path)
                            continue

                        segment_cache[cache_key] = segment.copy()
                        if len(segment_cache) > max_segment_cache:
                            segment_cache.popitem(last=False)

                    obj_h, obj_w = segment.shape[:2]
                    scale = np.random.uniform(0.8, 1.3)
                    max_obj_h = max(1, tgt_h // 2)
                    max_obj_w = max(1, tgt_w // 2)
                    obj_h = min(max(1, int(obj_h * scale)), max_obj_h)
                    obj_w = min(max(1, int(obj_w * scale)), max_obj_w)
                    if obj_h < min_size_threshold or obj_w < min_size_threshold:
                        continue

                    segment = cv2.resize(segment, (obj_w, obj_h), interpolation=cv2.INTER_LINEAR)

                    placed = False
                    for _ in range(50):
                        if self._sam3_should_stop():
                            break
                        max_x = max(1, tgt_w - obj_w + 1)
                        max_y = max(1, tgt_h - obj_h + 1)
                        px = np.random.randint(0, max_x)
                        py = np.random.randint(0, max_y)
                        new_bbox = (px, py, px + obj_w, py + obj_h)
                        if not self.check_overlap(new_bbox, existing_bboxes, min_spacing=10):
                            placed = True
                            break

                    if not placed:
                        continue

                    existing_bboxes.append(new_bbox)
                    image = self.overlay_object(image, segment, px, py)

                    pasted_label = self._transform_pasted_label(
                        bbox,
                        xyxy,
                        src_w,
                        src_h,
                        new_bbox,
                        tgt_w,
                        tgt_h,
                    )
                    annotations.append(pasted_label.to_str())
                    modified = True

                except torch.cuda.OutOfMemoryError as e:
                    logging.critical("CUDA out of memory during copy-paste augmentation: %s", e, exc_info=True)
                    self._safe_empty_cache()
                    self.stop_batch = True
                    break
                except Exception as e:
                    skipped_candidates += 1
                    logging.error("Skipping copy-paste candidate for %s: %s", image_file, e, exc_info=True)

            if modified and not self._sam3_should_stop():
                output_ext = self.augmentation_output_extension_for_image(image_file)
                img_name = f"copy_and_paste_{os.path.splitext(os.path.basename(image_file))[0]}{output_ext}"
                out_img_path = os.path.join(copy_paste_dir, img_name)
                out_txt_path = os.path.join(copy_paste_dir, os.path.splitext(img_name)[0] + ".txt")
                if self._save_sam3_output_pair(out_img_path, image, out_txt_path, annotations):
                    logging.info("Saved augmented image and annotations: %s, %s", out_img_path, out_txt_path)

        except Exception as e:
            logging.error("Copy-paste augmentation failed for %s: %s", image_file, e, exc_info=True)
            self._note_sam3_output_failure(image_file, e)
        finally:
            if idx % 3 == 0:
                self._safe_empty_cache()
                QtWidgets.QApplication.processEvents()
            maximum = (
                progress_total
                if progress_total is not None
                else self.label_progress.maximum() if hasattr(self, "label_progress") else progress_offset + len(target_image_files)
            )
            value = progress_offset + idx + 1
            self.set_label_progress(
                0,
                max(1, maximum),
                value=min(value, max(1, maximum)),
                progress_format=f"Copy/Paste {min(value, max(1, maximum))}/{max(1, maximum)}",
            )

    if skipped_candidates:
        logging.info("Copy-paste skipped %s candidate object(s) due to recoverable errors.", skipped_candidates)


def _selected_sam_batch_actions(self):
    return {
        "copy_paste": bool(getattr(self, "copy_paste_enabled", False)),
        "object_detail": bool(getattr(self, "object_detail_enabled", False)),
        "object_clarity": self._object_clarity_augmentation_selected(sam=True),
        "negatives": bool(getattr(self, "create_negatives_enabled", False)),
        "segmentation": self.segmentation_checkbox.isChecked(),
        "noise": bool(getattr(self, "is_noise_remove_enabled", False)),
        "shadow": bool(getattr(self, "shadow", False)),
        "obb": bool(getattr(self, "is_obb_generation_enabled", False)),
        "auto_fit": bool(getattr(self, "overwrite", False)),
    }


def _set_sam3_batch_button_state(self, active=False, stopping=False):
    button = getattr(self, "process_folder_btn", None)
    if button is None:
        return

    if stopping:
        button.setText("Stopping SAM3...")
        button.setEnabled(False)
        button.setToolTip("Waiting for the current SAM3 inference or file operation to finish safely.")
    elif active:
        button.setText("Stop SAM3")
        button.setEnabled(True)
        button.setToolTip("Stop after the current SAM3 inference or file operation finishes.")
    else:
        button.setText("Run SAM3")
        button.setEnabled(True)
        button.setToolTip("Run only the selected SAM3 conversion or augmentation tools.")


def _sam3_should_stop(self):
    return bool(
        getattr(self, "_sam3_cancel_requested", False)
        or getattr(self, "stop_batch", False)
        or getattr(self, "stop_labeling", False)
    )


def request_stop_sam3_batch(self):
    if not getattr(self, "_sam3_batch_active", False):
        return

    self._sam3_cancel_requested = True
    self.stop_batch = True
    self.stop_labeling = True
    self._set_sam3_batch_button_state(active=True, stopping=True)

    worker = getattr(self, "processing_thread", None)
    try:
        if worker is not None and worker.isRunning() and hasattr(worker, "cancel"):
            worker.cancel()
    except RuntimeError:
        pass

    if hasattr(self, "label_progress"):
        self.label_progress.setFormat("Stopping SAM3 safely...")
    logger.info("SAM3 batch cancellation requested.")


def _finish_sam3_batch_control(self, stopped=None):
    if stopped is None:
        stopped = self._sam3_should_stop()
    self._sam3_batch_active = False
    if getattr(self, "_clarity_augmentation_preview_pending", False):
        self._clarity_augmentation_preview_pending = False
        self.queue_augmentation_preview_update()
    self._sam3_cancel_requested = False
    self.stop_batch = False
    self.stop_labeling = False
    self._set_sam3_batch_button_state(active=False)

    if stopped and hasattr(self, "label_progress"):
        self.label_progress.setFormat("SAM3 batch stopped")
    elif getattr(self, "_sam3_output_failures", []) and hasattr(self, "label_progress"):
        self.label_progress.setFormat("SAM3 batch finished with output errors")


def _on_sam3_negative_generation_finished(self, thread):
    if thread is not getattr(self, "processing_thread", None):
        return
    if not getattr(self, "_sam3_batch_active", False):
        return
    stopped = self._sam3_should_stop() or bool(getattr(thread, "_cancel_requested", False))
    failures = list(getattr(thread, "output_failures", []))
    if getattr(thread, "failure_message", ""):
        failures.insert(0, thread.failure_message)
    if failures and not stopped:
        self._sam3_output_failures.extend(failures)
    self._finish_sam3_batch_control(stopped=stopped)
    if failures and not stopped:
        logger.error("SAM3 negative-image generation failed: %s", failures[0])
        QMessageBox.critical(self, "Negative Generation", "Negative generation finished with errors:\n" + "\n".join(failures[:3]))
    else:
        logger.info("SAM3 negative-image generation %s; generated %s image(s).", "stopped" if stopped else "completed", thread.counter)
        if not stopped and hasattr(self, "label_progress"):
            self.label_progress.setFormat(f"Negatives done: {thread.counter} generated, {thread.rejected_count} rejected")


@gpu_work("SAM3 augmentation", keep=sam_batch_resources, until="_sam3_batch_active")
def process_batch(self):
    if getattr(self, "_sam3_batch_active", False):
        self.request_stop_sam3_batch()
        return

    self._sam3_batch_active = True
    self._sam3_output_failures = []
    self._sam3_cancel_requested = False
    self.stop_labeling = False
    self.stop_batch = False
    self._set_sam3_batch_button_state(active=True)

    keep_control_active = False
    try:
        actions = self._selected_sam_batch_actions()
        needs_prompt = any(actions[name] for name in (
            "segmentation", "obb", "auto_fit", "copy_paste", "noise", "shadow", "object_detail"
        )) or actions.get("object_clarity", False)
        warmup = getattr(self, "_sam3_startup_warmup_thread", None)
        warming = (getattr(self, "_sam3_startup_warmup_active", False)
                   or (warmup is not None and warmup.is_alive()))
        cold = getattr(self, "sam_model", None) is None
        if (needs_prompt and (cold or warming) and self.current_dataset_directory()
                and callable(getattr(self, "start_sam3_startup_warmup", None))):
            if not warming:
                self.start_sam3_startup_warmup(requested=True, prompt_only=True)
                warmup = getattr(self, "_sam3_startup_warmup_thread", None)
                warming = (getattr(self, "_sam3_startup_warmup_active", False)
                           or (warmup is not None and warmup.is_alive()))
            if warming:
                self.set_label_progress_busy("Preparing SAM3 in the background...")
                # Reuse the existing model owner and background loader. Qt
                # control access and the batch itself remain on the GUI thread.
                def resume_batch():
                    worker = getattr(self, "_sam3_startup_warmup_thread", None)
                    if (getattr(self, "_sam3_startup_warmup_active", False)
                            or (worker is not None and worker.is_alive())):
                        QTimer.singleShot(50, resume_batch)
                        return
                    deferred = False
                    try:
                        if not self._sam3_should_stop():
                            if getattr(self, "sam_model", None) is None:
                                QMessageBox.critical(self, "Error", "SAM3 model could not be loaded.")
                            else:
                                deferred = bool(self._run_sam3_batch())
                    except Exception as error:
                        logger.exception("SAM3 batch failed after background preparation")
                        QMessageBox.critical(self, "SAM3 Batch", str(error))
                    finally:
                        if not deferred:
                            self._finish_sam3_batch_control()

                QTimer.singleShot(50, resume_batch)
                keep_control_active = True
                return
        keep_control_active = bool(self._run_sam3_batch())
    finally:
        if not keep_control_active:
            self._finish_sam3_batch_control()


def _run_sam3_batch(self):
    dataset_dir = self.current_dataset_directory()
    if not dataset_dir:
        QMessageBox.critical(self, "Error", "Please select a valid image directory first.")
        logging.error("No image directory selected. Aborting batch process.")
        return

    self.image_directory = dataset_dir
    self.load_num_classes()
    # IMPORTANT: Scan images first BEFORE changing the displayed index
    # This ensures image_files is populated before img_index_number_changed(0) accesses it
    self.image_files = self._scan_images(self.image_directory)
    self.filtered_image_files = list(self.image_files)  # Keep filtered list in sync
    total_images = len(self.image_files)
    
    # NOW display the first image after lists are populated
    self.img_index_number_changed(0)

    if not self.image_files:
        QMessageBox.critical(self, "Error", "No images found in the selected directory.")
        return

    actions = self._selected_sam_batch_actions()
    conversion_actions = any(actions[name] for name in ("segmentation", "obb", "auto_fit"))
    sam_augmentation_actions = any(actions[name] for name in ("copy_paste", "noise", "shadow", "object_detail", "negatives")) or actions.get("object_clarity", False)
    main_loop_actions = conversion_actions or actions["noise"] or actions["shadow"] or actions["object_detail"] or actions.get("object_clarity", False)
    main_loop_uses_sam = main_loop_actions
    sam_required = actions["copy_paste"] or main_loop_uses_sam

    sam_augmentation_images = self.sampled_sam_augmentation_images(self.image_files)
    sam_augmentation_set = set(sam_augmentation_images)
    main_loop_image_files = self.image_files if conversion_actions else sam_augmentation_images
    copy_paste_image_files = sam_augmentation_images if actions["copy_paste"] else []
    negative_image_files = sam_augmentation_images if actions["negatives"] else []
    progress_total = (
        (len(copy_paste_image_files) if actions["copy_paste"] else 0)
        + (len(main_loop_image_files) if main_loop_actions else 0)
    )

    if not actions["copy_paste"] and not actions["negatives"] and not main_loop_actions:
        QMessageBox.information(
            self,
            "No SAM3 Action Selected",
            "Choose a SAM3 conversion or augmentation before running the batch tool.",
        )
        return

    if sam_required and self._get_sam_model() is None:
        QMessageBox.critical(self, "Error", "SAM3 model could not be loaded.")
        return

    self.set_label_progress(
        0,
        max(1, progress_total),
        value=0,
        progress_format=f"SAM3 batch 0/{max(1, progress_total)}",
    )

    dirs = {}
    if actions["copy_paste"]:
        dirs["copy"] = self._prepare_dir("copy_and_paste")
        self.apply_copy_paste_augmentation(
            progress_offset=0,
            target_image_files=copy_paste_image_files,
            progress_total=max(1, progress_total),
        )

    if not main_loop_actions:
        stopped = self._sam3_should_stop()
        negative_started = False
        if actions["negatives"] and not stopped:
            negative_started = self.start_create_negative_images(
                target_image_files=negative_image_files,
                preselected=True,
            )
            message = "Negative generation is running." if negative_started else "No valid negative images were queued."
        else:
            message = "Process was stopped!" if stopped else "Finished!"
        self._show_sam3_batch_result(message)
        if not stopped and not negative_started:
            self.set_label_progress(
                0,
                max(1, progress_total),
                value=progress_total,
                progress_format="SAM3 batch done",
            )
        self.img_index_number_changed(0)
        logging.info(
            "SAM3 batch %s with augmentation-only actions.",
            "stopped" if stopped else "completed",
        )
        return negative_started

    progress_offset = len(copy_paste_image_files) if actions["copy_paste"] else 0

    if actions["segmentation"]:
        dirs["seg"] = self._prepare_dir("Segmented")
    if actions["noise"]:
        dirs["noise"] = self._prepare_dir("noise_reduced")
    if actions["shadow"]:
        dirs["shadow"] = self._prepare_dir("shadow")
    if actions["object_detail"]:
        dirs["object_detail"] = self._prepare_dir("object_detail")
    if actions.get("object_clarity", False):
        dirs["object_clarity"] = self._prepare_dir("object_clarity")
    if actions["obb"]:
        dirs["obb"] = self._prepare_dir("OBB")

    if sam_augmentation_actions:
        logger.info(
            "SAM3 augmentation percent: %s%% (%s of %s images selected for Copy/Paste, Clean BG, Shadow / Highlight, Object Detail, Object Clarity, and Negatives).",
            self.selected_augmentation_percent(),
            len(sam_augmentation_images),
            total_images,
        )

    for idx, image_file in enumerate(main_loop_image_files):
        QtWidgets.QApplication.processEvents()
        if self._sam3_should_stop():
            break

        if main_loop_uses_sam and self._resources_tight_for_sam_batch():
            self.stop_batch = True
            break

        label_file = self.get_label_file_safe(image_file)
        if not label_file:
            logging.info("[SKIP] %s - no label file found", image_file)
            continue

        labels = self._valid_annotation_lines_from_file(label_file)
        if not labels:
            logging.info("[SKIP] %s - no valid annotations", image_file)
            continue

        image = self._read_image_cv(image_file)
        if image is None:
            continue

        img_height, img_width = image.shape[:2]
        visualized_image = image.copy()
        obb_lines = []
        seg_lines = []
        adjusted_yolo_lines = []
        masks_for_bboxes = []
        polygon_parse_preference = self._polygon_label_parse_preference()

        for line in labels:
            QtWidgets.QApplication.processEvents()
            if self._sam3_should_stop():
                break

            output_line = str(line or "").strip()
            try:
                box_obj = BoundingBox.from_str(
                    line,
                    preferred_polygon_type=polygon_parse_preference,
                )
                if not box_obj:
                    if self.overwrite and output_line:
                        adjusted_yolo_lines.append(output_line)
                    continue

                mask, original_bbox, label_type = self._process_label_to_mask(
                    box_obj,
                    image,
                    img_width,
                    img_height,
                    use_sam=main_loop_uses_sam,
                )
                if mask is None:
                    if self.overwrite and output_line:
                        adjusted_yolo_lines.append(output_line)
                    continue

                seg_line = self._seg_line_from_mask(box_obj.class_id, mask, img_width, img_height)
                if seg_line:
                    seg_lines.append(seg_line)

                obb_line = self._obb_line_from_mask(box_obj.class_id, mask, img_width, img_height)
                if obb_line:
                    obb_lines.append(obb_line)

                contour = self._contour_from_mask(mask)
                if contour is None:
                    if self.overwrite and output_line:
                        adjusted_yolo_lines.append(output_line)
                    continue

                color = get_color(box_obj.class_id, self.num_classes)
                bgr_color = (color.blue(), color.green(), color.red())
                cv2.drawContours(visualized_image, [contour], -1, bgr_color, thickness=cv2.FILLED)

                bbox_from_mask = self._mask_bbox(mask)
                if bbox_from_mask is not None:
                    masks_for_bboxes.append((mask, bbox_from_mask))

                    if self.is_obb_generation_enabled:
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect).astype(np.int32)
                        cv2.polylines(visualized_image, [box], isClosed=True, color=bgr_color, thickness=2)
                    elif self.overwrite:
                        x1, y1, x2, y2 = bbox_from_mask
                        cv2.rectangle(visualized_image, (x1, y1), (x2, y2), bgr_color, 2)

                    if self.overwrite:
                        updated_bbox = self._update_box_obj_from_mask(box_obj, mask, label_type, img_width, img_height)
                        if updated_bbox is not None:
                            output_line = box_obj.to_str(remove_confidence=True)

                if self.overwrite and output_line:
                    adjusted_yolo_lines.append(output_line)

            except Exception as e:
                logging.error("Failed processing box in %s: %s", image_file, e, exc_info=True)
                if self.overwrite and output_line:
                    adjusted_yolo_lines.append(output_line)

        if self._sam3_should_stop():
            break

        base_name = os.path.basename(label_file)

        if "obb" in dirs and obb_lines:
            self._save_sam3_output_pair(
                os.path.join(dirs["obb"], os.path.basename(image_file)), image,
                os.path.join(dirs["obb"], base_name), obb_lines, source_image_path=image_file)

        if "seg" in dirs and seg_lines:
            self._save_sam3_output_pair(
                os.path.join(dirs["seg"], os.path.basename(image_file)), image,
                os.path.join(dirs["seg"], base_name), seg_lines, source_image_path=image_file)

        if self.overwrite and adjusted_yolo_lines:
            self._save_sam3_label_updates(label_file, adjusted_yolo_lines)

        if self.screen_update.isChecked():
            self.show_image(visualized_image, source_path=image_file)

        image_selected_for_sam_augmentation = image_file in sam_augmentation_set

        if "noise" in dirs and masks_for_bboxes and image_selected_for_sam_augmentation:
            self.apply_noise_reduction(image.copy(), masks_for_bboxes, image_file)

        if "shadow" in dirs and masks_for_bboxes and image_selected_for_sam_augmentation:
            self.create_shadow_image(image.copy(), masks_for_bboxes, image_file)

        if "object_detail" in dirs and masks_for_bboxes and image_selected_for_sam_augmentation:
            self.create_object_detail_image(image.copy(), masks_for_bboxes, image_file)
        if "object_clarity" in dirs and masks_for_bboxes and image_selected_for_sam_augmentation:
            self.create_object_clarity_image(image.copy(), masks_for_bboxes, image_file)

        value = progress_offset + idx + 1
        self.set_label_progress(
            0,
            max(1, progress_total),
            value=min(value, max(1, progress_total)),
            progress_format=f"SAM3 batch {min(value, max(1, progress_total))}/{max(1, progress_total)}",
        )
        QtWidgets.QApplication.processEvents()

    stopped = self._sam3_should_stop()
    negative_started = False
    if actions["negatives"] and not stopped:
        negative_started = self.start_create_negative_images(
            target_image_files=negative_image_files,
            preselected=True,
        )
        message = "SAM3 tools finished; negative generation is running." if negative_started else "SAM3 tools finished; no valid negative images were queued."
    else:
        message = "Process was stopped!" if stopped else "Finished!"
    self._show_sam3_batch_result(message)
    if not negative_started:
        self.set_label_progress(
            0,
            max(1, progress_total),
            value=progress_total,
            progress_format="SAM3 batch done",
        )
    self.img_index_number_changed(0)
    logging.info("Process completed and image index reset.")
    return negative_started


def convert_bbox_to_segmentation(self, image, annotation_file):
    image = self.ensure_rgb(image)
    if image is None:
        return []

    img_height, img_width = image.shape[:2]
    yolo_lines = self._read_label_lines(annotation_file)
    segmentation_labels = []
    polygon_parse_preference = self._polygon_label_parse_preference()

    for yolo_line in yolo_lines:
        try:
            box_obj = BoundingBox.from_str(
                yolo_line,
                preferred_polygon_type=polygon_parse_preference,
            )
            if not box_obj:
                continue
            if self._label_type(box_obj) != "bbox":
                continue

            bbox = self._bbox_xyxy_from_box_obj(box_obj, img_width, img_height)
            if bbox is None:
                continue

            mask = self.get_sam_mask(image, bbox)
            if mask is None or not mask.any():
                continue

            polygon = self.get_polygon_from_mask(mask, simplify=True, epsilon_factor=0.001)
            if not polygon or len(polygon) < 3:
                continue

            seg_box = BoundingBox(
                class_id=box_obj.class_id,
                x_center=0,
                y_center=0,
                width=0,
                height=0,
                segmentation=self._normalize_polygon(polygon, img_width, img_height),
            )
            segmentation_labels.append(seg_box.to_str())

        except Exception as e:
            logging.error("Error processing line '%s': %s", yolo_line, e, exc_info=True)

    return segmentation_labels


def get_all_labels(self):
    all_labels = []

    self.class_names = self.load_classes()

    for img_file in self.image_files:
        base = os.path.splitext(img_file)[0]
        txt_file = base + '.txt'

        if os.path.exists(txt_file) and os.path.getsize(txt_file) > 0:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for line_number, line in enumerate(lines, start=1):
                    if line.strip():
                        try:
                            class_index = int(line.strip().split()[0])
                            if class_index < len(self.class_names):
                                all_labels.append(self.class_names[class_index])
                            else:
                                logger.warning(f"Class index {class_index} in {txt_file} is out of range.")
                        except ValueError:
                            logger.warning(f"Failed parsing line {line_number} in {txt_file}: '{line.strip()}'")

    return all_labels


def start_create_negative_images(self, target_image_files=None, preselected=False):
    folder_path = self.current_dataset_directory()
    if not folder_path:
        QMessageBox.critical(self, "Error", "Please open a dataset first.")
        return False

    existing_thread = getattr(self, "processing_thread", None)
    try:
        if existing_thread is not None and existing_thread.isRunning():
            QMessageBox.warning(
                self,
                "Negatives Running",
                "Negative-image generation is already running.",
            )
            return False
    except RuntimeError:
        self.processing_thread = None

    self.image_directory = folder_path

    from darkfusion_negative_crops import resolve_negative_folder

    save_folder_path = resolve_negative_folder(folder_path, create=True)

    candidate_files = list(target_image_files or [])
    if not candidate_files:
        candidate_files = list(getattr(self, "image_files", []) or [])
    if not candidate_files:
        candidate_files = self._scan_images(folder_path)

    if not preselected:
        candidate_files = self.sampled_sam_augmentation_images(candidate_files)
    else:
        candidate_files = [
            self.normalize_path(path)
            for path in candidate_files
            if path and os.path.isfile(path) and not self.is_placeholder_file(path)
        ]

    image_files = []
    yolo_annotation_files = []

    for candidate_index, image_file in enumerate(candidate_files):
        if candidate_index % 32 == 0:
            QtWidgets.QApplication.processEvents()
        if getattr(self, "_sam3_batch_active", False) and self._sam3_should_stop():
            return False

        if os.path.splitext(image_file)[1].lower() not in DATASET_IMAGE_SUFFIXES:
            continue

        yolo_annotation_file = self.get_label_file_safe(image_file)
        if not yolo_annotation_file or not os.path.exists(yolo_annotation_file):
            logger.warning(f"No YOLO annotation found for {image_file}. Skipping.")
            continue

        if not self._valid_annotation_lines_from_file(yolo_annotation_file):
            logger.warning(f"No valid bbox/pose/segmentation/OBB labels found for negative generation: {image_file}")
            continue

        image_files.append(image_file)
        yolo_annotation_files.append(yolo_annotation_file)

    if not image_files:
        logger.info("Negative-image generation skipped because no valid labeled images were queued.")
        return False

    image_format = self.selected_augmentation_output_format_value()

    self.processing_thread = ImageProcessingThread(
        image_files,
        yolo_annotation_files,
        save_folder_path,
        100,
        image_format,
    )
    self.set_label_progress(0, 100, value=0, progress_format="Negatives 0%")
    self.processing_thread.progressSignal.connect(
        lambda value: self.set_label_progress(
            0,
            100,
            value=max(0, min(100, int(value))),
            progress_format=f"Negatives {max(0, min(100, int(value)))}%",
        )
    )
    self.processing_thread.finished.connect(
        lambda thread=self.processing_thread: self._on_sam3_negative_generation_finished(thread)
    )
    self.processing_thread.finished.connect(
        lambda thread=self.processing_thread: self._cleanup_worker_reference("processing_thread", thread)
    )
    self.processing_thread.start()
    logger.info(
        "Started SAM3 negative-image generation for %s image(s) into %s with output format %s.",
        len(image_files),
        save_folder_path,
        image_format,
    )
    return True


def get_head_class_id(self):
    """
        Retrieve the class ID for head labels.
        Prefer the Extra Label dialog's target ID, then fall back to a class named head.
        """
    self.class_labels = self.load_classes() or []
    configured_id = str(self.class_id.text() if hasattr(self, "class_id") else "").strip()
    if configured_id:
        try:
            class_id = int(configured_id)
            if 0 <= class_id < len(self.class_labels):
                return class_id
            logging.error("Extra-label class ID %s is outside the class list.", class_id)
            return None
        except ValueError:
            logging.error("Extra-label class ID must be a number: %s", configured_id)
            return None

    for index, class_name in enumerate(self.class_labels):
        if str(class_name).strip().casefold() == "head":
            return index

    logging.error("No target class ID was entered and no class named 'head' exists.")
    return None


def generate_head_labels(self, box, img_width, img_height, class_id):
    """
        Generate head labels for a bounding box if the head area checkbox is enabled.

        Args:
            box (tuple): The bounding box (x1, y1, x2, y2).
            img_width (int): Width of the image.
            img_height (int): Height of the image.
            class_id (int): Class ID of the object.

        Returns:
            str: A YOLO-style head label if applicable, otherwise None.
        """
    if not self.heads_area.isChecked():
        return None

    head_class_id = self.get_head_class_id()
    if head_class_id is None:
        logging.warning("Skipping head labels due to missing head class ID.")
        return None
    if int(class_id) == int(head_class_id):
        return None

    # Ensure `box` is absolute (x1, y1, x2, y2) and not YOLO format
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1

    # Calculate the head bounding box in absolute coordinates
    head_x, head_y, head_w, head_h = self.calculate_head_area(x1, y1, w, h)

    head_x = max(0, min(int(head_x), max(0, img_width - 1)))
    head_y = max(0, min(int(head_y), max(0, img_height - 1)))
    head_w = max(1, min(int(head_w), img_width - head_x))
    head_h = max(1, min(int(head_h), img_height - head_y))

    # Normalize coordinates for YOLO format (relative to image size).
    head_xc = (head_x + head_w / 2) / img_width
    head_yc = (head_y + head_h / 2) / img_height
    head_w /= img_width
    head_h /= img_height

    #  **Return YOLO-formatted label**
    return f"{head_class_id} {head_xc:.6f} {head_yc:.6f} {head_w:.6f} {head_h:.6f}"


@staticmethod
def _bbox_iou_xyxy(first, second):
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _sam3_head_candidates(self, predictor, image, prompt, min_confidence):
    """Run one semantic query and return mask-derived head boxes."""
    candidates = []
    if predictor is None or image is None:
        return candidates

    img_height, img_width = image.shape[:2]
    try:
        with sam_predictor_session(predictor), torch.inference_mode():
            predictor.set_image(np.ascontiguousarray(image))
            if not hasattr(predictor.model, "mask_threshold"):
                predictor.model.mask_threshold = 0.0
            results = predictor(text=[str(prompt)])

        for result in results or []:
            mask_data = getattr(getattr(result, "masks", None), "data", None)
            if mask_data is None:
                continue
            confidences = getattr(getattr(result, "boxes", None), "conf", None)
            if confidences is not None and hasattr(confidences, "detach"):
                confidences = confidences.detach().float().cpu().numpy().reshape(-1)

            for index, raw_mask in enumerate(mask_data):
                confidence = float(confidences[index]) if confidences is not None and index < len(confidences) else 0.0
                if confidence < float(min_confidence):
                    continue
                if hasattr(raw_mask, "detach"):
                    raw_mask = raw_mask.detach().float().cpu().numpy()
                mask = self._normalize_sam_mask_to_image(raw_mask, img_width, img_height)
                bbox = self._mask_bbox(mask) if mask is not None else None
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                if (x2 - x1) * (y2 - y1) < 16:
                    continue
                candidates.append({"bbox": (x1, y1, x2, y2), "confidence": confidence})
    except Exception as e:
        logger.warning("SAM3 nested-label query failed: %s", e, exc_info=True)
    return candidates


def _nested_label_scan_guides(self, parent_bbox):
    """Return the base guide plus optional SAHI tiles covering the parent."""
    px1, py1, px2, py2 = [float(value) for value in parent_bbox]
    parent_w = max(1.0, px2 - px1)
    parent_h = max(1.0, py2 - py1)
    guide_x, guide_y, guide_w, guide_h = self.calculate_head_area(
        px1, py1, parent_w, parent_h
    )
    guide_w = max(1.0, min(float(guide_w), parent_w))
    guide_h = max(1.0, min(float(guide_h), parent_h))

    dynamic_widget = getattr(self, "dynamic_guide_scan_checkbox", None)
    dynamic = bool(dynamic_widget.isChecked()) if dynamic_widget is not None else False
    base_x1 = max(px1, min(px2 - guide_w, float(guide_x)))
    base_y1 = max(py1, min(py2 - guide_h, float(guide_y)))
    base_guide = (base_x1, base_y1, base_x1 + guide_w, base_y1 + guide_h)
    if not dynamic:
        return [base_guide]

    local_slices = get_sahi_slice_bboxes(
        image_height=max(1, int(round(parent_h))),
        image_width=max(1, int(round(parent_w))),
        slice_height=max(1, int(round(guide_h))),
        slice_width=max(1, int(round(guide_w))),
        auto_slice_resolution=False,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
    )
    tiled_guides = [
        (px1 + sx1, py1 + sy1, px1 + sx2, py1 + sy2)
        for sx1, sy1, sx2, sy2 in local_slices
    ]
    base_center_x = (base_guide[0] + base_guide[2]) / 2.0
    base_center_y = (base_guide[1] + base_guide[3]) / 2.0
    tiled_guides.sort(
        key=lambda box: (
            ((box[0] + box[2]) / 2.0 - base_center_x) ** 2
            + ((box[1] + box[3]) / 2.0 - base_center_y) ** 2
        )
    )

    guides = [base_guide]
    for guide in tiled_guides:
        if not any(self._bbox_iou_xyxy(guide, existing) > 0.98 for existing in guides):
            guides.append(guide)
    return guides


def _select_heads_for_parent_boxes(self, candidates, parent_boxes):
    """Assign at most one geometrically safe SAM3 candidate to each parent."""
    selected = {}
    parent_guides = [
        self._nested_label_scan_guides(parent)
        for parent in parent_boxes
    ]
    for candidate in candidates:
        cx1, cy1, cx2, cy2 = candidate["bbox"]
        center_x = (cx1 + cx2) / 2.0
        center_y = (cy1 + cy2) / 2.0
        candidate_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))
        best_parent = None
        best_score = -1.0

        for parent_index, parent in enumerate(parent_boxes):
            px1, py1, px2, py2 = parent
            parent_w = max(1.0, px2 - px1)
            parent_h = max(1.0, py2 - py1)
            parent_area = parent_w * parent_h
            guide_candidate_overlap = 0.0
            center_in_guide = False
            matched_guide = None
            for guide in parent_guides[parent_index]:
                gx1, gy1, gx2, gy2 = guide
                inter_w = max(0.0, min(cx2, gx2) - max(cx1, gx1))
                inter_h = max(0.0, min(cy2, gy2) - max(cy1, gy1))
                overlap = (inter_w * inter_h) / candidate_area
                if overlap > guide_candidate_overlap:
                    guide_candidate_overlap = overlap
                    matched_guide = guide
                center_in_guide = center_in_guide or (
                    gx1 <= center_x <= gx2 and gy1 <= center_y <= gy2
                )

            threshold_widget = getattr(self, "guide_match_threshold", None)
            match_threshold = float(threshold_widget.value()) if threshold_widget is not None else 0.35
            if guide_candidate_overlap < match_threshold:
                continue
            if not (px1 <= center_x <= px2 and py1 <= center_y <= py2):
                continue

            inter_w = max(0.0, min(cx2, px2) - max(cx1, px1))
            inter_h = max(0.0, min(cy2, py2) - max(cy1, py1))
            inside_ratio = (inter_w * inter_h) / candidate_area
            area_ratio = candidate_area / parent_area
            if inside_ratio < 0.70 or not (0.004 <= area_ratio <= 0.35):
                continue
            if (cx2 - cx1) > parent_w * 0.90 or (cy2 - cy1) > parent_h * 0.65:
                continue

            vertical_position = max(0.0, min(1.0, (center_y - py1) / parent_h))
            score = (
                float(candidate["confidence"])
                + inside_ratio * 0.18
                + guide_candidate_overlap * 0.22
                + (0.08 if center_in_guide else 0.0)
                + max(0.0, 0.55 - vertical_position) * 0.08
            )
            if score > best_score:
                best_score = score
                best_parent = parent_index
                best_guide = matched_guide

        if best_parent is None:
            continue
        previous = selected.get(best_parent)
        if previous is None or best_score > previous[0]:
            selected[best_parent] = (
                best_score,
                candidate["bbox"],
                candidate["confidence"],
                best_guide,
            )
    return selected


def _head_line_from_bbox(self, bbox, class_id, img_width, img_height):
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = max(0.0, min(float(img_width - 1), x1))
    y1 = max(0.0, min(float(img_height - 1), y1))
    x2 = max(x1 + 1.0, min(float(img_width), x2))
    y2 = max(y1 + 1.0, min(float(img_height), y2))
    xc, yc, width, height = self._xyxy_to_yolo_xywh((x1, y1, x2, y2), img_width, img_height)
    return f"{int(class_id)} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"


def _auto_head_annotation_bounds(self, box_obj, img_width, img_height):
    """Return pixel bounds for bbox, pose, segmentation, or OBB annotations."""
    try:
        if getattr(box_obj, "segmentation", None):
            points = np.asarray(box_obj.segmentation, dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= img_width
            points[:, 1] *= img_height
            bbox = (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max())
        elif getattr(box_obj, "obb", None):
            points = np.asarray(box_obj.obb[:8], dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= img_width
            points[:, 1] *= img_height
            bbox = (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max())
        else:
            bbox = box_obj.to_xyxy(img_width, img_height)

        x1, y1, x2, y2 = [float(value) for value in bbox]
        x1 = max(0.0, min(float(img_width), x1))
        y1 = max(0.0, min(float(img_height), y1))
        x2 = max(0.0, min(float(img_width), x2))
        y2 = max(0.0, min(float(img_height), y2))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
    except Exception as e:
        logger.debug("Could not derive nested-label annotation bounds: %s", e)
        return None


def _current_auto_head_image_path(self):
    override = self.normalize_path(getattr(self, "_nested_review_override_image_file", ""))
    if override and os.path.isfile(override):
        return override
    viewer = getattr(self, "_nested_label_suggestion_viewer", None)
    review_files = getattr(self, "_nested_review_files", None) or []
    review_index = int(getattr(self, "_nested_review_index", 0) or 0)
    if (
        isinstance(viewer, NestedLabelSuggestionViewer)
        and viewer.isVisible()
        and 0 <= review_index < len(review_files)
        and os.path.isfile(review_files[review_index])
    ):
        return self.normalize_path(review_files[review_index])
    current = self.normalize_path(getattr(self, "current_file", ""))
    if current and os.path.isfile(current):
        return current

    dataset_dir = self.current_dataset_directory()
    current_name = os.path.basename(current) if current else ""
    if dataset_dir and current_name:
        candidate = os.path.join(dataset_dir, current_name)
        if os.path.isfile(candidate):
            return candidate
    return ""


def _show_auto_head_preview_image(self, image):
    preview = getattr(self, "auto_head_preview_label", None)
    if not isinstance(preview, QtWidgets.QLabel) or image is None:
        return
    rgb = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    qimage = QtGui.QImage(
        rgb.data,
        width,
        height,
        int(rgb.strides[0]),
        QtGui.QImage.Format_RGB888,
    ).copy()
    pixmap = QtGui.QPixmap.fromImage(qimage)
    self._auto_head_preview_full_pixmap = QtGui.QPixmap(pixmap)
    target_size = preview.contentsRect().size()
    if target_size.width() < 2 or target_size.height() < 2:
        target_size = QtCore.QSize(560, 315)
    preview.setPixmap(
        pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    )


def _ensure_nested_label_suggestion_viewer(self):
    viewer = getattr(self, "_nested_label_suggestion_viewer", None)
    if isinstance(viewer, NestedLabelSuggestionViewer):
        return viewer
    viewer = NestedLabelSuggestionViewer(self)
    # Auto-verify current image when navigating away
    viewer.previousRequested.connect(lambda: self._auto_verify_current_nested_image() or self.show_nested_label_review_index(self._nested_review_index - 1))
    viewer.nextRequested.connect(lambda: self._auto_verify_current_nested_image() or self.show_nested_label_review_index(self._nested_review_index + 1))
    viewer.refreshRequested.connect(lambda: self.show_nested_label_review_index(self._nested_review_index))
    viewer.applyCurrentRequested.connect(self.apply_current_nested_label_suggestion)
    viewer.applyAllRequested.connect(self.apply_all_nested_label_suggestions)
    self._nested_label_suggestion_viewer = viewer
    return viewer


def _auto_verify_current_nested_image(self):
    """Mark current nested review image as verified before navigation."""
    try:
        review_files = getattr(self, "_nested_review_files", None) or []
        if not review_files:
            return False
        
        index = getattr(self, "_nested_review_index", 0)
        if 0 <= index < len(review_files):
            current_image = review_files[index]
            dataset_root = getattr(self, "base_directory", None) or self.current_dataset_directory()
            if dataset_root and current_image:
                tracker = VerifiedImagesTracker()
                was_new = tracker.mark_image_verified(dataset_root, current_image)
    except Exception as e:
        # Don't block navigation on verification errors
        print(f"Warning: Could not verify image: {e}")
    return False  # Allow navigation to proceed


def _nested_review_images_for_source_class(self, dataset_dir, source_class_id):
    review_files = []
    for image_file in self._scan_images(dataset_dir):
        label_file = self.get_label_file_safe(image_file)
        label_lines = self._valid_annotation_lines_from_file(label_file)
        for line in label_lines:
            try:
                if int(float(str(line).split()[0])) == int(source_class_id):
                    review_files.append(image_file)
                    break
            except (ValueError, IndexError):
                continue
    return review_files


def open_nested_label_suggestion_review(self):
    self.refresh_auto_head_class_selectors()
    dataset_dir = self.current_dataset_directory()
    source_combo = getattr(self, "auto_head_source_class_combo", None)
    target_combo = getattr(self, "auto_head_target_class_combo", None)
    source_class_id = source_combo.currentData() if isinstance(source_combo, QComboBox) else None
    target_class_id = target_combo.currentData() if isinstance(target_combo, QComboBox) else None
    if not dataset_dir or source_class_id is None or target_class_id is None:
        QMessageBox.warning(self, "Nested Label Review", "Open a dataset and choose parent/output classes first.")
        return
    if int(source_class_id) == int(target_class_id):
        QMessageBox.warning(self, "Nested Label Review", "Parent and output classes must be different.")
        return

    review_files = self._nested_review_images_for_source_class(dataset_dir, int(source_class_id))
    if not review_files:
        QMessageBox.information(self, "Nested Label Review", "No images contain the selected parent class.")
        return

    self._nested_review_files = review_files
    current_file = self.normalize_path(getattr(self, "current_file", ""))
    normalized = [self.normalize_path(path) for path in review_files]
    self._nested_review_index = normalized.index(current_file) if current_file in normalized else 0

    viewer = self._ensure_nested_label_suggestion_viewer()
    viewer.show()
    viewer.raise_()
    viewer.activateWindow()
    self.show_nested_label_review_index(self._nested_review_index)


def show_nested_label_review_index(self, index):
    review_files = getattr(self, "_nested_review_files", None) or []
    viewer = self._ensure_nested_label_suggestion_viewer()
    if not review_files:
        viewer.status_label.setText("No review images are available.")
        return
    index = max(0, min(int(index), len(review_files) - 1))
    self._nested_review_index = index
    image_file = review_files[index]
    mode_name = "geometry suggestion" if self.selected_nested_label_mode() == "geometry" else "SAM3 suggestion"
    viewer.set_busy(True, f"Building {mode_name} for {os.path.basename(image_file)}...")
    QtWidgets.QApplication.processEvents()

    self._nested_review_override_image_file = image_file
    self._auto_head_preview_context = None
    self._auto_head_preview_full_pixmap = QPixmap()
    try:
        self.preview_sam3_auto_heads()
    finally:
        self._nested_review_override_image_file = ""

    context = getattr(self, "_auto_head_preview_context", None)
    pixmap = getattr(self, "_auto_head_preview_full_pixmap", None)
    if isinstance(context, dict) and isinstance(pixmap, QPixmap) and not pixmap.isNull():
        status = self.auto_head_status_label.text()
        viewer.set_busy(False)
        viewer.set_review(
            pixmap,
            image_file,
            index,
            len(review_files),
            status,
            can_apply=bool(context.get("proposed")),
        )
    else:
        viewer.set_busy(False, "No safe suggestion could be built for this image.")
        viewer.previous_btn.setEnabled(index > 0)
        viewer.next_btn.setEnabled(index + 1 < len(review_files))
        viewer.apply_current_btn.setEnabled(False)


def apply_current_nested_label_suggestion(self):
    context = getattr(self, "_auto_head_preview_context", None)
    viewer = self._ensure_nested_label_suggestion_viewer()
    if not isinstance(context, dict):
        viewer.status_label.setText("No current suggestion is available.")
        return
    proposed = list(context.get("proposed") or [])
    if not proposed:
        viewer.status_label.setText("There are no unapplied safe suggestions on this image.")
        viewer.apply_current_btn.setEnabled(False)
        return

    label_file = context.get("label_file", "")
    image = context.get("image")
    target_class_id = context.get("target_class_id")
    if not label_file or image is None or target_class_id is None:
        viewer.status_label.setText("The current suggestion is incomplete and was not saved.")
        return
    img_height, img_width = image.shape[:2]
    fresh_lines = self._valid_annotation_lines_from_file(label_file)
    new_lines = [
        self._head_line_from_bbox(item[0], target_class_id, img_width, img_height)
        for item in proposed
    ]
    if not self._write_label_lines(label_file, list(fresh_lines) + new_lines):
        viewer.status_label.setText("Could not write the current label file.")
        return

    context["label_lines"] = list(fresh_lines) + new_lines
    context["existing_head_boxes"] = list(context.get("existing_head_boxes") or []) + [
        item[0] for item in proposed
    ]
    target_name = self._selected_auto_head_target_name() or "output"
    self._render_cached_auto_head_preview()
    viewer.status_label.setText(
        f"Applied {len(new_lines)} '{target_name}' suggestion(s) to this image."
    )
    viewer.apply_current_btn.setEnabled(False)
    self.auto_head_status_label.setText(viewer.status_label.text())


def apply_all_nested_label_suggestions(self):
    self.run_sam3_auto_heads()
    if getattr(self, "_nested_review_files", None):
        self.show_nested_label_review_index(self._nested_review_index)


def queue_auto_head_geometry_preview_update(self, *_args):
    context = getattr(self, "_auto_head_preview_context", None)
    if not context:
        return
    cached_file = self.normalize_path(context.get("image_file", ""))
    if cached_file and cached_file != self.normalize_path(self._current_auto_head_image_path()):
        self._auto_head_preview_context = None
        return
    timer = getattr(self, "_auto_head_geometry_preview_timer", None)
    if isinstance(timer, QtCore.QTimer):
        timer.start(90)


def _render_cached_auto_head_preview(self):
    """Redraw cached SAM3 results plus the current live geometry guide."""
    context = getattr(self, "_auto_head_preview_context", None)
    if not isinstance(context, dict):
        return

    image = context.get("image")
    parent_boxes = context.get("parent_boxes") or []
    existing_head_boxes = context.get("existing_head_boxes") or []
    if image is None:
        return
    target_name = self._selected_auto_head_target_name() or "selected class"
    geometry_only = self.selected_nested_label_mode() == "geometry"
    candidates = context.get("candidates") or []
    selected = {} if geometry_only else self._select_heads_for_parent_boxes(candidates, parent_boxes)
    context["selected"] = dict(selected)

    geometry_boxes = []
    matched_scan_boxes = []
    proposed = []
    for parent_index, parent_bbox in enumerate(parent_boxes):
        px1, py1, px2, py2 = parent_bbox
        hx, hy, hw, hh = self.calculate_head_area(px1, py1, px2 - px1, py2 - py1)
        geometry_bbox = (hx, hy, hx + hw, hy + hh)
        geometry_boxes.append(geometry_bbox)

        selected_head = selected.get(parent_index)
        if selected_head is not None:
            head_bbox = selected_head[1]
            confidence = float(selected_head[2])
            source = "SAM3"
            if len(selected_head) > 3 and selected_head[3] is not None:
                matched_scan_boxes.append(selected_head[3])
        elif geometry_only or self.auto_head_geometry_fallback.isChecked():
            head_bbox = geometry_bbox
            confidence = None
            source = "geometry" if geometry_only else "fallback"
        else:
            continue
        if any(
            self._bbox_iou_xyxy(head_bbox, other) >= 0.40
            for other in existing_head_boxes + [item[0] for item in proposed]
        ):
            continue
        proposed.append((head_bbox, confidence, source))

    context["proposed"] = list(proposed)

    visual = image.copy()
    for bbox in parent_boxes:
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        cv2.rectangle(visual, (x1, y1), (x2, y2), (0, 165, 255), 2)
    for bbox in geometry_boxes:
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        guide_color = (255, 0, 255)
        cv2.rectangle(visual, (x1, y1), (x2, y2), guide_color, 1)
        cv2.putText(
            visual,
            "start guide",
            (x1, min(visual.shape[0] - 4, max(14, y2 + 14))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            guide_color,
            1,
            cv2.LINE_AA,
        )
    for bbox in matched_scan_boxes:
        if any(self._bbox_iou_xyxy(bbox, base) > 0.98 for base in geometry_boxes):
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        cv2.rectangle(visual, (x1, y1), (x2, y2), (180, 70, 180), 1)
        cv2.putText(
            visual,
            "matched window",
            (x1, min(visual.shape[0] - 4, max(14, y2 + 14))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (180, 70, 180),
            1,
            cv2.LINE_AA,
        )
    for bbox in existing_head_boxes:
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        cv2.rectangle(visual, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(visual, f"existing {target_name}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    for bbox, confidence, source in proposed:
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        cv2.rectangle(visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{target_name} {confidence:.2f}" if confidence is not None else f"{target_name} ({source})"
        cv2.putText(visual, label, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    self._show_auto_head_preview_image(visual)
    if geometry_only:
        mode_status = "Geometry Only (SAM3 Off); the base guide is the proposed label."
    elif getattr(self, "dynamic_guide_scan_checkbox", None) is not None and self.dynamic_guide_scan_checkbox.isChecked():
        mode_status = "SAHI scans each parent internally; only the start guide and winning window are shown."
    else:
        mode_status = "Hybrid mode is matching cached SAM3 masks against the base guide."
    if not geometry_only and not context.get("sam_evaluated", False):
        mode_status += " Use Refresh Suggestion to run SAM3 for this image."
    status = (
        f"Preview: {len(proposed)} proposed, {len(existing_head_boxes)} existing. {mode_status} "
            "Orange = parent, green = proposed, cyan = existing, magenta = start guide, purple = matched window. No files changed."
    )
    self.auto_head_status_label.setText(status)
    viewer = getattr(self, "_nested_label_suggestion_viewer", None)
    pixmap = getattr(self, "_auto_head_preview_full_pixmap", None)
    if isinstance(viewer, NestedLabelSuggestionViewer) and viewer.isVisible() and isinstance(pixmap, QPixmap):
        viewer.set_review(
            pixmap,
            context.get("image_file", ""),
            int(getattr(self, "_nested_review_index", 0) or 0),
            len(getattr(self, "_nested_review_files", None) or []),
            status,
            can_apply=bool(proposed),
        )


def preview_sam3_auto_heads(self):
    """Render proposed nested labels for the current image without writing labels."""
    self._auto_head_preview_context = None
    image_file = self._current_auto_head_image_path()
    if not image_file:
        QMessageBox.warning(self, "Nested Label Preview", "Display a labeled image first.")
        return

    source_combo = getattr(self, "auto_head_source_class_combo", None)
    target_combo = getattr(self, "auto_head_target_class_combo", None)
    source_class_id = source_combo.currentData() if isinstance(source_combo, QtWidgets.QComboBox) else None
    target_class_id = target_combo.currentData() if isinstance(target_combo, QtWidgets.QComboBox) else None
    if source_class_id is None or target_class_id is None or int(source_class_id) == int(target_class_id):
        QMessageBox.warning(self, "Nested Label Preview", "Choose different parent and output classes.")
        return

    geometry_only = self.selected_nested_label_mode() == "geometry"
    prompt = str(self.auto_head_prompt_edit.text()).strip()
    if not geometry_only and not prompt:
        QMessageBox.warning(self, "Nested Label Preview", "Enter a SAM3 concept prompt first.")
        return

    label_file = self.get_label_file_safe(image_file)
    label_lines = self._valid_annotation_lines_from_file(label_file)
    image = self._read_image_cv(image_file)
    if image is None or not label_lines:
        QMessageBox.warning(self, "Nested Label Preview", "The current image has no readable labels.")
        return

    img_height, img_width = image.shape[:2]
    parsed = []
    for line in label_lines:
        box_obj = BoundingBox.from_str(
            line,
            preferred_polygon_type=self._polygon_label_parse_preference(),
        )
        if box_obj is not None:
            parsed.append(box_obj)

    source_class_id = int(source_class_id)
    target_class_id = int(target_class_id)
    parent_boxes = [
        self._auto_head_annotation_bounds(box_obj, img_width, img_height)
        for box_obj in parsed
        if int(box_obj.class_id) == source_class_id
    ]
    parent_boxes = [bbox for bbox in parent_boxes if bbox is not None]
    existing_head_boxes = [
        self._auto_head_annotation_bounds(box_obj, img_width, img_height)
        for box_obj in parsed
        if int(box_obj.class_id) == target_class_id
    ]
    existing_head_boxes = [bbox for bbox in existing_head_boxes if bbox is not None]

    if not parent_boxes:
        self._show_auto_head_preview_image(image)
        self.auto_head_status_label.setText("No labels of the selected parent class exist in this image.")
        return

    predictor = None
    if not geometry_only:
        predictor = self.ensure_sam_semantic_predictor_loaded()
        if predictor is None:
            QMessageBox.critical(self, "Nested Label Preview", "The SAM3 semantic predictor could not be loaded.")
            return

    self.auto_head_preview_button.setEnabled(False)
    self.auto_head_run_button.setEnabled(False)
    self.auto_head_status_label.setText("Building current-image preview...")
    QtWidgets.QApplication.processEvents()
    try:
        candidates = []
        if not geometry_only:
            candidates = self._sam3_head_candidates(
                predictor,
                image,
                prompt,
                float(self.auto_head_min_confidence.value()),
            )
        selected = {} if geometry_only else self._select_heads_for_parent_boxes(candidates, parent_boxes)
        self._auto_head_preview_context = {
            "image_file": self.normalize_path(image_file),
            "label_file": self.normalize_path(label_file),
            "label_lines": list(label_lines),
            "target_class_id": target_class_id,
            "image": image.copy(),
            "parent_boxes": list(parent_boxes),
            "existing_head_boxes": list(existing_head_boxes),
            "candidates": list(candidates),
            "selected": dict(selected),
            "sam_evaluated": not geometry_only,
        }
        self._render_cached_auto_head_preview()
    finally:
        self.auto_head_preview_button.setEnabled(True)
        self.auto_head_run_button.setEnabled(True)
        self._safe_empty_cache()


@gpu_work("SAM3 label tools", keep=("sam_semantic", "sam_prompt"))
def run_sam3_auto_heads(self):
    """Append validated SAM3 nested boxes inside existing parent annotations."""
    dataset_dir = self.current_dataset_directory()
    source_combo = getattr(self, "auto_head_source_class_combo", None)
    target_combo = getattr(self, "auto_head_target_class_combo", None)
    if not dataset_dir or not isinstance(source_combo, QtWidgets.QComboBox) or not isinstance(target_combo, QtWidgets.QComboBox):
        QMessageBox.warning(self, "SAM3 Nested Labels", "Open a labeled image dataset first.")
        return

    source_class_id = source_combo.currentData()
    target_class_id = target_combo.currentData()
    if source_class_id is None or target_class_id is None:
        QMessageBox.warning(self, "SAM3 Nested Labels", "Choose both the parent and output classes.")
        return
    source_class_id = int(source_class_id)
    target_class_id = int(target_class_id)
    target_name = self._selected_auto_head_target_name() or f"class {target_class_id}"
    if source_class_id == target_class_id:
        QMessageBox.warning(self, "SAM3 Nested Labels", "The parent and output classes must be different.")
        return

    geometry_only = self.selected_nested_label_mode() == "geometry"
    prompt = str(self.auto_head_prompt_edit.text()).strip()
    if not geometry_only and not prompt:
        QMessageBox.warning(self, "SAM3 Nested Labels", "Enter a short SAM3 prompt such as 'human head' or 'face'.")
        return

    image_files = self._scan_images(dataset_dir)
    if not image_files:
        QMessageBox.warning(self, "SAM3 Nested Labels", "No images were found in the current dataset.")
        return

    method_text = "use the percentage geometry guide (SAM3 stays off)" if geometry_only else "search with SAM3 and match prompt masks to the geometry guide"
    answer = QMessageBox.question(
        self,
        f"Generate {target_name} Labels",
        f"Process {len(image_files)} image(s), {method_text}, and append '{target_name}' (class {target_class_id}) labels inside class {source_class_id} parents?\n\nExisting labels are kept.",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    if answer != QMessageBox.Yes:
        return

    predictor = None
    if not geometry_only:
        predictor = self.ensure_sam_semantic_predictor_loaded()
        if predictor is None:
            QMessageBox.critical(self, "SAM3 Nested Labels", "The SAM3 semantic predictor could not be loaded.")
            return

    min_confidence = float(self.auto_head_min_confidence.value())
    use_fallback = geometry_only or bool(self.auto_head_geometry_fallback.isChecked())
    added_total = 0
    changed_files = 0
    parent_total = 0
    self.auto_head_run_button.setEnabled(False)
    self.auto_head_status_label.setText(f"Generating {target_name} labels...")

    try:
        self.set_label_progress(0, max(1, len(image_files)), value=0, progress_format="Nested Labels 0/%s" % len(image_files))
        for image_index, image_file in enumerate(image_files):
            QtWidgets.QApplication.processEvents()
            label_file = self.get_label_file_safe(image_file)
            label_lines = self._valid_annotation_lines_from_file(label_file)
            if not label_lines:
                continue

            image = self._read_image_cv(image_file)
            if image is None:
                continue
            img_height, img_width = image.shape[:2]
            parsed = []
            for line in label_lines:
                box_obj = BoundingBox.from_str(line, preferred_polygon_type=self._polygon_label_parse_preference())
                if box_obj is not None:
                    parsed.append(box_obj)

            parent_boxes = [
                self._auto_head_annotation_bounds(box_obj, img_width, img_height)
                for box_obj in parsed
                if int(box_obj.class_id) == source_class_id
            ]
            parent_boxes = [bbox for bbox in parent_boxes if bbox is not None]
            if not parent_boxes:
                continue
            parent_total += len(parent_boxes)

            existing_head_boxes = [
                self._auto_head_annotation_bounds(box_obj, img_width, img_height)
                for box_obj in parsed
                if int(box_obj.class_id) == target_class_id
            ]
            existing_head_boxes = [bbox for bbox in existing_head_boxes if bbox is not None]

            candidates = [] if geometry_only else self._sam3_head_candidates(predictor, image, prompt, min_confidence)
            selected = {} if geometry_only else self._select_heads_for_parent_boxes(candidates, parent_boxes)
            new_boxes = []

            for parent_index, parent_bbox in enumerate(parent_boxes):
                selected_head = selected.get(parent_index)
                if selected_head is not None:
                    head_bbox = selected_head[1]
                elif use_fallback:
                    px1, py1, px2, py2 = parent_bbox
                    hx, hy, hw, hh = self.calculate_head_area(px1, py1, px2 - px1, py2 - py1)
                    head_bbox = (hx, hy, hx + hw, hy + hh)
                else:
                    continue

                if any(self._bbox_iou_xyxy(head_bbox, other) >= 0.40 for other in existing_head_boxes + new_boxes):
                    continue
                new_boxes.append(head_bbox)

            if new_boxes:
                new_lines = [
                    self._head_line_from_bbox(bbox, target_class_id, img_width, img_height)
                    for bbox in new_boxes
                ]
                if self._write_label_lines(label_file, list(label_lines) + new_lines):
                    added_total += len(new_lines)
                    changed_files += 1

            self.set_label_progress(
                0,
                max(1, len(image_files)),
                value=image_index + 1,
                progress_format=f"Nested Labels {image_index + 1}/{len(image_files)}",
            )

        result_text = (
            f"Added {added_total} '{target_name}' label(s) across {changed_files} file(s); "
                f"checked {parent_total} parent label(s)."
        )
        self.auto_head_status_label.setText(result_text)
        QMessageBox.information(self, "SAM3 Nested Labels", result_text)
        logger.info("Nested labels (%s): %s", "geometry" if geometry_only else "SAM3 hybrid", result_text)
    finally:
        self.auto_head_run_button.setEnabled(True)
        self._safe_empty_cache()


def calculate_head_area(self, x, y, w, h):
    # Get the values from the spin boxes
    head_w_factor = float(self.head_width.value()) / \
        100  # this is a percentage
    head_h_factor = float(self.head_height.value()) / 100
    head_x_factor = float(self.head_horizontal.value()) / 100
    head_y_factor = float(self.head_verticle.value()) / 100

    # Calculate the position and size of the head bounding box based on the main bounding box (class_id != 1)
    head_w = int(w * head_w_factor)
    head_h = int(h * head_h_factor)
    # center the head box horizontally
    head_x = x + int(w * head_x_factor) - head_w // 2
    # Adjust this value to control the vertical position of the head bounding box
    head_y = y + int(h * head_y_factor)
    return head_x, head_y, head_w, head_h


def apply_glass_effect(self, image):
    try:
        h, w = image.shape[:2]

        # --- Base layers ---
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        bubbled = cv2.GaussianBlur(image, (21, 21), 30)

        # --- Random refraction distortion (simulated light bending) ---
        # Create a subtle random displacement field
        displacement_strength = 3  # pixels
        flow_x = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32), (61, 61), 0
        ) * displacement_strength
        flow_y = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32), (61, 61), 0
        ) * displacement_strength

        # Create coordinate grid and apply distortion
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + flow_x).astype(np.float32)
        map_y = (map_y + flow_y).astype(np.float32)
        refracted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # --- Chromatic aberration (RGB channel offset) ---
        shift = 1  # pixels of color shift
        b, g, r = cv2.split(refracted)
        b_shifted = np.roll(b, shift, axis=1)
        r_shifted = np.roll(r, -shift, axis=1)
        aberration = cv2.merge((b_shifted, g, r_shifted))

        # --- Random streaks / gloss (like yours) ---
        streaks = np.random.rand(h, w) * 255
        streaks = cv2.GaussianBlur(streaks, (31, 31), 30).astype(np.uint8)
        _, streaks = cv2.threshold(streaks, 220, 255, cv2.THRESH_BINARY)
        streaks = np.stack([streaks] * 3, axis=-1)

        glossy = np.random.rand(h, w) * 255
        glossy = cv2.GaussianBlur(glossy, (11, 11), 60).astype(np.uint8)
        _, glossy = cv2.threshold(glossy, 250, 255, cv2.THRESH_BINARY)
        glossy = np.stack([glossy] * 3, axis=-1)

        # --- Combine all layers ---
        result = cv2.addWeighted(image, 0.75, blurred, 0.25, 0)
        result = cv2.addWeighted(result, 0.8, bubbled, 0.2, 0)
        result = cv2.addWeighted(result, 0.9, refracted, 0.1, 0)
        result = cv2.addWeighted(result, 0.95, aberration, 0.05, 0)
        result = cv2.addWeighted(result, 1.0, streaks, 0.05, 0)
        result = cv2.addWeighted(result, 1.0, glossy, 0.05, 0)

        # --- Apply soft blue tint ---
        tint = np.array([200, 200, 255], dtype=np.uint8)
        tinted_layer = np.full_like(result, tint)
        result = cv2.addWeighted(result, 0.85, tinted_layer, 0.15, 0)

        # --- Final clamp ---
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    except cv2.error as e:
        logger.error(f"OpenCV error occurred in apply_glass_effect: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in apply_glass_effect: {e}")
        return None


def glass_checkbox_state_changed(self, state):
    self.glass_effect = state == QtCore.Qt.Checked
    self.queue_augmentation_preview_update()


def apply_motion_blur_effect(self, image, kernel_size=15, angle=0):
    # Create the motion blur kernel
    k = np.zeros((kernel_size, kernel_size))
    k[(kernel_size - 1) // 2, :] = np.ones(kernel_size)
    k = k * (1.0 / np.sum(k))

    # Create the rotation matrix
    rot_mat = cv2.getRotationMatrix2D(
        (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1)

    # Rotate the kernel
    k_rotated = cv2.warpAffine(k, rot_mat, (kernel_size, kernel_size))

    # Apply the motion blur effect
    return cv2.filter2D(image, -1, k_rotated)


def motion_blur_checkbox_state_changed(self, state):
    if state == QtCore.Qt.Checked:
        self.motion_blur_effect = True
    else:
        self.motion_blur_effect = False
    self.queue_augmentation_preview_update()


def training_augmentation_checkbox_changed(self, attr_name, state):
    setattr(self, attr_name, state == Qt.Checked)
    self.queue_augmentation_preview_update()


def apply_brightness_contrast_augmentation(self, image, alpha=None, beta=None):
    if image is None:
        return None

    alpha = random.uniform(0.75, 1.35) if alpha is None else float(alpha)
    beta = random.uniform(-28, 28) if beta is None else float(beta)
    adjusted = image.astype(np.float32) * alpha + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_hsv_jitter_augmentation(self, image, hue_shift=None, sat_scale=None, val_scale=None):
    if image is None:
        return None

    hue_shift = random.uniform(-8, 8) if hue_shift is None else float(hue_shift)
    sat_scale = random.uniform(0.75, 1.35) if sat_scale is None else float(sat_scale)
    val_scale = random.uniform(0.80, 1.25) if val_scale is None else float(val_scale)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_gaussian_noise_augmentation(self, image, sigma=None):
    if image is None:
        return None

    sigma = random.uniform(3.0, 16.0) if sigma is None else float(sigma)
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_jpeg_compression_augmentation(self, image, quality=None):
    if image is None:
        return None

    quality = random.randint(38, 86) if quality is None else int(quality)
    quality = max(10, min(100, quality))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[:2] != image.shape[:2]:
        return image
    return decoded


def apply_flashbang_effect(self, image, intensity=0.8):
    # Ensure the image is valid
    if image is None:
        raise ValueError("Image is None")
    if not (0.0 <= intensity <= 1.0):
        raise ValueError("Intensity should be between 0.0 and 1.0")
    if image.dtype != np.uint8:
        raise TypeError("Image data type should be uint8")

    # Create a white flash mask
    flash_mask = np.full(image.shape, (255, 255, 255), dtype=np.float32)

    # Apply the flash effect using alpha blending
    flash_effect_alpha = intensity
    flash_image = cv2.addWeighted(image.astype(np.float32), 1 - flash_effect_alpha, flash_mask, flash_effect_alpha, 0)

    # Ensure the flash image is valid
    if flash_image is None:
        raise RuntimeError("Flash image is None")

    return flash_image.astype(np.uint8)


def flash_checkbox_changed(self, state):
    self.flash_effect = state == Qt.Checked
    logger.info(f"Flash effect changed: {self.flash_effect}")
    self.queue_augmentation_preview_update()


def apply_smoke_effect(self, image, intensity=0.62, speed=0.035):
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return image

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    h, w = image.shape[:2]
    self.time += speed

    base_noise = self.generate_smoke_mask(h, w, scale_factor=1.35, time_offset=self.time, octaves=5)
    soft_noise = self.generate_smoke_mask(h, w, scale_factor=3.25, time_offset=(self.time * 0.55) + 19.0, octaves=3)
    smoke_field = np.clip((base_noise * 0.65) + (soft_noise * 0.35), 0.0, 1.0)

    blur_radius = max(9, int(min(w, h) / 18))
    if blur_radius % 2 == 0:
        blur_radius += 1
    smoke_field = cv2.GaussianBlur(smoke_field, (blur_radius, blur_radius), 0)

    image_f = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Smoke in the reference behaves like a full-screen veil: low saturation,
    # low contrast, and very soft detail before the gray layer is applied.
    desaturated = (image_f * 0.32) + (gray_bgr * 0.68)
    low_contrast = np.clip(((desaturated - 0.5) * 0.48) + 0.5, 0.0, 1.0)
    softened = cv2.GaussianBlur(low_contrast, (3, 3), 0)

    haze_color = np.array([0.43, 0.44, 0.43], dtype=np.float32)  # BGR neutral gray.
    density = float(np.clip(intensity, 0.0, 0.85))
    alpha = density + ((smoke_field - 0.5) * 0.20)

    # Slight screen-position bias keeps it from looking like a flat opacity slider.
    x_gradient = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    y_gradient = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    alpha += (x_gradient * 0.035) + ((1.0 - y_gradient) * 0.025)
    alpha = np.clip(alpha, 0.28, 0.88)
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (blur_radius, blur_radius), 0)

    smoky = (softened * (1.0 - alpha[:, :, None])) + (haze_color * alpha[:, :, None])
    return np.clip(smoky * 255.0, 0, 255).astype(np.uint8)


def generate_smoke_mask(self, h, w, scale_factor=1.0, time_offset=0.0, octaves=6):
    if not isinstance(h, int) or not isinstance(w, int):
        raise TypeError(f"Expected integers for h and w, but got {type(h)} and {type(w)}")

    sample_w = min(w, 160)
    sample_h = min(h, 160)

    # Keep a stable noise field and move through it with fractional offsets.
    noise = PerlinNoise(octaves=octaves, seed=1337)
    drift_x = float(time_offset) * 0.55
    drift_y = float(time_offset) * 0.35
    sample_scale = max(1e-6, 20 * float(scale_factor))

    # Create a small noise field and upscale it. This keeps smoke usable on large images.
    noise_array = np.zeros((sample_h, sample_w), dtype=np.float32)

    if sample_h > 0 and sample_w > 0:
        from perlin_noise.tools import fade

        # Keep the library's seeded gradient vectors and scalar fade values.
        # Evaluate their four corner contributions together instead of calling
        # Python noise/weight/distance methods for every sampled pixel.
        ys = np.asarray([((i / sample_scale) + drift_y) * noise.octaves for i in range(sample_h)])
        xs = np.asarray([((j / sample_scale) + drift_x) * noise.octaves for j in range(sample_w)])
        by, bx = np.floor(ys).astype(np.int64), np.floor(xs).astype(np.int64)
        gy = np.unique(np.concatenate((by, by + 1)))
        gx = np.unique(np.concatenate((bx, bx + 1)))
        gradients = np.asarray([
            [noise.get_from_cache_of_create_new((int(y), int(x)), None).vec for x in gx]
            for y in gy
        ], dtype=np.float64)
        sampled = np.zeros((sample_h, sample_w), dtype=np.float64)
        for corner_y in (0, 1):
            yd = ys - (by + corner_y)
            yw = np.asarray([fade(1 - abs(float(value))) for value in yd])
            iy = np.searchsorted(gy, by + corner_y)
            for corner_x in (0, 1):
                xd = xs - (bx + corner_x)
                xw = np.asarray([fade(1 - abs(float(value))) for value in xd])
                ix = np.searchsorted(gx, bx + corner_x)
                vectors = gradients[iy[:, None], ix[None, :]]
                sampled += (yw[:, None] * xw[None, :]) * (
                    vectors[:, :, 0] * yd[:, None] + vectors[:, :, 1] * xd[None, :]
                )
        noise_array[:] = sampled

    # Normalize noise to [0,1]
    min_val, max_val = noise_array.min(), noise_array.max()
    if max_val - min_val > 0:  # Avoid division by zero
        noise_array = (noise_array - min_val) / (max_val - min_val)

    if sample_w != w or sample_h != h:
        noise_array = cv2.resize(noise_array, (w, h), interpolation=cv2.INTER_CUBIC)

    return np.clip(noise_array, 0.0, 1.0).astype(np.float32)


def smoke_checkbox_changed(self, state):
    self.smoke_effect = state == Qt.Checked
    logger.info(f"Smoke effect changed: {self.smoke_effect}")
    self.queue_augmentation_preview_update()


def sight_picture_checkbox_changed(self, state):
    self.sight_picture_effect = state == Qt.Checked
    self.queue_augmentation_preview_update()


@staticmethod
def _normalize_sight_aim_point(value, fallback=(0.50, 0.18)):
    try:
        if isinstance(value, dict):
            value = (value.get("x"), value.get("y"))
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            x = max(0.02, min(0.98, float(value[0])))
            y = max(0.02, min(0.98, float(value[1])))
            return x, y
    except Exception:
        pass
    return fallback


def sight_overlay_metadata(self):
    metadata_path = os.path.join(APP_DIR, "images", "sight_overlays.json")
    metadata = read_json_file(metadata_path, default={}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    files = metadata.get("files", {})
    if not isinstance(files, dict):
        files = {}
    return metadata, files


def sight_overlay_asset_specs(self):
    asset_dir = os.path.join(APP_DIR, "images")
    if not os.path.isdir(asset_dir):
        return []

    metadata, file_metadata = self.sight_overlay_metadata()
    default_aim = self._normalize_sight_aim_point(
        metadata.get("default_aim_point"),
        fallback=(0.50, 0.18),
    )
    known_aim_points = {
        "1.webp": (0.50, 0.19),
        "2.png": (0.50, 0.20),
        "3.png": (0.50, 0.19),
        "4.png": (0.50, 0.48),
        "5.png": (0.50, 0.24),
        "6.png": (0.58, 0.15),
        "7.png": (0.19, 0.14),
        "8.png": (0.56, 0.20),
        "9.png": (0.57, 0.20),
        "11.webp": (0.50, 0.16),
        "12.webp": (0.50, 0.16),
    }
    specs = []
    for name in sorted(os.listdir(asset_dir), key=str.lower):
        path = os.path.join(asset_dir, name)
        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(name)[1].lower()
        if extension not in DATASET_IMAGE_SUFFIXES:
            continue

        entry = file_metadata.get(name, {})
        if not isinstance(entry, dict):
            entry = {}
        if entry.get("enabled", True) is False:
            continue

        aim_point = self._normalize_sight_aim_point(
            entry.get("aim_point", known_aim_points.get(name, default_aim)),
            fallback=known_aim_points.get(name, default_aim),
        )
        specs.append((path, aim_point))

    return specs


@staticmethod
def _strengthen_sight_alpha(alpha):
    alpha_float = np.asarray(alpha, dtype=np.float32)
    visible = alpha_float > 4.0
    if not np.any(visible):
        return np.zeros_like(alpha, dtype=np.uint8)

    normalized = np.clip(alpha_float / 255.0, 0.0, 1.0)
    boosted = np.power(normalized, 0.42) * 255.0
    boosted[alpha_float >= 16.0] = np.maximum(boosted[alpha_float >= 16.0], 220.0)
    boosted[alpha_float >= 80.0] = np.maximum(boosted[alpha_float >= 80.0], 245.0)
    boosted[~visible] = 0.0
    return np.clip(boosted, 0, 255).astype(np.uint8)


@staticmethod
def _trim_sight_overlay_bgra(bgra, aim_point):
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        return None, aim_point

    alpha = bgra[:, :, 3]
    ys, xs = np.where(alpha > 3)
    if xs.size == 0 or ys.size == 0:
        return None, aim_point

    h, w = bgra.shape[:2]
    x1 = max(0, int(xs.min()) - 2)
    y1 = max(0, int(ys.min()) - 2)
    x2 = min(w, int(xs.max()) + 3)
    y2 = min(h, int(ys.max()) + 3)
    if x2 <= x1 or y2 <= y1:
        return None, aim_point

    aim_x = float(aim_point[0]) * w
    aim_y = float(aim_point[1]) * h
    cropped = bgra[y1:y2, x1:x2].copy()
    cropped_h, cropped_w = cropped.shape[:2]
    adjusted_aim = (
        max(0.0, min(1.0, (aim_x - x1) / max(1.0, float(cropped_w)))),
        max(0.0, min(1.0, (aim_y - y1) / max(1.0, float(cropped_h)))),
    )
    return cropped, adjusted_aim


def load_sight_overlay_assets(self):
    specs = self.sight_overlay_asset_specs()
    signature = []
    for path, aim_point in specs:
        try:
            signature.append((path, os.path.getmtime(path), os.path.getsize(path), aim_point))
        except OSError:
            continue
    signature = tuple(signature)

    if (
        getattr(self, "_sight_overlay_asset_signature", None) == signature
        and getattr(self, "_sight_overlay_assets", None)
    ):
        return self._sight_overlay_assets

    assets = []
    for path, aim_point in specs:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None or raw.size == 0:
            continue

        if raw.ndim != 3 or raw.shape[2] != 4:
            logger.warning(
                "Skipping sight overlay without transparency: %s",
                os.path.basename(path),
            )
            continue

        bgra = raw.copy()
        alpha = bgra[:, :, 3]
        if int(np.max(alpha)) <= 8:
            logger.warning("Skipping transparent/empty sight overlay: %s", os.path.basename(path))
            continue

        transparent_ratio = float(np.count_nonzero(alpha < 250)) / float(alpha.size)
        if transparent_ratio <= 0.002:
            logger.warning(
                "Skipping fully opaque sight overlay: %s",
                os.path.basename(path),
            )
            continue

        bgra[:, :, 3] = self._strengthen_sight_alpha(alpha)
        bgra, adjusted_aim = self._trim_sight_overlay_bgra(bgra, aim_point)
        if bgra is None:
            continue

        alpha = bgra[:, :, 3]
        visible_ratio = float(np.count_nonzero(alpha > 8)) / float(alpha.size)
        if visible_ratio <= 0.01:
            continue

        assets.append(
            {
                "path": path,
                "name": os.path.basename(path),
                "image": bgra,
                "aim_point": adjusted_aim,
            }
        )

    self._sight_overlay_asset_signature = signature
    self._sight_overlay_assets = assets
    if not assets:
        logger.warning("No usable sight overlay assets found in %s", os.path.join(APP_DIR, "images"))
    return assets


@staticmethod
def _blend_bgra_overlay(base_image, overlay_bgra, x, y, opacity=1.0, require_full_fit=False):
    if (
        base_image is None
        or overlay_bgra is None
        or not isinstance(base_image, np.ndarray)
        or not isinstance(overlay_bgra, np.ndarray)
        or overlay_bgra.ndim != 3
        or overlay_bgra.shape[2] != 4
    ):
        return base_image

    h, w = base_image.shape[:2]
    oh, ow = overlay_bgra.shape[:2]
    if h <= 0 or w <= 0 or oh <= 0 or ow <= 0:
        return base_image

    x = int(round(float(x)))
    y = int(round(float(y)))

    if require_full_fit and (x < 0 or y < 0 or x + ow > w or y + oh > h):
        return None

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + ow)
    y2 = min(h, y + oh)
    if x2 <= x1 or y2 <= y1:
        return base_image

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    overlay = overlay_bgra[oy1:oy2, ox1:ox2]
    alpha = (overlay[:, :, 3].astype(np.float32) / 255.0) * max(0.0, min(1.0, float(opacity)))
    alpha = alpha[:, :, None]

    roi = base_image[y1:y2, x1:x2].astype(np.float32)
    blended = overlay[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
    base_image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return base_image


def _augmentation_class_name_for_id(self, class_id):
    for mapping_name in ("id_to_class", "class_id_to_name", "class_mapping"):
        mapping = getattr(self, mapping_name, None)
        if isinstance(mapping, dict):
            value = mapping.get(class_id, mapping.get(str(class_id)))
            if value is not None:
                return str(value).strip().lower()
    return ""


def _sight_target_center(self, labels, pose_data=None):
    candidates = []
    preferred = []
    target_terms = ("person", "player", "human", "enemy", "operator", "soldier", "pedestrian")

    for label in list(labels or []):
        parts = str(label).strip().split()
        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))
        except Exception:
            class_id = None

        center = self._augmentation_label_center(label, pose_data)
        if center is None:
            continue

        x, y = center
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            continue

        candidates.append((float(x), float(y)))
        class_name = self._augmentation_class_name_for_id(class_id) if class_id is not None else ""
        if any(term in class_name for term in target_terms):
            preferred.append((float(x), float(y)))

    if preferred:
        return random.choice(preferred)
    if candidates:
        return random.choice(candidates)
    return random.uniform(0.42, 0.58), random.uniform(0.34, 0.55)


def _sight_label_cover_boxes(
    self,
    labels,
    image_width,
    image_height,
    pose_data=None,
    polygon_parse_preference=None,
):
    if not labels or image_width <= 0 or image_height <= 0:
        return []

    boxes = []
    if polygon_parse_preference is None:
        polygon_parse_preference = self._polygon_label_parse_preference()
    previous_expected = getattr(BoundingBox, "EXPECTED_KEYPOINT_COUNT", None)
    expected_keypoints = self._augmentation_keypoint_count(pose_data)

    try:
        BoundingBox.EXPECTED_KEYPOINT_COUNT = expected_keypoints or None

        for label in labels:
            try:
                box_obj = BoundingBox.from_str(
                    str(label),
                    preferred_polygon_type=polygon_parse_preference,
                )
                if not box_obj:
                    continue

                xyxy = self._label_seed_bbox(box_obj, image_width, image_height)
                if xyxy is None:
                    continue

                x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[:4]]
                x1 = max(0, min(image_width - 1, x1))
                y1 = max(0, min(image_height - 1, y1))
                x2 = max(0, min(image_width, x2))
                y2 = max(0, min(image_height, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                class_id = int(box_obj.class_id)
                boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        class_id,
                        self._label_type(box_obj),
                        self._augmentation_class_name_for_id(class_id),
                    )
                )
            except Exception as e:
                logger.debug("Skipped sight coverage label check for '%s': %s", label, e)
    finally:
        BoundingBox.EXPECTED_KEYPOINT_COUNT = previous_expected

    return boxes


@staticmethod
def _sight_overlay_overcovers_labels(
    overlay_bgra,
    overlay_x,
    overlay_y,
    label_boxes,
    max_cover_ratio=0.85,
    alpha_threshold=24,
):
    if not label_boxes:
        return False
    if (
        overlay_bgra is None
        or not isinstance(overlay_bgra, np.ndarray)
        or overlay_bgra.ndim != 3
        or overlay_bgra.shape[2] != 4
    ):
        return False

    oh, ow = overlay_bgra.shape[:2]
    if oh <= 0 or ow <= 0:
        return False

    overlay_x = int(round(float(overlay_x)))
    overlay_y = int(round(float(overlay_y)))
    cover_limit = max(0.0, min(1.0, float(max_cover_ratio)))
    target_terms = ("person", "player", "human", "enemy", "operator", "soldier", "pedestrian")

    for label_box in label_boxes:
        x1, y1, x2, y2 = label_box[:4]
        label_type = str(label_box[5]).lower() if len(label_box) > 5 else ""
        class_name = str(label_box[6]).lower() if len(label_box) > 6 else ""
        label_cover_limit = cover_limit
        if label_type in {"segmentation", "obb"}:
            label_cover_limit = min(label_cover_limit, 0.65)
        elif label_type == "keypoints":
            label_cover_limit = min(label_cover_limit, 0.75)
        if any(term in class_name for term in target_terms):
            label_cover_limit = min(label_cover_limit, 0.75)

        ix1 = max(int(x1), overlay_x)
        iy1 = max(int(y1), overlay_y)
        ix2 = min(int(x2), overlay_x + ow)
        iy2 = min(int(y2), overlay_y + oh)
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        ox1 = ix1 - overlay_x
        oy1 = iy1 - overlay_y
        ox2 = ox1 + (ix2 - ix1)
        oy2 = oy1 + (iy2 - iy1)
        alpha_crop = overlay_bgra[oy1:oy2, ox1:ox2, 3]
        if alpha_crop.size == 0:
            continue

        covered_pixels = int(np.count_nonzero(alpha_crop > alpha_threshold))
        label_area = max(1, (int(x2) - int(x1)) * (int(y2) - int(y1)))
        if covered_pixels / float(label_area) >= label_cover_limit:
            return True

    return False


@staticmethod
def _sight_overlay_covers_normalized_point(
    overlay_bgra,
    overlay_x,
    overlay_y,
    x_norm,
    y_norm,
    image_width,
    image_height,
    alpha_threshold=24,
):
    if (
        overlay_bgra is None
        or not isinstance(overlay_bgra, np.ndarray)
        or overlay_bgra.ndim != 3
        or overlay_bgra.shape[2] != 4
        or image_width <= 0
        or image_height <= 0
    ):
        return False

    try:
        px = int(round(max(0.0, min(1.0, float(x_norm))) * max(1, image_width - 1)))
        py = int(round(max(0.0, min(1.0, float(y_norm))) * max(1, image_height - 1)))
    except Exception:
        return False

    ox = px - int(round(float(overlay_x)))
    oy = py - int(round(float(overlay_y)))
    oh, ow = overlay_bgra.shape[:2]
    if ox < 0 or oy < 0 or ox >= ow or oy >= oh:
        return False

    return int(overlay_bgra[oy, ox, 3]) > int(alpha_threshold)


@staticmethod
def _format_pose_label_line(class_id, values):
    bbox_part = " ".join(f"{float(v):.6f}" for v in values[:4])
    keypoint_part = " ".join(
        f"{float(values[i]):.6f} {float(values[i + 1]):.6f} {int(float(values[i + 2]))}"
        for i in range(4, len(values), 3)
    )
    return f"{int(class_id)} {bbox_part} {keypoint_part}".strip()


def _sight_pose_labels_for_overlay(
    self,
    labels,
    overlay_bgra,
    overlay_x,
    overlay_y,
    image_width,
    image_height,
    pose_data=None,
    max_covered_ratio=0.60,
    max_covered_count=8,
    min_visible_for_ratio=4,
):
    labels = list(labels or [])
    if not labels:
        return labels, False, 0

    keypoint_count = self._augmentation_keypoint_count(pose_data)
    updated_labels = []
    changed_keypoints = 0

    for label in labels:
        stripped = str(label).strip()
        parts = stripped.split()
        if len(parts) < 5:
            updated_labels.append(stripped)
            continue

        try:
            class_id = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except Exception:
            updated_labels.append(stripped)
            continue

        if not self._is_pose_label_values(values, keypoint_count):
            updated_labels.append(stripped)
            continue

        adjusted = values[:]
        visible_count = 0
        covered_count = 0
        for i in range(4, len(values), 3):
            visibility = int(float(values[i + 2]))
            if visibility != 2:
                continue

            visible_count += 1
            if self._sight_overlay_covers_normalized_point(
                overlay_bgra,
                overlay_x,
                overlay_y,
                values[i],
                values[i + 1],
                image_width,
                image_height,
            ):
                adjusted[i + 2] = 1
                covered_count += 1

        if covered_count:
            changed_keypoints += covered_count
            covered_ratio = covered_count / float(max(1, visible_count))
            if (
                covered_count >= max_covered_count
                or (
                    visible_count >= min_visible_for_ratio
                    and covered_ratio >= max_covered_ratio
                )
            ):
                return labels, True, changed_keypoints

        updated_labels.append(self._format_pose_label_line(class_id, adjusted))

    return updated_labels, False, changed_keypoints


def apply_sight_overlay_augmentation(
    self,
    image,
    labels=None,
    pose_data=None,
    return_labels=False,
    polygon_parse_preference=None,
):
    labels = list(labels or [])

    def result(result_image, result_labels=None):
        if return_labels:
            return result_image, list(result_labels if result_labels is not None else labels)
        return result_image

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return result(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        return result(image)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    assets = self.load_sight_overlay_assets()
    if not assets:
        logger.warning("Sight augmentation selected, but no transparent sight assets were loaded.")
        return result(image)

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return result(image)

    target_x_norm, _target_y_norm = self._sight_target_center(labels, pose_data)
    label_cover_boxes = self._sight_label_cover_boxes(
        labels,
        w,
        h,
        pose_data,
        polygon_parse_preference=polygon_parse_preference,
    )
    attempts = max(18, min(60, len(assets) * 6))
    min_cover_width = max(32, min(max(1, w - 4), int(w * 0.34)))
    min_cover_height = max(32, min(max(1, h - 4), int(h * 0.28)))
    margin = max(2, int(min(w, h) * 0.004))
    coverage_rejections = 0
    pose_rejections = 0

    for _attempt in range(attempts):
        asset = random.choice(assets)
        overlay = asset["image"].copy()
        aim_x_norm, _aim_y_norm = asset["aim_point"]

        if random.random() < 0.35:
            overlay = cv2.flip(overlay, 1)
            aim_x_norm = 1.0 - float(aim_x_norm)

        oh, ow = overlay.shape[:2]
        if oh <= 0 or ow <= 0:
            continue
        aspect = max(0.05, float(ow) / float(oh))

        target_x = (float(target_x_norm) + random.uniform(-0.025, 0.025)) * w

        max_height_by_frame = max(1, h)
        max_height_by_width = int((w - (2 * margin)) / aspect)
        max_height_by_style = int(h * random.uniform(0.56, 0.82))
        max_allowed_height = min(max_height_by_frame, max_height_by_width, max_height_by_style)
        if max_allowed_height < 16:
            continue

        preferred_height = int(h * random.uniform(0.52, 0.74))
        target_height = min(max_allowed_height, preferred_height)
        target_width = int(target_height * aspect)

        if target_width > (w - (2 * margin)):
            target_width = w - (2 * margin)
            target_height = int(target_width / aspect)

        if target_width <= 0 or target_height <= 0:
            continue

        if target_width < min_cover_width and target_height < min_cover_height:
            continue

        interpolation = cv2.INTER_AREA if target_width < ow or target_height < oh else cv2.INTER_LINEAR
        resized = cv2.resize(overlay, (target_width, target_height), interpolation=interpolation)

        brightness = random.uniform(0.88, 1.08)
        offset = random.uniform(-4.0, 4.0)
        resized[:, :, :3] = np.clip(
            resized[:, :, :3].astype(np.float32) * brightness + offset,
            0,
            255,
        ).astype(np.uint8)

        if random.random() < 0.12:
            resized[:, :, :3] = cv2.GaussianBlur(resized[:, :, :3], (3, 3), 0)

        x_min = margin
        x_max = w - target_width - margin
        if x_max < x_min:
            continue

        x = int(round(target_x - (float(aim_x_norm) * target_width)))
        x += int(random.uniform(-0.025, 0.025) * w)
        x = max(x_min, min(x_max, x))

        y = h - target_height

        if self._sight_overlay_overcovers_labels(resized, x, y, label_cover_boxes):
            coverage_rejections += 1
            continue

        updated_labels, pose_rejected, changed_keypoints = self._sight_pose_labels_for_overlay(
            labels,
            resized,
            x,
            y,
            w,
            h,
            pose_data,
        )
        if pose_rejected:
            pose_rejections += 1
            continue

        blended = self._blend_bgra_overlay(
            image,
            resized,
            x,
            y,
            opacity=random.uniform(0.98, 1.0),
            require_full_fit=True,
        )
        if blended is not None:
            if changed_keypoints:
                logger.info(
                    "Sight augmentation marked %d visible keypoint(s) as occluded.",
                    changed_keypoints,
                )
            return result(blended, updated_labels)

    if coverage_rejections or pose_rejections:
        logger.warning(
            "Skipping sight augmentation: no label-safe full-fit placement found "
                "(%d bbox/shape rejection(s), %d pose rejection(s)).",
            coverage_rejections,
            pose_rejections,
        )
    else:
        logger.warning("Skipping sight augmentation: no full-fit sight placement found for this image.")
    return result(image)


def make_copy_checkbox_changed(self, state):
    self.make_copy_enabled = state == Qt.Checked
    self.queue_augmentation_preview_update()


def set_flip_images(self, state):
    self.flip_images = state == Qt.Checked
    self.queue_augmentation_preview_update()


def selected_augmentation_percent(self):
    widget = getattr(self, "augmentation_percent_spinbox", None)
    if widget is not None:
        return max(1, min(100, int(widget.value())))
    return max(1, min(100, int(getattr(self, "augmentation_percent", 100))))


def selected_augmentation_output_format_value(self):
    combo = getattr(self, "image_format_combo", None)
    if isinstance(combo, QtWidgets.QComboBox):
        value = combo.currentData()
        if value is None:
            value = combo.currentText()
        return normalize_augmentation_output_format(value, default="source")
    return normalize_augmentation_output_format(
        getattr(self, "augmentation_output_format", "source"),
        default="source",
    )


def augmentation_output_extension_for_image(self, image_path):
    output_format = self.selected_augmentation_output_format_value()
    if output_format == "source":
        return normalize_image_extension(os.path.splitext(str(image_path or ""))[1], default=".jpg")
    return normalize_image_extension(output_format, default=".jpg")


def sampled_augmentation_items(self, image_files_map):
    items = list(image_files_map.items())
    if not items:
        return []

    percent = self.selected_augmentation_percent()
    if percent >= 100:
        return items

    sample_count = max(1, math.ceil(len(items) * (percent / 100.0)))
    sample_count = min(sample_count, len(items))
    selected_indexes = sorted(random.sample(range(len(items)), sample_count))
    return [items[index] for index in selected_indexes]


def sampled_sam_augmentation_images(self, image_files):
    clean_files = []
    seen = set()

    for image_file in list(image_files or []):
        if not image_file or self.is_placeholder_file(image_file):
            continue

        image_file = self.normalize_path(image_file)
        if image_file in seen or not os.path.isfile(image_file):
            continue

        if os.path.splitext(image_file)[1].lower() not in DATASET_IMAGE_SUFFIXES:
            continue

        clean_files.append(image_file)
        seen.add(image_file)

    if not clean_files:
        return []

    percent = self.selected_augmentation_percent()
    if percent >= 100:
        return clean_files

    sample_count = max(1, math.ceil(len(clean_files) * (percent / 100.0)))
    sample_count = min(sample_count, len(clean_files))
    selected_indexes = sorted(random.sample(range(len(clean_files)), sample_count))
    return [clean_files[index] for index in selected_indexes]


def collect_current_augmentation_dataset(self):
    dataset_dir = self.current_dataset_directory()
    if not dataset_dir:
        return "", [], {}
    dataset_dir = self.normalize_path(dataset_dir)

    image_files = []
    seen = set()

    for image_file in list(getattr(self, "image_files", []) or []):
        if not image_file or self.is_placeholder_file(image_file):
            continue
        image_file = self.normalize_path(image_file)
        if image_file in seen or not os.path.isfile(image_file):
            continue
        try:
            common_dir = self.normalize_path(os.path.commonpath([dataset_dir, image_file]))
            if common_dir != dataset_dir:
                continue
        except Exception:
            continue
        if os.path.splitext(image_file)[1].lower() in DATASET_IMAGE_SUFFIXES:
            image_files.append(image_file)
            seen.add(image_file)

    if not image_files:
        image_files = [
            self.normalize_path(path)
            for path in self.get_image_files(dataset_dir, extensions=DATASET_IMAGE_SUFFIXES)
            if not self.is_placeholder_file(path)
        ]

    label_files_map = {}
    try:
        for entry in os.scandir(dataset_dir):
            if not entry.is_file():
                continue
            name = entry.name
            if not name.lower().endswith(".txt") or name.lower() == "classes.txt":
                continue
            if self.is_placeholder_file(name) or self.is_placeholder_file(entry.path):
                continue
            label_files_map[os.path.splitext(name)[0]] = self.normalize_path(entry.path)
    except Exception as e:
        logger.warning(f"Failed collecting augmentation label files: {e}")

    return dataset_dir, sorted(image_files), label_files_map


def open_augmentation_preview_viewer(self, source=None):
    """Open any Augmentation preview in the shared zoom/pan viewer."""
    if source is getattr(self, "auto_head_preview_label", None):
        pixmap = getattr(self, "_auto_head_preview_full_pixmap", None)
        title = "SAM3 Nested Label Preview"
    else:
        pixmap = getattr(self, "_augmentation_preview_full_pixmap", None)
        title = "Augmentation Preview"

    if not isinstance(pixmap, QPixmap) or pixmap.isNull():
        fallback = source.pixmap() if isinstance(source, QtWidgets.QLabel) else None
        pixmap = QPixmap(fallback) if isinstance(fallback, QPixmap) else QPixmap()
    if pixmap.isNull():
        return

    viewer = getattr(self, "_augmentation_preview_viewer", None)
    if not isinstance(viewer, AugmentationPreviewViewer):
        viewer = AugmentationPreviewViewer(self)
        self._augmentation_preview_viewer = viewer
    if not viewer.set_preview(QPixmap(pixmap), title):
        return
    viewer.show()
    viewer.raise_()
    viewer.activateWindow()


def augmentation_preview_visible(self):
    preview = getattr(self, "augmentation_preview_label", None)
    if not isinstance(preview, QtWidgets.QLabel):
        return False

    try:
        return bool(preview.isVisible() and preview.isEnabled())
    except RuntimeError:
        return False


def setup_augmentation_preview_timer(self):
    if getattr(self, "_augmentation_preview_timer_ready", False):
        return

    timer = QTimer(self)
    timer.setSingleShot(True)
    timer.timeout.connect(self.update_augmentation_preview)
    self._augmentation_preview_timer = timer
    self._augmentation_preview_timer_ready = True


def queue_augmentation_preview_update(self):
    if not self.augmentation_preview_visible():
        return

    timer = getattr(self, "_augmentation_preview_timer", None)
    if timer is not None:
        try:
            timer.start(120)
            return
        except RuntimeError:
            pass

    if hasattr(self, "update_augmentation_preview"):
        QTimer.singleShot(0, self.update_augmentation_preview)


def augmentation_preview_source(self):
    dataset_dir = self.current_dataset_directory()
    if not dataset_dir:
        return "", "", ""
    dataset_dir = self.normalize_path(dataset_dir)

    def inside_dataset(path):
        if not path:
            return False
        try:
            common_dir = self.normalize_path(os.path.commonpath([dataset_dir, path]))
            return common_dir == dataset_dir
        except Exception:
            return False

    current_file = self.normalize_path(getattr(self, "current_file", "") or "")
    image_path = ""

    if (
        current_file
        and inside_dataset(current_file)
        and os.path.splitext(current_file)[1].lower() in DATASET_IMAGE_SUFFIXES
        and os.path.isfile(current_file)
    ):
        image_path = current_file

    if not image_path:
        for candidate in getattr(self, "image_files", []) or []:
            candidate = self.normalize_path(candidate)
            if (
                candidate
                and not self.is_placeholder_file(candidate)
                and inside_dataset(candidate)
                and os.path.splitext(candidate)[1].lower() in DATASET_IMAGE_SUFFIXES
                and os.path.isfile(candidate)
            ):
                image_path = candidate
                break

    if not image_path:
        image_files = self.get_image_files(dataset_dir, extensions=DATASET_IMAGE_SUFFIXES)
        image_path = self.normalize_path(image_files[0]) if image_files else ""

    label_path = ""
    if image_path:
        candidate_label = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(candidate_label):
            label_path = self.normalize_path(candidate_label)

    return dataset_dir, image_path, label_path


def update_augmentation_preview(self):
    if self._object_clarity_augmentation_selected() and not gpu_model_allowed(self, "clarity_augmentation"):
        return
    if (getattr(self, "_clarity_augmentation_busy", False)
            or getattr(self, "_augmentation_job", None) is not None
            or getattr(self, "_sam3_batch_active", False)):
        self._clarity_augmentation_preview_pending = True
        return
    if not self.augmentation_preview_visible():
        return

    preview = getattr(self, "augmentation_preview_label", None)
    if not isinstance(preview, QtWidgets.QLabel):
        return

    _dataset_dir, image_path, label_path = self.augmentation_preview_source()
    if not image_path:
        preview.setPixmap(QPixmap())
        preview.setText("")
        return

    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        preview.setPixmap(QPixmap())
        preview.setText("No Preview")
        return

    labels = []
    if label_path and os.path.exists(label_path):
        try:
            with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
                labels = self._valid_annotation_lines(f.readlines())
        except Exception:
            labels = []

    try:
        image, _labels = self.apply_selected_augmentation_effects(image.copy(), labels)
    except Exception as error:
        logger.exception("Augmentation preview failed: %s", error)
        preview.setPixmap(QPixmap())
        preview.setText("Preview unavailable; check the log")
        return
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        preview.setPixmap(QPixmap())
        preview.setText("No Preview")
        return

    qimage = self.cv2_to_qimage(image)
    if qimage is None:
        preview.setPixmap(QPixmap())
        preview.setText("No Preview")
        return

    pixmap = QPixmap.fromImage(qimage)
    self._augmentation_preview_full_pixmap = QPixmap(pixmap)
    preview.setPixmap(
        pixmap.scaled(
            preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    )
    preview.setText("")


def _write_classes_file(self, out_dir: str):
    """
        Writes classes.txt to `out_dir` using current class list if available,
        otherwise tries to copy from the active augmentation/dataset folder.
        """
    classes_path = resolve_dataset_classes_path(out_dir, for_write=True, migrate=True)
    if os.path.exists(classes_path):
        return  # already there

    lines = []
    # Best: reconstruct from self.id_to_class (id -> name)
    try:
        if hasattr(self, "id_to_class") and self.id_to_class:
            max_id = max(self.id_to_class.keys())
            lines = [self.id_to_class.get(i, f"cls_{i}") for i in range(max_id + 1)]
    except Exception:
        pass

    # Fallback: try to copy classes.txt from the selected source dir.
    if not lines:
        for probe_dir in [
            getattr(self, "image_directory", None),
            os.path.dirname(getattr(self, "current_file", "") or ""),
        ]:
            if not probe_dir:
                continue
            candidate = resolve_dataset_classes_path(probe_dir, migrate=True)
            if os.path.isfile(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        raw = [ln.rstrip("\n") for ln in f.readlines()]
                    if raw:
                        lines = raw
                        break
                except Exception:
                    pass

    # Last fallback: write a minimal single-class file to avoid crashes
    if not lines:
        lines = ["object"]

    os.makedirs(out_dir, exist_ok=True)
    with open(classes_path, "w", encoding="utf-8") as f:
        for name in lines:
            f.write(name.strip() + "\n")


@pyqtSlot(int)

def update_progress_bar(self, value):
    maximum = self.label_progress.maximum() if hasattr(self, "label_progress") else 100
    self.set_main_progress(value, maximum)


def set_progress_bar_maximum(self, max_value):
    if hasattr(self, "label_progress"):
        self.label_progress.setMaximum(max(1, int(max_value)))


def _load_augmentation_pose_data(self):
    candidate_dirs = [
        getattr(self, "image_directory", None),
    ]

    try:
        if getattr(self, "current_file", None):
            candidate_dirs.append(os.path.dirname(self.current_file))
    except Exception:
        pass

    seen = set()
    for directory in candidate_dirs:
        if not directory or directory in seen:
            continue
        seen.add(directory)

        for file_name in ("points.json", "pose.json"):
            path = resolve_dataset_metadata_path(directory, file_name, migrate=True)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning(f"Failed to read augmentation pose data from {path}: {e}")

    return {}


def _augmentation_keypoint_count(self, pose_data=None):
    pose_data = pose_data or {}

    keypoints = pose_data.get("keypoints")
    if isinstance(keypoints, list) and keypoints:
        return len(keypoints)

    try:
        count = self.keypoint_list.rowCount()
        if count > 0:
            return count
    except Exception:
        pass

    return 0


def _augmentation_flip_idx(self, keypoint_count, pose_data=None):
    pose_data = pose_data or {}
    raw_flip_idx = pose_data.get("flip_idx")

    if isinstance(raw_flip_idx, list) and len(raw_flip_idx) == keypoint_count:
        try:
            flip_idx = [int(v) for v in raw_flip_idx]
            if all(0 <= idx < keypoint_count for idx in flip_idx):
                return flip_idx
        except Exception:
            pass

    return list(range(keypoint_count))


@staticmethod

def _looks_like_normalized_bbox(values):
    if len(values) < 4:
        return False

    x_center, y_center, width, height = values[:4]
    return (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    )


@staticmethod

def _format_augmented_label(class_id, values):
    return f"{class_id} " + " ".join(f"{float(v):.6f}" for v in values) + "\n"


def _is_pose_label_values(self, values, keypoint_count):
    if not self._looks_like_normalized_bbox(values):
        return False

    if len(values) <= 4 or (len(values) - 4) % 3 != 0:
        return False

    visibility_values = values[6::3]
    valid_visibility = bool(visibility_values) and all(
        float(v).is_integer() and int(float(v)) in (0, 1, 2)
        for v in visibility_values
    )

    if not valid_visibility:
        return False

    if keypoint_count > 0 and len(values) != 4 + (keypoint_count * 3):
        logger.warning(
            f"Pose label has {(len(values) - 4) // 3} keypoints, "
                f"but the loaded point list expects {keypoint_count}. Keeping it as pose anyway."
        )

    return True


def _flip_label_line_horizontal(self, label, pose_data=None):
    parts = label.strip().split()
    if not parts:
        return None

    try:
        class_id = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
    except Exception as e:
        logger.warning(f"Skipped malformed label during horizontal flip: {label.strip()} ({e})")
        return None

    keypoint_count = self._augmentation_keypoint_count(pose_data)

    if self._is_pose_label_values(values, keypoint_count):
        actual_keypoint_count = (len(values) - 4) // 3

        flipped = values[:]
        flipped[0] = 1.0 - flipped[0]

        keypoints = []
        for i in range(4, len(values), 3):
            x, y, visibility = values[i], values[i + 1], int(float(values[i + 2]))
            if visibility in (1, 2):
                x = 1.0 - x
            keypoints.append((x, y, visibility))

        flip_idx = self._augmentation_flip_idx(actual_keypoint_count, pose_data)
        if len(flip_idx) == len(keypoints):
            keypoints = [keypoints[idx] for idx in flip_idx]

        flipped = flipped[:4]
        for x, y, visibility in keypoints:
            flipped.extend([x, y, visibility])

        bbox_part = " ".join(f"{float(v):.6f}" for v in flipped[:4])
        keypoint_part = " ".join(
            f"{float(x):.6f} {float(y):.6f} {int(visibility)}"
            for x, y, visibility in keypoints
        )
        return f"{class_id} {bbox_part} {keypoint_part}\n"

    if len(values) in (4, 5) and self._looks_like_normalized_bbox(values):
        flipped = values[:]
        flipped[0] = 1.0 - flipped[0]
        return self._format_augmented_label(class_id, flipped)

    if len(values) >= 6 and len(values) % 2 == 0:
        flipped = values[:]
        for i in range(0, len(flipped), 2):
            flipped[i] = 1.0 - flipped[i]
        return self._format_augmented_label(class_id, flipped)

    logger.warning(f"Skipped unsupported label format during horizontal flip: {label.strip()}")
    return None


def _augmentation_label_center(self, label, pose_data=None):
    parts = label.strip().split()
    if len(parts) < 5:
        return None

    try:
        values = [float(v) for v in parts[1:]]
    except Exception:
        return None

    keypoint_count = self._augmentation_keypoint_count(pose_data)

    if len(values) in (4, 5) and self._looks_like_normalized_bbox(values):
        return values[0], values[1]

    if self._is_pose_label_values(values, keypoint_count):
        return values[0], values[1]

    if len(values) >= 6 and len(values) % 2 == 0:
        xs = values[::2]
        ys = values[1::2]
        if xs and ys:
            return sum(xs) / len(xs), sum(ys) / len(ys)

    return None


def copy_original_augmentation_files(self, image_files_map, label_files_map, output_directory):
    output_path = os.path.join(output_directory, "copy")
    os.makedirs(output_path, exist_ok=True)
    self._write_classes_file(output_path)

    copied = 0
    total_images = max(1, len(image_files_map))
    self.set_label_progress(
        0,
        total_images,
        value=0,
        progress_format=f"Copy originals 0/{total_images}",
    )

    for index, (image_base_name, image_path) in enumerate(image_files_map.items()):
        try:
            label_path = label_files_map.get(image_base_name)
            if self._copy_single_augmentation_source(image_base_name, image_path, label_path, output_path):
                copied += 1
        except Exception as e:
            logger.warning(f"Failed copying augmentation source {image_path}: {e}")

        self.set_label_progress(
            0,
            total_images,
            value=index + 1,
            progress_format=f"Copy originals {index + 1}/{total_images}",
        )
        if index % 10 == 0:
            QtWidgets.QApplication.processEvents()

    return copied


def apply_selected_augmentation_effects(
    self,
    image,
    labels=None,
    pose_data=None,
    polygon_parse_preference=None,
):
    labels = list(labels or [])
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None, labels

    if pose_data is None:
        pose_data = self._load_augmentation_pose_data()

    if getattr(self, "flip_images", False):
        image = cv2.flip(image, 1)
        flipped_labels = []
        for label in labels:
            new_label = self._flip_label_line_horizontal(label, pose_data)
            if new_label:
                flipped_labels.append(new_label.strip())
        labels = flipped_labels

    if getattr(self, "brightness_contrast_effect", False):
        image = self.apply_brightness_contrast_augmentation(image)
        if image is None:
            return None, labels

    if getattr(self, "hsv_jitter_effect", False):
        image = self.apply_hsv_jitter_augmentation(image)
        if image is None:
            return None, labels

    if labels and self._object_clarity_augmentation_selected():
        records = self._object_clarity_annotation_regions(image, labels, polygon_parse_preference)
        image = self._render_object_clarity_augmentation(image, records)
        if image is None:
            return None, labels

    if labels and getattr(self, "sight_picture_effect", False):
        image, labels = self.apply_sight_overlay_augmentation(
            image,
            labels,
            pose_data,
            return_labels=True,
            polygon_parse_preference=polygon_parse_preference,
        )
        if image is None:
            return None, labels

    if getattr(self, "smoke_effect", False):
        image = self.apply_smoke_effect(image)
        if image is None:
            return None, labels

    if getattr(self, "flash_effect", False):
        image = self.apply_flashbang_effect(image)
        if image is None:
            return None, labels

    if getattr(self, "motion_blur_effect", False):
        image = self.apply_motion_blur_effect(image)
        if image is None:
            return None, labels

    if getattr(self, "glass_effect", False):
        image = self.apply_glass_effect(image)
        if image is None:
            return None, labels

    if getattr(self, "gaussian_noise_effect", False):
        image = self.apply_gaussian_noise_augmentation(image)
        if image is None:
            return None, labels

    if getattr(self, "jpeg_compression_effect", False):
        image = self.apply_jpeg_compression_augmentation(image)
        if image is None:
            return None, labels

    return image, labels


@gpu_work("Augmentation", keep=lambda owner: ("clarity_augmentation",) if owner._object_clarity_augmentation_selected() else None,
          until="_augmentation_job")
def process_images_triggered(self):
    """
        Start or cancel the event-driven augmentation job.

        One image is processed per Qt event-loop turn. This keeps the application,
        progress display, and CUDA/System Monitor responsive without accessing
        QWidget objects from a background thread.
        """
    active_job = getattr(self, "_augmentation_job", None)
    if active_job is not None:
        active_job["cancel_requested"] = True
        if hasattr(self, "crop_button"):
            self.crop_button.setEnabled(False)
            self.crop_button.setText("Stopping Augmentation...")
        self.set_label_progress_busy("Stopping augmentation safely...")
        return

    logger.info("process_images_triggered called")

    output_directory, image_files, label_files_map = self.collect_current_augmentation_dataset()
    if not image_files:
        QMessageBox.warning(self, "Error", "Open an image dataset before running augmentation.")
        return

    image_effects = []
    if self.glass_effect:
        image_effects.append("glass")
    if self.motion_blur_effect:
        image_effects.append("motion_blur")
    if self.brightness_contrast_effect:
        image_effects.append("light")
    if self.hsv_jitter_effect:
        image_effects.append("hsv")
    if self.gaussian_noise_effect:
        image_effects.append("noise")
    if self.jpeg_compression_effect:
        image_effects.append("jpeg")
    if self.flash_effect:
        image_effects.append("flash")
    if self.smoke_effect:
        image_effects.append("smoke")
    if self.sight_picture_effect:
        image_effects.append("sight_picture")
    if self.flip_images:
        image_effects.append("mirror")
    if self._object_clarity_augmentation_selected():
        image_effects.append("object_clarity")

    # Load class names using `load_classes`
    class_names = self.load_classes(data_directory=output_directory)
    if not class_names:
        QMessageBox.warning(self, "Error", "No valid classes found in classes.txt.")
        return

    # Image and label mapping from the current opened dataset.
    image_files_map = {os.path.splitext(os.path.basename(p))[0]: p for p in image_files}
    selected_items = list(self.sampled_augmentation_items(image_files_map))
    total_source_images = len(image_files)
    percent = self.selected_augmentation_percent()

    total_images = len(selected_items)
    logger.info(
        "Augmentation percent: %s%% (%s of %s images selected).",
        percent,
        total_images,
        total_source_images,
    )

    if not image_effects and not self.make_copy_enabled:
        QMessageBox.warning(self, "No Augmentation Selected", "Select an augmentation or Copy before running.")
        return

    output_folder_name = "_".join(image_effects) if image_effects else ""
    output_path = os.path.join(output_directory, output_folder_name) if output_folder_name else ""
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        self._write_classes_file(output_path)

    copy_path = ""
    if self.make_copy_enabled:
        copy_path = os.path.join(output_directory, "copy")
        os.makedirs(copy_path, exist_ok=True)
        self._write_classes_file(copy_path)

    phases = []
    if self.make_copy_enabled:
        phases.append("copy")
    if image_effects:
        phases.append("augment")

    pose_data = dict(self._load_augmentation_pose_data() or {})
    if not pose_data.get("keypoints"):
        keypoint_count = self._augmentation_keypoint_count(pose_data)
        if keypoint_count > 0:
            pose_data["keypoints"] = [f"keypoint_{index}" for index in range(keypoint_count)]
    polygon_parse_preference = self._polygon_label_parse_preference()

    control_names = (
        "sight_picture",
        "smoke_checkbox",
        "flash_checkbox",
        "motion_blur",
        "glass_checkbox",
        "brightness_contrast_checkbox",
        "hsv_jitter_checkbox",
        "gaussian_noise_checkbox",
        "jpeg_compression_checkbox",
        "flip_checkbox",
        "make_copy",
        "object_clarity_augmentation_checkbox",
        "augmentation_percent_spinbox",
        "image_format_combo",
    )
    control_states = {}
    for name in control_names:
        widget = getattr(self, name, None)
        if isinstance(widget, QtWidgets.QWidget):
            control_states[name] = widget.isEnabled()
            widget.setEnabled(False)

    total_steps = max(1, total_images * len(phases))
    self._augmentation_job = {
        "items": selected_items,
        "label_files_map": dict(label_files_map),
        "output_directory": output_directory,
        "output_path": output_path,
        "copy_path": copy_path,
        "output_folder_name": output_folder_name,
        "pose_data": pose_data,
        "polygon_parse_preference": polygon_parse_preference,
        "phases": phases,
        "phase_index": 0,
        "item_index": 0,
        "completed_steps": 0,
        "total_steps": total_steps,
        "processed": 0,
        "errors": 0,
        "cancel_requested": False,
        "control_states": control_states,
        "started_at": time.monotonic(),
    }

    self.set_label_progress(
        0,
        total_steps,
        value=0,
        progress_format=f"Starting augmentation 0/{total_steps}",
    )
    if hasattr(self, "crop_button"):
        self.crop_button.setEnabled(True)
        self.crop_button.setText("Cancel Augmentation")

    QTimer.singleShot(0, self._process_next_augmentation_step)


def _copy_single_augmentation_source(
    self,
    image_base_name,
    image_path,
    label_path,
    output_path,
    pose_data=None,
):
    """Copy one labeled source image for the incremental augmentation job."""
    if not label_path or os.path.basename(label_path).lower() == "classes.txt":
        return False

    label_lines = self._valid_annotation_lines_from_file(label_path, pose_data=pose_data)
    if not label_lines:
        logger.info("Skipping copy source with no valid annotations: %s", image_path)
        return False

    output_ext = self.augmentation_output_extension_for_image(image_path)
    output_image_name = f"{os.path.splitext(os.path.basename(image_path))[0]}{output_ext}"
    output_image_path = os.path.join(output_path, output_image_name)
    source_ext = normalize_image_extension(os.path.splitext(image_path)[1], default=".jpg")

    output_label_path = os.path.join(output_path, os.path.basename(label_path))
    if output_ext == source_ext:
        image = image_path
        image_writer = lambda target, source: bool(shutil.copy2(source, target))
    else:
        image = self._read_image_cv(image_path)
        if image is None:
            raise ExtractionWriteError(f"Could not read augmentation source: {image_path}")
        image_writer = lambda target, frame: save_cv_image(target, frame, quality=100)
    return save_extraction_outputs(
        output_image_path, image, image_writer,
        label_path=output_label_path, labels=label_lines,
        label_writer=lambda path, lines, mode="w": self._write_label_lines(path, lines),
        preserve_existing_labels=False,
    )


def _process_next_augmentation_step(self):
    """Process one augmentation item, then yield back to Qt before continuing."""
    job = getattr(self, "_augmentation_job", None)
    if job is None:
        return
    if job.get("cancel_requested"):
        self._finish_augmentation_job(canceled=True)
        return

    phases = job["phases"]
    items = job["items"]
    if job["phase_index"] >= len(phases):
        self._finish_augmentation_job()
        return

    if job["item_index"] >= len(items):
        job["phase_index"] += 1
        job["item_index"] = 0
        QTimer.singleShot(0, self._process_next_augmentation_step)
        return

    phase = phases[job["phase_index"]]
    item_index = job["item_index"]
    image_base_name, image_path = items[item_index]
    label_path = job["label_files_map"].get(image_base_name)
    succeeded = False

    try:
        if phase == "copy":
            succeeded = self._copy_single_augmentation_source(
                image_base_name,
                image_path,
                label_path,
                job["copy_path"],
                pose_data=job.get("pose_data"),
            )
        elif label_path is not None and os.path.basename(label_path).lower() != "classes.txt":
            succeeded = bool(self.apply_augmentations(
                item_index,
                image_path,
                label_path,
                job["output_path"],
                job["output_folder_name"],
                len(items),
                update_progress=False,
                pose_data=job.get("pose_data"),
                polygon_parse_preference=job.get("polygon_parse_preference"),
            ))
    except Exception as e:
        job["errors"] += 1
        logger.exception("Augmentation failed for %s: %s", image_path, e)

    if succeeded:
        job["processed"] += 1
    job["item_index"] += 1
    job["completed_steps"] += 1

    phase_title = "Copying originals" if phase == "copy" else "Augmenting"
    self.set_label_progress(
        0,
        job["total_steps"],
        value=job["completed_steps"],
        progress_format=(
            f"{phase_title} {item_index + 1}/{max(1, len(items))} | "
                f"Overall {job['completed_steps']}/{job['total_steps']}"
        ),
    )
    QTimer.singleShot(0, self._process_next_augmentation_step)


def _finish_augmentation_job(self, canceled=False):
    job = getattr(self, "_augmentation_job", None)
    if job is None:
        return

    self._augmentation_job = None
    if getattr(self, "_clarity_augmentation_preview_pending", False):
        self._clarity_augmentation_preview_pending = False
        self.queue_augmentation_preview_update()
    for name, was_enabled in job.get("control_states", {}).items():
        widget = getattr(self, name, None)
        if isinstance(widget, QtWidgets.QWidget):
            widget.setEnabled(bool(was_enabled))

    if hasattr(self, "crop_button"):
        self.crop_button.setEnabled(True)
        self.crop_button.setText("Run Augmentation")

    elapsed = max(0.0, time.monotonic() - float(job.get("started_at", time.monotonic())))
    processed = int(job.get("processed", 0))
    errors = int(job.get("errors", 0))
    completed_steps = int(job.get("completed_steps", 0))
    total_steps = int(job.get("total_steps", 1))

    if canceled:
        self.set_label_progress(
            0,
            max(1, total_steps),
            value=min(completed_steps, max(1, total_steps)),
            progress_format="Augmentation canceled safely",
        )
        logger.info("Augmentation canceled after %d completed step(s).", completed_steps)
        QMessageBox.information(
            self,
            "Augmentation Canceled",
            f"Stopped safely.\n\nOutputs completed: {processed}\nErrors: {errors}\nElapsed: {elapsed:.1f} seconds",
        )
        return

    self.set_label_progress(
        0,
        max(1, total_steps),
        value=max(1, total_steps),
        progress_format="Augmentation done",
    )
    logger.info(
        "Augmentation completed: %d output(s), %d error(s), %.1f seconds.",
        processed,
        errors,
        elapsed,
    )
    if errors:
        QMessageBox.warning(
            self,
            "Augmentation Finished With Errors",
            f"{processed} complete output pair(s) were created.\nErrors: {errors}\n"
                f"Elapsed: {elapsed:.1f} seconds\nCheck the log for failed saves or processing errors.",
        )
    elif processed == 0:
        QMessageBox.warning(
            self,
            "No Valid Annotations",
            "No outputs were created because the selected images had no valid annotation labels.",
        )
    else:
        QMessageBox.information(
            self,
            "Success",
            f"{processed} outputs were created.\nErrors: {errors}\nElapsed: {elapsed:.1f} seconds",
        )


def apply_augmentations(
    self,
    current_image,
    image_path,
    label_path,
    output_path,
    output_folder_name,
    total_images,
    update_progress=True,
    pose_data=None,
    polygon_parse_preference=None,
):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return

    # Read labels
    if not os.path.exists(label_path):
        logger.warning(f"No label file found for: {image_path}")
        labels = []
    else:
        with open(label_path, 'r') as f:
            labels = self._valid_annotation_lines(f.readlines(), pose_data=pose_data)

    if not labels:
        logger.warning(f"Skipping augmentation for {image_path} (no valid annotations).")
        return

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image, labels = self.apply_selected_augmentation_effects(
        image,
        labels,
        pose_data=pose_data,
        polygon_parse_preference=polygon_parse_preference,
    )
    if image is None:
        return

    if not labels:
        logger.warning(f"Skipping augmentation for {image_path} (no valid labels remain).")
        return

    # Choose output format and file path.
    output_ext = self.augmentation_output_extension_for_image(image_path)

    output_image_path = os.path.join(
        output_path, f"{output_folder_name}_{timestamp}_{current_image}{output_ext}"
    )

    output_label_path = os.path.join(output_path, f"{output_folder_name}_{timestamp}_{current_image}.txt")
    save_extraction_outputs(
        output_image_path, image,
        lambda path, frame: save_cv_image(path, frame, quality=100),
        label_path=output_label_path, labels=labels,
        label_writer=lambda path, lines, mode="w": self._write_label_lines(path, lines),
        preserve_existing_labels=False,
    )

    if update_progress:
        self.set_label_progress(
            0,
            max(1, total_images),
            value=current_image + 1,
            progress_format=f"Augmenting {current_image + 1}/{max(1, total_images)}",
        )
    return True


def _compact_extra_label_panel(self):
    if getattr(self, "_extra_label_panel_compacted", False):
        return

    group = getattr(self, "groupBox_14", None)
    layout = group.layout() if group is not None else None
    if not isinstance(layout, QtWidgets.QGridLayout):
        return

    names = (
        "heads_area",
        "label_52",
        "class_id",
        "label_8",
        "head_width",
        "label_38",
        "head_height",
        "label_39",
        "head_horizontal",
        "label_40",
        "head_verticle",
    )
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)

    self._reset_layout(layout)
    layout.setContentsMargins(12, 16, 12, 12)
    layout.setHorizontalSpacing(8)
    layout.setVerticalSpacing(8)

    widgets["heads_area"].setText("Extra label")
    widgets["label_52"].setText("Class")
    widgets["label_8"].setText("W")
    widgets["label_38"].setText("H")
    widgets["label_39"].setText("X")
    widgets["label_40"].setText("Y")
    widgets["class_id"].setPlaceholderText("Class ID")
    widgets["class_id"].setToolTip(
        "Target class ID for the nested label. If blank, DarkFusion uses the class named 'head'."
    )
    widgets["head_width"].setToolTip("Nested-label width as a percentage of the parent label width.")
    widgets["head_height"].setToolTip("Nested-label height as a percentage of the parent label height.")
    widgets["head_horizontal"].setToolTip("Horizontal center position as a percentage of the parent label width.")
    widgets["head_verticle"].setToolTip("Vertical top position as a percentage of the parent label height.")

    enabled_row = self._row_layout()
    enabled_row.addWidget(widgets["heads_area"])
    enabled_row.addStretch(1)

    class_row = self._row_layout()
    class_row.addWidget(widgets["label_52"])
    class_row.addWidget(widgets["class_id"], 1)

    size_row = self._row_layout()
    size_row.addWidget(widgets["label_8"])
    size_row.addWidget(widgets["head_width"])
    size_row.addWidget(widgets["label_38"])
    size_row.addWidget(widgets["head_height"])
    size_row.addStretch(1)

    offset_row = self._row_layout()
    offset_row.addWidget(widgets["label_39"])
    offset_row.addWidget(widgets["head_horizontal"])
    offset_row.addWidget(widgets["label_40"])
    offset_row.addWidget(widgets["head_verticle"])
    offset_row.addStretch(1)

    for row, row_layout in enumerate((enabled_row, class_row, size_row, offset_row)):
        layout.addLayout(row_layout, row, 0)

    self._extra_label_panel_compacted = True

