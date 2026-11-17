"""Dataset Analysis dialog, health scanning, and analysis/report workers.

Statistics, image-check caching, filesystem policy, and model ownership remain
in their existing shared modules. The main window supplies the same parent API.
"""

import cv2
from darkfusion_environment import model_directory
import hashlib
import html
import json
import logging
import math
import numpy as np
import os
import re
import shutil
import torch
from PyQt5 import QtWidgets, sip
from PyQt5.QtCore import QObject, QSettings, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QDoubleSpinBox, QFileDialog, QHeaderView, QMessageBox,
    QSpinBox, QTableWidgetItem,
)
from collections import Counter, OrderedDict, defaultdict
from darkfusion_borderline_llm_verifier import BorderlineLLMVerifier
from darkfusion_dataset_operations import (
    PROJECT_SETTINGS_DIR, dataset_metadata_directory, read_json_file,
    resolve_dataset_classes_path, resolve_dataset_metadata_path,
    update_health_report_after_deletion, write_json_file_atomic,
)
from darkfusion_dataset_statistics import (
    DatasetStatisticsWidget, ImageCheckCache, QUALITY_DEFAULTS, build_statistics,
    image_quality_metrics, quality_findings,
)
from darkfusion_gpu_memory import analysis_resources, gpu_work, release_analysis_models
from darkfusion_image_loading import IMAGE_SUFFIXES
from darkfusion_inference import SAM3_DEFAULT_IMGSZ
from darkfusion_shape_verifier import (
    ShapeProfiler, ShapeReferenceCache, shape_annotation_crop, shape_verification_score,
)
from darkfusion_verified_images import get_verified_images_for_scan
from datetime import datetime
from pathlib import Path


APP_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("UltraDarkFusionLogger")


class DatasetAnalysisWorker(QThread):
    """Run the read-only dataset health scan without blocking the Qt UI."""

    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(object, str)
    failed = pyqtSignal(str)
    canceled = pyqtSignal()

    def __init__(self, scanner, dataset_dir, classes_file, scan_context, parent=None):
        super().__init__(parent)
        self.scanner = scanner
        self.dataset_dir = str(dataset_dir or "")
        self.classes_file = str(classes_file or "")
        self.scan_context = dict(scan_context or {})
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def is_cancel_requested(self):
        return bool(self._cancel_requested)

    @release_analysis_models
    def run(self):
        try:
            result = self.scanner._run_scan_background(
                self.dataset_dir,
                self.classes_file,
                self.scan_context,
                self.is_cancel_requested,
                self.progress.emit,
            )
            if self._cancel_requested or result is None:
                self.canceled.emit()
                return

            report, report_path = result
            self.completed.emit(report, report_path)
        except Exception as e:
            self.failed.emit(str(e))


class ScanAnnotations(QObject):
    """
    Safe dataset health check.

    This scanner only reads dataset files and writes its report and image-check
    cache under .darkfusion. The scan itself does not rewrite labels,
    strip metadata, delete JSON files, or move bad files. The report dialog can
    move a selected issue into review or delete files matched by its filters,
    only after the user confirms the action.
    """

    PROTECTED_JSON_NAMES = {
        "points.json",
        "pose.json",
        "keypoints.json",
        "statistics.json",
        "settings.json",
        "project.json",
        "scan_report.json",
        "review_state.json",
        "repair_log.json",
        "review_manifest.json",
        "darkfusion_project.json",
        "class_thresholds.json",
        "data.json",
        "dataset.json",
    }

    DUPLICATE_IOU_THRESHOLD = 0.85
    BBOX_EDGE_TOLERANCE_PX = 1.0
    TRAINING_TINY_OBJECT_SIDE_PX = 4.0
    CLASS_IMBALANCE_RATIO = 5.0
    VISUAL_OUTLIER_MIN_CLASS_SAMPLES = 10  # Increased from 5 - skip very small classes
    VISUAL_OUTLIER_MAX_CLASS_FRACTION = 0.03
    VISUAL_OUTLIER_EXACT_CLASS_LIMIT = 1000  # Reduced from 2000 - use sampling sooner
    VISUAL_OUTLIER_REFERENCE_LIMIT = 256
    VISUAL_OUTLIER_MAX_FINDINGS_PER_CLASS = None  # None = no per-class cap; driven by fraction only
    VISUAL_OUTLIER_DEFAULT_DEEP_PASSES = 1  # OPTIMIZED: 1 pass instead of 3 (3-4x speedup!)
    VISUAL_OUTLIER_TRUSTED_STRONG_MATCH = 0.86
    VISUAL_OUTLIER_TRUSTED_TOP3_MATCH = 0.82
    SHAPE_MIN_TRUSTED_REFERENCES = 3
    SHAPE_MAX_TRUSTED_REFERENCES_PER_CLASS = 32
    SCAN_CHECKS = (
        ("statistics", "Collect statistics"),
        ("file_checks", "Missing/empty labels and orphan files"),
        ("image_checks", "Image readability"),
        ("label_checks", "Label format, classes and geometry"),
        ("duplicate_checks", "Duplicate labels"),
        ("tiny_checks", "Tiny objects / training size"),
        ("quality_checks", "Image quality: blur, exposure and contrast (slower)"),
    )

    def _saved_scan_checks(self):
        settings = QSettings("UltraDarkFusion", "HealthCheck")
        return {key: str(settings.value(
            "scan_" + key,
            "false" if key == "quality_checks" else "true",
        )).lower()
                in ("1", "true", "yes") for key, _title in self.SCAN_CHECKS}

    def _saved_quality_thresholds(self):
        settings = QSettings("UltraDarkFusion", "HealthCheck")
        values = {}
        for key, default in QUALITY_DEFAULTS.items():
            try:
                value = float(settings.value("quality_" + key, default))
                values[key] = max(0.0, min(1000000.0 if key == "blur" else 255.0, value)) if math.isfinite(value) else default
            except (TypeError, ValueError):
                values[key] = default
        if values["dark"] >= values["bright"]:
            values["dark"], values["bright"] = QUALITY_DEFAULTS["dark"], QUALITY_DEFAULTS["bright"]
        return values

    def _scan_check_enabled(self, key):
        return bool((getattr(self, "_scan_checks_override", None) or {}).get(
            key, key not in ("statistics", "quality_checks")
        ))

    def _needs_annotation_records(self):
        return any(self._scan_check_enabled(key) for key in (
            "label_checks", "duplicate_checks", "tiny_checks", "statistics",
        )) or bool(self._scan_visual_outliers_override)

    def _selected_scan_issues(self):
        groups = {
            "missing_label_file": "file_checks", "empty_label_file": "file_checks",
            "label_without_image": "file_checks", "orphan_json": "file_checks",
            "duplicate_image_stem": "file_checks",
            "unreadable_image": "image_checks", "duplicate_box_candidate": "duplicate_checks",
            "duplicate_annotation_exact": "duplicate_checks",
            "tiny_box": "tiny_checks", "object_too_small_for_training_size": "tiny_checks",
            "blurry_image": "quality_checks", "underexposed_image": "quality_checks",
            "overexposed_image": "quality_checks", "low_contrast_image": "quality_checks",
        }
        return [issue for issue in self.issues
                if issue.get("issue_type") == "visual_class_outlier"
                or self._scan_check_enabled(groups.get(issue.get("issue_type"), "label_checks"))]

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.base_directory = ""
        self.valid_classes = []
        self.override_classes_file = ""
        self.valid_image_extensions = IMAGE_SUFFIXES
        self.scan_report = None
        self._analysis_worker = None
        self._scan_training_size_override = None
        self._scan_training_size_source_override = ""
        self._scan_polygon_preference_override = None
        self._scan_expected_annotation_family_override = None
        self._scan_duplicate_iou_override = None
        self._scan_visual_outliers_override = None
        self._scan_visual_outlier_threshold_override = None
        self._scan_visual_outlier_passes_override = None
        self._scan_visual_class_verify_override = None
        self._scan_shape_verify_override = None
        self._scan_checks_override = None
        self._scan_context_active = False
        self._reset_scan_state()

    def _reset_scan_state(self):
        self.total_images = 0
        self.total_label_files = 0
        self.total_labels = 0
        self.valid_labels = 0
        self.missing_label_files = 0
        self.empty_label_files = 0
        self.bad_images = 0
        self.duplicate_box_candidates = 0
        self.used_class_indices = set()
        self.class_label_counts = Counter()
        self.training_tiny_object_count = 0
        self.visual_outlier_candidates = 0
        self.visual_outlier_scan = {"status": "not_run", "backend": "not_run"}
        self.training_input_size_source = ""
        self.expected_keypoint_count = 0
        self.annotation_family_counts = Counter()
        self.annotation_family_records = []
        self.dominant_annotation_family = ""
        self._image_size_cache = {}
        self._image_check_cache = None
        self._statistics_label_states = {}
        self._quality_records = {}
        self._ambiguous_image_stems = set()
        self._scan_quality_thresholds = dict(QUALITY_DEFAULTS)
        self.statistics_report = None
        self.issues = []

    def _normalize_path(self, file_path):
        try:
            if hasattr(self.parent, "normalize_path"):
                return self.parent.normalize_path(file_path)
            return os.path.abspath(str(file_path)).replace("\\", "/")
        except Exception:
            return str(file_path).replace("\\", "/")

    def _dataset_directory(self):
        """Return the dataset that is open now, never a training-YAML fallback."""
        candidates = [getattr(self.parent, "image_directory", "")]
        current_file = getattr(self.parent, "current_file", "")
        if current_file and not (
            hasattr(self.parent, "is_placeholder_file")
            and self.parent.is_placeholder_file(current_file)
        ):
            candidates.append(os.path.dirname(current_file) if os.path.isfile(current_file) else current_file)
        candidates.append(self.base_directory)

        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return self._normalize_path(candidate)

        start_dir = (
            self.parent.dialog_start_directory("dataset")
            if hasattr(self.parent, "dialog_start_directory")
            else os.getcwd()
        )
        selected = QFileDialog.getExistingDirectory(self.parent, "Select Dataset Folder", start_dir)
        if selected:
            if hasattr(self.parent, "remember_dialog_selection"):
                self.parent.remember_dialog_selection("dataset", selected)
            return self._normalize_path(selected)
        return ""

    def _report_path(self, dataset_dir):
        return os.path.join(dataset_dir, PROJECT_SETTINGS_DIR, "scan_report.json")

    def _review_state_path(self, dataset_dir):
        return os.path.join(dataset_dir, PROJECT_SETTINGS_DIR, "review_state.json")

    def _relative_path(self, file_path):
        if not file_path:
            return ""
        try:
            return os.path.relpath(file_path, self.base_directory).replace("\\", "/")
        except Exception:
            return self._normalize_path(file_path)

    def _make_issue(self, issue_type, severity, file_path, message, line_number=None, label_line=None, suggestion=""):
        return {
            "severity": severity,
            "issue_type": issue_type,
            "file": self._relative_path(file_path),
            "absolute_path": self._normalize_path(file_path) if file_path else "",
            "line": line_number,
            "message": message,
            "label_line": label_line or "",
            "suggestion": suggestion,
        }

    def _issue_review_key(self, issue):
        issue_type = issue.get("issue_type", "")
        # Similarity scores and deep-pass positions can shift when nearby bad
        # labels are removed. Keep the review identity tied to the saved
        # annotation row so an already reviewed finding stays reviewed.
        message = "" if issue_type == "visual_class_outlier" else issue.get("message", "")
        return json.dumps([
            issue.get("file", ""),
            issue.get("line", None),
            issue_type,
            issue.get("label_line", ""),
            message,
        ], ensure_ascii=False)

    def _load_reviewed_issue_keys(self, dataset_dir):
        state = read_json_file(self._review_state_path(dataset_dir), {})
        if not isinstance(state, dict):
            return set()

        reviewed = state.get("reviewed_issues", [])
        if not isinstance(reviewed, list):
            return set()

        keys = {str(key) for key in reviewed}
        # Older builds included the changing similarity score in visual finding
        # keys. Add the stable annotation-row form in memory so those reviewed
        # findings remain hidden after upgrading to deep scans.
        for key in list(keys):
            try:
                parts = json.loads(key)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                isinstance(parts, list)
                and len(parts) >= 5
                and parts[2] == "visual_class_outlier"
            ):
                parts[4] = ""
                keys.add(json.dumps(parts, ensure_ascii=False))
        return keys

    def _load_visual_trusted_annotations(self, dataset_dir):
        """Load annotations the user explicitly reviewed and chose to keep.

        ``review_state.json`` cannot be used here because older versions stored
        both *Mark Finding Reviewed* and *Skip Finding* in the same set.  The
        decision history preserves that distinction.  Entries without a saved
        raw annotation are ignored rather than guessed.
        """
        from darkfusion_review_feedback import canonical_label_text

        history_path = os.path.join(
            self._normalize_path(dataset_dir), PROJECT_SETTINGS_DIR,
            "health_review_history.json",
        )
        history = read_json_file(history_path, {})
        decisions = history.get("decisions", {}) if isinstance(history, dict) else {}
        if not isinstance(decisions, dict):
            return set()
        queue_path = os.path.join(
            self._normalize_path(dataset_dir), PROJECT_SETTINGS_DIR,
            "health_review_queue.json",
        )
        queue_report = read_json_file(queue_path, {})
        queue_issues = (
            queue_report.get("issues", []) if isinstance(queue_report, dict) else []
        )
        issue_by_key = {
            str(issue.get("issue_key", "")): issue
            for issue in queue_issues
            if isinstance(issue, dict) and issue.get("issue_key")
        }
        trusted = set()
        reviewed_pages = history.get("reviewed_images", {})
        if not isinstance(reviewed_pages, dict):
            reviewed_pages = {}
        page_label_paths = set()
        for page in reviewed_pages.values():
            if not isinstance(page, dict):
                continue
            label_path = os.path.normcase(self._normalize_path(page.get("label_path", "")))
            page_label_paths.add(label_path)
            try:
                image_stat = os.stat(page.get("image_path", ""))
                if (image_stat.st_size, image_stat.st_mtime_ns) != (
                    page.get("image_size"), page.get("image_mtime_ns")
                ):
                    continue
            except OSError:
                continue
            for line in page.get("label_lines", []):
                identity = canonical_label_text(line)
                if label_path and identity:
                    trusted.add((label_path, identity))
        for issue_key, decision in decisions.items():
            if not isinstance(decision, dict):
                continue
            # Histories written before trusted-example learning did not copy
            # the raw label into each decision. Recover it from the saved
            # review queue when the stable issue key is still available.
            source_issue = issue_by_key.get(str(issue_key), {})
            if str(decision.get("status", "")) != "reviewed":
                continue
            issue_type = decision.get("type") or source_issue.get("type")
            if str(issue_type or "") != "visual_class_outlier":
                continue
            label_path = self._normalize_path(
                decision.get("label_path") or source_issue.get("label_path", "")
            )
            if os.path.normcase(label_path) in page_label_paths:
                # The latest confirmed page supersedes older individual keeps.
                continue
            raw_line = " ".join(
                str(
                    decision.get("annotation_label_text")
                    or source_issue.get("annotation_label_text", "")
                    or ""
                ).split()
            )
            if not label_path or not raw_line:
                continue
            trusted.add((os.path.normcase(label_path), canonical_label_text(raw_line)))
        return trusted

    def _visual_record_trust_key(self, record):
        from darkfusion_review_feedback import canonical_label_text

        return (
            os.path.normcase(self._normalize_path(record.get("file_path", ""))),
            canonical_label_text(record.get("line", "")),
        )

    def _trusted_visual_similarity(self, matrix, candidate_index, trusted_indices):
        """Return (protected, max, top-3 mean) against kept review examples."""
        if not trusted_indices:
            return False, None, None
        candidate_index = int(candidate_index)
        if candidate_index in trusted_indices:
            return True, 1.0, 1.0
        similarities = np.matmul(matrix[list(trusted_indices)], matrix[candidate_index])
        if similarities.size == 0:
            return False, None, None
        ordered = np.sort(similarities)[::-1]
        maximum = float(ordered[0])
        top3 = float(np.mean(ordered[:min(3, len(ordered))]))
        protected = maximum >= self.VISUAL_OUTLIER_TRUSTED_STRONG_MATCH
        if len(ordered) >= 3:
            protected = protected or top3 >= self.VISUAL_OUTLIER_TRUSTED_TOP3_MATCH
        return protected, maximum, top3

    def _save_reviewed_issue_keys(self, dataset_dir, reviewed_keys):
        state = OrderedDict([
            ("updated_at", datetime.now().isoformat(timespec="seconds")),
            ("reviewed_issues", sorted(str(key) for key in reviewed_keys)),
        ])
        write_json_file_atomic(self._review_state_path(dataset_dir), state)

    def _find_classes_file(self, dataset_dir):
        if self.override_classes_file and os.path.exists(self.override_classes_file):
            override_path = self._normalize_path(self.override_classes_file)
            try:
                normalized_dataset = self._normalize_path(dataset_dir)
                if os.path.normcase(os.path.commonpath([normalized_dataset, override_path])) == os.path.normcase(normalized_dataset):
                    return override_path
            except (OSError, ValueError):
                pass

        candidates = [resolve_dataset_classes_path(dataset_dir, migrate=True)]
        candidates.extend([
            os.path.join(dataset_dir, "obj.names"),
            os.path.join(dataset_dir, "names.txt"),
        ])

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate

        for name in os.listdir(dataset_dir):
            if name.lower().endswith(".names"):
                candidate = os.path.join(dataset_dir, name)
                if os.path.exists(candidate):
                    return candidate

        return ""

    def _load_classes(self, dataset_dir):
        classes_file = self._find_classes_file(dataset_dir)
        classes = []

        if classes_file:
            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    classes = [line.strip() for line in f if line.strip()]
            except UnicodeDecodeError:
                with open(classes_file, "r", encoding="latin-1", errors="ignore") as f:
                    classes = [line.strip() for line in f if line.strip()]
            except Exception as e:
                self.issues.append(self._make_issue(
                    "classes_read_error",
                    "error",
                    classes_file,
                    f"Could not read classes file: {e}",
                    suggestion="Check the classes file encoding or permissions.",
                ))

        if not classes and hasattr(self.parent, "load_classes"):
            try:
                classes = self.parent.load_classes(data_directory=dataset_dir, create_if_missing=False) or []
            except TypeError:
                classes = self.parent.load_classes(data_directory=dataset_dir) or []
            except Exception as e:
                self.issues.append(self._make_issue(
                    "classes_load_error",
                    "error",
                    dataset_dir,
                    f"Could not load classes automatically: {e}",
                ))

        if not classes:
            start_dir = dataset_dir
            if hasattr(self.parent, "dialog_start_directory"):
                start_dir = self.parent.dialog_start_directory("metadata", classes_file or dataset_dir)
            selected, _ = QFileDialog.getOpenFileName(
                self.parent,
                "Select Classes File",
                start_dir,
                "Class Files (classes.txt *.names *.txt);;All Files (*)"
            )
            if selected:
                if hasattr(self.parent, "remember_dialog_selection"):
                    self.parent.remember_dialog_selection("metadata", selected)
                classes_file = selected
                self.base_directory = self._normalize_path(os.path.dirname(selected))
                try:
                    with open(selected, "r", encoding="utf-8") as f:
                        classes = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    self.issues.append(self._make_issue(
                        "classes_read_error",
                        "error",
                        selected,
                        f"Could not read selected classes file: {e}",
                    ))

        return classes, classes_file

    def _load_expected_keypoint_count(self, dataset_dir):
        candidates = [
            resolve_dataset_metadata_path(dataset_dir, "points.json", migrate=True),
            resolve_dataset_metadata_path(dataset_dir, "pose.json", migrate=True),
        ]

        for candidate in candidates:
            if not os.path.exists(candidate):
                continue
            data = read_json_file(candidate, {})
            keypoints = data.get("keypoints") if isinstance(data, dict) else None
            if isinstance(keypoints, list) and keypoints:
                return len(keypoints)

            kpt_shape = data.get("kpt_shape") if isinstance(data, dict) else None
            if isinstance(kpt_shape, list) and kpt_shape:
                try:
                    return max(0, int(kpt_shape[0]))
                except Exception:
                    pass

        try:
            table = getattr(self.parent, "keypoint_list", None)
            if table is not None:
                return max(0, int(table.rowCount()))
        except Exception:
            pass

        return 0

    def _is_class_index_valid(self, class_index):
        if isinstance(self.valid_classes, dict):
            return class_index in self.valid_classes
        return 0 <= class_index < len(self.valid_classes)

    def _class_name_for_index(self, class_index):
        try:
            class_index = int(class_index)
            if isinstance(self.valid_classes, dict):
                return str(self.valid_classes.get(class_index, f"Class {class_index}"))
            if 0 <= class_index < len(self.valid_classes):
                return str(self.valid_classes[class_index])
        except Exception:
            pass
        return f"Class {class_index}"

    def _training_input_size(self):
        override = getattr(self, "_scan_training_size_override", None)
        if override is not None:
            try:
                width, height = int(override[0]), int(override[1])
                self.training_input_size_source = str(
                    getattr(self, "_scan_training_size_source_override", "") or ""
                )
                return max(0, width), max(0, height)
            except Exception:
                pass

        self.training_input_size_source = ""

        def parse_size(value):
            text = str(value or "").strip()
            if not text:
                return 0, 0

            numbers = [int(match) for match in re.findall(r"\d+", text)]
            if not numbers:
                return 0, 0

            if len(numbers) == 1:
                size = numbers[0]
                return (size, size) if size > 0 else (0, 0)

            width, height = numbers[0], numbers[1]
            return (width, height) if width > 0 and height > 0 else (0, 0)

        # Future training-preview controls can set a dedicated size without
        # changing the real training parameter until the user accepts it.
        for attr_name in ("training_preview_size", "train_preview_size"):
            value = getattr(self.parent, attr_name, None)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                try:
                    width, height = int(value[0]), int(value[1])
                    if width > 0 and height > 0:
                        self.training_input_size_source = "preview"
                        return width, height
                except Exception:
                    pass

        for widget_name in ("training_preview_imgsz_spinbox", "train_preview_imgsz_spinbox"):
            widget = getattr(self.parent, widget_name, None)
            if widget is not None:
                try:
                    size = int(widget.value())
                    if size > 0:
                        self.training_input_size_source = "preview"
                        return size, size
                except Exception:
                    pass

        settings = getattr(self.parent, "settings", {})
        if isinstance(settings, dict):
            width, height = parse_size(settings.get("trainingPreviewImgSize", ""))
            if width > 0 and height > 0:
                self.training_input_size_source = "preview"
                return width, height

        # Prefer the Ultralytics training imgsz field because this evaluator is
        # meant to match what YOLO receives when training from DarkFusion.
        imgsz_widget = getattr(self.parent, "imgsz_input", None)
        if imgsz_widget is not None:
            try:
                width, height = parse_size(imgsz_widget.text())
                if width > 0 and height > 0:
                    self.training_input_size_source = "ultralytics"
                    return width, height
            except Exception:
                pass

        # Fallback for Darknet projects that still use explicit network dims.
        try:
            width_widget = getattr(self.parent, "network_width", None)
            height_widget = getattr(self.parent, "network_height", None)
            if width_widget is not None and height_widget is not None:
                width = int(width_widget.value())
                height = int(height_widget.value())
                if width > 0 and height > 0:
                    self.training_input_size_source = "darknet"
                    return width, height
        except Exception:
            pass
        return 0, 0

    def _letterbox_label_size(self, bbox_width, bbox_height, image_size, train_size):
        img_width, img_height = image_size
        train_width, train_height = train_size

        if img_width <= 0 or img_height <= 0 or train_width <= 0 or train_height <= 0:
            return 0.0, 0.0, 0.0

        scale = min(train_width / img_width, train_height / img_height)
        return bbox_width * img_width * scale, bbox_height * img_height * scale, scale

    def _iter_image_files(self, dataset_dir):
        # Dataset Analysis deliberately reads the selected folder itself.  The
        # main labeler's list may be filtered, partially loaded, or temporarily
        # replaced by a review queue and must not define the scan scope.
        image_files = []
        for entry in os.scandir(dataset_dir):
            if not entry.is_file():
                continue
            if entry.name.lower().endswith(self.valid_image_extensions):
                image_files.append(self._normalize_path(entry.path))
        return sorted(image_files)

    def _iter_label_files(self, dataset_dir):
        label_files = []
        non_label_text_files = {
            "classes.txt",
            "train.txt",
            "valid.txt",
            "val.txt",
            "test.txt",
            "obj.names",
        }
        for entry in os.scandir(dataset_dir):
            if not entry.is_file():
                continue
            name = entry.name.lower()
            if not name.endswith(".txt"):
                continue
            if name in non_label_text_files or name.endswith(".names"):
                continue
            label_files.append(self._normalize_path(entry.path))
        return sorted(label_files)

    def _read_image_size(self, image_path, *, facts=None):
        try:
            cache = getattr(self, "_image_check_cache", None)
            verify = self._scan_check_enabled("image_checks")
            quality = self._scan_check_enabled("quality_checks")
            # Label/statistics-only scans need headers, not full-file hashing.
            read_image = (cache.read_image if cache is not None and (verify or quality)
                          else ImageCheckCache.inspect_uncached)
            if facts is None:
                facts = read_image(
                    image_path, verify=verify,
                    quality_function=image_quality_metrics if quality else None,
                )
            if facts["error"] is not None:
                raise OSError(facts["error"])
            metrics = facts["metrics"]
            if metrics is not None:
                self._quality_records[image_path] = metrics
                # Cached facts, fresh decisions: thresholds may have changed.
                for kind, message in quality_findings(metrics, self._scan_quality_thresholds):
                    self.issues.append(self._make_issue(
                        kind, "info", image_path, message,
                        suggestion="Review image quality before removing or replacing it. Content and resolution affect these scores."
                    ))
            return tuple(facts["size"])
        except Exception as e:
            self.bad_images += 1
            self.issues.append(self._make_issue(
                "unreadable_image",
                "error",
                image_path,
                f"Image could not be opened: {e}",
                suggestion="Move this image to review or replace it with a readable copy.",
            ))
            return None

    def _annotation_type(self, values, preferred_polygon_type=None):
        value_count = len(values)
        expected_pose_count = 4 + (self.expected_keypoint_count * 3)

        if self.expected_keypoint_count > 0 and value_count == expected_pose_count:
            return "bbox_keypoints"
        if value_count > 4 and (value_count - 4) % 3 == 0:
            x_center, y_center, width, height = values[:4]
            bbox_like = (
                0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            )
            visibility_values = values[6::3]
            if bbox_like and visibility_values and all(
                float(v).is_integer() and int(float(v)) in (0, 1, 2)
                for v in visibility_values
            ):
                return "bbox_keypoints"
        if value_count in (4, 5):
            return "bbox"
        if value_count == 8:
            if str(preferred_polygon_type or "").strip().lower() in {"seg", "segment", "segmentation", "polygon"}:
                return "segmentation"
            return "obb"
        if value_count >= 6 and value_count % 2 == 0:
            return "segmentation"
        return "unknown"

    @staticmethod
    def _annotation_family(annotation_type):
        return {
            "bbox": "bbox",
            "bbox_keypoints": "pose",
            "segmentation": "segmentation",
            "obb": "obb",
        }.get(str(annotation_type or "").strip().lower(), "unknown")

    @staticmethod
    def _annotation_family_for_task(task):
        return {
            "detect": "bbox",
            "detection": "bbox",
            "boxes": "bbox",
            "box": "bbox",
            "segment": "segmentation",
            "seg": "segmentation",
            "segmentation": "segmentation",
            "polygon": "segmentation",
            "pose": "pose",
            "keypoint": "pose",
            "keypoints": "pose",
            "obb": "obb",
        }.get(str(task or "").strip().lower(), "")

    @staticmethod
    def _annotation_family_display(family):
        return {
            "bbox": "detection box",
            "segmentation": "segmentation polygon",
            "pose": "pose/keypoints",
            "obb": "oriented box (OBB)",
        }.get(str(family or ""), str(family or "unknown annotation"))

    def _add_dataset_annotation_type_issues(self):
        counts = Counter({
            family: int(count)
            for family, count in self.annotation_family_counts.items()
            if family != "unknown" and int(count) > 0
        })
        if not counts:
            self.dominant_annotation_family = ""
            return

        max_count = max(counts.values())
        dominant_candidates = sorted(family for family, count in counts.items() if count == max_count)
        expected = str(getattr(self, "_scan_expected_annotation_family_override", "") or "")
        dominant = expected if expected in dominant_candidates else dominant_candidates[0]
        self.dominant_annotation_family = dominant

        if len(counts) <= 1:
            return

        count_text = ", ".join(
            f"{self._annotation_family_display(family)}={count}"
            for family, count in sorted(counts.items())
        )
        for record in self.annotation_family_records:
            family = record.get("family", "")
            if family in {"", "unknown", dominant}:
                continue
            self.issues.append(self._make_issue(
                "dataset_mixed_annotation_types",
                "error",
                record.get("file_path", ""),
                (
                    f"Dataset annotations are predominantly {self._annotation_family_display(dominant)}, "
                    f"but this row is {self._annotation_family_display(family)}. "
                    f"Dataset totals: {count_text}."
                ),
                record.get("line_number"),
                record.get("line"),
                suggestion=(
                    f"Convert or remove this {self._annotation_family_display(family)} row so the dataset "
                    f"uses only {self._annotation_family_display(dominant)} annotations."
                ),
            ))

    @staticmethod
    def _visual_outlier_bounds(parsed):
        """Return normalized xyxy bounds for any supported spatial annotation."""
        if not isinstance(parsed, dict):
            return None
        values = list(parsed.get("values", []) or [])
        annotation_type = str(parsed.get("annotation_type", "") or "")
        if annotation_type in {"bbox", "bbox_keypoints"} and len(values) >= 4:
            center_x, center_y, width, height = [float(value) for value in values[:4]]
            bounds = (
                center_x - width / 2.0,
                center_y - height / 2.0,
                center_x + width / 2.0,
                center_y + height / 2.0,
            )
        elif annotation_type in {"segmentation", "obb"} and len(values) >= 6:
            xs = [float(values[index]) for index in range(0, len(values), 2)]
            ys = [float(values[index]) for index in range(1, len(values), 2)]
            bounds = (min(xs), min(ys), max(xs), max(ys))
        else:
            return None
        x1, y1, x2, y2 = bounds
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _visual_outlier_feature(image, bounds):
        """Build a compact color/shape feature from an annotation crop."""
        if image is None or bounds is None:
            return None
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return None
        x1, y1, x2, y2 = bounds
        center_x = (x1 + x2) * 0.5 * width
        center_y = (y1 + y2) * 0.5 * height
        crop_width = max(2.0, (x2 - x1) * width * 1.25)
        crop_height = max(2.0, (y2 - y1) * height * 1.25)
        px1 = max(0, int(math.floor(center_x - crop_width / 2.0)))
        py1 = max(0, int(math.floor(center_y - crop_height / 2.0)))
        px2 = min(width, int(math.ceil(center_x + crop_width / 2.0)))
        py2 = min(height, int(math.ceil(center_y + crop_height / 2.0)))
        if px2 - px1 < 4 or py2 - py1 < 4:
            return None
        crop = image[py1:py2, px1:px2]
        crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]
        ).reshape(-1).astype(np.float32)
        hist_norm = float(np.linalg.norm(histogram))
        if hist_norm > 1e-9:
            histogram /= hist_norm

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        dct = cv2.dct(gray)[:8, :8].reshape(-1)
        dct[0] = 0.0
        dct_norm = float(np.linalg.norm(dct))
        if dct_norm > 1e-9:
            dct /= dct_norm

        feature = np.concatenate((histogram * 0.65, dct * 0.35)).astype(np.float32)
        feature_norm = float(np.linalg.norm(feature))
        return feature / feature_norm if feature_norm > 1e-9 else None

    def _trusted_shape_reference_records(self, class_ids, trusted_keys, excluded_keys=()):
        """Bounded, independent shape references from explicit keep decisions only."""
        class_ids = set(class_ids)
        excluded_keys = set(excluded_keys)
        grouped = defaultdict(list)
        seen = set()
        # Use annotation records, not DINO's filtered feature list: a verified
        # image can still supply a trusted reference without being scanned again.
        for record in self.annotation_family_records:
            class_id = int(record.get("class_id", -1))
            if class_id not in class_ids:
                continue
            key = self._visual_record_trust_key(record)
            if key not in trusted_keys or key in excluded_keys:
                continue
            bounds = self._visual_outlier_bounds(record.get("parsed"))
            image_path = self._normalize_path(record.get("image_path", ""))
            identity = (class_id, os.path.normcase(image_path), bounds)
            if bounds is None or not image_path or identity in seen or not os.path.isfile(image_path):
                continue
            seen.add(identity)
            grouped[class_id].append(dict(record, image_path=image_path, bounds=bounds))
        references = []
        limit = max(self.SHAPE_MIN_TRUSTED_REFERENCES, self.SHAPE_MAX_TRUSTED_REFERENCES_PER_CLASS)
        for records in grouped.values():
            if len(records) < self.SHAPE_MIN_TRUSTED_REFERENCES:
                continue
            records.sort(key=lambda record: (record["image_path"], record["bounds"]))
            if len(records) > limit:
                positions = np.linspace(0, len(records) - 1, limit, dtype=np.int32)
                records = [records[int(position)] for position in positions]
            references.extend(records)
        return references

    def _review_shape_predict_kwargs(self):
        if callable(getattr(self.parent, "_sam_predict_kwargs", None)):
            return dict(self.parent._sam_predict_kwargs())
        return {
            "imgsz": self.parent._sam_imgsz() if callable(getattr(self.parent, "_sam_imgsz", None)) else SAM3_DEFAULT_IMGSZ,
            "verbose": False, "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        }

    def _review_shape_model_identity(self, predict_kwargs):
        """Identify the loaded checkpoint, not a file replaced after it loaded."""
        try:
            model_path = self.parent._sam_model_path()
            model_path = os.path.normcase(os.path.realpath(os.path.abspath(model_path)))
            checkpoint = getattr(self.parent, "_sam_loaded_checkpoint_identity", None)
            if getattr(self.parent, "sam_model", None) is not None:
                if checkpoint is None:
                    return None  # Cannot safely identify a legacy in-memory model.
            else:
                stat = os.stat(model_path)
                checkpoint = (model_path, stat.st_size, stat.st_mtime_ns)
            import ultralytics
            return {"checkpoint": checkpoint, "predict": predict_kwargs,
                    "ultralytics": ultralytics.__version__, "torch": torch.__version__}
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    def _review_shape_mask(self, image, bounds, predict_kwargs=None):
        """Use the same BGR crop and SAM3 box prompt for references and candidates."""
        crop = shape_annotation_crop(image, bounds)
        if crop is None:
            return None
        height, width = crop.shape[:2]
        with torch.inference_mode():
            results = self.parent.sam_model.predict(
                source=crop, bboxes=[[0.0, 0.0, float(width), float(height)]],
                **(predict_kwargs if predict_kwargs is not None else self._review_shape_predict_kwargs()),
            )
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            return None
        mask = results[0].masks.data[0].detach().float().cpu().numpy()
        if mask.ndim != 2 or not np.isfinite(mask).all():
            return None
        mask = (mask > 0).astype(np.uint8)
        return mask if mask.any() else None

    def _scan_trusted_candidate_shapes(self, candidates, trusted_keys, should_cancel, status):
        """Never build a class shape baseline from the candidates being judged."""
        results = [None] * len(candidates)
        coverage = {
            "enabled": True, "status": "pending", "backend": "sam3", "device": "",
            "reference_source": "confirmed_review_examples", "reference_annotations": 0,
            "references_processed": 0, "candidates_processed": 0,
            "reference_cache_hits": 0,
            "class_profiles_built": 0, "shape_mismatches_detected": 0,
            "suppressed_strong_matches": 0, "semantic_conflicts": 0, "error": "",
        }
        cache = None
        try:
            references = self._trusted_shape_reference_records(
                {candidate["class_id"] for candidate in candidates}, trusted_keys,
                {self._visual_record_trust_key(candidate["record"]) for candidate in candidates},
            )
            coverage["reference_annotations"] = len(references)
            if should_cancel():
                coverage["status"] = "canceled"
                return results, coverage
            if not references:
                coverage["status"] = "insufficient_trusted_references"
                return results, coverage
            predict_kwargs = self._review_shape_predict_kwargs()
            coverage["device"] = str(predict_kwargs.get("device", ""))
            model_identity = self._review_shape_model_identity(predict_kwargs)
            cache_dir = getattr(self, "_shape_reference_cache_dir_override", None) or os.path.join(
                APP_DIR, ".darkfusion_cache", "review_similarity",
            )
            cache = ShapeReferenceCache(os.path.join(cache_dir, "shape_reference_metrics.sqlite3"), model_identity)
            model_ready = False
            def ensure_model_ready():
                nonlocal model_ready
                if not model_ready:
                    model_ready = bool(self.parent.ensure_sam_model_loaded())
                    # Never write under a planned identity if weights changed
                    # while loading. Unknown model identity just disables reuse.
                    if self._review_shape_model_identity(predict_kwargs) != model_identity:
                        cache.disabled = True
                return model_ready
            profiler = ShapeProfiler()
            reference_groups = OrderedDict()
            for reference in references:
                reference_groups.setdefault(reference["image_path"], []).append(reference)
            status(f"Generating SAM3 shapes for {len(references)} confirmed-good references...")
            for image_path, records in reference_groups.items():
                if should_cancel():
                    coverage["status"] = "canceled"
                    return results, coverage
                try:
                    image_bytes = Path(image_path).read_bytes()
                except OSError:
                    continue
                image = None
                for record in records:
                    if should_cancel():
                        coverage["status"] = "canceled"
                        return results, coverage
                    try:
                        cache_key = cache.key(image_path, record, image_bytes) if model_identity else None
                        metrics = cache.get(cache_key) if cache_key else None
                        if metrics is not None:
                            coverage["reference_cache_hits"] += 1
                        else:
                            if not ensure_model_ready():
                                coverage.update(status="unavailable", error="SAM3 model not available")
                                return results, coverage
                            if image is None:
                                image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if image is None:
                                continue
                            coverage["references_processed"] += 1
                            mask = self._review_shape_mask(image, record["bounds"], predict_kwargs)
                            metrics = profiler.analyze_mask_shape(mask)
                            if metrics and cache_key:
                                cache.put(cache_key, metrics)
                        if metrics:
                            profiler.add_shape_sample(int(record["class_id"]), metrics)
                    except Exception as error:
                        logger.debug("Trusted SAM3 shape unavailable for %s: %s", image_path, error)
            profiler.build_profiles(min_samples=self.SHAPE_MIN_TRUSTED_REFERENCES)
            coverage["class_profiles_built"] = len(profiler.class_profiles)
            if not profiler.class_profiles:
                coverage["status"] = "insufficient_trusted_masks"
                return results, coverage
            if should_cancel():
                coverage["status"] = "canceled"
                return results, coverage
            if not ensure_model_ready():
                coverage.update(status="unavailable", error="SAM3 model not available")
                return results, coverage
            candidate_groups = OrderedDict()
            for index, candidate in enumerate(candidates):
                if candidate["class_id"] not in profiler.class_profiles:
                    continue
                record = candidate["record"]
                candidate_groups.setdefault(record.get("image_path", ""), []).append((index, candidate))
            status("Checking unresolved candidates against confirmed-good SAM3 shapes...")
            for image_path, entries in candidate_groups.items():
                if should_cancel():
                    coverage["status"] = "canceled"
                    return results, coverage
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                for index, candidate in entries:
                    if should_cancel():
                        coverage["status"] = "canceled"
                        return results, coverage
                    coverage["candidates_processed"] += 1
                    try:
                        mask = self._review_shape_mask(image, candidate["record"].get("bounds"), predict_kwargs)
                        result = shape_verification_score(candidate["class_id"], mask, profiler)
                        results[index] = result
                        if result.get("class_matches") is False:
                            coverage["shape_mismatches_detected"] += 1
                    except Exception as error:
                        logger.debug("Candidate SAM3 shape unavailable for %s: %s", image_path, error)
            coverage["status"] = "complete"
        except Exception as error:
            coverage.update(status="canceled" if should_cancel() else "error", error=str(error)[:1000])
            logger.warning("Trusted shape verification unavailable: %s", error)
        finally:
            if cache is not None:
                cache.close()
        return results, coverage

    def _scan_visual_outliers(self, should_cancel=lambda: False, progress_callback=None):
        """Flag conservative within-class appearance outliers for human review."""
        class_totals = Counter(int(record.get("class_id", -1)) for record in self.annotation_family_records)
        try:
            requested_passes = int(
                getattr(self, "_scan_visual_outlier_passes_override", None)
                or self.VISUAL_OUTLIER_DEFAULT_DEEP_PASSES
            )
        except (TypeError, ValueError):
            requested_passes = self.VISUAL_OUTLIER_DEFAULT_DEEP_PASSES
        requested_passes = max(1, min(100, requested_passes))
        semantic_verification_enabled = bool(
            getattr(self, "_scan_visual_class_verify_override", False)
        )
        shape_verification_enabled = bool(
            getattr(self, "_scan_shape_verify_override", False)
        )
        coverage = {
            "status": "running", "backend": "not_run", "device": "", "fallback_reason": "",
            "total_annotations": len(self.annotation_family_records), "encoded_annotations": 0,
            "compared_annotations": 0, "skipped_unavailable": len(self.annotation_family_records),
            "skipped_small_class": 0, "sampled_classes": 0, "candidates_before_cap": 0,
            "queued_findings": 0, "omitted_by_cap": 0,
            "requested_passes": requested_passes, "completed_passes": 0,
            "queued_before_vetting": 0,
            "trusted_examples": 0,
            "suppressed_by_trusted_examples": 0,
            "class_semantic_verification": {
                "enabled": semantic_verification_enabled,
                "status": "pending" if semantic_verification_enabled else "disabled",
                "backend": "siglip2_or_clip" if semantic_verification_enabled else "not_run",
                "device": "",
                "candidates_checked": 0,
                "suppressed_strong_matches": 0,
                "high_priority_findings": 0,
                "moderate_priority_findings": 0,
                "cached_results": 0,
                "error": "",
            },
            "minimum_class_samples": self.VISUAL_OUTLIER_MIN_CLASS_SAMPLES,
            "maximum_class_fraction": self.VISUAL_OUTLIER_MAX_CLASS_FRACTION,
            "maximum_findings_per_class": self.VISUAL_OUTLIER_MAX_FINDINGS_PER_CLASS,
            "exact_class_limit": self.VISUAL_OUTLIER_EXACT_CLASS_LIMIT,
            "reference_limit": self.VISUAL_OUTLIER_REFERENCE_LIMIT,
            "classes": [
                {"class_id": class_id, "class_name": self._class_name_for_index(class_id),
                 "annotations": count, "encoded": 0, "compared": 0, "references": 0,
                 "sampled": False, "cutoff": None, "candidate_count": 0,
                 "queued": 0, "omitted_by_cap": 0, "passes_completed": 0,
                 "queued_before_vetting": 0, "trusted_examples": 0,
                 "suppressed_by_trusted_examples": 0,
                 "suppressed_by_class_semantics": 0,
                 "suppressed_by_trusted_shape": 0,
                 "suppressed_by_llm": 0,
                 "high_priority_findings": 0,
                 "moderate_priority_findings": 0,
                 "pass_details": [], "status": "unavailable"}
                for class_id, count in sorted(class_totals.items())
            ],
        }
        self.visual_outlier_scan = coverage
        coverage_by_class = {row["class_id"]: row for row in coverage["classes"]}
        records_by_image = OrderedDict()
        for record in self.annotation_family_records:
            bounds = self._visual_outlier_bounds(record.get("parsed"))
            image_path = self._normalize_path(record.get("image_path", ""))
            if bounds is not None and image_path and os.path.isfile(image_path):
                item = dict(record)
                item["bounds"] = bounds
                records_by_image.setdefault(image_path, []).append(item)
        
        # Filter out verified images to avoid re-flagging cleaned items
        dataset_root = self.base_directory
        if dataset_root:
            try:
                verified_images = get_verified_images_for_scan(dataset_root)
                original_count = len(records_by_image)
                records_by_image = OrderedDict(
                    (path, records) for path, records in records_by_image.items()
                    if not any(
                        str(path).replace("\\", "/").endswith(v) or 
                        os.path.normpath(os.path.relpath(path, dataset_root)).replace("\\", "/") in verified_images
                        for v in verified_images
                    )
                )
                skipped_verified = original_count - len(records_by_image)
                if skipped_verified > 0:
                    coverage["skipped_verified"] = skipped_verified
                    logger.info(f"Skipping {skipped_verified} verified images in visual outlier scan")
            except Exception as e:
                logger.warning(f"Could not filter verified images: {e}")

        feature_records = []
        matcher = None
        cancellation_type = ()
        try:
            # DINOv3-first descriptors; shared cache/models with the review filter.
            from darkfusion_review_similarity import ReviewEmbeddingMatcher, ReviewSimilarityCancelled
            cancellation_type = ReviewSimilarityCancelled
            app_dir = os.path.dirname(os.path.abspath(__file__))
            matcher_records = [
                {"image_file": image_path, "bounds": record["bounds"]}
                for image_path, records in records_by_image.items()
                for record in records
            ]
            matcher = ReviewEmbeddingMatcher(
                os.path.join(app_dir, ".darkfusion_cache", "review_similarity"),
                status=(
                    lambda m: progress_callback(0, 0, m)
                    if callable(progress_callback) else None
                ),
                cancelled=should_cancel,
                model_key="dinov3_base",
                models_dir=model_directory(os.path.join(app_dir, "Sam")),
                progress=(
                    lambda completed, total: progress_callback(
                        int(completed), max(1, int(total) + 1),
                        "Encoding annotations with DINOv3...",
                    )
                    if callable(progress_callback) else None
                ),
            )
            if not matcher_records:
                # Nothing to embed; skip loading the DINOv3 model entirely.
                matcher.close()
                matcher = None
                coverage["status"] = "no_usable_crops"
                return
            matcher.prepare()
            coverage["backend"] = "dinov3"
            coverage["device"] = str(getattr(matcher, "device", "") or "")
            if callable(progress_callback):
                progress_callback(
                    0, len(matcher_records) + 1,
                    "Encoding annotations with DINOv3...",
                )
            vectors = matcher.encode_records(matcher_records)
            for record, vector in zip(
                (r for rs in records_by_image.values() for r in rs), vectors
            ):
                if vector is not None:
                    record["feature"] = vector
                    feature_records.append(record)
        except cancellation_type:
            coverage["status"] = "canceled"
            return
        except Exception as exc:
            # Continue without blocking the UI, and report the method actually used.
            if matcher is not None:
                try:
                    matcher.close()
                except Exception:
                    pass
                matcher = None
            if should_cancel():
                # The user stopped the scan; do not start the CPU fallback.
                coverage["status"] = "canceled"
                return
            coverage.update(backend="cpu_histogram_dct", device="cpu", fallback_reason=str(exc)[:1000])
            logger.warning("Dataset Analysis visual scan is using CPU appearance features: %s", exc)
            if callable(progress_callback):
                progress_callback(0, max(1, len(records_by_image)),
                                  "DINOv3 unavailable; using CPU color/shape features...")
            feature_records = []
            total_images = len(records_by_image)
            for image_number, (image_path, records) in enumerate(records_by_image.items(), start=1):
                if should_cancel():
                    coverage["status"] = "canceled"
                    return
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                for record in records:
                    record.pop("feature", None)
                    feature = self._visual_outlier_feature(image, record.get("bounds"))
                    if feature is not None:
                        record["feature"] = feature
                        feature_records.append(record)
                if callable(progress_callback) and (
                    image_number == total_images or image_number % 50 == 0
                ):
                    progress_callback(
                        image_number,
                        max(1, total_images + 1),
                        "Checking within-class visual consistency...",
                    )
        finally:
            if matcher is not None:
                matcher.close()

        coverage["encoded_annotations"] = len(feature_records)
        coverage["skipped_unavailable"] = coverage["total_annotations"] - len(feature_records)
        grouped = OrderedDict()
        for record in feature_records:
            grouped.setdefault(int(record.get("class_id", -1)), []).append(record)

        comparison_total = len(feature_records)
        comparison_maximum = max(1, comparison_total * requested_passes + 1)
        comparison_completed = 0
        if callable(progress_callback):
            progress_callback(
                0, comparison_maximum,
                "Comparing same-class DINOv3 features...",
            )

        configured_threshold = getattr(
            self, "_scan_visual_outlier_threshold_override", None
        )
        try:
            configured_threshold = float(configured_threshold)
        except (TypeError, ValueError):
            configured_threshold = 0.45
        configured_threshold = max(0.20, min(0.90, configured_threshold))

        dataset_dir = self._normalize_path(
            self.base_directory
            or (os.path.dirname(feature_records[0].get("image_path", "")) if feature_records else "")
        )
        trusted_annotation_keys = self._load_visual_trusted_annotations(dataset_dir)
        pending_candidates = []

        for class_id, class_records in grouped.items():
            if should_cancel():
                coverage["status"] = "canceled"
                return
            count = len(class_records)
            class_coverage = coverage_by_class[class_id]
            class_coverage["encoded"] = count
            if count < self.VISUAL_OUTLIER_MIN_CLASS_SAMPLES:
                class_coverage["status"] = "too_few_examples"
                coverage["skipped_small_class"] += count
                comparison_completed += count
                if callable(progress_callback):
                    progress_callback(
                        comparison_completed, comparison_maximum,
                        "Comparing same-class DINOv3 features...",
                    )
                continue
            matrix = np.vstack([record["feature"] for record in class_records])
            gpu_matrix = None
            if (
                coverage.get("backend") == "dinov3"
                and str(coverage.get("device", "")).startswith("cuda")
                and torch.cuda.is_available()
            ):
                try:
                    gpu_matrix = torch.from_numpy(matrix).to(
                        device="cuda", dtype=torch.float32
                    )
                except (RuntimeError, MemoryError) as error:
                    logger.warning(
                        "DINOv3 deep comparison is using CPU after GPU allocation failed: %s",
                        error,
                    )
                    gpu_matrix = None
            class_coverage["comparison_device"] = "cuda" if gpu_matrix is not None else "cpu"
            active_indices = np.arange(count, dtype=np.int32)
            queued_indices = set()
            candidate_indices_seen = set()
            queued_results = []
            class_cutoffs = []
            sampled = False
            first_reference_count = 0
            passes_completed = 0

            for pass_index in range(requested_passes):
                if should_cancel():
                    coverage["status"] = "canceled"
                    return
                active_count = len(active_indices)
                if active_count < self.VISUAL_OUTLIER_MIN_CLASS_SAMPLES:
                    break

                if active_count <= self.VISUAL_OUTLIER_EXACT_CLASS_LIMIT:
                    reference_indices = active_indices.copy()
                else:
                    reference_positions = np.linspace(
                        0,
                        active_count - 1,
                        self.VISUAL_OUTLIER_REFERENCE_LIMIT,
                        dtype=np.int32,
                    )
                    reference_indices = active_indices[np.unique(reference_positions)]
                    sampled = True
                if not first_reference_count:
                    first_reference_count = len(reference_indices)
                reference_matrix = matrix[reference_indices] if gpu_matrix is None else None
                reference_tensor = None
                if gpu_matrix is not None:
                    reference_tensor = gpu_matrix.index_select(
                        0,
                        torch.as_tensor(
                            reference_indices, dtype=torch.long, device="cuda"
                        ),
                    )
                reference_columns = {
                    int(record_index): column
                    for column, record_index in enumerate(reference_indices)
                }
                nearest = np.full(active_count, -1.0, dtype=np.float32)
                chunk_size = 256
                for start in range(0, active_count, chunk_size):
                    if should_cancel():
                        coverage["status"] = "canceled"
                        return
                    stop = min(active_count, start + chunk_size)
                    batch_indices = active_indices[start:stop]
                    if gpu_matrix is not None:
                        with torch.inference_mode():
                            batch_tensor = gpu_matrix.index_select(
                                0,
                                torch.as_tensor(
                                    batch_indices, dtype=torch.long, device="cuda"
                                ),
                            )
                            similarities = torch.matmul(
                                batch_tensor, reference_tensor.T
                            )
                            for row, record_index in enumerate(batch_indices):
                                reference_column = reference_columns.get(int(record_index))
                                if reference_column is not None:
                                    similarities[row, reference_column] = -1.0
                            nearest[start:stop] = (
                                similarities.max(dim=1).values.cpu().numpy()
                            )
                    else:
                        similarities = np.matmul(
                            matrix[batch_indices], reference_matrix.T
                        )
                        for row, record_index in enumerate(batch_indices):
                            reference_column = reference_columns.get(int(record_index))
                            if reference_column is not None:
                                similarities[row, reference_column] = -1.0
                        nearest[start:stop] = similarities.max(axis=1)
                    if callable(progress_callback):
                        progress_callback(
                            comparison_completed + pass_index * count + stop,
                            comparison_maximum,
                            f"Comparing same-class DINOv3 features — deep pass {pass_index + 1}/{requested_passes}...",
                        )

                median = float(np.median(nearest))
                mad = float(np.median(np.abs(nearest - median)))
                adaptive_threshold = max(0.20, median - max(0.12, mad * 3.0))
                cutoff = min(configured_threshold, adaptive_threshold)
                candidate_positions = [
                    index for index, score in enumerate(nearest)
                    if float(score) < cutoff
                ]
                candidate_positions.sort(key=lambda index: float(nearest[index]))
                candidate_globals = [int(active_indices[index]) for index in candidate_positions]
                candidate_indices_seen.update(candidate_globals)

                fraction_limit = max(
                    1,
                    int(math.ceil(active_count * self.VISUAL_OUTLIER_MAX_CLASS_FRACTION)),
                )
                if self.VISUAL_OUTLIER_MAX_FINDINGS_PER_CLASS is not None:
                    per_class_cap = int(self.VISUAL_OUTLIER_MAX_FINDINGS_PER_CLASS)
                    if per_class_cap <= 0:
                        logger.warning(
                            "VISUAL_OUTLIER_MAX_FINDINGS_PER_CLASS is %s; using fraction limit instead",
                            per_class_cap,
                        )
                        maximum = fraction_limit
                    else:
                        maximum = min(per_class_cap, fraction_limit)
                else:
                    maximum = fraction_limit

                selected_positions = candidate_positions[:maximum]
                selected_globals = [int(active_indices[index]) for index in selected_positions]
                for position, global_index in zip(selected_positions, selected_globals):
                    if global_index in queued_indices:
                        continue
                    queued_indices.add(global_index)
                    queued_results.append((
                        global_index,
                        max(0.0, min(1.0, float(nearest[position]))),
                        pass_index + 1,
                        cutoff,
                    ))

                passes_completed = pass_index + 1
                class_cutoffs.append(cutoff)
                class_coverage["pass_details"].append({
                    "pass": passes_completed,
                    "active_annotations": active_count,
                    "references": len(reference_indices),
                    "cutoff": cutoff,
                    "candidates": len(candidate_positions),
                    "queued": len(selected_globals),
                })
                if not selected_globals:
                    break
                selected_set = set(selected_globals)
                active_indices = np.asarray(
                    [index for index in active_indices if int(index) not in selected_set],
                    dtype=np.int32,
                )

            omitted_indices = candidate_indices_seen - queued_indices
            trusted_indices = {
                index for index, record in enumerate(class_records)
                if self._visual_record_trust_key(record) in trusted_annotation_keys
            }
            class_coverage["trusted_examples"] = len(trusted_indices)
            coverage["trusted_examples"] += len(trusted_indices)
            vetted_results = []
            for index, score, pass_number, cutoff in queued_results:
                protected, trusted_score, trusted_top3 = self._trusted_visual_similarity(
                    matrix, index, trusted_indices
                )
                if protected:
                    class_coverage["suppressed_by_trusted_examples"] += 1
                    coverage["suppressed_by_trusted_examples"] += 1
                    continue
                vetted_results.append({
                    "class_id": class_id,
                    "class_coverage": class_coverage,
                    "record": class_records[index],
                    "score": score,
                    "pass_number": pass_number,
                    "cutoff": cutoff,
                    "trusted_similarity": trusted_score,
                    "trusted_top3_similarity": trusted_top3,
                })
            pending_candidates.extend(vetted_results)
            first_cutoff = class_cutoffs[0] if class_cutoffs else None
            class_coverage.update(
                status="compared",
                compared=count,
                references=first_reference_count,
                sampled=sampled,
                cutoff=first_cutoff,
                candidate_count=len(candidate_indices_seen),
                queued=0,
                queued_before_vetting=len(queued_indices),
                omitted_by_cap=len(omitted_indices),
                passes_completed=passes_completed,
            )
            coverage["sampled_classes"] += int(sampled)
            coverage["completed_passes"] = max(
                coverage["completed_passes"], passes_completed
            )
            coverage["compared_annotations"] += count
            coverage["candidates_before_cap"] += len(candidate_indices_seen)
            coverage["queued_before_vetting"] += len(queued_indices)
            coverage["omitted_by_cap"] += len(omitted_indices)

            if gpu_matrix is not None:
                del gpu_matrix

            comparison_completed += count * requested_passes
            if callable(progress_callback):
                progress_callback(
                    comparison_completed, comparison_maximum,
                    "Finalizing DINOv3 findings...",
                )

        semantic_results = [None] * len(pending_candidates)
        semantic_coverage = coverage["class_semantic_verification"]
        if semantic_verification_enabled and pending_candidates and coverage.get("backend") == "dinov3":
            verifier = None
            try:
                from darkfusion_class_semantics import ClassSemanticVerifier
                app_dir = os.path.dirname(os.path.abspath(__file__))
                verifier = ClassSemanticVerifier(
                    os.path.join(
                        app_dir, ".darkfusion_cache", "review_similarity",
                        "class_semantics.sqlite3",
                    ),
                    device="cuda",
                    status=(
                        lambda message: progress_callback(
                            comparison_maximum - 1, comparison_maximum, message
                        ) if callable(progress_callback) else None
                    ),
                    cancelled=should_cancel,
                )
                scored_semantics = verifier.score_records([
                    {
                        "image_file": candidate["record"].get("image_path", ""),
                        "bounds": candidate["record"].get("bounds"),
                        "class_name": self._class_name_for_index(candidate["class_id"]),
                    }
                    for candidate in pending_candidates
                ])
                # An optional verifier must not make annotations disappear if
                # it returns incomplete evidence. Missing entries stay unknown.
                semantic_results = list(scored_semantics or [])[:len(pending_candidates)]
                semantic_results.extend([None] * (len(pending_candidates) - len(semantic_results)))
                semantic_coverage.update(
                    status="complete" if len(scored_semantics or []) == len(pending_candidates) else "partial",
                    backend=str(getattr(verifier, "backend", "") or "unknown"),
                    device=str(getattr(verifier, "device", "") or ""),
                    candidates_checked=sum(isinstance(result, dict) and not result.get("error") for result in semantic_results),
                    cached_results=sum(isinstance(result, dict) and bool(result.get("cached")) for result in semantic_results),
                )
            except Exception as error:
                if should_cancel():
                    coverage["status"] = "canceled"
                    return
                semantic_coverage.update(status="unavailable", error=str(error)[:1000])
                logger.warning("Class-name verification is unavailable: %s", error)
            finally:
                if verifier is not None:
                    verifier.close()
        elif semantic_verification_enabled and not pending_candidates:
            semantic_coverage["status"] = "no_candidates"
        elif semantic_verification_enabled:
            semantic_coverage.update(
                status="skipped_without_dinov3",
                error="Class-name protection requires the DINOv3 scan backend.",
            )

        if should_cancel():
            coverage["status"] = "canceled"
            return

        if semantic_verification_enabled:
            try:
                from darkfusion_class_semantics import (
                    false_positive_evidence,
                    should_protect_class_match,
                    class_semantic_conflict,
                )
            except Exception:
                should_protect_class_match = lambda _result, **_kwargs: False
                class_semantic_conflict = lambda _result: True
                false_positive_evidence = lambda _result, **_kwargs: {
                    "priority": 0.0, "strength": "review",
                    "semantic_mismatch": 0.0, "dino_anomaly": 0.0,
                }
        else:
            should_protect_class_match = lambda _result, **_kwargs: False
            class_semantic_conflict = lambda _result: False
            false_positive_evidence = lambda _result, **_kwargs: {
                "priority": 0.0, "strength": "review",
                "semantic_mismatch": 0.0, "dino_anomaly": 0.0,
            }

        # Apply the existing conservative semantic rule before any SAM3 work.
        remaining_candidates, remaining_semantics = [], []
        for candidate, semantic_result in zip(pending_candidates, semantic_results):
            if should_protect_class_match(semantic_result, scan_pass=candidate["pass_number"]):
                candidate["class_coverage"]["suppressed_by_class_semantics"] += 1
                semantic_coverage["suppressed_strong_matches"] += 1
                continue
            remaining_candidates.append(candidate)
            remaining_semantics.append(semantic_result)
        pending_candidates, semantic_results = remaining_candidates, remaining_semantics

        shape_results = [None] * len(pending_candidates)
        shape_coverage = {
            "enabled": shape_verification_enabled,
            "status": "disabled" if not shape_verification_enabled else "no_candidates",
            "backend": "sam3", "device": "", "reference_source": "confirmed_review_examples",
            "reference_annotations": 0, "references_processed": 0, "candidates_processed": 0,
            "reference_cache_hits": 0,
            "class_profiles_built": 0, "shape_mismatches_detected": 0,
            "suppressed_strong_matches": 0, "semantic_conflicts": 0, "error": "",
            "skipped_after_semantics": semantic_coverage["suppressed_strong_matches"],
        }
        if shape_verification_enabled and pending_candidates:
            shape_results, checked_coverage = self._scan_trusted_candidate_shapes(
                pending_candidates, trusted_annotation_keys, should_cancel,
                lambda message: progress_callback(
                    comparison_maximum - 1, comparison_maximum, message
                ) if callable(progress_callback) else None,
            )
            shape_coverage.update(checked_coverage)
        coverage["shape_verification"] = shape_coverage
        if should_cancel() or shape_coverage["status"] == "canceled":
            coverage["status"] = "canceled"
            return

        # LLM-based borderline verification for 10-70% similarity range
        llm_coverage = {
            "enabled": True,
            "status": "not_needed" if coverage.get("backend") == "dinov3" else "skipped_without_dinov3",
            "model": "not_run",
            "candidates_checked": 0,
            "cached_results": 0,
            "high_confidence_protections": 0,
            "low_confidence_flags": 0,
            "evidence_conflicts": 0,
            "unavailable_reason": "",
            "error": "",
        }
        
        llm_verifier = None
        llm_setup_attempted = False

        for candidate, semantic_result, shape_result in zip(
            pending_candidates, semantic_results, shape_results
        ):
            if should_cancel():
                coverage["status"] = "canceled"
                return
            class_coverage = candidate["class_coverage"]
            record = candidate["record"]
            class_id = candidate["class_id"]
            score = candidate["score"]
            pass_number = candidate["pass_number"]
            cutoff = candidate["cutoff"]
            
            semantic_conflict = class_semantic_conflict(semantic_result)
            shape_matches = isinstance(shape_result, dict) and shape_result.get("class_matches") is True
            shape_mismatch = isinstance(shape_result, dict) and shape_result.get("class_matches") is False
            verification_conflict = bool(shape_matches and semantic_conflict)
            if verification_conflict:
                shape_coverage["semantic_conflicts"] += 1
            if shape_matches and not semantic_conflict:
                class_coverage["suppressed_by_trusted_shape"] += 1
                shape_coverage["suppressed_strong_matches"] += 1
                continue
            
            # LLM-based borderline verification for 10-70% similarity range
            # Limit: Only check top 50 borderline cases to avoid excessive slowdown
            llm_protected = False
            llm_result = None
            llm_candidates_checked = llm_coverage.get("candidates_checked", 0)
            max_llm_candidates = 50  # Limit to prevent timeout
            llm_eligible = (
                coverage.get("backend") == "dinov3" and 0.10 <= score <= 0.70
                and llm_candidates_checked < max_llm_candidates
            )
            # No server/model setup until an unresolved, eligible object actually
            # reaches this stage. Clear earlier results never incur LLM setup.
            if llm_eligible and not llm_setup_attempted:
                llm_setup_attempted = True
                try:
                    app_dir = os.path.dirname(os.path.abspath(__file__))
                    llm_verifier = BorderlineLLMVerifier(
                        cache_dir=os.path.join(app_dir, ".darkfusion_cache", "llm_borderline"),
                        enabled=True,
                    )
                    if llm_verifier.health_check():
                        llm_coverage.update(status="ready", model=str(llm_verifier.available_model))
                        logger.info("LLM borderline verifier ready: %s", llm_verifier.available_model)
                    else:
                        llm_coverage.update(status="unavailable", unavailable_reason="Ollama not running or no vision models available")
                        llm_verifier = None
                except Exception as error:
                    llm_coverage.update(status="error", error=str(error)[:500])
                    logger.warning("Could not initialize LLM verifier: %s", error)
                    llm_verifier = None
            if should_cancel():
                coverage["status"] = "canceled"
                return
            if (0.10 <= score <= 0.70 and llm_verifier and 
                llm_coverage.get("status") == "ready" and 
                llm_candidates_checked < max_llm_candidates):
                try:
                    image_path = record.get("image_path", "")
                    bounds = record.get("bounds")
                    class_name = self._class_name_for_index(class_id)
                    
                    if image_path and os.path.isfile(image_path):
                        llm_coverage["candidates_checked"] += 1
                        llm_result = llm_verifier.verify_borderline_candidate(
                            image_path, class_name, bounds
                        )
                        if llm_result.get("cached"):
                            llm_coverage["cached_results"] += 1
                        
                        # If LLM confirms label with high confidence, protect it
                        if (llm_result.get("verified") is True and 
                            llm_result.get("confidence") == "high" and not llm_result.get("error")):
                            if semantic_conflict or shape_mismatch:
                                verification_conflict = True
                                llm_coverage["evidence_conflicts"] += 1
                            else:
                                llm_protected = True
                                llm_coverage["high_confidence_protections"] += 1
                                class_coverage["suppressed_by_llm"] += 1
                        elif llm_result.get("verified") is False:
                            llm_coverage["low_confidence_flags"] += 1
                        if llm_protected:
                            logger.debug(
                                f"LLM protected {class_name} annotation (score={score:.2f}): "
                                f"{llm_result.get('reasoning', '')[:100]}"
                            )
                except Exception as e:
                    logger.warning(f"LLM verification error for candidate: {e}")
            
            if llm_protected:
                continue
            
            evidence = false_positive_evidence(
                semantic_result,
                visual_similarity=score,
                visual_cutoff=cutoff,
                scan_pass=pass_number,
            )
            evidence_strength = str(evidence.get("strength", "review"))
            if evidence_strength == "high":
                class_coverage["high_priority_findings"] += 1
                if isinstance(semantic_result, dict) and not semantic_result.get("error"):
                    semantic_coverage["high_priority_findings"] += 1
            elif evidence_strength == "moderate":
                class_coverage["moderate_priority_findings"] += 1
                if isinstance(semantic_result, dict) and not semantic_result.get("error"):
                    semantic_coverage["moderate_priority_findings"] += 1
            priority_prefix = {
                "high": "High-priority possible false positive. ",
                "moderate": "Moderate-priority possible false positive. ",
            }.get(evidence_strength, "")
            issue = self._make_issue(
                "visual_class_outlier",
                "warning" if evidence_strength == "high" else "info",
                record.get("file_path", ""),
                (
                    priority_prefix
                    + f"This {self._class_name_for_index(class_id)} annotation's nearest "
                    f"same-class visual match scored {score * 100.0:.1f}%."
                ),
                record.get("line_number"),
                record.get("line"),
                suggestion=(
                    "Review this as a possible false positive: check whether the object and class are correct. "
                    "DINOv3 similarity is supporting evidence, not proof that the label is wrong."
                ),
            )
            issue.update(
                verification_conflict=verification_conflict,
                visual_similarity=round(score, 6),
                visual_scan_pass=int(pass_number),
                visual_cutoff=round(float(cutoff), 6),
                trusted_similarity=(
                    round(float(candidate["trusted_similarity"]), 6)
                    if candidate["trusted_similarity"] is not None else None
                ),
                class_semantic_probability=(
                    semantic_result.get("class_probability")
                    if isinstance(semantic_result, dict) else None
                ),
                class_semantic_margin=(
                    semantic_result.get("semantic_margin")
                    if isinstance(semantic_result, dict) else None
                ),
                class_semantic_similarity=(
                    semantic_result.get("class_similarity")
                    if isinstance(semantic_result, dict) else None
                ),
                class_semantic_backend=(
                    semantic_result.get("backend")
                    if isinstance(semantic_result, dict) else None
                ),
                shape_score=(
                    round(float(shape_result.get("shape_score")), 6)
                    if isinstance(shape_result, dict) and shape_result.get("shape_score") is not None else None
                ),
                shape_class_match=(
                    shape_result.get("class_matches")
                    if isinstance(shape_result, dict) else None
                ),
                llm_verified=(
                    llm_result.get("verified")
                    if isinstance(llm_result, dict) else None
                ),
                llm_confidence=(
                    llm_result.get("confidence")
                    if isinstance(llm_result, dict) else None
                ),
                llm_reasoning=(
                    llm_result.get("reasoning")
                    if isinstance(llm_result, dict) else None
                ),
                llm_suggested_class=(
                    llm_result.get("suggested_class")
                    if isinstance(llm_result, dict) else None
                ),
                false_positive_priority=evidence.get("priority"),
                false_positive_strength=evidence_strength,
                semantic_mismatch=evidence.get("semantic_mismatch"),
                dino_anomaly=evidence.get("dino_anomaly"),
            )
            class_coverage["queued"] += 1
            coverage["queued_findings"] += 1
            self.visual_outlier_candidates += 1
            self.issues.append(issue)

        # Finalize LLM coverage
        if llm_verifier:
            try:
                llm_coverage["status"] = "complete"
            except Exception:
                pass
        coverage["llm_borderline_verification"] = llm_coverage

        coverage["status"] = "complete"

    @staticmethod
    def _visual_scan_summary(summary):
        """Format saved scan evidence without loading images or models in the UI."""
        coverage = summary.get("visual_outlier_scan") or {}
        status = coverage.get("status", "not_recorded")
        if status == "not_run":
            return "Possible false-positive scan was not run. Enable the DINOv3 similarity check and press Start Scan.", ""
        if status == "not_recorded":
            return "Possible false-positive coverage was not recorded in this older report. Press Start Scan to update it.", ""
        method = {"dinov3": "DINOv3", "cpu_histogram_dct": "CPU color/shape fallback", "not_run": "No visual method ran"}.get(
            coverage.get("backend"), "Unknown visual method")
        device = coverage.get("device", "")
        if device:
            method += " (" + ("GPU" if device.startswith("cuda") else device.upper()) + ")"
        total = int(coverage.get("total_annotations", 0))
        compared = int(coverage.get("compared_annotations", 0))
        minimum = int(coverage.get("minimum_class_samples", 5))
        text = (
            f"Possible false-positive scan: {method}. Compared {compared:,}/{total:,} annotation crops. "
            f"Skipped: {int(coverage.get('skipped_unavailable', 0)) + int(coverage.get('skipped_small_class', 0)):,} crops (see details)."
        )
        if status != "complete":
            text += f" Status: {status.replace('_', ' ')}."
        if coverage.get("backend") == "cpu_histogram_dct":
            text += " DINOv3 did not complete; see details for the reason."
        max_per_class = coverage.get('maximum_findings_per_class')
        cap_text = "no per-class cap" if max_per_class is None else f"maximum {int(max_per_class)}"
        text += (
            f" Per-class result cap: {float(coverage.get('maximum_class_fraction', .03)) * 100:g}% "
            f"(rounded up, {cap_text}). "
            f"{int(coverage.get('omitted_by_cap', 0)):,} candidates omitted by cap. "
            f"{int(coverage.get('sampled_classes', 0)):,} classes used sampled comparisons. "
            f"Deep passes completed: {int(coverage.get('completed_passes', 1))}/"
            f"{int(coverage.get('requested_passes', 1))}."
        )
        semantic = coverage.get("class_semantic_verification", {}) or {}
        if semantic.get("enabled"):
            text += (
                f" Class-name verification: "
                f"{str(semantic.get('status', 'unknown')).replace('_', ' ')}; "
                f"protected {int(semantic.get('suppressed_strong_matches', 0)):,} "
                "strong later-pass matches; ranked "
                f"{int(semantic.get('high_priority_findings', 0)):,} high and "
                f"{int(semantic.get('moderate_priority_findings', 0)):,} moderate "
                "possible false positives."
            )
        shape = coverage.get("shape_verification", {}) or {}
        if shape.get("enabled"):
            text += (
                f" SAM3 shape verification: "
                f"{str(shape.get('status', 'unknown')).replace('_', ' ')}; "
                f"processed {int(shape.get('candidates_processed', 0)):,} candidates; "
                f"protected {int(shape.get('suppressed_strong_matches', 0)):,} "
                "matches to confirmed-good shape references."
            )
        if int(coverage.get("suppressed_by_trusted_examples", 0)):
            text += (
                f" Protected {int(coverage.get('suppressed_by_trusted_examples', 0)):,} candidates "
                "using annotations you previously reviewed and kept."
            )
        details = [
            text,
            f"Skipped crops: {int(coverage.get('skipped_unavailable', 0)):,} unavailable; "
            f"{int(coverage.get('skipped_small_class', 0)):,} in classes with fewer than {minimum} usable examples.",
            f"Classes above {coverage.get('exact_class_limit', 2000):,} usable crops compare against up to "
            f"{coverage.get('reference_limit', 256):,} sampled reference crops, not every possible pair.",
            "Each deep pass temporarily excludes the strongest queued candidates from comparison, "
            "then searches the remaining annotations for the next outlier layer. Dataset labels are unchanged.",
            f"Candidates selected before vetting: {int(coverage.get('queued_before_vetting', coverage.get('queued_findings', 0))):,}; "
            f"final review findings: {int(coverage.get('queued_findings', 0)):,}; "
            f"trusted kept examples: {int(coverage.get('trusted_examples', 0)):,}.",
            "No visual findings does not prove that all labels are correct.",
        ]
        if semantic.get("enabled"):
            details.append(
                f"SigLIP 2 class-name verifier ({semantic.get('backend', 'unknown')}): "
                f"{semantic.get('status', 'unknown')}; "
                f"checked {int(semantic.get('candidates_checked', 0)):,}; "
                f"cache hits {int(semantic.get('cached_results', 0)):,}; "
                f"protected {int(semantic.get('suppressed_strong_matches', 0)):,}; "
                f"high-priority findings {int(semantic.get('high_priority_findings', 0)):,}; "
                f"moderate-priority findings {int(semantic.get('moderate_priority_findings', 0)):,}."
            )
            if semantic.get("error"):
                details.append("Class-name verifier detail: " + str(semantic["error"]))
        if shape.get("enabled"):
            details.append(
                f"SAM3 shape verifier ({shape.get('backend', 'sam3')}): "
                f"{shape.get('status', 'unknown')}; "
                f"confirmed-good references {int(shape.get('reference_annotations', 0)):,}; "
                f"reference cache hits {int(shape.get('reference_cache_hits', 0)):,}; "
                f"processed {int(shape.get('candidates_processed', 0)):,}; "
                f"profiles built {int(shape.get('class_profiles_built', 0)):,}; "
                f"mismatches detected {int(shape.get('shape_mismatches_detected', 0)):,}; "
                f"protected {int(shape.get('suppressed_strong_matches', 0)):,}; "
                f"semantic disagreements retained {int(shape.get('semantic_conflicts', 0)):,}."
            )
            if shape.get("reference_source") == "confirmed_review_examples":
                details.append(
                    "Shape references use only annotations you explicitly reviewed and kept, "
                    "never the suspicious objects being checked. Without enough confirmed-good "
                    "masks, shape evidence is unknown and the other checks continue."
                )
            if shape.get("error"):
                details.append("Shape verifier detail: " + str(shape["error"]))
        
        llm = coverage.get("llm_borderline_verification", {}) or {}
        if llm.get("enabled"):
            details.append(
                f"LLM borderline verifier ({llm.get('model', 'unknown')}): "
                f"{llm.get('status', 'unknown')}; "
                f"borderline cases checked {int(llm.get('candidates_checked', 0)):,}; "
                f"protected {int(llm.get('high_confidence_protections', llm.get('protected_by_high_confidence', 0))):,}; "
                f"cache hits {int(llm.get('cached_results', 0)):,}; "
                f"conflicting answers retained {int(llm.get('evidence_conflicts', 0)):,}."
            )
            if llm.get("error"):
                details.append("LLM verifier detail: " + str(llm["error"]))
            elif llm.get("unavailable_reason"):
                details.append("LLM verifier detail: " + str(llm["unavailable_reason"]))
        if coverage.get("fallback_reason"):
            details.extend(["", "DINOv3 fallback reason:", str(coverage["fallback_reason"])])
        for row in coverage.get("classes", []):
            cutoff = row.get("cutoff")
            details.extend(["", f"Class {row['class_id']}: {row['class_name']} — {row['status'].replace('_', ' ')}",
                             f"Annotations: {row['annotations']}; usable: {row['encoded']}; compared: {row['compared']}; "
                             f"references: {row['references']}{' (sampled)' if row['sampled'] else ''}; "
                             f"comparison device: {str(row.get('comparison_device', 'cpu')).upper()}.",
                             f"Candidates: {row['candidate_count']}; queued: {row['queued']}; omitted by cap: {row['omitted_by_cap']}; "
                             f"deep passes: {row.get('passes_completed', 1)}; "
                             f"selected before vetting: {row.get('queued_before_vetting', row['queued'])}; "
                             f"protected by kept examples: {row.get('suppressed_by_trusted_examples', 0)}; "
                             f"protected by class meaning: {row.get('suppressed_by_class_semantics', 0)}; "
                             f"high-priority possible false positives: {row.get('high_priority_findings', 0)}; "
                             f"moderate-priority: {row.get('moderate_priority_findings', 0)}."
                             + (f" Similarity cutoff: {cutoff:.3f}." if cutoff is not None else "")])
        return text, "\n".join(details)

    def _parse_label_line(self, line):
        parts = line.split()
        if not parts:
            return None, "blank label line"

        try:
            class_value = float(parts[0])
            if not math.isfinite(class_value) or not class_value.is_integer():
                return None, "class id must be a finite integer"
            class_id = int(class_value)
        except Exception:
            return None, "class id is not a number"

        try:
            values = [float(value) for value in parts[1:]]
        except Exception:
            return None, "annotation values must be numbers"
        if not all(math.isfinite(value) for value in values):
            return None, "annotation values must be finite"

        polygon_preference = getattr(self, "_scan_polygon_preference_override", None)
        if not getattr(self, "_scan_context_active", False):
            try:
                if hasattr(self.parent, "_polygon_label_parse_preference"):
                    polygon_preference = self.parent._polygon_label_parse_preference()
            except Exception:
                polygon_preference = None

        annotation_type = self._annotation_type(values, preferred_polygon_type=polygon_preference)
        if annotation_type == "unknown":
            return None, f"unsupported value count: {len(values)}"

        return {
            "class_id": class_id,
            "values": values,
            "annotation_type": annotation_type,
        }, ""

    def _check_normalized_pair_values(self, values, file_path, line_number, line, issue_type):
        issues = []
        if len(values) % 2 != 0:
            issues.append(self._make_issue(
                issue_type,
                "error",
                file_path,
                "Point annotation has an odd number of coordinates.",
                line_number,
                line,
            ))
            return issues

        for idx in range(0, len(values), 2):
            x = values[idx]
            y = values[idx + 1]
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                issues.append(self._make_issue(
                    issue_type,
                    "warning",
                    file_path,
                    f"Point {idx // 2 + 1} is outside normalized bounds.",
                    line_number,
                    line,
                    suggestion="Review this polygon or OBB before training.",
                ))
                break
        return issues

    def _check_bbox(self, parsed, image_size, file_path, line_number, line):
        issues = []
        values = parsed["values"]
        x_center, y_center, width, height = values[:4]
        img_width, img_height = image_size

        if width <= 0 or height <= 0:
            issues.append(self._make_issue(
                "invalid_box_size",
                "error",
                file_path,
                "Box width and height must be positive.",
                line_number,
                line,
                suggestion="Delete or redraw this label.",
            ))
            return issues

        if width > 1 or height > 1:
            issues.append(self._make_issue(
                "box_size_out_of_range",
                "error",
                file_path,
                "Box width or height is greater than 1.0.",
                line_number,
                line,
            ))

        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            issues.append(self._make_issue(
                "box_center_out_of_range",
                "error",
                file_path,
                "Box center is outside normalized bounds.",
                line_number,
                line,
            ))

        left = x_center - width / 2.0
        top = y_center - height / 2.0
        right = x_center + width / 2.0
        bottom = y_center + height / 2.0

        overflow_px = max(
            0.0,
            -left * img_width,
            -top * img_height,
            (right - 1.0) * img_width,
            (bottom - 1.0) * img_height,
        )

        if overflow_px > self.BBOX_EDGE_TOLERANCE_PX:
            issues.append(self._make_issue(
                "box_extends_outside_image",
                "warning",
                file_path,
                f"Box extends about {overflow_px:.1f}px outside the image boundary.",
                line_number,
                line,
                suggestion="Open this image and resave the label to clip the raw box if needed.",
            ))

        if self._scan_check_enabled("tiny_checks") and (width * img_width <= 1 or height * img_height <= 1):
            issues.append(self._make_issue(
                "tiny_box",
                "warning",
                file_path,
                "Box is 1 pixel or smaller in width or height.",
                line_number,
                line,
                suggestion="Review tiny boxes; they are often accidental labels.",
            ))

        train_width, train_height = self._training_input_size()
        if self._scan_check_enabled("tiny_checks") and train_width > 0 and train_height > 0:
            scaled_width, scaled_height, train_scale = self._letterbox_label_size(
                width,
                height,
                image_size,
                (train_width, train_height),
            )
            if min(scaled_width, scaled_height) < self.TRAINING_TINY_OBJECT_SIDE_PX:
                self.training_tiny_object_count += 1
                issues.append(self._make_issue(
                    "object_too_small_for_training_size",
                    "warning",
                    file_path,
                    (
                        f"Object will be about {scaled_width:.1f}x{scaled_height:.1f}px after "
                        f"letterbox to {train_width}x{train_height} training size."
                    ),
                    line_number,
                    line,
                    suggestion=(
                        "Try a larger training size in Train View, crop/tiling, "
                        "or collect closer examples for this class."
                    ),
                ))

        if parsed["annotation_type"] == "bbox_keypoints":
            keypoint_values = values[4:]
            for idx in range(0, len(keypoint_values), 3):
                x, y, visible = keypoint_values[idx:idx + 3]
                if int(visible) not in (0, 1, 2):
                    issues.append(self._make_issue(
                        "invalid_keypoint_visibility",
                        "error",
                        file_path,
                        "Keypoint visibility must be 0, 1, or 2.",
                        line_number,
                        line,
                    ))
                    break
                if int(visible) in (1, 2) and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    issues.append(self._make_issue(
                        "keypoint_out_of_range",
                        "warning",
                        file_path,
                        "Visible keypoint is outside normalized bounds.",
                        line_number,
                        line,
                    ))
                    break

        return issues

    def _bbox_iou(self, box_a, box_b):
        return self._bbox_overlap_metrics(box_a, box_b).get("iou", 0.0)

    @staticmethod
    def _bbox_overlap_metrics(box_a, box_b):
        """Calculate overlap metrics between two bounding boxes in normalized coordinates.
        Expects box_a and box_b to be sequences with at least 4 elements: [cx, cy, width, height].
        """
        if not isinstance(box_a, (list, tuple)) or len(box_a) < 4:
            raise ValueError(f"box_a must be a sequence with at least 4 elements, got {type(box_a).__name__} with length {len(box_a) if hasattr(box_a, '__len__') else 'unknown'}")
        if not isinstance(box_b, (list, tuple)) or len(box_b) < 4:
            raise ValueError(f"box_b must be a sequence with at least 4 elements, got {type(box_b).__name__} with length {len(box_b) if hasattr(box_b, '__len__') else 'unknown'}")
        try:
            ax, ay, aw, ah = [float(v) for v in box_a[:4]]
            bx, by, bw, bh = [float(v) for v in box_b[:4]]
        except (TypeError, ValueError) as e:
            raise ValueError(f"Could not convert box coordinates to float: {e}")

        a_left = ax - aw / 2.0
        a_top = ay - ah / 2.0
        a_right = ax + aw / 2.0
        a_bottom = ay + ah / 2.0

        b_left = bx - bw / 2.0
        b_top = by - bh / 2.0
        b_right = bx + bw / 2.0
        b_bottom = by + bh / 2.0

        inter_left = max(a_left, b_left)
        inter_top = max(a_top, b_top)
        inter_right = min(a_right, b_right)
        inter_bottom = min(a_bottom, b_bottom)

        inter_width = max(0.0, inter_right - inter_left)
        inter_height = max(0.0, inter_bottom - inter_top)
        intersection = inter_width * inter_height
        area_a = max(0.0, aw * ah)
        area_b = max(0.0, bw * bh)
        union = area_a + area_b - intersection
        smaller_area = max(1e-9, min(area_a, area_b))
        larger_area = max(1e-9, max(area_a, area_b))

        center_distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        avg_w = max(1e-9, (abs(aw) + abs(bw)) / 2.0)
        avg_h = max(1e-9, (abs(ah) + abs(bh)) / 2.0)
        avg_diag = max(1e-9, (avg_w * avg_w + avg_h * avg_h) ** 0.5)

        return {
            "iou": intersection / union if union > 0 else 0.0,
            "containment": intersection / smaller_area,
            "area_ratio": smaller_area / larger_area,
            "center_distance": center_distance,
            "relative_center_distance": center_distance / avg_diag,
        }

    @classmethod
    def _bbox_duplicate_geometry_match(cls, box_a, box_b, iou_threshold):
        metrics = cls._bbox_overlap_metrics(box_a, box_b)
        iou = float(metrics.get("iou", 0.0))
        containment = float(metrics.get("containment", 0.0))
        area_ratio = float(metrics.get("area_ratio", 0.0))
        rel_center = float(metrics.get("relative_center_distance", 999.0))

        if iou >= float(iou_threshold):
            return True, "iou", metrics
        if containment >= 0.92 and area_ratio >= 0.18 and rel_center <= 0.45:
            return True, "containment", metrics
        if containment >= 0.78 and area_ratio >= 0.35 and rel_center <= 0.35:
            return True, "overlap", metrics
        return False, "", metrics

    def _duplicate_iou_threshold(self):
        """Get duplicate IoU threshold with bounds validation [0.50, 0.99]."""
        override = getattr(self, "_scan_duplicate_iou_override", None)
        if override is not None:
            try:
                value = float(override)
                # Clamp override to valid range
                return max(0.50, min(0.99, value))
            except (TypeError, ValueError):
                pass

        threshold_getter = getattr(self.parent, "get_duplicate_iou_threshold", None)
        if callable(threshold_getter):
            try:
                value = float(threshold_getter())
                # Clamp to valid range
                return max(0.50, min(0.99, value))
            except (TypeError, ValueError):
                pass
        try:
            value = float(QSettings("UltraDarkFusion", "HealthCheck").value(
                "duplicate_iou_threshold", self.DUPLICATE_IOU_THRESHOLD
            ))
            return max(0.50, min(0.99, value))
        except (TypeError, ValueError):
            return float(self.DUPLICATE_IOU_THRESHOLD)

    def _scan_label_file(self, label_file, image_map):
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        if base_name.lower() in self._ambiguous_image_stems:
            # One label cannot be assigned safely when, for example, both
            # frame.jpg and frame.png exist.  The image-side findings explain
            # the collision without parsing the label against an arbitrary file.
            return
        image_path = image_map.get(base_name.lower())

        if not image_path:
            self.issues.append(self._make_issue(
                "label_without_image",
                "error",
                label_file,
                "Label file has no matching image file.",
                suggestion="Move this label to review or restore the missing image.",
            ))
            return

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()
        except UnicodeDecodeError:
            with open(label_file, "r", encoding="latin-1", errors="ignore") as f:
                raw_lines = f.readlines()
        except Exception as e:
            self._statistics_label_states[image_path] = {"state": "unreadable", "line_count": 0}
            self.issues.append(self._make_issue(
                "label_read_error",
                "error",
                label_file,
                f"Could not read label file: {e}",
            ))
            return

        stripped_lines = [(idx, raw.strip()) for idx, raw in enumerate(raw_lines, start=1) if raw.strip()]

        # Diagnostic: Log file read details for freshness validation
        try:
            file_stat = os.stat(label_file)
            file_mtime = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            logger.debug(
                f"Scanned label file: {os.path.basename(label_file)} "
                f"({len(stripped_lines)} annotations, modified {file_mtime})"
            )
        except Exception:
            pass  # If stat fails, don't block the scan

        self._statistics_label_states[image_path] = {
            "state": "annotated" if stripped_lines else "empty", "line_count": len(stripped_lines)
        }
        if not stripped_lines:
            self.empty_label_files += 1
            self.issues.append(self._make_issue(
                "empty_label_file",
                "info",
                label_file,
                "This image has an empty label file.",
                suggestion=(
                    "Keep it when this is an intentional negative image; otherwise open the image "
                    "in the Dataset Analysis queue and add the missing objects in Label Maker."
                ),
            ))
            return

        if not self._needs_annotation_records():
            return
        image_size = self._image_size_cache.get(image_path)
        if not image_size:
            return

        self.total_labels += len(stripped_lines)
        file_families = set()
        bbox_records = []
        first_annotation_line = {}

        for line_number, line in stripped_lines:
            parsed, error = self._parse_label_line(line)
            if parsed is None:
                self.issues.append(self._make_issue(
                    "malformed_label",
                    "error",
                    label_file,
                    error,
                    line_number,
                    line,
                    suggestion="Redraw or remove this label line.",
                ))
                continue

            class_id = parsed["class_id"]
            if not self._is_class_index_valid(class_id):
                self.issues.append(self._make_issue(
                    "invalid_class_id",
                    "error",
                    label_file,
                    f"Class id {class_id} is not present in classes.txt.",
                    line_number,
                    line,
                    suggestion="Fix the class id or update classes.txt.",
                ))
                continue

            self.used_class_indices.add(class_id)
            self.class_label_counts[class_id] += 1
            annotation_type = parsed["annotation_type"]
            annotation_key = (
                int(class_id),
                str(annotation_type),
                tuple(float(value) for value in parsed["values"]),
            )
            first_line_number = first_annotation_line.get(annotation_key)
            exact_duplicate = first_line_number is not None
            if not exact_duplicate:
                first_annotation_line[annotation_key] = line_number
            elif self._scan_check_enabled("duplicate_checks"):
                self.duplicate_box_candidates += 1
                self.issues.append(self._make_issue(
                    (
                        "duplicate_box_candidate"
                        if annotation_type in ("bbox", "bbox_keypoints")
                        else "duplicate_annotation_exact"
                    ),
                    "warning",
                    label_file,
                    f"This annotation exactly duplicates line {first_line_number}.",
                    line_number,
                    line,
                    suggestion="Remove one copy of the annotation before training.",
                ))
            family = self._annotation_family(annotation_type)
            file_families.add(family)
            self.annotation_family_counts[family] += 1
            self.annotation_family_records.append({
                "family": family,
                "file_path": label_file,
                "image_path": image_path,
                "line_number": line_number,
                "line": line,
                "class_id": class_id,
                "parsed": parsed,
            })

            if len(file_families) > 1:
                self.issues.append(self._make_issue(
                    "mixed_annotation_types",
                    "error",
                    label_file,
                    "This file contains more than one Ultralytics annotation type.",
                    line_number,
                    line,
                    suggestion="Keep one annotation family across the entire dataset.",
                ))

            if annotation_type in ("bbox", "bbox_keypoints"):
                self.issues.extend(self._check_bbox(parsed, image_size, label_file, line_number, line))
                if not exact_duplicate:
                    bbox_records.append((line_number, class_id, tuple(parsed["values"][:4]), line))
            elif annotation_type == "segmentation":
                if len(parsed["values"]) < 6:
                    self.issues.append(self._make_issue(
                        "short_segmentation",
                        "error",
                        label_file,
                        "Segmentation needs at least 3 points.",
                        line_number,
                        line,
                    ))
                self.issues.extend(self._check_normalized_pair_values(
                    parsed["values"], label_file, line_number, line, "segmentation_out_of_range"
                ))
            elif annotation_type == "obb":
                self.issues.extend(self._check_normalized_pair_values(
                    parsed["values"], label_file, line_number, line, "obb_out_of_range"
                ))

            self.valid_labels += 1

        if not self._scan_check_enabled("duplicate_checks"):
            return
        duplicate_iou_threshold = self._duplicate_iou_threshold()
        for i, first in enumerate(bbox_records):
            line_a, class_a, box_a, text_a = first
            for line_b, class_b, box_b, text_b in bbox_records[i + 1:]:
                if class_a != class_b:
                    continue
                try:
                    is_duplicate, reason, metrics = self._bbox_duplicate_geometry_match(
                        box_a,
                        box_b,
                        duplicate_iou_threshold,
                    )
                except (ValueError, TypeError) as e:
                    # If box format is invalid, skip duplicate check for this pair
                    logger.debug("Could not compare boxes on lines %s and %s: %s", line_a, line_b, e)
                    continue
                if is_duplicate:
                    self.duplicate_box_candidates += 1
                    self.issues.append(self._make_issue(
                        "duplicate_box_candidate",
                        "warning",
                        label_file,
                        (
                            f"Possible duplicate {self._class_name_for_index(class_a)} boxes "
                            f"on lines {line_a} and {line_b} "
                            f"({reason}, IoU {metrics.get('iou', 0.0):.2f}, "
                            f"containment {metrics.get('containment', 0.0):.2f})."
                        ),
                        line_b,
                        text_b,
                        suggestion="Use Remove Duplicate Boxes, or adjust Duplicate IoU if this is too strict or too loose.",
                    ))

    def _scan_orphan_json_files(self, dataset_dir, image_map):
        for entry in os.scandir(dataset_dir):
            if not entry.is_file() or not entry.name.lower().endswith(".json"):
                continue

            name = entry.name.lower()
            if name in self.PROTECTED_JSON_NAMES:
                continue

            base_name = os.path.splitext(entry.name)[0].lower()
            if base_name in image_map:
                continue

            self.issues.append(self._make_issue(
                "orphan_json",
                "info",
                entry.path,
                "JSON file does not match an image and is not a known protected dataset file.",
                suggestion="Review this file before deleting or moving it.",
            ))

    def _class_balance_summary(self):
        nonzero_counts = {
            class_id: count
            for class_id, count in self.class_label_counts.items()
            if count > 0
        }

        if len(nonzero_counts) < 2:
            return {}

        max_class_id, max_count = max(nonzero_counts.items(), key=lambda item: item[1])
        min_class_id, min_count = min(nonzero_counts.items(), key=lambda item: item[1])

        if min_count <= 0:
            return {}

        imbalance_ratio = max_count / min_count
        if imbalance_ratio < self.CLASS_IMBALANCE_RATIO:
            return {}

        return {
            "ratio": round(imbalance_ratio, 2),
            "max_class_id": int(max_class_id),
            "max_class_name": self._class_name_for_index(max_class_id),
            "max_count": int(max_count),
            "min_class_id": int(min_class_id),
            "min_class_name": self._class_name_for_index(min_class_id),
            "min_count": int(min_count),
            "message": (
                f"Class balance looks uneven: '{self._class_name_for_index(max_class_id)}' has {max_count} labels, "
                f"while '{self._class_name_for_index(min_class_id)}' has {min_count} "
                f"({imbalance_ratio:.1f}x difference). Open Statistics for the full class table."
            ),
        }

    def _build_report(self, dataset_dir, classes_file, image_files, label_files):
        class_balance = self._class_balance_summary()

        severity_counts = Counter(issue["severity"] for issue in self.issues)
        issue_type_counts = Counter(issue["issue_type"] for issue in self.issues)
        files_with_issues = {
            issue["file"] for issue in self.issues
            if issue.get("file") and issue.get("severity") in ("error", "warning")
        }

        if isinstance(self.valid_classes, dict):
            all_class_ids = set(int(cid) for cid in self.valid_classes.keys())
        else:
            all_class_ids = set(range(len(self.valid_classes)))

        unused_classes = sorted(all_class_ids - self.used_class_indices)
        training_size = self._training_input_size()
        training_size_text = "{}x{}".format(*training_size) if all(training_size) else ""

        return OrderedDict([
            ("generated_at", datetime.now().isoformat(timespec="seconds")),
            ("source", "dataset_folder"),
            ("dataset_dir", self._normalize_path(dataset_dir)),
            ("classes_file", self._normalize_path(classes_file) if classes_file else ""),
            ("summary", OrderedDict([
                ("total_images", len(image_files)),
                ("total_label_files", len(label_files)),
                ("total_label_lines", self.total_labels),
                ("valid_label_lines", self.valid_labels),
                ("missing_label_files", self.missing_label_files),
                ("empty_label_files", self.empty_label_files),
                ("duplicate_box_candidates", self.duplicate_box_candidates),
                ("duplicate_iou_threshold", self._duplicate_iou_threshold()),
                ("bad_images", self.bad_images),
                ("files_with_errors_or_warnings", len(files_with_issues)),
                ("errors", severity_counts.get("error", 0)),
                ("warnings", severity_counts.get("warning", 0)),
                ("info", severity_counts.get("info", 0)),
                ("issue_types", dict(sorted(issue_type_counts.items()))),
                ("class_label_counts", {
                    self._class_name_for_index(class_id): count
                    for class_id, count in sorted(self.class_label_counts.items())
                }),
                ("annotation_type_counts", dict(sorted(self.annotation_family_counts.items()))),
                ("dominant_annotation_type", self.dominant_annotation_family),
                ("class_balance", class_balance),
                ("training_input_size", training_size_text),
                ("training_input_size_source", self.training_input_size_source),
                ("training_tiny_object_count", self.training_tiny_object_count),
                ("visual_outliers_enabled", bool(self._scan_visual_outliers_override)),
                ("visual_outlier_threshold", float(
                    self._scan_visual_outlier_threshold_override or 0.45
                )),
                ("visual_class_verify_enabled", bool(
                    self._scan_visual_class_verify_override
                )),
                ("visual_shape_verify_enabled", bool(
                    self._scan_shape_verify_override
                )),
                ("visual_outlier_candidates", self.visual_outlier_candidates),
                ("visual_outlier_scan", self.visual_outlier_scan),
                ("scan_checks", {key: self._scan_check_enabled(key) for key, _title in self.SCAN_CHECKS}),
                ("unused_class_count", len(unused_classes)),
                ("unused_classes", unused_classes),
            ])),
            ("statistics", self.statistics_report),
            ("issues", self.issues),
        ])

    def _issue_matches_image(self, issue, image_path, dataset_dir):
        if not image_path:
            return False

        image_path = self._normalize_path(image_path)
        label_path = self._normalize_path(os.path.splitext(image_path)[0] + ".txt")
        source_path = self._normalize_path(self._issue_source_path(issue, dataset_dir))

        if source_path in {image_path, label_path}:
            return True

        matching_image = self._matching_image_for_issue(issue, dataset_dir)
        return bool(matching_image) and self._normalize_path(matching_image) == image_path

    def _issue_source_path(self, issue, dataset_dir):
        absolute_path = issue.get("absolute_path", "")
        if absolute_path:
            return self._normalize_path(absolute_path)

        relative_path = issue.get("file", "")
        if relative_path:
            return self._normalize_path(os.path.join(dataset_dir, relative_path))

        return ""

    def _matching_image_for_issue(self, issue, dataset_dir):
        source_path = self._issue_source_path(issue, dataset_dir)
        if not source_path:
            return ""

        extension = os.path.splitext(source_path)[1].lower()
        if extension in self.valid_image_extensions and os.path.exists(source_path):
            return source_path

        stem = os.path.splitext(os.path.basename(source_path))[0]
        search_dirs = [os.path.dirname(source_path), dataset_dir]
        seen_dirs = set()
        for directory in search_dirs:
            if not directory:
                continue

            normalized_dir = self._normalize_path(directory)
            if normalized_dir in seen_dirs:
                continue
            seen_dirs.add(normalized_dir)

            for image_extension in self.valid_image_extensions:
                candidate = self._normalize_path(os.path.join(normalized_dir, stem + image_extension))
                if os.path.exists(candidate):
                    return candidate

        return ""

    def _health_issue_to_review_issue(self, issue, dataset_dir):
        """Adapt a folder-analysis finding to the shared Label Maker queue schema."""
        if not isinstance(issue, dict):
            return None
        image_path = self._matching_image_for_issue(issue, dataset_dir)
        if not image_path or not os.path.isfile(image_path):
            return None
        image_path = self._normalize_path(image_path)
        source_path = self._issue_source_path(issue, dataset_dir)
        label_path = source_path if str(source_path).lower().endswith(".txt") else os.path.splitext(image_path)[0] + ".txt"
        task = "detect"
        parsed = None
        raw_line = str(issue.get("label_line", "") or "").strip()
        if raw_line:
            try:
                parsed, _error = self._parse_label_line(raw_line)
            except Exception:
                parsed = None
        annotation_type = str((parsed or {}).get("annotation_type", "") or "")
        task = {
            "bbox": "detect",
            "bbox_keypoints": "pose",
            "segmentation": "segment",
            "obb": "obb",
        }.get(annotation_type, "detect")

        ground_truth = None
        line_number = issue.get("line")
        try:
            from darkfusion_validation_review import parse_ground_truth

            objects = parse_ground_truth(image_path, task, list(self.valid_classes or []))
            if isinstance(line_number, int) and line_number > 0:
                ground_truth = next(
                    (item for item in objects if item.get("label_line") == line_number - 1),
                    None,
                )
        except Exception as error:
            logger.debug("Could not adapt health ground truth for %s: %s", image_path, error)

        reference = ground_truth or {}
        severity_text = str(issue.get("severity", "warning") or "warning").lower()
        severity = {"error": 1.0, "warning": 0.7, "info": 0.3}.get(severity_text, 0.5)
        health_key = self._issue_review_key(issue)
        suggestion = str(issue.get("suggestion", "") or "").strip()
        detail = str(issue.get("message", "") or "").strip()
        if suggestion:
            detail = f"{detail}\n\nSuggested action: {suggestion}"
        if (
            str(issue.get("issue_type", "")) == "visual_class_outlier"
            and issue.get("visual_scan_pass") is not None
        ):
            detail += (
                f"\n\nDeep scan pass: {int(issue['visual_scan_pass'])}. "
                "Later passes expose the next outlier layer without changing dataset labels."
            )
            strength = str(issue.get("false_positive_strength", "review") or "review")
            if strength in {"high", "moderate"}:
                probability = issue.get("class_semantic_probability")
                try:
                    probability_text = f"{float(probability) * 100.0:.1f}%"
                except (TypeError, ValueError):
                    probability_text = "unavailable"
                detail += (
                    f"\nFalse-positive priority: {strength.title()}. "
                    f"SigLIP 2 class-match score: {probability_text}. "
                    "This ranking combines DINOv3 visual outlier strength and class-name evidence; "
                    "inspect the highlighted annotation before changing it."
                )
        review_issue = {
            "id": "health_" + hashlib.sha256(health_key.encode("utf-8")).hexdigest()[:24],
            "image_path": image_path,
            "label_path": self._normalize_path(label_path),
            "task": task,
            "type": str(issue.get("issue_type", "dataset_health") or "dataset_health"),
            "severity": severity,
            "class_id": int(reference.get("class_id", -1)),
            "class_name": str(reference.get("class_name", "")),
            "confidence": None,
            "iou": None,
            "ground_truth": ground_truth,
            "prediction": None,
            "detail": detail,
            "review_status": "unreviewed",
            "source": "dataset_health",
            "health_issue_review_key": health_key,
            "health_review_state_path": self._review_state_path(dataset_dir),
            # Preserve the scan-time annotation identity and visual evidence for
            # the Label Maker Preview panel.  The raw label is more stable than
            # a line number after another annotation has been removed.
            "annotation_label_text": raw_line,
            "annotation_line_index": line_number - 1
            if isinstance(line_number, int) and line_number > 0 else None,
            "visual_similarity": issue.get("visual_similarity"),
            "visual_scan_pass": issue.get("visual_scan_pass"),
            "visual_cutoff": issue.get("visual_cutoff"),
            "trusted_similarity": issue.get("trusted_similarity"),
            "class_semantic_probability": issue.get("class_semantic_probability"),
            "class_semantic_margin": issue.get("class_semantic_margin"),
            "class_semantic_similarity": issue.get("class_semantic_similarity"),
            "class_semantic_backend": issue.get("class_semantic_backend"),
            "false_positive_priority": issue.get("false_positive_priority"),
            "false_positive_strength": issue.get("false_positive_strength"),
            "semantic_mismatch": issue.get("semantic_mismatch"),
            "dino_anomaly": issue.get("dino_anomaly"),
        }
        review_issue["issue_key"] = hashlib.sha256(
            ("dataset_health|" + health_key).encode("utf-8")
        ).hexdigest()
        return review_issue

    def _open_health_review_queue(self, selected_health_issue, health_issues, dataset_dir, source_dialog=None):
        selected = self._health_issue_to_review_issue(selected_health_issue, dataset_dir)
        if not selected:
            QMessageBox.information(
                self.parent,
                "Dataset Analysis",
                "This health finding is not attached to a viewable dataset image.",
            )
            return False
        queue = []
        for health_issue in health_issues:
            adapted = self._health_issue_to_review_issue(health_issue, dataset_dir)
            if adapted is not None:
                queue.append(adapted)
        if queue and all(
            str(item.get("type", "")) == "visual_class_outlier"
            for item in queue
        ):
            queue.sort(key=lambda item: (
                -float(item.get("false_positive_priority", 0.0) or 0.0),
                int(item.get("visual_scan_pass", 999999) or 999999),
                float(item.get("visual_similarity", 1.0) or 1.0),
                str(item.get("image_path", "")),
            ))
        selected_id = str(selected.get("id", ""))
        selected = next((item for item in queue if str(item.get("id", "")) == selected_id), selected)
        metadata_dir = os.path.join(dataset_dir, PROJECT_SETTINGS_DIR)
        os.makedirs(metadata_dir, exist_ok=True)
        bridge_path = self._normalize_path(os.path.join(metadata_dir, "health_review_queue.json"))
        history_path = self._normalize_path(os.path.join(metadata_dir, "health_review_history.json"))

        bridge_report = {
            "version": 1,
            "status": "complete",
            "stage": "dataset_health_review",
            "source": "dataset_health",
            "source_title": "Dataset Analysis",
            "model": "Folder-based dataset analysis",
            "data": self._normalize_path(dataset_dir),
            "dataset_dir": self._normalize_path(dataset_dir),
            "task": str(selected.get("task", "detect")),
            "class_names": list(self.valid_classes or []),
            "review_history_path": history_path,
            "issues": queue,
            "summary": dict(Counter(str(item.get("type", "")) for item in queue)),
            "processed_images": len({item.get("image_path") for item in queue}),
            "total_images": len({item.get("image_path") for item in queue}),
        }
        write_json_file_atomic(bridge_path, bridge_report)
        if source_dialog is not None:
            source_dialog.hide()
            self.parent._validation_review_source_dialog = source_dialog
        return self.parent.open_validation_review_issue_in_labeler(
            selected, bridge_path, queue=queue
        )

    def _flash_issue_annotation(self, issue, image_path):
        try:
            line_number = issue.get("line")
            if not isinstance(line_number, int) or line_number <= 0:
                return

            if not hasattr(self.parent, "flash_bounding_box"):
                return

            label_index = line_number - 1
            QTimer.singleShot(
                80,
                lambda idx=label_index, path=image_path: self.parent.flash_bounding_box(idx, path)
            )
        except Exception as e:
            logger.warning(f"Could not flash issue annotation: {e}")

    def show_scan_report_dialog(self, report, report_path):
        """Show a folder-analysis dashboard and launch cleanup in Label Maker."""
        issues = list(report.get("issues", []) or [])
        summary = report.get("summary", {}) or {}
        dataset_dir = self._normalize_path(
            report.get("dataset_dir") or self.base_directory or ""
        )
        if not dataset_dir or not os.path.isdir(dataset_dir):
            QMessageBox.warning(
                self.parent,
                "Dataset Analysis",
                "The dataset folder saved in this report is no longer available.",
            )
            return
        existing = getattr(self, "health_report_dialog", None)
        if existing is not None and not sip.isdeleted(existing):
            if getattr(existing, "_dataset_directory", "") == dataset_dir:
                existing._darkfusion_apply_scan_report(report, report_path)
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            existing.close()
        reviewed_issue_keys = self._load_reviewed_issue_keys(dataset_dir)

        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle("Dataset Analysis")
        dialog.setModal(False)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.setMinimumSize(760, 600)
        dialog.resize(1080, 720)
        self.health_report_dialog = dialog
        dialog._dataset_directory = dataset_dir
        dialog._scan_running = False
        def forget_dialog(_obj=None):
            if getattr(self, "health_report_dialog", None) is dialog:
                self.health_report_dialog = None
        dialog.destroyed.connect(forget_dialog)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title_label = QtWidgets.QLabel("Dataset Analysis")
        title_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        layout.addWidget(title_label)

        source_label = QtWidgets.QLabel(
            f"<b>Dataset folder:</b> {html.escape(dataset_dir)}<br>"
            "This scan reads the images and labels in this folder directly. "
            "It does not use a training YAML or a train/validation split."
        )
        source_label.setWordWrap(True)
        source_label.setMaximumHeight(72)
        source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        source_label.setStyleSheet(
            "QLabel { background: #17212b; border: 1px solid #344454; "
            "border-radius: 6px; padding: 9px; color: #dce6f0; }"
        )
        layout.addWidget(source_label)

        workflow_label = QtWidgets.QLabel(
            "Choose the checks and thresholds in Scan Settings, then press Start Scan. "
            "View counts and charts in Statistics, and review problems in Findings."
        )
        workflow_label.setWordWrap(True)
        workflow_label.setMaximumHeight(44)
        workflow_label.setStyleSheet("color: #9aa4af;")
        layout.addWidget(workflow_label)

        outer_layout = layout
        tabs = QtWidgets.QTabWidget(dialog)
        tabs.setObjectName("datasetAnalysisTabs")
        settings_scroll = QtWidgets.QScrollArea(dialog)
        settings_scroll.setWidgetResizable(True)
        settings_page = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_page)
        settings_scroll.setWidget(settings_page)
        results_page = QtWidgets.QWidget(dialog)
        layout = QtWidgets.QVBoxLayout(results_page)
        tabs.addTab(settings_scroll, "Scan Settings")
        tabs.addTab(results_page, "Findings")
        statistics_page = DatasetStatisticsWidget(dialog)
        tabs.insertTab(1, statistics_page, "Statistics")
        dialog._darkfusion_statistics = statistics_page
        dialog._darkfusion_analysis_tabs = tabs
        outer_layout.addWidget(tabs, 1)

        checks_group = QtWidgets.QGroupBox("Checks to Run", settings_page)
        checks_layout = QtWidgets.QGridLayout(checks_group)
        saved_checks = self._saved_scan_checks()
        scan_checkboxes = {}
        for index, (key, title) in enumerate(self.SCAN_CHECKS):
            checkbox = QtWidgets.QCheckBox(title, checks_group)
            checkbox.setObjectName("datasetScan_" + key)
            checkbox.setChecked(saved_checks[key])
            scan_checkboxes[key] = checkbox
            checks_layout.addWidget(checkbox, index // 2, index % 2)
        settings_layout.addWidget(checks_group)

        active_checks_label = QtWidgets.QLabel(dialog)
        active_checks_label.setObjectName("datasetAnalysisActiveChecks")
        active_checks_label.setWordWrap(True)
        active_checks_label.setTextFormat(Qt.PlainText)
        active_checks_label.setStyleSheet(
            "QLabel { background: #17212b; border: 1px solid #46708f; "
            "border-radius: 6px; padding: 7px 10px; color: #dce6f0; }"
        )
        # Keep the selected checks visible when another tab is open or the
        # settings controls are disabled for an active worker scan.
        outer_layout.insertWidget(3, active_checks_label)

        summary_values = {}
        summary_grid = QtWidgets.QGridLayout()
        summary_grid.setHorizontalSpacing(8)
        summary_grid.setVerticalSpacing(8)

        def add_summary_cell(row, col, title, value, accent="#9aa4af"):
            frame = QtWidgets.QFrame(dialog)
            frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            frame.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Preferred,
            )
            frame.setStyleSheet(
                "QFrame { border: 1px solid #30363d; border-radius: 6px; padding: 6px; }"
                "QLabel { border: none; }"
            )
            cell_layout = QtWidgets.QVBoxLayout(frame)
            cell_layout.setContentsMargins(8, 6, 8, 6)
            value_label = QtWidgets.QLabel(str(value))
            summary_values[title] = value_label
            value_label.setStyleSheet(
                f"font-size: 18px; font-weight: 700; color: {accent};"
            )
            name_label = QtWidgets.QLabel(title)
            name_label.setStyleSheet("color: #9aa4af;")
            cell_layout.addWidget(value_label)
            cell_layout.addWidget(name_label)
            summary_grid.addWidget(frame, row, col)

        add_summary_cell(0, 0, "Images", summary.get("total_images", 0))
        add_summary_cell(0, 1, "Label Files", summary.get("total_label_files", 0))
        add_summary_cell(0, 2, "Errors", summary.get("errors", 0), "#ff6b6b")
        add_summary_cell(0, 3, "Warnings", summary.get("warnings", 0), "#f0b45f")
        add_summary_cell(
            0, 4, "Files to Review",
            summary.get("files_with_errors_or_warnings", 0),
            "#7cc7ff",
        )
        add_summary_cell(0, 5, "Blank Labels", summary.get("empty_label_files", 0))
        add_summary_cell(1, 0, "Missing Labels", summary.get("missing_label_files", 0))
        add_summary_cell(
            1, 2, "Unreadable Images",
            summary.get("bad_images", 0),
            "#ff6b6b",
        )
        add_summary_cell(
            1, 1, "Possible Duplicates",
            summary.get("duplicate_box_candidates", 0),
        )
        add_summary_cell(
            1, 3, "Tiny at Train Size",
            summary.get("training_tiny_object_count", 0),
            "#f0b45f",
        )
        add_summary_cell(
            1, 4, "Possible False Positives",
            summary.get("visual_outlier_candidates", 0),
            "#9ecbff",
        )
        add_summary_cell(
            1, 5, "Unused Classes",
            summary.get("unused_class_count", 0),
        )
        layout.addLayout(summary_grid)

        visual_text, visual_details = self._visual_scan_summary(summary)
        visual_coverage_label = QtWidgets.QLabel(visual_text, dialog)
        visual_coverage_label.setObjectName("datasetVisualScanCoverage")
        visual_coverage_label.setWordWrap(True)
        visual_coverage_label.setTextFormat(Qt.PlainText)
        layout.addWidget(visual_coverage_label)
        visual_details_button = QtWidgets.QPushButton("False-Positive Scan Details", dialog)
        visual_details_button.setObjectName("datasetVisualScanDetails")
        def show_visual_scan_details():
            details_dialog = QtWidgets.QDialog(dialog)
            details_dialog.setWindowTitle("Possible False-Positive Scan Coverage")
            details_dialog.resize(720, 500)
            details_layout = QtWidgets.QVBoxLayout(details_dialog)
            text_view = QtWidgets.QPlainTextEdit(details_dialog)
            text_view.setReadOnly(True)
            text_view.setPlainText(visual_details)
            details_layout.addWidget(text_view)
            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, details_dialog)
            buttons.rejected.connect(details_dialog.reject)
            details_layout.addWidget(buttons)
            details_dialog.exec_()
        visual_details_button.clicked.connect(show_visual_scan_details)
        layout.addWidget(visual_details_button)

        issue_types = summary.get("issue_types", {}) or {}
        issue_label = QtWidgets.QLabel()
        if issue_types:
            top_items = sorted(
                issue_types.items(), key=lambda item: item[1], reverse=True
            )[:6]
            issue_label.setText(
                "Most common findings: "
                + "   ".join(
                    f"{str(issue_type).replace('_', ' ')}: {count}"
                    for issue_type, count in top_items
                )
            )
        else:
            issue_label.setText("No findings were generated.")
        issue_label.setWordWrap(True)
        issue_label.setStyleSheet("color: #c9d1d9; padding: 3px 0;")
        layout.addWidget(issue_label)

        class_balance_label = QtWidgets.QLabel()
        class_balance_label.setWordWrap(True)
        class_balance_label.setStyleSheet(
            "color: #f0b45f; padding: 6px; border: 1px solid #4b3a1f; border-radius: 4px;"
        )
        layout.addWidget(class_balance_label)
        checked_checks_label = QtWidgets.QLabel()
        checked_checks_label.setObjectName("datasetAnalysisChecksRun")
        checked_checks_label.setWordWrap(True)
        layout.addWidget(checked_checks_label)

        health_settings = QSettings("UltraDarkFusion", "HealthCheck")

        def saved_float(setting_key, default, minimum, maximum):
            try:
                value = float(health_settings.value(setting_key, default) or default)
            except (TypeError, ValueError):
                value = float(default)
            if not math.isfinite(value):
                value = float(default)
            return max(float(minimum), min(float(maximum), value))

        tools_group = QtWidgets.QGroupBox("Thresholds and Dataset Tools", dialog)
        tools_layout = QtWidgets.QGridLayout(tools_group)
        tools_layout.setContentsMargins(10, 8, 10, 8)
        tools_layout.setHorizontalSpacing(8)
        tools_layout.setVerticalSpacing(6)

        duplicate_iou_spin = QDoubleSpinBox(tools_group)
        duplicate_iou_spin.setObjectName("datasetAnalysisDuplicateIoU")
        duplicate_iou_spin.setRange(0.50, 0.99)
        duplicate_iou_spin.setDecimals(2)
        duplicate_iou_spin.setSingleStep(0.01)
        duplicate_iou_spin.setValue(
            self._duplicate_iou_threshold()
        )
        duplicate_iou_spin.setToolTip(
            "Lower catches looser duplicate labels; higher catches only near-identical labels."
        )

        sam_verify_check = QtWidgets.QCheckBox(
            "SAM verify overlaps", tools_group
        )
        sam_verify_check.setObjectName("datasetAnalysisSamVerify")
        sam_verify_check.setChecked(
            str(
                QSettings("UltraDarkFusion", "HealthCheck").value(
                    "sam_duplicate_verify", "false"
                )
            ).lower()
            in ("1", "true", "yes")
        )
        sam_verify_check.setToolTip(
            "Used by Remove Duplicates only. Slower, but safer when separate touching "
            "objects have heavily overlapping boxes."
        )
        visual_outliers_check = QtWidgets.QCheckBox(
            "Possible false positives — DINOv3 similarity (slower)", tools_group
        )
        visual_outliers_check.setChecked(
            str(health_settings.value("visual_outliers_enabled", "false")).lower()
            in ("1", "true", "yes")
        )
        visual_outliers_check.setObjectName("datasetScan_visual_outliers")
        visual_outliers_check.setToolTip(
            "Uses DINOv3 lookalike descriptors to compare each annotation crop with "
            "others in the same class and queue conservative outliers. Falls back to a "
            "CPU color/shape histogram if the model is unavailable. Review hint only."
        )

        def selected_check_titles():
            titles = [
                title for key, title in self.SCAN_CHECKS
                if scan_checkboxes[key].isChecked()
            ]
            if visual_outliers_check.isChecked():
                titles.append("Possible false positives (DINOv3)")
                if visual_class_verify_check.isChecked():
                    titles.append("Class verification (SigLIP 2)")
                if visual_shape_verify_check.isChecked():
                    titles.append("Shape verification (SAM3 + LLM)")
            return titles

        def show_selected_checks(prefix="Selected for next scan"):
            titles = selected_check_titles()
            active_checks_label.setText(
                prefix + ": " + ("  •  ".join(titles) if titles else "None")
            )

        for key, checkbox in scan_checkboxes.items():
            checkbox.setProperty("darkfusionReadyFocusPolicy", int(checkbox.focusPolicy()))
            checkbox.toggled.connect(lambda _checked: show_selected_checks())
            checkbox.toggled.connect(
                lambda checked, setting_key=key: health_settings.setValue(
                    "scan_" + setting_key, bool(checked)
                )
            )
        visual_outliers_check.setProperty(
            "darkfusionReadyFocusPolicy", int(visual_outliers_check.focusPolicy())
        )
        visual_outliers_check.toggled.connect(lambda _checked: show_selected_checks())
        visual_outlier_threshold = QDoubleSpinBox(tools_group)
        visual_outlier_threshold.setObjectName("datasetAnalysisOutlierThreshold")
        visual_outlier_threshold.setRange(0.20, 0.90)
        visual_outlier_threshold.setDecimals(2)
        visual_outlier_threshold.setSingleStep(0.05)
        try:
            saved_visual_threshold = float(health_settings.value("visual_outlier_threshold", 0.45))
        except (TypeError, ValueError):
            saved_visual_threshold = 0.45
        visual_outlier_threshold.setValue(saved_visual_threshold)
        visual_outlier_threshold.setToolTip(
            "Flag an annotation when its closest same-class match scores below this value. "
            "Lower is stricter and returns fewer candidates; higher returns more. "
            "A per-class statistical cutoff can make the final threshold stricter."
        )
        visual_outlier_passes = QSpinBox(tools_group)
        visual_outlier_passes.setObjectName("datasetAnalysisOutlierPasses")
        visual_outlier_passes.setRange(1, 100)
        try:
            saved_visual_passes = int(
                health_settings.value("visual_outlier_passes", 8) or 8
            )
        except (TypeError, ValueError):
            saved_visual_passes = 8
        visual_outlier_passes.setValue(max(1, min(100, saved_visual_passes)))
        visual_outlier_passes.setSuffix(" passes")
        visual_outlier_passes.setToolTip(
            "Each deep pass temporarily removes the strongest review candidates from "
            "the comparison pool and exposes the next outlier layer. This never deletes labels."
        )
        visual_class_verify_check = QtWidgets.QCheckBox(
            "Rank possible false positives with SigLIP 2", tools_group
        )
        visual_class_verify_check.setObjectName(
            "datasetAnalysisClassSemanticVerify"
        )
        visual_class_verify_check.setChecked(
            str(
                health_settings.value("visual_class_verify_enabled", "true")
            ).lower() in ("1", "true", "yes")
        )
        visual_class_verify_check.setToolTip(
            "After DINOv3 finds candidates, SigLIP 2 compares each saved crop with its "
            "classes.txt name and gameplay distractors. Candidates where both checks disagree "
            "with the saved label are ranked first for review. Decisive later-pass class matches "
            "are removed from the queue. HUD, icons, nameplates, shadows, effects, and teammate "
            "markers remain reviewable. The model downloads once; unchanged results are cached."
        )
        visual_shape_verify_check = QtWidgets.QCheckBox(
            "Verify with SAM3 shape profile (4-layer: DINOv3 → SigLIP2 → SAM3 → LLM)", tools_group
        )
        visual_shape_verify_check.setObjectName(
            "datasetAnalysisShapeVerify"
        )
        visual_shape_verify_check.setChecked(
            str(
                health_settings.value(
                    "visual_shape_verify_enabled",
                    health_settings.value("visual_foreground_verify_enabled", "true"),
                )
            ).lower() in ("1", "true", "yes")
        )
        visual_shape_verify_check.setToolTip(
            "Layer 1: DINOv3 finds visual outliers. "
            "Layer 2: SigLIP 2 checks if labeled class matches the object. "
            "Decisive later-pass matches skip later checks. "
            "Layer 3: SAM3 checks unresolved shapes against annotations you reviewed and kept, "
            "not against the suspicious candidates themselves. Without enough confirmed-good "
            "examples, shape evidence stays unknown. "
            "Layer 4: LLM (Ollama) runs only for unresolved borderline cases (10-70% similarity). "
            "Conflicting evidence stays in review. This scan never changes labels."
        )
        for checkbox in (
            sam_verify_check,
            visual_class_verify_check,
            visual_shape_verify_check,
        ):
            checkbox.setProperty(
                "darkfusionReadyFocusPolicy", int(checkbox.focusPolicy())
            )
        show_selected_checks()

        refresh_scan_btn = QtWidgets.QPushButton("Start Scan", dialog)
        refresh_scan_btn.setObjectName("datasetAnalysisStartScan")
        dedupe_btn = QtWidgets.QPushButton("Remove Duplicates", tools_group)

        tools_layout.addWidget(QtWidgets.QLabel("Duplicate IoU"), 0, 0)
        tools_layout.addWidget(duplicate_iou_spin, 0, 1)
        tools_layout.addWidget(sam_verify_check, 0, 2)
        tools_layout.addWidget(dedupe_btn, 0, 4)
        checks_layout.addWidget(visual_outliers_check, len(self.SCAN_CHECKS) // 2, len(self.SCAN_CHECKS) % 2)
        false_positive_threshold_label = QtWidgets.QLabel(
            "Flag if DINOv3 similarity is below"
        )
        false_positive_threshold_label.setObjectName(
            "datasetAnalysisFalsePositiveThresholdLabel"
        )
        tools_layout.addWidget(false_positive_threshold_label, 2, 0)
        tools_layout.addWidget(visual_outlier_threshold, 2, 1)
        visual_outlier_note = QtWidgets.QLabel()
        visual_outlier_note.setObjectName("datasetAnalysisFalsePositiveThresholdHelp")
        visual_outlier_note.setWordWrap(True)
        visual_outlier_note.setStyleSheet("color: #9aa4af;")
        tools_layout.addWidget(visual_outlier_note, 2, 2, 1, 3)
        tools_layout.addWidget(QtWidgets.QLabel("Deep false-positive passes"), 3, 0)
        tools_layout.addWidget(visual_outlier_passes, 3, 1)
        deep_pass_note = QtWidgets.QLabel(
            "More passes replace repeated rescans by adding deeper review candidates from the same cached embeddings."
        )
        deep_pass_note.setWordWrap(True)
        deep_pass_note.setStyleSheet("color: #9aa4af;")
        tools_layout.addWidget(deep_pass_note, 3, 2, 1, 3)
        tools_layout.addWidget(visual_class_verify_check, 4, 0, 1, 5)
        tools_layout.addWidget(visual_shape_verify_check, 5, 0, 1, 5)
        
        ollama_install_check = QtWidgets.QCheckBox("Auto-install Ollama for LLM verification (4th layer)", tools_group)
        ollama_install_check.setObjectName("datasetAnalysisOllamaInstall")
        saved_ollama = health_settings.value("ollama_auto_install", "false").lower() in ("1", "true", "yes")
        ollama_install_check.setChecked(saved_ollama)
        ollama_install_check.toggled.connect(
            lambda checked: health_settings.setValue("ollama_auto_install", "true" if checked else "false")
        )
        ollama_install_note = QtWidgets.QLabel(
            "First scan: 15-20 minutes (one-time download). Subsequent scans: instant."
        )
        ollama_install_note.setWordWrap(True)
        ollama_install_note.setStyleSheet("color: #9aa4af;")
        
        # LLM Model Selection (NEW)
        llm_model_label = QtWidgets.QLabel("LLM Model")
        llm_model_combo = QtWidgets.QComboBox(tools_group)
        llm_model_combo.setObjectName("datasetAnalysisLLMModel")
        llm_models = [
            "Auto (detect available)",
            "Qwen2-VL 4B (recommended - fast)",
            "LLaVA 7B (accurate but slower)",
            "Hermes 7B (reasoning-focused)",
        ]
        llm_model_combo.addItems(llm_models)
        saved_model = health_settings.value("llm_model_choice", "Auto (detect available)")
        llm_model_combo.setCurrentText(saved_model if saved_model in llm_models else "Auto (detect available)")
        llm_model_combo.currentTextChanged.connect(
            lambda text: health_settings.setValue("llm_model_choice", text)
        )
        llm_model_combo.setToolTip(
            "Choose which LLM model to use for borderline verification. "
            "Qwen2-VL is fastest (recommended). LLaVA is more accurate. "
            "Hermes is good for reasoning-heavy tasks."
        )
        
        # Presets (NEW)
        presets_label = QtWidgets.QLabel("Scan Presets")
        presets_layout = QtWidgets.QHBoxLayout()
        presets_layout.setSpacing(6)
        presets_layout.setContentsMargins(0, 0, 0, 0)
        
        def set_preset_fast():
            visual_outlier_passes.setValue(1)
            health_settings.setValue("visual_outlier_passes", "1")
        def set_preset_balanced():
            visual_outlier_passes.setValue(2)
            health_settings.setValue("visual_outlier_passes", "2")
        def set_preset_thorough():
            visual_outlier_passes.setValue(3)
            health_settings.setValue("visual_outlier_passes", "3")
        
        preset_fast = QtWidgets.QPushButton("⚡ Fast (3-4m)", tools_group)
        preset_fast.setMaximumWidth(120)
        preset_fast.clicked.connect(set_preset_fast)
        preset_fast.setToolTip("1 pass - finds 90% of false positives in 3-4 minutes")
        
        preset_balanced = QtWidgets.QPushButton("⚙️ Balanced (6-8m)", tools_group)
        preset_balanced.setMaximumWidth(120)
        preset_balanced.clicked.connect(set_preset_balanced)
        preset_balanced.setToolTip("2 passes - balances speed and accuracy")
        
        preset_thorough = QtWidgets.QPushButton("🔬 Thorough (12m+)", tools_group)
        preset_thorough.setMaximumWidth(120)
        preset_thorough.clicked.connect(set_preset_thorough)
        preset_thorough.setToolTip("3 passes - comprehensive but slow")
        
        presets_layout.addWidget(preset_fast)
        presets_layout.addWidget(preset_balanced)
        presets_layout.addWidget(preset_thorough)
        presets_layout.addStretch()
        
        tools_layout.addWidget(ollama_install_check, 6, 0, 1, 5)
        tools_layout.addWidget(ollama_install_note, 7, 0, 1, 5)
        tools_layout.addWidget(llm_model_label, 8, 0)
        tools_layout.addWidget(llm_model_combo, 8, 1, 1, 2)
        tools_layout.addWidget(presets_label, 9, 0)
        presets_container = QtWidgets.QWidget()
        presets_container.setLayout(presets_layout)
        tools_layout.addWidget(presets_container, 9, 1, 1, 4)
        settings_layout.addWidget(tools_group)

        quality_group = QtWidgets.QGroupBox("Image Quality Thresholds")
        quality_layout = QtWidgets.QGridLayout(quality_group)
        quality_spins = {}
        quality_values = self._saved_quality_thresholds()
        for index, (key, title) in enumerate((
            ("blur", "Low detail / blur below"), ("contrast", "Low contrast below"),
            ("dark", "Low brightness below"), ("bright", "High brightness above"),
        )):
            spin = QDoubleSpinBox(quality_group)
            spin.setObjectName("datasetQuality_" + key)
            spin.setRange(0, 1000000 if key == "blur" else 255)
            spin.setDecimals(1)
            spin.setValue(quality_values[key])
            spin.valueChanged.connect(
                lambda value, setting_key=key: health_settings.setValue(
                    "quality_" + setting_key, float(value)
                )
            )
            quality_spins[key] = spin
            quality_layout.addWidget(QtWidgets.QLabel(title), index // 2, (index % 2) * 2)
            quality_layout.addWidget(spin, index // 2, (index % 2) * 2 + 1)
        quality_group.setEnabled(scan_checkboxes["quality_checks"].isChecked())
        scan_checkboxes["quality_checks"].toggled.connect(
            lambda enabled: quality_group.setEnabled(enabled and not dialog._scan_running)
        )
        settings_layout.addWidget(quality_group)
        statistics_help = QtWidgets.QLabel(
            "Statistics share the same background scan. Opening tabs or changing charts reuses the last completed scan. "
            "Image-quality checks decode every image and run only when selected; their thresholds are review hints."
        )
        statistics_help.setWordWrap(True)
        settings_layout.addWidget(statistics_help)
        visual_help = QtWidgets.QLabel(
            "Possible false positives uses DINOv3 to find labels whose crop looks unlike other examples "
            "of the same class. These are candidates for review, because an unusual valid object can also be flagged. "
            "SigLIP 2 ranks candidates where appearance and class meaning both suggest a possible false positive. "
            "Reviewed-good examples and teammate/HUD checks protect valid later-pass labels. A lower similarity threshold is stricter "
            "and returns fewer candidates. "
            "Use validation with your trained weights to check model predictions."
        )
        visual_help.setWordWrap(True)
        settings_layout.addWidget(visual_help)
        settings_layout.addStretch(1)

        filter_row = QtWidgets.QHBoxLayout()
        severity_filter = QtWidgets.QComboBox(dialog)
        severity_filter.addItem("All Severities", "")
        severity_filter.addItem("Errors", "error")
        severity_filter.addItem("Warnings", "warning")
        severity_filter.addItem("Information", "info")

        issue_filter = QtWidgets.QComboBox(dialog)
        issue_filter.addItem("All Finding Types", "")
        for issue_type, _count in sorted(
            issue_types.items(), key=lambda item: item[1], reverse=True
        ):
            issue_filter.addItem(
                str(issue_type).replace("_", " ").title(), issue_type
            )

        search_filter = QtWidgets.QLineEdit(dialog)
        search_filter.setPlaceholderText("Filter by file, finding, or message")
        hide_reviewed_checkbox = QtWidgets.QCheckBox("Hide Reviewed", dialog)
        hide_reviewed_checkbox.setChecked(True)
        current_image_checkbox = QtWidgets.QCheckBox("Current Image Only", dialog)
        clear_filters_btn = QtWidgets.QPushButton("Clear", dialog)

        filter_row.addWidget(QtWidgets.QLabel("Severity"))
        filter_row.addWidget(severity_filter)
        filter_row.addWidget(QtWidgets.QLabel("Finding"))
        filter_row.addWidget(issue_filter)
        filter_row.addWidget(search_filter, 1)
        filter_row.addWidget(hide_reviewed_checkbox)
        filter_row.addWidget(current_image_checkbox)
        filter_row.addWidget(clear_filters_btn)
        layout.addLayout(filter_row)

        table = QtWidgets.QTableWidget(dialog)
        table.setObjectName("datasetAnalysisFindings")
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(
            ["Severity", "Finding", "File", "Line", "What was found"]
        )
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        layout.addWidget(table, 1)

        status_label = QtWidgets.QLabel(dialog)
        status_label.setStyleSheet("color: #9aa4af;")
        layout.addWidget(status_label)

        buttons = QtWidgets.QDialogButtonBox()
        review_in_labeler_btn = buttons.addButton(
            "Review in Label Maker",
            QtWidgets.QDialogButtonBox.AcceptRole,
        )
        review_in_labeler_btn.setToolTip(
            "Open all currently filtered, viewable findings as a cleanup queue."
        )
        review_in_labeler_btn.setObjectName("datasetAnalysisReview")
        delete_flagged_btn = buttons.addButton(
            "Delete Flagged",
            QtWidgets.QDialogButtonBox.DestructiveRole,
        )
        delete_flagged_btn.setToolTip(
            "Permanently delete all currently filtered findings from the report."
        )
        delete_flagged_btn.setObjectName("datasetAnalysisDeleteFlagged")
        open_folder_btn = buttons.addButton(
            "Report Folder", QtWidgets.QDialogButtonBox.ActionRole
        )
        layout.addWidget(buttons)

        scan_status = QtWidgets.QLabel("Ready. Choose checks and press Start Scan.", dialog)
        scan_status.setObjectName("datasetAnalysisStatus")
        scan_status.setWordWrap(True)
        outer_layout.addWidget(scan_status)
        
        ollama_status = QtWidgets.QLabel("")
        ollama_status.setObjectName("datasetAnalysisOllamaStatus")
        ollama_status.setWordWrap(True)
        ollama_status.setStyleSheet("color: #9ecbff;")
        outer_layout.addWidget(ollama_status)
        scan_progress = QtWidgets.QProgressBar(dialog)
        scan_progress.setObjectName("datasetAnalysisProgress")
        scan_progress.setRange(0, 100)
        scan_progress.setValue(0)
        outer_layout.addWidget(scan_progress)
        run_buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, dialog)
        run_buttons.addButton(refresh_scan_btn, QtWidgets.QDialogButtonBox.ActionRole)
        cancel_scan_btn = run_buttons.addButton("Cancel Scan", QtWidgets.QDialogButtonBox.ActionRole)
        cancel_scan_btn.setObjectName("datasetAnalysisCancelScan")
        outer_layout.addWidget(run_buttons)
        run_buttons.rejected.connect(dialog.reject)
        cancel_scan_btn.clicked.connect(lambda: self.cancel_active_scan())
        dialog.finished.connect(lambda _result: self.cancel_active_scan())
        dialog._darkfusion_scan_progress = scan_progress
        dialog._darkfusion_scan_status = scan_status

        visible_issues_for_navigation = []
        # QTableWidget creates five Python/Qt objects per row. Keeping the
        # initial view compact prevents a large findings set from freezing the
        # UI as soon as a background scan finishes; filters and review
        # navigation still operate on the full issue list.
        maximum_rows = 500

        def issue_key(issue):
            return self._issue_review_key(issue)

        def issue_display_name(issue_type):
            if str(issue_type) == "visual_class_outlier":
                return "Possible False Positive (DINOv3)"
            return str(issue_type).replace("_", " ").title()

        def current_image_path():
            return self._normalize_path(
                getattr(self.parent, "current_file", "") or ""
            )

        def selected_issue():
            row = table.currentRow()
            item = table.item(row, 0) if row >= 0 else None
            return item.data(Qt.UserRole) if item is not None else None

        def update_review_button():
            review_in_labeler_btn.setEnabled(
                not dialog._scan_running
                and isinstance(selected_issue(), dict)
                and bool(visible_issues_for_navigation)
            )
            delete_flagged_btn.setEnabled(
                not dialog._scan_running
                and bool(visible_issues_for_navigation)
            )

        def populate_table(select_key=""):
            severity_value = str(severity_filter.currentData() or "")
            issue_value = str(issue_filter.currentData() or "")
            search_value = search_filter.text().strip().lower()
            active_image = current_image_path()
            hide_reviewed = hide_reviewed_checkbox.isChecked()
            only_current = current_image_checkbox.isChecked()

            matching = []
            for issue in issues:
                if hide_reviewed and issue_key(issue) in reviewed_issue_keys:
                    continue
                if (
                    only_current
                    and active_image
                    and not self._issue_matches_image(
                        issue, active_image, dataset_dir
                    )
                ):
                    continue
                if (
                    severity_value
                    and str(issue.get("severity", "")).lower() != severity_value
                ):
                    continue
                if issue_value and str(issue.get("issue_type", "")) != issue_value:
                    continue
                haystack = " ".join(
                    [
                        str(issue.get("issue_type", "")),
                        str(issue.get("file", "")),
                        str(issue.get("message", "")),
                        "" if issue.get("line") is None else str(issue.get("line")),
                    ]
                ).lower()
                if search_value and search_value not in haystack:
                    continue
                matching.append(issue)

            if matching and all(
                str(issue.get("issue_type", "")) == "visual_class_outlier"
                for issue in matching
            ):
                matching.sort(key=lambda issue: (
                    -float(issue.get("false_positive_priority", 0.0) or 0.0),
                    int(issue.get("visual_scan_pass", 999999) or 999999),
                    float(issue.get("visual_similarity", 1.0) or 1.0),
                    str(issue.get("file", "")),
                    int(issue.get("line", 0) or 0),
                ))

            visible_issues_for_navigation[:] = matching
            displayed = matching[:maximum_rows]
            table.setUpdatesEnabled(False)
            table.setRowCount(len(displayed))
            selected_row = 0 if displayed else -1
            try:
                for row, issue in enumerate(displayed):
                    reviewed = issue_key(issue) in reviewed_issue_keys
                    values = [
                        str(issue.get("severity", "")).title(),
                        issue_display_name(issue.get("issue_type", "")),
                        str(issue.get("file", "")),
                        "" if issue.get("line") is None else str(issue.get("line")),
                        str(issue.get("message", "")),
                    ]
                    for column, value in enumerate(values):
                        item = QTableWidgetItem(value)
                        if column == 0:
                            item.setData(Qt.UserRole, issue)
                        if reviewed:
                            item.setForeground(QBrush(QColor("#7f8a98")))
                        elif column == 0 and str(issue.get("severity", "")).lower() == "error":
                            item.setForeground(QBrush(QColor("#ff6b6b")))
                        elif column == 0 and str(issue.get("severity", "")).lower() == "warning":
                            item.setForeground(QBrush(QColor("#f0b45f")))
                        table.setItem(row, column, item)
                    if select_key and issue_key(issue) == select_key:
                        selected_row = row
            finally:
                table.setUpdatesEnabled(True)

            if selected_row >= 0:
                table.selectRow(selected_row)
            shown_text = (
                f"Showing first {maximum_rows:,} of {len(matching):,} filtered findings"
                if len(matching) > maximum_rows
                else f"Showing {len(matching):,} of {len(issues):,} findings"
            )
            reviewed_count = sum(
                1 for issue in issues if issue_key(issue) in reviewed_issue_keys
            )
            status_label.setText(f"{shown_text} · {reviewed_count:,} reviewed")
            status_label.setToolTip(report_path)
            update_review_button()

        def clear_filters():
            severity_filter.setCurrentIndex(0)
            issue_filter.setCurrentIndex(0)
            search_filter.clear()
            current_image_checkbox.setChecked(False)

        def review_selected_issue_in_labeler():
            if dialog._scan_running:
                return
            issue = selected_issue()
            if not isinstance(issue, dict):
                return
            self._open_health_review_queue(
                issue,
                list(visible_issues_for_navigation),
                dataset_dir,
                source_dialog=dialog,
            )

        def reload_external_review_state():
            selected = selected_issue()
            selected_key = issue_key(selected) if isinstance(selected, dict) else ""
            reviewed_issue_keys.clear()
            reviewed_issue_keys.update(
                self._load_reviewed_issue_keys(dataset_dir)
            )
            populate_table(selected_key)

        def sync_duplicate_iou(value):
            value = float(value)
            QSettings("UltraDarkFusion", "HealthCheck").setValue(
                "duplicate_iou_threshold", value
            )
            parent_spin = getattr(self.parent, "dedupe_iou_spinbox", None)
            if parent_spin is not None:
                blocked = parent_spin.blockSignals(True)
                parent_spin.setValue(value)
                parent_spin.blockSignals(blocked)

        def set_visual_outlier_scan(enabled):
            QSettings("UltraDarkFusion", "HealthCheck").setValue(
                "visual_outliers_enabled", bool(enabled)
            )
            visual_outlier_threshold.setEnabled(bool(enabled) and not dialog._scan_running)
            visual_outlier_passes.setEnabled(bool(enabled) and not dialog._scan_running)
            visual_class_verify_check.setEnabled(bool(enabled) and not dialog._scan_running)
            visual_shape_verify_check.setEnabled(
                bool(enabled) and not dialog._scan_running
            )
            visual_outlier_note.setText(
                f"Candidate ceiling {visual_outlier_threshold.value() * 100:.0f}%; "
                "the class cutoff may be lower. Lower is stricter"
                if enabled else "Enable Possible false positives to use this threshold"
            )

        def set_visual_outlier_threshold(value):
            QSettings("UltraDarkFusion", "HealthCheck").setValue(
                "visual_outlier_threshold", float(value)
            )
            if visual_outliers_check.isChecked():
                visual_outlier_note.setText(
                    f"Candidate ceiling {float(value) * 100:.0f}%; "
                    "the class cutoff may be lower. Lower is stricter"
                )

        def refresh_full_scan():
            if dialog._scan_running:
                return
            checks = {key: box.isChecked() for key, box in scan_checkboxes.items()}
            if not any(checks.values()) and not visual_outliers_check.isChecked():
                scan_status.setText("Select at least one check before starting a scan.")
                tabs.setCurrentIndex(0)
                return
            if checks["quality_checks"] and quality_spins["dark"].value() >= quality_spins["bright"].value():
                scan_status.setText("Low brightness must be less than high brightness.")
                tabs.setCurrentIndex(0)
                return
            for key, enabled in checks.items():
                health_settings.setValue("scan_" + key, enabled)
            for key, spin in quality_spins.items():
                health_settings.setValue("quality_" + key, spin.value())
            sync_duplicate_iou(duplicate_iou_spin.value())
            set_visual_outlier_scan(visual_outliers_check.isChecked())
            set_visual_outlier_threshold(visual_outlier_threshold.value())
            health_settings.setValue(
                "visual_class_verify_enabled",
                bool(visual_class_verify_check.isChecked()),
            )
            health_settings.setValue(
                "visual_shape_verify_enabled",
                bool(visual_shape_verify_check.isChecked()),
            )
            
            # Auto-install Ollama if checkbox enabled
            if ollama_install_check.isChecked():
                from darkfusion_ollama_auto_setup import ensure_ollama_ready
                ollama_status.setText("Installing Ollama and pulling model (this may take 15-20 minutes on first run)...")
                ollama_status.setStyleSheet("color: #f0b45f;")
                dialog.update()
                QtWidgets.QApplication.processEvents()
                
                success, model = ensure_ollama_ready(silent=True)
                if success:
                    ollama_status.setText(f"✓ Ollama ready with {model} model. Scan will use full 4-layer verification.")
                    ollama_status.setStyleSheet("color: #a8e6a1;")
                else:
                    ollama_status.setText("⚠ Ollama installation failed or not available. Scan will use 3-layer verification instead.")
                    ollama_status.setStyleSheet("color: #ff9999;")
            else:
                ollama_status.setText("")
            
            self.base_directory = dataset_dir
            try:
                self._start_analysis_scan(dataset_dir)
            except Exception as exc:
                self._set_scan_controls_running(False)
                self._on_scan_failed(str(exc))

        def remove_duplicates_from_report():
            if not hasattr(self.parent, "deduplicate_dataset"):
                QMessageBox.warning(
                    dialog,
                    "Remove Duplicate Labels",
                    "Duplicate cleanup is not available.",
                )
                return
            iou_threshold = float(duplicate_iou_spin.value())
            reply = QMessageBox.question(
                dialog,
                "Remove Duplicate Labels",
                (
                    f"Remove duplicate boxes, segmentations, and OBBs using "
                    f"IoU {iou_threshold:.2f}?\n\n"
                    "The selected dataset folder will be rescanned afterward."
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            sync_duplicate_iou(iou_threshold)
            sam_verify = bool(sam_verify_check.isChecked())
            QSettings("UltraDarkFusion", "HealthCheck").setValue(
                "sam_duplicate_verify", sam_verify
            )
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                result = self.parent.deduplicate_dataset(
                    dataset_dir,
                    iou_threshold=iou_threshold,
                    sam_verify=sam_verify,
                )
            finally:
                QApplication.restoreOverrideCursor()
            QMessageBox.information(
                dialog,
                "Duplicate Cleanup Complete",
                (
                    f"Files checked: {int(result.get('files_scanned', 0))}\n"
                    f"Files changed: {int(result.get('files_changed', 0))}\n"
                    f"Duplicate labels removed: {int(result.get('labels_removed', 0))}\n\n"
                    "Refreshing Dataset Analysis now."
                ),
            )
            refresh_full_scan()

        def delete_flagged_findings():
            """Delete all currently filtered findings using the label maker system."""
            if dialog._scan_running:
                return
            if not visible_issues_for_navigation:
                QMessageBox.information(dialog, "No Findings", "No findings to delete with current filters.")
                return

            # Get filter state for user confirmation
            issue_value = str(issue_filter.currentData() or "")
            issue_type_display = issue_display_name(issue_value) if issue_value else "all findings"
            filter_desc = f"Delete {len(visible_issues_for_navigation):,} {issue_type_display}?"

            reply = QMessageBox.question(
                dialog,
                "Delete Flagged Findings",
                f"{filter_desc}\n\nThis will permanently remove these annotations from label files and the report.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

            # Use label maker's delete method
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                result = self.parent.delete_health_issues_from_dataset(
                    list(visible_issues_for_navigation),
                    dataset_dir,
                )
            finally:
                QApplication.restoreOverrideCursor()

            if not result:
                QMessageBox.warning(dialog, "Delete Failed", "Could not delete findings.")
                return

            # Failed writes keep their findings. Successful deletions shift
            # subsequent persisted line numbers, so reindex those findings too.
            reviewed_before = {id(issue) for issue in issues if issue_key(issue) in reviewed_issue_keys}
            deleted_count = update_health_report_after_deletion(issues, result)
            reviewed_issue_keys.clear()
            reviewed_issue_keys.update(issue_key(issue) for issue in issues if id(issue) in reviewed_before)

            # Update report summary
            if deleted_count > 0 and isinstance(summary, dict):
                severity_counts = Counter(issue["severity"] for issue in issues)
                issue_type_counts = Counter(issue["issue_type"] for issue in issues)
                summary["errors"] = severity_counts.get("error", 0)
                summary["warnings"] = severity_counts.get("warning", 0)
                summary["info"] = severity_counts.get("info", 0)
                summary["issue_types"] = dict(sorted(issue_type_counts.items()))
                summary["files_with_errors_or_warnings"] = len({
                    issue.get("absolute_path") or issue.get("file") for issue in issues
                    if issue.get("severity") in {"error", "warning"}
                })
                report["issues"] = list(issues)
                report["summary"] = summary

            # Save updated report
            try:
                write_json_file_atomic(report_path, report)
            except Exception as e:
                logger.warning("Could not update report: %s", e)

            # Refresh display
            apply_scan_report(report, report_path)

            # Show result
            msg = f"Files modified: {result.get('files_modified', 0)}\nAnnotations removed: {result.get('annotations_removed', 0)}\nFindings deleted: {deleted_count:,}"
            if result.get("errors"):
                msg += f"\n\nErrors:\n" + "\n".join(result.get("errors", [])[:5])
            QMessageBox.information(dialog, "Deletion Complete", msg)

        severity_filter.currentIndexChanged.connect(
            lambda _index: populate_table()
        )
        issue_filter.currentIndexChanged.connect(lambda _index: populate_table())
        search_filter.textChanged.connect(lambda _text: populate_table())
        hide_reviewed_checkbox.stateChanged.connect(
            lambda _state: populate_table()
        )
        current_image_checkbox.stateChanged.connect(
            lambda _state: populate_table()
        )
        clear_filters_btn.clicked.connect(lambda _checked=False: clear_filters())
        table.itemSelectionChanged.connect(update_review_button)
        table.itemDoubleClicked.connect(
            lambda _item: review_selected_issue_in_labeler()
        )
        duplicate_iou_spin.valueChanged.connect(sync_duplicate_iou)
        sam_verify_check.toggled.connect(
            lambda checked: health_settings.setValue(
                "sam_duplicate_verify", bool(checked)
            )
        )
        visual_outliers_check.toggled.connect(set_visual_outlier_scan)
        visual_outlier_threshold.valueChanged.connect(
            set_visual_outlier_threshold
        )
        visual_outlier_passes.valueChanged.connect(
            lambda value: health_settings.setValue(
                "visual_outlier_passes", int(value)
            )
        )
        visual_class_verify_check.toggled.connect(
            lambda checked: health_settings.setValue(
                "visual_class_verify_enabled", bool(checked)
            )
        )
        visual_class_verify_check.toggled.connect(
            lambda _checked: show_selected_checks()
        )
        visual_shape_verify_check.toggled.connect(
            lambda checked: health_settings.setValue(
                "visual_shape_verify_enabled", bool(checked)
            )
        )
        visual_shape_verify_check.toggled.connect(
            lambda _checked: show_selected_checks()
        )
        refresh_scan_btn.clicked.connect(refresh_full_scan)
        dedupe_btn.clicked.connect(remove_duplicates_from_report)
        review_in_labeler_btn.clicked.connect(review_selected_issue_in_labeler)
        delete_flagged_btn.clicked.connect(delete_flagged_findings)
        open_folder_btn.clicked.connect(
            lambda _checked=False: self._open_report_folder(report_path)
        )
        buttons.rejected.connect(dialog.reject)

        def set_running(running):
            dialog._scan_running = bool(running)
            if running:
                show_selected_checks("Scanning with")
            elif active_checks_label.text().startswith("Scanning with:"):
                show_selected_checks("Selected for next scan")
            # qt-material makes disabled checked indicators look empty. Keep
            # their checked appearance and lock interaction instead.
            checks_group.setEnabled(True)
            for checkbox in list(scan_checkboxes.values()) + [visual_outliers_check]:
                checkbox.setAttribute(Qt.WA_TransparentForMouseEvents, bool(running))
                checkbox.setFocusPolicy(
                    Qt.NoFocus
                    if running
                    else Qt.FocusPolicy(
                        int(checkbox.property("darkfusionReadyFocusPolicy"))
                    )
                )
            tools_group.setEnabled(True)
            for checkbox in (
                sam_verify_check,
                visual_class_verify_check,
                visual_shape_verify_check,
            ):
                checkbox.setAttribute(Qt.WA_TransparentForMouseEvents, bool(running))
                checkbox.setFocusPolicy(
                    Qt.NoFocus
                    if running
                    else Qt.FocusPolicy(
                        int(checkbox.property("darkfusionReadyFocusPolicy"))
                    )
                )
            duplicate_iou_spin.setEnabled(not running)
            dedupe_btn.setEnabled(not running)
            sam_verify_check.setEnabled(True)
            quality_group.setEnabled(not running and scan_checkboxes["quality_checks"].isChecked())
            refresh_scan_btn.setEnabled(not running)
            cancel_scan_btn.setEnabled(running)
            visual_outlier_threshold.setEnabled(visual_outliers_check.isChecked() and not running)
            visual_outlier_passes.setEnabled(visual_outliers_check.isChecked() and not running)
            visual_class_verify_check.setEnabled(visual_outliers_check.isChecked())
            visual_shape_verify_check.setEnabled(
                visual_outliers_check.isChecked()
            )
            if running:
                scan_progress.setRange(0, 0)
            elif scan_progress.maximum() == 0:
                scan_progress.setRange(0, 100)
                scan_progress.setValue(0)
            update_review_button()

        def apply_scan_report(new_report, new_path):
            nonlocal report, report_path, summary, visual_details, issue_types
            selected = selected_issue()
            selected_key = issue_key(selected) if isinstance(selected, dict) else ""
            report, report_path = new_report, new_path
            summary = report.get("summary", {}) or {}
            issues[:] = list(report.get("issues", []) or [])
            statistics_page.set_report(report)
            pending = report.get("status") == "not_started"
            ran_checks = summary.get("scan_checks", {})
            cells = {
                "Images": ("total_images", ""),
                "Label Files": ("total_label_files", ""),
                "Errors": ("errors", ""),
                "Warnings": ("warnings", ""),
                "Files to Review": ("files_with_errors_or_warnings", ""),
                "Blank Labels": ("empty_label_files", "file_checks"),
                "Missing Labels": ("missing_label_files", "file_checks"),
                "Unreadable Images": ("bad_images", "image_checks"),
                "Possible Duplicates": ("duplicate_box_candidates", "duplicate_checks"),
                "Tiny at Train Size": ("training_tiny_object_count", "tiny_checks"),
                "Possible False Positives": ("visual_outlier_candidates", "visual"),
                "Unused Classes": ("unused_class_count", "label_checks"),
            }
            for title, (key, check) in cells.items():
                skipped = (not summary.get("visual_outliers_enabled", False)
                           if check == "visual" else not ran_checks.get(check, True))
                summary_values[title].setText("—" if pending or skipped else str(summary.get(key, 0)))
                summary_values[title].setToolTip("Not scanned" if pending or skipped else "")
            visual_text, visual_details = self._visual_scan_summary(summary)
            visual_coverage_label.setText(visual_text)
            visual_coverage_label.setVisible(not pending)
            visual_details_button.setVisible(not pending and bool(visual_details))
            issue_types = summary.get("issue_types", {}) or {}
            top_items = sorted(issue_types.items(), key=lambda item: item[1], reverse=True)[:6]
            issue_label.setText(
                "No scan has run. Choose checks and press Start Scan." if pending else
                "Most common findings: " + "   ".join(
                    f"{issue_display_name(kind)}: {count}" for kind, count in top_items
                ) if top_items else "No findings from the selected checks."
            )
            balance = summary.get("class_balance", {}) or {}
            class_balance_label.setText(str(balance.get("message", "") or ""))
            class_balance_label.setVisible(
                not pending and bool(balance) and ran_checks.get("label_checks", True)
            )
            check_titles = [title for key, title in self.SCAN_CHECKS if ran_checks.get(key, True)]
            if summary.get("visual_outliers_enabled"):
                check_titles.append("Possible false positives (DINOv3)")
            if pending:
                show_selected_checks()
            else:
                active_checks_label.setText(
                    "Last scan included: "
                    + ("  •  ".join(check_titles) if check_titles else "None")
                )
            checked_checks_label.setText("Checks run: " + "; ".join(check_titles))
            checked_checks_label.setVisible(not pending)
            current_type = issue_filter.currentData()
            blocked = issue_filter.blockSignals(True)
            issue_filter.clear()
            issue_filter.addItem("All Finding Types", "")
            for kind, _count in sorted(issue_types.items(), key=lambda item: item[1], reverse=True):
                issue_filter.addItem(issue_display_name(kind), kind)
            issue_filter.setCurrentIndex(max(0, issue_filter.findData(current_type)))
            issue_filter.blockSignals(blocked)
            reviewed_issue_keys.clear()
            reviewed_issue_keys.update(self._load_reviewed_issue_keys(dataset_dir))
            populate_table(selected_key)
            if not pending:
                tabs.setCurrentWidget(results_page if issues else statistics_page if report.get("statistics") else results_page)
                scan_progress.setRange(0, 100)
                scan_progress.setValue(100)

        dialog._darkfusion_refresh_health_review = reload_external_review_state
        dialog._darkfusion_apply_scan_report = apply_scan_report
        dialog._darkfusion_set_running = set_running
        set_visual_outlier_scan(visual_outliers_check.isChecked())
        apply_scan_report(report, report_path)
        worker = getattr(self, "_analysis_worker", None)
        set_running(worker is not None and worker.isRunning())
        if report.get("status") != "not_started":
            scan_status.setText("Results from the previous scan. Change settings to scan again.")
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def open_statistics(self, plot_type=None):
        self.scan_annotations()
        dialog = getattr(self, "health_report_dialog", None)
        if dialog is None or sip.isdeleted(dialog):
            return
        dialog._darkfusion_analysis_tabs.setCurrentWidget(dialog._darkfusion_statistics)
        if plot_type:
            dialog._darkfusion_statistics.select_chart(
                {"bar": "classes", "histogram": "histogram", "scatter": "scatter"}.get(plot_type, plot_type)
            )

    def _open_report_folder(self, report_path):
        try:
            os.startfile(os.path.dirname(report_path))
        except Exception as e:
            QMessageBox.warning(self.parent, "Open Failed", f"Could not open report folder:\n{e}")

    def import_classes(self):
        dataset_dir = self._dataset_directory()
        if not dataset_dir:
            return

        classes_file = self._find_classes_file(dataset_dir)
        if classes_file:
            self.base_directory = dataset_dir
            self.valid_classes, _ = self._load_classes(dataset_dir)
            QMessageBox.information(self.parent, "Classes Found", f"Using:\n{classes_file}")
            return

        file_name, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Select Classes File",
            (
                self.parent.dialog_start_directory("metadata", dataset_dir)
                if hasattr(self.parent, "dialog_start_directory")
                else dataset_dir
            ),
            "Class Files (classes.txt *.names *.txt);;All Files (*)"
        )
        if not file_name:
            return
        if hasattr(self.parent, "remember_dialog_selection"):
            self.parent.remember_dialog_selection("metadata", file_name)

        canonical_classes = resolve_dataset_classes_path(
            dataset_dir, for_write=True, migrate=True
        )
        try:
            if os.path.normcase(os.path.abspath(file_name)) != os.path.normcase(os.path.abspath(canonical_classes)):
                shutil.copy2(file_name, canonical_classes)
            self.override_classes_file = self._normalize_path(canonical_classes)
        except OSError as e:
            QMessageBox.warning(
                self.parent,
                "Classes Import Failed",
                f"Could not copy the class list into .darkfusion:\n{e}",
            )
            return
        self.base_directory = dataset_dir
        self.valid_classes, _ = self._load_classes(dataset_dir)
        QMessageBox.information(self.parent, "Classes Loaded", f"Using:\n{canonical_classes}")

    def scan_annotations(self, dataset_dir=None):
        """Open configuration immediately; opening this window never starts a scan."""
        requested = os.fspath(dataset_dir) if isinstance(dataset_dir, (str, bytes, os.PathLike)) else ""
        directory = self._normalize_path(requested) if requested and os.path.isdir(requested) else self._dataset_directory()
        worker = getattr(self, "_analysis_worker", None)
        if worker is not None and worker.isRunning():
            directory = worker.dataset_dir
        if not directory:
            QMessageBox.information(self.parent, "Dataset Analysis", "No dataset folder selected.")
            return
        existing = getattr(self, "health_report_dialog", None)
        if existing is not None and not sip.isdeleted(existing) and getattr(existing, "_dataset_directory", "") == directory:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return
        report = self.scan_report
        if not isinstance(report, dict) or self._normalize_path(report.get("dataset_dir", "")) != directory:
            report = {"dataset_dir": directory, "status": "not_started", "summary": {}, "issues": []}
        self.show_scan_report_dialog(report, self._report_path(directory))

    @gpu_work("Dataset Analysis", keep=analysis_resources, until="_analysis_worker", owner_attr="parent")
    def _start_analysis_scan(self, dataset_dir=None):
        active_worker = getattr(self, "_analysis_worker", None)
        if active_worker is not None:
            try:
                if active_worker.isRunning():
                    self._set_scan_status("A dataset scan is already running.")
                    return
            except RuntimeError:
                self._analysis_worker = None

        requested_dataset_dir = (
            os.fspath(dataset_dir)
            if isinstance(dataset_dir, (str, bytes, os.PathLike))
            else ""
        )
        dataset_dir = (
            self._normalize_path(requested_dataset_dir)
            if requested_dataset_dir and os.path.isdir(requested_dataset_dir)
            else self._dataset_directory()
        )
        if not dataset_dir:
            QMessageBox.information(self.parent, "Dataset Analysis", "No dataset folder selected.")
            return

        self._reset_scan_state()
        self.base_directory = dataset_dir
        if hasattr(self.parent, "image_directory"):
            self.parent.image_directory = dataset_dir
        if hasattr(self.parent, "activate_project_settings"):
            self.parent.activate_project_settings(dataset_dir)
            self.parent.settings["last_dir"] = dataset_dir
            self.parent.saveSettings()

        checks = self._saved_scan_checks()
        health_settings = QSettings("UltraDarkFusion", "HealthCheck")
        visual_outliers_enabled = str(
            health_settings.value("visual_outliers_enabled", "false")
        ).lower() in ("1", "true", "yes")
        visual_class_verify_enabled = str(
            health_settings.value("visual_class_verify_enabled", "true")
        ).lower() in ("1", "true", "yes")
        visual_shape_verify_enabled = str(
            health_settings.value(
                "visual_shape_verify_enabled",
                health_settings.value("visual_foreground_verify_enabled", "true"),
            )
        ).lower() in ("1", "true", "yes")
        needs_annotations = any(checks[key] for key in (
            "label_checks", "duplicate_checks", "tiny_checks", "statistics",
        )) or visual_outliers_enabled
        if needs_annotations:
            self.valid_classes, classes_file = self._load_classes(dataset_dir)
        else:
            self.valid_classes, classes_file = [], self._find_classes_file(dataset_dir)
        if needs_annotations and not self.valid_classes:
            self._set_scan_status("A class list is required for the selected annotation checks.")
            QMessageBox.warning(
                self.parent,
                "Dataset Analysis",
                "No class list was found. Open the dataset to create .darkfusion/classes.txt "
                "or import a class list. File checks and image readability can run without classes."
            )
            return

        self.expected_keypoint_count = self._load_expected_keypoint_count(dataset_dir)

        training_size = self._training_input_size()
        training_size_source = self.training_input_size_source
        try:
            polygon_preference = self.parent._polygon_label_parse_preference()
        except Exception:
            polygon_preference = None
        task_text = ""
        task_combo = getattr(self.parent, "task_combobox", None)
        if task_combo is not None:
            try:
                task_text = task_combo.currentText()
            except Exception:
                task_text = ""
        expected_annotation_family = self._annotation_family_for_task(task_text)
        if not expected_annotation_family:
            try:
                expected_annotation_family = self._annotation_family_for_task(self.parent.current_mode())
            except Exception:
                expected_annotation_family = ""
        if expected_annotation_family == "segmentation":
            polygon_preference = "segmentation"
        elif expected_annotation_family == "obb":
            polygon_preference = "obb"

        try:
            visual_outlier_threshold = float(
                health_settings.value("visual_outlier_threshold", 0.45)
            )
        except (TypeError, ValueError):
            visual_outlier_threshold = 0.45
        try:
            visual_outlier_passes = int(
                health_settings.value("visual_outlier_passes", 8) or 8
            )
        except (TypeError, ValueError):
            visual_outlier_passes = 8

        scan_context = {
            "checks": checks,
            "quality_thresholds": self._saved_quality_thresholds(),
            "valid_classes": self.valid_classes,
            "expected_keypoint_count": self.expected_keypoint_count,
            "training_size": training_size,
            "training_size_source": training_size_source,
            "polygon_preference": polygon_preference,
            "expected_annotation_family": expected_annotation_family,
            "duplicate_iou_threshold": self._duplicate_iou_threshold(),
            "visual_outliers_enabled": visual_outliers_enabled,
            "visual_outlier_threshold": max(
                0.20, min(0.90, visual_outlier_threshold)
            ),
            "visual_outlier_passes": max(1, min(100, visual_outlier_passes)),
            "visual_class_verify_enabled": visual_class_verify_enabled,
            "visual_shape_verify_enabled": visual_shape_verify_enabled,
        }

        worker = DatasetAnalysisWorker(
            self,
            dataset_dir,
            classes_file,
            scan_context,
            parent=self,
        )
        self._analysis_worker = worker
        worker.progress.connect(self._on_scan_progress)
        worker.completed.connect(self._on_scan_completed)
        worker.failed.connect(self._on_scan_failed)
        worker.canceled.connect(self._on_scan_canceled)
        worker.finished.connect(lambda worker=worker: self._on_scan_worker_finished(worker))
        worker.finished.connect(worker.deleteLater)

        self._set_scan_controls_running(True)
        self._set_scan_status(f"Scanning dataset folder: {dataset_dir}")
        worker.start()

    def _run_scan_background(
        self,
        dataset_dir,
        classes_file,
        scan_context,
        should_cancel,
        progress_callback,
    ):
        """Worker-thread scan body. It must not read or update Qt widgets."""
        self._reset_scan_state()
        scan_start_time = datetime.now()
        self.base_directory = self._normalize_path(dataset_dir)
        self.valid_classes = scan_context.get("valid_classes", [])
        self.expected_keypoint_count = int(scan_context.get("expected_keypoint_count", 0) or 0)
        self._scan_training_size_override = tuple(scan_context.get("training_size", (0, 0)))
        self._scan_training_size_source_override = str(
            scan_context.get("training_size_source", "") or ""
        )
        self._scan_polygon_preference_override = scan_context.get("polygon_preference")
        self._scan_expected_annotation_family_override = scan_context.get("expected_annotation_family")
        self._scan_duplicate_iou_override = scan_context.get("duplicate_iou_threshold")
        self._scan_visual_outliers_override = bool(
            scan_context.get("visual_outliers_enabled", False)
        )
        self._scan_visual_outlier_threshold_override = scan_context.get(
            "visual_outlier_threshold", 0.45
        )
        self._scan_visual_outlier_passes_override = scan_context.get(
            "visual_outlier_passes", 8
        )
        self._scan_visual_class_verify_override = bool(
            scan_context.get("visual_class_verify_enabled", True)
        )
        self._scan_shape_verify_override = bool(
            scan_context.get("visual_shape_verify_enabled", True)
        )
        self._scan_checks_override = dict(scan_context.get("checks", {}) or {})
        self._scan_quality_thresholds = dict(QUALITY_DEFAULTS, **(scan_context.get("quality_thresholds") or {}))
        self._scan_context_active = True

        logger.info(f"Dataset Analysis scan started at {scan_start_time.isoformat()} for {dataset_dir}")

        image_reads = None
        try:
            # Connection ownership stays on this worker, never on the UI thread.
            self._image_check_cache = ImageCheckCache(os.path.join(
                dataset_dir, PROJECT_SETTINGS_DIR, "image_checks.sqlite3"
            ))
            progress_callback(0, 0, "Finding images and labels...")
            image_files = self._background_image_files(dataset_dir, scan_context)
            if should_cancel():
                return None

            label_files = self._iter_label_files(dataset_dir)
            if should_cancel():
                return None

            image_map = {
                os.path.splitext(os.path.basename(path))[0].lower(): path
                for path in image_files
            }
            image_stem_groups = defaultdict(list)
            for image_path in image_files:
                image_stem_groups[
                    os.path.splitext(os.path.basename(image_path))[0].lower()
                ].append(image_path)
            self._ambiguous_image_stems = {
                stem for stem, paths in image_stem_groups.items() if len(paths) > 1
            }
            if self._scan_check_enabled("file_checks"):
                for stem in sorted(self._ambiguous_image_stems):
                    paths = image_stem_groups[stem]
                    names = ", ".join(os.path.basename(path) for path in paths)
                    for image_path in paths:
                        self.issues.append(self._make_issue(
                            "duplicate_image_stem",
                            "error",
                            image_path,
                            (
                                f"Multiple images share the label name '{stem}.txt': {names}. "
                                "DarkFusion cannot know which image owns that label."
                            ),
                            suggestion="Rename one image and its label so every image has a unique basename.",
                        ))
            label_map = {
                os.path.splitext(os.path.basename(path))[0].lower(): path
                for path in label_files
            }

            self.total_images = len(image_files)
            self.total_label_files = len(label_files)
            scan_total = max(1, len(image_files) + len(label_files) + 1)
            scan_progress = 0
            progress_step = max(1, scan_total // 500)
            progress_callback(scan_progress, scan_total, "Reading image metadata...")

            verify_images = self._scan_check_enabled("image_checks")
            quality_images = self._scan_check_enabled("quality_checks")
            if verify_images or quality_images:
                image_reads = self._image_check_cache.read_images(
                    image_files, verify=verify_images,
                    quality_function=image_quality_metrics if quality_images else None,
                    should_cancel=should_cancel,
                )
            else:
                image_reads = ((path, None) for path in image_files)
            for image_file, image_facts in image_reads:
                if should_cancel():
                    return None

                base_name = os.path.splitext(os.path.basename(image_file))[0].lower()
                if (self._scan_check_enabled("image_checks") or self._scan_check_enabled("quality_checks")
                        or self._needs_annotation_records()):
                    image_size = self._read_image_size(image_file, facts=image_facts)
                    if image_size:
                        self._image_size_cache[image_file] = image_size
                if self._scan_check_enabled("file_checks") and base_name not in label_map:
                    self.missing_label_files += 1
                    self.issues.append(self._make_issue(
                        "missing_label_file",
                        "warning",
                        image_file,
                        "Image has no matching label file.",
                        suggestion=(
                            "Create an empty label file for an intentional negative image, or open "
                            "the image from Dataset Analysis in Label Maker and label its objects."
                        ),
                    ))

                scan_progress += 1
                if scan_progress % progress_step == 0:
                    progress_callback(scan_progress, scan_total, "Reading image metadata...")

            for label_file in label_files:
                if should_cancel():
                    return None

                if self._scan_check_enabled("file_checks") or self._needs_annotation_records():
                    self._scan_label_file(label_file, image_map)
                scan_progress += 1
                if scan_progress % progress_step == 0:
                    progress_callback(scan_progress, scan_total, "Parsing annotations...")

            if should_cancel():
                return None

            if self._scan_check_enabled("label_checks"):
                self._add_dataset_annotation_type_issues()
            if self._scan_visual_outliers_override:
                progress_callback(
                    0, max(1, len(image_files)),
                    "Checking within-class visual consistency...",
                )
                self._scan_visual_outliers(should_cancel, progress_callback)
                if should_cancel():
                    return None
            if self._scan_check_enabled("file_checks"):
                self._scan_orphan_json_files(dataset_dir, image_map)
            if self._scan_check_enabled("statistics") or self._scan_check_enabled("quality_checks"):
                progress_callback(0, 0, "Summarizing statistics from this scan...")
                self.statistics_report = build_statistics(
                    image_files, self._image_size_cache, self.annotation_family_records,
                    self._statistics_label_states, self.valid_classes, self._quality_records,
                    basic_enabled=self._scan_check_enabled("statistics"),
                    quality_enabled=self._scan_check_enabled("quality_checks"),
                    quality_thresholds=self._scan_quality_thresholds, should_cancel=should_cancel,
                )
                if should_cancel() or self.statistics_report is None:
                    return None
            self.issues = self._selected_scan_issues()
            # Report construction and the atomic disk write are still active
            # work. Keep the bar indeterminate so 100% only appears after the
            # completed signal has also refreshed the results UI.
            progress_callback(0, 0, "Writing analysis report...")

            # Diagnostic: Log data freshness and consistency
            logger.info(
                f"Dataset Analysis scan complete. Fresh data verified:"
                f" {len(image_files)} images, {self.total_label_files} label files,"
                f" {self.total_labels} annotations, {len(self.issues)} issues found."
            )

            # Log issue breakdown for debugging stale data
            issue_types = {}
            for issue in self.issues:
                issue_type = issue.get("issue_type", "unknown")
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            if issue_types:
                types_str = ", ".join(f"{k}={v}" for k, v in sorted(issue_types.items()))
                logger.debug(f"Issue breakdown by type: {types_str}")

            report = self._build_report(dataset_dir, classes_file, image_files, label_files)
            report_path = self._report_path(dataset_dir)
            if should_cancel():
                return None

            write_json_file_atomic(report_path, report)

            scan_end_time = datetime.now()
            scan_duration = (scan_end_time - scan_start_time).total_seconds()
            logger.info(
                f"Dataset Analysis scan finished at {scan_end_time.isoformat()} "
                f"(duration: {scan_duration:.1f}s). Report: {report_path}"
            )
            return report, report_path
        finally:
            if image_reads is not None:
                image_reads.close()
            image_cache = self._image_check_cache
            self._image_check_cache = None
            if image_cache is not None:
                image_cache.close()
                if image_cache.hits or image_cache.misses:
                    logger.info("Dataset Analysis image checks: %d reused, %d inspected.",
                                image_cache.hits, image_cache.misses)
            self._scan_training_size_override = None
            self._scan_training_size_source_override = ""
            self._scan_polygon_preference_override = None
            self._scan_expected_annotation_family_override = None
            self._scan_duplicate_iou_override = None
            self._scan_visual_outliers_override = None
            self._scan_visual_outlier_threshold_override = None
            self._scan_visual_outlier_passes_override = None
            self._scan_visual_class_verify_override = None
            self._scan_checks_override = None
            self._scan_context_active = False

    def _background_image_files(self, dataset_dir, scan_context):
        dataset_dir = self._normalize_path(dataset_dir)
        # Read the explicit dataset folder on every scan.  Cached/filtered
        # labeler lists are navigation state, not a reliable dataset manifest.
        image_files = []
        with os.scandir(dataset_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                if entry.name.lower().endswith(self.valid_image_extensions):
                    image_files.append(self._normalize_path(entry.path))
        return sorted(image_files)

    def _on_scan_progress(self, value, maximum, message):
        dialog = getattr(self, "health_report_dialog", None)
        if dialog is not None and not sip.isdeleted(dialog):
            progress = dialog._darkfusion_scan_progress
            progress.setRange(0, max(0, int(maximum)))
            if maximum > 0:
                progress.setValue(min(int(value), int(maximum)))
        if maximum <= 0:
            progress = getattr(self.parent, "label_progress", None)
            if progress is not None:
                progress.setRange(0, 0)
        elif hasattr(self.parent, "set_main_progress"):
            self.parent.set_main_progress(value, maximum)
        self._set_scan_status(message)

    def _on_scan_completed(self, report, report_path):
        self.scan_report = report
        self._set_scan_status("Loading Dataset Analysis results...")
        dialog = getattr(self, "health_report_dialog", None)
        if dialog is not None and not sip.isdeleted(dialog) and dialog.isVisible():
            self.show_scan_report_dialog(report, report_path)
        if hasattr(self.parent, "set_main_progress"):
            self.parent.set_main_progress(100, 100)
        self._set_scan_status("Dataset Analysis complete.", timeout_ms=4000)

    def _on_scan_failed(self, message):
        if hasattr(self.parent, "set_main_progress"):
            self.parent.set_main_progress(0, 100)
        self._set_scan_status("Dataset Analysis failed.", timeout_ms=4000)
        QMessageBox.critical(self.parent, "Dataset Analysis", f"Dataset analysis failed:\n{message}")

    def _on_scan_canceled(self):
        if hasattr(self.parent, "set_main_progress"):
            self.parent.set_main_progress(0, 100)
        self._set_scan_status("Dataset Analysis canceled.", timeout_ms=3000)

    def _on_scan_worker_finished(self, worker):
        if self._analysis_worker is worker:
            self._analysis_worker = None
            self._set_scan_controls_running(False)

    def _set_scan_controls_running(self, running):
        dialog = getattr(self, "health_report_dialog", None)
        if dialog is not None and not sip.isdeleted(dialog):
            dialog._darkfusion_set_running(bool(running))

    def _set_scan_status(self, message, timeout_ms=0):
        dialog = getattr(self, "health_report_dialog", None)
        if dialog is not None and not sip.isdeleted(dialog):
            dialog._darkfusion_scan_status.setText(str(message))
        try:
            status_bar = self.parent.statusBar()
            if status_bar is not None:
                status_bar.showMessage(str(message), int(timeout_ms))
        except Exception:
            pass

    def cancel_active_scan(self, wait_ms=0):
        worker = getattr(self, "_analysis_worker", None)
        if worker is None:
            return False
        try:
            if not worker.isRunning():
                return False
            worker.cancel()
            self._set_scan_status("Canceling Dataset Analysis...")
            if wait_ms:
                worker.wait(max(0, int(wait_ms)))
            return True
        except RuntimeError:
            self._analysis_worker = None
            self._set_scan_controls_running(False)
            return False


def _centralize_dataset_analysis_ui(self):
    """Keep dataset analysis visible as one hub instead of scattered menu shortcuts."""
    scan_action = getattr(self, "scan_health_report_action", None)
    if scan_action is not None:
        scan_action.setText("Dataset Analysis")
        scan_action.setToolTip("Open health checks, duplicate cleanup, statistics, and plots in one place.")
        scan_action.setVisible(True)

    dedupe_action = getattr(self, "deduplicate_labels_action", None)
    if dedupe_action is not None:
        dedupe_action.setVisible(False)

    for action_name in ("call_stats_button", "histogram_plot", "bar_plot", "scatter_plot"):
        action = getattr(self, action_name, None)
        if action is not None:
            action.setVisible(False)

    plots_menu = getattr(self, "menuplots", None)
    if plots_menu is not None:
        plots_menu.menuAction().setVisible(False)

    view_menu = getattr(self, "menufunctions", None)
    if view_menu is not None:
        view_menu.menuAction().setVisible(False)


def _compact_health_check_panel(self):
    group = getattr(self, "groupBox_4", None)
    scan_button = getattr(self, "scan_button", None)
    dedupe_button = getattr(self, "deduplicate_labels_button", None)
    if group is None or getattr(self, "_health_check_panel_compacted", False):
        return

    layout = group.layout()
    for widget_name, widget in (
        ("scan_button", scan_button),
        ("deduplicate_labels_button", dedupe_button),
    ):
        if isinstance(widget, QtWidgets.QWidget):
            self._take_layout_widget(layout, widget)
            widget.setParent(None)
            widget.deleteLater()
        setattr(self, widget_name, None)

    parent = group.parentWidget()
    parent_layout = parent.layout() if parent is not None else None
    self._take_layout_widget(parent_layout, group)
    group.setParent(None)
    group.deleteLater()
    setattr(self, "groupBox_4", None)

    self._health_check_panel_compacted = True


def create_plot(self, plot_type="scatter"):
    """Show a chart from the last completed Dataset Analysis scan."""
    self.scan_annotations.open_statistics(plot_type)


def display_stats(self):
    """Open the shared Statistics tab without scanning."""
    self.scan_annotations.open_statistics()


def toggle_image_quality_analysis(self, checked):
    self.image_quality_analysis_enabled = bool(checked)
    QSettings("UltraDarkFusion", "HealthCheck").setValue("scan_quality_checks", bool(checked))
    dialog = getattr(self.scan_annotations, "health_report_dialog", None)
    if dialog is not None and not sip.isdeleted(dialog):
        checkbox = dialog.findChild(QtWidgets.QCheckBox, "datasetScan_quality_checks")
        if checkbox is not None and checkbox.isEnabled():
            checkbox.setChecked(bool(checked))

