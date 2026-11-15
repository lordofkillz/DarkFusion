from darkfusion_environment import configure_runtime
configure_runtime()
from darkfusion_version import APP_VERSION
from darkfusion_environment import configure_gui_opencv_threads

import warnings
from darkfusion_windows_ml import configure as configure_onnx_runtime
from darkfusion_windows_ml import settings as onnx_runtime_settings
from darkfusion_inference import (
    load_inference_model, inference_backend_for_path, normalize_inference_backend, read_onnx_model_names,
    tracking_dependencies_available, ultralytics_kwargs_use_fp16,
    sam_predictor_session, sam_predict_with_image_cache,
)
from darkfusion_dataset_statistics import (
    DatasetStatisticsWidget, QUALITY_DEFAULTS, build_statistics,
    image_quality_metrics, quality_findings, ImageCheckCache,
)
from darkfusion_ui_controls import (
    CollapsibleSettingsGroup, CollapsibleSectionsController, MultipleOfSpinBox,
    PercentSliderControl, snap_int_to_multiple, blocked_signals,
    dock_chrome_stylesheet, connect_weak_bound_signal, DockHelpButtonController,
    RedBoxDelegate, ClassColorButton, ROIEdgeSliders,
)
from darkfusion_training_size import (
    TrainingSizeAnalysis, candidate_sizes as training_candidate_sizes,
    parse_training_batch, training_file_signature, training_image_dimensions,
)
from darkfusion_verified_images import VerifiedImagesTracker, get_verified_images_for_scan
from darkfusion_shape_verifier import ShapeProfiler, shape_verification_score, shape_annotation_crop, ShapeReferenceCache
from darkfusion_borderline_llm_verifier import BorderlineLLMVerifier
from darkfusion_ollama_auto_setup import ensure_ollama_ready, check_ollama_running
from darkfusion_video_links import (
    SUPPORTED_VIDEO_URL_EXTS, yt_dlp_runtime_options,
    video_stream_quality_key, video_stream_format_selector,
)
from darkfusion_dataset_operations import (
    delete_image_label_pair, natural_image_key, excluded_video_frames,
    sparse_video_navigation, update_health_report_after_deletion, is_placeholder_path,
    DatasetFileTransaction, normalize_dataset_path,
    PROJECT_SETTINGS_DIR, PROJECT_SETTINGS_FILE, DATASET_METADATA_FILES,
    dataset_metadata_directory, dataset_metadata_path, _files_have_same_content,
    resolve_dataset_metadata_path, resolve_dataset_classes_path,
    migrate_darkfusion_dataset_metadata, resolve_dataset_pose_schema_path,
)
from darkfusion_dataset_workers import (
    DatasetNaturalSortWorker, GridImageDeleteWorker, DatasetDirectoryScanWorker,
    FileRevealSignals, FileRevealWorker, BulkMoveWorker, FilteredExportWorker,
)
from darkfusion_dataset_review import (
    ImageListModel, GridThumbnailSignals, GridThumbnailWorker, GridThumbnailModel,
    GridReviewView, ReviewFilterWorker, ReviewSimilarityWorker,
)
from darkfusion_dataset_operations import read_json_file, write_json_file_atomic
from darkfusion_dataset_workers import ImageConverterRunnableSignals, ImageConverterRunnable
from darkfusion_image_loading import (
    DATASET_CONVERSION_TARGET_SUFFIXES, FORMAT_MAP, image_format_for_extension,
)
from darkfusion_video_capture import get_video_codec_fourcc, get_video_output_fps
from darkfusion_propagation import (
    _PropagationFrameCache, AdjacentPropagationWorker, Sam3PropagationWarmupWorker,
)
from darkfusion_inference import (
    SAM3_DEFAULT_IMGSZ, SAM3_MIN_IMGSZ, SAM3_STRIDE, normalize_sam_imgsz,
)
from darkfusion_dataset_analysis import DatasetAnalysisWorker, ScanAnnotations
from darkfusion_annotation_style import (
    _CLASS_COLOR_CACHE_LOCK, _CLASS_COLOR_SETTINGS_CACHE, _CLASS_COLOR_CACHE_CHECK_INTERVAL,
    invalidate_class_color_cache, _class_color_settings_signature, _saved_class_color_settings,
    get_color, safe_update_class_name_item_font, annotation_labels_hidden,
    update_annotation_label_badge, annotation_setting_number, annotation_outline_width,
    annotation_hover_width, pose_skeleton_width, annotation_handle_radius,
    annotation_pen, CheckboxDelegate,
)
from darkfusion_settings import (
    UI_SCALE_MIN_PERCENT, UI_SCALE_MAX_PERCENT, CLASS_HOTKEY_PREFIX,
    KEYBIND_GROUPS, KEYBIND_LABELS, normalize_keybind_value,
    class_hotkey_key, keybind_label_for_key, iter_keybind_entries,
    find_keybind_conflict, SettingsDialog,
)
from darkfusion_dataset_operations import label_file_version, write_label_text_atomic
from darkfusion_annotations import class_name_index
from darkfusion_dataset_workers import ClearLabelsWorker, ClassLabelRemapWorker
from darkfusion_class_editor import MergeClassDialog, ClassPickerDialog
from darkfusion_environment import (
    APP_DIR, APP_SETTINGS_FILE, APP_SETTINGS_PATH,
)
from darkfusion_video_capture import (
    video_playback_quality_key, video_playback_renderer_key,
)
from darkfusion_propagation import (
    LazyPropagationRecords, propagation_status, run_propagation_worker, sam3_frame_source,
    save_propagation_outputs,
    undo_propagation_outputs,
    find_yolo_box_gaps, compute_yolo_box_gap_proposals,
)
from darkfusion_extraction import ExtractionCheckpoint, ExtractionWriteError, save_extraction_outputs
from darkfusion_image_loading import (
    EditorImageSource, EditorLabelSource, IMAGE_SUFFIXES, OUTPUT_IMAGE_SUFFIXES,
    NON_LABELING_IMAGE_SUFFIXES, DATASET_IMAGE_SUFFIXES, normalize_image_extension, reflect_pad_pil,
)
from darkfusion_gpu_memory import (
    gpu_work, model_allowed as gpu_model_allowed, attach as attach_gpu_memory,
    ensure_detector as ensure_gpu_detector, auto_label_resources, sam_batch_resources,
    analysis_resources, release_worker_models, release_analysis_models, start_gpu_worker,
    adopt_saved_training, GpuWorkBusy,
)
from darkfusion_video_capture import optimize_video_capture, negotiated_capture_properties
import ast
import os
import xml.etree.ElementTree as ET

# Keep Ultralytics from failing startup if the default AppData settings path
# becomes locked or unreadable.
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    os.path.join(APP_DIR, ".codex-ultralytics"),
)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import cv2
import html

# Import necessary libraries and modules for the application
import logging
from logging.handlers import RotatingFileHandler
import concurrent.futures
import csv
import gc
import hashlib
import importlib.util
from datetime import datetime
from typing import List
import glob
import json
import threading
import random
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
import pyautogui
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_PATH
import torch
import numpy as np
import yaml
import psutil
import GPUtil
from darkfusion_system_metrics import GpuPowerSampler
from darkfusion_translation import TranslationController
from darkfusion_review_settings import (
    METHOD_OPTIONS, RECOMMENDED_THRESHOLDS, migrate_review_settings,
    review_method, review_threshold, select_review_method, set_review_threshold,
)
from threading import Thread
import functools
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import (QAbstractListModel, QEvent, QModelIndex, QObject,
                          QRectF, QRunnable, Qt, QThread, QThreadPool,
                          QTimer, QUrl, pyqtSignal, pyqtSlot, QPointF,
                          QRect, QSettings)
from PyQt5.QtGui import (QBrush, QColor, QFont, QImage, QImageReader,
                         QImageWriter, QMovie, QPainter, QPen,
                         QPixmap,  QStandardItem,
                         QStandardItemModel, QLinearGradient,QIcon,QCursor,QMouseEvent,QKeyEvent,QPainterPath,QPolygonF,QPalette,
                         QTextBlockFormat, QTextCursor)
from PyQt5.QtWidgets import (QApplication, QFileDialog,
                             QGraphicsDropShadowEffect, QGraphicsItem,
                             QGraphicsPixmapItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsTextItem, QGraphicsView,
                             QLabel, QMessageBox, QProgressBar,
                             QTableWidgetItem,QGraphicsPathItem,QColorDialog, QMenu,QTableView,QGraphicsLineItem, QVBoxLayout,QHBoxLayout,QFrame,QWidget,QHeaderView,QStyledItemDelegate,QStyle,QTabWidget,QStyleOptionButton,QGraphicsPolygonItem,QGraphicsEllipseItem,QCheckBox,QAbstractItemView)
from PyQt5 import QtGui
from concurrent.futures import ThreadPoolExecutor, as_completed
from qt_material import build_stylesheet, list_themes
import matplotlib
matplotlib.use('Qt5Agg')
from perlin_noise import PerlinNoise
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QAction
from darkfusion_sahi import SahiPredictWrapper
from sahi.slicing import get_slice_bboxes as get_sahi_slice_bboxes
from PyQt5.QtWidgets import (QDialog, QLineEdit,QPushButton, QSpinBox, QComboBox,QDoubleSpinBox)
from ui_ultradarkfusion import Ui_mainWindow
from PyQt5.QtCore import QItemSelectionModel
import uuid
from darkfusion_model_security import install_checkpoint_guard
install_checkpoint_guard()
from collections import defaultdict
try:
    import sip
except ModuleNotFoundError:
    from PyQt5 import sip
import mss
import codecs
from dotenv import load_dotenv, set_key
import math
from shapely.geometry import box as shapely_box
from shapely.geometry import Polygon
from ultralytics import SAM
from collections import Counter
from PyQt5.QtWidgets import QInputDialog
from splash_utils import show_splash
import faulthandler
import traceback
import contextlib
import weakref
from prediction_size_filter import (
    filter_overlay_to_size_limits as shared_filter_overlay_to_size_limits,
    filter_predictions_to_size_limits as shared_filter_predictions_to_size_limits,
    prediction_size_allowed as shared_prediction_size_allowed,
    prediction_size_allowed_xyxy,
    prediction_dimensions_allowed,
    prediction_size_limits,
    polygon_touches_roi_rect as shared_polygon_touches_roi_rect,
)



from darkfusion_training_widgets import TrainingMetricsPlot, TrainingBatchSpinBox, TrainingEvaluatorActivityIndicator


from darkfusion_image_viewers import ZoomableImageView, AugmentationPreviewViewer, TrainingArtifactViewer
from darkfusion_image_viewers import FastFrameViewer, graphics_image_dimensions, pg
from darkfusion_image_viewers import NestedLabelSuggestionViewer
from darkfusion_annotation_items import (
    BoundingBoxDrawer, BoxVertexHandle, KeypointHandle, KeypointDrawer,
    SegmentationDrawer, VertexHandle, OBBDrawer, SolidGradientBorderItem,
)
from darkfusion_canvas import PanPuckSettingsDialog, FloatingPanPuck, CustomGraphicsView
from darkfusion_autolabel import SahiWorker, SahiSettingsDialog, AutoLabelDialog, DinoWorker
from darkfusion_annotations import _LabelRectEntry


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

from darkfusion_canvas import SUPER_RES_MAX_INPUT_PIXELS

from darkfusion_annotations import DEFAULT_POSE_KEYPOINT_LABELS

from darkfusion_annotations import DEFAULT_POSE_SKELETON

from darkfusion_annotations import DEFAULT_POSE_FLIP_IDX


from darkfusion_settings import PROJECT_SETTING_KEYS

UI_LINE_HEIGHT_PERCENT = 122
UI_LAYOUT_SPACING = 8
from darkfusion_settings import CURRENT_UI_LAYOUT_VERSION

GLOBAL_LINE_SPACING_QSS = """
/* DarkFusion global control sizing. Avoid QSS line-height because Qt can clip text. */
QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox {
    min-height: 22px;
}

QTextEdit,
QPlainTextEdit {
    min-height: 26px;
}

QAbstractItemView::item {
    padding-top: 5px;
    padding-bottom: 5px;
}

QCheckBox {
    spacing: 6px;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
}

QGroupBox {
    margin-top: 12px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
}
"""

# Keep this block at the end of every selected theme. A number of legacy and
# third-party themes either omit the unchecked state or use an image that is
# unavailable, which makes checkbox indicators disappear into the panel.
GLOBAL_CHECKBOX_OUTLINE_QSS = """
/* DarkFusion checkbox visibility override. */
QCheckBox {
    spacing: 6px;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #8491a1;
    border-radius: 3px;
    background-color: #10151c;
}

QCheckBox::indicator:unchecked:hover {
    border: 2px solid #65b7ff;
    background-color: #172331;
}

QCheckBox::indicator:checked,
QCheckBox::indicator:indeterminate {
    border: 2px solid #9bd3ff;
    background-color: #1976c9;
}

QCheckBox::indicator:checked:hover,
QCheckBox::indicator:indeterminate:hover {
    border-color: #d5edff;
    background-color: #238de8;
}

QCheckBox::indicator:disabled {
    border: 1px solid #59616d;
    background-color: #272c33;
}
"""

DARKFUSION_SHELL_QSS = """
/* DarkFusion shell polish: keep generated Designer panels from feeling boxed-in. */
QDockWidget {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}
QDockWidget::title {
    padding: 5px 8px;
    text-align: left;
}
QDockWidget QWidget {
    background-color: transparent;
}
QTabWidget::pane {
    border: 1px solid #30363d;
    border-radius: 4px;
    background-color: #0d1117;
}
QTabBar::tab {
    padding: 7px 10px;
    margin-right: 2px;
}
QGroupBox {
    border: 1px solid #30363d;
    border-radius: 6px;
    margin-top: 12px;
    padding: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QFrame#frame_27,
QFrame#frame_11,
QFrame#frame_12,
QFrame#status_shell_panel,
QFrame#status_labeling_panel,
QFrame#display_compact_panel,
QFrame#effects_compact_panel,
QFrame#sam3_compact_panel,
QFrame#keypoint_compact_panel {
    background-color: transparent;
    border: none;
}
QPushButton#cuda_monitor_button {
    font-weight: 700;
    min-height: 28px;
}
QLabel#console_output {
    color: #c9d1d9;
    padding: 3px 0;
}
QLabel#image_label_2 {
    border: none;
    background-color: transparent;
}
QProgressBar#label_progress {
    min-height: 12px;
    max-height: 16px;
    border: 1px solid #30363d;
    border-radius: 7px;
    text-align: center;
}
"""



from darkfusion_settings import DEFAULT_KEYBINDS

from darkfusion_settings import PROJECT_SETTING_PREFIXES


from darkfusion_settings import is_project_setting_key


from darkfusion_image_loading import normalize_augmentation_output_format


from darkfusion_image_loading import cv2_write_params_for_extension


from darkfusion_image_loading import save_cv_image


from darkfusion_image_loading import extraction_frame_paths


from darkfusion_image_loading import existing_labeled_extraction_image


from darkfusion_image_loading import normalize_dataset_conversion_extension


from darkfusion_image_loading import prepare_image_for_dataset_format


from darkfusion_image_loading import convert_dataset_image_file


# Dump a Python-level traceback to stderr if the interpreter crashes.
faulthandler.enable()


def log_uncaught_exceptions(ex_cls, ex, tb):
    text = ''.join(traceback.format_exception(ex_cls, ex, tb))
    logging.critical("Uncaught exception:\n%s", text)
    print(f"\nUncaught exception:\n{text}")
sys.excepthook = log_uncaught_exceptions


dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path)
THUMBS_BY_CLASS_ROOT = "thumbnails_by_class"
from darkfusion_sam import DEVICE
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

dotenv_path = os.path.join(os.getcwd(), ".env")
try:
    load_dotenv(dotenv_path)
except Exception as e:
    print(f"Failed to load .env file: {e}")

MAX_LOG_LINE_LENGTH = 500

class TruncatingFormatter(logging.Formatter):

    def format(self, record):
        original = super().format(record)
        max_len = globals().get("MAX_LOG_LINE_LENGTH", 500)

        if len(original) > max_len:
            return original[:max_len] + "..."

        return original


def setup_logger():
    logger = logging.getLogger("UltraDarkFusionLogger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    log_dir = "debug"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "my_app.log")

    # Truncate the log file if it gets too large at startup.
    MAX_STARTUP_SIZE = 64 * 1024
    if os.path.exists(log_file) and os.path.getsize(log_file) > MAX_STARTUP_SIZE:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("")
        print("Cleared large log file at startup")

    # Rotating file handler
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=16 * 1024,
        backupCount=0,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = TruncatingFormatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = TruncatingFormatter(
        "%(name)-20s: %(levelname)-8s %(message)s"
    )

    # UTF-8 fix for Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "replace")

    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

from darkfusion_annotations import BoundingBox

# Base Video Processor Class

from darkfusion_dataset_workers import VideoProcessor

# Video processing worker for camera/video frame extraction.

from darkfusion_dataset_workers import GUIVideoProcessor


from darkfusion_annotations import InvalidAnnotationError


from darkfusion_image_loading import InvalidImageError


from darkfusion_training_dialog import UltralyticsExportWorker


# YouTube download worker.

from darkfusion_video_links import DownloadThread


from darkfusion_video_inference import VideoInferenceThread, VideoFrameReaderThread, DesktopFrameReaderThread


from darkfusion_video_capture import video_resize_policy_key


from darkfusion_video_capture import resize_interpolation_for


from darkfusion_video_capture import resize_frame_to_wh


from darkfusion_video_capture import letterbox_frame_to_wh


from darkfusion_video_capture import transform_video_frame_by_settings


from darkfusion_inference import REID_TRACKER_MODEL
from darkfusion_inference import ULTRALYTICS_TRACKER_TYPES

from darkfusion_inference import TRACKER_PRESETS


from darkfusion_inference import tracker_preset_key


from darkfusion_dataset_workers import FrameExtractionThread


from darkfusion_augmentations import ImageProcessingThread



from darkfusion_tiling import TileWorker


# Class dropdown delegate with checkbox toggles and class-color swatches.

from darkfusion_dataset_split import DatasetSplitDialog

# Video URL resolver worker. Runs off the UI thread to prevent freezing during yt-dlp resolve.

from darkfusion_video_links import VideoUrlResolverWorker

# Frame extraction URL resolver worker.

from darkfusion_video_links import FrameExtractorUrlResolverWorker

# Grounding DINO worker. Runs off the UI thread so labeling stays responsive.

class QLabelInfoLogHandler(logging.Handler):

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.last_message = ""

    def emit(self, record):
        try:
            msg = self.format(record)
            self.last_message = msg
        except Exception as e:
            logger.debug(f"Log handler error: {e}")


from darkfusion_system_monitor import DARKFUSION_DIALOG_QSS, SystemMonitorDialog


class UiLoader:

    def __init__(self, main_window):
        self.main_window = main_window

    def setup_ui(self, show_images=True):
        table = self.main_window.preview_list

        try:
            table.setUpdatesEnabled(False)
            table.clearContents()
            table.setRowCount(0)
            table.setColumnCount(5)
            table.setHorizontalHeaderLabels([
                "Crop",
                "Details",
                "",
                "",
                "",
            ])

            table.setColumnHidden(0, not show_images)
            table.setColumnHidden(2, True)
            table.setColumnHidden(3, True)
            table.setColumnHidden(4, True)

        finally:
            table.setUpdatesEnabled(True)
            table.viewport().update()

        logger.info("Preview table UI setup completed.")




class MainWindow(QtWidgets.QMainWindow, Ui_mainWindow):
    MAX_SIZE=200
    keyboard_interrupt_signal = pyqtSignal()
    progress_signal = pyqtSignal(int)
    clear_dropdown_signal = pyqtSignal()
    add_item_signal = pyqtSignal(str)
    rightClicked = pyqtSignal(QModelIndex)
    leftClicked = pyqtSignal(QModelIndex)
    signalThumbnailCreated = QtCore.pyqtSignal(int, QtGui.QPixmap)
    status_message_requested = pyqtSignal(str, int)
    voice_class_recognized = pyqtSignal(str)
    voice_mode_failed = pyqtSignal(str)
    sam3_startup_warmup_finished = pyqtSignal()

    @pyqtSlot(str, int)
    def _show_status_message(self, message, timeout):
        """Only this GUI-thread slot may create/update the status bar."""
        self.statusBar().showMessage(message, timeout)

    def post_status_message(self, message, timeout=3000):
        if QThread.currentThread() == self.thread():
            self._show_status_message(str(message), int(timeout))
        else:
            self.status_message_requested.emit(str(message), int(timeout))

    def _ensure_language_dropdown(self):
        dropdown = getattr(self, "language_dropdown", None)
        if isinstance(dropdown, QtWidgets.QComboBox):
            return dropdown

        dropdown = self.findChild(QtWidgets.QComboBox, "language_dropdown")
        if dropdown is None:
            logger.warning("Missing language_dropdown UI widget; creating fallback ComboBox.")
            dropdown = QtWidgets.QComboBox(self)
            dropdown.setObjectName("language_dropdown")
            dropdown.hide()

        self.language_dropdown = dropdown
        return dropdown

    def _theme_manifest_for_style(self, selected_style=None):
        """Resolve a selected QSS name to its colocated artwork manifest."""
        combo = getattr(self, "styleComboBox", None)
        candidate_style = selected_style
        if not candidate_style and combo is not None:
            candidate_style = combo.currentText()
        style_name = str(candidate_style or "").strip()
        if not style_name:
            return None
        assets_root = Path(APP_DIR) / "styles" / "df-assets"
        if not assets_root.is_dir():
            return None
        for manifest_path in assets_root.glob("*/theme.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if str(manifest.get("styleName", "")).strip() == style_name:
                manifest["_directory"] = str(manifest_path.parent)
                return manifest
        return None

    def _apply_theme_header_art(self, manifest=None):
        """Apply the original theme package's top-dock and idle-canvas artwork."""
        from darkfusion_theme_assets import apply_theme_assets

        style_name = self.styleComboBox.currentText()
        apply_theme_assets(self, style_name, Path(APP_DIR) / "styles")
        QtCore.QTimer.singleShot(0, self._reflow_status_label_tools)

    def recommended_ui_scale_percent(self, screen=None):
        """Return an extra app scale based on Qt's available logical pixels."""
        if screen is None:
            try:
                screen = self.screen()
            except Exception:
                screen = None
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return 100

        available = screen.availableGeometry()
        width = max(1, int(available.width()))
        height = max(1, int(available.height()))
        if width < 1050 or height < 650:
            return 75
        if width < 1200 or height < 740:
            return 85
        if width < 1450 or height < 850:
            return 90
        if width < 1650 or height < 950:
            return 95
        return 100

    def effective_ui_scale_percent(self):
        settings = getattr(self, "settings", {}) or {}
        if bool(settings.get("autoScaleUi", True)):
            return self.recommended_ui_scale_percent()
        try:
            percent = int(settings.get("uiScalePercent", 100) or 100)
        except (TypeError, ValueError):
            percent = 100
        return max(UI_SCALE_MIN_PERCENT, min(UI_SCALE_MAX_PERCENT, percent))

    def _stylesheet_with_ui_scale(self, stylesheet):
        base = stylesheet or ""
        if "DarkFusion effective UI scale:" in base:
            return base
        factor = self.effective_ui_scale_percent() / 100.0

        def scale_font_size(match):
            value = float(match.group(1))
            unit = match.group(2)
            scaled = max(7.0, value * factor)
            clean = f"{scaled:.2f}".rstrip("0").rstrip(".")
            return f"font-size: {clean}{unit}"

        base = re.sub(
            r"font-size\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(pt|px)",
            scale_font_size,
            base,
            flags=re.IGNORECASE,
        )
        return f"/* DarkFusion effective UI scale: {int(round(factor * 100))}% */\n{base}"

    def _style_with_global_line_spacing(self, stylesheet):
        base = self._stylesheet_with_ui_scale(stylesheet)
        base = re.sub(r"/\* DarkFusion dock chrome start \*/.*?/\* DarkFusion dock chrome end \*/",
                      "", base, flags=re.DOTALL).rstrip()
        prefix_additions = []
        if GLOBAL_LINE_SPACING_QSS.strip() not in base:
            prefix_additions.append(GLOBAL_LINE_SPACING_QSS)

        parts = []
        if prefix_additions:
            parts.append("\n\n".join(prefix_additions).rstrip())
        if base.strip():
            parts.append(base.rstrip())

        parts.append(dock_chrome_stylesheet(base, self._theme_manifest_for_style()).strip())

        # This must remain last so incomplete theme-specific indicator rules
        # cannot make unchecked boxes transparent or borderless.
        if GLOBAL_CHECKBOX_OUTLINE_QSS.strip() not in base:
            parts.append(GLOBAL_CHECKBOX_OUTLINE_QSS.strip())
        return "\n\n".join(parts)

    def apply_ui_scale(self, persist=False):
        percent = self.effective_ui_scale_percent()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            base_size = app.property("darkfusionBaseFontPointSize")
            if not base_size:
                base_size = app.font().pointSizeF()
                if base_size <= 0:
                    base_size = 9.0
                app.setProperty("darkfusionBaseFontPointSize", float(base_size))
            font = QtGui.QFont(app.font())
            font.setPointSizeF(max(7.0, float(base_size) * percent / 100.0))
            app.setFont(font)

        if hasattr(self, "styleComboBox"):
            self.apply_stylesheet()
        self.setProperty("effectiveUiScalePercent", percent)
        for widget in self.findChildren(QtWidgets.QWidget):
            widget.updateGeometry()
        if self.layout() is not None:
            self.layout().invalidate()
            self.layout().activate()
        self._fit_window_to_available_screen(center=False)
        QtCore.QTimer.singleShot(0, self._normalize_default_dock_sizes)
        if persist:
            self.saveSettings()

    def _connect_ui_scale_screen_tracking(self):
        handle = self.windowHandle()
        if handle is None or getattr(self, "_ui_scale_screen_tracking_ready", False):
            return
        handle.screenChanged.connect(self._on_ui_scale_screen_changed)
        self._ui_scale_screen_tracking_ready = True

    def _on_ui_scale_screen_changed(self, _screen):
        if bool(getattr(self, "settings", {}).get("autoScaleUi", True)):
            self.apply_ui_scale(persist=False)
        self._fit_window_to_available_screen(center=False)
        QtCore.QTimer.singleShot(0, self._normalize_default_dock_sizes)

    def _apply_global_line_spacing(self, root=None):
        root = root or self

        for layout in root.findChildren(QtWidgets.QLayout):
            if layout.property("preserveCompactSpacing"):
                continue
            if layout.spacing() < UI_LAYOUT_SPACING:
                layout.setSpacing(UI_LAYOUT_SPACING)
            try:
                margins = layout.contentsMargins()
                if min(margins.left(), margins.top(), margins.right(), margins.bottom()) < 4:
                    layout.setContentsMargins(
                        max(4, margins.left()),
                        max(4, margins.top()),
                        max(4, margins.right()),
                        max(4, margins.bottom()),
                    )
            except Exception:
                pass

        for editor_class in (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit):
            for editor in root.findChildren(editor_class):
                document = editor.document()
                cursor = QTextCursor(document)
                cursor.select(QTextCursor.Document)
                block_format = QTextBlockFormat()
                block_format.setLineHeight(UI_LINE_HEIGHT_PERCENT, QTextBlockFormat.ProportionalHeight)
                cursor.mergeBlockFormat(block_format)

        for label in root.findChildren(QtWidgets.QLabel):
            if label.margin() < 1:
                label.setMargin(1)
            if label.property("preserveWordWrap"):
                label.setWordWrap(True)
                label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
                continue
            label.setWordWrap(False)
            label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

    def _clean_restored_ui(self):
        """Restore review controls that belong in the active workflow."""
        widget = self.findChild(QtWidgets.QWidget, "frame_10")
        if widget is not None:
            widget.show()

    def _set_widget_copy(self, object_name, text=None, tooltip=None):
        widget = getattr(self, object_name, None)
        if widget is None:
            return

        if text is not None and hasattr(widget, "setText"):
            widget.setText(text)
        if tooltip:
            widget.setToolTip(tooltip)

    def _polish_loaded_ui(self):
        """Tighten generated UI text, spacing, and related control grouping."""
        self._clear_generated_inline_styles()

        labels = {
            "label_28": ("Confidence", "Minimum detection confidence."),
            "label_61": ("Network W", "Displayed as width. Ultralytics receives imgsz as [H, W]."),
            "label_60": ("Network H", "Displayed as height. Ultralytics receives imgsz as [H, W]."),
            "label_219": ("Batch", "Images processed per batch."),
            "label_42": ("NMS", "Duplicate-box suppression threshold."),
            "label_218": ("Precision", "Floating point precision mode."),
            "label_53": ("Min Box", "Smallest label size allowed."),
            "label_49": ("Auto Delay", "Delay while Auto Scan is active."),
            "label_7": ("Font Size", "Annotation label text size."),
            "label_54": ("Max Label %", "Largest label size as a percent of the image."),
            "label_40": ("Vertical", "Vertical offset."),
            "label_55": ("Format", "Export format."),
            "label_50": ("PT Model", "Model file used for export."),
            "label_73": ("Img Height", "Export image height."),
            "label_75": ("Img Width", "Export image width."),
        }

        for object_name, (text, tooltip) in labels.items():
            self._set_widget_copy(object_name, text, tooltip)

        button_copy = {
            "save_list_button": ("Save", "Save the current class or keypoint list."),
            "remove_input": ("Remove", "Remove the selected entry."),
            "filter_button": ("Run", "Run the selected label removal filter."),
            "clear_all_button": ("Clear Labels", "Remove boxes for the selected filter from the dataset."),
            "move_all_button": (
                "Move Filtered",
                "Blank-filter images and matching .txt files are moved into Blanks. "
                "Other filters export copies with class-specific labels. Grid View checkmarks limit the action."
            ),
            "model_input": ("Model", "Choose a model file."),
            "data_input": ("YAML", "Choose the dataset YAML."),
            "save_runs_directory": ("Save To", "Choose where training runs are saved."),
            "ultralytics": ("Train", "Start training."),
            "convertButton": ("Convert", "Export the selected model format."),
            "Load_pt_model": ("Load Model", "Choose the PT model to export."),
            "location_button": ("Save To", "Choose an output folder only. Extraction starts from Extract."),
            "extract_button": ("Extract", "Start frame extraction."),
            "stop_extract": ("Stop", "Stop extraction."),
            "download_video": ("Download", "Download the queued video."),
            "remove_video": ("Remove", "Remove the selected queued video."),
            "output_button": ("Save To", "Choose a health report output folder."),
            "scan_button": ("Open Dataset Analysis", "Open health checks, duplicate cleanup, statistics, and plots in one place."),
            "deduplicate_labels_button": ("Remove Duplicate Boxes", "Clean duplicate labels, then open Dataset Analysis to scan again."),
            "import_button": ("Images", "Choose the image folder."),
            "start_move_btn": ("Move Files", "Move files from source to target."),
            "convert_image": ("Convert Dataset", "Convert images from the current dataset."),
            "crop_img_button": ("Tile Dataset", "Tile the currently opened dataset images."),
            "split_data": ("Split", "Split the dataset."),
            "review_button": ("Pause Preview", "Pause or resume the annotation preview table."),
            "process_folder_btn": ("Run SAM3 Tools", "Run the selected SAM3 tools."),
            "adjust_label": ("Adjust", "Apply the label offset."),
            "btn_save_file": ("Save", "Save the edited CFG."),
        }

        for object_name, (text, tooltip) in button_copy.items():
            self._set_widget_copy(object_name, text, tooltip)

        titles = {
            "groupBox_15": "Auto Label",
            "groupBox_16": "Display",
            "groupBox_14": "Extra Label",
            "groupBox_7": "Remove Labels",
            "groupBox_10": "Training",
            "groupBox_8": "Export",
            "groupBox_9": "Train Params",
            "groupBox_12": "Run Options",
            "groupBox_11": "Train Files",
            "groupBox_20": "Capture / Display",
            "groupBox_19": "Video Playback",
            "groupBox_4": "Dataset Analysis",
            "groupBox": "Tile Images",
            "groupBox_17": "SAM3 Tools",
            "groupBox_6": "Adjust Labels",
            "groupBox_18": "CFG Editor",
            "groupBox_13": "Anchors",
        }

        for object_name, title in titles.items():
            widget = getattr(self, object_name, None)
            if widget is not None and hasattr(widget, "setTitle"):
                widget.setTitle(title)

        dock_titles = {
            "dockWidget_3": "Status",
            "menu_dock": "Workflow",
        }

        for object_name, title in dock_titles.items():
            widget = getattr(self, object_name, None)
            if isinstance(widget, QtWidgets.QDockWidget):
                widget.setWindowTitle(title)

        if hasattr(self, "filter_class_spinbox"):
            self.filter_class_spinbox.setToolTip(
                "Filter review by all images, blanks, or a specific class."
            )
        if hasattr(self, "img_index_search"):
            self.img_index_search.setPlaceholderText("Find image...")
            self.img_index_search.setToolTip("Search the current image list by file name.")
        if hasattr(self, "console_output"):
            self.console_output.setToolTip(
                "System resource summary. Open System Monitor for resource usage and GPU power."
            )

        self._group_main_review_controls()
        self._simplify_review_filter_toolbar()
        self._modernize_legacy_detection_panel()
        self._centralize_dataset_analysis_ui()
        self._compact_health_check_panel()
        self._compact_display_panel()
        self._polish_status_dock()
        self._polish_laptop_layouts()
        self._compact_keypoint_panel()
        self._compact_roi_extras_panel()
        self._compact_prepare_split_panels()
        self._compact_augment_convert_panel()
        self._compact_train_export_panels()
        self._compact_advanced_panels()
        self._consolidate_label_workflow()
        for page_name in ("tab_20", "tab_21", "tab_2"):
            page = getattr(self, page_name, None)
            if page is not None and page.layout() is not None:
                page.layout().setContentsMargins(8, 8, 8, 8)
                page.layout().setSpacing(10)
                page.layout().setAlignment(Qt.AlignTop)
        self.setStyleSheet(self._style_with_global_line_spacing(self.styleSheet()))

    def setup_dock_context_help(self):
        self._dock_help_controllers = []
        self.register_dock_context_help(
            getattr(self, "dockWidget_3", None),
            "Status dock",
            (
                "Shows current operation progress and the controls used while reviewing "
                "images: annotation mode, active class, navigation, Auto Scan, ROI, "
                "and shared Label Tools. Appearance and language are in Settings."
            ),
            "2-understand-the-workflow",
        )
        self.register_dock_context_help(
            getattr(self, "menu_dock", None),
            "Workflow dock",
            (
                "Contains the main workflow tabs for collecting data, review, dataset "
                "preparation, training/export, and advanced tools. Label Tools remain in the top dock. Resize or float "
                "this dock when a panel needs more horizontal space."
            ),
            "2-understand-the-workflow",
        )

    def register_dock_context_help(self, dock, help_title, summary, guide_section):
        if not isinstance(dock, QtWidgets.QDockWidget):
            return
        body = dock.widget()
        if body is not None and not body.property("darkfusionDockBody"):
            # Native floating docks do not paint the QDockWidget box. Frame the
            # existing body instead; never wrap/reparent the controls or viewer.
            body.setProperty("darkfusionDockBody", True)
            body.style().unpolish(body)
            body.style().polish(body)
            body.updateGeometry()
            body.update()
        if bool(dock.property("darkfusionContextHelpRegistered")):
            return

        dock.setProperty("darkfusionContextHelpRegistered", True)
        controller = DockHelpButtonController(
            dock,
            lambda _checked=False, t=help_title, s=summary, g=guide_section:
                self.show_dock_context_help(t, s, g),
            self,
        )
        self._dock_help_controllers.append(controller)

        view_menu = getattr(self, "menufunctions", None)
        if isinstance(view_menu, QtWidgets.QMenu):
            toggle_action = dock.toggleViewAction()
            toggle_action.setText(f"Show {dock.windowTitle()} Dock")
            if toggle_action not in view_menu.actions():
                view_menu.addAction(toggle_action)

    def show_dock_context_help(self, title, summary, guide_section=""):
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Information)
        dialog.setWindowTitle(title)
        dialog.setText(summary)
        dialog.setInformativeText(
            "Use the ? button for this explanation. The neighboring dock buttons "
            "float/restore the panel and close it. A closed dock can be restored from View."
        )
        guide_button = dialog.addButton("Open User Guide", QMessageBox.ActionRole)
        dialog.addButton(QMessageBox.Close)
        dialog.exec_()
        if dialog.clickedButton() is guide_button:
            self.open_user_guide(guide_section)

    def open_user_guide(self, guide_section=""):
        guide_path = Path(APP_DIR).parent / "USER_GUIDE.md"
        if guide_path.exists():
            QtGui.QDesktopServices.openUrl(QUrl.fromLocalFile(str(guide_path)))
            return
        section = f"#{guide_section}" if guide_section else ""
        QtGui.QDesktopServices.openUrl(
            QUrl(f"https://github.com/lordofkillz/DarkFusion/blob/main/USER_GUIDE.md{section}")
        )

    def _polish_laptop_layouts(self):
        """Relax Designer-era fixed sizing so docks can breathe on smaller screens."""
        self.setMinimumSize(560, 360)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        dock_specs = (
            ("dockWidget_3", 280, 120),
            ("menu_dock", 300, 220),
        )
        for object_name, min_width, min_height in dock_specs:
            dock = getattr(self, object_name, None)
            if isinstance(dock, QtWidgets.QDockWidget):
                dock.setMinimumSize(min_width, min_height)
                dock.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
                dock.setFeatures(
                    QtWidgets.QDockWidget.DockWidgetClosable
                    | QtWidgets.QDockWidget.DockWidgetMovable
                    | QtWidgets.QDockWidget.DockWidgetFloatable
                )

        for scroll_name in ("scrollArea",):
            scroll = getattr(self, scroll_name, None)
            if isinstance(scroll, QtWidgets.QScrollArea):
                scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
                scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                scroll.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
                scroll.setWidgetResizable(True)
                content = scroll.widget()
                if content is not None:
                    content.setMinimumWidth(0)
                    content.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
                    if content.layout() is not None:
                        content.layout().setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)

        workflow_content = getattr(self, "scrollAreaWidgetContents", None)
        for tab_widget in self.findChildren(QtWidgets.QTabWidget):
            tab_widget.setDocumentMode(True)
            tab_widget.setElideMode(Qt.ElideRight)
            tab_widget.setUsesScrollButtons(True)
            tab_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            tab_widget.setMinimumWidth(0)
            in_workflow_scroll = (
                isinstance(workflow_content, QtWidgets.QWidget)
                and (tab_widget is workflow_content or workflow_content.isAncestorOf(tab_widget))
            )
            if in_workflow_scroll:
                tab_widget.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
                stack = tab_widget.findChild(QtWidgets.QStackedWidget)
                if stack is not None:
                    stack.setMinimumWidth(0)
                    stack.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
                for page_index in range(tab_widget.count()):
                    page = tab_widget.widget(page_index)
                    if page is not None:
                        page.setMinimumWidth(0)
                        page.setSizePolicy(QtWidgets.QSizePolicy.Ignored, page.sizePolicy().verticalPolicy())
            if tab_widget.tabBar() is not None:
                tab_widget.tabBar().setUsesScrollButtons(True)
                tab_widget.tabBar().setExpanding(False)
                tab_widget.tabBar().setMinimumWidth(0)
                if in_workflow_scroll:
                    tab_widget.tabBar().setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)

        for group in self.findChildren(QtWidgets.QGroupBox):
            layout = group.layout()
            if layout is not None:
                layout.setContentsMargins(8, 8, 8, 8)
                layout.setSpacing(6)
                # Make buttons inside group boxes more consistent and allow them to expand
                try:
                    for btn in group.findChildren(QtWidgets.QPushButton):
                        try:
                            btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
                            btn.setMinimumSize(90, 36)
                            btn.setMaximumHeight(56)
                        except Exception:
                            pass
                except Exception:
                    pass

        # Specific polish for Video Playback controls to ensure even sizing and spacing
        try:
            playback_group = getattr(self, 'groupBox_19', None)
            if isinstance(playback_group, QtWidgets.QGroupBox):
                playback_layout = playback_group.layout()
                if playback_layout is not None:
                    playback_layout.setContentsMargins(12, 12, 12, 12)
                    playback_layout.setSpacing(12)
                for btn_name in ('remove_video_button', 'video_upload', 'play_video_button', 'forward_button', 'back_button', 'extract_button', 'stop_extract'):
                    btn = getattr(self, btn_name, None)
                    if isinstance(btn, QtWidgets.QPushButton):
                        btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
                        btn.setMinimumSize(82, 34)
                        btn.setMaximumHeight(56)
                        try:
                            btn.setMaximumWidth(220)
                        except Exception:
                            pass
                self._reflow_video_playback_controls(playback_group)
                # If there are horizontal layouts containing the big buttons, increase spacing
                for h in playback_group.findChildren(QtWidgets.QHBoxLayout):
                    try:
                        h.setSpacing(8)
                    except Exception:
                        pass
        except Exception:
            pass

        self._reflow_video_extract_controls()
        self._polish_video_tables()
        self._polish_collect_tab_layout()

        for frame_name in ("frame_27", "frame_11", "frame_12"):
            frame = getattr(self, frame_name, None)
            if isinstance(frame, QtWidgets.QFrame):
                frame.setFrameShape(QtWidgets.QFrame.NoFrame)
                frame.setFrameShadow(QtWidgets.QFrame.Plain)

    from darkfusion_video_inference import _reflow_video_playback_controls

    from darkfusion_video_inference import _ensure_video_player_controls

    from darkfusion_video_inference import _ensure_resize_policy_control

    from darkfusion_video_inference import _ensure_tracker_preset_control

    from darkfusion_video_inference import _reflow_video_extract_controls

    from darkfusion_video_inference import _polish_video_tables

    from darkfusion_video_inference import _polish_collect_tab_layout

    from darkfusion_video_inference import _show_collect_controls

    from darkfusion_video_inference import _update_collect_action_footer

    def _workflow_label(self, text, parent=None, width=72):
        label = QtWidgets.QLabel(text, parent)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label.setMinimumWidth(width)
        label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        return label

    def _reset_layout(self, layout):
        if layout is None:
            return
        while layout.count():
            layout.takeAt(0)

    def _compact_control(self, widget, min_height=30):
        if not isinstance(widget, QtWidgets.QWidget):
            return
        widget.setMinimumWidth(0)
        widget.setMinimumHeight(min_height)
        # Do not cap control height. Windows text scaling and translated labels
        # can make the native size hint taller than Designer's old 30-36 px
        # limits, which caused checkbox text to paint below an empty control.
        widget.setMaximumHeight(QtWidgets.QWIDGETSIZE_MAX)
        if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QComboBox)):
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            widget.setMaximumWidth(96)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        elif isinstance(widget, QtWidgets.QPushButton):
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        elif isinstance(widget, QtWidgets.QCheckBox):
            widget.setLayoutDirection(Qt.LeftToRight)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

    def _row_layout(self, spacing=8):
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(spacing)
        return row

    def _prepare_group_grid(self, group, margins=(12, 16, 12, 12), spacing=8):
        layout = group.layout() if group is not None else None
        if not isinstance(layout, QtWidgets.QGridLayout):
            return None
        group.setMinimumWidth(0)
        group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        layout.setContentsMargins(*margins)
        layout.setHorizontalSpacing(spacing)
        layout.setVerticalSpacing(spacing)
        for index in range(layout.count() - 1, -1, -1):
            layout.takeAt(index)
        for row in range(20):
            layout.setRowStretch(row, 0)
            layout.setRowMinimumHeight(row, 0)
        for column in range(8):
            layout.setColumnStretch(column, 0)
            layout.setColumnMinimumWidth(column, 0)
        layout.setColumnStretch(0, 1)
        return layout

    from darkfusion_canvas import _compact_roi_extras_panel

    from darkfusion_augmentations import _compact_extra_label_panel

    def _compact_prepare_split_panels(self):
        self._move_tile_images_panel_to_prepare()
        self._compact_prepare_split_toolbar()
        self._compact_move_images_panel()
        self._compact_image_format_panel()
        self._compact_tile_images_panel()
        self._reorder_prepare_split_panels()

    def _move_tile_images_panel_to_prepare(self):
        if getattr(self, "_tile_images_panel_moved_to_prepare", False):
            return

        group = getattr(self, "groupBox", None)
        prepare_tab = getattr(self, "tab_20", None)
        prepare_layout = prepare_tab.layout() if prepare_tab is not None else getattr(self, "gridLayout_9", None)
        if not isinstance(group, QtWidgets.QGroupBox) or not isinstance(prepare_layout, QtWidgets.QGridLayout):
            return

        parent = group.parentWidget()
        parent_layout = parent.layout() if parent is not None else None
        self._take_layout_widget(parent_layout, group)
        group.setParent(prepare_tab)
        prepare_layout.addWidget(group, 3, 0, 1, 1)
        self._tile_images_panel_moved_to_prepare = True

    def _reorder_prepare_split_panels(self):
        layout = getattr(self, "gridLayout_9", None)
        if not isinstance(layout, QtWidgets.QGridLayout) or getattr(self, "_prepare_split_panels_reordered", False):
            return

        ordered_widgets = [
            getattr(self, "frame_4", None),
            getattr(self, "groupBox_3", None),
            getattr(self, "groupBox", None),
            getattr(self, "groupBox_7", None) if getattr(self, "groupBox_7", None) is not None and self.groupBox_7.parentWidget() is self.tab_20 else None,
            getattr(self, "groupBox_2", None),
        ]
        ordered_widgets = [widget for widget in ordered_widgets if isinstance(widget, QtWidgets.QWidget)]

        for widget in ordered_widgets:
            self._take_layout_widget(layout, widget)
            widget.setMinimumWidth(0)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        # Designer's old bottom spacer occupied row 4. Adding Label Cleanup
        # moved Move Images into that same row, so the spacer stretched a huge
        # gap above it when the other categories were collapsed. Keep unused
        # space after the whole stack through top alignment, not inside a row.
        for index in range(layout.count() - 1, -1, -1):
            if layout.itemAt(index).spacerItem() is not None:
                layout.takeAt(index)
        for row in range(layout.rowCount()):
            layout.setRowStretch(row, 0)
            layout.setRowMinimumHeight(row, 0)

        for row, widget in enumerate(ordered_widgets):
            layout.addWidget(widget, row, 0, 1, 1)
            layout.setRowStretch(row, 0)

        layout.setAlignment(Qt.AlignTop)
        layout.setColumnStretch(0, 1)
        self._prepare_split_panels_reordered = True

    def _compact_prepare_split_toolbar(self):
        if getattr(self, "_prepare_split_toolbar_compacted", False):
            return

        frame = getattr(self, "frame_4", None)
        layout = frame.layout() if frame is not None else None
        if not isinstance(layout, QtWidgets.QGridLayout):
            return
        frame.setProperty("collapsibleSectionTitle", "Create Train / Valid")

        names = ("label_4", "dropdown", "valid_percent", "import_button", "output_button")
        widgets = {name: getattr(self, name, None) for name in names}
        if any(widget is None for widget in widgets.values()):
            return

        for widget in widgets.values():
            self._take_layout_widget(layout, widget)
            self._compact_control(widget)

        self._reset_layout(layout)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        widgets["label_4"].setText("List")
        widgets["label_4"].setMinimumWidth(44)
        widgets["dropdown"].setMaximumWidth(120)
        widgets["valid_percent"].setMaximumWidth(92)

        widgets["import_button"].setText("Choose Images...")
        widgets["output_button"].setText("Create Files...")
        widgets["import_button"].setToolTip("Choose the images used to create the training and validation lists.")
        options = self._row_layout()
        options.addWidget(widgets["label_4"])
        options.addWidget(widgets["dropdown"])
        options.addWidget(self._workflow_label("Validation %", frame, 86))
        options.addWidget(widgets["valid_percent"])
        options.addStretch(1)
        actions = self._row_layout()
        actions.addWidget(widgets["import_button"], 1)
        actions.addWidget(widgets["output_button"], 1)
        layout.addLayout(options, 0, 0)
        layout.addLayout(actions, 1, 0)

        self._prepare_split_toolbar_compacted = True

    from darkfusion_dataset_workers import _compact_move_images_panel

    from darkfusion_dataset_workers import _compact_image_format_panel

    from darkfusion_tiling import (
        ensure_tile_images_controls,
        _compact_tile_images_panel,
    )

    from darkfusion_training_dialog import _compact_train_export_panels

    from darkfusion_training_dialog import _setup_ultralytics_tab_launcher

    from darkfusion_training_dialog import _compact_ultralytics_train_panel

    from darkfusion_training_dialog import _compact_ultralytics_export_panel

    from darkfusion_training_dialog import _compact_train_params_panel

    from darkfusion_darknet import _compact_darknet_options_panel

    from darkfusion_darknet import _compact_darknet_train_panel

    from darkfusion_darknet import _compact_advanced_panels

    from darkfusion_darknet import _group_cfg_editor_contents

    from darkfusion_darknet import _sync_cfg_editor_visibility

    from darkfusion_darknet import _compact_cfg_editor_panel

    from darkfusion_darknet import _compact_anchor_panel

    def _polish_status_dock(self):
        if getattr(self, "_status_dock_polished", False):
            return

        dock = getattr(self, "dockWidget_3", None)
        contents = getattr(self, "dockWidgetContents_3", None)
        layout = getattr(self, "gridLayout_16", None)
        if dock is None or contents is None or layout is None:
            return

        widgets = {
            "button": getattr(self, "cuda_monitor_button", None),
            "console": getattr(self, "console_output", None),
            "preview": getattr(self, "image_label_2", None),
            "labeling": getattr(self, "frame_27", None),
            "progress": getattr(self, "label_progress", None),
        }
        if any(widgets[key] is None for key in ("button", "preview", "labeling", "progress")):
            return

        for widget in widgets.values():
            if widget is not None:
                self._take_layout_widget(layout, widget)
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if item is not None and item.widget() is None:
                layout.takeAt(i)

        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        shell = QtWidgets.QFrame(contents)
        shell.setObjectName("status_shell_panel")
        shell.setFrameShape(QtWidgets.QFrame.NoFrame)
        shell.setMinimumHeight(0)
        shell.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        shell_layout = QtWidgets.QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(6)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(10)

        button = widgets["button"]
        button.setMinimumHeight(28)
        button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        progress = widgets["progress"]
        progress.setMinimumHeight(12)
        progress.setMaximumHeight(16)
        progress.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        preview = widgets["preview"]
        preview.setObjectName("image_label_2")
        preview.setFrameShape(QtWidgets.QFrame.NoFrame)
        preview.setLineWidth(0)
        preview.setMidLineWidth(0)
        preview.setAlignment(Qt.AlignCenter)
        preview.setScaledContents(False)
        preview.setMinimumSize(96, 64)
        preview.setMaximumSize(120, 64)
        preview.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        preview.installEventFilter(self)

        top_row.addWidget(button)
        top_row.addWidget(progress, 1)
        console = widgets["console"]
        if console is not None:
            console.clear()
            console.hide()

        shell_layout.addLayout(top_row)

        labeling = widgets["labeling"]
        labeling.setObjectName("frame_27")
        labeling.setFrameShape(QtWidgets.QFrame.NoFrame)
        labeling.setFrameShadow(QtWidgets.QFrame.Plain)
        labeling.setMinimumHeight(0)
        labeling.setMaximumHeight(QtWidgets.QWIDGETSIZE_MAX)
        labeling.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        labeling_layout = labeling.layout()
        if labeling_layout is not None:
            labeling_layout.setContentsMargins(0, 0, 0, 0)
            labeling_layout.setSpacing(0)
        self._rebuild_status_labeling_panel(labeling)
        shell_layout.addWidget(labeling)
        # Themes may style every QFrame. Keep only the dock/category boundaries,
        # not a stack of decorative rectangles around the same controls.
        for frame in (shell, labeling, self.status_labeling_panel, self.status_labeling_controls):
            frame.setProperty("workflowChrome", True)
            frame.setStyleSheet('QFrame[workflowChrome="true"] { border: none; background: transparent; }')

        layout.addWidget(shell, 0, 0, 1, 2)
        # Let the contents determine the dock height, including an expanded ROI
        # selector or a second Label Tools row. Old saved dock heights must not
        # stretch the toolbar or force a fixed pixel cap under larger UI fonts.
        layout.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        for row in range(layout.rowCount()):
            layout.setRowStretch(row, 0)

        dock.setMinimumSize(320, 0)
        dock.setMaximumHeight(QtWidgets.QWIDGETSIZE_MAX)
        contents.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._status_dock_polished = True

    def _rebuild_status_labeling_panel(self, labeling):
        if getattr(self, "_status_labeling_panel_built", False):
            return

        layout = labeling.layout()
        if layout is None:
            return

        control_names = (
                "classes_dropdown",
                "segmentation_mode",
                "auto_scan_checkbox",
                "previous_button",
                "img_index_number",
                "total_images",
                "next_button",
                "delete_button",
                "grid_view_button",
                "styleComboBox",
                "gif_change",
                "language_dropdown",
            )
        controls = [getattr(self, name, None) for name in control_names]
        if any(widget is None for widget in controls):
            return

        for widget in controls:
            self._take_layout_widget(layout, widget)

        (
            classes_dropdown,
            segmentation_mode,
            auto_scan_checkbox,
            previous_button,
            img_index_number,
            total_images,
            next_button,
            delete_button,
            grid_view_button,
            style_combo,
            gif_combo,
            language_dropdown,
        ) = controls

        voice_btn = getattr(self, "voice_btn", None)
        if isinstance(voice_btn, QtWidgets.QPushButton):
            voice_parent = voice_btn.parentWidget()
            voice_parent_layout = voice_parent.layout() if voice_parent is not None else None
            self._take_layout_widget(voice_parent_layout, voice_btn)
            voice_btn.setParent(None)
            voice_btn.deleteLater()
            self.voice_btn = None

        voice_mode_checkbox = getattr(self, "voice_mode_checkbox", None)
        if isinstance(voice_mode_checkbox, QtWidgets.QCheckBox):
            voice_parent = voice_mode_checkbox.parentWidget()
            voice_parent_layout = voice_parent.layout() if voice_parent is not None else None
            self._take_layout_widget(voice_parent_layout, voice_mode_checkbox)
            voice_mode_checkbox.setParent(None)

        panel = QtWidgets.QFrame(labeling)
        panel.setObjectName("status_labeling_panel")
        panel.setFrameShape(QtWidgets.QFrame.NoFrame)
        panel.setMinimumHeight(0)
        panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.status_labeling_panel = panel

        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)

        top_line = QtWidgets.QHBoxLayout()
        top_line.setContentsMargins(0, 0, 0, 0)
        top_line.setSpacing(6)

        bottom_line = QtWidgets.QHBoxLayout()
        bottom_line.setObjectName("status_navigation_row")
        bottom_line.setContentsMargins(0, 0, 0, 0)
        bottom_line.setSpacing(10)

        classes_dropdown.setMinimumWidth(140)
        classes_dropdown.setMaximumWidth(520)
        classes_dropdown.setMinimumHeight(28)
        classes_dropdown.setMaximumHeight(30)
        classes_dropdown.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        fixed_controls = (
            (segmentation_mode, 100),
            (auto_scan_checkbox, 86),
            (style_combo, 112),
            (gif_combo, 112),
            (language_dropdown, 96),
        )
        for widget, width in fixed_controls:
            widget.setMinimumHeight(28)
            widget.setMaximumHeight(30)
            widget.setFixedWidth(width)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        for button in (previous_button, next_button, delete_button, grid_view_button):
            button.setMinimumHeight(28)
            button.setMaximumHeight(30)
            button.setMinimumWidth(58)
            button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        img_index_number.setMinimumHeight(28)
        img_index_number.setMaximumHeight(30)
        img_index_number.setFixedWidth(76)
        img_index_number.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        total_images.setMinimumHeight(28)
        total_images.setMaximumHeight(30)
        total_images.setMinimumWidth(68)
        total_images.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        top_line.addWidget(classes_dropdown, 1)
        top_line.addWidget(segmentation_mode)
        top_line.addStretch(1)

        bottom_line.addWidget(previous_button, 0, Qt.AlignTop)
        bottom_line.addWidget(img_index_number, 0, Qt.AlignTop)
        bottom_line.addWidget(total_images, 0, Qt.AlignTop)
        bottom_line.addWidget(next_button, 0, Qt.AlignTop)
        bottom_line.addWidget(delete_button, 0, Qt.AlignTop)
        bottom_line.addWidget(grid_view_button, 0, Qt.AlignTop)
        bottom_line.addWidget(auto_scan_checkbox, 0, Qt.AlignTop)

        roi_tools = QtWidgets.QFrame(panel)
        roi_tools.setObjectName("status_roi_tools")
        roi_tools.setFrameShape(QtWidgets.QFrame.NoFrame)
        roi_tools.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        roi_tools.setProperty("workflowChrome", True)
        roi_tools.setStyleSheet('QFrame[workflowChrome="true"] { border: none; background: transparent; }')
        roi_layout = QtWidgets.QVBoxLayout(roi_tools)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(6)
        enable_roi_button = QtWidgets.QPushButton("Enable ROI", roi_tools)
        enable_roi_button.setObjectName("enable_roi_button")
        enable_roi_button.setCheckable(True)
        enable_roi_button.setToolTip(
            "Turn ROI filtering on/off. Starts with ROI 1; select ROI 2 below to add the second region."
        )
        enable_roi_button.setStyleSheet(
            "QPushButton#enable_roi_button:checked { background-color: #237a3b; color: white; "
            "border: 1px solid #6fcb85; font-weight: 600; }"
            "QPushButton#enable_roi_button:checked:hover { background-color: #2d9149; }"
        )
        enable_roi_button.setMinimumHeight(28)
        enable_roi_button.setMaximumHeight(30)
        enable_roi_button.setMinimumWidth(126)
        enable_roi_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        enable_roi_button.toggled.connect(self._toggle_roi_master)
        self.enable_roi_button = enable_roi_button
        roi_layout.addWidget(enable_roi_button)
        self.roi_selector = QtWidgets.QComboBox(roi_tools)
        self.roi_selector.setObjectName("roi_selector")
        self.roi_selector.addItem("ROI 1", 1)
        self.roi_selector.addItem("ROI 2", 2)
        self.roi_selector.setMinimumHeight(28)
        self.roi_selector.setAccessibleName("ROI to adjust")
        self.roi_selector.setToolTip(
            "Choose the ROI to adjust with the view-edge sliders. Selecting ROI 2 adds it; ROI 1 stays enabled."
        )
        self.roi_selector.currentIndexChanged.connect(self._on_roi_selected)
        self.roi_selector.hide()
        roi_layout.addWidget(self.roi_selector)
        self.roi_clarity_checkbox = QtWidgets.QCheckBox("Clarity in ROI", roi_tools)
        self.roi_clarity_checkbox.setObjectName("roi_clarity_enabled")
        self.roi_clarity_checkbox.setChecked(bool(getattr(self, "settings", {}).get("objectClarityUseROI", False)))
        self.roi_clarity_checkbox.setToolTip(
            "Enhance enabled ROIs in the annotation view using the existing ROI sliders. "
            "The border turns cyan and shows Clarity active when ready. "
            "This checkbox leaves inference settings unchanged."
        )
        self.roi_clarity_checkbox.toggled.connect(self._roi_clarity_changed)
        roi_layout.addWidget(self.roi_clarity_checkbox)
        bottom_line.addWidget(roi_tools, 0, Qt.AlignTop)
        bottom_line.addStretch(1)

        for preference_combo in (style_combo, gif_combo, language_dropdown):
            preference_combo.hide()

        controls_panel = QtWidgets.QFrame(panel)
        controls_panel.setObjectName("status_labeling_controls")
        controls_panel.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls_panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        controls_panel.setMinimumWidth(0)
        self.status_labeling_controls = controls_panel

        controls_column = QtWidgets.QVBoxLayout(controls_panel)
        controls_column.setContentsMargins(0, 0, 0, 0)
        controls_column.setSpacing(6)
        controls_column.addLayout(top_line)
        controls_column.addLayout(bottom_line)

        body_line = QtWidgets.QHBoxLayout()
        body_line.setContentsMargins(0, 0, 0, 0)
        body_line.setSpacing(10)
        body_line.addWidget(controls_panel, 1, Qt.AlignTop)

        preview = getattr(self, "image_label_2", None)
        if preview is not None:
            body_line.addWidget(preview, 0, Qt.AlignRight | Qt.AlignVCenter)

        panel_layout.addLayout(body_line)

        # Reuse the original propagation/clear buttons, but remove the sparse
        # Designer grid rows that used to separate them from navigation.
        action_row = self._row_layout(6)
        action_row.setObjectName("status_annotation_actions")
        for name in ("clear_current_frame_labels_button", "propagate_labels_button",
                     "propagation_overwrite_checkbox", "propagate_range_button"):
            widget = getattr(self, name, None)
            if widget is not None:
                self._take_layout_widget(layout, widget)
                action_row.addWidget(widget)
        action_row.addStretch(1)
        panel_layout.addLayout(action_row)
        while layout.count():
            layout.takeAt(0)
        if isinstance(layout, QtWidgets.QGridLayout):
            for row in range(layout.rowCount()):
                layout.setRowStretch(row, 0)
                layout.setRowMinimumHeight(row, 0)

        if isinstance(layout, QtWidgets.QGridLayout):
            layout.addWidget(panel, 0, 0, 1, 12)
        else:
            layout.addWidget(panel)

        self._status_labeling_panel_built = True

    def _scale_status_gif(self):
        label = getattr(self, "image_label_2", None)
        movie = getattr(self, "movie", None)
        if label is None or movie is None:
            return

        size = label.size()
        if size.width() <= 0 or size.height() <= 0:
            size = QtCore.QSize(180, 120)

        source_size = movie.frameRect().size()
        if source_size.width() > 0 and source_size.height() > 0:
            size = source_size.scaled(size, Qt.KeepAspectRatio)
        movie.setScaledSize(size)

    from darkfusion_dataset_analysis import _centralize_dataset_analysis_ui

    from darkfusion_dataset_analysis import _compact_health_check_panel

    from darkfusion_dataset_workers import get_duplicate_iou_threshold

    from darkfusion_dataset_review import _simplify_review_filter_toolbar

    from darkfusion_dataset_review import _update_review_action_scope

    from darkfusion_autolabel import _modernize_legacy_detection_panel

    from darkfusion_ui_controls import _format_slider_value_label

    from darkfusion_ui_controls import update_slider_value_label

    from darkfusion_ui_controls import update_slider_value_labels

    from darkfusion_canvas import ensure_enhance_image_checkbox

    from darkfusion_canvas import ensure_visibility_preprocessing_button

    from darkfusion_canvas import (
        _object_clarity,
    )

    from darkfusion_canvas import _visibility_control_specs

    from darkfusion_canvas import _set_visibility_setting

    from darkfusion_canvas import show_visibility_preprocessing_dialog

    from darkfusion_canvas import _compact_display_panel

    def _consolidate_label_workflow(self):
        """Move shared live controls, then remove only the empty workflow page.

        Keep the hidden Designer backing widgets alive: settings, keypoint
        dialogs and existing inference paths still share those native values.
        """
        if getattr(self, "_label_workflow_consolidated", False):
            return
        shell = self.findChild(QtWidgets.QFrame, "status_shell_panel")
        old_panel = self.findChild(QtWidgets.QFrame, "display_compact_panel")
        prepare = getattr(self, "tab_20", None)
        cleanup = getattr(self, "groupBox_7", None)
        page = getattr(self, "tab_3", None)
        tabs = getattr(self, "tabWidget", None)
        if (shell is None or old_panel is None or prepare is None or cleanup is None
                or page is None or tabs is None or not getattr(self, "_extra_label_dialog_ready", False)):
            return
        labels = getattr(self, "_slider_value_labels", {})
        if not all(name in labels for name in ("shade_slider", "timmer_speed")):
            return

        panel = QtWidgets.QFrame(shell)
        panel.setObjectName("status_label_tools")
        panel.setFrameShape(QtWidgets.QFrame.NoFrame)
        panel.setProperty("workflowChrome", True)
        panel.setStyleSheet('QFrame[workflowChrome="true"] { border: none; background: transparent; }')
        panel.setMinimumWidth(0)
        panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        grid = QtWidgets.QGridLayout(panel)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        title = QtWidgets.QLabel("Label Tools", panel)
        title.setObjectName("label_tools_title")
        font = title.font()
        font.setBold(True)
        title.setFont(font)

        def move(widget):
            self._take_layout_widget(old_panel.layout(), widget)
            widget.setParent(panel)
            widget.show()
            return widget

        def slider_pair(label_name, slider_name, caption):
            pair = QtWidgets.QWidget(panel)
            pair.setObjectName("status_" + slider_name + "_controls")
            row = self._row_layout(6)
            pair.setLayout(row)
            label = move(getattr(self, label_name))
            label.setText(caption)
            label.setMinimumWidth(0)
            label.setMaximumWidth(QtWidgets.QWIDGETSIZE_MAX)
            label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            slider = move(getattr(self, slider_name))
            slider.setMinimumWidth(60)
            slider.setMaximumWidth(QtWidgets.QWIDGETSIZE_MAX)
            value = move(labels[slider_name])
            row.addWidget(label)
            row.addWidget(slider, 1)
            row.addWidget(value)
            return pair

        opacity = slider_pair("shade", "shade_slider", "Opacity")
        delay = slider_pair("label_49", "timmer_speed", "Scan delay")
        toggles = [move(getattr(self, name)) for name in ("rapid_del_checkbox", "zoom_lock", "outline_Checkbox")]
        for widget in toggles:
            widget.setMinimumWidth(0)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        visibility = move(self.visibility_preprocessing_button)
        visibility.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.status_label_tools = panel
        self._status_label_tool_widgets = (title, opacity, *toggles, visibility, delay)
        shell.layout().addWidget(panel)
        panel.installEventFilter(self)
        self._reflow_status_label_tools()
        old_panel.hide()

        parent = cleanup.parentWidget()
        self._take_layout_widget(parent.layout() if parent is not None else None, cleanup)
        cleanup.setParent(prepare)
        cleanup.setTitle("Label Cleanup")
        prepare.layout().addWidget(cleanup, prepare.layout().rowCount(), 0)
        cleanup.show()
        self._prepare_split_panels_reordered = False
        self._reorder_prepare_split_panels()

        # removeTab does not delete the backing widgets or change either viewer.
        index = tabs.indexOf(page)
        if index >= 0:
            tabs.removeTab(index)
        page.hide()
        self._label_workflow_consolidated = True
        # Do not let the global document/layout spacing pass inflate each nested
        # toolbar row again on startup, theme changes or UI-scale changes.
        for layout in shell.findChildren(QtWidgets.QLayout):
            layout.setProperty("preserveCompactSpacing", True)

    def _reflow_status_label_tools(self):
        panel = getattr(self, "status_label_tools", None)
        widgets = getattr(self, "_status_label_tool_widgets", ())
        if panel is None or len(widgets) != 7:
            return
        grid = panel.layout()
        # Leave room for native captions after theme/font changes; the former
        # right-side ROI column and stretch no longer need reserved space.
        button_widths = {"previous_button": 58, "next_button": 58, "delete_button": 58,
                         "grid_view_button": 58, "cuda_monitor_button": 0,
                         "clear_current_frame_labels_button": 96, "propagate_labels_button": 118,
                         "propagate_range_button": 122, "enable_roi_button": 126}
        for name, minimum in button_widths.items():
            button = getattr(self, name, None)
            if button is not None:
                button.setMinimumWidth(max(minimum, button.sizeHint().width()))
        scan = getattr(self, "auto_scan_checkbox", None)
        if scan is not None:
            scan.setFixedWidth(max(86, scan.sizeHint().width()))
        required = sum(max(widget.minimumWidth(), widget.minimumSizeHint().width()) for widget in widgets)
        required += grid.horizontalSpacing() * 6 + grid.contentsMargins().left() + grid.contentsMargins().right()
        narrow = panel.width() < required
        if getattr(self, "_status_label_tools_narrow", None) == narrow:
            return
        self._status_label_tools_narrow = narrow
        while grid.count():
            grid.takeAt(0)
        for column in range(7):
            grid.setColumnStretch(column, 0)
        title, opacity, rapid, zoom, snap, visibility, delay = widgets
        if narrow:
            for column, widget in enumerate((title, rapid, zoom, snap, visibility)):
                grid.addWidget(widget, 0, column)
            grid.addWidget(opacity, 1, 0, 1, 2)
            grid.addWidget(delay, 1, 2, 1, 3)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(4, 1)
        else:
            for column, widget in enumerate(widgets):
                grid.addWidget(widget, 0, column)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(6, 1)
        panel.updateGeometry()

    def _saved_main_workflow_index(self):
        """Restore page identity; migrate indices saved before Label was removed."""
        tabs = self.tabWidget
        saved_name = str(self.settings.get("mainWorkflowPage", "") or "")
        if not saved_name:
            try:
                legacy = max(0, min(5, int(self.settings.get("mainTabIndex", 0))))
            except (TypeError, ValueError):
                legacy = 0
            if getattr(self, "_label_workflow_consolidated", False):
                saved_name = ("Extract", "Trainer", "Trainer", "tab", "tab_6", "tab_2")[legacy]
            else:
                return min(legacy, max(0, tabs.count() - 1))
        for index in range(tabs.count()):
            if tabs.widget(index).objectName() == saved_name:
                return index
        return 0

    from darkfusion_class_editor import _compact_keypoint_panel

    from darkfusion_class_editor import _hide_legacy_keypoint_tab

    def _compact_augment_convert_panel(self):
        self._relocate_extra_label_to_augmentation_dialog()
        self._compact_effects_panel()
        self._compact_sam3_tools_panel()
        self._compact_shared_augmentation_percent_panel()
        self._reorder_augment_convert_panels()

    from darkfusion_augmentations import (
        _relocate_extra_label_to_augmentation_dialog,
        show_extra_label_dialog,
        _sync_auto_head_target_class_id,
        _selected_auto_head_target_name,
        selected_nested_label_mode,
        update_nested_label_mode_controls,
        refresh_auto_head_class_selectors,
        ensure_yolo_augmentation_controls,
        _compact_effects_panel,
        _compact_shared_augmentation_percent_panel,
        _ensure_augmentation_output_format_combo,
        _reorder_augment_convert_panels,
        _remove_legacy_create_blanks_panel,
        _compact_sam3_tools_panel,
        _appearance_augmentation_config,
        _restore_appearance_augmentation_controls,
        _sync_appearance_augmentation_controls,
        _on_appearance_style_changed,
        pick_appearance_augmentation_color,
    )

    from darkfusion_autolabel import (
        setup_unified_auto_label_menu,
        show_auto_label_dialog,
    )

    def _clear_generated_inline_styles(self):
        """Let selected QSS themes own colors instead of Designer's baked-in greys."""
        keep_styles = {
            "crosshair_color_btn",
            "preview_flash_color_btn",
            "preview_alt_flash_color_btn",
            "appearance_color_button",
            "enable_roi_button",
            "extract_button",
            "stop_extract",
            "systemMonitorDialog",
        }

        for widget in self.findChildren(QtWidgets.QWidget):
            object_name = widget.objectName()
            if object_name in keep_styles:
                continue
            # These rules control category spacing/transparency, not colors.
            # Retain them when a theme is changed with Settings still open.
            if (isinstance(widget, CollapsibleSettingsGroup) or widget.property("roiEdgeControl") or widget.property("settingsSectionContent")
                    or widget.property("collapsibleSectionBody") or widget.property("workflowChrome")):
                continue

            if isinstance(widget, (QtWidgets.QFrame, QtWidgets.QGroupBox, QtWidgets.QTabWidget,
                                   QtWidgets.QTabBar, QtWidgets.QScrollArea, QtWidgets.QWidget,
                                   QtWidgets.QPushButton, QtWidgets.QCheckBox, QtWidgets.QComboBox,
                                   QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QSlider,
                                   QtWidgets.QLineEdit, QtWidgets.QListView, QtWidgets.QTableWidget)):
                if widget.styleSheet():
                    widget.setStyleSheet("")
                widget.setAutoFillBackground(False)

        # Also clear styles on common popup/list widgets and menus that Designer may have styled
        try:
            for combo in self.findChildren(QtWidgets.QComboBox):
                try:
                    view = combo.view()
                    if view is not None and view.styleSheet():
                        view.setStyleSheet("")
                except Exception:
                    pass
        except Exception:
            pass

        try:
            for menu in self.findChildren(QtWidgets.QMenu):
                try:
                    if menu.styleSheet():
                        menu.setStyleSheet("")
                except Exception:
                    pass
        except Exception:
            pass

        if isinstance(getattr(self, "appearance_style_combo", None), QtWidgets.QComboBox):
            self._sync_appearance_augmentation_controls()

    from darkfusion_ui_controls import _take_layout_widget

    from darkfusion_dataset_review import _group_main_review_controls

    def __init__(self):
        super().__init__()
        self.status_message_requested.connect(self._show_status_message, Qt.QueuedConnection)
        self.voice_class_recognized.connect(self.apply_voice_class_selection, Qt.QueuedConnection)
        self.voice_mode_failed.connect(self.report_voice_mode_failure, Qt.QueuedConnection)
        # Startup theme/GIF signals must not collect designer defaults over
        # saved controls before restore_persistent_ui_state has applied them.
        self._restoring_settings = True
        self.setupUi(self)
        self._apply_global_line_spacing()
        self._clean_restored_ui()
        self.video_playback_fps_limit = 240
        self.language_dropdown = self._ensure_language_dropdown()
        self.setup_tab_workflow()
        self.ensure_yolo_augmentation_controls()
        self.ensure_tile_images_controls()
        self.ensure_enhance_image_checkbox()
        self.setup_control_copy()
        self._polish_loaded_ui()
        self.setup_dock_context_help()
        QtWidgets.QApplication.setStyle('Fusion')
        global start_time
        start_time = datetime.now()

        # --- App shell and startup state ---
        self.icons = [
            QIcon('styles/icons/Df1.png'),
            QIcon('styles/icons/Df2.png'),
        ]
        self.setWindowIcon(self.icons[0])
        self.last_logged_file_name = None
        # Cache GPU availability once during startup.
        self.pytorch_cuda_available = torch.cuda.is_available()
        self.opencv_cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

        self.heatmap_colormap_options = {
            "JET": cv2.COLORMAP_JET,
            "TURBO": cv2.COLORMAP_TURBO,
            "MAGMA": cv2.COLORMAP_MAGMA,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "PLASMA": cv2.COLORMAP_PLASMA,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS,
            "CIVIDIS": cv2.COLORMAP_CIVIDIS,
            "HOT": cv2.COLORMAP_HOT,
            "COOL": cv2.COLORMAP_COOL,
            "HSV": cv2.COLORMAP_HSV,
            "TWILIGHT": cv2.COLORMAP_TWILIGHT,
            "TWILIGHT_SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED,
            "BONE": cv2.COLORMAP_BONE,
            "OCEAN": cv2.COLORMAP_OCEAN,
            "PINK": cv2.COLORMAP_PINK,
            "SPRING": cv2.COLORMAP_SPRING,
            "SUMMER": cv2.COLORMAP_SUMMER,
            "AUTUMN": cv2.COLORMAP_AUTUMN,
            "WINTER": cv2.COLORMAP_WINTER,
            "RAINBOW": cv2.COLORMAP_RAINBOW,
            "PARULA": cv2.COLORMAP_PARULA,
            "DEEPGREEN": cv2.COLORMAP_DEEPGREEN
        }
        # --- Shared processors and background pools ---
        self.video_processor = GUIVideoProcessor(self.label_progress)
        self.set_main_progress(0, 100)
        self.label_progress.setToolTip(
            "Shows active job progress. During image review, it also reflects dataset navigation."
        )


        self.setWindowTitle(f"UltraDarkFusion {APP_VERSION}")

        # --- App timers, logger bridge, and top-level menu actions ---
        self.icon_timer = QTimer(self)
        self.icon_timer.timeout.connect(self.update_icon)
        self.icon_timer.start(500)
        self.current_icon_index = 0
        self.threadpool = QThreadPool()
        self._review_filter_worker = None
        self._review_similarity_threads = []
        self._review_filter_request_id = 0
        self._review_filter_label_cache = {}
        self._review_similarity_reference = None
        self._review_similarity_matches = []
        self._review_similarity_matches_by_image = {}
        self._review_similarity_origin_file = ""
        self._review_similarity_origin_index = -1
        self._review_similarity_results_ready = False
        self._review_filter_restore_file = ""
        self._review_filter_restore_index = -1
        self._review_filter_origin = None
        self.review_similarity_threshold = review_threshold(getattr(self, "settings", {}))
        self._bulk_move_worker = None
        self._filtered_export_worker = None
        self.logger = logging.getLogger("UltraDarkFusionLogger")

        self.label_log_handler = QLabelInfoLogHandler()
        self.label_log_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self.label_log_handler)

        self._setup_cuda_monitor_button()
        self.histogram_plot.triggered.connect(lambda: self.create_plot('histogram'))
        self.bar_plot.triggered.connect(lambda: self.create_plot('bar'))
        self.scatter_plot.triggered.connect(lambda: self.create_plot('scatter'))
        self.call_stats_button.triggered.connect(self.display_stats)

        self.label_file = None
        self.file_name = None
        # --- Main scene and graphics view ---
        self.view_references = []
        self.screen_view = CustomGraphicsView(main_window=self)
        self.screen_view.setBackgroundBrush(QBrush(Qt.black))
        self.screen_view.setRenderHint(QPainter.Antialiasing)
        self.screen_view.setRenderHint(QPainter.TextAntialiasing)
        self.frame_viewer = FastFrameViewer(self)
        self.grid_review_view = GridReviewView(self)
        self.grid_review_view.setObjectName("grid_review_view")
        self.grid_review_view.setViewMode(QtWidgets.QListView.IconMode)
        self.grid_review_view.setFlow(QtWidgets.QListView.LeftToRight)
        self.grid_review_view.setWrapping(True)
        self.grid_review_view.setResizeMode(QtWidgets.QListView.Adjust)
        self.grid_review_view.setMovement(QtWidgets.QListView.Static)
        self.grid_review_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.grid_review_view.setMouseTracking(True)
        self.grid_review_view.setUniformItemSizes(True)
        self.grid_review_view.setWordWrap(True)
        self.grid_review_view.setSpacing(8)

        # Load grid thumbnail size from settings (use default if settings not loaded yet)
        grid_thumb_size = 240  # Default size
        if hasattr(self, "settings") and self.settings:
            grid_thumb_size = int(self.settings.get("gridThumbnailSize", 240))
        grid_thumb_size = max(100, min(400, grid_thumb_size))
        grid_thumb_height = grid_thumb_size * 2 // 3
        grid_cell_width = grid_thumb_size + 24
        grid_cell_height = grid_thumb_height + 50

        self.grid_review_view.setIconSize(QtCore.QSize(grid_thumb_size, grid_thumb_height))
        self.grid_review_view.setGridSize(QtCore.QSize(grid_cell_width, grid_cell_height))
        self.grid_review_view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.grid_review_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid_review_view.setStyleSheet(
            "QListView#grid_review_view { background: #0e1116; border: 0; padding: 8px; }"
            "QListView#grid_review_view::item { color: #e8edf3; border: 2px solid transparent; "
            "border-radius: 7px; padding: 5px; }"
            "QListView#grid_review_view::item:hover { background: #202a34; border-color: #5c758d; }"
            "QListView#grid_review_view::item:selected { background: #173b29; border-color: #48e07b; }"
        )
        self.grid_thumbnail_model = GridThumbnailModel(
            thumbnail_size=QtCore.QSize(grid_thumb_size, grid_thumb_height),
            parent=self.grid_review_view,
        )
        self.grid_review_view.setModel(self.grid_thumbnail_model)
        self.grid_review_view.clicked.connect(self.open_grid_image)
        self.grid_review_view.entered.connect(self.schedule_grid_hover_preview)
        self.grid_review_view.checkedChanged.connect(self.grid_checked_selection_changed)
        self.grid_review_view.viewportChanged.connect(self.schedule_grid_visible_thumbnails)
        self.grid_review_view.verticalScrollBar().valueChanged.connect(
            self.schedule_grid_visible_thumbnails
        )
        self._grid_hover_preview_timer = QTimer(self)
        self._grid_hover_preview_timer.setSingleShot(True)
        self._grid_hover_preview_timer.setInterval(140)
        self._grid_hover_preview_timer.timeout.connect(self.show_grid_hover_preview)
        self._grid_hover_image_file = ""
        self.grid_mode_active = False
        self.viewer_stack = QtWidgets.QStackedWidget(self)
        self.viewer_stack.setContentsMargins(0, 0, 0, 0)
        self.viewer_stack.addWidget(self.screen_view)
        self.viewer_stack.addWidget(self.frame_viewer)
        self.viewer_stack.addWidget(self.grid_review_view)
        self.viewer_stack.setCurrentWidget(self.screen_view)
        self.setCentralWidget(self.viewer_stack)
        self.graphics_scene = QGraphicsScene(self)
        self.selected_bbox = None
        self.screen_view.setScene(self.graphics_scene)


        # --- Image navigation and delete controls ---
        self.img_video_button.triggered.connect(self.open_image_video)
        self.next_button.pressed.connect(self.next_frame)
        self.next_button.released.connect(self.on_next_button_released)
        self.next_timer = QTimer(self)
        self.next_timer.timeout.connect(self.next_frame)
        self.timmer_speed.valueChanged.connect(self.update_timer_speed)
        self.timer_interval = self.timmer_speed.value()
        self.previous_button.pressed.connect(self.previous_frame)
        self.previous_button.released.connect(self.on_previous_button_released)
        self.prev_timer = QTimer(self)
        self.prev_timer.timeout.connect(self.previous_frame)
        self.scan_direction = None
        self.auto_scan_checkbox.toggled.connect(self.toggle_auto_scan)
        self.navigation_busy = False

        # --- Annotation class controls and model thresholds ---
        self.clear_dropdown_signal.connect(self.clear_classes_dropdown)
        self.add_item_signal[str].connect(self.add_item_to_classes_dropdown)
        self.classes_dropdown.currentIndexChanged.connect(self.change_class_id)
        self._highlighted_row = None
        self._preview_dataset_analysis_target = None
        self._preview_dataset_analysis_target_image = ""
        self._preview_target_pulse_generation = 0
        self.weights_file = None
        self.cfg_file = None
        self.weights_files = []
        self.loaded_model_path = None

        self.current_class_id = 0
        self.confidence_threshold_spinbox.valueChanged.connect(self.threshold_spinbox_changed)
        self.nms_threshold_spinbox.valueChanged.connect(self.threshold_spinbox_changed)
        self.box_size.valueChanged.connect(self.on_prediction_size_filter_changed)
        self.max_label.valueChanged.connect(self.on_prediction_size_filter_changed)
        self.classes_dropdown.setContextMenuPolicy(Qt.CustomContextMenu)
        self.classes_dropdown.customContextMenuRequested.connect(self.classes_dropdown_menu)
        self.classes_dropdown.installEventFilter(self)
        self.classes_dropdown.setToolTip("Click to open the class and pose lists. Right-click for quick options.")
        self.class_picker_dialog = None
        # Initialize thresholds from current spinbox values
        self.confidence_threshold = self.confidence_threshold_spinbox.value()
        self.nms_threshold = self.nms_threshold_spinbox.value()

        # --- Crop and manual delete actions ---
        self.crop_img_button.clicked.connect(self.crop_images)

        self.delete_button.clicked.connect(self.delete_current_image)

        self.image_list = []
        self.class_to_id = {}
        self.zoom_mode = False
        self.original_transform = None

        # --- Settings and persistent dataset state ---
        self.settingsButton.triggered.connect(self.openSettingsDialog)
        self.settings = self.loadSettings()
        self._sync_preview_annotation_actions()
        # Training controls may open while the UI is restoring. Initialize
        # this setting immediately after loading so those controls never read
        # an attribute that has not been created yet.
        self.ultralytics_extra_train_args = str(
            self.settings.get("ultralyticsExtraTrainArgs", "") or ""
        )
        self.frame_viewer.set_quality_mode(
            self.settings.get("videoPlaybackQuality", "smooth")
        )
        self._sam_model_load_lock = threading.RLock()
        self._sam_video_inference_lock = threading.RLock()
        attach_gpu_memory(self)
        self._sam3_startup_warmup_active = False
        self._sam3_startup_warmup_thread = None
        self._sam3_propagation_warmup_worker = None
        self._sam3_video_propagation_ready = False
        self._sam3_video_propagation_unavailable = False
        self._sam3_propagation_batch_active = False
        self._sam3_propagation_shutting_down = False
        self._sam3_propagation_release_pending = False
        self._sam3_propagation_last_message = ""
        self.sam3_startup_warmup_finished.connect(
            self._start_sam3_propagation_preload, QtCore.Qt.QueuedConnection
        )
        if hasattr(self.screen_view, "pan_overlay"):
            self.screen_view.pan_overlay.apply_saved_settings()
        self.initialize_floating_point_mode_from_settings()
        self.train_view_active = False
        self.training_preview_always_on = bool(self.settings.get("trainingPreviewAlwaysOn", False))
        self.training_preview_size = None
        self.training_evaluator_dialog = None
        self.training_evaluator_last_result = None
        self.training_evaluator_tune_process = None
        self.training_evaluator_tune_timer = None
        self.training_evaluator_batch_process = None
        self.training_evaluator_batch_timer = None
        self.training_evaluator_active_run = None
        self.training_evaluator_last_run = None
        self.training_evaluator_active_training_run = None
        self.training_evaluator_last_training_run = None
        self.training_artifact_viewers = []
        self._settings_save_timer = QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.timeout.connect(self.saveSettings)


        self.class_names = []  # Initialize as an empty list

        self.image_directory = None  # Initialize the attribute to store the image directory
        self.THUMBS_BY_CLASS_ROOT = None
        self.file_observer = None
        self.image_files = []  # Initialize empty image files list
        self.filtered_image_files = []  # For filtered images
        self._image_file_index = {}
        self._image_file_index_list_id = id(self.image_files)
        self._image_file_index_length = 0
        self._dataset_directory_scan_generation = 0
        self._dataset_directory_scan_worker = None
        self._dataset_directory_refresh_pending = False
        self._watched_dataset_directory = ""
        self._dataset_directory_last_mtime_ns = None
        self._dataset_directory_scan_mtime_ns = None
        self._dataset_directory_scan_succeeded = False
        self._dataset_directory_refresh_timer = QTimer(self)
        self._dataset_directory_refresh_timer.setSingleShot(True)
        self._dataset_directory_refresh_timer.setInterval(350)
        self._dataset_directory_refresh_timer.timeout.connect(
            self._start_dataset_directory_refresh
        )
        self.current_image_index = 0  # To track current image
        self.current_file = None  # Current displayed file
        self.current_file = None
        self.label_file = None
        self.file_name = None

        self._loading_image = False
        self._saving_labels = False
        self._overlay_text_item = None
        self._overlay_timer = QTimer(self)
        # --- Review filters, image list, and annotation preview ---
        self.filter_class_spinbox.addItem("All (-1)")  # Special option: All
        self.filter_class_spinbox.addItem("Blanks (-2)")  # Special option: Blanks
        self.filter_class_spinbox.currentIndexChanged.connect(self.on_filter_class_spinbox_changed)
        self.img_index_number.valueChanged.connect(self.img_index_number_changed)
        self.img_index_search.textChanged.connect(self.img_index_number_changed)


        self.update_filter_spinbox()  # Populate with initial values
        self.List_view.clicked.connect(self.on_list_view_clicked)
        self.hide_labels = bool(self.settings.get("hideLabels", False))
        if hasattr(self, "hide_label_checkbox"):
            try:
                self.hide_label_checkbox.setText("Hide Class Labels (Text)")
                self.hide_label_checkbox.setToolTip(
                    "Checked hides class-name text and badges while keeping annotation shapes visible."
                )
                self.hide_label_checkbox.blockSignals(True)
                self.hide_label_checkbox.setChecked(self.hide_labels)
                self.hide_label_checkbox.blockSignals(False)
                self.hide_label_checkbox.triggered.connect(self.toggle_label_visibility)
            except Exception as e:
                logger.debug(f"Could not wire hide-label action: {e}")
        self.setup_list_view_with_delegate()
        self.image = None


        self.flash_color_rgb = (255, 0, 0)  # Default flash color
        self.alternate_flash_color_rgb = (0, 0, 255)  # Default alternate flash color
        default_flash_time = 1000
        self.flash_time_value = int(self.settings.get("previewFlashTimeMs", default_flash_time))
        self.screen_view.viewport().installEventFilter(self)
        self._image_size_value = int(self.settings.get("previewThumbnailSize", 128))
        self.show_images = True
        self.loader = UiLoader(self)
        self.loader.setup_ui()
        self.setup_preview_list_style()
        self.setup_review_panel_layout()
        self.setup_review_view_actions()

        self.apply_preview_settings()


        self.thumbnails_directory = None
        self.id_to_class = {}
        self.label_dict = {}
        self.current_bounding_boxes = []


        self.batch_directory = ""
        self.batch_running = False
        self.current_bbox_index = {}  # Track the current bounding box index for each image
        self.flash_time = QSpinBox()  # Placeholder for actual flash time value control

        QApplication.instance().installEventFilter(self)
        self.review_mode = False  # Track review state
        self.last_thumbnail_visibility = True  # For restoring state
        self.review_button.clicked.connect(self.toggle_review_mode)

        self.move_all_button.clicked.connect(self.move_filtered_images)
        self.clear_all_button.clicked.connect(self.clear_class_boxes)
        if hasattr(self, "go_to_review_file_button"):
            self.go_to_review_file_button.clicked.connect(self.go_to_selected_review_file)
        self.preview_list.viewport().installEventFilter(self)
        self.preview_list.verticalScrollBar().valueChanged.connect(
            self._on_preview_scroll_value_changed
        )
        self.rightClicked.connect(self.handle_right_click)
        self.leftClicked.connect(self.handle_left_click)

        # --- OpenCV DNN and auto-label runtime state ---
        self.setup_dataset_image_converter_controls()
        self._last_call = time.time()
        self.preview_list.installEventFilter(self)
        self.plot_counter = 0
        self.bounding_boxes = []
        self.blob_cache = {}
        self.auto_label_yolo_button.triggered.connect(lambda: self.show_auto_label_dialog("weights"))
        self.actionSahi_label.triggered.connect(lambda: self.show_auto_label_dialog("sahi"))
        self.stop_labeling = False
        self._sam3_batch_active = False
        self._sam3_cancel_requested = False
        self.process_timer = QTimer(self)
        self.process_timer.setSingleShot(True)
        self.process_timer.timeout.connect(self.process_current_image)
        self.process_timer_interval = 500
        self.cuda_available = False

        # --- Theme, GIF, and audio feedback ---
        self.movie = QMovie('styles/gifs/darkfusion.gif')

        self.populate_style_combo_box()
        self.styleComboBox.currentIndexChanged.connect(self.on_style_change)

        self.image_label_2.setMovie(self.movie)
        self._scale_status_gif()
        self.movie.start()
        QtCore.QTimer.singleShot(0, self._scale_status_gif)
        self.populateGifCombo()
        self.onGifChange(self.gif_change.currentIndex())
        self.gif_change.currentIndexChanged.connect(self.onGifChange)
        self.sound_player = QMediaPlayer()
        self.sound_player.setMuted(bool(self.settings.get("muteAudio", True)))
        self.update_counts = {}
        # --- Video playback, extraction, and download tools ---
        self.video_source_cache = {}
        self.setup_video_queue_table()
        self.video_upload.clicked.connect(self.on_add_video_clicked)
        self.setAcceptDrops(True)
        self.remove_video_button.clicked.connect(self.on_remove_video_clicked)
        self.custom_frames_checkbox.stateChanged.connect(self.on_custom_frames_toggled)
        self.custom_size_checkbox.stateChanged.connect(self.on_custom_size_checkbox_state_changed)
        self.crop_images_checkbox.stateChanged.connect(self.on_crop_images_checkbox_state_changed)
        for size_box in (self.width_box, self.height_box):
            size_box.setMinimum(1)
        self.selected_image_format = ".jpg"

        self.inference_checkbox.stateChanged.connect(self.toggle_inference)
        self.toggle_inference(self.inference_checkbox.checkState())
        self.tracking_enabled_checkbox.stateChanged.connect(self.toggle_tracking)
        self.toggle_tracking(self.tracking_enabled_checkbox.checkState(), restart=False)
        self.play_video_button.clicked.connect(self.toggle_play_pause)
        self.extract_mode_active = False
        self.playback_extraction_enabled = True  # Default true
        self.setup_mss()

        # Video playback speed controls.
        self.forward_button.clicked.connect(self.skip_forward)
        self.back_button.clicked.connect(self.skip_backward)

        self.video_slider.setEnabled(False)  # Disabled until video is loaded
        self.video_slider.sliderReleased.connect(self.slider_seek)
        self.video_slider.sliderPressed.connect(self.slider_pressed)
        self.slider_is_pressed = False
        self.connect_extraction_controls()
        self.is_playing = False
        self._live_timer_paused = False

        self.default_classes_path = os.getcwd()
        self.dialog_open = False
        self.output_path = ""
        self.add_video_running = False
        self.custom_frames_checkbox.stateChanged.connect(self.update_checkboxes)
        self.custom_size_checkbox.stateChanged.connect(self.update_checkboxes)
        self.crop_images_checkbox.stateChanged.connect(self.update_checkboxes)
        self.height_box.valueChanged.connect(self.on_size_spinbox_value_changed)
        self.width_box.valueChanged.connect(self.on_size_spinbox_value_changed)
        self.custom_input.textChanged.connect(self.on_custom_input_changed)
        self.video_processor.progress_updated.connect(self.update_progress)
        self.width_box.valueChanged.connect(self.force_live_redraw)
        self.height_box.valueChanged.connect(self.force_live_redraw)
        for network_spin in (self.network_width, self.network_height):
            try:
                network_spin.setSingleStep(32)
                network_spin.setKeyboardTracking(False)
            except Exception:
                pass
        self.network_width.valueChanged.connect(self.on_network_size_spinbox_value_changed)
        self.network_height.valueChanged.connect(self.on_network_size_spinbox_value_changed)
        self.setup_image_format_controls()


        self.paste_url = self.findChild(QtWidgets.QLineEdit, 'paste_url')
        self.video_download = self.findChild(QtWidgets.QTableWidget, 'video_download')
        self.download_video = self.findChild(QtWidgets.QPushButton, 'download_video')

        self.download_video.clicked.connect(self.download_videos)
        self.remove_video = self.findChild(QtWidgets.QPushButton, 'remove_video')
        self.remove_video.clicked.connect(self.remove_selected_video)
        self.download_threads = []
        self.setup_video_download_table()
        self.current_image = None
        self.capture = None
        self.extraction_thread = None
        self._active_extraction_settings = None
        self.is_camera_mode = False
        self.current_file_name = None
        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(self.display_camera_input)
        self.skip_frames_count = 1
        self.set_extraction_ui_state("idle")
        self.input_selection.currentIndexChanged.connect(self.on_input_source_changed)
        self.setup_input_sources()
        self.save_path = ""

        # --- Dataset import paths and CFG state ---
        self.weights_file_path = None
        self.cfg_file_path = None

        self.extracting_frames = False
        self.custom_size = None
        self.label_indices = {}
        self.processing = False
        self.current_img_index = 0
        self.cfg_dict = {}
        self.file_type = ""
        self.import_button.clicked.connect(self.import_images)
        self.output_button.clicked.connect(self.output_paths)

        # --- Darknet training ---
        self.browse_data.clicked.connect(self.browse_data_clicked)
        self.browse_weights.clicked.connect(self.browse_weights_clicked)
        self.train_button.clicked.connect(self.darknet_train)
        self.gpu1.toggled.connect(lambda checked: self.normalize_darknet_gpu_options("gpu1" if checked else None))
        self.gpu2.toggled.connect(lambda checked: self.normalize_darknet_gpu_options("gpu2" if checked else None))
        self.weights_button.triggered.connect(lambda: self.show_auto_label_dialog("weights"))
        self.import_data_button.clicked.connect(self.import_data)
        self.calculate_anchors_button.clicked.connect(self.calculate_anchors)
        self.browse_cfg.clicked.connect(self.browse_cfg_clicked)
        self.cfg_button.triggered.connect(self.open_cfg)


        # --- Ultralytics training ---
        self.model_input.clicked.connect(self.browse_pt_clicked)
        self.data_input.clicked.connect(self.browse_yaml_clicked)
        self.runs_directory = ''  # Initial empty value for save directory
        self.save_runs_directory.clicked.connect(self.on_save_dir_clicked)
        self.ultralytics.clicked.connect(self.ultralytics_train_clicked)
        if isinstance(getattr(self, "training_evaluator_button", None), QtWidgets.QPushButton):
            self.training_evaluator_button.clicked.connect(self.open_training_evaluator_dialog)
        if isinstance(getattr(self, "ultralytics_trainer_launcher_button", None), QtWidgets.QPushButton):
            self.ultralytics_trainer_launcher_button.clicked.connect(self.open_training_evaluator_dialog)
        self.model_config_path = None
        self.pretrained_model_path = None
        self.pt_path = None
        self.data_yaml_path = None
        self.selected_pytorch_file = None
        self.ultralytics_distill_enabled = bool(self.settings.get("ultralyticsDistillEnabled", False))
        self.ultralytics_distill_model_path = self.normalize_path(self.settings.get("ultralyticsDistillModelPath", ""))
        try:
            self.ultralytics_distill_weight = float(self.settings.get("ultralyticsDistillWeight", 6.0) or 6.0)
        except Exception:
            self.ultralytics_distill_weight = 6.0

        # --- CFG/YAML editor and anchor tools ---
        self.btn_open_file.clicked.connect(self.cfg_open_clicked)
        self.btn_save_file.clicked.connect(self.cfg_save_clicked)
        self.import_yaml_button.clicked.connect(self.import_yaml)
        self.cfg_table.cellChanged.connect(self.save_table_changes)
        self.hide_activation_checkbox.stateChanged.connect(self.toggle_activation_layers)
        self.file_paths = {"data": "", "cfg": "", "weights": []}
        self.parsed_yaml = None
        self.filename = None
        self.input_dir = None
        self.imported_anchors = None
        self.yaml_filename = ""
        self.STEP_PERCENTAGES = [0.7, 0.8, 0.9]
        self.DEFAULT_SCALES = "0.1,0.1,0.1"
        self.playback_speed = 1.0  # 1.0x default
        self.speed_levels = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 999.0]
        self.video_fps = 30.0
        self.total_frames = 0
        self.playback_interval_ms = 33
        self.loop_playback_enabled = False
        self.video_mark_in_frame = None
        self.video_mark_out_frame = None
        self.current_playback_source = ""
        self.current_playback_original_source = ""
        self.reset_video_performance_counters(update=False)
        self._video_inference_submit_interval = 0.0
        self._live_display_active = False
        self._live_display_size = None
        self.video_reader_thread = None
        self._video_reader_generation = 0
        self.video_inference_thread = None
        self._video_inference_generation = 0
        self._pending_live_inference_refresh_reason = ""
        self._live_inference_refresh_timer = QTimer(self)
        self._live_inference_refresh_timer.setSingleShot(True)
        self._live_inference_refresh_timer.timeout.connect(self.refresh_live_inference_worker_from_ui)
        self.latest_video_frame_number = 0
        self.last_overlay_frame_number = -1
        self.latest_video_overlay = None
        self._last_video_frame_bgr = None
        self.page_size = 100
        self.current_page = 0
        self.current_preview_image = None
        # --- Legacy training-file utilities ---
        if hasattr(self, "combine_txt_button"):
            self.combine_txt_button.triggered.connect(self.on_combine_txt_clicked)
        self.combine_txt_flag = False
        if hasattr(self, "clear_json"):
            self.clear_json.triggered.connect(self.on_clear_json_clicked)

        # --- Dataset tools, conversion, and health checks ---
        if isinstance(getattr(self, "process_image_button", None), QtWidgets.QPushButton):
            self.process_image_button.clicked.connect(self.on_button_click)
        self.convert_image.clicked.connect(self.convert_images)
        self.scan_annotations = ScanAnnotations(self)

        if isinstance(getattr(self, "scan_button", None), QtWidgets.QPushButton):
            self.scan_button.clicked.connect(self.scan_annotations.scan_annotations)
        self.scan_health_report_action = self.findChild(QAction, "scan_health_report_action")
        if self.scan_health_report_action is not None:
            self.scan_health_report_action.triggered.connect(self.scan_annotations.scan_annotations)
        self.deduplicate_labels_action = self.findChild(QAction, "deduplicate_labels_action")
        if self.deduplicate_labels_action is not None:
            self.deduplicate_labels_action.triggered.connect(self.deduplicate_current_dataset_clicked)
        if isinstance(getattr(self, "deduplicate_labels_button", None), QtWidgets.QPushButton):
            self.deduplicate_labels_button.clicked.connect(self.deduplicate_current_dataset_clicked)
        self.crop_button.clicked.connect(self.process_images_triggered)
        self.split_data_button = self.findChild(QPushButton, 'split_data')
        self.split_data_button.clicked.connect(self.launch_split_data_ui)
        self.image_quality_checkbox.toggled.connect(self.toggle_image_quality_analysis)
        self.image_quality_analysis_enabled = False
        self.move_from.clicked.connect(self.select_source_folder)
        self.move_to.clicked.connect(self.select_target_folder)
        self.start_move_btn.clicked.connect(self.start_moving_files)

        self.map_counter = 0
        self.subfolder_counter = 1
        self.valid_classes = []
        self.base_directory = None


        self.image_paths = []

        # --- Metrics and augmentation controls ---
        self.cpu_usage = 0
        self.gpu_usage = 0
        self.memory_usage = 0

        self.images = []
        self.counter = 1
        self.sight_picture.stateChanged.connect(self.sight_picture_checkbox_changed)
        self.sight_picture_effect = False
        self.smoke_checkbox.stateChanged.connect(self.smoke_checkbox_changed)
        self.smoke_effect = False
        self.flash_checkbox.stateChanged.connect(self.flash_checkbox_changed)
        self.flash_effect = False
        self.motion_blur.stateChanged.connect(self.motion_blur_checkbox_state_changed)
        self.motion_blur_effect = False
        self.glass_effect = False
        self.glass_checkbox.stateChanged.connect(self.glass_checkbox_state_changed)
        self.brightness_contrast_effect = False
        self.hsv_jitter_effect = False
        self.gaussian_noise_effect = False
        self.jpeg_compression_effect = False
        for widget_name, attr_name in (
            ("brightness_contrast_checkbox", "brightness_contrast_effect"),
            ("hsv_jitter_checkbox", "hsv_jitter_effect"),
            ("gaussian_noise_checkbox", "gaussian_noise_effect"),
            ("jpeg_compression_checkbox", "jpeg_compression_effect"),
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.stateChanged.connect(
                    lambda state, attr_name=attr_name: self.training_augmentation_checkbox_changed(attr_name, state)
                )
        self.make_copy_enabled = False
        self.make_copy.stateChanged.connect(self.make_copy_checkbox_changed)
        self.progress_signal.connect(self.update_progress)
        self._annotation_mode_auto_changing = False
        self._annotation_mode_manual_override = False
        self._last_auto_annotation_mode = None
        self.segmentation_mode.activated.connect(self.on_annotation_mode_user_selected)
        self.segmentation_mode.currentTextChanged.connect(self.on_segmentation_mode_changed)
        self.segmentation_mode.setCurrentText("Boxes")
        self.user_requested_stop = False
        self._augmentation_job = None


        self.augmentation_percent_spinbox.setRange(1, 100)
        self.augmentation_percent_spinbox.setSuffix("%")
        self.augmentation_percent_spinbox.setToolTip("Percent used for normal augmentation, SAM3 Copy/Paste, Clean BG, Shadow / Highlight, Object Detail, Object Clarity, and Negatives. SAM3 Auto-Fit, Seg, and OBB run on all images.")
        saved_augmentation_percent = int(self.settings.get("augmentationPercent", 100))
        self.augmentation_percent_spinbox.setValue(max(1, min(100, saved_augmentation_percent)))
        self.augmentation_percent_spinbox.valueChanged.connect(self.on_augmentation_percent_changed)
        self.augmentation_percent = self.augmentation_percent_spinbox.value()
        self.is_fp16 = False
        self.start_gathering_metrics()
        self.flip_images = False
        self.flip_checkbox.stateChanged.connect(self.set_flip_images)

        self.model = None

        # --- Keyboard shortcuts and display filters ---
        self.class_colors = []
        self.keyBuffer = ""

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.processKeyPresses)

        self.new_class_input = False

        self.slider_min_value = 50
        self.slider_max_value = 150

        self.bounding_boxes = {}

        self.outline_Checkbox.stateChanged.connect(self.checkbox_clicked)
        self.outline_Checkbox.toggled.connect(
            lambda checked: self.annotationDisplaySettings.save_snap_setting(checked)
            if hasattr(self, "annotationDisplaySettings")
            else None
        )
        self.super_resolution_Checkbox.triggered.connect(self.checkbox_clicked)
        self.grayscale_Checkbox.stateChanged.connect(self.checkbox_clicked)
        self.grey_scale_slider.valueChanged.connect(self.checkbox_clicked)
        self.super_resolution_Checkbox.toggled.connect(self.on_preprocessing_control_changed)
        self.grayscale_Checkbox.stateChanged.connect(self.on_preprocessing_control_changed)
        self.grey_scale_slider.valueChanged.connect(self.on_preprocessing_control_changed)

        try:
            self.debug_edge_display_checkbox.stateChanged.connect(self.checkbox_clicked)
        except Exception as e:
            logger.warning(f"Failed to connect debug_edge_display_checkbox: {e}")

        self.edge_slider_min.valueChanged.connect(self.edge_slider_changed)
        self.edge_slider_max.valueChanged.connect(self.edge_slider_changed)

        self.segmentation_checkbox.stateChanged.connect(self.on_segmentation_checkbox_checked)
        self.heatmap_Checkbox.stateChanged.connect(self.on_heatmap_toggle)
        self.setup_quick_toggle_buttons()

        # --- Thumbnail paging ---
        self.save_path = None
        self.yolo_files = []
        self._thumbnail_timer = QTimer(self)
        self._thumbnail_timer.setSingleShot(True)
        self._thumbnail_timer.timeout.connect(self._run_debounced_thumbnail_page)
        self._pending_thumbnail_args = None
        self.zoom_lock_state = {
            "enabled": False,
            "scale": 1.0,
            "center_x_ratio": 0.5,
            "center_y_ratio": 0.5,
        }

        try:
            self.zoom_lock.stateChanged.connect(self.on_zoom_lock_state_changed)
            self.zoom_lock.toggled.connect(
                lambda checked: self.annotationDisplaySettings.save_zoom_lock_setting(checked)
                if hasattr(self, "annotationDisplaySettings")
                else None
            )
        except Exception as e:
            logger.warning(f"zoom_lock checkbox setup failed: {e}")

        # --- Model conversion and precision controls ---
        self.Load_pt_model.clicked.connect(self.load_model)
        self.convertButton.clicked.connect(self.handle_conversion)
        self.convert_model.currentIndexChanged.connect(self.format_changed)
        self.half_true.stateChanged.connect(lambda: self.update_gui_elements(self.convert_model.currentText()))
        self.fp_select_combobox.currentIndexChanged.connect(self.switch_floating_point_mode)

        # --- SAM3/DINO assist tools and batch processing ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_labels = []
        self.input_points = []

        self.overwrite_var.stateChanged.connect(self.on_check_button_click)

        self.temp_bbox = []
        self.overwrite = False
        self.yolo_cache = {}
        self.shadow_var.stateChanged.connect(self.on_shadow_checkbox_click)
        self.shadow = False
        self.noise_remove_checkbox.stateChanged.connect(self.on_noise_remove_checked)
        self.copy_paste_checkbox.stateChanged.connect(self.on_copy_paste_checkbox_click)
        if isinstance(getattr(self, "object_detail_checkbox", None), QtWidgets.QCheckBox):
            self.object_detail_checkbox.stateChanged.connect(self.on_object_detail_checkbox_click)
        if isinstance(getattr(self, "create_negatives_checkbox", None), QtWidgets.QCheckBox):
            self.create_negatives_checkbox.stateChanged.connect(self.on_create_negatives_checkbox_click)
        self.is_noise_remove_enabled = False
        self.copy_paste_enabled = False
        self.object_detail_enabled = False
        self.create_negatives_enabled = False
        self.is_obb_generation_enabled = False
        self.process_folder_btn.clicked.connect(self.process_batch)
        self.num_classes = 1
        self.dino_label.triggered.connect(lambda: self.show_auto_label_dialog("dino"))
        self.setup_unified_auto_label_menu()

        # --- ROI and layout controls ---
        self.initial_geometry = self.saveGeometry()
        self.initial_state = self.saveState()
        self.roi_item_1 = None
        self.roi_item_2 = None
        self._roi_bounds_cache = {}  # roi_id -> (x, y, w, h) ints; safe to read off-thread
        # Edge sliders stuck to the display border; shown only when ROI is enabled.
        self.roi_edge_sliders = ROIEdgeSliders(self, self.viewer_stack)
        self.roi_edge_sliders.connect_real_sliders()
        self.roi_checkbox.stateChanged.connect(lambda state: self.toggle_roi(state, 1))
        self.roi_checkbox_2.stateChanged.connect(lambda state: self.toggle_roi(state, 2))
        self.reset_layout_action = self.findChild(QAction, "reset_layout")
        if self.reset_layout_action:
            self.reset_layout_action.triggered.connect(self.resetLayout)
        # ROI sliders and spin boxes
        self.width_spin_slider.valueChanged.connect(lambda: self._on_roi_slider_changed(1))
        self.width_spin_slider_2.valueChanged.connect(lambda: self._on_roi_slider_changed(2))
        self.height_spin_slider.valueChanged.connect(lambda: self._on_roi_slider_changed(1))
        self.height_spin_slider_2.valueChanged.connect(lambda: self._on_roi_slider_changed(2))
        self.x_spin_slider.valueChanged.connect(lambda: self._on_roi_slider_changed(1))
        self.x_spin_slider_2.valueChanged.connect(lambda: self._on_roi_slider_changed(2))
        self.y_spin_slider.valueChanged.connect(lambda: self._on_roi_slider_changed(1))
        self.y_spin_slider_2.valueChanged.connect(lambda: self._on_roi_slider_changed(2))

        # --- Annotation filters, OBB, and display overlays ---
        self.filter_button.clicked.connect(self.filter_bboxes)
        self.logged_label_files = set()
        self.obb_checkbox.stateChanged.connect(self.on_obb_checkbox_checked)
        self._deletion_in_progress = False

        self.tracking_enabled = self.tracking_enabled_checkbox.isChecked()

        self.current_item_index = None  # CHECK THIS
        self.thumbnail_index = 0
        self.time = 0.0

        self.all_frame_bounding_boxes = {}  # Dictionary to store bounding boxes for all frames
        self.class_visibility = {}  # Store visibility state for each class
        self.all_frame_segmentations = {}
        self.all_frame_obbs = {}
        self.font_size_slider.valueChanged.connect(self.update_font_size_display)
        self.shade_slider.valueChanged.connect(self.update_shading_opacity)
        self.heatmap_dropdown.currentIndexChanged.connect(self.on_heatmap_colormap_changed)

        # --- Keypoint list and monitor selection ---
        self.list_input.returnPressed.connect(self.add_keypoint_list_item)
        self.remove_input.clicked.connect(self.remove_keypoint_list_item)
        self.keypoint_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.keypoint_list.setToolTip(
            "States for the selected box: Visible checked = 2, neither checked = "
            "occluded (1), Missing checked = 0. Names and point count remain global."
        )

        self.current_monitor_index = 1  # Default to monitor 1

        self.detect_and_set_monitor_range()
        self.monitor.valueChanged.connect(self.set_monitor_index)
        # Initialize table
        self.keypoint_list.setColumnCount(5)
        self.keypoint_list.cellClicked.connect(self.store_selected_row)
        self.selected_keypoint_row = None  # To store index
        self.keypoint_list.setHorizontalHeaderLabels(["#", "Label", "Color", "Visible", "Missing"])
        self.keypoint_list.setColumnWidth(0, 40)
        self.keypoint_list.setColumnWidth(1, 120)
        self.keypoint_list.setColumnWidth(2, 60)
        self.keypoint_colors = {}  # Track color per index
        self.fast_label_checkbox.setChecked(True)
        self.fast_label_checkbox.setToolTip(
            "After placing a keypoint, advance once. Placement stops after the final point."
        )
        self.auto_sync_checkbox.setChecked(True)
        self.save_list_button.clicked.connect(self.save_keypoint_list)
        self.load_list_button.clicked.connect(lambda: self.load_keypoint_list_from_path())
        self._compact_keypoint_panel()
        self._hide_legacy_keypoint_tab()

        # --- Voice controls and UI translation ---
        self.voice_mode_checkbox.stateChanged.connect(self.toggle_voice_mode)
        self.language_dropdown = self._ensure_language_dropdown()

        self.language_dropdown.clear()
        languages = {   "English": "en",
                        "Français": "fr",
                        "Español": "es",
                        "Deutsch": "de",
                        "Italiano": "it",
                        "Português (Brasil)": "pt",
                        "Русский": "ru",
                        "中文 (简体)": "zh-CN",
                        "日本語": "ja",
                        "한국어": "ko",
                        "Türkçe": "tr",
                        "العربية": "ar",
                        "हिन्दी": "hi",
                        "Українська": "uk",
                        "Polski": "pl",
                        "Nederlands": "nl",
                        "Tiếng Việt": "vi",
                        "ไทย": "th",
                        "Indonesian": "id"
                    }

        for lang_display, lang_code in languages.items():
            self.language_dropdown.addItem(lang_display, lang_code)
        self.ui_translation = TranslationController(
            self, Path(__file__).resolve().parent / "translations",
            online=bool(self.settings.get("translationOnlineEnabled", True)),
        )
        self.language_dropdown.currentIndexChanged.connect(self.change_language)

        saved_language = self.settings.get("languageCode", "en")
        saved_language_index = self.language_dropdown.findData(saved_language)
        if saved_language_index > 0:
            self.language_dropdown.setCurrentIndex(saved_language_index)
        else:
            self.change_language(0)

        self.restore_persistent_ui_state()
        # Remembering the last directory is useful for file dialogs, but a
        # fresh DarkFusion window must never reopen it in the GraphicsView.
        # A dataset becomes active only after the user explicitly chooses one.
        self._graphics_dataset_explicitly_opened = False
        self.image_directory = None
        self.last_image_directory = None
        self.image_files = []
        self.filtered_image_files = []
        self.images = []
        self.current_file = None
        self.current_img_index = 0
        self.current_image_index = -1
        if getattr(self, "graphics_scene", None) is not None:
            self.graphics_scene.clear()
        self.setup_persistent_settings_bindings()
        QTimer.singleShot(0, self._connect_ui_scale_screen_tracking)
        self.collapsible_sections = CollapsibleSectionsController(self)
        self.show()
        # Let the main window finish drawing before model I/O and CUDA setup.
        # Offscreen runs are smoke tests and should not allocate production models.
        if os.environ.get("QT_QPA_PLATFORM", "").strip().lower() != "offscreen":
            QTimer.singleShot(2500, self.start_sam3_startup_warmup)

    # -----------------------------
    # MAIN WINDOW SETUP HELPERS
    # -----------------------------

    def update_progress(self, progress):
        self.set_main_progress(progress, 100)

    def setup_tab_workflow(self):
        """Put major tabs in the order most dataset sessions naturally use."""
        main_tabs = getattr(self, "tabWidget", None)
        labeling_tabs = getattr(self, "tabWidget_3", None)
        video_tabs = getattr(self, "tabWidget_4", None)
        dataset_tabs = getattr(self, "tabWidget_6", None)
        training_tabs = getattr(self, "tabWidget_5", None)

        self.rename_tab(main_tabs, "Video", "Collect")
        self.rename_tab(main_tabs, "Video Tools", "Collect")
        self.rename_tab(main_tabs, "Dataset Tools", "Prepare / Split")
        self.rename_tab(main_tabs, "Labeling", "Label")
        self.rename_tab(main_tabs, "Training", "Train / Export")
        self.rename_tab(main_tabs, "Train", "Train / Export")
        self.rename_tab(main_tabs, "CFG Editor", "Advanced")
        self.rename_tab(main_tabs, "CFG / Anchors", "Advanced")
        self.reorder_tabs_by_text(
            main_tabs,
            ["Collect", "Label", "Review", "Prepare / Split", "Train / Export", "Advanced"],
        )
        if main_tabs is not None:
            self.set_tab_tooltips(
                main_tabs,
                {
                    "Collect": "Download videos, capture/extract frames, or bring image sources into the project.",
                    "Label": "Draw, edit, auto-label, and tune the active annotation view.",
                    "Review": "Filter, inspect, reveal, move, or clear reviewed images.",
                    "Prepare / Split": "Analyze, clean, move, convert, augment, tile, and split datasets.",
                    "Train / Export": "Train Ultralytics models and export model formats.",
                    "Advanced": "CFG editing and anchor tools.",
                },
            )
            collect_index = self.tab_index_by_text(main_tabs, ("Collect",))
            main_tabs.setCurrentIndex(collect_index if collect_index >= 0 else 0)
            main_tabs.currentChanged.connect(self.on_main_workflow_tab_changed)

        self.rename_tab(labeling_tabs, "Extras", "ROI / Filters")
        self.rename_tab(labeling_tabs, "ROI / Extras", "ROI / Filters")
        self.rename_tab(labeling_tabs, "Detection & Display", "Display")
        self.rename_tab(video_tabs, "Playback & Extraction", "Playback / Extract")
        self.rename_tab(video_tabs, "Download", "Download Videos")
        if video_tabs is not None:
            video_tabs.setCurrentIndex(0)
        self.rename_tab(dataset_tabs, "Prepare Dataset", "Prepare")
        self.rename_tab(dataset_tabs, "Augmentation", "Augment / Convert")
        self.rename_tab(dataset_tabs, "Augment / Convert", "Augment")
        if training_tabs is not None and not getattr(self, "_ultralytics_tab_click_connected", False):
            training_tabs.tabBarClicked.connect(self.on_training_workflow_tab_clicked)
            self._ultralytics_tab_click_connected = True

    def on_training_workflow_tab_clicked(self, index):
        training_tabs = getattr(self, "tabWidget_5", None)
        ultralytics_tab = getattr(self, "tab_17", None)
        if training_tabs is None or ultralytics_tab is None:
            return
        if training_tabs.widget(int(index)) is ultralytics_tab:
            self.open_training_evaluator_dialog()

    def on_main_workflow_tab_changed(self, index):
        main_tabs = getattr(self, "tabWidget", None)
        if main_tabs is None:
            return

        current_text = self._ui_tab_source_text(main_tabs, index).strip()
        if getattr(self, "collect_action_footer", None) is not None:
            self._update_collect_action_footer()
        # Workflow tabs are controls, not ownership of a running capture job.
        # Stop/source changes/shutdown still perform the full runtime cleanup.
        if current_text == "Collect":
            self.restore_collect_video_surface()
            if getattr(self, "_extraction_progress_state", None):
                self.update_extraction_progress()

    def rename_tab(self, tab_widget, old_text, new_text):
        if tab_widget is None:
            return

        for index in range(tab_widget.count()):
            if self._ui_tab_source_text(tab_widget, index) == old_text:
                tab_widget.setTabText(index, new_text)
                return

    def tab_index_by_text(self, tab_widget, names):
        if tab_widget is None:
            return -1

        normalized_names = {
            str(name).strip().lower()
            for name in names
            if str(name).strip()
        }
        for index in range(tab_widget.count()):
            if self._ui_tab_source_text(tab_widget, index).strip().lower() in normalized_names:
                return index
        return -1

    def set_tab_tooltips(self, tab_widget, tooltip_by_text):
        if tab_widget is None:
            return

        for index in range(tab_widget.count()):
            text = tab_widget.tabText(index)
            tooltip = tooltip_by_text.get(text)
            if tooltip:
                tab_widget.setTabToolTip(index, tooltip)

    def reorder_tabs_by_text(self, tab_widget, desired_order):
        if tab_widget is None:
            return

        desired_index = 0
        for title in desired_order:
            current_index = -1
            for index in range(tab_widget.count()):
                if self._ui_tab_source_text(tab_widget, index) == title:
                    current_index = index
                    break

            if current_index < 0 or current_index == desired_index:
                desired_index += 1
                continue

            page = tab_widget.widget(current_index)
            icon = tab_widget.tabIcon(current_index)
            tooltip = tab_widget.tabToolTip(current_index)
            whats_this = tab_widget.tabWhatsThis(current_index)
            enabled = tab_widget.isTabEnabled(current_index)

            tab_widget.removeTab(current_index)
            tab_widget.insertTab(desired_index, page, icon, title)
            tab_widget.setTabToolTip(desired_index, tooltip)
            tab_widget.setTabWhatsThis(desired_index, whats_this)
            tab_widget.setTabEnabled(desired_index, enabled)
            desired_index += 1

    def setup_control_copy(self):
        """Normalize visible labels and tooltips without changing control behavior."""
        copy = {
            "classes_dropdown": (
                None,
                "Choose the active annotation class. Hotkeys can be adjusted in Settings.",
            ),
            "segmentation_mode": (
                None,
                "Choose the annotation type: boxes, OBB, segmentation, or keypoints.",
            ),
            "styleComboBox": (None, "Switch the application theme."),
            "gif_change": (None, "Switch the animated status visual."),
            "language_dropdown": (None, "Switch the interface language."),
            "next_button": ("Next", "Open the next image, or the next source video frame when labeling a video."),
            "previous_button": ("Prev", "Open the previous image, or the previous source video frame when labeling a video."),
            "delete_button": ("Delete", "Delete the current image and its matching label file."),
            "network_height": (None, "Network input height. Ultralytics receives imgsz as [H, W]."),
            "network_width": (None, "Network input width. Ultralytics receives imgsz as [H, W]."),
            "confidence_threshold_spinbox": (None, "Minimum detection confidence for OpenCV DNN labels."),
            "nms_threshold_spinbox": (None, "Overlap threshold used to suppress duplicate OpenCV DNN boxes."),
            "box_size": (None, "Minimum label width and height in image pixels, independent of zoom, for drawing, saving, and auto-labeling."),
            "max_label": (None, "Maximum label size as a percent of the image for drawing, saving, and auto-labeling."),
            "timmer_speed": (None, "Delay between images while Auto Scan is active."),
            "edge_slider_min": (None, "Lower Canny edge threshold for snap/edge preview."),
            "edge_slider_max": (None, "Upper Canny edge threshold for snap/edge preview."),
            "grey_scale_slider": (None, "Adjust grayscale preview brightness."),
            "shade_slider": (None, "Adjust annotation shading opacity."),
            "dot_size_slider": (None, "Adjust displayed pose keypoint size."),
            "font_size_slider": (None, "Adjust annotation label text size."),
            "x_spin_slider": (None, "Move ROI 1 horizontally."),
            "y_spin_slider": (None, "Move ROI 1 vertically."),
            "width_spin_slider": (None, "Set ROI 1 width."),
            "height_spin_slider": (None, "Set ROI 1 height."),
            "x_spin_slider_2": (None, "Move ROI 2 horizontally."),
            "y_spin_slider_2": (None, "Move ROI 2 vertically."),
            "width_spin_slider_2": (None, "Set ROI 2 width."),
            "height_spin_slider_2": (None, "Set ROI 2 height."),
            "monitor": (None, "Select the monitor used for screen capture."),
            "custom_size_checkbox": ("Resize Frames", "Resize video/display/extracted frames to Frame W x H."),
            "custom_frames_checkbox": ("Frame Step", "Extract every Nth frame."),
            "custom_input": (None, "Frame step value used when Frame Step is enabled."),
            "tracking_enabled_checkbox": ("Tracking", "Track objects across video frames."),
            "inference_checkbox": ("Detections", "Show live detections during video playback."),
            "crop_images_checkbox": ("Crop Frames", "Crop extracted frames before saving."),
            "height_box": (None, "Frame transform height for Crop/Resize as W x H. Resize defaults to Network H."),
            "width_box": (None, "Frame transform width for Crop/Resize as W x H. Resize defaults to Network W."),
            "video_slider": (None, "Video timeline. Drag left or right to seek."),
            "batch_input_spinbox": (None, "Batch size used during model export."),
            "conf_spinbox": (None, "Export confidence threshold."),
            "iou_spinbox": (None, "Export IoU threshold for NMS."),
            "process_folder_btn": ("Run Selected SAM3 Tools", "Run only the selected SAM3 conversion or augmentation tools."),
            "screen_update": ("Show Preview", "Show SAM3 previews as the batch tool moves through the dataset."),
            "overwrite_var": ("Auto-Fit Labels", "Use the SAM3 mask to tighten existing label geometry."),
            "segmentation_checkbox": ("Seg Labels", "Create YOLO segmentation labels in a Segmented subfolder."),
            "obb_checkbox": ("OBB Labels", "Create oriented bounding box labels in an OBB subfolder."),
            "noise_remove_checkbox": ("Clean BG", "Save label-focused variants with background noise softened."),
            "shadow_var": ("Shadow / Highlight", "Save one shadow or highlight variant per selected image, using existing SAM3 object masks and unchanged annotations."),
            "copy_paste_checkbox": ("Copy/Paste", "Create copy-paste augmentation samples from existing labels."),
            "object_detail_checkbox": ("Object Detail", "Create subtle natural-edge and local-contrast variants only inside labeled SAM3 masks."),
            "sam_object_clarity_checkbox": ("Object Clarity", "Create Real-ESRGAN variants inside labeled SAM3 masks. Restored detail is estimated; image size and label geometry are preserved."),
            "object_clarity_augmentation_checkbox": ("Object Clarity", "Create Real-ESRGAN variants inside existing boxes, polygons or OBBs. Restored detail is estimated; image size and label geometry are preserved."),
            "extra_label_dialog_button": ("Extra Label...", "Open nested-label geometry and SAM3 nested-label settings."),
            "enhance_image_checkbox": ("Enhance", "Enhance the current image display with super-resolution when available, otherwise clarity sharpening."),
            "tile_keep_empty_checkbox": ("Keep Empty", "Keep background-only tiles. Turn off to avoid flooding training with empty crops."),
            "tile_pad_edges_checkbox": ("Pad Edges", "Reflect-pad edge tiles so every output keeps the requested tile size."),
            "keep_duplicate_labels_checkbox": (
                "Keep Duplicates",
                "Training only: disable Ultralytics exact duplicate-label row removal. Use only for intentional repeated labels.",
            ),
            "crop_button": ("Run Augmentation", "Create the selected augmentation outputs."),
            "make_copy": ("Copy Originals", "Create unchanged image and label copies."),
            "flip_checkbox": ("Mirror", "Horizontally flip images and supported YOLO labels."),
            "motion_blur": ("Motion Blur", "Apply a horizontal motion blur effect."),
            "flash_checkbox": ("Flash", "Apply a bright exposure/flash effect."),
            "smoke_checkbox": ("Smoke", "Apply a dense gray tactical smoke haze."),
            "glass_checkbox": ("Glass", "Apply a glass/refraction effect."),
            "sight_picture": ("Sight Overlay", "Overlay a bottom-attached foreground gun/sight asset. It can shift left or right but stays anchored to the image bottom."),
            "brightness_contrast_checkbox": ("Light", "Randomly adjust brightness and contrast for YOLO lighting generalization."),
            "hsv_jitter_checkbox": ("HSV", "Randomly shift hue, saturation, and value without changing labels."),
            "gaussian_noise_checkbox": ("Noise", "Add light sensor/compression noise without changing labels."),
            "jpeg_compression_checkbox": ("JPEG", "Simulate stream/upload JPEG compression artifacts without changing labels."),
            "augmentation_percent_spinbox": (None, "Percent used for normal augmentation, SAM3 Copy/Paste, Clean BG, Shadow / Highlight, Object Detail, Object Clarity, and Negatives. SAM3 Auto-Fit, Seg, and OBB run on all images."),
        }

        titles = {
            "groupBox_17": "SAM3 Batch Tools",
        }

        for object_name, (text, tooltip) in copy.items():
            widget = getattr(self, object_name, None)
            if widget is None:
                continue

            if text is not None and hasattr(widget, "setText"):
                widget.setText(text)
            if tooltip:
                widget.setToolTip(tooltip)

        for object_name, title in titles.items():
            widget = getattr(self, object_name, None)
            if widget is not None and hasattr(widget, "setTitle"):
                widget.setTitle(title)

    def setup_quick_toggle_buttons(self):
        """Make the main quick toggles read like compact toolbar controls."""
        toggle_labels = {
            "zoom_lock": (
                "Zoom Lock",
                "Keep the current zoom and view position while moving between images.",
            ),
            "image_quality_checkbox": (
                "Quality",
                "Include blur, exposure, contrast, and resolution checks in dataset statistics.",
            ),
            "outline_Checkbox": (
                "Snapping",
                "Use edge-aware snapping while drawing or editing annotations.",
            ),
            "rapid_del_checkbox": (
                "Rapid Del",
                "Hold right-click over annotations to remove them repeatedly. The image Delete button always removes one image per click.",
            ),
            "auto_scan_checkbox": (
                "Auto Scan",
                "Repeat Next or Previous automatically, using your enabled tools and settings. "
                "Waits for each frame to finish before advancing again.",
            ),
            "voice_mode_checkbox": (
                "Voice",
                "Turn on continuous spoken class selection.",
            ),
            "enhance_image_checkbox": (
                "Enhance",
                "Enhance the current image display with super-resolution when available, otherwise clarity sharpening.",
            ),
        }

        for object_name, (label, tooltip) in toggle_labels.items():
            widget = getattr(self, object_name, None)
            if widget is None:
                continue

            widget.setText(label)
            widget.setToolTip(tooltip)
            widget.setStyleSheet("")
            widget.setCursor(Qt.PointingHandCursor)
            widget.setMinimumHeight(26)
            widget.setMaximumHeight(32)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    def set_main_progress(self, value=0, maximum=100):
        """Update the main progress bar with safe bounds and a consistent 0-based range."""
        if getattr(self, "_review_label_clear_active", False):
            return  # Review-position timers must not overwrite Clear Labels progress.
        if not hasattr(self, "label_progress") or self.label_progress is None:
            return

        try:
            maximum = max(1, int(maximum))
            value = max(0, min(int(value), maximum))
            self.label_progress.setRange(0, maximum)
            self.label_progress.setFormat("%p%")
            self.label_progress.setValue(value)
        except Exception as e:
            logger.debug(f"Could not update main progress bar: {e}")

    def update_dataset_progress(self):
        """Use the main progress bar as a quiet position indicator while reviewing images."""
        worker = self.__dict__.get("_review_filter_worker")
        if worker is not None:
            try:
                if worker.isRunning():
                    return  # Navigation must not erase an active filter/search's progress.
            except RuntimeError:
                pass
        image_files = getattr(self, "filtered_image_files", None) or getattr(self, "image_files", [])
        total_images = len(image_files)
        if total_images <= 0:
            self.set_main_progress(0, 100)
            return

        current_index = getattr(self, "current_img_index", 0)
        current_file = getattr(self, "current_file", None)
        try:
            current_index = int(current_index)
        except Exception:
            current_index = 0

        if current_file and not (0 <= current_index < total_images and self.normalize_path(image_files[current_index]) == self.normalize_path(current_file)):
            try:
                current_index = image_files.index(current_file)
            except ValueError:
                current_index = max(0, min(current_index, total_images - 1))

        self.set_main_progress(current_index + 1, total_images)

    def _reset_navigation_state(self):
        """
        Emergency recovery: Reset navigation busy flag if stuck.
        Use this if navigation stops responding after dataset analysis or deletion.
        """
        was_busy = getattr(self, "navigation_busy", False)
        self.navigation_busy = False
        logger.info(f"Navigation state reset. Was busy: {was_busy}")
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage("Navigation state reset. Try navigating again.", 3000)
        return not was_busy

    # -----------------------------
    # APP SHELL / STATUS HELPERS
    # -----------------------------

    def update_icon(self):
        self.current_icon_index = (self.current_icon_index + 1) % len(self.icons)
        self.setWindowIcon(self.icons[self.current_icon_index])

    def _setup_cuda_monitor_button(self):
        button = getattr(self, "cuda_monitor_button", None)
        if button is None:
            button = QtWidgets.QPushButton("System Monitor", self)
            button.setObjectName("cuda_monitor_button")
            layout = getattr(self, "gridLayout_16", None)
            if layout is not None:
                layout.addWidget(button, 0, 0, 1, 2)
            self.cuda_monitor_button = button

        button.setText("System Monitor")
        button.setToolTip("Open the compact system resource monitor.")

        try:
            button.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        button.clicked.connect(self.open_cuda_monitor)

    def open_cuda_monitor(self):
        dialog = getattr(self, "cuda_monitor_dialog", None)
        if dialog is None:
            dialog = SystemMonitorDialog(self)
            self.cuda_monitor_dialog = dialog

        dialog.refresh()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    from darkfusion_system_monitor import (
        gather_metrics,
    )

    def start_gathering_metrics(self):
        existing_thread = getattr(self, "metrics_thread", None)
        if existing_thread is not None and existing_thread.is_alive():
            return

        self.metrics_thread_active = True
        self.metrics_thread = Thread(target=self.gather_metrics, daemon=True)
        self.metrics_thread.start()

    # -----------------------------
    # SETTINGS AND PROJECT STATE
    # -----------------------------

    def openSettingsDialog(self):
        logger.info("Settings button was clicked.")
        settingsDialog = SettingsDialog(self)
        try:
            settingsDialog.exec_()
        finally:
            settingsDialog.deleteLater()

    from darkfusion_canvas import (
        refresh_annotation_drawing_styles,
        current_keybind_class_names,
        keybind_entries,
        keybind_matches_setting,
        keybind_setting_for_text,
        class_name_for_keybind,
        _navigation_direction_sign,
        _run_navigation_offset,
        handle_navigation_key_press,
        handle_navigation_key_release,
        run_navigation_keybind,
        run_mode_keybind,
        select_class_for_keybind,
    )

    from darkfusion_settings import (
        app_settings_path,
        dialog_history_key,
        dialog_start_directory,
        remember_dialog_selection,
        project_settings_path,
        load_project_settings,
        split_settings_for_storage,
        loadSettings,
        defaultSettings,
        queue_settings_save,
        _qbytearray_to_setting,
        _setting_to_qbytearray,
        _widget_int_value,
        _widget_float_value,
        _set_widget_int_value,
        _set_widget_float_value,
        _set_widget_checked_setting,
        _widget_text_value,
        _set_widget_text_value,
        _normalize_setting_path,
        _combo_text_value,
        _set_combo_text_value,
        initialize_floating_point_mode_from_settings,
        collect_persistent_ui_state,
        snapshot_classes_dropdown_checks_as_bools,
        current_roi_settings,
        current_layout_screen_metrics,
        _saved_layout_matches_current_screen,
        _fit_window_to_available_screen,
        _normalize_default_dock_sizes,
        restore_persistent_ui_state,
        restore_model_path_settings,
        restore_roi_settings,
        restore_project_ui_state,
        restore_annotation_mode_setting,
        restore_active_class_setting,
        restore_class_visibility_setting,
    )

    from darkfusion_dataset_review import restore_review_filter_setting

    from darkfusion_settings import (
        setup_persistent_settings_bindings,
        saveSettings,
        save_project_settings_for_directory,
        activate_project_settings,
    )

    def normalize_path(self, path):
        return normalize_dataset_path(path)

    from darkfusion_canvas import (
        get_saved_last_image_for_dir,
        save_current_dataset_position,
        restore_current_image_after_load,
    )

    def set_sound(self, sound_path):
        self.sound_player.setMedia(
            QMediaContent(QUrl.fromLocalFile(sound_path)))
        self.sound_player.play()

    # -----------------------------
    # TRANSLATION AND VOICE CONTROLS
    # -----------------------------

    def _refresh_runtime_translations(self):
        controller = getattr(self, "ui_translation", None)
        if controller is not None:
            controller.enqueue(self, tree=True)

    def _schedule_translation_refresh(self):
        # The translation controller observes individual widgets as they appear.
        # Kept for callers that explicitly rebuild a complete window.
        self._refresh_runtime_translations()

    def change_language(self, index):
        if index < 0:
            return
        code = self.language_dropdown.itemData(index) or "en"
        self.settings["languageCode"] = code
        self.queue_settings_save()
        self.ui_translation.set_language(code)

    def _ui_tab_source_text(self, tabs, index):
        controller = getattr(self, "ui_translation", None)
        return controller.source_text(tabs, "tab", index) if controller else tabs.tabText(index)

    from darkfusion_class_editor import toggle_voice_mode

    from darkfusion_class_editor import apply_voice_class_selection, report_voice_mode_failure

    from darkfusion_class_editor import continuous_voice_loop

    def closeEvent(self, event):
        """
        Ensure annotations are saved safely before application closes.
        """
        camera_discovery = getattr(self, "_camera_discovery_worker", None)
        if self.qthread_is_running(camera_discovery):
            self._camera_discovery_shutting_down = True
            camera_discovery.requestInterruption()
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        bulk_move = getattr(self, "_bulk_move_worker", None)
        if bulk_move is not None and self.qthread_is_running(bulk_move):
            bulk_move.cancel()
            self.statusBar().showMessage("Finishing current file moves before closing...")
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        if getattr(self, "_review_label_clear_active", False):
            worker = getattr(self, "_review_label_clear_worker", None)
            if worker is not None:
                worker.requestInterruption()
            self.statusBar().showMessage("Finishing the current label file before closing...")
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        class_worker = getattr(getattr(self, "class_picker_dialog", None), "_label_remap_worker", None)
        if self.qthread_is_running(class_worker):
            self.statusBar().showMessage("Finishing dataset class changes before closing...")
            event.ignore()
            QTimer.singleShot(100, self.close)
            return

        review_threads = [worker for worker in getattr(self, "_review_similarity_threads", [])
                          if self.qthread_is_running(worker)]
        if review_threads:
            for worker in review_threads:
                worker.cancel()
            self.statusBar().showMessage("Stopping similarity work before closing...")
            event.ignore()
            QTimer.singleShot(100, self.close)
            return

        if getattr(self, "_sam3_batch_active", False):
            if not getattr(self, "_sam3_cancel_requested", False):
                self.request_stop_sam3_batch()
            self.statusBar().showMessage("Finishing the current SAM3 operation before closing...")
            event.ignore()
            QTimer.singleShot(100, self.close)
            return

        model_loader = getattr(self, '_inference_model_load_worker', None)
        if model_loader is not None:
            model_loader.stop()
            self.statusBar().showMessage('Finishing model initialization before closing…')
            event.ignore()
            QTimer.singleShot(100, self.close)
            return

        self._sam3_propagation_shutting_down = True
        self._propagation_stop_requested = True
        propagation_workers = (
            getattr(self, "_sam3_propagation_warmup_worker", None),
            getattr(self, "_adjacent_propagation_worker", None),
            getattr(self, "_propagation_range_worker", None),
        )
        if any(self.qthread_is_running(worker) for worker in propagation_workers) or getattr(
            self, "_sam3_propagation_batch_active", False
        ):
            # CUDA calls cannot be forcibly interrupted safely. Keep the event
            # loop alive until they finish instead of blocking it in wait().
            for worker in propagation_workers:
                if self.qthread_is_running(worker):
                    worker.requestInterruption()
            self.statusBar().showMessage("Finishing propagation before closing...")
            event.ignore()
            QTimer.singleShot(100, self.close)
            return

        try:
            if (
                hasattr(self, "save_bounding_boxes")
                and hasattr(self, "current_file")
                and self.current_file
                and not self.is_placeholder_file(self.current_file)
            ):
                scene = self.screen_view.scene() if hasattr(self, "screen_view") else None

                if scene:
                    logger.info(f"Saving before close: {self.current_file}")

                    self.save_bounding_boxes(
                        self.current_file,
                        scene.width(),
                        scene.height()
                    )

                if hasattr(self, "_finalize_sparse_video_working_frame"):
                    self._finalize_sparse_video_working_frame()

        except Exception as e:
            logger.error(f"Failed saving during closeEvent: {e}")

        try:
            self.save_current_dataset_position()
            self.saveSettings()
        except Exception as e:
            logger.error(f"Failed saving persistent UI state during closeEvent: {e}")

        self.cleanup_runtime_resources()
        event.accept()

    def _stop_and_delete_timer(self, attr_name):
        timer = getattr(self, attr_name, None)
        if timer is None:
            return

        try:
            timer.stop()
        except RuntimeError:
            pass

        try:
            timer.timeout.disconnect()
        except (TypeError, RuntimeError):
            pass

        try:
            timer.deleteLater()
        except RuntimeError:
            pass

        setattr(self, attr_name, None)

    from darkfusion_video_inference import (
        _release_capture,
        stop_live_collect_runtime,
        _close_mss_resources,
    )

    def _cleanup_worker_reference(self, attr_name, worker=None):
        worker = worker or getattr(self, attr_name, None)
        if worker is None:
            return

        try:
            if hasattr(worker, "isRunning") and worker.isRunning():
                QTimer.singleShot(100, lambda: self._cleanup_worker_reference(attr_name, worker))
                return
        except RuntimeError:
            pass

        try:
            worker.deleteLater()
        except RuntimeError:
            pass

        if getattr(self, attr_name, None) is worker:
            setattr(self, attr_name, None)

    def _cleanup_download_thread(self, thread):
        try:
            if thread in self.download_threads:
                self.download_threads.remove(thread)
        except Exception:
            pass

        try:
            thread.deleteLater()
        except RuntimeError:
            pass

    def cleanup_runtime_resources(self):
        self._clarity_augmentation_shutting_down = True
        cancel = getattr(self, "_clarity_augmentation_cancel", None)
        if cancel is not None:
            cancel.set()
        executor = getattr(self, "_clarity_augmentation_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        clarity = getattr(self, "_object_clarity_controller", None)
        if clarity is not None:
            clarity.close()
        source = getattr(self, "_editor_image_source", None)
        if source is not None:
            source.close()
        self.metrics_thread_active = False
        self.voice_thread_active = False
        self._propagation_stop_requested = True

        augmentation_job = getattr(self, "_augmentation_job", None)
        if augmentation_job is not None:
            augmentation_job["cancel_requested"] = True
            self._augmentation_job = None
            logger.info("Canceled the active augmentation job during application shutdown.")

        self.stop_live_collect_runtime(reset_input=True)

        completion = getattr(self, "_extraction_completion_dialog", None)
        if completion is not None:
            completion.close()
            self._extraction_completion_dialog = None

        dialog = getattr(self, "cuda_monitor_dialog", None)
        if dialog is not None:
            try:
                dialog.close_for_cleanup()
            except RuntimeError:
                pass
            self.cuda_monitor_dialog = None

        # Clean up downloaded temp files to prevent disk space leak
        downloaded_files = getattr(self, "_downloaded_temp_files", set())
        if downloaded_files:
            for file_path in downloaded_files:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.debug(f"Cleaned up temp video file: {file_path}")
                except Exception as e:
                    logger.debug(f"Failed to cleanup temp file {file_path}: {e}")
            self._downloaded_temp_files = set()

        for attr_name in (
            "timer2",
            "_overlay_timer",
            "_thumbnail_timer",
            "process_timer",
            "next_timer",
            "prev_timer",
            "icon_timer",
            "timer",
        ):
            self._stop_and_delete_timer(attr_name)

        self._release_capture()
        self._close_mss_resources()
        self.stop_existing_workers(include_standalone_moves=True)

        self._sam3_propagation_shutting_down = True
        self._sam3_propagation_release_pending = True
        self._release_sam_video_propagation_runtime()

        export_worker = getattr(self, "training_export_worker", None)
        if export_worker is not None:
            try:
                if export_worker.isRunning():
                    logger.info("Waiting for the active Ultralytics export before shutdown...")
                    export_worker.wait()
            except RuntimeError:
                pass
            self.training_export_worker = None

        if hasattr(self, "threadpool") and self.threadpool is not None:
            self.threadpool.clear()

        for attr_name in ("metrics_thread", "voice_thread"):
            thread = getattr(self, attr_name, None)
            if thread is None:
                continue

            try:
                if thread.is_alive():
                    thread.join(timeout=0.2)
            except RuntimeError:
                pass

            try:
                if not thread.is_alive():
                    setattr(self, attr_name, None)
            except RuntimeError:
                setattr(self, attr_name, None)

        for thread in list(getattr(self, "download_threads", [])):
            try:
                if not thread.isRunning():
                    self._cleanup_download_thread(thread)
            except RuntimeError:
                self._cleanup_download_thread(thread)

        if hasattr(self, "movie") and self.movie is not None:
            self.movie.stop()

        if hasattr(self, "sound_player") and self.sound_player is not None:
            self.sound_player.stop()

        self._super_resolution_model = None
        self._super_resolution_model_path = None
        self._clarity_clahe = None

        if hasattr(self, "logger") and hasattr(self, "label_log_handler"):
            try:
                self.logger.removeHandler(self.label_log_handler)
            except Exception:
                pass

    #points.json and keypoint list management

    from darkfusion_class_editor import (
        ensure_points_json_exists,
        add_keypoint_list_item,
        clear_keypoint_highlight,
        highlight_keypoint_row,
        load_keypoint_list_from_path,
        load_keypoint_list_from_data,
        keypoint_row_is_ignored,
        set_keypoint_row_manual_ignore,
        keypoint_row_is_manually_ignored,
        next_labelable_keypoint_row,
        selected_keypoint_drawer,
        update_keypoint_active_status,
        restart_selected_box_keypoints,
        set_active_keypoint_row,
        mark_keypoint_row_ignored,
        sync_ignore_visibility,
        store_selected_row,
        lock_keypoint_checkboxes,
        rebind_keypoint_checkbox_signals,
        reset_keypoint_checkboxes,
        remove_keypoint_list_item,
        get_points_json_path,
        remove_keypoint_index_from_points_json,
        remove_keypoint_index_from_current_scene,
        remove_keypoint_index_from_dataset_labels,
        remove_keypoint_index_from_label_line,
        collect_dataset_label_files_for_keypoint_migration,
        sync_all_keypoint_drawers_to_table,
        reindex_keypoint_table,
        save_keypoint_list_to_path,
        autosave_keypoint_list_json,
        save_keypoint_list,
        update_keypoint_checkboxes_from_labels,
        update_keypoint_checkboxes_from_points,
    )

    from darkfusion_canvas import (
        update_shading_opacity,
        update_font_size_display,
        on_annotation_mode_user_selected,
        on_segmentation_mode_changed,
    )

    from darkfusion_class_editor import (
        load_skeleton_from_points_json_data,
    )

    from darkfusion_canvas import (
        current_mode,
        is_boxes_mode,
        is_segmentation_mode,
        is_keypoint_mode,
        is_obb_mode,
    )

    from darkfusion_class_editor import (
        _keypoint_label_key,
        _format_keypoint_label,
        _sanitize_kpt_shape,
        _sanitize_skeleton_pairs,
        _infer_skeleton_from_keypoint_labels,
        _sanitize_flip_idx,
        _infer_flip_idx_from_labels,
        _rgba_from_value,
        _keypoint_style_snapshot,
        _build_pose_list_data,
        _extract_keypoint_labels_from_value,
        _extract_pose_labels_from_mapping,
        _extract_pose_skeleton_from_mapping,
        _iter_pose_metadata_mappings,
        _read_pose_yaml_candidate,
        extract_pose_schema_from_current_model,
        sync_pose_list_from_current_model,
        pose_keypoint_color_qcolor,
        pose_keypoint_color_bgr,
        pose_list_is_active_for_current_model,
        ensure_pose_list_for_output_dir,
    )

    from darkfusion_video_inference import (
        redraw_current_video_overlay,
    )

    from darkfusion_autolabel import (
        annotation_mode_for_model_task,
        _task_from_mapping,
        detect_loaded_model_task,
        exported_model_task_hint,
        create_ultralytics_model,
        normalize_inference_backend,
        inference_backend_for_model,
        onnx_execution_provider,
        inference_model_identity,
        is_darkfusion_onnx_model,
        create_inference_model,
        active_inference_backend_label,
        reload_selected_inference_model,
        auto_set_annotation_mode_for_model,
    )

    from darkfusion_canvas import (
        edit_mode_active,
    )

    #split data ui

    def launch_split_data_ui(self):
        dialog = DatasetSplitDialog(self)
        dialog.exec_()

    from darkfusion_tiling import (
        collect_current_tile_images,
        tile_overlap_percent,
        tile_min_visible_percent,
        selected_tile_image_format,
        selected_tile_layout_mode,
        tile_keep_empty_enabled,
        tile_pad_edges_enabled,
        crop_images,
        on_tile_progress,
        on_tile_completed,
        on_tile_failed,
        on_tile_worker_finished,
    )

    from darkfusion_canvas import (
        _selected_roi_id,
        _toggle_roi_master,
        _on_roi_selected,
        _sync_roi_controls,
        _roi_clarity_changed,
        _update_roi_clarity_badge,
        _update_roi_edge_overlay,
        _init_roi_default_size,
        _on_roi_slider_changed,
        _apply_pending_roi_changes,
        _redraw_video_frame_for_roi,
        toggle_roi,
        validate_roi,
        filter_bboxes,
        get_dynamic_roi_polygons,
        filter_bboxes_in_file,
        _roi_bounds_size,
        _clamp_roi_rect,
        _sync_roi_slider_limits,
        draw_roi,
        remove_roi,
        update_roi,
        on_frame_change,
    )

    #sahi settings see

    from darkfusion_autolabel import (
        showSahiSettings,
    )

    def resetLayout(self):
        # Restore the responsive default instead of replaying geometry captured
        # on a different monitor or under a different Windows DPI setting.
        self.restoreState(self.initial_state)
        self._fit_window_to_available_screen(center=True)
        self._normalize_default_dock_sizes()
        self.apply_review_splitter_sizes()
        if hasattr(self, "settings"):
            self.settings["windowGeometry"] = ""
            self.settings["windowState"] = ""
            self.settings["layoutStateVersion"] = CURRENT_UI_LAYOUT_VERSION
            self.settings["layoutScreenMetrics"] = self.current_layout_screen_metrics()
            self.settings["reviewSplitterSizes"] = []
            self.saveSettings()

    from darkfusion_dataset_review import (
        setup_review_panel_layout,
        apply_review_splitter_sizes,
        setup_review_view_actions,
        set_label_preview_visible,
    )

    from darkfusion_augmentations import (
        on_augmentation_percent_changed,
    )

    from darkfusion_dataset_workers import (
        on_clear_json_clicked,
    )

    from darkfusion_augmentations import (
        on_check_button_click,
        on_shadow_checkbox_click,
        on_obb_checkbox_checked,
        on_noise_remove_checked,
        on_copy_paste_checkbox_click,
        on_object_detail_checkbox_click,
        on_create_negatives_checkbox_click,
        on_segmentation_checkbox_checked,
    )

    # -----------------------------
    # SAM3 PROMPT SNAP HELPERS
    # -----------------------------
    # Legacy method name kept for existing callers; this loads Sam/sam3.pt.

    from darkfusion_sam import (
        ensure_sam_model_loaded,
        ensure_sam_semantic_predictor_loaded,
    )

    from darkfusion_propagation import (
        ensure_sam_video_predictor_loaded,
        _warm_sam3_propagation_runtime,
        _set_navigation_propagation_button_state,
        _start_sam3_propagation_preload,
        start_sam3_propagation_warmup,
        _on_sam3_propagation_warmup_completed,
        _on_sam3_propagation_warmup_finished,
        _release_sam_video_propagation_runtime,
    )

    from darkfusion_sam import (
        start_sam3_startup_warmup,
        _sam3_startup_warmup_capacity,
        _run_sam3_startup_warmup,
    )

    from darkfusion_class_editor import (
        get_current_class_name_safe,
    )

    from darkfusion_sam import (
        mask_to_obb_points,
        pad_bbox_xyxy,
        validate_sam_mask_for_bbox,
        polygon_guide_mask,
        select_sam_mask_for_polygon,
        _annotation_snap_image,
        sam_snap_class_polygon_to_mask,
        sam_snap_bbox_to_mask,
        mask_to_bbox_rect,
        mask_to_polygon_points,
    )

    # -----------------------------
    # SAM3 BATCH CORE HELPERS
    # -----------------------------

    from darkfusion_augmentations import (
        generate_unique_name,
        process_directory,
        _scan_images,
        ensure_rgb,
        _read_image_cv,
        _safe_empty_cache,
        _safe_sync_cuda,
        _resources_tight_for_sam_batch,
    )

    from darkfusion_sam import (
        _sam_device,
        _sam_half_enabled,
        _sam_imgsz,
        _sam_predict_kwargs,
        _sam_model_path,
        _get_sam_model,
    )

    from darkfusion_canvas import (
        get_label_file_safe,
        _read_label_lines,
        _valid_annotation_lines,
        _valid_annotation_lines_from_file,
        _write_label_lines,
    )

    from darkfusion_augmentations import (
        _prepare_dir,
    )

    # -----------------------------
    # YOLO FILE-BOUNDARY CONVERSION
    # -----------------------------

    from darkfusion_canvas import (
        _yolo_xywh_to_xyxy,
        _xyxy_to_yolo_xywh,
        _bbox_xyxy_from_box_obj,
        _normalize_polygon,
        _label_type,
        _polygon_label_parse_preference,
        _label_seed_bbox,
    )

    # -----------------------------
    # MASK / POLYGON CONVERSION HELPERS
    # -----------------------------

    from darkfusion_sam import (
        _clean_mask,
        _normalize_sam_mask_to_image,
        _mask_bbox,
        _bbox_area_xyxy,
        _bbox_intersection_area_xyxy,
        _bbox_center_distance_xyxy,
        _keypoint_bbox_xyxy_from_box_obj,
        _choose_keypoint_adjusted_bbox,
        polygon_to_mask,
        polygon_to_mask_shapely,
        obb_to_mask,
        get_polygon_from_mask,
        _contour_from_mask,
        _seg_line_from_mask,
        _obb_line_from_mask,
        get_sam_mask,
        generate_segmentation_mask,
        preprocess_image_for_segmentation,
        _bbox_to_mask,
        _process_label_to_mask,
        _update_box_obj_from_mask,
        _segment_to_box,
    )

    # -----------------------------
    # PREVIEW DRAWING / VISUALIZATION
    # -----------------------------

    from darkfusion_augmentations import (
        draw_mask_and_bbox,
        show_image,
        _note_sam3_output_failure,
        _save_sam3_output_pair,
        _save_sam3_label_updates,
        _show_sam3_batch_result,
        apply_noise_reduction,
        _render_object_appearance,
        create_shadow_image,
        _object_clarity_augmentation_changed,
        _object_clarity_augmentation_selected,
        _object_clarity_annotation_regions,
        _render_object_clarity_augmentation,
        create_object_clarity_image,
        create_object_detail_image,
    )

    # -----------------------------
    # COPY / PASTE AUGMENTATION
    # -----------------------------

    from darkfusion_canvas import (
        get_active_rois_for_inference,
        get_active_roi_for_inference,
        roi_inference_snapshot,
        _polygon_touches_roi_rect,
        remap_roi_bboxes_to_full_frame,
        scale_roi_bounds_to_frame,
        extract_roi_region_for_inference,
    )

    from darkfusion_augmentations import (
        validate_segmentation_quality,
        extract_segmented_object,
        overlay_object,
        check_overlap,
        _transform_pasted_point,
        _transform_pasted_label,
        apply_copy_paste_augmentation,
    )

    # -----------------------------
    # SAM3 LIVE OBJECT PROCESSING
    # -----------------------------

    from darkfusion_autolabel import (
        predict_and_draw_yolo_objects,
    )

    # -----------------------------
    # SAM3 BATCH PROCESSING
    # -----------------------------

    from darkfusion_augmentations import (
        _selected_sam_batch_actions,
        _set_sam3_batch_button_state,
        _sam3_should_stop,
        request_stop_sam3_batch,
        _finish_sam3_batch_control,
        _on_sam3_negative_generation_finished,
        process_batch,
        _run_sam3_batch,
        convert_bbox_to_segmentation,
    )

    # -----------------------------
    # SAM3 BATCH MISC HELPERS
    # -----------------------------

    from darkfusion_class_editor import (
        copy_classes_txt,
        load_num_classes,
    )

    # function to auto label with dino.py

    from darkfusion_autolabel import (
        on_dino_label_clicked,
        start_dino_auto_label,
        on_dino_finished,
        on_dino_error,
        on_dino_progress,
        on_dino_preview,
    )

    from darkfusion_class_editor import (
        get_active_class_names_from_ui,
    )

    from darkfusion_canvas import (
        set_current_file_name,
    )

    from darkfusion_augmentations import (
        get_all_labels,
        start_create_negative_images,
    )

    from darkfusion_dataset_workers import (
        on_button_click,
        setup_dataset_image_converter_controls,
        current_dataset_directory,
        collect_current_dataset_conversion_files,
        dataset_conversion_output_path,
        copy_dataset_conversion_metadata,
        convert_current_dataset_images,
        convert_images,
    )

    # video download

    from darkfusion_video_links import (
        setup_video_queue_table,
        _set_video_queue_status,
        _set_video_queue_progress,
        paste_video_queue_sources,
        _video_queue_item_text,
        _first_empty_video_queue_row,
        setup_video_download_table,
        remove_selected_video,
        download_videos,
        update_download_progress,
        update_download_status,
        on_video_queue_download_complete,
    )

    # Video player, live input, and frame extraction.

    from darkfusion_video_links import (
        is_video_url, _is_direct_video_url, _clean_video_source_name,
        _safe_video_name_from_source, _pick_yt_dlp_stream_url, _capture_source_is_readable,
        _download_video_url_cache, _resolve_stream_url, _video_source_error_message,
        _prepare_video_capture_source,
    )

    def stop_program(self):
        self.stop_live_collect_runtime(reset_input=True)
        if hasattr(self, "set_extraction_ui_state") and not self.extraction_is_active():
            self.set_extraction_ui_state("idle", "Extraction stopped.")
        logger.info("Processing stopped and input source reset to 'null'.")

    from darkfusion_video_inference import (
        toggle_inference,
        toggle_tracking,
        restart_live_inference_if_running,
        reset_live_tracker_sequence,
        live_video_input_is_running,
        queue_live_inference_refresh,
        refresh_live_inference_worker_from_ui,
        get_input_devices,
        setup_input_sources,
        _apply_input_devices,
        _resume_pending_camera_input,
        on_input_source_changed,
        on_custom_input_changed,
        on_skip_frames_changed,
        on_custom_size_changed,
    )

    from darkfusion_class_editor import (
        initialize_classes,
    )

    from darkfusion_video_inference import (
        ensure_live_inference_model,
        draw_opencv_dnn_bboxes_on_frame,
        on_crop_images_checkbox_state_changed,
        clear_live_video_overlay,
    )

    @staticmethod
    def qthread_is_running(thread):
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except RuntimeError:
            return False

    def park_stopping_qthread(self, thread, label="thread"):
        if thread is None:
            return

        pending = getattr(self, "_stopping_qthreads", None)
        if not isinstance(pending, list):
            pending = []
            self._stopping_qthreads = pending

        if thread in pending:
            return

        pending.append(thread)

        def cleanup_finished_thread(thread=thread, label=label):
            pending_threads = getattr(self, "_stopping_qthreads", [])
            try:
                if thread in pending_threads:
                    pending_threads.remove(thread)
            except ValueError:
                pass

            try:
                thread.deleteLater()
            except RuntimeError:
                pass

            logger.debug("%s stopped and was cleaned up.", label)

        try:
            thread.finished.connect(cleanup_finished_thread)
        except (TypeError, RuntimeError):
            cleanup_finished_thread()

    from darkfusion_video_inference import (
        stop_pending_live_inference_refresh,
        prepare_runtime_for_frame_extraction,
        get_resize_policy_key,
        model_input_size_available,
        current_video_transform_settings,
        resize_frame,
        sync_frame_resize_to_network_size,
        on_network_size_spinbox_value_changed,
        on_resize_policy_changed,
        update_resize_policy_controls,
        positive_int,
        parse_positive_int,
        get_frame_transform_size_wh,
        get_network_size_wh,
        get_network_imgsz_hw,
        current_prediction_size_filter_values,
        get_export_imgsz_hw,
        get_frame_step_value,
        should_extract_frame,
        transform_frame_for_extraction,
        remove_stale_extraction_outputs,
        force_live_redraw,
        apply_live_video_settings_change,
        update_crop_dimensions_enabled,
        crop_frame,
        setup_image_format_controls,
        get_image_extension,
        change_monitor,
        detect_and_set_monitor_range,
        set_monitor_index,
        setup_mss,
        capture_with_mss,
        display_camera_input,
    )

    from darkfusion_dataset_review import (
        grid_review_is_active,
        _grid_thumbnail_render_context,
        refresh_grid_review_view,
        grid_checked_selection_changed,
        schedule_grid_visible_thumbnails,
        schedule_grid_hover_preview,
        show_grid_hover_preview,
        _set_grid_current_index,
        navigate_grid_selection,
        toggle_grid_view,
        open_grid_image,
    )

    from darkfusion_video_inference import (
        show_annotation_viewer,
        enter_image_annotation_mode,
        show_frame_viewer,
        video_surface_is_foreground,
        collect_workflow_is_active,
        restore_collect_video_surface,
        _draw_video_roi_dimensions,
        update_display,
        set_video_playback_quality,
    )

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "graphics_scene") and self.graphics_scene.items():
            if self.screen_view.auto_fit_enabled:
                self.screen_view.reset_zoom()

    from darkfusion_video_inference import (
        connect_extraction_controls,
        extraction_is_active,
    )

    _CAPTURE_ACTION_BUTTON_STYLES = {
        "idle_extract": "",
        "idle_stop": (
            "QPushButton { background-color: #202327; color: #7d858f; "
            "border: 1px solid #3a4048; border-radius: 5px; }"
            "QPushButton:disabled { background-color: #202327; color: #626a73; "
            "border: 1px solid #343941; }"
        ),
        "preparing": (
            "QPushButton { background-color: #493313; color: #f8d99a; "
            "border: 1px solid #c58a2a; border-radius: 5px; font-weight: 600; }"
            "QPushButton:disabled { background-color: #493313; color: #f8d99a; "
            "border: 1px solid #c58a2a; }"
        ),
        "running": (
            "QPushButton { background-color: #6f1d25; color: #fff0f2; "
            "border: 1px solid #d15361; border-radius: 5px; font-weight: 700; }"
            "QPushButton:disabled { background-color: #6f1d25; color: #ffd9de; "
            "border: 1px solid #d15361; }"
        ),
        "stopping": (
            "QPushButton { background-color: #3c2024; color: #e7a4ad; "
            "border: 1px solid #8d3b46; border-radius: 5px; font-weight: 600; }"
            "QPushButton:disabled { background-color: #3c2024; color: #b77880; "
            "border: 1px solid #69313a; }"
        ),
    }

    from darkfusion_video_inference import (
        update_capture_stop_button,
        set_extraction_ui_state,
        set_label_progress,
        reset_label_progress,
        set_label_progress_busy,
        next_extraction_generation,
        is_current_extraction_generation,
        on_stop_extract_clicked,
        set_output_directory,
        _snapshot_extraction_inference,
        _snapshot_extraction_settings,
        start_extraction_thread,
        start_camera_extraction_timer,
        stop_extraction,
        on_extraction_progress_range,
        update_extraction_progress,
        on_extraction_progress_value,
        on_extraction_progress_format,
        on_extraction_checkpoint_resume_request,
        on_extraction_error,
        on_extraction_finished,
        on_extraction_thread_finished,
        on_extract_button_clicked,
        extract_frames_from_desktop,
        extract_frames_from_camera,
        extract_frames_from_video,
        _do_direct_video_extraction,
        save_frame_on_inference,
        update_checkboxes,
        on_custom_frames_toggled,
        on_original_size_toggled,
        on_custom_size_checkbox_state_changed,
        on_size_spinbox_value_changed,
        on_image_format_changed,
        on_prediction_size_filter_changed,
        perform_yolo_inference,
        get_tracker_config,
        _tracker_combo_key,
        _sync_tracker_preset_combos,
        get_tracker_preset_key,
        get_tracker_preset,
        get_tracker_preset_label,
        on_tracker_preset_changed,
        get_selected_video_path,
        get_capture_fps,
        get_frame_interval_ms,
        set_video_playback_renderer,
        get_video_playback_fps_limit,
        get_live_input_emit_fps_limit,
        apply_video_reader_max_emit_fps,
        get_playback_interval_ms,
        playback_speed_text,
        on_playback_speed_combo_changed,
        set_playback_speed,
        on_video_loop_toggled,
        current_video_frame_index,
        update_video_frame_label,
        _trim_rate_window,
        _trim_numeric_window,
        measured_rate,
        average_numeric_window,
        reset_video_capture_sequence,
        reset_video_performance_counters,
        record_video_capture_frame,
        record_video_display_frame,
        record_video_inference_frame,
        record_video_inference_latency,
        record_video_inference_drop,
        should_submit_video_inference_frame,
        mark_video_inference_frame_submitted,
        update_video_status_label,
        get_video_mark_range,
        update_video_mark_label,
        apply_video_loop_range_to_reader,
        mark_video_in,
        mark_video_out,
        clear_video_marks,
        seek_video_frame,
        step_video_frame,
        configure_frame_timer,
        is_live_video_source,
        current_video_source_is_live_input,
        stop_video_reader_thread,
        next_video_reader_generation,
        is_current_video_reader_generation,
        connect_video_reader_signals,
        handle_video_reader_frame,
        stop_video_inference_thread,
        next_video_inference_generation,
        is_current_video_inference_generation,
        on_video_inference_error,
        prepare_playback_frame,
        optimize_live_yolo_model,
        get_live_model_kwargs,
        start_video_inference_worker,
        on_video_metadata_ready,
        get_capture_fps_value,
        video_overlay_is_fresh,
        video_overlay_max_lag_frames,
        on_video_frame_ready,
        _accept_prepared_video_frame,
        _display_paired_video_frame,
        update_video_playback_controls_light,
        on_video_inference_profile_ready,
        on_video_overlay_ready,
        class_display_names,
        class_display_name,
        class_display_color_bgr,
        draw_video_overlay_on_frame,
        on_video_reader_error,
        on_video_reader_finished,
        on_play_video_clicked,
    )

    from darkfusion_video_links import (
        _on_url_resolved,
        _on_url_resolution_error,
        _on_extraction_url_resolved,
        _on_extraction_url_resolution_error,
        _on_direct_extraction_url_error,
    )

    from darkfusion_video_inference import (
        _start_video_playback,
        play_video_frame,
        _refresh_paused_video_inference,
    )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        SUPPORTED_VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm',
                                '.wmv', '.m4v', '.mpg', '.mpeg', '.3gp', '.ts', '.ogv')

        existing_paths = {
            self._video_queue_item_text(row).lower()
            for row in range(self.video_table.rowCount())
            if self._video_queue_item_text(row)
        }

        for url in event.mimeData().urls():
            source = url.toLocalFile() if url.isLocalFile() else url.toString().strip()
            source_key = source.lower()

            if (
                source
                and (source_key.endswith(SUPPORTED_VIDEO_EXTS) or self.is_video_url(source))
                and source_key not in existing_paths
            ):
                self.add_video_to_table(source)
                existing_paths.add(source_key)
                logger.info(f"Video added via drop: {source}")
            elif source_key in existing_paths:
                logger.debug(f"Skipped duplicate video from drop: {source}")

    from darkfusion_video_inference import (
        on_add_video_clicked,
        add_video_to_table,
        on_remove_video_clicked,
        toggle_play_pause,
        stop_video_playback,
        resume_video_playback,
        slider_seek,
        slider_pressed,
        update_time_label,
        skip_forward,
        skip_backward,
        display_frame_direct,
    )

    # predict and review function

    from darkfusion_dataset_review import (
        preview_updates_enabled,
        _perform_size_adjustment,
        is_preview_hover_zoom_enabled,
        hide_preview_hover_zoom,
        position_preview_hover_popup,
        show_preview_hover_zoom,
    )

    def eventFilter(self, source, event):
        try:
            if source is getattr(self, "filter_class_spinbox", None) and event.type() == QEvent.EnabledChange:
                self._update_review_action_scope()
            if source is getattr(self, "status_label_tools", None) and event.type() in (
                QEvent.Resize, QEvent.LayoutRequest, QEvent.FontChange,
            ):
                self._reflow_status_label_tools()
                return False
            if event.type() in (QEvent.KeyPress, QEvent.KeyRelease):
                # Qt item views consume letter keys for incremental search.
                # Resolve annotation navigation before the canvas/file/review
                # view sees it, while leaving text editors and dialogs alone.
                views = (
                    getattr(self, "screen_view", None),
                    getattr(self, "List_view", None),
                    getattr(self, "preview_list", None),
                    getattr(self, "grid_review_view", None),
                )
                navigation_target = any(
                    view is not None and (source is view or source is view.viewport())
                    for view in views
                )
                if navigation_target:
                    setting_key = self.keybind_setting_for_text(
                        event.text(), ("nextButton", "previousButton")
                    )
                    if setting_key:
                        if event.type() == QEvent.KeyPress:
                            return self.handle_navigation_key_press(setting_key, event)
                        return self.handle_navigation_key_release(setting_key, event)

            if source is self and event.type() == QEvent.WindowActivate:
                # Reconcile external file changes when the user returns to the
                # app. Watching every directory write also reacts to each label
                # save and can contend with drawing on very large datasets.
                self._schedule_dataset_directory_refresh()

            if source is getattr(self, "image_label_2", None):
                if event.type() == QEvent.Resize:
                    QTimer.singleShot(0, self._scale_status_gif)
                return False

            if source is getattr(self, "classes_dropdown", None):
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    self.show_class_picker_popup()
                    return True

                if event.type() == QEvent.KeyPress and event.key() in (
                    Qt.Key_Space,
                    Qt.Key_Return,
                    Qt.Key_Enter,
                    Qt.Key_Down,
                ):
                    self.show_class_picker_popup()
                    return True

            if source in (
                getattr(self, "augmentation_preview_label", None),
                getattr(self, "auto_head_preview_label", None),
            ):
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    self.open_augmentation_preview_viewer(source)
                    return True

            preview_widget = getattr(self, "preview_list", None)
            preview_viewport = preview_widget.viewport() if preview_widget is not None else None

            if source is preview_viewport:
                if event.type() == QEvent.MouseMove:
                    index = self.preview_list.indexAt(event.pos())
                    if index.isValid():
                        self.show_preview_hover_zoom(index.row(), event.globalPos())
                    else:
                        self.hide_preview_hover_zoom()
                    return False

                if event.type() in (QEvent.Leave, QEvent.MouseButtonPress, QEvent.Wheel):
                    self.hide_preview_hover_zoom()

            if event.type() == QEvent.MouseButtonPress and source is preview_viewport:
                pos = event.pos()
                index = self.preview_list.indexAt(pos)
                if not index.isValid():
                    return False
                current_button = event.button()
                current_time = time.time()

                if hasattr(self, "_last_click_time") and hasattr(self, "_last_button_clicked"):
                    elapsed_time = current_time - self._last_click_time
                    if elapsed_time < 0.3 and self._last_button_clicked == current_button:
                        logger.debug("Ignored quick repeated click.")
                        return True

                self._last_click_time = current_time
                self._last_button_clicked = current_button

                if current_button == Qt.RightButton:
                    self.handle_right_click(index)
                    return True
                elif current_button == Qt.LeftButton:
                    self.handle_left_click(index)
                    return True

            return super().eventFilter(source, event)

        except Exception as e:
            logger.error(f"An unexpected error occurred in eventFilter: {e}")
            return False

    from darkfusion_dataset_review import (
        highlight_thumbnail,
        unhighlight_all_thumbnails,
        handle_right_click,
    )

    from darkfusion_canvas import (
        reinitialize_bounding_boxes,
    )

    from darkfusion_dataset_review import (
        _clamp_preview_page,
        _on_preview_scroll_value_changed,
        _append_next_preview_batch,
        delete_item,
        _discard_deleted_review_similarity_match,
        _show_next_review_similarity_file,
        _build_preview_pixmap_for_bbox,
        _thumbnail_label_style,
        _preview_details_card_style,
        _set_preview_widget_opacity,
        _preview_badge,
        _create_preview_details_widget,
        _preview_bbox_review_object,
        _canonical_preview_label_text,
        _preview_bounds_iou,
        _active_dataset_analysis_preview_issue,
        _resolve_dataset_analysis_preview_target,
        _preview_dataset_target_for_row,
        _style_dataset_analysis_preview_target,
        _focus_dataset_analysis_preview_target,
        _current_preview_annotation_line,
        _preview_row_for_annotation,
    )

    from darkfusion_canvas import (
        save_preview_annotation_negative,
    )

    from darkfusion_dataset_review import (
        change_preview_annotation_class,
        _selected_preview_annotation_context,
    )

    _PREVIEW_ANNOTATION_ACTION_TITLES = {
        'save_negative': 'Save Negative Crop',
        'save_negative_delete': 'Save Negative Crop + Delete Annotation',
        'find_similar': 'Find Similar Annotations',
        'remove_matches': 'Remove Similar Matches...',
        'negative_remove_matches': 'Save Representative Negatives + Remove Matches...',
        'change_class': 'Change Class...',
        'copy_label': 'Copy YOLO Label',
    }

    from darkfusion_dataset_review import (
        _sync_preview_annotation_actions,
        run_selected_preview_annotation_action,
        find_similar_selected_preview_annotations,
        _populate_selected_preview_annotation_menu,
        show_selected_preview_annotation_actions,
        find_similar_preview_annotations,
        _resolve_similarity_label_rows,
        _resolved_review_similarity_matches,
        _representative_similarity_matches,
        remove_review_similarity_matches,
        _populate_preview_annotation_menu,
        show_preview_annotation_actions,
        _create_thumbnail_widget,
        _invalidate_preview_for_filter_change,
        _preview_source_version,
        _populate_preview_list_for_image,
        _run_debounced_thumbnail_page_now,
    )

    from darkfusion_canvas import (
        load_label_lines,
        _editor_label_source,
        save_label_lines,
    )

    from darkfusion_dataset_review import (
        rebuild_preview_list_for_all_images,
        show_thumbnail_page,
        _run_debounced_thumbnail_page,
        _selected_review_preview_row,
        _review_preview_image_path,
        _selected_review_file_path,
        _select_review_preview_row,
        go_to_selected_review_file,
    )

    from darkfusion_dataset_review import (
        _on_file_reveal_completed,
    )


    from darkfusion_dataset_review import (
        handle_left_click,
        flash_bounding_box,
        _rgb_from_setting,
        apply_preview_settings,
        set_preview_thumbnail_size,
        set_grid_thumbnail_size,
        set_preview_batch_size,
        set_review_similarity_threshold,
        refresh_active_review_similarity_filter,
        set_preview_flash_time,
        set_preview_hover_zoom_enabled,
        set_preview_flash_color,
        synchronize_list_view,
        sync_list_view_selection,
        setup_list_view_with_delegate,
        toggle_review_mode,
        setup_preview_list_style,
    )

    # mute sounds

    from darkfusion_video_inference import (
        mute_player,
    )

    def keyPressEvent(self, event):
        key_code = event.key()
        modifiers = event.modifiers()

        # Block unwanted keys
        if key_code in (Qt.Key_Control, Qt.Key_Alt):
            event.accept()
            return

        if key_code == Qt.Key_Delete:
            event.accept()
            if event.isAutoRepeat():
                return
            self.stop_next_timer()
            self.stop_prev_timer()
            self.delete_current_image()
            return

        key = event.text()
        normalized_key = normalize_keybind_value(key)
        delete_binding = normalize_keybind_value(
            getattr(self, "settings", {}).get("deleteButton", DEFAULT_KEYBINDS["deleteButton"])
        )
        if event.isAutoRepeat() and normalized_key and delete_binding.startswith(normalized_key):
            event.accept()
            return
        setting_key = self.keybind_setting_for_text(key, ("nextButton", "previousButton", "deleteButton"))

        if setting_key in ("nextButton", "previousButton"):
            if self.handle_navigation_key_press(setting_key, event):
                event.accept()
                return

        if setting_key and self.run_navigation_keybind(setting_key, modifiers, auto_scan=False):
            event.accept()
            return

        else:
            if not normalized_key:
                super().keyPressEvent(event)
                return

            configured_bindings = {
                normalize_keybind_value(value)
                for _setting, _label, value in self.keybind_entries()
                if normalize_keybind_value(value)
            }
            class_bindings = {
                normalize_keybind_value(
                    getattr(self, "settings", {}).get(class_hotkey_key(class_name), "")
                )
                for class_name in self.current_keybind_class_names()
            }
            class_bindings.discard("")

            # Holding a class key should not manufacture a multi-key sequence
            # from keyboard auto-repeat (for example, zzz or 111).
            if event.isAutoRepeat() and any(
                binding.startswith(normalized_key) for binding in class_bindings
            ):
                event.accept()
                return

            if self.keyBuffer:
                candidate = normalize_keybind_value(self.keyBuffer + normalized_key)
                candidate_is_exact = candidate in configured_bindings
                candidate_has_longer_match = any(
                    binding.startswith(candidate) and binding != candidate
                    for binding in configured_bindings
                )

                # An already-started valid sequence takes precedence over a
                # single-key binding. This keeps 12 working even if 2 is also
                # configured as its own shortcut.
                if candidate_is_exact and not candidate_has_longer_match:
                    self.keyBuffer = candidate
                    self.processKeyPresses()
                    event.accept()
                    return

                if candidate_is_exact or candidate_has_longer_match:
                    self.keyBuffer = candidate
                    if self.timer.isActive():
                        self.timer.stop()
                    self.timer.start(300)
                    event.accept()
                    return

                # The old prefix and this key cannot form a configured
                # shortcut. Discard that incomplete prefix and evaluate this
                # key as the beginning of a fresh shortcut.
                self.keyBuffer = ""

            single_is_exact = normalized_key in configured_bindings
            single_has_longer_match = any(
                binding.startswith(normalized_key) and binding != normalized_key
                for binding in configured_bindings
            )

            # Exact single-key bindings such as Z and X must fire immediately.
            # Previously every key waited 300 ms, so rapid Z then X became the
            # invalid combined shortcut "zx" and neither class was selected.
            if single_is_exact and not single_has_longer_match:
                self.keyBuffer = normalized_key
                self.processKeyPresses()
                event.accept()
                return

            self.keyBuffer = normalized_key

            # Only ambiguous prefixes (such as 1 when 12 and 13 exist) need
            # the short wait used for multi-key hotkeys.
            if self.timer.isActive():
                self.timer.stop()
            self.timer.start(300)
            event.accept()

    def keyReleaseEvent(self, event):
        setting_key = self.keybind_setting_for_text(
            event.text(),
            ("nextButton", "previousButton"),
        )
        if setting_key and self.handle_navigation_key_release(setting_key, event):
            event.accept()
            return
        super().keyReleaseEvent(event)

    def processKeyPresses(self):
        key = self.keyBuffer
        modifiers = QApplication.keyboardModifiers()
        setting_key = self.keybind_setting_for_text(key)

        if not self.run_navigation_keybind(setting_key, modifiers, stop_timers=False):
            if not self.run_mode_keybind(setting_key):
                self.select_class_for_keybind(key)

        # Clear the buffer and stop the timer.
        self.keyBuffer = ""
        if self.timer.isActive():
            self.timer.stop()

    from darkfusion_canvas import (
        set_annotation_mode,
        flash_overlay_text,
    )

    # all part of the auto label function.

    from darkfusion_autolabel import (
        stop_auto_labeling,
        threshold_spinbox_changed,
        open_weights,
        clear_loaded_weights,
    )

    from darkfusion_darknet import (
        suggested_cfg_path_for_weights,
        prompt_for_matching_cfg,
        open_cfg,
    )

    from darkfusion_autolabel import (
        determine_model_type,
        read_engine_metadata,
        engine_size_from_metadata,
        engine_size_from_tensorrt,
        engine_size_from_filename,
        model_size_from_sidecar_json,
        model_size_from_onnx,
        model_size_from_loaded_model,
        detect_weight_input_size,
        detect_engine_input_size,
        apply_loaded_weight_input_size,
        merge_auto_label_bboxes,
        clear_auto_label_files_for_images,
        _qsettings_bool_value,
        auto_label_teammate_filter_config,
        _teammate_filter_bbox_bounds,
        _teammate_filter_pil_image,
        filter_teammate_auto_label_batches,
        filter_teammate_auto_labels,
        _refine_yolo_boxes_with_sam3,
        _yolo_gap_record,
        _yolo_gap_record_is_current,
        _finish_yolo_sam3_assist,
    )

    @gpu_work("Auto labeling", keep=auto_label_resources)
    def auto_label_images2(self, overwrite=None):
        if getattr(self, "_gpu_released_detector", False):
            ensure_gpu_detector(self)
        logging.info("Starting auto_label_images2")

        class_names = self.sync_class_labels_from_dropdown()

        if not class_names:
            class_names = self.get_class_names_for_current_model(
                data_directory=getattr(self, "image_directory", None)
            )

            self.class_labels = list(class_names) if class_names else []
            self.class_names = list(self.class_labels)
            self.id_to_class = {i: name for i, name in enumerate(self.class_labels)}

        if not self.class_labels:
            QMessageBox.critical(self, "Error", "No classes available from current model or classes.txt.")
            return

        if not hasattr(self, "model") or self.model is None:
            QMessageBox.critical(self, "Error", "Model is not initialized.")
            return

        if str(getattr(self.model, "task", "") or "").lower() in {"classify", "reid"}:
            QMessageBox.critical(self, "Auto Label", "Classification/ReID models cannot write object annotations. Existing labels were kept.")
            return

        if self.is_keypoint_mode():
            model_path = getattr(self, "weights_file_path", None) or getattr(self, "loaded_model_path", None)
            model_key = os.path.abspath(str(model_path)) if model_path else ""
            if model_key and getattr(self, "_last_pose_model_path", None) != model_key:
                self.sync_pose_list_from_current_model(
                    model_path=model_path,
                    write_points_file=True,
                    force=False,
                )

        active_class_ids = self.get_active_class_ids_from_ui()
        if not active_class_ids:
            logging.info("No classes checked in classes_dropdown. Falling back to all classes.")
            active_class_ids = set(range(len(self.class_labels)))

        self.stop_labeling = False
        self.img_index_number_changed(0)

        source_images = [
            image_file for image_file in self.image_files
            if not self.is_placeholder_file(image_file)
        ]

        total_images = len(source_images)
        self.set_label_progress(
            0, total_images, value=0, progress_format="Auto Label %v/%m"
        )

        if total_images == 0:
            QMessageBox.information(self, "Information", "No images available for autolabeling.")
            return

        if overwrite is None:
            overwrite_reply = QMessageBox.question(
                self,
                "Auto Label Existing Dataset",
                (
                    "How should auto-label handle existing labels?\n\n"
                    "Yes = delete all old labels and write only new checked-class detections.\n"
                    "No = keep old labels and stack/merge new checked-class detections.\n"
                    "Cancel = stop."
                ),
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.No
            )

            if overwrite_reply == QMessageBox.Cancel:
                logging.info("Labeling canceled by user.")
                return

            overwrite_existing = overwrite_reply == QMessageBox.Yes
        else:
            overwrite_existing = bool(overwrite)

        logging.info(f"Auto-label overwrite existing labels: {overwrite_existing}")
        self.reset_prediction_size_filter_stats("image auto-label")
        sam3_assist = bool(getattr(self, "auto_label_sam3_assist", False))
        gap_frames = [None] * total_images if sam3_assist else None
        if sam3_assist:
            self._yolo_sam3_stats = {"refined": 0, "kept": 0}
            self._yolo_sam3_unavailable = ""
        # Replace each file only after successful inference/decoding. Never
        # clear the whole dataset before the first model/provider has run.

        batch_size = self.batch_inference.value()
        if batch_size <= 0:
            batch_size = 1

        num_expected_keypoints = self.keypoint_list.rowCount()

        points_file = resolve_dataset_pose_schema_path(self.image_directory, migrate=True)
        if num_expected_keypoints <= 0 and os.path.exists(points_file):
            try:
                with open(points_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict) and isinstance(data.get("keypoints"), list):
                    num_expected_keypoints = len(data["keypoints"])
                    logging.info(f"Using {num_expected_keypoints} keypoints from points.json")

            except Exception as e:
                logging.warning(f"Failed to read points.json: {e}")

        failed_image_count = 0
        for batch_start in range(0, total_images, batch_size):
            if self.stop_labeling:
                logging.info("Labeling stopped by user.")
                self.log_prediction_size_filter_summary("image auto-label")
                return

            batch_files = source_images[batch_start:batch_start + batch_size]
            batch_imgs = []
            img_whs = []
            batch_roi_offsets = []  # Track auto-label ROI metadata for each image.

            # Check if ROI inference is enabled
            roi_enabled = self.settings.get('videoUseROIForInference', False)
            roi_bounds = None
            if roi_enabled:
                try:
                    roi_bounds = self.get_active_rois_for_inference()
                except Exception as e:
                    logging.debug(f"Failed to get ROI bounds for image auto-label: {e}")

            for image_offset, image_file in enumerate(batch_files):
                if self.stop_labeling:
                    logging.info("Labeling stopped by user.")
                    self.log_prediction_size_filter_summary("image auto-label")
                    return

                # Only the standard BGR reader shares with the editor. Custom
                # readers (for example SAM's unchanged-depth loader) keep their
                # own decoding and display semantics.
                reader = self.read_image
                shares_original = getattr(reader, "__func__", None) is MainWindow.read_image
                if shares_original:
                    image = reader(image_file, share_editor=True)
                else:
                    image = reader(image_file)
                if image is None:
                    logging.warning(f"Skipping invalid image: {image_file}")
                    failed_image_count += 1
                    batch_imgs.append(None)
                    img_whs.append((None, None))
                    batch_roi_offsets.append({"x": 0, "y": 0})
                    continue

                try:
                    self.current_display_image = None
                    previous_neighbors = getattr(self, '_editor_prefetch_paths', None)
                    next_index = batch_start + image_offset + 1
                    self._editor_prefetch_paths = source_images[next_index:next_index + 2] if shares_original else None
                    try:
                        self.display_image(image_file)
                    finally:
                        self._editor_prefetch_paths = previous_neighbors

                    with Image.open(image_file) as img:
                        img_width, img_height = img.size

                    preprocessed_image = getattr(self, "current_display_image", None)
                    if preprocessed_image is None or not isinstance(preprocessed_image, np.ndarray) or preprocessed_image.size == 0:
                        preprocessed_image, _ = self.apply_preprocessing(
                            image.copy(),
                            None,
                            img_width,
                            img_height
                        )
                    else:
                        preprocessed_image = preprocessed_image.copy()

                    if preprocessed_image is None or not isinstance(preprocessed_image, np.ndarray) or preprocessed_image.size == 0:
                        raise ValueError("Preprocessing returned an empty image.")

                    proc_height, proc_width = preprocessed_image.shape[:2]

                    # Keep full-frame inference and attach the auto-label ROI filter.
                    roi_offset = {"x": 0, "y": 0}
                    infer_image = preprocessed_image
                    if roi_enabled and roi_bounds is not None:
                        try:
                            roi_frame, offset_map = self.extract_roi_region_for_inference(
                                preprocessed_image, roi_bounds
                            )
                            if roi_frame is not None and roi_frame.size > 0:
                                infer_image = roi_frame
                                roi_offset = offset_map
                        except Exception as e:
                            logging.debug(f"Failed to extract ROI from preprocessed image: {e}")
                            infer_image = preprocessed_image

                    batch_imgs.append(infer_image)
                    img_whs.append((proc_width, proc_height))
                    batch_roi_offsets.append(roi_offset)

                except Exception as e:
                    logging.warning(f"Failed to prepare {image_file}: {e}")
                    failed_image_count += 1
                    batch_imgs.append(None)
                    img_whs.append((None, None))
                    batch_roi_offsets.append({"x": 0, "y": 0})

            valid_indices = [
                i for i, img in enumerate(batch_imgs)
                if img is not None and img_whs[i][0] is not None and img_whs[i][1] is not None
            ]

            if not valid_indices:
                continue

            batch_imgs = [batch_imgs[i] for i in valid_indices]
            valid_batch_files = [batch_files[i] for i in valid_indices]
            valid_img_whs = [img_whs[i] for i in valid_indices]
            valid_roi_offsets = [batch_roi_offsets[i] for i in valid_indices]

            try:
                model_kwargs = self.get_model_kwargs()

                if "classes" in model_kwargs and model_kwargs["classes"] == []:
                    logging.warning("No selected project classes match the model. Existing labels were kept.")
                    self.set_label_progress(0, total_images, value=batch_start, progress_format="Auto Label stopped: no matching classes")
                    QMessageBox.warning(self, "Auto Label", "None of the selected project classes match this model's classes. Existing labels were kept.")
                    return

                if (
                    not self.is_darkfusion_onnx_model(self.model)
                    and ultralytics_kwargs_use_fp16(model_kwargs)
                    and torch.cuda.is_available()
                ):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        results = self.model.predict(batch_imgs, **model_kwargs)
                else:
                    results = self.model.predict(batch_imgs, **model_kwargs)
                results = list(results)
                if len(results) != len(valid_batch_files):
                    raise ValueError(f"Model returned {len(results)} results for {len(valid_batch_files)} images.")
                if any(result is None for result in results):
                    raise ValueError("Model returned a missing image result.")

            except Exception as e:
                logging.error(f"Model inference error: {e}")
                self.set_label_progress(0, total_images, value=batch_start, progress_format="Auto Label stopped: inference failed")
                QMessageBox.critical(self, "Auto Label stopped", f"Model inference failed: {e}\nExisting labels for unprocessed images were kept.")
                return

            prepared_bbox_batches = []
            failed_prediction_indices = set()
            for i, (result, image_file) in enumerate(zip(results, valid_batch_files)):
                img_width, img_height = valid_img_whs[i]
                try:
                    infer_height, infer_width = batch_imgs[i].shape[:2]
                    new_bboxes = self.extract_bboxes_from_result(
                        result,
                        infer_width,
                        infer_height,
                        image_file,
                        None,
                        num_expected_keypoints,
                        conf_threshold=model_kwargs.get("conf")
                    )

                    new_bboxes = self.filter_bboxes_to_active_classes(new_bboxes, active_class_ids)
                    if sam3_assist:
                        new_bboxes = self._refine_yolo_boxes_with_sam3(batch_imgs[i], new_bboxes)

                    # Keep only annotations intersecting the active ROI while
                    # preserving their complete full-image geometry.
                    new_bboxes = self.remap_roi_bboxes_to_full_frame(
                        new_bboxes,
                        valid_roi_offsets[i],
                        (infer_width, infer_height),
                        (img_width, img_height),
                    )

                    new_bboxes = self.filter_bboxes_to_size_limits(
                        new_bboxes,
                        img_width,
                        img_height,
                        source=os.path.basename(image_file)
                    )
                    prepared_bbox_batches.append(new_bboxes)
                except Exception as e:
                    logging.error(f"Failed to prepare predictions for {image_file}: {e}")
                    failed_image_count += 1
                    failed_prediction_indices.add(i)
                    prepared_bbox_batches.append([])

            prepared_bbox_batches = self.filter_teammate_auto_label_batches(
                batch_imgs,
                prepared_bbox_batches,
                class_names=self.class_labels,
                source="weights auto label",
            )

            for i, (image_file, new_bboxes) in enumerate(zip(valid_batch_files, prepared_bbox_batches)):
                if i in failed_prediction_indices:
                    continue  # Decode failure is not a successful empty prediction.
                try:

                    label_file, label_exists = self.get_label_file(
                        image_file,
                        return_existence=True
                    )

                    existing_bboxes = []
                    if not overwrite_existing and label_exists and label_file and os.path.exists(label_file):
                        existing_bboxes = self.load_existing_bboxes(label_file)

                    if overwrite_existing:
                        final_bboxes = self.remove_near_duplicate_bounding_boxes(
                            new_bboxes,
                            iou_threshold=0.5,
                            class_aware=True
                        )

                    else:
                        final_bboxes = self.merge_auto_label_bboxes(existing_bboxes, new_bboxes)

                    labels_to_save = [bbox.to_str() for bbox in final_bboxes]

                    if label_file:
                        saved = self.save_labels_to_file(
                            label_file,
                            labels_to_save,
                            mode="w",
                            allow_empty=overwrite_existing,
                        )
                        if not saved and (labels_to_save or overwrite_existing):
                            raise IOError("The annotation file could not be replaced.")
                        logging.info(f"Labels saved to: {label_file}")
                        if sam3_assist:
                            gap_frames[batch_start + valid_indices[i]] = self._yolo_gap_record(
                                image_file, label_file, new_bboxes, valid_img_whs[i], valid_roi_offsets[i]
                            )

                        if (batch_start + i) % 3 == 0:
                            self.current_file = image_file
                            previous_neighbors = getattr(self, '_editor_prefetch_paths', None)
                            next_index = batch_start + len(batch_files)
                            shares_original = getattr(self.read_image, "__func__", None) is MainWindow.read_image
                            self._editor_prefetch_paths = source_images[next_index:next_index + 2] if shares_original else None
                            try:
                                self.display_image(image_file, rebuild_preview=False)
                            finally:
                                self._editor_prefetch_paths = previous_neighbors
                            QApplication.processEvents()
                    else:
                        logging.warning(f"Could not resolve label file for {image_file}")

                except Exception as e:
                    logging.error(f"Failed to save labels for {image_file}: {e}")
                    failed_image_count += 1

                self.label_progress.setValue(batch_start + i + 1)
                QApplication.processEvents()

        assist_summary = self._finish_yolo_sam3_assist(gap_frames) if sam3_assist else ""
        self.set_label_progress(
            0, total_images, value=total_images, progress_format="Auto Label complete %v/%m"
        )
        self.log_prediction_size_filter_summary("image auto-label")
        if failed_image_count:
            QMessageBox.warning(self, "Auto Label", f"Labeling finished with {failed_image_count} skipped/failed images. Their existing labels were kept. Check the log for details." + assist_summary)
        else:
            QMessageBox.information(self, "Information", "Labeling completed!" + assist_summary)
        logging.info("auto_label_images2 finished; skipped/failed images: %s", failed_image_count)
        self.img_index_number_changed(0)

    from darkfusion_autolabel import (
        class_supports_keypoints,
        extract_bboxes_from_result,
    )

    from darkfusion_class_editor import (
        get_class_names_for_current_model,
        sync_classes_from_current_model,
    )

    from darkfusion_autolabel import (
        load_existing_bboxes,
    )

    from darkfusion_class_editor import (
        map_model_class_id_to_local_class_id,
    )
    from darkfusion_autolabel import (
        has_visible_keypoints,
        obb_xywhr_to_corners,
        get_model_kwargs,
    )

    from darkfusion_class_editor import (
        load_model_classes,
    )

    from darkfusion_class_editor import (
        get_active_class_ids_from_ui,
    )

    from darkfusion_class_editor import (
        map_local_class_ids_to_model_class_ids,
        model_to_local_class_id_map,
        filter_bboxes_to_active_classes,
        classes_dropdown_menu,
        ensure_class_picker_dialog,
        show_class_picker_popup,
    )

    from darkfusion_canvas import (
        _editor_original_source,
        _prefetch_editor_neighbors,
    )

    from darkfusion_autolabel import (
        process_current_image,
        switch_floating_point_mode,
        set_floating_point_mode,
        apply_backend_and_target,
        get_dnn_output_layer_names,
        initialize_yolo,
    )

    from darkfusion_canvas import (
        read_image,
    )

    from darkfusion_autolabel import (
        auto_label_images,
        auto_label_yolo_button_clicked,
        infer_and_display_bounding_boxes,
    )

    from darkfusion_canvas import (
        get_annotation_size_limits,
        annotation_pixel_size_allowed,
        annotation_box_size_allowed,
    )

    from darkfusion_autolabel import (
        _prediction_size_stats_lock,
        reset_prediction_size_filter_stats,
        record_prediction_size_filter_skip,
        log_prediction_size_filter_summary,
        prediction_size_allowed,
        filter_predictions_to_size_limits,
        filter_bboxes_to_size_limits,
        filter_video_overlay_to_size_limits,
        get_dnn_size_limits,
        run_opencv_dnn_inference,
        display_dnn_bounding_boxes,
    )

    #polots and stats related to dataset analysis

    from darkfusion_canvas import (
        set_image_directory,
    )

    from darkfusion_dataset_analysis import create_plot

    from darkfusion_dataset_analysis import display_stats

    from darkfusion_dataset_analysis import toggle_image_quality_analysis

    from darkfusion_dataset_review import (
        initialize_filter_spinbox,
        update_filter_spinbox,
        on_filter_class_spinbox_changed,
        sync_selected_class_with_filter,
    )

    from darkfusion_dataset_review import (
        _capture_review_filter_origin,
        filter_class,
        _review_similarity_is_active,
        _clear_review_similarity_state,
        _cancel_review_filter_request,
        _stop_review_similarity_search,
        _start_review_similarity_filter,
        _set_review_similarity_busy,
        _review_similarity_worker_finished,
        _on_review_similarity_status,
        _on_review_similarity_progress,
        _on_review_similarity_completed,
        _apply_review_similarity_results,
        _review_placeholder_paths,
        _start_review_filter_worker,
        _on_review_filter_progress,
        _on_review_filter_completed,
        _on_review_filter_failed,
    )

    from darkfusion_dataset_review import (
        _apply_filtered_image_files,
        sync_class_checkboxes_with_filter,
    )

    from darkfusion_canvas import (
        update_bounding_box_count_display,
        clear_current_frame_labels,
        clear_class_boxes,
        refresh_bounding_box_display,
    )

    from darkfusion_dataset_workers import (
        move_filtered_images,
        on_filtered_export_progress,
    )

    from darkfusion_dataset_review import _remove_moved_images_from_review

    from darkfusion_dataset_workers import (
        on_filtered_export_completed,
        on_filtered_export_failed,
        move_file,
        is_file_matching_class_id,
    )

    from darkfusion_canvas import (
        update_list_view,
        _rebuild_image_file_index,
        _full_dataset_index_for_file,
        on_list_view_clicked,
        img_index_number_changed,
        get_image_files,
        _watch_dataset_directory,
        _schedule_dataset_directory_refresh,
        _start_dataset_directory_refresh,
        _dataset_directory_refresh_finished,
        _dataset_directory_refresh_failed,
        _apply_dataset_directory_refresh,
        apply_natural_sort_background,
        _apply_natural_sort_result,
        _natural_sort_finished,
        placeholder_image_path,
        display_placeholder,
        preprocess_placeholder_image,
        is_placeholder_file,
        cleanup_placeholder_artifacts,
        _prune_missing_dataset_files,
    )

    def open_image_video(self):
        """
        Open an image directory and display the first image quickly.
        Heavy dataset analytics are NOT run here.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        last_used_directory = os.getenv("ULTRADARKFUSION_LAST_DIR", None)
        start_directory = self.dialog_start_directory(
            "dataset",
            last_used_directory if last_used_directory and os.path.exists(last_used_directory) else None,
        )

        dir_name = QFileDialog.getExistingDirectory(self, "Open Image Directory", start_directory, options=options)
        if not dir_name:
            return
        self.remember_dialog_selection("dataset", dir_name)
        self._graphics_dataset_explicitly_opened = True

        self.set_main_progress(0, 100)
        self.clear_annotations()
        self.enter_image_annotation_mode(clear_video_surface=True)
        self.set_main_progress(5, 100)

        self.last_image_directory = dir_name
        self.image_directory = self.normalize_path(dir_name)
        self._watch_dataset_directory(self.image_directory)
        self.activate_project_settings(self.image_directory)
        self._review_filter_label_cache = {}
        self.set_main_progress(15, 100)

        self.settings["last_dir"] = self.image_directory
        set_key(dotenv_path, "ULTRADARKFUSION_LAST_DIR", self.image_directory)
        self.saveSettings()
        logger.info(f"Image directory: {self.image_directory}")

        if hasattr(self, "scan_annotations"):
            self.scan_annotations.base_directory = self.image_directory

        self.class_names = self.load_classes() or ['person']
        self.update_classes_dropdown(self.class_names)
        self.set_main_progress(30, 100)

        # Fast scan + fast sort first
        self.image_files = [
            self.normalize_path(f)
            for f in self.get_image_files(dir_name)
        ]
        self.image_files.sort()
        self.set_main_progress(50, 100)

        placeholder_image_path = 'styles/images/default.png'
        processed_placeholder = self.preprocess_placeholder_image(placeholder_image_path)
        if processed_placeholder:
            processed_placeholder = self.normalize_path(processed_placeholder)
            if processed_placeholder not in self.image_files:
                self.image_files.insert(0, processed_placeholder)

        self.filtered_image_files = list(self.image_files)
        if not self.image_files:
            QMessageBox.warning(None, 'No Images Found', "No image files found in the directory.")
            self.set_main_progress(0, 100)
            return

        # Default working list is the full image list and must be kept in sync
        # with the actual folder contents even when files were deleted externally.
        self.filtered_image_files = list(self.image_files)
        # get_image_files just scanned this directory. Do not synchronously
        # repeat an exists() check for every image in both copies of the list.
        self.filtered_image_files = [path for path in self.filtered_image_files if not self.is_placeholder_file(path)]
        self.image_files = list(self.filtered_image_files)
        self.restore_project_ui_state(apply_filter=True)

        # Show first real image if possible
        saved_last_image = self.get_saved_last_image_for_dir(self.image_directory)
        self.restore_current_image_after_load(saved_last_image)

        self.display_image(self.current_file)
        if self.determine_model_type(getattr(self, "weights_file_path", "")) == "weights":
            self.initialize_yolo()
        self.set_main_progress(75, 100)

        # Only ensure what is needed now
        self.ensure_points_json_exists()

        # Update UI list now without extra dataset analysis
        self.update_list_view(self.filtered_image_files, validate_files=False)
        self.set_main_progress(90, 100)

        if hasattr(self, "img_index_number"):
            self.img_index_number.blockSignals(True)
            self.img_index_number.setMaximum(max(0, len(self.filtered_image_files) - 1))
            self.img_index_number.setValue(self.current_img_index)
            self.img_index_number.blockSignals(False)

        self.set_main_progress(100, 100)

        # Apply sequential natural sort after UI is already responsive
        QTimer.singleShot(0, self.apply_natural_sort_background)

        self.cleanup_placeholder_artifacts()

    from darkfusion_canvas import (
        clear_annotations,
    )

    def stop_existing_workers(self, include_standalone_moves=False):
        """Stop dataset workers; independent folder moves survive dataset changes."""
        try:
            for name in ("_grid_delete_worker", "_natural_sort_worker"):
                worker = getattr(self, name, None)
                if worker is not None:
                    worker.requestInterruption()
                    # Cancellation is checked between pairs. Never terminate
                    # a thread halfway through a filesystem transaction.
                    worker.wait()
                    setattr(self, name, None)
                    worker.deleteLater()
            self._natural_sort_pending = False
            self._set_image_deletion_busy(False)
            dataset_scanner = getattr(self, "scan_annotations", None)
            if dataset_scanner is not None and hasattr(dataset_scanner, "cancel_active_scan"):
                dataset_scanner.cancel_active_scan(wait_ms=3000)

            tile_worker = getattr(self, "_tile_worker", None)
            if tile_worker is not None:
                try:
                    if tile_worker.isRunning():
                        logger.info("Stopping tile worker...")
                        tile_worker.cancel()
                        tile_worker.wait(3000)
                        logger.info("Tile worker stopped.")
                except RuntimeError:
                    pass
                finally:
                    self._tile_worker = None

            filter_worker = getattr(self, "_review_filter_worker", None)
            if filter_worker is not None:
                try:
                    if filter_worker.isRunning():
                        logger.info("Stopping review filter worker...")
                        filter_worker.cancel()
                        filter_worker.wait(3000)
                        logger.info("Review filter worker stopped.")
                except RuntimeError:
                    pass
                finally:
                    self._review_filter_worker = None

            bulk_move_worker = getattr(self, "_bulk_move_worker", None) if include_standalone_moves else None
            if bulk_move_worker is not None:
                try:
                    if bulk_move_worker.isRunning():
                        logger.info("Stopping bulk move worker...")
                        bulk_move_worker.cancel()
                        bulk_move_worker.wait(3000)
                        logger.info("Bulk move worker stopped.")
                except RuntimeError:
                    pass
                finally:
                    self._bulk_move_worker = None

            filtered_export_worker = getattr(self, "_filtered_export_worker", None)
            if filtered_export_worker is not None:
                try:
                    if filtered_export_worker.isRunning():
                        logger.info("Stopping filtered export worker...")
                        filtered_export_worker.cancel()
                        filtered_export_worker.wait(3000)
                        logger.info("Filtered export worker stopped.")
                except RuntimeError:
                    pass
                finally:
                    self._filtered_export_worker = None

            if hasattr(self, 'thumbnail_thread') and self.thumbnail_thread is not None:
                try:
                    if self.thumbnail_thread.isRunning():
                        logger.info("Stopping existing thumbnail thread...")
                        try:
                            self.thumbnail_worker.finished.disconnect(self.on_thumbnail_worker_finished)
                        except Exception:
                            pass
                        self.thumbnail_thread.quit()
                        self.thumbnail_thread.wait()
                        logger.info("Thumbnail thread stopped.")
                except RuntimeError:
                    pass
                finally:
                    self.thumbnail_thread = None
                    self.thumbnail_worker = None

            if hasattr(self, 'extraction_thread') and self.extraction_thread is not None:
                try:
                    if self.extraction_thread.isRunning():
                        logger.info("Stopping existing extraction thread...")
                        try:
                            self.extraction_worker.finished.disconnect(self.on_extraction_worker_finished)
                        except Exception:
                            pass
                        try:
                            self.extraction_worker.progress.disconnect()
                        except Exception:
                            pass
                        try:
                            self.extraction_worker.error.disconnect()
                        except Exception:
                            pass
                        self.extraction_thread.quit()
                        self.extraction_thread.wait()
                        logger.info("Extraction thread stopped.")
                except RuntimeError:
                    pass
                finally:
                    self.extraction_thread = None
                    self.extraction_worker = None

        except Exception as e:
            logger.error(f"Error stopping existing workers: {e}")

    from darkfusion_dataset_workers import (
        deduplicate_current_dataset_clicked,
        deduplicate_dataset,
        delete_health_issues_from_dataset,
        _image_path_for_label_file,
        _mask_pair_similarity,
        _sam_masks_indicate_duplicate,
        deduplicate_single_label_file,
        get_expected_keypoint_count_for_dedupe,
        _dedupe_visible_keypoint_count,
        _dedupe_bbox_priority,
        _dedupe_keypoints_compatible,
        _bbox_center_distance,
        _bbox_area_ratio,
        _dedupe_bbox_geometry_metrics,
        _dedupe_geometry_is_duplicate,
        _box_only_keypoint_match_is_tight,
        remove_near_duplicate_bounding_boxes,
        remove_duplicate_segmentations,
        normalize_polygon,
        remove_duplicate_obbs,
        calculate_iou,
    )

    from darkfusion_dataset_review import delete_checked_grid_images

    from darkfusion_canvas import (
        _set_image_deletion_busy,
    )

    from darkfusion_dataset_review import (
        _on_grid_delete_progress,
        _grid_delete_finished,
        _apply_checked_grid_deletion,
    )

    from darkfusion_canvas import (
        delete_current_image,
        delete_files,
        is_scene_empty,
        on_heatmap_toggle,
        on_heatmap_colormap_changed,
        display_image,
        set_screen_view_scene_and_rect,
        on_zoom_lock_state_changed,
        is_zoom_locked,
        capture_zoom_lock_state,
        restore_zoom_lock_state,
        checkbox_clicked,
        on_preprocessing_control_changed,
        edge_slider_changed,
        qimage_to_cv2,
        cv2_to_qimage,
        adjust_brightness_contrast,
        replace_extension_with_txt,
        clear_blank_overlay,
        sync_scene_annotation_cache,
        annotation_scene_signature,
    )

    from darkfusion_dataset_review import refresh_annotation_preview_after_save

    from darkfusion_canvas import (
        scene_has_annotations,
        update_blank_overlay_state,
        display_image_with_text,
    )

    from darkfusion_augmentations import (
        get_head_class_id,
        generate_head_labels,
        _bbox_iou_xyxy,
        _sam3_head_candidates,
        _nested_label_scan_guides,
        _select_heads_for_parent_boxes,
        _head_line_from_bbox,
        _auto_head_annotation_bounds,
        _current_auto_head_image_path,
        _show_auto_head_preview_image,
        _ensure_nested_label_suggestion_viewer,
        _auto_verify_current_nested_image,
        _nested_review_images_for_source_class,
        open_nested_label_suggestion_review,
        show_nested_label_review_index,
        apply_current_nested_label_suggestion,
        apply_all_nested_label_suggestions,
        queue_auto_head_geometry_preview_update,
        _render_cached_auto_head_preview,
        preview_sam3_auto_heads,
        run_sam3_auto_heads,
    )

    from darkfusion_canvas import (
        _prepare_clarity_image,
        _get_super_resolution_model,
        _enhance_image_clarity,
        _apply_super_resolution_or_clarity,
        _preprocessing_state_snapshot,
        _apply_gray_brightness,
        _apply_visibility_adjustments,
        _apply_propagation_preprocessing,
        apply_preprocessing,
    )

    from darkfusion_augmentations import (
        calculate_head_area,
    )

    from darkfusion_canvas import (
        toggle_label_visibility,
    )

    from darkfusion_class_editor import (
        change_class_id,
        get_current_class_id,
        clear_classes_dropdown,
        add_item_to_classes_dropdown,
        get_classes_from_dropdown,
        get_class_names,
        set_class_names,
        refresh_class_hotkey_map,
        get_class_hotkey,
        set_class_hotkey,
        get_class_color_override,
        set_class_color_override,
        reset_class_color,
        _move_class_metadata_value,
        _class_metadata_value_present,
        rename_class_metadata,
        remove_class_metadata,
        merge_class_metadata,
        refresh_annotation_class_metadata,
        refresh_class_ui,
        snapshot_classes_dropdown_checks,
        restore_classes_dropdown_checks,
        get_live_class_source_key,
        get_class_lookup_directory,
    )

    from darkfusion_video_inference import (
        get_video_output_dir,
        _video_annotation_source,
        _current_label_frame_source,
        _capture_annotation_source_details,
        capture_annotation_paths,
        _record_capture_annotation_frame,
        video_annotation_paths,
        video_annotation_label_path,
        _record_video_annotation_frame,
        _discard_unlabeled_sparse_video_frame,
        _finalize_sparse_video_working_frame,
        _ensure_video_annotation_manifest,
        video_annotation_context_for_image,
        _video_annotation_total_frame_count,
        _read_video_annotation_frame,
        _release_video_label_navigation_capture,
        _ensure_video_dataset_metadata,
        video_frame_has_saved_labels,
        _saved_video_label_overlay,
        draw_saved_video_labels,
        _open_video_frame_in_image_labeler,
        label_current_video_frame,
    )

    from darkfusion_class_editor import (
        sync_class_labels_from_dropdown,
        get_next_available_class_id,
        update_classes_dropdown,
        handle_class_visibility_change,
        update_bbox_visibility,
        set_all_class_checkboxes,
        check_all_classes,
        uncheck_all_classes,
        toggle_all_classes,
        refresh_class_picker_if_open,
        class_input_field_return_pressed,
        remove_class_button_clicked,
    )

    # loading creating saving updating bboxes and labels

    from darkfusion_canvas import (
        get_label_file,
        display_bounding_boxes,
        save_bounding_boxes,
        load_labels,
        set_selected,
    )
            # Update the class_input_field_label text

    from darkfusion_class_editor import (
        load_classes,
    )

    from darkfusion_canvas import (
        save_labels_to_file,
        update_bounding_boxes,
        load_files,
        clear_processed_logs,
    )

    # ------------------------------------------------------------------
    # Assisted label propagation (image sequences and video frames)
    # ------------------------------------------------------------------

    from darkfusion_propagation import (
        _propagation_video_annotation_context,
        toggle_image_navigation_propagation,
        propagation_overwrite_existing_enabled,
        set_propagation_overwrite_existing,
        _propagation_label_signature,
        _compute_adjacent_propagation,
        _apply_adjacent_propagation_result,
        _propagate_labels_to_adjacent_image,
        _start_adjacent_propagation,
        _on_adjacent_propagation_worker_finished,
        _on_adjacent_propagation_completed,
        open_label_propagation_dialog,
        stop_label_propagation,
        _note_propagation_stop,
        _propagation_seed_lines_from_image,
        _propagation_seed_lines_from_video,
        _propagation_pose_ready,
        _propagation_iou,
        _propagation_shift_bbox,
        _propagation_rect_mask,
        _propagation_mask_similarity,
        _propagation_mask_agrees_with_guide,
        _sam3_video_pair_masks,
        _native_propagation_candidate_score,
        _prepare_native_propagation_records,
        _propagation_lines_from_native_masks,
        _sam3_result_masks,
        _propagation_occlusion_grace_frames,
        _initialize_native_occlusion_tracks,
        _propagation_reference_bbox,
        _update_native_occlusion_tracks,
        _propagate_lines_to_frame_native,
        _propagate_lines_to_frame_best,
        _propagation_shifted_segmentation_guide,
        _propagation_pose_line,
        _propagation_pose_candidates,
        _propagation_unique_pose_assignment,
        _propagation_pose_lines,
        _update_pose_occlusion_tracks,
        _propagate_lines_to_frame,
        _begin_propagation_batch,
        _backup_propagation_label,
        _finish_propagation_batch,
        _merge_propagated_lines,
        _propagation_scene_changed,
        _sam3_sequence_prompt_bboxes,
        _reset_sam3_video_session,
        _run_propagation_frames_native_session,
        _run_propagation_frames_fallback,
        _compute_propagation_frames,
        _run_propagation_frames,
        propagate_image_sequence_labels,
        propagate_video_labels,
        undo_last_label_propagation,
    )

    from darkfusion_canvas import (
        navigate_frame,
    )

    from darkfusion_video_inference import (
        _navigate_video_annotation_by_offset,
    )

    from darkfusion_canvas import (
        navigate_by_offset,
        _navigation_step_pending,
        next_frame,
        previous_frame,
        on_next_button_released,
        on_previous_button_released,
        update_timer_speed,
        start_next_timer,
        start_prev_timer,
        stop_next_timer,
        stop_prev_timer,
        toggle_auto_scan,
    )

    def stop_on_user_input(self, event):
        # Stop both timers if any input is detected
        self.stop_next_timer()
        self.stop_prev_timer()

        # Check if the event is a mouse or a key event and handle accordingly
        if isinstance(event, QMouseEvent):
            super().mousePressEvent(event)  # Call the appropriate event method for mouse press
        elif isinstance(event, QKeyEvent):
            super().keyPressEvent(event)  # Call the appropriate event method for key press

    def eventFilter2(self, obj, event):
        if event.type() in [QEvent.MouseButtonPress, QEvent.KeyPress]:
            self.stop_on_user_input(event)
        return super().eventFilter2(obj, event)

    # augmentation effects

    from darkfusion_augmentations import (
        apply_glass_effect,
        glass_checkbox_state_changed,
        apply_motion_blur_effect,
        motion_blur_checkbox_state_changed,
        training_augmentation_checkbox_changed,
        apply_brightness_contrast_augmentation,
        apply_hsv_jitter_augmentation,
        apply_gaussian_noise_augmentation,
        apply_jpeg_compression_augmentation,
        apply_flashbang_effect,
        flash_checkbox_changed,
        apply_smoke_effect,
        generate_smoke_mask,
        smoke_checkbox_changed,
        sight_picture_checkbox_changed,
        _normalize_sight_aim_point,
        sight_overlay_metadata,
        sight_overlay_asset_specs,
        _strengthen_sight_alpha,
        _trim_sight_overlay_bgra,
        load_sight_overlay_assets,
        _blend_bgra_overlay,
        _augmentation_class_name_for_id,
        _sight_target_center,
        _sight_label_cover_boxes,
        _sight_overlay_overcovers_labels,
        _sight_overlay_covers_normalized_point,
        _format_pose_label_line,
        _sight_pose_labels_for_overlay,
        apply_sight_overlay_augmentation,
        make_copy_checkbox_changed,
        set_flip_images,
        selected_augmentation_percent,
        selected_augmentation_output_format_value,
        augmentation_output_extension_for_image,
        sampled_augmentation_items,
        sampled_sam_augmentation_images,
        collect_current_augmentation_dataset,
        open_augmentation_preview_viewer,
        augmentation_preview_visible,
        setup_augmentation_preview_timer,
        queue_augmentation_preview_update,
        augmentation_preview_source,
        update_augmentation_preview,
        _write_classes_file,
        update_progress_bar,
        set_progress_bar_maximum,
        _load_augmentation_pose_data,
        _augmentation_keypoint_count,
        _augmentation_flip_idx,
        _looks_like_normalized_bbox,
        _format_augmented_label,
        _is_pose_label_values,
        _flip_label_line_horizontal,
        _augmentation_label_center,
        copy_original_augmentation_files,
        apply_selected_augmentation_effects,
        process_images_triggered,
        _copy_single_augmentation_source,
        _process_next_augmentation_step,
        _finish_augmentation_job,
        apply_augmentations,
    )

    # Theme GIF helpers.

    def populateGifCombo(self):
        gif_directory = "styles/gifs"
        if os.path.exists(gif_directory):  # Ensure the directory exists
            gif_files = [f for f in os.listdir(gif_directory) if f.endswith(".gif")]
            self.gif_change.clear()
            self.gif_change.addItems(gif_files)
            saved_gif = self.settings.get("lastGif", "darkfusion.gif") if hasattr(self, "settings") else "darkfusion.gif"
            gif_index = self.gif_change.findText(saved_gif)
            if gif_index >= 0:
                self.gif_change.setCurrentIndex(gif_index)

    # Update the GIF when a new one is selected from the ComboBox

    def onGifChange(self, index):
        if index != -1:  # Ensure the index is valid
            selected_gif = self.gif_change.currentText()
            gif_path = f"styles/gifs/{selected_gif}"
            if os.path.exists(gif_path):  # Ensure the GIF file exists
                if hasattr(self, "settings"):
                    self.settings["lastGif"] = selected_gif
                    self.saveSettings()
                self.movie.stop()
                self.movie.setFileName(gif_path)
                self._scale_status_gif()
                self.movie.start()
                QtCore.QTimer.singleShot(0, self._scale_status_gif)

    # Populate the Style ComboBox with available stylesheets

    def populate_style_combo_box(self):
        style_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'styles')
        style_files = sorted(
            file
            for file in os.listdir(style_folder)
            if file.endswith(('.qss', '.css', '.stylesheet', '.ess'))
            and file.lower() != "default.css"
        ) if os.path.exists(style_folder) else []

        qt_material_styles = list_themes()  # Optional: Add qt_material themes if available
        style_files = ["Default"] + style_files + sorted(qt_material_styles)
        self.styleComboBox.clear()
        self.styleComboBox.addItems(style_files)

        # Load settings and apply the last used theme
        self.settings = self.loadSettings()
        last_theme = self.settings.get('lastTheme', 'Default')
        if last_theme == "default.css":
            last_theme = "Default"
        last_theme_index = self.styleComboBox.findText(last_theme)
        self.styleComboBox.setCurrentIndex(last_theme_index if last_theme_index >= 0 else 0)

        self.apply_stylesheet()  # Apply the stylesheet
        self.update_gif_to_match_theme()  # Ensure the GIF matches the loaded theme

    # Apply the selected stylesheet to the application

    def apply_stylesheet(self):
        selected_style = self.styleComboBox.currentText()
        style_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'styles')
        if selected_style == "Default":
            file_path = os.path.join(style_folder, "default.css")
            try:
                with open(file_path, 'r', encoding="utf-8") as f:
                    self.setStyleSheet(self._style_with_global_line_spacing(f.read()))
                self._apply_global_line_spacing()
                self._apply_default_item_view_palette()
            except Exception as e:
                logger.error(f"Failed to read default stylesheet file: {e}")
                self.setStyleSheet(self._style_with_global_line_spacing(""))
        elif selected_style in list_themes():
            stylesheet = build_stylesheet(selected_style)
            self.setStyleSheet(self._style_with_global_line_spacing(
                stylesheet if stylesheet is not None else self.styleSheet()
            ))
            self._apply_global_line_spacing()
        else:
            file_path = os.path.join(style_folder, selected_style)
            try:
                with open(file_path, 'r', encoding="utf-8") as f:
                    self.setStyleSheet(self._style_with_global_line_spacing(f.read()))
                self._apply_global_line_spacing()
            except Exception as e:
                logger.error(f"Failed to read stylesheet file: {e}")
                self.setStyleSheet(self._style_with_global_line_spacing(""))  # Reset to default if there's an error

        # Ensure any Designer-baked inline styles are cleared so widgets inherit
        # the freshly applied stylesheet (fixes buttons appearing white on restart)
        try:
            self._clear_generated_inline_styles()
        except Exception:
            # Be defensive: don't crash the app if clearing styles fails
            logger.debug("_clear_generated_inline_styles failed during apply_stylesheet")

        try:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.setStyleSheet(self.styleSheet())
        except Exception:
            logger.debug("Failed to propagate stylesheet/palette to QApplication instance")

        # Some widgets (combo popups, menus) may be created slightly later by the
        # event loop or by deferred initialization. Schedule delayed clears to
        # ensure we remove any remaining Designer-baked styles.
        try:
            QtCore.QTimer.singleShot(0, self._clear_generated_inline_styles)
            QtCore.QTimer.singleShot(50, self._clear_generated_inline_styles)
            QtCore.QTimer.singleShot(500, self._clear_generated_inline_styles)
        except Exception:
            pass

        self._apply_theme_header_art()

        monitor = getattr(self, "cuda_monitor_dialog", None)
        if monitor is not None:
            monitor.apply_theme()
        roi_controls = getattr(self, "roi_edge_sliders", None)
        if roi_controls is not None:
            roi_controls.apply_theme()

    def _apply_default_item_view_palette(self):
        """
        Keep Qt item views from falling back to Fusion's light Base/AlternateBase
        when a widget has a local stylesheet.
        """
        app = QtWidgets.QApplication.instance()
        palette = app.palette() if app is not None else self.palette()

        colors = {
            QPalette.Window: QColor("#111418"),
            QPalette.WindowText: QColor("#d9e0e7"),
            QPalette.Base: QColor("#101318"),
            QPalette.AlternateBase: QColor("#151a20"),
            QPalette.Text: QColor("#d9e0e7"),
            QPalette.Button: QColor("#252b33"),
            QPalette.ButtonText: QColor("#f2f5f8"),
            QPalette.Highlight: QColor("#2f6f9f"),
            QPalette.HighlightedText: QColor("#ffffff"),
            QPalette.ToolTipBase: QColor("#20252c"),
            QPalette.ToolTipText: QColor("#f2f5f8"),
        }

        for role, color in colors.items():
            palette.setColor(QPalette.Active, role, color)
            palette.setColor(QPalette.Inactive, role, color)

        disabled_text = QColor("#6f7884")
        for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
            palette.setColor(QPalette.Disabled, role, disabled_text)
        palette.setColor(QPalette.Disabled, QPalette.Base, QColor("#0f1216"))
        palette.setColor(QPalette.Disabled, QPalette.AlternateBase, QColor("#131820"))

        self.setPalette(palette)
        if app is not None:
            app.setPalette(palette)

        for view in self.findChildren(QtWidgets.QAbstractItemView):
            view.setPalette(palette)
            viewport = view.viewport()
            if viewport is not None:
                viewport.setPalette(palette)

        for combo in self.findChildren(QtWidgets.QComboBox):
            combo.setPalette(palette)
            try:
                combo.view().setPalette(palette)
            except Exception:
                pass

    # Update the GIF to match the current theme or style

    def update_gif_to_match_theme(self):
        selected_style = self.styleComboBox.currentText()
        manifest = self._theme_manifest_for_style(selected_style)
        self._apply_theme_header_art(manifest)
        corresponding_gif = ""
        gif_path = ""
        if manifest:
            corresponding_gif = f"{Path(selected_style).stem}.gif"
            candidate = Path(manifest.get("_directory", "")) / str(manifest.get("gif", "")).strip()
            if candidate.is_file():
                gif_path = str(candidate)
        if not gif_path:
            gif_directory = "styles/gifs"
            style_base_name = os.path.splitext(
                "darkfusion.gif" if selected_style == "Default" else selected_style
            )[0]
            corresponding_gif = f"{style_base_name}.gif"
            gif_path = os.path.join(gif_directory, corresponding_gif)
        if os.path.exists(gif_path):
            if hasattr(self, "gif_change"):
                gif_index = self.gif_change.findData(gif_path)
                if gif_index < 0:
                    gif_index = self.gif_change.findText(corresponding_gif)
                if gif_index >= 0:
                    self.gif_change.blockSignals(True)
                    self.gif_change.setCurrentIndex(gif_index)
                    self.gif_change.blockSignals(False)
            self.movie.stop()
            self.movie.setFileName(gif_path)
            self._scale_status_gif()
            self.movie.start()
            QtCore.QTimer.singleShot(0, self._scale_status_gif)
        else:
            logger.warning(f"No matching GIF found for theme: {selected_style}")

    # Theme switching.

    def on_style_change(self):
        self.apply_stylesheet()
        self.saveSettings()
        self.update_gif_to_match_theme()

    # Theme startup.

    def initialize_theme_and_gif(self):
        self.populate_style_combo_box()
        self.populateGifCombo()
        self.update_gif_to_match_theme()

    # Ultralytics training helpers.

    from darkfusion_training_dialog import (
        _parse_training_imgsz_value,
    )

    from darkfusion_training_dialog import (
        _training_eval_dataset_dir,
    )

    from darkfusion_class_editor import (
        sam_text_prompt_for_class,
    )

    from darkfusion_training_dialog import (
        _set_training_data_yaml_path,
        _clear_training_data_yaml_path,
        _training_eval_paths_overlap,
        _training_eval_yaml_dataset_roots,
        _training_eval_yaml_matches_dataset,
        _training_eval_managed_output_dir,
        _training_eval_is_managed_yaml,
        _training_eval_find_obj_yaml,
        autoload_training_data_yaml_for_dataset,
        _training_eval_valid_percent,
    )

    from darkfusion_canvas import _label_path_for_image

    from darkfusion_training_dialog import (
        _training_eval_image_class_ids,
        _training_eval_prepare_cache_context,
        _training_eval_image_class_counts,
        _training_eval_label_record,
        _training_eval_cached_image_scan,
        _training_eval_split_class_report,
        _training_eval_yaml_split_report,
        _training_eval_train_val_split,
        _training_eval_percent_value,
        _training_eval_validation_mode,
        _training_eval_image_files_from_directory,
        _training_eval_read_image_list_file,
        _training_eval_sample_images,
        _training_eval_write_image_list,
        _training_eval_write_text_atomic,
        _training_eval_prepare_validation_files,
        ensure_training_evaluator_data_yaml,
    )

    from darkfusion_training_dialog import _selected_training_weights_path

    from darkfusion_training_dialog import _effective_eval_weights_path

    from darkfusion_training_dialog import (
        inspect_training_eval_weights,
        training_eval_task_from_weights,
        _training_eval_candidate_sizes,
    )

    from darkfusion_training_widgets import (
        _letterbox_preview_params,
        _train_eval_label_extent,
        _parse_training_eval_bbox,
    )

    from darkfusion_training_dialog import (
        _gpu_vram_gb,
        _looks_like_official_ultralytics_weight,
        recommend_training_optimizer_args,
        _write_training_recommendation_yaml,
        _training_eval_format_arg_value,
        _training_eval_rebuild_train_args,
        _training_eval_command_value,
        _training_eval_option_value,
        _training_eval_tune_dir_from_command,
        _training_eval_model_factor_from_path,
        _training_eval_clamp_batch,
        _training_eval_setting_bool,
        _training_eval_setting_int,
        _training_eval_effective_batch,
        _training_eval_effective_validation_batch,
        _training_eval_cache_arg,
        _training_eval_workers_arg,
        _append_ultralytics_memory_guard_args,
    )

    from darkfusion_training_dialog import (
        _training_yaml_image_count,
        _training_eval_yaml_names,
        _training_eval_yaml_image_files,
        _training_eval_data_yaml_needs_auto_refresh,
        _training_eval_refresh_data_yaml_split,
        _training_eval_refresh_stale_data_yaml,
    )

    from darkfusion_training_dialog import (
        read_training_tune_best_hyperparameters,
        apply_training_tune_best_hyperparameters,
        _training_eval_selected_parameter_args,
        _training_eval_selected_config_path,
        _training_eval_job_context,
        _training_eval_job_context_is_current,
        read_training_batch_calibration,
        apply_training_batch_calibration,
        evaluate_training_setup,
        format_training_evaluation_report,
        apply_training_evaluation_recommendations,
        browse_training_eval_weights,
    )

    from darkfusion_training_widgets import (
        _training_eval_run_dir_from_command,
        _training_eval_log_path,
        _training_eval_run_modified_time,
        _training_eval_candidate_run_roots,
        _training_eval_discover_latest_run,
        _training_eval_record_run,
        _training_eval_latest_saved_run,
        _training_eval_run_is_active,
        _training_eval_read_args_yaml,
        _training_eval_restore_last_run_setup,
        _training_eval_int_value,
        _training_eval_last_completed_epoch,
        _training_eval_resume_info,
    )

    from darkfusion_training_dialog import (
        build_training_evaluator_resume_command,
        launch_training_evaluator_logged_command,
    )

    from darkfusion_training_widgets import (
        _training_eval_read_text_tail,
        _training_eval_read_results_csv,
        _training_eval_float_value,
        _training_eval_display_metric_headers,
        _training_eval_format_csv_value,
        _training_eval_update_metrics_table,
        _training_eval_metric_series,
        _training_eval_latest_metric_line,
        _training_eval_metric_header,
        _training_eval_metric_values_for_header,
        _training_eval_last_number,
        _training_eval_window_mean,
        _training_eval_loss_values,
        _training_eval_run_args_from_log,
        _training_eval_run_args_for_compare,
        _training_eval_resolve_compare_path,
        _training_eval_compare_arg_match,
        _training_eval_parse_all_metric_from_log,
        _training_eval_best_metric_snapshot,
        _training_eval_baseline_args_from_result,
        _training_eval_data_split_fingerprint,
        _training_eval_read_baseline_record,
        _training_eval_write_baseline_validation_record,
        _training_eval_baseline_candidate_record,
        _training_eval_find_matching_baseline_validation_for_args,
        _training_eval_find_matching_baseline_validation,
        _training_eval_find_matching_baseline_for_result,
        _training_eval_format_baseline_status,
        _training_eval_update_baseline_widget,
        _training_eval_baseline_run_name,
        _training_eval_run_data_yaml_path,
        _training_eval_results_diagnosis,
        _training_eval_update_diagnosis_widgets,
        _training_eval_artifact_images,
        _training_eval_set_artifact_preview,
        open_training_eval_artifact_viewer,
        _training_eval_update_artifact_combo,
        _training_eval_set_activity_indicator,
        _training_eval_activity_state,
        _training_eval_update_eta,
        _training_eval_refresh_run_widgets,
        watch_training_evaluator_run,
    )

    from darkfusion_training_dialog import (
        _training_health_run_ultralytics_checks,
        _training_health_status_line,
        _training_health_writable_dir,
        _training_health_parse_data_yaml,
        _training_health_dataset_dir,
        _training_health_find_classes_file,
        _training_health_find_points_file,
        _training_health_engine_paths,
        build_training_health_report,
    )

    from darkfusion_dataset_review import (
        _capture_validation_similarity_origin,
        _restore_review_class_visibility,
        _restore_validation_similarity_origin,
        _apply_validation_similarity_queue,
        _sync_validation_similarity_after_save,
        _refresh_validation_similarity_after_edit,
        _back_from_validation_review,
        _ensure_validation_review_dock,
        _clear_validation_review_overlay_on_main,
        _add_validation_review_graphic,
        _draw_validation_review_object_on_main,
        _draw_validation_review_overlay_on_main,
        _validation_review_issue_guidance,
        _write_active_validation_review_report,
        _validation_review_label_signature,
        _record_validation_review_decision,
        _refresh_active_validation_ground_truth,
        _save_active_validation_review_edits,
        _validation_prediction_to_yolo_line,
        apply_current_validation_prediction_to_ground_truth,
        _validation_negative_crop_paths,
        save_validation_false_positive_crop,
        save_current_validation_false_positive_crop,
        _fit_similarity_review_docks,
        _show_validation_review_issue,
        open_validation_review_issue_in_labeler,
        navigate_validation_review_issue,
        set_current_validation_review_status,
        quarantine_current_validation_review_image,
        close_validation_review_mode,
    )

    from darkfusion_training_dialog import open_training_evaluator_dialog

    from darkfusion_training_widgets import (
        _draw_training_preview_label_text,
        _training_preview_class_color,
        _draw_training_preview_labels,
        _add_training_preview_badge,
        is_training_preview_always_on,
        set_training_preview_always_on,
        refresh_training_preview_if_enabled,
        show_training_size_preview,
        restore_original_training_view,
    )

    from darkfusion_training_dialog import (
        browse_yaml_clicked,
        browse_pt_clicked,
        open_file_dialog,
        on_save_dir_clicked,
    )

    from darkfusion_training_dialog import (
        _training_eval_default_project_dir,
        _training_eval_project_dir,
        _ray_tune_available,
        _training_eval_model_path,
        _training_eval_model_scale_rank_from_path,
        _training_eval_distillation_sanity_notes,
        _training_eval_format_distillation_notes,
        _training_eval_model_command_args,
        _ultralytics_cli_command_prefix,
        _ultralytics_train_command_prefix,
        _version_tuple,
        ultralytics_version_at_least,
        _looks_like_ultralytics_model_reference,
        _is_ultralytics_export_model_path,
        _is_ultralytics_trainable_model_path,
        _validate_ultralytics_train_model_selection,
        _ultralytics_distillation_args,
        _training_eval_resolve_command_data_yaml,
        build_training_evaluator_command,
        build_training_evaluator_tune_command,
        build_training_evaluator_batch_command,
        _format_command_display,
        launch_training_evaluator_command,
        watch_training_evaluator_tune,
        watch_training_evaluator_batch,
        _ultralytics_extra_train_args,
        ultralytics_train_clicked,
        handle_stdout,
        handle_stderr,
        process_finished,
        ultralytics_export_formats,
        saved_export_batch_size,
        build_ultralytics_export_params,
        run_ultralytics_export,
        load_model,
        initiate_conversion,
        export_model_for_conversion,
        handle_conversion,
        is_parameter_supported,
        update_gui_elements,
        format_changed,
    )

    # -----------------------------
    # TRAIN/VALID TEXT EXPORT
    # -----------------------------

    from darkfusion_darknet import (
        import_images,
        output_paths,
        create_and_update_yaml_and_data_files,
        browse_file_clicked,
        browse_data_clicked,
        browse_cfg_clicked,
        browse_weights_clicked,
        normalize_darknet_gpu_options,
        build_darknet_train_command,
        format_command_for_log,
        darknet_train,
        import_data,
        update_classes_and_batches,
        calculate_anchors,
        get_annotation_dimensions,
        anchor_iou_matrix,
        initialize_anchor_centroids,
        calculate_anchor_clusters,
        cfg_section_ranges,
        count_cfg_sections,
        parse_cfg_section_params,
        set_cfg_section_param,
        mask_indices_from_value,
        yolo_mask_count,
        default_yolo_masks,
        update_cfg_yolo_masks,
        update_yolo_filters,
        update_cfg_param,
        update_table_param,
        calculate_avg_iou,
        update_cfg_anchors,
        clean_cfg_file,
        plot_anchors,
        saveAnchorsSettings,
        main,
        clear_table,
        import_yaml,
        save_table_changes,
        parse_value,
        import_anchors,
        cfg_open_clicked,
        toggle_activation_layers,
        parse_cfg_file,
        cfg_save_clicked,
    )

     # combine txt files

    from darkfusion_dataset_workers import (
        on_combine_txt_clicked,
        select_source_folder,
        select_target_folder,
        move_file_pair,
        bulk_move_files,
        start_moving_files,
        on_bulk_move_progress,
        on_bulk_move_completed,
        on_bulk_move_failed,
        run_bulk_move,
    )


def run_pyqt_app():
    configure_gui_opencv_threads()
    # These attributes must be configured before QApplication is created.
    # Qt then works in logical pixels and keeps icons sharp across monitors.
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
    except Exception as exc:
        logger.debug(f"Could not configure Qt high-DPI attributes: {exc}")

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    app._app_shutdown_requested = False
    app._main_window_started = False

    # Show splash screen
    splash = show_splash(app)

    def request_app_shutdown():
        app._app_shutdown_requested = True
        window = getattr(app, 'main_window', None)
        viewer = getattr(window, 'frame_viewer', None)
        if viewer is not None:
            viewer._close_native_video_surface()

    app.aboutToQuit.connect(request_app_shutdown)

    # Nested so it can close the splash before showing the main window.
    def start_main_window():
        if app._app_shutdown_requested or app._main_window_started:
            splash.close()
            return

        app._main_window_started = True
        splash.close()  # close splash after delay
        app.main_window = MainWindow()  # keep a reference for the life of the app
        app.main_window.show()

    # Wait 2000 ms (2 sec) before closing splash and showing main window
    QTimer.singleShot(2000, start_main_window)

    exit_code = app.exec_()
    sys.exit(exit_code)

if __name__ == "__main__":
    run_pyqt_app()
