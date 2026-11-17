"""Auto-label workflow dialogs and their existing SAHI/DINO worker wrappers."""

import cv2
import copy
import json
import logging
import math
import numpy as np
import os
import re
import threading
import torch
import traceback
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import QEventLoop, QSettings, QThread, QTimer, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
    QGraphicsRectItem, QGraphicsTextItem, QLabel, QLineEdit, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)
from collections import Counter, defaultdict
from darkfusion_annotation_items import BoundingBoxDrawer, OBBDrawer, SegmentationDrawer
from darkfusion_annotation_style import get_color
from darkfusion_annotations import BoundingBox, class_name_index
from darkfusion_dataset_operations import (
    label_file_version, read_json_file, resolve_dataset_classes_path,
    resolve_dataset_metadata_path, resolve_dataset_pose_schema_path,
)
from darkfusion_environment import APP_DIR, configure_opencv_dnn_cache
from darkfusion_gpu_memory import (
    GpuWorkBusy, auto_label_resources, ensure_detector as ensure_gpu_detector, gpu_work,
    model_allowed as gpu_model_allowed,
)
from darkfusion_inference import (
    inference_backend_for_path, load_inference_model,
    normalize_inference_backend as _normalize_inference_backend, read_onnx_model_names,
    read_model_class_sidecar, iter_opencv_dnn_candidates,
)
from darkfusion_propagation import (
    compute_yolo_box_gap_proposals, find_yolo_box_gaps, run_propagation_worker,
    save_propagation_outputs,
)
from darkfusion_sahi import SahiPredictWrapper
from darkfusion_ui_controls import (
    CollapsibleSectionsController, CollapsibleSettingsGroup, MultipleOfSpinBox,
    PercentSliderControl, blocked_signals, snap_int_to_multiple,
)
from darkfusion_windows_ml import (
    configure as configure_onnx_runtime, settings as onnx_runtime_settings,
)
from pathlib import Path
from prediction_size_filter import (
    filter_overlay_to_size_limits as shared_filter_overlay_to_size_limits,
    filter_predictions_to_size_limits as shared_filter_predictions_to_size_limits,
    prediction_size_allowed as shared_prediction_size_allowed, prediction_size_allowed_xyxy,
)
from typing import List
from ultralytics import YOLO


logger = logging.getLogger("UltraDarkFusionLogger")


class InferenceModelLoadCancelled(GpuWorkBusy):
    """The model finished constructing after its load was cancelled."""


class _InferenceModelLoadWorker(QThread):
    """Construct one model; GUI metadata and settings remain with the caller."""

    def __init__(self, factory, parent=None):
        super().__init__(parent)
        self._factory = factory
        self.result = None
        self.error = None
        self.cancel_requested = False

    def stop(self):
        self.cancel_requested = True
        self.requestInterruption()

    def run(self):
        try:
            if not self.cancel_requested:
                model = self._factory()
                if not self.cancel_requested:
                    self.result = model
        except Exception as error:
            self.error = error.with_traceback(None)
        finally:
            self._factory = None


def _construct_model_without_blocking_gui(owner, factory):
    app = QApplication.instance()
    if app is None or not isinstance(owner, QWidget) or QThread.currentThread() != app.thread():
        return factory()
    active = getattr(owner, '_inference_model_load_worker', None)
    if active is not None:
        raise GpuWorkBusy('A model is already loading. Finish or cancel it before loading another.')
    parent = QApplication.activeModalWidget() or owner
    progress = QtWidgets.QProgressDialog('Loading inference model…', 'Cancel', 0, 0, parent)
    progress.setWindowTitle('Loading Model')
    progress.setWindowModality(Qt.WindowModal)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    worker = _InferenceModelLoadWorker(factory, owner)
    owner._inference_model_load_worker = worker
    loop = QEventLoop()
    worker.finished.connect(loop.quit, Qt.QueuedConnection)
    progress.canceled.connect(worker.stop)
    # Keep the GUI responsive while a retired predictor drains. A new session
    # must not initialize concurrently with the old detector's final enqueue.
    from darkfusion_video_inference import VideoInferenceThread
    retired = [thread for thread in list(getattr(owner, '_stopping_qthreads', []) or [])
               if isinstance(thread, VideoInferenceThread)]
    gate = QTimer(progress)
    gate.setInterval(10)
    def launch():
        if worker.cancel_requested:
            gate.stop()
            loop.quit()
        elif not any(owner.qthread_is_running(thread) for thread in retired):
            gate.stop()
            worker.start()
    gate.timeout.connect(launch)
    try:
        progress.show()
        gate.start()
        loop.exec_()
        if worker.isRunning() or worker.cancel_requested:
            raise InferenceModelLoadCancelled('Model loading cancelled.')
        if worker.error is not None:
            raise worker.error
        return worker.result
    finally:
        progress.blockSignals(True)
        gate.stop()
        progress.close()
        progress.deleteLater()
        if worker.isRunning():
            worker.stop()
            worker.finished.connect(lambda: setattr(owner, '_inference_model_load_worker', None), Qt.QueuedConnection)
            owner.park_stopping_qthread(worker, 'Inference model loader')
        else:
            owner._inference_model_load_worker = None
            worker.result = None
            worker.deleteLater()


class SahiWorker(QThread):
    progress = pyqtSignal(int, int, str, int)
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)
    annotation_task = pyqtSignal(str, object)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = dict(config)

    def run(self):
        try:
            # This is a worker thread: use the immutable snapshot captured by
            # the dialog instead of reading Qt widgets on the worker thread.
            roi = dict(self.config.get("roi_inference", {}) or {})

            predictor = SahiPredictWrapper(
                model_type=self.config["model_type"],
                model_path=self.config["model_weights"],
                confidence_threshold=self.config["confidence_threshold"],
                device=self.config["device"],
                fp16=self.config.get("fp16", False),
                inference_backend=self.config.get("inference_backend", "auto"),
                onnx_provider=self.config.get("onnx_provider", "auto"),
                onnx_runtime_options=self.config.get("onnx_runtime_options"),
                postprocess_type=self.config["postprocess_type"],
                postprocess_match_metric=self.config["postprocess_match_metric"],
                postprocess_match_threshold=self.config["postprocess_match_threshold"],
                postprocess_class_agnostic=self.config["postprocess_class_agnostic"],
                perform_standard_pred=self.config["perform_standard_pred"],
                show_preview=False,
                min_size_px=self.config.get("min_size_px", 0.0),
                max_percent=self.config.get("max_percent", 1.0),
                ignore_teammates=self.config.get("ignore_teammates", False),
                teammate_threshold=self.config.get("teammate_threshold", 0.90),
                roi_enabled=bool(roi.get("enabled", False)),
                roi_bounds=roi.get("bounds_list") or roi.get("bounds"),
                roi_reference_size=roi.get("reference_size"),
            )

            selected_model = getattr(predictor, "onnx_runtime_model", None)
            if selected_model is None:
                selected_model = getattr(getattr(predictor, "detection_model", None), "model", None)
            self.annotation_task.emit(str(getattr(predictor, "task", "detect")),
                {"model": selected_model, "model_path": self.config["model_weights"]})
            summary = predictor.process_folder(
                folder_path=self.config["image_directory"],
                class_names_file=self.config["classes_txt_path"],
                slice_height=self.config["slice_height"],
                slice_width=self.config["slice_width"],
                overlap_height_ratio=self.config["overlap_height_ratio"],
                overlap_width_ratio=self.config["overlap_width_ratio"],
                desired_classes=self.config["selected_classes"],
                overwrite=self.config["overwrite"],
                progress_callback=self.progress.emit,
                class_names=self.config.get("class_names"),
            )
            self.completed.emit(summary)
        except Exception:
            self.failed.emit(traceback.format_exc())


class SahiSettingsDialog(QDialog):

    def __init__(self, parent=None, main_window=None, shared_settings_provider=None):
        super().__init__(parent)
        self.main_window = main_window or parent
        self.shared_settings_provider = shared_settings_provider
        self.uses_shared_auto_label_settings = callable(shared_settings_provider)
        self.worker = None
        self._applying_preset = False
        self.sahi_settings = QSettings("UltraDarkFusion", "SAHI")

        self.setWindowTitle("SAHI Slice Labeling")
        self.setMinimumSize(760, 700)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel("SAHI Slice Labeling")
        title_font = title.font()
        title_font.setPointSize(max(title_font.pointSize() + 4, 14))
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        subtitle = QLabel("Run tiled inference for large images and small objects, using your current dataset classes.")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        main_split = QtWidgets.QSplitter(Qt.Horizontal)
        main_split.setChildrenCollapsible(False)
        root.addWidget(main_split, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        main_split.addWidget(left_panel)
        self._hidden_class_panel = right_panel
        self._hidden_class_panel.hide()
        main_split.setStretchFactor(0, 1)

        self._build_model_group(left_layout)
        self._build_dataset_group(left_layout)
        self._build_slicing_group(left_layout)
        video_group = QtWidgets.QGroupBox('Video / live inference')
        video_layout = QVBoxLayout(video_group)
        self.videoSahiCheckBox = QCheckBox('Use SAHI for video / live inference')
        self.videoSahiCheckBox.setChecked(bool(getattr(self.main_window, 'settings', {}).get('videoSahiEnabled', False)))
        video_layout.addWidget(self.videoSahiCheckBox)
        self.useVideoWeightsBtn = QPushButton('Use these weights for video')
        self.useVideoWeightsBtn.clicked.connect(self.useWeightsForVideo)
        video_buttons = QtWidgets.QHBoxLayout()
        video_buttons.addWidget(self.useVideoWeightsBtn)
        self.fastVideoPresetBtn = QPushButton('Fast video preset')
        self.fastVideoPresetBtn.setToolTip('Uses larger slices to cover the current video in at most 2 x 2 tiles. Fewer passes are faster; smaller slices above give distant objects more detail. Keeps your overlap and full-image-pass choices.')
        self.fastVideoPresetBtn.clicked.connect(self.useFastVideoPreset)
        video_buttons.addWidget(self.fastVideoPresetBtn)
        video_layout.addLayout(video_buttons)
        note = QLabel('Uses the current video weights, confidence, checked classes and ROI, with the slicing settings here. More slices can improve small-object detection but reduce inference FPS.')
        note.setWordWrap(True)
        video_layout.addWidget(note)
        left_layout.addWidget(video_group)
        self._build_postprocess_group(left_layout)
        left_layout.addStretch(1)

        self._build_class_group(right_layout)

        self.statusLabel = QLabel("Ready.")
        self.statusLabel.setWordWrap(True)
        root.addWidget(self.statusLabel)

        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        root.addWidget(self.progressBar)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        self.runBtn = QPushButton("Run SAHI")
        self.runBtn.clicked.connect(self.confirmSettings)
        self.closeBtn = QPushButton("Close")
        self.closeBtn.clicked.connect(self.close)
        button_row.addWidget(self.runBtn)
        button_row.addWidget(self.closeBtn)
        root.addLayout(button_row)

        self._load_saved_settings()
        self._setup_settings_autosave()
        self.refreshClasses()

    def selectModelWeights(self):
        start_path = self.weightsLineEdit.text()
        if hasattr(self.main_window, "dialog_start_directory"):
            start_path = self.main_window.dialog_start_directory("model", start_path)
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Weights File",
            start_path or os.getcwd(),
            "Model Files (*.pt *.onnx *.engine);;All Files (*)",
        )
        if fileName:
            if hasattr(self.main_window, "remember_dialog_selection"):
                self.main_window.remember_dialog_selection("model", fileName)
            self.weightsLineEdit.setText(fileName)
            self.refreshClasses()

    def useWeightsForVideo(self):
        path = self.weightsLineEdit.text().strip()
        if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in {'.pt', '.onnx', '.engine'}:
            QMessageBox.warning(self, 'SAHI video', 'Select an existing .pt, .onnx or .engine model first.')
            return
        owner = self.main_window
        if owner is None or not hasattr(owner, 'open_weights'):
            return
        if not gpu_model_allowed(owner, 'detector'):
            self.statusLabel.setText('Video model loading is unavailable while a GPU job has released the detector.')
            return
        owner.open_weights(path)
        if getattr(owner, 'model', None) is not None and os.path.normcase(os.path.abspath(getattr(owner, 'loaded_model_path', '') or '')) == os.path.normcase(os.path.abspath(path)):
            self.videoSahiCheckBox.setChecked(True)
            self._save_settings()
            owner.queue_live_inference_refresh('SAHI video weights selected')
            self.statusLabel.setText('SAHI video is enabled. Play a video with Detect enabled.')

    def slicingOptions(self):
        return {
            'slice_height': self.sliceHeightSpinBox.value(),
            'slice_width': self.sliceWidthSpinBox.value(),
            'overlap_height_ratio': self.overlapHeightRatioSpinBox.value(),
            'overlap_width_ratio': self.overlapWidthRatioSpinBox.value(),
            'postprocess_type': self.postprocessTypeComboBox.currentText(),
            'postprocess_match_metric': self.postprocessMetricComboBox.currentText(),
            'postprocess_match_threshold': self.postprocessThresholdSpinBox.value(),
            'postprocess_class_agnostic': self.classAgnosticCheckBox.isChecked(),
            'perform_standard_pred': self.performStandardPredCheckBox.isChecked(),
        }

    def useFastVideoPreset(self):
        frame = getattr(self.main_window, '_last_video_frame_bgr', None)
        if not isinstance(frame, np.ndarray) or frame.ndim < 2 or not frame.size:
            self.statusLabel.setText('Play or pause a video first, then apply its fast slicing preset.')
            return
        height, width = frame.shape[:2]
        for dimension, overlap, control in (
            (height, self.overlapHeightRatioSpinBox.value(), self.sliceHeightSpinBox),
            (width, self.overlapWidthRatioSpinBox.value(), self.sliceWidthSpinBox),
        ):
            # Round upward so the two-slice coverage is never shortened.
            value = int(math.ceil(dimension / (2. - overlap) / 32.)) * 32
            control.setValue(max(control.value(), value))
        self._save_settings()
        self.statusLabel.setText('Fast video preset applied: larger slices, up to 2 x 2 tiles. Smaller slices provide more distant-object detail.')

    def selectImageDirectory(self):
        start_path = self.imageDirLineEdit.text()
        if hasattr(self.main_window, "dialog_start_directory"):
            start_path = self.main_window.dialog_start_directory("dataset", start_path)
        dirName = QFileDialog.getExistingDirectory(
            self,
            "Select Image Directory",
            start_path or os.getcwd(),
        )
        if dirName:
            if hasattr(self.main_window, "remember_dialog_selection"):
                self.main_window.remember_dialog_selection("dataset", dirName)
            self.imageDirLineEdit.setText(dirName)
            self.refreshClasses()

    def _build_model_group(self, parent_layout):
        group = QtWidgets.QGroupBox("Model")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)

        self.modelTypeComboBox = QComboBox()
        self.modelTypeComboBox.addItem("Ultralytics YOLO", "ultralytics")
        self.modelTypeComboBox.addItem("YOLOv5 legacy", "yolov5")
        self.modelTypeComboBox.addItem("YOLOv8 ONNX", "yolov8onnx")
        self.modelTypeComboBox.addItem("RT-DETR", "rtdetr")
        form.addRow("Type", self.modelTypeComboBox)

        weights_row = QtWidgets.QHBoxLayout()
        self.weightsLineEdit = QLineEdit()
        self.weightsLineEdit.setPlaceholderText("Select a .pt, .onnx, or .engine model")
        self.weightsLineEdit.editingFinished.connect(lambda: self.refreshClasses(prefer_weights=True))
        selectWeightsBtn = QPushButton("Browse")
        selectWeightsBtn.clicked.connect(self.selectModelWeights)
        weights_row.addWidget(self.weightsLineEdit, 1)
        weights_row.addWidget(selectWeightsBtn)
        form.addRow("Weights", weights_row)

        self.deviceComboBox = QComboBox()
        self.deviceComboBox.addItem("Auto", "auto")
        self.deviceComboBox.addItem("CUDA 0", "cuda:0")
        self.deviceComboBox.addItem("CPU", "cpu")
        form.addRow("Device", self.deviceComboBox)

        self.confidenceThresholdSpinBox = PercentSliderControl(0.01, 1.0, 0.35)
        self.confidenceThresholdSpinBox.setRange(0.01, 1.0)
        self.confidenceThresholdSpinBox.setSingleStep(0.05)
        self.confidenceThresholdSpinBox.setDecimals(2)
        self.confidenceThresholdSpinBox.setValue(0.35)
        form.addRow("Confidence", self.confidenceThresholdSpinBox)

        parent_layout.addWidget(group)

    def _build_dataset_group(self, parent_layout):
        group = QtWidgets.QGroupBox(
            "Output" if self.uses_shared_auto_label_settings else "Dataset"
        )
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)

        self.imageDirLineEdit = QLineEdit(group)
        self.imageDirLineEdit.setPlaceholderText("Dataset folder containing images")
        self.imageDirLineEdit.editingFinished.connect(self.refreshClasses)
        self.overwriteCheckBox = QCheckBox("Overwrite existing label files", group)
        self.overwriteCheckBox.setChecked(True)
        if not self.uses_shared_auto_label_settings:
            image_row = QtWidgets.QHBoxLayout()
            selectImageDirBtn = QPushButton("Browse")
            selectImageDirBtn.clicked.connect(self.selectImageDirectory)
            image_row.addWidget(self.imageDirLineEdit, 1)
            image_row.addWidget(selectImageDirBtn)
            form.addRow("Images", image_row)
            form.addRow("", self.overwriteCheckBox)
        else:
            self.imageDirLineEdit.hide()
            self.overwriteCheckBox.hide()

        self.showPreviewCheckBox = QCheckBox("Preview results in DarkFusion graphics view")
        self.showPreviewCheckBox.setToolTip("Refresh the main DarkFusion viewer as SAHI writes labels. Turn this off for maximum speed.")
        self.showPreviewCheckBox.setChecked(True)
        form.addRow("", self.showPreviewCheckBox)

        parent_layout.addWidget(group)

    def _build_slicing_group(self, parent_layout):
        group = QtWidgets.QGroupBox("Slicing")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)

        self.presetComboBox = QComboBox()
        self.presetComboBox.addItem("Balanced", {"height": 640, "width": 640, "overlap": 0.20})
        self.presetComboBox.addItem("Small Objects", {"height": 512, "width": 512, "overlap": 0.25})
        self.presetComboBox.addItem("Tiny Objects", {"height": 384, "width": 384, "overlap": 0.30})
        self.presetComboBox.addItem("Fast Pass", {"height": 768, "width": 768, "overlap": 0.15})
        self.presetComboBox.addItem("Custom", None)
        self.presetComboBox.currentIndexChanged.connect(self.applyPreset)
        form.addRow("Preset", self.presetComboBox)

        size_row = QtWidgets.QHBoxLayout()
        self.sliceHeightSpinBox = MultipleOfSpinBox(32)
        self.sliceHeightSpinBox.setRange(128, 4096)
        self.sliceHeightSpinBox.setSingleStep(32)
        self.sliceHeightSpinBox.setValue(640)
        self.sliceHeightSpinBox.setMaximumWidth(120)
        self.sliceWidthSpinBox = MultipleOfSpinBox(32)
        self.sliceWidthSpinBox.setRange(128, 4096)
        self.sliceWidthSpinBox.setSingleStep(32)
        self.sliceWidthSpinBox.setValue(640)
        self.sliceWidthSpinBox.setMaximumWidth(120)
        size_row.addWidget(self.sliceHeightSpinBox)
        size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.sliceWidthSpinBox)
        form.addRow("Slice H x W", size_row)

        overlap_row = QtWidgets.QHBoxLayout()
        self.overlapHeightRatioSpinBox = PercentSliderControl(0.0, 0.9, 0.20)
        self.overlapHeightRatioSpinBox.setRange(0.0, 0.9)
        self.overlapHeightRatioSpinBox.setSingleStep(0.05)
        self.overlapHeightRatioSpinBox.setDecimals(2)
        self.overlapHeightRatioSpinBox.setValue(0.20)
        self.overlapWidthRatioSpinBox = PercentSliderControl(0.0, 0.9, 0.20)
        self.overlapWidthRatioSpinBox.setRange(0.0, 0.9)
        self.overlapWidthRatioSpinBox.setSingleStep(0.05)
        self.overlapWidthRatioSpinBox.setDecimals(2)
        self.overlapWidthRatioSpinBox.setValue(0.20)
        overlap_row.addWidget(self.overlapHeightRatioSpinBox)
        overlap_row.addWidget(QLabel("x"))
        overlap_row.addWidget(self.overlapWidthRatioSpinBox)
        form.addRow("Overlap H x W", overlap_row)

        self.performStandardPredCheckBox = QCheckBox("Also run full-image pass")
        self.performStandardPredCheckBox.setChecked(True)
        form.addRow("", self.performStandardPredCheckBox)

        for widget in (
            self.sliceHeightSpinBox,
            self.sliceWidthSpinBox,
            self.overlapHeightRatioSpinBox,
            self.overlapWidthRatioSpinBox,
        ):
            widget.valueChanged.connect(self.markCustomPreset)

        parent_layout.addWidget(group)

    def _build_postprocess_group(self, parent_layout):
        group = QtWidgets.QGroupBox("Merge Overlapping Slices")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)

        self.postprocessTypeComboBox = QComboBox()
        self.postprocessTypeComboBox.addItems(["GREEDYNMM", "NMS", "NMM"])
        form.addRow("Mode", self.postprocessTypeComboBox)

        self.postprocessMetricComboBox = QComboBox()
        self.postprocessMetricComboBox.addItems(["IOS", "IOU"])
        form.addRow("Metric", self.postprocessMetricComboBox)

        self.postprocessThresholdSpinBox = PercentSliderControl(0.05, 0.95, 0.50)
        self.postprocessThresholdSpinBox.setRange(0.05, 0.95)
        self.postprocessThresholdSpinBox.setSingleStep(0.05)
        self.postprocessThresholdSpinBox.setDecimals(2)
        self.postprocessThresholdSpinBox.setValue(0.50)
        form.addRow("Threshold", self.postprocessThresholdSpinBox)

        self.classAgnosticCheckBox = QCheckBox("Merge across classes")
        self.classAgnosticCheckBox.setChecked(False)
        form.addRow("", self.classAgnosticCheckBox)

        parent_layout.addWidget(group)

    def _build_class_group(self, parent_layout):
        group = QtWidgets.QGroupBox("Classes")
        layout = QVBoxLayout(group)

        self.classSummaryLabel = QLabel("0 selected")
        layout.addWidget(self.classSummaryLabel)

        self.classSearchLineEdit = QLineEdit()
        self.classSearchLineEdit.setPlaceholderText("Search classes")
        self.classSearchLineEdit.textChanged.connect(self.filterClassList)
        layout.addWidget(self.classSearchLineEdit)

        action_row = QtWidgets.QHBoxLayout()
        select_all_btn = QPushButton("All")
        select_all_btn.clicked.connect(lambda: self.setClassChecks(True))
        clear_btn = QPushButton("None")
        clear_btn.clicked.connect(lambda: self.setClassChecks(False))
        checked_btn = QPushButton("Use Checked")
        checked_btn.clicked.connect(self.useCheckedClassesFromParent)
        weights_btn = QPushButton("Weights")
        weights_btn.clicked.connect(lambda: self.refreshClasses(prefer_weights=True))
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self.refreshClasses)
        action_row.addWidget(select_all_btn)
        action_row.addWidget(clear_btn)
        action_row.addWidget(checked_btn)
        action_row.addWidget(weights_btn)
        action_row.addWidget(reload_btn)
        layout.addLayout(action_row)

        self.classListWidget = QtWidgets.QListWidget()
        self.classListWidget.setAlternatingRowColors(True)
        self.classListWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.classListWidget.setUniformItemSizes(True)
        self.classListWidget.itemChanged.connect(self.updateClassSummary)
        layout.addWidget(self.classListWidget, 1)

        parent_layout.addWidget(group, 1)

    def applyPreset(self):
        preset = self.presetComboBox.currentData()
        if not preset:
            return

        self._applying_preset = True
        try:
            self.sliceHeightSpinBox.setValue(preset["height"])
            self.sliceWidthSpinBox.setValue(preset["width"])
            self.overlapHeightRatioSpinBox.setValue(preset["overlap"])
            self.overlapWidthRatioSpinBox.setValue(preset["overlap"])
        finally:
            self._applying_preset = False

    def markCustomPreset(self):
        if self._applying_preset:
            return
        custom_index = self.presetComboBox.findText("Custom")
        if custom_index >= 0 and self.presetComboBox.currentIndex() != custom_index:
            self.presetComboBox.setCurrentIndex(custom_index)

    def _load_saved_settings(self):
        self._loading_settings = True
        try:
            self._restoreComboByData(
                self.modelTypeComboBox,
                self.sahi_settings.value("model_type", "ultralytics"),
            )
            self._restoreComboByData(
                self.deviceComboBox,
                self.sahi_settings.value("device", "auto"),
            )

            parent_weights = getattr(self.main_window, "weights_file_path", None)
            saved_weights = self.sahi_settings.value("model_weights", "")
            self.weightsLineEdit.setText(parent_weights or saved_weights or "")

            parent_image_dir = getattr(self.main_window, "image_directory", None)
            saved_image_dir = self.sahi_settings.value("image_directory", "")
            self.imageDirLineEdit.setText(parent_image_dir or saved_image_dir or "")

            self.sliceHeightSpinBox.setValue(int(self.sahi_settings.value("slice_height", 640)))
            self.sliceWidthSpinBox.setValue(int(self.sahi_settings.value("slice_width", 640)))
            self.overlapHeightRatioSpinBox.setValue(float(self.sahi_settings.value("overlap_height_ratio", 0.20)))
            self.overlapWidthRatioSpinBox.setValue(float(self.sahi_settings.value("overlap_width_ratio", 0.20)))
            self.confidenceThresholdSpinBox.setValue(float(self.sahi_settings.value("confidence_threshold", 0.35)))
            self.postprocessTypeComboBox.setCurrentText(self.sahi_settings.value("postprocess_type", "GREEDYNMM"))
            self.postprocessMetricComboBox.setCurrentText(self.sahi_settings.value("postprocess_match_metric", "IOS"))
            self.postprocessThresholdSpinBox.setValue(float(self.sahi_settings.value("postprocess_match_threshold", 0.50)))
            self.overwriteCheckBox.setChecked(self._settings_bool("overwrite", True))
            self.performStandardPredCheckBox.setChecked(self._settings_bool("perform_standard_pred", True))
            self.classAgnosticCheckBox.setChecked(self._settings_bool("postprocess_class_agnostic", False))
            self.showPreviewCheckBox.setChecked(self._settings_bool("show_preview", True))
            saved_preset = str(self.sahi_settings.value("slice_preset", "Custom") or "Custom")
            preset_index = self.presetComboBox.findText(saved_preset)
            if preset_index >= 0:
                self.presetComboBox.setCurrentIndex(preset_index)
            else:
                self.markCustomPreset()
        finally:
            self._loading_settings = False

    def _save_settings(self):
        if getattr(self, "_loading_settings", False):
            return

        self.sahi_settings.setValue("model_type", self.modelTypeComboBox.currentData())
        self.sahi_settings.setValue("device", self.deviceComboBox.currentData())
        self.sahi_settings.setValue("model_weights", self.weightsLineEdit.text().strip())
        if not self.uses_shared_auto_label_settings:
            self.sahi_settings.setValue("image_directory", self.imageDirLineEdit.text().strip())
        self.sahi_settings.setValue("slice_preset", self.presetComboBox.currentText())
        self.sahi_settings.setValue("slice_height", self.sliceHeightSpinBox.value())
        self.sahi_settings.setValue("slice_width", self.sliceWidthSpinBox.value())
        self.sahi_settings.setValue("overlap_height_ratio", self.overlapHeightRatioSpinBox.value())
        self.sahi_settings.setValue("overlap_width_ratio", self.overlapWidthRatioSpinBox.value())
        self.sahi_settings.setValue("confidence_threshold", self.confidenceThresholdSpinBox.value())
        self.sahi_settings.setValue("postprocess_type", self.postprocessTypeComboBox.currentText())
        self.sahi_settings.setValue("postprocess_match_metric", self.postprocessMetricComboBox.currentText())
        self.sahi_settings.setValue("postprocess_match_threshold", self.postprocessThresholdSpinBox.value())
        if not self.uses_shared_auto_label_settings:
            self.sahi_settings.setValue("overwrite", self.overwriteCheckBox.isChecked())
        self.sahi_settings.setValue("perform_standard_pred", self.performStandardPredCheckBox.isChecked())
        self.sahi_settings.setValue("postprocess_class_agnostic", self.classAgnosticCheckBox.isChecked())
        self.sahi_settings.setValue("show_preview", self.showPreviewCheckBox.isChecked())
        owner = self.main_window
        settings = getattr(owner, 'settings', None)
        if isinstance(settings, dict):
            enabled = self.videoSahiCheckBox.isChecked()
            options = self.slicingOptions()
            changed = settings.get('videoSahiEnabled', False) != enabled
            changed = changed or (enabled and settings.get('videoSahiOptions') != options)
            settings.update(videoSahiEnabled=enabled, videoSahiOptions=options)
            if hasattr(owner, 'queue_settings_save'):
                owner.queue_settings_save()
            if changed and hasattr(owner, 'queue_live_inference_refresh'):
                owner.queue_live_inference_refresh('SAHI video settings changed')

    def _setup_settings_autosave(self):
        def save(*_args):
            self._save_settings()

        for widget, signal_name in (
            (getattr(self, "modelTypeComboBox", None), "currentIndexChanged"),
            (getattr(self, "deviceComboBox", None), "currentIndexChanged"),
            (getattr(self, "weightsLineEdit", None), "textChanged"),
            (getattr(self, "imageDirLineEdit", None), "textChanged"),
            (getattr(self, "presetComboBox", None), "currentIndexChanged"),
            (getattr(self, "sliceHeightSpinBox", None), "valueChanged"),
            (getattr(self, "sliceWidthSpinBox", None), "valueChanged"),
            (getattr(self, "overlapHeightRatioSpinBox", None), "valueChanged"),
            (getattr(self, "overlapWidthRatioSpinBox", None), "valueChanged"),
            (getattr(self, "confidenceThresholdSpinBox", None), "valueChanged"),
            (getattr(self, "postprocessTypeComboBox", None), "currentIndexChanged"),
            (getattr(self, "postprocessMetricComboBox", None), "currentIndexChanged"),
            (getattr(self, "postprocessThresholdSpinBox", None), "valueChanged"),
            (getattr(self, "overwriteCheckBox", None), "stateChanged"),
            (getattr(self, "performStandardPredCheckBox", None), "stateChanged"),
            (getattr(self, "classAgnosticCheckBox", None), "stateChanged"),
            (getattr(self, "showPreviewCheckBox", None), "stateChanged"),
            (getattr(self, "videoSahiCheckBox", None), "stateChanged"),
        ):
            if widget is None:
                continue
            try:
                getattr(widget, signal_name).connect(save)
            except Exception:
                pass

    def _settings_bool(self, key, default):
        value = self.sahi_settings.value(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _restoreComboByData(self, combo, value):
        for index in range(combo.count()):
            if str(combo.itemData(index)) == str(value):
                combo.setCurrentIndex(index)
                return

    def _normalize_model_names(self, names):
        if isinstance(names, dict):
            ordered = []
            try:
                keys = sorted(names.keys(), key=lambda key: int(key))
            except Exception:
                keys = sorted(names.keys())

            for key in keys:
                name = str(names[key]).strip()
                if name:
                    ordered.append(name)
            return ordered

        if isinstance(names, (list, tuple)):
            return [str(name).strip() for name in names if str(name).strip()]

        return []

    def _read_sidecar_model_classes(self, model_path):
        if not model_path:
            return []
        try:
            names = read_model_class_sidecar(model_path)
            return [names[index] for index in sorted(names)]
        except Exception as e:
            logger.warning(f"Could not read SAHI model class file: {e}")
            return []

    def _read_classes_from_selected_weights(self):
        model_path = self.weightsLineEdit.text().strip()
        if not model_path or not os.path.exists(model_path):
            return []

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            if model_path.lower().endswith(".onnx"):
                options = getattr(self.main_window, "settings", {}).get("onnxRuntimeOptions", {})
                names = self._normalize_model_names(read_onnx_model_names(model_path, options.get("names", "")))
                if names:
                    return names
                return self._read_sidecar_model_classes(model_path)
            if not gpu_model_allowed(self.main_window, "detector"):
                return self._read_sidecar_model_classes(model_path)
            model = load_inference_model(model_path)
            names = self._normalize_model_names(getattr(model, "names", None))
            try:
                del model
            except Exception:
                pass
            if names:
                return names
        except Exception as e:
            logger.warning(f"Could not read class names from SAHI weights {model_path}: {e}")
        finally:
            QApplication.restoreOverrideCursor()

        return self._read_sidecar_model_classes(model_path)

    def refreshClasses(self, prefer_weights=False):
        image_directory = self.imageDirLineEdit.text().strip()
        classes_txt_path = (
            resolve_dataset_classes_path(image_directory, migrate=True)
            if image_directory else ""
        )
        classes = []
        checked_names = None
        source_label = ""

        parent = self.main_window
        parent_check_state = {}
        if parent is not None and hasattr(parent, "get_classes_from_dropdown"):
            if hasattr(parent, "snapshot_classes_dropdown_checks"):
                parent_check_state = parent.snapshot_classes_dropdown_checks()

        if parent is not None and hasattr(parent, "get_classes_from_dropdown"):
            classes = parent.get_classes_from_dropdown()
            if classes:
                source_label = "DarkFusion class list"
                checked_names = {
                    name for name, state in parent_check_state.items()
                    if state == Qt.Checked
                } if parent_check_state else set(classes)

        if not classes and prefer_weights:
            classes = self._read_classes_from_selected_weights()
            if classes:
                source_label = "selected weights"
                checked_names = {
                    name for name in classes
                    if parent_check_state.get(name, Qt.Checked) == Qt.Checked
                }

        if not classes and classes_txt_path and os.path.exists(classes_txt_path):
            try:
                with open(classes_txt_path, "r", encoding="utf-8") as f:
                    classes = [line.strip() for line in f if line.strip()]
                source_label = "classes.txt"
            except Exception as e:
                self.statusLabel.setText(f"Could not read classes.txt: {e}")

        self.setClassItems(classes, checked_names=checked_names)
        if classes:
            self.statusLabel.setText(f"Loaded {len(classes)} classes from {source_label}.")
        else:
            self.statusLabel.setText("No classes loaded yet. Choose a dataset with .darkfusion/classes.txt.")

    def useCheckedClassesFromParent(self):
        if self.main_window is None or not hasattr(self.main_window, "get_active_class_names_from_ui"):
            return

        checked = set(self.main_window.get_active_class_names_from_ui())
        if not checked:
            return

        for row in range(self.classListWidget.count()):
            item = self.classListWidget.item(row)
            item.setCheckState(Qt.Checked if item.text() in checked else Qt.Unchecked)
        self.updateClassSummary()

    def setClassItems(self, classes, checked_names=None):
        checked_names = set(classes or []) if checked_names is None else set(checked_names)
        self.classListWidget.blockSignals(True)
        self.classListWidget.clear()

        for class_name in classes:
            item = QtWidgets.QListWidgetItem(class_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if class_name in checked_names else Qt.Unchecked)
            self.classListWidget.addItem(item)

        self.classListWidget.blockSignals(False)
        self.filterClassList(self.classSearchLineEdit.text())
        self.updateClassSummary()

    def setClassChecks(self, checked):
        state = Qt.Checked if checked else Qt.Unchecked
        self.classListWidget.blockSignals(True)
        for row in range(self.classListWidget.count()):
            self.classListWidget.item(row).setCheckState(state)
        self.classListWidget.blockSignals(False)
        self.updateClassSummary()

    def filterClassList(self, text):
        needle = (text or "").strip().lower()
        for row in range(self.classListWidget.count()):
            item = self.classListWidget.item(row)
            item.setHidden(bool(needle and needle not in item.text().lower()))

    def selectedClassNames(self):
        selected = []
        for row in range(self.classListWidget.count()):
            item = self.classListWidget.item(row)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected

    def allClassNames(self):
        return [
            self.classListWidget.item(row).text()
            for row in range(self.classListWidget.count())
            if self.classListWidget.item(row) is not None
        ]

    def updateClassSummary(self):
        selected = len(self.selectedClassNames())
        total = self.classListWidget.count()
        self.classSummaryLabel.setText(f"{selected} of {total} selected")

    def _resolved_device(self):
        device = self.deviceComboBox.currentData()
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    def _set_running(self, running):
        self.runBtn.setEnabled(not running)
        self.closeBtn.setEnabled(not running)
        for widget in (
            self.modelTypeComboBox,
            self.weightsLineEdit,
            self.imageDirLineEdit,
            self.sliceHeightSpinBox,
            self.sliceWidthSpinBox,
            self.overlapHeightRatioSpinBox,
            self.overlapWidthRatioSpinBox,
            self.confidenceThresholdSpinBox,
            self.postprocessTypeComboBox,
            self.postprocessMetricComboBox,
            self.postprocessThresholdSpinBox,
            self.overwriteCheckBox,
            self.performStandardPredCheckBox,
            self.classAgnosticCheckBox,
            self.showPreviewCheckBox,
            self.videoSahiCheckBox,
            self.useVideoWeightsBtn,
            self.fastVideoPresetBtn,
            self.classListWidget,
            self.classSearchLineEdit,
        ):
            widget.setEnabled(not running)

    def confirmSettings(self):
        shared = {}
        if self.uses_shared_auto_label_settings:
            shared = dict(self.shared_settings_provider() or {})
        elif self.main_window is not None:
            # The older standalone SAHI menu must use the same runtime profile
            # as the integrated Auto Label tab, not reset its provider to Auto.
            project_settings = getattr(self.main_window, "settings", {}) or {}
            shared = {
                "inference_backend": project_settings.get("inferenceBackend", "auto"),
                "onnx_provider": project_settings.get("onnxExecutionProvider", "auto"),
                "onnx_runtime_options": dict(project_settings.get("onnxRuntimeOptions", {}) or {}),
                "fp16": bool(getattr(self.main_window, "fp_mode", 0) == 1),
            }
            if hasattr(self.main_window, "auto_label_teammate_filter_config"):
                ignore, threshold = self.main_window.auto_label_teammate_filter_config()
                shared.update(ignore_teammates=ignore, teammate_threshold=threshold)
        model_type = self.modelTypeComboBox.currentData()
        model_weights = self.weightsLineEdit.text().strip()
        image_directory = str(
            shared.get("image_directory") or self.imageDirLineEdit.text().strip()
        )
        self.imageDirLineEdit.setText(image_directory)
        slice_height = self.sliceHeightSpinBox.value()
        slice_width = self.sliceWidthSpinBox.value()
        overlap_height_ratio = self.overlapHeightRatioSpinBox.value()
        overlap_width_ratio = self.overlapWidthRatioSpinBox.value()
        confidence_threshold = self.confidenceThresholdSpinBox.value()
        selected_classes = self.selectedClassNames()
        all_class_names = self.allClassNames()

        if not model_weights:
            QMessageBox.warning(self, "Error", "Please select model weights.")
            return

        if not os.path.exists(model_weights):
            QMessageBox.warning(self, "Error", f"Model file not found: {model_weights}")
            return

        if not image_directory or not os.path.isdir(image_directory):
            QMessageBox.warning(self, "Error", "Please select an image directory.")
            return

        classes_txt_path = resolve_dataset_classes_path(image_directory, migrate=True)
        if not os.path.exists(classes_txt_path) and not all_class_names:
            QMessageBox.warning(self, "Error", f"No DarkFusion class list was found for: {image_directory}")
            return

        if not selected_classes:
            QMessageBox.warning(self, "Error", "Turn on at least one class in the main DarkFusion class list.")
            return

        self._save_settings()
        min_size_px, max_percent = (0.0, 1.0)
        if self.main_window is not None and hasattr(self.main_window, "current_prediction_size_filter_values"):
            min_size_px, max_percent = self.main_window.current_prediction_size_filter_values()
        ignore_teammates = bool(shared.get("ignore_teammates", False))
        teammate_threshold = float(shared.get("teammate_threshold", 0.90))
        if not shared and self.main_window is not None and hasattr(self.main_window, "auto_label_teammate_filter_config"):
            ignore_teammates, teammate_threshold = self.main_window.auto_label_teammate_filter_config()
        roi_inference = (
            self.main_window.roi_inference_snapshot()
            if self.main_window is not None and hasattr(self.main_window, "roi_inference_snapshot")
            else {"enabled": False}
        )

        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.statusLabel.setText("Loading model and starting SAHI...")
        self._set_running(True)

        config = {
            "model_type": model_type,
            "model_weights": model_weights,
            "image_directory": image_directory,
            "classes_txt_path": classes_txt_path if os.path.exists(classes_txt_path) else None,
            "class_names": all_class_names,
            **self.slicingOptions(),
            "confidence_threshold": confidence_threshold,
            "device": self._resolved_device(),
            "selected_classes": selected_classes,
            "overwrite": bool(shared.get("overwrite", self.overwriteCheckBox.isChecked())),
            "fp16": bool(shared.get("fp16", False)),
            "inference_backend": str(shared.get("inference_backend", "auto") or "auto"),
            "onnx_provider": str(shared.get("onnx_provider", "auto") or "auto"),
            "onnx_runtime_options": dict(shared.get("onnx_runtime_options", {}) or {}),
            "show_preview": self.showPreviewCheckBox.isChecked(),
            "min_size_px": min_size_px,
            "max_percent": max_percent,
            "ignore_teammates": ignore_teammates,
            "teammate_threshold": teammate_threshold,
            "roi_inference": roi_inference,
        }

        self.worker = SahiWorker(config, self)
        self.worker.annotation_task.connect(self.onSahiAnnotationTask)
        self.worker.progress.connect(self.onSahiProgress)
        self.worker.completed.connect(self.onSahiCompleted)
        self.worker.failed.connect(self.onSahiError)
        self.worker.start()

    def onSahiAnnotationTask(self, task, model_info=None):
        # Runs on the GUI thread once, before progress previews. A four-point
        # segmentation must not be reloaded as OBB merely because a previous
        # workflow left the editor in a different mode. Project classes/colors
        # and compatible custom pose definitions are not replaced here.
        mode = {"detect": "Boxes", "segment": "Segmentation", "pose": "Keypoints", "obb": "OBB"}.get(task)
        if mode and self.main_window is not None:
            parent = self.main_window
            if task == "pose" and model_info and hasattr(parent, "extract_pose_schema_from_current_model"):
                schema = parent.extract_pose_schema_from_current_model(**model_info)
                table = getattr(parent, "keypoint_list", None)
                if schema and (table is None or table.rowCount() != int(schema["kpt_shape"][0])):
                    parent.sync_pose_list_from_current_model(**model_info, write_points_file=True, force=True)
            self.main_window.set_annotation_mode(mode)

    def _show_sahi_result_in_graphics_view(self, image_path):
        parent = self.main_window
        if parent is None or not image_path or not os.path.exists(image_path):
            return

        image_path = parent.normalize_path(image_path) if hasattr(parent, "normalize_path") else os.path.abspath(image_path).replace("\\", "/")
        image_dir = parent.normalize_path(os.path.dirname(image_path)) if hasattr(parent, "normalize_path") else os.path.dirname(image_path)

        if hasattr(parent, "image_directory"):
            parent.image_directory = image_dir

        image_files = getattr(parent, "image_files", [])
        if not isinstance(image_files, list):
            image_files = []
        if image_path not in image_files:
            image_files.append(image_path)
            parent.image_files = image_files

        filtered_image_files = getattr(parent, "filtered_image_files", [])
        if not isinstance(filtered_image_files, list):
            filtered_image_files = []
        if image_path not in filtered_image_files:
            filtered_image_files = list(image_files)
            parent.filtered_image_files = filtered_image_files
            if hasattr(parent, "update_list_view"):
                parent.update_list_view(filtered_image_files)

        search_list = getattr(parent, "filtered_image_files", []) or [image_path]
        current_index = search_list.index(image_path) if image_path in search_list else 0
        parent.current_file = image_path
        parent.current_img_index = current_index
        parent.current_image_index = current_index

        if hasattr(parent, "display_image"):
            parent.display_image(image_path, rebuild_preview=True)

        if hasattr(parent, "img_index_number"):
            parent.img_index_number.blockSignals(True)
            parent.img_index_number.setMaximum(max(0, len(search_list) - 1))
            parent.img_index_number.setValue(current_index)
            parent.img_index_number.blockSignals(False)

        if hasattr(parent, "sync_list_view_selection"):
            parent.sync_list_view_selection(image_path)

        if hasattr(parent, "save_current_dataset_position"):
            parent.save_current_dataset_position(image_path)

    def onSahiProgress(self, current, total, image_path, labels_written):
        if total > 0:
            self.progressBar.setRange(0, total)
            self.progressBar.setValue(current)

        self._last_sahi_image_path = image_path
        image_name = os.path.basename(image_path)
        self.statusLabel.setText(f"{current}/{total}: {image_name}  |  labels: {labels_written}")

        if self.showPreviewCheckBox.isChecked():
            self._show_sahi_result_in_graphics_view(image_path)

    def onSahiCompleted(self, summary):
        self._set_running(False)
        self.worker = None
        images = int(summary.get("images", 0))
        images_with_detections = int(summary.get("images_with_detections", 0))
        labels = int(summary.get("labels", 0))
        skipped_size = int(summary.get("skipped_size", 0) or 0)
        skipped_teammates = int(summary.get("skipped_teammates", 0) or 0)
        self.statusLabel.setText(
            f"Done. Processed {images} images, wrote {labels} labels on {images_with_detections} images."
        )
        skipped_text = f"\nSkipped {skipped_size} predictions outside annotation size limits." if skipped_size else ""
        teammate_text = f"\nIgnored {skipped_teammates} likely teammate predictions." if skipped_teammates else ""
        QMessageBox.information(
            self,
            "SAHI Complete",
            f"Processed {images} images.\nWrote {labels} labels on {images_with_detections} images."
            f"{skipped_text}{teammate_text}",
        )

        if not self.showPreviewCheckBox.isChecked():
            image_path = (
                summary.get("last_detected_image_path")
                or summary.get("last_image_path")
                or getattr(self, "_last_sahi_image_path", "")
            )
            self._show_sahi_result_in_graphics_view(image_path)

    def onSahiError(self, error_message):
        self._set_running(False)
        self.worker = None
        self.statusLabel.setText("SAHI failed. See error details.")
        QMessageBox.critical(self, "SAHI Error", error_message)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "SAHI Running", "SAHI is still running. Wait for it to finish first.")
            event.ignore()
            return
        super().closeEvent(event)

    def showSahiSettings(self):
        self.sahiSettingsDialog = SahiSettingsDialog(self)
        self.sahiSettingsDialog.exec_()


class AutoLabelDialog(QDialog):
    """One Auto Label entry point with separate workflow tabs."""

    TAB_KEYS = {
        "weights": 0,
        "dino": 1,
        "sahi": 2,
    }

    def __init__(self, main_window, initial_tab="weights"):
        super().__init__(main_window)
        self.main_window = main_window
        self.auto_settings = QSettings("UltraDarkFusion", "AutoLabel")
        self._auto_label_loading_settings = False

        self.setWindowTitle("Auto Label")
        self.setMinimumSize(760, 600)
        self.resize(1040, 820)
        parent_style = main_window.styleSheet() if main_window is not None else ""
        if parent_style:
            self.setStyleSheet(parent_style)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        title = QLabel("Auto Label")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        header.addWidget(title)
        header.addStretch()
        self.dataset_label = QLabel("")
        self.dataset_label.setStyleSheet("color: #9aa4af;")
        self.dataset_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.addWidget(self.dataset_label)
        root.addLayout(header)

        self._build_shared_settings_group(root)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self._build_weights_tab()
        self._build_dino_tab()
        self._build_sahi_tab()

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #9aa4af;")
        root.addWidget(self.status_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setObjectName("auto_label_progress")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setToolTip("Progress for the active Auto Label workflow.")
        root.addWidget(self.progress_bar)

        main_progress = getattr(self.main_window, "label_progress", None)
        if isinstance(main_progress, QProgressBar):
            def mirror_main_progress(*_args):
                try:
                    self.progress_bar.setRange(main_progress.minimum(), main_progress.maximum())
                    self.progress_bar.setFormat(main_progress.format())
                    self.progress_bar.setValue(main_progress.value())
                except RuntimeError:
                    pass

            main_progress.valueChanged.connect(mirror_main_progress)
            mirror_main_progress()
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_from_parent)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_row.addWidget(refresh_btn)
        button_row.addWidget(close_btn)
        root.addLayout(button_row)

        self.refresh_from_parent()
        self.load_saved_settings()
        self.setup_auto_label_settings_bindings()
        self.tabs.setCurrentIndex(self.TAB_KEYS.get(initial_tab, 0))

    def _build_shared_settings_group(self, parent_layout):
        group = QtWidgets.QGroupBox("Shared Auto Label Settings")
        self.shared_settings_group = group
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        note = QLabel(
            "These settings apply to Weights, DINO/YOLOE, and SAHI whenever the selected "
            "model supports them."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9aa4af;")
        layout.addWidget(note)

        self.shared_overwrite_check = QCheckBox("Overwrite existing labels")
        self.shared_overwrite_check.setToolTip(
            "Replace existing label files instead of merging new detections with saved labels."
        )

        self.shared_fp16_check = QCheckBox("Use FP16 when supported")
        self.shared_fp16_check.setToolTip(
            "Use FP16 for compatible CUDA .pt models in every auto-label workflow. "
            "ONNX and TensorRT precision is determined by the exported model and provider."
        )

        self.inference_backend_combo = QComboBox()
        self.inference_backend_combo.addItem("Automatic (Windows ML / ORT for .onnx)", "auto")
        self.inference_backend_combo.addItem("Windows ML / ONNX Runtime", "onnxruntime")
        self.inference_backend_combo.addItem("Ultralytics", "ultralytics")
        self.inference_backend_combo.setToolTip(
            "Shared by Weights, compiled YOLOE, and SAHI. Automatic uses DarkFusion's "
            "ONNX Runtime backend for .onnx files and Ultralytics for .pt/.engine files."
        )

        self.onnx_provider_combo = QComboBox()
        
        # Get available providers dynamically
        provider_choices = [("Automatic (recommended)", "auto")]
        try:
            from darkfusion_onnx_runtime import available_execution_providers
            from darkfusion_windows_ml import is_experimental
            
            available = available_execution_providers()
            provider_map = {
                "TensorrtExecutionProvider": ("TensorRT", "tensorrt"),
                "TensorRTPluginExecutionProvider": ("TensorRT (classic)", "tensorrt"),
                "NvTensorRTRTXExecutionProvider": ("TensorRT RTX", "tensorrt_rtx"),
                "CUDAExecutionProvider": ("CUDA", "cuda"),
                "DmlExecutionProvider": ("DirectML", "directml"),
                "ROCMExecutionProvider": ("ROCm", "rocm"),
                "MIGraphXExecutionProvider": ("MIGraphX", "migraphx"),
                "OpenVINOExecutionProvider": ("OpenVINO", "openvino"),
                "CoreMLExecutionProvider": ("CoreML", "coreml"),
                "QNNExecutionProvider": ("QNN", "qnn"),
                "WebGpuExecutionProvider": ("WebGPU", "webgpu"),
                "XnnpackExecutionProvider": ("XNNPACK", "xnnpack"),
                "CPUExecutionProvider": ("CPU", "cpu"),
            }
            
            # Add only available providers
            for provider in available:
                if not is_experimental(provider):
                    label, alias = provider_map.get(provider, (provider, provider))
                    provider_choices.append((label, alias))
        except Exception as e:
            # Do not advertise GPU providers we could not discover. The setup
            # panel reports the failure and can refresh the list in the background.
            logger.warning("Could not discover ONNX providers: %s", e)
        
        # Add all provider options to combo
        for label, provider in provider_choices:
            self.onnx_provider_combo.addItem(label, provider)
        
        self.onnx_provider_combo.setToolTip(
            "Execution provider used by DarkFusion's ONNX backend for Weights, YOLOE, and SAHI. "
            "Automatic compares compatible providers once per model/input shape and caches the choice. "
            "Only installed or compatible downloadable providers are shown. Experimental providers require opt-in under Advanced / troubleshooting."
        )

        self.onnx_provider_status = QLabel("")
        self.onnx_provider_status.setWordWrap(True)
        self.onnx_provider_status.setStyleSheet("color: #9aa4af;")

        self.ignore_teammates_check = QCheckBox(
            "Ignore friendly players with an overhead nameplate / colored marker"
        )
        self.ignore_teammates_check.setToolTip(
            "Shared by every auto-label workflow. Combines colored-marker/nameplate layout, "
            "visual semantics, and gamer-tag OCR near the predicted player's head. OCR also "
            "rejects common HUD words so unrelated interface text is kept."
        )

        self.teammate_threshold_spin = PercentSliderControl(0.50, 0.999, 0.90)
        self.teammate_threshold_spin.setRange(0.50, 0.999)
        self.teammate_threshold_spin.setSingleStep(0.025)
        self.teammate_threshold_spin.setDecimals(3)
        self.teammate_threshold_spin.setToolTip(
            "Visual-marker certainty. Higher values reject fewer predictions. OCR still requires "
            "independent nameplate and colored-symbol evidence near the player."
        )

        option_row = QtWidgets.QHBoxLayout()
        option_row.setSpacing(18)
        option_row.addWidget(self.shared_overwrite_check)
        option_row.addWidget(self.shared_fp16_check)
        option_row.addWidget(self.ignore_teammates_check)
        option_row.addStretch(1)
        layout.addLayout(option_row)

        teammate_row = QtWidgets.QHBoxLayout()
        teammate_row.setSpacing(8)
        teammate_label = QLabel("Teammate certainty:")
        teammate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        teammate_row.addWidget(teammate_label)
        teammate_row.addWidget(self.teammate_threshold_spin, 1)
        layout.addLayout(teammate_row)
        parent_layout.addWidget(group)

        # Provider controls are a sibling category, not nested inside Shared.
        # Collapsing Shared must never hide provider selection/setup.
        self.onnx_runtime_section = CollapsibleSettingsGroup("ONNX / Windows ML", expanded=False)
        self.onnx_runtime_section.setObjectName("onnx_provider_setup")
        self.onnx_runtime_section.setProperty("persistCollapseState", True)
        self.onnx_runtime_toggle = self.onnx_runtime_section.header
        self.onnx_runtime_toggle.setToolTip("Expand ONNX provider selection, setup, and troubleshooting.")
        provider_content = QWidget()
        provider_content.setObjectName("auto_label_onnx_settings_body")
        provider_content.setProperty("settingsSectionContent", True)
        provider_layout = QVBoxLayout(provider_content)
        provider_layout.setContentsMargins(6, 6, 6, 6)
        provider_layout.setSpacing(6)
        provider_note = QLabel("These settings affect ONNX files. .pt files use Ultralytics; .engine files use TensorRT.")
        provider_note.setWordWrap(True)
        provider_note.setStyleSheet("color: #9aa4af;")
        provider_layout.addWidget(provider_note)

        backend_row = QtWidgets.QHBoxLayout()
        backend_row.setSpacing(8)
        provider_label = QLabel("ONNX provider:")
        provider_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        backend_row.addWidget(provider_label)
        backend_row.addWidget(self.onnx_provider_combo, 1)
        provider_layout.addLayout(backend_row)

        detail_row = QtWidgets.QHBoxLayout()
        detail_row.setSpacing(8)
        available_label = QLabel("Available now:")
        available_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        detail_row.addWidget(available_label)
        detail_row.addWidget(self.onnx_provider_status, 1)
        provider_layout.addLayout(detail_row)

        from darkfusion_onnx_settings import OnnxRuntimeSettings
        self.onnx_runtime_widget = OnnxRuntimeSettings(
            self.onnx_provider_combo, self.onnx_provider_status,
            self.main_window.settings.get("onnxRuntimeOptions", {}), self,
            backend_combo=self.inference_backend_combo,
        )
        self.onnx_runtime_widget.setProperty("collapsibleSectionSkip", True)
        self.onnx_runtime_widget.setProperty("collapsibleSectionBody", True)
        self.onnx_runtime_widget.setStyleSheet(CollapsibleSectionsController.BODY_STYLE)
        provider_layout.addWidget(self.onnx_runtime_widget)
        self.onnx_runtime_scroll = QtWidgets.QScrollArea()
        self.onnx_runtime_scroll.setObjectName("auto_label_onnx_settings_scroll")
        self.onnx_runtime_scroll.setWidgetResizable(True)
        self.onnx_runtime_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.onnx_runtime_scroll.setMinimumHeight(200)
        self.onnx_runtime_scroll.setMaximumHeight(300)
        self.onnx_runtime_scroll.setWidget(provider_content)
        section_layout = QVBoxLayout(self.onnx_runtime_section.content)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self.onnx_runtime_scroll)
        parent_layout.addWidget(self.onnx_runtime_section)

    def _path_row(self, line_edit, browse_callback, placeholder=""):
        row = QtWidgets.QHBoxLayout()
        line_edit.setPlaceholderText(placeholder)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(browse_callback)
        row.addWidget(line_edit, 1)
        row.addWidget(browse_btn)
        return row

    def _build_weights_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        content = QWidget(tab)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        scroll = QtWidgets.QScrollArea(tab)
        scroll.setObjectName("weights_auto_label_scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll, 1)

        model_group = QtWidgets.QGroupBox("Model / Weights")
        model_form = QtWidgets.QFormLayout(model_group)
        model_form.setLabelAlignment(Qt.AlignRight)

        self.weights_path_edit = QLineEdit()
        self.weights_path_edit.setReadOnly(True)
        load_btn = QPushButton("Load Model / Weights")
        load_btn.clicked.connect(self.load_weights_from_dialog)
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Unload the current weights/model used by auto-labeling and live video detections.")
        clear_btn.clicked.connect(self.clear_weights_from_dialog)
        weights_row = QtWidgets.QHBoxLayout()
        weights_row.addWidget(self.weights_path_edit, 1)
        weights_row.addWidget(load_btn)
        weights_row.addWidget(clear_btn)
        model_form.addRow("Weights:", weights_row)

        self.cfg_path_edit = QLineEdit()
        self.cfg_path_edit.setReadOnly(True)
        self.cfg_btn = QPushButton("Load CFG")
        self.cfg_btn.clicked.connect(self.load_cfg_from_dialog)
        cfg_row = QtWidgets.QHBoxLayout()
        cfg_row.addWidget(self.cfg_path_edit, 1)
        cfg_row.addWidget(self.cfg_btn)
        self.cfg_label = QLabel("Darknet CFG:")
        model_form.addRow(self.cfg_label, cfg_row)

        layout.addWidget(model_group)

        detect_group = QtWidgets.QGroupBox("Detection")
        detect_form = QtWidgets.QFormLayout(detect_group)
        detect_form.setLabelAlignment(Qt.AlignRight)

        self.weight_conf_spin = PercentSliderControl(0.0, 1.0, 0.30)
        self.weight_conf_spin.setRange(0.0, 1.0)
        self.weight_conf_spin.setSingleStep(0.01)
        self.weight_conf_spin.setDecimals(2)
        detect_form.addRow("Confidence:", self.weight_conf_spin)

        self.weight_iou_spin = PercentSliderControl(0.0, 1.0, 0.45)
        self.weight_iou_spin.setRange(0.0, 1.0)
        self.weight_iou_spin.setSingleStep(0.01)
        self.weight_iou_spin.setDecimals(2)
        detect_form.addRow("NMS / IOU:", self.weight_iou_spin)

        size_row = QtWidgets.QHBoxLayout()
        self.weight_height_spin = MultipleOfSpinBox(32)
        self.weight_height_spin.setRange(32, 5000)
        self.weight_height_spin.setSingleStep(32)
        self.weight_height_spin.setMaximumWidth(120)
        self.weight_width_spin = MultipleOfSpinBox(32)
        self.weight_width_spin.setRange(32, 5000)
        self.weight_width_spin.setSingleStep(32)
        self.weight_width_spin.setMaximumWidth(120)
        size_row.addWidget(self.weight_height_spin)
        size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.weight_width_spin)
        detect_form.addRow("Input H x W:", size_row)

        self.weight_batch_spin = QSpinBox()
        self.weight_batch_spin.setRange(1, 128)
        self.weight_batch_spin.setMaximumWidth(120)
        detect_form.addRow("Batch size:", self.weight_batch_spin)

        self.weights_sam3_assist_check = QCheckBox("SAM3 assist: refine boxes and fill short gaps")
        self.weights_sam3_assist_check.setToolTip(
            "Refine new YOLO boxes when a SAM3 mask agrees; uncertain adjustments "
            "keep the YOLO box. For numbered sequential images, also fill gaps of "
            "up to 3 images when YOLO detects the object on both sides and "
            "forward/backward propagation agrees. Existing saved labels are kept "
            "unless overwrite is enabled. Teammate-ignore, ROI and class filters "
            "still apply. Detection boxes only; adds processing time."
        )
        detect_form.addRow(self.weights_sam3_assist_check)

        layout.addWidget(detect_group)

        action_row = QtWidgets.QHBoxLayout()
        action_row.addStretch()
        apply_btn = QPushButton("Apply Settings")
        apply_btn.clicked.connect(self.apply_weight_settings)
        run_btn = QPushButton("Run Weights Auto Label")
        run_btn.clicked.connect(self.run_weights_auto_label)
        action_row.addWidget(apply_btn)
        action_row.addWidget(run_btn)
        tab_layout.addLayout(action_row)
        layout.addStretch()

        self.tabs.addTab(tab, "Weights")

    def _build_dino_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        content = QWidget(tab)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        scroll = QtWidgets.QScrollArea(tab)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll, 1)

        model_group = QtWidgets.QGroupBox("DINO / YOLOE Model")
        model_form = QtWidgets.QFormLayout(model_group)
        model_form.setLabelAlignment(Qt.AlignRight)

        self.yoloe_model_edit = QLineEdit()
        model_form.addRow(
            "YOLOE:",
            self._path_row(
                self.yoloe_model_edit,
                self.browse_yoloe_model,
                "yoloe-26s-seg.pt, local .pt, .onnx, or a prebuilt .engine",
            ),
        )
        compiled_note = QLabel(
            "YOLOE .pt accepts the selected classes as live text prompts. A compiled .onnx or "
            ".engine must be exported with the same classes already embedded. ONNX uses the "
            "shared backend and provider above."
        )
        compiled_note.setWordWrap(True)
        compiled_note.setMinimumHeight(34)
        compiled_note.setStyleSheet("color: #9aa4af;")
        model_form.addRow("", compiled_note)

        self.dino_weights_edit = QLineEdit()
        model_form.addRow(
            "DINO weights:",
            self._path_row(
                self.dino_weights_edit,
                self.browse_dino_weights,
                "Sam/groundingdino_swint_ogc.pth",
            ),
        )

        self.dino_config_edit = QLineEdit()
        model_form.addRow(
            "DINO config:",
            self._path_row(
                self.dino_config_edit,
                self.browse_dino_config,
                "GroundingDINO_SwinT_OGC.py",
            ),
        )

        layout.addWidget(model_group)

        detect_group = QtWidgets.QGroupBox("Detection / Fusion")
        detect_form = QtWidgets.QFormLayout(detect_group)
        detect_form.setLabelAlignment(Qt.AlignRight)

        self.yoloe_conf_spin = PercentSliderControl(0.01, 1.0, 0.15)
        self.yoloe_conf_spin.setRange(0.01, 1.0)
        self.yoloe_conf_spin.setSingleStep(0.01)
        self.yoloe_conf_spin.setDecimals(2)
        detect_form.addRow("YOLOE confidence:", self.yoloe_conf_spin)

        self.yoloe_iou_spin = PercentSliderControl(0.01, 1.0, 0.45)
        self.yoloe_iou_spin.setRange(0.01, 1.0)
        self.yoloe_iou_spin.setSingleStep(0.01)
        self.yoloe_iou_spin.setDecimals(2)
        detect_form.addRow("YOLOE IoU:", self.yoloe_iou_spin)

        self.dino_text_spin = PercentSliderControl(0.01, 1.0, 0.35)
        self.dino_text_spin.setRange(0.01, 1.0)
        self.dino_text_spin.setSingleStep(0.01)
        self.dino_text_spin.setDecimals(2)
        detect_form.addRow("DINO text threshold:", self.dino_text_spin)

        self.dino_box_spin = PercentSliderControl(0.01, 1.0, 0.18)
        self.dino_box_spin.setRange(0.01, 1.0)
        self.dino_box_spin.setSingleStep(0.01)
        self.dino_box_spin.setDecimals(2)
        detect_form.addRow("DINO box threshold:", self.dino_box_spin)

        self.dino_batch_spin = QSpinBox()
        self.dino_batch_spin.setRange(1, 32)
        self.dino_batch_spin.setMaximumWidth(120)
        detect_form.addRow("Batch size:", self.dino_batch_spin)

        self.dino_predict_size_spin = MultipleOfSpinBox(32)
        self.dino_predict_size_spin.setRange(320, 4096)
        self.dino_predict_size_spin.setSingleStep(32)
        self.dino_predict_size_spin.setMaximumWidth(120)
        detect_form.addRow("Predict image size:", self.dino_predict_size_spin)

        self.dino_fp16_fallback_check = QCheckBox("Retry DINO in FP32 if shared FP16 fails")
        detect_form.addRow("", self.dino_fp16_fallback_check)

        layout.addWidget(detect_group)

        action_row = QtWidgets.QHBoxLayout()
        action_row.addStretch()
        run_btn = QPushButton("Run DINO / YOLOE")
        run_btn.clicked.connect(self.run_dino_yoloe_auto_label)
        action_row.addWidget(run_btn)
        layout.addLayout(action_row)
        layout.addStretch()

        self.tabs.addTab(tab, "DINO / YOLOE")

    def _build_sahi_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.sahi_panel = SahiSettingsDialog(
            tab,
            main_window=self.main_window,
            shared_settings_provider=self.shared_auto_label_config,
        )
        self.sahi_panel.setWindowFlags(Qt.Widget)
        self.sahi_panel.setMinimumSize(640, 560)
        self.sahi_panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        if hasattr(self.sahi_panel, "closeBtn"):
            self.sahi_panel.closeBtn.hide()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self.sahi_panel)
        layout.addWidget(scroll, 1)

        self.tabs.addTab(tab, "SAHI")

    def _settings_bool(self, key, default):
        value = self.auto_settings.value(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def refresh_from_parent(self):
        dataset = getattr(self.main_window, "image_directory", "") or "No dataset loaded"
        self.dataset_label.setText(str(dataset))

        self.weights_path_edit.setText(str(getattr(self.main_window, "weights_file_path", "") or ""))
        self.cfg_path_edit.setText(str(getattr(self.main_window, "cfg_file_path", "") or ""))
        self.update_cfg_row_visibility()

        backend = str(self.main_window.settings.get("inferenceBackend", "auto") or "auto")
        backend_index = self.inference_backend_combo.findData(backend)
        self.inference_backend_combo.setCurrentIndex(backend_index if backend_index >= 0 else 0)
        provider = str(self.main_window.settings.get("onnxExecutionProvider", "auto") or "auto")
        provider_index = self.onnx_provider_combo.findData(provider)
        self.onnx_provider_combo.setCurrentIndex(provider_index if provider_index >= 0 else 0)
        self.refresh_onnx_provider_status()
        self.update_cfg_row_visibility()

        self.weight_conf_spin.setValue(float(self.main_window.confidence_threshold_spinbox.value()))
        self.weight_iou_spin.setValue(float(self.main_window.nms_threshold_spinbox.value()))
        self.weight_height_spin.setValue(snap_int_to_multiple(int(self.main_window.network_height.value()), 32, 32, 5000))
        self.weight_width_spin.setValue(snap_int_to_multiple(int(self.main_window.network_width.value()), 32, 32, 5000))
        self.weight_batch_spin.setValue(int(self.main_window.batch_inference.value()))

        fp_widget = getattr(self.main_window, "fp_select_combobox", None)
        if isinstance(fp_widget, QComboBox):
            self.shared_fp16_check.setChecked(fp_widget.currentIndex() == 1)
        else:
            self.shared_fp16_check.setChecked(bool(getattr(self.main_window, "fp_mode", 0) == 1))

        if hasattr(self, "sahi_panel"):
            self.sahi_panel.imageDirLineEdit.setText(
                str(getattr(self.main_window, "image_directory", "") or "")
            )
            self.sahi_panel.refreshClasses()

    def update_cfg_row_visibility(self):
        weights_path = getattr(self.main_window, "weights_file_path", None) or self.weights_path_edit.text()
        is_darknet = str(weights_path or "").lower().endswith(".weights")
        self.cfg_path_edit.setVisible(is_darknet)
        self.cfg_btn.setVisible(is_darknet)
        self.cfg_label.setVisible(is_darknet)

        backend = str(self.inference_backend_combo.currentData() or "auto")
        self.onnx_provider_combo.setEnabled(backend != "ultralytics")
        if hasattr(self, "onnx_runtime_toggle"):
            # Keep the troubleshooting loader reachable even for an existing
            # explicit Ultralytics override; its own controls handle ONNX gating.
            self.onnx_runtime_toggle.setEnabled(True)
            self.onnx_runtime_widget.setEnabled(True)
            self.onnx_runtime_widget._update_enabled()

    def refresh_onnx_provider_status(self):
        if hasattr(self, "onnx_runtime_widget"):
            self.onnx_runtime_widget.refresh_providers()
            return
        try:
            from darkfusion_onnx_runtime import available_execution_providers

            available = available_execution_providers()
            friendly = {
                "TensorrtExecutionProvider": "TensorRT",
                "CUDAExecutionProvider": "CUDA",
                "DmlExecutionProvider": "DirectML",
                "ROCMExecutionProvider": "ROCm",
                "MIGraphXExecutionProvider": "MIGraphX",
                "OpenVINOExecutionProvider": "OpenVINO",
                "CoreMLExecutionProvider": "CoreML",
                "QNNExecutionProvider": "QNN",
                "WebGpuExecutionProvider": "WebGPU",
                "XnnpackExecutionProvider": "XNNPACK",
                "CPUExecutionProvider": "CPU",
            }
            labels = [friendly.get(provider, provider) for provider in available]
            status_text = ", ".join(labels) if labels else "No providers detected"
            self.onnx_provider_status.setText(status_text)
            
            # Disable combo items that aren't available
            for i in range(self.onnx_provider_combo.count()):
                label = self.onnx_provider_combo.itemText(i)
                if "(not installed)" in label:
                    self.onnx_provider_combo.model().item(i).setEnabled(False)
        except Exception as e:
            self.onnx_provider_status.setText(f"ONNX Runtime unavailable: {e}")

    def load_saved_settings(self):
        self._auto_label_loading_settings = True
        try:
            # Keep the user's detector thresholds when upgrading, but always use
            # a compatible YOLOE weight in the new model field.
            legacy_yoloe_settings = {
                "world_conf": "yoloe_conf",
                "world_iou": "yoloe_iou",
                "world_fp16": "yoloe_fp16",
            }
            for old_key, new_key in legacy_yoloe_settings.items():
                if not self.auto_settings.contains(new_key) and self.auto_settings.contains(old_key):
                    self.auto_settings.setValue(new_key, self.auto_settings.value(old_key))
            for old_key in (*legacy_yoloe_settings, "world_model_name"):
                self.auto_settings.remove(old_key)

            if self.auto_settings.contains("weights_conf"):
                self.weight_conf_spin.setValue(float(self.auto_settings.value("weights_conf", self.weight_conf_spin.value())))
            if self.auto_settings.contains("weights_iou"):
                self.weight_iou_spin.setValue(float(self.auto_settings.value("weights_iou", self.weight_iou_spin.value())))
            if self.auto_settings.contains("weights_height"):
                self.weight_height_spin.setValue(
                    snap_int_to_multiple(
                        int(self.auto_settings.value("weights_height", self.weight_height_spin.value())),
                        32,
                        32,
                        5000,
                    )
                )
            if self.auto_settings.contains("weights_width"):
                self.weight_width_spin.setValue(
                    snap_int_to_multiple(
                        int(self.auto_settings.value("weights_width", self.weight_width_spin.value())),
                        32,
                        32,
                        5000,
                    )
                )
            if self.auto_settings.contains("weights_batch_size"):
                self.weight_batch_spin.setValue(int(self.auto_settings.value("weights_batch_size", self.weight_batch_spin.value())))
            self.weights_sam3_assist_check.setChecked(
                self._settings_bool("weights_sam3_assist", False)
            )
            self.main_window.auto_label_sam3_assist = self.weights_sam3_assist_check.isChecked()
            if self.auto_settings.contains("shared_fp16"):
                shared_fp16 = self._settings_bool("shared_fp16", self.shared_fp16_check.isChecked())
            elif self.auto_settings.contains("weights_fp16"):
                shared_fp16 = self._settings_bool("weights_fp16", self.shared_fp16_check.isChecked())
            elif self.auto_settings.contains("yoloe_fp16"):
                shared_fp16 = self._settings_bool("yoloe_fp16", self.shared_fp16_check.isChecked())
            else:
                shared_fp16 = self._settings_bool("dino_fp16", self.shared_fp16_check.isChecked())
            self.shared_fp16_check.setChecked(shared_fp16)
            # Project/application settings own the runtime. A stale global
            # auto-label dialog cache must not overwrite an active project.
            backend = str(self.main_window.settings.get("inferenceBackend", "auto") or "auto")
            provider = str(self.main_window.settings.get("onnxExecutionProvider", "auto") or "auto")
            index = self.inference_backend_combo.findData(backend)
            self.inference_backend_combo.setCurrentIndex(index if index >= 0 else 0)
            index = self.onnx_provider_combo.findData(provider)
            self.onnx_provider_combo.setCurrentIndex(index if index >= 0 else 0)
            self.onnx_runtime_widget.set_options(self.main_window.settings.get("onnxRuntimeOptions", {}))
            options = configure_onnx_runtime(self.onnx_runtime_widget.options())
            self.main_window.settings["onnxRuntimeOptions"] = options

            if self.auto_settings.contains("shared_overwrite"):
                shared_overwrite = self._settings_bool("shared_overwrite", False)
            elif self.auto_settings.contains("weights_overwrite"):
                shared_overwrite = self._settings_bool("weights_overwrite", False)
            else:
                shared_overwrite = self._settings_bool("dino_overwrite", False)
            self.shared_overwrite_check.setChecked(shared_overwrite)
            self.ignore_teammates_check.setChecked(self._settings_bool("ignore_teammates", False))
            self.teammate_threshold_spin.setValue(
                float(self.auto_settings.value("teammate_threshold", 0.90))
            )

            self.yoloe_model_edit.setText(str(self.auto_settings.value("yoloe_model_name", "yoloe-26s-seg.pt")))
            from darkfusion_environment import model_directory
            self.dino_weights_edit.setText(str(self.auto_settings.value("dino_weights_path", str(Path(model_directory("Sam")) / "groundingdino_swint_ogc.pth"))))
            self.dino_config_edit.setText(str(self.auto_settings.value("dino_config_path", "")))
            self.yoloe_conf_spin.setValue(float(self.auto_settings.value("yoloe_conf", 0.15)))
            self.yoloe_iou_spin.setValue(float(self.auto_settings.value("yoloe_iou", 0.45)))
            self.dino_text_spin.setValue(float(self.auto_settings.value("dino_text_threshold", 0.35)))
            self.dino_box_spin.setValue(float(self.auto_settings.value("dino_box_threshold", 0.18)))
            self.dino_batch_spin.setValue(int(self.auto_settings.value("dino_batch_size", 2)))
            self.dino_predict_size_spin.setValue(
                snap_int_to_multiple(
                    int(self.auto_settings.value("dino_predict_imgsz", 640)),
                    32,
                    320,
                    4096,
                )
            )
            self.dino_fp16_fallback_check.setChecked(self._settings_bool("dino_fp16_fallback", True))
            self.main_window.auto_label_ignore_teammates = self.ignore_teammates_check.isChecked()
            self.main_window.auto_label_teammate_threshold = self.teammate_threshold_spin.value()
            for legacy_key in (
                "weights_fp16",
                "yoloe_fp16",
                "dino_fp16",
                "weights_overwrite",
                "dino_overwrite",
            ):
                self.auto_settings.remove(legacy_key)
            self.update_cfg_row_visibility()
        finally:
            self._auto_label_loading_settings = False

    def setup_auto_label_settings_bindings(self):
        if getattr(self, "_auto_label_settings_bindings_ready", False):
            return

        def save_weights(*_args):
            self.save_weight_settings()

        def save_dino(*_args):
            self.save_dino_settings()

        def save_shared(*_args):
            self.save_shared_settings()

        for widget, signal_name in (
            (self.weight_conf_spin, "valueChanged"),
            (self.weight_iou_spin, "valueChanged"),
            (self.weight_height_spin, "valueChanged"),
            (self.weight_width_spin, "valueChanged"),
            (self.weight_batch_spin, "valueChanged"),
            (self.weights_sam3_assist_check, "stateChanged"),
        ):
            try:
                getattr(widget, signal_name).connect(save_weights)
            except Exception:
                pass

        for widget, signal_name in (
            (self.shared_overwrite_check, "stateChanged"),
            (self.shared_fp16_check, "stateChanged"),
            (self.inference_backend_combo, "currentIndexChanged"),
            (self.onnx_provider_combo, "currentIndexChanged"),
            (self.ignore_teammates_check, "stateChanged"),
            (self.teammate_threshold_spin, "valueChanged"),
        ):
            try:
                getattr(widget, signal_name).connect(save_shared)
            except Exception:
                pass

        for widget, signal_name in (
            (self.yoloe_model_edit, "textChanged"),
            (self.dino_weights_edit, "textChanged"),
            (self.dino_config_edit, "textChanged"),
            (self.yoloe_conf_spin, "valueChanged"),
            (self.yoloe_iou_spin, "valueChanged"),
            (self.dino_text_spin, "valueChanged"),
            (self.dino_box_spin, "valueChanged"),
            (self.dino_batch_spin, "valueChanged"),
            (self.dino_predict_size_spin, "valueChanged"),
            (self.dino_fp16_fallback_check, "stateChanged"),
        ):
            try:
                getattr(widget, signal_name).connect(save_dino)
            except Exception:
                pass

        self._auto_label_settings_bindings_ready = True
        self.onnx_runtime_widget.changed.connect(save_shared)
        self.inference_backend_combo.currentIndexChanged.connect(
            lambda _index: self.update_cfg_row_visibility()
        )

    def save_weight_settings(self):
        if getattr(self, "_auto_label_loading_settings", False):
            return

        self.auto_settings.setValue("weights_conf", self.weight_conf_spin.value())
        self.auto_settings.setValue("weights_iou", self.weight_iou_spin.value())
        self.auto_settings.setValue("weights_height", self.weight_height_spin.value())
        self.auto_settings.setValue("weights_width", self.weight_width_spin.value())
        self.auto_settings.setValue("weights_batch_size", self.weight_batch_spin.value())
        self.auto_settings.setValue("weights_sam3_assist", self.weights_sam3_assist_check.isChecked())
        self.main_window.auto_label_sam3_assist = self.weights_sam3_assist_check.isChecked()
        self.save_shared_settings()

    def save_shared_settings(self):
        if getattr(self, "_auto_label_loading_settings", False):
            return
        enabled = bool(self.ignore_teammates_check.isChecked())
        threshold = float(self.teammate_threshold_spin.value())
        backend = str(self.inference_backend_combo.currentData() or "auto")
        provider = str(self.onnx_provider_combo.currentData() or "auto")
        self.auto_settings.setValue("shared_overwrite", self.shared_overwrite_check.isChecked())
        self.auto_settings.setValue("shared_fp16", self.shared_fp16_check.isChecked())
        self.auto_settings.setValue("inference_backend", backend)
        self.auto_settings.setValue("onnx_provider", provider)
        options = configure_onnx_runtime(self.onnx_runtime_widget.options())
        self.auto_settings.setValue("onnx_runtime_options", json.dumps(options, sort_keys=True))
        self.main_window.settings["onnxRuntimeOptions"] = options
        self.auto_settings.setValue("ignore_teammates", enabled)
        self.auto_settings.setValue("teammate_threshold", threshold)
        self.main_window.auto_label_ignore_teammates = enabled
        self.main_window.auto_label_teammate_threshold = threshold
        self.main_window.settings["inferenceBackend"] = backend
        self.main_window.settings["onnxExecutionProvider"] = provider
        self.update_cfg_row_visibility()
        if hasattr(self.main_window, "queue_settings_save"):
            self.main_window.queue_settings_save()

    # Kept for callers from older builds; teammate filtering is now part of the
    # single shared auto-label configuration.
    def save_teammate_settings(self):
        self.save_shared_settings()

    def shared_auto_label_config(self):
        return {
            "image_directory": str(getattr(self.main_window, "image_directory", "") or ""),
            "overwrite": bool(self.shared_overwrite_check.isChecked()),
            "fp16": bool(self.shared_fp16_check.isChecked()),
            "inference_backend": str(self.inference_backend_combo.currentData() or "auto"),
            "onnx_provider": str(self.onnx_provider_combo.currentData() or "auto"),
            "onnx_runtime_options": dict(self.main_window.settings.get("onnxRuntimeOptions", {}) or {}),
            "ignore_teammates": bool(self.ignore_teammates_check.isChecked()),
            "teammate_threshold": float(self.teammate_threshold_spin.value()),
        }

    def save_dino_settings(self):
        if getattr(self, "_auto_label_loading_settings", False):
            return

        self.auto_settings.setValue("yoloe_model_name", self.yoloe_model_edit.text().strip())
        self.auto_settings.setValue("dino_weights_path", self.dino_weights_edit.text().strip())
        self.auto_settings.setValue("dino_config_path", self.dino_config_edit.text().strip())
        self.auto_settings.setValue("yoloe_conf", self.yoloe_conf_spin.value())
        self.auto_settings.setValue("yoloe_iou", self.yoloe_iou_spin.value())
        self.auto_settings.setValue("dino_text_threshold", self.dino_text_spin.value())
        self.auto_settings.setValue("dino_box_threshold", self.dino_box_spin.value())
        self.auto_settings.setValue("dino_batch_size", self.dino_batch_spin.value())
        self.auto_settings.setValue("dino_predict_imgsz", self.dino_predict_size_spin.value())
        self.auto_settings.setValue("dino_fp16_fallback", self.dino_fp16_fallback_check.isChecked())
        self.save_shared_settings()

    def browse_yoloe_model(self):
        start_path = self.main_window.dialog_start_directory(
            "model", self.yoloe_model_edit.text().strip()
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLOE Model",
            start_path,
            "Model Files (*.pt *.engine *.onnx);;All Files (*)",
        )
        if path:
            self.main_window.remember_dialog_selection("model", path)
            self.yoloe_model_edit.setText(path)

    def browse_dino_weights(self):
        start_path = self.main_window.dialog_start_directory(
            "model", self.dino_weights_edit.text().strip()
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DINO Weights",
            start_path,
            "DINO Weights (*.pth);;All Files (*)",
        )
        if path:
            self.main_window.remember_dialog_selection("model", path)
            self.dino_weights_edit.setText(path)

    def browse_dino_config(self):
        start_path = self.main_window.dialog_start_directory(
            "model", self.dino_config_edit.text().strip()
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DINO Config",
            start_path,
            "Python Config (*.py);;All Files (*)",
        )
        if path:
            self.main_window.remember_dialog_selection("model", path)
            self.dino_config_edit.setText(path)

    def apply_weight_settings(self):
        shared = self.shared_auto_label_config()
        previous_backend = str(self.main_window.settings.get("inferenceBackend", "auto") or "auto")
        previous_provider = str(self.main_window.settings.get("onnxExecutionProvider", "auto") or "auto")
        # Combo changes save pending preferences before Apply. Roll back to the
        # loaded session, not those already-edited pending preferences.
        previous_runtime = dict(getattr(getattr(self.main_window, "model", None), "_darkfusion_runtime_settings", {}) or {})
        if previous_runtime:
            previous_backend = str(previous_runtime.get("inferenceBackend", previous_backend))
            previous_provider = str(previous_runtime.get("onnxExecutionProvider", previous_provider))
        selected_backend = shared["inference_backend"]
        selected_provider = shared["onnx_provider"]
        self.main_window.settings["inferenceBackend"] = selected_backend
        self.main_window.settings["onnxExecutionProvider"] = selected_provider

        conf_blocked = self.main_window.confidence_threshold_spinbox.blockSignals(True)
        nms_blocked = self.main_window.nms_threshold_spinbox.blockSignals(True)
        try:
            self.main_window.confidence_threshold_spinbox.setValue(self.weight_conf_spin.value())
            self.main_window.nms_threshold_spinbox.setValue(self.weight_iou_spin.value())
        finally:
            self.main_window.confidence_threshold_spinbox.blockSignals(conf_blocked)
            self.main_window.nms_threshold_spinbox.blockSignals(nms_blocked)

        network_h = snap_int_to_multiple(self.weight_height_spin.value(), 32, 32, 5000)
        network_w = snap_int_to_multiple(self.weight_width_spin.value(), 32, 32, 5000)
        self.weight_height_spin.setValue(network_h)
        self.weight_width_spin.setValue(network_w)
        self.main_window.network_height.setValue(network_h)
        self.main_window.network_width.setValue(network_w)
        self.main_window.batch_inference.setValue(self.weight_batch_spin.value())

        fp_widget = getattr(self.main_window, "fp_select_combobox", None)
        fp_mode = 1 if shared["fp16"] else 0
        if isinstance(fp_widget, QComboBox):
            fp_widget.setCurrentIndex(fp_mode)
        if hasattr(self.main_window, "set_floating_point_mode"):
            self.main_window.set_floating_point_mode(fp_mode)

        if hasattr(self.main_window, "threshold_spinbox_changed"):
            self.main_window.confidence_threshold = self.weight_conf_spin.value()
            self.main_window.nms_threshold = self.weight_iou_spin.value()

        self.save_weight_settings()
        if hasattr(self.main_window, "queue_settings_save"):
            self.main_window.queue_settings_save()
        loaded_path = str(getattr(self.main_window, "weights_file_path", "") or "")
        desired_identity = self.main_window.inference_model_identity(loaded_path)
        loaded_identity = getattr(
            getattr(self.main_window, "model", None),
            "_darkfusion_load_identity",
            None,
        )
        backend_changed = selected_backend != previous_backend or selected_provider != previous_provider
        needs_reload = (
            loaded_path.lower().endswith(".onnx")
            and (backend_changed or loaded_identity != desired_identity)
        )
        if needs_reload:
            try:
                self.main_window.reload_selected_inference_model()
                actual = self.main_window.active_inference_backend_label()
                self.status_label.setText(f"Weights settings applied. Active: {actual}.")
            except Exception as e:
                self.main_window.settings["inferenceBackend"] = previous_backend
                self.main_window.settings["onnxExecutionProvider"] = previous_provider
                if previous_runtime:
                    self.main_window.settings.update(previous_runtime)
                    was_loading = self._auto_label_loading_settings
                    self._auto_label_loading_settings = True
                    try:
                        self.inference_backend_combo.setCurrentIndex(self.inference_backend_combo.findData(previous_backend))
                        self.onnx_provider_combo.setCurrentIndex(self.onnx_provider_combo.findData(previous_provider))
                        self.onnx_runtime_widget.set_options(previous_runtime["onnxRuntimeOptions"])
                        old_fp = int(previous_runtime.get("fpMode", 0))
                        self.shared_fp16_check.setChecked(old_fp == 1)
                        if isinstance(fp_widget, QComboBox):
                            with blocked_signals(fp_widget):
                                fp_widget.setCurrentIndex(old_fp)
                        self.main_window.set_floating_point_mode(old_fp)
                    finally:
                        self._auto_label_loading_settings = was_loading
                    self.save_shared_settings()
                self.main_window.restart_live_inference_if_running('backend change cancelled or failed')
                if isinstance(e, InferenceModelLoadCancelled):
                    self.status_label.setText('Backend change cancelled. Previous settings retained.')
                else:
                    self.status_label.setText(f"Could not switch backend: {e}")
                    QMessageBox.critical(self, "Inference Backend", str(e))
                return False
        else:
            self.status_label.setText(
                f"Weights settings applied. Active: {self.main_window.active_inference_backend_label()}."
            )
        return True

    def load_weights_from_dialog(self):
        self.main_window.open_weights()
        self.refresh_from_parent()

    def clear_weights_from_dialog(self):
        if hasattr(self.main_window, "clear_loaded_weights"):
            self.main_window.clear_loaded_weights()
        self.refresh_from_parent()
        self.status_label.setText("Weights cleared.")

    def load_cfg_from_dialog(self):
        if hasattr(self.main_window, "open_cfg"):
            self.main_window.open_cfg()
            self.refresh_from_parent()

    def run_weights_auto_label(self):
        if not self.apply_weight_settings():
            return
        if not getattr(self.main_window, "weights_file_path", None):
            QMessageBox.warning(self, "Auto Label", "Load model weights first.")
            return
        if not getattr(self.main_window, "image_directory", None):
            QMessageBox.warning(self, "Auto Label", "Load an image dataset first.")
            return
        self.status_label.setText("Running Weights Auto Label...")
        QApplication.processEvents()
        shared = self.shared_auto_label_config()
        if not self.main_window.auto_label_yolo_button_clicked(
            overwrite=shared["overwrite"]
        ):
            QMessageBox.warning(self, "Auto Label", "Auto-label could not start. Check the loaded model, CFG, and classes.")
            self.status_label.setText("Weights Auto Label could not start.")
        else:
            self.status_label.setText("Weights Auto Label complete.")

    def _dino_runtime_config(self):
        self.save_dino_settings()
        shared = self.shared_auto_label_config()
        min_size_px, max_percent = self.main_window.current_prediction_size_filter_values()
        return {
            "yoloe_model_name": self.yoloe_model_edit.text().strip() or "yoloe-26s-seg.pt",
            "dino_weights_path": self.dino_weights_edit.text().strip() or None,
            "dino_config_path": self.dino_config_edit.text().strip() or None,
            "runtime_overrides": {
                "YOLOE_CONF": float(self.yoloe_conf_spin.value()),
                "YOLOE_IOU": float(self.yoloe_iou_spin.value()),
                "TEXT_THRESHOLD": float(self.dino_text_spin.value()),
                "BBOX_THRESHOLD": float(self.dino_box_spin.value()),
                "BATCH_SIZE": int(self.dino_batch_spin.value()),
                "PREDICT_IMGSZ": int(self.dino_predict_size_spin.value()),
                "DINO_FP16": shared["fp16"],
                "DINO_FP16_FALLBACK": bool(self.dino_fp16_fallback_check.isChecked()),
                "YOLOE_FP16": shared["fp16"],
                "YOLOE_ONNX_BACKEND": shared["inference_backend"],
                "YOLOE_ONNX_PROVIDER": shared["onnx_provider"],
                "YOLOE_ONNX_RUNTIME_OPTIONS": dict(shared["onnx_runtime_options"]),
                "PREDICTION_MIN_SIZE_PX": float(min_size_px),
                "PREDICTION_MAX_PERCENT": float(max_percent),
                "IGNORE_TEAMMATES": shared["ignore_teammates"],
                "TEAMMATE_THRESHOLD": shared["teammate_threshold"],
            },
        }

    def run_dino_yoloe_auto_label(self):
        self.save_dino_settings()
        dino_weights = self.dino_weights_edit.text().strip()
        if dino_weights and not os.path.exists(dino_weights):
            QMessageBox.warning(self, "Auto Label", f"DINO weights were not found:\n{dino_weights}")
            return

        dino_config = self.dino_config_edit.text().strip()
        if dino_config and not os.path.exists(dino_config):
            QMessageBox.warning(self, "Auto Label", f"DINO config was not found:\n{dino_config}")
            return

        if not getattr(self.main_window, "image_directory", None):
            QMessageBox.warning(self, "Auto Label", "Load an image dataset first.")
            return

        shared = self.shared_auto_label_config()
        self.main_window.start_dino_auto_label(
            overwrite=shared["overwrite"],
            config=self._dino_runtime_config(),
        )

    def open_sahi_settings(self):
        self.tabs.setCurrentIndex(self.TAB_KEYS["sahi"])


class DinoWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    preview = pyqtSignal(str, object, object, object, object)

    def __init__(self, image_directory, overwrite=False, class_names=None, review_dir=None, config=None, parent=None):
        super().__init__()
        self.image_directory = image_directory
        self.overwrite = overwrite
        self.class_names = [
            str(name).strip()
            for name in (class_names or [])
            if str(name).strip()
        ]
        self.review_dir = review_dir
        self.config = dict(config or {})
        self.parent = parent  # Store parent reference for ROI access

    def _emit_progress(self, processed, total):
        self.progress.emit(processed, total)

    def _emit_preview(self, image_path, xyxy_list, score_list, label_list, class_names):
        self.preview.emit(str(image_path), xyxy_list, score_list, label_list, class_names)

    def run(self):
        try:
            self.progress.emit(0, 0)

            if not self.image_directory:
                self.error.emit("No image directory provided.")
                return

            if not os.path.isdir(self.image_directory):
                self.error.emit(f"Image directory does not exist: {self.image_directory}")
                return

            if not self.class_names:
                self.error.emit("No classes selected from DarkFusion dropdown.")
                return

            import darkfusion_dino as dino_runtime

            # The UI captures this before the worker starts. Never access Qt
            # widgets from this thread just to discover the active ROI.
            roi = dict(self.config.get("roi_inference", {}) or {})

            runtime_overrides = self.config.get("runtime_overrides", {})
            ok = dino_runtime.run_groundingdino(
                self.image_directory,
                self.overwrite,
                review_dir=self.review_dir,
                class_names=self.class_names,
                dino_config_path=self.config.get("dino_config_path"),
                dino_weights_path=self.config.get("dino_weights_path"),
                yoloe_model_name=self.config.get("yoloe_model_name"),
                runtime_overrides=runtime_overrides,
                progress_callback=self._emit_progress,
                preview_callback=self._emit_preview,
                roi_enabled=bool(roi.get("enabled", False)),
                roi_bounds=roi.get("bounds_list") or roi.get("bounds"),
                roi_reference_size=roi.get("reference_size"),
                project_class_names=self.config.get("project_class_names"),
            )

            if ok is False:
                self.error.emit("DINO / YOLOE did not complete. Check the DINO weights, config, classes, and log output.")
                return

            self.finished.emit()

        except Exception as e:
            logger.exception("DINO worker failed")
            self.error.emit(str(e))


def setup_unified_auto_label_menu(self):
    """Make Annotate a single Auto Label entry point."""
    if hasattr(self, "auto_label_yolo_button"):
        self.auto_label_yolo_button.setText("Auto Label")
        self.auto_label_yolo_button.setToolTip("Open weights, DINO/YOLOE, and SAHI auto-label settings.")

    for action_name in ("weights_button", "dino_label", "actionSahi_label"):
        action = getattr(self, action_name, None)
        if action is not None:
            action.setVisible(False)

    self._modernize_legacy_detection_panel()


def show_auto_label_dialog(self, initial_tab="weights"):
    self.auto_label_dialog = AutoLabelDialog(self, initial_tab=initial_tab)
    self.auto_label_dialog.exec_()


@staticmethod
def annotation_mode_for_model_task(task):
    task = str(task or "").strip().lower()
    if task in {"segment", "seg", "segmentation"}:
        return "Segmentation"
    if task in {"pose", "keypoint", "keypoints"}:
        return "Keypoints"
    if task in {"obb", "oriented", "oriented_bbox", "rotated"}:
        return "OBB"
    if task in {"detect", "detection", "box", "boxes"}:
        return "Boxes"
    return None


def _task_from_mapping(self, mapping):
    if not isinstance(mapping, dict):
        return None

    for key in ("task", "model_task", "yolo_task"):
        value = mapping.get(key)
        if value:
            return value

    for key in ("args", "train_args", "overrides", "metadata"):
        value = mapping.get(key)
        if isinstance(value, dict):
            task = self._task_from_mapping(value)
            if task:
                return task

    return None


def detect_loaded_model_task(self, model_path=None):
    model_path = str(model_path or getattr(self, "weights_file_path", "") or "")

    model = getattr(self, "model", None)
    lower_path = model_path.lower()
    candidates = []
    # Exported backends need their embedded task checked before model.task.
    # Ultralytics must choose the predictor while YOLO(...) is constructed,
    # and a generically named engine such as "last.engine" can otherwise be
    # guessed as detection even though its metadata says segmentation.
    if lower_path.endswith(".engine"):
        candidates.append(self._task_from_mapping(self.read_engine_metadata(model_path)))

    sidecar_path = model_path + ".json" if model_path else ""
    if sidecar_path and os.path.exists(sidecar_path):
        candidates.append(self._task_from_mapping(read_json_file(sidecar_path, {})))

    candidates.extend([
        getattr(model, "task", None),
        getattr(getattr(model, "model", None), "task", None),
        self._task_from_mapping(getattr(model, "overrides", None)),
        self._task_from_mapping(getattr(getattr(model, "model", None), "args", None)),
        self._task_from_mapping(getattr(getattr(model, "model", None), "yaml", None)),
        self._task_from_mapping(getattr(model, "ckpt", None)),
    ])

    for candidate in candidates:
        mode = self.annotation_mode_for_model_task(candidate)
        if mode:
            return str(candidate).strip().lower(), mode

    name = os.path.basename(lower_path)
    if re.search(r"(^|[-_.])seg(ment)?($|[-_.])", name):
        return "segment", "Segmentation"
    if re.search(r"(^|[-_.])pose($|[-_.])", name):
        return "pose", "Keypoints"
    if re.search(r"(^|[-_.])obb($|[-_.])", name):
        return "obb", "OBB"

    if lower_path.endswith(".weights"):
        return "detect", "Boxes"

    return None, None


def exported_model_task_hint(self, model_path):
    """Return a task that must be supplied while constructing an exported model."""
    model_path = self.normalize_path(model_path)
    if not model_path:
        return None

    task = None
    lower_path = model_path.lower()
    if lower_path.endswith(".engine"):
        task = self._task_from_mapping(self.read_engine_metadata(model_path))

    sidecar_path = model_path + ".json"
    if not task and os.path.exists(sidecar_path):
        task = self._task_from_mapping(read_json_file(sidecar_path, {}))

    if not task:
        name = os.path.basename(lower_path)
        if re.search(r"(^|[-_.])seg(ment)?($|[-_.])", name):
            task = "segment"
        elif re.search(r"(^|[-_.])pose($|[-_.])", name):
            task = "pose"
        elif re.search(r"(^|[-_.])obb($|[-_.])", name):
            task = "obb"

    task = str(task or "").strip().lower()
    aliases = {
        "seg": "segment",
        "segmentation": "segment",
        "detection": "detect",
        "keypoint": "pose",
        "keypoints": "pose",
    }
    task = aliases.get(task, task)
    return task if task in {"detect", "segment", "classify", "pose", "obb", "semantic"} else None


def create_ultralytics_model(self, model_path):
    """Construct YOLO with an explicit exported-model task when metadata provides one."""
    model_path = self.normalize_path(model_path)
    task = self.exported_model_task_hint(model_path)
    if task:
        logger.info("Loading Ultralytics model as task=%s from %s", task, model_path)
        return YOLO(model_path, task=task)
    return YOLO(model_path)


@staticmethod
def normalize_inference_backend(value):
    return _normalize_inference_backend(value)


def inference_backend_for_model(self, model_path=None):
    model_path = self.normalize_path(
        model_path or getattr(self, "weights_file_path", "") or ""
    )
    return inference_backend_for_path(model_path, self.settings.get("inferenceBackend", "auto"))


def onnx_execution_provider(self):
    return str(self.settings.get("onnxExecutionProvider", "auto") or "auto").strip().lower()


def inference_model_identity(self, model_path=None):
    model_path = self.normalize_path(
        model_path or getattr(self, "weights_file_path", "") or ""
    )
    backend = self.inference_backend_for_model(model_path)
    provider = self.onnx_execution_provider() if backend == "onnxruntime" else ""
    precision = "fp16" if backend == "onnxruntime" and int(getattr(self, "fp_mode", 0) or 0) == 1 else "fp32"
    normalized_path = os.path.normcase(os.path.abspath(model_path)) if model_path else ""
    options = json.dumps(onnx_runtime_settings(self.settings.get("onnxRuntimeOptions", {})), sort_keys=True) if backend == "onnxruntime" else ""
    return normalized_path, backend, provider, precision, options


@staticmethod
def is_darkfusion_onnx_model(model):
    return bool(getattr(model, "_darkfusion_inference_backend", "") == "onnxruntime")


def create_inference_model(self, model_path):
    """Create the selected backend while keeping one Ultralytics-style result contract."""
    if not gpu_model_allowed(self, "detector"):
        raise GpuWorkBusy("Detector loading is paused while another GPU workflow is running.")
    model_path = self.normalize_path(model_path)
    backend = self.inference_backend_for_model(model_path)
    identity = self.inference_model_identity(model_path)
    runtime_settings = {
        'inferenceBackend': self.settings.get('inferenceBackend', 'auto'),
        'onnxExecutionProvider': self.settings.get('onnxExecutionProvider', 'auto'),
        'onnxRuntimeOptions': copy.deepcopy(self.settings.get('onnxRuntimeOptions', {}) or {}),
        'fpMode': int(getattr(self, 'fp_mode', 0) or 0),
    }
    if backend == "onnxruntime":
        if not model_path.lower().endswith(".onnx"):
            raise ValueError("The standalone ONNX Runtime backend requires a .onnx model.")
        task = self.exported_model_task_hint(model_path)
        provider = self.onnx_execution_provider()
        def construct():
            model = load_inference_model(model_path, backend=backend, provider=provider, task=task,
                fp16=runtime_settings['fpMode'] == 1, runtime_options=runtime_settings['onnxRuntimeOptions'])
            logger.info('Loaded standalone ONNX Runtime backend with providers: %s', ', '.join(model.providers))
            for diagnostic in model.setup_diagnostics:
                logger.warning(diagnostic)
            return model
    else:
        native_factory = self.create_ultralytics_model
        def construct():
            return load_inference_model(model_path, backend=backend, native_factory=native_factory)
    model = _construct_model_without_blocking_gui(self, construct)
    model._darkfusion_inference_backend = backend
    model._darkfusion_load_identity = identity
    model._darkfusion_runtime_settings = runtime_settings
    self.active_inference_backend = backend
    self._gpu_released_detector = False
    return model


def active_inference_backend_label(self):
    model = getattr(self, "model", None)
    backend = str(getattr(model, "_darkfusion_inference_backend", "") or "")
    if backend == "onnxruntime":
        providers = list(getattr(model, "providers", []) or [])
        runtime_name = str(getattr(model, "_darkfusion_runtime_name", "ONNX Runtime"))
        return runtime_name + (f" ({providers[0]})" if providers else "")
    if backend == "ultralytics":
        return "Ultralytics"
    if getattr(self, "net", None) is not None:
        return "OpenCV DNN"
    return self.inference_backend_for_model()


def reload_selected_inference_model(self):
    model_path = self.normalize_path(getattr(self, "weights_file_path", "") or "")
    if not model_path or not os.path.isfile(model_path):
        raise ValueError("Select an existing model before changing its inference backend.")
    if self.determine_model_type(model_path) == "weights":
        if not self.initialize_yolo():
            raise RuntimeError("OpenCV DNN could not initialize the selected Darknet model.")
        return self.net

    previous_path = self.normalize_path(getattr(self, "loaded_model_path", "") or "")
    same_model = bool(previous_path) and os.path.normcase(os.path.abspath(previous_path)) == os.path.normcase(os.path.abspath(model_path))
    self.stop_video_inference_thread(wait_ms=0)
    self.model = self.create_inference_model(model_path)
    self.loaded_model_path = model_path
    self.model_directory = os.path.dirname(model_path)
    self.apply_loaded_weight_input_size(model_path)
    target_dir = getattr(self, "image_directory", None) or getattr(self, "output_path", None) or os.getcwd()
    if not same_model:
        self.sync_classes_from_current_model(data_directory=target_dir, write_classes_file=False)
        self.sync_pose_list_from_current_model(model_path=model_path, write_points_file=True, force=True)
        self.auto_set_annotation_mode_for_model(model_path, force=True)
    self.restart_live_inference_if_running("inference backend changed")
    return self.model


def auto_set_annotation_mode_for_model(self, model_path=None, force=False):
    _task, mode = self.detect_loaded_model_task(model_path)
    if not mode:
        logger.info("Could not detect annotation mode from model: %s", model_path)
        return False

    current = self.segmentation_mode.currentText()
    if current == mode and not force:
        self._last_auto_annotation_mode = mode
        return False

    self._annotation_mode_auto_changing = True
    try:
        self.set_annotation_mode(mode)
    finally:
        self._annotation_mode_auto_changing = False

    self._annotation_mode_manual_override = False
    self._last_auto_annotation_mode = mode
    self.statusBar().showMessage(f"Annotation mode set from model: {mode}", 3500)
    return True


def showSahiSettings(self):
    self.sahiSettingsDialog = SahiSettingsDialog(self)
    self.sahiSettingsDialog.exec_()


def predict_and_draw_yolo_objects(self, image, image_file_path):
    try:
        image = self.ensure_rgb(image)
        if image is None or not os.path.isfile(image_file_path):
            logging.error("Invalid image or path in predict_and_draw_yolo_objects.")
            return None

        img_height, img_width = image.shape[:2]
        image_with_boxes = image.copy()
        image_copy = image.copy()

        yolo_file_path = self.get_label_file_safe(image_file_path)
        if not yolo_file_path:
            logging.info("Skipping %s due to missing annotation file.", image_file_path)
            return None

        yolo_lines = self._valid_annotation_lines_from_file(yolo_file_path)
        if not yolo_lines:
            logging.info("Skipping %s due to missing valid annotations.", image_file_path)
            return None

        masks_for_bboxes = []
        adjusted_boxes = []
        polygon_parse_preference = self._polygon_label_parse_preference()

        for yolo_line in yolo_lines:
            output_line = str(yolo_line or "").strip()
            try:
                box_obj = BoundingBox.from_str(
                    yolo_line,
                    preferred_polygon_type=polygon_parse_preference,
                )
                if box_obj is None:
                    if self.overwrite and output_line:
                        adjusted_boxes.append(output_line)
                    continue

                mask, original_bbox, label_type = self._process_label_to_mask(box_obj, image, img_width, img_height)
                if mask is None:
                    if self.overwrite and output_line:
                        adjusted_boxes.append(output_line)
                    continue

                image_with_boxes, new_bbox = self.draw_mask_and_bbox(
                    image_with_boxes,
                    mask,
                    original_bbox=original_bbox,
                    obb_points=box_obj.obb if getattr(box_obj, "obb", None) else None,
                    class_id=box_obj.class_id,
                    num_classes=self.classes_dropdown.count(),
                )

                if new_bbox == (0, 0, 0, 0):
                    if self.overwrite and output_line:
                        adjusted_boxes.append(output_line)
                    continue

                masks_for_bboxes.append((mask, new_bbox))

                if self.overwrite:
                    updated_bbox = self._update_box_obj_from_mask(box_obj, mask, label_type, img_width, img_height)
                    if updated_bbox is not None:
                        output_line = box_obj.to_str(remove_confidence=True)

                if self.overwrite and output_line:
                    adjusted_boxes.append(output_line)

            except Exception as e:
                logging.error("Error processing YOLO line '%s': %s", yolo_line, e, exc_info=True)
                if self.overwrite and output_line:
                    adjusted_boxes.append(output_line)

        if self.overwrite and adjusted_boxes:
            self._save_sam3_label_updates(yolo_file_path, adjusted_boxes)

        if self.is_noise_remove_enabled:
            self.apply_noise_reduction(image_copy.copy(), masks_for_bboxes, image_file_path)

        if self.shadow and masks_for_bboxes:
            self.create_shadow_image(image_copy.copy(), masks_for_bboxes, image_file_path)

        return image_with_boxes if self.screen_update.isChecked() else None

    except Exception as e:
        logging.critical("Unhandled crash in predict_and_draw_yolo_objects: %s", e, exc_info=True)
        self._safe_empty_cache()
        return None


def on_dino_label_clicked(self):
    self.show_auto_label_dialog("dino")


@gpu_work("DINO auto labeling", keep=("analysis",), until="dino_worker")
def start_dino_auto_label(self, overwrite=False, config=None):
    self.dino_label.setEnabled(False)

    try:
        if self.image_directory is not None:
            existing_worker = getattr(self, "dino_worker", None)
            try:
                if existing_worker is not None and existing_worker.isRunning():
                    QMessageBox.warning(self, "DINO Running", "DINO / YOLOE auto-label is already running.")
                    self.dino_label.setEnabled(True)
                    return
            except RuntimeError:
                self.dino_worker = None

            self.img_index_number_changed(0)
            self.set_label_progress_busy("Loading DINO / YOLOE...")
            try:
                self.statusBar().showMessage("Loading DINO / YOLOE models...", 5000)
            except Exception:
                pass

            active_class_names = self.get_active_class_names_from_ui()

            worker_config = dict(config or {})
            worker_config["roi_inference"] = self.roi_inference_snapshot()
            worker_config["project_class_names"] = list(self.class_labels)
            self.dino_worker = DinoWorker(
                self.image_directory,
                overwrite,
                active_class_names,
                config=worker_config,
                parent=self,
            )
            self.dino_worker.finished.connect(self.on_dino_finished)
            self.dino_worker.error.connect(self.on_dino_error)
            self.dino_worker.progress.connect(self.on_dino_progress)
            self.dino_worker.preview.connect(self.on_dino_preview)
            self.dino_worker.start()
        else:
            QMessageBox.warning(
                self,
                "Directory Not Selected",
                "Please select an image directory first."
            )
            self.open_image_video()
            self.dino_label.setEnabled(True)

    except Exception as e:
        QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        self.dino_label.setEnabled(True)


def on_dino_finished(self):
    """Called when the DINO worker finishes."""
    maximum = 100
    try:
        maximum = max(1, int(self.label_progress.maximum()))
    except Exception:
        pass
    self.set_label_progress(0, maximum, value=maximum, progress_format="DINO done")
    QMessageBox.information(self, "DINO Completed", "Auto-labeling completed successfully!")
    self.dino_label.setEnabled(True)  # Re-enable the button
    self._cleanup_worker_reference("dino_worker")


def on_dino_error(self, error_message):
    """Handle errors in the DINO worker."""
    self.reset_label_progress(0)
    QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
    self.dino_label.setEnabled(True)  # Re-enable the button
    self._cleanup_worker_reference("dino_worker")


def on_dino_progress(self, processed, total):
    try:
        processed = int(processed)
        total = int(total)
    except Exception:
        processed, total = 0, 0

    if total <= 0:
        self.set_label_progress_busy("Loading DINO / YOLOE...")
        return

    processed = max(0, min(processed, total))
    self.set_label_progress(0, total, value=processed, progress_format=f"DINO {processed}/{total}")
    try:
        self.statusBar().showMessage(f"DINO / YOLOE auto-label: {processed}/{total}", 2500)
    except Exception:
        pass


def on_dino_preview(self, image_path, xyxy_list=None, score_list=None, label_list=None, class_names=None):
    if not image_path:
        return
    try:
        image_path = str(image_path)
        if not os.path.exists(image_path):
            return
        self.current_file = image_path
        self.display_image(image_path, rebuild_preview=False)

        # Draw detected boxes on the preview
        if xyxy_list and self.screen_view and self.screen_view.scene():
            scene = self.screen_view.scene()

            # Clear previous preview boxes
            for item in list(scene.items()):
                if hasattr(item, "_is_dino_preview_box"):
                    scene.removeItem(item)

            # Draw new detected boxes
            for box_idx, (xyxy, score, label_id) in enumerate(zip(
                xyxy_list or [],
                score_list or [],
                label_list or []
            )):
                try:
                    if not xyxy or len(xyxy) < 4:
                        continue

                    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                    width = x2 - x1
                    height = y2 - y1

                    # Create preview box
                    preview_box = QGraphicsRectItem(x1, y1, width, height)
                    preview_box._is_dino_preview_box = True

                    # Dense prompt IDs are not project IDs; use the same
                    # project palette as the editor and video renderers.
                    preview_name = str(class_names[label_id]) if class_names and 0 <= int(label_id) < len(class_names) else ""
                    project_names = list(getattr(self, "class_labels", []) or [])
                    project_id = class_name_index(preview_name, project_names)
                    if project_id is None:
                        continue
                    pen = QPen(get_color(project_id, len(project_names), alpha=255, class_names=project_names))
                    pen.setWidth(2)
                    preview_box.setPen(pen)
                    preview_box.setBrush(QBrush(get_color(project_id, len(project_names), alpha=20, class_names=project_names)))
                    preview_box.setZValue(10)

                    scene.addItem(preview_box)

                    # Add label text if available
                    if class_names and isinstance(label_id, int) and 0 <= label_id < len(class_names):
                        class_name = str(project_names[project_id])
                        confidence = f"{float(score)*100:.0f}%" if score else ""
                        label_text = f"{class_name} {confidence}".strip()

                        text_item = QGraphicsTextItem(label_text, preview_box)
                        text_item.setPos(x1 + 5, y1 + 5)
                        text_item.setDefaultTextColor(pen.color())
                        text_item.setZValue(11)

                except Exception as e:
                    logger.debug(f"Could not draw DINO preview box {box_idx}: {e}")

        QApplication.processEvents()
    except Exception as e:
        logger.debug(f"Could not update DINO preview for {image_path}: {e}")


def stop_auto_labeling(self):
    self.stop_labeling = True
    self.reset_progress_bar()
    torch.cuda.empty_cache()  # Clear CUDA cache


def threshold_spinbox_changed(self):
    self.confidence_threshold = self.confidence_threshold_spinbox.value()
    self.nms_threshold = self.nms_threshold_spinbox.value()

    self.queue_live_inference_refresh("threshold changed")

    if getattr(self, 'net', None) is not None and getattr(self, 'current_file', None) and os.path.exists(self.current_file):
        self.process_timer.start(self.process_timer_interval)
    else:
        logging.debug("Thresholds updated but no valid net/file to run inference.")


def open_weights(self, file_name=None):
    if getattr(self, '_inference_model_load_worker', None) is not None:
        self.post_status_message('A model is still loading. Wait for it to finish before selecting another.', 5000)
        return
    # QAction/button signals pass a boolean; an explicit path reuses this
    # same loader from the SAHI panel without opening another file picker.
    if not isinstance(file_name, (str, os.PathLike)):
        file_name = None
    elif file_name:
        file_name = os.fspath(file_name)
    if not file_name:
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        options |= QFileDialog.DontUseCustomDirectoryIcons
        # Keep the picker above the Auto Label dialog that launched it.
        parent = QApplication.activeModalWidget() or self
        dialog = QFileDialog(
            parent,
            "Open Weights/Model",
            self.dialog_start_directory("model"),
            "Model Files (*.pt *.engine *.onnx *.weights)",
        )
        dialog.setOptions(options)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        # Reach model/run folders directly instead of enumerating a dataset with
        # hundreds of thousands of images on the way to its runs subdirectory.
        places = dialog.sidebarUrls()
        candidates = [os.path.join(APP_DIR,"weights"),
                      self.settings.get("ultralyticsRunsDirectory", ""),
                      self.settings.get("trainingEvaluatorRunsDir", "")]
        recent_log = self.settings.get("trainingEvaluatorLastTrainingRunLog", "")
        if recent_log:
            candidates.append(os.path.dirname(os.path.dirname(recent_log)))
        existing = {os.path.normcase(os.path.abspath(url.toLocalFile())) for url in places if url.isLocalFile()}
        for candidate in candidates:
            if not candidate or not os.path.isdir(candidate):
                continue
            candidate = os.path.abspath(candidate)
            if os.path.normcase(candidate) not in existing:
                places.append(QUrl.fromLocalFile(candidate))
                existing.add(os.path.normcase(candidate))
        dialog.setSidebarUrls(places)
        try:
            files = dialog.selectedFiles() if dialog.exec_() == QDialog.Accepted else []
            file_name = files[0] if files else ""
        finally:
            dialog.deleteLater()

    if file_name:
        self.remember_dialog_selection("model", file_name)
        self.weights_file_path = file_name
        self.model_directory = os.path.dirname(file_name)
        self.model_type = self.determine_model_type(file_name)
        self.settings["lastWeightsPath"] = self.normalize_path(file_name)
        self.settings["lastModelPath"] = self.normalize_path(file_name)
        if self.model_type in {"pt", "trt", "onnx"}:
            self.settings["lastPTWeightsPath"] = self.normalize_path(file_name)
        self.saveSettings()

        model_size_applied = self.apply_loaded_weight_input_size(file_name)

        if self.model_type == "weights":
            self.model = None
            self.loaded_model_path = None
            self.net = None
            self.cfg_file_path = None
            target_dir = getattr(self, "image_directory", None) or getattr(self, "output_path", None) or os.getcwd()
            class_names = self.load_classes(data_directory=target_dir)

            if class_names:
                self.class_labels = list(class_names)
                self.class_names = list(class_names)
                self.id_to_class = {i: name for i, name in enumerate(class_names)}

            initialized = self.prompt_for_matching_cfg(file_name)
            self.auto_set_annotation_mode_for_model(file_name, force=True)
            self.saveSettings()

            QMessageBox.information(
                self,
                "Darknet Weights Loaded",
                "Darknet weights loaded. Select the matching CFG file before auto-labeling."
                if not getattr(self, "cfg_file_path", None)
                else (
                    "Darknet weights and CFG are ready."
                    if initialized
                    else "Darknet weights and CFG are selected, but the model could not initialize yet. Check the CFG and classes.txt or load the image dataset before auto-labeling."
                )
            )
            return

        try:
            self.net = None
            self.stop_video_inference_thread(wait_ms=0)
            self.model = None
            self.model = self.create_inference_model(file_name)
            self.loaded_model_path = self.normalize_path(file_name)
            if not model_size_applied:
                model_size_applied = self.apply_loaded_weight_input_size(file_name)
            logging.info(f"Loaded {self.model_type} model from {file_name}")
            target_dir = getattr(self, "image_directory", None) or getattr(self, "output_path", None) or os.getcwd()

            # Sync classes from the currently loaded model
            class_names = self.sync_classes_from_current_model(
                data_directory=target_dir,
                write_classes_file=True
            )

            # Keep in-memory class labels aligned too
            if class_names:
                self.class_labels = list(class_names)
                self.class_names = list(class_names)
                self.id_to_class = {i: name for i, name in enumerate(class_names)}

            self.sync_pose_list_from_current_model(
                model_path=file_name,
                write_points_file=True,
                force=True,
            )
            self.auto_set_annotation_mode_for_model(file_name, force=True)

            size_note = ""
            if model_size_applied:
                network_w, network_h = self.get_network_size_wh()
                size_note = f"\nNetwork size set to W={network_w}, H={network_h} (YOLO imgsz [H, W])."
            QMessageBox.information(
                self,
                "Model Loaded",
                f"Successfully loaded {self.model_type} model with "
                    f"{self.active_inference_backend_label()}.{size_note}",
            )
        except InferenceModelLoadCancelled:
            logger.info('Model loading cancelled: %s', file_name)
            self.model = None
            self.post_status_message('Model loading cancelled; no model is loaded.', 5000)
        except Exception as e:
            logging.error(f"Failed to load model from {file_name}: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load the model: {str(e)}")
            self.model = None


def clear_loaded_weights(self):
    """Unload model weights used by auto-labeling and live video/stream detection."""
    self._gpu_released_detector = False
    self.weights_file_path = None
    self.cfg_file_path = None
    self.model_directory = None
    self.model_type = ""
    self.weights_file = None
    self.weights_files = []

    self.model = None
    self.loaded_model_path = None
    self.active_inference_backend = ""
    self.net = None
    self.layer_names = []
    self.classes = []
    self._last_classes_model_path = None
    self._last_classes_source_key = None

    for key in ("lastWeightsPath", "lastModelPath", "lastPTWeightsPath", "lastCfgPath"):
        self.settings[key] = ""

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.debug(f"Could not clear CUDA cache after unloading weights: {e}")

    self.saveSettings()
    logger.info("Cleared loaded weights/model state.")


def determine_model_type(self, file_path):
    file_path = str(file_path or "").lower()
    if file_path.endswith('.pt'):
        return 'pt'
    elif file_path.endswith('.engine'):
        return 'trt'
    elif file_path.endswith('.onnx'):
        return 'onnx'
    elif file_path.endswith('.weights'):
        return 'weights'
    else:
        return 'unknown'


def read_engine_metadata(self, engine_path):
    """Read the Ultralytics JSON header embedded at the start of many .engine files."""
    try:
        with open(engine_path, "rb") as f:
            header = f.read(4)
            if len(header) != 4:
                return None
            metadata_length = int.from_bytes(header, "little")
            if metadata_length <= 0 or metadata_length > 10_000_000:
                return None
            metadata_bytes = f.read(metadata_length)
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        return metadata if isinstance(metadata, dict) else None
    except Exception as e:
        logger.debug(f"Could not read engine metadata from {engine_path}: {e}")
        return None


def engine_size_from_metadata(self, metadata):
    if not metadata:
        return None

    imgsz = metadata.get("imgsz") or metadata.get("img_size") or metadata.get("image_size")
    try:
        if isinstance(imgsz, int):
            height = width = imgsz
        elif isinstance(imgsz, (list, tuple)) and len(imgsz) >= 2:
            height, width = int(imgsz[0]), int(imgsz[1])
        elif isinstance(imgsz, (list, tuple)) and len(imgsz) == 1:
            height = width = int(imgsz[0])
        else:
            return None
    except (TypeError, ValueError):
        return None

    if width <= 0 or height <= 0:
        return None
    return width, height


def engine_size_from_tensorrt(self, engine_path):
    if not gpu_model_allowed(self, "detector"):
        return None
    try:
        import tensorrt as trt
    except Exception as e:
        logger.debug(f"TensorRT is unavailable for engine size detection: {e}")
        return None

    try:
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            return None

        if hasattr(engine, "num_io_tensors"):
            tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            for tensor_name in tensor_names:
                if engine.get_tensor_mode(tensor_name) != trt.TensorIOMode.INPUT:
                    continue
                shape = tuple(int(dim) for dim in engine.get_tensor_shape(tensor_name))
                if len(shape) >= 4 and shape[-1] > 0 and shape[-2] > 0:
                    return shape[-1], shape[-2]
        else:
            for index in range(engine.num_bindings):
                if not engine.binding_is_input(index):
                    continue
                shape = tuple(int(dim) for dim in engine.get_binding_shape(index))
                if len(shape) >= 4 and shape[-1] > 0 and shape[-2] > 0:
                    return shape[-1], shape[-2]
    except Exception as e:
        logger.debug(f"Could not deserialize engine for size detection: {e}")

    return None


def engine_size_from_filename(self, engine_path):
    name = os.path.basename(str(engine_path or "")).lower()
    match = re.search(r"imgsz(\d+)(?:x(\d+))?", name)
    if not match:
        return None

    height = int(match.group(1))
    width = int(match.group(2) or height)
    return width, height


def model_size_from_sidecar_json(self, model_path):
    sidecar_path = os.path.splitext(str(model_path or ""))[0] + os.path.splitext(str(model_path or ""))[1] + ".json"
    if not os.path.exists(sidecar_path):
        return None

    metadata = read_json_file(sidecar_path, {})
    return self.engine_size_from_metadata(metadata)


def model_size_from_onnx(self, model_path):
    try:
        import onnx

        model = onnx.load(model_path, load_external_data=False)
        for input_info in model.graph.input:
            dims = input_info.type.tensor_type.shape.dim
            shape = [
                dim.dim_value
                for dim in dims
                if getattr(dim, "dim_value", 0) and dim.dim_value > 0
            ]
            if len(shape) >= 4:
                height, width = int(shape[-2]), int(shape[-1])
                if width > 0 and height > 0:
                    return width, height
    except Exception as e:
        logger.debug(f"Could not read ONNX input size from {model_path}: {e}")

    return None


def model_size_from_loaded_model(self):
    model = getattr(self, "model", None)
    candidates = [
        getattr(model, "overrides", None),
        getattr(getattr(model, "model", None), "args", None),
        getattr(getattr(model, "model", None), "yaml", None),
    ]

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        size = self.engine_size_from_metadata(candidate)
        if size:
            return size

    return None


def detect_weight_input_size(self, model_path):
    model_type = self.determine_model_type(model_path)

    if model_type == "trt":
        metadata_size = self.engine_size_from_metadata(self.read_engine_metadata(model_path))
        if metadata_size:
            return metadata_size, "engine metadata"

        tensorrt_size = self.engine_size_from_tensorrt(model_path)
        if tensorrt_size:
            return tensorrt_size, "TensorRT binding"

    if model_type == "onnx":
        onnx_size = self.model_size_from_onnx(model_path)
        if onnx_size:
            return onnx_size, "ONNX input shape"

    loaded_size = self.model_size_from_loaded_model()
    if loaded_size:
        return loaded_size, "loaded model metadata"

    sidecar_size = self.model_size_from_sidecar_json(model_path)
    if sidecar_size:
        return sidecar_size, "sidecar metadata"

    filename_size = self.engine_size_from_filename(model_path)
    if filename_size:
        return filename_size, "file name"

    return None, None


def detect_engine_input_size(self, engine_path):
    metadata_size = self.engine_size_from_metadata(self.read_engine_metadata(engine_path))
    if metadata_size:
        return metadata_size, "engine metadata"

    tensorrt_size = self.engine_size_from_tensorrt(engine_path)
    if tensorrt_size:
        return tensorrt_size, "TensorRT binding"

    filename_size = self.engine_size_from_filename(engine_path)
    if filename_size:
        return filename_size, "file name"

    return None, None


def apply_loaded_weight_input_size(self, model_path):
    size, source = self.detect_weight_input_size(model_path)
    if not size:
        self._loaded_model_input_size_wh = None
        logger.info(f"Could not detect fixed input size for: {model_path}")
        return False

    width, height = size
    self._loaded_model_input_size_wh = (int(width), int(height))
    for widget, value in ((self.network_width, width), (self.network_height, height)):
        if widget.maximum() < value:
            widget.setMaximum(value)
        with blocked_signals(widget):
            widget.setValue(value)

    self.sync_frame_resize_to_network_size(enable_resize=True, reason=f"{source} input size")
    self.update_resize_policy_controls()

    logger.info(
        "Set network size to W=%s H=%s from %s: %s. Ultralytics imgsz will be [H, W].",
        width,
        height,
        source,
        os.path.basename(model_path),
    )
    return True


def merge_auto_label_bboxes(
    self,
    existing_bboxes,
    new_bboxes,
    duplicate_iou_threshold=0.5,
    keypoint_update_iou_threshold=0.85,
):
    final_bboxes = list(existing_bboxes or [])

    for new_box in new_bboxes or []:
        best_existing_box = None
        best_iou = 0.0

        for existing_box in final_bboxes:
            try:
                if int(new_box.class_id) != int(existing_box.class_id):
                    continue

                current_iou = new_box.iou(existing_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_existing_box = existing_box
            except Exception:
                continue

        if best_existing_box is None or best_iou <= duplicate_iou_threshold:
            final_bboxes.append(new_box)
            continue

        new_keypoints = getattr(new_box, "keypoints", None) or []
        existing_keypoints = getattr(best_existing_box, "keypoints", None) or []
        new_has_visible_keypoints = self.has_visible_keypoints(new_keypoints)
        existing_has_visible_keypoints = self.has_visible_keypoints(existing_keypoints)

        if self.is_keypoint_mode() and new_has_visible_keypoints:
            if not existing_has_visible_keypoints:
                final_bboxes.append(new_box)
                continue

            if not self._dedupe_keypoints_compatible(new_keypoints, existing_keypoints):
                final_bboxes.append(new_box)
                continue

            if best_iou < keypoint_update_iou_threshold:
                final_bboxes.append(new_box)
                continue

    return self.remove_near_duplicate_bounding_boxes(
        final_bboxes,
        iou_threshold=duplicate_iou_threshold,
        class_aware=True
    )


def clear_auto_label_files_for_images(self, image_files):
    """Clear existing label files before an overwrite auto-label pass."""
    cleared = 0
    for image_file in image_files or []:
        try:
            label_file = self.get_label_file(image_file)
            if not label_file or not os.path.exists(label_file):
                continue

            self.save_labels_to_file(label_file, [], mode="w", allow_empty=True)
            cleared += 1
        except Exception as e:
            logging.warning(f"Failed to clear old labels for {image_file}: {e}")

    if cleared:
        logging.info(f"Cleared {cleared} existing label file(s) before auto-label overwrite.")
    return cleared


@staticmethod
def _qsettings_bool_value(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def auto_label_teammate_filter_config(self):
    settings = QSettings("UltraDarkFusion", "AutoLabel")
    enabled = getattr(self, "auto_label_ignore_teammates", None)
    if enabled is None:
        enabled = self._qsettings_bool_value(settings.value("ignore_teammates", False))
    threshold = getattr(self, "auto_label_teammate_threshold", None)
    if threshold is None:
        try:
            threshold = float(settings.value("teammate_threshold", 0.90))
        except Exception:
            threshold = 0.90
    return bool(enabled), max(0.50, min(0.999, float(threshold)))


@staticmethod
def _teammate_filter_bbox_bounds(bbox):
    try:
        if float(bbox.width) > 0 and float(bbox.height) > 0:
            return [
                max(0.0, float(bbox.x_center) - float(bbox.width) / 2.0),
                max(0.0, float(bbox.y_center) - float(bbox.height) / 2.0),
                min(1.0, float(bbox.x_center) + float(bbox.width) / 2.0),
                min(1.0, float(bbox.y_center) + float(bbox.height) / 2.0),
            ]
        points = list(getattr(bbox, "segmentation", []) or getattr(bbox, "obb", []) or [])
        xs = [float(points[index]) for index in range(0, len(points) - 1, 2)]
        ys = [float(points[index]) for index in range(1, len(points), 2)]
        if xs and ys:
            return [max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys))]
    except Exception:
        pass
    return None


@staticmethod
def _teammate_filter_pil_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray) and image.size:
        if image.ndim == 2:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        if image.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if isinstance(image, (str, os.PathLike)) and os.path.isfile(image):
        with Image.open(image) as source:
            return source.convert("RGB")
    return None


def filter_teammate_auto_label_batches(
    self,
    images,
    bbox_batches,
    class_names=None,
    enabled=None,
    threshold=None,
    source="auto label",
):
    """Reject likely friendly-player predictions in one batched CLIP pass."""
    configured_enabled, configured_threshold = self.auto_label_teammate_filter_config()
    enabled = configured_enabled if enabled is None else bool(enabled)
    threshold = configured_threshold if threshold is None else max(0.50, min(0.999, float(threshold)))
    output = [list(batch or []) for batch in bbox_batches]
    if not enabled or not any(output):
        return output

    class_names = list(class_names or getattr(self, "class_labels", []) or getattr(self, "class_names", []) or [])
    filter_images = []
    bounds_batches = []
    bbox_index_maps = []
    try:
        from darkfusion_teammate_review import TeammateMarkerClassifier

        classifier = getattr(self, "_auto_label_teammate_classifier", None)
        if classifier is None:
            classifier = TeammateMarkerClassifier("cuda" if torch.cuda.is_available() else "cpu")
            self._auto_label_teammate_classifier = classifier

        for image_index, (image, bboxes) in enumerate(zip(images, output)):
            pil_image = self._teammate_filter_pil_image(image)
            filter_images.append(pil_image)
            image_bounds = []
            image_indices = []
            if pil_image is None:
                bounds_batches.append(image_bounds)
                bbox_index_maps.append(image_indices)
                continue
            for bbox_index, bbox in enumerate(bboxes):
                class_id = int(getattr(bbox, "class_id", -1))
                class_name = class_names[class_id].lower() if 0 <= class_id < len(class_names) else ""
                if class_name and not any(token in class_name for token in ("person", "player", "enemy", "character")):
                    continue
                bounds = self._teammate_filter_bbox_bounds(bbox)
                if not bounds:
                    continue
                image_bounds.append(bounds)
                image_indices.append(bbox_index)
            bounds_batches.append(image_bounds)
            bbox_index_maps.append(image_indices)

        self.post_status_message(
            "Checking overhead teammate markers and gamer tags...", 3000
        )
        rejected_local = classifier.friendly_indices(
            filter_images,
            bounds_batches,
            threshold=threshold,
            batch_size=96,
            ocr=True,
        )
    except Exception as error:
        logger.warning("Teammate auto-label filter unavailable: %s", error)
        if not getattr(self, "_teammate_filter_warning_shown", False):
            self._teammate_filter_warning_shown = True
            self.post_status_message(
                f"Teammate filter unavailable; predictions were kept: {error}", 6000
            )
        return output

    rejected = defaultdict(set)
    for image_index, local_indices in enumerate(rejected_local):
        index_map = bbox_index_maps[image_index] if image_index < len(bbox_index_maps) else []
        for local_index in local_indices:
            if 0 <= int(local_index) < len(index_map):
                rejected[image_index].add(index_map[int(local_index)])
    if classifier.last_ocr_error:
        logger.warning(
            "Teammate OCR unavailable; visual marker filtering remains active: %s",
            classifier.last_ocr_error,
        )
        if not getattr(self, "_teammate_ocr_warning_shown", False):
            self._teammate_ocr_warning_shown = True
            self.post_status_message(
                "Teammate OCR unavailable; colored marker/nameplate filtering is still active.",
                6000,
            )
    if rejected:
        output = [
            [bbox for index, bbox in enumerate(batch) if index not in rejected.get(image_index, set())]
            for image_index, batch in enumerate(output)
        ]
    rejected_count = sum(len(indices) for indices in rejected.values())
    if rejected_count:
        evidence = dict(getattr(classifier, "last_evidence_counts", {}) or {})
        logger.info(
            "Ignored %d teammate prediction(s) in %s at certainty %.3f "
                "(visual=%d, OCR=%d).",
            rejected_count,
            source,
            threshold,
            int(evidence.get("visual_marker", 0)),
            int(evidence.get("ocr_gamer_tag", 0)),
        )
        self.post_status_message(
            f"Ignored {rejected_count} teammate prediction(s) · visual marker + gamer-tag OCR",
            3000,
        )
    return output


def filter_teammate_auto_labels(self, image, bboxes, **kwargs):
    batches = self.filter_teammate_auto_label_batches([image], [bboxes], **kwargs)
    return batches[0] if batches else []


def _refine_yolo_boxes_with_sam3(self, image, bboxes):
    """Refine only new detection boxes; never infer a class or drop a YOLO box."""
    output = list(bboxes or [])
    eligible = [index for index, box in enumerate(output) if self._label_type(box) == "bbox"]
    stats = self.__dict__.setdefault("_yolo_sam3_stats", {"refined": 0, "kept": 0})
    if not eligible or getattr(self, "_yolo_sam3_unavailable", ""):
        return output
    try:
        image = self.ensure_rgb(image)  # Legacy helper returns BGR for Ultralytics.
        if image is None or not image.size:
            return output
        height, width = image.shape[:2]
        model = self._get_sam_model()
        if model is None:
            raise RuntimeError("SAM3 is not available; original YOLO boxes were kept.")
        bounds = [self._bbox_xyxy_from_box_obj(output[index], width, height) for index in eligible]
        prompts = [self.pad_bbox_xyxy(box, width, height, pad_ratio=.05) for box in bounds]
        # All prompts share one image encoding. SAM's class field identifies
        # the PROMPT, not a dataset class; masks may be filtered or reordered.
        with torch.inference_mode():
            results = model.predict(source=image, bboxes=prompts, conf=0.0,
                                    **self._sam_predict_kwargs())
        result = results[0] if results else None
        masks = getattr(getattr(result, "masks", None), "data", None)
        boxes = getattr(result, "boxes", None)

        def cpu_array(value):
            return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)

        if masks is None or boxes is None:
            stats["kept"] += len(eligible)
            return output
        ids, scores = cpu_array(boxes.cls).reshape(-1), cpu_array(boxes.conf).reshape(-1)
        if (len(ids) != len(masks) or len(scores) != len(masks)
                or not np.isfinite(ids).all() or not np.isfinite(scores).all()
                or any(value != int(value) or not 0 <= int(value) < len(eligible) for value in ids)
                or len(set(int(value) for value in ids)) != len(ids)):
            logger.warning("SAM3 prompt identities were ambiguous; original YOLO boxes were kept.")
            stats["kept"] += len(eligible)
            return output
        accepted = 0
        for mask_index, prompt_id in enumerate(ids):
            prompt_id = int(prompt_id)
            if scores[mask_index] < .70:
                continue
            index = eligible[prompt_id]
            original = output[index]
            old_bounds = bounds[prompt_id]
            mask = self._normalize_sam_mask_to_image(cpu_array(masks[mask_index]), width, height)
            class_name = self.get_current_class_name_safe(original.class_id)
            if not self.validate_sam_mask_for_bbox(mask, old_bounds, class_name=class_name):
                continue
            rect = self.mask_to_bbox_rect(mask)
            if rect is None:
                continue
            x, y, box_w, box_h = rect
            new_bounds = (x, y, x + box_w, y + box_h)
            old_w, old_h = old_bounds[2] - old_bounds[0], old_bounds[3] - old_bounds[1]
            ratio = box_w * box_h / max(1., old_w * old_h)
            movement = math.hypot((new_bounds[0] + new_bounds[2] - old_bounds[0] - old_bounds[2]) / 2,
                                  (new_bounds[1] + new_bounds[3] - old_bounds[1] - old_bounds[3]) / 2)
            if (not .5 <= ratio <= 2. or movement > .20 * max(1., math.hypot(old_w, old_h))
                    or self._propagation_iou(old_bounds, new_bounds) < .50):
                continue
            xc, yc, bw, bh = self._xyxy_to_yolo_xywh(new_bounds, width, height)
            output[index] = BoundingBox(original.class_id, xc, yc, bw, bh,
                                        confidence=original.confidence)
            accepted += 1
        stats["refined"] += accepted
        stats["kept"] += len(eligible) - accepted
    except Exception as error:
        # Disable only this optional assist for the rest of the run. The
        # normal YOLO result, classes and output formats remain intact.
        self._yolo_sam3_unavailable = str(error)
        logger.warning("SAM3 auto-label assist unavailable: %s", error)
    return output


def _yolo_gap_record(self, image_path, label_path, boxes, image_size, roi_metadata):
    return dict(image_path=image_path, label_path=label_path,
                boxes=[box for box in boxes if self._label_type(box) == "bbox"],
                image_size=tuple(image_size), roi_metadata=dict(roi_metadata),
                image_version=label_file_version(image_path),
                label_version=label_file_version(label_path))


@staticmethod
def _yolo_gap_record_is_current(record):
    try:
        return (os.path.isfile(record["image_path"])
                and label_file_version(record["image_path"]) == record["image_version"]
                and label_file_version(record["label_path"]) == record["label_version"])
    except OSError:
        return False


def _finish_yolo_sam3_assist(self, records):
    """Fill verified short gaps without changing any existing annotation row."""
    stats = getattr(self, "_yolo_sam3_stats", {})
    summary = f"\nSAM3: {stats.get('refined', 0)} box(es) refined."
    if getattr(self, "_yolo_sam3_unavailable", ""):
        return summary + " Assist unavailable; original YOLO boxes were kept where needed."
    if getattr(self, "stop_labeling", False):
        return summary + " Gap filling stopped."
    candidates = find_yolo_box_gaps(records)
    if not candidates:
        return summary + " No confirmed short gaps to fill."
    if getattr(self, "_sam3_propagation_batch_active", False) or any(
        self.qthread_is_running(worker) for worker in (
            getattr(self, "_sam3_propagation_warmup_worker", None),
            getattr(self, "_adjacent_propagation_worker", None),
        )
    ):
        return summary + " Gap filling skipped because the propagation tracker is busy."

    status_label = QLabel(self)
    proposals = []
    self._propagation_stop_requested = False
    self._last_propagation_stop_reasons = []
    self._sam3_propagation_batch_active = True
    self._active_propagation_preprocessing_state = self._preprocessing_state_snapshot()
    self._active_propagation_polygon_preference = None  # This assist is boxes-only.
    self._active_propagation_class_names = tuple(self.class_labels)
    dialog_parent = getattr(self, "auto_label_dialog", None)
    if dialog_parent is None or not dialog_parent.isVisible():
        dialog_parent = self
    try:
        run_propagation_worker(
            self,
            lambda status: compute_yolo_box_gap_proposals(
                self, records, candidates, status,
                self._active_propagation_preprocessing_state, proposals),
            status_label, len(candidates), self.stop_label_propagation,
            lambda: bool(getattr(self, "_propagation_stop_requested", False)),
            unit="short gaps",
            dialog_parent=dialog_parent, window_title="SAM3 Auto Label Assist",
        )
    finally:
        self._active_propagation_preprocessing_state = None
        self._active_propagation_polygon_preference = None
        self._active_propagation_class_names = None
        self._sam3_propagation_batch_active = False
        self._sam3_propagation_release_pending = True
        self._release_sam_video_propagation_runtime()
        status_label.deleteLater()
    if self._propagation_stop_requested or self._last_propagation_stop_reasons:
        return summary + " Gap filling stopped; no pending additions were saved."

    by_frame = {}
    for proposal in proposals:
        by_frame.setdefault(proposal["index"], []).append(proposal)
    manifest = None
    added, failures = 0, 0
    try:
        for index, frame_proposals in sorted(by_frame.items()):
            record = records[index]
            involved = {index} | {item[key] for item in frame_proposals for key in ("start", "end")}
            if not all(self._yolo_gap_record_is_current(records[item]) for item in involved):
                continue  # A deletion or newer manual correction wins.
            image = self._read_image_cv(record["image_path"])
            if image is None:
                continue
            boxes = [item["box"] for item in frame_proposals]
            size = record["image_size"]
            boxes = self.remap_roi_bboxes_to_full_frame(boxes, record["roi_metadata"], size, size)
            boxes = self.filter_bboxes_to_active_classes(boxes)
            boxes = self.filter_bboxes_to_size_limits(boxes, *size, source="SAM3 YOLO gap fill")
            boxes = self.filter_teammate_auto_labels(image, boxes, class_names=self.class_labels,
                                                     source="SAM3 YOLO gap fill")
            existing = self.load_label_lines(record["label_path"])
            # Include all saved formats/classes in the duplicate check. A
            # tracking proposal must not relabel an existing object.
            existing_objects = [BoundingBox.from_str(line) for line in existing]
            unique = []
            for box in boxes:
                new_bounds = self._label_seed_bbox(box, *size)
                old_bounds = [self._label_seed_bbox(old, *size)
                              for old in existing_objects + unique if old is not None]
                if not any(bounds is not None and self._propagation_iou(new_bounds, bounds) >= .15
                           for bounds in old_bounds):
                    unique.append(box)
            if not unique or not all(self._yolo_gap_record_is_current(records[item]) for item in involved):
                continue
            if manifest is None:
                manifest = self._begin_propagation_batch(self.image_directory, "yolo_sam3_gap_fill")
            if save_propagation_outputs(self, record, image,
                                        existing + [box.to_str() for box in unique], manifest):
                added += len(unique)
                record["label_version"] = label_file_version(record["label_path"])
            else:
                failures += 1
    except Exception as error:
        failures += 1
        logger.warning("SAM3 YOLO gap fill stopped: %s", error, exc_info=True)
    finally:
        if manifest is not None:
            self._finish_propagation_batch(manifest)
    return (summary + f" {added} gap box(es) added."
            + (f" {failures} save failure(s); check the log." if failures else "")
            + (" Undo Last Batch can remove the gap additions." if added else ""))


def class_supports_keypoints(self, class_id):
    try:
        if self.keypoint_list.rowCount() > 0:
            return True
    except Exception:
        pass

    for directory in (
        getattr(self, "image_directory", None),
        os.path.dirname(getattr(self, "current_file", "") or ""),
        getattr(self, "output_path", None),
    ):
        if not directory:
            continue

        for file_name in ("points.json", "pose.json"):
            path = resolve_dataset_metadata_path(directory, file_name, migrate=True)
            if not os.path.exists(path):
                continue

            data = read_json_file(path, {})
            if not isinstance(data, dict):
                continue

            keypoints = data.get("keypoints")
            if isinstance(keypoints, list) and keypoints:
                return True

            kpt_shape = data.get("kpt_shape")
            if isinstance(kpt_shape, list) and kpt_shape:
                try:
                    if int(kpt_shape[0]) > 0:
                        return True
                except Exception:
                    pass

    try:
        self.sync_class_labels_from_dropdown()
        return self.class_labels[int(class_id)].lower() in {"person", "human", "player", "hand", "hands"}
    except Exception:
        return False


def extract_bboxes_from_result(
    self,
    result,
    img_width,
    img_height,
    image_file,
    mp_pose,
    num_expected_keypoints=0,
    conf_threshold=None,
    inference_context=None,
):
    new_bboxes = []
    annotation_mode = inference_context["annotation_mode"] if inference_context else None
    if inference_context and conf_threshold is None:
        conf_threshold = inference_context["confidence"]

    def map_class_id(model_id):
        if inference_context:
            try:
                return inference_context["class_map"].get(int(model_id))
            except (TypeError, ValueError, OverflowError):
                return None
        return self.map_model_class_id_to_local_class_id(model_id)

    if conf_threshold is None:
        try:
            conf_threshold = float(self.confidence_threshold_spinbox.value())
        except Exception:
            conf_threshold = 0.35
    else:
        try:
            conf_threshold = float(conf_threshold)
        except Exception:
            conf_threshold = 0.35

    if result is None:
        return new_bboxes

    def _to_numpy_attr(obj, attr_name):
        try:
            value = getattr(obj, attr_name, None)
            if value is None:
                return []
            if hasattr(value, "cpu"):
                return value.cpu().numpy()
            return np.asarray(value)
        except Exception:
            return []

    def _resolve_expected_keypoint_count():
        count = 0

        try:
            if num_expected_keypoints and int(num_expected_keypoints) > 0:
                count = int(num_expected_keypoints)
        except Exception:
            count = 0

        if count <= 0 and not inference_context:
            try:
                count = self.keypoint_list.rowCount()
            except Exception:
                count = 0

        if count > 0:
            return count

        candidate_files = []

        try:
            if image_file:
                candidate_files.append(resolve_dataset_pose_schema_path(
                    os.path.dirname(image_file), migrate=True
                ))
        except Exception:
            pass

        try:
            if getattr(self, "image_directory", None):
                candidate_files.append(resolve_dataset_pose_schema_path(
                    self.image_directory, migrate=True
                ))
        except Exception:
            pass

        try:
            if getattr(self, "output_path", None):
                candidate_files.append(resolve_dataset_pose_schema_path(
                    self.output_path, migrate=True
                ))
        except Exception:
            pass

        seen = set()

        for points_file in candidate_files:
            if not points_file or points_file in seen:
                continue

            seen.add(points_file)

            if os.path.exists(points_file):
                try:
                    with open(points_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if isinstance(data, dict) and isinstance(data.get("keypoints"), list):
                        file_count = len(data["keypoints"])
                        if file_count > 0:
                            logging.info(f"Using {file_count} keypoints from {points_file}")
                            return file_count

                except Exception as e:
                    logging.warning(f"Failed to load {points_file}: {e}")

        return max(count, 0)

    def _normalize_box_xyxy(box):
        if box is None or len(box) < 4:
            return None

        if img_width <= 0 or img_height <= 0:
            return None

        x1, y1, x2, y2 = map(float, box[:4])
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            return None

        x1 = float(np.clip(x1, 0.0, img_width))
        y1 = float(np.clip(y1, 0.0, img_height))
        x2 = float(np.clip(x2, 0.0, img_width))
        y2 = float(np.clip(y2, 0.0, img_height))

        if x2 <= x1 or y2 <= y1:
            return None

        xc = float(np.clip((x1 + x2) / 2.0 / img_width, 0.0, 1.0))
        yc = float(np.clip((y1 + y2) / 2.0 / img_height, 0.0, 1.0))
        w = float(np.clip((x2 - x1) / img_width, 0.0, 1.0))
        h = float(np.clip((y2 - y1) / img_height, 0.0, 1.0))

        if w <= 0.0 or h <= 0.0:
            return None

        return xc, yc, w, h

    def _bbox_bounds_from_normalized(xc, yc, w, h):
        x_min = max(0.0, xc - (w / 2.0))
        y_min = max(0.0, yc - (h / 2.0))
        x_max = min(1.0, xc + (w / 2.0))
        y_max = min(1.0, yc + (h / 2.0))
        return x_min, y_min, x_max, y_max

    def _point_inside_bbox(nx, ny, bounds, tolerance=0.015):
        x_min, y_min, x_max, y_max = bounds
        return (
            (x_min - tolerance) <= nx <= (x_max + tolerance) and
            (y_min - tolerance) <= ny <= (y_max + tolerance)
        )

    boxes_obj = getattr(result, "boxes", None)
    keypoints_obj = getattr(result, "keypoints", None)
    masks_obj = getattr(result, "masks", None)
    obb_obj = getattr(result, "obb", None)

    has_boxes = boxes_obj is not None
    has_keypoints = keypoints_obj is not None and boxes_obj is not None
    has_masks = masks_obj is not None and boxes_obj is not None
    has_obb = obb_obj is not None

    if (annotation_mode == "segmentation" if inference_context else self.is_segmentation_mode()):
        if not has_masks:
            return new_bboxes

        masks = getattr(masks_obj, "xy", None)
        class_ids = _to_numpy_attr(boxes_obj, "cls")
        confidences = _to_numpy_attr(boxes_obj, "conf")

        if masks is None:
            return new_bboxes

        for mask, class_id, conf in zip(masks, class_ids, confidences):
            try:
                conf = float(conf)
            except Exception:
                continue

            if conf < conf_threshold:
                continue

            local_class_id = map_class_id(class_id)
            if local_class_id is None:
                continue

            try:
                mask = np.array(mask, dtype=np.float32).reshape(-1, 2)
                if mask.size == 0:
                    continue

                mask[:, 0] /= img_width
                mask[:, 1] /= img_height
                mask[:, 0] = np.clip(mask[:, 0], 0.0, 1.0)
                mask[:, 1] = np.clip(mask[:, 1], 0.0, 1.0)

                new_bboxes.append(
                    BoundingBox(
                        int(local_class_id),
                        0,
                        0,
                        0,
                        0,
                        confidence=conf,
                        segmentation=mask.flatten().tolist()
                    )
                )

            except Exception as e:
                logging.warning(f"Failed to process segmentation result for {image_file}: {e}")

        return new_bboxes

    if (annotation_mode == "obb" if inference_context else self.is_obb_mode()):
        if not has_obb:
            return new_bboxes

        xywhr = _to_numpy_attr(obb_obj, "xywhr")
        class_ids = _to_numpy_attr(obb_obj, "cls")
        confidences = _to_numpy_attr(obb_obj, "conf")

        for obb, class_id, conf in zip(xywhr, class_ids, confidences):
            try:
                conf = float(conf)
            except Exception:
                continue

            if conf < conf_threshold or obb is None or len(obb) != 5:
                continue

            local_class_id = map_class_id(class_id)
            if local_class_id is None:
                continue

            try:
                x_center, y_center, width, height, angle = map(float, obb[:5])
                corners = self.obb_xywhr_to_corners(x_center, y_center, width, height, angle)
                corners = np.array(corners, dtype=np.float32).reshape(-1, 2)

                if corners.size == 0:
                    continue

                corners[:, 0] /= img_width
                corners[:, 1] /= img_height
                corners[:, 0] = np.clip(corners[:, 0], 0.0, 1.0)
                corners[:, 1] = np.clip(corners[:, 1], 0.0, 1.0)

                new_bboxes.append(
                    BoundingBox(
                        int(local_class_id),
                        0,
                        0,
                        0,
                        0,
                        confidence=conf,
                        obb=corners.flatten().tolist()
                    )
                )

            except Exception as e:
                logging.warning(f"Failed to process OBB result for {image_file}: {e}")

        return new_bboxes

    if (annotation_mode == "keypoints" if inference_context else self.is_keypoint_mode()):
        expected_kpt_count = _resolve_expected_keypoint_count()

        if has_keypoints:
            boxes = _to_numpy_attr(boxes_obj, "xyxy")
            confidences = _to_numpy_attr(boxes_obj, "conf")
            class_ids = _to_numpy_attr(boxes_obj, "cls")
            keypoints_xy = _to_numpy_attr(keypoints_obj, "xy")
            keypoints_conf = _to_numpy_attr(keypoints_obj, "conf")

            if expected_kpt_count <= 0:
                try:
                    if len(keypoints_xy) > 0:
                        expected_kpt_count = len(keypoints_xy[0])
                except Exception:
                    expected_kpt_count = 0

            for det_index, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                try:
                    conf = float(conf)
                except Exception:
                    continue

                if not math.isfinite(conf) or conf < conf_threshold:
                    continue

                local_class_id = map_class_id(class_id)
                if local_class_id is None:
                    continue

                try:
                    norm_box = _normalize_box_xyxy(box)
                    if norm_box is None:
                        continue

                    xc, yc, w, h = norm_box
                    bounds = _bbox_bounds_from_normalized(xc, yc, w, h)

                    final_kpts = [(0.0, 0.0, 0)] * expected_kpt_count
                    try:
                        yolo_kpts = keypoints_xy[det_index] if len(keypoints_xy) > det_index else []
                    except (TypeError, IndexError):
                        yolo_kpts = []

                    det_kpt_conf = []
                    try:
                        if len(keypoints_conf) > det_index:
                            det_kpt_conf = keypoints_conf[det_index]
                    except Exception:
                        det_kpt_conf = []

                    for idx, kp in enumerate(yolo_kpts):
                        if idx >= expected_kpt_count:
                            break

                        if kp is None or len(kp) < 2:
                            final_kpts[idx] = (0.0, 0.0, 0)
                            continue

                        try:
                            kp_x = float(kp[0])
                            kp_y = float(kp[1])
                        except Exception:
                            final_kpts[idx] = (0.0, 0.0, 0)
                            continue

                        try:
                            kp_conf = float(det_kpt_conf[idx]) if len(det_kpt_conf) > idx else 1.0
                        except Exception:
                            kp_conf = 1.0

                        if not all(math.isfinite(value) for value in (kp_x, kp_y, kp_conf)):
                            final_kpts[idx] = (0.0, 0.0, 0)
                            continue

                        if kp_x <= 0.0 and kp_y <= 0.0:
                            final_kpts[idx] = (0.0, 0.0, 0)
                            continue

                        nx = float(np.clip(kp_x / img_width, 0.0, 1.0))
                        ny = float(np.clip(kp_y / img_height, 0.0, 1.0))

                        if nx <= 0.0 and ny <= 0.0:
                            final_kpts[idx] = (0.0, 0.0, 0)
                            continue

                        if kp_conf <= 0.01:
                            final_kpts[idx] = (0.0, 0.0, 0)
                            continue

                        if not _point_inside_bbox(nx, ny, bounds):
                            final_kpts[idx] = (0.0, 0.0, 0)
                            continue

                        final_kpts[idx] = (nx, ny, 2)

                    new_bboxes.append(
                        BoundingBox(
                            int(local_class_id),
                            xc,
                            yc,
                            w,
                            h,
                            confidence=conf,
                            keypoints=final_kpts
                        )
                    )

                except Exception as e:
                    logging.warning(f"Failed to process keypoint result for {image_file}: {e}")

            return new_bboxes

        if not has_boxes:
            return new_bboxes

        boxes = _to_numpy_attr(boxes_obj, "xyxy")
        confidences = _to_numpy_attr(boxes_obj, "conf")
        class_ids = _to_numpy_attr(boxes_obj, "cls")

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            try:
                conf = float(conf)
            except Exception:
                continue

            if not math.isfinite(conf) or conf < conf_threshold:
                continue

            local_class_id = map_class_id(class_id)
            if local_class_id is None:
                continue

            try:
                norm_box = _normalize_box_xyxy(box)
                if norm_box is None:
                    continue

                xc, yc, w, h = norm_box
                ghost_kpts = [(0.0, 0.0, 0)] * expected_kpt_count

                new_bboxes.append(
                    BoundingBox(
                        int(local_class_id),
                        xc,
                        yc,
                        w,
                        h,
                        confidence=conf,
                        keypoints=ghost_kpts
                    )
                )

            except Exception as e:
                logging.warning(f"Failed to process bbox fallback in keypoint mode for {image_file}: {e}")

        return new_bboxes

    if (annotation_mode == "boxes" if inference_context else self.is_boxes_mode()):
        if not has_boxes:
            return new_bboxes

        boxes = _to_numpy_attr(boxes_obj, "xyxy")
        confidences = _to_numpy_attr(boxes_obj, "conf")
        class_ids = _to_numpy_attr(boxes_obj, "cls")

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            try:
                conf = float(conf)
            except Exception:
                continue

            if conf < conf_threshold:
                continue

            local_class_id = map_class_id(class_id)
            if local_class_id is None:
                continue

            try:
                norm_box = _normalize_box_xyxy(box)
                if norm_box is None:
                    continue

                xc, yc, w, h = norm_box

                new_bboxes.append(
                    BoundingBox(
                        int(local_class_id),
                        xc,
                        yc,
                        w,
                        h,
                        confidence=conf
                    )
                )

            except Exception as e:
                logging.warning(f"Failed to process bbox result for {image_file}: {e}")

        return new_bboxes

    if has_boxes:
        boxes = _to_numpy_attr(boxes_obj, "xyxy")
        confidences = _to_numpy_attr(boxes_obj, "conf")
        class_ids = _to_numpy_attr(boxes_obj, "cls")

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            try:
                conf = float(conf)
            except Exception:
                continue

            if conf < conf_threshold:
                continue

            local_class_id = map_class_id(class_id)
            if local_class_id is None:
                continue

            try:
                norm_box = _normalize_box_xyxy(box)
                if norm_box is None:
                    continue

                xc, yc, w, h = norm_box

                new_bboxes.append(
                    BoundingBox(
                        int(local_class_id),
                        xc,
                        yc,
                        w,
                        h,
                        confidence=conf
                    )
                )

            except Exception as e:
                logging.warning(f"Failed to process default bbox result for {image_file}: {e}")

    return new_bboxes


def load_existing_bboxes(self, label_file):
    try:
        expected = self.keypoint_list.rowCount()
    except Exception:
        expected = 0

    if expected <= 0:
        try:
            expected = self.get_expected_keypoint_count_for_dedupe(os.path.dirname(label_file))
        except Exception:
            expected = 0

    BoundingBox.EXPECTED_KEYPOINT_COUNT = expected or None
    polygon_parse_preference = self._polygon_label_parse_preference()

    with open(label_file, 'r') as f:
        bboxes = []
        for line in f:
            bbox = BoundingBox.from_str(
                line.strip(),
                preferred_polygon_type=polygon_parse_preference,
            )
            if bbox is not None:
                if self.is_keypoint_mode() and self.class_supports_keypoints(bbox.class_id) and len(bbox.keypoints) < expected:
                    bbox.keypoints = list(bbox.keypoints) + [(0.0, 0.0, 0)] * (expected - len(bbox.keypoints))
                bboxes.append(bbox)
            else:
                logging.warning(f"Malformed bbox line skipped: {line.strip()}")
        return bboxes


@staticmethod

def has_visible_keypoints(keypoints):
    return any(len(kp) >= 3 and kp[2] > 0 for kp in keypoints)


@staticmethod

def obb_xywhr_to_corners(x_center, y_center, width, height, angle_rad):
    """
        Converts an OBB from (x_center, y_center, width, height, angle in radians)
        to 4 corner points in clockwise order.
        """
    import numpy as np

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Rectangle corners around the origin
    half_w, half_h = width / 2, height / 2
    corners = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h]
    ])

    # Rotation matrix
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Rotate and shift
    rotated = np.dot(corners, R.T)
    shifted = rotated + np.array([x_center, y_center])

    return shifted.flatten().tolist()


def get_model_kwargs(self):
    """
        Builds model inference kwargs using checked classes from classes_dropdown.
        If none are checked, falls back to all classes like the old behavior.
        """

    conf_threshold = self.confidence_threshold_spinbox.value()
    iou_threshold = self.nms_threshold_spinbox.value()
    batch_size = self.batch_inference.value()
    use_fp16 = getattr(self, 'fp_mode', 0) == 1 and torch.cuda.is_available()
    device = 0 if torch.cuda.is_available() else "cpu"

    # Keep physical frame transforms separate from YOLO's network input size.
    # Resize/Crop changes the frame that is displayed or saved; network_width /
    # network_height are the explicit imgsz controls for Ultralytics inference.
    width, height = self.get_network_size_wh()

    dropdown_classes = self.get_classes_from_dropdown()

    if dropdown_classes:
        self.class_labels = list(dropdown_classes)
        self.class_names = list(dropdown_classes)
        self.id_to_class = {i: name for i, name in enumerate(dropdown_classes)}
    elif not hasattr(self, "class_labels") or not self.class_labels:
        self.class_labels = self.load_classes() or []
        self.class_names = list(self.class_labels)
        self.id_to_class = {i: name for i, name in enumerate(self.class_labels)}

    model_training_classes = self.load_model_classes()

    active_local_class_ids = self.get_active_class_ids_from_ui()

    if not active_local_class_ids:
        active_local_class_ids = set(range(len(self.class_labels)))

    # Only the mapping is cached: thresholds, precision and network size
    # above/below still come from the current UI on every call. Rebuild (and
    # report missing names once) when the model, names or selection changes.
    mapping_key = (
        id(getattr(self, "model", None)),
        str(getattr(self, "weights_file_path", "") or ""),
        tuple(self.class_labels),
        tuple(model_training_classes),
        tuple(sorted(active_local_class_ids)),
    )
    cached_mapping = getattr(self, "_inference_class_mapping_cache", None)
    if cached_mapping is None or cached_mapping[0] != mapping_key:
        class_indices = []
        for local_id in sorted(active_local_class_ids):
            if 0 <= local_id < len(self.class_labels):
                class_name = self.class_labels[local_id]
                model_idx = class_name_index(class_name, model_training_classes)
                if model_idx is not None and class_name_index(
                    model_training_classes[model_idx], self.class_labels
                ) == local_id:
                    class_indices.append(model_idx)
                else:
                    logging.warning(
                        f"Class '{class_name}' has no unambiguous project/model mapping."
                    )
        if not class_indices:
            logging.warning("No selected project classes match model classes.")
        cached_mapping = (mapping_key, tuple(class_indices))
        self._inference_class_mapping_cache = cached_mapping
    class_indices = list(cached_mapping[1])

    model_kwargs = {
        'conf': conf_threshold,
        'iou': iou_threshold,
        'imgsz': self.get_network_imgsz_hw(),
        'batch': batch_size,
        'device': device,
        'classes': class_indices,
        'quantize': 16 if use_fp16 else None,
        'agnostic_nms': True,
        'max_det': 100,
        'retina_masks': True,
        'verbose': False,
    }

    logging.debug(
        "Model kwargs for inference: %s (UI Network W=%s H=%s -> imgsz [H, W]=%s)",
        model_kwargs,
        width,
        height,
        model_kwargs["imgsz"],
    )
    return model_kwargs


def process_current_image(self):
    image_file = getattr(self, "current_file", None)

    if not image_file:
        logger.warning("No image file is currently opened.")
        return

    image = self.read_image(image_file, share_editor=True)
    if image is None:
        logger.warning(f"Failed to load current image: {image_file}")
        return

    self.infer_and_display_bounding_boxes(image)

    if getattr(self, "current_file", None) != image_file:
        logger.warning(
            f"Skipped save because current_file changed during processing: "
                f"{image_file} -> {self.current_file}"
        )
        return

    self.save_bounding_boxes(
        image_file,
        self.screen_view.scene().width(),
        self.screen_view.scene().height()
    )


def switch_floating_point_mode(self):
    current_text = self.fp_select_combobox.currentText()
    if current_text == "FP32":
        self.set_floating_point_mode(0)  # 0 for FP32
    elif current_text == "FP16":
        self.set_floating_point_mode(1)  # 1 for FP16


def set_floating_point_mode(self, mode):
    try:
        mode = int(mode)
    except Exception:
        mode = 0

    mode = 1 if mode == 1 and torch.cuda.is_available() else 0
    self.fp_mode = mode
    self.yolo_floating_point = mode

    if hasattr(self, "settings"):
        self.settings["fpMode"] = mode
        self.queue_settings_save()

    if hasattr(self, "net") and self.net is not None:
        self.apply_backend_and_target()


def apply_backend_and_target(self):
    """Use a supported DNN target, rather than general CUDA device presence."""
    if not hasattr(self, "net") or self.net is None:
        logger.warning("OpenCV DNN network is not initialized.")
        return

    try:
        cuda_targets = set(cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA))
    except cv2.error as error:
        logger.debug("OpenCV DNN CUDA target discovery failed: %s", error)
        cuda_targets = set()

    prefer_fp16 = bool(getattr(self, "yolo_floating_point", 0))
    cuda_target = (cv2.dnn.DNN_TARGET_CUDA_FP16 if prefer_fp16 and
                   cv2.dnn.DNN_TARGET_CUDA_FP16 in cuda_targets else cv2.dnn.DNN_TARGET_CUDA)
    if cuda_target in cuda_targets:
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cuda_target)
        precision = 'FP16' if cuda_target == cv2.dnn.DNN_TARGET_CUDA_FP16 else 'FP32'
        logger.info("Using OpenCV DNN CUDA target: %s", precision)
    elif (cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()
          and cv2.ocl.Device_getDefault().type() & cv2.ocl.Device_TYPE_GPU):
        configure_opencv_dnn_cache()
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        logger.info("Using OpenCV DNN OpenCL FP32 target: %s", cv2.ocl.Device_getDefault().name())
    else:
        logger.info("Using OpenCV DNN CPU target")

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def get_dnn_output_layer_names(self):
    if hasattr(self.net, "getUnconnectedOutLayersNames"):
        return list(self.net.getUnconnectedOutLayersNames())

    layer_names = self.net.getLayerNames()
    indices = np.array(self.net.getUnconnectedOutLayers()).flatten()
    return [layer_names[int(i) - 1] for i in indices]


def initialize_yolo(self):
    if not gpu_model_allowed(self, "detector"):
        return False
    try:
        self.classes = []

        if not getattr(self, 'weights_file_path', None) or not getattr(self, 'cfg_file_path', None):
            logger.warning("Weights and CFG files must both be selected for Darknet models.")
            return False

        file_extension = os.path.splitext(self.weights_file_path)[1].lower()
        if file_extension != '.weights':
            logger.warning("Unsupported OpenCV DNN file extension. Select a .weights file.")
            return False

        if not os.path.exists(self.weights_file_path):
            logger.warning(f"Weights file does not exist: {self.weights_file_path}")
            return False

        if not os.path.exists(self.cfg_file_path):
            logger.warning(f"CFG file does not exist: {self.cfg_file_path}")
            return False

        cfg_path, weights_path = self.cfg_file_path, self.weights_file_path
        self.net = _construct_model_without_blocking_gui(
            self, lambda: cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        )

        self.yolo_floating_point = 1 if int(getattr(self, "fp_mode", 0) or 0) == 1 else 0

        self.apply_backend_and_target()
        self.layer_names = self.get_dnn_output_layer_names()
        # OpenCV's Darknet Region layer already multiplies each class score
        # by objectness. Raw YOLO tensors use a different score contract.
        self._opencv_scores_include_objectness = bool(self.layer_names) and all(
            self.net.getLayer(self.net.getLayerId(name)).type == "Region"
            for name in self.layer_names
        )
        self.class_labels = self.load_classes() or []
        self.class_names = list(self.class_labels)
        self.id_to_class = {i: name for i, name in enumerate(self.class_labels)}

        if not self.class_labels:
            logger.warning("classes.txt not found or empty. Select an image directory with classes.txt.")
            return False

        logger.info(
            f"Initialized OpenCV DNN model with {len(self.layer_names)} output layer(s) "
                f"and {len(self.class_labels)} classes."
        )
        return True
    except Exception as e:
        logger.error(f"Error initializing YOLO: {e}")
        self.net = None
        return False


@gpu_work("Auto labeling", keep=auto_label_resources)
def auto_label_images(self, overwrite=None):
    """
        Auto-label images using YOLO model with .weights and .cfg files.
        Includes overwrite, append, and cancel functionality.
        """
    if getattr(self, "_gpu_released_detector", False):
        ensure_gpu_detector(self)
    if getattr(self, "model_type", "") == 'weights' and (
        not getattr(self, 'weights_file_path', None) or not getattr(self, 'cfg_file_path', None)
    ):
        QMessageBox.warning(self, "Missing Files", "Both .weights and .cfg files are required for this model type.")
        return

    self.class_labels = self.load_classes() or []
    if not self.class_labels:
        QMessageBox.critical(self, "Error", "No classes available. Select an image folder with classes.txt.")
        return

    if not getattr(self, "net", None) and not self.initialize_yolo():
        QMessageBox.critical(self, "Error", "OpenCV DNN model could not be initialized.")
        return

    source_images = [
        image_file for image_file in self.image_files
        if not self.is_placeholder_file(image_file)
    ]
    total_images = len(source_images)
    self.set_label_progress(
        0, total_images, value=0, progress_format="Auto Label %v/%m"
    )

    if total_images == 0:
        QMessageBox.information(self, "Information", "No images available for auto-labeling.")
        return

    if overwrite is None:
        overwrite_reply = QMessageBox.question(
            self, 'Overwrite Labels',
            "Do you want to overwrite existing labels?\n\n"
                "- Yes: Overwrite existing label files\n"
                "- No: Append new labels to existing\n"
                "- Cancel: Do nothing",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No
        )

        if overwrite_reply == QMessageBox.Cancel:
            logging.info("Auto-labeling canceled by user.")
            return

        overwrite = overwrite_reply == QMessageBox.Yes
    else:
        overwrite = bool(overwrite)

    logging.info(f"OpenCV auto-label overwrite existing labels: {overwrite}")
    self.stop_labeling = False
    self.reset_prediction_size_filter_stats("OpenCV image auto-label")
    sam3_assist = bool(getattr(self, "auto_label_sam3_assist", False))
    gap_frames = [None] * total_images if sam3_assist else None
    if sam3_assist:
        self._yolo_sam3_stats = {"refined": 0, "kept": 0}
        self._yolo_sam3_unavailable = ""

    roi_enabled = self.settings.get('videoUseROIForInference', False)
    roi_bounds = self.get_active_rois_for_inference() if roi_enabled else None

    for idx, image_file in enumerate(source_images):
        if self.stop_labeling:
            logging.info("Labeling stopped by user.")
            break

        self.current_file = image_file
        label_file = self.get_label_file(image_file)

        image = self.read_image(image_file)
        if image is None:
            logging.error(f"Failed to load image {image_file}. Skipping.")
            continue

        try:
            new_bboxes = self.run_opencv_dnn_inference(image)
        except Exception as e:
            logger.error(f"OpenCV DNN inference failed for {image_file}: {e}")
            self.set_label_progress(0, total_images, value=idx, progress_format="Auto Label stopped: inference failed")
            QMessageBox.critical(self, "Auto Label stopped", f"OpenCV inference failed: {e}\nExisting labels for unprocessed images were kept.")
            return

        img_h, img_w = image.shape[:2]
        new_bboxes = self.filter_bboxes_to_active_classes(new_bboxes)
        if sam3_assist:
            new_bboxes = self._refine_yolo_boxes_with_sam3(image, new_bboxes)
        roi_metadata = {"x": 0, "y": 0}
        if roi_bounds is not None:
            _full_image, roi_metadata = self.extract_roi_region_for_inference(
                image, roi_bounds
            )
        new_bboxes = self.remap_roi_bboxes_to_full_frame(
            new_bboxes,
            roi_metadata,
            (img_w, img_h),
            (img_w, img_h),
        )
        new_bboxes = self.filter_bboxes_to_size_limits(
            new_bboxes,
            img_w,
            img_h,
            source=os.path.basename(image_file),
        )
        new_bboxes = self.filter_teammate_auto_labels(
            image,
            new_bboxes,
            class_names=self.class_labels,
            source="OpenCV image auto label",
        )

        if overwrite:
            final_bboxes = self.remove_near_duplicate_bounding_boxes(
                new_bboxes,
                iou_threshold=0.5,
                class_aware=True,
            )
        else:
            existing_bboxes = []
            if label_file and os.path.exists(label_file):
                existing_bboxes = self.load_existing_bboxes(label_file)

            final_bboxes = self.remove_near_duplicate_bounding_boxes(
                existing_bboxes + new_bboxes,
                iou_threshold=0.5,
                class_aware=True,
            )

        labels = [bbox.to_str(remove_confidence=True) for bbox in final_bboxes]

        if label_file:
            saved = self.save_labels_to_file(
                label_file,
                labels,
                mode="w",
                allow_empty=overwrite,
            )
            if sam3_assist and (saved or (not labels and not overwrite)):
                gap_frames[idx] = self._yolo_gap_record(
                    image_file, label_file, new_bboxes, (img_w, img_h), roi_metadata
                )

        if idx % 3 == 0:
            self.display_image(image_file, rebuild_preview=False)

        self.label_progress.setValue(idx + 1)
        QApplication.processEvents()

    assist_summary = self._finish_yolo_sam3_assist(gap_frames) if sam3_assist else ""
    self.set_label_progress(
        0, total_images, value=total_images, progress_format="Auto Label complete %v/%m"
    )
    self.log_prediction_size_filter_summary("OpenCV image auto-label")
    QMessageBox.information(self, "Auto-Labeling", "Finished auto-labeling all images." + assist_summary)
    self.stop_labeling = False
    self.img_index_number_changed(0)


def auto_label_yolo_button_clicked(self, overwrite=None) -> bool:
    logging.info("auto_label_yolo_button_clicked called")

    if not hasattr(self, 'image_directory') or self.image_directory is None:
        logging.error("Image directory not selected.")
        return False

    if not hasattr(self, 'weights_file_path') or self.weights_file_path is None:
        logging.error("Weights file path not selected.")
        return False

    weights_ext = os.path.splitext(str(self.weights_file_path))[1].lower()

    if weights_ext == '.weights' and (not hasattr(self, 'cfg_file_path') or self.cfg_file_path is None):
        logging.error("CFG file path not selected for .weights model.")
        return False

    self.class_labels = self.load_classes()  # Use the centralized function
    if not self.class_labels:
        logging.error("Classes file not found or empty.")
        return False

    if weights_ext in ('.pt', '.engine', '.onnx'):
        logging.info(
            "%s model detected. Running shared auto-label flow with %s.",
            weights_ext,
            self.active_inference_backend_label(),
        )
        self.auto_label_images2(overwrite=overwrite)
    elif weights_ext == '.weights':
        logging.info("Darknet model detected. Running auto_label_images.")
        self.auto_label_images(overwrite=overwrite)
    return True


def infer_and_display_bounding_boxes(self, image) -> List[List[int]]:
    if not getattr(self, 'net', None):
        logger.warning("Network not initialized. Please select weights file and cfg file.")
        return []

    bboxes = self.run_opencv_dnn_inference(image)
    img_h, img_w = image.shape[:2]
    bboxes = self.filter_bboxes_to_size_limits(
        bboxes,
        img_w,
        img_h,
        source="DNN preview",
        record=False,
    )
    self.display_dnn_bounding_boxes(bboxes, image.shape)

    return [
        [
            int((bbox.x_center - bbox.width / 2) * img_w),
            int((bbox.y_center - bbox.height / 2) * img_h),
            int(bbox.width * img_w),
            int(bbox.height * img_h),
        ]
        for bbox in bboxes
    ]


def _prediction_size_stats_lock(self):
    lock = getattr(self, "_prediction_size_filter_stats_lock", None)
    if lock is None:
        lock = threading.Lock()
        self._prediction_size_filter_stats_lock = lock
    return lock


def reset_prediction_size_filter_stats(self, context=""):
    lock = self._prediction_size_stats_lock()
    with lock:
        self._prediction_size_filter_stats = {
            "context": str(context or ""),
            "skipped": 0,
            "sources": Counter(),
        }


def record_prediction_size_filter_skip(self, source, skipped):
    try:
        skipped = int(skipped or 0)
    except Exception:
        skipped = 0
    if skipped <= 0:
        return

    lock = self._prediction_size_stats_lock()
    with lock:
        stats = getattr(self, "_prediction_size_filter_stats", None)
        if not isinstance(stats, dict):
            stats = {"context": "", "skipped": 0, "sources": Counter()}
            self._prediction_size_filter_stats = stats
        stats["skipped"] = int(stats.get("skipped", 0) or 0) + skipped
        sources = stats.get("sources")
        if not isinstance(sources, Counter):
            sources = Counter()
            stats["sources"] = sources
        sources[str(source or "predictions")] += skipped


def log_prediction_size_filter_summary(self, context=None, reset=True):
    lock = self._prediction_size_stats_lock()
    with lock:
        stats = dict(getattr(self, "_prediction_size_filter_stats", {}) or {})
        sources = stats.get("sources")
        if isinstance(sources, Counter):
            sources = Counter(sources)
        else:
            sources = Counter()
        if reset:
            self._prediction_size_filter_stats = {
                "context": "",
                "skipped": 0,
                "sources": Counter(),
            }

    skipped = int(stats.get("skipped", 0) or 0)
    if skipped <= 0:
        return

    context = str(context or stats.get("context") or "prediction filtering")
    source_bits = [
        f"{name}: {count}"
        for name, count in sources.most_common(4)
        if count
    ]
    source_text = f" ({', '.join(source_bits)})" if source_bits else ""
    logger.info(
        "Skipped %s prediction(s) outside annotation size limits during %s%s.",
        skipped,
        context,
        source_text,
    )


def prediction_size_allowed(self, prediction, img_width, img_height):
    min_size_px, max_percent = self.current_prediction_size_filter_values()
    return shared_prediction_size_allowed(
        prediction,
        img_width,
        img_height,
        min_size_px=min_size_px,
        max_percent=max_percent,
    )


def filter_predictions_to_size_limits(
    self,
    predictions,
    img_width,
    img_height,
    source="predictions",
    record=True,
    size_limits=None,
):
    if not predictions:
        return predictions

    min_size_px, max_percent = (size_limits if size_limits is not None
                                else self.current_prediction_size_filter_values())
    kept, skipped = shared_filter_predictions_to_size_limits(
        predictions,
        img_width,
        img_height,
        min_size_px=min_size_px,
        max_percent=max_percent,
    )
    if record and skipped:
        self.record_prediction_size_filter_skip(source, skipped)
    return kept


def filter_bboxes_to_size_limits(self, bboxes, img_width, img_height, source="labels", record=True,
                                size_limits=None):
    return self.filter_predictions_to_size_limits(
        bboxes,
        img_width,
        img_height,
        source=source,
        record=record,
        **({"size_limits": size_limits} if size_limits is not None else {}),
    )


def filter_video_overlay_to_size_limits(self, overlay, source="video overlay", record=False):
    if not isinstance(overlay, dict):
        return overlay

    frame_shape = overlay.get("frame_shape")
    if not frame_shape or len(frame_shape) < 2:
        return overlay

    try:
        img_h, img_w = int(frame_shape[0]), int(frame_shape[1])
    except Exception:
        return overlay

    min_size_px, max_percent = self.current_prediction_size_filter_values()
    signature = (img_w, img_h, float(min_size_px), float(max_percent))
    if not record and overlay.get("_size_filter_signature") == signature:
        return overlay
    filtered, skipped = shared_filter_overlay_to_size_limits(
        overlay,
        img_width=img_w,
        img_height=img_h,
        min_size_px=min_size_px,
        max_percent=max_percent,
    )
    filtered["_size_filter_signature"] = signature
    if record and skipped:
        self.record_prediction_size_filter_skip(source, skipped)
    return filtered


def get_dnn_size_limits(self, img_width, img_height):
    return self.get_annotation_size_limits(img_width, img_height)


def run_opencv_dnn_inference(self, image, inference_context=None) -> List[BoundingBox]:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return []

    if getattr(self, "net", None) is None:
        if inference_context or not self.initialize_yolo():
            return []

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        logger.error(f"Unsupported image shape for OpenCV DNN inference: {image.shape}")
        return []

    img_h, img_w = image.shape[:2]
    net_w, net_h = (inference_context["network_size"] if inference_context
                    else self.get_network_size_wh())

    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1 / 255.0,
        size=(net_w, net_h),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )

    self.net.setInput(blob)
    output_layer_names = getattr(self, "layer_names", None) or self.get_dnn_output_layer_names()
    outputs = self.net.forward(output_layer_names)

    active_class_ids = (set(inference_context["active_class_ids"]) if inference_context
                        else self.get_active_class_ids_from_ui())
    if not active_class_ids:
        active_class_ids = set(range(len(getattr(self, "class_labels", []) or [])))

    boxes = []
    confidences = []
    class_ids = []
    skipped_size_count = 0

    min_size_px, max_percent = (inference_context["size_limits"] if inference_context
                                else self.current_prediction_size_filter_values())
    conf_threshold = (inference_context["confidence"] if inference_context
                      else float(self.confidence_threshold_spinbox.value()))
    combined_scores = (inference_context.get("scores_include_objectness", False) if inference_context
                       else getattr(self, "_opencv_scores_include_objectness", False))

    for detection, class_id, confidence in iter_opencv_dnn_candidates(
            outputs, active_class_ids, conf_threshold, combined_scores):
        center_x = float(detection[0]) * img_w
        center_y = float(detection[1]) * img_h
        width = float(detection[2]) * img_w
        height = float(detection[3]) * img_h

        if width <= 0 or height <= 0:
            continue

        xyxy = (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
        if not prediction_size_allowed_xyxy(
            xyxy,
            img_w,
            img_h,
            min_size_px=min_size_px,
            max_percent=max_percent,
        ):
            skipped_size_count += 1
            continue

        x = int(round(center_x - width / 2))
        y = int(round(center_y - height / 2))
        w = int(round(width))
        h = int(round(height))

        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))

        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    if skipped_size_count:
        self.record_prediction_size_filter_skip("OpenCV DNN", skipped_size_count)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        conf_threshold,
        inference_context["iou"] if inference_context else float(self.nms_threshold_spinbox.value()),
    )
    indices = np.array(indices).flatten().tolist() if len(indices) else []

    bboxes = []
    for idx in indices:
        x, y, w, h = boxes[idx]
        bboxes.append(
            BoundingBox(
                class_ids[idx],
                (x + w / 2) / img_w,
                (y + h / 2) / img_h,
                w / img_w,
                h / img_h,
                confidence=confidences[idx],
            )
        )

    return bboxes


def display_dnn_bounding_boxes(self, bboxes, image_shape):
    scene = self.screen_view.scene()
    if scene is None:
        return

    for item in list(scene.items()):
        if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
            try:
                scene.removeItem(item)
            except RuntimeError:
                pass

    current_file = os.path.abspath(getattr(self, "current_file", "")).replace("\\", "/")
    img_h, img_w = image_shape[:2]
    bboxes = self.filter_bboxes_to_size_limits(
        bboxes,
        img_w,
        img_h,
        source="DNN display",
        record=False,
    )

    for row, bbox in enumerate(bboxes):
        rect = bbox.to_rect(img_w, img_h)
        unique_id = f"{current_file}_dnn_{row}"
        item = BoundingBoxDrawer(
            rect.x(),
            rect.y(),
            rect.width(),
            rect.height(),
            main_window=self,
            class_id=bbox.class_id,
            confidence=bbox.confidence,
            unique_id=unique_id,
        )
        item.true_line_index = row
        item.file_name = current_file
        scene.addItem(item)
        self.bounding_boxes[unique_id] = item

    self.update_blank_overlay_state(scene)
    scene.update()


def _modernize_legacy_detection_panel(self):
    """Keep old auto-label widgets alive while moving visible settings to Auto Label."""
    group = getattr(self, "groupBox_15", None)
    if group is None:
        return

    group.setVisible(False)
    group.setTitle("Auto Label")
    group.setToolTip("Open the unified Auto Label panel for weights, DINO/YOLOE, and SAHI settings.")
    group.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    group.setMaximumHeight(96)

    hidden_legacy_controls = (
        "frame_12",
        "label_28",
        "label_42",
        "label_60",
        "label_61",
        "label_218",
        "label_219",
        "confidence_threshold_spinbox",
        "nms_threshold_spinbox",
        "network_height",
        "network_width",
        "batch_inference",
        "fp_select_combobox",
    )

    for object_name in hidden_legacy_controls:
        widget = getattr(self, object_name, None)
        if widget is not None:
            widget.setVisible(False)

    layout = group.layout()
    if layout is not None:
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

    old_frame = getattr(self, "frame_12", None)
    if old_frame is not None and layout is not None:
        layout.removeWidget(old_frame)

    if getattr(self, "_legacy_detection_launcher_built", False):
        return

    launcher = QtWidgets.QFrame(group)
    launcher.setObjectName("auto_label_legacy_launcher")
    launcher.setFrameShape(QtWidgets.QFrame.NoFrame)
    launcher.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    launcher_layout = QtWidgets.QHBoxLayout(launcher)
    launcher_layout.setContentsMargins(0, 0, 0, 0)
    launcher_layout.setSpacing(8)

    summary = QtWidgets.QLabel("Weights, DINO / YOLOE, and SAHI settings")
    summary.setObjectName("auto_label_legacy_summary")
    summary.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    open_button = QtWidgets.QPushButton("Open Auto Label")
    open_button.setObjectName("open_auto_label_settings_button")
    open_button.setMinimumHeight(28)
    open_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    open_button.clicked.connect(lambda: self.show_auto_label_dialog("weights"))

    launcher_layout.addWidget(summary, 1)
    launcher_layout.addWidget(open_button)

    if isinstance(layout, QtWidgets.QGridLayout):
        layout.addWidget(launcher, 0, 0, 1, 1)
        layout.setRowMinimumHeight(1, 0)
        layout.setRowStretch(1, 0)
    elif layout is not None:
        layout.addWidget(launcher)

    self._legacy_detection_launcher_built = True

