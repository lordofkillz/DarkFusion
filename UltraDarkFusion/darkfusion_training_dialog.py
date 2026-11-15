"""Trainer/exporter workflow, dataset preparation, commands, and existing callbacks.

The main window supplies shared state through its existing owner API.
Importing this module does not start training or export a model.
"""

import GPUtil
import csv
import cv2
import glob
import html
import importlib.util
import json
import logging
import math
import numpy as np
import os
import psutil
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
import torch
import yaml
from PyQt5 import QtWidgets, sip
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QTableWidgetItem
from collections import Counter, OrderedDict
from darkfusion_dataset_analysis import ScanAnnotations
from darkfusion_dataset_operations import (
    PROJECT_SETTINGS_DIR, dataset_metadata_directory, resolve_dataset_classes_path,
    resolve_dataset_pose_schema_path,
)
from darkfusion_gpu_memory import (
    GpuWorkBusy, gpu_work, model_allowed as gpu_model_allowed, start_gpu_worker,
)
from darkfusion_image_loading import IMAGE_SUFFIXES
from darkfusion_training_size import (
    TrainingSizeAnalysis, candidate_sizes as training_candidate_sizes, parse_training_batch,
    training_file_signature, training_image_dimensions,
)
from darkfusion_training_widgets import (
    TrainingBatchSpinBox, TrainingEvaluatorActivityIndicator, TrainingMetricsPlot,
    TrainingHyperparametersWidget,
)
from darkfusion_ui_controls import blocked_signals, snap_int_to_multiple, CollapsibleSettingsGroup
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_PATH


APP_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("UltraDarkFusionLogger")


def _training_precision(owner, result=None, use_recommended=None):
    """Keep the selected cfg's native AMP value, including BF16 and FP32."""
    from darkfusion_tune_settings import read_training_configuration
    settings = getattr(owner, "settings", {}) or {}
    if use_recommended is None:
        use_recommended = bool(settings.get("trainingEvaluatorUseOptimizedArgs", True))
    path = getattr(owner, "recommended_args_path", None)
    if path is None:
        path = (result or {}).get("training_recommendations_yaml", "")
    if use_recommended and path and os.path.isfile(path):
        try:
            return read_training_configuration(path).get("amp", DEFAULT_CFG_DICT.get("amp", True))
        except (ValueError, OSError, yaml.YAMLError):
            pass  # The config/command validation will report a malformed selection.
    return ((result or {}).get("recommendations", {}).get("amp",
            settings.get("ultralyticsAmp", True)))


class UltralyticsExportWorker(QThread):
    """Export an Ultralytics checkpoint without blocking the Qt UI thread."""

    completed = pyqtSignal(str, object)
    failed = pyqtSignal(str)

    def __init__(self, model_path, export_params, parent=None):
        super().__init__(parent)
        self.model_path = str(model_path or "")
        self.export_params = dict(export_params or {})

    def run(self):
        try:
            model = YOLO(self.model_path)
            output = model.export(**self.export_params)
            self.completed.emit(str(output), model)
        except Exception as error:
            self.failed.emit(str(error))


def open_training_evaluator_dialog(self):
    existing = getattr(self, "training_evaluator_dialog", None)
    if existing is not None and not sip.isdeleted(existing):
        existing.show()
        existing.raise_()
        existing.activateWindow()
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle("Ultralytics Trainer / Exporter")
    dialog.setAttribute(Qt.WA_DeleteOnClose, True)
    dialog.setSizeGripEnabled(True)
    dialog.setMinimumSize(560, 420)
    dialog.resize(900, 660)
    if self.styleSheet():
        dialog.setStyleSheet(self.styleSheet())

    outer_layout = QtWidgets.QVBoxLayout(dialog)
    outer_layout.setContentsMargins(12, 12, 12, 12)
    outer_layout.setSpacing(8)

    body_scroll = QtWidgets.QScrollArea(dialog)
    body_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    body_scroll.setWidgetResizable(True)
    body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    body_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    body_scroll.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
    body_scroll.setMinimumSize(0, 0)
    body_scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    body_widget = QtWidgets.QWidget(body_scroll)
    body_widget.setMinimumSize(0, 0)
    body_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

    layout = QtWidgets.QVBoxLayout(body_widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)

    title = QtWidgets.QLabel("Training")
    title.setStyleSheet("font-size: 15px; font-weight: 700;")
    layout.addWidget(title)

    controls = QtWidgets.QGridLayout()
    controls.setHorizontalSpacing(8)
    controls.setVerticalSpacing(8)

    restored_setup = self._training_eval_restore_last_run_setup()
    restored_args = restored_setup.get("args", {}) if isinstance(restored_setup, dict) else {}
    restored_checkpoint = self.normalize_path(restored_setup.get("checkpoint", ""))
    restored_data_yaml = self.normalize_path(restored_setup.get("data_yaml", ""))
    restored_run_dir = self.normalize_path(restored_setup.get("run_dir", ""))

    use_training_weights = QtWidgets.QCheckBox("Use Train Setup")
    use_training_weights.setChecked(True)
    use_training_weights.setToolTip("When checked, evaluation inspects the model selected in Train Setup.")
    eval_weights_edit = QtWidgets.QLineEdit(
        restored_checkpoint or self.settings.get("trainingEvaluatorWeightsPath", "")
    )
    eval_weights_edit.setPlaceholderText("Optional eval weights path")
    eval_weights_edit.setToolTip("Optional .pt or model .yaml used only for evaluator stride/task inspection.")
    browse_weights_btn = QtWidgets.QPushButton("Browse")

    current_model_path = self.normalize_path(
        restored_checkpoint
        or getattr(self, "model_config_path", None)
        or getattr(self, "pt_path", None)
        or self.settings.get("trainingEvaluatorModelPath", "")
        or self.settings.get("ultralyticsModelConfigPath", "")
        or self.settings.get("ultralyticsPtPath", "")
        or ""
    )
    current_pretrained_path = self.normalize_path(
        getattr(self, "pretrained_model_path", None)
        or self.settings.get("ultralyticsPretrainedModelPath", "")
        or ""
    )
    current_distill_path = self.normalize_path(
        getattr(self, "ultralytics_distill_model_path", None)
        or self.settings.get("ultralyticsDistillModelPath", "")
        or ""
    )
    current_data_yaml = self.normalize_path(
        restored_data_yaml
        or getattr(self, "data_yaml_path", None)
        or self.settings.get("trainingEvaluatorDataYamlPath", "")
        or self.settings.get("ultralyticsDataYamlPath", "")
        or ""
    )
    if not current_data_yaml:
        current_data_yaml = self.autoload_training_data_yaml_for_dataset(self._training_eval_dataset_dir())
    current_runs_dir = self.normalize_path(
        restored_args.get("project", "")
        or (os.path.dirname(restored_run_dir) if restored_run_dir else "")
        or self.settings.get("trainingEvaluatorRunsDir", "")
        or getattr(self, "runs_directory", "")
        or self.settings.get("ultralyticsRunsDirectory", "")
        or self._training_eval_project_dir()
    )

    setup_group = QtWidgets.QGroupBox("Train Setup")
    setup_layout = QtWidgets.QGridLayout(setup_group)
    setup_layout.setContentsMargins(10, 14, 10, 10)
    setup_layout.setHorizontalSpacing(8)
    setup_layout.setVerticalSpacing(8)

    data_yaml_edit = QtWidgets.QLineEdit(current_data_yaml)
    data_yaml_edit.setPlaceholderText("Dataset YAML")
    data_yaml_edit.setToolTip("Ultralytics data YAML used for training.")
    browse_data_yaml_btn = QtWidgets.QPushButton("YAML")
    browse_data_yaml_btn.setToolTip("Choose the training data YAML.")

    train_model_edit = QtWidgets.QLineEdit(current_model_path)
    train_model_edit.setPlaceholderText("Model .pt or model .yaml")
    train_model_edit.setToolTip("Choose a .pt model, or a model .yaml with optional pretrained weights.")
    browse_train_model_btn = QtWidgets.QPushButton("Model")
    browse_train_model_btn.setToolTip("Choose a .pt weights file or model .yaml.")

    pretrained_edit = QtWidgets.QLineEdit(current_pretrained_path)
    pretrained_edit.setPlaceholderText("Optional pretrained .pt for model YAML")
    pretrained_edit.setToolTip(
        "Used automatically when the model field is a model YAML. "
        "Leave this empty to build the student model from the YAML without transfer weights."
    )
    browse_pretrained_btn = QtWidgets.QPushButton("Pretrained")
    browse_pretrained_btn.setToolTip("Choose pretrained weights for transfer learning from a model YAML.")
    clear_pretrained_btn = QtWidgets.QPushButton("Clear")
    clear_pretrained_btn.setToolTip("Clear optional pretrained transfer weights.")

    runs_dir_edit = QtWidgets.QLineEdit(current_runs_dir)
    runs_dir_edit.setPlaceholderText("Defaults to the dataset runs folder")
    runs_dir_edit.setToolTip("Optional run folder. Leave blank to use the active dataset's runs folder.")
    browse_runs_dir_btn = QtWidgets.QPushButton("Browse")
    browse_runs_dir_btn.setToolTip("Choose a custom runs folder, or leave blank for the dataset default.")

    current_size = self._parse_training_imgsz_value(
        restored_args.get(
            "imgsz",
            self.settings.get("trainingPreviewImgSize", self.settings.get("ultralyticsTrainImgSize", "640")),
        )
    )
    imgsz_spin = QtWidgets.QSpinBox()
    imgsz_spin.setRange(160, 2048)
    imgsz_spin.setSingleStep(32)
    imgsz_spin.setValue(current_size)
    imgsz_spin.setToolTip("Training image size used by evaluator commands.")

    def _setting_int_value(name, default, minimum=1, maximum=10000):
        try:
            value = int(float(str(self.settings.get(name, default)).strip()))
        except Exception:
            value = int(default)
        return max(int(minimum), min(int(maximum), value))

    train_batch_spin = TrainingBatchSpinBox()
    try:
        # A finished AutoBatch run records its measured positive batch. Keep
        # the user's saved Auto/-1 choice when reopening instead of replacing
        # it with that earlier run's resolved batch.
        train_batch_spin.setValue(self.settings.get("ultralyticsTrainBatch", restored_args.get("batch", 8)))
    except ValueError:
        train_batch_spin.setValue(-1)
    train_batch_spin.setToolTip("Auto (-1): let Ultralytics profile GPU memory at training start. Use a positive integer for a fixed batch, or 0.7 for a 70% memory target.")

    train_epochs_spin = QtWidgets.QSpinBox()
    train_epochs_spin.setRange(1, 10000)
    restored_epochs = self._training_eval_int_value(restored_args.get("epochs"), 0)
    train_epochs_spin.setValue(
        max(1, min(10000, restored_epochs))
        if restored_epochs > 0
        else _setting_int_value("ultralyticsTrainEpochs", 50, 1, 10000)
    )
    train_epochs_spin.setToolTip("Number of epochs for Start Training.")

    task_setup_combo = QtWidgets.QComboBox()
    for task_name in ("detect", "segment", "classify", "pose", "obb"):
        task_setup_combo.addItem(task_name)
    current_task = str(
        restored_setup.get("task", "")
        or (getattr(self, "task_combobox", None).currentText()
        if getattr(self, "task_combobox", None) is not None
        else self.settings.get("ultralyticsTask", "detect"))
    ).strip().lower()
    task_index = task_setup_combo.findText(current_task, Qt.MatchFixedString)
    task_setup_combo.setCurrentIndex(task_index if task_index >= 0 else 0)

    train_rect_checkbox = QtWidgets.QCheckBox("Rect")
    train_rect_checkbox.setChecked(bool(restored_args.get("rect", self.settings.get("ultralyticsRect", False))))
    train_rect_checkbox.setToolTip("Use rectangular training batches.")
    duplicate_labels_checkbox = QtWidgets.QCheckBox("Keep Duplicate Labels")
    duplicate_labels_checkbox.setChecked(bool(self.settings.get("ultralyticsKeepDuplicateLabels", False)))
    duplicate_labels_checkbox.setToolTip("Disable Ultralytics duplicate-label row removal through the DarkFusion wrapper.")

    distill_checkbox = QtWidgets.QCheckBox("Distill")
    distill_checkbox.setChecked(bool(self.settings.get("ultralyticsDistillEnabled", False)))
    distill_checkbox.setToolTip("Use Ultralytics Knowledge Distillation. Requires Ultralytics v8.4.77+ and a larger teacher .pt model.")
    distill_model_edit = QtWidgets.QLineEdit(current_distill_path)
    distill_model_edit.setPlaceholderText("Teacher model .pt, e.g. yolo26s.pt")
    distill_model_edit.setToolTip("Teacher model used as distill_model during training. Detect is benchmarked first; other tasks are supported by Ultralytics but less validated.")
    browse_distill_btn = QtWidgets.QPushButton("Teacher")
    browse_distill_btn.setToolTip("Choose a larger teacher .pt model for Knowledge Distillation.")
    clear_distill_btn = QtWidgets.QPushButton("Clear")
    clear_distill_btn.setToolTip("Disable and clear the teacher model.")
    distill_weight_spin = QtWidgets.QDoubleSpinBox()
    distill_weight_spin.setRange(0.0, 10.0)
    distill_weight_spin.setSingleStep(0.1)
    distill_weight_spin.setDecimals(2)
    distill_weight_spin.setValue(float(self.settings.get("ultralyticsDistillWeight", 6.0) or 6.0))
    distill_weight_spin.setMaximumWidth(82)
    distill_weight_spin.setToolTip("Ultralytics distillation loss weight, passed as dis=.")

    train_patience_checkbox = QtWidgets.QCheckBox("Patience")
    restored_patience = self._training_eval_int_value(restored_args.get("patience"), 0)
    train_patience_checkbox.setChecked(
        restored_patience > 0 or bool(self.settings.get("ultralyticsPatienceEnabled", False))
    )
    train_patience_spin = QtWidgets.QSpinBox()
    train_patience_spin.setRange(1, 10000)
    train_patience_spin.setValue(
        max(1, min(10000, restored_patience))
        if restored_patience > 0
        else _setting_int_value("ultralyticsPatience", 40, 1, 10000)
    )
    train_patience_spin.setToolTip("Early-stop patience.")

    train_freeze_checkbox = QtWidgets.QCheckBox("Freeze")
    restored_freeze = self._training_eval_int_value(restored_args.get("freeze"), 0)
    train_freeze_checkbox.setChecked(
        restored_freeze > 0 or bool(self.settings.get("ultralyticsFreezeEnabled", False))
    )
    train_freeze_spin = QtWidgets.QSpinBox()
    train_freeze_spin.setRange(0, 1000)
    train_freeze_spin.setValue(
        max(0, min(1000, restored_freeze))
        if restored_freeze > 0
        else _setting_int_value("ultralyticsFreeze", 0, 0, 1000)
    )
    train_freeze_spin.setToolTip("Freeze first N layers.")

    validation_mode_combo = QtWidgets.QComboBox()
    validation_mode_combo.setToolTip("Choose how evaluator-managed obj.yaml creates valid.txt.")
    validation_mode_combo.addItem("Auto Split", "auto")
    validation_mode_combo.addItem("Valid Folder", "folder")
    validation_mode_combo.addItem("Valid TXT", "txt")
    validation_mode_combo.addItem("Train as Val", "train")
    validation_mode_index = validation_mode_combo.findData(
        self._training_eval_validation_mode(self.settings.get("trainingEvaluatorValidationMode", "auto"))
    )
    validation_mode_combo.setCurrentIndex(validation_mode_index if validation_mode_index >= 0 else 0)

    validation_percent_combo = QtWidgets.QComboBox()
    validation_percent_combo.setToolTip("Percent of images/split to use for validation. Applies to all validation modes.")
    for percent_text in ("5%", "10%", "15%", "20%", "25%", "30%", "40%", "50%", "75%", "100%"):
        validation_percent_combo.addItem(percent_text)
    # Consolidate both old settings into one unified value (prefer the specific one, fall back to old one)
    validation_percent_text = str(
        self.settings.get("trainingEvaluatorValidationPercent", 
                         self.settings.get("trainingEvaluatorValidationSourcePercent",
                                         self.settings.get("datasetValidPercent", "20%")))
        or "20%"
    )
    validation_percent_index = validation_percent_combo.findText(validation_percent_text, Qt.MatchFixedString)
    validation_percent_combo.setCurrentIndex(validation_percent_index if validation_percent_index >= 0 else 3)

    validation_source_edit = QtWidgets.QLineEdit(str(self.settings.get("trainingEvaluatorValidationSource", "") or ""))
    validation_source_edit.setPlaceholderText("Optional validation folder or valid.txt")
    validation_source_edit.setToolTip("Used only when Validation is set to Valid Folder or Valid TXT.")
    browse_validation_source_btn = QtWidgets.QPushButton("Source")
    browse_validation_source_btn.setToolTip("Choose validation folder or valid.txt.")

    for path_edit in (
        eval_weights_edit,
        data_yaml_edit,
        train_model_edit,
        pretrained_edit,
        runs_dir_edit,
        distill_model_edit,
        validation_source_edit,
    ):
        path_edit.setMinimumWidth(0)
        path_edit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    data_row = self._row_layout(8)
    data_row.addWidget(data_yaml_edit, 1)
    data_row.addWidget(browse_data_yaml_btn)
    model_row = self._row_layout(8)
    model_row.addWidget(train_model_edit, 1)
    model_row.addWidget(browse_train_model_btn)
    pretrained_row = self._row_layout(8)
    pretrained_row.addWidget(pretrained_edit, 1)
    pretrained_row.addWidget(browse_pretrained_btn)
    pretrained_row.addWidget(clear_pretrained_btn)
    runs_row = self._row_layout(8)
    runs_row.addWidget(runs_dir_edit, 1)
    runs_row.addWidget(browse_runs_dir_btn)

    setup_layout.addWidget(QtWidgets.QLabel("Data"), 0, 0)
    setup_layout.addLayout(data_row, 0, 1, 1, 3)
    setup_layout.addWidget(QtWidgets.QLabel("Model"), 1, 0)
    setup_layout.addLayout(model_row, 1, 1, 1, 3)
    setup_layout.addWidget(QtWidgets.QLabel("Student Init"), 2, 0)
    setup_layout.addLayout(pretrained_row, 2, 1, 1, 3)
    setup_layout.addWidget(QtWidgets.QLabel("Runs"), 3, 0)
    setup_layout.addLayout(runs_row, 3, 1, 1, 3)
    setup_layout.addWidget(QtWidgets.QLabel("Task"), 4, 0)
    setup_layout.addWidget(task_setup_combo, 4, 1)
    setup_layout.addWidget(QtWidgets.QLabel("ImgSz"), 4, 2)
    setup_layout.addWidget(imgsz_spin, 4, 3)
    setup_layout.addWidget(QtWidgets.QLabel("Batch"), 5, 0)
    setup_layout.addWidget(train_batch_spin, 5, 1)
    setup_layout.addWidget(QtWidgets.QLabel("Epochs"), 5, 2)
    setup_layout.addWidget(train_epochs_spin, 5, 3)

    advanced_train_row = self._row_layout(8)
    advanced_train_row.addWidget(train_rect_checkbox)
    advanced_train_row.addWidget(duplicate_labels_checkbox)
    advanced_train_row.addWidget(train_patience_checkbox)
    advanced_train_row.addWidget(train_patience_spin)
    advanced_train_row.addWidget(train_freeze_checkbox)
    advanced_train_row.addWidget(train_freeze_spin)
    advanced_train_row.addStretch(1)
    setup_layout.addLayout(advanced_train_row, 6, 0, 1, 4)

    distill_row = self._row_layout(8)
    distill_row.addWidget(distill_checkbox)
    distill_row.addWidget(distill_model_edit, 1)
    distill_row.addWidget(browse_distill_btn)
    distill_row.addWidget(clear_distill_btn)
    distill_row.addWidget(QtWidgets.QLabel("dis"))
    distill_row.addWidget(distill_weight_spin)
    setup_layout.addWidget(QtWidgets.QLabel("KD"), 7, 0)
    setup_layout.addLayout(distill_row, 7, 1, 1, 3)

    validation_row = self._row_layout(8)
    validation_row.addWidget(validation_mode_combo)
    validation_row.addWidget(validation_percent_combo)
    validation_row.addWidget(validation_source_edit, 1)
    validation_row.addWidget(browse_validation_source_btn)
    setup_layout.addWidget(QtWidgets.QLabel("Valid"), 8, 0)
    setup_layout.addLayout(validation_row, 8, 1, 1, 3)
    setup_layout.setColumnStretch(1, 1)
    setup_layout.setColumnStretch(3, 1)
    layout.addWidget(setup_group)

    saved_candidates = str(self.settings.get("trainingEvaluatorCandidates", "Auto") or "Auto")
    if saved_candidates.replace(" ", "") == "320,416,512,640,832":
        saved_candidates = "Auto"  # Migrate the former restricted default.
    candidates_edit = QtWidgets.QLineEdit(saved_candidates)
    candidates_edit.setPlaceholderText("Auto, or e.g. 512,640,1024")
    candidates_edit.setToolTip("Auto compares 320-2048 plus a size calculated from training objects. A custom list is compared along with the current ImgSz. Sizes round up to the model stride.")
    candidates_edit.setMinimumWidth(0)
    candidates_edit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    always_preview_checkbox = QtWidgets.QCheckBox("Train View")
    always_preview_checkbox.setChecked(self.is_training_preview_always_on())
    always_preview_checkbox.setToolTip("Toggle read-only train-size preview while browsing images.")
    auto_yaml_checkbox = QtWidgets.QCheckBox("Auto YAML")
    auto_yaml_checkbox.setChecked(bool(self.settings.get("trainingEvaluatorAutoYaml", True)))
    auto_yaml_checkbox.setToolTip("Generate obj.yaml and train/validation lists for the loaded dataset, then load the generated YAML paths. Valid TXT and Valid Folder require a source.")
    optimized_args_checkbox = QtWidgets.QCheckBox("Recommended Args")
    optimized_args_checkbox.setChecked(bool(self.settings.get("trainingEvaluatorUseOptimizedArgs", True)))
    optimized_args_checkbox.setToolTip("Use the selected training configuration when checked. When unchecked, tuning saves parameters in the selected Data YAML and training reads them there. Browsing never rewrites either file.")
    skip_image_scan_checkbox = QtWidgets.QCheckBox("Skip Image Scan")
    skip_image_scan_checkbox.setChecked(bool(self.settings.get("trainingEvaluatorSkipImageScan", False)))
    skip_image_scan_checkbox.setToolTip("Skip the full image scan and use quick parameter estimation. Useful after you've already scanned this dataset. Uncheck to do a full re-scan with detailed statistics.")

    # Display field for train_recommendations.yaml path
    current_recommended_args_yaml = self.normalize_path(
        self.settings.get("trainingEvaluatorRecommendedArgsYaml", "")
        or ""
    )
    self.recommended_args_path = current_recommended_args_yaml
    recommended_args_display = QtWidgets.QLineEdit(current_recommended_args_yaml)
    recommended_args_display.setPlaceholderText("train_recommendations.yaml path will appear here or paste/browse")
    recommended_args_display.setToolTip("Path to train_recommendations.yaml file. Auto-populated after Generate Files + Parameters, or browse to load a pre-existing one.")
    browse_recommended_args_btn = QtWidgets.QPushButton("Browse")
    browse_recommended_args_btn.setToolTip(
        "Select an existing training configuration YAML. Browsing only remembers "
        "the selected path; it does not generate or overwrite the file."
    )

    probe_epochs_spin = QtWidgets.QSpinBox()
    probe_epochs_spin.setRange(1, 20)
    probe_epochs_spin.setValue(int(self.settings.get("trainingEvaluatorProbeEpochs", 5) or 5))
    probe_epochs_spin.setToolTip("Epochs per Ultralytics tuning trial.")
    tune_iterations_spin = QtWidgets.QSpinBox()
    tune_iterations_spin.setRange(1, 300)
    tune_iterations_spin.setValue(int(self.settings.get("trainingEvaluatorTuneIterations", 10) or 10))
    tune_iterations_spin.setToolTip("Number of Ultralytics tuning trials. Total work is trials multiplied by epochs per trial.")
    ray_tune_checkbox = QtWidgets.QCheckBox("Ray Tune")
    ray_tune_checkbox.setChecked(bool(self.settings.get("trainingEvaluatorUseRayTune", False)))
    ray_tune_checkbox.setEnabled(self._ray_tune_available())
    ray_tune_checkbox.setToolTip(
        "Use Ultralytics Ray Tune. Install with: pip install \"ray[tune]\""
        if not self._ray_tune_available()
        else "Use Ray Tune instead of the built-in tuner. DarkFusion allocates one CUDA GPU per trial when available."
    )
    tune_distill_checkbox = QtWidgets.QCheckBox("Tune Distill")
    tune_distill_checkbox.setChecked(bool(self.settings.get("trainingEvaluatorTuneDistill", False)))
    tune_distill_checkbox.setToolTip(
        "Experimental: include the selected distill_model in each hyperparameter tuning trial. "
        "Usually tune first, then train with distillation."
    )
    batch_override_checkbox = QtWidgets.QCheckBox("Batch")
    batch_override_checkbox.setChecked(bool(self.settings.get("trainingEvaluatorBatchOverrideEnabled", False)))
    batch_override_checkbox.setToolTip(
        "Override evaluator training batch with Auto, a memory fraction, or a fixed batch. Validation uses batch 1."
    )
    batch_override_spin = TrainingBatchSpinBox()
    try:
        batch_override_spin.setValue(self.settings.get("trainingEvaluatorBatchOverride", train_batch_spin.value()))
    except ValueError:
        batch_override_spin.setValue(-1)
    batch_override_spin.setMaximumWidth(82)
    workers_override_checkbox = QtWidgets.QCheckBox("Set Workers")
    restored_workers_present = "workers" in restored_args and restored_args.get("workers") is not None
    workers_override_checkbox.setChecked(
        restored_workers_present or bool(self.settings.get("trainingEvaluatorWorkersOverrideEnabled", True))
    )
    workers_override_checkbox.setToolTip("Override Ultralytics dataloader workers. 0 is safest on Windows.")
    workers_override_spin = QtWidgets.QSpinBox()
    workers_override_spin.setRange(0, 32)
    restored_workers = self._training_eval_int_value(restored_args.get("workers"), 0)
    workers_override_spin.setValue(
        max(0, min(32, restored_workers))
        if restored_workers_present
        else int(self.settings.get("trainingEvaluatorWorkersOverride", 0 if sys.platform.startswith("win") else 2) or 0)
    )
    workers_override_spin.setMaximumWidth(70)
    cache_mode_combo = QtWidgets.QComboBox()
    cache_mode_combo.setToolTip("Dataset cache mode. Off is safest; RAM is fastest but can consume lots of memory.")
    cache_mode_combo.addItem("Cache Off", "off")
    cache_mode_combo.addItem("Cache RAM", "ram")
    cache_mode_combo.addItem("Cache Disk", "disk")
    restored_cache = restored_args.get("cache") if "cache" in restored_args else None
    if isinstance(restored_cache, str) and restored_cache.strip().lower() in {"disk", "ram"}:
        cache_mode = restored_cache.strip().lower()
    elif restored_cache is True:
        cache_mode = "ram"
    elif restored_cache is False:
        cache_mode = "off"
    else:
        cache_mode = str(self.settings.get("trainingEvaluatorCacheMode", "off")).lower()
    cache_index = cache_mode_combo.findData(cache_mode)
    cache_mode_combo.setCurrentIndex(cache_index if cache_index >= 0 else 0)

    controls.addWidget(QtWidgets.QLabel("Candidates"), 0, 0)
    controls.addWidget(candidates_edit, 0, 1)
    flags_row = self._row_layout(8)
    flags_row.addWidget(auto_yaml_checkbox)
    flags_row.addWidget(optimized_args_checkbox)
    flags_row.addWidget(skip_image_scan_checkbox)
    flags_row.addWidget(always_preview_checkbox)
    flags_row.addStretch(1)
    controls.addLayout(flags_row, 0, 2)
    controls.addWidget(QtWidgets.QLabel("Recommended Args YAML"), 1, 0)
    recommended_args_layout = self._row_layout(8)
    recommended_args_layout.addWidget(recommended_args_display, 1)
    recommended_args_layout.addWidget(browse_recommended_args_btn)
    controls.addLayout(recommended_args_layout, 1, 1, 1, 2)
    controls.addWidget(QtWidgets.QLabel("Epochs/Trial"), 2, 0)
    controls.addWidget(probe_epochs_spin, 2, 1)
    tune_row = self._row_layout(8)
    tune_row.addWidget(QtWidgets.QLabel("Trials"))
    tune_row.addWidget(tune_iterations_spin)
    tune_row.addWidget(ray_tune_checkbox)
    tune_row.addWidget(tune_distill_checkbox)
    tune_row.addStretch(1)
    controls.addLayout(tune_row, 2, 2)
    override_row = self._row_layout(8)
    override_row.addWidget(batch_override_checkbox)
    override_row.addWidget(batch_override_spin)
    override_row.addWidget(workers_override_checkbox)
    override_row.addWidget(workers_override_spin)
    override_row.addWidget(cache_mode_combo)
    override_row.addStretch(1)
    controls.addWidget(QtWidgets.QLabel("Runtime"), 3, 0)
    controls.addLayout(override_row, 3, 1, 1, 2)
    controls.setColumnStretch(1, 1)
    tools_group = QtWidgets.QGroupBox("Evaluator Tools")
    tools_group.setLayout(controls)
    layout.addWidget(tools_group)

    hyperparameter_summary = QtWidgets.QLabel("Hyperparameters: generate or select a training configuration YAML.")
    hyperparameter_summary.setObjectName("hyperparameters_summary")
    hyperparameter_summary.setWordWrap(True)
    layout.addWidget(hyperparameter_summary)
    hyperparameter_group = CollapsibleSettingsGroup("Hyperparameters", expanded=False)
    hyperparameter_group.setObjectName("trainer_hyperparameters_group")
    hyperparameter_layout = QtWidgets.QVBoxLayout(hyperparameter_group.content)
    hyperparameters = TrainingHyperparametersWidget(hyperparameter_group.content)
    hyperparameter_layout.addWidget(hyperparameters)
    layout.addWidget(hyperparameter_group)

    evaluator_tabs = QtWidgets.QTabWidget()

    results_text = QtWidgets.QTextEdit()
    results_text.setReadOnly(True)
    results_text.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
    results_text.setText("Choose the setup, then click Generate Files + Parameters.")
    evaluator_tabs.addTab(results_text, "Summary")

    metrics_tab = QtWidgets.QWidget()
    metrics_layout = QtWidgets.QVBoxLayout(metrics_tab)
    metrics_layout.setContentsMargins(8, 8, 8, 8)
    metrics_layout.setSpacing(8)
    activity_indicator = TrainingEvaluatorActivityIndicator()
    run_state_label = QtWidgets.QLabel("No evaluator run is active.")
    run_state_label.setStyleSheet("font-weight: 700;")
    status_row = self._row_layout(8)
    status_row.addWidget(activity_indicator)
    status_row.addWidget(run_state_label, 1)
    metrics_layout.addLayout(status_row)
    eta_panel = QtWidgets.QFrame()
    eta_panel.setObjectName("trainingEtaPanel")
    eta_panel.setStyleSheet(
        "QFrame#trainingEtaPanel { background: #142839; border: 1px solid #4fa3ff; "
        "border-radius: 6px; }"
    )
    eta_layout = QtWidgets.QVBoxLayout(eta_panel)
    eta_layout.setContentsMargins(12, 8, 12, 8)
    eta_layout.setSpacing(3)
    eta_label = QtWidgets.QLabel("Estimated completion: waiting for training")
    eta_label.setObjectName("trainingEtaLabel")
    eta_label.setWordWrap(True)
    eta_label.setStyleSheet("font-size: 20px; font-weight: 800; color: #8dccff; border: none;")
    eta_detail_label = QtWidgets.QLabel("An estimate appears after the first completed epoch.")
    eta_detail_label.setWordWrap(True)
    eta_detail_label.setStyleSheet("color: #dce6f0; border: none;")
    eta_panel.setToolTip(
        "Estimated from the last five completed epoch timings, including validation. "
        "The finish time updates as training progresses; early stopping and final saving can change it."
    )
    eta_layout.addWidget(eta_label)
    eta_layout.addWidget(eta_detail_label)
    metrics_layout.addWidget(eta_panel)
    run_dir_label = QtWidgets.QLabel("Run: none")
    run_dir_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    run_dir_label.setStyleSheet("color: #9aa4af;")
    metrics_label = QtWidgets.QLabel("Waiting for results.csv.")
    metrics_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    baseline_label = QtWidgets.QLabel("Baseline: evaluate dataset and validate weights first.")
    baseline_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    baseline_label.setStyleSheet("color: #9aa4af; font-weight: 700;")
    metrics_layout.addWidget(run_dir_label)
    metrics_layout.addWidget(metrics_label)
    metrics_layout.addWidget(baseline_label)
    resume_note_label = QtWidgets.QLabel("")
    resume_note_label.setWordWrap(True)
    resume_note_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    resume_note_label.setStyleSheet("color: #f0d264; font-weight: 700;")
    resume_note_label.hide()
    metrics_layout.addWidget(resume_note_label)
    metrics_splitter = QtWidgets.QSplitter(Qt.Vertical)
    metrics_table = QtWidgets.QTableWidget()
    metrics_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    metrics_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    metrics_table.setAlternatingRowColors(True)
    metrics_table.verticalHeader().setVisible(False)
    metrics_table.horizontalHeader().setStretchLastSection(False)
    metrics_table.setMinimumHeight(110)
    log_text = QtWidgets.QPlainTextEdit()
    log_text.setReadOnly(True)
    log_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
    log_text.setMaximumBlockCount(1200)
    log_text.setPlaceholderText("Validation/training console output will appear here.")
    metrics_splitter.addWidget(metrics_table)
    metrics_splitter.addWidget(log_text)
    metrics_splitter.setStretchFactor(0, 2)
    metrics_splitter.setStretchFactor(1, 3)
    metrics_layout.addWidget(metrics_splitter, 1)
    evaluator_tabs.addTab(metrics_tab, "Live Metrics")

    charts_tab = QtWidgets.QWidget()
    charts_layout = QtWidgets.QVBoxLayout(charts_tab)
    charts_layout.setContentsMargins(8, 8, 8, 8)
    charts_layout.setSpacing(8)
    diagnosis_panel = QtWidgets.QFrame()
    diagnosis_panel.setObjectName("trainingDiagnosisPanel")
    diagnosis_layout = QtWidgets.QHBoxLayout(diagnosis_panel)
    diagnosis_layout.setContentsMargins(10, 8, 10, 8)
    diagnosis_layout.setSpacing(8)
    diagnosis_light = QtWidgets.QFrame()
    diagnosis_light.setFixedSize(14, 14)
    diagnosis_title = QtWidgets.QLabel("Waiting for metrics")
    diagnosis_title.setMinimumWidth(130)
    diagnosis_detail = QtWidgets.QLabel("No results.csv rows yet.")
    diagnosis_detail.setWordWrap(True)
    diagnosis_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
    diagnosis_layout.addWidget(diagnosis_light)
    diagnosis_layout.addWidget(diagnosis_title)
    diagnosis_layout.addWidget(diagnosis_detail, 1)
    charts_layout.addWidget(diagnosis_panel)
    score_plot = TrainingMetricsPlot("mAP / Precision / Recall", value_range=(0.0, 1.0))
    loss_plot = TrainingMetricsPlot("Training / Validation Loss")
    charts_layout.addWidget(score_plot, 1)
    charts_layout.addWidget(loss_plot, 1)
    artifact_row = self._row_layout(8)
    artifact_combo = QtWidgets.QComboBox()
    artifact_combo.setToolTip("Ultralytics images from the active run, such as results.png, PR curves, confusion matrix, or val batch previews.")
    open_artifact_btn = QtWidgets.QPushButton("Open Viewer")
    open_artifact_btn.setToolTip("Open the selected visual in a zoomable image viewer.")
    artifact_row.addWidget(QtWidgets.QLabel("Visual"))
    artifact_row.addWidget(artifact_combo, 1)
    artifact_row.addWidget(open_artifact_btn)
    charts_layout.addLayout(artifact_row)
    artifact_preview = QtWidgets.QLabel("No visual artifact yet")
    artifact_preview.setAlignment(Qt.AlignCenter)
    artifact_preview.setMinimumHeight(120)
    artifact_preview.setFrameShape(QtWidgets.QFrame.StyledPanel)
    artifact_preview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    artifact_preview.setCursor(Qt.PointingHandCursor)
    charts_layout.addWidget(artifact_preview, 2)
    evaluator_tabs.addTab(charts_tab, "Charts")

    validation_review_tab = QtWidgets.QWidget()
    validation_review_layout = QtWidgets.QVBoxLayout(validation_review_tab)
    validation_review_layout.setContentsMargins(8, 8, 8, 8)
    validation_review_layout.setSpacing(8)

    review_workflow_help = QtWidgets.QLabel(
        "<b>Model Validation Review</b><br>"
        "1. Choose <b>Entire Dataset</b>, then click <b>Find Validation Problems</b> to compare the "
        "selected checkpoint with every unique original image and saved label. Limited runs "
        "continue in batches.<br>"
        "2. Select a finding and open it in <b>Label Maker</b>. Use the normal drawing tools, choose "
        "<b>Use Prediction as Ground Truth</b>, or keep the current label. Your normal "
        "Previous/Next controls advance through the issue queue and save the original labels.<br>"
        "Predictions on intentionally blank images are separated as <b>Hard Negatives</b>; "
        "keep those images blank when the detected object should be ignored.<br>"
        "3. Run this analysis again after cleanup, then retrain on the corrected dataset.<br>"
        "<i>Metrics Only</i> calculates mAP/precision/recall; it does not create an editable image queue."
    )
    review_workflow_help.setWordWrap(True)
    review_workflow_help.setStyleSheet(
        "QLabel { background: #17212b; border: 1px solid #344454; "
        "border-radius: 6px; padding: 9px; color: #dce6f0; }"
    )
    validation_review_layout.addWidget(review_workflow_help)

    model_review_source_label = QtWidgets.QLabel()
    model_review_source_label.setWordWrap(True)
    model_review_source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    model_review_source_label.setStyleSheet("color: #9aa4af; padding: 2px 0;")
    validation_review_layout.addWidget(model_review_source_label)

    review_controls = QtWidgets.QGridLayout()
    review_split_combo = QtWidgets.QComboBox()
    review_split_combo.addItem("Entire Dataset", "all")
    review_split_combo.addItem("Validation", "val")
    review_split_combo.addItem("Test", "test")
    review_split_combo.addItem("Training", "train")
    saved_review_split = str(
        self.settings.get("trainingValidationReviewSplit", "all") or "all"
    )
    saved_review_split_index = review_split_combo.findData(saved_review_split)
    review_split_combo.setCurrentIndex(
        saved_review_split_index if saved_review_split_index >= 0 else 0
    )
    review_conf_spin = QtWidgets.QDoubleSpinBox()
    review_conf_spin.setRange(0.001, 1.0)
    review_conf_spin.setDecimals(3)
    review_conf_spin.setSingleStep(0.025)
    review_conf_spin.setValue(float(self.settings.get("trainingValidationReviewConfidence", 0.25) or 0.25))
    review_iou_spin = QtWidgets.QDoubleSpinBox()
    review_iou_spin.setRange(0.05, 0.95)
    review_iou_spin.setDecimals(2)
    review_iou_spin.setSingleStep(0.05)
    review_iou_spin.setValue(float(self.settings.get("trainingValidationReviewMatchIou", 0.50) or 0.50))
    review_limit_spin = QtWidgets.QSpinBox()
    review_limit_spin.setRange(0, 10000000)
    review_limit_spin.setSpecialValueText("All")
    review_limit_spin.setValue(int(self.settings.get("trainingValidationReviewLimit", 0) or 0))
    review_limit_spin.setToolTip(
        "Use All to scan the complete selected dataset split. Limited runs automatically "
        "continue with the next batch instead of rescanning the first images."
    )
    review_analyze_btn = QtWidgets.QPushButton("1. Find Validation Problems")
    review_analyze_btn.setToolTip(
        "Run the trained model on original dataset images and compare every prediction "
        "with the matching saved annotation. Predictions on blank images become hard negatives."
    )
    review_validate_metrics_btn = QtWidgets.QPushButton("Metrics Only (mAP)")
    review_validate_metrics_btn.setToolTip(
        "Calculate aggregate validation scores. This does not build the editable problem-image queue."
    )
    review_load_btn = QtWidgets.QPushButton("Open Saved Review")
    review_compare_btn = QtWidgets.QPushButton("Compare Report")
    review_export_btn = QtWidgets.QPushButton("Export CSV")
    review_reset_history_btn = QtWidgets.QPushButton("Reset Reviewed")
    review_reset_history_btn.setToolTip(
        "Forget kept, ignored, accepted, and manually edited issue decisions so they can be reviewed again."
    )
    review_stop_btn = QtWidgets.QPushButton("Stop Analysis")
    review_stop_btn.setEnabled(False)

    review_controls.addWidget(QtWidgets.QLabel("Split"), 0, 0)
    review_controls.addWidget(review_split_combo, 0, 1)
    review_controls.addWidget(QtWidgets.QLabel("Prediction confidence"), 0, 2)
    review_controls.addWidget(review_conf_spin, 0, 3)
    review_controls.addWidget(QtWidgets.QLabel("Match IoU"), 0, 4)
    review_controls.addWidget(review_iou_spin, 0, 5)
    review_controls.addWidget(QtWidgets.QLabel("Images to scan"), 1, 0)
    review_controls.addWidget(review_limit_spin, 1, 1)
    review_action_row = self._row_layout(8)
    review_action_row.addWidget(review_validate_metrics_btn)
    review_action_row.addWidget(review_analyze_btn)
    review_action_row.addWidget(review_stop_btn)
    review_action_row.addWidget(review_load_btn)
    review_action_row.addWidget(review_compare_btn)
    review_action_row.addWidget(review_export_btn)
    review_action_row.addWidget(review_reset_history_btn)
    review_action_row.addStretch(1)
    review_controls.addLayout(review_action_row, 1, 2, 1, 4)
    review_controls.setColumnStretch(1, 1)
    validation_review_layout.addLayout(review_controls)

    review_status_label = QtWidgets.QLabel(
        "Ready to compare the selected checkpoint against the validation source shown above."
    )
    review_status_label.setWordWrap(True)
    review_status_label.setStyleSheet("color: #9aa4af; font-weight: 700;")
    validation_review_layout.addWidget(review_status_label)

    review_label_state_label = QtWidgets.QLabel()
    review_label_state_label.setObjectName("validationLabelCoverage")
    review_label_state_label.setWordWrap(True)
    review_label_state_label.setTextFormat(Qt.PlainText)
    validation_review_layout.addWidget(review_label_state_label)

    review_summary_row = self._row_layout(8)
    review_summary_labels = {}
    for key, title in (
        ("hard_negative", "Hard Neg"),
        ("unverified_prediction", "Unverified"),
        ("false_positive", "False +"),
        ("false_negative", "Missed"),
        ("wrong_class", "Class"),
        ("weak_localization", "Weak"),
        ("duplicate_prediction", "Duplicate"),
        ("poor_keypoints", "Keypoints"),
    ):
        label = QtWidgets.QLabel(f"{title}: 0")
        label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumWidth(86)
        label.setStyleSheet("padding: 5px; font-weight: 700;")
        review_summary_labels[key] = label
        review_summary_row.addWidget(label)
    review_summary_row.addStretch(1)
    validation_review_layout.addLayout(review_summary_row)

    review_filter_row = self._row_layout(8)
    review_filter_combo = QtWidgets.QComboBox()
    review_filter_combo.addItem("All issues", "all")
    review_filter_combo.addItem("Hard negatives (empty labels)", "hard_negative")
    review_filter_combo.addItem("Unverified predictions", "unverified_prediction")
    review_filter_combo.addItem("Missing label files", "missing_label_file")
    review_filter_combo.addItem("Invalid label files", "invalid_label_file")
    review_filter_combo.addItem("Unreadable label files", "unreadable_label_file")
    review_filter_combo.addItem("False positives", "false_positive")
    review_filter_combo.addItem("Missed objects", "false_negative")
    review_filter_combo.addItem("Wrong classes", "wrong_class")
    review_filter_combo.addItem("Weak localization", "weak_localization")
    review_filter_combo.addItem("Duplicate predictions", "duplicate_prediction")
    review_filter_combo.addItem("Poor keypoints", "poor_keypoints")
    review_filter_combo.addItem("Unreviewed only", "unreviewed")
    review_search_edit = QtWidgets.QLineEdit()
    review_search_edit.setPlaceholderText("Filter by filename or class")
    review_filter_row.addWidget(QtWidgets.QLabel("Show"))
    review_filter_row.addWidget(review_filter_combo)
    review_filter_row.addWidget(review_search_edit, 1)
    validation_review_layout.addLayout(review_filter_row)

    review_issue_table = QtWidgets.QTableWidget()
    review_issue_table.setColumnCount(5)
    review_issue_table.setHorizontalHeaderLabels(["Type", "Class", "Conf", "IoU", "Image"])
    review_issue_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    review_issue_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    review_issue_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    review_issue_table.verticalHeader().setVisible(False)
    review_issue_table.horizontalHeader().setStretchLastSection(True)
    review_issue_table.setColumnWidth(0, 125)
    review_issue_table.setColumnWidth(1, 90)
    review_issue_table.setColumnWidth(2, 64)
    review_issue_table.setColumnWidth(3, 64)
    review_issue_table.setMinimumWidth(420)
    validation_review_layout.addWidget(review_issue_table, 1)

    review_queue_row = self._row_layout(8)
    review_queue_hint = QtWidgets.QLabel(
        "Select a finding and open it in Label Maker to edit the original dataset. "
        "Double-clicking a row does the same thing."
    )
    review_queue_hint.setWordWrap(True)
    review_queue_hint.setStyleSheet("color: #9aa4af;")
    review_open_labeler_btn = QtWidgets.QPushButton("2. Review Problems in Label Maker")
    review_open_labeler_btn.setToolTip(
        "Open the filtered model findings as a cleanup queue in the normal Label Maker."
    )
    review_open_labeler_btn.setDefault(True)
    review_open_labeler_btn.setEnabled(False)
    review_queue_row.addWidget(review_queue_hint, 1)
    review_queue_row.addWidget(review_open_labeler_btn)
    validation_review_layout.addLayout(review_queue_row)
    evaluator_tabs.addTab(validation_review_tab, "Validation Review")

    export_tab = QtWidgets.QWidget()
    export_layout = QtWidgets.QVBoxLayout(export_tab)
    export_layout.setContentsMargins(10, 10, 10, 10)
    export_layout.setSpacing(10)

    export_help = QtWidgets.QLabel(
        "<b>Export the trained Ultralytics checkpoint</b><br>"
        "Choose Best/Last from the active training run or browse to <b>any compatible .pt weights</b>. "
        "INT8 becomes ready when the selected format supports it and a calibration dataset YAML "
        "is available. Exported files are written beside the selected checkpoint by Ultralytics.<br>"
        "For DarkFusion's Windows ML / ONNX backend, export as <b>ONNX</b> with <b>Embed NMS off</b>. "
        "This preserves native NMS-free heads when available. Confidence stays adjustable; "
        "NMS/IOU applies only to models that need suppression, not NMS-free heads."
    )
    export_help.setWordWrap(True)
    export_help.setStyleSheet(
        "QLabel { background: #17212b; border: 1px solid #344454; "
        "border-radius: 6px; padding: 9px; color: #dce6f0; }"
    )
    export_layout.addWidget(export_help)

    export_form = QtWidgets.QGridLayout()
    export_form.setHorizontalSpacing(8)
    export_form.setVerticalSpacing(8)

    def latest_run_export_checkpoint():
        record = self._training_eval_latest_saved_run(training_only=True)
        run_dirs = [
            self.normalize_path((record or {}).get("run_dir", "")),
            restored_run_dir,
            self.normalize_path(self.settings.get("trainingEvaluatorLastTrainingRunDir", "")),
        ]
        seen_run_dirs = set()
        for run_dir in run_dirs:
            key = os.path.normcase(run_dir)
            if not run_dir or key in seen_run_dirs:
                continue
            seen_run_dirs.add(key)
            for name in ("best.pt", "last.pt"):
                candidate = self.normalize_path(os.path.join(run_dir, "weights", name))
                if not candidate or not os.path.isfile(candidate):
                    continue
                info = self.inspect_training_eval_weights(candidate)
                if info.get("loaded"):
                    return candidate
        return ""

    def usable_export_checkpoint(*candidates):
        for raw_path in candidates:
            candidate = self.normalize_path(raw_path)
            if not candidate or not os.path.isfile(candidate) or not candidate.lower().endswith(".pt"):
                continue
            info = self.inspect_training_eval_weights(candidate)
            if info.get("loaded"):
                return candidate
        return latest_run_export_checkpoint()

    export_checkpoint_edit = QtWidgets.QLineEdit(
        usable_export_checkpoint(
            restored_checkpoint,
            self.settings.get("exportModelPath", ""),
            eval_weights_edit.text(),
            train_model_edit.text(),
        )
    )
    export_checkpoint_edit.setObjectName("trainer_export_checkpoint")
    export_checkpoint_edit.setPlaceholderText("Trained .pt checkpoint")
    export_checkpoint_edit.setToolTip("The trained Ultralytics .pt checkpoint to export.")
    export_checkpoint_browse = QtWidgets.QPushButton("Browse")
    export_checkpoint_browse.setObjectName("trainer_export_browse")
    export_checkpoint_use_run = QtWidgets.QPushButton("Use Best / Last")
    export_checkpoint_use_run.setObjectName("trainer_export_use_run")
    checkpoint_row = self._row_layout(8)
    checkpoint_row.addWidget(export_checkpoint_edit, 1)
    checkpoint_row.addWidget(export_checkpoint_use_run)
    checkpoint_row.addWidget(export_checkpoint_browse)
    export_form.addWidget(QtWidgets.QLabel("Checkpoint"), 0, 0)
    export_form.addLayout(checkpoint_row, 0, 1, 1, 5)

    export_data_edit = QtWidgets.QLineEdit(current_data_yaml)
    export_data_edit.setObjectName("trainer_export_data_yaml")
    export_data_edit.setPlaceholderText("Dataset YAML (required for INT8 calibration)")
    export_data_browse = QtWidgets.QPushButton("YAML")
    export_data_browse.setObjectName("trainer_export_data_browse")
    export_data_use_training = QtWidgets.QPushButton("Use Training Dataset")
    export_data_use_training.setObjectName("trainer_export_use_training_data")
    data_export_row = self._row_layout(8)
    data_export_row.addWidget(export_data_edit, 1)
    data_export_row.addWidget(export_data_use_training)
    data_export_row.addWidget(export_data_browse)
    export_form.addWidget(QtWidgets.QLabel("Calibration data"), 1, 0)
    export_form.addLayout(data_export_row, 1, 1, 1, 5)

    export_format_combo = QtWidgets.QComboBox()
    export_format_combo.setObjectName("trainer_export_format")
    legacy_format_combo = getattr(self, "convert_model", None)
    legacy_formats = []
    if isinstance(legacy_format_combo, QtWidgets.QComboBox):
        legacy_formats = [
            legacy_format_combo.itemText(index).strip()
            for index in range(legacy_format_combo.count())
            if legacy_format_combo.itemText(index).strip()
        ]
    for format_name in self.ultralytics_export_formats() or legacy_formats or (
        "torchscript", "onnx", "openvino", "engine", "coreml",
        "saved_model", "pb", "tflite", "edgetpu",
    ):
        export_format_combo.addItem(format_name)
    saved_export_format = str(self.settings.get("exportFormat", "") or "").strip()
    if saved_export_format == "tflite" and export_format_combo.findText("litert") >= 0:
        saved_export_format = "litert"
    format_index = export_format_combo.findText(saved_export_format, Qt.MatchFixedString)
    export_format_combo.setCurrentIndex(format_index if format_index >= 0 else 0)

    export_precision_combo = QtWidgets.QComboBox()
    export_precision_combo.setObjectName("trainer_export_precision")
    export_precision_combo.addItem("FP32", "fp32")
    export_precision_combo.addItem("FP16", "fp16")
    export_precision_combo.addItem("INT8", "int8")
    export_precision_combo.addItem("INT8 weights / 16-bit activations", "w8a16")
    export_precision_combo.addItem("Dynamic INT8", "w8a32")
    if bool(self.settings.get("exportInt8", False)):
        export_precision_combo.setCurrentIndex(2)
    elif bool(self.settings.get("exportHalf", False)):
        export_precision_combo.setCurrentIndex(1)
    saved_precision_index = export_precision_combo.findData(self.settings.get("exportPrecision", ""))
    if saved_precision_index >= 0:
        export_precision_combo.setCurrentIndex(saved_precision_index)

    export_gpu_checkbox = QtWidgets.QCheckBox("Use GPU")
    export_gpu_checkbox.setObjectName("trainer_export_use_gpu")
    export_gpu_checkbox.setChecked(bool(self.settings.get("exportUseGpu", True)) and torch.cuda.is_available())
    export_gpu_checkbox.setEnabled(torch.cuda.is_available())

    export_width_spin = QtWidgets.QSpinBox()
    export_width_spin.setObjectName("trainer_export_width")
    export_width_spin.setRange(32, 8192)
    export_width_spin.setSingleStep(32)
    export_height_spin = QtWidgets.QSpinBox()
    export_height_spin.setObjectName("trainer_export_height")
    export_height_spin.setRange(32, 8192)
    export_height_spin.setSingleStep(32)
    try:
        export_width_spin.setValue(int(self.settings.get("exportWidth", 640) or 640))
    except Exception:
        export_width_spin.setValue(640)
    try:
        export_height_spin.setValue(int(self.settings.get("exportHeight", 640) or 640))
    except Exception:
        export_height_spin.setValue(640)

    export_form.addWidget(QtWidgets.QLabel("Format"), 2, 0)
    export_form.addWidget(export_format_combo, 2, 1)
    export_form.addWidget(QtWidgets.QLabel("Precision"), 2, 2)
    export_form.addWidget(export_precision_combo, 2, 3)
    export_form.addWidget(export_gpu_checkbox, 2, 4)
    export_form.addWidget(QtWidgets.QLabel("Width"), 3, 0)
    export_form.addWidget(export_width_spin, 3, 1)
    export_form.addWidget(QtWidgets.QLabel("Height"), 3, 2)
    export_form.addWidget(export_height_spin, 3, 3)

    export_batch_checkbox = QtWidgets.QCheckBox("Custom batch")
    export_batch_checkbox.setObjectName("trainer_export_batch_enabled")
    export_batch_checkbox.setChecked(bool(self.settings.get("exportBatchEnabled", False)))
    export_batch_checkbox.setToolTip(
        "Unchecked exports batch 1. Use batch 1 for single-frame video inference. "
        "This does not change the training batch or enable dynamic shapes."
    )
    export_batch_spin = QtWidgets.QSpinBox()
    export_batch_spin.setObjectName("trainer_export_batch")
    export_batch_spin.setRange(1, 512)
    export_batch_spin.setValue(self.saved_export_batch_size())
    export_batch_spin.setEnabled(export_batch_checkbox.isChecked())
    export_dynamic_checkbox = QtWidgets.QCheckBox("Dynamic shapes")
    export_dynamic_checkbox.setObjectName("trainer_export_dynamic")
    export_dynamic_checkbox.setChecked(bool(self.settings.get("exportDynamic", False)))
    export_simplify_checkbox = QtWidgets.QCheckBox("Simplify graph")
    export_simplify_checkbox.setObjectName("trainer_export_simplify")
    export_simplify_checkbox.setChecked(bool(self.settings.get("exportSimplify", False)))
    export_form.addWidget(export_batch_checkbox, 4, 0)
    export_form.addWidget(export_batch_spin, 4, 1)
    export_form.addWidget(export_dynamic_checkbox, 4, 2)
    export_form.addWidget(export_simplify_checkbox, 4, 3)

    export_nms_checkbox = QtWidgets.QCheckBox("Embed NMS")
    export_nms_checkbox.setObjectName("trainer_export_nms")
    export_nms_checkbox.setChecked(bool(self.settings.get("exportNms", False)))
    export_agnostic_checkbox = QtWidgets.QCheckBox("Class agnostic")
    export_agnostic_checkbox.setObjectName("trainer_export_agnostic")
    export_agnostic_checkbox.setChecked(bool(self.settings.get("exportAgnosticNms", False)))
    export_conf_spin = QtWidgets.QDoubleSpinBox()
    export_conf_spin.setObjectName("trainer_export_conf")
    export_conf_spin.setRange(0.001, 1.0)
    export_conf_spin.setDecimals(3)
    export_conf_spin.setSingleStep(0.025)
    export_conf_spin.setValue(float(self.settings.get("exportConf", 0.25) or 0.25))
    export_iou_spin = QtWidgets.QDoubleSpinBox()
    export_iou_spin.setObjectName("trainer_export_iou")
    export_iou_spin.setRange(0.01, 0.99)
    export_iou_spin.setDecimals(2)
    export_iou_spin.setSingleStep(0.05)
    export_iou_spin.setValue(float(self.settings.get("exportIou", 0.7) or 0.7))
    export_form.addWidget(export_nms_checkbox, 5, 0)
    export_form.addWidget(export_agnostic_checkbox, 5, 1)
    export_form.addWidget(QtWidgets.QLabel("Confidence"), 5, 2)
    export_form.addWidget(export_conf_spin, 5, 3)
    export_form.addWidget(QtWidgets.QLabel("IoU"), 5, 4)
    export_form.addWidget(export_iou_spin, 5, 5)
    export_extra_edit = QtWidgets.QLineEdit(str(self.settings.get("exportExtraArgs", "") or ""))
    export_extra_edit.setObjectName("trainer_export_extra_args")
    export_extra_edit.setPlaceholderText("opset=17 workspace=4 fraction=0.5")
    export_extra_edit.setToolTip(
        "Native Ultralytics key=value export arguments. Explicit arguments override the controls above. "
        "Use quotes around paths with spaces. Optional targets require their own platform/SDK dependencies."
    )
    export_form.addWidget(QtWidgets.QLabel("Extra arguments"), 6, 0)
    export_form.addWidget(export_extra_edit, 6, 1, 1, 5)
    export_form.setColumnStretch(1, 1)
    export_form.setColumnStretch(3, 1)
    export_layout.addLayout(export_form)

    export_readiness = QtWidgets.QLabel()
    export_readiness.setObjectName("trainer_export_readiness")
    export_readiness.setWordWrap(True)
    export_readiness.setTextInteractionFlags(Qt.TextSelectableByMouse)
    export_layout.addWidget(export_readiness)

    export_action_row = self._row_layout(8)
    export_button = QtWidgets.QPushButton("Export Model")
    export_button.setObjectName("trainer_export_button")
    export_open_output = QtWidgets.QPushButton("Open Output Folder")
    export_open_output.setObjectName("trainer_export_open_output")
    export_open_output.setEnabled(False)
    export_action_row.addWidget(export_button)
    export_action_row.addWidget(export_open_output)
    export_action_row.addStretch(1)
    export_layout.addLayout(export_action_row)

    export_status = QtWidgets.QLabel("Choose a trained checkpoint and export format.")
    export_status.setObjectName("trainer_export_status")
    export_status.setWordWrap(True)
    export_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
    export_status.setStyleSheet("color: #9aa4af; font-weight: 700;")
    export_layout.addWidget(export_status)
    export_layout.addStretch(1)
    evaluator_tabs.addTab(export_tab, "Export")

    export_state = {"output": "", "running": False}

    def refresh_export_controls():
        export_format = export_format_combo.currentText().strip().lower()
        precision_model = export_precision_combo.model()
        fp16_supported = self.is_parameter_supported(export_format, "half")
        int8_supported = self.is_parameter_supported(export_format, "int8")
        precision_model.item(0).setEnabled(self.is_parameter_supported(export_format, "fp32"))
        precision_model.item(1).setEnabled(fp16_supported)
        precision_model.item(2).setEnabled(int8_supported)
        precision_model.item(3).setEnabled(self.is_parameter_supported(export_format, "w8a16"))
        precision_model.item(4).setEnabled(self.is_parameter_supported(export_format, "w8a32"))
        if not precision_model.item(export_precision_combo.currentIndex()).isEnabled():
            for index in range(export_precision_combo.count()):
                if precision_model.item(index).isEnabled():
                    export_precision_combo.setCurrentIndex(index)
                    break
        export_dynamic_checkbox.setEnabled(self.is_parameter_supported(export_format, "dynamic"))
        export_simplify_checkbox.setEnabled(self.is_parameter_supported(export_format, "simplify"))
        batch_supported = self.is_parameter_supported(export_format, "batch")
        export_batch_checkbox.setEnabled(batch_supported)
        export_batch_spin.setEnabled(batch_supported and export_batch_checkbox.isChecked())
        nms_supported = self.is_parameter_supported(export_format, "nms")
        export_nms_checkbox.setEnabled(nms_supported)
        nms_options_enabled = nms_supported and export_nms_checkbox.isChecked()
        export_agnostic_checkbox.setEnabled(nms_options_enabled)
        export_conf_spin.setEnabled(nms_options_enabled)
        export_iou_spin.setEnabled(nms_options_enabled)

        checkpoint_path = self.normalize_path(export_checkpoint_edit.text().strip())
        checkpoint_ready = bool(
            checkpoint_path
            and os.path.isfile(checkpoint_path)
            and checkpoint_path.lower().endswith(".pt")
        )
        calibration_path = self.normalize_path(export_data_edit.text().strip())
        calibration_ready = bool(
            calibration_path
            and os.path.isfile(calibration_path)
            and calibration_path.lower().endswith((".yaml", ".yml"))
        )
        int8_selected = (
            str(export_precision_combo.currentData() or "fp32") in {"int8", "w8a16"}
            and self.is_parameter_supported(export_format, "data")
        )
        # Native overrides may supply data/precision (including QAT); the builder
        # performs the final validation rather than silently ignoring them.
        export_ready = checkpoint_ready and (not int8_selected or calibration_ready or bool(export_extra_edit.text().strip()))
        export_button.setEnabled(export_ready and not export_state.get("running", False))

        if not checkpoint_ready:
            message = "● Select any existing Ultralytics .pt weights to enable export."
            color = "#ff6b6b"
        elif int8_selected and not calibration_ready:
            message = (
                "● INT8 needs a calibration dataset YAML. Click Use Training Dataset "
                "or browse to the dataset that matches these weights."
            )
            color = "#f0b35a"
        elif int8_selected:
            message = (
                f"● INT8 ready — {os.path.basename(checkpoint_path)} will calibrate with "
                f"{os.path.basename(calibration_path)}."
            )
            color = "#45d483"
        elif calibration_ready and int8_supported:
            message = (
                f"● Ready to export {os.path.basename(checkpoint_path)}. "
                f"INT8 is also available with {os.path.basename(calibration_path)}."
            )
            color = "#45d483"
        else:
            message = f"● Ready to export {os.path.basename(checkpoint_path)}."
            color = "#45d483"
        export_readiness.setText(message)
        export_readiness.setStyleSheet(
            f"QLabel {{ color: {color}; background: #141c24; border: 1px solid {color}; "
            "border-radius: 5px; padding: 7px; font-weight: 700; }"
        )

    def browse_export_checkpoint():
        file_name, _selected_filter = QFileDialog.getOpenFileName(
            dialog,
            "Select Trained Ultralytics Checkpoint",
            self.dialog_start_directory(
                "export", export_checkpoint_edit.text().strip()
            ),
            "Ultralytics Checkpoints (*.pt);;All Files (*)",
        )
        if file_name:
            self.remember_dialog_selection("export", file_name)
            export_checkpoint_edit.setText(self.normalize_path(file_name))

    def use_latest_export_checkpoint():
        checkpoint = latest_run_export_checkpoint()
        if checkpoint:
            export_checkpoint_edit.setText(checkpoint)
            export_status.setText(f"Using trained checkpoint: {checkpoint}")
        else:
            QMessageBox.warning(dialog, "Export", "No usable best.pt or last.pt checkpoint was found.")

    def browse_export_data():
        file_name, _selected_filter = QFileDialog.getOpenFileName(
            dialog,
            "Select Calibration Dataset YAML",
            self.dialog_start_directory(
                "training", export_data_edit.text().strip()
            ),
            "Dataset YAML (*.yaml *.yml);;All Files (*)",
        )
        if file_name:
            self.remember_dialog_selection("training", file_name)
            export_data_edit.setText(self.normalize_path(file_name))

    def use_training_export_data():
        candidate = self.normalize_path(
            data_yaml_edit.text().strip()
            or current_data_yaml
            or self.settings.get("ultralyticsDataYamlPath", "")
        )
        if candidate and os.path.isfile(candidate):
            export_data_edit.setText(candidate)
            export_status.setText(f"Using training calibration dataset: {candidate}")
        else:
            QMessageBox.warning(
                dialog,
                "INT8 Calibration Dataset",
                "The current Train Setup does not contain an existing dataset YAML.",
            )

    def sync_legacy_export_state():
        export_format = export_format_combo.currentText().strip()
        precision = str(export_precision_combo.currentData() or "fp32")
        self.settings.update({
            "exportModelPath": self.normalize_path(export_checkpoint_edit.text().strip()),
            "exportFormat": export_format,
            "exportWidth": str(export_width_spin.value()),
            "exportHeight": str(export_height_spin.value()),
            "exportHalf": precision == "fp16",
            "exportInt8": precision == "int8",
            "exportPrecision": precision,
            "exportExtraArgs": export_extra_edit.text().strip(),
            "exportUseGpu": bool(export_gpu_checkbox.isChecked()),
            "exportDynamic": bool(export_dynamic_checkbox.isChecked()),
            "exportSimplify": bool(export_simplify_checkbox.isChecked()),
            "exportNms": bool(export_nms_checkbox.isChecked()),
            "exportAgnosticNms": bool(export_agnostic_checkbox.isChecked()),
            "exportBatchEnabled": bool(export_batch_checkbox.isChecked()),
            "exportBatchSize": int(export_batch_spin.value()),
            "exportConf": float(export_conf_spin.value()),
            "exportIou": float(export_iou_spin.value()),
        })
        legacy_values = {
            "imgsz_input_W": str(export_width_spin.value()),
            "imgsz_input_H": str(export_height_spin.value()),
        }
        for name, value in legacy_values.items():
            widget = getattr(self, name, None)
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(value)
        legacy_combo = getattr(self, "convert_model", None)
        if isinstance(legacy_combo, QtWidgets.QComboBox):
            index = legacy_combo.findText(export_format, Qt.MatchFixedString)
            if index >= 0:
                legacy_combo.setCurrentIndex(index)
        for name, checked in (
            ("half_true", precision == "fp16"),
            ("int8_true", precision == "int8"),
            ("use_gpu_checkbox", export_gpu_checkbox.isChecked()),
            ("dynamic_true", export_dynamic_checkbox.isChecked()),
            ("simplify", export_simplify_checkbox.isChecked()),
            ("nms_true_checkbox", export_nms_checkbox.isChecked()),
            ("agnostic_checkbox", export_agnostic_checkbox.isChecked()),
            ("batch_size_checkbox", export_batch_checkbox.isChecked()),
        ):
            widget = getattr(self, name, None)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(checked))
        for name, value in (
            ("batch_input_spinbox", export_batch_spin.value()),
            ("conf_spinbox", export_conf_spin.value()),
            ("iou_spinbox", export_iou_spin.value()),
        ):
            widget = getattr(self, name, None)
            if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                widget.setValue(value)
        self.queue_settings_save()

    def perform_trainer_export():
        try:
            current_worker = getattr(self, "training_export_worker", None)
            if current_worker is not None and current_worker.isRunning():
                QMessageBox.information(dialog, "Export", "A model export is already running.")
                return

            sync_legacy_export_state()
            params = self.build_ultralytics_export_params(
                export_format_combo.currentText(),
                export_width_spin.value(),
                export_height_spin.value(),
                use_gpu=export_gpu_checkbox.isChecked(),
                precision=export_precision_combo.currentData(),
                dynamic=export_dynamic_checkbox.isChecked(),
                simplify=export_simplify_checkbox.isChecked(),
                nms=export_nms_checkbox.isChecked(),
                agnostic_nms=export_agnostic_checkbox.isChecked(),
                confidence=export_conf_spin.value(),
                iou=export_iou_spin.value(),
                batch_enabled=export_batch_checkbox.isChecked(),
                batch=export_batch_spin.value(),
                data_yaml=export_data_edit.text().strip(),
                extra_args=export_extra_edit.text().strip(),
            )
            checkpoint_path = self.normalize_path(export_checkpoint_edit.text().strip())
            export_state["running"] = True
            refresh_export_controls()
            export_status.setText(
                f"Exporting {os.path.basename(checkpoint_path)} "
                f"to {export_format_combo.currentText()} in the background..."
            )

            worker = UltralyticsExportWorker(checkpoint_path, params, self)
            self.training_export_worker = worker

            def export_completed(output, model):
                output = self.normalize_path(output)
                # Export uses its own checkpoint; do not replace the live
                # annotation/video model when that background job completes.
                self.export_model_path = checkpoint_path
                self.settings["exportModelPath"] = checkpoint_path
                self.queue_settings_save()
                export_state["output"] = output
                export_state["running"] = False
                if sip.isdeleted(dialog):
                    return
                export_open_output.setEnabled(bool(output and os.path.exists(output)))
                export_status.setText(f"Export complete:\n{output}")
                refresh_export_controls()
                QMessageBox.information(dialog, "Export Complete", f"Exported model:\n{output}")

            def export_failed(message):
                export_state["running"] = False
                if sip.isdeleted(dialog):
                    return
                export_status.setText(f"Export failed: {message}")
                refresh_export_controls()
                QMessageBox.critical(dialog, "Export Failed", message)

            def export_finished():
                if getattr(self, "training_export_worker", None) is worker:
                    self.training_export_worker = None
                worker.deleteLater()

            worker.completed.connect(export_completed)
            worker.failed.connect(export_failed)
            worker.finished.connect(export_finished)
            try:
                start_gpu_worker(self, worker, "Model export", exclusive=True)
            except Exception:
                export_finished()
                raise
        except Exception as e:
            export_state["running"] = False
            export_status.setText(f"Export failed: {e}")
            QMessageBox.critical(dialog, "Export Failed", str(e))
            refresh_export_controls()

    def open_export_output_folder():
        output = self.normalize_path(export_state.get("output", ""))
        if not output:
            return
        directory = output if os.path.isdir(output) else os.path.dirname(output)
        if directory and os.path.isdir(directory):
            os.startfile(directory)

    export_checkpoint_browse.clicked.connect(browse_export_checkpoint)
    export_checkpoint_use_run.clicked.connect(use_latest_export_checkpoint)
    export_data_browse.clicked.connect(browse_export_data)
    export_data_use_training.clicked.connect(use_training_export_data)
    export_format_combo.currentIndexChanged.connect(lambda _index: refresh_export_controls())
    export_precision_combo.currentIndexChanged.connect(lambda _index: refresh_export_controls())
    export_batch_checkbox.toggled.connect(lambda _checked: refresh_export_controls())
    export_nms_checkbox.toggled.connect(lambda _checked: refresh_export_controls())
    export_checkpoint_edit.textChanged.connect(lambda _text: refresh_export_controls())
    export_data_edit.textChanged.connect(lambda _text: refresh_export_controls())
    export_extra_edit.textChanged.connect(lambda _text: refresh_export_controls())
    export_button.clicked.connect(perform_trainer_export)
    export_open_output.clicked.connect(open_export_output_folder)
    refresh_export_controls()

    layout.addWidget(evaluator_tabs, 1)

    run_widgets = {
        "tabs": evaluator_tabs,
        "charts_tab": charts_tab,
        "activity_indicator": activity_indicator,
        "run_state_label": run_state_label,
        "eta_label": eta_label,
        "eta_detail_label": eta_detail_label,
        "run_dir_label": run_dir_label,
        "metrics_label": metrics_label,
        "baseline_label": baseline_label,
        "resume_note_label": resume_note_label,
        "metrics_table": metrics_table,
        "log_text": log_text,
        "diagnosis_panel": diagnosis_panel,
        "diagnosis_light": diagnosis_light,
        "diagnosis_title": diagnosis_title,
        "diagnosis_detail": diagnosis_detail,
        "score_plot": score_plot,
        "loss_plot": loss_plot,
        "artifact_combo": artifact_combo,
        "artifact_preview": artifact_preview,
    }
    self.training_evaluator_run_widgets = run_widgets
    self._training_eval_update_diagnosis_widgets(
        run_widgets,
        {"state": "idle", "title": "Waiting for metrics", "detail": "No results.csv rows yet.", "issues": []},
    )
    self._training_eval_update_baseline_widget(run_widgets)
    artifact_combo.currentIndexChanged.connect(lambda _index: self._training_eval_set_artifact_preview(artifact_combo, artifact_preview))
    open_artifact_btn.clicked.connect(lambda _checked=False: self.open_training_eval_artifact_viewer(artifact_combo.currentData()))
    artifact_preview.mouseDoubleClickEvent = lambda _event: self.open_training_eval_artifact_viewer(artifact_combo.currentData())

    health_tab = QtWidgets.QWidget()
    health_layout = QtWidgets.QVBoxLayout(health_tab)
    health_layout.setContentsMargins(8, 8, 8, 8)
    health_layout.setSpacing(8)
    health_text = QtWidgets.QTextEdit()
    health_text.setReadOnly(True)
    health_text.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
    health_text.setPlaceholderText("Click Setup Check to verify CUDA, Ultralytics, files, dataset access, and TensorRT.")
    health_text.setText("Setup Check verifies the training environment and selected paths; it does not inspect annotation quality.")
    health_layout.addWidget(health_text, 1)
    evaluator_tabs.addTab(health_tab, "Setup Check")

    status_label = QtWidgets.QLabel("")
    status_label.setStyleSheet("color: #9aa4af;")
    body_scroll.setWidget(body_widget)
    outer_layout.addWidget(body_scroll, 1)
    outer_layout.addWidget(status_label)
    self.training_evaluator_status_label = status_label

    review_state = {
        "report": {},
        "report_path": "",
        "visible_issues": [],
        "process": None,
        "timer": None,
        "report_signature": None,
    }

    def active_review_checkpoint():
        selected = self.normalize_path(
            train_model_edit.text().strip()
            if use_training_weights.isChecked()
            else eval_weights_edit.text().strip()
        )
        if selected:
            return selected

        record = self._training_eval_latest_saved_run(training_only=True)
        run_dir = self.normalize_path((record or {}).get("run_dir", ""))
        for name in ("best.pt", "last.pt"):
            candidate = self.normalize_path(os.path.join(run_dir, "weights", name))
            if candidate and os.path.isfile(candidate):
                return candidate
        return ""

    def refresh_model_review_source(*_args):
        checkpoint = active_review_checkpoint()
        data_yaml = self.normalize_path(data_yaml_edit.text().strip())
        model_review_source_label.setText(
            f"<b>Checkpoint:</b> {html.escape(checkpoint or '(not selected)')}<br>"
            f"<b>Validation dataset YAML:</b> {html.escape(data_yaml or '(not selected)')}<br>"
            "Validation Review intentionally uses this YAML to resolve the selected split. "
            "Folder-based Dataset Analysis is a separate tool and never uses this YAML."
        )

    use_training_weights.toggled.connect(refresh_model_review_source)
    eval_weights_edit.textChanged.connect(refresh_model_review_source)
    train_model_edit.textChanged.connect(refresh_model_review_source)
    data_yaml_edit.textChanged.connect(refresh_model_review_source)
    refresh_model_review_source()

    def review_report_path_for(checkpoint, split):
        checkpoint = self.normalize_path(checkpoint)
        checkpoint_parent = os.path.dirname(checkpoint)
        if os.path.basename(checkpoint_parent).lower() == "weights":
            base_dir = os.path.dirname(checkpoint_parent)
        else:
            base_dir = self._training_eval_project_dir()
        folder_name = f"{Path(checkpoint).stem}_{str(split or 'val')}"
        return self.normalize_path(os.path.join(base_dir, "darkfusion_review", folder_name, "error_report.json"))

    def validation_review_history_path(data_yaml):
        """Keep review decisions with the active dataset, not beside an arbitrary YAML."""
        data_yaml = self.normalize_path(data_yaml or "")
        dataset_dir = self._training_eval_dataset_dir()
        if dataset_dir and data_yaml and not self._training_eval_yaml_matches_dataset(
            data_yaml, dataset_dir
        ):
            dataset_dir = ""
        if not dataset_dir and data_yaml and os.path.isfile(data_yaml):
            yaml_data, yaml_dir = self._training_health_parse_data_yaml(data_yaml)
            dataset_dir = self._training_health_dataset_dir(
                data_yaml, yaml_data, yaml_dir
            )
        if dataset_dir:
            return self.normalize_path(
                os.path.join(
                    dataset_metadata_directory(dataset_dir, create=True),
                    "validation_review_history.json",
                )
            )
        return self.normalize_path(
            os.path.join(
                dataset_metadata_directory(os.path.dirname(data_yaml), create=True),
                "validation_review_history.json",
            )
        )

    def review_issue_type_text(issue_type):
        return {
            "hard_negative": "Hard negative candidate",
            "unverified_prediction": "Unverified prediction",
            "missing_label_file": "Missing label file",
            "invalid_label_file": "Invalid label file",
            "unreadable_label_file": "Unreadable label file",
            "false_positive": "False positive",
            "false_negative": "Missed object",
            "wrong_class": "Wrong class",
            "weak_localization": "Weak localization",
            "duplicate_prediction": "Duplicate",
            "poor_keypoints": "Poor keypoints",
        }.get(str(issue_type or ""), str(issue_type or "").replace("_", " ").title())

    def selected_review_issue():
        row = review_issue_table.currentRow()
        if row < 0:
            return None
        item = review_issue_table.item(row, 0)
        return item.data(Qt.UserRole) if item is not None else None

    def display_selected_review_issue():
        issue = selected_review_issue()
        enabled = isinstance(issue, dict)
        review_open_labeler_btn.setEnabled(enabled)
        if not enabled:
            review_open_labeler_btn.setText("2. Review Problems in Label Maker")
            return
        current_row = review_issue_table.currentRow()
        review_open_labeler_btn.setText(
            f"2. Review from Finding {current_row + 1:,} in Label Maker"
        )

    def refresh_review_issue_table(select_issue_id=""):
        report = review_state.get("report", {}) or {}
        issues = list(report.get("issues", []) or [])
        filter_value = str(review_filter_combo.currentData() or "all")
        search = review_search_edit.text().strip().lower()
        visible = []
        for issue in issues:
            if filter_value == "unreviewed":
                if str(issue.get("review_status", "unreviewed")) != "unreviewed":
                    continue
            elif filter_value != "all" and str(issue.get("type", "")) != filter_value:
                continue
            haystack = f"{issue.get('image_path', '')} {issue.get('class_name', '')}".lower()
            if search and search not in haystack:
                continue
            visible.append(issue)
        review_state["visible_issues"] = visible
        display_issues = visible[:2000]
        review_issue_table.setUpdatesEnabled(False)
        review_issue_table.setRowCount(len(display_issues))
        selected_row = 0 if display_issues else -1
        try:
            for row, issue in enumerate(display_issues):
                values = [
                    review_issue_type_text(issue.get("type")),
                    str(issue.get("class_name", "")),
                    "" if issue.get("confidence") is None else f"{float(issue.get('confidence')):.3f}",
                    "" if issue.get("iou") is None else f"{float(issue.get('iou')):.3f}",
                    os.path.basename(str(issue.get("image_path", ""))),
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column == 0:
                        item.setData(Qt.UserRole, issue)
                    if str(issue.get("review_status", "unreviewed")) != "unreviewed":
                        item.setForeground(QBrush(QColor("#7f8a98")))
                    review_issue_table.setItem(row, column, item)
                if select_issue_id and str(issue.get("id")) == str(select_issue_id):
                    selected_row = row
        finally:
            review_issue_table.setUpdatesEnabled(True)
        if selected_row >= 0:
            review_issue_table.selectRow(selected_row)
        else:
            display_selected_review_issue()

    def update_review_summary():
        report = review_state.get("report", {}) or {}
        summary = report.get("summary", {}) or {}
        titles = {
            "hard_negative": "Hard Neg",
            "unverified_prediction": "Unverified",
            "false_positive": "False +",
            "false_negative": "Missed",
            "wrong_class": "Class",
            "weak_localization": "Weak",
            "duplicate_prediction": "Duplicate",
            "poor_keypoints": "Keypoints",
        }
        for key, label in review_summary_labels.items():
            label.setText(f"{titles[key]}: {int(summary.get(key, 0) or 0)}")
        label_counts = report.get("label_state_counts")
        if isinstance(label_counts, dict):
            if str(report.get("task", "")) == "classify":
                coverage_text = f"Labels checked: {int(label_counts.get('classification', 0)):,} images labeled by class folder."
            else:
                invalid = sum(int(label_counts.get(key, 0)) for key in ("invalid", "partial"))
                coverage_text = (
                    f"Labels checked: {int(label_counts.get('annotated', 0)):,} annotated; "
                    f"{int(label_counts.get('empty', 0)):,} empty; {int(label_counts.get('missing', 0)):,} missing; "
                    f"{invalid:,} invalid or incomplete; {int(label_counts.get('unreadable', 0)):,} unreadable. "
                    "Missing or unusable labels produce unverified predictions. Empty files need a visual check before confirming background."
                )
            review_label_state_label.setText(coverage_text)
        else:
            review_label_state_label.setText(
                "This report has no label coverage information. Run Find Validation Problems again to distinguish missing and empty labels."
                if report else ""
            )
        processed = int(report.get("processed_images", 0) or 0)
        total = int(report.get("total_images", 0) or 0)
        dataset_total = int(report.get("dataset_total_images", total) or total)
        status = str(report.get("status", "") or "")
        stage = str(report.get("stage", "") or "").replace("_", " ").strip().title()
        issue_count = len(report.get("issues", []) or [])
        issue_images = int(report.get("issue_image_count", 0) or 0)
        suppressed = int(report.get("suppressed_issue_count", 0) or 0)
        scan_start = int(report.get("scan_start_index", 0) or 0)
        if not issue_images and issue_count:
            issue_images = len({
                str(issue.get("image_path", ""))
                for issue in report.get("issues", []) or []
                if issue.get("image_path")
            })
        scan_scope = f"{processed:,}/{total:,} scanned"
        if total < dataset_total:
            scan_end = min(dataset_total, scan_start + total)
            scan_scope += (
                f" (dataset images {scan_start + 1:,}-{scan_end:,} "
                f"of {dataset_total:,})"
            )
        suppressed_text = f" · {suppressed:,} reviewed issue(s) skipped" if suppressed else ""
        scanner_name = os.path.basename(str(report.get("model", "") or "checkpoint"))
        review_status_label.setText(
            f"{stage or status.title() or 'Report'}: {scan_scope} · "
            f"{issue_count:,} issues on {issue_images:,} original images · "
            f"{scanner_name}"
            f"{suppressed_text}"
        )

    def load_review_report(report_path, preserve_selection=True):
        report_path = self.normalize_path(report_path)
        if not report_path or not os.path.isfile(report_path):
            return False
        selected = selected_review_issue() if preserve_selection else None
        selected_id = str((selected or {}).get("id", ""))
        try:
            with open(report_path, "r", encoding="utf-8", errors="replace") as handle:
                report = json.load(handle)
            if not isinstance(report, dict):
                raise ValueError("The report root is not an object.")
            if str(report.get("source", "")) in {"dataset_folder", "dataset_health"}:
                review_status_label.setText(
                    "That is a Dataset Analysis report. Open it from Dataset Analysis; "
                    "Validation Review accepts model-comparison reports only."
                )
                return False
            if str(report.get("scanner", "")) == "clip_teammate_marker":
                review_status_label.setText(
                    "The separate teammate review has been retired. Run Find Validation Problems "
                    "to review model-driven hard negatives instead."
                )
                return False
        except Exception as e:
            review_status_label.setText(f"Could not read validation report: {e}")
            return False
        review_state["report"] = report
        review_state["report_path"] = report_path
        self.settings["trainingValidationReviewLastReport"] = report_path
        try:
            stat = os.stat(report_path)
            review_state["report_signature"] = (int(stat.st_size), int(stat.st_mtime_ns))
        except Exception:
            review_state["report_signature"] = None
        update_review_summary()
        refresh_review_issue_table(selected_id)
        return True

    def browse_review_report():
        suggested_dir = os.path.dirname(
            review_state.get("report_path")
            or review_report_path_for(active_review_checkpoint(), review_split_combo.currentData())
        )
        start_dir = self.dialog_start_directory("training", suggested_dir)
        file_name, _selected_filter = QFileDialog.getOpenFileName(
            dialog,
            "Open Validation Review Report",
            start_dir,
            "Validation Reports (*.json);;All Files (*)",
        )
        if file_name:
            self.remember_dialog_selection("training", file_name)
            load_review_report(file_name, preserve_selection=False)

    def compare_review_report():
        current = review_state.get("report", {}) or {}
        if not current:
            QMessageBox.information(dialog, "Compare Validation Reports", "Open or generate the first report.")
            return
        start_dir = self.dialog_start_directory(
            "training", os.path.dirname(review_state.get("report_path", "") or self._training_eval_project_dir())
        )
        file_name, _selected_filter = QFileDialog.getOpenFileName(
            dialog,
            "Select Validation Report to Compare",
            start_dir,
            "Validation Reports (*.json);;All Files (*)",
        )
        if not file_name:
            return
        self.remember_dialog_selection("training", file_name)
        try:
            with open(file_name, "r", encoding="utf-8", errors="replace") as handle:
                other = json.load(handle)
        except Exception as e:
            QMessageBox.warning(dialog, "Compare Validation Reports", str(e))
            return
        keys = (
            "hard_negative",
            "false_positive",
            "false_negative",
            "wrong_class",
            "weak_localization",
            "duplicate_prediction",
            "poor_keypoints",
        )
        current_summary = current.get("summary", {}) or {}
        other_summary = other.get("summary", {}) or {}
        lines = [
            f"Current: {os.path.basename(str(current.get('model', '') or 'model'))}",
            f"Compare: {os.path.basename(str(other.get('model', '') or 'model'))}",
            "",
            "Negative delta means the current report has fewer errors.",
            "",
        ]
        for key in keys:
            first = int(current_summary.get(key, 0) or 0)
            second = int(other_summary.get(key, 0) or 0)
            lines.append(f"{review_issue_type_text(key)}: {first} vs {second}  ({first - second:+d})")
        QMessageBox.information(dialog, "Validation Report Comparison", "\n".join(lines))

    def export_review_csv():
        report = review_state.get("report", {}) or {}
        issues = list(report.get("issues", []) or [])
        if not issues:
            QMessageBox.information(dialog, "Export Validation Review", "There are no report issues to export.")
            return
        default_path = os.path.splitext(review_state.get("report_path", "validation_review.json"))[0] + ".csv"
        file_name, _selected_filter = QFileDialog.getSaveFileName(
            dialog,
            "Export Validation Review CSV",
            default_path,
            "CSV Files (*.csv)",
        )
        if not file_name:
            return
        self.remember_dialog_selection("export", file_name)
        fields = [
            "id", "review_status", "type", "task", "class_id", "class_name",
            "confidence", "iou", "severity", "image_path", "label_path", "label_state", "invalid_label_lines", "detail",
        ]
        try:
            with open(file_name, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for issue in issues:
                    writer.writerow({field: issue.get(field, "") for field in fields})
        except Exception as e:
            QMessageBox.warning(dialog, "Export Validation Review", str(e))
            return
        review_status_label.setText(f"Exported {len(issues)} issues to {file_name}")

    def stop_review_analysis():
        process = review_state.get("process")
        if process is None:
            return
        try:
            parent = psutil.Process(int(process.pid))
            children = parent.children(recursive=True)
            for child in reversed(children):
                child.terminate()
            parent.terminate()
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
        review_status_label.setText("Stopping validation analysis...")

    def poll_review_analysis():
        process = review_state.get("process")
        report_path = review_state.get("report_path", "")
        try:
            stat = os.stat(report_path)
            signature = (int(stat.st_size), int(stat.st_mtime_ns))
        except Exception:
            signature = None
        if signature and signature != review_state.get("report_signature"):
            load_review_report(report_path)
        if process is None:
            return
        exit_code = process.poll()
        if exit_code is None:
            return
        timer = review_state.get("timer")
        if timer is not None:
            timer.stop()
        review_state["process"] = None
        review_analyze_btn.setEnabled(True)
        review_stop_btn.setEnabled(False)
        if os.path.isfile(report_path):
            load_review_report(report_path)
        if exit_code == 0:
            review_status_label.setText(
                review_status_label.text()
                + " · Ready: select a finding, then open it in Label Maker."
            )
        else:
            review_status_label.setText(
                f"Validation analysis stopped with exit code {exit_code}. "
                f"Partial results remain available in {report_path}."
            )

    def analyze_review_errors():
        # Qt slots otherwise print preparation errors only to the console,
        # leaving the button looking as though it did nothing.
        process = review_state.get("process")
        if process is not None and process.poll() is None:
            review_status_label.setText("Validation analysis is already running.")
            return
        review_status_label.setText("Preparing validation problem scan...")
        review_analyze_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            start_review_analysis()
        except Exception as error:
            logger.exception("Could not start validation problem scan")
            review_status_label.setText(f"Could not start validation analysis: {error}")
            QMessageBox.warning(dialog, "Validation Review", str(error))
        finally:
            process = review_state.get("process")
            if process is None or process.poll() is not None:
                review_analyze_btn.setEnabled(True)
                review_stop_btn.setEnabled(False)

    def start_review_analysis():
        checkpoint = active_review_checkpoint()
        data_yaml = self.normalize_path(data_yaml_edit.text().strip())
        task = str(task_setup_combo.currentText() or "detect").strip().lower()
        split = str(review_split_combo.currentData() or "val")
        if not checkpoint or not os.path.isfile(checkpoint):
            QMessageBox.warning(dialog, "Validation Review", "Select an existing .pt model in Train Setup first.")
            return
        if not data_yaml or not os.path.isfile(data_yaml):
            QMessageBox.warning(dialog, "Validation Review", "Select an existing dataset YAML first.")
            return
        active_record = self._training_eval_latest_saved_run(training_only=True)
        if self._training_eval_run_is_active(active_record or {}):
            QMessageBox.information(
                dialog,
                "Training Is Active",
                "Training has priority for GPU memory. Finish or stop training before starting a separate validation analysis.",
            )
            return
        script_path = os.path.join(APP_DIR, "darkfusion_validation_review.py")
        if not os.path.isfile(script_path):
            QMessageBox.warning(dialog, "Validation Review", f"Analyzer not found:\n{script_path}")
            return
        report_path = review_report_path_for(checkpoint, split)
        review_dir = os.path.dirname(report_path)
        os.makedirs(review_dir, exist_ok=True)
        batch_state_path = self.normalize_path(
            os.path.join(review_dir, "validation_batch_state.json")
        )
        history_path = validation_review_history_path(data_yaml)
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        scan_limit = int(review_limit_spin.value())
        scan_start = 0
        cursor_source = batch_state_path if os.path.isfile(batch_state_path) else report_path
        if scan_limit > 0 and os.path.isfile(cursor_source):
            try:
                with open(cursor_source, "r", encoding="utf-8", errors="replace") as handle:
                    previous_report = json.load(handle) or {}
                same_scan = (
                    self.normalize_path(previous_report.get("model", "")) == checkpoint
                    and self.normalize_path(previous_report.get("data", "")) == data_yaml
                    and str(previous_report.get("split", "")) == split
                    and str(previous_report.get("task", "")) == task
                    and int(previous_report.get("scan_limit", 0) or 0) == scan_limit
                )
                completed_state = cursor_source == batch_state_path
                if same_scan and (
                    completed_state or str(previous_report.get("status", "")) == "complete"
                ):
                    scan_start = max(0, int(previous_report.get("next_start_index", 0) or 0))
            except Exception as e:
                logger.debug("Could not continue prior validation review batch: %s", e)
        command = [
            sys.executable,
            script_path,
            f"--model={checkpoint}",
            f"--data={data_yaml}",
            f"--task={task}",
            f"--split={split}",
            f"--output={report_path}",
            f"--conf={float(review_conf_spin.value())}",
            f"--match-iou={float(review_iou_spin.value())}",
            f"--good-iou={max(float(review_iou_spin.value()), 0.75)}",
            f"--imgsz={int(imgsz_spin.value())}",
            f"--device={'0' if torch.cuda.is_available() else 'cpu'}",
            f"--max-images={scan_limit}",
            f"--start-index={scan_start}",
            f"--review-history={history_path}",
            f"--batch-state={batch_state_path}",
        ]
        try:
            process, _run_dir, _log_path = self.launch_training_evaluator_logged_command(
                command,
                "Validation Error Review",
                run_dir=review_dir,
            )
        except Exception as e:
            QMessageBox.warning(dialog, "Validation Review", str(e))
            return
        review_state["process"] = process
        review_state["report_path"] = report_path
        review_state["report_signature"] = None
        self.settings["trainingValidationReviewConfidence"] = float(review_conf_spin.value())
        self.settings["trainingValidationReviewMatchIou"] = float(review_iou_spin.value())
        self.settings["trainingValidationReviewLimit"] = int(review_limit_spin.value())
        self.settings["trainingValidationReviewSplit"] = split
        self.queue_settings_save()
        review_analyze_btn.setEnabled(False)
        review_stop_btn.setEnabled(True)
        evaluator_tabs.setCurrentWidget(validation_review_tab)
        image_scope = "every original image" if scan_limit == 0 else (
            f"the next {scan_limit:,} original images starting at {scan_start + 1:,}"
        )
        review_status_label.setText(
            f"Finding prediction/label disagreements across {image_scope} in the "
            f"{str(review_split_combo.currentText()).lower()} split..."
        )
        timer = review_state.get("timer")
        if timer is None:
            timer = QTimer(dialog)
            timer.setInterval(1500)
            timer.timeout.connect(poll_review_analysis)
            review_state["timer"] = timer
        timer.start()
        QTimer.singleShot(300, poll_review_analysis)

    def reset_review_history():
        data_yaml = self.normalize_path(data_yaml_edit.text().strip())
        if not data_yaml:
            return
        history_path = validation_review_history_path(data_yaml)
        if not os.path.isfile(history_path):
            review_status_label.setText("No saved validation review decisions were found.")
            return
        reply = QMessageBox.question(
            dialog,
            "Reset Reviewed Validation Issues",
            (
                "Allow previously kept, ignored, accepted, and manually edited issues "
                "to appear in future validation reviews again?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            os.remove(history_path)
            review_status_label.setText("Reviewed validation history was reset.")
        except Exception as e:
            QMessageBox.warning(dialog, "Reset Reviewed", str(e))

    def open_selected_review_in_labeler():
        issue = selected_review_issue()
        if not isinstance(issue, dict):
            return
        self.open_validation_review_issue_in_labeler(
            issue,
            review_state.get("report_path", ""),
            list(review_state.get("visible_issues", []) or []),
        )

    review_analyze_btn.clicked.connect(analyze_review_errors)
    review_stop_btn.clicked.connect(stop_review_analysis)
    review_load_btn.clicked.connect(browse_review_report)
    review_compare_btn.clicked.connect(compare_review_report)
    review_export_btn.clicked.connect(export_review_csv)
    review_reset_history_btn.clicked.connect(reset_review_history)
    review_filter_combo.currentIndexChanged.connect(lambda _index: refresh_review_issue_table())
    review_search_edit.textChanged.connect(lambda _text: refresh_review_issue_table())
    review_issue_table.itemSelectionChanged.connect(display_selected_review_issue)
    review_issue_table.itemDoubleClicked.connect(lambda _item: open_selected_review_in_labeler())
    review_open_labeler_btn.clicked.connect(open_selected_review_in_labeler)

    def autoload_validation_review_report():
        candidates = [
            self.normalize_path(self.settings.get("trainingValidationReviewLastReport", "")),
            review_report_path_for(
                active_review_checkpoint(),
                review_split_combo.currentData(),
            ),
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate) and load_review_report(candidate, preserve_selection=False):
                break

    QTimer.singleShot(0, autoload_validation_review_report)

    def refresh_setup_model_controls():
        model_path = self.normalize_path(train_model_edit.text().strip())
        uses_yaml = model_path.lower().endswith((".yaml", ".yml"))
        pretrained_edit.setEnabled(uses_yaml)
        browse_pretrained_btn.setEnabled(uses_yaml)
        clear_pretrained_btn.setEnabled(uses_yaml)
        if not uses_yaml and pretrained_edit.text().strip():
            pretrained_edit.clear()

    def refresh_advanced_train_controls():
        train_patience_spin.setEnabled(bool(train_patience_checkbox.isChecked()))
        train_freeze_spin.setEnabled(bool(train_freeze_checkbox.isChecked()))
        distill_enabled = bool(distill_checkbox.isChecked())
        distill_model_edit.setEnabled(distill_enabled)
        browse_distill_btn.setEnabled(distill_enabled)
        clear_distill_btn.setEnabled(distill_enabled)
        distill_weight_spin.setEnabled(distill_enabled)
        tune_distill_checkbox.setEnabled(distill_enabled)
        if not distill_enabled and tune_distill_checkbox.isChecked():
            with blocked_signals(tune_distill_checkbox):
                tune_distill_checkbox.setChecked(False)
        if distill_enabled and not self.ultralytics_version_at_least("8.4.77"):
            distill_checkbox.setToolTip(
                "Knowledge Distillation requires Ultralytics v8.4.77+. "
                "Your current fusion environment is older; upgrade before training with distill_model."
            )
        else:
            distill_checkbox.setToolTip(
                "Use Ultralytics Knowledge Distillation with distill_model=. "
                "Detect is benchmarked first; segment, pose, and OBB are supported but less validated."
            )

    def refresh_validation_controls():
        mode = self._training_eval_validation_mode(validation_mode_combo.currentData())
        uses_source = mode in {"folder", "txt"}
        # Validation percent now applies to all modes, always enabled
        validation_percent_combo.setEnabled(True)
        validation_source_edit.setEnabled(uses_source)
        browse_validation_source_btn.setEnabled(uses_source)
        if mode == "folder":
            validation_source_edit.setPlaceholderText("Validation image folder")
            browse_validation_source_btn.setText("Folder")
        elif mode == "txt":
            validation_source_edit.setPlaceholderText("valid.txt")
            browse_validation_source_btn.setText("TXT")
        else:
            validation_source_edit.setPlaceholderText("Optional validation folder or valid.txt")
            browse_validation_source_btn.setText("Source")

    def overlay_current_setup_on_result(result):
        if not isinstance(result, dict):
            return result
        rec = result.setdefault("recommendations", {})
        data_yaml = self.normalize_path(data_yaml_edit.text().strip())
        model_path = self.normalize_path(train_model_edit.text().strip())
        rec["data_yaml_path"] = data_yaml
        result["data_yaml_path"] = data_yaml
        result["training_recommendations_yaml"] = self.normalize_path(recommended_args_display.text().strip())
        rec["task"] = str(task_setup_combo.currentText()).strip().lower() or "detect"
        rec["imgsz"] = int(imgsz_spin.value())
        rec["batch"] = train_batch_spin.value()
        rec["epochs"] = int(train_epochs_spin.value())
        rec["amp"] = _training_precision(self, result, optimized_args_checkbox.isChecked())
        rec["rect"] = bool(train_rect_checkbox.isChecked())
        rec["patience"] = int(train_patience_spin.value()) if train_patience_checkbox.isChecked() else int(rec.get("patience", 40) or 40)
        rec["freeze"] = int(train_freeze_spin.value()) if train_freeze_checkbox.isChecked() else 0
        existing_weight_path = self.normalize_path(rec.get("weights_path", ""))
        if model_path and (model_path.lower().endswith(".pt") or not existing_weight_path):
            rec["weights_path"] = model_path
        return result

    def sync_training_setup_to_state(update_result=True):
        data_yaml = self.normalize_path(data_yaml_edit.text().strip())
        model_path = self.normalize_path(train_model_edit.text().strip())
        pretrained_path = self.normalize_path(pretrained_edit.text().strip())
        selected_task = str(task_setup_combo.currentText()).strip().lower() or "detect"

        self.data_yaml_path = data_yaml or None
        runs_dir = self.normalize_path(runs_dir_edit.text().strip())
        if not runs_dir:
            runs_dir = self._training_eval_default_project_dir()
            with blocked_signals(runs_dir_edit):
                runs_dir_edit.setText(runs_dir)
        self.runs_directory = runs_dir
        if model_path.lower().endswith(".pt"):
            self.pt_path = model_path
            self.model_config_path = None
            self.pretrained_model_path = None
            pretrained_path = ""
        elif model_path.lower().endswith((".yaml", ".yml")):
            self.model_config_path = model_path
            self.pt_path = None
            self.pretrained_model_path = pretrained_path or None
        elif model_path:
            self.pt_path = model_path
            self.model_config_path = None
            self.pretrained_model_path = None
        else:
            self.pt_path = None
            self.model_config_path = None
            self.pretrained_model_path = None

        if pretrained_edit.text().strip() != pretrained_path:
            with blocked_signals(pretrained_edit):
                pretrained_edit.setText(pretrained_path)

        if getattr(self, "yaml_label", None) is not None:
            self.yaml_label.setText(f"Data YAML: {data_yaml}" if data_yaml else "No Data YAML selected")
        if getattr(self, "pt_label", None) is not None:
            if self.model_config_path:
                label = f"Model Config: {self.model_config_path}"
                if self.pretrained_model_path:
                    label += f"; Pretrained: {self.pretrained_model_path}"
                self.pt_label.setText(label)
            elif self.pt_path:
                self.pt_label.setText(f"Model: {self.pt_path}")
            else:
                self.pt_label.setText("No Model selected")
        if getattr(self, "task_combobox", None) is not None:
            index = self.task_combobox.findText(selected_task, Qt.MatchFixedString)
            if index >= 0:
                self.task_combobox.setCurrentIndex(index)
        for widget_name, value in (
            ("imgsz_input", int(imgsz_spin.value())),
            ("batch_input", train_batch_spin.value()),
            ("epochs_input", int(train_epochs_spin.value())),
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None and hasattr(widget, "setText"):
                widget.setText(str(value))
        for widget, checked in (
            (getattr(self, "rect_train_checkbox", None), train_rect_checkbox.isChecked()),
            (getattr(self, "keep_duplicate_labels_checkbox", None), duplicate_labels_checkbox.isChecked()),
            (getattr(self, "patience_checkbox", None), train_patience_checkbox.isChecked()),
            (getattr(self, "freeze_checkbox", None), train_freeze_checkbox.isChecked()),
        ):
            if widget is not None and hasattr(widget, "setChecked"):
                widget.setChecked(bool(checked))
        if getattr(self, "patience_input", None) is not None and hasattr(self.patience_input, "setValue"):
            self.patience_input.setValue(int(train_patience_spin.value()))
        if getattr(self, "freeze_input", None) is not None and hasattr(self.freeze_input, "setValue"):
            self.freeze_input.setValue(int(train_freeze_spin.value()))

        # extra_args no longer used; cfg parameter handles all hyperparameters
        self.ultralytics_distill_enabled = bool(distill_checkbox.isChecked())
        self.ultralytics_distill_model_path = self.normalize_path(distill_model_edit.text().strip())
        try:
            self.ultralytics_distill_weight = float(distill_weight_spin.value())
        except Exception:
            self.ultralytics_distill_weight = 6.0
        refresh_setup_model_controls()
        refresh_advanced_train_controls()

        self.settings["ultralyticsTask"] = selected_task
        self.settings["ultralyticsTrainImgSize"] = str(int(imgsz_spin.value()))
        self.settings["ultralyticsTrainBatch"] = str(train_batch_spin.value())
        self.settings["ultralyticsTrainEpochs"] = str(int(train_epochs_spin.value()))
        self.settings["ultralyticsRunsDirectory"] = runs_dir
        self.settings["ultralyticsDataYamlPath"] = data_yaml
        self.settings["ultralyticsPtPath"] = self.pt_path or ""
        self.settings["ultralyticsModelConfigPath"] = self.model_config_path or ""
        self.settings["ultralyticsPretrainedModelPath"] = self.pretrained_model_path or ""
        self.settings["ultralyticsRect"] = bool(train_rect_checkbox.isChecked())
        self.settings["ultralyticsKeepDuplicateLabels"] = bool(duplicate_labels_checkbox.isChecked())
        self.settings["ultralyticsPatienceEnabled"] = bool(train_patience_checkbox.isChecked())
        self.settings["ultralyticsPatience"] = int(train_patience_spin.value())
        self.settings["ultralyticsFreezeEnabled"] = bool(train_freeze_checkbox.isChecked())
        self.settings["ultralyticsFreeze"] = int(train_freeze_spin.value())
        self.settings["ultralyticsDistillEnabled"] = bool(distill_checkbox.isChecked())
        self.settings["ultralyticsDistillModelPath"] = self.ultralytics_distill_model_path
        self.settings["ultralyticsDistillWeight"] = float(self.ultralytics_distill_weight)
        self.settings["ultralyticsExtraTrainArgs"] = self.ultralytics_extra_train_args
        self.settings["ultralyticsAmp"] = _training_precision(self, use_recommended=optimized_args_checkbox.isChecked())
        if update_result:
            result = getattr(self, "training_evaluator_last_result", None)
            if isinstance(result, dict):
                self.training_evaluator_last_result = overlay_current_setup_on_result(result)

    def apply_result_to_training_controls(result):
        rec = (result or {}).get("recommendations", {})
        if not rec:
            return
        with blocked_signals(imgsz_spin):
            imgsz_spin.setValue(int(rec.get("imgsz", imgsz_spin.value()) or imgsz_spin.value()))
        with blocked_signals(train_batch_spin):
            train_batch_spin.setValue(rec.get("batch", train_batch_spin.value()))
        if batch_override_checkbox.isChecked():
            with blocked_signals(batch_override_spin):
                batch_override_spin.setValue(train_batch_spin.value())
        with blocked_signals(train_epochs_spin):
            train_epochs_spin.setValue(int(rec.get("epochs", train_epochs_spin.value()) or train_epochs_spin.value()))
        with blocked_signals(train_patience_spin):
            train_patience_spin.setValue(int(rec.get("patience", train_patience_spin.value()) or train_patience_spin.value()))
        with blocked_signals(train_rect_checkbox):
            train_rect_checkbox.setChecked(bool(rec.get("rect", train_rect_checkbox.isChecked())))
        freeze_value = int(rec.get("freeze", 0) or 0)
        with blocked_signals(train_freeze_checkbox):
            train_freeze_checkbox.setChecked(freeze_value > 0)
        with blocked_signals(train_freeze_spin):
            train_freeze_spin.setValue(freeze_value)
        data_yaml = self.normalize_path(rec.get("data_yaml_path", "") or result.get("data_yaml_path", ""))
        if data_yaml:
            with blocked_signals(data_yaml_edit):
                data_yaml_edit.setText(data_yaml)
        recommended_args_yaml = str(result.get("training_recommendations_yaml", "") or "").strip()
        with blocked_signals(recommended_args_display):
            recommended_args_display.setText(recommended_args_yaml)
        task = str(rec.get("task", "") or "").strip().lower()
        if task:
            index = task_setup_combo.findText(task, Qt.MatchFixedString)
            if index >= 0:
                with blocked_signals(task_setup_combo):
                    task_setup_combo.setCurrentIndex(index)
        optimized_text = str(rec.get("train_args_text", "") or "").strip()
        if optimized_args_checkbox.isChecked() and optimized_text:
            # cfg YAML file updated by tuner, no UI update needed
            pass
        sync_settings()

    def browse_setup_data_yaml():
        file_name = self.open_file_dialog("Select Training YAML", "YAML Files (*.yaml *.yml);;All Files (*)")
        if not file_name:
            return
        data_yaml_edit.setText(self.normalize_path(file_name))
        auto_yaml_checkbox.setChecked(False)
        sync_settings()

    def sync_task_from_loaded_weights(weights_path, label="model"):
        weights_path = self.normalize_path(weights_path or "")
        if not weights_path or not os.path.exists(weights_path):
            return False

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            task, source, info = self.training_eval_task_from_weights(weights_path)
        finally:
            QApplication.restoreOverrideCursor()

        if not task:
            detail = str(info.get("error", "") or "The model task could not be detected.")
            status_label.setText(f"Could not set task from {label}: {detail}")
            logger.warning("Training Evaluator could not detect task for %s: %s", weights_path, detail)
            return False

        index = task_setup_combo.findText(task, Qt.MatchFixedString)
        if index < 0:
            status_label.setText(f"Unsupported model task detected: {task}")
            return False

        changed = task_setup_combo.currentIndex() != index
        if changed:
            with blocked_signals(task_setup_combo):
                task_setup_combo.setCurrentIndex(index)

        self.training_evaluator_last_result = None
        sync_training_setup_to_state(update_result=False)
        status_label.setText(
            f"Task set to {task} from {os.path.basename(weights_path)} ({source}). "
            "Run Generate Files + Parameters to rebuild files and recommendations."
        )
        logger.info(
            "Training Evaluator task set to %s from %s using %s.",
            task,
            weights_path,
            source,
        )
        return changed

    def browse_eval_weights():
        was_using_training_model = use_training_weights.isChecked()
        file_name = self.browse_training_eval_weights(eval_weights_edit, use_training_weights)
        if file_name and not was_using_training_model:
            sync_task_from_loaded_weights(file_name, label="evaluation weights")
        sync_settings()

    def browse_setup_model():
        file_name = self.open_file_dialog("Select Training Model", "Model Files (*.pt *.yaml *.yml);;All Files (*)")
        if not file_name:
            return
        file_name = self.normalize_path(file_name)
        train_model_edit.setText(file_name)
        if file_name.lower().endswith(".pt"):
            pretrained_edit.clear()
        sync_task_from_loaded_weights(file_name, label="training model")
        sync_settings()

    def browse_setup_pretrained():
        file_name = self.open_file_dialog("Select Pretrained Model", "Model Files (*.pt);;All Files (*)")
        if not file_name:
            return
        pretrained_edit.setText(self.normalize_path(file_name))
        sync_settings()

    def browse_setup_distill_model():
        file_name = self.open_file_dialog("Select Teacher Model", "Model Files (*.pt);;All Files (*)")
        if not file_name:
            return
        distill_model_edit.setText(self.normalize_path(file_name))
        distill_checkbox.setChecked(True)
        sync_settings()

    def browse_setup_runs_dir():
        sync_training_setup_to_state(update_result=False)
        start_dir = self.dialog_start_directory(
            "training", runs_dir_edit.text().strip() or self._training_eval_project_dir()
        )
        directory = QFileDialog.getExistingDirectory(dialog, "Select Training Runs Folder", start_dir)
        directory = self.normalize_path(directory or "")
        if not directory:
            return
        self.remember_dialog_selection("training", directory)
        os.makedirs(directory, exist_ok=True)
        runs_dir_edit.setText(directory)

    def browse_recommended_args_yaml():
        file_name = self.open_file_dialog("Select Training Recommendations YAML", "YAML Files (*.yaml *.yml);;All Files (*)")
        if not file_name:
            return
        file_name = self.normalize_path(file_name)
        # textChanged already synchronizes the selected path once. Browsing
        # must not generate recommendations or try to open a runs folder.
        recommended_args_display.setText(file_name)

    def refresh_hyperparameters(force=False):
        from darkfusion_tune_settings import read_training_configuration, training_hyperparameter_values
        path = self.normalize_path(recommended_args_display.text().strip())
        if hyperparameters.is_dirty() and not force and path == getattr(hyperparameters, "path", ""):
            return
        if not path or not os.path.isfile(path):
            hyperparameters.set_unavailable()
            hyperparameter_summary.setText("Hyperparameters: generate or select a training configuration YAML.")
            return
        try:
            values = training_hyperparameter_values(read_training_configuration(path))
            active = optimized_args_checkbox.isChecked()
            task = task_setup_combo.currentText().strip().lower()
            hyperparameters.set_values(values, path, task, enabled=active)
            precision = {True: "FP16 AMP", False: "FP32", "bf16": "BF16 AMP", "fp16": "FP16 AMP", "fp32": "FP32"}.get(values.get("amp"), str(values.get("amp")))
            qat = " · INT8 QAT" if values.get("quantize") in (8, "int8", "8", "w8a8") else ""
            learning = "Automatic learning" if values.get('optimizer') == 'auto' else "Manual learning"
            augmentation = (f"image mixing {float(values.get('mosaic', 0)) * 100:g}% · "
                            f"brightness variation {float(values.get('hsv_v', 0)) * 100:g}%"
                            if task != 'classify' else f"augmentation: {values.get('auto_augment')}")
            hyperparameter_summary.setText(
                f"Hyperparameters: {precision}{qat} · {learning} · {augmentation}"
                if active else "Hyperparameters: selected YAML is inactive. Enable Recommended Args to use and edit it.")
        except (ValueError, TypeError, OverflowError, OSError, yaml.YAMLError) as error:
            hyperparameters.set_unavailable(str(error))
            hyperparameter_summary.setText("Hyperparameters: selected configuration could not be read.")

    def save_hyperparameters(changes):
        from darkfusion_tune_settings import save_training_hyperparameters
        path = self.normalize_path(recommended_args_display.text().strip())
        try:
            if not optimized_args_checkbox.isChecked() or path != getattr(hyperparameters, "path", ""):
                raise ValueError("Reload the active training configuration before saving.")
            payload = save_training_hyperparameters(path, changes, self._training_eval_write_text_atomic,
                data_yaml=data_yaml_edit.text().strip(), baseline=hyperparameters.source_values)
            self.settings["ultralyticsAmp"] = payload.get("amp", True)
            result = getattr(self, "training_evaluator_last_result", None)
            if isinstance(result, dict):
                rec = result.setdefault("recommendations", {})
                rec["amp"] = payload.get("amp", True)
                rec.setdefault("hyperparams", {}).update(changes)
                self._training_eval_rebuild_train_args(rec)
                results_text.setText(self.format_training_evaluation_report(result))
            sync_settings()
            refresh_hyperparameters(force=True)
            status_label.setText(f"Saved {len(changes)} hyperparameter change(s) to {os.path.basename(path)}")
            return True
        except (ValueError, OSError, yaml.YAMLError, TypeError) as error:
            QMessageBox.warning(dialog, "Hyperparameters", str(error))
            return False

    def browse_validation_source():
        mode = self._training_eval_validation_mode(validation_mode_combo.currentData())
        if mode == "folder":
            start_dir = self.dialog_start_directory(
                "dataset", validation_source_edit.text().strip() or self._training_eval_dataset_dir()
            )
            directory = QFileDialog.getExistingDirectory(dialog, "Select Validation Image Folder", start_dir)
            directory = self.normalize_path(directory or "")
            if directory:
                self.remember_dialog_selection("dataset", directory)
                validation_source_edit.setText(directory)
                sync_settings()
        elif mode == "txt":
            file_name = self.open_file_dialog("Select Validation TXT", "Text Files (*.txt);;All Files (*)")
            if file_name:
                validation_source_edit.setText(self.normalize_path(file_name))
                sync_settings()

    def refresh_latest_run_from_disk():
        update_training_control_buttons()
        record = None
        active_record = getattr(self, "training_evaluator_active_run", None)
        active_process = (active_record or {}).get("process") if isinstance(active_record, dict) else None
        if active_process is not None:
            try:
                if active_process.poll() is None:
                    record = active_record
            except Exception:
                pass
        if record is None:
            record = self._training_eval_latest_saved_run()
        if not record:
            return
        title_text = str(record.get("title") or "Training")
        active = self._training_eval_run_is_active(record)
        message = f"{title_text} running..." if active else f"{title_text} results loaded."
        self._training_eval_refresh_run_widgets(
            run_widgets,
            record.get("run_dir", ""),
            record.get("log_path", ""),
            message,
            include_diagnosis=False,
        )

    latest_run_timer = QTimer(dialog)
    latest_run_timer.setInterval(5000)
    latest_run_timer.timeout.connect(refresh_latest_run_from_disk)
    latest_run_timer.start()
    QTimer.singleShot(0, refresh_latest_run_from_disk)

    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
    evaluate_btn = buttons.addButton("Generate Files + Parameters", QtWidgets.QDialogButtonBox.ActionRole)
    evaluate_btn.setToolTip(
        "Rescan the loaded dataset, overwrite train/valid metadata files, rebuild obj.yaml, "
        "and calculate current training parameters."
    )
    health_check_btn = buttons.addButton("Setup Check", QtWidgets.QDialogButtonBox.ActionRole)
    preview_cmd_btn = buttons.addButton("Commands", QtWidgets.QDialogButtonBox.ActionRole)
    calibrate_batch_btn = buttons.addButton("Calibrate Batch", QtWidgets.QDialogButtonBox.ActionRole)
    tune_btn = buttons.addButton("Tune HParams", QtWidgets.QDialogButtonBox.ActionRole)
    resume_btn = buttons.addButton("Resume Last Run", QtWidgets.QDialogButtonBox.ActionRole)
    resume_btn.setEnabled(False)
    train_btn = buttons.addButton("Start Training", QtWidgets.QDialogButtonBox.ActionRole)
    stop_after_epoch_btn = buttons.addButton("Stop After Epoch", QtWidgets.QDialogButtonBox.ActionRole)
    stop_now_btn = buttons.addButton("Stop Now", QtWidgets.QDialogButtonBox.ActionRole)
    open_run_btn = buttons.addButton("Open Run Folder", QtWidgets.QDialogButtonBox.ActionRole)
    open_best_btn = buttons.addButton("Open best.pt", QtWidgets.QDialogButtonBox.ActionRole)
    stop_after_epoch_btn.setEnabled(False)
    stop_now_btn.setEnabled(False)
    open_run_btn.setEnabled(False)
    open_best_btn.setEnabled(False)
    buttons.rejected.connect(dialog.reject)
    buttons_scroll = QtWidgets.QScrollArea(dialog)
    buttons_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    buttons_scroll.setWidgetResizable(False)
    buttons_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    buttons_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    buttons_scroll.setMinimumWidth(0)
    buttons_scroll.setMinimumHeight(buttons.sizeHint().height() + 8)
    buttons_scroll.setMaximumHeight(buttons.sizeHint().height() + 22)
    buttons_scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    buttons.setMinimumWidth(buttons.sizeHint().width())
    buttons_scroll.setWidget(buttons)
    outer_layout.addWidget(buttons_scroll)
    run_widgets["resume_button"] = resume_btn

    def sync_settings():
        sync_training_setup_to_state(update_result=True)
        self.training_preview_size = (int(imgsz_spin.value()), int(imgsz_spin.value()))
        self.settings["trainingPreviewImgSize"] = str(imgsz_spin.value())
        self.settings["trainingPreviewAlwaysOn"] = bool(always_preview_checkbox.isChecked())
        self.training_preview_always_on = bool(always_preview_checkbox.isChecked())
        self.settings["trainingEvaluatorWeightsPath"] = eval_weights_edit.text().strip()
        self.settings["trainingEvaluatorUseTrainingWeights"] = bool(use_training_weights.isChecked())
        self.settings["trainingEvaluatorCandidates"] = candidates_edit.text().strip()
        self.settings["trainingEvaluatorAutoYaml"] = bool(auto_yaml_checkbox.isChecked())
        self.settings["trainingEvaluatorUseOptimizedArgs"] = bool(optimized_args_checkbox.isChecked())
        self.settings["trainingEvaluatorSkipImageScan"] = bool(skip_image_scan_checkbox.isChecked())
        self.settings["trainingEvaluatorDataYamlPath"] = self.normalize_path(data_yaml_edit.text().strip())
        self.settings["trainingEvaluatorModelPath"] = self.normalize_path(train_model_edit.text().strip())
        self.settings["trainingEvaluatorRunsDir"] = self.normalize_path(runs_dir_edit.text().strip())
        # trainingEvaluatorExtraArgs removed; cfg file path in recommended_args_display
        self.settings["trainingEvaluatorRecommendedArgsYaml"] = self.normalize_path(recommended_args_display.text().strip())
        self.recommended_args_path = self.normalize_path(recommended_args_display.text().strip())
        self.settings["trainingEvaluatorProbeEpochs"] = int(probe_epochs_spin.value())
        self.settings["trainingEvaluatorTuneIterations"] = int(tune_iterations_spin.value())
        self.settings["trainingEvaluatorUseRayTune"] = bool(ray_tune_checkbox.isChecked())
        self.settings["trainingEvaluatorTuneDistill"] = bool(tune_distill_checkbox.isChecked() and distill_checkbox.isChecked())
        self.settings["trainingEvaluatorBatchOverrideEnabled"] = bool(batch_override_checkbox.isChecked())
        self.settings["trainingEvaluatorBatchOverride"] = batch_override_spin.value()
        self.settings["trainingEvaluatorWorkersOverrideEnabled"] = bool(workers_override_checkbox.isChecked())
        self.settings["trainingEvaluatorWorkersOverride"] = int(workers_override_spin.value())
        self.settings["trainingEvaluatorCacheMode"] = str(cache_mode_combo.currentData() or "off")
        self.settings["trainingEvaluatorValidationMode"] = str(validation_mode_combo.currentData() or "auto")
        self.settings["trainingEvaluatorValidationSource"] = self.normalize_path(validation_source_edit.text().strip())
        # Save unified validation percent to all keys for compatibility
        unified_percent = validation_percent_combo.currentText()
        self.settings["trainingEvaluatorValidationPercent"] = unified_percent
        self.settings["trainingEvaluatorValidationSourcePercent"] = unified_percent
        self.settings["datasetValidPercent"] = unified_percent
        self.queue_settings_save()

    def validation_settings_changed(refresh_controls=False):
        if refresh_controls:
            refresh_validation_controls()
        self.training_evaluator_last_result = None
        sync_settings()

    def set_runtime_override_controls_enabled():
        batch_override_spin.setEnabled(bool(batch_override_checkbox.isChecked()))
        workers_override_spin.setEnabled(bool(workers_override_checkbox.isChecked()))
        sync_settings()

    def set_eval_weights_enabled(sync_task=True):
        use_panel_model = bool(use_training_weights.isChecked())
        eval_weights_edit.setEnabled(not use_panel_model)
        browse_weights_btn.setEnabled(not use_panel_model)
        selected_weights = train_model_edit.text().strip() if use_panel_model else eval_weights_edit.text().strip()
        if sync_task and selected_weights:
            sync_task_from_loaded_weights(
                selected_weights,
                label="training model" if use_panel_model else "evaluation weights",
            )
        sync_settings()

    def refresh_preview_if_visible():
        if always_preview_checkbox.isChecked():
            self.refresh_training_preview_if_enabled()

    def set_always_preview(checked):
        self.set_training_preview_always_on(bool(checked), update_view=True)
        status_label.setText(
            f"Train View is on at {imgsz_spin.value()}x{imgsz_spin.value()}." if checked else "Showing normal editable label view."
        )

    def run_evaluation():
        if optimized_args_checkbox.isChecked() and hyperparameters.is_dirty():
            if not save_hyperparameters(hyperparameters.changes()):
                return
        sync_settings()
        self.training_evaluator_last_result = None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._training_eval_set_activity_indicator(run_widgets, "running", "Working", "Generating files and parameters")
        status_label.setText("Rescanning dataset, overwriting training files, and calculating parameters...")
        QApplication.processEvents()

        def show_evaluation_progress(message):
            status_label.setText(str(message))
            self._training_eval_set_activity_indicator(
                run_widgets, "running", "Working", str(message)
            )
            QApplication.processEvents()

        try:
            result = self.evaluate_training_setup(
                candidate_sizes=candidates_edit.text(),
                weights_path=eval_weights_edit.text().strip(),
                use_training_weights=use_training_weights.isChecked(),
                auto_prepare_yaml=auto_yaml_checkbox.isChecked(),
                validation_mode=validation_mode_combo.currentData(),
                validation_source=validation_source_edit.text().strip(),
                validation_percent=validation_percent_combo.currentText(),
                validation_source_percent=validation_percent_combo.currentText(),
                skip_image_scan=skip_image_scan_checkbox.isChecked(),
                progress_callback=show_evaluation_progress,
            )
            self.training_evaluator_last_result = result
            apply_result_to_training_controls(self.training_evaluator_last_result)
            result = self.training_evaluator_last_result
            refresh_hyperparameters(force=True)
            results_text.setText(self.format_training_evaluation_report(result))
            self._training_eval_update_baseline_widget(run_widgets, result)
            rec = result.get("recommendations", {})
            status_label.setText(
                f"YAML files loaded. Recommended imgsz={rec.get('imgsz')} batch={rec.get('batch')} "
                f"epochs={rec.get('epochs')} patience={rec.get('patience')}."
            )
            self._training_eval_set_activity_indicator(run_widgets, "success", "Ready", "Files and parameters generated")
        except Exception as e:
            self.training_evaluator_last_result = None
            logger.exception("Training file/parameter generation failed")
            QMessageBox.warning(dialog, "Training Evaluator", str(e))
            results_text.setText(f"File/parameter generation failed: {e}")
            status_label.setText(f"File/parameter generation failed: {e}")
            self._training_eval_set_activity_indicator(run_widgets, "error", "Error", "Generation failed")
        finally:
            QApplication.restoreOverrideCursor()

    def run_health_check():
        sync_settings()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        status_label.setText("Running Ultralytics yolo checks and DarkFusion setup checks...")
        self._training_eval_set_activity_indicator(run_widgets, "running", "Working", "Running health check")
        QApplication.processEvents()
        try:
            report = self.build_training_health_report(
                data_yaml_path=data_yaml_edit.text().strip(),
                model_path=train_model_edit.text().strip(),
                pretrained_path=pretrained_edit.text().strip(),
                runs_dir=runs_dir_edit.text().strip(),
                task=task_setup_combo.currentText(),
            )
            health_text.setText(report)
            evaluator_tabs.setCurrentWidget(health_tab)
            status_label.setText("Training setup check complete.")
            self._training_eval_set_activity_indicator(run_widgets, "success", "Ready", "Training setup check complete")
        except Exception as e:
            evaluator_tabs.setCurrentWidget(health_tab)
            health_text.setText(f"Training setup check failed:\n{e}")
            QMessageBox.warning(dialog, "Training Setup Check", str(e))
            status_label.setText(f"Training setup check failed: {e}")
            self._training_eval_set_activity_indicator(run_widgets, "error", "Error", "Training setup check failed")
        finally:
            QApplication.restoreOverrideCursor()

    def ensure_evaluation_result():
        # Visible edits take precedence over the saved cfg at launch time.
        # Preserve the editor's conflict check instead of using stale YAML.
        if optimized_args_checkbox.isChecked() and hyperparameters.is_dirty():
            if not save_hyperparameters(hyperparameters.changes()):
                return None
        result = getattr(self, "training_evaluator_last_result", None)
        if not result:
            # A prior Generate Files + Parameters result is convenient,
            # but it is not required to start an already prepared setup.
            # Reconstruct only the command inputs from the fields already
            # selected in the dialog. This reads existing YAMLs and never
            # creates, refreshes, or scans any dataset files.
            data_yaml = self.normalize_path(data_yaml_edit.text().strip())
            cfg_path = self.normalize_path(recommended_args_display.text().strip())
            if not data_yaml or not os.path.isfile(data_yaml):
                status_label.setText("Select the existing obj.yaml before training.")
                self._training_eval_set_activity_indicator(
                    run_widgets, "warning", "Needs setup", "Existing obj.yaml is required"
                )
                return None
            if optimized_args_checkbox.isChecked():
                if not cfg_path or not os.path.isfile(cfg_path):
                    status_label.setText("Select the existing train_recommendations.yaml before training.")
                    self._training_eval_set_activity_indicator(
                        run_widgets, "warning", "Needs setup", "Existing training config is required"
                    )
                    return None
                try:
                    with open(cfg_path, "r", encoding="utf-8") as handle:
                        if not isinstance(yaml.safe_load(handle) or {}, dict):
                            raise ValueError("YAML must contain a mapping.")
                except Exception as error:
                    status_label.setText(f"Could not read training config: {error}")
                    self._training_eval_set_activity_indicator(
                        run_widgets, "error", "Error", "Training config could not be read"
                    )
                    return None

            model_path = self.normalize_path(train_model_edit.text().strip())
            result = {
                "data_yaml_path": data_yaml,
                "training_recommendations_yaml": cfg_path,
                "image_count": 0,
                "weight_info": {},
                "recommendations": {
                    "data_yaml_path": data_yaml,
                    "weights_path": model_path,
                    "task": str(task_setup_combo.currentText()).strip().lower() or "detect",
                    "imgsz": int(imgsz_spin.value()),
                    "batch": train_batch_spin.value(),
                    "epochs": int(train_epochs_spin.value()),
                    "patience": int(train_patience_spin.value()),
                    "amp": _training_precision(self),
                    "rect": bool(train_rect_checkbox.isChecked()),
                    "freeze": int(train_freeze_spin.value()) if train_freeze_checkbox.isChecked() else 0,
                    "hyperparams": {},
                    "train_args": [],
                    "train_args_text": "",
                },
            }
            self.training_evaluator_last_result = result
            status_label.setText("Using the selected existing YAML files; no dataset files were generated.")
        elif isinstance(result, dict):
            result = overlay_current_setup_on_result(result)
            self.training_evaluator_last_result = result
        return result

    def preview_command():
        sync_settings()
        result = ensure_evaluation_result()
        if not result:
            return
        try:
            use_train_args = optimized_args_checkbox.isChecked()
            train_cmd = self.build_training_evaluator_command(
                result,
                mode="train",
                probe_epochs=None,
                apply_optimized_args=use_train_args,
            )
            val_cmd = self.build_training_evaluator_command(
                result,
                mode="val",
                apply_optimized_args=False,
            )
            tune_cmd = self.build_training_evaluator_tune_command(
                result,
                iterations=int(tune_iterations_spin.value()),
                epochs_per_trial=int(probe_epochs_spin.value()),
                use_ray=ray_tune_checkbox.isChecked(),
                include_distillation=tune_distill_checkbox.isChecked(),
            )
            batch_cmd = self.build_training_evaluator_batch_command(result)
            resume_info = self._training_eval_resume_info(self._training_eval_latest_saved_run(training_only=True))
            resume_text = ""
            if resume_info.get("resumable"):
                resume_cmd = self.build_training_evaluator_resume_command(resume_info.get("run_dir"))
                resume_text = "\n\nResume last interrupted run command\n" + self._format_command_display(resume_cmd)
            report = self.format_training_evaluation_report(result)
            results_text.setText(
                report
                + "\n\nFinal train command\n"
                + self._format_command_display(train_cmd)
                + "\n\nValidate baseline weights command\n"
                + self._format_command_display(val_cmd)
                + "\n\nCalibrate batch command\n"
                + self._format_command_display(batch_cmd)
                + "\n\nUltralytics tune command\n"
                + self._format_command_display(tune_cmd)
                + resume_text
            )
            status_label.setText("Command preview updated.")
            self._training_eval_set_activity_indicator(run_widgets, "success", "Ready", "Command preview updated")
        except Exception as e:
            QMessageBox.warning(dialog, "Training Evaluator", str(e))
            status_label.setText(f"Command preview failed: {e}")
            self._training_eval_set_activity_indicator(run_widgets, "error", "Error", "Command preview failed")

    def run_validation_metrics():
        sync_settings()
        status_label.setText("Checking checkpoint and launching validation...")
        self._training_eval_set_activity_indicator(
            run_widgets, "running", "Working", "Preparing validation"
        )
        QApplication.processEvents()
        selected_weights = self._effective_eval_weights_path(
            use_training_weights.isChecked(),
            eval_weights_edit.text().strip(),
        )
        if not selected_weights:
            QMessageBox.warning(
                dialog,
                "Training Evaluator",
                "Select a model checkpoint before starting validation.",
            )
            self._training_eval_set_activity_indicator(
                run_widgets, "warning", "Needs attention", "No validation checkpoint selected"
            )
            return

        weight_info = self.inspect_training_eval_weights(selected_weights)
        if not weight_info.get("loaded"):
            detail = str(weight_info.get("error") or "Unknown checkpoint load error")
            message = (
                f"The selected checkpoint cannot be loaded:\n{selected_weights}\n\n{detail}\n\n"
                "The file may be incomplete or corrupt. If this is best.pt, try last.pt from "
                "the same weights folder."
            )
            QMessageBox.warning(dialog, "Invalid Validation Checkpoint", message)
            status_label.setText(f"Validation not started: {os.path.basename(selected_weights)} could not be loaded.")
            self._training_eval_set_activity_indicator(
                run_widgets, "error", "Error", "Checkpoint could not be loaded"
            )
            return

        data_yaml = self.normalize_path(data_yaml_edit.text().strip())
        yaml_data, _yaml_dir = self._training_health_parse_data_yaml(data_yaml)
        yaml_names = yaml_data.get("names") if isinstance(yaml_data, dict) else None
        if isinstance(yaml_names, (dict, list, tuple)):
            dataset_class_count = len(yaml_names)
        else:
            try:
                dataset_class_count = int(yaml_data.get("nc", 0) or 0)
            except (TypeError, ValueError):
                dataset_class_count = 0
        checkpoint_class_count = int(weight_info.get("class_count", 0) or 0)
        if checkpoint_class_count and dataset_class_count and checkpoint_class_count != dataset_class_count:
            message = (
                f"The checkpoint and dataset do not match:\n\n"
                f"Checkpoint: {os.path.basename(selected_weights)} ({checkpoint_class_count} classes)\n"
                f"Dataset YAML: {data_yaml} ({dataset_class_count} classes)\n\n"
                "Select the obj.yaml that belongs to this checkpoint's dataset before validating."
            )
            QMessageBox.warning(dialog, "Checkpoint / Dataset Mismatch", message)
            status_label.setText("Validation not started: checkpoint and dataset class counts do not match.")
            self._training_eval_set_activity_indicator(
                run_widgets, "warning", "Needs attention", "Checkpoint / dataset mismatch"
            )
            return

        # Validation does not need the expensive recommendation evaluator.
        # Build its command directly from the visible, already-restored run
        # settings so a large dataset is never scanned on the Qt UI thread.
        model_task = str(weight_info.get("task", "") or "").strip().lower()
        selected_task = model_task if model_task in {"detect", "segment", "classify", "pose", "obb"} else str(
            task_setup_combo.currentText() or "detect"
        ).strip().lower()
        result = {
            "data_yaml_path": data_yaml,
            "metrics_split": str(review_split_combo.currentData() or "all"),
            "image_count": 0,
            "weight_info": weight_info,
            "recommendations": {
                "data_yaml_path": data_yaml,
                "weights_path": selected_weights,
                "task": selected_task,
                "imgsz": int(imgsz_spin.value()),
                "batch": train_batch_spin.value(),
                "epochs": int(train_epochs_spin.value()),
                "patience": int(train_patience_spin.value()),
                "rect": bool(train_rect_checkbox.isChecked()),
                "freeze": int(train_freeze_spin.value()) if train_freeze_checkbox.isChecked() else 0,
                "amp": _training_precision(self),
                "train_args": [],
                "train_args_text": "",
            },
        }
        self.training_evaluator_last_metrics_result = result
        try:
            split = result["metrics_split"]
            run_name = self._training_eval_baseline_run_name(result)
            if split != "val":
                run_name = f"metrics_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            command = self.build_training_evaluator_command(result, mode="val", apply_optimized_args=False, run_name=run_name)
            run_title = "Baseline Validation" if split == "val" else f"Metrics Only ({review_split_combo.currentText()})"
            title = f"Training Evaluator {run_title}"
            if torch.cuda.is_available():
                reply = QMessageBox.question(
                    dialog,
                    "GPU Validation",
                    (
                        "GPU validation can draw a hard power spike.\n\n"
                        "Start GPU validation anyway?"
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    status_label.setText("GPU validation canceled.")
                    self._training_eval_set_activity_indicator(run_widgets, "warning", "Needs attention", "GPU validation canceled")
                    return
            process, run_dir, log_path = self.launch_training_evaluator_logged_command(
                command,
                title,
            )
            evaluator_tabs.setCurrentWidget(metrics_tab)
            self.watch_training_evaluator_run(
                process,
                command,
                run_title,
                run_dir,
                log_path,
                widgets=run_widgets,
                status_label=status_label,
            )
            status_label.setText(f"{run_title} started. Live output will appear in the Live Metrics tab.")
            self._training_eval_set_activity_indicator(run_widgets, "running", "Working", f"{run_title} running")
        except Exception as e:
            QMessageBox.warning(dialog, "Training Evaluator", str(e))
            status_label.setText(f"Validation failed to start: {e}")
            self._training_eval_set_activity_indicator(run_widgets, "error", "Error", "Validation failed to start")

    def calibrate_batch():
        sync_settings()
        result = ensure_evaluation_result()
        if not result:
            return
        if not torch.cuda.is_available():
            QMessageBox.warning(dialog, "Training Evaluator", "CUDA is not available, so batch calibration cannot run.")
            self._training_eval_set_activity_indicator(run_widgets, "warning", "Needs attention", "CUDA unavailable")
            return
        reply = QMessageBox.question(
            dialog,
            "Calibrate Batch",
            "Run Ultralytics auto-batch calibration now? This tests VRAM once and then sets a fixed batch for training.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            command = self.build_training_evaluator_batch_command(result)
            process, run_dir, log_path = self.launch_training_evaluator_logged_command(
                command,
                "Training Evaluator Batch Calibration",
            )
            evaluator_tabs.setCurrentWidget(metrics_tab)
            self.watch_training_evaluator_run(
                process,
                command,
                "Batch calibration",
                run_dir,
                log_path,
                widgets=run_widgets,
                status_label=status_label,
            )
            self.watch_training_evaluator_batch(
                process,
                command,
                result,
                results_text=results_text,
                status_label=status_label,
                on_calibrated=lambda updated_result: apply_result_to_training_controls(updated_result),
            )
            status_label.setText("Batch calibration started. Live output will appear in the Live Metrics tab.")
            self._training_eval_set_activity_indicator(run_widgets, "running", "Working", "Batch calibration running")
        except Exception as e:
            QMessageBox.warning(dialog, "Training Evaluator", str(e))
            status_label.setText(f"Batch calibration failed to start: {e}")
            self._training_eval_set_activity_indicator(run_widgets, "error", "Error", "Batch calibration failed to start")

    def tune_hyperparams():
        sync_settings()

        # VALIDATION: Check required settings BEFORE spawning any processes
        data_yaml = self.normalize_path(data_yaml_edit.text().strip())
        model_path = self.normalize_path(train_model_edit.text().strip())
        runs_dir = self.normalize_path(runs_dir_edit.text().strip())

        missing = []
        if not data_yaml or not os.path.exists(data_yaml):
            missing.append(f"• Data YAML not found: {data_yaml if data_yaml else '(empty)'}")
        if not model_path or not os.path.exists(model_path):
            missing.append(f"• Model path not found: {model_path if model_path else '(empty)'}")

        if missing:
            QMessageBox.critical(
                dialog,
                "Cannot Start Tuning",
                "Please configure the required fields first:\n\n" + "\n".join(missing) + 
                "\n\n1. Click 'YAML' to select your training data.yaml\n" +
                "2. Click 'Model' to select a .pt weights file\n" +
                "3. Then click 'Generate Files + Parameters'\n" +
                "4. Finally, click 'Tune HParams'"
            )
            self._training_eval_set_activity_indicator(run_widgets, "error", "Error", "Missing required configuration")
            return

        result = ensure_evaluation_result()
        if not result:
            return
        epochs = int(probe_epochs_spin.value())
        iterations = int(tune_iterations_spin.value())
        total_epochs = epochs * iterations
        if ray_tune_checkbox.isChecked() and not self._ray_tune_available():
            QMessageBox.warning(
                dialog,
                "Training Evaluator",
                "Ray Tune is not installed. Install it with:\n\npip install \"ray[tune]\"\n\nOr turn Ray Tune off to use Ultralytics' built-in tuner.",
            )
            self._training_eval_set_activity_indicator(run_widgets, "warning", "Needs attention", "Ray Tune unavailable")
            return
        reply = QMessageBox.question(
            dialog,
            "Tune Hyperparameters",
            (
                f"Start Ultralytics hyperparameter tuning?\n\n"
                f"Trials: {iterations}\n"
                f"Epochs per trial: {epochs}\n"
                f"Approx total epochs: {total_epochs}\n"
                f"Ray Tune: {'on, GPU per trial auto' if ray_tune_checkbox.isChecked() else 'off'}\n"
                f"Distillation during tune: {'on' if tune_distill_checkbox.isChecked() else 'off'}"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            # Show warming up message immediately
            warming_msg = "Ray Tune warming up: initializing GPU workers and search space (1-2 minutes)..." if ray_tune_checkbox.isChecked() else "Tuning warming up: preparing tuning engine..."
            status_label.setText(warming_msg)
            self._training_eval_set_activity_indicator(run_widgets, "running", "Starting", "Warming up")

            command = self.build_training_evaluator_tune_command(
                result,
                iterations=iterations,
                epochs_per_trial=epochs,
                use_ray=ray_tune_checkbox.isChecked(),
                include_distillation=tune_distill_checkbox.isChecked(),
            )
            process, run_dir, log_path = self.launch_training_evaluator_logged_command(
                command,
                "Training Evaluator Hyperparameter Tune",
            )
            evaluator_tabs.setCurrentWidget(metrics_tab)
            self.watch_training_evaluator_run(
                process,
                command,
                "Hyperparameter tune",
                run_dir,
                log_path,
                widgets=run_widgets,
                status_label=status_label,
            )
            self.watch_training_evaluator_tune(
                process,
                command,
                result,
                results_text=results_text,
                status_label=status_label,
                optimized_args_checkbox=optimized_args_checkbox,
                on_tuned=lambda updated_result: apply_result_to_training_controls(updated_result),
            )
            # Compute trial info for status message
            epochs = int(probe_epochs_spin.value())
            iterations = int(tune_iterations_spin.value())
            total_epochs_approx = epochs * iterations
            status_label.setText(
                f"Warmup complete! Tuning now active: testing {iterations} hyperparameter combinations with {epochs} epochs each (~{total_epochs_approx} total epochs). "
                f"Each trial runs independently. Live logs appear below in the Live Metrics tab."
            )
            self._training_eval_set_activity_indicator(run_widgets, "running", "Working", "Tuning running")
        except Exception as e:
            QMessageBox.warning(dialog, "Training Evaluator", str(e))
            status_label.setText(f"Hyperparameter tuning failed to start: {e}")
            self._training_eval_set_activity_indicator(run_widgets, "error", "Error", "Tuning failed to start")

    def resume_last_run():
        sync_settings()
        record = self._training_eval_latest_saved_run(training_only=True)
        info = self._training_eval_resume_info(record)
        if not info.get("resumable"):
            QMessageBox.information(dialog, "Resume Training", info.get("note") or "No resumable training run was found.")
            self._training_eval_set_activity_indicator(run_widgets, "warning", "Needs attention", "No resumable run")
            return
        reply = QMessageBox.question(
            dialog,
            "Resume Training",
            (
                f"{info.get('note')}\n\n"
                "Resume continues the existing run from weights/last.pt. "
                "Start Training creates a fresh run instead."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            command = self.build_training_evaluator_resume_command(info.get("run_dir"))
            process, run_dir, log_path = self.launch_training_evaluator_logged_command(
                command,
                "Training Evaluator Resume",
                run_dir=info.get("run_dir"),
            )
            evaluator_tabs.setCurrentWidget(metrics_tab)
            self.watch_training_evaluator_run(
                process,
                command,
                "Training resume",
                run_dir,
                log_path,
                widgets=run_widgets,
                status_label=status_label,
            )
            status_label.setText("Resume started. Live output will appear in the Live Metrics tab.")
            self._training_eval_set_activity_indicator(run_widgets, "running", "Working", "Resume running")
        except Exception as e:
            QMessageBox.warning(dialog, "Resume Training", str(e))
            status_label.setText(f"Resume failed to start: {e}")
            self._training_eval_set_activity_indicator(run_widgets, "error", "Error", "Resume failed to start")

    def latest_training_control_record():
        record = self._training_eval_latest_saved_run(training_only=True)
        run_dir = self.normalize_path((record or {}).get("run_dir", ""))
        if not run_dir or not os.path.isdir(run_dir):
            return {}
        clean = dict(record or {})
        clean["run_dir"] = run_dir
        clean["log_path"] = self.normalize_path(
            clean.get("log_path") or self._training_eval_log_path(run_dir)
        )
        return clean

    def training_processes_for_record(record):
        from darkfusion_training_runs import training_processes_for_record as find_processes
        return find_processes(record, psutil)

    def terminate_training_record(record, reason):
        roots = training_processes_for_record(record)
        if not roots:
            return False, "The active training process could not be located."
        processes = {}
        for root in roots:
            try:
                processes[root.pid] = root
                for child in root.children(recursive=True):
                    processes[child.pid] = child
            except Exception:
                continue
        ordered = list(processes.values())
        def child_count(process):
            try:
                return len(process.children(recursive=True))
            except Exception:
                return 0
        ordered.sort(key=child_count)
        for process in ordered:
            try:
                process.terminate()
            except Exception:
                pass
        try:
            _gone, alive = psutil.wait_procs(ordered, timeout=0.2)
        except Exception:
            alive = []
        for process in alive:
            try:
                process.kill()
            except Exception:
                pass
        logger.warning("%s: terminated DarkFusion training process tree for %s", reason, record.get("run_dir", ""))
        eta_context = run_widgets.get("_eta_context", {})
        if eta_context.get("run_dir") == self.normalize_path(record.get("run_dir", "")):
            eta_context["stopped"] = True
        return True, ""

    def update_training_control_buttons():
        if sip.isdeleted(dialog):
            return
        record = latest_training_control_record()
        run_dir = self.normalize_path(record.get("run_dir", ""))
        active = bool(record and self._training_eval_run_is_active(record))
        best_path = self.normalize_path(os.path.join(run_dir, "weights", "best.pt"))
        stop_after_epoch_btn.setEnabled(active)
        stop_now_btn.setEnabled(active)
        open_run_btn.setEnabled(bool(run_dir and os.path.isdir(run_dir)))
        open_best_btn.setEnabled(bool(best_path and os.path.isfile(best_path)))
        if not active and stop_after_epoch_btn.text() != "Stop After Epoch":
            stop_after_epoch_btn.setText("Stop After Epoch")

    def open_latest_run_folder():
        record = latest_training_control_record()
        run_dir = self.normalize_path(record.get("run_dir", ""))
        if not run_dir or not os.path.isdir(run_dir):
            QMessageBox.information(dialog, "Training Run", "No training run folder is available.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(run_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", run_dir])
            else:
                subprocess.Popen(["xdg-open", run_dir])
        except Exception as e:
            QMessageBox.information(dialog, "Training Run", f"{run_dir}\n\nCould not open it automatically: {e}")

    def open_latest_best_checkpoint():
        record = latest_training_control_record()
        best_path = self.normalize_path(os.path.join(record.get("run_dir", ""), "weights", "best.pt"))
        if not best_path or not os.path.isfile(best_path):
            QMessageBox.information(dialog, "best.pt", "The active/latest run does not have best.pt yet.")
            return
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer.exe", f"/select,{best_path}"])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", best_path])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(best_path)])
        except Exception as e:
            QMessageBox.information(dialog, "best.pt", f"{best_path}\n\nCould not reveal it automatically: {e}")

    def request_stop_after_current_epoch():
        record = latest_training_control_record()
        if not record or not self._training_eval_run_is_active(record):
            QMessageBox.information(dialog, "Stop After Epoch", "No active training run was found.")
            update_training_control_buttons()
            return
        run_dir = self.normalize_path(record.get("run_dir", ""))
        reply = QMessageBox.question(
            dialog,
            "Stop After Current Epoch",
            (
                "Finish the current epoch, complete validation, save last.pt/best.pt, "
                "and then stop training?\n\n"
                "No running process will be stopped until that checkpoint is saved."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        _headers, starting_rows, _csv_path = self._training_eval_read_results_csv(run_dir)
        marker_path = self.normalize_path(os.path.join(run_dir, ".darkfusion_stop_after_epoch"))
        request_payload = {
            "requested_at": datetime.now().isoformat(timespec="seconds"),
            "completed_rows_at_request": len(starting_rows),
        }
        try:
            with open(marker_path, "w", encoding="utf-8") as handle:
                json.dump(request_payload, handle, indent=2)
        except Exception as e:
            QMessageBox.warning(dialog, "Stop After Epoch", f"Could not create the stop request:\n{e}")
            return

        stop_after_epoch_btn.setEnabled(False)
        stop_after_epoch_btn.setText("Stop Queued")
        status_label.setText("Stop requested. DarkFusion will wait for the current epoch checkpoint.")
        request_time = time.time()
        initial_rows = len(starting_rows)
        control_timer = QTimer(dialog)
        control_timer.setInterval(250)

        def watch_for_clean_checkpoint():
            if not self._training_eval_run_is_active(record):
                control_timer.stop()
                stop_after_epoch_btn.setText("Stop After Epoch")
                update_training_control_buttons()
                status_label.setText("Training stopped after completing its checkpoint.")
                return
            ack_path = os.path.join(run_dir, ".darkfusion_stop_ack.json")
            if os.path.isfile(ack_path):
                status_label.setText("Trainer accepted the stop request and is finishing cleanly.")
                return
            _headers, rows, csv_path = self._training_eval_read_results_csv(run_dir)
            last_path = os.path.join(run_dir, "weights", "last.pt")
            try:
                checkpoint_ready = (
                    len(rows) > initial_rows
                    and os.path.isfile(last_path)
                    and os.path.getmtime(last_path) >= request_time
                    and os.path.getmtime(last_path) >= os.path.getmtime(csv_path)
                )
            except Exception:
                checkpoint_ready = False
            if not checkpoint_ready:
                return
            stopped, error = terminate_training_record(record, "Stop after current epoch")
            if stopped:
                try:
                    os.remove(marker_path)
                except OSError:
                    pass
                control_timer.stop()
                stop_after_epoch_btn.setText("Stop After Epoch")
                status_label.setText("Current epoch and checkpoint completed; training has stopped.")
                QTimer.singleShot(500, update_training_control_buttons)
            elif error:
                status_label.setText(error)

        control_timer.timeout.connect(watch_for_clean_checkpoint)
        control_timer.start()

    def stop_training_now():
        record = latest_training_control_record()
        if not record or not self._training_eval_run_is_active(record):
            QMessageBox.information(dialog, "Stop Training", "No active training run was found.")
            update_training_control_buttons()
            return
        reply = QMessageBox.warning(
            dialog,
            "Stop Training Now",
            (
                "Stop training immediately?\n\n"
                "The current incomplete epoch will be discarded. The last clean last.pt and best.pt "
                "checkpoint from the previous completed epoch will remain available."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        stopped, error = terminate_training_record(record, "Stop now")
        if not stopped:
            QMessageBox.warning(dialog, "Stop Training", error)
            return
        for marker_name in (".darkfusion_stop_after_epoch", ".darkfusion_stop_now"):
            try:
                os.remove(os.path.join(record.get("run_dir", ""), marker_name))
            except OSError:
                pass
        status_label.setText("Training stopped. The last completed checkpoint remains available.")
        QTimer.singleShot(500, update_training_control_buttons)

    def start_training():
        sync_settings()
        result = ensure_evaluation_result()
        if not result:
            return
        use_train_args = optimized_args_checkbox.isChecked()
        self.apply_training_evaluation_recommendations(
            result,
            apply_weights=False,
            # cfg=/dataset metadata is authoritative; do not copy stale
            # cached recommendations back into legacy extra-args state.
            apply_optimized_args=False,
        )
        sync_training_setup_to_state(update_result=True)
        result = overlay_current_setup_on_result(result)
        self.training_evaluator_last_result = result
        rec = result.get("recommendations", {})
        reply = QMessageBox.question(
            dialog,
            "Start Training",
            (
                f"Start Ultralytics training with imgsz={rec.get('imgsz')}, "
                f"batch={rec.get('batch')}, epochs={rec.get('epochs')}?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                command = self.build_training_evaluator_command(
                    result,
                    mode="train",
                    probe_epochs=None,
                    apply_optimized_args=use_train_args,
                )
                process, run_dir, log_path = self.launch_training_evaluator_logged_command(
                    command,
                    "Training Evaluator Final Train",
                )
                evaluator_tabs.setCurrentWidget(metrics_tab)
                self.watch_training_evaluator_run(
                    process,
                    command,
                    "Training",
                    run_dir,
                    log_path,
                    widgets=run_widgets,
                    status_label=status_label,
                )
                status_label.setText("Training started. Live output will appear in the Live Metrics tab.")
                self._training_eval_set_activity_indicator(run_widgets, "running", "Working", "Training running")
            except Exception as e:
                QMessageBox.warning(dialog, "Training Evaluator", str(e))
                status_label.setText(f"Training failed to start: {e}")
                self._training_eval_set_activity_indicator(run_widgets, "error", "Error", "Training failed to start")

    use_training_weights.toggled.connect(lambda _checked: set_eval_weights_enabled())
    always_preview_checkbox.toggled.connect(set_always_preview)
    browse_data_yaml_btn.clicked.connect(browse_setup_data_yaml)
    browse_weights_btn.clicked.connect(browse_eval_weights)
    browse_train_model_btn.clicked.connect(browse_setup_model)
    browse_pretrained_btn.clicked.connect(browse_setup_pretrained)
    clear_pretrained_btn.clicked.connect(lambda: (pretrained_edit.clear(), sync_settings()))
    browse_runs_dir_btn.clicked.connect(browse_setup_runs_dir)
    browse_recommended_args_btn.clicked.connect(browse_recommended_args_yaml)
    data_yaml_edit.textChanged.connect(lambda _text: sync_settings())
    train_model_edit.textChanged.connect(lambda _text: (refresh_setup_model_controls(), sync_settings()))
    train_model_edit.editingFinished.connect(
        lambda: sync_task_from_loaded_weights(train_model_edit.text().strip(), label="training model")
    )
    pretrained_edit.textChanged.connect(lambda _text: sync_settings())
    runs_dir_edit.textChanged.connect(lambda _text: sync_settings())
    task_setup_combo.currentIndexChanged.connect(lambda _index: sync_settings())
    imgsz_spin.valueChanged.connect(lambda _value: (sync_settings(), refresh_preview_if_visible()))
    train_batch_spin.valueChanged.connect(lambda _value: sync_settings())
    train_epochs_spin.valueChanged.connect(lambda _value: sync_settings())
    train_rect_checkbox.toggled.connect(lambda _checked: sync_settings())
    duplicate_labels_checkbox.toggled.connect(lambda _checked: sync_settings())
    train_patience_checkbox.toggled.connect(lambda _checked: (refresh_advanced_train_controls(), sync_settings()))
    train_patience_spin.valueChanged.connect(lambda _value: sync_settings())
    train_freeze_checkbox.toggled.connect(lambda _checked: (refresh_advanced_train_controls(), sync_settings()))
    train_freeze_spin.valueChanged.connect(lambda _value: sync_settings())
    distill_checkbox.toggled.connect(lambda _checked: (refresh_advanced_train_controls(), sync_settings()))
    distill_model_edit.textChanged.connect(lambda _text: sync_settings())
    browse_distill_btn.clicked.connect(browse_setup_distill_model)
    clear_distill_btn.clicked.connect(lambda: (distill_model_edit.clear(), distill_checkbox.setChecked(False), sync_settings()))
    distill_weight_spin.valueChanged.connect(lambda _value: sync_settings())
    # extra_args_edit removed; cfg file updates handled by file path changes
    recommended_args_display.textChanged.connect(lambda _text: sync_settings())
    recommended_args_display.textChanged.connect(lambda _text: refresh_hyperparameters(force=True))
    hyperparameters.saveRequested.connect(save_hyperparameters)
    hyperparameters.reloadRequested.connect(lambda: refresh_hyperparameters(force=True))
    hyperparameter_group.expandedChanged.connect(lambda expanded: refresh_hyperparameters() if expanded else None)
    optimized_args_checkbox.toggled.connect(lambda _checked: refresh_hyperparameters(force=True))
    task_setup_combo.currentIndexChanged.connect(lambda _index: refresh_hyperparameters(force=True))
    validation_mode_combo.currentIndexChanged.connect(lambda _index: validation_settings_changed(refresh_controls=True))
    validation_percent_combo.currentIndexChanged.connect(lambda _index: validation_settings_changed())
    validation_source_edit.textChanged.connect(lambda _text: validation_settings_changed())
    browse_validation_source_btn.clicked.connect(browse_validation_source)
    candidates_edit.textChanged.connect(lambda _text: sync_settings())
    eval_weights_edit.textChanged.connect(lambda _text: sync_settings())
    eval_weights_edit.editingFinished.connect(
        lambda: (
            sync_task_from_loaded_weights(eval_weights_edit.text().strip(), label="evaluation weights")
            if not use_training_weights.isChecked()
            else None
        )
    )
    auto_yaml_checkbox.toggled.connect(lambda _checked: sync_settings())
    optimized_args_checkbox.toggled.connect(lambda _checked: sync_settings())
    probe_epochs_spin.valueChanged.connect(lambda _value: sync_settings())
    tune_iterations_spin.valueChanged.connect(lambda _value: sync_settings())
    ray_tune_checkbox.toggled.connect(lambda _checked: sync_settings())
    tune_distill_checkbox.toggled.connect(lambda _checked: sync_settings())
    batch_override_checkbox.toggled.connect(lambda _checked: set_runtime_override_controls_enabled())
    batch_override_spin.valueChanged.connect(lambda _value: sync_settings())
    workers_override_checkbox.toggled.connect(lambda _checked: set_runtime_override_controls_enabled())
    workers_override_spin.valueChanged.connect(lambda _value: sync_settings())
    cache_mode_combo.currentIndexChanged.connect(lambda _index: sync_settings())
    evaluate_btn.clicked.connect(run_evaluation)
    health_check_btn.clicked.connect(run_health_check)
    preview_cmd_btn.clicked.connect(preview_command)
    review_validate_metrics_btn.clicked.connect(run_validation_metrics)
    calibrate_batch_btn.clicked.connect(calibrate_batch)
    tune_btn.clicked.connect(tune_hyperparams)
    resume_btn.clicked.connect(resume_last_run)
    train_btn.clicked.connect(start_training)
    stop_after_epoch_btn.clicked.connect(request_stop_after_current_epoch)
    stop_now_btn.clicked.connect(stop_training_now)
    open_run_btn.clicked.connect(open_latest_run_folder)
    open_best_btn.clicked.connect(open_latest_best_checkpoint)

    set_eval_weights_enabled(sync_task=False)
    refresh_setup_model_controls()
    refresh_advanced_train_controls()
    refresh_validation_controls()
    set_runtime_override_controls_enabled()
    update_training_control_buttons()
    refresh_hyperparameters(force=True)
    QTimer.singleShot(
        0,
        lambda: sync_task_from_loaded_weights(
            train_model_edit.text().strip() if use_training_weights.isChecked() else eval_weights_edit.text().strip(),
            label="training model" if use_training_weights.isChecked() else "evaluation weights",
        ),
    )
    self.training_evaluator_dialog = dialog
    def clear_training_evaluator_dialog(_obj=None):
        self.training_evaluator_dialog = None
        self.training_evaluator_run_widgets = None
        self.training_evaluator_status_label = None
    dialog.destroyed.connect(clear_training_evaluator_dialog)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def _training_eval_default_project_dir(self, result=None):
    rec = (result or {}).get("recommendations", {})
    data_yaml = self.normalize_path(rec.get("data_yaml_path", "") or (result or {}).get("data_yaml_path", ""))
    if not data_yaml:
        data_yaml = self.normalize_path(getattr(self, "data_yaml_path", "") or "")
    if data_yaml:
        yaml_dir = os.path.dirname(data_yaml)
        marker = os.path.normcase(os.path.join(PROJECT_SETTINGS_DIR, "training"))
        norm_yaml_dir = os.path.normcase(yaml_dir.replace("\\", os.sep).replace("/", os.sep))
        if norm_yaml_dir.endswith(marker):
            return os.path.join(os.path.dirname(os.path.dirname(yaml_dir)), "runs").replace("\\", "/")
        if os.path.basename(norm_yaml_dir) == os.path.normcase(PROJECT_SETTINGS_DIR):
            return os.path.join(os.path.dirname(yaml_dir), "runs").replace("\\", "/")
        return os.path.join(yaml_dir, "runs").replace("\\", "/")

    dataset_dir = self._training_eval_dataset_dir()
    if dataset_dir:
        return os.path.join(dataset_dir, "runs").replace("\\", "/")

    return self.normalize_path(os.path.join(APP_DIR, "runs"))


def _training_eval_project_dir(self, result=None):
    settings = getattr(self, "settings", {}) if isinstance(getattr(self, "settings", None), dict) else {}
    for path in (
        getattr(self, "runs_directory", ""),
        settings.get("ultralyticsRunsDirectory", ""),
    ):
        path = self.normalize_path(path or "")
        if path:
            return path
    return self._training_eval_default_project_dir(result)


def _ray_tune_available(self):
    return importlib.util.find_spec("ray") is not None


def _training_eval_model_path(self, result=None, prefer_recommendation=True):
    rec = (result or {}).get("recommendations", {})
    if prefer_recommendation:
        weights_path = self.normalize_path(rec.get("weights_path", ""))
        if weights_path and os.path.exists(weights_path):
            return weights_path

    if getattr(self, "model_config_path", None):
        return self.normalize_path(self.model_config_path)
    if getattr(self, "pt_path", None):
        return self.normalize_path(self.pt_path)
    if getattr(self, "pretrained_model_path", None):
        return self.normalize_path(self.pretrained_model_path)
    return ""


def _training_eval_model_scale_rank_from_path(self, model_path=""):
    name = os.path.splitext(os.path.basename(str(model_path or "")))[0].lower()
    if not name:
        return 0, ""
    for token, rank in (("x", 5), ("l", 4), ("m", 3), ("s", 2), ("n", 1)):
        if re.search(rf"(^|[^a-z]){token}([^a-z]|$)", name):
            return rank, token
    return 0, ""


def _training_eval_distillation_sanity_notes(self, student_path="", teacher_path="", weight=None, task="detect"):
    notes = []
    student_path = self.normalize_path(student_path or "")
    teacher_path = self.normalize_path(teacher_path or "")
    if not teacher_path:
        return notes
    student_rank, student_scale = self._training_eval_model_scale_rank_from_path(student_path)
    teacher_rank, teacher_scale = self._training_eval_model_scale_rank_from_path(teacher_path)
    if student_path and teacher_path and os.path.normcase(student_path) == os.path.normcase(teacher_path):
        notes.append({"severity": "warning", "text": "Distillation teacher and student are the same file."})
    elif student_rank and teacher_rank and teacher_rank < student_rank:
        notes.append({
            "severity": "warning",
            "text": f"Distillation teacher looks smaller ({teacher_scale}) than student ({student_scale}).",
        })
    elif student_rank and teacher_rank and teacher_rank == student_rank:
        notes.append({
            "severity": "info",
            "text": f"Teacher and student appear to use the same scale ({teacher_scale}).",
        })
    try:
        weight_value = float(weight if weight is not None else getattr(self, "ultralytics_distill_weight", 6.0))
    except Exception:
        weight_value = 6.0
    if math.isfinite(weight_value) and weight_value > 2.0:
        notes.append({
            "severity": "warning",
            "text": f"Distillation weight dis={weight_value:g} is strong; start lower if the teacher is not proven better.",
        })
    if str(task or "").strip().lower() not in {"", "detect"}:
        notes.append({
            "severity": "info",
            "text": f"Distillation for task={task} is less battle-tested than detect.",
        })
    return notes


def _training_eval_format_distillation_notes(self, notes):
    return "\n".join(f"- {note.get('text', '')}" for note in notes or [] if note.get("text"))


def _training_eval_model_command_args(self, result=None, mode="train"):
    if mode == "train" and getattr(self, "model_config_path", None):
        model_config_path = self.normalize_path(self.model_config_path)
        if model_config_path and os.path.exists(model_config_path):
            args = [f"model={model_config_path}"]
            pretrained_path = self.normalize_path(getattr(self, "pretrained_model_path", "") or "")
            if pretrained_path and os.path.exists(pretrained_path):
                if not pretrained_path.lower().endswith(".pt"):
                    raise ValueError("Transfer-learning pretrained weights must be a .pt file.")
                args.append(f"pretrained={pretrained_path}")
            return args

    model_path = self._training_eval_model_path(result, prefer_recommendation=True)
    if not model_path:
        return []
    if mode == "val" and not model_path.lower().endswith(".pt"):
        raise ValueError("Validation needs a trained .pt weights file. Select eval weights or a training .pt model first.")
    if mode == "train" and not self._is_ultralytics_trainable_model_path(model_path, allow_yaml=False):
        raise ValueError(
            "Training needs a .pt model, or a model .yaml with optional .pt pretrained weights. "
                "Exported .onnx/.engine models only support predict/val."
        )
    return [f"model={model_path}"]


def _ultralytics_cli_command_prefix(self):
    wrapper_path = os.path.join(APP_DIR, "darkfusion_ultralytics_cli.py")
    if os.path.isfile(wrapper_path):
        return [sys.executable, wrapper_path]
    return ["yolo"]


def _ultralytics_train_command_prefix(self):
    keep_duplicate_labels = bool(
        getattr(self, "keep_duplicate_labels_checkbox", None)
        and self.keep_duplicate_labels_checkbox.isChecked()
    )
    wrapper_path = os.path.join(APP_DIR, "darkfusion_ultralytics_train.py")
    if os.path.isfile(wrapper_path):
        prefix = [sys.executable, wrapper_path]
        if keep_duplicate_labels:
            prefix.append("--darkfusion-keep-duplicates")
        return prefix
    return self._ultralytics_cli_command_prefix()


@staticmethod
def _version_tuple(value):
    parts = []
    for piece in re.split(r"[^\d]+", str(value or "")):
        if piece == "":
            continue
        try:
            parts.append(int(piece))
        except Exception:
            parts.append(0)
        if len(parts) >= 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def ultralytics_version_at_least(self, minimum):
    try:
        import ultralytics
        current = getattr(ultralytics, "__version__", "0.0.0")
    except Exception:
        current = "0.0.0"
    return self._version_tuple(current) >= self._version_tuple(minimum)


@staticmethod
def _looks_like_ultralytics_model_reference(value):
    text = str(value or "").strip()
    if not text:
        return False
    if any(sep in text for sep in ("/", "\\")):
        return False
    return text.lower().endswith((".pt", ".yaml", ".yml"))


@staticmethod
def _is_ultralytics_export_model_path(value):
    return str(value or "").strip().lower().endswith((".onnx", ".engine"))


@staticmethod
def _is_ultralytics_trainable_model_path(value, allow_yaml=True):
    lower = str(value or "").strip().lower()
    if not lower:
        return False
    if lower.endswith(".pt"):
        return True
    return bool(allow_yaml and lower.endswith((".yaml", ".yml")))


def _validate_ultralytics_train_model_selection(self):
    model_config_path = self.normalize_path(getattr(self, "model_config_path", "") or "")
    pretrained_path = self.normalize_path(getattr(self, "pretrained_model_path", "") or "")
    pt_path = self.normalize_path(getattr(self, "pt_path", "") or "")

    if model_config_path:
        if not model_config_path.lower().endswith((".yaml", ".yml")):
            raise ValueError("Training model config must be a .yaml/.yml file.")
        if not os.path.exists(model_config_path):
            raise ValueError(f"Training model config was not found:\n{model_config_path}")
        if pretrained_path:
            if not pretrained_path.lower().endswith(".pt"):
                raise ValueError("Transfer-learning pretrained weights must be a .pt file.")
            if not os.path.exists(pretrained_path):
                raise ValueError(f"Pretrained weights were not found:\n{pretrained_path}")
        return

    if pt_path:
        if self._is_ultralytics_export_model_path(pt_path):
            raise ValueError(
                "Exported models such as .onnx and .engine cannot start Ultralytics training. "
                    "Select a .pt model, or select a model .yaml with optional .pt pretrained weights."
            )
        if not pt_path.lower().endswith(".pt"):
            raise ValueError("Training weights must be a .pt file, or use a model .yaml config.")
        if not os.path.exists(pt_path):
            raise ValueError(f"Training weights were not found:\n{pt_path}")
        return

    raise ValueError("Select a .pt model or a model .yaml config before training.")


def _ultralytics_distillation_args(self, mode="train", task="detect", raise_on_error=True):
    if mode != "train":
        return []

    if not bool(getattr(self, "ultralytics_distill_enabled", False)):
        return []

    teacher_path = self.normalize_path(getattr(self, "ultralytics_distill_model_path", "") or "")
    if not teacher_path:
        if raise_on_error:
            raise ValueError("Knowledge Distillation is enabled, but no teacher model is selected.")
        return []

    if not os.path.exists(teacher_path) and not self._looks_like_ultralytics_model_reference(teacher_path):
        if raise_on_error:
            raise ValueError(f"Knowledge Distillation teacher model was not found:\n{teacher_path}")
        return []

    if not self.ultralytics_version_at_least("8.4.77"):
        if raise_on_error:
            raise ValueError(
                "Knowledge Distillation requires Ultralytics v8.4.77 or newer. "
                    "Upgrade the fusion environment before using distill_model."
            )
        return []

    args = [f"distill_model={teacher_path}"]
    try:
        weight = float(getattr(self, "ultralytics_distill_weight", 6.0))
    except Exception:
        weight = 6.0
    if math.isfinite(weight):
        weight = max(0.0, min(10.0, weight))
        args.append(f"dis={weight:.6g}")

    student_path = self.normalize_path(
        getattr(self, "model_config_path", "") or getattr(self, "pt_path", "") or self._training_eval_model_path()
    )
    for note in self._training_eval_distillation_sanity_notes(student_path, teacher_path, weight, task):
        message = note.get("text", "")
        if not message:
            continue
        if note.get("severity") == "warning":
            logger.warning(message)
        else:
            logger.info(message)

    if str(task or "").strip().lower() != "detect":
        logger.info(
            "Knowledge Distillation is enabled for task=%s. Ultralytics currently benchmarks detect first.",
            task,
        )
    return args


def _training_eval_resolve_command_data_yaml(self, result=None):
    """Return the selected existing dataset YAML without modifying it."""
    result = result if isinstance(result, dict) else {}
    rec = result.get("recommendations", {}) if isinstance(result.get("recommendations", {}), dict) else {}
    data_yaml = self.normalize_path(
        rec.get("data_yaml_path", "")
        or result.get("data_yaml_path", "")
        or getattr(self, "data_yaml_path", "")
        or ""
    )
    # Training uses the explicitly selected data YAML. The folder open in
    # the image browser is relevant only to Generate Files + Parameters,
    # not to launching an already prepared training run.
    if data_yaml and os.path.isfile(data_yaml):
        return data_yaml

    if not data_yaml:
        raise ValueError("Select the existing obj.yaml before starting training.")
    raise ValueError(
        f"The selected training YAML no longer exists:\n{data_yaml}\n\n"
            "Starting training does not recreate dataset files. Select an existing obj.yaml, "
            "or explicitly click Generate Files + Parameters."
    )


def build_training_evaluator_command(self, result, mode="train", probe_epochs=None, apply_optimized_args=True, run_name=None):
    if not result:
        raise ValueError("Run Generate Files + Parameters first.")

    rec = result.get("recommendations", {})
    task = str(rec.get("task") or (getattr(self, "task_combobox", None).currentText() if getattr(self, "task_combobox", None) is not None else "detect")).strip().lower()
    if task not in {"detect", "segment", "classify", "pose", "obb"}:
        raise ValueError(f"Invalid Ultralytics task: {task}")

    data_yaml = self._training_eval_resolve_command_data_yaml(result)

    model_args = self._training_eval_model_command_args(result, mode=mode)
    if not model_args:
        raise ValueError("No model/weights selected for validation or training.")

    project_dir = self._training_eval_project_dir(result)
    os.makedirs(project_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not run_name:
        if mode == "val":
            run_name = f"eval_val_{timestamp}"
        else:
            run_name = f"{'probe_train' if probe_epochs is not None else 'evaluator_train'}_{timestamp}"

    prefix = self._ultralytics_train_command_prefix() if mode == "train" else self._ultralytics_cli_command_prefix()
    model_factor = float((result.get("weight_info", {}) or {}).get("model_factor", 1.0) or 1.0)
    if mode == "val":
        batch_value = self._training_eval_effective_validation_batch(rec, result.get("image_count", 0), model_factor)
    else:
        batch_value = self._training_eval_effective_batch(rec, result.get("image_count", 0), model_factor)
    cmd = [
        *prefix,
        task,
        mode,
        f"data={data_yaml}",
        *model_args,
        f"imgsz={int(rec.get('imgsz', self._parse_training_imgsz_value()))}",
        f"batch={batch_value}",
        f"project={project_dir}",
        f"plots={'False' if mode == 'val' else 'True'}",
    ]

    if mode == "val":
        split = str(result.get("metrics_split", "val"))
        if split not in {"all", "train", "val", "test"}:
            raise ValueError(f"Invalid metrics split: {split}")
        if task == "classify" and split == "all":
            raise ValueError("Select Validation or Test for classification metrics; Entire Dataset requires an annotation dataset YAML.")
        cmd.extend([
            f"name={run_name if run_name else f'eval_val_{timestamp}'}",
            "save_json=True",
            f"split={split}",
            "exist_ok=True",
        ])
        self._append_ultralytics_memory_guard_args(cmd)
        return cmd

    epochs = int(probe_epochs) if probe_epochs is not None else int(rec.get("epochs", 100))
    cmd.extend([
        f"epochs={epochs}",
        f"patience={max(3, min(int(rec.get('patience', 40)), max(3, epochs // 2 if probe_epochs else int(rec.get('patience', 40)))))}",
        f"name={run_name}",
        f"amp={_training_precision(self, result, apply_optimized_args)}",
        f"rect={'True' if rec.get('rect', False) else 'False'}",
        "exist_ok=True",
    ])

    freeze_value = int(rec.get("freeze", 0) or 0)
    cmd.append(f"freeze={freeze_value}")

    cmd.extend(self._ultralytics_distillation_args(mode=mode, task=task))

    # With recommendations off, read the exact dataset's persisted tuning
    # values, including after reopening DarkFusion without Generate.
    cmd.extend(self._training_eval_selected_parameter_args(result, apply_optimized_args))

    self._append_ultralytics_memory_guard_args(cmd)
    return cmd


def build_training_evaluator_tune_command(
    self,
    result,
    iterations=10,
    epochs_per_trial=10,
    use_ray=False,
    include_distillation=False,
):
    if not result:
        raise ValueError("Run Generate Files + Parameters first.")

    rec = result.get("recommendations", {})
    task = str(rec.get("task") or (getattr(self, "task_combobox", None).currentText() if getattr(self, "task_combobox", None) is not None else "detect")).strip().lower()
    if task not in {"detect", "segment", "classify", "pose", "obb"}:
        raise ValueError(f"Invalid Ultralytics task: {task}")

    data_yaml = self._training_eval_resolve_command_data_yaml(result)

    model_args = self._training_eval_model_command_args(result, mode="train")
    model_path = ""
    pretrained_arg = ""
    for arg in model_args:
        if str(arg).startswith("model="):
            model_path = str(arg).split("=", 1)[1]
        elif str(arg).startswith("pretrained="):
            pretrained_arg = str(arg)
    if not model_path:
        raise ValueError("No model selected for tuning.")

    tune_script = os.path.join(APP_DIR, "darkfusion_ultralytics_tune.py")
    if not os.path.exists(tune_script):
        raise ValueError(f"Tune launcher not found:\n{tune_script}")

    project_dir = self._training_eval_project_dir(result)
    os.makedirs(project_dir, exist_ok=True)
    run_name = f"evaluator_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_factor = float((result.get("weight_info", {}) or {}).get("model_factor", 1.0) or 1.0)
    batch_value = self._training_eval_effective_batch(rec, result.get("image_count", 0), model_factor)

    cmd = [
        sys.executable,
        tune_script,
        f"--model={model_path}",
        f"--iterations={max(1, int(iterations or 1))}",
        "--",
        f"data={data_yaml}",
        f"imgsz={int(rec.get('imgsz', self._parse_training_imgsz_value()))}",
        f"batch={batch_value}",
        f"epochs={max(1, int(epochs_per_trial or 1))}",
        f"project={project_dir}",
        f"name={run_name}",
        f"amp={_training_precision(self, result)}",
        f"rect={'True' if rec.get('rect', False) else 'False'}",
        "plots=False",
        "save=False",
        "val=True",
        "exist_ok=True",
    ]
    if use_ray:
        cmd.insert(3, "--use-ray")
        cmd.insert(4, "--gpu-per-trial=auto")

    if pretrained_arg:
        cmd.append(pretrained_arg)

    freeze_value = int(rec.get("freeze", 0) or 0)
    if freeze_value > 0:
        cmd.append(f"freeze={freeze_value}")

    if include_distillation:
        cmd.extend(self._ultralytics_distillation_args(mode="train", task=task))

    self._append_ultralytics_memory_guard_args(cmd)
    return cmd


def build_training_evaluator_batch_command(self, result, fraction=0.60):
    if not result:
        raise ValueError("Run Generate Files + Parameters first.")

    rec = result.get("recommendations", {})
    data_yaml = self._training_eval_resolve_command_data_yaml(result)

    model_args = self._training_eval_model_command_args(result, mode="train")
    model_path = ""
    for arg in model_args:
        if str(arg).startswith("model="):
            model_path = str(arg).split("=", 1)[1]
            break
    if not model_path:
        raise ValueError("No model selected for batch calibration.")

    batch_script = os.path.join(APP_DIR, "darkfusion_ultralytics_batch.py")
    if not os.path.exists(batch_script):
        raise ValueError(f"Batch calibration launcher not found:\n{batch_script}")

    project_dir = self._training_eval_project_dir(result)
    run_name = f"batch_calibrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = self.normalize_path(os.path.join(project_dir, run_name))
    output_path = self.normalize_path(os.path.join(run_dir, "batch_calibration.json"))
    os.makedirs(run_dir, exist_ok=True)

    return [
        sys.executable,
        batch_script,
        f"--model={model_path}",
        f"--data={data_yaml}",
        f"--imgsz={int(rec.get('imgsz', self._parse_training_imgsz_value()))}",
        f"--amp={str(_training_precision(self, result)).lower()}",
        f"--fraction={float(fraction):.3f}",
        f"--output={output_path}",
    ]


def _format_command_display(self, command):
    try:
        return self.format_command_for_log(command)
    except Exception:
        return subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command)


@gpu_work("Training / evaluation", keep=(), external=True, exclusive=True, nested_launcher=True)
def launch_training_evaluator_command(self, command, title="Training Evaluator", keep_open=True):
    if not command:
        raise ValueError("No command to launch.")

    command_display = self._format_command_display(command)
    working_dir = APP_DIR
    logger.info(f"{title} command: {command_display}")

    if sys.platform.startswith("win"):
        if keep_open:
            return subprocess.Popen(["cmd.exe", "/k", command_display], cwd=working_dir)
        return subprocess.Popen(command, cwd=working_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        if not keep_open:
            return subprocess.Popen(command, cwd=working_dir)
        try:
            return subprocess.Popen(["gnome-terminal", "--working-directory", working_dir, "--"] + command)
        except (FileNotFoundError, TypeError):
            return subprocess.Popen(command, cwd=working_dir)


def watch_training_evaluator_tune(self, process, command, result, results_text=None, status_label=None, optimized_args_checkbox=None, on_tuned=None):
    if process is None:
        return

    context = self._training_eval_job_context(result)
    result = context["result"]
    tune_dir = self._training_eval_tune_dir_from_command(command)
    self.training_evaluator_tune_process = process
    widgets = getattr(self, "training_evaluator_run_widgets", None)

    timer = QTimer(self)
    timer.setInterval(5000)
    self.training_evaluator_tune_timer = timer
    finished = False

    def widget_alive(widget):
        return widget is not None and not sip.isdeleted(widget)

    def finish(exit_code):
        nonlocal finished
        if finished:
            return
        finished = True
        timer.stop()
        if self.training_evaluator_tune_timer is timer:
            self.training_evaluator_tune_timer = None
        if self.training_evaluator_tune_process is process:
            self.training_evaluator_tune_process = None

        current_setup = self._training_eval_job_context_is_current(context)
        try:
            best_hyperparams, source_path = self.read_training_tune_best_hyperparameters(tune_dir)
            applied = self.apply_training_tune_best_hyperparameters(
                result, best_hyperparams, source_path,
                use_recommended=context["use_recommended"], target_yaml=context["target_yaml"],
            ) if best_hyperparams else False
        except Exception as error:
            logger.warning("Could not save tuning results to %s: %s", context["target_yaml"], error)
            if current_setup and widget_alive(status_label):
                status_label.setText(f"Could not save tune results: {error}. Original output remains in {tune_dir}.")
                self._training_eval_set_activity_indicator(widgets, "error", "Error", "Tune settings could not be saved")
            return
        if applied:
            self.training_evaluator_last_tune_result = result
            if not current_setup:
                logger.info("Tune saved to its original setup: %s; current controls left unchanged.", context["target_yaml"])
                return
            self.training_evaluator_last_result = result
            if callable(on_tuned):
                try:
                    on_tuned(result)
                except Exception as e:
                    logger.warning(f"Could not sync tune results into evaluator controls: {e}")
            if widget_alive(results_text):
                results_text.setText(self.format_training_evaluation_report(result))
            if widget_alive(status_label):
                status_label.setText("Tune finished. Best hyperparameters are loaded; Start Training will use them.")
            self._training_eval_set_activity_indicator(widgets, "success", "Ready", "Tune finished")
        elif current_setup:
            if widget_alive(status_label):
                if exit_code not in (0, None):
                    status_label.setText(f"Tune exited with code {exit_code}; no best_hyperparameters.yaml found.")
                    self._training_eval_set_activity_indicator(widgets, "error", "Error", "Tune exited without usable results")
                else:
                    status_label.setText(f"Tune finished, but no best_hyperparameters.yaml was found in {tune_dir}.")
                    self._training_eval_set_activity_indicator(widgets, "warning", "Needs attention", "Tune finished without best hyperparameters")

    def poll():
        if finished:
            return
        try:
            exit_code = process.poll()
        except Exception as e:
            timer.stop()
            if widget_alive(status_label):
                status_label.setText(f"Tune watch failed: {e}")
            self._training_eval_set_activity_indicator(widgets, "error", "Error", "Tune watch failed")
            return

        if exit_code is None:
            # The regular run monitor owns live trial metrics and status.
            # This watcher only loads the best settings when tuning exits.
            return
        finish(exit_code)

    timer.timeout.connect(poll)
    timer.start()
    QTimer.singleShot(1000, poll)


def watch_training_evaluator_batch(self, process, command, result, results_text=None, status_label=None, on_calibrated=None):
    if process is None:
        return

    context = self._training_eval_job_context(result)
    result = context["result"]
    output_path = self._training_eval_option_value(command, "--output")
    self.training_evaluator_batch_process = process
    widgets = getattr(self, "training_evaluator_run_widgets", None)

    timer = QTimer(self)
    timer.setInterval(2000)
    self.training_evaluator_batch_timer = timer
    finished = False

    def widget_alive(widget):
        return widget is not None and not sip.isdeleted(widget)

    def finish(exit_code):
        nonlocal finished
        if finished:
            return
        finished = True
        timer.stop()
        if self.training_evaluator_batch_timer is timer:
            self.training_evaluator_batch_timer = None
        if self.training_evaluator_batch_process is process:
            self.training_evaluator_batch_process = None

        current_setup = self._training_eval_job_context_is_current(context)
        calibration, source_path = self.read_training_batch_calibration(output_path)
        if calibration.get("error"):
            if current_setup and widget_alive(status_label):
                status_label.setText(f"Batch calibration failed: {calibration.get('error')}")
                self._training_eval_set_activity_indicator(widgets, "error", "Error", "Batch calibration failed")
            return

        try:
            applied = self.apply_training_batch_calibration(
                result, calibration, source_path, target_yaml=context["target_yaml"],
                use_recommended=context["use_recommended"], sync_controls=current_setup,
            )
        except Exception as error:
            logger.warning("Could not save batch calibration to %s: %s", context["target_yaml"], error)
            if current_setup and widget_alive(status_label):
                status_label.setText(f"Could not save batch calibration: {error}. Original output remains in {output_path}.")
                self._training_eval_set_activity_indicator(widgets, "error", "Error", "Batch settings could not be saved")
            return
        if applied:
            self.training_evaluator_last_batch_result = result
            if not current_setup:
                logger.info("Batch calibration saved for its original setup; current controls left unchanged.")
                return
            if callable(on_calibrated):
                try:
                    on_calibrated(result)
                except Exception as e:
                    logger.warning(f"Could not sync batch calibration into evaluator controls: {e}")
            if widget_alive(results_text):
                results_text.setText(self.format_training_evaluation_report(result))
            if widget_alive(status_label):
                status_label.setText(f"Batch calibrated to {calibration.get('batch')}. Start Training will use it.")
            self._training_eval_set_activity_indicator(widgets, "success", "Ready", "Batch calibration complete")
        elif current_setup:
            if widget_alive(status_label):
                if exit_code not in (0, None):
                    status_label.setText(f"Batch calibration exited with code {exit_code}; no usable result found.")
                    self._training_eval_set_activity_indicator(widgets, "error", "Error", "Batch calibration exited without result")
                else:
                    status_label.setText("Batch calibration finished, but no usable batch result was found.")
                    self._training_eval_set_activity_indicator(widgets, "warning", "Needs attention", "No usable batch result")

    def poll():
        if finished:
            return
        try:
            exit_code = process.poll()
        except Exception as e:
            timer.stop()
            if widget_alive(status_label):
                status_label.setText(f"Batch calibration watch failed: {e}")
            self._training_eval_set_activity_indicator(widgets, "error", "Error", "Batch calibration watch failed")
            return

        if exit_code is None:
            if widget_alive(status_label):
                status_label.setText("Batch calibration running...")
            self._training_eval_set_activity_indicator(widgets, "running", "Working", "Batch calibration running")
            return
        finish(exit_code)

    timer.timeout.connect(poll)
    timer.start()
    QTimer.singleShot(1000, poll)


def _ultralytics_extra_train_args(self):
    args_text = ""
    if isinstance(getattr(self, "settings", None), dict):
        args_text = str(self.settings.get("ultralyticsExtraTrainArgs", "") or "")
    args_text = str(getattr(self, "ultralytics_extra_train_args", args_text) or args_text).strip()
    if not args_text:
        return []

    try:
        parts = shlex.split(args_text, posix=False)
    except Exception:
        parts = args_text.split()

    cleaned = []
    blocked_keys = {
        "data",
        "model",
        "imgsz",
        "epochs",
        "batch",
        "project",
        "amp",
        "resume",
        "rect",
        "freeze",
        "patience",
        "workers",
        "cache",
        "distill_model",
        "dis",
    }
    for part in parts:
        token = str(part).strip().strip('"').strip("'")
        if not token or "=" not in token:
            continue
        key = token.split("=", 1)[0].strip().lower()
        if key in blocked_keys:
            continue
        cleaned.append(token)
    return cleaned


def ultralytics_train_clicked(self):
    self.data_yaml_path = self.normalize_path(getattr(self, "data_yaml_path", "") or "")
    if not self.data_yaml_path or not os.path.exists(self.data_yaml_path):
        QMessageBox.warning(
            self,
            "Training YAML Required",
            "Select an existing Data YAML before training. Starting training never creates or refreshes obj.yaml."
        )
        return

    selected_task = self.task_combobox.currentText().lower()
    if selected_task not in ["detect", "segment", "classify", "pose", "obb"]:
        QMessageBox.critical(self, "Error", f"Invalid task selected: {selected_task}")
        return

    try:
        self._validate_ultralytics_train_model_selection()
    except ValueError as e:
        QMessageBox.warning(self, "Ultralytics Training", str(e))
        return

    keep_duplicate_labels = bool(
        getattr(self, "keep_duplicate_labels_checkbox", None)
        and self.keep_duplicate_labels_checkbox.isChecked()
    )
    cmd_prefix = self._ultralytics_train_command_prefix()
    if keep_duplicate_labels:
        wrapper_path = os.path.join(APP_DIR, "darkfusion_ultralytics_train.py")
        if not os.path.isfile(wrapper_path):
            QMessageBox.critical(self, "Error", f"Training wrapper not found:\n{wrapper_path}")
            return

    model_path_for_batch = ""
    if self.model_config_path:
        model_path_for_batch = self.model_config_path
    elif hasattr(self, "pt_path") and self.pt_path:
        model_path_for_batch = self.pt_path
    dataset_size = self._training_yaml_image_count(self.data_yaml_path)
    if not dataset_size:
        dataset_size = len(getattr(self, "images", []) or [])
    model_factor = self._training_eval_model_factor_from_path(model_path_for_batch)
    requested_batch = self.batch_input.text()
    batch_value = self._training_eval_clamp_batch(requested_batch, dataset_size, model_factor)
    project_dir = self._training_eval_project_dir()
    run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cmd = [
        *cmd_prefix, selected_task, "train",
        f"data={self.data_yaml_path}",
        f"imgsz={self.imgsz_input.text()}",
        f"epochs={self.epochs_input.text()}",
        f"batch={batch_value}",
        f"project={project_dir}",
        f"name={run_name}",
    ]

    if self.model_config_path:
        cmd.append(f"model={self.model_config_path}")
        if hasattr(self, "pretrained_model_path") and self.pretrained_model_path:
            cmd.append(f"pretrained={self.pretrained_model_path}")
    elif hasattr(self, "pt_path") and self.pt_path:
        cmd.append(f"model={self.pt_path}")

    precision = _training_precision(self)
    if getattr(self, "amp_true", None) is not None:
        self.amp_true.setChecked(precision not in (False, "fp32"))
    cmd.append(f"amp={precision}")
    if self.resume_checkbox.isChecked():
        cmd.append("resume=True")
    if getattr(self, "rect_train_checkbox", None) is not None and self.rect_train_checkbox.isChecked():
        cmd.append("rect=True")
    if self.freeze_checkbox.isChecked():
        cmd.append(f"freeze={self.freeze_input.value()}")
    if self.patience_checkbox.isChecked():
        cmd.append(f"patience={self.patience_input.text()}")
    distill_notes = []
    if bool(getattr(self, "ultralytics_distill_enabled", False)):
        distill_notes = self._training_eval_distillation_sanity_notes(
            model_path_for_batch,
            getattr(self, "ultralytics_distill_model_path", ""),
            getattr(self, "ultralytics_distill_weight", 6.0),
            selected_task,
        )
    try:
        cmd.extend(self._ultralytics_distillation_args(mode="train", task=selected_task))
    except ValueError as e:
        QMessageBox.warning(self, "Knowledge Distillation", str(e))
        return
    
    # Keep the older training entry point on the same parameter source as
    # the trainer dialog. It must not re-enable cfg= when recommendations
    # are off or lose tuning saved in the selected dataset YAML.
    try:
        cmd.extend(self._training_eval_selected_parameter_args(
            {"data_yaml_path": self.data_yaml_path},
            bool(self.settings.get("trainingEvaluatorUseOptimizedArgs", True)),
        ))
    except (ValueError, OSError) as error:
        QMessageBox.warning(self, "Training Parameters", str(error))
        return
    
    self._append_ultralytics_memory_guard_args(cmd, use_evaluator_overrides=True)

    # Start through the evaluator logger so crash recovery can find the latest run.
    try:
        run_dir_hint = self.normalize_path(os.path.join(project_dir, run_name))
        process, run_dir, log_path = self.launch_training_evaluator_logged_command(
            cmd,
            "Ultralytics Training",
            run_dir=run_dir_hint,
        )
        widgets = getattr(self, "training_evaluator_run_widgets", None)
        status_label = getattr(self, "training_evaluator_status_label", None)
        tabs = widgets.get("tabs") if isinstance(widgets, dict) else None
        if tabs is not None and not sip.isdeleted(tabs):
            tabs.setCurrentIndex(1)
        if isinstance(widgets, dict):
            self.watch_training_evaluator_run(
                process,
                cmd,
                "Training",
                run_dir,
                log_path,
                widgets=widgets,
                status_label=status_label,
            )
        else:
            self._training_eval_record_run(
                "Ultralytics Training",
                run_dir,
                log_path,
                command=cmd,
                process=process,
            )

        message = (
            "Training is running in a logged console. The evaluator can reopen this run, "
                "show live metrics, and resume from weights/last.pt if it is interrupted."
        )
        message += f"\n\nRun: {run_dir}"
        if parse_training_batch(requested_batch) != batch_value:
            message += f"\n\nBatch was limited from {requested_batch} to the {batch_value} training images available."
        if keep_duplicate_labels:
            message += "\n\nUltralytics duplicate-label row removal is disabled for this training run."
        if distill_notes:
            message += "\n\nDistillation notes:\n" + self._training_eval_format_distillation_notes(distill_notes)
        
        cfg_path = self._training_eval_command_value(cmd, "cfg")
        if cfg_path:
            message += f"\n\nTraining config: {cfg_path}"
        
        QMessageBox.information(self, "Training Started", message)
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to start training: {str(e)}")


def handle_stdout(self):
    output = self.process.readAllStandardOutput().data().decode()

    # Keep progress output out of the persistent logger.
    if "it/s" in output or "%" in output:
        print(output, end="\r")  # Show progress inline, avoid cluttering logs
    else:
        print(output)


def handle_stderr(self):
    error = self.process.readAllStandardError().data().decode()

    if "it/s" in error or "%" in error:
        print(error, end="\r")
    else:
        print(f"\033[91m{error}\033[0m")  # Red color for errors


def process_finished(self):
    logger.info("Training process finished.")
    self.process = None


@staticmethod
def ultralytics_export_formats():
    """Use the installed exporter's catalog, not a stale nine-format list."""
    try:
        from ultralytics.engine.exporter import export_formats
        return [str(value) for value in export_formats()["Argument"] if value != "-"]
    except Exception as error:
        logger.debug("Could not read Ultralytics export formats: %s", error)
        return []


def saved_export_batch_size(self):
    """Restore a custom export batch only when explicitly enabled."""
    if not self.settings.get("exportBatchEnabled", False):
        return 1
    try:
        return max(1, int(self.settings.get("exportBatchSize", 1) or 1))
    except (TypeError, ValueError, OverflowError):
        return 1


def build_ultralytics_export_params(
    self,
    format_selected,
    width,
    height,
    use_gpu=True,
    precision="fp32",
    dynamic=False,
    simplify=False,
    nms=False,
    agnostic_nms=False,
    confidence=0.25,
    iou=0.7,
    batch_enabled=False,
    batch=1,
    data_yaml="",
    extra_args="",
):
    """Build validated kwargs for the installed Ultralytics exporter."""
    export_format = str(format_selected or "").strip().lower()
    if not export_format:
        raise ValueError("Choose an export format.")
    formats = self.ultralytics_export_formats()
    if formats and export_format not in formats:
        # This alias was replaced upstream by LiteRT; retain old saved settings.
        if export_format == "tflite" and "litert" in formats:
            export_format = "litert"
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    width = self.parse_positive_int(width, "Export width")
    height = self.parse_positive_int(height, "Export height")
    precision = str(precision or "fp32").strip().lower()

    params = {
        "format": export_format,
        "imgsz": [height, width],
        "device": "0" if bool(use_gpu) and torch.cuda.is_available() else "cpu",
    }

    from ultralytics.engine import exporter
    native_quantize = hasattr(exporter, "FP16_FORMATS")
    precision_values = {"fp32": None, "fp16": 16, "int8": 8, "w8a16": "w8a16", "w8a32": "w8a32"}
    if precision not in precision_values:
        raise ValueError(f"Unknown export precision: {precision}")
    capability = "half" if precision == "fp16" else precision
    if not self.is_parameter_supported(export_format, capability):
        raise ValueError(f"{export_format} does not support {precision.upper()} export.")
    if native_quantize:
        params["quantize"] = precision_values[precision]
    else:
        params.update(half=precision == "fp16", int8=precision == "int8")
    if data_yaml and self.is_parameter_supported(export_format, "data"):
        params["data"] = self.normalize_path(data_yaml)

    if self.is_parameter_supported(export_format, "dynamic"):
        params["dynamic"] = bool(dynamic)
    if self.is_parameter_supported(export_format, "simplify"):
        params["simplify"] = bool(simplify)
    if self.is_parameter_supported(export_format, "batch"):
        params["batch"] = self.parse_positive_int(batch, "Export batch") if batch_enabled else 1
    # False retains native NMS-free inference when the checkpoint supports it.
    # None explicitly selects raw one-to-many output in recent Ultralytics,
    # so it is NOT interchangeable with an unchecked Embed NMS control.
    # Legacy heads without one-to-one weights still export their raw outputs.
    params["nms"] = bool(nms)

    if bool(nms):
        if not self.is_parameter_supported(export_format, "nms"):
            raise ValueError(f"{export_format} does not support embedded NMS.")
        params.update({
            "nms": True,
            "conf": max(0.001, min(1.0, float(confidence))),
            "iou": max(0.01, min(0.99, float(iou))),
            "agnostic_nms": bool(agnostic_nms),
        })

    if str(extra_args or "").strip():
        from ultralytics.cfg import merge_equals_args, parse_key_value_pair
        native_export_keys = set().union(*exporter.export_formats()["Arguments"]) | {
            "imgsz", "device", "verbose", "conf", "iou", "agnostic_nms", "max_det",
            "quantize", "half", "int8",
        }
        lexer = shlex.shlex(str(extra_args), posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        lexer.escape = ""  # Preserve Windows path backslashes, including inside quotes.
        for token in merge_equals_args(list(lexer)):
            if "=" not in token:
                raise ValueError(f"Export arguments must use key=value: {token}")
            key, value = parse_key_value_pair(token)
            if key in {"model", "format", "mode", "task", "cfg"}:
                raise ValueError(f"Select {key} using the export controls, not extra arguments.")
            if key not in native_export_keys:
                raise ValueError(f"Not a native export argument: {key}")
            params[key] = value
    if params.get("nms") and not self.is_parameter_supported(export_format, "nms"):
        raise ValueError(f"{export_format} does not support embedded NMS.")
    if params.get("opset") is not None:
        if isinstance(params["opset"], bool) or not isinstance(params["opset"], int) or params["opset"] < 1:
            raise ValueError("Export opset must be a positive integer or None.")
    # Use the same type and per-target validation as the native exporter.
    from ultralytics.cfg import get_cfg
    config = get_cfg(overrides=params)
    table = exporter.export_formats()
    index = list(table["Argument"]).index(export_format)
    exporter.validate_args(export_format, config, table["Arguments"][index])
    calibration_precision = getattr(config, "quantize", None) if native_quantize else (8 if config.int8 else None)
    if calibration_precision in {8, "w8a16"} and self.is_parameter_supported(export_format, "data"):
        calibration_path = self.normalize_path(config.data or "")
        if not calibration_path or not os.path.isfile(calibration_path) or not calibration_path.lower().endswith((".yaml", ".yml")):
            raise ValueError("INT8 export requires an existing dataset YAML for calibration.")
        params["data"] = calibration_path
    return params


@gpu_work("Model export", keep=("export",))
def run_ultralytics_export(self, model_path, export_params):
    """Load a PT checkpoint and export it with already validated parameters."""
    model_path = self.normalize_path(model_path)
    if not model_path or not os.path.isfile(model_path):
        raise ValueError("Select an existing Ultralytics .pt checkpoint to export.")
    if not model_path.lower().endswith(".pt"):
        raise ValueError("Ultralytics export requires a .pt checkpoint.")

    model = YOLO(model_path)
    output = model.export(**dict(export_params or {}))
    self._export_model = model
    self._export_model_path = model_path
    self.export_model_path = model_path
    self.settings["exportModelPath"] = model_path
    self.queue_settings_save()
    return self.normalize_path(str(output))


def load_model(self):
    if not gpu_model_allowed(self, "export"):
        return
    fname, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        'Open file',
        self.dialog_start_directory("export"),
        "Ultralytics checkpoints (*.pt)",
    )

    if fname:
        self.remember_dialog_selection("export", fname)
        logger.info(f"Loading model from file: {fname}")
        if not os.path.isfile(fname) or not fname.lower().endswith(".pt"):
            QMessageBox.warning(self, "Export Model", "Select an existing Ultralytics .pt checkpoint.")
            return
        try:
            export_model = YOLO(fname)
        except Exception as error:
            QMessageBox.critical(self, "Export Model", str(error))
            return
        self._export_model = export_model
        self._export_model_path = fname
        self.export_model_path = fname
        self.queue_settings_save()
        logger.info(f"Model loaded successfully from {fname}")
    else:
        logger.warning("No model file selected.")


def initiate_conversion(self):
    # Compatibility entry point: never convert an unrelated hard-coded model.onnx.
    self.handle_conversion()


def export_model_for_conversion(self):
    if not gpu_model_allowed(self, "export"):
        raise GpuWorkBusy("Export model loading is paused while another GPU workflow is running.")
    model = getattr(self, "_export_model", None)
    path = self.normalize_path(getattr(self, "export_model_path", "") or getattr(self, "weights_file_path", ""))
    cached_path = self.normalize_path(getattr(self, "_export_model_path", ""))
    if model is not None and (not path or os.path.normcase(cached_path) == os.path.normcase(path)):
        return model
    if not path or not os.path.isfile(path) or not path.lower().endswith(".pt"):
        raise ValueError("Select an existing Ultralytics .pt checkpoint for export.")
    # Export can fuse/change a checkpoint in place. Never use the live
    # inference object, even when it refers to the same .pt file.
    self._export_model = YOLO(path)
    self._export_model_path = path
    self.export_model_path = path
    return self._export_model


def handle_conversion(self):
    try:
        export_model = self.export_model_for_conversion()
    except Exception as error:
        QMessageBox.warning(self, "Export Model", str(error))
        logger.warning("Conversion has no usable export checkpoint: %s", error)
        return

    format_selected = self.convert_model.currentText()
    logger.info(f"Handling conversion for format: {format_selected}")
    self.update_gui_elements(format_selected)  # Sync GUI before collecting inputs

    try:
        imgsz_list = self.get_export_imgsz_hw()
        logger.info(
            "Export imgsz set to [H, W]=%s from UI W=%s H=%s",
            imgsz_list,
            self.imgsz_input_W.text(),
            self.imgsz_input_H.text(),
        )
    except ValueError as e:
        QMessageBox.critical(self, "Conversion Error", str(e))
        logger.error(f"Invalid export image size: {e}")
        return

    try:
        half = self.half_true.isChecked() and self.half_true.isEnabled()
        int8 = self.int8_true.isChecked() and self.int8_true.isEnabled()
        if half and int8:
            raise ValueError("Choose either FP16 or INT8, not both.")
        selected_precision = "int8" if int8 else "fp16" if half else "fp32"
        export_params = self.build_ultralytics_export_params(
            format_selected, imgsz_list[1], imgsz_list[0],
            use_gpu=self.use_gpu_checkbox.isChecked(),
            precision=selected_precision,
            dynamic=self.dynamic_true.isChecked() and self.dynamic_true.isEnabled(),
            simplify=self.simplify.isChecked() and self.simplify.isEnabled(),
            nms=self.nms_true_checkbox.isChecked() and self.nms_true_checkbox.isEnabled(),
            agnostic_nms=self.agnostic_checkbox.isChecked(),
            confidence=self.conf_spinbox.value(), iou=self.iou_spinbox.value(),
            batch_enabled=self.batch_size_checkbox.isChecked() and self.batch_size_checkbox.isEnabled(),
            batch=self.batch_input_spinbox.value(),
            data_yaml=getattr(self, "data_yaml_path", ""),
            extra_args=self.settings.get("exportExtraArgs", ""),
        )
        output = export_model.export(**export_params)
        self.settings["exportPrecision"] = selected_precision
        QMessageBox.information(self, "Conversion Successful", f'Exported model:\n{output}')
        logger.info(f"Export completed with parameters: {export_params}")
    except Exception as e:
        QMessageBox.critical(self, "Conversion Error", str(e))
        logger.error(f"Model export failed: {e}")


def is_parameter_supported(self, format_selected, param):
    export_format = str(format_selected or "").strip().lower()
    parameter = str(param or "").strip().lower()
    try:
        from ultralytics.engine import exporter
        precision_sets = {"half": "FP16_FORMATS", "int8": "INT8_FORMATS",
                          "w8a16": "W8A16_FORMATS", "w8a32": "W8A32_FORMATS"}
        if parameter == "fp32":
            return export_format not in getattr(exporter, "FP32_UNSUPPORTED_FORMATS", ())
        if parameter in precision_sets and hasattr(exporter, precision_sets[parameter]):
            return export_format in getattr(exporter, precision_sets[parameter])
        if parameter in {"w8a16", "w8a32"}:
            return False

        formats = exporter.export_formats()
        arguments = list(formats.get("Argument", ()) or ())
        supported = list(formats.get("Arguments", ()) or ())
        if export_format in arguments:
            index = arguments.index(export_format)
            return parameter in set(supported[index] if index < len(supported) else ())
    except Exception as e:
        logger.debug("Could not read live Ultralytics export capabilities: %s", e)

    # Compatibility fallback for older Ultralytics builds.
    supported_params = {
        'torchscript': {'imgsz', 'half', 'optimize', 'nms', 'batch', 'device'},
        'onnx': {'imgsz', 'half', 'dynamic', 'simplify', 'opset', 'nms', 'batch', 'device'},
        'openvino': {'imgsz', 'half', 'dynamic', 'int8', 'nms', 'batch', 'device'},
        'engine': {'imgsz', 'half', 'dynamic', 'simplify', 'workspace', 'int8', 'nms', 'batch', 'data', 'fraction', 'device'},
        'coreml': {'imgsz', 'half', 'int8', 'nms', 'batch', 'device'},
        'saved_model': {'imgsz', 'keras', 'int8', 'nms', 'batch', 'device'},
        'pb': {'imgsz', 'batch', 'device'},
        'tflite': {'imgsz', 'half', 'int8', 'nms', 'batch', 'data', 'fraction', 'device'},
        'edgetpu': {'imgsz', 'device'},
        'tfjs': {'imgsz', 'half', 'int8', 'nms', 'batch', 'device'},
        'paddle': {'imgsz', 'batch', 'device'},
        'mnn': {'imgsz', 'batch', 'int8', 'half', 'device'},
        'ncnn': {'imgsz', 'half', 'batch', 'device'},
        'imx': {'imgsz', 'int8', 'data', 'fraction', 'device'},
        'rknn': {'imgsz', 'batch', 'name', 'device'}
    }

    logger.debug(f"Checking if parameter '{parameter}' is supported for format '{export_format}'")
    return parameter in supported_params.get(export_format, set())


def update_gui_elements(self, format_selected):
    self.half_true.setEnabled(self.is_parameter_supported(format_selected, 'half'))
    if not self.is_parameter_supported(format_selected, 'half'):
        self.half_true.setChecked(False)

    supports_nms = self.is_parameter_supported(format_selected, 'nms')
    self.nms_true_checkbox.setEnabled(supports_nms)
    if not supports_nms:
        self.nms_true_checkbox.setChecked(False)

    self.int8_true.setEnabled(self.is_parameter_supported(format_selected, 'int8'))
    if not self.is_parameter_supported(format_selected, 'int8'):
        self.int8_true.setChecked(False)

    self.simplify.setEnabled(self.is_parameter_supported(format_selected, 'simplify'))
    if not self.is_parameter_supported(format_selected, 'simplify'):
        self.simplify.setChecked(False)

    self.dynamic_true.setEnabled(self.is_parameter_supported(format_selected, 'dynamic'))
    if not self.is_parameter_supported(format_selected, 'dynamic'):
        self.dynamic_true.setChecked(False)

    self.batch_size_checkbox.setEnabled(self.is_parameter_supported(format_selected, 'batch'))
    if not self.is_parameter_supported(format_selected, 'batch'):
        self.batch_size_checkbox.setChecked(False)

    self.conf_spinbox.setEnabled(supports_nms and self.nms_true_checkbox.isChecked())
    self.iou_spinbox.setEnabled(supports_nms and self.nms_true_checkbox.isChecked())
    self.agnostic_checkbox.setEnabled(supports_nms and self.nms_true_checkbox.isChecked())

    # Prevent stacking only our own format-refresh connection. Do not
    # disconnect unrelated slots such as persistent settings autosave.
    previous_slot = getattr(self, "_nms_format_refresh_slot", None)
    if previous_slot is not None:
        try:
            self.nms_true_checkbox.stateChanged.disconnect(previous_slot)
        except TypeError:
            pass

    self._nms_format_refresh_slot = lambda *_args: self.update_gui_elements(format_selected)
    self.nms_true_checkbox.stateChanged.connect(self._nms_format_refresh_slot)


def format_changed(self):
    format_selected = self.convert_model.currentText()
    logger.debug(f"Format changed to: {format_selected}")
    self.update_gui_elements(format_selected)


def _parse_training_imgsz_value(self, value=None, fallback=640):
    text = str(value if value is not None else getattr(self, "imgsz_input", None).text() if getattr(self, "imgsz_input", None) is not None else "").strip()
    numbers = [int(match) for match in re.findall(r"\d+", text)]
    if not numbers:
        try:
            return int(fallback)
        except Exception:
            return 640
    return snap_int_to_multiple(numbers[0], 32, minimum=160, maximum=2048)

def inspect_training_eval_weights(self, weights_path):
    info = {
        "path": self.normalize_path(weights_path) if weights_path else "",
        "name": os.path.basename(weights_path) if weights_path else "No weights loaded",
        "task": "",
        "class_count": 0,
        "strides": [8, 16, 32],
        "loaded": False,
        "error": "",
        "model_factor": 1.0,
    }

    name = info["name"].lower()
    if re.search(r"(^|[^a-z])n([^a-z]|$)|nano", name):
        info["model_factor"] = 0.65
    elif re.search(r"(^|[^a-z])s([^a-z]|$)|small", name):
        info["model_factor"] = 0.85
    elif re.search(r"(^|[^a-z])m([^a-z]|$)|medium", name):
        info["model_factor"] = 1.15
    elif re.search(r"(^|[^a-z])l([^a-z]|$)|large", name):
        info["model_factor"] = 1.55
    elif re.search(r"(^|[^a-z])x([^a-z]|$)|xlarge", name):
        info["model_factor"] = 2.1

    if not weights_path:
        info["error"] = "No eval weights selected; using default YOLO stride estimate."
        return info

    if not os.path.exists(weights_path):
        info["error"] = "Eval weights path does not exist."
        return info

    if not gpu_model_allowed(self, "detector"):
        info["error"] = "Weight inspection is paused while another GPU workflow is running."
        return info

    try:
        model = YOLO(weights_path)
        info["loaded"] = True
        info["task"] = str(getattr(model, "task", "") or "")
        model_names = getattr(model, "names", None)
        if isinstance(model_names, (dict, list, tuple)):
            info["class_count"] = len(model_names)

        raw_stride = getattr(getattr(model, "model", None), "stride", None)
        if raw_stride is not None:
            if hasattr(raw_stride, "detach"):
                raw_stride = raw_stride.detach().cpu().numpy()
            elif hasattr(raw_stride, "cpu"):
                raw_stride = raw_stride.cpu().numpy()

            strides = np.asarray(raw_stride, dtype=float).reshape(-1).tolist()
            strides = sorted({int(round(value)) for value in strides if value > 0})
            if strides:
                info["strides"] = strides

        del model
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception as e:
        info["error"] = f"Could not inspect weights: {e}"

    return info

def training_eval_task_from_weights(self, weights_path):
    weights_path = self.normalize_path(weights_path or "")
    info = self.inspect_training_eval_weights(weights_path)
    raw_task = str(info.get("task", "") or "").strip().lower()
    aliases = {
        "detect": "detect",
        "detection": "detect",
        "segment": "segment",
        "seg": "segment",
        "segmentation": "segment",
        "classify": "classify",
        "classification": "classify",
        "pose": "pose",
        "keypoint": "pose",
        "keypoints": "pose",
        "obb": "obb",
    }
    task = aliases.get(raw_task, "")
    source = "model metadata" if task else ""

    if not task:
        name = os.path.basename(weights_path).lower()
        if re.search(r"(?:^|[-_.])(?:seg|segment)(?:[-_.]|$)", name):
            task = "segment"
        elif re.search(r"(?:^|[-_.])(?:pose|keypoint|keypoints)(?:[-_.]|$)", name):
            task = "pose"
        elif re.search(r"(?:^|[-_.])obb(?:[-_.]|$)", name):
            task = "obb"
        elif re.search(r"(?:^|[-_.])(?:cls|classify|classification)(?:[-_.]|$)", name):
            task = "classify"
        elif weights_path.lower().endswith((".pt", ".yaml", ".yml")):
            task = "detect"
        if task:
            source = "filename fallback"

    return task, source, info

def _training_eval_candidate_sizes(self, text, current_size, stride=32):
    return training_candidate_sizes(text, current_size, stride)

def _gpu_vram_gb(self):
    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.get_device_properties(0).total_memory) / (1024 ** 3)
    except Exception:
        try:
            gpus = GPUtil.getGPUs()
            return float(gpus[0].memoryTotal) / 1024.0 if gpus else 0.0
        except Exception:
            return 0.0

def _looks_like_official_ultralytics_weight(self, weights_path):
    name = os.path.basename(str(weights_path or "")).lower()
    stem = os.path.splitext(name)[0]
    return bool(re.match(r"^yolo(?:v?\d+)?(?:[nslmx])?(?:[-_][a-z0-9]+)*$", stem))

def recommend_training_optimizer_args(self, total_labels, recommended_row, imbalance_ratio, weight_info, weights_path, task_text):
    # A filename cannot establish whether a checkpoint needs fine-tuning,
    # nor how many layers are safe to freeze. Let the installed trainer
    # select its supported optimizer from the actual training workload.
    hyperparams = {"optimizer": "auto"}
    train_args = ["optimizer=auto"]
    # Preserve explicit extra arguments (including mask_ratio used by the
    # size analysis). Generating recommendations must not silently discard
    # a user's augmentation or optimizer settings.
    extras = self._ultralytics_extra_train_args()
    explicit_optimizer = any(arg.split("=", 1)[0].strip().lower() == "optimizer" for arg in extras)
    if explicit_optimizer:
        train_args = []
        hyperparams = {}
    train_args.extend(extras)
    for arg in extras:
        key, separator, value = arg.partition("=")
        if separator:
            try:
                hyperparams[key] = yaml.safe_load(value)
            except yaml.YAMLError:
                hyperparams[key] = value
    # The selected Hyperparameters configuration takes precedence over legacy
    # extra-argument text when generating again after a user edit.
    config_path = getattr(self, "recommended_args_path", "")
    use_config = bool(getattr(self, "settings", {}).get("trainingEvaluatorUseOptimizedArgs", True))
    selected_config = bool(use_config and config_path and os.path.isfile(config_path))
    if selected_config:
        from darkfusion_tune_settings import read_training_configuration, training_hyperparameter_groups
        selected = read_training_configuration(config_path)
        allowed = {key for keys in training_hyperparameter_groups(task_text).values() for key in keys}
        hyperparams.update({key: value for key, value in selected.items() if key in allowed})
        train_args = [f"{key}={self._training_eval_format_arg_value(value)}" for key, value in hyperparams.items()]
    optimizer_reason = (
        "Selected Hyperparameters configuration preserved." if selected_config else
        "Explicit extra arguments preserved." if explicit_optimizer else
        "Ultralytics selects the optimizer and learning rate using the actual training workload; geometry alone cannot determine them."
    )
    freeze_widget = getattr(self, "freeze_checkbox", None)
    freeze_input = getattr(self, "freeze_input", None)
    freeze = int(freeze_input.value()) if freeze_widget is not None and freeze_widget.isChecked() and freeze_input is not None else 0
    return {
        "hyperparams": hyperparams,
        "train_args": train_args,
        "train_args_text": " ".join(train_args),
        "optimizer_reason": optimizer_reason,
        "freeze": freeze,
        "batch_note": "Batch is an unmeasured starting estimate. Use Calibrate Batch with the chosen model and image size to profile memory." if torch.cuda.is_available() else "CPU/no CUDA detected: batch is an unmeasured conservative estimate.",
    }

def _write_training_recommendation_yaml(self, result, target_yaml_path=""):
    """Create/update the selected Ultralytics-compatible training cfg.

        A recommendation file is an actual Ultralytics ``cfg=`` file, rather
        than DarkFusion-only report metadata.  On its first creation it starts
        as a copy of Ultralytics' installed ``default.yaml`` (the same source
        used by ``yolo copy-cfg``), in the dataset/config directory.
        """
    result = result or {}
    rec = result.get("recommendations", {}) or {}
    data_yaml = self.normalize_path(rec.get("data_yaml_path", "") or result.get("data_yaml_path", ""))
    if not data_yaml:
        return ""

    selected_path = self.normalize_path(target_yaml_path or "")
    from darkfusion_training_runs import path_identity
    if selected_path and path_identity(selected_path) == path_identity(data_yaml):
        raise ValueError("The training configuration and dataset YAML must be different files.")
    if selected_path:
        output_path = selected_path
    else:
        output_dir = os.path.dirname(data_yaml)
        if not output_dir:
            return ""
        output_path = self.normalize_path(os.path.join(output_dir, "train_recommendations.yaml"))

    try:
        # Ultralytics' ``yolo copy-cfg`` always copies to the current
        # directory.  Copy its DEFAULT_CFG_PATH here instead so the
        # generated cfg remains beside the dataset, as DarkFusion expects.
        if not os.path.isfile(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(DEFAULT_CFG_PATH, output_path)

        from darkfusion_tune_settings import read_training_configuration
        existing = read_training_configuration(output_path)
        if not isinstance(existing, dict):
            raise ValueError("Training configuration YAML must contain a mapping.")

        # Rebase legacy metadata-only recommendation files on the current
        # Ultralytics defaults.  Preserve only recognized cfg settings so
        # that cfg= never receives DarkFusion report keys.
        payload = dict(DEFAULT_CFG_DICT)
        payload.update({key: value for key, value in existing.items() if key in DEFAULT_CFG_DICT})

        # A legacy recommendation file kept optimized values under this
        # nested key.  Migrate them into valid top-level Ultralytics args.
        legacy_hyperparams = existing.get("hyperparams", {})
        if isinstance(legacy_hyperparams, dict):
            payload.update({
                key: value for key, value in legacy_hyperparams.items()
                if key in DEFAULT_CFG_DICT
            })

        payload.update({
            "task": rec.get("task") or payload.get("task") or "detect",
            "mode": "train",
            "data": data_yaml,
            "imgsz": int(rec.get("imgsz") or payload.get("imgsz") or 640),
            "batch": parse_training_batch(rec.get("batch") or payload.get("batch") or 16),
            "epochs": int(rec.get("epochs") or payload.get("epochs") or 100),
            "patience": int(rec.get("patience") or payload.get("patience") or 100),
            "amp": rec.get("amp", payload.get("amp", True)),
            "rect": bool(rec.get("rect", payload.get("rect", False))),
            "freeze": int(rec.get("freeze", 0) or 0),
        })
        if rec.get("weights_path"):
            payload["model"] = self.normalize_path(rec["weights_path"])
        payload.update({
            key: value for key, value in (rec.get("hyperparams", {}) or {}).items()
            if key in DEFAULT_CFG_DICT
        })

        self._training_eval_write_text_atomic(
            output_path,
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        )
        return output_path
    except Exception as e:
        logger.warning(f"Could not write training recommendation YAML: {e}")
        raise ValueError(f"Could not write training recommendation YAML:\n{e}") from e

def _training_eval_format_arg_value(self, value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

def _training_eval_rebuild_train_args(self, recommendations):
    hyperparams = OrderedDict((recommendations or {}).get("hyperparams", {}) or {})
    train_args = [
        f"{key}={self._training_eval_format_arg_value(value)}"
        for key, value in hyperparams.items()
    ]
    recommendations["train_args"] = train_args
    recommendations["train_args_text"] = " ".join(train_args)

def _training_eval_command_value(self, command, key):
    prefix = f"{key}="
    for part in command or []:
        text = str(part)
        if text.startswith(prefix):
            return text[len(prefix):]
    return ""

def _training_eval_option_value(self, command, option):
    option = str(option)
    for index, part in enumerate(command or []):
        text = str(part)
        if text == option and index + 1 < len(command):
            return str(command[index + 1])
        if text.startswith(option + "="):
            return text.split("=", 1)[1]
    return ""

def _training_eval_tune_dir_from_command(self, command):
    project_dir = self._training_eval_command_value(command, "project")
    run_name = self._training_eval_command_value(command, "name")
    if project_dir and run_name:
        return self.normalize_path(os.path.join(project_dir, run_name))
    return ""

def _training_eval_model_factor_from_path(self, model_path=""):
    name = os.path.splitext(os.path.basename(str(model_path or "")))[0].lower()
    if not name:
        return 1.0
    if re.search(r"(^|[^a-z])x([^a-z]|$)|xlarge", name):
        return 2.1
    if re.search(r"(^|[^a-z])l([^a-z]|$)|large", name):
        return 1.55
    if re.search(r"(^|[^a-z])m([^a-z]|$)|medium", name):
        return 1.15
    if re.search(r"(^|[^a-z])s([^a-z]|$)|small", name):
        return 0.85
    if re.search(r"(^|[^a-z])n([^a-z]|$)|nano", name):
        return 0.65
    return 1.0

def _training_eval_clamp_batch(self, batch, dataset_size=0, model_factor=1.0):
    batch = parse_training_batch(batch)
    if batch < 1:
        return batch
    # A measured or explicitly selected batch must not be reduced by a
    # filename-based model estimate. The trainer owns memory profiling.
    return min(batch, int(dataset_size)) if int(dataset_size or 0) > 0 else batch

def _training_eval_setting_bool(self, key, default=False):
    if not isinstance(getattr(self, "settings", None), dict):
        return bool(default)
    return bool(self.settings.get(key, default))

def _training_eval_setting_int(self, key, default=0, minimum=None, maximum=None):
    value = self.settings.get(key, default) if isinstance(getattr(self, "settings", None), dict) else default
    try:
        value = int(float(str(value).strip()))
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    return value

def _training_eval_effective_batch(self, rec, dataset_size=0, model_factor=1.0):
    if self._training_eval_setting_bool("trainingEvaluatorBatchOverrideEnabled", False):
        requested = self.settings.get("trainingEvaluatorBatchOverride", 8)
    else:
        requested = (rec or {}).get(
            "batch",
            getattr(self, "batch_input", None).text() if getattr(self, "batch_input", None) is not None else 4,
        )
    return self._training_eval_clamp_batch(requested, dataset_size, model_factor)

def _training_eval_effective_validation_batch(self, rec, dataset_size=0, model_factor=1.0):
    # Validation is often used as a quick safety check; keep it much gentler than training.
    return 1

def _training_eval_cache_arg(self):
    mode = str(
        self.settings.get("trainingEvaluatorCacheMode", "off")
        if isinstance(getattr(self, "settings", None), dict)
        else "off"
    ).strip().lower()
    if mode in {"ram", "memory", "true", "on", "1"}:
        return "cache=ram"
    if mode in {"disk", "ssd"}:
        return "cache=disk"
    return "cache=False"

def _training_eval_workers_arg(self):
    if self._training_eval_setting_bool("trainingEvaluatorWorkersOverrideEnabled", False):
        workers = self._training_eval_setting_int("trainingEvaluatorWorkersOverride", 0, minimum=0, maximum=32)
    else:
        workers = 0 if sys.platform.startswith("win") else 2
    return f"workers={workers}"

def _append_ultralytics_memory_guard_args(self, command, use_evaluator_overrides=True):
    keys = {
        str(part).split("=", 1)[0].strip().lower()
        for part in command or []
        if "=" in str(part)
    }
    if "workers" not in keys:
        command.append(self._training_eval_workers_arg() if use_evaluator_overrides else ("workers=0" if sys.platform.startswith("win") else "workers=2"))
    if "cache" not in keys:
        command.append(self._training_eval_cache_arg() if use_evaluator_overrides else "cache=False")
    return command

def read_training_tune_best_hyperparameters(self, tune_dir):
    from darkfusion_tune_monitor import resolve_tune_directory

    tune_dir = self.normalize_path(tune_dir)
    if not tune_dir:
        return {}, ""
    # This reader is also used while inspecting ordinary training runs. Those
    # runs do not need a sibling-directory scan on every metrics refresh.
    if Path(tune_dir).name.lower().startswith(("evaluator_tune_", "tune")):
        tune_dir = resolve_tune_directory(tune_dir)

    candidates = [
        os.path.join(tune_dir, "best_hyperparameters.yaml"),
        os.path.join(tune_dir, "best_hyperparameters.yml"),
    ]
    for path in candidates:
        path = self.normalize_path(path)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                cleaned = {
                    str(key): value
                    for key, value in data.items()
                    if not str(key).strip().startswith("#")
                }
                return cleaned, path
        except Exception as e:
            logger.warning(f"Could not read tune hyperparameters from {path}: {e}")
    return {}, ""

def apply_training_tune_best_hyperparameters(self, result, best_hyperparams, source_path="", use_recommended=None, target_yaml=""):
    from darkfusion_tune_settings import tune_target_path, write_dataset_tuned_hyperparameters

    if not result or not isinstance(best_hyperparams, dict) or not best_hyperparams:
        return False

    rec = result.get("recommendations", {})
    if use_recommended is None:
        use_recommended = bool(getattr(self, "settings", {}).get("trainingEvaluatorUseOptimizedArgs", True))
    data_yaml = self.normalize_path(rec.get("data_yaml_path") or result.get("data_yaml_path", ""))
    if use_recommended:
        # Tuning must update the cfg currently displayed/loaded by the
        # evaluator, not quietly create a second cfg beside data.yaml.
        target_yaml = self.normalize_path(
            target_yaml or getattr(self, "recommended_args_path", "") or tune_target_path(data_yaml, True)
        )
        if not os.path.isfile(target_yaml):
            target_yaml = self._write_training_recommendation_yaml(
                result, target_yaml_path=target_yaml
            )
    else:
        target_yaml = self.normalize_path(target_yaml or tune_target_path(data_yaml, False))
    blocked = {
        "data",
        "model",
        "imgsz",
        "epochs",
        "batch",
        "project",
        "name",
        "amp",
        "rect",
        "resume",
        "freeze",
        "patience",
        "device", "workers", "cache", "save", "val", "plots", "exist_ok",
        "pretrained", "distill_model", "dis", "gpu_per_trial",
    }
    cleaned = {}
    for key, value in best_hyperparams.items():
        key = str(key).strip()
        if not key or key.lower() in blocked or key not in DEFAULT_CFG_DICT:
            continue
        cleaned[key] = value
    if not cleaned:
        return False
    if use_recommended:
        if not target_yaml or not os.path.isfile(target_yaml):
            raise ValueError("Could not create the selected training configuration YAML.")
        with open(target_yaml, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Recommendation YAML must contain a mapping: {target_yaml}")
        current = dict(rec.get("hyperparams", {}) or {})
        current.update(cleaned)
        # This is a real Ultralytics cfg, so optimized values belong at
        # the top level; nested DarkFusion metadata would make cfg= fail.
        payload.update({key: value for key, value in cleaned.items() if key in DEFAULT_CFG_DICT})
        self._training_eval_write_text_atomic(target_yaml, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
        rec["hyperparams"] = current
        self._training_eval_rebuild_train_args(rec)
        result["training_recommendations_yaml"] = target_yaml
    else:
        write_dataset_tuned_hyperparameters(target_yaml, cleaned, source_path)
    result["tune_best_hyperparameters"] = cleaned
    result["tune_best_path"] = source_path
    result["tune_target_yaml"] = target_yaml
    return True

def _training_eval_selected_parameter_args(self, result, use_recommended):
    from darkfusion_tune_settings import read_dataset_tuned_hyperparameters

    rec = (result or {}).get("recommendations", {})
    data_yaml = self.normalize_path(rec.get("data_yaml_path") or (result or {}).get("data_yaml_path") or getattr(self, "data_yaml_path", ""))
    if not data_yaml:
        return list(rec.get("train_args", [])) if use_recommended else []
    if use_recommended:
        # cfg= already supplies its settings. Cached recommendation args
        # must not silently override a newly browsed or hand-edited cfg.
        cfg_path = self._training_eval_selected_config_path(result)
        if not cfg_path and hasattr(self, "recommended_args_path"):
            raise ValueError("Select the existing training configuration, or turn Recommended Args off.")
        if cfg_path:
            from darkfusion_training_runs import path_identity
            if path_identity(cfg_path) == path_identity(data_yaml):
                raise ValueError("The training configuration and dataset YAML must be different files.")
            if not os.path.isfile(cfg_path):
                raise ValueError(f"Selected training configuration does not exist: {cfg_path}")
            with open(cfg_path, "r", encoding="utf-8") as handle:
                if not isinstance(yaml.safe_load(handle), dict):
                    raise ValueError(f"Invalid training configuration in {cfg_path}")
            return [f"cfg={cfg_path}"]
        return list(rec.get("train_args", []))
    else:
        hyperparams = read_dataset_tuned_hyperparameters(data_yaml)
    blocked = {"data", "model", "imgsz", "epochs", "batch", "project", "name", "resume", "device", "workers", "cache", "save", "val", "plots", "amp", "rect", "freeze", "patience", "exist_ok", "pretrained", "distill_model", "dis", "gpu_per_trial"}
    return [f"{key}={self._training_eval_format_arg_value(value)}" for key, value in hyperparams.items()
            if key in DEFAULT_CFG_DICT and str(key).lower() not in blocked]

def _training_eval_selected_config_path(self, result):
    # An existing (even cleared) UI selection takes precedence over cached
    # evaluation output. Headless/restored callers can use their result.
    return self.normalize_path(getattr(self, "recommended_args_path", (result or {}).get("training_recommendations_yaml", "")) or "")

def _training_eval_job_context(self, result):
    """Freeze output ownership before a background tuning/calibration job."""
    from copy import deepcopy
    from darkfusion_tune_settings import tune_target_path

    snapshot = deepcopy(result)
    settings = getattr(self, "settings", {})
    use_recommended = bool(settings.get("trainingEvaluatorUseOptimizedArgs", True))
    data_yaml = snapshot.get("recommendations", {}).get("data_yaml_path") or snapshot.get("data_yaml_path", "")
    target = (self._training_eval_selected_config_path(snapshot) or tune_target_path(data_yaml, True)) if use_recommended else data_yaml
    keys = (
        "ultralyticsDataYamlPath", "ultralyticsPtPath", "ultralyticsModelConfigPath",
        "ultralyticsPretrainedModelPath", "ultralyticsTask", "ultralyticsTrainImgSize",
        "ultralyticsTrainBatch", "ultralyticsTrainEpochs", "ultralyticsRect",
        "ultralyticsFreezeEnabled", "ultralyticsFreeze", "ultralyticsPatienceEnabled",
        "ultralyticsPatience", "ultralyticsRunsDirectory", "trainingEvaluatorUseOptimizedArgs",
        "trainingEvaluatorRecommendedArgsYaml",
    )
    return {"result": snapshot, "use_recommended": use_recommended,
            "target_yaml": self.normalize_path(target),
            "selected_cfg": self._training_eval_selected_config_path(snapshot),
            "setup": {key: deepcopy(settings.get(key)) for key in keys}}

def _training_eval_job_context_is_current(self, context):
    from darkfusion_training_runs import path_identity

    settings = getattr(self, "settings", {})

    def comparable(values):
        values = dict(values)
        values["ultralyticsTask"] = str(values.get("ultralyticsTask") or "detect").lower()
        # Saving global settings reads the legacy controls too. Case and
        # disabled spin-box minimums must not make an unchanged setup look
        # like a different job (e.g. detect/Detect, unchecked freeze 0/1).
        if not values.get("ultralyticsFreezeEnabled"):
            values["ultralyticsFreeze"] = 0
        if not values.get("ultralyticsPatienceEnabled"):
            values["ultralyticsPatience"] = 0
        for key in ("ultralyticsDataYamlPath", "ultralyticsPtPath", "ultralyticsModelConfigPath",
                    "ultralyticsPretrainedModelPath", "ultralyticsRunsDirectory", "trainingEvaluatorRecommendedArgsYaml"):
            values[key] = path_identity(values.get(key))
        return values

    return (comparable({key: settings.get(key) for key in context["setup"]}) == comparable(context["setup"])
            and path_identity(self._training_eval_selected_config_path(context["result"])) == path_identity(context["selected_cfg"]))

def read_training_batch_calibration(self, output_path):
    output_path = self.normalize_path(output_path)
    if not output_path or not os.path.exists(output_path):
        return {}, ""
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if isinstance(data, dict):
            return data, output_path
    except Exception as e:
        logger.warning(f"Could not read batch calibration from {output_path}: {e}")
    return {}, ""

def apply_training_batch_calibration(self, result, calibration, source_path="", target_yaml=None, use_recommended=None, sync_controls=True):
    if not result or not isinstance(calibration, dict):
        return False
    if calibration.get("error"):
        return False
    raw_batch = int(calibration.get("batch") or calibration.get("raw_batch") or 0)
    if raw_batch <= 0:
        return False

    rec = result.get("recommendations", {})
    dataset_size = int(calibration.get("dataset_size") or result.get("image_count") or 0)
    model_factor = float((result.get("weight_info", {}) or {}).get("model_factor", 1.0) or 1.0)
    batch = self._training_eval_clamp_batch(raw_batch, dataset_size, model_factor)
    batch_cap = dataset_size or batch
    rec["batch"] = batch
    note = (
        f"Batch calibrated by Ultralytics auto-batch at imgsz={calibration.get('imgsz')} "
            f"using {calibration.get('fraction', 0.6):.0%} VRAM target."
    )
    if batch != raw_batch:
        note += f" Limited from {raw_batch} to the {batch} training images available."
    rec["batch_note"] = note
    calibration_record = dict(calibration)
    calibration_record["raw_batch"] = int(calibration.get("raw_batch") or raw_batch)
    calibration_record["batch_cap"] = int(calibration.get("batch_cap") or batch_cap)
    calibration_record["batch"] = batch
    result["batch_calibration"] = calibration_record
    result["batch_calibration_path"] = source_path
    if use_recommended is None:
        use_recommended = bool(getattr(self, "settings", {}).get("trainingEvaluatorUseOptimizedArgs", True))
    if use_recommended:
        loaded_yaml = self.normalize_path(target_yaml if target_yaml is not None else self._training_eval_selected_config_path(result))
        from darkfusion_training_runs import path_identity
        data_yaml = rec.get("data_yaml_path") or result.get("data_yaml_path", "")
        if loaded_yaml and path_identity(loaded_yaml) == path_identity(data_yaml):
            raise ValueError("The training configuration and dataset YAML must be different files.")
        if loaded_yaml and os.path.isfile(loaded_yaml):
            with open(loaded_yaml, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid training configuration in {loaded_yaml}")
            # Calibration owns batch only; do not overwrite a concurrently
            # edited/tuned configuration with cached recommendations.
            payload["batch"] = batch
            self._training_eval_write_text_atomic(loaded_yaml, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
        else:
            loaded_yaml = self._write_training_recommendation_yaml(result, target_yaml_path=loaded_yaml)
        result["training_recommendations_yaml"] = loaded_yaml
    if sync_controls:
        self.training_evaluator_last_result = result
        self.apply_training_evaluation_recommendations(result, apply_weights=False, apply_optimized_args=False)
    return True

def evaluate_training_setup(
    self,
    candidate_sizes=None,
    weights_path="",
    use_training_weights=True,
    auto_prepare_yaml=True,
    validation_mode=None,
    validation_source="",
    validation_percent=None,
    validation_source_percent=None,
    skip_image_scan=False,
    progress_callback=None,
):
    def report_progress(message):
        if callable(progress_callback):
            try:
                progress_callback(str(message))
            except Exception:
                pass

    self._training_eval_candidate_sizes(candidate_sizes, 640)  # Validate before generating files.
    self._training_eval_last_validation_prep = {}
    report_progress("Preparing dataset YAML and validation split...")
    data_yaml_path = self.normalize_path(getattr(self, "data_yaml_path", "") or "")
    data_yaml_error = ""
    validation_note = ""
    if auto_prepare_yaml:
        data_yaml_path = self.ensure_training_evaluator_data_yaml(
            force=False,
            prepare_validation=True,
            validation_mode=validation_mode,
            validation_source=validation_source,
            validation_percent=validation_percent,
            validation_source_percent=validation_source_percent,
        )
        prep = getattr(self, "_training_eval_last_validation_prep", {}) or {}
        validation_note = prep.get("validation_note") or "Auto YAML found or created obj.yaml for the current training setup."
    elif data_yaml_path and os.path.exists(data_yaml_path):
        validation_note = "Using the selected training YAML as the dataset source."
    if not data_yaml_path or not os.path.isfile(data_yaml_path):
        raise ValueError("Select an existing dataset YAML or enable Auto YAML to generate one for the loaded dataset.")

    report_progress("Resolving dataset paths and classes...")
    loaded_dataset_dir = self._training_eval_dataset_dir()
    if (
        data_yaml_path
        and loaded_dataset_dir
        and not self._training_eval_yaml_matches_dataset(data_yaml_path, loaded_dataset_dir)
    ):
        raise ValueError(
            "The selected Data YAML belongs to a different dataset. "
                "Run Generate Files + Parameters again with Auto YAML enabled for the currently loaded dataset."
        )
    yaml_data, yaml_dir = self._training_health_parse_data_yaml(data_yaml_path)
    dataset_dir = loaded_dataset_dir or self._training_health_dataset_dir(data_yaml_path, yaml_data, yaml_dir)
    if not dataset_dir:
        raise ValueError("Select obj.yaml or open a dataset folder before evaluating training setup.")
    self._training_eval_prepare_cache_context(dataset_dir)

    current_size = self._parse_training_imgsz_value(getattr(self, "imgsz_input", None).text() if getattr(self, "imgsz_input", None) is not None else "640")
    weights_path = self._effective_eval_weights_path(use_training_weights, weights_path)
    report_progress(
        f"Loading checkpoint {os.path.basename(weights_path) if weights_path else 'metadata'}..."
    )
    weight_info = self.inspect_training_eval_weights(weights_path)
    strides = weight_info.get("strides") or [8, 16, 32]
    min_stride = max(1, min(strides))
    task_text = str(
        getattr(self, "task_combobox", None).currentText()
        if getattr(self, "task_combobox", None) is not None
        else ""
    ).strip().lower()
    model_task = str(weight_info.get("task", "") or "").strip().lower()

    # Ultralytics downsamples segmentation targets by mask_ratio while
    # training. A target can therefore be large enough for the detection
    # head but still have too few mask cells to learn a useful outline.
    mask_ratio = 4
    extra_train_args = str(getattr(self, "ultralytics_extra_train_args", "") or "")
    mask_ratio_match = re.search(r"(?:^|\s)mask_ratio\s*=\s*(\d+)", extra_train_args)
    if mask_ratio_match:
        try:
            mask_ratio = max(1, int(mask_ratio_match.group(1)))
        except (TypeError, ValueError):
            mask_ratio = 4

    scanner = getattr(self, "scan_annotations", None)
    if not isinstance(scanner, ScanAnnotations):
        scanner = ScanAnnotations(self)
    scanner.base_directory = dataset_dir
    yaml_names = self._training_eval_yaml_names(yaml_data)
    if yaml_names:
        scanner.valid_classes = yaml_names
        classes_file = data_yaml_path
    else:
        scanner.valid_classes, classes_file = scanner._load_classes(dataset_dir)
    try:
        kpt_shape = yaml_data.get("kpt_shape") if isinstance(yaml_data, dict) else None
        scanner.expected_keypoint_count = int(kpt_shape[0]) if isinstance(kpt_shape, list) and kpt_shape else 0
    except Exception:
        scanner.expected_keypoint_count = 0
    if not scanner.expected_keypoint_count:
        scanner.expected_keypoint_count = scanner._load_expected_keypoint_count(dataset_dir)

    if not scanner.valid_classes:
        raise ValueError("No classes were found. Add classes.txt or select a classes file first.")

    report_progress("Building the training image list...")
    image_files = self._training_eval_yaml_image_files(data_yaml_path, yaml_data, "train") if data_yaml_path else []
    if data_yaml_path and not image_files:
        raise ValueError("The selected YAML has no readable training image paths. Check its train entry.")
    if not image_files:
        image_files = scanner._iter_image_files(dataset_dir)
    unique_images = {}
    for image_path in image_files:
        normalized = self.normalize_path(image_path)
        unique_images.setdefault(os.path.normcase(os.path.abspath(normalized)), normalized)
    image_files = list(unique_images.values())
    if not image_files:
        raise ValueError("No training images were found in the selected dataset.")

    analysis = TrainingSizeAnalysis()
    label_files = set()
    missing_labels = empty_labels = unreadable_images = unreadable_labels = invalid_labels = 0
    expected_keypoints = scanner.expected_keypoint_count or None
    polygon_preference = self._polygon_label_parse_preference() if hasattr(self, "_polygon_label_parse_preference") else None
    if (model_task or task_text) in {"segment", "obb"}:
        polygon_preference = model_task or task_text
    total_images_to_scan = len(image_files)
    
    if not skip_image_scan:
        report_progress(f"Scanning {total_images_to_scan:,} training images...")
        for image_index, image_path in enumerate(image_files, start=1):
            if image_index == 1 or image_index % 100 == 0 or image_index == total_images_to_scan:
                report_progress(f"Scanning training image {image_index:,} / {total_images_to_scan:,}...")
            # Resolve each full path independently; stems are not unique across
            # dataset subdirectories or train/validation splits.
            label_path = self._label_path_for_image(image_path)
            if label_path and os.path.isfile(label_path):
                label_files.add(os.path.normcase(os.path.abspath(label_path)))
            scan_record = self._training_eval_cached_image_scan(image_path, label_path, polygon_preference, expected_keypoints)
            if scan_record.get("image_error"):
                unreadable_images += 1
                continue
            missing_labels += int(bool(scan_record.get("missing_label")))
            empty_labels += int(bool(scan_record.get("empty_label")))
            unreadable_labels += int(bool(scan_record.get("label_read_error")))
            invalid_labels += int(scan_record.get("invalid_labels", 0))
            valid_annotations = []
            for annotation in scan_record.get("annotations", []):
                if not 0 <= annotation[0] < len(scanner.valid_classes):
                    invalid_labels += 1
                    continue
                valid_annotations.append(annotation)
            analysis.add_image(int(scan_record.get("width", 0)), int(scan_record.get("height", 0)), valid_annotations)
    else:
        # Skip image scan: use quick defaults
        report_progress(f"Quick mode: estimating parameters for {total_images_to_scan:,} images without scanning...")
    
    class_counts = analysis.class_counts
    annotation_counts = analysis.annotation_counts
    total_labels = len(analysis.native_sides)
    pose_labels = annotation_counts.get("pose", 0)
    visible_keypoints = analysis.visible_keypoints
    invalid_labels += analysis.invalid_labels
    if not analysis.images and not skip_image_scan:
        raise ValueError("None of the training images could be read. Check the dataset before generating parameters.")

    class_names = {}
    for class_id, count in class_counts.items():
        class_names[class_id] = scanner._class_name_for_index(class_id)

    annotation_task = ""
    if annotation_counts:
        annotation_task = {
            "seg": "segment",
            "bbox": "detect",
            "obb": "obb",
            "pose": "pose",
        }.get(annotation_counts.most_common(1)[0][0], "")
    effective_size_task = model_task or task_text or annotation_task or "detect"
    if annotation_task == "segment" and effective_size_task == "detect":
        # Prefer the labels for size analysis when a generic/mismatched
        # detect fallback was inferred from a filename.
        effective_size_task = "segment"

    report_progress("Comparing training image sizes...")
    size_analysis = analysis.recommend(candidate_sizes, current_size, strides, effective_size_task, mask_ratio)
    candidate_rows = size_analysis["candidate_rows"]
    recommended_row = size_analysis["recommended_row"]
    recommended_imgsz = int(recommended_row["imgsz"])
    class_risk_rows = size_analysis["class_risk_rows"]
    for row in class_risk_rows:
        row["class_name"] = class_names.get(row["class_id"], str(row["class_id"]))
    if unreadable_images or unreadable_labels or invalid_labels:
        size_analysis["warnings"].append(
            f"Skipped {unreadable_images} unreadable images, {unreadable_labels} unreadable label files, "
                f"and {invalid_labels} invalid annotations. Run Setup Check before training."
        )

    vram_gb = self._gpu_vram_gb()
    training_image_count = len(image_files) if skip_image_scan else len(analysis.images)
    batch_widget = getattr(self, "batch_input", None)
    requested_batch = (batch_widget.text() if batch_widget is not None else
                       getattr(self, "settings", {}).get("ultralyticsTrainBatch", -1 if torch.cuda.is_available() else 4))
    batch_settings = getattr(self, "settings", {})
    if batch_settings.get("trainingEvaluatorBatchOverrideEnabled", False):
        requested_batch = batch_settings.get("trainingEvaluatorBatchOverride", requested_batch)
    recommended_batch = self._training_eval_clamp_batch(requested_batch, training_image_count)

    imbalance_ratio = 1.0
    nonzero_counts = [count for count in class_counts.values() if count > 0]
    if len(nonzero_counts) >= 2 and min(nonzero_counts) > 0:
        imbalance_ratio = max(nonzero_counts) / min(nonzero_counts)

    # An epoch is a pass over training images. Object count does not
    # measure dataset size (one crowded image may contain many labels).
    recommended_epochs = 200 if training_image_count < 500 else 150 if training_image_count < 2000 else 100
    recommended_patience = max(25, round(recommended_epochs * 0.25))

    task_warning = ""
    if model_task and task_text and model_task != task_text:
        task_warning = f"Loaded eval weights are task '{model_task}', while the training panel is set to '{task_text}'."

    rect_widget = getattr(self, "rect_train_checkbox", None)
    rect_recommended = bool(rect_widget.isChecked()) if rect_widget is not None else False
    if not rect_recommended and float(recommended_row.get("avg_padding_waste", 0)) >= 0.25:
        size_analysis["warnings"].append("Square inputs leave substantial padding. Consider Rect after comparing its augmentation and validation results.")
    optimizer_rec = self.recommend_training_optimizer_args(
        total_labels,
        recommended_row,
        imbalance_ratio,
        weight_info,
        weights_path,
        task_text or model_task or "detect",
    )

    recommendations = {
        "imgsz": recommended_imgsz,
        "batch": recommended_batch,
        "epochs": recommended_epochs,
        "patience": recommended_patience,
        "amp": _training_precision(self),
        "rect": bool(rect_recommended),
        "task": task_text or model_task or "detect",
        "weights_path": weights_path,
        "data_yaml_path": data_yaml_path,
        "hyperparams": optimizer_rec.get("hyperparams", {}),
        "train_args": optimizer_rec.get("train_args", []),
        "train_args_text": optimizer_rec.get("train_args_text", ""),
        "optimizer_reason": optimizer_rec.get("optimizer_reason", ""),
        "freeze": int(optimizer_rec.get("freeze", 0) or 0),
        "batch_note": ("Ultralytics AutoBatch profiles memory once when training starts."
                       if recommended_batch == -1 else
                       f"Ultralytics AutoBatch targets {recommended_batch:.0%} of available GPU memory when training starts."
                       if recommended_batch < 1 else
                       "Fixed batch from your training setup. Select Auto to let Ultralytics profile memory, or use Calibrate Batch."),
        "size_note": size_analysis["size_note"],
        "parameter_note": "Epochs and patience are starting budgets based on training images. Use validation curves and early stopping; extra epochs cannot fix missing image detail or class coverage.",
    }

    validation_prep = getattr(self, "_training_eval_last_validation_prep", {}) or {}
    split_class_report = (
        validation_prep.get("split_class_report")
        if isinstance(validation_prep, dict)
        else {}
    ) or self._training_eval_yaml_split_report(data_yaml_path, yaml_data)

    result = {
        "dataset_dir": dataset_dir,
        "data_yaml_path": data_yaml_path,
        "data_yaml_error": data_yaml_error,
        "validation_note": validation_note,
        "validation_prep": validation_prep,
        "split_class_report": split_class_report,
        "classes_file": classes_file,
        "image_count": len(image_files),
        "training_image_count": training_image_count,
        "scanned_image_count": len(analysis.images),
        "analysis_scope": "training split only",
        "native_summary": size_analysis["native_summary"],
        "size_warnings": size_analysis["warnings"],
        "unreadable_images": unreadable_images,
        "unreadable_labels": unreadable_labels,
        "invalid_labels": invalid_labels,
        "label_file_count": len(label_files),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "total_labels": total_labels,
        "class_counts": dict(sorted((class_names.get(cid, str(cid)), count) for cid, count in class_counts.items())),
        "annotation_counts": dict(annotation_counts),
        "candidate_rows": candidate_rows,
        "class_risk_rows": class_risk_rows[:12],
        "recommended_row": recommended_row,
        "recommendations": recommendations,
        "weight_info": weight_info,
        "stride_min": min_stride,
        "effective_size_task": effective_size_task,
        "mask_ratio": mask_ratio,
        "vram_gb": round(vram_gb, 2),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "pose_labels": pose_labels,
        "visible_keypoints": visible_keypoints,
        "task_warning": task_warning,
        "skip_image_scan": skip_image_scan,
    }
    report_progress("Finalizing training recommendations...")
    # Try to update currently loaded YAML if available, otherwise create new one
    loaded_yaml = self.normalize_path(getattr(self, "recommended_args_path", "") or "")
    result["training_recommendations_yaml"] = self._write_training_recommendation_yaml(result, target_yaml_path=loaded_yaml)
    if skip_image_scan:
        report_progress("Quick mode: parameters generated without full image scan.")
    else:
        report_progress("Dataset scan complete.")
    return result

def format_training_evaluation_report(self, result):
    if not result:
        return "No evaluation has been run yet."

    rec = result.get("recommendations", {})
    weight_info = result.get("weight_info", {})
    lines = [
        "Training Evaluator",
        "",
        f"Dataset: {result.get('dataset_dir', '')}",
        f"Training YAML: {result.get('data_yaml_path') or 'not selected'}",
        f"Images: {result.get('image_count', 0)}   Label files: {result.get('label_file_count', 0)}   Labels: {result.get('total_labels', 0)}",
        f"Missing labels: {result.get('missing_labels', 0)}   Empty labels: {result.get('empty_labels', 0)}",
        "",
        f"Eval weights: {weight_info.get('name', 'No weights loaded')}",
        f"Task: {weight_info.get('task') or 'unknown'}   Stride: {', '.join(str(s) for s in weight_info.get('strides', []))}",
    ]

    if weight_info.get("error"):
        lines.append(f"Weight note: {weight_info['error']}")
    if result.get("skip_image_scan"):
        lines.append("NOTE: Quick mode was used—image scan was skipped. For detailed analysis, uncheck 'Skip Image Scan' and regenerate.")
    if result.get("data_yaml_error"):
        lines.append(f"YAML note: {result['data_yaml_error']}")
    if result.get("validation_note"):
        lines.append(f"Validation note: {result['validation_note']}")
    validation_prep = result.get("validation_prep", {}) if isinstance(result.get("validation_prep", {}), dict) else {}
    if validation_prep:
        lines.append(
            f"Split files: train={validation_prep.get('train_count', 0)} "
                f"valid={validation_prep.get('valid_count', 0)}"
        )
    split_report = result.get("split_class_report", {}) if isinstance(result.get("split_class_report", {}), dict) else {}
    if split_report:
        lines.append(
            f"Split coverage: train images={split_report.get('train_images', 0)} "
                f"valid images={split_report.get('valid_images', 0)} "
                f"overlap={split_report.get('overlap_images', 0)}"
        )
        if split_report.get("same_split"):
            lines.append("Split warning: train and validation contain the same images.")
        missing_valid = split_report.get("missing_valid", []) or []
        missing_train = split_report.get("missing_train", []) or []
        if missing_valid:
            preview = ", ".join(
                f"{row.get('class_name')}({row.get('train_labels', 0)})"
                for row in missing_valid[:10]
            )
            more = f" +{len(missing_valid) - 10} more" if len(missing_valid) > 10 else ""
            lines.append(f"Classes missing from valid: {preview}{more}")
        if missing_train:
            preview = ", ".join(
                f"{row.get('class_name')}({row.get('valid_labels', 0)})"
                for row in missing_train[:10]
            )
            more = f" +{len(missing_train) - 10} more" if len(missing_train) > 10 else ""
            lines.append(f"Classes missing from train: {preview}{more}")
    if result.get("task_warning"):
        lines.append(f"Task warning: {result['task_warning']}")
    if bool(getattr(self, "ultralytics_distill_enabled", False)):
        distill_notes = self._training_eval_distillation_sanity_notes(
            rec.get("weights_path", "") or getattr(self, "model_config_path", "") or getattr(self, "pt_path", ""),
            getattr(self, "ultralytics_distill_model_path", ""),
            getattr(self, "ultralytics_distill_weight", 6.0),
            rec.get("task", ""),
        )
        if distill_notes:
            lines.append("Distillation notes:")
            lines.extend(f"- {note.get('text', '')}" for note in distill_notes if note.get("text"))

    segment_size_analysis = result.get("effective_size_task") == "segment"
    lines.extend([
        "",
        "Candidate sizes (detail retention is a heuristic, not predicted accuracy)",
        (
            "imgsz | detail% | <4px | <stride | <2*stride | p10 input | p10 mask | mask<4 | mask<8 | padding"
            if segment_size_analysis
            else "imgsz | detail% | <4px | <stride | <2*stride | p10 min side | risky classes | padding waste"
        ),
    ])

    for row in result.get("candidate_rows", []):
        marker = "*" if int(row.get("imgsz", 0)) == int(rec.get("imgsz", 0)) else " "
        if segment_size_analysis:
            lines.append(
                f"{marker}{row['imgsz']:>4} | {row['score']:>5.1f} | "
                    f"{row['tiny4_pct']:>5.1f}% | {row['under_stride_pct']:>7.1f}% | "
                    f"{row['under_2stride_pct']:>9.1f}% | {row.get('p10_min_side', 0):>7.1f}px | "
                    f"{row.get('mask_p10_min_side', 0):>6.1f} | "
                    f"{row.get('mask_under_4_pct', 0):>6.1f}% | "
                    f"{row.get('mask_under_8_pct', 0):>6.1f}% | "
                    f"{row['avg_padding_waste'] * 100:>5.1f}%"
            )
        else:
            lines.append(
                f"{marker}{row['imgsz']:>4} | {row['score']:>5.1f} | "
                    f"{row['tiny4_pct']:>5.1f}% | {row['under_stride_pct']:>7.1f}% | "
                    f"{row['under_2stride_pct']:>9.1f}% | {row.get('p10_min_side', 0):>7.1f}px | "
                    f"{row.get('risky_class_count', 0):>5} | "
                    f"{row['avg_padding_waste'] * 100:>5.1f}%"
            )

    lines.extend([
        "",
        "Recommended training setup",
        f"imgsz={rec.get('imgsz')}   batch={rec.get('batch')}   epochs={rec.get('epochs')}   patience={rec.get('patience')}",
        f"amp={rec.get('amp', True)}   rect={rec.get('rect')}   freeze={rec.get('freeze', 0)}   task={rec.get('task')}",
    ])

    if rec.get("weights_path"):
        lines.append(f"weights={rec.get('weights_path')}")
    if rec.get("train_args_text"):
        lines.extend([
            "",
            "Starting train args",
            str(rec.get("train_args_text")),
        ])
    if rec.get("optimizer_reason"):
        lines.append(f"Optimizer note: {rec.get('optimizer_reason')}")
    if rec.get("size_note"):
        lines.append(f"Size note: {rec.get('size_note')}")
    if rec.get("parameter_note"):
        lines.append(f"Parameter note: {rec['parameter_note']}")
    if result.get("native_summary"):
        native = result["native_summary"]
        lines.append(f"Analysis: {result.get('analysis_scope', 'training images')}; median source long side {native.get('median_long_side', 0):.0f}px; p10 source object short side {native.get('p10_object_short_side', 0):.1f}px.")
        lines.append(f"At selected size, {result.get('recommended_row', {}).get('upscaled_images_pct', 0):.1f}% of images are upscaled.")
    lines.extend(f"Size note: {warning}" for warning in result.get("size_warnings", []))
    if rec.get("batch_note"):
        lines.append(f"Batch note: {rec.get('batch_note')}")
    if result.get("batch_calibration_path"):
        lines.append(f"Batch calibration: {result.get('batch_calibration_path')}")
    if result.get("tune_best_path"):
        lines.append(f"Tuned hyperparameters: {result.get('tune_best_path')}")
    if result.get("tune_target_yaml"):
        lines.append(f"Tuning updated YAML: {result.get('tune_target_yaml')}")
    if result.get("training_recommendations_yaml"):
        lines.append(f"Recommendation YAML: {result.get('training_recommendations_yaml')}")

    if result.get("class_risk_rows"):
        lines.extend([
            "",
            "Riskiest classes at recommended size",
            "class | labels | <stride | <2*stride | median min side",
        ])
        for row in result["class_risk_rows"][:8]:
            lines.append(
                f"{row['class_name']} | {row['labels']} | "
                    f"{row['under_stride_pct']:.1f}% | {row['under_2stride_pct']:.1f}% | "
                    f"{row['median_min_side']:.1f}px"
            )

    if result.get("pose_labels", 0):
        rec_row = result.get("recommended_row", {})
        lines.extend([
            "",
            f"Pose labels: {result.get('pose_labels', 0)}   Visible keypoints: {result.get('visible_keypoints', 0)}",
            f"Nearest visible-keypoint spacing under stride at recommended size: {rec_row.get('pose_spread_under_stride_pct', 0):.1f}% ({rec_row.get('pose_spread_under_stride', 0)} labels)",
        ])

    if result.get("class_counts"):
        top_classes = sorted(result["class_counts"].items(), key=lambda item: item[1], reverse=True)[:8]
        lines.extend(["", "Top classes"])
        lines.extend(f"{name}: {count}" for name, count in top_classes)

    if result.get("imbalance_ratio", 1.0) >= 8:
        lines.append("")
        lines.append(f"Class balance note: largest/smallest class ratio is about {result.get('imbalance_ratio')}x.")

    if result.get("vram_gb", 0) > 0:
        lines.append(f"VRAM estimate: {result.get('vram_gb')} GB detected.")
    else:
        lines.append("VRAM estimate: CUDA GPU not detected; batch suggestion is conservative.")

    return "\n".join(lines)

def apply_training_evaluation_recommendations(self, result, apply_weights=False, apply_optimized_args=True):
    if not result:
        return

    rec = result.get("recommendations", {})
    imgsz = int(rec.get("imgsz", self._parse_training_imgsz_value()))
    batch = parse_training_batch(rec.get("batch", 4))
    epochs = int(rec.get("epochs", 100))
    patience = int(rec.get("patience", 40))

    if getattr(self, "imgsz_input", None) is not None:
        self.imgsz_input.setText(str(imgsz))
    if getattr(self, "batch_input", None) is not None:
        self.batch_input.setText(str(batch))
    if getattr(self, "epochs_input", None) is not None:
        self.epochs_input.setText(str(epochs))

    if getattr(self, "patience_checkbox", None) is not None:
        self.patience_checkbox.setChecked(True)
    if getattr(self, "patience_input", None) is not None:
        self.patience_input.setValue(patience)
    if getattr(self, "amp_true", None) is not None:
        self.amp_true.setChecked(True)
    if getattr(self, "rect_train_checkbox", None) is not None:
        self.rect_train_checkbox.setChecked(bool(rec.get("rect", False)))
    freeze_value = int(rec.get("freeze", 0) or 0)
    if getattr(self, "freeze_checkbox", None) is not None:
        self.freeze_checkbox.setChecked(freeze_value > 0)
    if freeze_value > 0 and getattr(self, "freeze_input", None) is not None:
        self.freeze_input.setValue(freeze_value)

    task = str(rec.get("task", "") or "").strip()
    if task and getattr(self, "task_combobox", None) is not None:
        index = self.task_combobox.findText(task, Qt.MatchFixedString)
        if index >= 0:
            self.task_combobox.setCurrentIndex(index)

    weights_path = self.normalize_path(rec.get("weights_path", ""))
    if apply_weights and weights_path and os.path.exists(weights_path) and weights_path.lower().endswith(".pt"):
        self.pt_path = weights_path
        self.model_config_path = None
        self.pretrained_model_path = None
        if getattr(self, "pt_label", None) is not None:
            self.pt_label.setText(f"Model: {weights_path}")

    data_yaml_path = self.normalize_path(rec.get("data_yaml_path", "") or result.get("data_yaml_path", ""))
    if data_yaml_path and os.path.exists(data_yaml_path):
        self.data_yaml_path = data_yaml_path
        if getattr(self, "yaml_label", None) is not None:
            self.yaml_label.setText(f"Data YAML: {data_yaml_path}")

    optimized_args_text = str(rec.get("train_args_text", "") or "").strip()
    if apply_optimized_args and optimized_args_text:
        self.ultralytics_extra_train_args = optimized_args_text

    self.training_preview_size = (imgsz, imgsz)
    if isinstance(getattr(self, "settings", None), dict):
        self.settings["trainingPreviewImgSize"] = str(imgsz)
        if weights_path:
            self.settings["trainingEvaluatorWeightsPath"] = weights_path
        if data_yaml_path:
            self.settings["ultralyticsDataYamlPath"] = data_yaml_path
        if apply_optimized_args and optimized_args_text:
            self.settings["ultralyticsExtraTrainArgs"] = optimized_args_text
    self.queue_settings_save()

def browse_training_eval_weights(self, line_edit, use_training_weights_checkbox=None):
    file_name = self.open_file_dialog("Select Evaluation Weights", "Model Files (*.pt *.yaml);;All Files (*)")
    if file_name:
        file_name = self.normalize_path(file_name)
        line_edit.setText(file_name)
        if use_training_weights_checkbox is not None:
            use_training_weights_checkbox.setChecked(False)
        self.settings["trainingEvaluatorWeightsPath"] = file_name
        self.settings["trainingEvaluatorUseTrainingWeights"] = False
        self.queue_settings_save()
        return file_name
    return ""

def build_training_evaluator_resume_command(self, run_dir=None):
    info = self._training_eval_resume_info(run_dir or self._training_eval_latest_saved_run(training_only=True))
    if not info.get("resumable"):
        raise ValueError(info.get("note") or "No resumable training run was found.")
    last_pt = self.normalize_path(info.get("last_pt", ""))
    if not last_pt or not os.path.exists(last_pt):
        raise ValueError("Cannot resume because weights/last.pt is missing.")

    args = info.get("args", {}) or {}
    task = str(args.get("task") or "detect").strip().lower()
    if task not in {"detect", "segment", "classify", "pose", "obb"}:
        task = "detect"
    cmd = [
        *self._ultralytics_train_command_prefix(),
        task,
        "train",
        f"model={last_pt}",
        "resume=True",
    ]
    self._append_ultralytics_memory_guard_args(cmd)
    return cmd

@gpu_work("Training / evaluation", keep=(), external=True, exclusive=True)
def launch_training_evaluator_logged_command(self, command, title="Training Evaluator", run_dir=None):
    if not command:
        raise ValueError("No command to launch.")

    run_dir = self.normalize_path(run_dir or self._training_eval_run_dir_from_command(command))
    if not run_dir:
        run_dir = self.normalize_path(os.path.join(self._training_eval_project_dir(), f"evaluator_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    os.makedirs(run_dir, exist_ok=True)

    runner_path = os.path.join(APP_DIR, "darkfusion_command_runner.py")
    if not os.path.exists(runner_path):
        return self.launch_training_evaluator_command(command, title=title, keep_open=False), run_dir, ""

    log_path = self._training_eval_log_path(run_dir)
    runner_command = [
        sys.executable,
        runner_path,
        f"--log={log_path}",
        f"--cwd={APP_DIR}",
        "--max-commit-gb=48",
        "--min-system-available-gb=4",
        "--",
        *command,
    ]
    logger.info(f"{title} command: {self._format_command_display(command)}")
    logger.info(f"{title} log: {log_path}")

    kwargs = {
        "cwd": APP_DIR,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(runner_command, **kwargs)
    return process, run_dir, log_path

def _training_health_run_ultralytics_checks(self, timeout=45):
    command = [
        sys.executable,
        "-c",
        "from ultralytics.utils.checks import collect_system_info; collect_system_info()",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=APP_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(5, int(timeout)),
        )
        output = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part and part.strip()
        ).strip()
        return completed.returncode == 0, output or "Ultralytics checks completed with no console output."
    except subprocess.TimeoutExpired as e:
        output = "\n".join(
            str(part).strip()
            for part in (getattr(e, "stdout", ""), getattr(e, "stderr", ""))
            if part and str(part).strip()
        ).strip()
        return False, f"Ultralytics checks timed out after {timeout} seconds.\n{output}".strip()
    except Exception as e:
        return False, f"Could not run Ultralytics checks with {sys.executable}: {e}"

def _training_health_status_line(self, status, label, detail=""):
    icon = {
        "ok": "OK",
        "warn": "WARN",
        "fail": "FAIL",
    }.get(str(status).lower(), "INFO")
    detail = str(detail or "").strip()
    return f"[{icon}] {label}" + (f": {detail}" if detail else "")

def _training_health_writable_dir(self, directory):
    """Report likely write access without creating or deleting probe files."""
    directory = self.normalize_path(directory or "")
    if not directory or not os.path.isdir(directory):
        return False, "folder not found"
    try:
        writable = bool(os.access(directory, os.W_OK))
        return writable, directory if writable else "folder is not writable"
    except Exception as e:
        return False, str(e)

def _training_health_parse_data_yaml(self, data_yaml_path):
    data_yaml_path = self.normalize_path(data_yaml_path or "")
    if not data_yaml_path or not os.path.exists(data_yaml_path):
        return {}, ""
    try:
        with open(data_yaml_path, "r", encoding="utf-8", errors="ignore") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}, os.path.dirname(data_yaml_path)
    except Exception:
        return {}, os.path.dirname(data_yaml_path)

def _training_health_dataset_dir(self, data_yaml_path="", yaml_data=None, yaml_dir=""):
    data_yaml_path = self.normalize_path(data_yaml_path or "")
    yaml_data = yaml_data if isinstance(yaml_data, dict) else {}
    yaml_dir = self.normalize_path(yaml_dir or (os.path.dirname(data_yaml_path) if data_yaml_path else ""))

    candidates = []
    for value in (
        yaml_data.get("path"),
        getattr(self, "image_directory", None),
        getattr(self, "output_path", None),
        yaml_dir,
    ):
        value = self.normalize_path(value or "")
        if not value:
            continue
        if not os.path.isabs(value) and yaml_dir:
            value = self.normalize_path(os.path.join(yaml_dir, value))
        if os.path.isdir(value):
            candidates.append(value)

    return candidates[0] if candidates else (yaml_dir if yaml_dir and os.path.isdir(yaml_dir) else "")

def _training_health_find_classes_file(self, dataset_dir="", data_yaml_path="", yaml_data=None):
    yaml_data = yaml_data if isinstance(yaml_data, dict) else {}
    names = yaml_data.get("names")
    if isinstance(names, (list, tuple, dict)) and names:
        return "", f"names found in YAML ({len(names)} classes)"

    candidates = []
    for directory in (
        dataset_dir,
        os.path.dirname(self.normalize_path(data_yaml_path or "")),
        getattr(self, "image_directory", None),
        getattr(self, "output_path", None),
        APP_DIR,
    ):
        directory = self.normalize_path(directory or "")
        if directory:
            candidates.append(resolve_dataset_classes_path(directory, migrate=False))

    for path in candidates:
        if path and os.path.exists(path):
            return self.normalize_path(path), "classes.txt found"
    return "", "classes.txt missing and YAML names not found"

def _training_health_find_points_file(self, dataset_dir="", model_path="", data_yaml_path=""):
    candidates = []
    for directory in (
        dataset_dir,
        os.path.dirname(self.normalize_path(data_yaml_path or "")),
        os.path.dirname(self.normalize_path(model_path or "")),
        getattr(self, "image_directory", None),
        getattr(self, "output_path", None),
        APP_DIR,
    ):
        directory = self.normalize_path(directory or "")
        if directory:
            candidates.append(resolve_dataset_pose_schema_path(directory, migrate=False))

    seen = set()
    for path in candidates:
        key = os.path.normcase(self.normalize_path(path))
        if key in seen:
            continue
        seen.add(key)
        if os.path.exists(path):
            return self.normalize_path(path)
    return ""

def _training_health_engine_paths(self, model_path="", data_yaml_path="", runs_dir=""):
    roots = []
    for path in (
        model_path,
        getattr(self, "weights_file_path", None),
        getattr(self, "loaded_model_path", None),
    ):
        path = self.normalize_path(path or "")
        if path.lower().endswith(".engine") and os.path.exists(path):
            roots.append(path)
        elif path and os.path.isdir(path):
            roots.append(path)
        elif path:
            parent = os.path.dirname(path)
            if parent:
                roots.append(parent)

    for directory in (
        runs_dir,
        os.path.dirname(self.normalize_path(data_yaml_path or "")),
        os.path.join(APP_DIR, "weights"),
        APP_DIR,
    ):
        directory = self.normalize_path(directory or "")
        if directory:
            roots.append(directory)

    engines = []
    seen_roots = set()
    for root in roots:
        root = self.normalize_path(root)
        if not root:
            continue
        if os.path.isfile(root) and root.lower().endswith(".engine"):
            engines.append(root)
            continue
        if not os.path.isdir(root):
            continue
        root_key = os.path.normcase(os.path.abspath(root))
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        try:
            engines.extend(self.normalize_path(path) for path in glob.glob(os.path.join(root, "*.engine")))
        except Exception:
            continue

    seen = set()
    unique = []
    for path in engines:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen and os.path.exists(path):
            seen.add(key)
            unique.append(path)
    return unique

def build_training_health_report(self, data_yaml_path="", model_path="", pretrained_path="", runs_dir="", task="detect"):
    data_yaml_path = self.normalize_path(data_yaml_path or getattr(self, "data_yaml_path", "") or "")
    model_path = self.normalize_path(model_path or getattr(self, "pt_path", "") or getattr(self, "model_config_path", "") or "")
    pretrained_path = self.normalize_path(pretrained_path or getattr(self, "pretrained_model_path", "") or "")
    runs_dir = self.normalize_path(runs_dir or getattr(self, "runs_directory", "") or "")
    task = str(task or "detect").strip().lower()

    lines = [
        "DarkFusion Training Setup Check",
        f"Python: {sys.executable}",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Core Checks",
    ]

    cuda_ok = bool(torch.cuda.is_available())
    lines.append(self._training_health_status_line("ok" if cuda_ok else "warn", "CUDA available", str(cuda_ok)))
    if cuda_ok:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_gb = float(getattr(props, "total_memory", 0) or 0) / (1024 ** 3)
            lines.append(self._training_health_status_line("ok", "GPU", f"{gpu_name} ({total_gb:.1f} GB VRAM)"))
        except Exception as e:
            lines.append(self._training_health_status_line("warn", "GPU name/VRAM", str(e)))
    else:
        lines.append(self._training_health_status_line("warn", "GPU", "training will run on CPU or fail for CUDA-only paths"))

    torch_cuda_version = getattr(torch.version, "cuda", None) or "not built with CUDA"
    lines.append(self._training_health_status_line("ok" if torch_cuda_version != "not built with CUDA" else "warn", "PyTorch CUDA version", torch_cuda_version))

    try:
        import ultralytics
        lines.append(self._training_health_status_line("ok", "Ultralytics import", getattr(ultralytics, "__version__", "unknown")))
    except Exception as e:
        lines.append(self._training_health_status_line("fail", "Ultralytics import", str(e)))

    yaml_data, yaml_dir = self._training_health_parse_data_yaml(data_yaml_path)
    dataset_dir = self._training_health_dataset_dir(data_yaml_path, yaml_data, yaml_dir)

    if data_yaml_path:
        lines.append(self._training_health_status_line("ok" if os.path.exists(data_yaml_path) else "fail", "Data YAML", data_yaml_path))
    else:
        lines.append(self._training_health_status_line("warn", "Data YAML", "not selected"))

    if model_path:
        lines.append(self._training_health_status_line("ok" if os.path.exists(model_path) else "fail", "Model", model_path))
    else:
        lines.append(self._training_health_status_line("warn", "Model", "not selected"))

    if model_path.lower().endswith((".yaml", ".yml")) and pretrained_path:
        lines.append(self._training_health_status_line("ok" if os.path.exists(pretrained_path) else "fail", "Pretrained weights", pretrained_path))

    if bool(getattr(self, "ultralytics_distill_enabled", False)):
        distill_path = self.normalize_path(getattr(self, "ultralytics_distill_model_path", "") or "")
        if not self.ultralytics_version_at_least("8.4.77"):
            lines.append(self._training_health_status_line("fail", "Knowledge Distillation", "requires Ultralytics v8.4.77+"))
        elif distill_path and (os.path.exists(distill_path) or self._looks_like_ultralytics_model_reference(distill_path)):
            lines.append(self._training_health_status_line("ok", "Knowledge Distillation teacher", distill_path))
        elif distill_path:
            lines.append(self._training_health_status_line("fail", "Knowledge Distillation teacher", f"missing: {distill_path}"))
        else:
            lines.append(self._training_health_status_line("fail", "Knowledge Distillation teacher", "enabled but no teacher model selected"))
        if task != "detect":
            lines.append(self._training_health_status_line("warn", "Knowledge Distillation task", "Ultralytics benchmarks detect first; this task is less validated"))
    else:
        lines.append(self._training_health_status_line("ok", "Knowledge Distillation", "off"))

    if dataset_dir:
        writable, detail = self._training_health_writable_dir(dataset_dir)
        lines.append(self._training_health_status_line("ok" if writable else "fail", "Writable dataset folder", detail))
    else:
        lines.append(self._training_health_status_line("warn", "Writable dataset folder", "dataset folder could not be resolved"))

    if runs_dir:
        writable, detail = self._training_health_writable_dir(runs_dir)
        lines.append(self._training_health_status_line("ok" if writable else "warn", "Writable runs folder", detail))
    else:
        lines.append(self._training_health_status_line("warn", "Writable runs folder", "not selected; Ultralytics will choose default project path"))

    classes_path, classes_detail = self._training_health_find_classes_file(dataset_dir, data_yaml_path, yaml_data)
    lines.append(self._training_health_status_line("ok" if classes_path or "names found" in classes_detail else "fail", "Classes", classes_path or classes_detail))

    points_path = self._training_health_find_points_file(dataset_dir, model_path, data_yaml_path)
    points_required = task in {"pose", "keypoint", "keypoints"}
    points_status = "ok" if points_path else ("fail" if points_required else "warn")
    points_detail = points_path or ("required for pose training" if points_required else "not found; only needed for pose/keypoints")
    lines.append(self._training_health_status_line(points_status, "points.json", points_detail))

    engines = self._training_health_engine_paths(model_path, data_yaml_path, runs_dir)
    if engines:
        preview = ", ".join(os.path.basename(path) for path in engines[:5])
        if len(engines) > 5:
            preview += f", +{len(engines) - 5} more"
        lines.append(self._training_health_status_line("ok", "TensorRT engines", preview))
    else:
        lines.append(self._training_health_status_line("warn", "TensorRT engines", "none found near selected model, runs folder, or app weights"))

    lines.extend(["", "Ultralytics yolo checks"])
    yolo_ok, yolo_output = self._training_health_run_ultralytics_checks()
    lines.append(self._training_health_status_line("ok" if yolo_ok else "fail", "Ultralytics checks command", "completed" if yolo_ok else "failed"))
    lines.append("")
    lines.append(yolo_output)
    return "\n".join(lines)


def _training_eval_dataset_dir(self):
    try:
        current_dir = self.current_dataset_directory()
        if current_dir and os.path.isdir(current_dir):
            return self.normalize_path(current_dir)
    except Exception:
        pass

    for candidate in (
        getattr(self, "image_directory", ""),
        getattr(self, "output_path", ""),
    ):
        if candidate and os.path.isdir(candidate):
            return self.normalize_path(candidate)
    return ""


def _set_training_data_yaml_path(self, yaml_path):
    yaml_path = self.normalize_path(yaml_path or "")
    if not yaml_path:
        return ""
    self.data_yaml_path = yaml_path
    if getattr(self, "yaml_label", None) is not None:
        self.yaml_label.setText(f"Data YAML: {yaml_path}")
    if isinstance(getattr(self, "settings", None), dict):
        self.settings["ultralyticsDataYamlPath"] = yaml_path
    return yaml_path


def _clear_training_data_yaml_path(self):
    self.data_yaml_path = None
    if getattr(self, "yaml_label", None) is not None:
        self.yaml_label.setText("Data YAML: Not selected")
    if isinstance(getattr(self, "settings", None), dict):
        self.settings["ultralyticsDataYamlPath"] = ""


@staticmethod
def _training_eval_paths_overlap(first_path, second_path):
    """Return True when either dataset path contains the other."""
    try:
        first_value = str(first_path or "").strip()
        second_value = str(second_path or "").strip()
        if not first_value or not second_value:
            return False
        first = os.path.normcase(os.path.abspath(first_value))
        second = os.path.normcase(os.path.abspath(second_value))
        common = os.path.commonpath((first, second))
        return common == first or common == second
    except (OSError, ValueError, TypeError):
        return False


def _training_eval_yaml_dataset_roots(self, data_yaml_path, yaml_data=None, yaml_dir=""):
    """Resolve paths owned by a YAML without consulting the currently open dataset."""
    data_yaml_path = self.normalize_path(data_yaml_path or "")
    yaml_data = yaml_data if isinstance(yaml_data, dict) else None
    if yaml_data is None:
        yaml_data, parsed_dir = self._training_health_parse_data_yaml(data_yaml_path)
        yaml_dir = yaml_dir or parsed_dir
    yaml_dir = self.normalize_path(yaml_dir or (os.path.dirname(data_yaml_path) if data_yaml_path else ""))
    yaml_data = yaml_data if isinstance(yaml_data, dict) else {}
    roots = []

    def add(path):
        path = self.normalize_path(path or "")
        if path and path not in roots:
            roots.append(path)

    configured_root = str(yaml_data.get("path") or "").strip()
    if configured_root:
        if not os.path.isabs(configured_root) and yaml_dir:
            configured_root = os.path.join(yaml_dir, configured_root)
        configured_root = self.normalize_path(configured_root)
        add(configured_root)
    base_dir = configured_root or yaml_dir

    def add_dataset_entry(entry):
        if isinstance(entry, (list, tuple)):
            for item in entry:
                add_dataset_entry(item)
            return
        if not isinstance(entry, str) or not entry.strip():
            return
        entry_path = entry.strip()
        if not os.path.isabs(entry_path) and base_dir:
            entry_path = os.path.join(base_dir, entry_path)
        entry_path = self.normalize_path(entry_path)
        if entry_path.lower().endswith(".txt"):
            if os.path.isfile(entry_path):
                try:
                    with open(entry_path, "r", encoding="utf-8", errors="ignore") as file:
                        for index, line in enumerate(file):
                            if index >= 100:
                                break
                            image_path = line.strip()
                            if not image_path:
                                continue
                            if not os.path.isabs(image_path):
                                image_path = os.path.join(os.path.dirname(entry_path), image_path)
                            add(os.path.dirname(image_path))
                except OSError:
                    pass
            return
        if os.path.splitext(entry_path)[1].lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
        }:
            add(os.path.dirname(entry_path))
        else:
            add(entry_path)

    for split_name in ("train", "val", "valid", "test"):
        add_dataset_entry(yaml_data.get(split_name))
    if not roots and yaml_dir:
        add(yaml_dir)
    return roots


def _training_eval_yaml_matches_dataset(self, data_yaml_path, dataset_dir):
    data_yaml_path = self.normalize_path(data_yaml_path or "")
    dataset_dir = self.normalize_path(dataset_dir or "")
    if not data_yaml_path or not dataset_dir or not os.path.exists(data_yaml_path):
        return False
    yaml_data, yaml_dir = self._training_health_parse_data_yaml(data_yaml_path)
    if "train" not in yaml_data:
        return False  # A training-arguments/model YAML is not a dataset YAML.
    return any(
        self._training_eval_paths_overlap(root, dataset_dir)
        for root in self._training_eval_yaml_dataset_roots(data_yaml_path, yaml_data, yaml_dir)
    )


def _training_eval_managed_output_dir(self, dataset_dir, create=False):
    dataset_dir = self.normalize_path(dataset_dir or "")
    if not dataset_dir:
        return ""
    output_dir = self.normalize_path(dataset_metadata_directory(dataset_dir, create=create))
    if create:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _training_eval_is_managed_yaml(self, data_yaml_path, dataset_dir):
    data_yaml_path = self.normalize_path(data_yaml_path or "")
    managed_dir = self._training_eval_managed_output_dir(dataset_dir, create=False)
    if not data_yaml_path or not managed_dir:
        return False
    yaml_dir = os.path.normcase(os.path.dirname(data_yaml_path))
    legacy_dir = os.path.normcase(os.path.join(managed_dir, "training"))
    return yaml_dir in {os.path.normcase(managed_dir), legacy_dir}


def _training_eval_find_obj_yaml(self, dataset_dir=""):
    candidates = []
    dataset_dir = self.normalize_path(dataset_dir or "")

    def add(path):
        path = self.normalize_path(path or "")
        if path and path not in candidates:
            candidates.append(path)

    if dataset_dir and os.path.isdir(dataset_dir):
        add(os.path.join(dataset_dir, "obj.yaml"))
        add(os.path.join(dataset_dir, "data.yaml"))
        managed_dir = os.path.join(dataset_dir, PROJECT_SETTINGS_DIR)
        add(os.path.join(managed_dir, "obj.yaml"))
        add(os.path.join(managed_dir, "data.yaml"))
        managed_training_dir = os.path.join(managed_dir, "training")
        add(os.path.join(managed_training_dir, "obj.yaml"))
        add(os.path.join(managed_training_dir, "data.yaml"))

    for path in (
        getattr(self, "data_yaml_path", ""),
        self.settings.get("ultralyticsDataYamlPath", "") if isinstance(getattr(self, "settings", None), dict) else "",
    ):
        add(path)

    for directory in (
        getattr(self, "train_valid_output_directory", ""),
        self.settings.get("trainValidOutputDirectory", "") if isinstance(getattr(self, "settings", None), dict) else "",
        getattr(self, "image_directory", ""),
        getattr(self, "output_path", ""),
    ):
        directory = self.normalize_path(directory or "")
        if directory and os.path.isdir(directory):
            add(os.path.join(directory, "obj.yaml"))
            add(os.path.join(directory, "data.yaml"))

    for path in candidates:
        if path and os.path.exists(path) and path.lower().endswith((".yaml", ".yml")):
            if dataset_dir and not self._training_eval_yaml_matches_dataset(path, dataset_dir):
                continue
            return self.normalize_path(path)
    return ""


def autoload_training_data_yaml_for_dataset(self, dataset_dir=""):
    dataset_dir = self.normalize_path(dataset_dir or self._training_eval_dataset_dir())
    yaml_path = self._training_eval_find_obj_yaml(dataset_dir)
    if yaml_path:
        self._set_training_data_yaml_path(yaml_path)
        self.queue_settings_save()
    else:
        current_yaml = self.normalize_path(getattr(self, "data_yaml_path", "") or "")
        if current_yaml and dataset_dir and not self._training_eval_yaml_matches_dataset(current_yaml, dataset_dir):
            logger.info("Cleared stale training YAML after dataset switch: %s", current_yaml)
            self._clear_training_data_yaml_path()
            self.queue_settings_save()
    return yaml_path


def _training_eval_valid_percent(self):
    value = (
        self.settings.get("datasetValidPercent", "20%")
        if isinstance(getattr(self, "settings", None), dict)
        else "20%"
    )
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    try:
        percent = float(match.group(0)) if match else 20.0
    except Exception:
        percent = 20.0
    return max(5.0, min(40.0, percent))


def _training_eval_image_class_ids(self, image_path):
    return set(self._training_eval_image_class_counts(image_path))


def _training_eval_prepare_cache_context(self, dataset_dir):
    """Keep evaluator caches bounded to the dataset currently being analyzed."""
    dataset_dir = self.normalize_path(dataset_dir or "")
    context = os.path.normcase(os.path.abspath(dataset_dir)) if dataset_dir else ""
    if getattr(self, "_training_eval_cache_context", "") != context:
        self._training_eval_cache_context = context
        self._training_eval_label_class_cache = {}
        self._training_eval_image_scan_cache = {}


def _training_eval_image_class_counts(self, image_path):
    label_path = self._label_path_for_image(image_path)
    return Counter(self._training_eval_label_record(label_path).get("counts", {}))


def _training_eval_label_record(self, label_path, signature=None):
    """Share one label read between split coverage and geometry analysis."""
    signature = training_file_signature(label_path) if signature is None else signature
    cache = getattr(self, "_training_eval_label_class_cache", None)
    if cache is None:
        cache = self._training_eval_label_class_cache = {}
    cache_key = os.path.normcase(os.path.abspath(label_path)) if label_path else ""
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and cached.get("signature") == signature:
        return cached
    counts = Counter()
    lines = []
    read_error = False
    if signature is not None:
        try:
            with open(label_path, "rb") as stream:
                raw = stream.read()
            for line in raw.decode("utf-8", errors="ignore").splitlines():
                parts = line.split()
                if parts:
                    try:
                        counts[int(float(parts[0]))] += 1
                    except (ValueError, OverflowError):
                        pass
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
        except OSError:
            read_error = True
    record = {"signature": signature, "counts": dict(counts), "lines": lines, "read_error": read_error}
    # A file edited during the read must be read again on the next request.
    if training_file_signature(label_path) == signature and not read_error:
        cache[cache_key] = record
    return record


def _training_eval_cached_image_scan(self, image_path, label_path, polygon_preference, expected_keypoints):
    """Read an image/label pair once and reuse it until either file changes."""
    image_path = self.normalize_path(image_path or "")
    label_path = self.normalize_path(label_path or "")

    image_signature = training_file_signature(image_path)
    label_signature = training_file_signature(label_path)
    parser_key = (2, repr(polygon_preference), int(expected_keypoints or 0))
    cache_key = os.path.normcase(os.path.abspath(image_path))
    normalized_label_path = os.path.normcase(os.path.abspath(label_path)) if label_path else ""
    cache = getattr(self, "_training_eval_image_scan_cache", {}) or {}
    cached = cache.get(cache_key)
    if (
        isinstance(cached, dict)
        and cached.get("label_path", "") == normalized_label_path
        and tuple(cached.get("image_signature") or ()) == tuple(image_signature or ())
        and tuple(cached.get("label_signature") or ()) == tuple(label_signature or ())
        and tuple(cached.get("parser_key") or ()) == parser_key
    ):
        return cached.get("record") or {}

    record = {
        "width": 0,
        "height": 0,
        "image_error": False,
        "missing_label": label_signature is None,
        "empty_label": False,
        "label_read_error": False,
        "invalid_labels": 0,
        "annotations": [],
    }
    try:
        if isinstance(cached, dict) and cached.get("image_signature") == image_signature:
            old_record = cached.get("record") or {}
            if old_record.get("image_error"):
                raise OSError("Unreadable image")
            record["width"], record["height"] = old_record["width"], old_record["height"]
        else:
            record["width"], record["height"] = training_image_dimensions(image_path)
    except Exception:
        record["image_error"] = True
        cache[cache_key] = {
            "label_path": normalized_label_path,
            "image_signature": image_signature,
            "label_signature": label_signature,
            "parser_key": parser_key,
            "record": record,
        }
        self._training_eval_image_scan_cache = cache
        return record

    if label_signature is not None:
        label_record = self._training_eval_label_record(label_path, label_signature)
        lines = label_record["lines"]
        record["label_read_error"] = label_record["read_error"]
        record["empty_label"] = not bool(lines) and not record["label_read_error"]
        for line in lines:
            try:
                raw_values = [float(value) for value in line.split()]
                if not all(math.isfinite(value) for value in raw_values) or not raw_values[0].is_integer():
                    raise ValueError("Invalid annotation numbers")
            except (ValueError, IndexError):
                record["invalid_labels"] += 1
                continue
            bbox = self._parse_training_eval_bbox(line, polygon_preference, expected_keypoints or None)
            if bbox is None:
                record["invalid_labels"] += 1
                continue
            if bbox.obb or bbox.segmentation:
                valid_geometry = all(0 <= value <= 1 for value in (bbox.obb or bbox.segmentation))
            else:
                valid_geometry = (len(raw_values) >= 5 and all(0 <= value <= 1 for value in raw_values[1:5])
                                  and raw_values[3] > 0 and raw_values[4] > 0)
            if not valid_geometry:
                record["invalid_labels"] += 1
                continue
            extent = self._train_eval_label_extent(bbox)
            if extent is None:
                continue
            _xc, _yc, norm_width, norm_height = extent
            if bbox.obb:
                # Measure the rotated rectangle in source pixels, not its
                # inflated axis-aligned envelope (especially thin objects).
                points = np.asarray(bbox.obb, dtype=np.float32).reshape(-1, 2)
                points *= np.asarray([record["width"], record["height"]], dtype=np.float32)
                _center, (edge_a, edge_b), _angle = cv2.minAreaRect(points)
                short_side = min(edge_a, edge_b)
                norm_width = short_side / record["width"]
                norm_height = short_side / record["height"]
            visible_pose_points = [
                (float(x), float(y))
                for x, y, visibility in (bbox.keypoints or [])
                if int(visibility) > 0
            ]
            pose_span_x = 0.0
            pose_span_y = 0.0
            if visible_pose_points:
                xs = [point[0] for point in visible_pose_points]
                ys = [point[1] for point in visible_pose_points]
                pose_span_x = max(xs) - min(xs)
                pose_span_y = max(ys) - min(ys)
            pose_spacing = 0.0
            if len(visible_pose_points) >= 2:
                points = np.asarray(visible_pose_points, dtype=float)
                points *= np.asarray([record["width"], record["height"]])
                distances = np.sqrt(np.sum((points[:, None] - points[None, :]) ** 2, axis=2))
                distinct = distances[distances > 1e-6]
                if distinct.size:
                    pose_spacing = float(distinct.min())
            annotation_kind = "pose" if bbox.keypoints else "seg" if bbox.segmentation else "obb" if bbox.obb else "bbox"
            record["annotations"].append((
                int(bbox.class_id),
                float(norm_width),
                float(norm_height),
                annotation_kind,
                len(visible_pose_points),
                float(pose_span_x),
                float(pose_span_y),
                float(pose_spacing),
            ))

    cache[cache_key] = {
        "label_path": normalized_label_path,
        "image_signature": image_signature,
        "label_signature": label_signature,
        "parser_key": parser_key,
        "record": record,
    }
    self._training_eval_image_scan_cache = cache
    return record


def _training_eval_split_class_report(self, train_files, valid_files, class_names=None):
    train_files = [self.normalize_path(path) for path in train_files or [] if path]
    valid_files = [self.normalize_path(path) for path in valid_files or [] if path]
    class_names = list(class_names or [])
    train_counts = Counter()
    valid_counts = Counter()
    for image_path in train_files:
        train_counts.update(self._training_eval_image_class_counts(image_path))
    for image_path in valid_files:
        valid_counts.update(self._training_eval_image_class_counts(image_path))

    class_ids = set(train_counts) | set(valid_counts) | set(range(len(class_names)))
    rows = []
    missing_valid = []
    missing_train = []
    for class_id in sorted(class_ids):
        train_count = int(train_counts.get(class_id, 0))
        valid_count = int(valid_counts.get(class_id, 0))
        total = train_count + valid_count
        if total <= 0:
            continue
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        row = {
            "class_id": int(class_id),
            "class_name": str(class_name),
            "train_labels": train_count,
            "valid_labels": valid_count,
            "total_labels": total,
            "valid_pct": round((valid_count / total) * 100.0, 2) if total else 0.0,
        }
        rows.append(row)
        if train_count > 0 and valid_count == 0:
            missing_valid.append(row)
        if valid_count > 0 and train_count == 0:
            missing_train.append(row)

    overlap = len(set(train_files) & set(valid_files))
    return {
        "train_images": len(train_files),
        "valid_images": len(valid_files),
        "overlap_images": overlap,
        "same_split": bool(train_files and valid_files and set(train_files) == set(valid_files)),
        "rows": rows,
        "missing_valid": missing_valid,
        "missing_train": missing_train,
    }


def _training_eval_yaml_split_report(self, data_yaml_path, yaml_data=None):
    data_yaml_path = self.normalize_path(data_yaml_path or "")
    if not data_yaml_path or not os.path.exists(data_yaml_path):
        return {}
    try:
        yaml_mtime = os.path.getmtime(data_yaml_path)
    except Exception:
        yaml_mtime = 0.0
    cache = getattr(self, "_training_eval_yaml_split_report_cache", {}) or {}
    cache_key = os.path.normcase(os.path.abspath(data_yaml_path))
    cached = cache.get(cache_key)
    now = time.time()
    if (
        isinstance(cached, dict)
        and float(cached.get("mtime", -1.0)) == float(yaml_mtime)
        and now - float(cached.get("at", 0.0) or 0.0) < 30.0
    ):
        return dict(cached.get("report") or {})
    yaml_data = yaml_data if isinstance(yaml_data, dict) else None
    if yaml_data is None:
        yaml_data, _yaml_dir = self._training_health_parse_data_yaml(data_yaml_path)
    class_names = self._training_eval_yaml_names(yaml_data or {})
    train_files = self._training_eval_yaml_image_files(data_yaml_path, yaml_data, "train")
    valid_files = self._training_eval_yaml_image_files(data_yaml_path, yaml_data, "val")
    report = self._training_eval_split_class_report(train_files, valid_files, class_names)
    cache[cache_key] = {"mtime": yaml_mtime, "at": now, "report": dict(report)}
    self._training_eval_yaml_split_report_cache = cache
    return report


def _training_eval_train_val_split(self, image_files, valid_percent=None):
    image_files = [self.normalize_path(path) for path in image_files or [] if path]
    image_files = list(dict.fromkeys(image_files))
    if len(image_files) < 2:
        return image_files, []

    valid_percent = self._training_eval_valid_percent() if valid_percent is None else float(valid_percent)
    target_valid = int(round(len(image_files) * (valid_percent / 100.0)))
    target_valid = max(1, min(len(image_files) - 1, target_valid))

    image_classes = {path: self._training_eval_image_class_ids(path) for path in image_files}
    class_counts = Counter()
    for class_ids in image_classes.values():
        class_counts.update(class_ids)

    selected = []
    selected_set = set()
    selected_class_counts = Counter()

    def keeps_training_coverage(path):
        for class_id in image_classes.get(path, set()):
            if class_counts.get(class_id, 0) - selected_class_counts.get(class_id, 0) <= 1:
                return False
        return True

    def add_image(path):
        if path in selected_set or len(selected) >= target_valid:
            return False
        if not keeps_training_coverage(path):
            return False
        selected.append(path)
        selected_set.add(path)
        selected_class_counts.update(image_classes.get(path, set()))
        return True

    by_rare_class = sorted(class_counts, key=lambda class_id: (class_counts[class_id], class_id))
    for class_id in by_rare_class:
        if class_counts[class_id] < 2:
            continue
        candidates = [
            path for path in image_files
            if path not in selected_set and class_id in image_classes.get(path, set())
        ]
        if not candidates:
            continue
        candidates = [
            path for path in candidates
            if keeps_training_coverage(path)
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda path: (len(image_classes.get(path, set())), path))
        add_image(candidates[0])
        if len(selected) >= target_valid:
            break

    remaining = [
        path for path in image_files
        if path not in selected_set
        and keeps_training_coverage(path)
    ]
    random.Random(0).shuffle(remaining)
    for path in remaining:
        if len(selected) >= target_valid:
            break
        add_image(path)

    def remove_image(path):
        if path not in selected_set:
            return False
        selected_set.remove(path)
        try:
            selected.remove(path)
        except ValueError:
            pass
        for class_id in image_classes.get(path, set()):
            selected_class_counts[class_id] -= 1
            if selected_class_counts[class_id] <= 0:
                selected_class_counts.pop(class_id, None)
        return True

    def can_remove_without_losing_valid_coverage(path):
        for class_id in image_classes.get(path, set()):
            if class_counts.get(class_id, 0) >= 2 and selected_class_counts.get(class_id, 0) <= 1:
                return False
        return True

    feasible_classes = [class_id for class_id, count in class_counts.items() if count >= 2]
    for class_id in sorted(feasible_classes, key=lambda cid: (selected_class_counts.get(cid, 0), class_counts[cid], cid)):
        if selected_class_counts.get(class_id, 0) > 0:
            continue
        candidates = [
            path for path in image_files
            if path not in selected_set
            and class_id in image_classes.get(path, set())
            and keeps_training_coverage(path)
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda path: (
                -sum(1 for cid in image_classes.get(path, set()) if selected_class_counts.get(cid, 0) == 0 and class_counts.get(cid, 0) >= 2),
                len(image_classes.get(path, set())),
                path,
            )
        )
        if len(selected) < target_valid:
            add_image(candidates[0])
            continue
        removable = [
            path for path in selected
            if can_remove_without_losing_valid_coverage(path)
        ]
        removable.sort(
            key=lambda path: (
                sum(1 for cid in image_classes.get(path, set()) if selected_class_counts.get(cid, 0) <= 1),
                len(image_classes.get(path, set())),
                path,
            )
        )
        if removable and remove_image(removable[0]):
            add_image(candidates[0])

    if not selected:
        fallback = [path for path in image_files if path not in selected_set]
        random.Random(0).shuffle(fallback)
        for path in fallback:
            if add_image(path):
                break

    valid_set = set(selected)
    train_files = [path for path in image_files if path not in valid_set]
    valid_files = [path for path in image_files if path in valid_set]
    if not train_files:
        train_files, valid_files = image_files[:-1], image_files[-1:]
    return train_files, valid_files


def _training_eval_percent_value(self, value, default=20.0, minimum=1.0, maximum=100.0):
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    try:
        percent = float(match.group(0)) if match else float(default)
    except Exception:
        percent = float(default)
    return max(float(minimum), min(float(maximum), percent))


def _training_eval_validation_mode(self, value=None):
    mode = str(
        value
        if value is not None
        else self.settings.get("trainingEvaluatorValidationMode", "auto")
        if isinstance(getattr(self, "settings", None), dict)
        else "auto"
    ).strip().lower()
    aliases = {
        "auto split": "auto",
        "auto_split": "auto",
        "folder": "folder",
        "valid folder": "folder",
        "validation folder": "folder",
        "txt": "txt",
        "valid txt": "txt",
        "valid.txt": "txt",
        "train": "train",
        "train as val": "train",
        "train_as_val": "train",
    }
    return aliases.get(mode, mode if mode in {"auto", "folder", "txt", "train"} else "auto")


def _training_eval_image_files_from_directory(self, directory):
    directory = self.normalize_path(directory or "")
    if not directory or not os.path.isdir(directory):
        return []
    image_exts = {str(ext).lower() for ext in IMAGE_SUFFIXES}
    files = []
    for root, _dirs, names in os.walk(directory):
        for name in names:
            if os.path.splitext(name)[1].lower() in image_exts:
                files.append(self.normalize_path(os.path.join(root, name)))
    return sorted(dict.fromkeys(files))


def _training_eval_read_image_list_file(self, list_path, dataset_dir=""):
    list_path = self.normalize_path(list_path or "")
    if not list_path or not os.path.isfile(list_path):
        return []
    image_exts = {str(ext).lower() for ext in IMAGE_SUFFIXES}
    list_dir = os.path.dirname(list_path)
    dataset_dir = self.normalize_path(dataset_dir or "")
    files = []
    try:
        with open(list_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = list(f)
    except Exception:
        return []

    for line in lines:
        text = str(line or "").strip().strip('"').strip("'")
        if not text or text.startswith("#"):
            continue
        if os.path.isabs(text):
            candidate = self.normalize_path(text)
        else:
            candidate = ""
            for base in (list_dir, dataset_dir):
                base = self.normalize_path(base or "")
                if not base:
                    continue
                possible = self.normalize_path(os.path.join(base, text))
                if os.path.exists(possible):
                    candidate = possible
                    break
            if not candidate:
                candidate = self.normalize_path(os.path.join(list_dir, text))
        if os.path.isfile(candidate) and os.path.splitext(candidate)[1].lower() in image_exts:
            files.append(candidate)
    return list(dict.fromkeys(files))


def _training_eval_sample_images(self, image_files, percent=100.0, seed=0):
    image_files = [self.normalize_path(path) for path in image_files or [] if path]
    image_files = list(dict.fromkeys(image_files))
    if not image_files:
        return []
    percent = self._training_eval_percent_value(percent, default=100.0, minimum=1.0, maximum=100.0)
    if percent >= 99.5:
        return image_files
    count = int(round(len(image_files) * (percent / 100.0)))
    count = max(1, min(len(image_files), count))
    shuffled = list(image_files)
    random.Random(seed).shuffle(shuffled)
    selected = set(shuffled[:count])
    return [path for path in image_files if path in selected]


def _training_eval_write_image_list(self, path, image_files):
    path = self.normalize_path(path or "")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = "".join(
        os.path.abspath(image_file).replace("\\", "/") + "\n"
        for image_file in image_files or []
    )
    self._training_eval_write_text_atomic(path, content)


def _training_eval_write_text_atomic(self, path, content):
    path = self.normalize_path(path or "")
    if not path:
        raise ValueError("Training metadata output path is empty.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            file.write(str(content))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def _training_eval_prepare_validation_files(
    self,
    dataset_dir,
    output_dir,
    source_yaml_path="",
    validation_mode=None,
    validation_source="",
    validation_percent=None,
    validation_source_percent=None,
    target_yaml_path="",
):
    dataset_dir = self.normalize_path(dataset_dir or "")
    source_yaml_path = self.normalize_path(source_yaml_path or "")
    source_yaml_data, source_yaml_dir = (
        self._training_health_parse_data_yaml(source_yaml_path)
        if source_yaml_path and os.path.exists(source_yaml_path)
        else ({}, "")
    )
    if source_yaml_path and not dataset_dir:
        dataset_dir = self._training_health_dataset_dir(source_yaml_path, source_yaml_data, source_yaml_dir)

    output_dir = self.normalize_path(output_dir or source_yaml_dir or dataset_dir)
    if not dataset_dir or not os.path.isdir(dataset_dir):
        raise ValueError("Open or select a dataset folder before preparing obj.yaml.")
    if not output_dir:
        output_dir = dataset_dir

    mode = self._training_eval_validation_mode(validation_mode)
    validation_source = self.normalize_path(
        validation_source
        or self.settings.get("trainingEvaluatorValidationSource", "")
        if isinstance(getattr(self, "settings", None), dict)
        else ""
    )
    valid_pool = []
    if mode == "folder":
        valid_pool = self._training_eval_image_files_from_directory(validation_source)
        if not valid_pool:
            raise ValueError("Validation folder is empty or missing. Select a Valid Folder source or choose Auto Split / Train as Val.")
    elif mode == "txt":
        valid_pool = self._training_eval_read_image_list_file(validation_source, dataset_dir)
        if not valid_pool:
            raise ValueError("Validation TXT is empty or missing. Select a Valid TXT source or choose Auto Split / Train as Val.")

    os.makedirs(output_dir, exist_ok=True)

    source_train_files = []
    source_valid_files = []
    image_files = []
    source_is_managed = self._training_eval_is_managed_yaml(source_yaml_path, dataset_dir)
    if source_yaml_path and os.path.exists(source_yaml_path) and not source_is_managed:
        source_train_files = self._training_eval_yaml_image_files(source_yaml_path, source_yaml_data, "train")
        source_valid_files = self._training_eval_yaml_image_files(source_yaml_path, source_yaml_data, "val")
        image_files = list(source_train_files)
        if source_train_files:
            logger.info("Using training images from obj.yaml train list: %s", source_yaml_path)

    if not image_files:
        scanner = ScanAnnotations(self)
        scanner.base_directory = dataset_dir
        image_files = scanner._iter_image_files(dataset_dir)
    if not image_files:
        image_files = self._training_eval_image_files_from_directory(dataset_dir)
    image_files = [self.normalize_path(path) for path in image_files or [] if path]
    image_files = sorted(dict.fromkeys(image_files))
    try:
        from darkfusion_negative_crops import resolve_negative_folder

        blanks_dir = resolve_negative_folder(dataset_dir, create=False)
        blank_files = (
            self._training_eval_image_files_from_directory(blanks_dir)
            if os.path.isdir(blanks_dir)
            else []
        )
    except Exception:
        blank_files = []
    blank_files = sorted(dict.fromkeys(
        self.normalize_path(path) for path in blank_files if path
    ))
    if blank_files:
        image_files = sorted(dict.fromkeys([*image_files, *blank_files]))
        logger.info("Included %s image(s) from the shared blanks folder.", len(blank_files))
    blank_set = set(blank_files)
    if not image_files:
        raise ValueError("No dataset images were found for training YAML creation.")

    if source_valid_files:
        combined_files = sorted(dict.fromkeys(
            self.normalize_path(path)
            for path in [*source_train_files, *source_valid_files]
            if path
        ))
        if len(combined_files) > len(image_files):
            image_files = combined_files
            logger.info(
                "Using obj.yaml train+val union as split source: %s image(s)",
                len(image_files),
            )
    split_percent = self._training_eval_percent_value(
        validation_percent
        if validation_percent is not None
        else self.settings.get("trainingEvaluatorValidationPercent", self.settings.get("datasetValidPercent", "20%"))
        if isinstance(getattr(self, "settings", None), dict)
        else "20%",
        default=20.0,
        minimum=5.0,
        maximum=100.0 if mode == "train" else 50.0,
    )
    source_percent = self._training_eval_percent_value(
        validation_source_percent
        if validation_source_percent is not None
        else self.settings.get("trainingEvaluatorValidationSourcePercent", "100%")
        if isinstance(getattr(self, "settings", None), dict)
        else "100%",
        default=100.0,
        minimum=1.0,
        maximum=100.0,
    )
    note = ""
    if mode == "folder":
        valid_files = self._training_eval_sample_images(valid_pool, source_percent, seed=1)
        excluded = set(valid_pool)
        train_files = [path for path in image_files if path not in excluded]
        note = f"Validation uses {len(valid_files)} image(s) from folder ({source_percent:g}% of folder)."
    elif mode == "txt":
        valid_files = self._training_eval_sample_images(valid_pool, source_percent, seed=1)
        excluded = set(valid_pool)
        train_files = [path for path in image_files if path not in excluded]
        note = f"Validation uses {len(valid_files)} image(s) from valid.txt ({source_percent:g}% of list)."
    elif mode == "train":
        train_files = list(image_files)
        valid_files = self._training_eval_sample_images(image_files, split_percent, seed=1)
        note = f"Validation uses {split_percent:g}% of the training images ({len(valid_files)} image(s)) for smoke-checking."
    else:
        split_source = [path for path in image_files if path not in blank_set]
        train_files, valid_files = self._training_eval_train_val_split(
            split_source or image_files,
            split_percent,
        )
        if not valid_files:
            valid_files = list(train_files)
            note = "Dataset is too small for a holdout split; validation uses the training list."
        else:
            note = f"Validation auto-split uses {len(valid_files)} image(s) ({split_percent:g}%)."

    # Background-only examples belong in training.  Keeping the shared
    # blanks folder out of validation avoids diluting validation metrics.
    if blank_set:
        train_files = list(dict.fromkeys([*train_files, *blank_files]))
        valid_files = [path for path in valid_files if path not in blank_set]
        # If filtering blanks emptied valid_files (especially for "train as val" mode),
        # resample from non-blank images to preserve the percentage split
        if not valid_files and mode == "train":
            non_blank_images = [path for path in image_files if path not in blank_set]
            if non_blank_images:
                valid_files = self._training_eval_sample_images(non_blank_images, split_percent, seed=1)

    train_files = [path for path in train_files if path]
    valid_files = [path for path in valid_files if path]
    if not train_files:
        raise ValueError("Validation selection leaves no training images. Choose a smaller validation source.")
    if not valid_files:
        valid_files = list(train_files)

    class_names = [] if source_is_managed else self._training_eval_yaml_names(source_yaml_data)
    if not class_names:
        try:
            class_names = list(self.load_classes(data_directory=dataset_dir) or [])
        except Exception:
            class_names = []
    split_class_report = self._training_eval_split_class_report(train_files, valid_files, class_names)

    train_txt_path = os.path.join(output_dir, "train.txt").replace("\\", "/")
    valid_txt_path = os.path.join(output_dir, "valid.txt").replace("\\", "/")
    self._training_eval_write_image_list(train_txt_path, train_files)
    self._training_eval_write_image_list(valid_txt_path, valid_files)

    self.image_directory = dataset_dir
    self.images = image_files
    self.train_valid_output_directory = output_dir
    if isinstance(getattr(self, "settings", None), dict):
        self.settings["trainValidOutputDirectory"] = output_dir

    yaml_options = {"source_yaml_path": source_yaml_path}
    if target_yaml_path:
        yaml_options["data_yaml_path"] = target_yaml_path
    self.create_and_update_yaml_and_data_files(output_dir, "valid.txt", **yaml_options)
    yaml_path = self.normalize_path(target_yaml_path) if target_yaml_path else os.path.join(output_dir, "obj.yaml").replace("\\", "/")
    if not os.path.exists(yaml_path):
        raise ValueError(f"Training YAML was not created:\n{yaml_path}")

    return {
        "yaml_path": self.normalize_path(yaml_path),
        "train_txt_path": self.normalize_path(train_txt_path),
        "valid_txt_path": self.normalize_path(valid_txt_path),
        "train_count": len(train_files),
        "valid_count": len(valid_files),
        "source_count": len(image_files),
        "validation_mode": mode,
        "validation_note": note,
        "split_class_report": split_class_report,
    }


def ensure_training_evaluator_data_yaml(
    self,
    force=False,
    prepare_validation=False,
    validation_mode=None,
    validation_source="",
    validation_percent=None,
    validation_source_percent=None,
):
    self._training_eval_last_validation_prep = {}
    loaded_dataset_dir = self._training_eval_dataset_dir()
    existing_yaml = self.normalize_path(getattr(self, "data_yaml_path", "") or "")
    if existing_yaml and loaded_dataset_dir and (
        not os.path.isfile(existing_yaml)
        or not self._training_eval_yaml_matches_dataset(existing_yaml, loaded_dataset_dir)
    ):
        if os.path.isfile(existing_yaml):
            logger.warning(
                "Ignoring training YAML from a different dataset: %s (loaded dataset: %s)",
                existing_yaml,
                loaded_dataset_dir,
            )
        else:
            logger.warning("Ignoring missing training YAML: %s", existing_yaml)
        existing_yaml = ""
        self._clear_training_data_yaml_path()
        self.queue_settings_save()
    existing_yaml_data, existing_yaml_dir = (
        self._training_health_parse_data_yaml(existing_yaml)
        if existing_yaml and os.path.exists(existing_yaml)
        else ({}, "")
    )
    if existing_yaml and os.path.exists(existing_yaml) and not prepare_validation:
        refreshed_yaml = self._training_eval_refresh_stale_data_yaml(existing_yaml)
        if refreshed_yaml:
            return refreshed_yaml
        return existing_yaml

    dataset_dir = loaded_dataset_dir
    if not dataset_dir and existing_yaml and os.path.exists(existing_yaml):
        dataset_dir = self._training_health_dataset_dir(existing_yaml, existing_yaml_data, existing_yaml_dir)
    if not dataset_dir:
        raise ValueError("Open or select a dataset folder before preparing training YAML.")
    self._training_eval_prepare_cache_context(dataset_dir)

    obj_yaml = self._training_eval_find_obj_yaml(dataset_dir)
    if obj_yaml and not prepare_validation:
        refreshed_yaml = self._training_eval_refresh_stale_data_yaml(obj_yaml)
        if refreshed_yaml:
            return refreshed_yaml
        self._set_training_data_yaml_path(obj_yaml)
        self.queue_settings_save()
        return obj_yaml
    source_yaml_path = existing_yaml if existing_yaml and os.path.exists(existing_yaml) else obj_yaml

    output_dir = getattr(self, "train_valid_output_directory", "")
    if not output_dir and isinstance(getattr(self, "settings", None), dict):
        output_dir = self.settings.get("trainValidOutputDirectory", "")
    output_dir = self.normalize_path(output_dir)
    if output_dir and not self._training_eval_paths_overlap(output_dir, dataset_dir):
        logger.info(
            "Ignoring training output folder from a different dataset: %s",
            output_dir,
        )
        output_dir = ""
    if output_dir and not os.path.isdir(output_dir):
        logger.info("Ignoring missing training output folder: %s", output_dir)
        output_dir = ""
    if not output_dir and source_yaml_path:
        output_dir = os.path.dirname(source_yaml_path)
    if not output_dir:
        output_dir = self._training_eval_managed_output_dir(dataset_dir, create=True)
    output_dir = self.normalize_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if prepare_validation:
        prep = self._training_eval_prepare_validation_files(
            dataset_dir,
            output_dir,
            source_yaml_path=source_yaml_path,
            validation_mode=validation_mode,
            validation_source=validation_source,
            validation_percent=validation_percent,
            validation_source_percent=validation_source_percent,
            target_yaml_path=existing_yaml,
        )
        self._training_eval_last_validation_prep = prep
        yaml_path = prep.get("yaml_path", "")
        self._set_training_data_yaml_path(yaml_path)
        self.queue_settings_save()
        return yaml_path

    scanner = ScanAnnotations(self)
    scanner.base_directory = dataset_dir
    image_files = []
    if (
        source_yaml_path
        and os.path.exists(source_yaml_path)
        and not self._training_eval_is_managed_yaml(source_yaml_path, dataset_dir)
    ):
        source_yaml_data, _source_yaml_dir = self._training_health_parse_data_yaml(source_yaml_path)
        image_files = self._training_eval_yaml_image_files(source_yaml_path, source_yaml_data, "train")
    if not image_files:
        image_files = scanner._iter_image_files(dataset_dir)
    if not image_files:
        image_files = self._training_eval_image_files_from_directory(dataset_dir)
    if not image_files:
        raise ValueError("No dataset images were found for training YAML creation.")

    train_txt_path = os.path.join(output_dir, "train.txt").replace("\\", "/")
    valid_txt_path = os.path.join(output_dir, "valid.txt").replace("\\", "/")
    if not os.path.exists(train_txt_path) or force:
        self._training_eval_write_image_list(train_txt_path, image_files)

    self.image_directory = dataset_dir
    self.images = image_files
    self.train_valid_output_directory = output_dir
    if isinstance(getattr(self, "settings", None), dict):
        self.settings["trainValidOutputDirectory"] = output_dir

    selected_option = "valid.txt" if os.path.exists(valid_txt_path) else "train.txt"
    self.create_and_update_yaml_and_data_files(output_dir, selected_option, source_yaml_path=source_yaml_path)

    yaml_path = os.path.join(output_dir, "obj.yaml").replace("\\", "/")
    if not os.path.exists(yaml_path):
        raise ValueError(f"Training YAML was not created:\n{yaml_path}")

    self._set_training_data_yaml_path(yaml_path)
    self.queue_settings_save()
    return yaml_path


def _training_yaml_image_count(self, data_yaml_path):
    data_yaml_path = self.normalize_path(data_yaml_path)
    if not data_yaml_path or not os.path.exists(data_yaml_path):
        return 0

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def resolve(value, base_dir):
        value = str(value or "").strip()
        if not value:
            return ""
        return self.normalize_path(value if os.path.isabs(value) else os.path.join(base_dir, value))

    try:
        with open(data_yaml_path, "r", encoding="utf-8", errors="ignore") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return 0

    yaml_dir = os.path.dirname(data_yaml_path)
    entries = data.get("train")
    entries = entries if isinstance(entries, list) else [entries]
    count = 0
    for entry in entries:
        path = resolve(entry, yaml_dir)
        if not path:
            continue
        if os.path.isfile(path) and path.lower().endswith(".txt"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    count += sum(1 for line in f if line.strip())
            except Exception:
                continue
        elif os.path.isdir(path):
            for _root, _dirs, files in os.walk(path):
                count += sum(1 for name in files if os.path.splitext(name)[1].lower() in image_exts)
        elif os.path.isfile(path) and os.path.splitext(path)[1].lower() in image_exts:
            count += 1
    return count


def _training_eval_yaml_names(self, yaml_data):
    names = (yaml_data or {}).get("names") if isinstance(yaml_data, dict) else None
    if isinstance(names, dict):
        def sort_key(item):
            key = item[0]
            try:
                return int(key)
            except Exception:
                return str(key)
        return [str(value) for _key, value in sorted(names.items(), key=sort_key)]
    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]
    return []


def _training_eval_yaml_image_files(self, data_yaml_path, yaml_data=None, split="train"):
    data_yaml_path = self.normalize_path(data_yaml_path)
    if not data_yaml_path or not os.path.exists(data_yaml_path):
        return []

    yaml_data = yaml_data if isinstance(yaml_data, dict) else None
    yaml_dir = os.path.dirname(data_yaml_path)
    if yaml_data is None:
        try:
            with open(data_yaml_path, "r", encoding="utf-8", errors="ignore") as f:
                yaml_data = yaml.safe_load(f) or {}
        except Exception:
            yaml_data = {}

    image_exts = {str(ext).lower() for ext in IMAGE_SUFFIXES}
    dataset_root = self.normalize_path((yaml_data or {}).get("path", "") or "")
    if dataset_root and not os.path.isabs(dataset_root):
        dataset_root = self.normalize_path(os.path.join(yaml_dir, dataset_root))
    if not dataset_root:
        dataset_root = yaml_dir

    def resolve_path(value, base_dir=None):
        value = str(value or "").strip().strip('"').strip("'")
        if not value:
            return ""
        if os.path.isabs(value):
            return self.normalize_path(value)
        for base in (base_dir, dataset_root, yaml_dir):
            base = self.normalize_path(base or "")
            if not base:
                continue
            candidate = self.normalize_path(os.path.join(base, value))
            if os.path.exists(candidate):
                return candidate
        return self.normalize_path(os.path.join(dataset_root, value))

    def collect_from_entry(entry, base_dir=None):
        path = resolve_path(entry, base_dir)
        if not path:
            return []
        ext = os.path.splitext(path)[1].lower()
        if os.path.isfile(path) and ext == ".txt":
            files = []
            list_dir = os.path.dirname(path)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        resolved = resolve_path(line, list_dir)
                        if resolved and os.path.isfile(resolved) and os.path.splitext(resolved)[1].lower() in image_exts:
                            files.append(resolved)
            except Exception:
                return []
            return files
        if os.path.isdir(path):
            files = []
            for root, _dirs, names in os.walk(path):
                for name in names:
                    if os.path.splitext(name)[1].lower() in image_exts:
                        files.append(self.normalize_path(os.path.join(root, name)))
            return files
        if os.path.isfile(path) and ext in image_exts:
            return [path]
        return []

    entries = (yaml_data or {}).get(split)
    entries = entries if isinstance(entries, list) else [entries]
    files = []
    for entry in entries:
        files.extend(collect_from_entry(entry))
    return sorted(dict.fromkeys(path for path in files if path))


def _training_eval_data_yaml_needs_auto_refresh(self, data_yaml_path):
    data_yaml_path = self.normalize_path(data_yaml_path or "")
    if not data_yaml_path or not os.path.exists(data_yaml_path):
        return False
    if os.path.basename(data_yaml_path).lower() != "obj.yaml":
        return False
    settings = getattr(self, "settings", {}) if isinstance(getattr(self, "settings", None), dict) else {}
    if not bool(settings.get("trainingEvaluatorAutoYaml", True)):
        return False
    if self._training_eval_validation_mode() == "train":
        return False

    yaml_data, _yaml_dir = self._training_health_parse_data_yaml(data_yaml_path)
    if not isinstance(yaml_data, dict) or not yaml_data:
        return False

    train_entry = yaml_data.get("train")
    val_entry = yaml_data.get("val")
    if not train_entry:
        return False
    if not val_entry:
        return True
    if str(train_entry).strip().strip('"').strip("'") == str(val_entry).strip().strip('"').strip("'"):
        return True

    train_files = self._training_eval_yaml_image_files(data_yaml_path, yaml_data, "train")
    val_files = self._training_eval_yaml_image_files(data_yaml_path, yaml_data, "val")
    if not val_files:
        return True
    if train_files and set(train_files) == set(val_files):
        return True
    return False


def _training_eval_refresh_data_yaml_split(
    self,
    data_yaml_path,
    validation_mode=None,
    validation_source="",
    validation_percent=None,
    validation_source_percent=None,
):
    data_yaml_path = self.normalize_path(data_yaml_path or "")
    if not data_yaml_path or not os.path.exists(data_yaml_path):
        raise ValueError("Selected obj.yaml does not exist.")

    yaml_data, yaml_dir = self._training_health_parse_data_yaml(data_yaml_path)
    dataset_dir = self._training_health_dataset_dir(data_yaml_path, yaml_data, yaml_dir)
    if not dataset_dir:
        dataset_dir = self._training_eval_dataset_dir()
    if not dataset_dir:
        raise ValueError("Could not resolve the dataset folder from obj.yaml.")

    output_dir = yaml_dir if yaml_dir and os.path.isdir(yaml_dir) else dataset_dir
    prep = self._training_eval_prepare_validation_files(
        dataset_dir,
        output_dir,
        source_yaml_path=data_yaml_path,
        validation_mode=validation_mode,
        validation_source=validation_source,
        validation_percent=validation_percent,
        validation_source_percent=validation_source_percent,
    )
    yaml_path = prep.get("yaml_path", "")
    if yaml_path:
        self._training_eval_last_validation_prep = prep
        self._set_training_data_yaml_path(yaml_path)
        self.queue_settings_save()
        logger.info("Auto-refreshed obj.yaml validation split: %s", yaml_path)
    return yaml_path


def _training_eval_refresh_stale_data_yaml(self, data_yaml_path):
    if not self._training_eval_data_yaml_needs_auto_refresh(data_yaml_path):
        return ""
    return self._training_eval_refresh_data_yaml_split(
        data_yaml_path,
        validation_mode=self.settings.get("trainingEvaluatorValidationMode", "auto")
        if isinstance(getattr(self, "settings", None), dict)
        else "auto",
        validation_source=self.settings.get("trainingEvaluatorValidationSource", "")
        if isinstance(getattr(self, "settings", None), dict)
        else "",
        validation_percent=self.settings.get("trainingEvaluatorValidationPercent", "20%")
        if isinstance(getattr(self, "settings", None), dict)
        else "20%",
        validation_source_percent=self.settings.get("trainingEvaluatorValidationSourcePercent", "100%")
        if isinstance(getattr(self, "settings", None), dict)
        else "100%",
    )


def browse_yaml_clicked(self):
    file_name = self.open_file_dialog(
        "Select YAML File", "YAML Files (*.yaml);;All Files (*)")
    if file_name:
        self._set_training_data_yaml_path(file_name)
    else:
        self.data_yaml_path = None
        self.yaml_label.setText("No Data YAML selected")
    self.queue_settings_save()


def browse_pt_clicked(self):
    file_name = self.open_file_dialog("Select Model File", "Model Files (*.pt *.yaml *.yml);;All Files (*)")
    if not file_name:
        return
    file_name = self.normalize_path(file_name)
    if file_name.lower().endswith('.pt'):
        self.pt_path = file_name
        self.pt_label.setText(f"Model: {file_name}")
        self.model_config_path = None  # Reset .yaml config path if a .pt is chosen
        self.pretrained_model_path = None
    elif file_name.lower().endswith(('.yaml', '.yml')):
        self.model_config_path = file_name
        self.pt_path = None
        if getattr(self, "pretrained_model_path", None):
            self.pt_label.setText(
                f"Model Config: {file_name}; Pretrained: {self.pretrained_model_path}"
            )
        else:
            self.pt_label.setText(f"Model Config: {file_name}")
    else:
        self.model_config_path = None
        self.pretrained_model_path = None
        self.pt_label.setText("No Model selected")
    self.queue_settings_save()


def open_file_dialog(self, caption, file_filter, multiple=False, category=None, start_path=None):
    if category is None:
        combined = f"{caption} {file_filter}".lower()
        if any(token in combined for token in ("model", "weight", "checkpoint", ".pt", ".engine", ".onnx")):
            category = "model"
        elif any(token in combined for token in ("yaml", "validation", "dataset", ".txt")):
            category = "training"
        else:
            category = "generic"
    start_directory = self.dialog_start_directory(category, start_path)
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    if multiple:
        file_names, _ = QFileDialog.getOpenFileNames(
            self, caption, start_directory, file_filter, options=options)
        if file_names:
            self.remember_dialog_selection(category, file_names[0])
        return file_names
    else:
        file_name, _ = QFileDialog.getOpenFileName(
            self, caption, start_directory, file_filter, options=options)
        if file_name:
            self.remember_dialog_selection(category, file_name)
        return file_name


def on_save_dir_clicked(self):
    logger.debug("on_save_dir_clicked called")
    directory = self._training_eval_project_dir()
    if not directory:
        QMessageBox.information(self, "Training Runs", "Open a dataset folder first.")
        return
    os.makedirs(directory, exist_ok=True)
    self.runs_directory = directory
    if isinstance(getattr(self, "settings", None), dict):
        self.settings["ultralyticsRunsDirectory"] = directory
    self.queue_settings_save()
    try:
        os.startfile(directory)
    except Exception as e:
        QMessageBox.information(self, "Training Runs", f"Runs folder:\n{directory}\n\nCould not open it automatically: {e}")


def _compact_train_export_panels(self):
    self._compact_ultralytics_train_panel()
    self._compact_ultralytics_export_panel()
    self._compact_train_params_panel()
    self._compact_darknet_options_panel()
    self._compact_darknet_train_panel()
    self._setup_ultralytics_tab_launcher()


def _setup_ultralytics_tab_launcher(self):
    """Replace the legacy embedded Ultralytics controls with one trainer entry point."""
    if getattr(self, "_ultralytics_tab_launcher_ready", False):
        return

    tab = getattr(self, "tab_17", None)
    layout = getattr(self, "gridLayout_5", None)
    if not isinstance(tab, QtWidgets.QWidget) or not isinstance(layout, QtWidgets.QGridLayout):
        return

    # Keep the legacy widgets alive as compatibility state holders for saved
    # settings and older backend methods, but remove them from the visible
    # workflow. The unified Training dialog now owns all visible controls.
    for name in ("groupBox_10", "groupBox_8", "groupBox_9"):
        widget = getattr(self, name, None)
        if isinstance(widget, QtWidgets.QWidget):
            widget.hide()

    launcher = QtWidgets.QGroupBox("Ultralytics Trainer / Exporter", tab)
    launcher.setObjectName("ultralytics_trainer_launcher")
    launcher_layout = QtWidgets.QVBoxLayout(launcher)
    launcher_layout.setContentsMargins(14, 18, 14, 14)
    launcher_layout.setSpacing(10)

    description = QtWidgets.QLabel(
        "Training, validation review, metrics, checkpoints, and model export now live "
        "in one Ultralytics workflow window."
    )
    description.setWordWrap(True)
    launcher_layout.addWidget(description)

    open_button = QtWidgets.QPushButton("Open Ultralytics Trainer / Exporter")
    open_button.setObjectName("ultralytics_trainer_launcher_button")
    open_button.setToolTip("Open the unified Ultralytics training, validation, and export workflow.")
    launcher_layout.addWidget(open_button)

    layout.addWidget(launcher, 0, 0, 1, 1)
    layout.setRowStretch(1, 1)
    self.ultralytics_trainer_launcher = launcher
    self.ultralytics_trainer_launcher_button = open_button
    self._ultralytics_tab_launcher_ready = True


def _compact_ultralytics_train_panel(self):
    if getattr(self, "_ultralytics_train_panel_compacted", False):
        return

    group = getattr(self, "groupBox_10", None)
    layout = self._prepare_group_grid(group)
    if layout is None:
        return

    if not isinstance(getattr(self, "training_evaluator_button", None), QtWidgets.QPushButton):
        self.training_evaluator_button = QtWidgets.QPushButton("Open Trainer", group)
        self.training_evaluator_button.setObjectName("training_evaluator_button")
        self.training_evaluator_button.setToolTip("Open the unified Ultralytics training evaluator, metrics, and resume window.")
    else:
        self.training_evaluator_button.setText("Open Trainer")
        self.training_evaluator_button.setToolTip("Open the unified Ultralytics training evaluator, metrics, and resume window.")

    names = (
        "model_input",
        "pt_label",
        "data_input",
        "yaml_label",
        "task_combobox",
        "imgsz_input",
        "batch_input",
        "epochs_input",
        "save_runs_directory",
        "training_evaluator_button",
        "ultralytics",
    )
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)

    for legacy_name in (
        "model_input",
        "pt_label",
        "data_input",
        "yaml_label",
        "task_combobox",
        "imgsz_input",
        "batch_input",
        "epochs_input",
        "save_runs_directory",
        "ultralytics",
    ):
        widgets[legacy_name].setEnabled(False)
        widgets[legacy_name].hide()

    action_row = self._row_layout()
    widgets["training_evaluator_button"].setEnabled(True)
    widgets["training_evaluator_button"].show()
    action_row.addWidget(widgets["training_evaluator_button"], 1)

    layout.addLayout(action_row, 0, 0)

    self._ultralytics_train_panel_compacted = True


def _compact_ultralytics_export_panel(self):
    if getattr(self, "_ultralytics_export_panel_compacted", False):
        return

    group = getattr(self, "groupBox_8", None)
    layout = self._prepare_group_grid(group)
    if layout is None:
        return

    names = (
        "Load_pt_model",
        "label_50",
        "convert_model",
        "label_55",
        "imgsz_input_W",
        "imgsz_input_H",
        "label_75",
        "label_73",
        "batch_size_checkbox",
        "batch_input_spinbox",
        "label_2",
        "conf_spinbox",
        "label_3",
        "iou_spinbox",
        "half_true",
        "use_gpu_checkbox",
        "int8_true",
        "dynamic_true",
        "simplify",
        "nms_true_checkbox",
        "agnostic_checkbox",
        "convertButton",
    )
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)

    for frame_name in ("frame_9", "frame_6"):
        frame = getattr(self, frame_name, None)
        if isinstance(frame, QtWidgets.QFrame):
            frame.hide()

    widgets["label_50"].setVisible(False)
    widgets["label_55"].setVisible(False)
    widgets["convert_model"].setMaximumWidth(150)
    widgets["label_2"].setText("Conf")
    widgets["label_3"].setText("IoU")
    widgets["dynamic_true"].setText("Dynamic")
    widgets["agnostic_checkbox"].setText("Agnostic")
    widgets["imgsz_input_W"].setToolTip("Export width. Ultralytics export receives imgsz as [H, W].")
    widgets["imgsz_input_H"].setToolTip("Export height. Ultralytics export receives imgsz as [H, W].")
    for field_name in ("imgsz_input_W", "imgsz_input_H"):
        widgets[field_name].setFixedWidth(76)
    widgets["batch_input_spinbox"].setFixedWidth(76)
    widgets["conf_spinbox"].setFixedWidth(82)
    widgets["iou_spinbox"].setFixedWidth(82)

    model_row = self._row_layout()
    model_row.addWidget(widgets["Load_pt_model"], 1)
    model_row.addWidget(widgets["convert_model"], 1)

    export_action_row = self._row_layout()
    export_action_row.addWidget(widgets["convertButton"], 1)

    size_row = self._row_layout()
    size_row.addWidget(self._workflow_label("Export W", group, 62))
    size_row.addWidget(widgets["imgsz_input_W"])
    size_row.addWidget(self._workflow_label("Export H", group, 62))
    size_row.addWidget(widgets["imgsz_input_H"])
    size_row.addStretch(1)

    batch_row = self._row_layout()
    batch_row.addWidget(widgets["batch_size_checkbox"])
    batch_row.addWidget(widgets["batch_input_spinbox"])
    batch_row.addStretch(1)

    threshold_row = self._row_layout()
    threshold_row.addWidget(widgets["label_2"])
    threshold_row.addWidget(widgets["conf_spinbox"])
    threshold_row.addWidget(widgets["label_3"])
    threshold_row.addWidget(widgets["iou_spinbox"])
    threshold_row.addStretch(1)

    toggles_grid = QtWidgets.QGridLayout()
    toggles_grid.setContentsMargins(0, 0, 0, 0)
    toggles_grid.setHorizontalSpacing(10)
    toggles_grid.setVerticalSpacing(4)
    for index, name in enumerate(
        ("half_true", "use_gpu_checkbox", "int8_true", "dynamic_true", "simplify", "nms_true_checkbox", "agnostic_checkbox")
    ):
        toggles_grid.addWidget(widgets[name], index // 4, index % 4)
        toggles_grid.setColumnStretch(index % 4, 1)

    layout.addLayout(model_row, 0, 0)
    layout.addLayout(export_action_row, 1, 0)
    layout.addLayout(size_row, 2, 0)
    layout.addLayout(batch_row, 3, 0)
    layout.addLayout(threshold_row, 4, 0)
    layout.addLayout(toggles_grid, 5, 0)

    self._ultralytics_export_panel_compacted = True


def _compact_train_params_panel(self):
    if getattr(self, "_train_params_panel_compacted", False):
        return

    group = getattr(self, "groupBox_9", None)
    layout = self._prepare_group_grid(group)
    if layout is None:
        return

    names = (
        "resume_checkbox",
        "amp_true",
        "tensorboardCheckbox",
        "keep_duplicate_labels_checkbox",
        "patience_checkbox",
        "patience_input",
        "freeze_checkbox",
        "freeze_input",
    )
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    if not isinstance(getattr(self, "rect_train_checkbox", None), QtWidgets.QCheckBox):
        self.rect_train_checkbox = QtWidgets.QCheckBox("Rect", group)
        self.rect_train_checkbox.setObjectName("rect_train_checkbox")
        self.rect_train_checkbox.setToolTip("Use Ultralytics rect=True rectangular training batches when image aspect ratios waste a lot of padding.")
    widgets["rect_train_checkbox"] = self.rect_train_checkbox

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)
        widget.setEnabled(False)
        widget.hide()

    group.setEnabled(False)
    group.hide()

    self._train_params_panel_compacted = True


def _selected_training_weights_path(self):
    for path in (
        getattr(self, "pt_path", None),
        getattr(self, "pretrained_model_path", None),
        getattr(self, "export_model_path", None),
        getattr(self, "model_config_path", None),
    ):
        if path and os.path.exists(path):
            return self.normalize_path(path)
    return ""


def _effective_eval_weights_path(self, use_training_weights=True, eval_path=""):
    if use_training_weights:
        return self._selected_training_weights_path()
    eval_path = self.normalize_path(eval_path)
    return eval_path if eval_path and os.path.exists(eval_path) else ""

