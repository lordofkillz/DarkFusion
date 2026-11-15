"""Training widgets, run history, and existing monitor/preview callbacks."""

import csv
import cv2
import glob
import hashlib
import json
import logging
import math
import numpy as np
import os
import psutil
import re
import shlex
import time
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QPointF, QRect, QRectF, QTimer, Qt
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QTextCursor,
)
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsRectItem,
    QGraphicsScene, QGraphicsTextItem, QMessageBox, QTableWidgetItem, QWidget,
)
from darkfusion_annotation_items import KeypointDrawer
from darkfusion_annotation_style import get_color
from darkfusion_annotations import BoundingBox, DEFAULT_POSE_SKELETON
from darkfusion_environment import APP_DIR
from darkfusion_image_viewers import TrainingArtifactViewer
from darkfusion_training_size import parse_training_batch
from datetime import datetime
from pathlib import Path


try:
    import sip
except ModuleNotFoundError:
    from PyQt5 import sip


logger = logging.getLogger("UltraDarkFusionLogger")


class TrainingParameterSlider(QWidget):
    """Training amounts outside percentage ranges, with a read-only value label."""

    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, minimum, maximum, step, parent=None):
        super().__init__(parent)
        self._minimum, self._maximum = minimum, maximum
        self._step = step
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(0, round((maximum - minimum) / step))
        self.value_label = QtWidgets.QLabel()
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setMinimumWidth(105)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(lambda _: self.valueChanged.emit(self.value()))

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def value(self):
        return round(self._minimum + self.slider.value() * self._step, 10)

    def setValue(self, value):
        self.slider.setValue(round((value - self._minimum) / self._step))

    def setToolTip(self, text):
        super().setToolTip(text)
        self.slider.setToolTip(text)
        self.value_label.setToolTip(text)


class TrainingHyperparametersWidget(QWidget):
    """Edit the selected training cfg; constructing/browsing never saves it."""

    saveRequested = QtCore.pyqtSignal(dict)
    reloadRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        from darkfusion_tune_settings import training_hyperparameter_groups
        from ultralytics.utils import DEFAULT_CFG_DICT
        self.setObjectName("training_hyperparameters")
        self.setMinimumWidth(0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self._outer_viewport = None
        self._easy_column_count = 0
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._update_responsive_layout)
        self._supported = {key for task in ('detect', 'pose', 'segment', 'obb', 'classify')
                           for keys in training_hyperparameter_groups(task).values() for key in keys}
        self._values = {}
        self._visible = set()
        self._initial = {}
        self._loading = False
        self._available = False
        self.source_values = {}
        self._defaults = DEFAULT_CFG_DICT
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.source_label = QtWidgets.QLabel("Generate Files + Parameters or select a training configuration YAML.")
        self.source_label.setWordWrap(True)
        self.source_label.setMinimumWidth(0)
        self.source_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        layout.addWidget(self.source_label)
        self.easy_page = QtWidgets.QScrollArea()
        self.easy_page.setWidgetResizable(True)
        self.easy_page.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.easy_page.setMinimumHeight(170)
        layout.addWidget(self.easy_page)
        self._build_easy_controls()
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        row = QtWidgets.QHBoxLayout()
        self.reload_button = QtWidgets.QPushButton("Reload YAML")
        self.reload_button.setObjectName("hyperparameters_reload")
        self.save_button = QtWidgets.QPushButton("Save to YAML")
        self.save_button.setObjectName("hyperparameters_save")
        row.addWidget(self.reload_button)
        row.addStretch(1)
        row.addWidget(self.save_button)
        layout.addLayout(row)
        self.reload_button.clicked.connect(self.reloadRequested)
        self.save_button.clicked.connect(self._request_save)
        self.set_unavailable()

    def set_unavailable(self, message="Generate Files + Parameters or select a training configuration YAML."):
        self._available = False
        self.source_label.setText(message)
        self.easy_page.setEnabled(False)
        self.save_button.setEnabled(False)
        self.note.setText("")

    def set_values(self, values, path, task="detect", enabled=True):
        from darkfusion_tune_settings import training_hyperparameter_groups
        self._loading = True
        self._available = bool(enabled)
        self.source_values = dict(values)
        self.path = str(path)
        self.task = task
        visible = {key for keys in training_hyperparameter_groups(task).values() for key in keys}
        self._visible = visible
        self._values = {}
        for key in self._supported:
            value = values.get(key, self._defaults[key])
            if key == "amp":
                value = {"fp16": True, "fp32": False}.get(str(value).lower(), value)
            if key == "quantize" and str(value).lower() in {"8", "int8", "w8a8"}:
                value = 8
            self._values[key] = value
        self._initial = self.values()
        self._loading = False
        self.augmentation_preset.setCurrentIndex(0)
        self.apply_preset_button.setEnabled(False)
        self.preset_note.setText(
            "Less image mixing reduces augmentation work. More variation may cost time; "
            "speed and accuracy still depend on your data and hardware.")
        self._sync_easy_controls()
        self.easy_page.setEnabled(enabled)
        self.source_label.setText("Configuration: " + os.path.basename(str(path)))
        self.source_label.setToolTip(str(path))
        self._edited()
        self._layout_timer.start(0)

    def values(self):
        return {key: value for key, value in self._values.items() if key in self._visible}

    def changes(self):
        return {key: value for key, value in self.values().items() if value != self._initial.get(key)}

    def is_dirty(self):
        return bool(self.changes())

    def _edited(self, *_):
        if self._loading:
            return
        count = len(self.changes())
        self._sync_easy_controls()
        self.note.setText(f"{count} changed value(s). Save to overwrite the selected training YAML."
                          if count else "Showing current values. Unset options use Ultralytics defaults.")
        if self._values.get('optimizer') == 'auto':
            self.note.setText(self.note.text() + " Automatic selects the optimizer and learning rate when training starts.")
        self.save_button.setEnabled(self._available and bool(count))

    def _request_save(self):
        self.saveRequested.emit(self.changes())

    def showEvent(self, event):
        super().showEvent(event)
        ancestor = self.parentWidget()
        while ancestor is not None:
            if isinstance(ancestor, QtWidgets.QScrollArea):
                viewport = ancestor.viewport()
                if viewport is not self._outer_viewport:
                    if self._outer_viewport is not None and not sip.isdeleted(self._outer_viewport):
                        self._outer_viewport.removeEventFilter(self)
                    self._outer_viewport = viewport
                    viewport.installEventFilter(self)
                break
            ancestor = ancestor.parentWidget()
        self._layout_timer.start(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_layout_timer'):
            self._layout_timer.start(0)

    def eventFilter(self, source, event):
        if source is self._outer_viewport and event.type() == QtCore.QEvent.Resize:
            self._layout_timer.start(0)
        return super().eventFilter(source, event)

    def _update_responsive_layout(self):
        if sip.isdeleted(self) or not hasattr(self, '_easy_columns'):
            return
        viewport = self._outer_viewport
        if viewport is not None and not sip.isdeleted(viewport):
            # Other trainer controls may have a wider minimum size. Fit this
            # editor to the visible viewport rather than that hidden overflow.
            width = max(1, viewport.width() - 32)
            if self.maximumWidth() != width:
                self.setMaximumWidth(width)
            height = viewport.height()
        else:
            width = self.width()
            height = self.window().height()
        columns = 3 if width >= 1400 else 2 if width >= 900 else 1
        if columns != self._easy_column_count:
            arrangements = {
                1: (('Learning', 'Training', 'Appearance', 'Image mixing', 'Task settings', 'Run controls'),),
                2: (('Learning', 'Training', 'Run controls'), ('Appearance', 'Image mixing', 'Task settings')),
                3: (('Learning', 'Training'), ('Appearance', 'Run controls'), ('Image mixing', 'Task settings')),
            }
            for column in self._easy_columns:
                column_layout = column.layout()
                while column_layout.count():
                    column_layout.takeAt(0)
            for index, column in enumerate(self._easy_columns):
                column.setVisible(index < columns)
                if index < columns:
                    for name in arrangements[columns][index]:
                        column.layout().addWidget(self._easy_sections[name])
                    column.layout().addStretch(1)
            self._easy_column_count = columns
        self._easy_body.layout().activate()
        body_layout = self._easy_body.layout()
        content_height = body_layout.totalHeightForWidth(max(1, width - 12))
        if content_height < 0:
            content_height = body_layout.sizeHint().height()
        limit = max(220, min(760, height - 160))
        page_height = min(limit, max(220, content_height + 12))
        if self.easy_page.height() != page_height:
            self.easy_page.setFixedHeight(page_height)

    def _build_easy_controls(self):
        from darkfusion_ui_controls import PercentSliderControl, CollapsibleSettingsGroup
        from darkfusion_tune_settings import TRAINING_HYPERPARAMETER_GROUPS
        body = QWidget()
        self._easy_body = body
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(6, 6, 6, 6)
        intro = QtWidgets.QLabel(
            "Use Automatic, or adjust the sliders. Save to YAML when ready.")
        intro.setWordWrap(True)
        intro.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        layout.addWidget(intro)
        preset_row = QtWidgets.QHBoxLayout()
        self.augmentation_preset = QtWidgets.QComboBox()
        self.augmentation_preset.setObjectName("hyperparameter_augmentation_preset")
        self.augmentation_preset.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.augmentation_preset.setMinimumContentsLength(12)
        for text, value in [("Choose an augmentation preset…", None),
                            ("Lighter workload — no image mixing", "lighter"),
                            ("Ultralytics augmentation defaults", "defaults"),
                            ("More image mixing and size variation", "mixing")]:
            self.augmentation_preset.addItem(text, value)
        self.apply_preset_button = QtWidgets.QPushButton("Use preset")
        self.apply_preset_button.setObjectName("hyperparameter_apply_preset")
        self.apply_preset_button.setEnabled(False)
        self.augmentation_preset.currentIndexChanged.connect(
            lambda _: self.apply_preset_button.setEnabled(self.augmentation_preset.currentData() is not None))
        self.apply_preset_button.clicked.connect(self._apply_augmentation_preset)
        preset_row.addWidget(self.augmentation_preset, 1)
        preset_row.addWidget(self.apply_preset_button)
        self.preset_note = QtWidgets.QLabel(
            "Less image mixing reduces augmentation work. More variation may cost time; "
            "speed and accuracy still depend on your data and hardware.")
        self.preset_note.setWordWrap(True)
        self.preset_note.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.easy_controls = {}
        self.easy_rows = {}
        self._syncing_easy = False
        self._easy_units = {}
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(12)
        layout.addLayout(columns)
        self._easy_columns = []
        for _ in range(3):
            column = QWidget()
            column.setMinimumWidth(0)
            column.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
            column_layout = QtWidgets.QVBoxLayout(column)
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.setSpacing(12)
            columns.addWidget(column, 1)
            self._easy_columns.append(column)
        self._easy_sections = {}
        forms = {}
        for name in ('Learning', 'Training', 'Appearance', 'Image mixing', 'Task settings', 'Run controls'):
            section = CollapsibleSettingsGroup(name, body)
            section.setObjectName('hyperparameter_section_' + name.lower().replace(' ', '_'))
            section.setMinimumWidth(0)
            section.expandedChanged.connect(lambda _: self._layout_timer.start(0))
            form = QtWidgets.QFormLayout(section.content)
            form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            self._easy_sections[name] = section
            forms[name] = form

        def add(key, title, control, help_text, ends=None):
            label = QtWidgets.QLabel(title)
            label.setWordWrap(True)
            label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
            panel = QWidget()
            row = QtWidgets.QVBoxLayout(panel)
            row.setContentsMargins(0, 0, 0, 6)
            row.setSpacing(2)
            row.addWidget(control)
            if ends:
                captions = QtWidgets.QHBoxLayout()
                for index, text in enumerate(ends):
                    caption = QtWidgets.QLabel(text)
                    caption.setWordWrap(True)
                    caption.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
                    caption.setAlignment((Qt.AlignLeft if index == 0 else Qt.AlignRight) | Qt.AlignVCenter)
                    captions.addWidget(caption, 1)
                row.addLayout(captions)
            label.setToolTip(help_text)
            control.setToolTip(help_text)
            control.setObjectName("easy_hyperparameter_" + key)
            self.easy_controls[key] = control
            self.easy_rows[key] = (label, panel)
            section = ('Learning' if key in TRAINING_HYPERPARAMETER_GROUPS['Learning'] else
                       'Training' if key in TRAINING_HYPERPARAMETER_GROUPS['Precision and performance'] else
                       'Run controls' if key in TRAINING_HYPERPARAMETER_GROUPS['Run controls'] else
                       'Task settings' if key in TRAINING_HYPERPARAMETER_GROUPS['Loss and task'] else
                       'Image mixing' if key in {'mosaic', 'close_mosaic', 'mixup', 'cutmix', 'multi_scale',
                                                'copy_paste', 'copy_paste_mode'} else 'Appearance')
            forms[section].addRow(label)
            forms[section].addRow(panel)

        for key, title, options, help_text in [
            ("optimizer", "Learning mode",
             [("Automatic — choose for me", "auto"), ("Adaptive learning (AdamW)", "AdamW"),
              ("Standard learning (SGD)", "SGD"), ("Mixed learning (MuSGD)", "MuSGD"),
              *[("Adaptive learning (" + name + ")", name)
                for name in ("Adam", "Adamax", "NAdam", "RAdam", "RMSProp")]],
             "Automatic selects the optimizer and learning rate when training starts. "
             "Current manual settings are kept unless you choose Automatic."),
            ("amp", "Training precision",
             [("FP16 AMP — lower memory, usual GPU choice", True),
              ("BF16 AMP — wider numeric range", "bf16"), ("FP32 — uses more memory", False)],
             "Controls GPU training precision. Hardware and model determine which is fastest."),
            ("quantize", "Prepare for INT8 export (QAT)",
             [("Off — normal training", None), ("On — INT8-aware training", 8)],
             "Fine-tunes the model to tolerate INT8 export. Training still uses the precision above; "
             "this does not make training run in INT8. Requires NVIDIA ModelOpt. "
             "Turn Training compilation off before saving."),
            ("compile", "Training compilation",
             [("Off — start without compilation", False), ("On — standard optimization", True),
              ("On — reduce launch overhead", "reduce-overhead"),
              ("On — extended optimization", "max-autotune-no-cudagraphs")],
             "Compilation adds startup work and can use extra memory. Its speed benefit depends on your model. "
             "Keep it off for INT8 preparation."),
            ("channels_last", "GPU memory layout",
             [("Automatic — use the platform default", None), ("Channels last", True),
              ("Standard layout", False)],
             "Use Automatic unless you have measured a benefit from a different CUDA memory layout."),
            ("cos_lr", "Learning schedule",
             [("Normal taper", False), ("Smooth cosine taper", True)],
             "Controls how the learning rate decreases over the run. Final learning pace sets its ending level."),
            ("nbs", "Reference batch for learning scaling",
             [("Smaller reference (16)", 16), ("Small reference (32)", 32), ("Standard reference (64)", 64),
              ("Larger reference (128)", 128), ("Large reference (256)", 256)],
             "Reference batch used to scale accumulation and weight decay. This does not set your actual GPU batch size."),
            ("deterministic", "Repeatability or speed",
             [("Repeatable runs", True), ("Allow faster algorithm choices", False)],
             "Allowing non-deterministic algorithms may run faster, but repeated runs can differ."),
            ("mask_ratio", "Training mask detail",
             [("Most detail — more memory", 1), ("More detail", 2),
              ("Balanced detail and memory", 4), ("Less detail — less memory", 8)],
             "Controls training mask downsampling. Smaller values retain more mask detail and use more memory."),
            ("overlap_mask", "Overlapping segmentation masks",
             [("Merge overlapping masks", True), ("Keep separate masks", False)],
             "Choose how overlapping instance masks are represented during segmentation training."),
            ("copy_paste_mode", "Object copy/paste method",
             [("Mirror objects from the same image", "flip"), ("Mix objects from another image", "mixup")],
             "Chooses the native copy/paste strategy. Copy labeled objects controls how much it is used."),
            ("auto_augment", "Classification appearance policy",
             [("Default random policy", "randaugment"), ("Learned policy", "autoaugment"),
              ("Mixed policy", "augmix"), ("Off — use color variation sliders", None)],
             "Classification appearance augmentation. The color variation sliders apply when this policy is off."),
            ("close_mosaic", "Finish training without four-image mixing",
             [("Keep mixing through the end", 0), ("Stop in the last 5 epochs", 5),
              ("Stop in the last 10 epochs", 10), ("Stop in the last 20 epochs", 20),
              ("Stop in the last 30 epochs", 30)],
             "Stops mosaic augmentation near the end of training to stabilize the final learning steps."),
            ("seed", "Repeatable random pattern",
             [("Default pattern", 0), ("Alternative pattern A", 1), ("Alternative pattern B", 42),
              ("Alternative pattern C", 123), ("Alternative pattern D", 1234)],
             "Random seed for initialization and data variation. Reusing a pattern helps compare training runs."),
            ("save_period", "Extra checkpoint frequency",
             [("Only standard best/last checkpoints", -1), ("Every epoch", 1), ("Every 5 epochs", 5),
              ("Every 10 epochs", 10), ("Every 25 epochs", 25), ("Every 50 epochs", 50)],
             "Optional periodic checkpoints in addition to the normal best and last model files."),
            ("time", "Training time limit",
             [("No limit — use the epoch count", None), ("15 minutes", .25), ("30 minutes", .5),
              ("1 hour", 1.), ("2 hours", 2.), ("4 hours", 4.), ("8 hours", 8.),
              ("12 hours", 12.), ("24 hours", 24.)],
             "Optional training time budget. When enabled, this takes precedence over the requested epoch count."),
        ]:
            if key not in self._supported:
                continue
            control = QtWidgets.QComboBox()
            control.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            control.setMinimumContentsLength(18)
            for text, value in options:
                control.addItem(text, value)
            control.currentIndexChanged.connect(lambda _, k=key, c=control: self._easy_changed(k, c.currentData()))
            add(key, title, control, help_text)

        self.learning_pace = QtWidgets.QSlider(Qt.Horizontal)
        self.learning_pace.setObjectName("easy_learning_pace_slider")
        self.learning_pace.setRange(0, 100)
        self.learning_pace.setPageStep(10)
        self.learning_pace.valueChanged.connect(self._learning_pace_changed)
        self.learning_pace_value = QtWidgets.QLabel()
        pace_panel = QWidget()
        pace_layout = QtWidgets.QHBoxLayout(pace_panel)
        pace_layout.setContentsMargins(0, 0, 0, 0)
        pace_layout.addWidget(self.learning_pace, 1)
        pace_layout.addWidget(self.learning_pace_value)
        add("lr0", "Learning pace", pace_panel,
            "Gentle learning takes smaller steps. Faster learning takes larger steps and can overshoot. "
            "This changes learning rate, not how fast each batch runs. Moving this slider from Automatic "
            "selects AdamW so your learning rate is actually used.", ("Gentle / slower learning", "Larger / faster steps"))
        self.easy_controls['lr0'] = self.learning_pace
        pace_note = QtWidgets.QLabel(
            "Keep Automatic if unsure. Moving Learning pace or Momentum uses manual learning settings. "
            "You can return to Automatic at any time.")
        pace_note.setWordWrap(True)
        pace_note.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        forms['Image mixing'].addRow(preset_row)
        forms['Image mixing'].addRow(self.preset_note)

        sliders = [
            ("momentum", "Momentum — how smoothly learning changes", "Lower momentum reacts more quickly to new batches. Higher momentum keeps more of the previous direction, smoothing updates. Moving this slider from Automatic selects AdamW with a 0.001 starting learning rate.", "momentum", ("React quickly", "Smoother / slower to change")),
            ("lrf", "Final learning pace", "Ending learning rate as a share of the starting rate.", "strength", ("Finish gently", "Keep initial pace")),
            ("warmup_momentum", "Momentum at the start", "Momentum during the initial warmup period.", "strength", ("React quickly", "Smoother start")),
            ("warmup_bias_lr", "Bias learning pace at the start", "Initial bias learning rate during warmup.", "amount", ("Gentler", "Larger steps")),
            ("hsv_h", "Color hue variation", "Strength of random hue changes.", "strength", ("Off", "Stronger")),
            ("hsv_v", "Brightness variation", "Strength of random lighting changes.", "strength", ("Off", "Stronger")),
            ("hsv_s", "Color saturation variation", "Strength of random changes to color saturation.", "strength", ("Off", "Stronger")),
            ("scale", "Object size variation", "Random zoom range. A custom min/max range remains unchanged until you move this slider.", "size", ("No zoom", "Wider zoom range")),
            ("translate", "Object position variation", "How far objects can shift within the training image.", "shift", ("No shift", "Larger shifts")),
            ("fliplr", "Horizontal flips", "How often training images are mirrored left to right.", "images", ("Never", "Every image")),
            ("flipud", "Vertical flips", "How often training images are flipped upside down. Usually off for upright gameplay.", "images", ("Never", "Every image")),
            ("bgr", "Swap red and blue channels", "How often training swaps RGB and BGR color channels.", "images", ("Never", "Every image")),
            ("mosaic", "Combine four images (mosaic)", "How often four images are combined. This can make already small objects smaller.", "images", ("Never", "Every image")),
            ("mixup", "Blend two images (MixUp)", "How often two images are blended. More mixing adds preparation work.", "images", ("Never", "Every image")),
            ("cutmix", "Mix image patches (CutMix)", "How often patches from different images are combined.", "images", ("Never", "Every image")),
            ("multi_scale", "Vary training resolution", "Random size range around ImgSz. Larger inputs may need more GPU memory.", "size", ("Fixed resolution", "Wider size range")),
            ("copy_paste", "Copy labeled objects", "Amount of object copy/paste during segmentation or OBB training. Its interpretation depends on the selected copy/paste strategy.", "%", ("Off", "More copy/paste")),
            ("fraction", "Training data to use", "Use all data for normal training, or a smaller fraction for a quick test. Existing image counts or split lists are retained until you move the slider.", "data", ("Quick subset", "All data")),
            ("cls_pw", "Balance rare classes", "Strength of inverse-frequency class weighting. Off uses the normal unweighted class loss.", "strength", ("Off", "More balancing")),
            ("dropout", "Classification dropout", "Amount of training features randomly dropped to discourage overfitting.", "strength", ("Off", "More dropout")),
            ("erasing", "Hide random image patches", "Probability of random erasing during classification training.", "images", ("Never", "Every image")),
        ]
        for key, title, help_text, unit, ends in sliders:
            if key not in self._supported:
                continue
            control = PercentSliderControl(minimum=.01 if key == "fraction" else 0,
                maximum=.99 if key == "momentum" else .5 if key == "multi_scale" else 1)
            control.value_label.setMinimumWidth(105)
            control.valueChanged.connect(lambda value, k=key: self._easy_changed(k, value))
            self._easy_units[key] = unit
            add(key, title, control, help_text, ends)
        for key, title, maximum, step, unit, help_text, ends in [
            ('weight_decay', 'Resistance to overfitting', .01, .00001, 'amount',
             'Weight decay penalizes large weights. Stronger values can reduce overfitting but can also prevent fitting the data.',
             ('Off', 'Stronger regularization')),
            ('warmup_epochs', 'Time to ease into learning', 20, .1, 'epochs',
             'Number of initial epochs used to ramp up learning. Existing longer or fractional warmups are preserved.',
             ('No warmup', 'Longer warmup')),
            ('degrees', 'Image rotation', 180, 1, 'degrees',
             'Maximum random rotation in either direction.', ('No rotation', 'More rotation')),
            ('shear', 'Image slant', 45, .1, 'degrees',
             'Maximum random shear angle in either direction.', ('No slant', 'More slant')),
            ('perspective', 'Perspective distortion', .001, .000001, 'amount',
             'Strength of perspective warping. Small native values can already have a visible effect.',
             ('Off', 'Stronger distortion')),
            *[(key, title, maximum, .05, 'weight', help_text, ('Less emphasis', 'More emphasis'))
              for key, title, maximum, help_text in [
                  ('box', 'Focus on box position and size', 30, 'Weight given to box localization loss.'),
                  ('cls', 'Focus on correct classes', 5, 'Weight given to class prediction loss.'),
                  ('dfl', 'Focus on box edge precision', 10, 'Weight given to the model\'s box-distance loss.'),
                  ('pose', 'Focus on keypoint placement', 30, 'Weight given to keypoint position loss.'),
                  ('kobj', 'Focus on keypoint presence', 10, 'Weight given to keypoint objectness loss.'),
                  ('rle', 'Focus on keypoint uncertainty', 10, 'Weight given to residual log-likelihood pose loss.'),
                  ('angle', 'Focus on rotated-box angles', 10, 'Weight given to oriented bounding-box angle loss.'),
              ]],
        ]:
            if key not in self._supported:
                continue
            control = TrainingParameterSlider(0, maximum, step)
            control.valueChanged.connect(lambda value, k=key: self._easy_changed(k, value))
            self._easy_units[key] = unit
            add(key, title, control, help_text, ends)
        forms['Learning'].addRow(pace_note)
        self.easy_page.setWidget(body)

    def _set_parameter_value(self, key, value):
        self._values[key] = value
        if not self._loading:
            self._edited()

    def _easy_changed(self, key, value):
        if self._loading or self._syncing_easy or not self._available:
            return
        if key == 'momentum' and self._values.get('optimizer') == 'auto':
            self._loading = True
            try:
                self._set_parameter_value('optimizer', 'AdamW')
                self._set_parameter_value('lr0', .001)
                self._set_parameter_value(key, value)
            finally:
                self._loading = False
            self._edited()
            return
        self._set_parameter_value(key, value)

    def _learning_rate_limits(self):
        optimizer = str(self._values.get('optimizer', 'auto')).lower()
        return (1e-5, .003) if optimizer == 'auto' or 'adam' in optimizer else (.0001, .03)

    def _learning_pace_changed(self, position):
        if self._loading or self._syncing_easy or not self._available:
            return
        low, high = self._learning_rate_limits()
        rate = low * (high / low) ** (position / 100)
        self._loading = True
        try:
            if self._values.get('optimizer') == 'auto':
                self._set_parameter_value('optimizer', 'AdamW')
            self._set_parameter_value('lr0', round(rate, 8))
        finally:
            self._loading = False
        self._edited()

    def _sync_easy_controls(self):
        if not hasattr(self, "easy_controls"):
            return
        self._syncing_easy = True
        try:
            values = self.values()
            for key, control in self.easy_controls.items():
                visible = key in values
                for widget in self.easy_rows[key]:
                    widget.setVisible(visible)
                if not visible:
                    continue
                active = not (self.task == 'classify' and key in {'hsv_h', 'hsv_v', 'hsv_s'}
                              and values.get('auto_augment') is not None)
                for widget in self.easy_rows[key]:
                    widget.setEnabled(active)
                value = values[key]
                blocker = QtCore.QSignalBlocker(control)
                if key == 'lr0':
                    low, high = self._learning_rate_limits()
                    automatic = values.get('optimizer') == 'auto'
                    rate = .001 if automatic else value
                    position = round(100 * math.log(max(low, rate) / low) / math.log(high / low))
                    self.learning_pace.setValue(max(0, min(100, position)))
                    level = "Gentle" if position < 35 else "Moderate" if position < 70 else "Larger steps"
                    self.learning_pace_value.setText('Automatic' if automatic else f'{level} ({value:g})')
                elif isinstance(control, QtWidgets.QComboBox):
                    index = next((i for i in range(control.count())
                                  if type(control.itemData(i)) is type(value) and control.itemData(i) == value), -1)
                    if index < 0:
                        control.addItem(f"Current custom setting: {value}", value)
                        index = control.count() - 1
                    control.setCurrentIndex(index)
                elif isinstance(value, (int, float)) and not isinstance(value, bool) and control.minimum() <= value <= control.maximum():
                    control.setValue(value)
                    percent = f"{value * 100:.6g}%"
                    unit = self._easy_units[key]
                    default = self._defaults[key]
                    text = (f'{value:g} epochs' if unit == 'epochs' else
                            f'±{value:g}°' if unit == 'degrees' else
                            f'{"Standard" if value == default else "Lower" if value < default else "Higher"} ({value:g})'
                            if unit == 'weight' else
                            f'{value:g}' if unit == 'amount' else
                            "Automatic" if key == 'momentum' and values.get('optimizer') == 'auto' else
                            f"{percent} carry-over" if unit == 'momentum' else
                            f"±{percent} size" if unit == "size" else
                            f"{percent} images" if unit == "images" else
                            "All data" if unit == "data" and value == 1 else
                            f"{percent} data" if unit == "data" else
                            f"{percent} shift" if unit == "shift" else percent)
                    control.value_label.setText("Off" if value == 0 else text)
                else:
                    control.setValue(control.minimum())
                    text = (f"{value} images" if key == "fraction" and isinstance(value, (int, float))
                            else f"Custom: {value}")
                    control.value_label.setText(text)
                del blocker
            for section in self._easy_sections.values():
                section.setVisible(any(label.parentWidget() is section.content and key in values
                                       for key, (label, _) in self.easy_rows.items()))
        finally:
            self._syncing_easy = False

    def _apply_augmentation_preset(self):
        from darkfusion_tune_settings import training_augmentation_preset
        if not self._available or self.augmentation_preset.currentData() is None:
            return
        values = training_augmentation_preset(self.augmentation_preset.currentData(), self.task)
        self._loading = True
        try:
            for key, value in values.items():
                self._set_parameter_value(key, value)
        finally:
            self._loading = False
        self._edited()
        self.preset_note.setText("Preset staged: " + self.augmentation_preset.currentText()
                                 + ". Save to YAML to apply it to training.")


class TrainingMetricsPlot(QWidget):
    def __init__(self, title="", parent=None, value_range=None):
        super().__init__(parent)
        self.title = title
        self.value_range = value_range
        self.series = []
        self.hidden_series = set()
        self._legend_hit_rects = []
        self._hovered_series = ""
        self.setMinimumHeight(170)
        self.setMouseTracking(True)
        self.setToolTip("Click a legend item to show or hide that metric. Right-click the legend to show all.")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_series(self, series, title=None, value_range=None):
        self.series = list(series or [])
        available = {str(item[0]) for item in self.series if item}
        self.hidden_series = {name for name in self.hidden_series if name in available}
        if title is not None:
            self.title = str(title)
        if value_range is not None:
            self.value_range = value_range
        self.update()

    def mouseMoveEvent(self, event):
        hovered = ""
        for name, rect in self._legend_hit_rects:
            if rect.contains(event.pos()):
                hovered = name
                break
        if hovered != self._hovered_series:
            self._hovered_series = hovered
            self.setCursor(Qt.PointingHandCursor if hovered else Qt.ArrowCursor)
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hovered_series:
            self._hovered_series = ""
            self.setCursor(Qt.ArrowCursor)
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        for name, rect in self._legend_hit_rects:
            if not rect.contains(event.pos()):
                continue
            if event.button() == Qt.RightButton:
                self.hidden_series.clear()
            elif event.button() == Qt.LeftButton:
                if name in self.hidden_series:
                    self.hidden_series.remove(name)
                else:
                    self.hidden_series.add(name)
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._legend_hit_rects = []
        rect = self.rect().adjusted(10, 8, -10, -10)
        painter.fillRect(rect, QColor(13, 17, 22))
        painter.setPen(QPen(QColor(52, 61, 71), 1))
        painter.drawRoundedRect(rect, 6, 6)

        title_rect = QRect(rect.left() + 10, rect.top() + 8, rect.width() - 20, 22)
        painter.setPen(QColor(235, 239, 245))
        painter.setFont(QFont(painter.font().family(), 9, QFont.Bold))
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, self.title or "Metrics")

        plot_rect = rect.adjusted(44, 38, -14, -42)
        if plot_rect.width() < 20 or plot_rect.height() < 20:
            return

        painter.setPen(QPen(QColor(45, 52, 61), 1))
        for i in range(5):
            y = plot_rect.top() + (plot_rect.height() * i / 4.0)
            painter.drawLine(plot_rect.left(), int(y), plot_rect.right(), int(y))
        painter.drawRect(plot_rect)

        usable_series = []
        all_values = []
        for name, values, color in self.series:
            numeric = []
            for value in values or []:
                try:
                    numeric.append(float(value))
                except Exception:
                    numeric.append(None)
            clean = [value for value in numeric if value is not None and math.isfinite(value)]
            if clean:
                usable_series.append((str(name), numeric, color))
                if str(name) not in self.hidden_series:
                    all_values.extend(clean)

        if not usable_series:
            painter.setPen(QColor(140, 151, 166))
            painter.setFont(QFont(painter.font().family(), 9))
            painter.drawText(plot_rect, Qt.AlignCenter, "No results.csv data yet")
            return

        visible_series = [
            (name, values, color)
            for name, values, color in usable_series
            if name not in self.hidden_series
        ]

        if not visible_series:
            painter.setPen(QColor(140, 151, 166))
            painter.setFont(QFont(painter.font().family(), 9))
            painter.drawText(plot_rect, Qt.AlignCenter, "All metrics hidden. Click legend items to show them.")
            self._draw_metrics_legend(painter, rect, plot_rect, usable_series)
            return

        if self.value_range:
            y_min, y_max = self.value_range
        else:
            y_min, y_max = min(all_values), max(all_values)
            if math.isclose(y_min, y_max):
                y_min -= 1.0
                y_max += 1.0
            padding = (y_max - y_min) * 0.08
            y_min -= padding
            y_max += padding

        if math.isclose(float(y_min), float(y_max)):
            y_max = y_min + 1.0

        painter.setPen(QColor(140, 151, 166))
        painter.setFont(QFont(painter.font().family(), 8))
        painter.drawText(QRect(rect.left() + 8, plot_rect.top() - 6, 34, 18), Qt.AlignRight | Qt.AlignVCenter, f"{y_max:.3g}")
        painter.drawText(QRect(rect.left() + 8, plot_rect.bottom() - 10, 34, 18), Qt.AlignRight | Qt.AlignVCenter, f"{y_min:.3g}")

        def point_at(index, value, count):
            x = plot_rect.left() if count <= 1 else plot_rect.left() + (plot_rect.width() * index / (count - 1))
            ratio = (float(value) - float(y_min)) / max(1e-9, float(y_max) - float(y_min))
            y = plot_rect.bottom() - (plot_rect.height() * max(0.0, min(1.0, ratio)))
            return QPointF(float(x), float(y))

        for name, values, color in visible_series[:8]:
            qcolor = color if isinstance(color, QColor) else QColor(str(color))
            points = [
                point_at(index, value, len(values))
                for index, value in enumerate(values)
                if value is not None and math.isfinite(float(value))
            ]
            if len(points) >= 2:
                path = QPainterPath(points[0])
                for point in points[1:]:
                    path.lineTo(point)
                painter.setPen(QPen(qcolor, 2))
                painter.drawPath(path)
            elif len(points) == 1:
                painter.setPen(QPen(qcolor, 4))
                painter.drawPoint(points[0])

        self._draw_metrics_legend(painter, rect, plot_rect, usable_series)

    def _draw_metrics_legend(self, painter, rect, plot_rect, series):
        legend_x = plot_rect.left()
        legend_y = plot_rect.bottom() + 10
        max_right = rect.right() - 8
        painter.setFont(QFont(painter.font().family(), 8))
        font_metrics = painter.fontMetrics()

        for name, _values, color in list(series or [])[:8]:
            qcolor = color if isinstance(color, QColor) else QColor(str(color))
            hidden = name in self.hidden_series
            hovered = name == self._hovered_series
            text_width = font_metrics.horizontalAdvance(name)
            item_width = max(94, min(190, text_width + 43))
            if legend_x + item_width > max_right and legend_x > plot_rect.left():
                legend_x = plot_rect.left()
                legend_y += 22

            item_rect = QRect(int(legend_x), int(legend_y - 2), int(item_width), 20)
            self._legend_hit_rects.append((name, item_rect))

            bg = QColor(22, 29, 37)
            if hovered:
                bg = QColor(34, 43, 54)
            if hidden:
                bg = QColor(15, 19, 25)
            painter.setPen(QPen(QColor(58, 69, 82), 1))
            painter.setBrush(QBrush(bg))
            painter.drawRoundedRect(item_rect, 4, 4)

            line_color = QColor(qcolor)
            text_color = QColor(216, 222, 230)
            if hidden:
                line_color.setAlpha(95)
                text_color = QColor(120, 130, 143)
            painter.setPen(QPen(line_color, 2))
            painter.drawLine(item_rect.left() + 9, item_rect.center().y(), item_rect.left() + 27, item_rect.center().y())
            painter.setPen(text_color)
            painter.drawText(QRect(item_rect.left() + 33, item_rect.top(), item_rect.width() - 38, item_rect.height()), Qt.AlignLeft | Qt.AlignVCenter, name)
            if hidden:
                painter.setPen(QPen(QColor(120, 130, 143), 1))
                y = item_rect.center().y()
                painter.drawLine(item_rect.left() + 33, y, item_rect.right() - 7, y)

            legend_x += item_width + 8


class TrainingBatchSpinBox(QtWidgets.QDoubleSpinBox):
    """Fixed image batches or Ultralytics' native automatic memory targets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(6)
        self.setRange(-1, 512)
        self.setSpecialValueText("Auto")
        self.setKeyboardTracking(False)
        self.setValue(-1)

    def value(self):
        return parse_training_batch(super().value())

    def setValue(self, value):
        value = parse_training_batch(value)
        if 0 < value < 1 and not 0 < round(value, self.decimals()) < 1:
            raise ValueError("The batch memory fraction is outside the control's precision.")
        super().setValue(float(value))

    def textFromValue(self, value):
        return str(int(value)) if float(value).is_integer() else f"{value:.6f}".rstrip("0")

    def valueFromText(self, text):
        try:
            return float(parse_training_batch(text))
        except ValueError:
            return super().value()

    def validate(self, text, position):
        if text.strip().lower() in {"", "-", "a", "au", "aut", "0", "0."}:
            return QtGui.QValidator.Intermediate, text, position
        try:
            value = parse_training_batch(text)
            valid = value <= self.maximum() and (not 0 < value < 1 or 0 < round(value, self.decimals()) < 1)
        except ValueError:
            valid = False
        return (QtGui.QValidator.Acceptable if valid else QtGui.QValidator.Invalid), text, position

    def stepBy(self, steps):
        current = super().value()
        target = steps if current < 1 and steps > 0 else int(current) + steps
        self.setValue(min(self.maximum(), target) if target >= 1 else -1)


class TrainingEvaluatorActivityIndicator(QWidget):
    STATE_COLORS = {
        "idle": QColor("#637083"),
        "running": QColor("#42c6ff"),
        "success": QColor("#58d68d"),
        "warning": QColor("#f0d264"),
        "error": QColor("#ff6473"),
    }

    STATE_LABELS = {
        "idle": "Idle",
        "running": "Working",
        "success": "Ready",
        "warning": "Needs attention",
        "error": "Error",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = "idle"
        self._label = self.STATE_LABELS["idle"]
        self._detail = ""
        self._pulse = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self.setMinimumHeight(34)
        self.setMinimumWidth(172)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    def sizeHint(self):
        return QtCore.QSize(190, 34)

    def set_state(self, state="idle", label="", detail=""):
        state = str(state or "idle").lower()
        if state not in self.STATE_COLORS:
            state = "idle"
        self._state = state
        self._label = str(label or self.STATE_LABELS.get(state, "Idle"))
        self._detail = str(detail or "")
        if state == "running":
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()
            self._pulse = 0.0
        self.update()

    def _tick(self):
        self._pulse = (self._pulse + 0.08) % 1.0
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        color = QColor(self.STATE_COLORS.get(self._state, self.STATE_COLORS["idle"]))

        bg = QColor(18, 24, 31)
        border = QColor(color)
        border.setAlpha(150 if self._state != "idle" else 90)
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1))
        painter.drawRoundedRect(rect, 7, 7)

        center = QPointF(rect.left() + 18, rect.center().y())
        if self._state == "running":
            glow_alpha = int(45 + 70 * (0.5 + 0.5 * math.sin(self._pulse * math.tau)))
            glow = QColor(color)
            glow.setAlpha(glow_alpha)
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center, 13, 13)

        ring = QColor(color)
        ring.setAlpha(235)
        painter.setBrush(QBrush(QColor(12, 17, 23)))
        painter.setPen(QPen(ring, 2))
        painter.drawEllipse(center, 7, 7)
        painter.setBrush(QBrush(ring))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, 3, 3)

        text_rect = QRect(rect.left() + 35, rect.top() + 2, rect.width() - 42, rect.height() - 4)
        painter.setPen(QColor(235, 239, 245))
        painter.setFont(QFont(painter.font().family(), 8, QFont.Bold))
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, self._label)


def _training_eval_read_text_tail(self, path, max_chars=160000):
    path = self.normalize_path(path)
    if not path or not os.path.exists(path):
        return ""
    def clean_log_text(text):
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(text))
        return text.replace("\r\n", "\n").replace("\r", "\n")
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_chars:
                f.seek(max(0, size - max_chars))
                data = f.read()
                text = clean_log_text(data.decode("utf-8", errors="replace"))
                return "... log trimmed to the most recent output ...\n" + text
            return clean_log_text(f.read().decode("utf-8", errors="replace"))
    except Exception as e:
        return f"Could not read log: {e}"


def _training_eval_read_results_csv(self, run_dir):
    csv_path = self.normalize_path(os.path.join(self.normalize_path(run_dir), "results.csv"))
    if not os.path.exists(csv_path):
        return [], [], ""
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            headers = [str(header).strip() for header in (reader.fieldnames or []) if str(header).strip()]
            rows = []
            for row in reader:
                clean = {}
                for key, value in (row or {}).items():
                    key = str(key).strip()
                    if key:
                        clean[key] = str(value).strip()
                if clean:
                    rows.append(clean)
        return headers, rows, csv_path
    except Exception as e:
        logger.warning(f"Could not read Ultralytics results CSV from {csv_path}: {e}")
        return [], [], csv_path


def _training_eval_float_value(self, value):
    try:
        parsed = float(str(value).strip())
        return parsed if math.isfinite(parsed) else None
    except Exception:
        return None


def _training_eval_display_metric_headers(self, headers):
    headers = [str(header).strip() for header in headers or [] if str(header).strip()]
    score_headers = [
        header for header in headers
        if header.lower().startswith("metrics/")
        and any(token in header.lower() for token in ("map", "precision", "recall"))
    ]
    loss_headers = [
        header for header in headers
        if "loss" in header.lower()
        and (header.lower().startswith("train/") or header.lower().startswith("val/"))
    ]
    lr_headers = [header for header in headers if header.lower().startswith("lr/")]
    preferred = ["epoch", "time"]
    ordered = []
    for header in preferred + score_headers + loss_headers[:10] + lr_headers[:3]:
        if header in headers and header not in ordered:
            ordered.append(header)
    return ordered or headers[:12]


def _training_eval_format_csv_value(self, value):
    parsed = self._training_eval_float_value(value)
    if parsed is None:
        return str(value)
    if abs(parsed) >= 1000:
        return f"{parsed:.1f}"
    if abs(parsed) >= 10:
        return f"{parsed:.3f}"
    return f"{parsed:.5f}".rstrip("0").rstrip(".")


def _training_eval_update_metrics_table(self, table, headers, rows):
    if table is None or sip.isdeleted(table):
        return
    table.setUpdatesEnabled(False)
    try:
        display_headers = self._training_eval_display_metric_headers(headers)
        display_rows = list(rows or [])[-200:]
        table.clear()
        table.setColumnCount(len(display_headers))
        table.setRowCount(len(display_rows))
        table.setHorizontalHeaderLabels(display_headers)
        for row_index, row in enumerate(display_rows):
            for col_index, header in enumerate(display_headers):
                item = QTableWidgetItem(self._training_eval_format_csv_value(row.get(header, "")))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_index, col_index, item)
        if display_rows:
            table.selectRow(len(display_rows) - 1)
        table.resizeColumnsToContents()
    finally:
        table.setUpdatesEnabled(True)


def _training_eval_metric_series(self, headers, rows, kind):
    if not rows:
        return []
    palette = [
        QColor("#d8a64f"),
        QColor("#53c7f0"),
        QColor("#7bd88f"),
        QColor("#ff6b7a"),
        QColor("#b98cff"),
        QColor("#f0d264"),
        QColor("#4fd1b8"),
        QColor("#ff9f43"),
    ]
    if kind == "score":
        keys = [
            header for header in headers
            if header.lower().startswith("metrics/")
            and any(token in header.lower() for token in ("map", "precision", "recall"))
        ]
    else:
        keys = [
            header for header in headers
            if "loss" in header.lower()
            and (header.lower().startswith("train/") or header.lower().startswith("val/"))
        ]

    series = []
    for index, key in enumerate(keys[:8]):
        values = [self._training_eval_float_value(row.get(key, "")) for row in rows]
        if any(value is not None for value in values):
            label = key.replace("metrics/", "").replace("train/", "tr/").replace("val/", "val/")
            series.append((label, values, palette[index % len(palette)]))
    return series


def _training_eval_latest_metric_line(self, headers, rows):
    if not rows:
        return "Waiting for results.csv..."
    latest = rows[-1]
    pieces = []
    for header in self._training_eval_display_metric_headers(headers)[:8]:
        if header in latest:
            pieces.append(f"{header}={self._training_eval_format_csv_value(latest.get(header, ''))}")
    return "Latest: " + "   ".join(pieces) if pieces else f"Rows: {len(rows)}"


def _training_eval_metric_header(self, headers, required=(), any_of=(), rejected=()):
    for header in headers or []:
        lower = str(header).strip().lower()
        if not lower:
            continue
        if required and not all(token in lower for token in required):
            continue
        if any_of and not any(token in lower for token in any_of):
            continue
        if rejected and any(token in lower for token in rejected):
            continue
        return header
    return ""


def _training_eval_metric_values_for_header(self, rows, header):
    values = []
    if not header:
        return values
    for row in rows or []:
        values.append(self._training_eval_float_value((row or {}).get(header, "")))
    return values


def _training_eval_last_number(self, values):
    for value in reversed(values or []):
        if value is not None:
            return float(value)
    return None


def _training_eval_window_mean(self, values, take_last=True, window=3):
    clean = [float(value) for value in values or [] if value is not None]
    if not clean:
        return None
    window = max(1, min(int(window or 1), len(clean)))
    sample = clean[-window:] if take_last else clean[:window]
    return sum(sample) / len(sample)


def _training_eval_loss_values(self, headers, rows, prefix):
    keys = [
        header for header in headers or []
        if str(header).lower().startswith(prefix)
        and "loss" in str(header).lower()
    ]
    values = []
    for row in rows or []:
        row_values = [
            self._training_eval_float_value((row or {}).get(header, ""))
            for header in keys
        ]
        row_values = [value for value in row_values if value is not None]
        values.append(sum(row_values) / len(row_values) if row_values else None)
    return values


def _training_eval_run_args_from_log(self, log_path):
    log_path = self.normalize_path(log_path)
    head_text = ""
    try:
        if log_path and os.path.exists(log_path):
            with open(log_path, "rb") as f:
                head_text = f.read(60000).decode("utf-8", errors="replace")
    except Exception:
        head_text = ""
    text = head_text + "\n" + self._training_eval_read_text_tail(log_path, max_chars=40000)
    command_text = ""
    for line in text.splitlines():
        if line.startswith("Command:"):
            command_text = line.split("Command:", 1)[1].strip()
    if not command_text:
        return {}
    try:
        tokens = shlex.split(command_text, posix=False)
    except Exception:
        tokens = command_text.split()
    tokens = [str(token).strip().strip('"').strip("'") for token in tokens if str(token).strip()]
    args = {}
    tasks = {"detect", "segment", "classify", "pose", "obb"}
    modes = {"train", "val", "predict", "export", "track", "benchmark", "tune"}
    for index, token in enumerate(tokens[:-1]):
        low = token.lower()
        next_low = tokens[index + 1].lower()
        if low in tasks and next_low in modes:
            args["task"] = low
            args["mode"] = next_low
            break
    for token in tokens:
        if "=" not in token or token.startswith("--"):
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        if key:
            args[key] = value.strip().strip('"').strip("'")
    return args


def _training_eval_run_args_for_compare(self, run_dir):
    args, _args_path = self._training_eval_read_args_yaml(run_dir)
    if isinstance(args, dict) and args:
        return {str(key): value for key, value in args.items()}
    log_path = self._training_eval_log_path(run_dir)
    return self._training_eval_run_args_from_log(log_path)


def _training_eval_resolve_compare_path(self, value, base_dir=""):
    value = str(value or "").strip().strip('"').strip("'")
    if not value:
        return ""
    candidates = []
    if os.path.isabs(value):
        candidates.append(value)
    else:
        if base_dir:
            candidates.append(os.path.join(base_dir, value))
        candidates.append(os.path.join(APP_DIR, value))
        candidates.append(os.path.join(os.getcwd(), value))
    for candidate in candidates:
        candidate = self.normalize_path(candidate)
        if os.path.exists(candidate):
            return os.path.normcase(os.path.abspath(candidate))
    return value.replace("\\", "/").lower()


def _training_eval_compare_arg_match(self, left, right, base_dir="", kind=""):
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left or not right:
        return False
    if kind in {"data", "model"}:
        left_path = self._training_eval_resolve_compare_path(left, base_dir)
        right_path = self._training_eval_resolve_compare_path(right, base_dir)
        if left_path == right_path:
            return True
        return os.path.basename(left_path).lower() == os.path.basename(right_path).lower()
    return left.lower() == right.lower()


def _training_eval_parse_all_metric_from_log(self, log_path):
    text = self._training_eval_read_text_tail(log_path, max_chars=120000)
    matches = re.findall(
        r"(?m)^\s*all\s+(\d+)\s+(\d+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)",
        text,
    )
    if not matches:
        return {}
    images, instances, precision, recall, map50, map5095 = matches[-1]
    return {
        "images": int(images),
        "instances": int(instances),
        "precision": float(precision),
        "recall": float(recall),
        "map50": float(map50),
        "map5095": float(map5095),
    }


def _training_eval_best_metric_snapshot(self, headers, rows):
    if not rows:
        return {}
    map5095_key = self._training_eval_metric_header(headers, required=("metrics/",), any_of=("map50-95", "map50_95"))
    map50_key = self._training_eval_metric_header(headers, required=("metrics/", "map50"), rejected=("50-95", "50_95"))
    precision_key = self._training_eval_metric_header(headers, required=("metrics/", "precision"))
    recall_key = self._training_eval_metric_header(headers, required=("metrics/", "recall"))
    best = None
    best_value = None
    for row in rows or []:
        value = self._training_eval_float_value((row or {}).get(map5095_key, "")) if map5095_key else None
        if value is None:
            continue
        if best is None or value > best_value:
            best = row
            best_value = value
    if not best:
        return {}
    return {
        "epoch": self._training_eval_float_value(best.get("epoch", "")),
        "precision": self._training_eval_float_value(best.get(precision_key, "")) if precision_key else None,
        "recall": self._training_eval_float_value(best.get(recall_key, "")) if recall_key else None,
        "map50": self._training_eval_float_value(best.get(map50_key, "")) if map50_key else None,
        "map5095": best_value,
    }


def _training_eval_baseline_args_from_result(self, result):
    rec = (result or {}).get("recommendations", {}) if isinstance(result, dict) else {}
    data_yaml = self.normalize_path(rec.get("data_yaml_path", "") or (result or {}).get("data_yaml_path", "") or getattr(self, "data_yaml_path", ""))
    model_path = self.normalize_path(rec.get("weights_path", "") or self._training_eval_model_path(result, prefer_recommendation=True))
    imgsz = str(int(rec.get("imgsz", self._parse_training_imgsz_value()) or self._parse_training_imgsz_value()))
    task = str(rec.get("task") or "detect").strip().lower() or "detect"
    return {
        "task": task,
        "mode": "val",
        "split": "val",
        "data": data_yaml,
        "model": model_path,
        "imgsz": imgsz,
        "split_fingerprint": self._training_eval_data_split_fingerprint(data_yaml),
    }


def _training_eval_data_split_fingerprint(self, data_yaml_path):
    data_yaml_path = self.normalize_path(data_yaml_path or "")
    if not data_yaml_path or not os.path.exists(data_yaml_path):
        return ""
    try:
        yaml_data, yaml_dir = self._training_health_parse_data_yaml(data_yaml_path)
        digest = hashlib.sha256()
        digest.update(data_yaml_path.encode("utf-8", errors="ignore"))
        try:
            with open(data_yaml_path, "rb") as f:
                digest.update(f.read())
        except Exception:
            pass
        dataset_root = self.normalize_path((yaml_data or {}).get("path", "") or yaml_dir)
        if dataset_root and not os.path.isabs(dataset_root):
            dataset_root = self.normalize_path(os.path.join(yaml_dir, dataset_root))
        for split in ("train", "val"):
            digest.update(f"\n[{split}]\n".encode("utf-8"))
            entries = (yaml_data or {}).get(split)
            entries = entries if isinstance(entries, list) else [entries]
            for entry in entries:
                value = str(entry or "").strip().strip('"').strip("'")
                if not value:
                    continue
                path = self.normalize_path(
                    value if os.path.isabs(value) else os.path.join(dataset_root or yaml_dir, value)
                )
                digest.update(path.encode("utf-8", errors="ignore"))
                digest.update(b"\n")
                # Hash list files sequentially instead of issuing tens of
                # thousands of os.path.isfile calls for their image paths.
                if os.path.isfile(path) and os.path.splitext(path)[1].lower() == ".txt":
                    try:
                        with open(path, "rb") as f:
                            while True:
                                block = f.read(1024 * 1024)
                                if not block:
                                    break
                                digest.update(block)
                    except Exception:
                        pass
                elif os.path.exists(path):
                    try:
                        stat = os.stat(path)
                        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
                    except Exception:
                        pass
        return digest.hexdigest()[:16]
    except Exception as e:
        logger.debug(f"Could not fingerprint training split {data_yaml_path}: {e}")
        return ""


def _training_eval_read_baseline_record(self, run_dir):
    path = self.normalize_path(os.path.join(self.normalize_path(run_dir), "baseline_validation.json"))
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if isinstance(data, dict):
            data["record_path"] = path
            return data
    except Exception as e:
        logger.debug(f"Could not read baseline validation record {path}: {e}")
    return {}


def _training_eval_write_baseline_validation_record(self, run_dir, command=None, log_path=""):
    run_dir = self.normalize_path(run_dir)
    log_path = self.normalize_path(log_path or self._training_eval_log_path(run_dir))
    args = self._training_eval_run_args_from_log(log_path)
    if str(args.get("mode", "")).lower() != "val":
        return {}
    if str(args.get("split", "val") or "val") != "val":
        return {}
    metrics = self._training_eval_parse_all_metric_from_log(log_path)
    if not metrics:
        return {}
    record = {
        "generated_by": "DarkFusion Training Evaluator",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "log_path": log_path,
        "args": args,
        "split_fingerprint": self._training_eval_data_split_fingerprint(args.get("data", "")),
        "metrics": metrics,
        "command": list(command or []),
    }
    path = os.path.join(run_dir, "baseline_validation.json").replace("\\", "/")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        record["record_path"] = self.normalize_path(path)
        self._training_eval_baseline_validation_cache = {}
        return record
    except Exception as e:
        logger.warning(f"Could not write baseline validation record: {e}")
        return {}


def _training_eval_baseline_candidate_record(self, run_dir):
    record = self._training_eval_read_baseline_record(run_dir)
    if record:
        return {
            "run_dir": self.normalize_path(record.get("run_dir", "") or run_dir),
            "log_path": self.normalize_path(record.get("log_path", "") or self._training_eval_log_path(run_dir)),
            "args": record.get("args", {}) or {},
            "split_fingerprint": record.get("split_fingerprint", ""),
            "metrics": record.get("metrics", {}) or {},
            "record_path": record.get("record_path", ""),
        }
    log_path = os.path.join(self.normalize_path(run_dir), "darkfusion_console.log")
    if not os.path.exists(log_path):
        return {}
    args = self._training_eval_run_args_from_log(log_path)
    if str(args.get("mode", "")).lower() != "val":
        return {}
    metrics = self._training_eval_parse_all_metric_from_log(log_path)
    if not metrics:
        return {}
    return {
        "run_dir": self.normalize_path(run_dir),
        "log_path": self.normalize_path(log_path),
        "args": args,
        "split_fingerprint": "",
        "metrics": metrics,
    }


def _training_eval_find_matching_baseline_validation_for_args(self, target_args, reference_run_dir="", reference_time=None):
    target_args = dict(target_args or {})
    target_data = target_args.get("data", "")
    target_model = target_args.get("model", "")
    target_imgsz = str(target_args.get("imgsz", "") or "").strip()
    target_split_fingerprint = str(target_args.get("split_fingerprint", "") or "").strip()
    if not target_split_fingerprint and target_data:
        target_split_fingerprint = self._training_eval_data_split_fingerprint(target_data)
    if not target_data or not target_model or not target_imgsz:
        return {}

    reference_run_dir = self.normalize_path(reference_run_dir)
    reference_time = float(reference_time or (self._training_eval_run_modified_time(reference_run_dir) if reference_run_dir else 0.0) or time.time())
    cache_key = (
        os.path.normcase(os.path.abspath(reference_run_dir)) if reference_run_dir else "current",
        self._training_eval_resolve_compare_path(target_data, APP_DIR),
        self._training_eval_resolve_compare_path(target_model, APP_DIR),
        target_imgsz,
        str(target_args.get("split", "val") or "val"),
        target_split_fingerprint,
        round(float(reference_time), 3),
    )
    now = time.time()
    cache = getattr(self, "_training_eval_baseline_validation_cache", {}) or {}
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and now - float(cached.get("at", 0.0) or 0.0) < 30.0:
        return dict(cached.get("record") or {})

    best = {}
    best_distance = None
    checked = 0
    max_checked = 5000

    for root in self._training_eval_candidate_run_roots():
        root = self.normalize_path(root)
        if not root or not os.path.isdir(root):
            continue
        try:
            for current, dirs, files in os.walk(root):
                checked += 1
                if checked > max_checked:
                    break
                current = self.normalize_path(current)
                rel = os.path.relpath(current, root)
                depth = 0 if rel == "." else rel.count(os.sep) + 1
                if depth >= 4:
                    dirs[:] = []
                dirs[:] = [
                    d for d in dirs
                    if d.lower() not in {"weights", "__pycache__"} and not d.startswith(".")
                ]
                record = self._training_eval_baseline_candidate_record(current)
                if not record:
                    continue
                args = record.get("args", {}) or {}
                if str(args.get("mode", "")).lower() != "val":
                    continue
                if str(args.get("split", "val") or "val") != str(target_args.get("split", "val") or "val"):
                    continue
                if not self._training_eval_compare_arg_match(target_data, args.get("data", ""), APP_DIR, kind="data"):
                    continue
                if not self._training_eval_compare_arg_match(target_model, args.get("model", ""), APP_DIR, kind="model"):
                    continue
                if str(args.get("imgsz", "")).strip() != target_imgsz:
                    continue
                record_split_fingerprint = str(record.get("split_fingerprint", "") or args.get("split_fingerprint", "") or "").strip()
                if target_split_fingerprint and record_split_fingerprint and record_split_fingerprint != target_split_fingerprint:
                    continue
                metrics = record.get("metrics", {}) or {}
                log_path = self.normalize_path(record.get("log_path", ""))
                modified = self._training_eval_run_modified_time(current)
                if not modified and log_path and os.path.exists(log_path):
                    modified = os.path.getmtime(log_path)
                if not modified:
                    modified = time.time()
                distance = abs(modified - reference_time)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best = dict(record)
        except Exception as e:
            logger.debug(f"Could not scan baseline validation under {root}: {e}")
    cache[cache_key] = {"at": now, "record": dict(best)}
    self._training_eval_baseline_validation_cache = cache
    return best


def _training_eval_find_matching_baseline_validation(self, train_run_dir, train_args=None):
    train_run_dir = self.normalize_path(train_run_dir)
    train_args = dict(train_args or self._training_eval_run_args_for_compare(train_run_dir) or {})
    return self._training_eval_find_matching_baseline_validation_for_args(
        train_args,
        reference_run_dir=train_run_dir,
    )


def _training_eval_find_matching_baseline_for_result(self, result):
    return self._training_eval_find_matching_baseline_validation_for_args(
        self._training_eval_baseline_args_from_result(result),
        reference_time=time.time(),
    )


def _training_eval_format_baseline_status(self, result=None):
    target_args = self._training_eval_baseline_args_from_result(result or {})
    data_yaml = self.normalize_path(target_args.get("data", ""))
    model_path = self.normalize_path(target_args.get("model", ""))
    imgsz = str(target_args.get("imgsz", "") or "").strip()
    if not data_yaml or not model_path or not imgsz:
        return "Baseline: evaluate dataset and select weights first.", ""

    model_name = os.path.basename(model_path) or model_path
    baseline = self._training_eval_find_matching_baseline_validation_for_args(target_args, reference_time=time.time())
    metrics = (baseline or {}).get("metrics", {}) if isinstance(baseline, dict) else {}
    if metrics:
        map5095 = metrics.get("map5095")
        map50 = metrics.get("map50")
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        text = (
            f"Baseline: {model_name} @ {imgsz} | "
                f"mAP50-95 {float(map5095):.3f} | mAP50 {float(map50):.3f} | current"
        )
        tooltip = (
            f"Baseline validation is matched to this model, data YAML, image size, and split.\n"
                f"Precision: {float(precision):.3f}\n"
                f"Recall: {float(recall):.3f}\n"
                f"Run: {baseline.get('run_dir', '')}\n"
                f"Log: {baseline.get('log_path', '')}"
        )
        return text, tooltip

    text = f"Baseline: missing for {model_name} @ {imgsz}. Run Metrics Only (mAP) in Validation Review."
    tooltip = "No matching validation baseline was found for the current model, obj.yaml split, and image size."
    return text, tooltip


def _training_eval_update_baseline_widget(self, widgets, result=None):
    widgets = widgets or {}
    label = widgets.get("baseline_label")
    if label is None or sip.isdeleted(label):
        return
    text, tooltip = self._training_eval_format_baseline_status(result or getattr(self, "training_evaluator_last_result", None) or {})
    label.setText(text)
    label.setToolTip(tooltip)


def _training_eval_baseline_run_name(self, result=None, force_cpu=False):
    rec = (result or {}).get("recommendations", {}) if isinstance(result, dict) else {}
    model_path = self.normalize_path(
        rec.get("weights_path", "") or self._training_eval_model_path(result, prefer_recommendation=True)
    )
    model_stem = os.path.splitext(os.path.basename(str(model_path or "weights")))[0]
    model_stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_stem).strip("._-") or "weights"
    imgsz = str(rec.get("imgsz", "") or self._parse_training_imgsz_value())
    device = "cpu" if force_cpu else "gpu"
    return f"baseline_{device}_{model_stem}_{imgsz}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _training_eval_run_data_yaml_path(self, run_dir):
    args, _args_path = self._training_eval_read_args_yaml(run_dir)
    data_yaml = self.normalize_path(args.get("data", "") if isinstance(args, dict) else "")
    if data_yaml and os.path.exists(data_yaml):
        return data_yaml
    if data_yaml and not os.path.isabs(data_yaml):
        for base in (APP_DIR, run_dir, os.getcwd()):
            candidate = self.normalize_path(os.path.join(base, data_yaml))
            if os.path.exists(candidate):
                return candidate
    return ""


def _training_eval_results_diagnosis(self, headers, rows, run_dir=""):
    if not rows:
        return {
            "state": "idle",
            "title": "Waiting for metrics",
            "detail": "No results.csv rows yet.",
            "issues": [],
        }

    issues = []
    advice = []

    def add(severity, text):
        issues.append({"severity": severity, "text": text})

    watchdog_path = self.normalize_path(
        os.path.join(self.normalize_path(run_dir), ".darkfusion_training_health.json")
    ) if run_dir else ""
    watchdog_record = {}
    if watchdog_path and os.path.isfile(watchdog_path):
        try:
            with open(watchdog_path, "r", encoding="utf-8", errors="ignore") as handle:
                watchdog_record = json.load(handle) or {}
        except Exception:
            watchdog_record = {}
    watchdog_state = str(watchdog_record.get("state", "") or "").strip().lower()
    watchdog_detail = str(watchdog_record.get("detail", "") or "").strip()
    if watchdog_state in {"stopped_abnormal", "stopped_collapse", "abnormal"}:
        add("error", f"Training watchdog: {watchdog_detail or watchdog_state}")
        healthy_checkpoint = self.normalize_path(
            os.path.join(self.normalize_path(run_dir), "weights", "darkfusion_healthy.pt")
        )
        if os.path.isfile(healthy_checkpoint):
            advice.append(f"Use the preserved healthy checkpoint: {healthy_checkpoint}")

    train_args = self._training_eval_run_args_for_compare(run_dir) if run_dir else {}
    if train_args and not train_args.get("split_fingerprint") and train_args.get("data"):
        train_args["split_fingerprint"] = self._training_eval_data_split_fingerprint(train_args.get("data", ""))
    data_yaml = self._training_eval_run_data_yaml_path(run_dir)
    split_report = self._training_eval_yaml_split_report(data_yaml) if data_yaml else {}
    if split_report:
        train_images = int(split_report.get("train_images", 0) or 0)
        valid_images = int(split_report.get("valid_images", 0) or 0)
        overlap = int(split_report.get("overlap_images", 0) or 0)
        if not valid_images:
            add("error", "No validation images found")
        elif split_report.get("same_split"):
            add("error", "Train and validation use the same images")
        elif overlap:
            ratio = overlap / max(1, valid_images)
            severity = "error" if ratio >= 0.5 else "warning"
            add(severity, f"{overlap} validation image(s) also appear in train")
        missing_valid = split_report.get("missing_valid", []) or []
        if missing_valid:
            severity = "warning" if len(missing_valid) < max(3, train_images // 20) else "error"
            preview = ", ".join(str(row.get("class_name")) for row in missing_valid[:4])
            add(severity, f"{len(missing_valid)} class(es) missing from validation: {preview}")

    row_count = len(rows)
    map5095_key = self._training_eval_metric_header(headers, required=("metrics/",), any_of=("map50-95", "map50_95"))
    map50_key = self._training_eval_metric_header(headers, required=("metrics/", "map50"), rejected=("50-95", "50_95"))
    precision_key = self._training_eval_metric_header(headers, required=("metrics/", "precision"))
    recall_key = self._training_eval_metric_header(headers, required=("metrics/", "recall"))

    map5095_values = self._training_eval_metric_values_for_header(rows, map5095_key)
    map50_values = self._training_eval_metric_values_for_header(rows, map50_key)
    precision_values = self._training_eval_metric_values_for_header(rows, precision_key)
    recall_values = self._training_eval_metric_values_for_header(rows, recall_key)
    train_loss_values = self._training_eval_loss_values(headers, rows, "train/")
    val_loss_values = self._training_eval_loss_values(headers, rows, "val/")

    last_map5095 = self._training_eval_last_number(map5095_values)
    last_map50 = self._training_eval_last_number(map50_values)
    last_precision = self._training_eval_last_number(precision_values)
    last_recall = self._training_eval_last_number(recall_values)
    best_snapshot = self._training_eval_best_metric_snapshot(headers, rows)
    best_map5095 = best_snapshot.get("map5095")
    best_epoch = best_snapshot.get("epoch")
    baseline_comparison = {}
    if run_dir and best_snapshot:
        baseline = self._training_eval_find_matching_baseline_validation(run_dir, train_args)
        baseline_metrics = (baseline or {}).get("metrics", {}) if isinstance(baseline, dict) else {}
        baseline_map5095 = baseline_metrics.get("map5095")
        if baseline_map5095 is not None and best_map5095 is not None:
            delta = float(best_map5095) - float(baseline_map5095)
            baseline_comparison = {
                "run_dir": baseline.get("run_dir", ""),
                "log_path": baseline.get("log_path", ""),
                "baseline": baseline_metrics,
                "best": best_snapshot,
                "delta_map5095": round(delta, 4),
            }
            if delta < -0.005:
                add("warning", f"Pretrained baseline beats trained best by {abs(delta):.3f} mAP50-95")
                advice.append("Keep the baseline or restart from the last good checkpoint with a much lower learning rate.")
            elif delta < 0.025:
                add("info", f"Fine-tune only slightly beats baseline (+{delta:.3f} mAP50-95)")
                advice.append("Use a short refresh fine-tune and keep best.pt; longer training is unlikely to help without new/cleaner data.")
            else:
                add("info", f"Fine-tune beats baseline by {delta:.3f} mAP50-95")

    if row_count < 3:
        add("info", "Collecting more epochs before strong diagnosis")

    if row_count >= 4 and last_map5095 is not None:
        if best_map5095 is None:
            best_map5095 = max(value for value in map5095_values if value is not None)
        drop = best_map5095 - last_map5095
        if drop >= 0.15:
            add("error", f"mAP50-95 dropped {drop:.3f} from best")
        elif drop >= 0.07:
            add("warning", f"mAP50-95 dropped {drop:.3f} from best")

    train_start = self._training_eval_window_mean(train_loss_values, take_last=False)
    train_end = self._training_eval_window_mean(train_loss_values, take_last=True)
    val_start = self._training_eval_window_mean(val_loss_values, take_last=False)
    val_end = self._training_eval_window_mean(val_loss_values, take_last=True)
    if row_count >= 5 and None not in (train_start, train_end, val_start, val_end):
        train_improving = train_end < train_start * 0.92
        val_worse = val_end > val_start * 1.10
        if train_improving and val_worse:
            severity = "error" if val_end > val_start * 1.25 else "warning"
            add(severity, "Training loss is falling while validation loss is rising")
        elif last_map50 is not None and last_map50 < 0.30 and train_end >= train_start * 0.97:
            add("warning", "Low mAP with little train-loss improvement")

        early_best = best_epoch is not None and float(best_epoch) <= max(1.0, min(3.0, row_count * 0.2))
        best_drop = (float(best_map5095) - float(last_map5095)) if best_map5095 is not None and last_map5095 is not None else 0.0
        if early_best and best_drop >= 0.05 and (train_improving or val_worse):
            add("warning", "Refresh fine-tune pattern: best validation arrived very early")
            model_arg = str(train_args.get("model", "") or "")
            custom_weights = bool(model_arg and not self._looks_like_official_ultralytics_weight(model_arg))
            refresh_lr = 0.0005 if custom_weights else 0.001
            advice.insert(0,
                "Refresh fine-tune suggestion: start from the best/current weights with "
                    f"epochs=10-20 patience=5 lr0={refresh_lr:g} lrf=0.05 warmup_epochs=1 "
                    "freeze=10 mosaic=0 close_mosaic=0 cache=False."
            )

    if last_precision is not None and last_recall is not None:
        gap = abs(last_precision - last_recall)
        if gap >= 0.35:
            direction = "recall is low" if last_precision > last_recall else "precision is low"
            add("error", f"Precision/recall imbalance: {direction}")
        elif gap >= 0.20:
            direction = "recall is low" if last_precision > last_recall else "precision is low"
            add("warning", f"Precision/recall imbalance: {direction}")

    if last_map50 is not None and last_map5095 is not None and last_map50 >= 0.45:
        gap = last_map50 - last_map5095
        if gap >= 0.45:
            add("error", f"mAP50 vs mAP50-95 gap is large ({gap:.3f})")
        elif gap >= 0.30:
            add("warning", f"mAP50 vs mAP50-95 gap is wide ({gap:.3f})")

    if row_count >= 5 and last_map50 is not None and last_map5095 is not None:
        if last_map50 < 0.25 and last_map5095 < 0.12:
            add("warning", "Metrics are still very low after several epochs")

    severity_order = {"error": 3, "warning": 2, "info": 1}
    worst = max((severity_order.get(issue.get("severity"), 0) for issue in issues), default=0)
    if worst >= 3:
        state = "error"
        title = "Training needs attention"
    elif worst == 2:
        state = "warning"
        title = "Training watch"
    elif worst == 1:
        state = "info"
        title = "Collecting signal"
    else:
        state = "success"
        title = "Training looks good"

    if not issues and row_count >= 3:
        detail = "Validation, losses, and score curves are not showing obvious problems."
    else:
        detail_parts = [issue["text"] for issue in issues[:2]]
        if advice:
            detail_parts.append(f"Advice: {advice[0]}")
        else:
            detail_parts.extend(issue["text"] for issue in issues[2:3])
        detail = "; ".join(detail_parts)
    return {
        "state": state,
        "title": title,
        "detail": detail,
        "issues": issues,
        "advice": advice,
        "best_metrics": best_snapshot,
        "baseline_comparison": baseline_comparison,
        "split_report": split_report,
        "data_yaml": data_yaml,
    }


def _training_eval_update_diagnosis_widgets(self, widgets, diagnosis):
    widgets = widgets or {}
    diagnosis = diagnosis or {}
    state = str(diagnosis.get("state") or "idle")
    colors = {
        "success": ("#38d978", "rgba(56, 217, 120, 0.16)"),
        "warning": ("#f0b84c", "rgba(240, 184, 76, 0.16)"),
        "error": ("#ff5c78", "rgba(255, 92, 120, 0.16)"),
        "info": ("#4fa3ff", "rgba(79, 163, 255, 0.16)"),
        "idle": ("#7c8794", "rgba(124, 135, 148, 0.12)"),
    }
    color, background = colors.get(state, colors["idle"])
    panel = widgets.get("diagnosis_panel")
    if panel is not None and not sip.isdeleted(panel):
        panel.setStyleSheet(
            "QFrame#trainingDiagnosisPanel {"
                f"border: 1px solid {color};"
                f"background: {background};"
                "border-radius: 8px;"
                "}"
        )
    light = widgets.get("diagnosis_light")
    if light is not None and not sip.isdeleted(light):
        light.setStyleSheet(
            f"background: {color}; border-radius: 7px; min-width: 14px; max-width: 14px;"
                f"min-height: 14px; max-height: 14px; border: 1px solid {color};"
        )
    title = widgets.get("diagnosis_title")
    if title is not None and not sip.isdeleted(title):
        title.setText(str(diagnosis.get("title") or "Waiting for metrics"))
        title.setStyleSheet(f"font-weight: 800; color: {color};")
    detail = widgets.get("diagnosis_detail")
    if detail is not None and not sip.isdeleted(detail):
        detail.setText(str(diagnosis.get("detail") or ""))
        tooltip_lines = [issue.get("text", "") for issue in diagnosis.get("issues", []) or []]
        tooltip_lines.extend(str(item) for item in diagnosis.get("advice", []) or [] if item)
        detail.setToolTip("\n".join(line for line in tooltip_lines if line))


def _training_eval_artifact_images(self, run_dir):
    run_dir = self.normalize_path(run_dir)
    if not run_dir or not os.path.isdir(run_dir):
        return []

    paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(os.path.join(run_dir, pattern)))

    def priority(path):
        name = os.path.basename(path).lower()
        order = 50
        if name == "results.png":
            order = 0
        elif "confusion_matrix" in name:
            order = 4
        elif "pr_curve" in name:
            order = 8
        elif "f1_curve" in name:
            order = 9
        elif "val_batch" in name and "pred" in name:
            order = 12
        elif "val_batch" in name and "label" in name:
            order = 13
        elif "train_batch" in name:
            order = 16
        return (order, os.path.basename(path).lower())

    return sorted({self.normalize_path(path) for path in paths}, key=priority)


def _training_eval_set_artifact_preview(self, combo, label):
    if combo is None or label is None or sip.isdeleted(combo) or sip.isdeleted(label):
        return
    path = combo.currentData()
    if not path or not os.path.exists(path):
        label.setText("No visual artifact yet")
        label.setPixmap(QPixmap())
        label.setToolTip("")
        return
    pixmap = QPixmap(path)
    if pixmap.isNull():
        label.setText(f"Could not load {os.path.basename(path)}")
        label.setPixmap(QPixmap())
        label.setToolTip(path)
        return
    label.setText("")
    label.setToolTip("Double-click to open a zoomable viewer.")
    target_size = label.size()
    if target_size.width() < 40 or target_size.height() < 40:
        target_size = QtCore.QSize(520, 260)
    label.setPixmap(pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))


def open_training_eval_artifact_viewer(self, path):
    path = self.normalize_path(path)
    if not path or not os.path.exists(path):
        QMessageBox.information(self, "Training Visual", "No visual artifact is selected yet.")
        return
    viewer = TrainingArtifactViewer(
        path,
        self,
        paths=self._training_eval_artifact_images(os.path.dirname(path)),
    )
    viewer.setAttribute(Qt.WA_DeleteOnClose, True)
    if self.styleSheet():
        viewer.setStyleSheet(self.styleSheet())
    if not isinstance(getattr(self, "training_artifact_viewers", None), list):
        self.training_artifact_viewers = []
    self.training_artifact_viewers.append(viewer)

    def cleanup(_obj=None, dialog=viewer):
        try:
            if dialog in self.training_artifact_viewers:
                self.training_artifact_viewers.remove(dialog)
        except Exception:
            pass

    viewer.destroyed.connect(cleanup)
    viewer.show()
    viewer.raise_()
    viewer.activateWindow()


def _training_eval_update_artifact_combo(self, combo, label, run_dir):
    if combo is None or label is None or sip.isdeleted(combo) or sip.isdeleted(label):
        return
    images = self._training_eval_artifact_images(run_dir)
    current = combo.currentData()
    existing = [combo.itemData(i) for i in range(combo.count())]
    if images != existing:
        combo.blockSignals(True)
        combo.clear()
        for path in images:
            combo.addItem(os.path.basename(path), path)
        if current and current in images:
            combo.setCurrentIndex(images.index(current))
        elif images:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)
        self._training_eval_set_artifact_preview(combo, label)


def _training_eval_set_activity_indicator(self, widgets, state="idle", label="", detail=""):
    indicator = (widgets or {}).get("activity_indicator")
    if indicator is not None and not sip.isdeleted(indicator) and hasattr(indicator, "set_state"):
        indicator.set_state(state, label=label, detail=detail)


def _training_eval_activity_state(self, resume_info=None, message="", has_run=False, has_error=False):
    message_lower = str(message or "").lower()
    resume_info = resume_info or {}
    status = str(resume_info.get("status", "") or "").lower()
    exit_code = resume_info.get("exit_code", None)

    if has_error or "failed" in message_lower or "watch failed" in message_lower:
        return "error", "Error"
    if "running" in message_lower or status == "running":
        return "running", "Working"
    if exit_code not in (None, 0):
        return "error", "Error"
    if status in {"interrupted", "interrupted_no_checkpoint", "unknown"}:
        return "warning", "Needs attention"
    if "finished with exit code 0" in message_lower or status == "complete":
        return "success", "Ready"
    if has_run:
        return "success", "Ready"
    return "idle", "Idle"


def _training_eval_update_eta(self, widgets, run_dir, resume_info, rows, csv_signature, message=""):
    from training_eta import training_eta

    eta_label = widgets.get("eta_label")
    eta_detail_label = widgets.get("eta_detail_label")
    if eta_label is None or sip.isdeleted(eta_label):
        return
    context = widgets.get("_eta_context", {})
    if context.get("run_dir") != run_dir:
        context = {}
    args = dict(context.get("args", {}))
    args.update(resume_info.get("args", {}) or {})
    mode = str(args.get("mode", "train") or "train").lower()
    status = str(resume_info.get("status", "none"))
    exit_match = re.search(r"finished with exit code (-?\d+)", message, flags=re.IGNORECASE)
    if exit_match:
        status = "complete" if int(exit_match.group(1)) == 0 else "failed"
        context["final_status"] = status
    elif context.get("final_status"):
        status = context["final_status"]
    elif "running" in message.lower():
        status = "running"
    elif resume_info.get("exit_code") not in (None, 0):
        status = "failed"
    elif resume_info.get("exit_code") == 0:
        status = "complete"
    if context.get("stopped"):
        status = "stopped"
    tune_snapshot = widgets.get("_tune_snapshot")
    if tune_snapshot:
        trial_label = tune_snapshot.get("trial_label") or "Waiting for a trial"
        total = tune_snapshot.get("total_trials", 0)
        completed = tune_snapshot.get("completed_trials", 0)
        failed = tune_snapshot.get("failed_trials", 0)
        progress = f"{completed}/{total} trials completed" if total else f"{completed} trials completed"
        if failed:
            progress += f" · {failed} failed"
        if status == "complete":
            text, detail = "Tuning complete", progress
        elif status == "failed":
            text, detail = "Tuning failed", progress
        elif status != "running":
            text, detail = "Tuning stopped", progress
        else:
            # Each trial starts its own epoch counter and cumulative timer.
            # A trial finishing does not mean that the tuning run is done.
            estimate = training_eta(
                rows, args.get("epochs", 0), status="running", now=time.time(),
                last_epoch_at=(csv_signature[1] / 1_000_000_000) if csv_signature else None,
                starting_epoch=0, mode="train",
            )
            trial_text = estimate["text"].replace("training", "tuning trial").replace("Training", "Tuning trial")
            text = f"Tuning · {trial_label} · {trial_text}"
            detail = f"{progress} · {estimate['detail']} · estimate is for this trial"
        eta_label.setText(text)
        if eta_detail_label is not None and not sip.isdeleted(eta_detail_label):
            eta_detail_label.setText(detail)
        return
    estimate = training_eta(
        rows,
        args.get("epochs", resume_info.get("target_epochs", 0)),
        status=status,
        now=time.time(),
        last_epoch_at=(csv_signature[1] / 1_000_000_000) if csv_signature else None,
        starting_epoch=context.get("starting_epoch"),
        mode=mode,
    )
    eta_label.setText(estimate["text"])
    if eta_detail_label is not None and not sip.isdeleted(eta_detail_label):
        eta_detail_label.setText(estimate["detail"])


def _training_eval_refresh_run_widgets(
    self,
    widgets,
    run_dir,
    log_path,
    message="",
    include_diagnosis=True,
):
    from darkfusion_tune_monitor import tuning_snapshot

    widgets = widgets or {}
    run_dir = self.normalize_path(run_dir)
    log_path = self.normalize_path(log_path or self._training_eval_log_path(run_dir))
    record = {"run_dir": run_dir, "log_path": log_path}
    active_record = getattr(self, "training_evaluator_active_run", None)
    if isinstance(active_record, dict) and self.normalize_path(active_record.get("run_dir", "")) == run_dir:
        record.update(active_record)
        if active_record.get("_eta_context"):
            widgets["_eta_context"] = active_record["_eta_context"]
    resume_info = self._training_eval_resume_info(record)

    try:
        stat = os.stat(log_path)
        log_signature = (int(stat.st_size), int(stat.st_mtime_ns), log_path)
    except OSError:
        log_signature = None
    if widgets.get("_log_signature") != log_signature or widgets.get("_log_path") != log_path:
        content = self._training_eval_read_text_tail(log_path, max_chars=40000)
        widgets["_log_cache"] = content
        widgets["_log_signature"] = log_signature
        widgets["_log_path"] = log_path
        log_text = widgets.get("log_text")
        if log_text is not None and not sip.isdeleted(log_text):
            log_text.setPlainText(content)
            log_text.moveCursor(QTextCursor.End)
    snapshot = tuning_snapshot(
        run_dir, log_text=widgets.get("_log_cache", ""),
        iterations=self._training_eval_int_value(
            self._training_eval_option_value(record.get("command", []), "--iterations"), 0,
        ),
    )
    metrics_run_dir = self.normalize_path(snapshot.get("metrics_dir") or run_dir)
    if snapshot:
        widgets["_tune_snapshot"] = snapshot
        trial_args, _trial_args_path = self._training_eval_read_args_yaml(metrics_run_dir)
        resume_info.update({"args": trial_args, "run_kind": "tune", "resumable": False})
        trial_label = snapshot.get("trial_label") or "Waiting for a trial"
        resume_info["note"] = f"{trial_label}. Live metrics show this trial; tuning continues between trials."
        message = f"{message or 'Tuning monitor active.'}  {trial_label}"
    else:
        widgets.pop("_tune_snapshot", None)
    widgets["_run_state_message"] = message

    state_label = widgets.get("run_state_label")
    if state_label is not None and not sip.isdeleted(state_label):
        state_label.setText(message or "Run monitor active.")

    has_run = bool(run_dir and os.path.isdir(run_dir))
    state, label = self._training_eval_activity_state(resume_info, message, has_run=has_run)
    if snapshot and state == "running":
        label = "Tuning"
    self._training_eval_set_activity_indicator(widgets, state, label, message or "")

    run_dir_label = widgets.get("run_dir_label")
    if run_dir_label is not None and not sip.isdeleted(run_dir_label):
        directory_text = f"Run: {run_dir}" if run_dir else "Run: waiting for Ultralytics output"
        if snapshot and metrics_run_dir != run_dir:
            directory_text += f"\nTrial: {metrics_run_dir}"
        run_dir_label.setText(directory_text)

    csv_path = self.normalize_path(os.path.join(metrics_run_dir, "results.csv"))
    try:
        stat = os.stat(csv_path)
        csv_signature = (int(stat.st_size), int(stat.st_mtime_ns), csv_path)
    except Exception:
        csv_signature = None
    metrics_changed = (widgets.get("_csv_signature") != csv_signature
                       or widgets.get("_metrics_run_dir") != metrics_run_dir)
    if metrics_changed:
        headers, rows, csv_path = self._training_eval_read_results_csv(metrics_run_dir)
        widgets["_metrics_cache"] = (headers, rows, csv_path)
        widgets["_csv_signature"] = csv_signature
        widgets["_metrics_run_dir"] = metrics_run_dir
    else:
        headers, rows, csv_path = widgets.get("_metrics_cache", ([], [], csv_path if os.path.exists(csv_path) else ""))
    metrics_label = widgets.get("metrics_label")
    if metrics_label is not None and not sip.isdeleted(metrics_label):
        if rows:
            metrics_label.setText(f"{self._training_eval_latest_metric_line(headers, rows)}   ({len(rows)} CSV rows)")
        elif csv_path:
            metrics_label.setText("results.csv exists but has no rows yet.")
        elif snapshot:
            metrics_label.setText("Waiting for this trial's first completed epoch...")
        else:
            metrics_label.setText("Waiting for results.csv. Validation runs may only write plots/log output.")
    self._training_eval_update_baseline_widget(widgets)
    self._training_eval_update_eta(widgets, run_dir, resume_info, rows, csv_signature, message)

    resume_note_label = widgets.get("resume_note_label")
    if resume_note_label is not None and not sip.isdeleted(resume_note_label):
        note = str(resume_info.get("note", "") or "")
        if note and resume_info.get("status") in {"interrupted", "interrupted_no_checkpoint", "running"}:
            resume_note_label.setText(note)
            resume_note_label.show()
        else:
            resume_note_label.clear()
            resume_note_label.hide()

    resume_button = widgets.get("resume_button")
    if resume_button is not None and not sip.isdeleted(resume_button):
        resume_button.setEnabled(bool(resume_info.get("resumable")))
        resume_button.setToolTip(str(resume_info.get("note", "") or "Resume the last interrupted training run."))

    if metrics_changed:
        self._training_eval_update_metrics_table(widgets.get("metrics_table"), headers, rows)
    if metrics_changed and include_diagnosis:
        self._training_eval_update_diagnosis_widgets(
            widgets,
            self._training_eval_results_diagnosis(headers, rows, metrics_run_dir),
        )

    score_plot = widgets.get("score_plot")
    if metrics_changed and score_plot is not None and not sip.isdeleted(score_plot):
        score_plot.set_series(self._training_eval_metric_series(headers, rows, "score"), "mAP / Precision / Recall", value_range=(0.0, 1.0))

    loss_plot = widgets.get("loss_plot")
    if metrics_changed and loss_plot is not None and not sip.isdeleted(loss_plot):
        loss_plot.set_series(self._training_eval_metric_series(headers, rows, "loss"), "Tuning Trial / Validation Loss" if snapshot else "Training / Validation Loss")

    tabs = widgets.get("tabs")
    artifact_combo = widgets.get("artifact_combo")
    charts_tab = widgets.get("charts_tab")
    charts_visible = tabs is None or sip.isdeleted(tabs) or charts_tab is None or tabs.currentWidget() is charts_tab
    now = time.monotonic()
    if charts_visible and now - float(widgets.get("_artifact_refresh_at", 0.0) or 0.0) >= 5.0:
        widgets["_artifact_refresh_at"] = now
        self._training_eval_update_artifact_combo(artifact_combo, widgets.get("artifact_preview"), metrics_run_dir)


def watch_training_evaluator_run(self, process, command, title, run_dir, log_path, widgets=None, status_label=None):
    if process is None:
        return

    record = self._training_eval_record_run(title, run_dir, log_path, command=command, process=process)
    if widgets is not None:
        command_args = {}
        for part in command or []:
            token = str(part).strip()
            if "=" in token:
                key, value = token.split("=", 1)
                command_args[key.strip().lower()] = value.strip()
            elif token.lower() in {"train", "val", "tune", "predict", "export"}:
                command_args["mode"] = token.lower()
        _headers, starting_rows, _csv_path = self._training_eval_read_results_csv(run_dir)
        widgets["_eta_context"] = {
            "run_dir": self.normalize_path(run_dir),
            "args": command_args,
            "starting_epoch": self._training_eval_last_completed_epoch(starting_rows),
        }
        record["_eta_context"] = widgets["_eta_context"]
        # A newly launched/resumed run must not inherit cached metrics or
        # a completed estimate from the previously watched process.
        widgets.pop("_csv_signature", None)
        widgets.pop("_metrics_run_dir", None)
        widgets.pop("_metrics_cache", None)
        widgets.pop("_tune_snapshot", None)
    timer = QTimer(self)
    timer.setInterval(2500)
    if not isinstance(getattr(self, "training_evaluator_run_timers", None), list):
        self.training_evaluator_run_timers = []
    self.training_evaluator_run_timers.append(timer)

    def widget_alive(widget):
        return widget is not None and not sip.isdeleted(widget)

    def poll():
        try:
            exit_code = process.poll()
        except Exception as e:
            timer.stop()
            if timer in self.training_evaluator_run_timers:
                self.training_evaluator_run_timers.remove(timer)
            if widget_alive(status_label):
                status_label.setText(f"{title} monitor failed: {e}")
            self._training_eval_set_activity_indicator(widgets, "error", "Error", f"{title} monitor failed")
            if widgets is not None:
                widgets.get("_eta_context", {})["final_status"] = "failed"
                _headers, rows, _csv_path = widgets.get("_metrics_cache", ([], [], ""))
                self._training_eval_update_eta(
                    widgets, self.normalize_path(run_dir), {"status": "failed"}, rows, None,
                )
            return

        running = exit_code is None
        state = f"{title} running..." if running else f"{title} finished with exit code {exit_code}."
        self._training_eval_refresh_run_widgets(widgets, run_dir, log_path, state)
        if widget_alive(status_label):
            status_label.setText((widgets or {}).get("_run_state_message", state))
        if not running:
            if exit_code == 0:
                baseline_record = self._training_eval_write_baseline_validation_record(run_dir, command=command, log_path=log_path)
                if baseline_record:
                    self._training_eval_update_baseline_widget(widgets)
            timer.stop()
            if timer in self.training_evaluator_run_timers:
                self.training_evaluator_run_timers.remove(timer)

    timer.timeout.connect(poll)
    timer.start()
    QTimer.singleShot(250, poll)


def _letterbox_preview_params(self, width, height, target_width, target_height):
    if width <= 0 or height <= 0 or target_width <= 0 or target_height <= 0:
        return 1.0, 0, 0, target_width, target_height

    scale = min(target_width / width, target_height / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    pad_x = target_width - new_width
    pad_y = target_height - new_height
    left = int(round(pad_x / 2 - 0.1))
    top = int(round(pad_y / 2 - 0.1))
    return scale, left, top, new_width, new_height

def _train_eval_label_extent(self, bbox):
    if bbox is None:
        return None

    if getattr(bbox, "segmentation", None):
        pts = bbox.segmentation
        xs = pts[0::2]
        ys = pts[1::2]
    elif getattr(bbox, "obb", None):
        pts = bbox.obb
        xs = pts[0::2]
        ys = pts[1::2]
    else:
        return (
            float(bbox.x_center),
            float(bbox.y_center),
            float(bbox.width),
            float(bbox.height),
        )

    if not xs or not ys:
        return None

    left = max(0.0, min(xs))
    right = min(1.0, max(xs))
    top = max(0.0, min(ys))
    bottom = min(1.0, max(ys))
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    return ((left + right) / 2.0, (top + bottom) / 2.0, width, height)

def _parse_training_eval_bbox(self, line, polygon_preference=None, expected_keypoints=None):
    previous_expected = getattr(BoundingBox, "EXPECTED_KEYPOINT_COUNT", None)
    BoundingBox.EXPECTED_KEYPOINT_COUNT = expected_keypoints or None
    try:
        return BoundingBox.from_str(line, preferred_polygon_type=polygon_preference)
    finally:
        BoundingBox.EXPECTED_KEYPOINT_COUNT = previous_expected

def _training_eval_run_dir_from_command(self, command):
    output_path = self._training_eval_option_value(command, "--output")
    if output_path:
        return self.normalize_path(os.path.dirname(output_path))

    project_dir = self._training_eval_command_value(command, "project")
    run_name = self._training_eval_command_value(command, "name")
    if project_dir and run_name:
        return self.normalize_path(os.path.join(project_dir, run_name))
    return ""

def _training_eval_log_path(self, run_dir):
    run_dir = self.normalize_path(run_dir)
    if not run_dir:
        return ""
    return self.normalize_path(os.path.join(run_dir, "darkfusion_console.log"))

def _training_eval_run_modified_time(self, run_dir):
    run_dir = self.normalize_path(run_dir)
    if not run_dir or not os.path.isdir(run_dir):
        return 0.0
    candidates = [
        run_dir,
        os.path.join(run_dir, "args.yaml"),
        os.path.join(run_dir, "results.csv"),
        os.path.join(run_dir, "darkfusion_console.log"),
        os.path.join(run_dir, "weights", "last.pt"),
        os.path.join(run_dir, "weights", "best.pt"),
    ]
    newest = 0.0
    for path in candidates:
        try:
            if os.path.exists(path):
                newest = max(newest, os.path.getmtime(path))
        except Exception:
            continue
    return newest

def _training_eval_candidate_run_roots(self):
    settings = getattr(self, "settings", {}) if isinstance(getattr(self, "settings", None), dict) else {}
    roots = []
    seen = set()

    def add_root(path):
        path = self.normalize_path(path or "")
        if not path:
            return
        try:
            key = os.path.normcase(os.path.abspath(path))
        except Exception:
            key = os.path.normcase(path)
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    add_root(getattr(self, "runs_directory", ""))
    add_root(settings.get("ultralyticsRunsDirectory", ""))

    saved_run_dir = settings.get("trainingEvaluatorLastRunDir", "")
    if saved_run_dir:
        add_root(os.path.dirname(self.normalize_path(saved_run_dir)))

    data_yaml = self.normalize_path(
        getattr(self, "data_yaml_path", "") or settings.get("ultralyticsDataYamlPath", "")
    )
    if data_yaml:
        add_root(os.path.join(os.path.dirname(data_yaml), "runs"))

    try:
        dataset_dir = self._training_eval_dataset_dir()
    except Exception:
        dataset_dir = ""
    if dataset_dir:
        add_root(os.path.join(dataset_dir, "runs"))

    try:
        add_root(self._training_eval_project_dir())
    except Exception:
        pass

    return roots

def _training_eval_discover_latest_run(self, force=False):
    now = time.time()
    cached = getattr(self, "_training_eval_discovered_run_cache", None)
    if not force and isinstance(cached, dict) and now - float(cached.get("at", 0.0) or 0.0) < 10.0:
        record = cached.get("record") or {}
        run_dir = self.normalize_path(record.get("run_dir", ""))
        if run_dir and os.path.isdir(run_dir):
            return dict(record)
        if not record:
            return {}

    best_dir = ""
    best_time = 0.0
    max_depth = 3
    max_checked = 5000
    checked = 0

    for root in self._training_eval_candidate_run_roots():
        if not root or not os.path.isdir(root):
            continue
        try:
            walker = os.walk(root)
            for current, dirs, files in walker:
                checked += 1
                if checked > max_checked:
                    break
                current = self.normalize_path(current)
                rel = os.path.relpath(current, root)
                depth = 0 if rel == "." else rel.count(os.sep) + 1
                if depth >= max_depth:
                    dirs[:] = []
                dirs[:] = [
                    d for d in dirs
                    if d.lower() not in {"weights", "__pycache__"} and not d.startswith(".")
                ]

                marker_paths = [
                    os.path.join(current, "args.yaml"),
                    os.path.join(current, "results.csv"),
                    os.path.join(current, "weights", "last.pt"),
                    os.path.join(current, "weights", "best.pt"),
                ]
                if not any(os.path.exists(path) for path in marker_paths):
                    continue

                args, _args_path = self._training_eval_read_args_yaml(current)
                mode = str(args.get("mode", "train") if args else "train").strip().lower()
                if mode and mode != "train":
                    continue

                modified = self._training_eval_run_modified_time(current)
                if modified > best_time:
                    best_time = modified
                    best_dir = current
        except Exception as e:
            logger.debug(f"Could not scan training runs under {root}: {e}")

    if not best_dir:
        self._training_eval_discovered_run_cache = {"at": now, "record": {}}
        return {}
    record = {
        "title": "Discovered Training Run",
        "run_dir": best_dir,
        "log_path": self._training_eval_log_path(best_dir),
        "process": None,
        "pid": None,
    }
    self._training_eval_discovered_run_cache = {"at": now, "record": dict(record)}
    return record

def _training_eval_record_run(self, title, run_dir, log_path, command=None, process=None):
    from darkfusion_training_runs import is_training_command

    run_dir = self.normalize_path(run_dir)
    log_path = self.normalize_path(log_path or self._training_eval_log_path(run_dir))
    record = {
        "title": str(title or "Training"),
        "run_dir": run_dir,
        "log_path": log_path,
        "command": list(command or []),
        "process": process,
        "pid": int(process.pid) if process is not None and getattr(process, "pid", None) else None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "kind": "train" if is_training_command(command) else "evaluation",
    }
    if record["pid"] and psutil is not None:
        try:
            record["process_created_at"] = psutil.Process(record["pid"]).create_time()
        except Exception:
            pass
    self.training_evaluator_active_run = record
    self.training_evaluator_last_run = dict(record)
    if record["kind"] == "train":
        self.training_evaluator_active_training_run = record
        self.training_evaluator_last_training_run = dict(record)
    self._training_eval_discovered_run_cache = None

    if isinstance(getattr(self, "settings", None), dict):
        self.settings["trainingEvaluatorLastRunDir"] = run_dir
        self.settings["trainingEvaluatorLastRunLog"] = log_path
        self.settings["trainingEvaluatorLastRunTitle"] = record["title"]
        if record["kind"] == "train":
            self.settings["trainingEvaluatorLastTrainingRunDir"] = run_dir
            self.settings["trainingEvaluatorLastTrainingRunLog"] = log_path
            self.settings["trainingEvaluatorLastTrainingRunTitle"] = record["title"]
        self.queue_settings_save()
    return record

def _training_eval_latest_saved_run(self, training_only=False):
    settings = getattr(self, "settings", {}) if isinstance(getattr(self, "settings", None), dict) else {}
    candidates = []
    for record in (
        getattr(self, "training_evaluator_active_training_run", None) if training_only else None,
        getattr(self, "training_evaluator_last_training_run", None) if training_only else None,
        getattr(self, "training_evaluator_active_run", None),
        getattr(self, "training_evaluator_last_run", None),
    ):
        if isinstance(record, dict) and record.get("run_dir"):
            candidates.append(record)

    saved_prefix = "trainingEvaluatorLastTrainingRun" if training_only else "trainingEvaluatorLastRun"
    saved_run_dir = settings.get(saved_prefix + "Dir", "")
    if saved_run_dir:
        candidates.append({
            "title": settings.get(saved_prefix + "Title", "Training"),
            "run_dir": saved_run_dir,
            "log_path": settings.get(saved_prefix + "Log", ""),
            "process": None,
            "pid": None,
            "kind": "train" if training_only else "",
        })

    discovered = self._training_eval_discover_latest_run()
    if discovered:
        candidates.append(discovered)

    valid = []
    seen_run_dirs = set()
    for record in candidates:
        run_dir = self.normalize_path(record.get("run_dir", ""))
        if run_dir and os.path.isdir(run_dir):
            from darkfusion_training_runs import path_identity
            run_key = path_identity(run_dir)
            if run_key in seen_run_dirs:
                continue
            if training_only:
                from darkfusion_training_runs import is_training_command
                if record.get("kind"):
                    if record["kind"] != "train":
                        continue
                elif record.get("command"):
                    if not is_training_command(record["command"]):
                        continue
                else:
                    args, _ = self._training_eval_read_args_yaml(run_dir)
                    if str(args.get("mode", "train") or "train").lower() != "train":
                        continue
            seen_run_dirs.add(run_key)
            clean = dict(record)
            clean["run_dir"] = run_dir
            clean["log_path"] = self.normalize_path(clean.get("log_path") or self._training_eval_log_path(run_dir))
            clean["_modified"] = self._training_eval_run_modified_time(run_dir)
            clean["_active"] = self._training_eval_run_is_active(record)
            valid.append(clean)
    for clean in valid:
        if clean.pop("_active", False):
            clean.pop("_modified", None)
            return clean
    if valid:
        valid.sort(key=lambda item: float(item.get("_modified") or 0.0), reverse=True)
        clean = valid[0]
        clean.pop("_modified", None)
        clean.pop("_active", None)
        return clean
    return {}

def _training_eval_run_is_active(self, record):
    process = (record or {}).get("process")
    if process is not None:
        try:
            return process.poll() is None
        except Exception:
            pass

    pid = (record or {}).get("pid")
    if pid and psutil is not None:
        try:
            from darkfusion_training_runs import process_matches_training_record
            proc = psutil.Process(int(pid))
            if (proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
                    and process_matches_training_record(proc.cmdline(), record)):
                created = (record or {}).get("process_created_at")
                if created is not None and abs(proc.create_time() - float(created)) > 0.01:
                    return False
                return True
        except Exception:
            pass

    # Runs launched by an older DarkFusion build have no persisted runner PID.
    # Ultralytics updates the captured progress log continuously, so recent
    # writes provide a cheap, non-blocking way to reattach after an app restart.
    log_path = self.normalize_path(
        (record or {}).get("log_path")
        or self._training_eval_log_path((record or {}).get("run_dir", ""))
    )
    try:
        recent = bool(log_path and os.path.isfile(log_path) and time.time() - os.path.getmtime(log_path) <= 20.0)
        if not recent:
            return False
        tail = self._training_eval_read_text_tail(log_path, max_chars=4096)
        current_command = tail.rsplit("DarkFusion command started:", 1)[-1]
        return not re.search(r"DarkFusion command finished:|^Exit code:\s*-?\d+\s*$", current_command, re.MULTILINE)
    except Exception:
        return False

def _training_eval_read_args_yaml(self, run_dir):
    run_dir = self.normalize_path(run_dir)
    args_path = self.normalize_path(os.path.join(run_dir, "args.yaml"))
    if not run_dir or not os.path.exists(args_path):
        return {}, args_path
    try:
        with open(args_path, "r", encoding="utf-8", errors="replace") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}, args_path
    except Exception as e:
        logger.warning(f"Could not read Ultralytics args from {args_path}: {e}")
        return {}, args_path

def _training_eval_restore_last_run_setup(self):
    """Return one verified, internally consistent setup from the latest training run."""
    settings = getattr(self, "settings", {}) if isinstance(getattr(self, "settings", None), dict) else {}
    candidate_dirs = []

    def add_run_dir(path):
        path = self.normalize_path(path or "")
        if path and os.path.isdir(path) and path not in candidate_dirs:
            args, _args_path = self._training_eval_read_args_yaml(path)
            if str(args.get("mode", "train") or "train").strip().lower() == "train":
                candidate_dirs.append(path)

    add_run_dir(settings.get("trainingEvaluatorLastTrainingRunDir", ""))
    for model_path in (
        getattr(self, "pt_path", ""),
        settings.get("ultralyticsPtPath", ""),
        settings.get("trainingEvaluatorWeightsPath", ""),
    ):
        model_path = self.normalize_path(model_path or "")
        parent = os.path.dirname(model_path)
        if os.path.basename(parent).lower() == "weights":
            add_run_dir(os.path.dirname(parent))

    # A validation launch may be the most recently recorded command. Recover
    # its model path from the log, then walk back from weights/*.pt to the
    # training run whose args.yaml is authoritative.
    last_command_run = self.normalize_path(settings.get("trainingEvaluatorLastRunDir", ""))
    last_command_log = self.normalize_path(
        settings.get("trainingEvaluatorLastRunLog", "")
        or self._training_eval_log_path(last_command_run)
    )
    logged_args = self._training_eval_run_args_from_log(last_command_log)
    logged_model = self.normalize_path(str(logged_args.get("model", "") or ""))
    logged_parent = os.path.dirname(logged_model)
    if os.path.basename(logged_parent).lower() == "weights":
        add_run_dir(os.path.dirname(logged_parent))

    discovered = self._training_eval_discover_latest_run(force=True) or {}
    add_run_dir(discovered.get("run_dir", ""))
    if not candidate_dirs:
        return {}

    run_dir = max(candidate_dirs, key=self._training_eval_run_modified_time)
    record = {
        "title": "Restored Training Run",
        "run_dir": run_dir,
        "log_path": self._training_eval_log_path(run_dir),
        "process": None,
        "pid": None,
    }
    if not run_dir:
        return {}

    args, args_path = self._training_eval_read_args_yaml(run_dir)
    if not args:
        return {}

    data_yaml = self.normalize_path(str(args.get("data", "") or ""))
    if data_yaml and not os.path.isabs(data_yaml):
        data_yaml = self.normalize_path(os.path.join(run_dir, data_yaml))
    if not data_yaml or not os.path.isfile(data_yaml):
        data_yaml = ""

    # Restoring a historical run is a fallback. A newly selected/generated
    # dataset must survive closing and reopening the trainer.
    selected_data = (getattr(self, "data_yaml_path", None)
                     or settings.get("trainingEvaluatorDataYamlPath", "")
                     or settings.get("ultralyticsDataYamlPath", ""))
    if selected_data and data_yaml:
        from darkfusion_training_runs import path_identity
        if path_identity(selected_data) != path_identity(data_yaml):
            return {}

    # Prefer the trained best checkpoint, but never select it merely because
    # the file exists. Interrupted writes can leave a large yet invalid .pt.
    model_from_args = self.normalize_path(str(args.get("model", "") or ""))
    candidates = [
        self.normalize_path(os.path.join(run_dir, "weights", "best.pt")),
        self.normalize_path(os.path.join(run_dir, "weights", "last.pt")),
        model_from_args,
    ]
    checkpoint = ""
    weight_info = {}
    seen = set()
    for candidate in candidates:
        key = os.path.normcase(candidate) if candidate else ""
        if not candidate or key in seen or not os.path.isfile(candidate):
            continue
        seen.add(key)
        info = self.inspect_training_eval_weights(candidate)
        if info.get("loaded"):
            checkpoint = candidate
            weight_info = info
            break
        logger.warning(
            "Skipping unusable checkpoint while restoring evaluator setup: %s (%s)",
            candidate,
            info.get("error") or "load failed",
        )

    task = str(args.get("task", "") or weight_info.get("task", "") or "").strip().lower()
    if task not in {"detect", "segment", "classify", "pose", "obb"}:
        task = str(weight_info.get("task", "") or "").strip().lower()

    if checkpoint and data_yaml and isinstance(getattr(self, "settings", None), dict):
        self.settings["trainingEvaluatorLastTrainingRunDir"] = run_dir

    return {
        "record": record,
        "run_dir": run_dir,
        "args_path": args_path,
        "args": args,
        "data_yaml": data_yaml,
        "checkpoint": checkpoint,
        "weight_info": weight_info,
        "task": task,
    }

def _training_eval_int_value(self, value, default=0):
    try:
        parsed = int(float(str(value).strip()))
        return parsed
    except Exception:
        return int(default)

def _training_eval_last_completed_epoch(self, rows):
    for row in reversed(rows or []):
        value = self._training_eval_float_value((row or {}).get("epoch", ""))
        if value is not None:
            return max(0, int(value))
    return 0

def _training_eval_resume_info(self, record=None):
    if isinstance(record, str):
        from darkfusion_training_runs import path_identity
        saved = self._training_eval_latest_saved_run(training_only=True) or {}
        record = dict(saved) if path_identity(saved.get("run_dir")) == path_identity(record) else {"run_dir": record}
    record = dict(record or self._training_eval_latest_saved_run(training_only=True) or {})
    run_dir = self.normalize_path(record.get("run_dir", ""))
    log_path = self.normalize_path(record.get("log_path") or self._training_eval_log_path(run_dir))
    info = {
        "status": "none",
        "run_dir": run_dir,
        "log_path": log_path,
        "args": {},
        "args_path": "",
        "last_pt": "",
        "target_epochs": 0,
        "last_epoch": 0,
        "resumable": False,
        "note": "No previous training run is available.",
    }
    if not run_dir or not os.path.isdir(run_dir):
        return info

    args, args_path = self._training_eval_read_args_yaml(run_dir)
    headers, rows, _csv_path = self._training_eval_read_results_csv(run_dir)
    del headers
    last_epoch = self._training_eval_last_completed_epoch(rows)
    target_epochs = self._training_eval_int_value(args.get("epochs", 0), 0)
    last_pt = self.normalize_path(os.path.join(run_dir, "weights", "last.pt"))
    has_last_pt = os.path.exists(last_pt)
    active = self._training_eval_run_is_active(record)
    checkpoint_label = (
        f"epoch {last_epoch}/{target_epochs}" if target_epochs else f"epoch {last_epoch}"
    )
    if active:
        info.update({
            "args": args,
            "args_path": args_path,
            "last_pt": last_pt,
            "target_epochs": target_epochs,
            "last_epoch": last_epoch,
            "exit_code": None,
            "status": "running",
            "note": f"Training is still running at {checkpoint_label}. Live metrics will keep updating.",
        })
        return info

    log_tail = self._training_eval_read_text_tail(log_path, max_chars=60000)
    completed_marker = bool(
        re.search(r"\bepochs completed\b|Results saved to|Exit code:\s*0\b", log_tail, flags=re.IGNORECASE)
    )
    exit_code_match = re.search(r"Exit code:\s*(-?\d+)", log_tail, flags=re.IGNORECASE)
    exit_code = int(exit_code_match.group(1)) if exit_code_match else None
    info.update({
        "args": args,
        "args_path": args_path,
        "last_pt": last_pt,
        "target_epochs": target_epochs,
        "last_epoch": last_epoch,
        "exit_code": exit_code,
    })

    tune_best, tune_best_path = self.read_training_tune_best_hyperparameters(run_dir)
    if tune_best and exit_code in (None, 0):
        info["status"] = "complete"
        info["run_kind"] = "tune"
        info["tune_best_path"] = tune_best_path
        info["note"] = (
            f"Hyperparameter tuning completed successfully. Best settings are available in "
                f"{os.path.basename(tune_best_path)}."
        )
        return info

    if not args:
        info["status"] = "unknown"
        info["note"] = "Last run is missing args.yaml, so DarkFusion cannot safely resume it."
        return info

    if (target_epochs > 0 and last_epoch >= target_epochs) or (completed_marker and not target_epochs):
        info["status"] = "complete"
        info["note"] = f"Last training run completed at {checkpoint_label}. Start Training will create a new run."
        return info

    if has_last_pt:
        info["status"] = "interrupted"
        info["resumable"] = True
        info["note"] = (
            f"Last training run appears interrupted at {checkpoint_label}. "
                "Resume is available from weights/last.pt. Use Resume Last Run to continue it, "
                "or Start Training to start a fresh run on this dataset."
        )
        return info

    if not has_last_pt:
        info["status"] = "interrupted_no_checkpoint"
        info["note"] = (
            f"Last training run appears interrupted at {checkpoint_label}, but weights/last.pt was not found. "
                "Start Training will create a new run."
        )
        return info

    info["status"] = "complete" if last_epoch >= target_epochs > 0 else "unknown"
    info["note"] = f"Last training run status: {checkpoint_label}. Start Training will create a new run."
    return info

def _draw_training_preview_label_text(self, scene, text, x, y, color):
    item = QGraphicsTextItem(str(text))
    item.setDefaultTextColor(QColor(255, 255, 255))
    font = item.font()
    font.setPointSize(max(7, min(12, int(getattr(self, "font_size_slider", None).value() if getattr(self, "font_size_slider", None) is not None else 9))))
    font.setBold(True)
    item.setFont(font)
    item.setPos(max(2, float(x)), max(2, float(y)))
    item.setZValue(2000)
    scene.addItem(item)

def _training_preview_class_color(self, class_id, alpha=255):
    class_names = [
        self.classes_dropdown.itemText(i)
        for i in range(self.classes_dropdown.count())
    ] if getattr(self, "classes_dropdown", None) is not None else []
    return get_color(int(class_id), max(1, len(class_names)), alpha=alpha, class_names=class_names)

def _draw_training_preview_labels(self, scene, image_file, orig_width, orig_height, scale, pad_left, pad_top, target_width, target_height):
    label_file = self.replace_extension_with_txt(image_file)
    if not label_file or not os.path.exists(label_file):
        return 0

    try:
        with open(label_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except UnicodeDecodeError:
        with open(label_file, "r", encoding="latin-1", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.warning(f"Could not load Train View labels: {e}")
        return 0

    expected_keypoints = self.keypoint_list.rowCount() if getattr(self, "keypoint_list", None) is not None else None
    polygon_preference = self._polygon_label_parse_preference() if hasattr(self, "_polygon_label_parse_preference") else None

    def tx(x):
        return float(x) * orig_width * scale + pad_left

    def ty(y):
        return float(y) * orig_height * scale + pad_top

    label_count = 0
    for line in lines:
        bbox = self._parse_training_eval_bbox(line, polygon_preference, expected_keypoints)
        if bbox is None:
            continue

        color = self._training_preview_class_color(bbox.class_id, alpha=255)
        fill = self._training_preview_class_color(bbox.class_id, alpha=45)
        pen = QPen(color, max(1, min(3, int(target_width / 320))))
        class_name = self.classes_dropdown.itemText(bbox.class_id) if 0 <= bbox.class_id < self.classes_dropdown.count() else f"cls_{bbox.class_id}"

        if bbox.segmentation or bbox.obb:
            points = bbox.segmentation if bbox.segmentation else bbox.obb
            polygon = QPolygonF([
                QPointF(tx(points[i]), ty(points[i + 1]))
                for i in range(0, len(points), 2)
            ])
            item = QGraphicsPolygonItem(polygon)
            item.setPen(pen)
            item.setBrush(QBrush(fill))
            item.setZValue(20 if bbox.segmentation else 30)
            item.setEnabled(False)
            scene.addItem(item)
            br = polygon.boundingRect()
            self._draw_training_preview_label_text(scene, class_name, br.left(), br.top() - 18, color)
            label_count += 1
            continue

        x1 = tx(bbox.x_center - bbox.width / 2)
        y1 = ty(bbox.y_center - bbox.height / 2)
        x2 = tx(bbox.x_center + bbox.width / 2)
        y2 = ty(bbox.y_center + bbox.height / 2)
        rect = QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized()
        rect_item = QGraphicsRectItem(rect)
        rect_item.setPen(pen)
        rect_item.setBrush(QBrush(fill))
        rect_item.setZValue(40)
        rect_item.setEnabled(False)
        scene.addItem(rect_item)
        self._draw_training_preview_label_text(scene, class_name, rect.left(), rect.top() - 18, color)

        if bbox.keypoints:
            train_points = []
            flags = []
            for kpt_index, (kx, ky, visible) in enumerate(bbox.keypoints):
                flags.append(int(visible))
                point = QPointF(tx(kx), ty(ky))
                train_points.append(point)

            for i1, i2 in getattr(KeypointDrawer, "SKELETON_CONNECTIONS", []) or DEFAULT_POSE_SKELETON:
                if i1 >= len(train_points) or i2 >= len(train_points):
                    continue
                if flags[i1] <= 0 or flags[i2] <= 0:
                    continue
                line_item = scene.addLine(
                    train_points[i1].x(),
                    train_points[i1].y(),
                    train_points[i2].x(),
                    train_points[i2].y(),
                    QPen(color, max(1, int(target_width / 420))),
                )
                line_item.setZValue(55)
                line_item.setEnabled(False)

            radius = max(2, min(5, int(target_width / 160)))
            for kpt_index, point in enumerate(train_points):
                if flags[kpt_index] <= 0:
                    continue
                try:
                    kpt_color = self.pose_keypoint_color_qcolor(kpt_index, alpha=255, fallback=color)
                except Exception:
                    kpt_color = color
                dot = QGraphicsEllipseItem(point.x() - radius, point.y() - radius, radius * 2, radius * 2)
                dot.setBrush(QBrush(kpt_color))
                dot.setPen(QPen(QColor(255, 255, 255, 210), 1))
                dot.setZValue(60)
                dot.setEnabled(False)
                scene.addItem(dot)

        label_count += 1

    return label_count

def _add_training_preview_badge(self, scene, text):
    badge = QGraphicsTextItem(text)
    badge.setDefaultTextColor(QColor("#f2f5f8"))
    font = badge.font()
    font.setPointSize(10)
    font.setBold(True)
    badge.setFont(font)
    badge.setPos(8, 6)
    badge.setZValue(5000)
    scene.addItem(badge)

def is_training_preview_always_on(self):
    return bool(
        getattr(self, "training_preview_always_on", False)
        or (
            isinstance(getattr(self, "settings", None), dict)
            and self.settings.get("trainingPreviewAlwaysOn", False)
        )
    )

def set_training_preview_always_on(self, checked, update_view=True):
    checked = bool(checked)
    self.training_preview_always_on = checked
    if isinstance(getattr(self, "settings", None), dict):
        self.settings["trainingPreviewAlwaysOn"] = checked
    self.queue_settings_save()

    if not update_view:
        return

    if checked:
        self.refresh_training_preview_if_enabled()
    elif getattr(self, "train_view_active", False):
        self.restore_original_training_view(disable_always_on=False)

def refresh_training_preview_if_enabled(self):
    if not self.is_training_preview_always_on():
        return False

    if getattr(self, "_loading_image", False):
        QTimer.singleShot(0, self.refresh_training_preview_if_enabled)
        return False

    image_file = getattr(self, "current_file", None)
    if not image_file or self.is_placeholder_file(image_file) or not os.path.exists(image_file):
        return False

    return self.show_training_size_preview(quiet=True)

def show_training_size_preview(self, imgsz=None, quiet=False):
    image_file = getattr(self, "current_file", None)
    if not image_file or self.is_placeholder_file(image_file) or not os.path.exists(image_file):
        if not quiet:
            QMessageBox.information(self, "Train View", "Open a dataset image before using Train View.")
        return False

    imgsz = self._parse_training_imgsz_value(imgsz or self.settings.get("trainingPreviewImgSize", self.settings.get("ultralyticsTrainImgSize", "640")))
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        if not quiet:
            QMessageBox.warning(self, "Train View", f"Could not read image:\n{image_file}")
        return False

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    orig_height, orig_width = image.shape[:2]
    scale, left, top, new_width, new_height = self._letterbox_preview_params(orig_width, orig_height, imgsz, imgsz)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR) if (new_width, new_height) != (orig_width, orig_height) else image.copy()
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=resized.dtype)
    canvas[top:top + new_height, left:left + new_width] = resized

    qimage = self.cv2_to_qimage(canvas)
    if qimage is None:
        return False

    self._loading_image = True
    try:
        self.train_view_active = True
        self.annotation_scene_active = False
        self.training_preview_size = (imgsz, imgsz)
        self.settings["trainingPreviewImgSize"] = str(imgsz)

        self.image = QPixmap.fromImage(qimage)
        self.processed_image = canvas.copy()
        scene = QGraphicsScene(0, 0, imgsz, imgsz)
        scene.setProperty("darkfusion_train_view_scene", True)
        scene.setProperty("darkfusion_source_image", os.path.abspath(image_file))
        pixmap_item = QGraphicsPixmapItem(self.image)
        pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        pixmap_item.setZValue(-1000)
        scene.addItem(pixmap_item)
        label_count = self._draw_training_preview_labels(scene, image_file, orig_width, orig_height, scale, left, top, imgsz, imgsz)
        waste = 1.0 - (new_width * new_height) / max(1.0, imgsz * imgsz)
        self._add_training_preview_badge(
            scene,
            f"Train View {imgsz}x{imgsz} | scale {scale:.3f} | pad {left},{top} | labels {label_count} | padding {waste * 100:.1f}%",
        )
        self.set_screen_view_scene_and_rect(scene)
        if self.screen_view.scene():
            self.screen_view.scene().update()
        self.screen_view.viewport().update()
        self.queue_settings_save()
        return True
    finally:
        self._loading_image = False

def restore_original_training_view(self, disable_always_on=False):
    if disable_always_on:
        self.training_preview_always_on = False
        if isinstance(getattr(self, "settings", None), dict):
            self.settings["trainingPreviewAlwaysOn"] = False
        self.queue_settings_save()

    self.train_view_active = False
    self.annotation_scene_active = True
    current_file = getattr(self, "current_file", None)
    if current_file and os.path.exists(current_file):
        self.display_image(current_file, rebuild_preview=False)
