import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
import os
import cv2
# Import necessary libraries and modules for the application
from typing import Optional
import logging
from logging.handlers import WatchedFileHandler
import concurrent.futures
import cProfile
from datetime import datetime
from typing import List, Tuple
import glob
import json
import threading
import queue
import random
import re
import shutil
import subprocess
import sys
import time
import pyautogui
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Type
from ultralytics import YOLO
import torch
import numpy as np
import yaml
import psutil
import GPUtil
from threading import Thread
import functools
from PIL import Image
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtCore import (QEvent, QModelIndex, QObject,
                          QRectF, QRunnable, Qt, QThread, QThreadPool,
                          QTimer, QUrl, pyqtSignal, pyqtSlot, QPointF,QModelIndex,Qt,QEvent,QPropertyAnimation, QEasingCurve,QRect,QProcess,QRectF,QMetaObject,QSettings,QLineF)
from PyQt5.QtGui import (QBrush, QColor, QFont, QImage, QImageReader,
                         QImageWriter, QMovie, QPainter, QPen,
                         QPixmap,  QStandardItem,
                         QStandardItemModel, QTransform, QLinearGradient,QIcon,QCursor,QStandardItemModel, QStandardItem,QMouseEvent,QKeyEvent,QPainterPath,QPolygonF,QPalette)
from PyQt5.QtWidgets import (QApplication, QFileDialog,
                             QGraphicsDropShadowEffect, QGraphicsItem,
                             QGraphicsPixmapItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsTextItem, QGraphicsView,
                             QLabel, QMessageBox, QProgressBar,
                             QTableWidgetItem, QColorDialog, QMenu,QSplashScreen,QTableView, QVBoxLayout,QWidget,QHeaderView,QStyledItemDelegate,QStyle,QTabWidget,QStyleOptionButton,QGraphicsPolygonItem,QGraphicsEllipseItem,QCheckBox,QAbstractItemView)
from PyQt5 import QtWidgets, QtGui
import weakref
from qt_material import apply_stylesheet, list_themes
from segment_anything import sam_model_registry, SamPredictor
import pybboxes as pbx
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from perlin_noise import PerlinNoise
from dinov5 import run_groundingdino
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
import webbrowser
import threading
from PIL import Image
from rectpack import newPacker
from PyQt5.QtWidgets import QAction
from sahi_predict_wrapperv5 import SahiPredictWrapper
from PyQt5.QtWidgets import (QDialog, QLineEdit,QPushButton, QSpinBox, QComboBox,QDoubleSpinBox)
from PyQt5.QtWidgets import QAbstractItemView
from PyQt5.QtCore import QItemSelectionModel
from PyQt5.QtCore import pyqtSlot, QModelIndex
from PyQt5.QtCore import QThread, pyqtSignal
import uuid
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from torchvision import transforms
from sklearn.cluster import MiniBatchKMeans
from yt_dlp import YoutubeDL
from PIL import Image
from collections import defaultdict
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import mediapipe as mp
import sip
import mss 
import codecs
from logging.handlers import WatchedFileHandler
from dotenv import load_dotenv, set_key
from torch.amp import autocast
import codecs
from logging.handlers import RotatingFileHandler
import torch
import torchvision.transforms.functional as F
import mediapipe as mp
import numpy as np
import mss
import logging
import os
import sys
import codecs
from logging.handlers import RotatingFileHandler
import speech_recognition as sr
from shapely.geometry import box as shapely_box
from shapely.geometry import Polygon

import math
mp_pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
# global functions
def get_color(class_id, num_classes, alpha=150):
    if class_id is None or num_classes == 0:
        return QColor(255, 255, 255, alpha)  # Default white
    hue_step = 360 / num_classes
    hue = (class_id * hue_step) % 360
    return QColor.fromHsv(int(hue), 255, 255, alpha)

# Setup logger configuration
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path)

MAX_LOG_LINE_LENGTH = 500  # adjust this as needed

class TruncatingFormatter(logging.Formatter):
    def format(self, record):
        original = super().format(record)
        if len(original) > MAX_LOG_LINE_LENGTH:
            return original[:MAX_LOG_LINE_LENGTH] + "..."  # truncate with ellipsis
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

    # ✂️ Truncate log file if it gets too large at startup (>64 KB)
    MAX_STARTUP_SIZE = 64 * 1024
    if os.path.exists(log_file) and os.path.getsize(log_file) > MAX_STARTUP_SIZE:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("")  # clear log at startup
        print("🧹 Cleared large log file at startup")

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




app = QApplication([])

# Set up a splash screen

class SplashScreen(QSplashScreen):
    def __init__(self, gif_path):
        pixmap = QPixmap()  # Create an empty QPixmap
        super(SplashScreen, self).__init__(pixmap, Qt.WindowStaysOnTopHint) # type: ignore

        # Create a QMovie from the provided GIF path and connect the frameChanged signal
        # to the set_frame method for dynamic updating.
        self.movie = QMovie(gif_path)
        self.movie.frameChanged.connect(self.set_frame)
        self.movie.start()

        # Create a QMediaPlayer for playing a sound and set the media content.
        self.sound_player = QMediaPlayer()
        self.sound_player.setMedia(QMediaContent(QUrl.fromLocalFile("sounds/Darkfusion.wav")))

        # Play the sound.
        self.sound_player.play()

    def set_frame(self):
        self.setPixmap(self.movie.currentPixmap())

# Create an instance of the SplashScreen class with the provided GIF path.
splash = SplashScreen("styles/gifs/darkfusion.gif")

# Show the splash screen.
splash.show()

# Allow Qt to process events and update the GUI.
app.processEvents()

# Set up a timer to close the splash screen after 2 seconds (2000 milliseconds).
timer = QTimer()
timer.singleShot(2000, splash.close) # type: ignore


class CustomGraphicsView(QGraphicsView):
    """
    A custom graphics view for displaying and interacting with images and bounding boxes.

    This class extends QGraphicsView to provide functionality for image annotation,
    including zooming, panning, drawing bounding boxes, handling right-click context
    menus, and managing UI interactions with checkboxes and sliders.

    Attributes:
        main_window (QMainWindow): Reference to the main GUI window for accessing UI elements.
        zoom_scale (float): The current zoom level of the view.
        show_crosshair (bool): Whether the crosshair overlay is displayed.
        sound_player (QMediaPlayer): Media player for playing sound effects.
        right_click_timer (QTimer): Timer for handling rapid deletion of bounding boxes.
        graphics_scene (QGraphicsScene): The scene that holds the images and annotations.
    """

    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window

        # Set up initial state and render settings
        self._setup_initial_state()
        self._setup_render_settings()
        self.selected_bbox = None
        # Create and configure sound player
        self.sound_player = QMediaPlayer()
        self.sound_player.setMuted(True)
        self.main_window.muteCheckBox.setChecked(True)  # type: ignore
        self.main_window.muteCheckBox.stateChanged.connect(self.mute_player)  # type: ignore

        # Initialize additional variables
        self.zoom_scale = 1.0
        self.fitInView_scale = 1.0
        self.auto_fit_enabled = True  # <-- Add this to control refitting behavior


        self.show_crosshair = False
        self.right_click_timer = QTimer()
        self.right_click_timer.setInterval(100)
        self.right_click_timer.timeout.connect(self.remove_item_under_cursor)

        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setBlurRadius(10)
        self.setGraphicsEffect(shadow)
        # Enable mouse tracking and configure view settings
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setRenderHints(
            QPainter.Antialiasing | 
            QPainter.SmoothPixmapTransform | 
            QPainter.HighQualityAntialiasing | 
            QPainter.TextAntialiasing
        )
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)        
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.setFrameShape(QGraphicsView.NoFrame)


    def _setup_initial_state(self):
        self.drawing = False
        self.drawing_obb = False               
        self.current_obb = None                   

        self.start_point = None
        self.current_bbox = None
        self.selected_bbox = None
        self.moving_view = False
        self.current_segmentation = None
        self.bboxes = []
        self.dragStartPos = None
        self.current_keypoint_drawer = None
        self.current_obb_drawer = None

        self.crosshair_position = QPointF()
        self.xy_lines_checkbox = self.main_window.xy_lines_checkbox  # type: ignore
        self.xy_lines_checkbox.toggled.connect(self.toggle_crosshair)
        self.main_window.crosshair_color.triggered.connect(self.pick_color)  # type: ignore 
        self.main_window.box_size.valueChanged.connect(self.update_bbox_size)  # type: ignore

        self.graphics_scene = QGraphicsScene(self)
        self.auto_fit_enabled = False

            
    def _setup_render_settings(self):
        self.setBackgroundBrush(QColor(0, 0, 0))  # Set the background color to black
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)


    def mute_player(self, state):
        """Mute or unmute the sound player based on checkbox state"""
        self.sound_player.setMuted(state == Qt.Checked)  # type: ignore

    def set_sound(self, sound_path):
        """Set the media content for sound effects"""
        self.sound_player.setMedia(QMediaContent(QUrl.fromLocalFile(sound_path)))

    def safe_remove_item(self, item):
        """Safely removes a bounding box from the scene and clears references."""
        if item and item.scene() and item.scene() == self.scene():
            if item == self.selected_bbox:
                self.selected_bbox = None  # ✅ Clear reference before deleting
            self.scene().removeItem(item)
            logger.debug("✅ Successfully removed item from scene.")
        else:
            logger.warning("⚠️ Attempted to remove an item that does not belong to the current scene or is None.")


    def contextMenuEvent(self, event):
        menu = QMenu(self)
        right_click_pos = self.mapToScene(event.pos())
        last_bbox = self._get_last_drawn_bbox()

        if last_bbox and last_bbox.contains(right_click_pos):
            delete_action = menu.addAction("Delete Last Drawn Box")
            action = menu.exec_(event.globalPos())

            if action == delete_action:
                self._remove_last_drawn_bbox(last_bbox)

    def remove_item_under_cursor(self):
        """Rapidly deletes BoundingBoxDrawer or SegmentationDrawer items under the cursor."""
        cursor_pos = self.mapToScene(self.mapFromGlobal(QCursor.pos()))

        for item in reversed(self.scene().items(cursor_pos)):
            if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer)):
                self._play_sound_and_remove_bbox(item)

                # Specifically handle SegmentationDrawer differently if needed
                if isinstance(item, SegmentationDrawer):
                    item.remove_self()

                # Immediately break after deleting one item to prevent accidental multiple deletions
                break



    def _get_last_drawn_bbox(self):
        if self.bboxes:
            return self.bboxes[-1]

    def _remove_last_drawn_bbox(self, bbox):
        """Remove the last drawn bounding box safely."""
        if bbox:
            self.safe_remove_item(bbox)
            self.safe_remove_item(bbox.class_name_item)  # Ensure the text label is removed too
            self.bboxes.remove(bbox)
            self.save_bounding_boxes()

    def update_bbox_size(self, value):
        BoundingBoxDrawer.MIN_SIZE = value
        BoundingBoxDrawer.MAX_SIZE = value

    def pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.crosshair_color_rgb = color.getRgb()

    def wheelEvent(self, event):
        logging.info("Wheel event triggered.")

        # Default: Zooming in and out
        zoomInFactor = 1.15
        zoomOutFactor = 1 / zoomInFactor
        oldPos = self.mapToScene(event.pos())

        # Enable SmoothPixmapTransform for better zoom rendering
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if event.angleDelta().y() > 0:  # Zoom in
            zoomFactor = zoomInFactor
            self.zoom_scale *= zoomInFactor
            logging.info("Zooming in; new zoom scale: %s", self.zoom_scale)
        else:  # Zoom out
            # Prevent zooming out too far
            min_zoom_scale = max(self.fitInView_scale, 0.1)
            if self.zoom_scale * zoomOutFactor < min_zoom_scale:
                logging.warning("Zoom out prevented to maintain minimum zoom scale.")
                self.setRenderHint(QPainter.SmoothPixmapTransform, False)
                return

            zoomFactor = zoomOutFactor
            self.zoom_scale *= zoomOutFactor
            logging.info("Zooming out; new zoom scale: %s", self.zoom_scale)

        # Scale the view smoothly
        self.scale(zoomFactor, zoomFactor)
        newPos = self.mapToScene(event.pos())
        delta = newPos - oldPos
        self.centerOn(self.mapToScene(self.viewport().rect().center()) - delta)

        # Disable SmoothPixmapTransform after zooming
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit_enabled and self.scene():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            self.fitInView_scale = self.transform().m11()

    def reset_zoom(self):
        """Manually reset zoom to fit in view."""
        if self.scene():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            self.fitInView_scale = self.transform().m11()
            self.zoom_scale = 1.0


    def mousePressEvent(self, event):
        if self.main_window.edit_mode_active():
            if not self.main_window.is_segmentation_mode():
                logger.warning("⚠️ Edit mode is only available in segmentation mode.")
                return
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        self.last_mouse_pos = scene_pos

        # ========================================
        # ✅ 1️⃣ OBB Mode Logic
        # ========================================
        if self.main_window.is_obb_mode():
            # 🔁 Begin or continue drawing an OBB
            if event.button() == Qt.LeftButton:
                if not self.current_obb_drawer:
                    print("🟢 Starting OBB Drawing")
                    self.current_obb_drawer = OBBDrawer(self.main_window)
                    self.scene().addItem(self.current_obb_drawer)

                if not self.current_obb_drawer.centerline_start:
                    self.current_obb_drawer.centerline_start = scene_pos
                    self.current_obb_drawer.current_state = "defining_centerline"
                    self.drawing_obb = True  # ✅ Add this!
                    print(f"🔹 OBB Centerline Start: {scene_pos}")
                    color = get_color(
                        self.current_obb_drawer.class_id,
                        self.main_window.classes_dropdown.count(),
                        alpha=255
                    )
                    pen = QPen(color, 2)

                    self.current_obb_drawer.centerline_item = self.scene().addLine(
                        scene_pos.x(), scene_pos.y(), scene_pos.x(), scene_pos.y(),
                        pen
                    )

                    return

            elif event.button() == Qt.RightButton:
                # 🔥 Delete the current OBB if it exists
                if self.current_obb_drawer:
                    print("🗑️ Right-click: Removing current OBB regardless of state or mode.")
                    self.current_obb_drawer.remove_self()
                    self.current_obb_drawer = None
                    return

                # Right click fallback
                self._handle_right_button_press(event)
                return

            elif event.button() == Qt.MiddleButton:
                if self.current_obb_drawer and self.current_obb_drawer.current_state == "bb_defined":
                    print("🚚 Starting to move OBB")
                    self.current_obb_drawer.start_moving(scene_pos)
                    return

            return  # Stop here if in OBB mode (don’t leak into other modes)


        # ✅ KEYPOINT MODE
        if self.main_window.is_keypoint_mode():
            if event.button() == Qt.RightButton:
                self._handle_right_button_press(event)
                return
            clicked_pos = self.mapToScene(event.pos())
            found_box = False

            for item in self.scene().items(clicked_pos):
                if isinstance(item, BoundingBoxDrawer):
                    is_new_selection = item != self.selected_bbox

                    # 👇 Deselect previous box and select new one
                    for bbox in self.bboxes:
                        bbox.set_selected(False)
                    item.set_selected(True)
                    self.selected_bbox = item
                    found_box = True

                    # 👇 Create keypoint drawer if needed
                    if not hasattr(item, 'keypoint_drawer') or item.keypoint_drawer is None:
                        keypoint_drawer = KeypointDrawer(
                            points=[],
                            main_window=self.main_window,
                            class_id=item.class_id,
                            file_name=self.main_window.label_file or self.main_window.current_file
                        )
                        keypoint_drawer.setParentItem(item)
                        item.keypoint_drawer = keypoint_drawer
                        self.scene().addItem(keypoint_drawer)

                        bbox_data = getattr(item, "bbox_data", None)
                        if bbox_data and bbox_data.keypoints:
                            keypoint_drawer.points = [(x, y) for x, y, v in bbox_data.keypoints]
                            keypoint_drawer.visibility_flags = [int(v) for x, y, v in bbox_data.keypoints]
                            keypoint_drawer.update_points()
                            keypoint_drawer.update_class_name_item()
                            self.scene().update() 
                    # ✅ Only sync checkboxes if auto_sync_checkbox is checked
                    if is_new_selection:
                        drawer = item.keypoint_drawer
                        if self.main_window.auto_sync_checkbox.isChecked():
                            flags = drawer.visibility_flags
                            for i in range(self.main_window.keypoint_list.rowCount()):
                                v = flags[i] if i < len(flags) else 0
                                visible_cb = self.main_window.keypoint_list.cellWidget(i, 3)
                                ignore_cb = self.main_window.keypoint_list.cellWidget(i, 4)

                                if v == 0:
                                    if ignore_cb: ignore_cb.setChecked(True)
                                    if visible_cb: visible_cb.setChecked(False)
                                elif v == 1:
                                    if ignore_cb: ignore_cb.setChecked(False)
                                    if visible_cb: visible_cb.setChecked(False)
                                elif v == 2:
                                    if ignore_cb: ignore_cb.setChecked(False)
                                    if visible_cb: visible_cb.setChecked(True)

                        # ⏩ Fast label: start from keypoint 0 on new box
                        if self.main_window.fast_label_checkbox.isChecked():
                            self.main_window.selected_keypoint_row = 0
                            self.main_window.keypoint_list.selectRow(0)
                        else:
                            self.main_window.selected_keypoint_row = None
                            self.main_window.keypoint_list.clearSelection()

                    break

            if not found_box:
                logger.warning("⚠️ No bounding box selected under cursor.")
                return

            bbox = self.selected_bbox
            drawer = bbox.keypoint_drawer
            row = self.main_window.selected_keypoint_row

            # 🎯 MOVE EXISTING POINT
            if row is not None:
                ignore_cb = self.main_window.keypoint_list.cellWidget(row, 4)
                if ignore_cb and ignore_cb.isChecked():
                    logger.info(f"🚫 Ignored keypoint row {row} — recording as (0.0, 0.0, 0)")

                    while len(drawer.points) <= row:
                        drawer.points.append((0.0, 0.0))

                    while len(drawer.visibility_flags) <= row:
                        drawer.visibility_flags.append(0)

                    drawer.points[row] = (0.0, 0.0)
                    drawer.visibility_flags[row] = 0  # visibility flag for ignored

                    # Optional: visually update labels and state
                    drawer.update_points()
                    drawer.update_class_name_item()

                    # 👉 Auto-advance row
                    next_row = row + 1
                    if next_row < self.main_window.keypoint_list.rowCount():
                        self.main_window.selected_keypoint_row = next_row
                        self.main_window.keypoint_list.selectRow(next_row)
                    else:
                        self.main_window.selected_keypoint_row = None
                        self.main_window.keypoint_list.clearSelection()

                    # 💾 Save updated annotation file
                    self.main_window.save_bounding_boxes(
                        self.main_window.label_file or self.main_window.current_file,
                        self.main_window.image.width(),
                        self.main_window.image.height()
                    )

                    return  # 🚫 Don't place a visual point

                # ✅ Regular point logic
                if bbox.rect().contains(clicked_pos):
                    x = clicked_pos.x() / self.main_window.image.width()
                    y = clicked_pos.y() / self.main_window.image.height()

                    while len(drawer.points) <= row:
                        drawer.points.append((0.0, 0.0))
                    drawer.points[row] = (x, y)

                    visible_cb = self.main_window.keypoint_list.cellWidget(row, 3)
                    v = 2 if visible_cb and visible_cb.isChecked() else 1

                    while len(drawer.visibility_flags) <= row:
                        drawer.visibility_flags.append(0)
                    drawer.visibility_flags[row] = v

                    # ✅ 🔥 Force the visual refresh immediately
                    drawer.update_points()
                    drawer.update_class_name_item()

                    self.main_window.keypoint_list.selectRow(row)
                    self.main_window.clear_keypoint_highlight()
                    self.main_window.highlight_keypoint_row(row)

                    next_row = row + 1
                    if next_row < self.main_window.keypoint_list.rowCount():
                        self.main_window.selected_keypoint_row = next_row
                        self.main_window.keypoint_list.selectRow(next_row)
                    else:
                        self.main_window.selected_keypoint_row = None
                        self.main_window.keypoint_list.clearSelection()

                    self.main_window.save_bounding_boxes(
                        self.main_window.label_file or self.main_window.current_file,
                        self.main_window.image.width(),
                        self.main_window.image.height()
                    )

                else:
                    logger.warning("⚠️ Clicked outside the selected bounding box.")
                return
            # 🆕 PLACE NEW POINT (when no row is selected)
            if self.selected_bbox and self.selected_bbox.rect().contains(clicked_pos):
                # 👇 If fast mode is on and no row selected, assign first unused row
                if self.main_window.fast_label_checkbox.isChecked() and self.main_window.selected_keypoint_row is None:
                    self.main_window.selected_keypoint_row = 0
                    self.main_window.keypoint_list.selectRow(0)

                drawer.append_point(clicked_pos)
                logger.info("✅ Keypoint added inside bounding box.")
                self.scene().invalidate(self.scene().sceneRect(), QGraphicsScene.BackgroundLayer)
                self.viewport().update()

                logger.warning("⚠️ Tried to add keypoint outside selected bounding box.")
            return
        # 🧭 FALLBACK MODES
        if event.button() == Qt.RightButton:
            self._handle_right_button_press(event)
            return  # 🛑 Prevent other actions on right-click

        if self.main_window.is_segmentation_mode():
            self._start_segmentation_drawing(event)
        elif self.main_window.is_keypoint_mode():
            self._start_keypoint_drawing(event)
        else:
            self._start_drawing(event)

        self.update()

    def mouseMoveEvent(self, event):
        if self.main_window.edit_mode_active():
            super().mouseMoveEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())

        # ========================================
        # ✅ 1️⃣ OBB Mode (Live Drag Logic)
        # ========================================
        if self.main_window.is_obb_mode() and self.current_obb_drawer:
            state = self.current_obb_drawer.current_state

            if state == "defining_centerline":
                self.current_obb_drawer.centerline_end = scene_pos
                self.current_obb_drawer.drawing_in_progress = True
                if self.current_obb_drawer.centerline_item:
                    self.current_obb_drawer.centerline_item.setLine(
                        self.current_obb_drawer.centerline_start.x(),
                        self.current_obb_drawer.centerline_start.y(),
                        scene_pos.x(),
                        scene_pos.y()
                    )
                return

            elif state == "awaiting_width_point":
                self.current_obb_drawer.update_preview(scene_pos)
                return

            elif state == "moving_bb":
                self.current_obb_drawer.update_position(scene_pos)
                return

        # ========================================
        # ✅ 2️⃣ Segmentation Drawing
        # ========================================
        if self.main_window.is_segmentation_mode() and self.current_segmentation:
            self._handle_segmentation_drawing(event)
            return

        # ========================================
        # ✅ 3️⃣ Keypoint Mode
        # ========================================
        if self.main_window.is_keypoint_mode() and self.selected_bbox:
            if event.buttons() & Qt.LeftButton:
                if self.current_keypoint_drawer:
                    self.current_keypoint_drawer.update_point(scene_pos)
            return

        # ========================================
        # ✅ 4️⃣ Bounding Box Drawing
        # ========================================
        if self.drawing and self.current_bbox:
            self._handle_drawing_bbox(event)
            return

        # ========================================
        # ✅ 5️⃣ View Movement
        # ========================================
        if self.moving_view:
            self._handle_moving_view(event)
            return

        # ========================================
        # ✅ 6️⃣ Crosshair
        # ========================================
        if self.show_crosshair:
            self._update_crosshair(event.pos())

        # ========================================
        # ✅ 7️⃣ Highlighting Selected BBox
        # ========================================
        if self.selected_bbox and not self.drawing:
            if sip.isdeleted(self.selected_bbox) or self.selected_bbox.scene() is None:
                logger.warning("⚠️ Attempted to move a deleted bounding box. Resetting selection.")
                self.selected_bbox = None
            elif event.buttons() & Qt.LeftButton:
                self.selected_bbox.setPen(QPen(QColor(0, 255, 0), 2))
            return

        # ========================================
        # ✅ 8️⃣ Fallback — let Qt handle anything else
        # ========================================
        super().mouseMoveEvent(event)



    def mouseReleaseEvent(self, event):
        """
        Handles mouse release events for OBB drawing, segmentation, and bounding boxes.
        """
        if self.main_window.edit_mode_active():
            print("📝 Edit Mode Active")
            super().mouseReleaseEvent(event)
            return

        # ========================================
        # ✅ 1️⃣ OBB Mode Logic (Finalization)
        # ========================================
        if self.main_window.is_obb_mode() and self.current_obb_drawer:
            scene_pos = self.mapToScene(event.pos())

            if event.button() == Qt.LeftButton:
                if self.current_obb_drawer.current_state == "defining_centerline":
                    print("✏️ Centerline defined. Waiting for width adjustment...")
                    self.current_obb_drawer.centerline_end = scene_pos
                    self.current_obb_drawer.current_state = "awaiting_width_point"
                    self.drawing_obb = False  # we are done dragging now

                    if self.current_obb_drawer.centerline_item:
                        self.scene().removeItem(self.current_obb_drawer.centerline_item)
                        self.current_obb_drawer.centerline_item = None
                    return

                elif self.current_obb_drawer.current_state == "awaiting_width_point":
                    print("✅ Finalizing OBB...")
                    self.current_obb_drawer.finalize()
                    self.current_obb_drawer = None
                    return

            elif event.button() == Qt.MiddleButton:
                print("🛑 Stopped Moving OBB")
                self.current_obb_drawer.stop_moving()
                return



        # ========================================
        # ✅ 2️⃣ Segmentation Finalization
        # ========================================
        if event.button() == Qt.LeftButton and self.drawing:
            if self.main_window.is_segmentation_mode() and self.current_segmentation:
                if len(self.current_segmentation.points) >= 3:
                    print("✅ Finalizing Segmentation...")
                    self.current_segmentation.finalize()
                    self.scene().addItem(self.current_segmentation)
                else:
                    logger.warning("⚠️ Segmentation must have at least 3 points.")
                    if self.current_segmentation.scene():
                        self.scene().removeItem(self.current_segmentation)
                self.current_segmentation = None
                self.drawing = False

            # ✅ 3️⃣ Finalize Normal BBox
            elif self.current_bbox:
                print("✅ Finalizing Bounding Box...")
                self._finalize_bbox()
                self.drawing = False

        # ========================================
        # ✅ 4️⃣ Right Click Stop
        # ========================================
        if event.button() == Qt.RightButton:
            self.right_click_timer.stop()

        # ========================================
        # ➡️ Fallback: Default Qt Behavior
        # ========================================
        super().mouseReleaseEvent(event)
        self.setCursor(Qt.ArrowCursor)



    def _start_keypoint_drawing(self, event):
        if not self.selected_bbox:
            logger.warning("⚠️ No bounding box selected for keypoint drawing.")
            return

        if not self.main_window.is_keypoint_mode():
            logger.warning("⚠️ Ignored: not in keypoint mode.")
            return

        click_pos = self.mapToScene(event.pos())
        if not self.selected_bbox.rect().contains(click_pos):
            logger.warning("⚠️ Click outside the selected bounding box.")
            return

        if not hasattr(self.selected_bbox, 'keypoint_drawer') or self.selected_bbox.keypoint_drawer is None:
            self.selected_bbox.keypoint_drawer = KeypointDrawer(
                points=[],
                main_window=self.main_window,
                class_id=self.selected_bbox.class_id,
                file_name=self.main_window.current_file
            )
            self.scene().addItem(self.selected_bbox.keypoint_drawer)

        self.selected_bbox.keypoint_drawer.append_point(click_pos)
        logger.info("✅ Keypoint added inside bounding box.")






    def _start_segmentation_drawing(self, event):
        """Starts drawing a segmentation mask just like bounding boxes."""
        self.drawing = True
        self.start_point = self.mapToScene(event.pos())

        # Ensure we pass an empty list of points, not self.main_window!
        self.current_segmentation = SegmentationDrawer(
            points=[],  # ✅ Fix: Pass an actual list
            main_window=self.main_window, 
            class_id=self.main_window.get_current_class_id()
        )

        self.scene().addItem(self.current_segmentation)


    def _start_drawing(self, event):
        self.drawing = True
        self.start_point = self.mapToScene(event.pos())

        self.setCursor(Qt.CrossCursor)  # ✅ Explicitly set cursor here

        self.current_bbox = BoundingBoxDrawer(
            self.start_point.x(), self.start_point.y(), 0, 0,
            main_window=self.main_window, class_id=self.main_window.get_current_class_id()
        )

        self.current_bbox.set_z_order(bring_to_front=True)
        self.scene().addItem(self.current_bbox)



    def _bring_to_front(self, item):
        """Ensure the selected bounding box moves to the top layer, but only cycle through boxes under cursor."""
        if not isinstance(item, BoundingBoxDrawer):
            return

        scene_pos = item.scenePos()
        items_under_cursor = [
            i for i in self.scene().items(scene_pos) if isinstance(i, BoundingBoxDrawer)
        ]

        if not items_under_cursor:
            return

        # Sort boxes under cursor by Z-value (lowest first)
        items_under_cursor.sort(key=lambda x: x.zValue())

        # If the clicked box is at the top, cycle to the next box below it
        if items_under_cursor[-1] == item and len(items_under_cursor) > 1:
            next_box = items_under_cursor[-2]
            self.selected_bbox = next_box
        else:
            self.selected_bbox = item

        # Update Z-values for only the relevant boxes under cursor
        base_z = 1.0
        for box in items_under_cursor:
            if box != self.selected_bbox:
                box.setZValue(base_z)
                base_z += 0.1

        self.selected_bbox.setZValue(base_z + 1.0)
        self.selected_bbox.setOpacity(1.0)

        # Ensure only the affected area updates
        update_rect = self.selected_bbox.boundingRect().united(
            max((box.boundingRect() for box in items_under_cursor), key=lambda r: r.width() * r.height())
        )
        self.scene().update(update_rect)


    def _handle_right_button_press(self, event):
        click_pos = self.mapToScene(event.pos())
        tolerance = 10

        if self.main_window.rapid_del_checkbox.isChecked():
            self.right_click_timer.start()
            return

        # Go in reverse order to hit topmost items first
        for item in reversed(self.scene().items(click_pos)):

            # ✅ Delete segmentation
            if isinstance(item, SegmentationDrawer):
                self._play_sound_and_remove_bbox(item)
                item.remove_self()
                break

            # ✅ Delete regular bounding box
            elif isinstance(item, BoundingBoxDrawer):
                if item.rect().adjusted(-tolerance, -tolerance, tolerance, tolerance).contains(click_pos):
                    self._play_sound_and_remove_bbox(item)
                    break

            # ✅ Delete OBB in segmentation or OBB mode
            elif isinstance(item, OBBDrawer):
                if self.main_window.is_segmentation_mode() or self.main_window.is_obb_mode():
                    print("🗑️ Right-click OBB in segmentation/obb mode")
                    item.remove_self()
                    break



    def _play_sound_and_remove_bbox(self, item):
        """Plays sound effect and removes the bounding box safely."""
        if sip.isdeleted(item):  # ✅ Prevent operating on deleted items
            logger.warning("⚠️ Attempted to remove a deleted bounding box.")
            return

        self.set_sound('sounds/shotgun.wav')
        self.sound_player.play()

        if item == self.selected_bbox:
            self.selected_bbox = None  # ✅ Clear reference before removing

        self.safe_remove_item(item)


    def _re_add_bounding_boxes(self):
        for bbox in self.bboxes:
            if bbox.scene() is None:  # Check if it's already in the scene
                self.scene().addItem(bbox)
                bbox.setZValue(0.5)
        if self.selected_bbox:
            self.selected_bbox.setZValue(1)



    def _handle_segmentation_drawing(self, event):
        """Handles dynamic polygon drawing while dragging, clamping points within image boundaries."""
        try:
            end_point = self.mapToScene(event.pos())

            # Get image boundaries
            img_width = self.scene().width()
            img_height = self.scene().height()

            # Clamp point inside image boundaries
            clamped_x = max(0, min(img_width - 1, end_point.x()))
            clamped_y = max(0, min(img_height - 1, end_point.y()))
            clamped_point = QPointF(clamped_x, clamped_y)

            # Append clamped point
            self.current_segmentation.append_point(clamped_point)

            # Update the polygon dynamically
            self.current_segmentation.update_polygon()

        except RuntimeError as e:
            logger.error(f"Error while drawing segmentation: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in segmentation drawing: {e}")


    def _update_crosshair(self, pos):
        self.crosshair_position = pos
        self.viewport().update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        if self.show_crosshair:
            painter.setPen(QColor(*self.crosshair_color_rgb) if hasattr(self, 'crosshair_color_rgb') else Qt.yellow)
            center_x = int(self.crosshair_position.x())
            center_y = int(self.crosshair_position.y())
            painter.drawLine(center_x, self.viewport().rect().top(), center_x, self.viewport().rect().bottom())
            painter.drawLine(self.viewport().rect().left(), center_y, self.viewport().rect().right(), center_y)

        painter.end()


    def _handle_moving_view(self, event):
        if self.last_mouse_position is None:
            self.last_mouse_position = event.pos()
        else:
            self._update_scrollbars(event)

    def _update_scrollbars(self, event):
        delta = event.pos() - self.last_mouse_position
        self.last_mouse_position = event.pos()
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    def _handle_drawing_bbox(self, event):
        try:
            end_point = self.mapToScene(event.pos())
            start_point = np.array([self.start_point.x(), self.start_point.y()], dtype=int)
            end_point_np = np.array([end_point.x(), end_point.y()], dtype=int)

            scene_rect = self.sceneRect()
            min_point = np.clip(
                np.minimum(start_point, end_point_np),
                [scene_rect.left(), scene_rect.top()],
                [scene_rect.right(), scene_rect.bottom()]
            ).astype(int)

            max_point = np.clip(
                np.maximum(start_point, end_point_np),
                [scene_rect.left(), scene_rect.top()],
                [scene_rect.right(), scene_rect.bottom()]
            ).astype(int)

            # ✅ Remove live snapping here!
            dimensions = np.clip(max_point - min_point, BoundingBoxDrawer.MIN_SIZE, None)
            x, y = min_point
            width, height = dimensions

            self.current_bbox.setRect(x, y, width, height)

        except RuntimeError as e:
            logger.error(f"Error drawing bbox: {e}")
        except Exception as e:
            logger.error(f"bbox error: {e}")



    def preprocess_roi_for_contours(self, roi):
        blurred = cv2.GaussianBlur(roi, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges
    def snap_bbox_to_contour(self, gray_img, min_point, max_point):
        x1, y1 = min_point
        x2, y2 = max_point

        # Clamp to valid image dimensions
        img_h, img_w = gray_img.shape[:2]
        x1, y1 = np.clip([x1, y1], [0, 0], [img_w - 1, img_h - 1])
        x2, y2 = np.clip([x2, y2], [0, 0], [img_w - 1, img_h - 1])

        roi = gray_img[y1:y2, x1:x2]
        processed_roi = self.preprocess_roi_for_contours(roi)

        contours, _ = cv2.findContours(processed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            snapped_min_point = np.array([x1 + x, y1 + y])
            snapped_max_point = snapped_min_point + [w, h]
            return snapped_min_point, snapped_max_point

        # If no contours found, return original points
        return min_point, max_point

    def _finalize_bbox(self):
        try:
            if (
                self.current_bbox
                and self.current_bbox.rect().width() >= BoundingBoxDrawer.MIN_SIZE
                and self.current_bbox.rect().height() >= BoundingBoxDrawer.MIN_SIZE
            ):
                if self.main_window.outline_Checkbox.isChecked() and hasattr(self.main_window, 'processed_image'):
                    gray_img = self.main_window.processed_image
                    if len(gray_img.shape) == 3 and gray_img.shape[2] == 3:
                        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
                         
                    rect = self.current_bbox.rect()
                    min_point = np.array([int(rect.x()), int(rect.y())], dtype=int)
                    max_point = np.array([int(rect.x() + rect.width()), int(rect.y() + rect.height())], dtype=int)

                    min_point, max_point = self.snap_bbox_to_contour(gray_img, min_point, max_point)

                    x, y = min_point
                    w, h = max_point - min_point
                    self.current_bbox.setRect(x, y, w, h)
                    self.current_bbox.update_class_name_item()  # force label to snap too


                self._save_and_play_sound()
            else:
                if self.current_bbox.scene():
                    self.scene().removeItem(self.current_bbox)
                    if hasattr(self.current_bbox, "class_name_item") and self.current_bbox.class_name_item.scene():
                        self.scene().removeItem(self.current_bbox.class_name_item)
        except RuntimeError as e:
            logger.error(f"Runtime error finalizing bounding box: {e}")
        except Exception as e:
            logger.error(f"Unexpected error finalizing bounding box: {e}")
        finally:
            self.drawing = False
            self.current_bbox = None
            self.clear_selection()
            self.setCursor(Qt.ArrowCursor)


    def _get_bbox_coordinates(self, end_point):
        x = max(0, min(self.start_point.x(), end_point.x()))
        y = max(0, min(self.start_point.y(), end_point.y()))
        return x, y

    def _get_bbox_dimensions(self, end_point, x, y):
        width = min(max(BoundingBoxDrawer.MIN_SIZE, abs(self.start_point.x() - end_point.x())), self.scene().width() - x)
        height = min(max(BoundingBoxDrawer.MIN_SIZE, abs(self.start_point.y() - end_point.y())), self.scene().height() - y)
        return width, height



    def _handle_left_button_release(self, event):
        if self.selected_bbox:
            self._update_selected_bbox()
        if self.drawing and self.current_bbox:
            self._handle_drawing_current_bbox()
        if event.modifiers() == Qt.ControlModifier and self.selected_bbox:
            self._reset_after_bbox_move()
        elif self.selected_bbox:
            self._reset_selected_bbox()

    def _update_selected_bbox(self):
        self.selected_bbox.update_position_and_size()
        self.selected_bbox.update_bbox()
        self.main_window.save_bounding_boxes(
            self.main_window.label_file, self.scene().width(), self.scene().height())

    def _handle_drawing_current_bbox(self):
        try:
            if self.current_bbox and self.current_bbox.rect().width() >= 6 and self.current_bbox.rect().height() >= 6:
                self._save_and_play_sound()
            else:
                self._remove_current_bbox_items()
        except RuntimeError as e:
            logger.error(f"An error occurred: {e}")
        finally:
            self.drawing = False
            self.current_bbox = None
            self.clear_selection()

    def _save_and_play_sound(self):
        self.main_window.set_selected(None)
        self.main_window.remove_bbox_from_classes_txt(self.current_bbox)
        self.main_window.save_bounding_boxes(
            self.main_window.label_file, self.main_window.screen_view.scene().width(),
            self.main_window.screen_view.scene().height())
        self.set_sound('sounds/createcock.wav')
        self.sound_player.play()

    def _remove_current_bbox_items(self):
        self.scene().removeItem(self.current_bbox)
        self.scene().removeItem(self.current_bbox.class_name_item)

    def _reset_after_bbox_move(self):
        self.moving_view = False
        self.moving_bbox = False
        self.initial_mouse_pos = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.clear_selection()

    def _reset_selected_bbox(self):
        self.selected_bbox.set_selected(False)
        self.selected_bbox = None
        self.clear_selection()

    def clear_selection(self):
        """Safely clears the selection without crashing."""
        if self.selected_bbox:
            if not sip.isdeleted(self.selected_bbox):
                if self.selected_bbox.boundingRect().isValid():
                    self.selected_bbox.set_selected(False)
            self.selected_bbox = None
        


    def toggle_crosshair(self, checked):
        self.show_crosshair = checked
        self.viewport().update()



class BoundingBoxDrawer(QGraphicsRectItem):
    MIN_SIZE = 6
    MAX_SIZE = 100
    """
    A graphical bounding box representation for object detection in a GUI.

    This class extends QGraphicsRectItem to provide interactive bounding boxes that 
    can be moved, resized, highlighted, and labeled. It allows users to visualize, 
    modify, and manage bounding boxes within an annotation tool.

    Attributes:
        MIN_SIZE (int): Minimum size constraint for bounding boxes.
        MAX_SIZE (int): Maximum size constraint for bounding boxes.
        unique_id (int, optional): A unique identifier for the bounding box.
        main_window (QMainWindow): Reference to the main GUI window for accessing UI elements.
        class_id (int): The assigned class ID of the bounding box.
        confidence (float, optional): The confidence score of the detection (if available).
        dragStartPos (QPointF, optional): Stores the initial mouse position when dragging.
        final_pos (QPointF, optional): Stores the final position after dragging.
        hover_opacity (float): Opacity level when hovering over the bounding box.
        normal_opacity (float): Default opacity level when not hovered.
        flash_color (QColor): The color used when flashing.
        alternate_flash_color (QColor): The alternate flashing color.
        flash_timer (QTimer): Timer to handle flashing animations.
        scroll_timer (QTimer): Timer to stop flashing after a duration.
    """    
    def __init__(self, x, y, width, height, main_window, class_id=None, confidence=None, unique_id=None):
        super().__init__(x, y, width, height)
        self.normalize_rect()
        self.set_z_order(bring_to_front=True)  
        self.unique_id = unique_id
        self.main_window = main_window
        self.class_id = 0 if class_id is None else class_id
        self.confidence = confidence
        self.dragStartPos = None
        self.final_pos = None
        self.keypoint_drawer = None        
        # Cache color and pen
        self._class_color = get_color(self.class_id, main_window.classes_dropdown.count())
        self._pen = QPen(self._class_color, 2)
        
        # Initialize graphics properties
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsFocusable)
        self.setPen(self._pen)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)        
        # Initialize text properties
        self._text_pos = QPointF(5, 5)
        self._last_rect = self.rect()
        self.class_name_item = QGraphicsTextItem(self)
        self.update_class_name_item()
        
        # Initialize hover properties
        self.setAcceptHoverEvents(True)
        self.hover_opacity = 1.0
        self.normal_opacity = 0.6
        self._is_hovered = False
        self.setOpacity(self.normal_opacity)
        
        # Initialize flash properties
        self.flash_color = QColor(255, 0, 0)
        self.alternate_flash_color = QColor(0, 0, 255)
        self.flash_timer = QTimer()
        self.flash_timer.timeout.connect(self.toggle_flash_color)
        self.scroll_timer = QTimer()
        self.scroll_timer.setSingleShot(True)
        self.scroll_timer.timeout.connect(self.stop_flashing)
        # Set initial visibility
        self.class_name_item.setVisible(not self.main_window.hide_label_checkbox.isChecked())
        self.main_window.font_size_slider.valueChanged.connect(self.update_class_name_item_font)
    def set_z_order(self, bring_to_front=False):
        """ Adjust Z-values while preserving initial order. """
        if bring_to_front:
            self.setZValue(1.0)  # Bring to front
        elif self.zValue() < 1.0:  #  `zValue()` exists and is correct
            self.setZValue(0.5)

    def normalize_rect(self):
        """Ensure bounding box has positive width/height and updates rect safely."""
        raw_rect = self.rect()
        normalized = QRectF(
            min(raw_rect.left(), raw_rect.right()),
            min(raw_rect.top(), raw_rect.bottom()),
            abs(raw_rect.width()),
            abs(raw_rect.height())
        )
        self.setRect(normalized)

    def hoverEnterEvent(self, event):
        if self._is_hovered:
            return

        scene = self.scene()
        if scene:
            scene_pos = self.mapToScene(event.pos())

            # Get all bounding boxes under the cursor
            items_under_cursor = [
                item for item in scene.items(scene_pos)
                if isinstance(item, BoundingBoxDrawer)
            ]

            # Sort boxes by Z-value (top-most first)
            items_under_cursor.sort(key=lambda x: x.zValue(), reverse=True)

            # If the hovered box is at the top, bring it forward

            if items_under_cursor and items_under_cursor[0] == self:
                self.set_z_order(bring_to_front=True)
            else:
                # If another box is already on top, push the current box slightly forward
                self.setZValue(self.zValue() + 0.1)

        self.setOpacity(self.hover_opacity)
        super().hoverEnterEvent(event)
        self._is_hovered = True

        # Cache scene items to avoid repeated lookups
        scene = self.scene()
        if not scene:
            return
            
        # Get boxes at position more efficiently
        scene_pos = self.mapToScene(event.pos())
        boxes = [item for item in scene.items(scene_pos)
                if isinstance(item, BoundingBoxDrawer)]

        if not boxes:
            return

        # Optimize z-value checks
        boxes.sort(key=lambda x: x.zValue())
        if boxes[0] == self:  # We're the bottom-most box
            views = scene.views()
            if views:
                view = views[0]
                if isinstance(view, CustomGraphicsView):
                    view._bring_to_front(self)
        else:
            self.setOpacity(1.0)

    def hoverLeaveEvent(self, event):
        if not self._is_hovered:
            return

        # Only reset Z-value if this box was moved to the front during hover
        if self.zValue() > 1.0:
            self.setZValue(self.zValue() - 0.1)

        self.setOpacity(self.normal_opacity)
        super().hoverLeaveEvent(event)
        self._is_hovered = False

        
    def start_flashing(self, interval, duration):
        self.flash_timer.start(interval)
        QTimer.singleShot(duration, self.stop_flashing)

    def stop_flashing(self):
        self.flash_timer.stop()
        self.setPen(QPen(get_color(self.class_id, self.main_window.classes_dropdown.count()), 2))

    def toggle_flash_color(self):
        current_color = self.pen().color()
        self.setPen(QPen(self.flash_color if current_color == self.alternate_flash_color else self.alternate_flash_color, 2))


    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)

        rect = self.rect()

        # Optional shaded background
        if self.main_window.shade_checkbox.isChecked():
            shade_value = self.main_window.shade_slider.value()
            color = self.pen().color()
            mapped_alpha = int((shade_value / 100) * 255) if self.confidence is not None else shade_value
            shaded_color = QColor(color.red(), color.green(), color.blue(), mapped_alpha)
            painter.setBrush(shaded_color)
            painter.drawRect(rect)

        # Label drawing
        if not self.main_window.hide_label_checkbox.isChecked():
            label_rect = self.class_name_item.boundingRect()
            label_pos = self.class_name_item.pos()
            label_rect.moveTopLeft(label_pos)

            path = QPainterPath()
            path.addRect(label_rect)

            # Determine connector direction (upward or downward)
            box_top = rect.top()
            box_bottom = rect.bottom()
            label_bottom = label_rect.bottom()
            label_top = label_rect.top()

            if label_bottom < box_top:
                # Label is above box – draw downward connector
                connector_top = label_bottom
                connector_bottom = box_top
            elif label_top > box_bottom:
                # Label is below box – draw upward connector
                connector_top = box_bottom
                connector_bottom = label_top
            else:
                # Label overlaps box – skip connector
                connector_top = connector_bottom = None

            # Draw connector only if it doesn’t overlap the box
            if connector_top is not None and connector_bottom is not None:
                connector_rect = QRectF(
                    rect.left(),
                    connector_top,
                    label_rect.width(),
                    connector_bottom - connector_top
                )
                path.addRect(connector_rect)

            # Draw label + connector
            pen_color = self.pen().color()
            painter.setBrush(pen_color)
            painter.setPen(Qt.NoPen)
            painter.drawRect(label_rect.adjusted(-2, -2, 2, 0))
            painter.drawPath(path)



    def update_class_name_item(self):
        """Update class name label text, color, font, and position (smart placement above or below)."""
        normalized_rect = self.rect()
        self._last_rect = normalized_rect

        # Set label text and color
        class_name = self.main_window.classes_dropdown.itemText(self.class_id)
        self.class_name_item.setPlainText(class_name)
        self.class_name_item.setDefaultTextColor(QColor(255, 255, 255))

        # Set font size
        font = self.class_name_item.font()
        font.setPointSize(self.main_window.font_size_slider.value())
        self.class_name_item.setFont(font)

        # Measure label height
        label_height = self.class_name_item.boundingRect().height()
        label_offset = 2

        # Smart positioning logic
        image_top_buffer = 0
        label_above_y = normalized_rect.top() - (label_height + label_offset)
        label_below_y = normalized_rect.bottom() + label_offset

        if label_above_y >= image_top_buffer:
            # Plenty of space above
            label_pos = QPointF(normalized_rect.left(), label_above_y)
        else:
            # Not enough space, place below
            label_pos = QPointF(normalized_rect.left(), label_below_y)

        self.class_name_item.setPos(label_pos)

        # Respect visibility toggle
        self.class_name_item.setVisible(not self.main_window.hide_label_checkbox.isChecked())

    def get_formatted_class_text(self):
        return self.main_window.classes_dropdown.itemText(self.class_id)

    def update_class_color_and_position(self):
        offset = 14
        position_x, position_y = self.rect().x(), self.rect().y() - offset
        self.class_name_item.setPos(position_x, position_y)
        self.class_name_item.setDefaultTextColor(QColor(255, 255, 255))


    def update_class_name_item_font(self):
        font = self.class_name_item.font()
        font.setPointSize(self.main_window.font_size_slider.value())
        self.class_name_item.setFont(font)
        self.update_class_name_item()  # optionally reposition

    def set_class_id(self, class_id):
        """
        Set the class ID and update the bounding box label safely.
        """
        if sip.isdeleted(self):  #  Prevent updates on deleted objects
            logger.warning("⚠️ Attempted to update a deleted BoundingBoxDrawer object.")
            return  

        self.class_id = class_id
        self.update_class_name_item()

    def update_bbox(self):
        if self.scene() and self in self.scene().items():  # Avoid re-adding
            return  
        rect = self.rect()
        if self.scene():
            img_width = self.scene().width()
            img_height = self.scene().height()
            x_center = (rect.x() + rect.width() / 2) / img_width
            y_center = (rect.y() + rect.height() / 2) / img_height
            width = rect.width() / img_width
            height = rect.height() / img_height
            self.bbox = BoundingBox(int(self.class_id), x_center, y_center, width, height, self.confidence)



    def set_selected(self, selected):
        """Set selection state safely."""
        if sip.isdeleted(self):
            return
        self.setSelected(selected)
        
        # Update Z-value based on selection
        self.set_z_order(bring_to_front=selected)



    def mouseDoubleClickEvent(self, event):
        """Ensures the cursor remains an arrow after a double-click, with mode guard."""
        if (
            event.button() == Qt.LeftButton
            and not event.modifiers() & Qt.ControlModifier
            and not self.main_window.is_segmentation_mode()
            and not self.main_window.is_keypoint_mode()
        ):
            self.setCursor(QCursor(Qt.ClosedHandCursor))

            # Ensure bounding box can't move outside the image
            rect = self.rect()
            img_width = self.main_window.image.width()
            img_height = self.main_window.image.height()

            if rect.right() > img_width or rect.bottom() > img_height:
                return  # Prevent activation if out of bounds

            self.setFlag(QGraphicsItem.ItemIsMovable, True)  # ✅ Enable only if within bounds
            self.dragStartPos = event.pos() - self.rect().topLeft()
            self.setPen(QPen(QColor(0, 255, 0), 2))
        else:
            super().mouseDoubleClickEvent(event)




    def reset_color(self):
        self.setPen(QPen(get_color(self.class_id, self.main_window.classes_dropdown.count()), 2))

    def mouseMoveEvent(self, event):
        if self.dragStartPos is None:
            return  # Don't process movement if dragging isn't active

        newPos = event.pos() - self.dragStartPos
        newRect = QRectF(newPos, self.rect().size())

        # Get actual image dimensions
        img_width = self.main_window.image.width()
        img_height = self.main_window.image.height()

        # Compute valid bounds for the bounding box
        maxLeft = 0
        maxTop = 0
        maxRight = img_width - self.rect().width()
        maxBottom = img_height - self.rect().height()

        # **Clamping to prevent the bounding box from going outside**
        clamped_x = max(maxLeft, min(newRect.left(), maxRight))
        clamped_y = max(maxTop, min(newRect.top(), maxBottom))

        # Set new position, ensuring it stays within valid image bounds
        self.setRect(QRectF(clamped_x, clamped_y, self.rect().width(), self.rect().height()))
        self.normalize_rect()
        # Ensure consistent Z-value when dragging
        self.set_z_order(bring_to_front=True)

        self.update_class_name_position()

        super().mouseMoveEvent(event)  # ✅ Keep default behavior intact



    def update_class_name_position(self):
        self.update_class_name_item()
        offset = 14
        top_left = self.rect().topLeft()
        new_label_pos = QPointF(top_left.x(), top_left.y() - offset)
        self.class_name_item.setPos(new_label_pos)
        self.class_name_item.update()

    def mouseReleaseEvent(self, event):
        """Handles bounding box movement and release events, ensuring it stays in bounds."""
        if sip.isdeleted(self):  # ✅ Prevent operations on deleted objects
            return

        if self.dragStartPos is not None:
            img_width = self.main_window.image.width()
            img_height = self.main_window.image.height()

            # Re-clamp position in case of unwanted movement on release
            rect = self.rect()
            clamped_x = max(0, min(rect.x(), img_width - rect.width()))
            clamped_y = max(0, min(rect.y(), img_height - rect.height()))
            self.setRect(QRectF(clamped_x, clamped_y, rect.width(), rect.height()))

            self.dragStartPos = None
            
            self.normalize_rect()
            self.update_bbox()
            self.setFlag(QGraphicsItem.ItemIsMovable, False)  # ✅ Disable moving after release
            self.setPen(QPen(get_color(self.class_id, self.main_window.classes_dropdown.count()), 2))

        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            self.set_selected(False)
            self.unsetCursor()
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.setFlag(QGraphicsItem.ItemIsFocusable, True)

        self.set_z_order(bring_to_front=False)
        super().mouseReleaseEvent(event)

class KeypointDrawer(QGraphicsItem):
    def __init__(self, points, main_window, class_id=None, unique_id=None, file_name=None, visibility_flags=None):
        super().__init__()

        self.main_window = main_window
        self.class_id = class_id if class_id is not None else main_window.get_current_class_id()
        self.unique_id = unique_id
        self.file_name = file_name if file_name else main_window.current_file
        self.points = points if isinstance(points, list) else []
        self.visibility_flags = visibility_flags if visibility_flags else [2] * len(self.points)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.setAcceptHoverEvents(True)
        self.main_window.dot_size_slider.valueChanged.connect(self.update_dot_radius)

        self.point_items = []
        self.class_name_item = QGraphicsTextItem(self)
        self.class_name_item.setDefaultTextColor(QColor(255, 255, 255))
        self.class_name_item.setFont(QFont("Arial", self.main_window.font_size_slider.value(), QFont.Bold))

        self.update_class_name_item()
        self.update_points()
        self.current_index = 0
        self.main_window.font_size_slider.valueChanged.connect(self.update_class_name_item_font)

    def update_dot_radius(self, value=None):
        self.update()


    def get_keypoint_string(self):
        keypoint_strs = []

        total_rows = self.main_window.keypoint_list.rowCount()
        num_points = len(self.points)
        num_flags = len(self.visibility_flags)

        for i in range(total_rows):
            # Handle missing points
            x, y = self.points[i] if i < num_points else (0.0, 0.0)

            # Handle visibility flags smartly
            if i < num_flags:
                v = self.visibility_flags[i]
            else:
                # Fallback: use 2 (visible) if point is placed, 0 if it's not
                v = 2 if (x != 0.0 or y != 0.0) else 0

            keypoint_strs.append(f"{x:.6f} {y:.6f} {v}")

        return " ".join(keypoint_strs)




    def boundingRect(self):
        if not self.points:
            return QRectF()
        img_width = self.main_window.image.width()
        img_height = self.main_window.image.height()
        scaled_points = [QPointF(x * img_width, y * img_height) for x, y in self.points]
        min_x = min(p.x() for p in scaled_points)
        min_y = min(p.y() for p in scaled_points)
        max_x = max(p.x() for p in scaled_points)
        max_y = max(p.y() for p in scaled_points)
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)

        color = get_color(self.class_id, self.main_window.classes_dropdown.count(), alpha=200)
        pen = QPen(color, 2)
        brush = QBrush(color)
        painter.setPen(pen)
        painter.setBrush(brush)

        img_width = self.main_window.image.width()
        img_height = self.main_window.image.height()
        zoom_factor = self.main_window.screen_view.transform().m11()

        # Get dot size from slider and apply zoom scaling
        slider_value = self.main_window.dot_size_slider.value()  # Assume 1–10 range
        base_radius = slider_value / 2.0  # Map to something like 0.5–5.0
        radius = base_radius / zoom_factor
        radius = max(0.5, min(radius, 6.0))  # Clamp for sanity

        for i, (x, y) in enumerate(self.points):
            if i < len(self.visibility_flags) and self.visibility_flags[i] == 0:
                continue  # Skip ignored keypoints

            # ✅ Use full image dimensions instead of bbox-relative
            x_abs = x * img_width
            y_abs = y * img_height
            point = QPointF(x_abs, y_abs)

            painter.drawEllipse(point, radius, radius)




    def append_point(self, point):
        if not hasattr(self, "main_window") or not hasattr(self.main_window, "screen_view"):
            logger.warning("Main window or screen_view missing.")
            return

        # ✅ Get the bounding box parent and make sure point is inside it
        parent = self.parentItem()
        if isinstance(parent, BoundingBoxDrawer) and not parent.rect().contains(point):
            logger.warning("⚠️ Cannot place keypoint outside bounding box.")
            return

        img_width = self.main_window.image.width()
        img_height = self.main_window.image.height()
        normalized_point = (point.x() / img_width, point.y() / img_height)

        # ✅ Get total number of keypoints allowed from UI
        max_points = self.main_window.keypoint_list.rowCount()
        if len(self.points) >= max_points:
            logger.warning("⚠️ Maximum number of keypoints reached.")
            return

        # ✅ Pad point list if needed
        while len(self.points) < self.current_index:
            self.points.append((0.0, 0.0))

        if self.current_index < len(self.points):
            self.points[self.current_index] = normalized_point
        else:
            self.points.append(normalized_point)

        # ✅ Pull visibility/ignore state from the UI checkboxes
        visible_cb = self.main_window.keypoint_list.cellWidget(self.current_index, 3)
        ignore_cb = self.main_window.keypoint_list.cellWidget(self.current_index, 4)

        if ignore_cb and ignore_cb.isChecked():
            # ⛔ Do not place a visual point
            while len(self.points) <= self.current_index:
                self.points.append((0.0, 0.0))

            while len(self.visibility_flags) <= self.current_index:
                self.visibility_flags.append(0)

            self.visibility_flags[self.current_index] = 0  # ignored
            self.points[self.current_index] = (0.0, 0.0)

            self.main_window.keypoint_list.selectRow(self.current_index)
            self.current_index += 1

            if self.current_index >= max_points:
                self.main_window.keypoint_list.clearSelection()

            self.update_class_name_item()

            logger.info(f"🚫 Ignored point for row {self.current_index - 1}")
            return  # ⛔ EXIT EARLY to skip adding the point to scene
        else:
            v = 2 if visible_cb and visible_cb.isChecked() else 1

        while len(self.visibility_flags) < self.current_index:
            self.visibility_flags.append(0)

        if self.current_index < len(self.visibility_flags):
            self.visibility_flags[self.current_index] = v
        else:
            self.visibility_flags.append(v)

        # ✅ Highlight selected row and advance index
        self.main_window.keypoint_list.selectRow(self.current_index)
        self.current_index += 1

        if self.current_index >= max_points:
            self.main_window.keypoint_list.clearSelection()

        self.update_points()
        self.update()
        self.scene().update()
        self.update_class_name_item()

        # 💾 Sync keypoints into bbox object before saving
        if isinstance(self.parentItem(), BoundingBoxDrawer):
            parent_bbox = self.parentItem()
            if hasattr(parent_bbox, 'bbox') and parent_bbox.bbox:
                parent_bbox.bbox.keypoints = [
                    (x, y, v) for (x, y), v in zip(self.points, self.visibility_flags)
                ]

        # 🔍 Log detailed info about the point added
        try:
            point_index = self.current_index - 1
            label_name = self.main_window.keypoint_list.item(point_index, 1).text()
            v_flag = self.visibility_flags[point_index] if point_index < len(self.visibility_flags) else "?"
            
            # 🟦 Grab the color
            color_item = self.main_window.keypoint_list.item(point_index, 2)
            if color_item:
                color = color_item.background().color()
                color_hex = color.name()
            else:
                color_hex = "N/A"

            logger.info(
                f"🟢 Keypoint placed → Label: '{label_name}', Flag: {v_flag}, Color: {color_hex}, File: {os.path.basename(self.file_name)}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to log keypoint label info: {e}")

        # ✅ Trigger save if we're in keypoint mode
        if hasattr(self.main_window, "save_bounding_boxes") and self.main_window.is_keypoint_mode():
            self.main_window.save_bounding_boxes(
                self.file_name,
                img_width,
                img_height
            )




    def update_points(self):
        for item in self.point_items:
            if self.scene():
                self.scene().removeItem(item)
        self.point_items.clear()

        img_width = self.main_window.image.width()
        img_height = self.main_window.image.height()
        total_rows = self.main_window.keypoint_list.rowCount()
        zoom_factor = self.main_window.screen_view.transform().m11()

        slider_value = self.main_window.dot_size_slider.value()
        base_radius = slider_value / 2.0
        radius = base_radius / zoom_factor
        radius = max(0.5, min(radius, 6.0))

        for i, (x, y) in enumerate(self.points):
            if i < len(self.visibility_flags) and self.visibility_flags[i] == 0:
                continue

            x_abs = x * img_width
            y_abs = y * img_height

            ellipse = QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius, self)
            ellipse.setPos(QPointF(x_abs, y_abs))

            if i < total_rows:
                color_item = self.main_window.keypoint_list.item(i, 2)
                color = color_item.background().color() if color_item else QColor(255, 255, 255)
            else:
                color = QColor(255, 255, 255)

            ellipse.setBrush(QBrush(color))
            ellipse.setPen(QPen(color))
            self.point_items.append(ellipse)


    def update_class_name_item_font(self):
        font = self.class_name_item.font()
        font.setPointSize(self.main_window.font_size_slider.value())
        self.class_name_item.setFont(font)
        self.update_class_name_item()

    def update_class_name_item(self):
        if not self.points:
            return

        try:
            point_index = len(self.points) - 1
            point_label = self.main_window.keypoint_list.item(point_index, 1).text()
        except Exception:
            point_label = f"pt{point_index}"

        self.class_name_item.setPlainText(point_label)

        img_width = self.main_window.image.width()
        img_height = self.main_window.image.height()
        x_abs = self.points[-1][0] * img_width
        y_abs = self.points[-1][1] * img_height
        self.class_name_item.setPos(QPointF(x_abs + 6, y_abs - 12))



    def remove_self(self):
        if self.scene():
            self.scene().removeItem(self)
        self.remove_keypoint_from_file()

    def remove_keypoint_from_file(self):
        if not self.file_name:
            logger.error("❌ Error: Cannot remove keypoints - file_name is None!")
            return
        label_file = os.path.splitext(self.file_name)[0] + ".txt"
        if not os.path.exists(label_file):
            logger.warning(f"⚠️ Label file does not exist: {label_file}")
            return

        try:
            with open(label_file, "r") as f:
                lines = f.readlines()
            updated_lines = [line for line in lines if f"{self.class_id} " not in line]
            with open(label_file, "w") as f:
                f.writelines(updated_lines)
            logging.info(f"✅ Keypoints removed from {label_file}")
        except Exception as e:
            logger.error(f"❌ Error removing keypoints from {label_file}: {e}")

class SegmentationDrawer(QGraphicsPolygonItem):
    def __init__(self, points, main_window, class_id=None, unique_id=None, file_name=None):
        super().__init__()
        self.main_window = main_window
        self.class_id = class_id if class_id is not None else main_window.get_current_class_id()
        self.unique_id = unique_id
        self.file_name = file_name if file_name else main_window.current_file
        self.points = points if isinstance(points, list) else []

        self.simplify_points()  # 🔧 Simplify immediately

        img_width = self.main_window.image.width()
        img_height = self.main_window.image.height()
        self.polygon = QPolygonF([QPointF(x * img_width, y * img_height) for x, y in self.points])
        self.setPolygon(self.polygon)

        self._base_color = get_color(self.class_id, self.main_window.classes_dropdown.count(), alpha=150)
        self.setPen(QPen(self._base_color, 2))
        self.setBrush(QBrush(self._base_color))

        self.class_name_item = QGraphicsTextItem(self)
        self.class_name_item.setDefaultTextColor(QColor(255, 255, 255))
        self.class_name_item.setFont(QFont("Arial", self.main_window.font_size_slider.value(), QFont.Bold))
        self.update_class_name_item()

        self.setAcceptHoverEvents(True)
        self.main_window.shade_slider.valueChanged.connect(self.update_opacity)
        self.main_window.shade_checkbox.stateChanged.connect(self.update_opacity)
        self.main_window.font_size_slider.valueChanged.connect(self.update_class_name_item_font)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)

        if self.file_name is None:
            logging.error("❌ Error: SegmentationDrawer file_name is None! Ensure it's passed correctly.")

        self.class_name_item.setVisible(not self.main_window.hide_labels)
        self.edit_mode = False
        self.vertex_handles = []

    def start_flashing(self, interval=100, duration=1000):
        if hasattr(self, "_flash_timer") and self._flash_timer.isActive():
            self._flash_timer.stop()

        self._flash_timer = QTimer()
        self._flash_timer.timeout.connect(self._toggle_flash)
        self._flash_duration = duration
        self._flash_start_time = time.time()
        self._flashing_on = False

        # ⚡ NO resetting flash_color or alternate_flash_color here
        if not hasattr(self, 'flash_color'):
            self.flash_color = QColor(255, 0, 0)  # fallback red if none set

        if not hasattr(self, 'alternate_flash_color'):
            self.alternate_flash_color = self._base_color  # fallback to base color

        self._flash_timer.start(interval)


    def _toggle_flash(self):
        elapsed_time = (time.time() - self._flash_start_time) * 1000  # ms

        if elapsed_time >= self._flash_duration:
            self.setBrush(QBrush(self._base_color))  # Reset to normal
            self._flash_timer.stop()
            return

        if self._flashing_on:
            self.setBrush(QBrush(self._base_color))
        else:
            self.setBrush(QBrush(self.flash_color))

        self._flashing_on = not self._flashing_on



    def set_z_order(self, bring_to_front=False):
        """Set z-value to control stacking order of segmentations."""
        if bring_to_front:
            self.setZValue(1.0)
        elif self.zValue() < 1.0:
            self.setZValue(0.5)


    def hoverEnterEvent(self, event):
        if not self.scene():
            return

        scene_pos = self.mapToScene(event.pos())
        overlapping = [
            item for item in self.scene().items(scene_pos)
            if isinstance(item, SegmentationDrawer)
        ]

        overlapping.sort(key=lambda x: x.zValue(), reverse=True)

        # Only bring to front if we're the top-most
        if overlapping and overlapping[0] == self:
            self.set_z_order(bring_to_front=True)
        else:
            self.setZValue(self.zValue() + 0.1)

        self.setOpacity(0.6)
        self.update()
        super().hoverEnterEvent(event)


    def hoverLeaveEvent(self, event):
        # Only reset Z if we bumped it up
        if self.zValue() > 1.0:
            self.setZValue(self.zValue() - 0.1)

        self.setOpacity(1.0)
        self.update()
        super().hoverLeaveEvent(event)

    def update_class_name_item_font(self):
        font = self.class_name_item.font()
        font.setPointSize(self.main_window.font_size_slider.value())
        self.class_name_item.setFont(font)
        self.update_class_name_item()

    def update_class_name_item(self):
        class_name = self.main_window.classes_dropdown.itemText(self.class_id)
        self.class_name_item.setPlainText(class_name)
        font = self.class_name_item.font()
        font.setPointSize(self.main_window.font_size_slider.value())
        self.class_name_item.setFont(font)
        if self.points:
            first_point = self.polygon.boundingRect().topLeft()
            self.class_name_item.setPos(first_point)
        self.class_name_item.setVisible(not self.main_window.hide_labels)


    def append_point(self, point):
        if self.edit_mode:
            return
        img_width = self.main_window.image.width()
        img_height = self.main_window.image.height()
        x, y = int(point.x()), int(point.y())
        snapped_point = point

        if self.main_window.outline_Checkbox.isChecked() and hasattr(self.main_window, 'processed_image'):
            gray_img = cv2.cvtColor(self.main_window.processed_image, cv2.COLOR_BGR2GRAY)
            roi_size = 30
            x_min = max(0, x - roi_size)
            y_min = max(0, y - roi_size)
            x_max = min(gray_img.shape[1], x + roi_size)
            y_max = min(gray_img.shape[0], y + roi_size)
            roi = gray_img[y_min:y_max, x_min:x_max]
            blurred = cv2.GaussianBlur(roi, (3, 3), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if contours:
                min_dist = float('inf')
                nearest_point = (x, y)
                for contour in contours:
                    for pt in contour:
                        cx, cy = pt[0]
                        global_x = cx + x_min
                        global_y = cy + y_min
                        dist = np.hypot(global_x - x, global_y - y)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_point = (global_x, global_y)
                snapped_point = QPointF(nearest_point[0], nearest_point[1])

        self.points.append((snapped_point.x() / img_width, snapped_point.y() / img_height))
        self.polygon.append(snapped_point)
        self.setPolygon(self.polygon)

    def update_polygon(self):
        self.setPolygon(QPolygonF([
            QPointF(x * self.main_window.image.width(), y * self.main_window.image.height()) 
            for x, y in self.points
        ]))
        self.update_class_name_item()


    def remove_self(self):
        if self.scene():
            self.scene().removeItem(self)
        self.remove_segmentation_from_file()

    def remove_segmentation_from_file(self):
        if not self.file_name:
            logger.error("❌ Error: Cannot remove segmentation - file_name is None!")
            return
        label_file = os.path.splitext(self.file_name)[0] + ".txt"
        if not os.path.exists(label_file):
            logger.warning(f"⚠️ Label file does not exist: {label_file}")
            return
        current_label = f"{self.class_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in self.points)
        try:
            with open(label_file, "r") as f:
                lines = f.readlines()
            updated_lines = [line for line in lines if line.strip() != current_label]
            with open(label_file, "w") as f:
                f.writelines(updated_lines)
            logger.info(f"✅ Segmentation removed from {label_file}")
        except Exception as e:
            logger.error(f"❌ Error removing segmentation from {label_file}: {e}")

    def update_opacity(self):
        if self.main_window.shade_checkbox.isChecked():
            shade_value = self.main_window.shade_slider.value()
            mapped_alpha = int((shade_value / 100) * 255)
        else:
            mapped_alpha = 150

        shaded_color = QColor(
            self._base_color.red(),
            self._base_color.green(),
            self._base_color.blue(),
            mapped_alpha,
        )
        self.setBrush(QBrush(shaded_color))
        self.update()



    def mousePressEvent(self, event):
        scene_pos = event.scenePos()

        if self.edit_mode:
            # Handle moving with left click
            if event.button() == Qt.LeftButton:
                for handle in self.vertex_handles:
                    if handle.contains(handle.mapFromScene(scene_pos)):
                        handle.setCursor(Qt.ClosedHandCursor)
                        handle.mousePressEvent(event)
                        return

            # ➕ Right-click to insert point
            elif event.button() == Qt.RightButton:
                nearest_index = self.find_nearest_edge_index(scene_pos)
                if nearest_index is not None:
                    self.insert_point_at(scene_pos, nearest_index)
                    return

        # Dragging the whole polygon if not in edit mode
        if event.button() == Qt.LeftButton:
            self.setOpacity(0.5)
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.dragStartPos = scene_pos
            self.startPolygonPos = self.pos()

        super().mousePressEvent(event)

    def find_nearest_edge_index(self, scene_pos, threshold=10):
        min_dist = float('inf')
        best_index = None

        for i in range(len(self.points)):
            p1 = self.polygon[i]
            p2 = self.polygon[(i + 1) % len(self.points)]  # Wrap around

            # Convert to vectors
            x1, y1 = p1.x(), p1.y()
            x2, y2 = p2.x(), p2.y()
            px, py = scene_pos.x(), scene_pos.y()

            dx, dy = x2 - x1, y2 - y1
            if dx == dy == 0:
                continue  # Avoid zero-length lines

            # Project point onto line segment
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = ((proj_x - px) ** 2 + (proj_y - py) ** 2) ** 0.5

            if dist < min_dist and dist <= threshold:
                min_dist = dist
                best_index = i + 1  # Insert after this index

        return best_index


    def insert_point_at(self, scene_pos, insert_index):
        img_width = self.main_window.image.width()
        img_height = self.main_window.image.height()
        
        local_pt = self.mapFromScene(scene_pos)
        norm_x = local_pt.x() / img_width
        norm_y = local_pt.y() / img_height

        # Insert point
        self.points.insert(insert_index, (norm_x, norm_y))
        self.polygon.insert(insert_index, local_pt)
        self.setPolygon(self.polygon)

        # Rebuild all handles from scratch
        for handle in self.vertex_handles:
            self.scene().removeItem(handle)
        self.vertex_handles = []

        for idx, pt in enumerate(self.polygon):
            handle = VertexHandle(self, idx)
            handle.setPos(self.mapToScene(pt))
            handle.setCursor(Qt.OpenHandCursor)
            self.scene().addItem(handle)
            self.vertex_handles.append(handle)

    def mouseMoveEvent(self, event):
        if self.edit_mode:
            for handle in self.vertex_handles:
                if handle.cursor() == Qt.ClosedHandCursor:
                    handle.mouseMoveEvent(event)
                    return
        if self.isSelected() and event.buttons() & Qt.LeftButton:
            newPos = event.scenePos()
            delta = newPos - self.dragStartPos
            self.setPos(self.startPolygonPos + delta)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.edit_mode:
            # Let the event bubble naturally to handles
            super().mouseReleaseEvent(event)
            return

        self.setOpacity(1.0)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        super().mouseReleaseEvent(event)


    def toggle_edit_mode(self):
        if not self.edit_mode:
            self.enter_edit_mode()
        else:
            self.finalize_edit_mode()
    def simplify_points(self, min_distance=6, angle_threshold=130):
        if len(self.points) < 3:
            return

        img_width = self.main_window.image.width()
        img_height = self.main_window.image.height()
        pts = np.array([
            [int(x * img_width), int(y * img_height)] for x, y in self.points
        ], dtype=np.int32)

        if len(pts) < 3:
            return  # Don't try to simplify

        def angle_between(a, b, c):
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        simplified_pts = [pts[0]]
        last_kept = pts[0]

        for i in range(1, len(pts) - 1):
            current = pts[i]
            next_pt = pts[i + 1]
            dist = np.linalg.norm(current - last_kept)
            angle = angle_between(pts[i - 1], current, next_pt)

            if dist < 1.0:
                continue  # Still remove extremely close points

            # Allow keeping more gentle angles if spaced out
            if angle > angle_threshold and dist < (min_distance * 1.2):
                continue  # Skip this point


            simplified_pts.append(current)
            last_kept = current

        if not np.array_equal(simplified_pts[-1], pts[-1]):
            simplified_pts.append(pts[-1])

        if len(simplified_pts) < 3:
            logger.warning("❌ Simplification resulted in too few points. Keeping original.")
            return

        self.points = [(pt[0] / img_width, pt[1] / img_height) for pt in simplified_pts]


    def enter_edit_mode(self):
        self.edit_mode = True
        self.vertex_handles = []
        self.simplify_points()  # 🔁 Ensure handle points match
        self.polygon = QPolygonF([
            QPointF(x * self.main_window.image.width(), y * self.main_window.image.height())
            for x, y in self.points
        ])
        self.setPolygon(self.polygon)

        for idx, pt in enumerate(self.polygon):
            handle = VertexHandle(self, idx)
            handle.setPos(self.mapToScene(pt))
            handle.setCursor(Qt.OpenHandCursor)
            self.scene().addItem(handle)
            self.vertex_handles.append(handle)

    def finalize(self):
        self.simplify_points()  # 💾 Save simplified version

        if len(self.points) < 3:
            logging.warning(f"❌ Cannot finalize — only {len(self.points)} points (need at least 3).")
            if self.scene():
                self.scene().removeItem(self)
            return  # 🔒 Don't proceed

        img_width = self.main_window.image.width()
        img_height = self.main_window.image.height()
        polygon = QPolygonF([
            QPointF(x * img_width, y * img_height)
            for x, y in self.points
        ])
        self.setPolygon(polygon)

        if not self.scene() or self not in self.scene().items():
            self.scene().addItem(self)

            self.update_class_name_item()
            self.main_window.save_bounding_boxes(
                self.file_name,
                self.main_window.image.width(),
                self.main_window.image.height()
            )

        logging.info(f"✅ Segmentation finalized and saved for {self.file_name}")


    def finalize_edit_mode(self):
        if len(self.points) < 3:
            logging.warning("⚠️ Cannot finalize edit mode — segmentation has fewer than 3 points.")
            return  # 🔒 Don't exit edit mode or remove handles

        self.edit_mode = False
        for handle in self.vertex_handles:
            self.scene().removeItem(handle)
        self.vertex_handles = []
        self.finalize()


    def update_point(self, index, new_scene_pos):
        if index >= len(self.points):
            logging.warning(f"⚠️ Tried to update point at index {index}, but points has only {len(self.points)} items.")
            return

        x, y = int(new_scene_pos.x()), int(new_scene_pos.y())
        snapped_scene_pos = new_scene_pos

        if self.main_window.outline_Checkbox.isChecked() and hasattr(self.main_window, 'processed_image'):
            gray_img = cv2.cvtColor(self.main_window.processed_image, cv2.COLOR_BGR2GRAY)
            roi_size = 30
            x_min = max(0, x - roi_size)
            y_min = max(0, y - roi_size)
            x_max = min(gray_img.shape[1], x + roi_size)
            y_max = min(gray_img.shape[0], y + roi_size)
            roi = gray_img[y_min:y_max, x_min:x_max]
            blurred = cv2.GaussianBlur(roi, (3, 3), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if contours:
                min_dist = float('inf')
                nearest_point = (x, y)
                for contour in contours:
                    for pt in contour:
                        cx, cy = pt[0]
                        global_x = cx + x_min
                        global_y = cy + y_min
                        dist = np.hypot(global_x - x, global_y - y)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_point = (global_x, global_y)
                snapped_scene_pos = QPointF(nearest_point[0], nearest_point[1])

        local_pt = self.mapFromScene(snapped_scene_pos)

        if index >= len(self.polygon):
            logging.warning(f"⚠️ Polygon index {index} out of bounds. Polygon has {len(self.polygon)} points.")
            return

        self.polygon[index] = local_pt
        self.setPolygon(self.polygon)
        self.points[index] = (
            local_pt.x() / self.main_window.image.width(),
            local_pt.y() / self.main_window.image.height()
        )



    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_edit_mode()
        super().mouseDoubleClickEvent(event)


class VertexHandle(QGraphicsEllipseItem):
    def __init__(self, drawer, index, radius=3):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.drawer = drawer
        self.index = index
        self.active_color = QColor(0, 255, 0)
        self.default_color = QColor(Qt.red)
        self.setBrush(QBrush(self.default_color))
        self.setCursor(Qt.OpenHandCursor)
        self.setZValue(10)

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)

    def mousePressEvent(self, event):
        self.setBrush(QBrush(self.active_color))
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.drawer.update_point(self.index, self.scenePos())

    def mouseReleaseEvent(self, event):
        self.setBrush(QBrush(self.default_color))
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemScenePositionHasChanged:
            self.drawer.update_point(self.index, value)
        return super().itemChange(change, value)

class OBBDrawer(QGraphicsPolygonItem):
    def __init__(self, main_window, class_id=None, unique_id=None, file_name=None,
                 obb_points=None, angle=None):
        super().__init__()
        self.main_window = main_window
        self.class_id = class_id if class_id is not None else main_window.get_current_class_id()
        self.unique_id = unique_id
        self.file_name = file_name if file_name else main_window.current_file

        # Core geometry states
        self.obb_points = obb_points or []
        self.angle = angle if angle is not None else 0.0
        self.centerline_start = None
        self.centerline_end = None
        self.current_width = 0
        self.drawing_in_progress = False
        self.current_state = "awaiting_centerline_start"
        self.class_name_item = QGraphicsTextItem(self)
        self.class_name_item.setDefaultTextColor(QColor(255, 255, 255))
        self.class_name_item.setFont(QFont("Arial", self.main_window.font_size_slider.value(), QFont.Bold))
        self.update_class_name_item()
        self.main_window.font_size_slider.valueChanged.connect(self.update_class_name_item_font)

        # Visual state
        self.centerline_item = None

        color = get_color(self.class_id, self.main_window.classes_dropdown.count())
        self.setPen(QPen(color, 2))
        self.setBrush(QBrush(Qt.NoBrush))

        if self.obb_points:
            self.angle = angle if angle is not None else 0.0
            self.restore_centerline_from_obb()
            self.render_from_points()

    def update_class_name_item(self):
        if not self.obb_points:
            return

        img_w = self.main_window.image.width()
        img_h = self.main_window.image.height()

        # Convert to absolute points
        abs_points = [QPointF(x * img_w, y * img_h) for x, y in self.obb_points]
        top_left = min(abs_points, key=lambda p: p.y())  # find topmost point

        # Get class label
        class_name = self.main_window.classes_dropdown.itemText(self.class_id)
        self.class_name_item.setPlainText(class_name)
        self.class_name_item.setPos(top_left.x() + 5, top_left.y() - 20)
        self.class_name_item.setVisible(not self.main_window.hide_label_checkbox.isChecked())
    def update_class_name_item_font(self):
        font = self.class_name_item.font()
        font.setPointSize(self.main_window.font_size_slider.value())
        self.class_name_item.setFont(font)
        self.update_class_name_item()
    def set_class_id(self, class_id):
        self.class_id = class_id
        self.update_class_name_item()

    def render_from_points(self):
        """
        Converts normalized `obb_points` to image-space and draws the polygon.
        """
        img_w = self.main_window.image.width()
        img_h = self.main_window.image.height()
        polygon = QPolygonF([QPointF(x * img_w, y * img_h) for x, y in self.obb_points])
        self.update_class_name_item()
        self.setPolygon(polygon)
    def update_opacity(self, alpha=255):
        color = get_color(self.class_id, self.main_window.classes_dropdown.count(), alpha=alpha)
        self.setPen(QPen(color, 2))
        self.setBrush(QBrush(Qt.NoBrush))

    def calculate_obb_corners(self, half_width):
        if not self.centerline_start or not self.centerline_end:
            return []

        p1 = np.array((self.centerline_start.x(), self.centerline_start.y()))
        p2 = np.array((self.centerline_end.x(), self.centerline_end.y()))
        vec = p2 - p1
        vec_len = np.linalg.norm(vec)
        if vec_len == 0:
            print("⚠️ Centerline length is zero.")
            return []

        unit_vec = vec / vec_len
        perp_vec = np.array([-unit_vec[1], unit_vec[0]])

        p1_topleft = p1 + perp_vec * half_width
        p2_topright = p2 + perp_vec * half_width
        p3_bottomright = p2 - perp_vec * half_width
        p4_bottomleft = p1 - perp_vec * half_width
        return [p1_topleft, p2_topright, p3_bottomright, p4_bottomleft]

    def update_preview(self, current_mouse_pos):
        try:
            if self.scene() is None:
                print("⚠️ OBBDrawer's scene is None — likely deleted.")
                return
        except RuntimeError:
            print("⚠️ OBBDrawer object was deleted — skipping preview update.")
            return

        if self.centerline_start and self.centerline_end and self.drawing_in_progress:
            vec = np.array((self.centerline_end.x(), self.centerline_end.y())) - \
                np.array((self.centerline_start.x(), self.centerline_start.y()))
            length = np.linalg.norm(vec)
            if length == 0:
                return

            unit_vec = vec / length
            perp_vec = np.array([-unit_vec[1], unit_vec[0]])
            mouse_vec = np.array((current_mouse_pos.x(), current_mouse_pos.y())) - \
                        np.array((self.centerline_start.x(), self.centerline_start.y()))
            self.current_width = np.dot(mouse_vec, perp_vec)

            half_width = abs(self.current_width) / 2
            corners = self.calculate_obb_corners(half_width)
            if not corners:
                return

            polygon = QPolygonF([QPointF(x, y) for x, y in corners])
            self.setPolygon(polygon)


    def finalize(self):
        self.current_state = "bb_defined"
        print(f"🚀 Finalized OBB with width: {self.current_width}")

        half_width = abs(self.current_width) / 2
        corners = self.calculate_obb_corners(half_width)
        if not corners or len(corners) != 4:
            print("❌ Invalid corners.")
            return

        img_w = self.main_window.image.width()
        img_h = self.main_window.image.height()

        # 🧠 Optional snapping based on contour
        if self.main_window.outline_Checkbox.isChecked() and hasattr(self.main_window, 'processed_image'):
            gray_img = self.main_window.processed_image
            if len(gray_img.shape) == 3 and gray_img.shape[2] == 3:
                gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

            x1 = int(min(x for x, y in corners))
            y1 = int(min(y for x, y in corners))
            x2 = int(max(x for x, y in corners))
            y2 = int(max(y for x, y in corners))

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            roi = gray_img[y1:y2, x1:x2]
            if roi.size > 0:
                blurred = cv2.GaussianBlur(roi, (3, 3), 0)
                edges = cv2.Canny(blurred, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    biggest = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(biggest)
                    (cx, cy), (w, h), angle = rect
                    box = cv2.boxPoints(rect)
                    box[:, 0] += x1
                    box[:, 1] += y1
                    corners = [tuple(p) for p in box]
                    self.angle = np.deg2rad(angle)
                else:
                    dx = self.centerline_end.x() - self.centerline_start.x()
                    dy = self.centerline_end.y() - self.centerline_start.y()
                    self.angle = np.arctan2(dy, dx)
            else:
                dx = self.centerline_end.x() - self.centerline_start.x()
                dy = self.centerline_end.y() - self.centerline_start.y()
                self.angle = np.arctan2(dy, dx)
        else:
            dx = self.centerline_end.x() - self.centerline_start.x()
            dy = self.centerline_end.y() - self.centerline_start.y()
            self.angle = np.arctan2(dy, dx)

        # Normalize and store corners
        self.obb_points = [(x / img_w, y / img_h) for x, y in corners]
        self.update_class_name_item()
        self.render_from_points()  # 🔥 Ensures shape updates immediately
        self.scene().update()
        self.main_window.screen_view.viewport().update()
        self._save_to_txt()

        # ✅ Save full label set immediately
        image_file = self.main_window.current_file
        if hasattr(self.main_window, "save_bounding_boxes"):
            self.main_window.save_bounding_boxes(image_file, img_w, img_h)

    def snap_corners_to_contour(self, gray_img, corners, img_w, img_h):
        x1 = int(max(min(x for x, y in corners), 0))
        y1 = int(max(min(y for x, y in corners), 0))
        x2 = int(min(max(x for x, y in corners), img_w - 1))
        y2 = int(min(max(y for x, y in corners), img_h - 1))

        roi = gray_img[y1:y2, x1:x2]
        if roi.size == 0:
            return corners  # fallback

        blurred = cv2.GaussianBlur(roi, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            biggest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(biggest)
            box = cv2.boxPoints(rect)
            box[:, 0] += x1
            box[:, 1] += y1
            return [tuple(p) for p in box]

        return corners

    def _save_to_txt(self):
        label_file = os.path.splitext(self.file_name)[0] + ".txt"

        # Convert 4-point polygon to center, width, height
        if len(self.obb_points) != 4:
            print("⚠️ Cannot convert to OBB format — invalid point count.")
            return

        # Unpack 4 points
        p1, p2, p3, p4 = [np.array([x, y]) for x, y in self.obb_points]

        # Compute center
        cx = np.mean([p[0] for p in [p1, p2, p3, p4]])
        cy = np.mean([p[1] for p in [p1, p2, p3, p4]])

        # Compute width and height using distances
        width = np.linalg.norm(p1 - p2)
        height = np.linalg.norm(p2 - p3)

        line = f"{self.class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f} {self.angle:.6f}\n"

        try:
            with open(label_file, "a") as f:
                f.write(line)
            print(f"💾 Saved Ultralytics OBB format to {label_file}")
        except Exception as e:
            print(f"❌ Error saving OBB: {e}")

    def remove_self(self):
        scene = self.scene()
        if scene:
            scene.removeItem(self)
        if self.centerline_item:
            scene = self.scene()
            if scene:
                scene.removeItem(self.centerline_item)
            self.centerline_item = None

        # 👇 Safely remove this OBBDrawer itself
        if self.scene():
            self.scene().removeItem(self)

        # 💥 Reset internal data
        self.centerline_start = None
        self.centerline_end = None
        self.obb_points = []
        self.drawing_in_progress = False
        self.current_state = "awaiting_centerline_start"

        # 🗑 Remove from file
        self.remove_obb_from_file()


    def _remove_from_txt(self):
        label_file = os.path.splitext(self.file_name)[0] + ".txt"
        if not os.path.exists(label_file):
            print("⚠️ Label file does not exist.")
            return

        try:
            with open(label_file, "r") as f:
                lines = f.readlines()

            current_line = f"{self.class_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in self.obb_points)
            lines = [l for l in lines if l.strip() != current_line]

            with open(label_file, "w") as f:
                f.writelines(lines)

            print(f"🗑️ Removed OBB from {label_file}")
        except Exception as e:
            print(f"❌ Failed to remove OBB: {e}")

    def clear(self):
        self.centerline_start = None
        self.centerline_end = None
        self.current_width = 0
        self.drawing_in_progress = False
        self.setPolygon(QPolygonF())  # Clear the shape


    def remove_obb_from_file(self):
        if not self.file_name:
            print("⚠️ No file path to remove OBB from.")
            return

        if not self.obb_points or len(self.obb_points) != 4:
            print("⚠️ Cannot remove OBB — obb_points not set.")
            return

        label_file = os.path.splitext(self.file_name)[0] + ".txt"
        if not os.path.exists(label_file):
            print(f"⚠️ Label file not found: {label_file}")
            return

        try:
            # Build your OBB float array for comparison
            target = [coord for pt in self.obb_points for coord in pt] + [self.angle]

            with open(label_file, "r") as f:
                lines = f.readlines()

            kept_lines = []
            removed = False

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 10:
                    kept_lines.append(line)
                    continue

                try:
                    class_id = int(parts[0])
                    floats = list(map(float, parts[1:]))

                    if (
                        class_id == self.class_id and
                        np.allclose(floats, target, atol=1e-5)
                    ):
                        removed = True
                        continue  # Skip this line
                except Exception:
                    pass

                kept_lines.append(line)

            with open(label_file, "w") as f:
                f.writelines(kept_lines)

            if removed:
                print(f"🗑️ Removed OBB from {label_file}")
            else:
                print(f"⚠️ No exact OBB match found to remove in {label_file}")

        except Exception as e:
            print(f"❌ Failed to remove OBB from file: {e}")

    def restore_centerline_from_obb(self):
        """
        Reconstruct centerline from the OBB points. This is useful when loading from file.
        """
        try:
            if not self.obb_points or len(self.obb_points) != 4:
                print("⚠️ Cannot restore centerline: obb_points not set or invalid.")
                return

            img_w = self.main_window.image.width()
            img_h = self.main_window.image.height()

            # Convert normalized points to absolute coordinates
            pts = [QPointF(x * img_w, y * img_h) for x, y in self.obb_points]

            # Define centerline as midpoint of top edge and bottom edge
            self.centerline_start = (pts[0] + pts[3]) / 2
            self.centerline_end = (pts[1] + pts[2]) / 2

            # Approximate width from perpendicular vector
            edge_vec = np.array([pts[1].x() - pts[0].x(), pts[1].y() - pts[0].y()])
            perp_vec = np.array([-edge_vec[1], edge_vec[0]])
            height_vec = np.array([pts[0].x() - pts[3].x(), pts[0].y() - pts[3].y()])
            self.current_width = np.linalg.norm(height_vec)

            self.current_state = "bb_defined"
        except Exception as e:
            print(f"❌ Failed to restore centerline: {e}")


class BoundingBox:
    """
    Represents a bounding box for object detection.

    This class stores bounding box information in a normalized format 
    (relative to image dimensions) and provides methods for conversion 
    between different formats, such as QRectF (used in Qt) and string representation.

    Attributes:
        class_id (int): The class ID of the detected object.
        x_center (float): The normalized x-coordinate of the bounding box center.
        y_center (float): The normalized y-coordinate of the bounding box center.
        width (float): The normalized width of the bounding box.
        height (float): The normalized height of the bounding box.
        confidence (float, optional): The confidence score of the detection.
        segmentation (list, optional): Segmentation points if using instance segmentation.
        keypoints (list, optional): Keypoints if detected.
        obb (list, optional): Oriented Bounding Box coordinates (x1, y1, x2, y2, x3, y3, x4, y4, angle).
    """

    def __init__(self, class_id, x_center, y_center, width, height, confidence=None, 
                segmentation=None, keypoints=None, obb=None):
        self.class_id = int(class_id)
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.width = float(width)
        self.height = float(height)
        self.confidence = float(confidence) if confidence is not None else None
        self.segmentation = segmentation if segmentation is not None else []
        self.keypoints = keypoints if keypoints is not None else []
        self.obb = obb if obb is not None else []

    @classmethod
    def from_rect(cls, rect, img_width, img_height, class_id, confidence=None):
        x_center = (rect.x() + rect.width() / 2) / img_width
        y_center = (rect.y() + rect.height() / 2) / img_height
        width = rect.width() / img_width
        height = rect.height() / img_height
        return cls(class_id, x_center, y_center, width, height, confidence)

    def to_rect(self, img_width, img_height):
        x = self.x_center * img_width - self.width * img_width / 2
        y = self.y_center * img_height - self.height * img_height / 2
        width = self.width * img_width
        height = self.height * img_height
        return QRectF(x, y, width, height)

    @staticmethod
    def from_str(label_str):
        if not label_str.strip():
            logging.warning("Empty label string encountered.")
            return None

        parts = label_str.strip().split()
        if len(parts) < 2:
            logging.warning(f"Incomplete label string: {label_str}")
            return None

        try:
            class_id = int(parts[0])
            floats = list(map(float, parts[1:]))
        except (ValueError, IndexError) as e:
            logging.warning(f"Failed to parse floats from label string '{label_str}': {e}")
            return None

        # Your existing logic unchanged from here
        if len(floats) == 8:
            return BoundingBox(class_id, 0, 0, 0, 0, obb=floats)

        if len(floats) >= 6 and len(floats) % 2 == 0 and len(floats) != 8:
            return BoundingBox(class_id, 0, 0, 0, 0, segmentation=floats)

        if len(floats) == 4:
            x_center, y_center, width, height = floats
            return BoundingBox(class_id, x_center, y_center, width, height)

        if len(floats) >= 7 and (len(floats) - 4) % 3 == 0:
            x_center, y_center, width, height = floats[:4]
            keypoints = [(floats[i], floats[i + 1], int(floats[i + 2])) for i in range(4, len(floats), 3)]
            return BoundingBox(class_id, x_center, y_center, width, height, keypoints=keypoints)

        if len(floats) == 5:
            x_center, y_center, width, height, confidence = floats
            return BoundingBox(class_id, x_center, y_center, width, height, confidence=confidence)

        logging.warning(f"Unrecognized bbox format: {label_str}")
        return None

    def to_str(self, remove_confidence=False):
        class_id_str = f"{int(self.class_id)}"  # ✅ force integer formatting

        if self.obb:
            obb_str = ' '.join(f"{coord:.6f}" for coord in self.obb)
            return f"{class_id_str} {obb_str}"

        if self.segmentation:
            seg_str = ' '.join(f"{coord:.6f}" for coord in self.segmentation)
            return f"{class_id_str} {seg_str}"

        bbox_str = f"{class_id_str} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
        
        if self.keypoints:
            kpt_str = ' '.join(f"{x:.6f} {y:.6f} {v:d}" for x, y, v in self.keypoints)
            return f"{bbox_str} {kpt_str}"

        if self.confidence is not None and not remove_confidence:
            bbox_str += f" {self.confidence:.6f}"

        return bbox_str

    def to_dict(self):
        return {
            "class_id": self.class_id,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "segmentation": self.segmentation,
            "keypoints": self.keypoints,
            "obb": self.obb
        }

    @property
    def confidence_safe(self):
        try:
            return float(self.confidence)
        except (TypeError, ValueError):
            return 0.0
    def num_visible_keypoints(self):
        return sum(1 for kp in self.keypoints if isinstance(kp, (list, tuple)) and len(kp) == 3 and kp[2] == 2)
    
    def to_polygon(self, image_width, image_height):
        if self.segmentation:
            points = [(self.segmentation[i] * image_width, self.segmentation[i + 1] * image_height)
                    for i in range(0, len(self.segmentation), 2)]
            return Polygon(points)

        if self.obb:
            points = [
                (self.obb[i] * image_width, self.obb[i + 1] * image_height)
                for i in range(0, 8, 2)
            ]
            return Polygon(points)

        # Regular bounding box
        rect = self.to_rect(image_width, image_height)
        return shapely_box(rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())

# Creates a graphical user interface (GUI) for the settings dialog.
class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(700, 750)

        self.classHotkeyInputs = {}
        self.classColorPickers = {}

        self.tabs = QtWidgets.QTabWidget(self)
        self.class_tab = QtWidgets.QWidget()
        self.keybinds_tab = QtWidgets.QWidget()

        self.tabs.addTab(self.class_tab, "Classes & Hotkeys")
        self.tabs.addTab(self.keybinds_tab, "Keybinds")

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.tabs)

        self.init_class_tab()
        self.init_keybinds_tab()

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn)

    def init_class_tab(self):
        layout = QtWidgets.QVBoxLayout(self.class_tab)

        # Class List
        self.class_list = QtWidgets.QListWidget()
        layout.addWidget(QtWidgets.QLabel("Classes (with ID):"))
        layout.addWidget(self.class_list)

        # Controls
        controls = QtWidgets.QHBoxLayout()
        self.class_input = QtWidgets.QLineEdit()
        self.class_input.setPlaceholderText("Enter new or renamed class name")

        self.add_btn = QtWidgets.QPushButton("Add")
        self.rename_btn = QtWidgets.QPushButton("Rename")
        self.remove_btn = QtWidgets.QPushButton("Remove")

        controls.addWidget(self.class_input)
        controls.addWidget(self.add_btn)
        controls.addWidget(self.rename_btn)
        controls.addWidget(self.remove_btn)
        layout.addLayout(controls)

        self.add_btn.clicked.connect(self.add_class)
        self.remove_btn.clicked.connect(self.remove_class)
        self.rename_btn.clicked.connect(self.rename_class)
        self.class_input.returnPressed.connect(self.add_class)

        # Hotkeys + Colors
        layout.addWidget(QtWidgets.QLabel("Hotkeys and Color:"))
        self.hotkey_grid = QtWidgets.QGridLayout()
        self.hotkey_area = QtWidgets.QScrollArea()
        self.hotkey_area.setWidgetResizable(True)
        self.hotkey_widget = QtWidgets.QWidget()
        self.hotkey_widget.setLayout(self.hotkey_grid)
        self.hotkey_area.setWidget(self.hotkey_widget)
        layout.addWidget(self.hotkey_area)

        self.refresh_class_list()
        self.rebuild_hotkeys_ui()

    def init_keybinds_tab(self):
        layout = QtWidgets.QFormLayout(self.keybinds_tab)
        self.inputs = {}
        keys = [
            'nextButton', 'previousButton', 'deleteButton',
            'autoLabel', 'modeBox', 'modeSegmentation', 'modeKeypoint', 'modeOBB'
        ]
        for key in keys:
            field = QtWidgets.QLineEdit(self.parent().settings.get(key, ''))
            field.textChanged.connect(lambda val, k=key: self.save_keybind(k, val))
            layout.addRow(key.replace("Button", " Button").replace("mode", "Mode: "), field)
            self.inputs[key] = field

    def refresh_class_list(self):
        self.class_list.clear()
        self.class_names = self.parent().load_classes()
        for i, name in enumerate(self.class_names):
            self.class_list.addItem(f"{i}: {name}")

    def rebuild_hotkeys_ui(self):
        from PyQt5.QtWidgets import QLabel
        from PyQt5.QtCore import QSize

        # Clear existing widgets
        for i in reversed(range(self.hotkey_grid.count())):
            widget = self.hotkey_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self.classHotkeyInputs.clear()
        self.classColorPickers.clear()

        classes = self.parent().load_classes()
        num_classes = len(classes)

        for i, class_name in enumerate(classes):
            row = i

            # ID + Class Label
            label = QLabel(f"{i}: {class_name}")
            self.hotkey_grid.addWidget(label, row, 0)

            # Hotkey input
            hotkey = QtWidgets.QLineEdit(self.parent().settings.get(f'classHotkey_{class_name}', ''))
            hotkey.setMaximumWidth(50)
            hotkey.textChanged.connect(lambda val, cls=class_name: self.parent().settings.update({f'classHotkey_{cls}': val}))
            self.hotkey_grid.addWidget(hotkey, row, 1)
            self.classHotkeyInputs[class_name] = hotkey

            # Color preview (non-clickable)
            color = get_color(i, num_classes, alpha=255)
            color_label = QtWidgets.QLabel()
            color_label.setFixedSize(QSize(30, 20))
            color_label.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #333;")
            self.hotkey_grid.addWidget(color_label, row, 2)


    def pick_color(self, btn, class_name):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            hex_val = color.name()
            btn.setStyleSheet(f"background-color: {hex_val}")
            self.parent().settings[f'classColor_{class_name}'] = hex_val
            self.parent().saveSettings()

    def add_class(self):
        new_class = self.class_input.text().strip()
        if not new_class:
            return
        classes = self.parent().load_classes()
        if new_class in classes:
            QtWidgets.QMessageBox.warning(self, "Exists", "This class already exists.")
            return
        classes.append(new_class)
        self._save_classes(classes)
        self.parent().update_classes_dropdown(classes)
        self.refresh_class_list()
        self.rebuild_hotkeys_ui()
        self.class_input.clear()

    def remove_class(self):
        selected = self.class_list.currentItem()
        if not selected:
            return
        text = selected.text()
        class_name = text.split(": ", 1)[-1]
        classes = self.parent().load_classes()
        if class_name not in classes:
            return
        classes.remove(class_name)
        self._save_classes(classes)
        self.parent().update_classes_dropdown(classes)
        self.refresh_class_list()
        self.rebuild_hotkeys_ui()

    def rename_class(self):
        selected = self.class_list.currentItem()
        new_name = self.class_input.text().strip()
        if not selected or not new_name:
            return
        old_name = selected.text().split(": ", 1)[-1]
        classes = self.parent().load_classes()
        if new_name in classes:
            QtWidgets.QMessageBox.warning(self, "Duplicate", "Class already exists.")
            return
        index = classes.index(old_name)
        classes[index] = new_name
        self._save_classes(classes)
        self.parent().update_classes_dropdown(classes)
        self.refresh_class_list()
        self.rebuild_hotkeys_ui()
        self.class_input.clear()

    def _save_classes(self, class_list):
        path = os.path.join(
            self.parent().image_directory or self.parent().output_path or os.getcwd(),
            "classes.txt"
        )
        try:
            with open(path, "w") as f:
                f.write("\n".join(class_list) + "\n")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def save_keybind(self, key, value):
        self.parent().settings[key] = value
        self.parent().saveSettings()




# Base Video Processor Class
class VideoProcessor(QObject):
    progress_updated = pyqtSignal(int)  # Signal to indicate progress updates

    def __init__(self):
        super().__init__()
        self.stop_processing = False

    def process_video(self, video_path, output_dir, extract_rate=1, custom_size=None):
        """
        Process a video, extracting frames at the specified rate.

        Args:
            video_path (str): Path to the input video.
            output_dir (str): Directory to save extracted frames.
            extract_rate (int): Extract every nth frame.
            custom_size (tuple): Custom resize dimensions (width, height).
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error(f"Error: Unable to open video file {video_path}")
            return

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        os.makedirs(output_dir, exist_ok=True)

        while True:
            ret, frame = video.read()
            if not ret:
                break

            if custom_size:
                frame = cv2.resize(frame, custom_size, interpolation=cv2.INTER_LANCZOS4)

            if frame_count % extract_rate == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(frame_path, frame)

            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            self.progress_updated.emit(progress)

            if self.stop_processing:
                logger.warning("Processing stopped.")
                break

        video.release()


# thread for camera processing and frame extraction.
class GUIVideoProcessor(VideoProcessor):
    def __init__(self, progress_bar: QProgressBar):
        super().__init__()
        self.videos = []
        self.extract_all_frames = False
        self.custom_frame_count = None
        self.custom_size = None
        self.output_path = ""
        self.label_progress = progress_bar  # Set progress bar from GUI
        self.stop_processing = False

        # Connect the parent progress_updated signal directly to the QProgressBar
        self.progress_updated.connect(self.label_progress.setValue)
    def stop(self):
        """
        Stop the current processing operation.
        """
        self.stop_processing = True
        logger.warning("Processing stopped by user.")


    def add_video(self, video_path):
        if video_path not in self.videos:
            self.videos.append(video_path)
            logger.info(f"Added video: {video_path}")

    def remove_video(self):
        if self.videos:
            removed_video = self.videos.pop()
            logger.info(f"Removed video: {removed_video}")
            return removed_video
        else:
            logger.info("No videos to remove.")
            return None

    def set_output_path(self, path):
        self.output_path = path

    def set_extract_all_frames(self, value):
        self.extract_all_frames = value

    def set_custom_frame_count(self, count):
        self.custom_frame_count = count

    def set_custom_size(self, size):
        self.custom_size = size

    def run(self):
        """
        Processes all videos added to the list.
        """
        if not self.output_path:
            QMessageBox.warning(None, "Warning", "Output directory not set!")
            return

        self.label_progress.reset()
        self.label_progress.setMaximum(len(self.videos))

        for idx, video_path in enumerate(self.videos):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(self.output_path, f"{video_name}_Frames")

            extract_rate = 1 if self.extract_all_frames else (self.custom_frame_count or 1)
            try:
                self.process_video(video_path, output_dir, extract_rate, self.custom_size)
            except Exception as e:
                logger.error(f"Error processing video {video_path}: {e}")

            self.label_progress.setValue(idx + 1)

        QMessageBox.information(None, "Info", "Video frame extraction completed.")




# scan class for handling invalid annotations
class InvalidAnnotationError(Exception):
    pass

class InvalidImageError(Exception):
    pass

class ScanAnnotations:
    def __init__(self, parent):
        self.parent = parent
        self.valid_classes = []
        self.base_directory = ""
        self.total_images = 0
        self.total_labels = 0
        self.bad_labels = 0
        self.bad_images = 0
        self.review_folder = "review"
        self.metadata_queue = queue.Queue()
        self.metadata_thread = threading.Thread(target=self.metadata_removal_thread)
        self.metadata_thread.daemon = True
        self.metadata_thread.start()
        self.review_folder = os.path.join(self.base_directory, "review")
        os.makedirs(self.review_folder, exist_ok=True)
        self.valid_image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    def is_class_index_valid(self, class_index):
        if isinstance(self.valid_classes, dict):
            return class_index in self.valid_classes
        elif isinstance(self.valid_classes, list):
            return 0 <= class_index < len(self.valid_classes)
        return False

    def save_labels_to_file(self, file_path, lines, mode='w'):
        try:
            with open(file_path, mode) as f:
                for line in lines:
                    f.write(line.strip() + '\n')
        except Exception as e:
            logging.error(f"Error saving cleaned labels to {file_path}: {e}")

    def check_annotation_file(self, file_path, image_folder):
        issues = []
        lines_to_keep = []
        img_file = None

        # 🔍 Find matching image file
        for ext in self.valid_image_extensions:
            potential_img_file = os.path.basename(file_path).replace(".txt", ext)
            potential_path = os.path.join(image_folder, potential_img_file)
            if os.path.exists(potential_path):
                img_file = potential_img_file
                break

        if img_file is None:
            return [f"Warning: No image file found for '{os.path.basename(file_path)}'"], lines_to_keep, False

        img_path = os.path.join(image_folder, img_file)
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            return [f"Warning: Error opening image file: {e}"], lines_to_keep, False

        # 🧠 Use your BoundingBox class for parsing and validation
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                bbox = BoundingBox.from_str(line)
                if bbox is None:
                    issues.append((line.strip(), ["Unrecognized or malformed annotation"]))
                    continue

                # ✅ Validate class index
                if not self.is_class_index_valid(bbox.class_id):
                    issues.append((line.strip(), [f"Invalid class index {bbox.class_id}"]))
                    continue

                # ✅ Validate width/height if bbox-based (not segmentation or obb-only)
                if not bbox.segmentation and not bbox.obb:
                    if bbox.width <= 0 or bbox.height <= 0:
                        issues.append((line.strip(), ["Width and height must be positive"]))
                        continue

                    abs_width = bbox.width * img_width
                    abs_height = bbox.height * img_height
                    if abs_width <= 1 or abs_height <= 1:
                        issues.append((line.strip(), ["Bounding box too small"]))
                        continue

                    # Clamp box values to [0,1]
                    bbox.x_center = min(max(bbox.x_center, 0.0), 1.0)
                    bbox.y_center = min(max(bbox.y_center, 0.0), 1.0)
                    bbox.width = min(max(bbox.width, 0.0), 1.0)
                    bbox.height = min(max(bbox.height, 0.0), 1.0)

                lines_to_keep.append(bbox.to_str())

        # 📝 Save cleaned labels
        if lines_to_keep:
            self.save_labels_to_file(file_path, lines_to_keep, 'w')

        return issues, lines_to_keep, bool(issues)

    


    def remove_metadata(self, image_path, output_path=None):
        try:
            with Image.open(image_path) as img:
                data = list(img.getdata())
                img_without_metadata = Image.new(img.mode, img.size)
                img_without_metadata.putdata(data)
                if output_path:
                    img_without_metadata.save(output_path)
                else:
                    img_without_metadata.save(image_path)
        except Exception as e:
            logging.error(f"Failed to remove metadata from {image_path}: {e}")

    def metadata_removal_thread(self):
        while True:
            try:
                image_path, output_path = self.metadata_queue.get(timeout=1)
                if os.path.exists(image_path):
                    self.remove_metadata(image_path, output_path)
                else:
                    logging.warning(f"File not found: {image_path}")
                self.metadata_queue.task_done()
            except queue.Empty:
                time.sleep(0.1)

    def process_image_for_metadata_removal(self, image_path):
        self.metadata_queue.put((image_path, None))

    def remove_orphan_json_files(self):
        json_files = glob.glob(os.path.join(self.base_directory, "*.json"))
        for json_file in json_files:
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            image_exists = any(
                os.path.exists(os.path.join(self.base_directory, f"{base_name}{ext}"))
                for ext in self.valid_image_extensions
            )
            if not image_exists:
                os.remove(json_file)
                logging.info(f"Removed orphan JSON file: {json_file}")

    def is_valid_image(self, file_path):
        try:
            with Image.open(file_path) as img:
                img.verify()
                img.close()
            with Image.open(file_path) as img:
                img.load()
            return True
        except (IOError, SyntaxError, OSError) as e:
            logging.error(f"Invalid image {file_path}: {e}")
            return False

    def remove_corrupted_images(self):
        for root, _, files in os.walk(self.base_directory):
            for file_name in files:
                if file_name.lower().endswith(self.valid_image_extensions):
                    file_path = os.path.join(root, file_name)
                    if not self.is_valid_image(file_path):
                        logging.info(f"Removing corrupted image: {file_path}")
                        os.remove(file_path)
                        annotation_file = os.path.splitext(file_path)[0] + ".txt"
                        if os.path.exists(annotation_file):
                            os.rename(annotation_file, os.path.join(self.review_folder, os.path.basename(annotation_file)))

    def import_classes(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self.parent, "Import Classes", "", "Classes Files (classes.txt)", options=options)
        if file_name:
            self.base_directory = os.path.dirname(file_name)
            self.valid_classes = self.parent.load_classes(data_directory=self.base_directory)
            if not self.valid_classes:
                QMessageBox.warning(self.parent, "Error", "The selected classes.txt file is empty or invalid.")
                return
            os.makedirs(os.path.join(self.base_directory, self.review_folder), exist_ok=True)
            QMessageBox.information(self.parent, "Success", "Classes imported successfully!")

    def move_files(self, annotation_folder, review_folder, file_name):
        annotation_folder = Path(annotation_folder)
        review_folder = Path(review_folder)
        txt_path = annotation_folder / file_name
        if not txt_path.exists():
            logging.error(f"Error: {file_name} not found in {annotation_folder}.")
            return
        try:
            shutil.move(str(txt_path), str(review_folder / file_name))
        except Exception as e:
            logging.error(f"Error moving file {file_name}: {e}")
        for ext in self.valid_image_extensions:
            image_file = file_name.replace(".txt", ext)
            image_path = annotation_folder / image_file
            if image_path.exists():
                try:
                    shutil.move(str(image_path), str(review_folder / image_file))
                except Exception as e:
                    logging.error(f"Error moving image {image_file}: {e}")
                break

    def process_files(self, file_name, annotation_folder, image_folder, review_folder, statistics):
        try:
            logging.info(f"Processing file: {file_name}")
            file_path = os.path.join(annotation_folder, file_name)

            if not os.path.exists(file_path):
                logging.warning(f"Annotation file {file_name} does not exist.")
                return

            # Scan the file for valid annotations
            issues, lines_to_keep, should_move = self.check_annotation_file(file_path, image_folder)

            # Count only good labels
            self.total_labels += len(lines_to_keep)

            # Record what was wrong with it
            if issues:
                statistics[file_name] = issues

            # If problems are detected, move to the review folder
            if should_move or len(issues) > 0:
                self.bad_labels += 1
                logging.info(f"Bad annotations found in {file_name}, moving to review folder.")
                self.move_files(annotation_folder, review_folder, file_name)

        except Exception as e:
            logging.error(f"Error while processing {file_name}: {e}")

    def scan_annotations(self):
        if not hasattr(self.parent, "load_classes"):
            logging.error("Parent object does not have 'load_classes' method.")
            return
        class_mapping = self.parent.load_classes(self.base_directory)
        if not class_mapping:
            logging.warning("No valid classes found in the provided directory.")
            return
        if isinstance(class_mapping, dict):
            self.valid_classes = list(class_mapping.keys())
        elif isinstance(class_mapping, list):
            self.valid_classes = list(range(len(class_mapping)))
        else:
            logging.error("Invalid class mapping format. Expected dict or list.")
            return
        if not self.base_directory:
            logging.warning("No base directory found. Please import a dataset first.")
            return
        annotation_folder = self.base_directory
        image_folder = self.base_directory
        review_folder = os.path.join(self.base_directory, "review")
        os.makedirs(review_folder, exist_ok=True)
        txt_files_set = {f for f in os.listdir(annotation_folder) if f.endswith(".txt") and f != "classes.txt" and not f.endswith(".names")}
        img_files_set = {f for f in os.listdir(image_folder) if f.lower().endswith(self.valid_image_extensions)}
        statistics = {}
        num_cores = os.cpu_count()
        max_workers = num_cores - 1 if num_cores > 1 else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for file_name in txt_files_set:
                executor.submit(self.process_files, file_name, annotation_folder, image_folder, review_folder, statistics)
        self.remove_orphan_json_files()
        self.total_images = len(img_files_set)
        for img_file in img_files_set:
            image_path = os.path.join(image_folder, img_file)
            self.process_image_for_metadata_removal(image_path)
        save_path = os.path.join(image_folder, 'statistics.json')
        try:
            with open(save_path, "w") as f:
                json.dump(OrderedDict([
                    ("total_images", self.total_images),
                    ("total_labels", self.total_labels),
                    ("bad_labels", self.bad_labels),
                    ("bad_images", self.bad_images),
                    ("problems", statistics)
                ]), f, indent=4)
            QMessageBox.information(self.parent, "Information", f"Statistics saved successfully to {save_path}!")
        except Exception as e:
            logging.error(f"Failed to save statistics: {e}")
            QMessageBox.critical(self.parent, "Error", f"Failed to save statistics: {e}")

#yt downloader  thread
class DownloadThread(QThread):
    update_status = pyqtSignal(int, str)  # Signal for status updates (e.g., "Download complete")
    update_progress = pyqtSignal(int, int)  # Signal for progress updates (row, progress)

    def __init__(self, url, row, directory):
        super().__init__()
        self.url = url
        self.row = row
        self.directory = directory

    def clean_filename(self, filename):
        # Clean and format the filename
        filename = re.sub(r'\W+', ' ', filename)  # Remove special characters
        filename = re.sub(r'\d+', '', filename)  # Remove numbers
        filename = filename.strip()  # Remove leading/trailing spaces
        return filename

    def run(self):
        retries = 3
        ydl_opts = {
            'format': 'bestvideo',
            'outtmpl': f'{self.directory}/%(title)s.%(ext)s',
            'progress_hooks': [self.yt_dlp_progress_hook],
            'noplaylist': True,
        }

        for attempt in range(retries):
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.url, download=True)
                    title = info.get('title', 'unknown_video')
                    filename = self.clean_filename(title)
                    logger.info(f"Downloaded video: {filename}")
                    self.update_progress.emit(self.row, 100)  # Set progress bar to 100%
                break

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(random.uniform(1, 3))
                else:
                    logger.error(f"Download failed with error: {e}")
                    self.update_status.emit(self.row, f'Failed: {str(e)}')  # Show error in table



    def yt_dlp_progress_hook(self, d):
        if d['status'] == 'downloading':
            # Get information about the current download state
            total_bytes = d.get('total_bytes', 0)
            downloaded_bytes = d.get('downloaded_bytes', 0)
            total_fragments = d.get('fragment_count', 1)  # Total number of fragments, if available
            current_fragment = d.get('fragment_index', 0)  # Current fragment index, if available

            # Calculate the progress percentage
            if total_bytes:
                # Progress based on total bytes
                progress = int((downloaded_bytes / total_bytes) * 100)
            elif total_fragments:
                # Progress based on fragment count (useful if total_bytes is not available)
                progress = int((current_fragment / total_fragments) * 100)
            else:
                progress = 0

            logger.debug(f"Progress: {progress}% for row {self.row}")
            self.update_progress.emit(self.row, progress)  # Emit progress signal for UI

        elif d['status'] == 'finished':
            # When the download finishes, set progress to 100%
            logger.info("Download finished, now processing...")
            self.update_progress.emit(self.row, 100)



class ImageConverterRunnableSignals(QObject):
    finished = pyqtSignal()  # Signal emitted when a file is processed


class ImageConverterRunnable(QRunnable):
    def __init__(self, directory, file, target_format, target_directory):
        super().__init__()
        self.signals = ImageConverterRunnableSignals()
        self.directory = directory
        self.file = file
        self.target_format = target_format
        self.target_directory = target_directory

    def run(self):
        try:
            reader = QImageReader(os.path.join(self.directory, self.file))
            image = reader.read()
            if image.isNull():
                raise Exception(f"Could not read {self.file}")

            target_file = os.path.join(
                self.target_directory, f"{os.path.splitext(self.file)[0]}.{self.target_format.lower()}"
            )
            writer = QImageWriter(target_file, self.target_format.encode())
            if not writer.write(image):
                raise Exception(f"Could not write {self.file}")

            txt_file = os.path.join(self.directory, f"{os.path.splitext(self.file)[0]}.txt")
            if os.path.isfile(txt_file):
                shutil.copy(txt_file, self.target_directory)
        finally:
            logger.info(f"Finished processing {self.file}")
            self.signals.finished.emit()  # Emit finished signal
            self.signals.finished.emit()  # Emit finished signal




# to make blanks

class ImageProcessingThread(QThread):
    progressSignal = pyqtSignal(int)

    def __init__(self, image_files, yolo_annotation_files, save_folder_path, limit_percentage, image_format):
        QThread.__init__(self)
        self.image_files = image_files
        self.yolo_annotation_files = yolo_annotation_files
        self.save_folder_path = save_folder_path
        self.limit_percentage = limit_percentage
        self.image_format = image_format
        self.counter = 0
        self.stable_diffusion_pipeline = self.load_stable_diffusion_pipeline()

    def load_stable_diffusion_pipeline(self):
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
            )
            pipe.to(DEVICE)
            # Enable memory-efficient attention (optional)
            pipe.enable_attention_slicing()
            return pipe
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def run(self):
        if self.stable_diffusion_pipeline is None:
            logger.critical("Pipeline not loaded. Exiting...")
            return

        # Get the total number of images
        total_images = len(self.image_files)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            # Enumerate and process each image file
            for counter, _ in enumerate(
                    executor.map(
                        lambda args: self.process_single_image(*args),
                        zip(
                            self.image_files,
                            self.yolo_annotation_files,
                            [self.save_folder_path] * total_images,
                            [self.limit_percentage] * total_images,
                            [self.image_format] * total_images,
                        ),
                    )
            ):
                # Emit progress signal as a percentage
                self.progressSignal.emit(int((counter / total_images) * 100))

        # Emit 100% when done
        self.progressSignal.emit(100)

    def process_single_image(self, image_path, yolo_annotation_path, save_folder_path, limit_percentage, image_format):
        """
        Process a single image using YOLO annotations and Stable Diffusion for inpainting.

        :param image_path: Path to the image file.
        :param yolo_annotation_path: Path to the YOLO annotation file.
        :param save_folder_path: Path to the folder where processed images will be saved.
        :param limit_percentage: The limit percentage for cropping.
        :param image_format: The image format to save (e.g., 'jpg', 'png').
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Error reading image: {image_path}")
            return
        original_img = img.copy()

        img_height, img_width, _ = img.shape
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # Read YOLO annotations
        with open(yolo_annotation_path, 'r') as f:
            yolo_data = f.readlines()

        for data in yolo_data:
            data = data.split()
            x_center, y_center, w, h = [float(x) for x in data[1:]]
            x_min = int((x_center - w / 2) * img_width)
            x_max = int((x_center + w / 2) * img_width)
            y_min = int((y_center - h / 2) * img_height)
            y_max = int((y_center + h / 2) * img_height)
            mask[y_min:y_max, x_min:x_max] = 255
            img[y_min:y_max, x_min:x_max] = 0

        img = self.fill_cropped_area(original_img, img, mask)

        if img is None:
            logger.error(f"Failed to inpaint image: {image_path}")
            return

        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        negative_filename = f'{base_filename}_no_labels_negative.{image_format}'
        negative_img = img.copy()
        self.save_image_with_limit(negative_img, save_folder_path, negative_filename, limit_percentage, image_format)

    def fill_cropped_area(self, original_img, img, mask):
        # Adjust the prompt to better guide the model
        prompt = "A photo of trees and bushes, detailed foliage, natural scenery"
        # Add a negative prompt to discourage generating humans
        negative_prompt = "person, people, human, man, woman, child, portrait, face, body"
        try:
            mask_image = Image.fromarray(mask).convert("RGB")
            image = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            result = self.stable_diffusion_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask_image,
                num_inference_steps=50,  # You can adjust this for quality vs. speed
                guidance_scale=7.5        # Adjust for prompt adherence
            )
            inpainted_img = result.images[0]
            return cv2.cvtColor(np.array(inpainted_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error in inpainting: {e}")
            return None

    def save_image_with_limit(self, img, save_folder_path, filename, limit, image_format):
        save_negative_image = random.choices([True, False], weights=[limit, 100 - limit], k=1)[0]
        if save_negative_image and not self.is_solid_color(img):
            negative_filename = os.path.join(save_folder_path, filename)
            cv2.imwrite(negative_filename, img)
            with open(negative_filename.replace(f'.{image_format}', '.txt'), 'w'):
                pass
            self.counter += 1

    def is_solid_color(self, img):
        return np.all(img == img[0, 0])

#border gradient
class SolidGradientBorderItem(QGraphicsItem):
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.gradient = QLinearGradient(0, 0, width, 0)
        self.gradient.setColorAt(0.0, QColor(255, 0, 0, 255))    # Red
        self.gradient.setColorAt(0.4, QColor(0, 255, 0, 255))    # Green
        self.gradient.setColorAt(0.6, QColor(0, 0, 255, 255))    # Blue
        self.gradient.setColorAt(1.0, QColor(
            255, 0, 0, 255))    # Red (to reconnect)

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget):
        pen = QPen()
        pen.setWidth(5)
        pen.setBrush(self.gradient)
        painter.setPen(pen)
        painter.drawRect(0, 0, self.width, self.height)

class SahiSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAHI Settings")
        layout = QVBoxLayout()

        # Model type selection
        self.modelTypeComboBox = QComboBox()
        self.modelTypeComboBox.addItems(["yolov8", "yolov7", "yolov5", "yolov9"])  # Add other models as needed
        layout.addWidget(QLabel("Model Type:"))
        layout.addWidget(self.modelTypeComboBox)

        # Model weights selection
        self.weightsLineEdit = QLineEdit()
        selectWeightsBtn = QPushButton("Select Model Weights")
        selectWeightsBtn.clicked.connect(self.selectModelWeights)
        layout.addWidget(QLabel("Model Weights:"))
        layout.addWidget(self.weightsLineEdit)
        layout.addWidget(selectWeightsBtn)

        # Image directory selection
        self.imageDirLineEdit = QLineEdit()
        selectImageDirBtn = QPushButton("Select Image Directory")
        selectImageDirBtn.clicked.connect(self.selectImageDirectory)
        layout.addWidget(QLabel("Image Directory:"))
        layout.addWidget(self.imageDirLineEdit)
        layout.addWidget(selectImageDirBtn)

        # Slice height and width with default values
        self.sliceHeightSpinBox = QSpinBox()
        self.sliceHeightSpinBox.setRange(100, 1024)
        self.sliceHeightSpinBox.setValue(256)  # Default value
        self.sliceWidthSpinBox = QSpinBox()
        self.sliceWidthSpinBox.setRange(100, 1024)
        self.sliceWidthSpinBox.setValue(256)  # Default value
        layout.addWidget(QLabel("Slice Height:"))
        layout.addWidget(self.sliceHeightSpinBox)
        layout.addWidget(QLabel("Slice Width:"))
        layout.addWidget(self.sliceWidthSpinBox)

        # Overlap ratio
        self.overlapHeightRatioSpinBox = QDoubleSpinBox()
        self.overlapHeightRatioSpinBox.setRange(0.0, 1.0)
        self.overlapHeightRatioSpinBox.setSingleStep(0.1)
        self.overlapHeightRatioSpinBox.setValue(0.2)  # Default value
        layout.addWidget(QLabel("Overlap Height Ratio:"))
        layout.addWidget(self.overlapHeightRatioSpinBox)

        self.overlapWidthRatioSpinBox = QDoubleSpinBox()
        self.overlapWidthRatioSpinBox.setRange(0.0, 1.0)
        self.overlapWidthRatioSpinBox.setSingleStep(0.1)
        self.overlapWidthRatioSpinBox.setValue(0.2)  # Default value
        layout.addWidget(QLabel("Overlap Width Ratio:"))
        layout.addWidget(self.overlapWidthRatioSpinBox)

        # Confidence threshold
        self.confidenceThresholdSpinBox = QDoubleSpinBox()
        self.confidenceThresholdSpinBox.setRange(0.0, 1.0)
        self.confidenceThresholdSpinBox.setSingleStep(0.1)
        self.confidenceThresholdSpinBox.setValue(0.4)  # Default value
        layout.addWidget(QLabel("Confidence Threshold:"))
        layout.addWidget(self.confidenceThresholdSpinBox)

        # Confirm settings button
        confirmBtn = QPushButton("Confirm Settings")
        confirmBtn.clicked.connect(self.confirmSettings)
        layout.addWidget(confirmBtn)

        self.setLayout(layout)

    def selectModelWeights(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Model Weights File", "", "Model Weights (*.pt)")
        self.weightsLineEdit.setText(fileName)

    def selectImageDirectory(self):
        dirName = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        self.imageDirLineEdit.setText(dirName)

    def confirmSettings(self):
        model_type = self.modelTypeComboBox.currentText()
        model_weights = self.weightsLineEdit.text()
        image_directory = self.imageDirLineEdit.text()
        slice_height = self.sliceHeightSpinBox.value()
        slice_width = self.sliceWidthSpinBox.value()
        overlap_height_ratio = self.overlapHeightRatioSpinBox.value()
        overlap_width_ratio = self.overlapWidthRatioSpinBox.value()
        confidence_threshold = self.confidenceThresholdSpinBox.value()

        # Check if model weights are selected
        if not model_weights:
            QMessageBox.warning(self, "Error", "Please select model weights.")
            return

        # Check if image directory is selected
        if not image_directory:
            QMessageBox.warning(self, "Error", "Please select an image directory.")
            return

        # Check if classes.txt exists in the image directory
        classes_txt_path = os.path.join(image_directory, 'classes.txt')
        if not os.path.exists(classes_txt_path):
            QMessageBox.warning(self, "Error", f"classes.txt not found in the image directory: {image_directory}")
            return

        try:
            # Instantiate the SahiPredictWrapper with user-defined settings
            sahi_predictor = SahiPredictWrapper(
                model_type=model_type,
                model_path=model_weights,
                confidence_threshold=confidence_threshold,
                device="cuda:0"  # or "cpu", based on user input
            )

            # Start the prediction process
            sahi_predictor.process_folder(
                folder_path=image_directory,
                class_names_file=classes_txt_path,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio
            )

            self.accept()  # Close the dialog
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def showSahiSettings(self):
        self.sahiSettingsDialog = SahiSettingsDialog(self)
        self.sahiSettingsDialog.exec_()
   
        
# to put sam on different thread.        
class ImageProcessor(QThread):
    update_signal = pyqtSignal(np.ndarray)

    def __init__(self, image_files, parent=None):
        super(ImageProcessor, self).__init__(parent)
        self.image_files = image_files

    def run(self):
        for file_path in self.image_files:
            image = cv2.imread(file_path)
            if image is not None:
                processed_image = self.predict_and_draw_yolo_objects(image, file_path)
                self.update_signal.emit(processed_image)  # Signal to update GUI

class UiLoader:
    def __init__(self, main_window):
        self.main_window = main_window

    def setup_ui(self, show_images=True):
        self.main_window.preview_list.setUpdatesEnabled(False)  # 🚀 Prevent UI from updating mid-setup
        self.main_window.preview_list.setRowCount(0)
        self.main_window.preview_list.setColumnCount(5)
        self.main_window.preview_list.setHorizontalHeaderLabels(['Image', 'Class Name', 'ID', 'Size', 'Bounding Box'])

        # Set up placeholder image
        placeholder_pixmap = QPixmap(128, 128)
        placeholder_pixmap.fill(Qt.gray)  

        # Column widths
        column_widths = [100 if show_images else 0, 50, 25, 50, 250]
        for col, width in enumerate(column_widths):
            self.main_window.preview_list.setColumnWidth(col, width)

        self.main_window.preview_list.setUpdatesEnabled(True)  # ✅ Re-enable UI updates after setup

        logger.info("UI setup completed.")



class CropWorker(QThread):
    update_progress = pyqtSignal(int)  # Signal to update the progress bar

    def __init__(self, directory, width, height):
        super().__init__()
        self.directory = directory
        self.width = width
        self.height = height
        
    def run(self):
        image_files = [f for f in os.listdir(self.directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        total = len(image_files)
        cropped_dir = os.path.join(self.directory, 'cropped')
        os.makedirs(cropped_dir, exist_ok=True)

        for i, filename in enumerate(image_files):
            image_path = os.path.join(self.directory, filename)
            label_path = image_path.rsplit('.', 1)[0] + '.txt'
            image = Image.open(image_path)
            bboxes = self.read_labels(label_path, image.width, image.height)

            for j, bbox in enumerate(bboxes):
                cropped_image = self.crop_and_adjust_bbox(image, bbox)
                cropped_image_path = os.path.join(cropped_dir, f"{filename.rsplit('.', 1)[0]}_{j}.png")
                cropped_image.save(cropped_image_path)

            self.update_progress.emit((i + 1) * 100 // total)

    def read_labels(self, label_path, img_width, img_height):
        bboxes = []
        try:
            with open(label_path, 'r') as file:
                for line in file.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    x1 = (x_center - width / 2) * img_width
                    y1 = (y_center - height / 2) * img_height
                    x2 = (x_center + width / 2) * img_width
                    y2 = (y_center + height / 2) * img_height
                    bboxes.append((class_id, x1, y1, x2, y2))
        except FileNotFoundError:
            logger.warning(f"No label file found for {label_path}")
        return bboxes

    def crop_and_adjust_bbox(self, image, bbox):
        _, x1, y1, x2, y2 = bbox  # Corrected unpacking with an underscore for class_id
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integer

        # Adjust the cropping size to ensure it does not exceed the image boundaries
        if (x2 - x1) < self.width:
            shift_x = min((self.width - (x2 - x1)) // 2, max(0, x1), max(0, image.width - x2))
            x1 = max(0, x1 - shift_x)
            x2 = min(image.width, x2 + shift_x)
        if (y2 - y1) < self.height:
            shift_y = min((self.height - (y2 - y1)) // 2, max(0, y1), max(0, image.height - y2))
            y1 = max(0, y1 - shift_y)
            y2 = min(image.height, y2 + shift_y)

        # Crop the image using the adjusted coordinates
        cropped = image.crop((x1, y1, x2, y2))

        # Resize the cropped area if it is smaller than desired size (no padding)
        if cropped.size[0] < self.width or cropped.size[1] < self.height:
            cropped = cropped.resize((self.width, self.height), Image.LANCZOS)

        return cropped

class RedBoxDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        """
        Custom painting to draw a red border around the selected item.
        """
        # Call the base class paint method to do the default rendering
        super().paint(painter, option, index)

        # Check if the item is selected
        if option.state & QStyle.State_Selected:
            # Create a red pen for drawing the border
            pen = QPen(QColor("red"), 2, Qt.SolidLine)
            painter.setPen(pen)

            # Draw a rectangle around the item (to simulate a border)
            rect = option.rect
            painter.drawRect(rect.adjusted(1, 1, -1, -1))  # Draw inside the item boundaries

# class that adds checkboxes to class drop down list
class CheckboxDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
     
    def paint(self, painter, option, index):
        painter.save()

        # Get the item text and checkbox state
        text = index.data(Qt.DisplayRole)
        checked = index.data(Qt.CheckStateRole) == Qt.Checked

        # Draw the background if selected or hovered
        if option.state & QStyle.State_Selected or option.state & QStyle.State_MouseOver:
            painter.fillRect(option.rect, option.palette.highlight())

        # Draw the checkbox
        checkbox_style = QApplication.style()
        checkbox_option = QStyleOptionButton()
        checkbox_rect = checkbox_style.subElementRect(QStyle.SE_CheckBoxIndicator, checkbox_option, None)
        checkbox_option.rect = QRect(
            option.rect.left(),
            option.rect.top() + (option.rect.height() - checkbox_rect.height()) // 2,
            checkbox_rect.width(),
            checkbox_rect.height()
        )
        checkbox_option.state = QStyle.State_Enabled | (QStyle.State_On if checked else QStyle.State_Off)
        checkbox_style.drawControl(QStyle.CE_CheckBox, checkbox_option, painter)

        # Determine the color for the circle
        class_id = index.row()
        num_classes = self.parent().model().rowCount()
        circle_color = get_color(class_id, num_classes)

        # Set text color based on state
        if option.state & QStyle.State_Selected or option.state & QStyle.State_MouseOver:
            painter.setPen(option.palette.color(QPalette.HighlightedText))
        else:
            painter.setPen(option.palette.color(QPalette.Text))

        # Draw the text
        text_rect = option.rect.adjusted(checkbox_rect.width() + 5, 0, -20, 0)
        painter.drawText(text_rect, Qt.AlignVCenter, text)

        # Draw the colored circle
        circle_diameter = 10
        circle_x = text_rect.right() + 5
        circle_y = option.rect.top() + (option.rect.height() - circle_diameter) // 2
        painter.setBrush(circle_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(circle_x, circle_y, circle_diameter, circle_diameter))

        painter.restore()


    
    def editorEvent(self, event, model, option, index):
        if event.type() == QEvent.MouseButtonPress:
            # Determine if the click is within the checkbox bounds
            checkbox_style = QApplication.style()
            checkbox_option = QStyleOptionButton()
            checkbox_rect = QApplication.style().subElementRect(QStyle.SE_CheckBoxIndicator, checkbox_option, None)
            checkbox_rect.moveTopLeft(option.rect.topLeft())
            checkbox_rect.setTop(option.rect.top() + (option.rect.height() - checkbox_rect.height()) // 2)

            if checkbox_rect.contains(event.pos()):
                # Toggle the checkbox state
                current_state = index.data(Qt.CheckStateRole)
                new_state = Qt.Unchecked if current_state == Qt.Checked else Qt.Checked
                model.setData(index, new_state, Qt.CheckStateRole)
                return True
        return super().editorEvent(event, model, option, index)



#class to run grounding dino on seperate thread. to avoid interfering with the ui. 

class DinoWorker(QThread):
    finished = pyqtSignal()  # Signal emitted when the task is done
    error = pyqtSignal(str)  # Signal emitted if there's an error

    def __init__(self, image_directory, overwrite):
        super().__init__()
        self.image_directory = image_directory
        self.overwrite = overwrite

    def run(self):
        try:
            # Call the function from dino.py
            run_groundingdino(self.image_directory, self.overwrite)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()



class DeduplicationWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, image_directory, get_image_files_func, remove_duplicates_func, parent=None):
        super().__init__(parent)  # ✅ Pass parent to QThread
        self.image_directory = image_directory
        self.get_image_files = get_image_files_func
        self.remove_duplicates = remove_duplicates_func


    def run(self):
        txt_files = [os.path.splitext(file)[0] + '.txt' for file in self.get_image_files(self.image_directory)]
        total = len(txt_files)

        parent = self.parent()
        for idx, txt_file in enumerate(txt_files):
            if hasattr(parent, 'deduplicate_single_label_file'):
                parent.deduplicate_single_label_file(txt_file)
            else:
                logger.error("❌ Parent does not have deduplicate_single_label_file")

            self.progress.emit(int((idx + 1) / total * 100))

        self.finished.emit()


class ImageLoaderThread(QThread):
    update_progress = pyqtSignal(int)  # Signal to update UI progress
    images_loaded = pyqtSignal(list)   # Signal to update UI with images

    def __init__(self, image_directory):
        super().__init__()
        self.image_directory = image_directory

    def run(self):
        """Load images in a separate thread to prevent UI freezing."""
        image_files = [
            os.path.join(self.image_directory, f).replace("\\", "/")
            for f in os.listdir(self.image_directory)
            if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.webp'}
        ]

        sorted_files = sorted(image_files, key=lambda x: os.path.getsize(x), reverse=True)

        for i, _ in enumerate(sorted_files):
            self.update_progress.emit(int((i / len(sorted_files)) * 100))  # Update progress bar

        self.images_loaded.emit(sorted_files)  # Send loaded images to UI

class QLabelInfoLogHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.last_message = ""

    def emit(self, record):
        try:
            msg = self.format(record)
            self.last_message = msg
        except Exception as e:
            print(f"Log handler error: {e}")


ui_file: Path = Path(__file__).resolve().parent / "UltraDarkFusion_v5.ui"
with open(ui_file, "r", encoding="utf-8") as file:
    Ui_MainWindow: Type[QtWidgets.QMainWindow]
    QtBaseClass: Type[object]
    Ui_MainWindow, QtBaseClass = uic.loadUiType(file)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    MAX_SIZE=200
    keyboard_interrupt_signal = pyqtSignal()
    progress_signal = pyqtSignal(int)
    clear_dropdown_signal = pyqtSignal()
    add_item_signal = pyqtSignal(str)
    rightClicked = pyqtSignal(QModelIndex)
    leftClicked = pyqtSignal(QModelIndex)
    signalThumbnailCreated = QtCore.pyqtSignal(int, QtGui.QPixmap)  
    def __init__(self):
        super().__init__()
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        global start_time
        start_time = datetime.now()

        # Set up icons for window
        self.icons = [
            QIcon('styles/icons/Df1.png'),
            QIcon('styles/icons/Df2.png'),
            # Add paths to other frames here if needed
        ]
        self.setWindowIcon(self.icons[0])
        self.last_logged_file_name = None 
        # Check for CUDA availability in PyTorch and OpenCV
        self.pytorch_cuda_available = torch.cuda.is_available()
        self.opencv_cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        self.heatmap_colormap_options = {
            "JET": cv2.COLORMAP_JET,                         # 🔥 Standard go-to. Bright, high contrast.
            "TURBO": cv2.COLORMAP_TURBO,                     # 🌈 Cleaner, perceptually uniform. Great for AI.
            "MAGMA": cv2.COLORMAP_MAGMA,                     # 🧨 Darker, scientific. Good on black UIs.
            "INFERNO": cv2.COLORMAP_INFERNO,                 # 🔥🔥 Similar to MAGMA but hotter.
            "PLASMA": cv2.COLORMAP_PLASMA,                   # 🔮 Great for gradients. Smooth.
            "VIRIDIS": cv2.COLORMAP_VIRIDIS,                 # 🌿 Beautiful and colorblind friendly.
            "CIVIDIS": cv2.COLORMAP_CIVIDIS,                 # ♿ High accessibility (colorblind safe).
            "HOT": cv2.COLORMAP_HOT,                         # 🥵 Very intense red-yellow-white. Sharp.
            "COOL": cv2.COLORMAP_COOL,                       # 🧊 Clean cyan-to-magenta transition.
            "HSV": cv2.COLORMAP_HSV,                         # 🌈 Classic hue circle, great for segment ID.
            "BONE": cv2.COLORMAP_BONE,                       # ☠️ Subtle grayscale tone. Great for X-ray feel.
            "OCEAN": cv2.COLORMAP_OCEAN,                     # 🌊 Smooth dark blues. Good for depth maps.
            "PINK": cv2.COLORMAP_PINK,                       # 🎀 Looks like soft medical heatmaps.
            "SPRING": cv2.COLORMAP_SPRING,                   # 💐 Bright magenta-yellow. Kinda wild.
            "SUMMER": cv2.COLORMAP_SUMMER,                   # 🏖️ Lime green to yellow. Feels fresh.
            "AUTUMN": cv2.COLORMAP_AUTUMN,                   # 🍁 Red to yellow. Fall vibe.
            "WINTER": cv2.COLORMAP_WINTER,                   # ❄️ Blue to green. Cold-themed.
            "RAINBOW": cv2.COLORMAP_RAINBOW,                 # 🌈 Classic full-spectrum map.
            "PARULA": cv2.COLORMAP_PARULA,                   # 🧪 MATLAB style, easy on the eyes.
            "DEEPGREEN": cv2.COLORMAP_DEEPGREEN              # 🌲 Experimental but stylish.
        }        
        # Video processor setup
        self.video_processor = GUIVideoProcessor(self.label_progress)
     

        # Set window title
        self.setWindowTitle("UltraDarkFusion")

        # Icon timer setup
        self.icon_timer = QTimer(self)
        self.icon_timer.timeout.connect(self.update_icon)
        self.icon_timer.start(500)
        self.current_icon_index = 0
        self.threadpool = QThreadPool()
        # Console output timer setup
        self.logger = logging.getLogger("UltraDarkFusionLogger")

        # Create and attach QLabelInfoLogHandler
        self.label_log_handler = QLabelInfoLogHandler()
        self.label_log_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self.label_log_handler)

        self.console_output_timer = QTimer(self)
        self.console_output_timer.timeout.connect(self.update_console_output)
        self.console_output_timer.start(1000)
        #plots and graphics
        self.histogram_plot.triggered.connect(lambda: self.create_plot('histogram'))
        self.bar_plot.triggered.connect(lambda: self.create_plot('bar'))
        self.scatter_plot.triggered.connect(lambda: self.create_plot('scatter'))
        self.call_stats_button.triggered.connect(self.display_stats)        
        self.current_file = None
        self.label_file = None
        self.file_name = None 
        # Initialize the screen view and graphics scene
        self.view_references = []
        self.screen_view = CustomGraphicsView(main_window=self)
        self.screen_view.setBackgroundBrush(QBrush(Qt.black))
        self.setCentralWidget(self.screen_view)
        self.screen_view.setRenderHint(QPainter.Antialiasing)
        self.screen_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_scene = QGraphicsScene(self)
        self.selected_bbox = None
        self.screen_view.setScene(self.graphics_scene)         
       
        
        # Initialize buttons and timers
        self.next_button.clicked.connect(self.next_frame)
        self.next_timer = QTimer()
        self.next_timer.timeout.connect(self.next_frame)
        self.next_button.pressed.connect(self.start_next_timer)
        self.next_button.released.connect(self.stop_next_timer)
        self.timmer_speed.valueChanged.connect(self.update_timer_speed)
        self.timer_interval = self.timmer_speed.value()
        self.previous_button.clicked.connect(self.previous_frame)
        self.prev_timer = QTimer()
        self.prev_timer.timeout.connect(self.previous_frame)
        self.previous_button.pressed.connect(self.start_prev_timer)
        self.previous_button.released.connect(self.stop_prev_timer) 
        
        #bounding box and class dropdown
        self.clear_dropdown_signal.connect(self.clear_classes_dropdown)
        self.add_item_signal[str].connect(self.add_item_to_classes_dropdown)  # Fix: Specify the argument type for add_item_signal
        self.classes_dropdown.currentIndexChanged.connect(self.change_class_id)
        
        #see auto save bounding boxes
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self.auto_save_bounding_boxes)
        self.auto_save_timer.start(10000)  # Save every -n seconds

        self.weights_file = None
        self.cfg_file = None
        self.weights_files = []

        #class input field and class id
     
        self.current_class_id = 0
        #confidence and nms threshold
        self.confidence_threshold_slider.valueChanged.connect(self.slider_value_changed)
        self.confidence_threshold_slider.setRange(0, 100)
        self.confidence_threshold = self.confidence_threshold_slider.value() / 100
        self.nms_threshold_slider.valueChanged.connect(self.slider_value_changed)
        self.nms_threshold_slider.setRange(0, 100)       
        self.nms_threshold = self.nms_threshold_slider.value() / 100


        self.crop_img_button.clicked.connect(self.crop_images)
        self.select_images_crop.clicked.connect(self.select_directory)

        self.delete_timer = QTimer(self)
        self.delete_timer.setInterval(100)  # The interval in milliseconds
        self.delete_timer.timeout.connect(self.delete_current_image)
        self.delete_button.pressed.connect(self.delete_timer.start)
        self.delete_button.released.connect(self.delete_timer.stop)

        self.image_list = []
        self.class_to_id = {}
        self.zoom_mode = False
        self.original_transform = None

        # settings dialog see def openSettingsDialog
        self.settingsButton.triggered.connect(self.openSettingsDialog)
        self.settings = self.loadSettings()


        
        self.class_names = []  # Initialize as an empty list

        self.image_files = []  # Initialize empty image files list
        self.filtered_image_files = []  # For filtered images
        self.current_image_index = 0  # To track current image
        self.current_file = None  # Current displayed file

        # Delay ComboBox initialization
        self.filter_class_spinbox.addItem("All (-1)")  # Special option: All
        self.filter_class_spinbox.addItem("Blanks (-2)")  # Special option: Blanks
        self.filter_class_spinbox.currentIndexChanged.connect(self.on_filter_class_spinbox_changed)


        self.update_filter_spinbox()  # Populate with initial values
        self.List_view.clicked.connect(self.on_list_view_clicked)
        self.filtered_image_files = []
        self.hide_label_checkbox.toggled.connect(self.toggle_label_visibility)
        self.hide_label_checkbox.setChecked(False)
        self.setup_list_view_with_delegate()
        self.image = None


        # preview function see def extract_and_display_data

        self.flash_time_spinbox.valueChanged.connect(self.update_flash_time)
        self.flash_color_button.clicked.connect(self.pick_flash_color)
        self.alternate_flash_color_button.clicked.connect(self.pick_alternate_flash_color)
        
        self.flash_color_rgb = (255, 0, 0)  # Default flash color
        self.alternate_flash_color_rgb = (0, 0, 255)  # Default alternate flash color
        self.flash_time_value = self.flash_time_spinbox.value()
        self.screen_view.viewport().installEventFilter(self)

        self.dont_show_img_checkbox.stateChanged.connect(self.toggle_image_display)
        self.show_images = True 
        self.ui_loader = UiLoader(self)

        self.batch_size_spinbox.valueChanged.connect(self.update_batch_size)
        self.image_size.valueChanged.connect(self.start_debounce_timer)                
        self.ui_loader.setup_ui()

        self.thumbnails_directory = None
        self.id_to_class = {}
        self.label_dict = {}
        self.current_bounding_boxes = []

        self.batch_directory = ""
        self.batch_running = False
        self.current_bbox_index = {}  # Track the current bounding box index for each image
        self.flash_time = QSpinBox()  # Placeholder for actual flash time value control

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._perform_size_adjustment)
        self._debounce_timer.setInterval(50)  # 100 milliseconds #event filter
        
        QApplication.instance().installEventFilter(self)
        self.preview_button.clicked.connect(self.extract_and_display_data)
        self.process_folder_btn.clicked.connect(self.process_batch)

        self.move_all_button.clicked.connect(self.move_filtered_images)
        self.clear_all_button.clicked.connect(self.clear_class_boxes)        
        self.preview_list.viewport().installEventFilter(self)
        self.rightClicked.connect(self.handle_right_click)
        self.leftClicked.connect(self.handle_left_click)
        self.review_off.clicked.connect(self.stop_processing_and_clear)
        self.image_select.clicked.connect(self.convert_directory)
        # opencv dnn see auto_label_current_image
        self._last_call = time.time()
       
        self.plot_counter = 0
        self.bounding_boxes = []
        self.image_cache = {}
        self.blob_cache = {}
        self.auto_label_yolo_button.triggered.connect(self.auto_label_yolo_button_clicked)        
        self.stop_labeling = False
        self.process_timer = QTimer()
        self.process_timer.setSingleShot(True)
        self.process_timer.timeout.connect(self.process_current_image)
        self.process_timer_interval = 500  # Adjust the delay as needed
        self.cuda_available = False
        self.current_cropped_directory = None
        #syles,gif, sound and mute
        self.movie = QMovie('styles/gifs/darkfusion.gif')
        self.app = app
        self.current_file = None  
        self.populate_style_combo_box()
        self.styleComboBox.currentIndexChanged.connect(self.on_style_change)

        self.image_label_2.setMovie(self.movie)
        self.movie.start()
        self.populateGifCombo()
        self.gif_change.currentIndexChanged.connect(self.onGifChange)
        self.sound_player = QMediaPlayer()
        self.muteCheckBox.stateChanged.connect(self.mute_player)
        self.sound_player.setMuted(True)  # Set mute state to True by default
        self.muteCheckBox.setChecked(True)
        self.update_counts = {}
        # see download and extrct video
        self.video_upload.clicked.connect(self.on_add_video_clicked)
        self.setAcceptDrops(True)
        self.remove_video_button.clicked.connect(self.on_remove_video_clicked)
        self.custom_frames_checkbox.stateChanged.connect(self.on_custom_frames_toggled)
        self.custom_size_checkbox.stateChanged.connect(self.on_custom_size_checkbox_state_changed)
        self.image_format.currentTextChanged.connect(self.on_image_format_changed)
        self.image_format = ".jpg"  # Default format
        self.inference_checkbox.stateChanged.connect(self.toggle_inference)
        self.play_video_button.clicked.connect(self.toggle_play_pause)
        
        # ⏩ Forward
        # ⏱ Create timers first
        self.forward_timer = QTimer(self)
        self.backward_timer = QTimer(self)

        # ⏱ Set intervals
        self.forward_timer.setInterval(100)
        self.backward_timer.setInterval(100)

        # ⏩ Forward
        self.forward_button.clicked.connect(lambda: self.skip_forward(frames=1))       # single press
        self.forward_timer.timeout.connect(lambda: self.skip_forward(frames=1))        # hold press
        self.forward_button.pressed.connect(self.forward_timer.start)
        self.forward_button.released.connect(self.forward_timer.stop)

        # ⏪ Backward
        self.back_button.clicked.connect(lambda: self.skip_backward(frames=1))         # single press
        self.backward_timer.timeout.connect(lambda: self.skip_backward(frames=1))      # hold press
        self.back_button.pressed.connect(self.backward_timer.start)
        self.back_button.released.connect(self.backward_timer.stop)
        self.video_slider.setEnabled(False)  # Disabled until video is loaded
        self.video_slider.sliderReleased.connect(self.slider_seek)
        self.video_slider.sliderPressed.connect(self.slider_pressed)
        self.slider_is_pressed = False
        self.extract_button.clicked.connect(self.on_extract_button_clicked)
        self.stop_extract.clicked.connect(self.video_processor.stop)
        self.stop_extract.clicked.connect(self.stop_program)
        self.is_playing = False 
        
        self.default_classes_path = os.getcwd()
        self.grey_scale_slider.valueChanged.connect(self.apply_grayscale)
        # 3) Now your slot can use the same mapping
        self.dialog_open = False
        self.output_path = ""
        self.add_video_running = False
        self.custom_frames_checkbox.stateChanged.connect(self.update_checkboxes)
        self.custom_size_checkbox.stateChanged.connect(self.update_checkboxes)
        self.height_box.valueChanged.connect(self.on_size_spinbox_value_changed)
        self.width_box.valueChanged.connect(self.on_size_spinbox_value_changed)
        self.custom_input.textChanged.connect(self.on_custom_input_changed)
        self.video_processor.progress_updated.connect(self.update_progress)
        self.width_box.valueChanged.connect(self.force_live_redraw)
        self.height_box.valueChanged.connect(self.force_live_redraw)


        self.paste_url = self.findChild(QtWidgets.QLineEdit, 'paste_url')
        self.video_download = self.findChild(QtWidgets.QTableWidget, 'video_download')
        self.download_video = self.findChild(QtWidgets.QPushButton, 'download_video')

        self.download_video.clicked.connect(self.download_videos)
        self.remove_video = self.findChild(QtWidgets.QPushButton, 'remove_video')
        self.remove_video.clicked.connect(self.remove_selected_video)
        self.download_threads = []
        self.progress_bars = [
            self.findChild(QtWidgets.QProgressBar, f'video_{i+1}') for i in range(5)
        ]
        
        # Set table row count for videos
        self.video_download.setRowCount(5)
        for i in range(5):
            self.video_download.setItem(i, 1, QtWidgets.QTableWidgetItem(''))
            self.video_download.setItem(i, 2, QtWidgets.QTableWidgetItem(''))
            self.video_download.setItem(i, 3, QtWidgets.QTableWidgetItem(''))
            self.video_download.setItem(i, 4, QtWidgets.QTableWidgetItem(''))
            self.video_download.setItem(i, 5, QtWidgets.QTableWidgetItem(''))
        self.current_image = None
        self.capture = None
        self.is_camera_mode = False
        self.current_file_name = None
        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.display_camera_input)
        self.skip_frames_count = 1 
        self.location_button.clicked.connect(self.set_output_directory)
        self.input_selection.currentIndexChanged.connect(self.on_input_source_changed)
        self.setup_input_sources()       
        self.save_path = ""
        self.model = None
        self.weights_file_path = None    
        self.segment = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
       
        self.extracting_frames = False
        self.skip_frames_count = 0
        self.custom_size = None
        # view cfg and auto cal anchors
        self.label_indices = {}
        self.processing = True
        self.current_img_index = 0
        self.cfg_dict = {}
        self.parsed_yaml = {}
        self.filename = ""
        self.file_type = ""
        self.import_button.clicked.connect(self.import_images)
        self.output_button.clicked.connect(self.output_paths)

        # darknet training
        self.browse_data.clicked.connect(self.browse_data_clicked)
        self.browse_weights.clicked.connect(self.browse_weights_clicked)
        self.train_button.clicked.connect(self.darknet_train)
        self.weights_button.triggered.connect(self.open_weights)
        self.import_data_button.clicked.connect(self.import_data)
        self.calculate_anchors_button.clicked.connect(self.calculate_anchors)
        self.browse_cfg.clicked.connect(self.browse_cfg_clicked)
        self.cfg_button.triggered.connect(self.open_cfg)


        #ultralyics training
        self.model_input.clicked.connect(self.browse_pt_clicked)
        self.data_input.clicked.connect(self.browse_yaml_clicked)
        self.runs_directory = ''  # Initial empty value for save directory
        self.save_runs_directory.clicked.connect(self.on_save_dir_clicked)
        self.ultralytics.clicked.connect(self.ultralytics_train_clicked)
        self.model_config_path = None
        self.pretrained_model_path = None
        self.pt_path = None
        self.data_yaml_path = None
        self.selected_pytorch_file = None

        #ANCHORS CFG AND YAML       
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
      
        self.combine_txt_button.triggered.connect(self.on_combine_txt_clicked)
        self.combine_txt_flag = False

        self.process_image_button.clicked.connect(self.on_button_click)
        self.convert_image.clicked.connect(self.convert_images)
        self.images_import = []
        self.scan_annotations = ScanAnnotations(self)

        self.import_images_button.clicked.connect(self.import_images_triggered)
        self.scan_button.clicked.connect(self.scan_annotations.scan_annotations)
        self.import_classes_button.clicked.connect(self.scan_annotations.import_classes)
        self.crop_button.clicked.connect(self.process_images_triggered)
      
        self.map_counter = 0
        self.subfolder_counter = 1
        self.valid_classes = []
        self.base_directory = None

        self.image_files = []
        self.image_paths = []

        #augmentation
        self.images = []
        self.counter = 1
        self.sight_picture.stateChanged.connect(self.sight_picture_checkbox_changed)
        self.sight_picture = False
        self.smoke_checkbox.stateChanged.connect(self.smoke_checkbox_changed)
        self.smoke_effect = False
        self.flash_checkbox.stateChanged.connect(self.flash_checkbox_changed)
        self.flash_effect = False
        self.motion_blur.stateChanged.connect(self.motion_blur_checkbox_state_changed)
        self.motion_blur_effect = False
        self.time = 0
        self.glass_effect = False
        self.glass_checkbox.stateChanged.connect(self.glass_checkbox_state_changed)
        self.progress_signal.connect(self.label_progress.setValue)
        self.segmentation_mode.setCurrentText("Boxes") 
        



        self.mosaic_checkbox.stateChanged.connect(self.mosaic_checkbox_changed)
        self.size_input.valueChanged.connect(self.on_size_input_changed)
        self.size_number.valueChanged.connect(self.on_size_number_changed)
        self.augmentation_size = 416  # Default size
        self.augmentation_count = 100  # Default count
        self.mosaic_effect = False
        self.is_fp16 = False
        self.start_gathering_metrics()
        self.flip_images = False
        self.flip_checkbox.stateChanged.connect(self.set_flip_images)

        self.model = None
        #settings dialog
        self.class_names = []
        self.class_colors = []
        self.keyBuffer = ""
        self.keyTime = time.time()
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.processKeyPresses)
        self.new_class_input = False
        self.outline_Checkbox.clicked.connect(self.checkbox_clicked)
        self.super_resolution_Checkbox.triggered.connect(self.checkbox_clicked)
        self.edge_slider_min.valueChanged.connect(self.edge_slider_changed)
        self.edge_slider_max.valueChanged.connect(self.edge_slider_changed)
        self.grayscale_Checkbox.stateChanged.connect(self.checkbox_clicked)
        self.slider_min_value = 50   # sensible default for edge detection
        self.slider_max_value = 150  # Canny works well with 50–150
        self.bounding_boxes = {}
        self.segmentation_checkbox.stateChanged.connect(self.on_segmentation_checkbox_checked)
        self.heatmap_Checkbox.stateChanged.connect(self.on_heatmap_toggle)

        self.hide_labels = False
               
        #random checkboxes sliders to sort
        self.save_path = None
        self.populate_combo_box()
        self.display_help.currentIndexChanged.connect(self.load_file)

        self.fp_select_combobox.currentIndexChanged.connect(self.switch_floating_point_mode)
        self.img_video_button.triggered.connect(self.open_image_video)
        self.img_index_number.valueChanged.connect(self.img_index_number_changed)
        self.img_index_search.textChanged.connect(self.img_index_number_changed)        
              
        self.app = app
        self.yolo_files = []
        
        
        #convert ultralyics 
        self.Load_pt_model.clicked.connect(self.load_model)
        self.convertButton.clicked.connect(self.handle_conversion)
        self.convert_model.currentIndexChanged.connect(self.format_changed)# Connect this signal only once.
        self.half_true.stateChanged.connect(lambda: self.update_gui_elements(self.convert_model.currentText()))
        self.threadpool = QThreadPool()
        # Sam model     
        self.device = "cuda"
        self.image_files = []
        self.input_labels = []
        self.input_points = []
        self.model_type = "vit_b"
        self.sam_checkpoint = 'Sam/sam_vit_b_01ec64.pth'  # Path to your model checkpoint
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.overwrite_var.stateChanged.connect(self.on_check_button_click)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        self.temp_bbox = []
        self.overwrite = False
        self.yolo_cache = {}
        self.sam_files = os.listdir('Sam')
        self.sam_model.addItems(self.sam_files)  # Updated this line
        self.sam_model.currentIndexChanged.connect(self.on_dropdown_changed)  # Updated this line
        self.processor = ImageProcessor(self.image_files)
        self.processor.update_signal.connect(self.show_image)
        self.shadow_var.stateChanged.connect(self.on_shadow_checkbox_click)
        self.shadow = False
        self.processor.start()           
        self.noise_remove_checkbox.stateChanged.connect(self.on_noise_remove_checked)
        self.copy_paste_checkbox.stateChanged.connect(self.on_copy_paste_checkbox_click)
        self.is_noise_remove_enabled = False
        self.copy_paste_enabled = False
        self.is_obb_generation_enabled = False
        self.num_classes = 1
        self.dino_label.triggered.connect(self.on_dino_label_clicked)

        self.image_directory = None  # Initialize the attribute to store the image directory
        self.file_observer = None
        self.clear_json.triggered.connect(self.on_clear_json_clicked)
        self.sct = mss.mss() 
        # Save ROI related UI elements
        self.initial_geometry = self.saveGeometry()
        self.initial_state = self.saveState()
        self.roi_checkbox.stateChanged.connect(lambda state: self.toggle_roi(state, 1))
        self.roi_checkbox_2.stateChanged.connect(lambda state: self.toggle_roi(state, 2))        
        self.reset_layout_action = self.findChild(QAction, "reset_layout")
        if self.reset_layout_action:
            self.reset_layout_action.triggered.connect(self.resetLayout)
        # ROI sliders and spin boxes    
        self.actionSahi_label.triggered.connect(self.showSahiSettings)
        self.width_spin_slider.valueChanged.connect(lambda: self.update_roi(1))
        self.width_spin_slider_2.valueChanged.connect(lambda: self.update_roi(2))
        self.height_spin_slider.valueChanged.connect(lambda: self.update_roi(1))
        self.height_spin_slider_2.valueChanged.connect(lambda: self.update_roi(2))
        self.x_spin_slider.valueChanged.connect(lambda: self.update_roi(1))
        self.x_spin_slider_2.valueChanged.connect(lambda: self.update_roi(2))
        self.y_spin_slider.valueChanged.connect(lambda: self.update_roi(1))
        self.y_spin_slider_2.valueChanged.connect(lambda: self.update_roi(2))
        self.filter_button.clicked.connect(self.filter_bboxes)
        self.update_current_bbox_class()
        self.logged_label_files = set() 
        self.obb_checkbox.stateChanged.connect(self.on_obb_checkbox_checked)



        self.current_item_index = None  # CHECK THIS
        self.thumbnail_index = 0 
        self.time = 0.0          

        self.split_data_button = self.findChild(QPushButton, 'split_data')
        self.split_data_button.clicked.connect(self.launch_split_data_ui)
        self.image_quality_checkbox.toggled.connect(self.toggle_image_quality_analysis)
        self.image_quality_analysis_enabled = False 

        self.all_frame_bounding_boxes = {}  # Dictionary to store bounding boxes for all frames
        self.class_visibility = {}  # Store visibility state for each class
        self.all_frame_bounding_boxes = {}  
        self.all_frame_segmentations = {}
        self.all_frame_obbs = {}        
        self.font_size_slider.valueChanged.connect(self.update_font_size_display)
        self.shade_slider.valueChanged.connect(self.update_shading_opacity)
        self.heatmap_dropdown.currentIndexChanged.connect(self.on_heatmap_colormap_changed)
        self.list_input.returnPressed.connect(self.add_keypoint_list_item)
        self.remove_input.clicked.connect(self.remove_keypoint_list_item)
        self.keypoint_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.current_file = None 
        self.label_file = None
        self.current_monitor_index = 1  # Default to monitor 1

        self.detect_and_set_monitor_range()
        self.monitor.valueChanged.connect(self.set_monitor_index)
        # Initialize table
        self.keypoint_list.setColumnCount(5)
        self.keypoint_list.cellClicked.connect(self.store_selected_row)
        self.selected_keypoint_row = None  # To store index      
        self.keypoint_list.setHorizontalHeaderLabels(["#", "Label", "Color", "Visible", "Ignore"])
        self.keypoint_list.setColumnWidth(0, 40)
        self.keypoint_list.setColumnWidth(1, 120)
        self.keypoint_list.setColumnWidth(2, 60)
        self.keypoint_colors = {}  # Track color per index
        self.save_list_button.clicked.connect(self.save_keypoint_list)
        self.load_list_button.clicked.connect(lambda: self.load_keypoint_list_from_path())

        self.inference_checkbox.stateChanged.connect(self.toggle_inference)
        self.toggle_inference(self.inference_checkbox.checkState())
        self.voice_mode_checkbox.stateChanged.connect(self.toggle_voice_mode)
        self.voice_btn.clicked.connect(self.listen_for_label)


        self.show()
        
    def toggle_voice_mode(self, state):
        if state == Qt.Checked:
            self.voice_thread_active = True
            self.voice_btn.setEnabled(False)
            threading.Thread(target=self.continuous_voice_loop, daemon=True).start()
            self.statusBar().showMessage("🎧 Voice Labeling Activated")
        else:
            self.voice_thread_active = False
            self.voice_btn.setEnabled(True)
            self.statusBar().showMessage("🛑 Voice Labeling Stopped")

    def listen_for_label(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.statusBar().showMessage("🎙️ Listening... say a class name")
            try:
                audio = recognizer.listen(source, phrase_time_limit=2)
                text = recognizer.recognize_google(audio).strip().lower()
                self.statusBar().showMessage(f"🔈 Heard: {text}")

                # Match to class name or hotkey
                for i in range(self.classes_dropdown.count()):
                    class_name = self.classes_dropdown.itemText(i).lower()
                    hotkey = self.settings.get(f'classHotkey_{class_name}', '').lower()
                    if text == class_name or text == hotkey:
                        self.classes_dropdown.setCurrentIndex(i)
                        self.statusBar().showMessage(f"✅ Selected class: {class_name}")
                        return

                self.statusBar().showMessage("❌ No matching class or hotkey")
            except sr.UnknownValueError:
                self.statusBar().showMessage("❌ Could not understand")
            except sr.RequestError:
                self.statusBar().showMessage("❌ Speech Recognition API unavailable")
    def continuous_voice_loop(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)

        while self.voice_thread_active:
            try:
                with mic as source:
                    self.statusBar().showMessage("🎙️ Listening for class...")
                    audio = recognizer.listen(source, phrase_time_limit=2)

                text = recognizer.recognize_google(audio).strip().lower()
                self.statusBar().showMessage(f"🔈 Heard: {text}")

                for i in range(self.classes_dropdown.count()):
                    class_name = self.classes_dropdown.itemText(i).lower()
                    hotkey = self.settings.get(f'classHotkey_{class_name}', '').lower()

                    if text == class_name or text == hotkey:
                        self.classes_dropdown.setCurrentIndex(i)
                        self.statusBar().showMessage(f"✅ Selected class: {class_name}")
                        break
                else:
                    self.statusBar().showMessage("❌ No match for voice input")

            except sr.UnknownValueError:
                self.statusBar().showMessage("🤔 Didn't catch that...")
            except sr.RequestError:
                self.statusBar().showMessage("❌ Speech API error")
            except Exception as e:
                self.statusBar().showMessage(f"⚠️ Error: {str(e)}")
        
    def closeEvent(self, event):
        if hasattr(self, "save_bounding_boxes"):
            logger.info(f"🔒 Saving before close: {self.current_file}")
            self.save_bounding_boxes(
                self.current_file,
                self.image.width(),
                self.image.height()
            )
        event.accept()

    def add_keypoint_list_item(self):
        label_text = self.list_input.text().strip()
        if not label_text:
            return

        for row in range(self.keypoint_list.rowCount()):
            existing_label = self.keypoint_list.item(row, 1).text()
            if label_text.lower() == existing_label.lower():
                logger.warning("⚠️ Duplicate label. Entry not added.")
                return

        row = self.keypoint_list.rowCount()
        self.keypoint_list.insertRow(row)

        index_item = QTableWidgetItem(str(row))
        index_item.setFlags(Qt.ItemIsEnabled)
        self.keypoint_list.setItem(row, 0, index_item)

        name_item = QTableWidgetItem(label_text)
        name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.keypoint_list.setItem(row, 1, name_item)

        total_rows = self.keypoint_list.rowCount()
        color = get_color(row, num_classes=total_rows)
        color_item = QTableWidgetItem()
        color_item.setBackground(QBrush(color))
        color_item.setFlags(Qt.ItemIsEnabled)
        self.keypoint_list.setItem(row, 2, color_item)

        checkbox = QCheckBox()
        checkbox.setChecked(True)
        checkbox.setStyleSheet("margin-left:10px;")
        self.keypoint_list.setCellWidget(row, 3, checkbox)

        ignore_checkbox = QCheckBox()
        ignore_checkbox.setChecked(False)
        ignore_checkbox.setStyleSheet("margin-left:10px;")
        self.keypoint_list.setCellWidget(row, 4, ignore_checkbox)

        checkbox.stateChanged.connect(lambda state, row=row: self.sync_ignore_visibility(row, source="visible"))
        ignore_checkbox.stateChanged.connect(lambda state, row=row: self.sync_ignore_visibility(row, source="ignore"))

        self.list_input.clear()
        self.reindex_keypoint_table()
        self.rebind_keypoint_checkbox_signals()

    def clear_keypoint_highlight(self):
        for row in range(self.keypoint_list.rowCount()):
            for col in range(self.keypoint_list.columnCount()):
                # ❌ Skip color column
                if col == 2:
                    continue

                item = self.keypoint_list.item(row, col)
                if item:
                    item.setBackground(QColor("white"))
                    item.setForeground(QColor("black"))

                widget = self.keypoint_list.cellWidget(row, col)
                if widget:
                    widget.setStyleSheet("")

    def highlight_keypoint_row(self, row):
        for col in range(self.keypoint_list.columnCount()):
            # 🟥 Don't override the color preview column!
            if col == 2:
                continue

            item = self.keypoint_list.item(row, col)
            if item:
                item.setBackground(QColor("#FFDDDD"))
                item.setForeground(QColor("black"))

            widget = self.keypoint_list.cellWidget(row, col)
            if widget:
                widget.setStyleSheet("border: 2px solid red;")


    def load_keypoint_list_from_path(self, file_path=None):
        if file_path is None:
            # 🛑 No file given? ASK the user to select manually
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Keypoint List",
                os.getcwd(),
                "JSON Files (*.json);;All Files (*)"
            )
            if not file_path:
                logger.info("🚫 User canceled keypoint list load.")
                return  # User cancelled manually

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"An error occurred while loading:\n{str(e)}")
            return

        self.keypoint_list.setRowCount(0)

        for idx, kp in enumerate(data.get("keypoints", [])):
            self.keypoint_list.insertRow(idx)

            index_item = QTableWidgetItem(str(idx))
            index_item.setFlags(Qt.ItemIsEnabled)
            self.keypoint_list.setItem(idx, 0, index_item)

            name_item = QTableWidgetItem(kp["label"])
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.keypoint_list.setItem(idx, 1, name_item)

            r, g, b, a = kp["color"]
            color = QColor(r, g, b, a)
            color_item = QTableWidgetItem()
            color_item.setBackground(QBrush(color))
            color_item.setFlags(Qt.ItemIsEnabled)
            self.keypoint_list.setItem(idx, 2, color_item)

            checkbox = QCheckBox()
            checkbox.setChecked(kp.get("visible", False))
            checkbox.setStyleSheet("margin-left:10px;")
            self.keypoint_list.setCellWidget(idx, 3, checkbox)

            ignore_checkbox = QCheckBox()
            ignore_checkbox.setChecked(kp.get("ignore", False))
            ignore_checkbox.setStyleSheet("margin-left:10px;")
            self.keypoint_list.setCellWidget(idx, 4, ignore_checkbox)

            # ✅ Correct partial connection
            checkbox.stateChanged.connect(functools.partial(self.sync_ignore_visibility, idx, source="visible"))
            ignore_checkbox.stateChanged.connect(functools.partial(self.sync_ignore_visibility, idx, source="ignore"))

        self.reindex_keypoint_table()
        self.rebind_keypoint_checkbox_signals()
        logger.info(f"✅ Successfully loaded keypoint list from {file_path}")


    def sync_ignore_visibility(self, row, source):
        visible_cb = self.keypoint_list.cellWidget(row, 3)
        ignore_cb = self.keypoint_list.cellWidget(row, 4)

        if source == "visible" and visible_cb and visible_cb.isChecked():
            if ignore_cb:
                ignore_cb.setChecked(False)
        elif source == "ignore" and ignore_cb and ignore_cb.isChecked():
            if visible_cb:
                visible_cb.setChecked(False)

    def store_selected_row(self, row, column):
        if row >= self.keypoint_list.rowCount():
            return
        self.selected_keypoint_row = row
        if self.selected_bbox and hasattr(self.selected_bbox, 'keypoint_drawer'):
            drawer = self.main_window.selected_bbox.keypoint_drawer
            if row < len(drawer.points):
                x, y = drawer.points[row]
                img_w = self.image.width()
                img_h = self.image.height()
                point_pos = QPointF(x * img_w, y * img_h)
                self.screen_view.centerOn(point_pos)
    def lock_keypoint_checkboxes(self, locked=True):
        for i in range(self.keypoint_list.rowCount()):
            for col in [3, 4]:  # Visibility and Ignore
                cb = self.keypoint_list.cellWidget(i, col)
                if cb:
                    cb.setEnabled(not locked)
                    
    def rebind_keypoint_checkbox_signals(self):
        for row in range(self.keypoint_list.rowCount()):
            visible_cb = self.keypoint_list.cellWidget(row, 3)
            ignore_cb = self.keypoint_list.cellWidget(row, 4)

            if visible_cb:
                try:
                    visible_cb.stateChanged.disconnect()
                except TypeError:
                    pass
                visible_cb.stateChanged.connect(lambda state, row=row: self.sync_ignore_visibility(row, source="visible"))

            if ignore_cb:
                try:
                    ignore_cb.stateChanged.disconnect()
                except TypeError:
                    pass
                ignore_cb.stateChanged.connect(lambda state, row=row: self.sync_ignore_visibility(row, source="ignore"))
                
    def reset_keypoint_checkboxes(self):
        for i in range(self.keypoint_list.rowcount()):
            visible_cb = self.keypoint_list.cellwidget(i, 3)
            ignore_cb = self.keypoint_list.cellwidget(i, 4)
            if visible_cb: visible_cb.setchecked(true)
            if ignore_cb: ignore_cb.setchecked(false)
           
    def remove_keypoint_list_item(self):
        if self.selected_keypoint_row is None:
            logger.warning("⚠️ No row selected to remove.")
            return

        self.keypoint_list.removeRow(self.selected_keypoint_row)
        self.selected_keypoint_row = None
        self.reindex_keypoint_table()
        self.rebind_keypoint_checkbox_signals()
        self.keypoint_list.clearSelection()


    def reindex_keypoint_table(self):
        for row in range(self.keypoint_list.rowCount()):
            index_item = QTableWidgetItem(str(row))
            index_item.setFlags(Qt.ItemIsEnabled)
            self.keypoint_list.setItem(row, 0, index_item)


    def save_keypoint_list(self):
        settings = QSettings()
        last_dir = settings.value("last_save_dir", os.getcwd())

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Keypoint List",
            os.path.join(last_dir, "points.json"),
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return  # User canceled

        settings.setValue("last_save_dir", os.path.dirname(file_path))

        keypoints = []
        label_to_index = {}
        for row in range(self.keypoint_list.rowCount()):
            label_item = self.keypoint_list.item(row, 1)
            color_item = self.keypoint_list.item(row, 2)
            checkbox = self.keypoint_list.cellWidget(row, 3)
            ignore_checkbox = self.keypoint_list.cellWidget(row, 4)

            if not label_item or not color_item or not checkbox or not ignore_checkbox:
                continue

            label = label_item.text()
            color = color_item.background().color()
            visible = checkbox.isChecked()
            ignore = ignore_checkbox.isChecked()

            keypoints.append({
                "label": label,
                "color": [color.red(), color.green(), color.blue(), color.alpha()],
                "visible": visible,
                "ignore": ignore
            })

            label_to_index[label.lower()] = row

        # 🔗 Auto-connect each point to the next for a basic skeleton
        skeleton = [[i, i + 1] for i in range(len(keypoints) - 1)]

        # 🔁 Auto-generate flip_idx based on label names (Left ↔ Right)
        flip_idx = list(range(len(keypoints)))  # fallback to identity
        for i, kp in enumerate(keypoints):
            label = kp["label"].lower()
            if "left" in label:
                match = label.replace("left", "right")
            elif "right" in label:
                match = label.replace("right", "left")
            else:
                continue

            j = label_to_index.get(match)
            if j is not None:
                flip_idx[i] = j

        data = {
            "kpt_shape": [len(keypoints), 3],
            "keypoints": keypoints,
            "skeleton": skeleton,
            "flip_idx": flip_idx
        }

        # 💣 Remove file before writing
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                QMessageBox.warning(self, "Overwrite Failed", f"Could not remove existing file:\n{str(e)}")
                return

        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4, separators=(",", ": "))
            QMessageBox.information(self, "Save Successful", f"Keypoints + skeleton + flip_idx saved to {file_path}.")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"An error occurred while saving:\n{str(e)}")


    def update_keypoint_checkboxes_from_labels(self, label_file_path):
        """
        Parses keypoint data from a YOLO label file without updating the UI.

        Returns:
            List of tuples: [(x, y), ...]
            List of visibility flags: [v, ...]
        """
        if not os.path.exists(label_file_path):
            logger.warning(f"⚠️ Label file not found: {label_file_path}")
            return [], []

        try:
            with open(label_file_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"❌ Failed to read label file: {e}")
            return [], []

        if not lines:
            return [], []

        # Parse keypoints from each line (each bbox)
        all_keypoints = []
        all_flags = []

        for line in lines:
            parts = line.split()
            if len(parts) < 6:
                logger.warning("⚠️ Not enough data in label to parse keypoints.")
                continue

            keypoint_values = parts[5:]  # After class_id xc yc w h
            num_kpts = len(keypoint_values) // 3

            points = []
            flags = []

            for i in range(num_kpts):
                try:
                    x = float(keypoint_values[i * 3])
                    y = float(keypoint_values[i * 3 + 1])
                    v = int(keypoint_values[i * 3 + 2])
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse keypoint {i}: {e}")
                    continue

                points.append((x, y))
                flags.append(v)

            all_keypoints.append(points)
            all_flags.append(flags)

        logger.info(f"✔️ Parsed {len(all_keypoints)} bounding box keypoint sets from {os.path.basename(label_file_path)}")
        return all_keypoints, all_flags


            
    def update_keypoint_checkboxes_from_points(self, points, visibility_flags=None):
        if not points or not visibility_flags:
            logger.info("⚠️ Skipping UI update: no points or flags to apply.")
            return

        for i in range(self.keypoint_list.rowCount()):
            visible_cb = self.keypoint_list.cellWidget(i, 3)
            ignore_cb = self.keypoint_list.cellWidget(i, 4)

            v = visibility_flags[i] if i < len(visibility_flags) else 0

            if ignore_cb:
                ignore_cb.setChecked(v == 0)
            if visible_cb:
                visible_cb.setChecked(v == 2)



    def update_shading_opacity(self, value):
        logger = logging.getLogger("UltraDarkFusionLogger")
        percent = int((value / 100) * 100)
        logger.info(f"🎨 Shading updated → Slider: {value}/100 → Opacity: {percent}%")


    def update_font_size_display(self, value):
        logger.info(f"📏 Font size slider changed to: {value} pt")
    def is_boxes_mode(self):
        return self.segmentation_mode.currentText() == "Boxes"
    def is_segmentation_mode(self):
        return self.segmentation_mode.currentText() == "Segmentation" 
    def is_keypoint_mode(self):
        return self.segmentation_mode.currentText() == "Keypoints"
    def is_obb_mode(self):
        return self.segmentation_mode.currentText() == "OBB"
    def edit_mode_active(self):
        if hasattr(self.screen_view, 'scene'):
            for item in self.screen_view.scene().items():
                if isinstance(item, SegmentationDrawer) and item.edit_mode:
                    return True
        return False
        
    def launch_split_data_ui(self):
        # Replace 'python' with the correct path if needed
        subprocess.Popen(["python", "splitdatav5.py"])

              
    def select_directory(self):
        self.directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.directory:
            self.statusBar().showMessage(f"Selected directory: {self.directory}")

    def crop_images(self):
        width = self.w_img.value()  # Assuming w_img is a QSpinBox
        height = self.h_img.value()  # Assuming h_img is a QSpinBox
        if not self.directory:
            QMessageBox.warning(self, "Warning", "Please select a directory first.")
            return

        self.worker = CropWorker(self.directory, width, height)
        self.worker.update_progress.connect(self.label_progress.setValue)  # Corrected to use the right progress bar name
        self.worker.start()




    def toggle_roi(self, state, roi_id):
        """Toggle visibility for the specified ROI based on checkbox state."""
        if roi_id == 1:
            if state == Qt.Checked:
                if not hasattr(self, 'roi_item_1') or self.roi_item_1 is None:
                    self.update_roi(1)  # Create ROI 1 if it doesn't exist
                else:
                    self.roi_item_1.show()
            else:
                self.remove_roi(1)
        elif roi_id == 2:
            if state == Qt.Checked:
                if not hasattr(self, 'roi_item_2') or self.roi_item_2 is None:
                    self.update_roi(2)  # Create ROI 2 if it doesn't exist
                else:
                    self.roi_item_2.show()
            else:
                self.remove_roi(2)



    def validate_roi(self):
        """Ensure at least one valid ROI exists before filtering."""
        def is_valid_roi(roi):
            """Helper function to check if ROI exists and is visible."""
            try:
                return roi is not None and roi.isVisible()
            except RuntimeError:  # Object has been deleted
                return False

        valid_roi_1 = is_valid_roi(getattr(self, 'roi_item_1', None))
        valid_roi_2 = is_valid_roi(getattr(self, 'roi_item_2', None))

        if not (valid_roi_1 or valid_roi_2):
            QMessageBox.warning(self, "No ROI Defined", "Please define and activate at least one ROI before filtering.")
            return False

        if not hasattr(self, 'image_directory') or not self.image_directory:
            QMessageBox.warning(self, "No Directory Selected", "Please select an image directory first.")
            return False

        return True



    def filter_bboxes(self):
        """Filter bounding boxes based on active ROIs (supports regular, OBB, and segmentation formats)."""
        if not self.validate_roi():
            return

        # Get visible ROI rectangles
        roi_rects = []
        for roi_attr in ['roi_item_1', 'roi_item_2']:
            try:
                roi_item = getattr(self, roi_attr, None)
                if roi_item and roi_item.isVisible():
                    roi_rects.append(roi_item.rect())
            except RuntimeError:
                setattr(self, roi_attr, None)

        if not roi_rects:
            QMessageBox.warning(self, "No ROI Defined", "Please activate at least one ROI before filtering.")
            return

        total_images = len(self.image_files)
        self.label_progress.setMinimum(0)
        self.label_progress.setMaximum(total_images)
        self.label_progress.setValue(0)

        for idx, image_file in enumerate(self.image_files):
            if image_file.endswith("default.png"):
                continue

            image = QPixmap(image_file)
            if image.isNull():
                logger.error(f"Failed to load image: {image_file}")
                continue

            image_width, image_height = image.width(), image.height()

            # Scale ROI rectangles to match current image dimensions
            scaled_rois = self.get_dynamic_rois(
                roi_rects[0] if len(roi_rects) > 0 else None,
                roi_rects[1] if len(roi_rects) > 1 else roi_rects[0],
                image_width, image_height
            )

            roi_polygons = []
            for roi in scaled_rois:
                if roi:
                    x, y, w, h = roi
                    roi_polygons.append(shapely_box(x, y, x + w, y + h))

            txt_file_path, exists = self.get_label_file(image_file, return_existence=True)
            if not exists:
                logger.warning(f"No annotation file found for {image_file}, skipping.")
                continue

            try:
                bounding_boxes = self.load_bounding_boxes(txt_file_path, image_width, image_height)
                if not bounding_boxes:
                    continue

                filtered_bboxes = []
                for bbox in bounding_boxes:
                    try:
                        poly = bbox.to_polygon(image_width, image_height)
                        if poly.is_valid and any(poly.intersects(roi) for roi in roi_polygons):
                            filtered_bboxes.append(bbox)
                    except Exception as poly_err:
                        logger.warning(f"Skipping invalid bbox: {bbox.to_str()} - {poly_err}")

                if filtered_bboxes:
                    labels = [bbox.to_str() for bbox in filtered_bboxes]
                    self.save_labels_to_file(txt_file_path, labels, mode="w")
                else:
                    open(txt_file_path, "w").close()

            except Exception as e:
                logger.error(f"Error processing {txt_file_path}: {e}")

            self.label_progress.setValue(idx + 1)
            QtWidgets.QApplication.processEvents()

        QMessageBox.information(self, "Filter Complete", "Bounding boxes outside the ROIs have been processed.")
        self.label_progress.setValue(total_images)




    def get_dynamic_rois(self, roi1_rect, roi2_rect, image_width, image_height):
        """
        Scale two ROI rectangles dynamically to fit the current image dimensions.
        
        Returns:
            Two tuples: (x_min, y_min, width, height) for each ROI.
        """
        x_scale = image_width / self.image.width()
        y_scale = image_height / self.image.height()

        scaled_x1 = int(roi1_rect.x() * x_scale)
        scaled_y1 = int(roi1_rect.y() * y_scale)
        scaled_width1 = int(roi1_rect.width() * x_scale)
        scaled_height1 = int(roi1_rect.height() * y_scale)

        scaled_x2 = int(roi2_rect.x() * x_scale)
        scaled_y2 = int(roi2_rect.y() * y_scale)
        scaled_width2 = int(roi2_rect.width() * x_scale)
        scaled_height2 = int(roi2_rect.height() * y_scale)

        return (scaled_x1, scaled_y1, scaled_width1, scaled_height1), (scaled_x2, scaled_y2, scaled_width2, scaled_height2)



    def filter_bboxes_in_file(self, txt_file_path, roi_1, roi_2, image_width, image_height):
        try:
            bounding_boxes = self.load_bounding_boxes(txt_file_path, image_width, image_height)
            if not bounding_boxes:
                return

            # Build ROI polygons
            roi_polygons = []
            if roi_1:
                x1, y1, w1, h1 = roi_1
                roi_polygons.append(shapely_box(x1, y1, x1 + w1, y1 + h1))
            if roi_2:
                x2, y2, w2, h2 = roi_2
                roi_polygons.append(shapely_box(x2, y2, x2 + w2, y2 + h2))

            filtered_bboxes = []

            for bbox in bounding_boxes:
                poly = bbox.to_polygon(image_width, image_height)

                if any(poly.intersects(roi) for roi in roi_polygons):
                    filtered_bboxes.append(bbox)

            if filtered_bboxes:
                labels = [bbox.to_str() for bbox in filtered_bboxes]
                self.save_labels_to_file(txt_file_path, labels, mode="w")
            else:
                open(txt_file_path, "w").close()

        except Exception as e:
            logger.error(f"Error processing {txt_file_path}: {e}")



    def draw_roi(self, width, height, x, y, roi_id):
        """Draw or update the ROI rectangle dynamically on the image."""
        pixmap_width = self.image.width()
        pixmap_height = self.image.height()

        # Constrain ROI to image bounds
        x = max(0, min(x, pixmap_width - width))
        y = max(0, min(y, pixmap_height - height))
        width = min(width, pixmap_width - x)
        height = min(height, pixmap_height - y)

        # Determine which ROI to update
        if roi_id == 1:
            if not hasattr(self, 'roi_item_1') or self.roi_item_1 not in self.screen_view.scene().items():
                self.roi_item_1 = QGraphicsRectItem(x, y, width, height)
                pen = QPen(Qt.yellow)
                pen.setWidth(3)
                self.roi_item_1.setPen(pen)
                self.screen_view.scene().addItem(self.roi_item_1)
            else:
                self.roi_item_1.setRect(x, y, width, height)
                self.roi_item_1.show()
        
        elif roi_id == 2:
            if not hasattr(self, 'roi_item_2') or self.roi_item_2 not in self.screen_view.scene().items():
                self.roi_item_2 = QGraphicsRectItem(x, y, width, height)
                pen = QPen(Qt.green)  # Different color for second ROI
                pen.setWidth(3)
                self.roi_item_2.setPen(pen)
                self.screen_view.scene().addItem(self.roi_item_2)
            else:
                self.roi_item_2.setRect(x, y, width, height)
                self.roi_item_2.show()



    def remove_roi(self, roi_id):
        """Safely hide and delete the specified ROI item."""
        try:
            if roi_id == 1 and hasattr(self, 'roi_item_1') and self.roi_item_1 is not None:
                self.roi_item_1.hide()
                self.roi_item_1 = None  # Prevents invalid object references
            elif roi_id == 2 and hasattr(self, 'roi_item_2') and self.roi_item_2 is not None:
                self.roi_item_2.hide()
                self.roi_item_2 = None  # Prevents invalid object references
        except RuntimeError:
            if roi_id == 1:
                self.roi_item_1 = None
            if roi_id == 2:
                self.roi_item_2 = None


    def update_roi(self, roi_id):
        """Update the ROI rectangle based on the current slider values."""
        if not hasattr(self, 'image') or self.image is None:
            if self.roi_checkbox.isChecked() or self.roi_checkbox_2.isChecked():
                logger.warning("Warning: No valid image loaded. ROI cannot be updated.")
            return

        # Get width and height from sliders
        if roi_id == 1:
            width_offset = self.width_spin_slider.value() // 2
            height_offset = self.height_spin_slider.value() // 2
            x_offset = self.x_spin_slider.value()
            y_offset = self.y_spin_slider.value()
        elif roi_id == 2:
            width_offset = self.width_spin_slider_2.value() // 2
            height_offset = self.height_spin_slider_2.value() // 2
            x_offset = self.x_spin_slider_2.value()
            y_offset = self.y_spin_slider_2.value()
        else:
            return

        # Calculate center of the image
        center_x = self.image.width() // 2
        center_y = self.image.height() // 2

        x = center_x - width_offset + x_offset
        y = center_y - height_offset + y_offset

        # Draw or update the selected ROI
        self.draw_roi(width_offset * 2, height_offset * 2, x, y, roi_id)


    def on_frame_change(self):
        # Update the displayed image
        if hasattr(self, 'current_image_path') and os.path.exists(self.current_image_path):
            self.image = QPixmap(self.current_image_path)

            # Update the scene with the new image
            scene = QGraphicsScene(0, 0, self.image.width(), self.image.height())
            pixmap_item = QGraphicsPixmapItem(self.image)
            pixmap_item.setTransformationMode(Qt.SmoothTransformation)
            scene.addItem(pixmap_item)
            self.screen_view.setScene(scene)

            # If ROI is enabled, update and redraw the ROI
        if self.roi_checkbox.isChecked():
            self.update_roi(1)  # Update first ROI

        if self.roi_checkbox_2.isChecked():
            self.update_roi(2)  # Update second ROI

    #sahi settings see
    def showSahiSettings(self):
        self.sahiSettingsDialog = SahiSettingsDialog(self)
        self.sahiSettingsDialog.exec_()
    def resetLayout(self):
        # Restore the initial layout state
        self.restoreGeometry(self.initial_geometry)
        self.restoreState(self.initial_state)

    def on_size_input_changed(self, value):
        self.augmentation_size = value

    def on_size_number_changed(self, value):
        self.augmentation_count = value

    def on_clear_json_clicked(self):
        """
        This method is triggered when the 'Clear JSON' button is clicked.
        It deletes all .json files in the current image directory.
        """
        if self.image_directory:  # Check if the image directory is set
            json_files = glob.glob(os.path.join(self.image_directory, '*.json'))
            if json_files:  # Check if there are any .json files to delete
                for json_file in json_files:
                    try:
                        os.remove(json_file)
                        logger.info(f"Deleted {json_file}")
                    except Exception as e:
                        logger.error(f"Error deleting {json_file}: {str(e)}")
                QMessageBox.information(None, 'JSON Files Cleared', f"All JSON files have been deleted from {self.image_directory}.")
            else:
                QMessageBox.information(None, 'No JSON Files', "No JSON files found in the current directory.")
        else:
            QMessageBox.warning(None, 'Directory Not Set', "The image directory has not been set. Please open an image directory first.")
            
    def on_dropdown_changed(self, index):
        selected_file = self.sam_model.itemText(index)
        self.sam_checkpoint = f'Sam/{selected_file}'

        model_mapping = {
            'vit_h': 'vit_h',
            'vit_b': 'vit_b',
            'vit_l': 'vit_l'
        }

        self.model_type = next((model for key, model in model_mapping.items() if key in selected_file), None)

        if self.model_type:
            logger.info(f'Model type changed to: {self.model_type}')
        else:
            logger.warning('No matching model type found.')

        logger.info(f'SAM checkpoint changed to: {self.sam_checkpoint}')

    def on_check_button_click(self):
        self.overwrite = self.overwrite_var.isChecked()

    def process_directory(self):
        self.image_files = glob.glob(os.path.join(self.batch_directory, '*.[jp][pn]g'))

    def on_shadow_checkbox_click(self):
        self.shadow = self.shadow_var.isChecked()

    def on_obb_checkbox_checked(self, state):
        self.is_obb_generation_enabled = state == QtCore.Qt.Checked
        logging.info(f"OBB generation {'enabled' if self.is_obb_generation_enabled else 'disabled'}.")


    def on_noise_remove_checked(self, state):
        self.is_noise_remove_enabled = state == QtCore.Qt.Checked
        
    def generate_unique_name(self):
        """Generate a unique name using the current datetime and a UUID."""
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]  # Shortened UUID
        return f"{current_time}_{unique_id}"
              
    def on_copy_paste_checkbox_click(self, state: int):
        """
        Slot to handle the state change of the copy_paste_checkbox.
        """
        self.copy_paste_enabled = state == QtCore.Qt.Checked
        logging.info(f"Copy-paste augmentation {'enabled' if self.copy_paste_enabled else 'disabled'}.")

    def on_segmentation_checkbox_checked(self, state):
        """
        Triggered when the segmentation checkbox is toggled.
        Enables segmentation processing and updates the UI.
        """
        self.is_segmentation_enabled = state == QtCore.Qt.Checked
        logging.info(f"Segmentation mode {'enabled' if self.is_segmentation_enabled else 'disabled'}.")

    def copy_and_rename_files(self, image_path, txt_path, target_directory):
        """Copy and rename image and corresponding txt file with unique names."""
        os.makedirs(target_directory, exist_ok=True)

        unique_name = self.generate_unique_name()

        # Construct new image path with unique name
        new_image_path = os.path.join(target_directory, f"{unique_name}{os.path.splitext(image_path)[1]}")
        try:
            shutil.copy2(image_path, new_image_path)
            logging.info(f"Copied image to {new_image_path}")
        except Exception as e:
            logging.error(f"Failed to copy image {image_path} to {new_image_path}: {e}")

        # Construct new annotation path with unique name
        new_txt_path = os.path.join(target_directory, f"{unique_name}.txt")
        _, label_exists = self.get_label_file(image_path, return_existence=True)
        if label_exists:
            try:
                with open(txt_path, "r") as f:
                    labels = [line.strip() for line in f.readlines()]
                self.save_labels_to_file(new_txt_path, labels, mode="w")
                logging.info(f"Copied annotation to {new_txt_path}")
            except Exception as e:
                logging.error(f"Failed to copy annotation {txt_path} to {new_txt_path}: {e}")
        else:
            logging.error(f"Annotation file not found: {txt_path}")

        return new_image_path, new_txt_path


    def polygon_to_mask(self, segmentation, img_width, img_height):
        """
        Converts a normalized YOLO-style polygon to a binary mask.

        Args:
            segmentation (list): List of floats representing x1, y1, x2, y2, ..., xn, yn (normalized 0–1).
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            mask (np.ndarray): Binary mask of shape (img_height, img_width).
        """
        if len(segmentation) < 6 or len(segmentation) % 2 != 0:
            logging.warning("Invalid segmentation length. Must be even and >= 6.")
            return np.zeros((img_height, img_width), dtype=np.uint8)

        points = np.array(segmentation, dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= img_width
        points[:, 1] *= img_height
        int_points = np.round(points).astype(np.int32)

        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(mask, [int_points], 1)  # Fill polygon with 1s
        return mask.astype(bool)

    def get_polygon_from_mask(self, mask, simplify=True, epsilon_factor=0.001):
        """
        Converts a binary mask to a polygon. Useful for exporting segmentation or OBB points.

        Args:
            mask (np.ndarray): Binary mask.
            simplify (bool): Whether to simplify the polygon.
            epsilon_factor (float): Simplification aggressiveness.

        Returns:
            List of (x, y) tuples representing the polygon, or [] if no contour found.
        """
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []

        largest = max(contours, key=cv2.contourArea)

        if simplify:
            arc_len = cv2.arcLength(largest, True)
            epsilon = epsilon_factor * arc_len
            largest = cv2.approxPolyDP(largest, epsilon, True)

        return largest.reshape(-1, 2).tolist()

    def preprocess_image_for_segmentation(self,image):
        """
        Preprocesses the image to reduce background noise before segmentation.
        Applies grayscale conversion, adaptive thresholding, and morphological filtering.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian Blur

        # Adaptive Thresholding to segment foreground and background
        thresholded = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Apply morphological closing to remove noise
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

        return closed

    def predict_and_draw_yolo_objects(self, image, image_file_path):
        if not self.predictor or not self.sam or not self.segment or not os.path.isfile(image_file_path):
            logging.error('Predictor, SAM, or MediaPipe is not available.')
            QMessageBox.critical(self, "Error", "Predictor, SAM, or MediaPipe is not available.")
            return None

        img_height, img_width = image.shape[:2]
        preprocessed_image, _ = self.apply_preprocessing(
            image.copy(), bounding_boxes=None, img_width=img_width, img_height=img_height
        )

        image_with_boxes = preprocessed_image.copy()
        image_copy = preprocessed_image.copy()
        yolo_file_path = os.path.splitext(image_file_path)[0] + ".txt"
        if not os.path.exists(yolo_file_path) or os.path.getsize(yolo_file_path) == 0:
            logging.info(f"Skipping {image_file_path} due to missing or empty annotation file.")
            return None

        adjusted_boxes = []

        try:
            with open(yolo_file_path, "r") as f:
                yolo_lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.error(f"Failed to read YOLO file {yolo_file_path}: {e}")
            return None

        self.predictor.set_image(preprocessed_image)
        masks_for_bboxes = []

        for yolo_line in yolo_lines:
            try:
                box_obj = BoundingBox.from_str(yolo_line)
                if box_obj is None:
                    continue

                if box_obj.segmentation:
                    mask = self.polygon_to_mask(box_obj.segmentation, img_width, img_height)
                    polygon = self.get_polygon_from_mask(mask)
                else:
                    voc_box = pbx.convert_bbox(
                        (box_obj.x_center, box_obj.y_center, box_obj.width, box_obj.height),
                        from_type="yolo", to_type="voc", image_size=(img_width, img_height)
                    )
                    x_min, y_min, x_max, y_max = map(int, map(round, voc_box))
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(img_width - 1, x_max), min(img_height - 1, y_max)

                    if x_max <= x_min or y_max <= y_min:
                        logging.warning(f"Skipped invalid bounding box: {x_min, y_min, x_max, y_max}")
                        continue

                        input_box = np.array([x_min, y_min, x_max, y_max])
                        with torch.amp.autocast('cuda'):
                            masks, _, _ = self.predictor.predict(box=input_box[None, :], multimask_output=False)
                        mask = masks[0].astype(bool)


                    bbox_crop = preprocessed_image[y_min:y_max, x_min:x_max]
                    if bbox_crop.size == 0:
                        continue

                    bbox_rgb = cv2.cvtColor(bbox_crop, cv2.COLOR_BGR2RGB)
                    results = self.segment.process(bbox_rgb)
                    if results.segmentation_mask is not None:
                        media_mask = (results.segmentation_mask > 0.5).astype(bool)
                        media_mask_resized = cv2.resize(media_mask.astype(np.uint8), (bbox_crop.shape[1], bbox_crop.shape[0])).astype(bool)
                        mask[y_min:y_max, x_min:x_max] |= media_mask_resized

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                image_with_boxes, new_bbox = self.draw_mask_and_bbox(
                    image_with_boxes,
                    mask,
                    original_bbox=(x_min, y_min, x_max, y_max) if 'x_min' in locals() else None,
                    obb_points=box_obj.obb if box_obj.obb else None,
                    class_id=box_obj.class_id,
                    num_classes=self.classes_dropdown.count()
                )

                if new_bbox == (0, 0, 0, 0):
                    continue

                masks_for_bboxes.append((mask, new_bbox))

                if self.overwrite:
                    if box_obj.segmentation:
                        polygon = self.get_polygon_from_mask(mask)
                        if polygon:
                            # Normalize segmentation coordinates
                            box_obj.segmentation = [coord for pt in polygon for coord in (pt[0]/img_width, pt[1]/img_height)]

                    elif box_obj.obb:
                        polygon = self.get_polygon_from_mask(mask)
                        if polygon and len(polygon) >= 4:
                            # Normalize the top 4 points for OBB (x1 y1 x2 y2 x3 y3 x4 y4)
                            obb_points = [coord for pt in polygon[:4] for coord in (pt[0]/img_width, pt[1]/img_height)]
                            box_obj.obb = obb_points

                    else:
                        # Regular bounding box overwrite
                        new_yolo = pbx.convert_bbox(new_bbox, "voc", "yolo", (img_width, img_height))
                        box_obj.x_center, box_obj.y_center, box_obj.width, box_obj.height = new_yolo

                    adjusted_boxes.append(box_obj.to_str(remove_confidence=True))


            except Exception as e:
                logging.error(f"Error processing YOLO line: {e}")

        if self.overwrite and adjusted_boxes:
            self.save_labels_to_file(yolo_file_path, adjusted_boxes, mode="w")

        if self.is_noise_remove_enabled:
            self.apply_noise_reduction(image_copy, masks_for_bboxes, image_file_path)

        if self.shadow and masks_for_bboxes:
            self.create_shadow_image(image_copy, masks_for_bboxes, image_file_path)

        return image_with_boxes if self.screen_update.isChecked() else None






    def apply_copy_paste_augmentation(self, progress_offset=0):
        """
        Performs copy-paste augmentation using BoundingBox abstraction.
        Only modified images and annotations are saved.
        """
        if not self.image_directory or not os.path.exists(self.image_directory):
            QMessageBox.critical(self, "Error", "Please select a valid image directory first.")
            return

        copy_paste_dir = os.path.join(self.image_directory, "copy_and_paste")
        os.makedirs(copy_paste_dir, exist_ok=True)

        total_images = len(self.image_files)
        min_size_threshold = 10

        for idx, image_file in enumerate(self.image_files):
            image = cv2.imread(image_file)
            if image is None:
                logging.warning(f"Failed to load image: {image_file}. Skipping.")
                continue

            annotation_file = os.path.splitext(image_file)[0] + ".txt"
            annotations = []
            if os.path.exists(annotation_file):
                with open(annotation_file, "r") as f:
                    annotations = [line.strip() for line in f if line.strip()]

            existing_bboxes = []
            for ann in annotations:
                bbox = BoundingBox.from_str(ann)
                if bbox and not bbox.segmentation and not bbox.obb:
                    voc_bbox = pbx.convert_bbox(
                        (bbox.x_center, bbox.y_center, bbox.width, bbox.height),
                        from_type="yolo", to_type="voc",
                        image_size=image.shape[:2][::-1]
                    )
                    existing_bboxes.append(voc_bbox)

            objects_to_add = max(0, np.random.randint(3, 6) - len(existing_bboxes))
            modified = False

            for _ in range(objects_to_add):
                src_img_path = np.random.choice(self.image_files)
                src_img = cv2.imread(src_img_path)
                if src_img is None:
                    logging.warning(f"Failed to load source image: {src_img_path}")
                    continue

                src_txt_path = os.path.splitext(src_img_path)[0] + ".txt"
                if not os.path.exists(src_txt_path):
                    continue

                with open(src_txt_path, "r") as f:
                    src_anns = [line.strip() for line in f if line.strip()]
                if not src_anns:
                    continue

                bbox = BoundingBox.from_str(np.random.choice(src_anns))
                if not bbox or bbox.segmentation or bbox.obb:
                    continue

                voc = pbx.convert_bbox(
                    (bbox.x_center, bbox.y_center, bbox.width, bbox.height),
                    from_type="yolo", to_type="voc",
                    image_size=src_img.shape[:2][::-1]
                )
                x_min, y_min, x_max, y_max = map(int, voc)

                mask = self.generate_segmentation_mask(src_img, (x_min, y_min, x_max, y_max))
                segment = self.extract_segmented_object(src_img, mask, x_min, y_min, x_max, y_max)
                if segment is None:
                    continue

                tgt_h, tgt_w = image.shape[:2]
                obj_h, obj_w = segment.shape[:2]

                scale = np.random.uniform(0.8, 1.5)
                obj_h, obj_w = int(obj_h * scale), int(obj_w * scale)
                obj_h = max(min_size_threshold, min(obj_h, tgt_h // 2))
                obj_w = max(min_size_threshold, min(obj_w, tgt_w // 2))
                segment = cv2.resize(segment, (obj_w, obj_h))

                for _ in range(50):
                    px, py = np.random.randint(0, tgt_w - obj_w), np.random.randint(0, tgt_h - obj_h)
                    new_bbox = (px, py, px + obj_w, py + obj_h)
                    if not self.check_overlap(new_bbox, existing_bboxes, min_spacing=10):
                        break
                else:
                    continue

                existing_bboxes.append(new_bbox)
                image = self.overlay_object(image, segment, px, py)

                new_xc = (px + obj_w / 2) / tgt_w
                new_yc = (py + obj_h / 2) / tgt_h
                new_w = obj_w / tgt_w
                new_h = obj_h / tgt_h

                new_bb = BoundingBox(class_id=bbox.class_id, x_center=new_xc, y_center=new_yc, width=new_w, height=new_h)
                annotations.append(new_bb.to_str())
                modified = True

            if modified:
                img_name = f"copy_and_paste_{os.path.basename(image_file)}"
                out_img_path = os.path.join(copy_paste_dir, img_name)
                cv2.imwrite(out_img_path, image)

                txt_name = img_name.replace(".png", ".txt").replace(".jpg", ".txt")
                out_txt_path = os.path.join(copy_paste_dir, txt_name)
                with open(out_txt_path, "w") as f:
                    f.write("\n".join(annotations))

                logging.info(f"Saved augmented image and annotations: {out_img_path}, {out_txt_path}")

            self.label_progress.setValue(progress_offset + idx + 1)
            QtWidgets.QApplication.processEvents()



    def check_overlap(self, new_bbox, existing_bboxes, min_spacing=10):
        """
        Checks if a new bounding box overlaps with any existing bounding boxes.

        Args:
            new_bbox: The new bounding box (x_min, y_min, x_max, y_max).
            existing_bboxes: List of existing bounding boxes.
            min_spacing: Minimum spacing required between bounding boxes.

        Returns:
            True if overlap is found, otherwise False.
        """
        x_min1, y_min1, x_max1, y_max1 = new_bbox

        for bbox in existing_bboxes:
            x_min2, y_min2, x_max2, y_max2 = bbox

            # Add spacing to the bounding boxes
            x_min2 -= min_spacing
            y_min2 -= min_spacing
            x_max2 += min_spacing
            y_max2 += min_spacing

            # Check for overlap
            if not (x_max1 < x_min2 or x_min1 > x_max2 or y_max1 < y_min2 or y_min1 > y_max2):
                return True

        return False




    def generate_segmentation_mask(self, image, bbox):
        """
        Generates a refined segmentation mask for the object within the bounding box.
        Refines the mask with morphological operations and selects the best mask.
        """
        x_min, y_min, x_max, y_max = bbox
        input_box = np.array([x_min, y_min, x_max, y_max])

        # Set the image in the predictor before predicting the mask
        self.predictor.set_image(image)

        # Generate multiple masks with SAM for better selection
        with torch.amp.autocast('cuda'):
            masks, _, _ = self.predictor.predict(box=input_box[None, :], multimask_output=True)

        if masks is not None and len(masks) > 0:
            # Morphological refinement
            refined_masks = []
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            for mask in masks:
                refined_mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
                refined_masks.append(refined_mask)

            # Select the best mask (e.g., largest by area)
            best_mask = max(refined_masks, key=lambda mask: np.sum(mask > 0))
            return best_mask.astype(bool)
        else:
            logging.warning("No masks generated by SAM.")
            return None



    def extract_segmented_object(self, image, mask, x_min, y_min, x_max, y_max):
        """
        Extracts the segmented object from the image using the mask.
        """
        if mask is None:
            return None

        # Crop the object from the original image
        cropped_object = image[y_min:y_max, x_min:x_max]
        mask_cropped = mask[y_min:y_max, x_min:x_max]

        # Apply the mask to the cropped object
        segmented_object = np.zeros_like(cropped_object, dtype=np.uint8)
        for c in range(3):  # For each color channel
            segmented_object[..., c] = np.where(mask_cropped, cropped_object[..., c], 0)

        return segmented_object


    def overlay_object(self, target_image, object_image, x, y):
        """
        Overlays the segmented object on the target image at the specified location.
        """
        obj_h, obj_w = object_image.shape[:2]

        # Overlay the object onto the target image
        target_image[y:y+obj_h, x:x+obj_w] = np.where(
            object_image > 0,
            object_image,
            target_image[y:y+obj_h, x:x+obj_w]
        )
        return target_image


    def create_shadow_image(self, image, masks_for_bboxes, image_file_path):
        """
        Creates an image where only the segmented objects are converted to grayscale, while the background remains unchanged.
        Ensures only shadow-processed images are saved.
        """
        original_image = image.copy()  # Preserve original image
        shadow_image = image.copy()
        modified = False  # Track if modifications happen

        for mask, (x_min, y_min, x_max, y_max) in masks_for_bboxes:
            object_region = shadow_image[y_min:y_max, x_min:x_max]
            grayscale_region = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)

            for c in range(3):
                object_region[..., c] = np.where(
                    mask[y_min:y_max, x_min:x_max],
                    grayscale_region,
                    object_region[..., c]
                )

            shadow_image[y_min:y_max, x_min:x_max] = object_region
            modified = True  # Mark that a change occurred

        # Save only if modifications were made
        if modified and not np.array_equal(shadow_image, original_image):
            shadow_folder = os.path.join(os.path.dirname(image_file_path), "shadow")
            os.makedirs(shadow_folder, exist_ok=True)
            shadow_image_name = f"shadow_{self.generate_unique_name()}.png"
            shadow_image_path = os.path.join(shadow_folder, shadow_image_name)
            cv2.imwrite(shadow_image_path, shadow_image)

            # Copy YOLO annotations only if shadow image is saved
            yolo_file_path, label_exists = self.get_label_file(image_file_path, return_existence=True)
            if label_exists:
                shadow_txt_name = os.path.splitext(shadow_image_name)[0] + ".txt"
                shadow_txt_path = os.path.join(shadow_folder, shadow_txt_name)
                with open(yolo_file_path, "r") as f:
                    labels = [line.strip() for line in f.readlines()]
                self.save_labels_to_file(shadow_txt_path, labels, mode="w")
            else:
                logging.warning(f"Label file does not exist for {image_file_path}.")
        else:
            logging.info(f"No shadow processing applied to {image_file_path}, skipping save.")

        return shadow_image  # Return processed image




    def draw_mask_and_bbox(self, image, mask, class_id=None, num_classes=1, original_bbox=None, obb_points=None, prioritize="largest"):
        """
        Draws mask filled with class color, axis-aligned bbox, optional original bbox, and optional OBB.
        """
        img_height, img_width = image.shape[:2]

        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logging.warning("No contours found in mask.")
            return image, (0, 0, 0, 0)

        # Select contour based on strategy
        if prioritize == "largest":
            target_contour = max(contours, key=cv2.contourArea)
        elif prioritize == "topmost":
            target_contour = min(contours, key=lambda c: cv2.boundingRect(c)[1])
        elif prioritize == "bottommost":
            target_contour = max(contours, key=lambda c: cv2.boundingRect(c)[1])
        else:
            logging.warning(f"Unknown prioritize method: {prioritize}")
            return image, (0, 0, 0, 0)

        # Bounding box
        x_min, y_min, w, h = cv2.boundingRect(target_contour)
        x_max, y_max = x_min + w, y_min + h
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_width - 1, x_max), min(img_height - 1, y_max)

        if x_max <= x_min or y_max <= y_min:
            logging.warning("Invalid axis-aligned bbox.")
            return image, (0, 0, 0, 0)

        # 🎨 Class-specific color
        qt_color = get_color(class_id, num_classes)
        bgr_color = (qt_color.blue(), qt_color.green(), qt_color.red())

        # Mask overlay
        overlay = image.copy()
        alpha = 0.4
        cv2.drawContours(overlay, [target_contour], -1, bgr_color, thickness=cv2.FILLED)
        image_with_boxes = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


        # 🟥 Draw AABB only if overwrite is enabled
        if getattr(self, "overwrite", False) and not self.is_obb_generation_enabled:
            cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), bgr_color, 2)

        # 🔷 Original bbox if available
        if original_bbox and not getattr(self, "overwrite", False):
            orig_x_min, orig_y_min, orig_x_max, orig_y_max = original_bbox
            cv2.rectangle(image_with_boxes, (orig_x_min, orig_y_min), (orig_x_max, orig_y_max), bgr_color, 2)

        # 🟨 OBB
        if self.is_obb_generation_enabled:
            if obb_points and len(obb_points) == 8:
                pts = np.array([
                    [int(obb_points[0] * img_width), int(obb_points[1] * img_height)],
                    [int(obb_points[2] * img_width), int(obb_points[3] * img_height)],
                    [int(obb_points[4] * img_width), int(obb_points[5] * img_height)],
                    [int(obb_points[6] * img_width), int(obb_points[7] * img_height)]
                ], dtype=np.int32)
            else:
                rect = cv2.minAreaRect(target_contour)
                box = cv2.boxPoints(rect)
                pts = np.int0(box)

            cv2.polylines(image_with_boxes, [pts], isClosed=True, color=bgr_color, thickness=2)

        return image_with_boxes, (x_min, y_min, x_max, y_max)





        
    def apply_noise_reduction(self, image, masks_for_bboxes, image_file_path):
        """
        Apply noise reduction selectively to background areas, preserving foreground sharpness.
        Only saves an image if modifications are made.
        """
        original_image = image.copy()  # Preserve the original

        modified = False  # Track if any modifications happen

        for mask, (x_min, y_min, x_max, y_max) in masks_for_bboxes:
            inverted_mask = ~mask[y_min:y_max, x_min:x_max]

            # Process each color channel
            for c in range(3):
                roi = image[y_min:y_max, x_min:x_max, c]
                filtered_roi = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
                image[y_min:y_max, x_min:x_max, c] = np.where(inverted_mask, filtered_roi, roi)

            modified = True  # Mark that a change occurred

        # Only save the image if modifications happened
        if modified and not np.array_equal(image, original_image):
            nr_folder = os.path.join(os.path.dirname(image_file_path), "noise_reduced")
            os.makedirs(nr_folder, exist_ok=True)
            nr_image_name = f"nr_{self.generate_unique_name()}.png"
            nr_image_path = os.path.join(nr_folder, nr_image_name)
            cv2.imwrite(nr_image_path, image)

            # Copy YOLO annotations only if noise-reduced image is saved
            yolo_file_path, label_exists = self.get_label_file(image_file_path, return_existence=True)
            if label_exists:
                nr_txt_name = os.path.basename(nr_image_path).replace('.png', '.txt')
                nr_txt_path = os.path.join(nr_folder, nr_txt_name)
                with open(yolo_file_path, "r") as f:
                    labels = [line.strip() for line in f.readlines()]
                self.save_labels_to_file(nr_txt_path, labels, mode="w")
            else:
                logging.warning(f"Label file does not exist for {image_file_path}.")
        else:
            logging.info(f"No noise reduction applied to {image_file_path}, skipping save.")

        return image  # Return processed image




    def show_image(self, img):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, channel = img_rgb.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(img_rgb.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

        pixmap = QtGui.QPixmap.fromImage(qImg)
        scene = QtWidgets.QGraphicsScene(self)  # Create a new scene for each image
        scene.addPixmap(pixmap)
        self.screen_view.setScene(scene)
        self.screen_view.setSceneRect(QtCore.QRectF(pixmap.rect()))  # Ensure the scene rect fits the pixmap
        self.screen_view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


    def process_batch(self):
        if not self.image_directory or not isinstance(self.image_directory, str):
            QMessageBox.critical(self, "Error", "Please select a valid image directory first.")
            logging.error("No image directory selected. Aborting batch process.")
            return

        self.img_index_number_changed(0)
        self.load_num_classes()
        self.stop_labeling = False
        self.stop_batch = False
        self.image_files = sorted(
            sum([glob.glob(os.path.join(self.image_directory, ext)) for ext in (
                "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tif", "*.webp")], []))
        total_images = len(self.image_files)

        if not self.image_files:
            QMessageBox.critical(self, "Error", "No images found in the selected directory.")
            return

        if not self.sam or not self.predictor:
            QMessageBox.critical(self, "Error", "Please select a folder and model first.")
            return

        def prepare_dir(name):
            path = os.path.join(self.image_directory, name)
            os.makedirs(path, exist_ok=True)
            self.copy_classes_txt(path)
            return path

        dirs = {}
        if self.copy_paste_enabled:
            dirs['copy'] = prepare_dir("copy_and_paste")
            self.apply_copy_paste_augmentation()
        if self.segmentation_checkbox.isChecked():
            dirs['seg'] = prepare_dir("Segmented")
        if self.is_noise_remove_enabled:
            dirs['noise'] = prepare_dir("noise_reduced")
        if self.shadow:
            dirs['shadow'] = prepare_dir("shadow")
        if self.is_obb_generation_enabled:
            dirs['obb'] = prepare_dir("OBB")

        for idx, image_file in enumerate(self.image_files):
            if self.stop_batch or self.stop_labeling:
                break

            label_file = os.path.splitext(image_file)[0] + ".txt"
            if not os.path.exists(label_file):
                logging.info(f"[SKIP] {image_file} - no label file found")
                continue

            with open(label_file, "r") as f:
                labels = [line.strip() for line in f if line.strip()]

            if not labels:
                logging.info(f"[SKIP] {image_file} - empty label file")
                continue

            image = cv2.imread(image_file)
            if image is None:
                continue

            img_height, img_width = image.shape[:2]
            image_original = image.copy()
            visualized_image = image.copy()
            self.predictor.set_image(image)

            obb_lines = []
            seg_lines = []
            adjusted_yolo_lines = []
            alpha = 0.4

            for line in labels:
                try:
                    box_obj = BoundingBox.from_str(line)
                    if not box_obj:
                        continue

                    voc = pbx.convert_bbox(
                        (box_obj.x_center, box_obj.y_center, box_obj.width, box_obj.height),
                        from_type="yolo", to_type="voc", image_size=(img_width, img_height))

                    with torch.amp.autocast('cuda'):
                        masks, _, _ = self.predictor.predict(
                            box=np.array([list(map(int, voc))])[None, :], multimask_output=False)
                    mask = masks[0].astype(np.uint8) * 255

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    contour = max(contours, key=cv2.contourArea)

                    seg_points = contour.reshape(-1, 2)
                    seg_line = f"{box_obj.class_id} " + " ".join(
                        f"{x / img_width:.6f} {y / img_height:.6f}" for x, y in seg_points)
                    seg_lines.append(seg_line)

                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.clip(box, [0, 0], [img_width, img_height])
                    obb_line = f"{box_obj.class_id} " + " ".join(
                        f"{v / img_width:.6f}" if i % 2 == 0 else f"{v / img_height:.6f}"
                        for i, v in enumerate(box.flatten()))
                    obb_lines.append(obb_line)

                    color = get_color(box_obj.class_id, self.num_classes)
                    bgr_color = (color.blue(), color.green(), color.red())
                    cv2.drawContours(visualized_image, [contour], -1, bgr_color, thickness=cv2.FILLED)

                    if self.is_obb_generation_enabled:
                        cv2.polylines(visualized_image, [np.int32(box)], isClosed=True, color=bgr_color, thickness=2)
                    elif self.overwrite:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(visualized_image, (x, y), (x + w, y + h), bgr_color, 2)

                    if self.overwrite:
                        x, y, w, h = cv2.boundingRect(contour)
                        new_bbox = (x, y, x + w, y + h)
                        new_yolo = pbx.convert_bbox(new_bbox, "voc", "yolo", (img_width, img_height))
                        box_obj.x_center, box_obj.y_center, box_obj.width, box_obj.height = new_yolo
                        adjusted_yolo_lines.append(box_obj.to_str(remove_confidence=True))

                except Exception as e:
                    logging.error(f"Failed processing box in {image_file}: {e}")

            base_name = os.path.basename(label_file)

            if 'obb' in dirs and obb_lines:
                with open(os.path.join(dirs['obb'], base_name), "w") as f:
                    f.write("\n".join(obb_lines))
                shutil.copy2(image_file, os.path.join(dirs['obb'], os.path.basename(image_file)))

            if 'seg' in dirs and seg_lines:
                with open(os.path.join(dirs['seg'], base_name), "w") as f:
                    f.write("\n".join(seg_lines))
                shutil.copy2(image_file, os.path.join(dirs['seg'], os.path.basename(image_file)))

            if self.overwrite and adjusted_yolo_lines:
                with open(label_file, "w") as f:
                    f.write("\n".join(adjusted_yolo_lines))

            if self.overwrite or self.screen_update.isChecked():
                self.show_image(visualized_image)

            if 'noise' in dirs:
                self.apply_noise_reduction(image.copy(), [], image_file)

            if 'shadow' in dirs:
                self.create_shadow_image(image.copy(), [], image_file)

            self.label_progress.setValue(idx + 1)
            QtWidgets.QApplication.processEvents()

        QMessageBox.information(self, "Information", "Finished!" if not self.stop_labeling else "Process was stopped!")
        self.label_progress.setValue(total_images)
        self.img_index_number_changed(0)
        logging.info("Process completed and image index reset.")







    def copy_classes_txt(self, target_directory):
        """
        Copies the `classes.txt` file to the specified target directory if it exists.

        Args:
            target_directory (str): The directory where the `classes.txt` file should be copied.
        """
        classes_txt_path = os.path.join(self.image_directory, "classes.txt")

        if os.path.exists(classes_txt_path):
            os.makedirs(target_directory, exist_ok=True)
            new_path = os.path.join(target_directory, "classes.txt")

            try:
                shutil.copy2(classes_txt_path, new_path)  # Copy instead of move
                logging.info(f"Copied classes.txt to {new_path}")
            except Exception as e:
                logging.error(f"Failed to copy classes.txt: {e}")
        else:
            logging.warning("classes.txt file not found. Skipping copy operation.")





    def convert_bbox_to_segmentation(self, image, annotation_file):
        """
        Convert YOLO bounding box labels to segmentation mask labels.
        Uses GrabCut to refine the object boundary before SAM segmentation.
        """
        img_height, img_width = image.shape[:2]
        self.predictor.set_image(image)

        segmentation_labels = []

        with open(annotation_file, "r") as f:
            yolo_lines = [line.strip() for line in f if line.strip()]

        for yolo_line in yolo_lines:
            try:
                box_obj = BoundingBox.from_str(yolo_line)
                if box_obj is None:
                    continue

                # Skip if the object already has segmentation or obb
                if box_obj.segmentation and len(box_obj.segmentation) > 0:
                    continue
                if box_obj.obb and len(box_obj.obb) > 0:
                    continue

                # Convert YOLO box to VOC format
                voc_box = pbx.convert_bbox(
                    (box_obj.x_center, box_obj.y_center, box_obj.width, box_obj.height),
                    from_type="yolo",
                    to_type="voc",
                    image_size=(img_width, img_height)
                )
                x_min, y_min, x_max, y_max = map(lambda v: int(round(v)), voc_box)

                # Refine with GrabCut
                grabcut_mask = self.refine_with_grabcut(image, [x_min, y_min, x_max, y_max])

                # Segment with SAM
                input_box = np.array([x_min, y_min, x_max, y_max])
                with torch.amp.autocast('cuda'):
                    masks, _, _ = self.predictor.predict(box=input_box[None, :], multimask_output=False)


                if masks is None or masks[0].sum() == 0:
                    continue

                sam_mask = (masks[0] > 0).astype(np.uint8) * 255
                refined_mask = cv2.bitwise_and(grabcut_mask, grabcut_mask, mask=sam_mask)

                contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not contours:
                    continue

                largest_contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)

                # Simplify the contour
                contour_length = cv2.arcLength(largest_contour, True)
                epsilon = 0.0001 * contour_length
                simplified = cv2.approxPolyDP(largest_contour, epsilon, True).reshape(-1, 2)

                # Ensure the polygon is closed
                if not np.array_equal(simplified[0], simplified[-1]):
                    simplified = np.vstack([simplified, simplified[0]])

                # Normalize
                simplified[:, 0] /= img_width
                simplified[:, 1] /= img_height

                # Use your BoundingBox class to serialize it properly
                seg_box = BoundingBox(
                    class_id=box_obj.class_id,
                    x_center=0, y_center=0, width=0, height=0,
                    segmentation=simplified.flatten().tolist()
                )
                segmentation_labels.append(seg_box.to_str())

            except Exception as e:
                logging.error(f"Error processing line '{yolo_line}': {e}")

        return segmentation_labels



    def refine_with_grabcut(self, image, bbox, iterations=5):
        """
        Refine segmentation using GrabCut with optimized initialization.
        Uses edge detection and dynamic bbox expansion to improve mask accuracy.
        """
        x_min, y_min, x_max, y_max = bbox
        mask = np.zeros(image.shape[:2], np.uint8)

        # **Dynamically Expand Bbox Based on Object Size**
        expand_pixels = max(5, int(0.05 * (x_max - x_min)))  # Scale with object size
        x_min = max(x_min - expand_pixels, 0)
        y_min = max(y_min - expand_pixels, 0)
        x_max = min(x_max + expand_pixels, image.shape[1])
        y_max = min(y_max + expand_pixels, image.shape[0])

        # **Define Initial Foreground and Background**
        mask[y_min:y_max, x_min:x_max] = 3  # Probable foreground
        mask[y_min+3:y_max-3, x_min+3:x_max-3] = 1  # Definite foreground

        # **Edge Detection Helps Define Better Object Boundaries**
        edges = cv2.Canny(image, 50, 150)
        mask[edges > 0] = 1  # Mark edges as foreground

        # **Ensure `grabCut()` has properly initialized background & foreground models**
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # **Apply GrabCut Using CPU**
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)

        # **Convert Output Mask to Binary**
        final_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

        # **Apply Morphological Refinement**
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)  # Remove noise

        return final_mask

    def load_num_classes(self):
        path = os.path.join(self.image_directory, "classes.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.num_classes = len([line.strip() for line in f if line.strip()])
            logging.info(f"✅ Loaded num_classes: {self.num_classes}")
        else:
            self.num_classes = 1
            logging.warning("⚠️ classes.txt not found, defaulting to 1 class.")


    # function to auto label with dino.py
    def on_dino_label_clicked(self):
        self.dino_label.setEnabled(False)  # Disable the button to prevent multiple clicks

        try:
            if self.image_directory is not None:
                self.img_index_number_changed(0)  # Reset image index

                # 🧠 Ask with Cancel support
                overwrite_reply = QMessageBox.question(
                    self, 'Overwrite Labels',
                    "Do you want to overwrite existing label files?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.No
                )

                # ❌ Cancel check
                if overwrite_reply == QMessageBox.Cancel:
                    logging.info("🛑 Dino labeling canceled by user.")
                    self.dino_label.setEnabled(True)
                    return

                overwrite = overwrite_reply == QMessageBox.Yes

                # ✅ Start worker
                self.dino_worker = DinoWorker(self.image_directory, overwrite)
                self.dino_worker.finished.connect(self.on_dino_finished)
                self.dino_worker.error.connect(self.on_dino_error)
                self.dino_worker.start()
            else:
                QMessageBox.warning(self, 'Directory Not Selected', "Please select an image directory first.")
                self.open_image_video()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"An error occurred: {e}")
            self.dino_label.setEnabled(True)



    def on_dino_finished(self):
        """Called when the DINO worker finishes."""
        QMessageBox.information(self, "DINO Completed", "Auto-labeling completed successfully!")
        self.dino_label.setEnabled(True)  # Re-enable the button

    def on_dino_error(self, error_message):
        """Handle errors in the DINO worker."""
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        self.dino_label.setEnabled(True)  # Re-enable the button


    def update_icon(self):
        self.current_icon_index = (self.current_icon_index + 1) % len(self.icons)
        self.setWindowIcon(self.icons[self.current_icon_index])

    def update_console_output(self):
        global start_time  # Ensure start_time is available globally
        now = datetime.now()
        elapsed_time = now - start_time  # Calculate elapsed time since start_time
        elapsed_time_str = str(elapsed_time).split('.')[0]  # Remove microseconds for display

        # Get CPU, GPU, and memory usage
        cpu_usage = self.cpu_usage
        gpu_usage = self.gpu_usage
        memory_usage = self.memory_usage

        # Create the initial message for CUDA and tip
        init_msg = (
            f"<b>PyTorch CUDA:</b> {'Enabled' if self.pytorch_cuda_available else 'Disabled'}<br>"
            f"<b>OpenCV CUDA:</b> {'Enabled' if self.opencv_cuda_available else 'Disabled'}<br>"
            f"Happy Labeling! Don't forget to check for updates on my <a href='https://github.com/lordofkillz/DarkFusion'>GitHub</a>.<br>"
            f"<b>Tip:</b> For optimal performance, process around 10,000 images at a time.<br>"
        )

        # Construct the full message with CPU, GPU, memory usage, and elapsed time
        text = (
            f'<font color="red">'
            f'UltraDarkLabel Initialized at: {start_time.strftime("%Y-%m-%d %H:%M:%S")}<br>'
            f'Running Time: {elapsed_time_str}<br>'
            f'CPU Usage: {cpu_usage}%<br>'
            f'GPU Usage: {gpu_usage}%<br>'
            f'Memory Usage: {memory_usage}%<br>'
            f'{init_msg}'  # Add init_msg to the final text
            f'</font>'
        )

        # Set the console output text in the PyQt widget
        self.console_output.setText(text)

        # Ensure rich text formatting and clickable links in QLabel
        self.console_output.setTextFormat(QtCore.Qt.RichText)
        self.console_output.setOpenExternalLinks(True)  # Make the link clickable
        # Append the last log message at the bottom
        last_log = getattr(self.label_log_handler, 'last_message', None)
        if last_log:
            text += f"<br><b><span style='color:#000000;'>Last Info:</span></b> <b><span style='color:#007700;'>{last_log}</span></b>"
        self.console_output.setText(text)
        self.console_output.setTextFormat(QtCore.Qt.RichText)
        self.console_output.setOpenExternalLinks(True)

        
    def gather_metrics(self):
        while True:
            self.cpu_usage = psutil.cpu_percent(interval=0.1)
            gpu_devices = GPUtil.getGPUs()
            self.gpu_usage = gpu_devices[0].load if gpu_devices else "No GPU found"
            memory_info = psutil.virtual_memory()
            self.memory_usage = memory_info.percent
            time.sleep(1)  # adjust sleep time to control the frequency of updates

    def start_gathering_metrics(self):
        # Start the thread to gather metrics in the background
        self.metrics_thread = Thread(target=self.gather_metrics)
        self.metrics_thread.start()

    def populate_combo_box(self):
        documents_folder = 'documents'
        if os.path.exists(documents_folder):  # Make sure the folder exists
            text_files = [f for f in os.listdir(documents_folder) if f.endswith('.txt') and f is not None]
            if text_files:  # Check that the list is not empty
                self.display_help.addItems(text_files)
            else:
                logging.warning("No text files found in the directory.")
        else:
            logging.error(f"The folder {documents_folder} does not exist.")

    def load_file(self):
        selected_file = self.display_help.currentText()
        if selected_file and selected_file != 'None':  # Check for valid selection
            file_path = os.path.join('documents', selected_file)
            if os.path.exists(file_path):  # Make sure the file exists
                file_content = self.read_text_file(file_path)
                self.text_edit.setText(file_content)
            else:
                logging.error(f"File {file_path} does not exist.")
        else:
            self.text_edit.clear()  # Clear the text edit if "None" is selected
            logging.warning("No valid file selected.")

    @staticmethod
    def read_text_file(file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def set_current_file_name(self, file_name):
        self.current_file_name = file_name

    def get_all_labels(self):
        all_labels = []

        # Load class names from the classes.txt file
        self.class_names = self.load_classes()

        for img_file in self.image_files:
            base = os.path.splitext(img_file)[0]
            txt_file = base + '.txt'

            # Check if file exists and is not empty
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
                                    logger.warning(f"Warning: Class index {class_index} in {txt_file} is out of range.")
                            except ValueError:
                                logger.warning(f"Error parsing line {line_number} in {txt_file}: '{line.strip()}'")

        return all_labels


    # for blanks
    def on_button_click(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly

        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Images and YOLO Annotation Files", "", options=options)
        if not folder_path:
            return

        # Find an available folder name (blanks, blanks_1, blanks_2, etc.)
        save_folder_name = "blanks"
        save_folder_path = os.path.join(folder_path, save_folder_name)
        counter = 1
        while os.path.exists(save_folder_path):
            save_folder_name = f"blanks_{counter}"
            save_folder_path = os.path.join(folder_path, save_folder_name)
            counter += 1

        # Create the new folder
        os.makedirs(save_folder_path)

        # Get the selected percentage from the combo box
        limit_percentage = int(self.limit_blanks.currentText().rstrip('%'))

        # Get the selected image format from the combo box
        image_format = self.image_format_combo.currentText().lower()

        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_files = []
        yolo_annotation_files = []

        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                image_file = os.path.join(folder_path, file)
                image_files.append(image_file)

                # Check for YOLO annotation for each image and add to yolo_annotation_files
                yolo_annotation_file = os.path.splitext(image_file)[0] + '.txt'
                if not os.path.exists(yolo_annotation_file):
                    logger.warning(f"No YOLO annotation found for {image_file}. Skipping...")
                    continue
                yolo_annotation_files.append(yolo_annotation_file)

        self.processing_thread = ImageProcessingThread(
            image_files, yolo_annotation_files, save_folder_path, limit_percentage, image_format)
        self.processing_thread.progressSignal.connect(
            self.label_progress.setValue)
        self.processing_thread.start()
    # convert format

    def convert_directory(self):
        self.directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.directory:
            # Define valid image file extensions
            valid_extensions = ('.bmp', '.gif', '.jpg', '.jpeg', '.png',
                                '.pbm', '.pgm', '.ppm', '.xbm', '.xpm', '.svg', '.tiff', '.tif')

            # Get all image files in the selected directory
            self.files = [f for f in os.listdir(self.directory) if f.lower().endswith(valid_extensions)]

    def convert_images(self):
        if not hasattr(self, 'directory') or not self.directory:
            QMessageBox.warning(self, "Error", "Image directory not selected.")
            return

        if not hasattr(self, 'files') or not self.files:
            QMessageBox.warning(self, "Error", "No files to convert.")
            return

        target_format = self.select_format.currentText()
        self.target_directory = os.path.join(self.directory, target_format)
        os.makedirs(self.target_directory, exist_ok=True)

        for file in self.files:
            runnable = ImageConverterRunnable(
                self.directory, file, target_format, self.target_directory
            )
            self.threadpool.start(runnable)  # Start the runnable task

    # video download
    
    def remove_selected_video(self):
        current_row = self.video_download.currentRow()
        if current_row != -1:
            self.video_download.removeRow(current_row)

    def download_videos(self):
        logger.debug("download_videos is called")

        directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select download directory')
        if not directory:
            return  # Ensure a directory is selected
        
        for row in range(self.video_download.rowCount()):
            item = self.video_download.item(row, 0)  # Fetch the URL from the first column
            if item and item.text().strip():
                url = item.text().strip()
                logger.debug(f"Processing URL from row {row}: {url}")
                
                # Create and start the download thread
                download_thread = DownloadThread(url, row, directory)
                download_thread.update_status.connect(self.update_download_status)
                download_thread.update_progress.connect(self.update_download_progress)
                self.download_threads.append(download_thread)
                download_thread.start()
            else:
                logger.warning(f"No valid URL found in row {row}, column 0")

    def update_download_progress(self, row, progress):
        # Update the corresponding progress bar based on the row
        self.progress_bars[row].setValue(progress)

    def update_download_status(self, row, status):
        # Only update the status in the table if there is an error or specific message to show.
        if "Failed" in status:
            self.video_download.setItem(row, 1, QTableWidgetItem(status))  # Show errors only

    #VIEO PLAYER FRAME EXTRACTION
    def stop_program(self):
        if hasattr(self, 'timer2') and self.timer2 is not None:
            self.timer2.stop()
            self.timer2.timeout.disconnect()  # Disconnect the signal to avoid residual calls
            self.timer2 = None

        # Stop extraction loops
        self.extracting_frames = False

        # Release capture resources
        self.is_camera_mode = False
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
            self.capture = None

            logger.warning("Processing stopped by user.")


    def toggle_inference(self, state):
        self.inference_enabled = (state == QtCore.Qt.Checked)
        logger.info(f"YOLO inference {'enabled' if self.inference_enabled else 'disabled'}.")


    def get_input_devices(self):
        """Get a list of connected camera devices."""
        index = 0
        devices = []
        while True:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            ret, _ = cap.read()
            cap.release()
            if not ret:
                break
            devices.append(index)
            index += 1
        return devices

    def setup_input_sources(self):
        """Populate the input sources dropdown."""
        self.input_selection.clear()
        self.input_selection.addItem("null")
        self.input_selection.addItem("Desktop")

        devices = self.get_input_devices()
        for dev in devices:
            self.input_selection.addItem(str(dev))  # Camera devices

        # ✅ Set UI initialized flag AFTER setting up input sources
        self.ui_initialized = True

    def on_input_source_changed(self, index):
        if not hasattr(self, 'ui_initialized') or not self.ui_initialized:
            logger.debug("🚀 UI still initializing, skipping input source change.")
            return

        self.stop_program()

        # 💥 Full wipe: detach any old scene, create fresh
        self.screen_view.setScene(None)
        self.graphics_scene = QGraphicsScene(self.screen_view)
        self.screen_view.setScene(self.graphics_scene)
        self.pixmap_item = None

        # 💥 SUPER IMPORTANT: Reset zoom, fit, center cleanly
        self.screen_view.resetTransform()
        self.screen_view.zoom_scale = 1.0
        self.screen_view.fitInView_scale = 1.0
        self.screen_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
        self.screen_view.centerOn(self.graphics_scene.sceneRect().center())

        # Release old capture if needed
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
            self.capture = None

        current_text = self.input_selection.currentText()
        if current_text == "null":
            logger.debug("No input source selected.")
            return

        # Initialize based on new input source
        if current_text == "Desktop":
            self.initialize_classes(input_type="desktop")
        elif current_text.isdigit():
            self.initialize_classes(input_type="webcam")
            device_index = int(current_text)
            self.capture = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                logger.error(f"❌ Unable to access webcam at index {device_index}.")
                self.capture = None
                return
        else:
            self.initialize_classes(input_type="video")
            video_path = current_text
            self.capture = cv2.VideoCapture(video_path)
            if not self.capture.isOpened():
                logger.error(f"❌ Unable to open video file: {video_path}")
                self.capture = None
                return

        logger.info("🔄 Reloading classes from working directory.")
        self.load_classes(data_directory=os.getcwd())

        # 👉 Now display first frame manually
        logger.info("🖼️ Displaying first frame manually after input switch...")
        self.display_camera_input()

        # 🔥 THEN start timer AFTER first frame
        if not hasattr(self, 'timer2') or self.timer2 is None:
            self.timer2 = QTimer()
            self.timer2.timeout.connect(self.display_camera_input)

        self.timer2.start(30)
        logger.info("🟢 Timer started for real-time frame updates.")





    def initialize_input_source(self, input_type):
        """
        Initialize an input source (video, webcam, etc.) and ensure classes.txt exists.
        """
        if input_type == "video" or input_type == "webcam":
            # Use output directory or fallback to default
            directory = getattr(self, 'output_path', self.default_classes_path)
            class_names = self.load_classes(data_directory=directory)
            logger.debug(f"Loaded classes for {input_type}: {class_names}")


    def on_custom_input_changed(self, text):
        if text.isdigit():
            self.video_processor.set_custom_frame_count(int(text))
        else:
            self.video_processor.set_custom_frame_count(None)

    @pyqtSlot(int)
    def on_skip_frames_changed(self, value):
        self.skip_frames_count = value

    @pyqtSlot(int, int)
    def on_custom_size_changed(self, width, height):
        self.custom_size = (width, height)


    def initialize_classes(self, input_type=None):
        """
        Ensure that classes.txt exists for the selected input type.
        Args:
            input_type (str): Type of input (e.g., "image", "video", "webcam").
        """
        if input_type == "image" and hasattr(self, 'image_directory'):
            directory = self.image_directory  # Image mode loads from self.image_directory
        elif input_type in ["video", "webcam", "desktop"]:
            directory = self.output_path or self.default_classes_path or os.getcwd()
        else:
            directory = os.getcwd()  # Fallback directory (current working directory)

        # 🔹 Prevent redundant class loading
        if hasattr(self, "class_names") and self.class_names:
            logger.debug(f"✅ Classes already initialized. Skipping redundant load.")
            return  

        self.class_names = self.load_classes(data_directory=directory)

        # 🔹 Ensure dropdown updates only once
        if not hasattr(self, 'dropdown_initialized') or not self.dropdown_initialized:
            self.update_classes_dropdown(self.class_names)
            self.dropdown_initialized = True  # Mark dropdown as initialized

        logger.debug(f"✅ Initialized classes for {input_type}, loaded from {directory}.")





    def on_crop_images_checkbox_state_changed(self, state):
        """Enable or disable crop dimensions based on checkbox state."""
        self.update_crop_dimensions_enabled(state == QtCore.Qt.Checked)
        self.force_live_redraw()  # Ensure immediate update if cropping is enabled
        if state == QtCore.Qt.Checked:
            # Ensure immediate update if cropping is enabled
            self.display_camera_input()
        
    def resize_frame(self, frame):
        if self.custom_size_checkbox.isChecked():
            width = max(self.width_box.value(), 1)  # Ensure width is at least 1
            height = max(self.height_box.value(), 1)  # Ensure height is at least 1
            return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        return frame
    def force_live_redraw(self):
        """If webcam or video is active, force a redraw using current crop/resize settings."""
        current_text = self.input_selection.currentText()
        if current_text.isdigit() or os.path.isfile(current_text):
            # Webcam or video mode is active — force redraw
            self.display_camera_input()


    def update_crop_dimensions_enabled(self, enabled):
        """Enable or disable the width and height controls for both cropping and resizing."""
        if self.crop_images_checkbox.isChecked() or self.custom_size_checkbox.isChecked():
            self.width_box.setEnabled(True)
            self.height_box.setEnabled(True)
        else:
            self.width_box.setEnabled(False)
            self.height_box.setEnabled(False)


    def crop_frame(self, frame):
        if not self.crop_images_checkbox.isChecked():
            return frame

        frame_height, frame_width = frame.shape[:2]
        width = min(max(self.width_box.value(), 1), frame_width)
        height = min(max(self.height_box.value(), 1), frame_height)

        start_x = max(0, (frame_width - width) // 2)
        start_y = max(0, (frame_height - height) // 2)
        end_x = start_x + width
        end_y = start_y + height

        return frame[start_y:end_y, start_x:end_x]


    def on_crop_images_checkbox_state_changed(self, state):
        """Ensure width/height boxes update immediately when crop is enabled."""
        self.update_crop_dimensions_enabled(state == QtCore.Qt.Checked)
        if state == QtCore.Qt.Checked:
            self.display_camera_input()


        
    def get_image_extension(self):
        """
        Returns the image file extension based on the selected image format.
        """
        if hasattr(self, "image_format"):
            return self.image_format
        else:
            # Fallback to a default if image_format is not set
            return ".jpg"
    def change_monitor(self):
        try:
            selected_index = int(self.monitor_selector.currentText())
            self.current_monitor_index = selected_index
            logger.info(f"🖥️ Switched to monitor {selected_index}")
        except Exception as e:
            logger.error(f"❌ Failed to switch monitor: {e}")
    def detect_and_set_monitor_range(self):
        try:
            with mss.mss() as sct:
                monitor_count = len(sct.monitors)  # includes monitor[0] = full virtual screen
                self.monitor.setMinimum(0)  # skip monitor[0] unless you want full screen
                self.monitor.setMaximum(monitor_count - 1)
                self.current_monitor_index = self.monitor.value()
                logger.info(f"🖥️ Detected {monitor_count - 1} monitor(s). Defaulting to monitor {self.current_monitor_index}")
        except Exception as e:
                logger.error(f"❌ Failed to detect monitors: {e}")
    def set_monitor_index(self, value):
        self.current_monitor_index = value
        logger.info(f"🔁 Monitor index switched to: {value}")
        
    def display_camera_input(self):
        try:
            current_text = self.input_selection.currentText()
            frame = None
            is_rgb_source = False

            # === DESKTOP (MSS) ===
            if current_text == "Desktop":
                self.initialize_classes("desktop")
                with mss.mss() as sct:
                    monitor_index = min(self.current_monitor_index, len(sct.monitors) - 1)
                    monitor = sct.monitors[monitor_index]
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)  # BGRA
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    frame = np.ascontiguousarray(frame)
                    is_rgb_source = False

            # === WEBCAM or CAPTURE CARD ===
            elif current_text.isdigit():
                self.initialize_classes("webcam")
                cam_index = int(current_text)
                if not hasattr(self, 'capture') or self.capture is None or not self.capture.isOpened():
                    self.capture = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    self.capture.set(cv2.CAP_PROP_FPS, 60)  # Realistic max, adjust as needed
                    self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ret, frame = self.capture.read()
                if not ret or frame is None:
                    logger.error("❌ Webcam/CaptureCard read failed.")
                    return

            # === VIDEO FILE ===
            else:
                self.initialize_classes("video")
                if not hasattr(self, 'capture') or self.capture is None or not self.capture.isOpened():
                    self.capture = cv2.VideoCapture(current_text)
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    logger.error("❌ Video read failed.")
                    return

            # === Color Channel Fix ===
            if frame is not None and frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            if frame is None or frame.size == 0:
                logger.warning("❌ Invalid frame, skipping.")
                return

            # === Torch Conversion (GPU) ===
            frame_tensor = torch.from_numpy(frame).to('cuda', non_blocking=True).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)  # CHW

            # === Resize or Crop ===
            if self.custom_size_checkbox.isChecked():
                width = max(self.width_box.value(), 1)
                height = max(self.height_box.value(), 1)
                frame_tensor = F.resize(frame_tensor, [height, width])
            elif self.crop_images_checkbox.isChecked():
                _, h, w = frame_tensor.shape
                crop_w = min(max(self.width_box.value(), 1), w)
                crop_h = min(max(self.height_box.value(), 1), h)
                top = (h - crop_h) // 2
                left = (w - crop_w) // 2
                frame_tensor = F.crop(frame_tensor, top, left, crop_h, crop_w)

            if is_rgb_source:
                frame_tensor = frame_tensor[[2, 1, 0], :, :]  # RGB → BGR

            # === Inference Pipeline ===
            frame_np = (frame_tensor * 255).byte().permute(1, 2, 0).cpu().numpy()
            processed_frame, head_labels = self.apply_preprocessing(np.ascontiguousarray(frame_np))
            results = self.perform_yolo_inference(processed_frame)

            frame_to_show = processed_frame
            if results and results[0] is not None:
                try:
                    frame_to_show = results[0].plot()
                except Exception as e:
                    logger.warning(f"⚠️ Plot fail: {e}")

            display_frame = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            self.update_display(display_frame)

        except Exception as e:
            logger.error(f"❌ display_camera_input error: {e}")



    def update_display(self, frame, segmentations=None):
        try:
            if not self.screen_view:
                logger.error("Error: screen_view is not initialized.")
                return

            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                logger.warning("❌ Invalid frame received. Skipping display update.")
                return

            frame = np.ascontiguousarray(frame)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            # 🛡️ FIX: If the scene is being replaced, reset pixmap item too
            scene_rect = QRectF(pixmap.rect())
            if not hasattr(self, "graphics_scene") or self.graphics_scene is None:
                self.graphics_scene = QGraphicsScene(self.screen_view)
                self.screen_view.setScene(self.graphics_scene)

            # 🖼️ Clear and update the scene
            self.graphics_scene.clear()
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.graphics_scene.addItem(self.pixmap_item)

            # Set the scene rect to exactly the image size
            self.graphics_scene.setSceneRect(scene_rect)

            # 🚀 Force a full refresh and reset zoom
            self.screen_view.resetTransform()
            self.screen_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
            self.screen_view.setSceneRect(scene_rect)
            self.screen_view.centerOn(self.graphics_scene.sceneRect().center())
            self.screen_view.update()
            


        except Exception as e:
            logger.error(f"❌ Error updating display: {e}")

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "graphics_scene") and self.graphics_scene.items():
            if self.screen_view.auto_fit_enabled:
                self.screen_view.reset_zoom()



    def set_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_path = directory  # Update output path
            logger.info(f"Output directory set to: {directory}")

    @pyqtSlot()
    def on_extract_button_clicked(self):
        """
        Handle the 'Extract' button click: start extracting frames from the selected input source.
        """
        if not self.output_path:
            QMessageBox.warning(self, "Error", "Please set the output directory first.")
            self.set_output_directory()
            if not self.output_path:
                return

        # Initialize classes for the output directory
        self.initialize_classes(input_type="video")

        # Reset extracting_frames
        self.extracting_frames = True

        # Check if the model is loaded
        model_loaded = hasattr(self, 'model') and self.model and getattr(self, 'inference_enabled', True)

        if model_loaded:
            logger.info("YOLO inference enabled: Only extracting frames with detections.")
        else:
            if hasattr(self, 'model') and self.model:
                logger.info("YOLO inference disabled: Extracting all frames without inference.")
            else:
                logger.info("No YOLO model loaded: Extracting all frames.")


        # First check if a video is selected in the table
        video_path = self.get_selected_video_path()
        if video_path:  # If a video is selected
            if not os.path.exists(video_path):
                QMessageBox.warning(self, "Error", f"Selected video file does not exist: {video_path}")
                return

            logger.info(f"Extracting frames from video: {video_path}")
            self.extract_frames_from_video(video_path)
        else:
            # If no video is selected in the table, use the input source dropdown
            current_source = self.input_selection.currentText()
            if current_source == "null":
                QMessageBox.warning(self, "Error", "No input source selected!")
                return

            # Determine source type and process accordingly
            if current_source == "Desktop":
                self.extract_frames_from_desktop()
            elif current_source.isdigit():
                self.extract_frames_from_camera(int(current_source))
            else:
                QMessageBox.warning(self, "Error", f"Invalid input source: {current_source}")

        self.extracting_frames = False
        QMessageBox.information(self, "Info", "Frame extraction completed.")


    def extract_frames_from_desktop(self):
        """Extract frames from the desktop."""
        output_dir = os.path.join(self.output_path, "Desktop_Extracted_Frames")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Extracting desktop frames to: {output_dir}")

        # ✅ Check if model is loaded
        model_loaded = hasattr(self, 'model') and self.model and getattr(self, 'inference_enabled', True)

        frame_count = 0
        while self.extracting_frames:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 🔄 If model is not loaded, save the raw frame instead
            if not model_loaded:
                frame_filename = f"frame_{frame_count}.jpg"
                frame_filepath = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_filepath, frame)
                logger.info(f"💾 Saved desktop frame (no detection): {frame_filepath}")
            else:
                # Use save_frame_on_inference() to process and save
                self.save_frame_on_inference(
                    frame=frame,
                    results=None,
                    frame_count=frame_count,
                    output_dir=output_dir
                )

            frame_count += 1
            QtCore.QCoreApplication.processEvents()

        logger.info(f"Completed desktop frame extraction. Output directory: {output_dir}")




    def extract_frames_from_camera(self, device_index):
        """Extract frames from a camera."""
        if not self.output_path:
            QMessageBox.warning(self, "Error", "Output directory not set!")
            return

        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", f"Unable to access camera: {device_index}")
            return

        output_dir = os.path.join(self.output_path, f"Camera_{device_index}_Extracted_Frames")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Extracting camera frames to: {output_dir}")

        # ✅ Check if model is loaded
        model_loaded = hasattr(self, 'model') and self.model and getattr(self, 'inference_enabled', True)

        frame_count = 0
        while self.extracting_frames:
            ret, frame = cap.read()
            if not ret:
                break  # camera feed ended or error

            # 🔄 If model is not loaded, save the raw frame instead
            if not model_loaded:
                frame_filename = f"frame_{frame_count}.jpg"
                frame_filepath = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_filepath, frame)
                logger.info(f"💾 Saved camera frame (no detection): {frame_filepath}")
            else:
                # Use save_frame_on_inference() to process and save
                self.save_frame_on_inference(
                    frame=frame,
                    results=None,
                    frame_count=frame_count,
                    output_dir=output_dir
                )

            frame_count += 1
            QtCore.QCoreApplication.processEvents()

        cap.release()
        self.extracting_frames = False
        logger.info(f"Completed camera frame extraction. Output directory: {output_dir}")




    def extract_frames_from_video(self, video_path):
        """Extract frames from the given video file."""
        logger.info(f"Starting frame extraction for video: {video_path}")
        
        # ✅ Check if model is loaded — moved here from the button click method
        model_loaded = hasattr(self, 'model') and self.model and getattr(self, 'inference_enabled', True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", f"Unable to open video: {video_path}")
            return

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(self.output_path, f"{video_name}_Extracted_Frames")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created: {output_dir}")

        frame_count = 0
        while self.extracting_frames:
            ret, frame = cap.read()
            if not ret:
                logger.warning("End of video or read failure.")
                break

            # 🔄 If model is not loaded, save the raw frame instead
            if not model_loaded:
                frame_filename = f"frame_{frame_count}.jpg"
                frame_filepath = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_filepath, frame)
                logger.info(f"💾 Saved frame (no detection): {frame_filepath}")
            else:
                # Use the updated save_frame_on_inference logic
                self.save_frame_on_inference(
                    frame=frame,
                    results=None,  # Inference handled in save_frame_on_inference
                    frame_count=frame_count,
                    output_dir=output_dir
                )

            frame_count += 1
            QtCore.QCoreApplication.processEvents()

        cap.release()
        self.extracting_frames = False
        logger.info(f"Completed frame extraction for video: {video_path}")



    def save_frame_on_inference(self, frame, results=None, frame_count=None, output_dir=None):
        try:
            if self.custom_size_checkbox.isChecked():
                frame = self.resize_frame(frame)
            elif self.crop_images_checkbox.isChecked():
                frame = self.crop_frame(frame)

            if not output_dir:
                current_source = self.input_selection.currentText()
                if current_source == "Desktop":
                    output_dir = os.path.join(self.output_path, "Desktop_Frames")
                elif current_source.isdigit():
                    output_dir = os.path.join(self.output_path, f"Camera_{current_source}_Frames")
                else:
                    video_name = os.path.splitext(os.path.basename(current_source))[0]
                    output_dir = os.path.join(self.output_path, f"{video_name}_Frames")
            os.makedirs(output_dir, exist_ok=True)

            if frame_count is not None and hasattr(self, 'skip_frames_count'):
                if self.skip_frames_count <= 0:
                    self.skip_frames_count = 1
                if frame_count % self.skip_frames_count != 0:
                    return

            frame_filename = (
                f"frame_{frame_count}{self.image_format}" if frame_count is not None
                else f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}{self.image_format}"
            )
            frame_filepath = os.path.join(output_dir, frame_filename)

            if not self.inference_enabled:
                cv2.imwrite(frame_filepath, frame)
                return

            if results is None:
                processed_frame, _ = self.apply_preprocessing(frame)
                model_kwargs = self.get_model_kwargs()
                results = self.model(processed_frame, **model_kwargs)

            if results is None or not results or results[0] is None:
                cv2.imwrite(frame_filepath, frame)
                return

            result = results[0]
            img_height, img_width = frame.shape[:2]

            is_keypoint = self.is_keypoint_mode()
            is_segmentation = self.is_segmentation_mode()
            is_obb = self.is_obb_mode()

            has_keypoints = hasattr(result, "keypoints") and result.keypoints is not None
            has_masks = hasattr(result, "masks") and result.masks is not None
            has_boxes = hasattr(result, "boxes") and result.boxes is not None

            labels = []

            if is_keypoint and has_keypoints:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                keypoints = result.keypoints.xy.cpu().numpy()
                num_expected_keypoints = self.keypoint_list.rowCount()

                for box_idx, (box, class_id) in enumerate(zip(boxes, class_ids)):
                    x1, y1, x2, y2 = box
                    xc = (x1 + x2) / 2 / img_width
                    yc = (y1 + y2) / 2 / img_height
                    w = (x2 - x1) / img_width
                    h = (y2 - y1) / img_height

                    final_kpt_strs = [(0.0, 0.0, 0)] * num_expected_keypoints
                    if len(keypoints) > box_idx:
                        for idx, (x, y) in enumerate(keypoints[box_idx]):
                            if idx < num_expected_keypoints:
                                nx = x / img_width
                                ny = y / img_height
                                final_kpt_strs[idx] = (nx, ny, 2 if (nx > 0 or ny > 0) else 0)

                    final_kpt_str = " ".join(f"{x:.6f} {y:.6f} {v}" for x, y, v in final_kpt_strs)
                    labels.append(f"{int(class_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {final_kpt_str}")

            elif is_segmentation and has_masks:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                masks = result.masks.xy
                for mask, class_id in zip(masks, class_ids):
                    mask[:, 0] /= img_width
                    mask[:, 1] /= img_height
                    labels.append(f"{int(class_id)} " + " ".join(f"{p:.6f}" for p in mask.flatten()))

            elif is_obb and has_boxes:
                boxes = result.boxes.xywh.cpu().numpy()  # (x_center, y_center, width, height)
                class_ids = result.boxes.cls.cpu().numpy()
                img_height, img_width = frame.shape[:2]

                for (xc, yc, w, h), class_id in zip(boxes, class_ids):
                    # Use the full box area as region for angle estimation
                    x1 = int((xc - w / 2))
                    y1 = int((yc - h / 2))
                    x2 = int((xc + w / 2))
                    y2 = int((yc + h / 2))

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)

                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue

                    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
                    ((box_x, box_y), (box_w, box_h), angle_deg) = rect

                    angle_rad = np.deg2rad(angle_deg)
                    corners = BoundingBox.obb_xywhr_to_corners(
                        (box_x + x1) / img_width,
                        (box_y + y1) / img_height,
                        box_w / img_width,
                        box_h / img_height,
                        angle_rad
                    )
                    labels.append(f"{int(class_id)} " + " ".join(f"{v:.6f}" for v in corners))

            # Save image and labels
            cv2.imwrite(frame_filepath, frame)
            if labels:
                annotation_filepath = frame_filepath.replace(self.image_format, ".txt")
                self.save_labels_to_file(annotation_filepath, labels, mode="w")

        except Exception as e:
            logger.error(f"❌ Error during save_frame_on_inference: {e}")






    @pyqtSlot(int)
    def update_progress(self, progress):
        self.label_progress.setValue(progress)

    def update_checkboxes(self):
        self.height_box.setEnabled(self.custom_size_checkbox.isChecked())
        self.width_box.setEnabled(self.custom_size_checkbox.isChecked())



    def on_custom_frames_toggled(self, state):
        if state == QtCore.Qt.Checked:
            custom_input_text = self.custom_input.text()
            if custom_input_text.isdigit():
                self.video_processor.set_custom_frame_count(int(custom_input_text))
            else:
                self.custom_frames_checkbox.setChecked(False)

    def on_original_size_toggled(self, state):
        self.video_processor.set_original_size(state == QtCore.Qt.Checked)

    def on_custom_size_checkbox_state_changed(self, state):
        self.height_box.setEnabled(state)
        self.width_box.setEnabled(state)
        if state:
            self.on_size_spinbox_value_changed()

    def on_size_spinbox_value_changed(self):
        if self.custom_size_checkbox.isChecked():
            height = self.height_box.value()
            width = self.width_box.value()
            self.video_processor.set_custom_size((width, height))
        else:
            self.video_processor.set_custom_size(None)

    def on_image_format_changed(self, text):
        """
        Update the selected image format based on dropdown text.
        """
        format_mapping = {
            "jpg": ".jpg",
            "jpeg": ".jpeg",
            "gif": ".gif",
            "bmp": ".bmp",
            "png": ".png"  # Match exactly with Qt Designer values
        }
        # Use lowercase to ensure consistency
        self.image_format = format_mapping.get(text.lower(), ".jpg")  # Default to .jpg
        logger.debug(f"Image format set to: {self.image_format}")






    def perform_yolo_inference(self, frame):
        """Perform YOLO inference and return detection results (boxes, masks, keypoints)."""
        if self.model is None or not getattr(self, 'inference_enabled', True):
            return None

        try:
            model_kwargs = self.get_model_kwargs()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                results = self.model(frame, **model_kwargs)  # Run YOLO model inference

            if not results:
                logger.warning("⚠️ YOLO model returned no results.")
                return None

            detections = []
            segmentations = []
            keypoints_list = []

            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                for box in results[0].boxes:
                    class_index = int(box.cls.item())
                    confidence = float(box.conf.item())
                    class_name = self.model.names.get(class_index, f"unknown_{class_index}")
                    detections.append(f"{class_name} {confidence:.2f}")

            if hasattr(results[0], 'masks') and results[0].masks is not None:
                masks = results[0].masks.xy
                class_ids = results[0].boxes.cls.cpu().numpy()
                for seg, class_id in zip(masks, class_ids):
                    segmentations.append((seg, int(class_id)))

            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                keypoints_xy = results[0].keypoints.xy.cpu().numpy()
                keypoints_list = keypoints_xy  # This is (N, K, 2) — N = detections, K = keypoints

            logger.debug(f"YOLO detections: {detections}")
            logger.debug(f"YOLO segmentations: {len(segmentations)} masks found.")
            logger.debug(f"YOLO keypoints: {len(keypoints_list)} detected.")

            return results

        except Exception as e:
            logger.error(f"❌ Error during YOLO inference: {e}")
            return None

    def get_selected_video_path(self):
        current_row = self.video_table.currentRow()
        if current_row != -1:
            return self.video_table.item(current_row, 0).text()
        return None
    
    def on_play_video_clicked(self):
        """Handles video playback when the play button is clicked."""
        video_path = self.get_selected_video_path()

        if not video_path or not os.path.exists(video_path):
            logger.warning("❌ No valid video selected for playback.")
            return

        # Release existing capture
        if hasattr(self, 'capture') and self.capture:
            self.capture.release()

        self.capture = cv2.VideoCapture(video_path)

        if not self.capture.isOpened():
            logger.error(f"⚠️ Failed to open video: {video_path}")
            return

        # Determine FPS and setup timer
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or fps > 240:
            logger.warning(f"⚠️ Unusual FPS ({fps:.2f}), defaulting to 30.")
            fps = 30.0

        interval_ms = int(1000 / fps)
        logger.info(f"🎬 Loaded video: {video_path}")
        logger.info(f"🎞️ FPS: {fps:.2f} | Interval: {interval_ms} ms")

        if not hasattr(self, 'timer2') or self.timer2 is None:
            self.timer2 = QTimer()

        try:
            self.timer2.timeout.disconnect()
        except TypeError:
            pass

        self.timer2.timeout.connect(self.play_video_frame)
        self.timer2.start(interval_ms)

        # Prepare UI elements
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_slider.setMaximum(self.total_frames - 1)
        self.video_slider.setValue(0)
        self.video_slider.setEnabled(True)
        self.slider_is_pressed = False
        self.is_playing = True
        self.play_video_button.setText("Pause")
        self.frame_counter = 0

        logger.info("▶️ Starting video playback...")


    def play_video_frame(self):
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret or frame is None:
                self.stop_video_playback()
                logger.warning("Video complete or read failure.")
                return

            if not isinstance(frame, np.ndarray) or frame.size == 0:
                logger.warning("Invalid frame.")
                return

            try:
                # === CROP & RESIZE BEFORE TENSOR CONVERSION ===
                frame = self.crop_frame(frame)   # 🛠️ Apply cropping
                frame = self.resize_frame(frame) # 🛠️ Apply resizing

                # Perform inference only if model and inference are enabled
                results = None
                if self.model and getattr(self, 'inference_enabled', True):
                    frame_tensor = torch.from_numpy(frame).to('cuda').float() / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1)

                    frame_np = (frame_tensor * 255).byte().permute(1, 2, 0).cpu().numpy()
                    processed_frame, head_labels = self.apply_preprocessing(np.ascontiguousarray(frame_np))
                    results = self.perform_yolo_inference(processed_frame)

                    if hasattr(self, 'output_path') and callable(self.save_frame_on_inference):
                        if results is not None:
                            self.save_frame_on_inference(
                                frame=frame,
                                results=results,
                                frame_count=self.frame_counter,
                                output_dir=self.output_path
                            )
                            self.frame_counter += 1

                # Display frame (with inference if available)
                frame_to_show = frame
                if results and results[0] is not None:
                    frame_to_show = results[0].plot()

                display_frame = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
                self.update_display(display_frame)

                if not self.slider_is_pressed:
                    current_frame = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
                    self.video_slider.setValue(current_frame)
                    self.update_time_label(current_frame)

            except Exception as e:
                logger.error(f"❌ Frame error: {e}")

                
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith(('.mp4', '.avi')):
                self.add_video_to_table(file_path)
    def on_add_video_clicked(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Videos (*.mp4 *.avi)")
        if file_dialog.exec_():
            video_path = file_dialog.selectedFiles()[0]
            # Check for duplicates
            for row in range(self.video_table.rowCount()):
                if self.video_table.item(row, 0).text() == video_path:
                    QMessageBox.information(self, "Info", "Video already added.")
                    return

            row_position = self.video_table.rowCount()
            self.video_table.insertRow(row_position)
            self.video_table.setItem(row_position, 0, QTableWidgetItem(video_path))
            logger.info(f"Video added: {video_path}")

    def add_video_to_table(self, video_path):
        # Prevent duplicates
        for row in range(self.video_table.rowCount()):
            if self.video_table.item(row, 0).text() == video_path:
                logger.info(f"Video already in table: {video_path}")
                return

        row_position = self.video_table.rowCount()
        self.video_table.insertRow(row_position)
        self.video_table.setItem(row_position, 0, QTableWidgetItem(video_path))
        logger.info(f"🎞️ Video added via drop: {video_path}")

    def on_remove_video_clicked(self):
        current_row = self.video_table.currentRow()
        if current_row != -1:
            # Remove the selected row from the table
            self.video_table.removeRow(current_row)
            logger.info(f"Video at row {current_row} removed from the table.")

            # Optionally, you can remove the corresponding entry from self.video_processor.videos if needed
            if hasattr(self, 'video_processor') and hasattr(self.video_processor, 'videos'):
                if len(self.video_processor.videos) > current_row:
                    removed_video = self.video_processor.videos.pop(current_row)
                    logger.info(f"Video '{removed_video}' removed from video_processor.videos.")

    def toggle_play_pause(self):
        if not self.capture or not self.capture.isOpened():
            self.on_play_video_clicked()
            self.is_playing = True
            self.play_video_button.setText("Pause")
            return

        if self.is_playing:
            self.timer2.stop()
            self.is_playing = False
            self.play_video_button.setText("Play")
            logger.info("⏸️ Paused video playback.")
        else:
            self.timer2.start()
            self.is_playing = True
            self.play_video_button.setText("Pause")
            logger.info("▶️ Resumed video playback.")
            
    def stop_video_playback(self):
        """Stop video playback and release resources safely."""
        if hasattr(self, 'timer2') and self.timer2 is not None:
            self.timer2.stop()
            self.timer2.timeout.disconnect()
            self.timer2 = None
            self.is_playing = False
            self.play_video_button.setText("Play")
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
            self.capture = None

        logger.info("Video playback stopped.")

    def resume_video_playback(self):
        if self.capture and self.capture.isOpened():
            if hasattr(self, "video_slider"):
                total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_slider.setMaximum(total_frames - 1)
            self.timer2.start(30)
            logger.info("Video playback resumed.")

    def slider_seek(self):
        """Jump to frame when slider is released and resume playback if it was playing."""
        if self.capture and self.capture.isOpened():
            frame = self.video_slider.value()
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame)

            # Immediately display the frame directly
            ret, frame_data = self.capture.read()
            if ret and frame_data is not None:
                self.display_frame_direct(frame_data, frame)
            else:
                logger.error(f"Could not read frame {frame}")

            self.slider_is_pressed = False

            # Resume playback only if it was playing before slider was pressed
            if self.is_playing:
                fps = self.capture.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or fps > 240:
                    fps = 30.0
                interval_ms = int(1000 / fps)
                self.timer2.start(interval_ms)
                logger.info("Playback resumed after slider adjustment.")




    def slider_pressed(self):
        self.slider_is_pressed = True
        if hasattr(self, 'timer2') and self.timer2.isActive():
            self.timer2.stop()  # Pause playback clearly while dragging
            logger.info("Playback paused during slider adjustment.")




    def update_time_label(self, frame):
        """Optional: update time_label QLabel (00:00 / 03:45)"""
        if hasattr(self, "total_frames") and self.capture:
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                return
            total_sec = int(self.total_frames / fps)
            current_sec = int(frame / fps)
            total_str = f"{total_sec // 60:02}:{total_sec % 60:02}"
            curr_str = f"{current_sec // 60:02}:{current_sec % 60:02}"
            self.time_label.setText(f"{curr_str} / {total_str}")



    def skip_forward(self, frames=30):
        """Skip forward in the video by `frames` (default 30)."""
        if self.capture and self.capture.isOpened():
            current_frame = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            new_frame = min(current_frame + frames, total_frames - 1)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            logger.info(f"⏩ Skipped forward to frame {new_frame}")
            self.play_video_frame()  # display updated frame

    def skip_backward(self, frames=30):
        if self.capture and self.capture.isOpened():
            current_frame = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = max(current_frame - frames, 0)

            # Seek and flush the internal buffer reliably
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = self.capture.read()
            if not ret or frame is None:
                logger.error(f"Error reading frame {new_frame}")
                return

            # Reset position again after flush
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

            # Call a safer method directly
            self.display_frame_direct(frame, new_frame)

            logger.info(f"⏪ Skipped backward to frame {new_frame}")

    def display_frame_direct(self, frame, frame_number):
        """Directly updates display without inference or saving."""
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.update_display(display_frame)
        self.video_slider.setValue(frame_number)
        self.update_time_label(frame_number)


    def resume_video_playback(self):
        if self.capture and self.capture.isOpened():
            self.timer2.start(30)  # Restart the timer for frame-by-frame playback
            logger.info("Video playback resumed.")
        


    # predict and reivew function       
    def process_images_cuda(self, image_files: List[str]) -> List[Image.Image]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()

        def load_and_process(image_file):
            try:
                img = Image.open(image_file).convert('RGB')
                img_tensor = to_tensor(img).to(device, non_blocking=True)
                img_pil = to_pil(img_tensor.cpu())
                return img_pil
            except Exception as e:
                logger.error(f"[process_images_cuda] Error processing {image_file}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            processed_images = [img_pil for img_pil in executor.map(load_and_process, image_files) if img_pil]

        return processed_images




    def lazy_image_batch_generator(self, image_files: List[str], batch_size: int):
        """Lazy generator for image batches."""
        for i in range(0, len(image_files), batch_size):
            yield image_files[i:i + batch_size]

    def batch_process(self, image_directory, batch_size, thumbnails_directory):
        os.makedirs(thumbnails_directory, exist_ok=True)

        # Gather image files
        image_files = [
            f for f in glob.glob(os.path.join(image_directory, '*'))
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

        if not image_files:
            logger.warning("No images found in the directory.")
            return

        batch_generator = self.lazy_image_batch_generator(image_files, batch_size)

        with ThreadPoolExecutor(max_workers=8) as executor:
            for batch_index, batch in enumerate(batch_generator):
                logger.info(f"Processing batch {batch_index + 1}/{(len(image_files) // batch_size) + 1}...")  # ✅ Log progress
                pil_images = self.process_images_cuda(batch)

                futures = []
                for image_file, pil_image in zip(batch, pil_images):
                    label_file = os.path.splitext(image_file)[0] + '.txt'
                    if not os.path.exists(label_file):
                        logger.warning(f"No label file found for {image_file}. Skipping.")
                        continue

                    with open(label_file, 'r') as file:
                        labels = file.readlines()

                    # Process each label and create thumbnails asynchronously
                    for label_index, label_data in enumerate(labels):
                        futures.append(
                            executor.submit(
                                self.create_thumbnail_for_label, 
                                image_file, pil_image, label_data, label_index, thumbnails_directory
                            )
                        )

                # Wait for all tasks to complete
                for future in futures:
                    future.result()
        logger.info(f"Processing completed for {len(image_files)} images.")



    def create_thumbnail_for_label(self, image_file, pil_image, label_data, label_index, thumbnails_directory):
        bbox = BoundingBox.from_str(label_data)
        if bbox is None:
            logger.warning(f"Invalid label data: {label_data}")
            return

        img_width, img_height = pil_image.size
        polygon = bbox.to_polygon(img_width, img_height)

        x_min, y_min, x_max, y_max = map(int, polygon.bounds)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_width, x_max), min(img_height, y_max)

        if x_min >= x_max or y_min >= y_max:
            logger.warning(f"Skipping invalid crop region for {image_file}, label {label_index}")
            return

        cropped_array = np.array(pil_image)[y_min:y_max, x_min:x_max]

        if cropped_array.size == 0:
            logger.warning(f"Crop resulted in empty array for {image_file}, label {label_index}")
            return

        base_file = os.path.splitext(os.path.basename(image_file))[0]
        thumbnail_filename = os.path.join(thumbnails_directory, f"{base_file}_{label_index}.jpeg")

        try:
            Image.fromarray(cropped_array).save(thumbnail_filename, "JPEG")
            logger.info(f"Saved thumbnail: {thumbnail_filename}")
        except Exception as e:
            logger.error(f"Failed to save thumbnail {thumbnail_filename}: {e}")






    def start_debounce_timer(self, value):
        self._image_size_value = min(value, self.MAX_SIZE)
        self._debounce_timer.start(300)

    def _perform_size_adjustment(self):
        """Resize thumbnails based on user input."""
        if not hasattr(self, '_image_size_value'):  
            self._image_size_value = 128  # Default value if missing

        for row in range(self.preview_list.rowCount()):
            thumbnail_label = self.preview_list.cellWidget(row, 0)
            if thumbnail_label:
                current_size = thumbnail_label.pixmap().size()
                if current_size.width() != self._image_size_value:
                    resized_pixmap = thumbnail_label.pixmap().scaled(
                        self._image_size_value, self._image_size_value, 
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    thumbnail_label.setPixmap(resized_pixmap)


    def update_batch_size(self, value):
        self.settings['batchSize'] = value
        self.saveSettings()
                

    def stop_processing_and_clear(self):
        self.processing = False

        # Check the checkbox and update UI immediately
        self.dont_show_img_checkbox.setChecked(True)
        self.toggle_image_display()
        QApplication.processEvents()

        # Clear table contents
        self.preview_list.setRowCount(0)
        QApplication.processEvents()

        logger.info("[stop_processing_and_clear] Processing stopped and UI cleared.")

        if self.preview_list.rowCount() != 0:
            logger.error(f"[stop_processing_and_clear] Error: Table still contains {self.preview_list.rowCount()} rows after clear attempt.")
        else:
            logger.info("[stop_processing_and_clear] Table cleared successfully.")

        # Slight delay before unchecking again to ensure UI updates
        QTimer.singleShot(200, self.reset_image_display_checkbox)

    def reset_image_display_checkbox(self):
        self.dont_show_img_checkbox.setChecked(False)
        self.toggle_image_display()
        QApplication.processEvents()

    def toggle_image_display(self):
        show_images = not self.dont_show_img_checkbox.isChecked()
        self.preview_list.setColumnWidth(0, 200 if show_images else 0)

    def eventFilter(self, source, event):
        try:
            if event.type() == QEvent.MouseButtonPress and source is self.preview_list.viewport():
                item = self.preview_list.itemAt(event.pos())
                if item is not None:
                    index = self.preview_list.indexFromItem(item)
                    if not self._debounce_timer.isActive() or self.last_row_clicked != index.row():
                        self.last_row_clicked = index.row()
                        self._debounce_timer.start(300)  # Prevent repeated events
                        if event.button() == Qt.RightButton:
                            self.handle_right_click(index)
                        elif event.button() == Qt.LeftButton:
                            self.handle_left_click(index)
                    else:
                        logger.debug(f"Ignored event for row: {index.row()}, column: {index.column()}")
            return super().eventFilter(source, event)
        except Exception as e:
            logger.error(f"An unexpected error occurred in eventFilter: {e}")
            return False

    


    def highlight_thumbnail(self, row):
        thumbnail_label = self.preview_list.cellWidget(row, 0)
        thumbnail_label.setStyleSheet("border: 2px solid red;")

    def unhighlight_all_thumbnails(self):
        for row in range(self.preview_list.rowCount()):
            thumbnail_label = self.preview_list.cellWidget(row, 0)
            thumbnail_label.setStyleSheet("")

    @pyqtSlot(QModelIndex)
    def handle_right_click(self, index):
        logger.debug(f"Right-clicked on row: {index.row()}")
        row = index.row()
        try:
            image_item = self.preview_list.item(row, 0)
            label_item = self.preview_list.item(row, 4)
            if image_item is not None and label_item is not None:
                bbox_preview_index = label_item.data(Qt.UserRole)             # Not used for deletion
                label_type = label_item.data(Qt.UserRole + 1)
                true_line_index = label_item.data(Qt.UserRole + 2)            # ✅ Actual line to delete

                image_file = image_item.text()

                self.delete_item(row, image_file, true_line_index, label_type)

                # ✅ After delete, rebuild thumbnails to match
                self.update_thumbnail_indices(image_file)

                QApplication.processEvents()
            else:
                logger.error("Image or label item is None.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in handle_right_click: {str(e)}")




    def delete_item(self, row, image_file, true_line_index, label_type):
        # 1️⃣ Verify deletion first!
        deletion_successful = self.update_label_file(image_file, true_line_index)
        
        if not deletion_successful:
            logger.error("Deletion unsuccessful! Aborting further deletion steps.")
            return

        # ✅ Proceed ONLY if deletion successful
        self.delete_thumbnail(image_file, true_line_index)

        # Remove graphical items from the QGraphicsScene
        scene = self.screen_view.scene()
        for item in scene.items():
            if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
                if getattr(item, 'file_name', None) == image_file:
                    item_index = getattr(item, 'unique_id', '').split('_')[-1]
                    item_label_type = getattr(item, 'label_type', 'bbox')
                    if item_label_type == label_type and item_index.isdigit() and int(item_index) == true_line_index:
                        scene.removeItem(item)
                        logger.info(f"Removed graphical item {item.unique_id} from scene.")


        # Remove the row from preview list
        self.preview_list.removeRow(row)

        # Update thumbnails and UI
        self.update_thumbnail_indices(image_file)
        self.rebuild_preview_list(image_file)

        # No further saving here!

        # Synchronize ListView
        self.synchronize_list_view(image_file)




    def rebuild_preview_list(self, image_file):
        label_file = os.path.splitext(image_file)[0] + '.txt'
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        self.preview_list.setRowCount(0)

        pixmap = QPixmap(image_file)
        img_width, img_height = pixmap.width(), pixmap.height()

        for line_index, line in enumerate(lines):
            bbox = BoundingBox.from_str(line)
            if bbox is None:
                continue

            polygon = bbox.to_polygon(img_width, img_height)

            # Use polygon.bounds to get accurate bounding rectangle
            x1, y1, x2, y2 = polygon.bounds
            x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)])

            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid bounding area for label: {line.strip()}")
                continue

            cropped_pixmap = pixmap.copy(x1, y1, x2 - x1, y2 - y1)
            resized_pixmap = cropped_pixmap.scaled(
                self._image_size_value, self._image_size_value, 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            thumbnail_label = QLabel()
            thumbnail_label.setPixmap(resized_pixmap)
            thumbnail_label.setAlignment(Qt.AlignCenter)

            row_count = self.preview_list.rowCount()
            self.preview_list.insertRow(row_count)
            self.preview_list.setItem(row_count, 0, QTableWidgetItem(image_file))
            self.preview_list.setCellWidget(row_count, 0, thumbnail_label)

            class_name = self.id_to_class.get(bbox.class_id, "Unknown")
            bbox_width, bbox_height = x2 - x1, y2 - y1

            label_type = "bbox"
            if bbox.segmentation:
                label_type = "seg"
            elif bbox.obb:
                label_type = "obb"

            self.preview_list.setItem(row_count, 1, QTableWidgetItem(class_name))
            self.preview_list.setItem(row_count, 2, QTableWidgetItem(str(bbox.class_id)))
            self.preview_list.setItem(row_count, 3, QTableWidgetItem(f"{bbox_width}x{bbox_height}"))

            bbox_item = QTableWidgetItem(line.strip())
            bbox_item.setData(Qt.UserRole, line_index)          # flashing index
            bbox_item.setData(Qt.UserRole + 1, label_type)      # type
            bbox_item.setData(Qt.UserRole + 2, line_index)      # exact file line index
            self.preview_list.setItem(row_count, 4, bbox_item)

        self.preview_list.resizeRowsToContents()
        self.preview_list.resizeColumnsToContents()
        QApplication.processEvents()





    def get_image_dimensions(self, image_file):
        """
        Retrieve the width and height of an image.
        Args:
            image_file (str): Path to the image file.
        Returns:
            Tuple[int, int]: Width and height of the image.
        """
        try:
            img = cv2.imread(image_file)
            if img is not None:
                return img.shape[1], img.shape[0]  # (width, height)
            else:
                raise ValueError(f"Could not load image: {image_file}")
        except Exception as e:
            logger.error(f"Error getting dimensions for {image_file}: {str(e)}")
            return 0, 0


    def update_label_file(self, image_file, true_line_index):
        label_file = os.path.splitext(image_file)[0] + '.txt'
        logger.info(f"Attempting to delete label index {true_line_index} from file: {label_file}")

        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            logger.info(f"Lines BEFORE deletion ({len(lines)} lines): {lines}")

            if true_line_index < 0 or true_line_index >= len(lines):
                logger.error(f"Invalid label index {true_line_index} for {label_file}")
                return False  # Return false to signify failure

            deleted_line = lines.pop(true_line_index)
            logger.info(f"Deleted line: {deleted_line.strip()}")

            with open(label_file, 'w') as f:
                f.writelines(lines)

            logger.info(f"Lines AFTER deletion ({len(lines)} lines): {lines}")
            return True  # Return true for success

        except Exception as e:
            logger.error(f"Error updating label file {label_file}: {str(e)}")
            return False






    def delete_thumbnail(self, image_file, bbox_index):
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        thumbnail_png = os.path.join(self.thumbnails_directory, f"{base_filename}_{bbox_index}.png")
        thumbnail_jpeg = os.path.join(self.thumbnails_directory, f"{base_filename}_{bbox_index}.jpeg")

        if not self.attempt_delete_thumbnail(thumbnail_png):
            logger.error(f"Thumbnail not found: {thumbnail_png}")
        if not self.attempt_delete_thumbnail(thumbnail_jpeg):
            logger.error(f"Thumbnail not found: {thumbnail_jpeg}")

    def update_thumbnail_indices(self, image_file):
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.splitext(image_file)[0] + '.txt'

        if not os.path.exists(label_file):
            logger.error(f"Label file not found for {image_file}")
            return

        with open(label_file, 'r') as f:
            updated_labels = f.readlines()

        # Clear existing thumbnails
        for filename in os.listdir(self.thumbnails_directory):
            if filename.startswith(base_filename):
                os.remove(os.path.join(self.thumbnails_directory, filename))
                logger.info(f"Deleted old thumbnail: {filename}")

        img = cv2.imread(image_file)
        if img is None:
            logger.error(f"Could not open image: {image_file}")
            return

        img_height, img_width = img.shape[:2]

        for new_index, line in enumerate(updated_labels):
            bbox = BoundingBox.from_str(line)
            if bbox is None:
                logger.warning(f"Skipping invalid label: {line}")
                continue

            polygon = bbox.to_polygon(img_width, img_height)

            x_min, y_min, x_max, y_max = polygon.bounds
            x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
            x_max, y_max = min(img_width, int(x_max)), min(img_height, int(y_max))

            if x_min >= x_max or y_min >= y_max:
                logger.warning(f"Invalid crop for {image_file}, line {new_index}")
                continue

            cropped_img = img[y_min:y_max, x_min:x_max]
            new_thumbnail_path = os.path.join(self.thumbnails_directory, f"{base_filename}_{new_index}.jpeg")
            cv2.imwrite(new_thumbnail_path, cropped_img)
            logger.info(f"Saved updated thumbnail: {new_thumbnail_path}")


    def attempt_delete_thumbnail(self, thumbnail_path):
        if os.path.exists(thumbnail_path):
            try:
                os.remove(thumbnail_path)
                logger.info(f"Successfully deleted {thumbnail_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete {thumbnail_path}: {str(e)}")
        else:
            logger.error(f"Thumbnail not found: {thumbnail_path}")
        return False

    def realign_remaining_entries(self):
        for i in range(self.preview_list.rowCount()):
            bbox_item = self.preview_list.item(i, 4)
            if bbox_item:
                class_id = int(bbox_item.text().split()[0])
                class_name = self.id_to_class.get(class_id, "Unknown")
                self.preview_list.setItem(i, 1, QTableWidgetItem(class_name))


    def update_flash_time(self, value):
        self.flash_time_value = value
    
    def pick_flash_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.flash_color_rgb = color.getRgb()[:3]
    
    def pick_alternate_flash_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.alternate_flash_color_rgb = color.getRgb()[:3]
    
    @pyqtSlot(QModelIndex)
    def handle_left_click(self, index):
        row = index.row()
        image_item = self.preview_list.item(row, 0)
        if image_item is None:
            logger.error("No image item found.")
            return

        image_file = image_item.text()
        self.display_image(image_file)
        self.unhighlight_all_thumbnails()
        self.highlight_thumbnail(row)
        if self.hide_labels:
            self.hide_labels = False
            self.toggle_label_visibility()

        bounding_box_item = self.preview_list.item(row, 4)
        if bounding_box_item:
            bounding_box_index = bounding_box_item.data(Qt.UserRole)
            label_type = bounding_box_item.data(Qt.UserRole + 1)
            logger.info(f"Attempting to flash bounding box with index: {bounding_box_index} for file: {image_file}")
            if bounding_box_index is not None:
                self.flash_bounding_box(bounding_box_index, image_file)
            else:
                logger.error("Bounding box index not found.")
        else:
            logger.error("Bounding box item not found.")
        self.synchronize_list_view(image_file)
        
    def flash_bounding_box(self, bbox_index, image_file):
        if bbox_index is None:
            logger.error(f"Invalid bounding box index: {bbox_index} for file: {image_file}")
            return

        found = False
        base_unique_id = f"{image_file}_{bbox_index}"

        # 🔎 Try normal bbox id
        for item in self.screen_view.scene().items():
            if hasattr(item, "unique_id") and item.unique_id == base_unique_id:
                if isinstance(item, BoundingBoxDrawer):
                    item.flash_color = QColor(*self.flash_color_rgb)
                    item.alternate_flash_color = QColor(*self.alternate_flash_color_rgb)
                elif isinstance(item, SegmentationDrawer):
                    item.flash_color = QColor(*self.flash_color_rgb)
                    item.alternate_flash_color = item._base_color

                item.start_flashing(100, self.flash_time_value)
                found = True
                break

        # 🔎 Try segmentation if not found
        if not found:
            for item in self.screen_view.scene().items():
                seg_unique_id = f"{image_file}_seg_{bbox_index}"
                if hasattr(item, "unique_id") and item.unique_id == seg_unique_id:
                    if isinstance(item, SegmentationDrawer):
                        item.flash_color = QColor(*self.flash_color_rgb)
                        item.alternate_flash_color = item._base_color
                        item.start_flashing(100, self.flash_time_value)
                        found = True
                        break

        if found:
            logger.info(f"Flashing initiated successfully for {base_unique_id} or {base_unique_id}_seg.")
        else:
            logger.warning(f"No matching bounding box or segmentation found for {base_unique_id}. Possibly deleted.")




    def synchronize_list_view(self, image_file):
        file_basename = os.path.basename(image_file)
        self.List_view.clearSelection()
        list_view_model = self.List_view.model()
        for i in range(list_view_model.rowCount()):
            if list_view_model.item(i).text() == file_basename:
                matching_index = list_view_model.index(i, 0)  # Assuming the image file name is in the first column
                self.List_view.scrollTo(matching_index, QAbstractItemView.PositionAtCenter)
                self.List_view.selectionModel().select(matching_index, QItemSelectionModel.Select)
                logger.info(f"ListView synchronized for {file_basename}")
                break  # Exit loop after finding the first match
        self.List_view.setStyleSheet("""
        QListView::item:selected {
            border: 2px solid red;
            background-color: none;
        }
        QListView::item:selected:!active {
            border: 2px solid red;
            background-color: none;
        }
        """)
        
    def sync_list_view_selection(self, file_name):
        """
        Synchronize the ListView selection with the displayed image and highlight it.
        """
        if file_name:
            file_name_base = os.path.basename(file_name)
            model = self.List_view.model()

            if model is None:
                logger.error("List_view model is not set.")
                return

            # Iterate through the list view model to find the matching file name
            for row in range(model.rowCount()):
                item = model.item(row)

                # If the item matches the current image file name, select it
                if item.text() == file_name_base:
                    index = model.indexFromItem(item)
                    self.List_view.setCurrentIndex(index)
                    break

    def setup_list_view_with_delegate(self):
        """
        Set up the ListView to use the custom delegate that adds a red border around selected items.
        """
        delegate = RedBoxDelegate(self.List_view)
        self.List_view.setItemDelegate(delegate)
                

    def adjust_column_width(self, value):
        self._image_size_value = min(value, self.MAX_SIZE)
        self._perform_size_adjustment()




    def extract_and_display_data(self):
        self.processing = True
        self.preview_list.clearContents()
        self.preview_list.setRowCount(0)
        if not hasattr(self, '_image_size_value'):
            self._image_size_value = 128

        try:
            if self.image_directory is None:
                QMessageBox.warning(self.loader.main_window, "No Directory Selected", "Please select a directory before previewing images.")
                return

            data_directory = self.image_directory
            self.ui_loader.setup_ui(show_images=not self.dont_show_img_checkbox.isChecked())
            self.thumbnails_directory = os.path.join(data_directory, "thumbnails")
            os.makedirs(self.thumbnails_directory, exist_ok=True)

            image_files = self.filtered_image_files if self.filtered_image_files else self.image_files
            if not image_files:
                QMessageBox.warning(self.loader.main_window, "No Images Found", "No images found. Please load images before adjusting the slider.")
                return

            self.load_classes(data_directory)

            current_filter_text = self.filter_class_spinbox.currentText()
            filter_mapping = {"All": -1, "Blanks": -2, "Size: Small": -3, "Size: Medium": -4, "Size: Large": -5}
            current_filter = filter_mapping.get(current_filter_text.split(" (")[0], -1)

            if current_filter not in filter_mapping.values():
                try:
                    current_filter = int(current_filter_text.split(":")[0])
                except ValueError:
                    current_filter = -1

            self.label_progress.setMaximum(len(image_files))
            self.label_progress.setValue(0)

            batch_size = self.batch_size_spinbox.value()

            for i in range(0, len(image_files), batch_size):
                if not self.processing:
                    break

                batch_images = image_files[i:i + batch_size]
                pil_images = self.process_images_cuda(batch_images)

                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = []

                    for idx, image_file in enumerate(batch_images):
                        if idx >= len(pil_images):
                            continue

                        pixmap = QPixmap(image_file)
                        img_width, img_height = pil_images[idx].size
                        label_path = os.path.splitext(image_file)[0] + '.txt'

                        if not os.path.exists(label_path):
                            if current_filter == -2:
                                row_count = self.preview_list.rowCount()
                                self.preview_list.insertRow(row_count)
                                self.preview_list.setItem(row_count, 0, QTableWidgetItem(image_file))
                                self.label_progress.setValue(self.label_progress.value() + 1)
                            continue

                        with open(label_path, 'r') as file:
                            lines = file.readlines()

                        if current_filter == -2:
                            continue

                        small_thresh = 32 * 32
                        medium_thresh = 96 * 96

                        for line_index, line in enumerate(lines):
                            bbox = BoundingBox.from_str(line)
                            if bbox is None:
                                continue

                            polygon = bbox.to_polygon(img_width, img_height)
                            x1, y1, x2, y2 = map(int, polygon.bounds)
                            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)

                            width, height = x2 - x1, y2 - y1
                            area = width * height

                            if current_filter == -3 and area >= small_thresh:
                                continue
                            elif current_filter == -4 and (area < small_thresh or area >= medium_thresh):
                                continue
                            elif current_filter == -5 and area < medium_thresh:
                                continue
                            elif current_filter >= 0 and bbox.class_id != current_filter:
                                continue

                            if width <= 0 or height <= 0:
                                continue

                            cropped_pixmap = pixmap.copy(x1, y1, width, height)
                            thumbnail_filename = os.path.join(self.thumbnails_directory, f"{os.path.basename(image_file)}_{line_index}.jpeg")
                            futures.append(executor.submit(cropped_pixmap.save, thumbnail_filename, "JPEG"))

                            if not self.dont_show_img_checkbox.isChecked():
                                resized_pixmap = cropped_pixmap.scaled(
                                    self._image_size_value, self._image_size_value, 
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                                )
                                thumbnail_label = QLabel()
                                thumbnail_label.setPixmap(resized_pixmap)
                                thumbnail_label.setAlignment(Qt.AlignCenter)

                                row_count = self.preview_list.rowCount()
                                self.preview_list.insertRow(row_count)
                                self.preview_list.setItem(row_count, 0, QTableWidgetItem(image_file))
                                self.preview_list.setCellWidget(row_count, 0, thumbnail_label)

                                class_name = self.id_to_class.get(bbox.class_id, "Unknown")
                                label_type = "seg" if bbox.segmentation else "obb" if bbox.obb else "bbox"

                                self.preview_list.setItem(row_count, 1, QTableWidgetItem(class_name))
                                self.preview_list.setItem(row_count, 2, QTableWidgetItem(str(bbox.class_id)))
                                self.preview_list.setItem(row_count, 3, QTableWidgetItem(f"{width}x{height}"))

                                bbox_item = QTableWidgetItem(line.strip())
                                bbox_item.setData(Qt.UserRole, line_index)               # For flashing
                                bbox_item.setData(Qt.UserRole + 1, label_type)           # Label type
                                bbox_item.setData(Qt.UserRole + 2, line_index)           # Exact file line index
                                self.preview_list.setItem(row_count, 4, bbox_item)

                                self.label_progress.setValue(self.label_progress.value() + 1)
                                self.preview_list.resizeRowsToContents()
                                self.preview_list.resizeColumnsToContents()
                                QApplication.processEvents()

                    for f in futures:
                        try:
                            f.result()
                        except Exception as e:
                            logger.error(f"Thumbnail save error: {e}")

            logger.info("Extract and display process completed.")

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")




                 


    # mute sounds

    def mute_player(self, state):
        # Check the state of the muteCheckBox
        is_muted = state == QtCore.Qt.Checked

        # Set the muted state of the sound_player
        self.sound_player.setMuted(is_muted)
        # settings


    def keyPressEvent(self, event):
        key_code = event.key()
        modifiers = event.modifiers()

        # Block unwanted keys
        if key_code in (Qt.Key_Control, Qt.Key_Alt):
            event.accept()
            return

        if key_code == Qt.Key_Delete:
            event.accept()
            self.stop_next_timer()
            self.stop_prev_timer()
            self.delete_current_image()
            return

        key = event.text()

        if key == self.settings.get('nextButton'):
            if modifiers == Qt.NoModifier:
                self.stop_next_timer()
                self.stop_prev_timer()
                self.next_frame()
                if self.auto_scan_checkbox.isChecked():
                    self.start_next_timer()
            elif modifiers == Qt.ControlModifier:
                self.quick_next_navigation()

        elif key == self.settings.get('previousButton'):
            if modifiers == Qt.NoModifier:
                self.stop_next_timer()
                self.stop_prev_timer()
                self.previous_frame()
                if self.auto_scan_checkbox.isChecked():
                    self.start_prev_timer()
            elif modifiers == Qt.ControlModifier:
                self.quick_previous_navigation()

        elif key == self.settings.get('deleteButton'):
            self.stop_next_timer()
            self.stop_prev_timer()
            self.delete_current_image()

        else:
            # Append to buffer for multi-key handling
            self.keyBuffer += key

            # Restart timer every keystroke (for combos like "11", "a1", etc.)
            if self.timer.isActive():
                self.timer.stop()
            self.timer.start(300)  # Adjust timeout as needed


    def processKeyPresses(self):
        key = self.keyBuffer
        modifiers = QApplication.keyboardModifiers()
        class_name = None

        # ⏩ Navigation
        if key == self.settings.get('nextButton'):
            if modifiers == Qt.NoModifier:
                self.next_frame()
            elif modifiers == Qt.ControlModifier:
                self.quick_next_navigation()

        elif key == self.settings.get('previousButton'):
            if modifiers == Qt.NoModifier:
                self.previous_frame()
            elif modifiers == Qt.ControlModifier:
                self.quick_previous_navigation()

        # 🔀 Mode switching
        elif key == self.settings.get('modeBox'):
            self.segmentation_mode.setCurrentText("Boxes")
            self.flash_overlay_text("📦 Boxes Mode", color=QColor(100, 200, 255))  # Light blue

        elif key == self.settings.get('modeSegmentation'):
            self.segmentation_mode.setCurrentText("Segmentation")
            self.flash_overlay_text("✏️ Segmentation Mode", color=QColor(255, 215, 0))  # Gold

        elif key == self.settings.get('modeKeypoint'):
            self.segmentation_mode.setCurrentText("Keypoints")
            self.flash_overlay_text("📍 Keypoints Mode", color=QColor(255, 105, 180))  # Pink

        # 🔄 New OBB Mode integration
        elif key == self.settings.get('modeOBB'):
            self.segmentation_mode.setCurrentText("OBB")
            self.flash_overlay_text("🧊 OBB Mode", color=QColor(0, 204, 102))  # Green shade

        # 🗑️ Delete
        elif key == self.settings.get('deleteButton'):
            self.delete_current_image()

        # 🧠 Class hotkey handling
        else:
            class_hotkeys = {
                k: v for k, v in self.settings.items() if k.startswith('classHotkey_')
            }
            for class_key, hotkey in class_hotkeys.items():
                if hotkey == key:
                    class_name = class_key.split('classHotkey_')[-1]
                    break

        # 🎯 Class switching and flash feedback
        if class_name:
            logger.info(f"Class name found: {class_name}")
            index = self.classes_dropdown.findText(class_name)
            if index >= 0:
                self.classes_dropdown.setCurrentIndex(index)

                class_id = index
                num_classes = self.classes_dropdown.count()
                color = get_color(class_id, num_classes, alpha=255)

                # 🎲 Random icon flair
                random_icons = ["🎯", "🚀", "✨", "🔥", "🎲", "🎉", "📌", "🔍", "🧠", "🖍️"]
                icon = random.choice(random_icons)

                self.flash_overlay_text(
                    f"{icon} {class_name}",
                    duration=1800,
                    color=color,
                    font_size=36
                )
            else:
                logger.warning(f"⚠️ Class '{class_name}' not found in dropdown.")

        # ⛔ Clear buffer and stop timer always
        self.keyBuffer = ""
        if self.timer.isActive():
            self.timer.stop()



    def openSettingsDialog(self):
        logger.info("Settings button was clicked.")
        settingsDialog = SettingsDialog(self)
        settingsDialog.exec_()

    def loadSettings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
        except FileNotFoundError:
            settings = self.defaultSettings()
        except json.JSONDecodeError:
            logger.error("Error decoding JSON, using default settings")
            settings = self.defaultSettings()
        return settings

    def defaultSettings(self):
        return {
            'lastTheme': 'Default',
            'lastWeightsPath': '',
            'lastCfgPath': '',
            'lastPTWeightsPath': '',
            'last_dir': '',
            'lastImage': '',
            'anchors': [],
            'batchSize': 1000,
            'modeBox': '1',
            'modeSegmentation': '2',
            'modeKeypoint': '3',
            'modeOBB': '4',    # <-- Added OBB default key
            'nextButton': 'd',
            'previousButton': 'a',
            'deleteButton': 'e',
            'autoLabel': 'l',
        }


    
    def saveSettings(self):
        # Update lastTheme in settings to currently applied theme
        self.settings['lastTheme'] = self.styleComboBox.currentText()
        with open('settings.json', 'w') as f:
            json.dump(self.settings, f, indent=4)

    def set_sound(self, sound_path):
        self.sound_player.setMedia(
            QMediaContent(QUrl.fromLocalFile(sound_path)))
        self.sound_player.play()

    # all part of the auto label function.

    def stop_auto_labeling(self):
        self.stop_labeling = True
        self.reset_progress_bar()
        torch.cuda.empty_cache()  # Clear CUDA cache



    def slider_value_changed(self):
        # Get raw slider values
        conf_val = self.confidence_threshold_slider.value()
        nms_val = self.nms_threshold_slider.value()

        # Update thresholds
        self.confidence_threshold = conf_val / 100
        self.nms_threshold = nms_val / 100

        # Log both raw and threshold values
        logging.info(
            f"📊 Confidence Slider: {conf_val}/100 → Threshold: {self.confidence_threshold:.2f} | "
            f"NMS Slider: {nms_val}/100 → Threshold: {self.nms_threshold:.2f}"
        )

        # Optionally update inference only if net is initialized and current file is valid
        if hasattr(self, 'net') and hasattr(self, 'current_file') and os.path.exists(self.current_file):
            self.process_timer.start(self.process_timer_interval)
        else:
            logging.debug("Net not ready or no valid file selected — thresholds updated but inference not triggered.")




    @functools.lru_cache(maxsize=None)  # No size limit for the cache
    def get_cached_image(self, file_path: str) -> Optional[np.ndarray]:
        image = cv2.imread(file_path)
        if image is None:
            logging.error(f"Failed to load image at {file_path}")
        return image

    @functools.lru_cache(maxsize=None)  # No size limit for the cache
    def get_cached_blob(self, file_path: str) -> Optional[np.ndarray]:
        image = self.get_cached_image(file_path)

        if image is None:
            return None 
        return image  # Return the original image if cropping is not possible or not enabled


    def open_weights(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Weights/Model", "", "Model Files (*.pt *.engine *.onnx *.weights)", options=options
        )

        if file_name:
            self.weights_file_path = file_name  # Store the weights file path
            try:
                # Initialize the YOLO model
                self.model = YOLO(file_name)
                self.model_type = self.determine_model_type(file_name)

                logging.info(f"Loaded {self.model_type} model from {file_name}")
                QMessageBox.information(self, "Model Loaded", f"Successfully loaded {self.model_type} model.")
            except Exception as e:
                logging.error(f"Failed to load model from {file_name}: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load the model: {str(e)}")
                self.model = None


    def open_cfg(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CFG", "", "Config Files (*.cfg)", options=options)
        if file_name:
            self.cfg_file_path = file_name  # Store the cfg file path
            QMessageBox.information(self, "CFG File Loaded", "Successfully loaded the CFG file.")
            # No need to immediately initialize YOLO or start auto-labeling after loading cfg




    def determine_model_type(self, file_path):
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


    def auto_label_images2(self):
        logging.info("Starting auto_label_images2")

        self.class_labels = self.load_classes()
        if not self.class_labels:
            QMessageBox.critical(self, "Error", "Classes file not found or empty.")
            return

        if not hasattr(self, 'model') or self.model is None:
            QMessageBox.critical(self, "Error", "Model is not initialized.")
            return

        self.img_index_number_changed(0)
        total_images = len(self.image_files)
        self.label_progress.setRange(0, total_images)

        overwrite = QMessageBox.question(
            self, 'Overwrite Labels',
            "Do you want to overwrite existing labels?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No
        )

        if overwrite == QMessageBox.Cancel:
            logging.info("❌ Labeling canceled by user.")
            return

        import mediapipe as mp
        mp_pose = mp.solutions.pose.Pose(static_image_mode=True)

        # Use keypoint count from keypoint list or fallback to points.json
        num_expected_keypoints = self.keypoint_list.rowCount()
        points_file = os.path.join(self.image_directory, "points.json")
        if os.path.exists(points_file):
            try:
                with open(points_file, "r") as f:
                    data = json.load(f)
                    if "keypoints" in data:
                        num_expected_keypoints = len(data["keypoints"])
                        logging.info(f"✅ Using {num_expected_keypoints} keypoints from points.json")
            except Exception as e:
                logging.warning(f"⚠️ Failed to read points.json: {e}")

        for idx, image_file in enumerate(self.image_files):
            if self.stop_labeling:
                logging.info("Labeling stopped by user.")
                break

            image = self.read_image(image_file)
            if image is None:
                logging.warning(f"Skipping invalid image: {image_file}")
                continue

            self.display_image(image_file)
            img_width, img_height = Image.open(image_file).size

            try:
                preprocessed_image, _ = self.apply_preprocessing(self.current_display_image.copy(), None, img_width, img_height)
                with torch.amp.autocast(device_type='cuda'):
                    results = self.model.predict(preprocessed_image, **self.get_model_kwargs())
                result = results[0]
            except Exception as e:
                logging.error(f"Model inference error: {e}")
                QMessageBox.critical(self, "Error", f"Error processing {image_file}: {e}")
                continue

            new_bboxes = self.extract_bboxes_from_result(result, img_width, img_height, image_file, mp_pose, num_expected_keypoints)
            label_file, label_exists = self.get_label_file(image_file, return_existence=True)

            if overwrite == QMessageBox.Yes or not label_exists:
                deduped_bboxes = new_bboxes
            else:
                existing_bboxes = self.load_existing_bboxes(label_file)
                if self.is_keypoint_mode():
                    self.inject_keypoints(existing_bboxes, new_bboxes, image_file, mp_pose, num_expected_keypoints)
                    combined = existing_bboxes
                else:
                    combined = existing_bboxes + new_bboxes

                deduped_bboxes = self.remove_near_duplicate_bounding_boxes(combined, iou_threshold=0.5, class_aware=True)

            labels_to_save = [bbox.to_str() for bbox in deduped_bboxes]
            self.save_labels_to_file(label_file, labels_to_save, mode='w')
            logging.info(f"✅ Labels saved to: {label_file}")

            self.label_progress.setValue(idx + 1)
            QApplication.processEvents()

        self.label_progress.setValue(total_images)
        QMessageBox.information(self, "Information", "Labeling completed!")
        logging.info("auto_label_images2 completed successfully")
        self.img_index_number_changed(0)


    # Additional helper methods for clarity and simplicity:

    def extract_bboxes_from_result(self, result, img_width, img_height, image_file, mp_pose, num_expected_keypoints=0):


        new_bboxes = []

        if self.is_segmentation_mode() and hasattr(result, "masks") and result.masks:
            masks = result.masks.xy
            class_ids = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for mask, class_id, conf in zip(masks, class_ids, confidences):
                if conf < 0.35:
                    continue
                mask[:, 0] /= img_width
                mask[:, 1] /= img_height
                new_bboxes.append(BoundingBox(class_id, 0, 0, 0, 0, segmentation=mask.flatten().tolist()))

        elif self.is_obb_mode() and hasattr(result, "obb") and result.obb:
            obbs = result.obb.xywhr.cpu().numpy()
            class_ids = result.obb.cls.cpu().numpy()
            confidences = result.obb.conf.cpu().numpy()
            for obb, class_id, conf in zip(obbs, class_ids, confidences):
                if conf < 0.35 or len(obb) != 5:
                    continue
                x_center, y_center, width, height, angle = obb
                corners = self.obb_xywhr_to_corners(x_center, y_center, width, height, angle)
                corners = np.array(corners).reshape(-1, 2)
                corners[:, 0] /= img_width
                corners[:, 1] /= img_height
                new_bboxes.append(BoundingBox(class_id, 0, 0, 0, 0, obb=corners.flatten().tolist()))

        elif self.is_keypoint_mode() and hasattr(result, "keypoints") and result.keypoints:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy()

            # 🔁 Load keypoint count override from points.json (if exists)
            num_expected_keypoints = self.keypoint_list.rowCount()
            points_file = os.path.join(os.path.dirname(self.image_files[0]), "points.json")
            if os.path.exists(points_file):
                try:
                    with open(points_file, "r") as f:
                        data = json.load(f)
                        if "keypoints" in data:
                            num_expected_keypoints = len(data["keypoints"])
                            logging.info(f"✅ Using {num_expected_keypoints} keypoints from points.json")
                except Exception as e:
                    logging.warning(f"⚠️ Failed to load points.json: {e}")


            for box, class_id, conf, yolo_kpts in zip(boxes, class_ids, confidences, keypoints):
                if conf < 0.35:
                    continue

                x1, y1, x2, y2 = box
                xc = (x1 + x2) / 2 / img_width
                yc = (y1 + y2) / 2 / img_height
                w = (x2 - x1) / img_width
                h = (y2 - y1) / img_height

                final_kpts = [(0.0, 0.0, 0)] * num_expected_keypoints

                visible_count = 0
                for idx, (kp_x, kp_y) in enumerate(yolo_kpts):
                    if idx >= num_expected_keypoints:
                        break
                    nx = np.clip(kp_x / img_width, 0, 1)
                    ny = np.clip(kp_y / img_height, 0, 1)

                    if nx == 0.0 and ny == 0.0:
                        visibility = 0
                    else:
                        visibility = 2
                        visible_count += 1

                    final_kpts[idx] = (nx, ny, visibility)

                # MediaPipe fallback if YOLO detections insufficient
                if visible_count < num_expected_keypoints * 0.6:
                    cropped = cv2.imread(str(image_file))[int(y1):int(y2), int(x1):int(x2)]
                    if cropped.size != 0:
                        results = mp_pose.process(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                        if results.pose_landmarks:
                            for row in range(num_expected_keypoints):
                                name = self.keypoint_list.item(row, 1).text().lower()
                                landmark_enum = getattr(mp.solutions.pose.PoseLandmark, name.upper().replace(" ", "_"), None)
                                if landmark_enum is not None:
                                    lm = results.pose_landmarks.landmark[landmark_enum]
                                    x_abs = x1 + np.clip(lm.x, 0, 1) * (x2 - x1)
                                    y_abs = y1 + np.clip(lm.y, 0, 1) * (y2 - y1)
                                    nx = np.clip(x_abs / img_width, 0, 1)
                                    ny = np.clip(y_abs / img_height, 0, 1)
                                    
                                    if nx == 0.0 and ny == 0.0:
                                        visibility = 0
                                    else:
                                        visibility = 2 if lm.visibility > 0.5 else 1
                                        
                                    final_kpts[row] = (nx, ny, visibility)

                bbox = BoundingBox(class_id, xc, yc, w, h, confidence=conf, keypoints=final_kpts)
                new_bboxes.append(bbox)



        elif self.is_boxes_mode() and hasattr(result, "boxes") and result.boxes:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for box, class_id, conf in zip(boxes, class_ids, confidences):
                if conf < 0.35:
                    continue
                x1, y1, x2, y2 = box
                xc = (x1 + x2) / 2 / img_width
                yc = (y1 + y2) / 2 / img_height
                w = (x2 - x1) / img_width
                h = (y2 - y1) / img_height
                new_bboxes.append(BoundingBox(int(class_id), xc, yc, w, h, confidence=conf))

        return new_bboxes



    def load_existing_bboxes(self, label_file):
        with open(label_file, 'r') as f:
            bboxes = []
            for line in f:
                bbox = BoundingBox.from_str(line.strip())
                if bbox is not None:  # Explicitly filter None
                    bboxes.append(bbox)
                else:
                    logging.warning(f"Malformed bbox line skipped: {line.strip()}")
            return bboxes


    def inject_keypoints(self, existing_bboxes, new_bboxes, image_file, mp_pose, num_expected_keypoints):

        image_bgr = cv2.imread(image_file)
        img_height, img_width = image_bgr.shape[:2]

        pose_result = mp_pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        num_expected_keypoints = self.keypoint_list.rowCount()
        keypoint_names = [
            self.keypoint_list.item(row, 1).text().lower().replace(' ', '_')
            for row in range(num_expected_keypoints)
        ]

        landmark_indices = [
            getattr(mp.solutions.pose.PoseLandmark, kp_name.upper(), None)
            for kp_name in keypoint_names
        ]

        for bbox in existing_bboxes:
            if not bbox.keypoints or len(bbox.keypoints) != num_expected_keypoints:
                bbox.keypoints = [(0.0, 0.0, 0)] * num_expected_keypoints

            if bbox is None:
                continue

            matched_bbox = next((new for new in new_bboxes if self.calculate_iou(new, bbox) > 0.5), None)

            if matched_bbox and matched_bbox.keypoints:
                bbox.keypoints = matched_bbox.keypoints
            elif pose_result.pose_landmarks:
                final_keypoints = []
                for landmark_enum in landmark_indices:
                    if landmark_enum is not None:
                        lm = pose_result.pose_landmarks.landmark[landmark_enum]
                        final_keypoints.append((
                            lm.x,
                            lm.y,
                            int(lm.visibility > 0.5)
                        ))
                    else:
                        final_keypoints.append((0.0, 0.0, 0))
                bbox.keypoints = final_keypoints
            else:
                # Explicitly set ghost points
                bbox.keypoints = [(0.0, 0.0, 0)] * num_expected_keypoints
                logging.warning(f"No keypoints found for {image_file}; set to ghost points.")







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
        conf_threshold = self.confidence_threshold_slider.value() / 100
        iou_threshold = self.nms_threshold_slider.value() / 100
        batch_size = self.batch_inference.value()
        use_fp16 = True if getattr(self, 'fp_mode', 0) == 1 else False

        # Dynamically use resized or cropped shape
        if self.custom_size_checkbox.isChecked() or self.crop_images_checkbox.isChecked():
            width = max(self.width_box.value(), 1)
            height = max(self.height_box.value(), 1)
        else:
            width = self.network_width.value()
            height = self.network_height.value()

        if not hasattr(self, "class_names") or not self.class_names:
            self.load_classes()

        class_indices = list(range(len(self.class_names))) if self.class_names else None

        model_kwargs = {
            'conf': conf_threshold,
            'iou': iou_threshold,
            'imgsz': [width, height],     # Use actual frame size
            'batch': batch_size,
            'device': 0,
            'classes': class_indices,
            'half': use_fp16,
            'agnostic_nms': True,
            'max_det': 100,
            'retina_masks': True
        }

        logger.debug(f"Model kwargs for inference: {model_kwargs}")
        return model_kwargs




    def process_current_image(self):
        if not hasattr(self, 'current_file'):
            logger.warning("No image file is currently opened.")
            return

        # Load the image and blob from cache
        image = self.get_cached_image(self.current_file)

        # Perform object detection and display the result
        self.infer_and_display_bounding_boxes(image)

        # Save the detected bounding boxes to a file
        label_file = self.get_label_file(self.current_file)
        self.save_bounding_boxes(label_file, self.screen_view.scene(
        ).width(), self.screen_view.scene().height())



    def switch_floating_point_mode(self):
        current_text = self.fp_select_combobox.currentText()
        if current_text == "FP32":
            self.set_floating_point_mode(0)  # 0 for FP32
        elif current_text == "FP16":
            self.set_floating_point_mode(1)  # 1 for FP16

    def set_floating_point_mode(self, mode):
        if not hasattr(self, 'net'):
            logger.error("Error: net is not initialized. Call initialize_yolo() first.")
            return

        self.yolo_floating_point = mode
        self.apply_backend_and_target()

    def apply_backend_and_target(self):
        """
        Sets the preferable backend and target based on the system's hardware capabilities.

        This function checks the available hardware acceleration options and configures the neural network accordingly.
        """
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Check if CUDA-enabled GPU is available
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

            # Choose the target based on user's preference for floating point precision
            target = cv2.dnn.DNN_TARGET_CUDA_FP16 if self.yolo_floating_point else cv2.dnn.DNN_TARGET_CUDA
            logger.info('Using CUDA:', 'FP16' if self.yolo_floating_point else 'FP32')

            # Set the preferable target
            self.net.setPreferableTarget(target)

        elif cv2.ocl.haveOpenCL():
            # Check if OpenCL is available (usually for some CPUs and GPUs)
            logger.info('Using OpenCL')

            # Set OpenCL as the backend
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

            # Set OpenCL as the target
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        else:
            # If no hardware acceleration is available, use CPU
            logger.info('Using CPU')

            # Set default CPU backend
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

            # Set CPU as the target
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



    def initialize_yolo(self):
        try:
            self.classes = []  # This can probably be removed later if unused

            if hasattr(self, 'weights_file_path') and hasattr(self, 'cfg_file_path'):
                file_extension = os.path.splitext(self.weights_file_path)[1]

                if file_extension == '.weights':
                    self.net = cv2.dnn.readNet(self.weights_file_path, self.cfg_file_path)

                    current_text = self.fp_select_combobox.currentText()
                    self.yolo_floating_point = 0 if current_text == "FP32" else 1

                    self.apply_backend_and_target()

                    # Directly get the names of the unconnected output layers
                    self.layer_names = self.net.getUnconnectedOutLayersNames()

                    #  Load classes dynamically
                    self.class_labels = self.load_classes()

                    if not self.class_labels:
                        logger.error("Error: classes.txt not found or empty. Ensure it exists in the image directory.")
                else:
                    logger.error("Unsupported file extension for weights file. Please use a .weights file.")
            else:
                logger.error("Weights and/or CFG files not selected. Please ensure both files are selected for .weights models.")
        except Exception as e:
            logger.error(f"Error initializing YOLO: {e}")








    def save_cropped_image(self, image, file_path):
        """
        Saves the cropped image to a 'cropped' subdirectory. If 'cropped' exists in a subsequent run,
        save to 'cropped_1', if that exists, save to 'cropped_2', and so on.
        """
        directory = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)

        if self.current_cropped_directory is None:
            counter = 0
            while os.path.exists(os.path.join(directory, f'cropped_{counter}')):
                counter += 1
            self.current_cropped_directory = os.path.join(directory, f'cropped_{counter}')
            os.makedirs(self.current_cropped_directory, exist_ok=True)

        cropped_file_path = os.path.join(self.current_cropped_directory, base_name)
        cv2.imwrite(cropped_file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return cropped_file_path

    def center_crop(self, image: np.ndarray, desired_height: int, desired_width: int, file_path: str = None) -> np.ndarray:
        """
        Crops the image from its center and saves it to a 'cropped' subdirectory if file_path is provided.
        """
        if image is None:
            logging.warning("Image is None. Cannot perform cropping.")
            return None

        height, width, _ = image.shape

        if desired_height > height or desired_width > width:
            logging.warning(f"Desired dimensions ({desired_height}x{desired_width}) are larger than the original dimensions ({height}x{width}). Cannot perform cropping.")
            return image

        left = (width - desired_width) // 2
        top = (height - desired_height) // 2
        right = (width + desired_width) // 2
        bottom = (height + desired_height) // 2

        cropped_image = image[top:bottom, left:right]

        if file_path:
            self.save_cropped_image(cropped_image, file_path)

        return cropped_image

    def read_image(self, file_path: str) -> np.ndarray:
        """
        Reads an image from the given file path and returns it as a numpy array.
        Handles errors gracefully by logging them and returning None.
        """
        try:
            if file_path is None:
                raise ValueError("File path is None")
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Could not read image from {file_path}")
            return image
        except Exception as e:
            logging.error(f"Failed to read image: {e}")
            return None

    def save_yolo_label_for_cropped_image(self, label_file, cropped_image, cropped_region):
        try:
            with open(label_file, 'r') as f:
                original_label_data = f.readlines()

            cropped_label_file = label_file.replace('.txt', '_cropped.txt')

            with open(cropped_label_file, 'w') as f:
                for line in original_label_data:
                    parts = line.strip().split()
                    class_id = parts[0]
                    x_center, y_center, width, height = map(float, parts[1:])

                    x_center_rel = (x_center - cropped_region[0]) / cropped_region[2]
                    y_center_rel = (y_center - cropped_region[1]) / cropped_region[3]
                    width_rel = width / cropped_region[2]
                    height_rel = height / cropped_region[3]

                    f.write(f"{class_id} {x_center_rel} {y_center_rel} {width_rel} {height_rel}\n")

            logger.info(f"Saved YOLO label for cropped image: {cropped_label_file}")
        except Exception as e:
            logger.error(f"Error saving YOLO label for cropped image: {e}")

    def process_image(self, overwrite):
        if not hasattr(self, 'current_file'):
            logger.warning("No image file is currently opened.")
            return

        label_file = self.get_label_file(self.current_file)

        if label_file is None:
            logger.warning("Label file could not be generated. Skipping image.")
            return

        if os.path.exists(label_file) and not overwrite:
            return

        image = self.read_image(self.current_file)

        # Perform cropping if enabled

        # Apply preprocessing (grayscale, edge detection, super-resolution)
        image, head_labels = self.apply_preprocessing(image)

        # Overwrite bounding boxes during inference
        self.infer_and_display_bounding_boxes(image)
        
        # Save the detected bounding boxes
        self.save_bounding_boxes(label_file, self.screen_view.scene().width(), self.screen_view.scene().height(), extra_labels=head_labels)




    def auto_label_images(self):
        """
        Auto-label images using YOLO model with .weights and .cfg files.
        Includes overwrite, append, and cancel functionality.
        """
        if self.model_type == 'weights' and (not hasattr(self, 'weights_file_path') or not hasattr(self, 'cfg_file_path')):
            QMessageBox.warning(self, "Missing Files", "Both .weights and .cfg files are required for this model type.")
            return

        # Load class labels
        self.class_labels = self.load_classes()
        if not self.class_labels:
            logging.error("Failed to load class labels.")
            return

        # Initialize YOLO if not already initialized
        if not hasattr(self, 'net'):
            self.initialize_yolo()

        total_images = len(self.image_files)
        self.label_progress.setRange(0, total_images)

        # Prompt user to overwrite, append, or cancel
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
            logging.info("🛑 Auto-labeling cancelled by user.")
            return

        overwrite = overwrite_reply == QMessageBox.Yes

        # Process each image
        for idx, image_file in enumerate(self.image_files):
            if self.stop_labeling:
                logging.info("Labeling stopped by user.")
                break

            self.current_file = image_file
            label_file = self.get_label_file(self.current_file)

            # Load and preprocess the image
            original_image = self.read_image(self.current_file)
            if original_image is None:
                logging.error(f"Failed to load image {self.current_file}. Skipping.")
                continue

            self.display_image(self.current_file)

            # Perform YOLO inference
            new_labels = self.perform_yolo_inference(self.current_file)

            if os.path.exists(label_file):
                if overwrite:
                    self.save_labels(label_file, new_labels)
                else:
                    existing_labels = self.load_labels(label_file)
                    combined_labels = self.combine_labels(existing_labels, new_labels)
                    self.save_labels(label_file, combined_labels)
            else:
                self.save_labels(label_file, new_labels)

            self.label_progress.setValue(idx + 1)
            QApplication.processEvents()

        QMessageBox.information(self, "Auto-Labeling", "Finished auto-labeling all images.")
        self.stop_labeling = False


    def auto_label_yolo_button_clicked(self) -> bool:
        logging.info("auto_label_yolo_button_clicked called")

        if not hasattr(self, 'image_directory') or self.image_directory is None:
            logging.error("Image directory not selected.")
            return False

        if not hasattr(self, 'weights_file_path') or self.weights_file_path is None:
            logging.error("Weights file path not selected.")
            return False

        if self.weights_file_path.endswith('.weights') and (not hasattr(self, 'cfg_file_path') or self.cfg_file_path is None):
            logging.error("CFG file path not selected for .weights model.")
            return False

        self.class_labels = self.load_classes()  # Use the centralized function
        if not self.class_labels:
            logging.error("Classes file not found or empty.")
            return False

        if self.weights_file_path.endswith(('.pt', '.engine', '.onnx')):
            logging.info("PyTorch model detected. Running auto_label_images2.")
            self.auto_label_images2()
        elif self.weights_file_path.endswith('.weights'):
            logging.info("Darknet model detected. Running auto_label_images.")
            if not hasattr(self, 'net'):
                self.initialize_yolo()
            self.img_index_number_changed(0)
            total_images = len(self.image_files)
            self.label_progress.setRange(0, total_images)
            overwrite = QMessageBox.question(self, 'Overwrite Labels', "Do you want to overwrite existing labels?",
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            for idx, image_file in enumerate(self.image_files):
                if self.stop_labeling:
                    break

                self.current_file = image_file
                label_file = self.get_label_file(self.current_file)
                if os.path.exists(label_file) and overwrite == QMessageBox.No:
                    with open(label_file, 'r') as f:
                        label_data = f.read().strip()
                    if label_data:
                        continue

                original_image = self.read_image(self.current_file)
                if original_image is None:
                    logging.error(f"Failed to load image at {self.current_file}")
                    continue
                else:
                    self.display_image(original_image)

                self.display_image(self.current_file)
                self.process_image(overwrite)
                if self.hide_labels:
                    self.hide_labels = False
                    self.toggle_label_visibility()

                self.label_progress.setValue(idx + 1)
                QApplication.processEvents()

            QMessageBox.information(self, "Auto-Labeling", "Finished auto-labeling all images.")
            self.stop_labeling = False
            self.img_index_number_changed(0)
            logging.info("Image index reset to 0 and display updated.")
        return True



    def infer_and_display_bounding_boxes(self, image) -> List[List[int]]:
        # No longer clearing bounding boxes; they will be replaced directly.
        
        if not hasattr(self, 'net'):
            logger.error("Network not initialized. Please select weights file and cfg file.")
            return []

        # Alternative scales to try
        scales = [1/255.0, 1/224.0, 1/192.0]

        all_boxes = []
        all_confidences = []
        all_class_ids = []

        # Get output layer names and indices once, outside the loop
        output_layer_indices = self.net.getUnconnectedOutLayers()
        output_layer_names = [self.net.getLayerNames()[i - 1] for i in output_layer_indices]

        for scale in scales:
            # Creating a blob with the current scale
            blob = cv2.dnn.blobFromImage(image, scale, (self.network_width.value(), self.network_height.value()), swapRB=True)

            self.net.setInput(blob)
            outputs = self.net.forward(output_layer_names)
            boxes, confidences, class_ids = self.get_detections(outputs, image.shape)

            # Collecting boxes, confidences, and class_ids from the current scale
            all_boxes.extend(boxes)
            all_confidences.extend(confidences)
            all_class_ids.extend(class_ids)

        # Perform NMS on all collected boxes, confidences, and class_ids
        indices = cv2.dnn.NMSBoxes(all_boxes, all_confidences, self.confidence_threshold, self.nms_threshold)
        indices = np.array(indices).flatten()  # Convert to NumPy array once

        # Create bounding boxes for the indices retained after NMS
        for idx in indices:
            x, y, w, h = all_boxes[idx]
            confidence = all_confidences[idx]
            self.class_labels = self.load_classes()  # Ensure latest classes are used
            label = self.class_labels[all_class_ids[idx]] if all_class_ids[idx] < len(self.class_labels) else f"obj{all_class_ids[idx]+1}"

            
            # Directly overwrite the existing bounding boxes instead of clearing them
            self.create_bounding_box(x, y, w, h, label, confidence)

        return [all_boxes[i] for i in indices]  # Use the converted NumPy array


    def get_detections(self, outputs: List[np.ndarray], shape: Tuple[int, int]) -> Tuple[List[List[int]], List[float], List[int]]:
        """
        Extracts and filters object detections from the YOLO model's output.

        Args:
            outputs (List[np.ndarray]): List of YOLO model output layers.
            shape (Tuple[int, int]): Shape of the input image (height, width).

        Returns:
            Tuple[List[List[int]], List[float], List[int]]: A tuple containing lists of detected boxes, confidences, and class IDs.
        """
        # Initialize lists to store detected boxes, confidences, and class IDs
        boxes, confidences, class_ids = [], [], []

        # Iterate through each output layer's detections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Check if the confidence threshold is met
                if confidence > self.confidence_threshold:
                    # Calculate the bounding box coordinates and dimensions
                    box = detection[0:4] * np.array([shape[1], shape[0], shape[1], shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Calculate minimum and maximum width and height based on user settings
                    min_width_px, max_width_px = self.box_size.value() / 100 * shape[1], self.max_label.value() / 100 * shape[1]
                    min_height_px, max_height_px = self.box_size.value() / 100 * shape[0], self.max_label.value() / 100 * shape[0]

                    # Check if the detected object's dimensions meet the size criteria
                    if min_width_px <= width <= max_width_px and min_height_px <= height <= max_height_px:
                        x, y = int(centerX - (width / 2)), int(centerY - (height / 2))

                        # Append the detected box, confidence, and class ID to the respective lists
                        boxes.append([x, y, int(max(min_width_px, width)), int(max(min_height_px, height))])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        # Return the lists of detected boxes, confidences, and class IDs
        return boxes, confidences, class_ids


    def set_image_directory(self, new_directory):
        """Set image directory as a class attribute and refresh statistics."""
        if not os.path.exists(new_directory):
            logger.error(f"Error: The directory {new_directory} does not exist.")
            return

        self.image_directory = new_directory
        logger.info(f"Image directory set to {self.image_directory}.")
        # Trigger the stats calculation and display immediately
        stats = self.process_labels_and_generate_stats()
        self.settings['stats'] = stats
        self.display_stats()  # Immediately display updated stats

    # Creating plots without calling stats display
    def create_plot(self, plot_type='scatter'):
        """Unified function to create specific plots without refreshing stats."""
        if not hasattr(self, 'image_directory') or not self.image_directory:
            logger.error("No image directory selected.")
            return

        # Ensure stats are processed, but don't display them
        self.process_labels_and_generate_stats()  # No argument passed here

        # Conditional plot generation based on plot_type
        if plot_type == 'histogram':
            self.plot_histogram()
        elif plot_type == 'bar':
            self.plot_bar()
        elif plot_type == 'scatter':
            self.plot_data()
    def toggle_image_quality_analysis(self, checked):
        """Toggle image quality analysis based on the checkbox."""
        self.image_quality_analysis_enabled = checked

        if checked:
            QMessageBox.warning(self, "Performance Warning",
                                "Enabling image quality analysis may slow down computations significantly.")
        else:
            QMessageBox.information(self, "Disabled",
                                    "Image quality analysis is now disabled. Metrics will show 'N/A'.")

        # Recalculate stats and refresh the UI
        self.update_stats_with_na()



    def update_stats_with_na(self):
        """Updates the stats table to reflect the current image quality analysis setting."""
        stats = self.process_labels_and_generate_stats()  # Get only the stats dictionary

        if not self.image_quality_analysis_enabled:
            # Update stats for image quality metrics with "N/A"
            stats.update({
                'Blurred Images': "N/A",
                'Underexposed Images': "N/A",
                'Overexposed Images': "N/A",
                'Low Contrast Images': "N/A",
            })

        self.settings['stats'] = stats
        self.display_stats()


    def process_labels_and_generate_stats(self):
        stats = {}
        self.all_center_points_and_areas = []
        self.all_label_classes = []

        directory_path = self.image_directory
        if not directory_path:
            logger.error("Image directory is not set.")
            return {}

        # Initialize trackers
        label_counts = defaultdict(int)
        pos_counts = defaultdict(int)
        size_counts = defaultdict(int)

        total_labels = 0
        labeled_images = 0

        smallest_bbox_area = float("inf")
        smallest_bbox_width = smallest_bbox_height = 0
        smallest_image_width = smallest_image_height = 0

        blurred_images = underexposed_images = overexposed_images = low_contrast_images = 0

        txt_files = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith('.txt') and not self.is_placeholder_file(f)
        ]

        class_names = self.load_classes()
        class_id_to_name = {i: name for i, name in enumerate(class_names)}

        image_files = {os.path.splitext(os.path.basename(f))[0]: f for f in self.get_image_files(self.image_directory)}

        for txt_file in txt_files:
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            image_path = image_files.get(base_name)

            if not image_path or not os.path.exists(image_path):
                logger.warning(f"❌ No matching image found for {txt_file}")
                continue

            if self.is_placeholder_file(os.path.basename(image_path)):
                continue

            with Image.open(image_path) as img:
                image_width, image_height = img.size

            with open(txt_file, 'r') as file:
                annotations = [line.strip() for line in file if line.strip()]
                if annotations:
                    labeled_images += 1
                    for line in annotations:
                        parts = line.split()
                        class_id = int(parts[0])

                        if len(parts) == 5:
                            # Bounding box
                            center_x, center_y, width, height = map(float, parts[1:])
                            bbox_width_pixels = width * image_width
                            bbox_height_pixels = height * image_height
                            bbox_area = bbox_width_pixels * bbox_height_pixels

                            # Track smallest bbox
                            if bbox_area < smallest_bbox_area:
                                smallest_bbox_area = bbox_area
                                smallest_bbox_width = bbox_width_pixels
                                smallest_bbox_height = bbox_height_pixels
                                smallest_image_width = image_width
                                smallest_image_height = image_height

                        elif len(parts) > 5 and len(parts) % 2 == 1:
                            # Segmentation
                            points = [
                                (float(parts[i]) * image_width, float(parts[i+1]) * image_height)
                                for i in range(1, len(parts), 2)
                            ]
                            xs, ys = zip(*points)
                            bbox_width_pixels = max(xs) - min(xs)
                            bbox_height_pixels = max(ys) - min(ys)
                            bbox_area = bbox_width_pixels * bbox_height_pixels

                            center_x = (min(xs) + max(xs)) / (2 * image_width)
                            center_y = (min(ys) + max(ys)) / (2 * image_height)

                            # Track smallest bbox
                            if bbox_area < smallest_bbox_area:
                                smallest_bbox_area = bbox_area
                                smallest_bbox_width = bbox_width_pixels
                                smallest_bbox_height = bbox_height_pixels
                                smallest_image_width = image_width
                                smallest_image_height = image_height
                        else:
                            continue

                        # Track general stats
                        self.all_center_points_and_areas.append(((center_x, center_y), bbox_area))
                        self.all_label_classes.append(class_id)
                        label_counts[class_id_to_name.get(class_id, f"Class {class_id}")] += 1
                        total_labels += 1

                        # Positional bias
                        if center_x < 0.33:
                            pos_counts['left'] += 1
                        elif center_x > 0.66:
                            pos_counts['right'] += 1
                        else:
                            pos_counts['center'] += 1

                        if center_y < 0.33:
                            pos_counts['top'] += 1
                        elif center_y > 0.66:
                            pos_counts['bottom'] += 1
                        else:
                            pos_counts['middle'] += 1

                        # Size bias (normalized by image area)
                        normalized_area = bbox_area / (image_width * image_height)
                        if normalized_area < 0.1:
                            size_counts['small'] += 1
                        elif normalized_area < 0.3:
                            size_counts['medium'] += 1
                        else:
                            size_counts['large'] += 1

            # Image quality (optional)
            if self.image_quality_analysis_enabled:
                blur, brightness, contrast = self.analyze_image_quality(image_path)
                blurred_images += blur < 100
                underexposed_images += brightness < 50
                overexposed_images += brightness > 200
                low_contrast_images += contrast < 10

        # Compute final stats
        labeling_progress = (labeled_images / len(txt_files)) * 100 if txt_files else 0
        self.optimal_network_size = self.calculate_optimal_size(
            smallest_bbox_width,
            smallest_bbox_height,
            smallest_image_width,
            smallest_image_height
        )

        stats = {
            'Total Images': len(txt_files),
            'Labeled Images': labeled_images,
            'Unlabeled Images': len(txt_files) - labeled_images,
            'Total Labels': total_labels,
            'Labels per Image (average)': round(total_labels / labeled_images, 1) if labeled_images else 0,
            'Labeling Progress (%)': round(labeling_progress, 1),
            'Blurred Images': blurred_images,
            'Underexposed Images': underexposed_images,
            'Overexposed Images': overexposed_images,
            'Low Contrast Images': low_contrast_images,
            'Class Counts': dict(label_counts),
            'Class Balance Difference': {
                class_name: max(label_counts.values()) - count
                for class_name, count in label_counts.items()
            },
            'Size Bias': dict(size_counts),
            'Smallest BBox (Width)': round(smallest_bbox_width, 2),
            'Smallest BBox (Height)': round(smallest_bbox_height, 2),
            'Smallest Image (Width)': smallest_image_width,
            'Smallest Image (Height)': smallest_image_height,
        }
        stats['Optimal Network Size'] = self.optimal_network_size
        self.settings['stats'] = stats
        return stats


    def calculate_optimal_size(self, smallest_bbox_width, smallest_bbox_height, smallest_image_width, smallest_image_height):
        """
        Calculate the optimal image input size to scale the smallest bbox to 16x16 pixels.
        Ensures dimensions are divisible by 32 for YOLO compatibility.
        """
        logger.info("[🔍] Calculating optimal network size...")

        if smallest_bbox_width <= 0 or smallest_bbox_height <= 0:
            logger.warning("[⚠️] Invalid smallest bbox dimensions — skipping size calculation.")
            return "N/A"

        if smallest_image_width <= 0 or smallest_image_height <= 0:
            logger.warning("[⚠️] Invalid image dimensions — skipping size calculation.")
            return "N/A"

        # Compute scale factors needed to enlarge the smallest box to at least 16x16
        scale_factor_width = 16 / smallest_bbox_width
        scale_factor_height = 16 / smallest_bbox_height
        scale_factor = max(scale_factor_width, scale_factor_height)

        scaled_width = smallest_image_width * scale_factor
        scaled_height = smallest_image_height * scale_factor

        # Round up to nearest multiple of 32 (YOLO likes divisible-by-32 inputs)
        optimal_width = (int(scaled_width) + 31) // 32 * 32
        optimal_height = (int(scaled_height) + 31) // 32 * 32

        # Optional: clamp to a reasonable upper bound
        max_size = 2048
        optimal_width = min(optimal_width, max_size)
        optimal_height = min(optimal_height, max_size)

        logger.info(f"[✅] Optimal input size: Width={optimal_width}, Height={optimal_height}")
        return f"{optimal_width}x{optimal_height}"  # ✅ Width × Height




    def display_stats(self):
        """Display statistics in a GUI with separate tabs."""
        stats = self.settings.get('stats', {})
        if not stats:
            QMessageBox.information(self, 'Labeling Statistics', 'No statistics available.')
            return

        self.stats_widget = QWidget()
        self.stats_widget.setWindowTitle("Labeling Statistics")
        self.stats_widget.resize(800, 600)

        # Create Tab Widget
        tab_widget = QTabWidget()

        # Add Tabs
        general_stats_tab = self.create_general_stats_tab(stats)
        class_counts_tab = self.create_class_counts_tab(stats)


        tab_widget.addTab(general_stats_tab, "General Stats")
        logger.info("Added General Stats tab.")  # Debugging
        tab_widget.addTab(class_counts_tab, "Class Counts")
        logger.info("Added Class Counts tab.")  # Debugging


        # Ensure Tab Widget is added to Layout
        layout = QVBoxLayout()
        layout.addWidget(tab_widget)
        self.stats_widget.setLayout(layout)
        self.stats_widget.show()

    def create_general_stats_tab(self, stats):
        widget = QWidget()
        layout = QVBoxLayout()

        # General Stats Keys - Now includes smallest image size & smallest bbox area
        general_stats_keys = [
            'Total Images', 'Labeled Images', 'Unlabeled Images',
            'Total Labels', 'Labels per Image (average)', 'Labeling Progress (%)',
            'Blurred Images', 'Underexposed Images', 'Overexposed Images', 'Low Contrast Images',
            'Optimal Network Size',
  
        ]

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Statistic", "Value"])

        for key in general_stats_keys:
            item = QStandardItem(key)
            value = QStandardItem(str(stats.get(key, "N/A")))
            model.appendRow([item, value])

        table = QTableView()
        table.setModel(model)
        self.style_table(table)

        layout.addWidget(table)
        widget.setLayout(layout)
        return widget

    def create_class_counts_tab(self, stats):
        widget = QWidget()
        layout = QVBoxLayout()

        # Class Counts
        class_counts = stats.get('Class Counts', {})
        balance_diff = stats.get('Class Balance Difference', {})

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Class", "Count", "Balance Difference"])

        # Order by classes.txt
        classes = self.load_classes()
        for class_name in classes:
            count = class_counts.get(class_name, 0)
            difference = balance_diff.get(class_name, 0)
            model.appendRow([
                QStandardItem(class_name),
                QStandardItem(str(count)),
                QStandardItem(f"{difference:+}")  # Show as + or - difference
            ])

        table = QTableView()
        table.setModel(model)
        self.style_table(table)

        layout.addWidget(table)
        widget.setLayout(layout)

        logger.info("Class Counts tab created successfully.")  # Debugging
        return widget


    def style_table(self, table):
        """Style the table widget dynamically."""
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.setAlternatingRowColors(True)  # Enable alternating row colors
        
        # Ensure table inherits the global stylesheet
        table.setStyleSheet(self.styleSheet())




    def plot_histogram(self):
        """Generate a histogram plot for label areas."""
        if not self.all_center_points_and_areas:
            logger.info("No data available for histogram plot.")
            return

        # Extract the area data from self.all_center_points_and_areas
        areas = [area for _, area in self.all_center_points_and_areas]

        fig, ax = plt.subplots()
        ax.hist(areas, bins=50, color='skyblue', edgecolor='black')
        ax.set_xlabel('Label Area')  # You can add fontdict or labelpad here if needed
        ax.set_ylabel('Count')
        ax.set_title("Distribution of Label Areas")
        plt.show()

    def plot_bar(self):
        """Generate a bar plot showing class distribution, including segmentations."""
        from collections import Counter
        class_counts = Counter(self.all_label_classes)  # 🔥 Includes segmentation classes

        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        fig, ax = plt.subplots()
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
        ax.bar(classes, counts, color=colors)
        ax.set_xlabel('Class')  
        ax.set_ylabel('Count')
        ax.set_title("Label & Segmentation Class Distribution")
        plt.show()


    def plot_data(self):
        """Plot data using scatter plot, including segmentation polygons."""
        if not self.all_center_points_and_areas:
            logger.info("No data available for scatter plot.")
            return

        x, y = zip(*[point[0] for point in self.all_center_points_and_areas])
        areas = [point[1] for point in self.all_center_points_and_areas]
        
        # Normalize point sizes
        sizes = (np.array(areas) - np.min(areas)) / (np.max(areas) - np.min(areas)) * 50 + 10  

        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, c=areas, cmap='viridis', alpha=0.7, s=sizes)
        plt.colorbar(scatter, label='Label & Segmentation Area', orientation='vertical')
        ax.set_title(f"Label Count: {len(self.all_center_points_and_areas)}")
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        plt.show()





    def analyze_image_quality(self, image_path):
        """Analyze blur, brightness, and contrast of an image using CPU."""
        # Load image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.error(f"Failed to load image at {image_path}")
            return None, None, None

        # Calculate blurriness using Sobel filters on the CPU
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        blur_value = (grad_x.var() + grad_y.var()) / 2  # Blurriness score based on gradient variance

        # Calculate brightness and contrast on the CPU
        brightness = np.mean(image)
        contrast = np.std(image)

        return blur_value, brightness, contrast


    def analyze_images_in_directory(self, directory_path):
        """Analyze image quality for all images in a directory using batch processing on the CPU."""
        # List all .jpg images in the directory
        image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.jpg')]

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit analysis tasks for each image
            futures = {executor.submit(self.analyze_image_quality, path): path for path in image_paths}

            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    # Collect the results as (image_path, blur, brightness, contrast)
                    results.append((path, *future.result()))
                except Exception as e:
                    logger.error(f"Exception occurred for {path}: {e}")

        return results





    def initialize_filter_spinbox(self):
        """
        Initialize the filter spinbox for class filtering.
        Includes options for 'All' (-1) and 'Blanks' (-2).
        Only enables spinbox after loading images.
        """
        self.filter_class_spinbox.setMinimum(-5)  
        self.filter_class_spinbox.setMaximum(-1)  # Initially, no classes are loaded
        self.filter_class_spinbox.setEnabled(False)  # Disabled until images are loaded
        self.filter_class_spinbox.valueChanged.connect(self.filter_class)

        # Adjust img_index_number limits dynamically
        self.filter_class_spinbox.valueChanged.connect(
            lambda: self.img_index_number.setMaximum(len(self.filtered_image_files) - 1)
        )


    def update_filter_spinbox(self, classes=None):
        """
        Populate the filter spinbox with class names, but only after an image directory is loaded.
        Log only the total number of classes instead of listing each one.
        """
        if not hasattr(self, "image_directory") or not self.image_directory:
            logger.warning("⚠️ Filter spinbox update skipped: image directory not set.")
            return

        if classes is None:
            classes = self.class_names  # Use stored class names if not provided

        num_classes = len(classes)
        logger.info(f"✅ Adding {num_classes} classes to filter spinbox.")  # Clean single log line

        self.filter_class_spinbox.blockSignals(True)
        self.filter_class_spinbox.clear()

        # Add filter options
        self.filter_class_spinbox.addItem("All (-1)")
        self.filter_class_spinbox.addItem("Blanks (-2)")
        # Add size filters
        self.filter_class_spinbox.addItem("Size: Small (-3)")
        self.filter_class_spinbox.addItem("Size: Medium (-4)")
        self.filter_class_spinbox.addItem("Size: Large (-5)")

        # Add class names (quietly, no log spam)
        for idx, class_name in enumerate(classes):
            self.filter_class_spinbox.addItem(f"{idx}: {class_name}")

        self.filter_class_spinbox.setEnabled(True)
        self.filter_class_spinbox.blockSignals(False)




        
    def on_filter_class_spinbox_changed(self, index):
        """
        Handle changes in the class filter ComboBox.
        """
        current_text = self.filter_class_spinbox.currentText()
        logger.info(f"Filter spinbox changed: {current_text}")

        if current_text.startswith("All"):
            self.filter_class(-1)  # All
        elif current_text.startswith("Blanks"):
            self.filter_class(-2)  # Blanks
        elif current_text.startswith("Size: Small"):
            self.filter_class(-3)  # Small bbox
        elif current_text.startswith("Size: Medium"):
            self.filter_class(-4)  # Medium bbox
        elif current_text.startswith("Size: Large"):
            self.filter_class(-5)  # Large bbox
        else:
            try:
                class_index = int(current_text.split(":")[0])
                self.filter_class(class_index)
            except ValueError:
                logger.error(f"Invalid class index from filter: {current_text}")



    def filter_class(self, filter_index):
        """
        Filter images based on class ID, blanks, or bounding box size (small/medium/large).
        """
        self.filtered_image_files = []
        placeholder_file = 'styles/images/default.png'

        for img_file in self.image_files:
            if img_file == placeholder_file:
                continue

            label_file = os.path.splitext(img_file)[0] + '.txt'
            if not os.path.exists(label_file):
                if filter_index == -2:  # Blanks
                    self.filtered_image_files.append(img_file)
                continue

            with open(label_file, 'r') as file:
                lines = [line.strip() for line in file if line.strip()]

            if not lines:
                if filter_index == -2:  # Blanks
                    self.filtered_image_files.append(img_file)
                continue

            if filter_index == -1:  # All
                self.filtered_image_files.append(img_file)
            elif filter_index >= 0:  # Specific class
                for line in lines:
                    try:
                        class_id = int(line.split()[0])
                        if class_id == filter_index:
                            self.filtered_image_files.append(img_file)
                            break
                    except Exception as e:
                        logger.error(f"Error parsing label in {label_file}: {e}")
                if filter_index in [-3, -4, -5]:
                    try:
                        img_width, img_height = Image.open(img_file).size
                        img_area = img_width * img_height
                        for line in lines:
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            _, cx, cy, w, h = map(float, parts[:5])
                            bbox_area = (w * img_width) * (h * img_height)
                            norm_area = bbox_area / img_area
                            if (
                                (filter_index == -3 and norm_area < 0.1) or
                                (filter_index == -4 and 0.1 <= norm_area < 0.3) or
                                (filter_index == -5 and norm_area >= 0.3)
                            ):
                                self.filtered_image_files.append(img_file)
                                break
                    except Exception as e:
                        logger.error(f"Error calculating size for {img_file}: {e}")


        # Logging & UI update
        logger.info(f"Filter index: {filter_index}")
        logger.info(f"Filtered images count: {len(self.filtered_image_files)}")

        if placeholder_file not in self.filtered_image_files:
            self.filtered_image_files.insert(0, placeholder_file)

        self.total_images.setText(f"Total Images: {len(self.filtered_image_files)}")
        self.update_list_view(self.filtered_image_files)

        if self.filtered_image_files:
            self.current_file = self.filtered_image_files[0]
            self.display_image(self.current_file)
        else:
            logger.info("No images matching the filter criteria.")





    def update_bounding_box_count_display(self):
        # Implementation of the method
        pass

    def clear_class_boxes(self):
        """
        Clears bounding boxes for the selected class.
        If 'All' (-1) is selected, it clears all bounding boxes.
        """
        placeholder_file = 'styles/images/default.png'
        current_text = self.filter_class_spinbox.currentText()
        self.img_index_number_changed(0)  # Reset the image index to 0

        if current_text.startswith("All"):
            class_index = -1
        elif current_text.startswith("Blanks"):
            class_index = -2
        elif current_text.startswith("Size:"):
            QMessageBox.information(self, "Unsupported Operation",
                "Clearing bounding boxes is only supported for actual classes, 'All', or 'Blanks'.")
            return
        else:
            try:
                class_index = int(current_text.split(":")[0])
            except ValueError:
                logger.error(f"Invalid filter value: {current_text}")
                return



        # Process each image in the filtered list.
        for img_file in self.filtered_image_files:
            if img_file == placeholder_file:  # Skip placeholder
                continue

            label_file = os.path.splitext(img_file)[0] + '.txt'
            if os.path.exists(label_file):
                try:
                    with open(label_file, 'r+') as file:
                        lines = file.readlines()
                        file.seek(0)
                        file.truncate()

                        if class_index == -1:
                            # If class -1 (All) is selected, delete all bounding boxes
                            logging.info(f"Clearing ALL bounding boxes for {img_file}")
                            continue  # Skip writing back anything (fully clears file)

                        for line in lines:
                            try:
                                if class_index == -2 or int(line.split()[0]) != class_index:
                                    file.write(line)
                            except Exception as parse_error:
                                logging.error(f"Error parsing line in {label_file}: {parse_error}")
                except Exception as e:
                    logging.error(f"Error processing {label_file}: {e}")

        logging.info(f"Cleared bounding boxes for class index: {class_index}")

        # Immediately refresh the UI so that the changes are reflected.
        self.refresh_bounding_box_display()

    def refresh_bounding_box_display(self):
        """
        Refresh the current image’s bounding boxes in the scene.
        This method re-reads the label file for the currently displayed image
        and updates the scene (or UI elements) accordingly.
        """
        current_label_file = os.path.splitext(self.current_file)[0] + '.txt'
        # Clear the current bounding boxes from the scene.
        for item in self.screen_view.scene().items():
            if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer)):  # ✅ Fixed
                self.screen_view.scene().removeItem(item)

        # If the label file exists, load and display the new boxes.
        if os.path.exists(current_label_file):
            try:
                with open(current_label_file, 'r') as file:
                    lines = file.readlines()
                for line in lines:
                    if line.strip():
                        bbox = BoundingBox.from_str(line.strip())
                        # Create a new BoundingBoxDrawer (or however you render your box)
                        box_item = BoundingBoxDrawer(bbox, self.current_file)
                        self.screen_view.scene().addItem(box_item)
            except Exception as e:
                logging.error(f"Error refreshing bounding boxes from {current_label_file}: {e}")

        # Optionally, update any UI elements that show the count of boxes, etc.
        self.update_bounding_box_count_display()

    


    def move_filtered_images(self):
        """
        Moves filtered images based on the ComboBox-selected class or filter.
        Updates label files so that the selected class is set to index 0.
        Creates a new classes.txt file in the destination folder.
        """
        placeholder_file = 'styles/images/default.png'
        current_text = self.filter_class_spinbox.currentText()

        # Handle special cases:
        if current_text.startswith("Blanks"):
            class_name = "Blanks"
            class_index = -1  # Use -1 as a flag for blanks
        elif current_text.startswith("Size:"):
            class_name = current_text.replace("Size: ", "Size_")
            class_index = -2  # Use -2 as a flag for size-based filtering
        else:
            try:
                class_index = int(current_text.split(":")[0])
                class_name = self.class_names[class_index]
            except ValueError:
                logger.error(f"Invalid class filter format: {current_text}")
                return

        # Create a destination folder
        class_folder = os.path.join(self.image_directory, class_name)
        os.makedirs(class_folder, exist_ok=True)

        for file_path in self.filtered_image_files:
            if file_path == placeholder_file:  # Skip placeholder
                continue

            # Move the image file
            self.move_file(file_path, class_folder)

            # Process the corresponding label file
            txt_file = os.path.splitext(file_path)[0] + '.txt'

            if os.path.exists(txt_file):
                new_txt_file = os.path.join(class_folder, os.path.basename(txt_file))
                with open(txt_file, 'r') as file:
                    lines = file.readlines()

                # Rewrite the labels
                with open(new_txt_file, 'w') as file:
                    for line in lines:
                        parts = line.strip().split()

                        # Handle Blanks
                        if class_index == -1:  # -1 is a flag for Blanks
                            if len(parts) == 0:  # If it's empty, don't write it
                                continue

                        # Handle Size-based filtering (skip writing labels, move only images)
                        elif class_index == -2:
                            file.write(line + '\n')
                            continue

                        # Handle normal class-based filtering
                        elif int(parts[0]) == class_index:
                            parts[0] = '0'  # Set the class index to 0
                            file.write(' '.join(parts) + '\n')

            logger.info(f"Moved images and updated labels for '{class_name}' to {class_folder}.")

        # Create a new classes.txt file in the destination folder
        classes_txt_path = os.path.join(class_folder, 'classes.txt')
        with open(classes_txt_path, 'w') as file:
            file.write(f"{class_name}\n")

        logger.info(f"Created {classes_txt_path} with the updated class mapping.")

        # Refresh the bounding box display
        self.refresh_bounding_box_display()

    def move_file(self, file_path, destination_folder):
        """
        Moves a file to the specified destination folder.
        """
        try:
            os.makedirs(destination_folder, exist_ok=True)
            destination_path = os.path.join(destination_folder, os.path.basename(file_path))
            shutil.move(file_path, destination_path)
            logger.info(f"Moved {file_path} to {destination_path}")
        except Exception as e:
            logger.error(f"Error moving {file_path} to {destination_folder}: {e}")



    def is_file_matching_class_id(self, file_path, class_id):
        """
        Checks if a file matches the given class ID.
        """
        label_file = os.path.splitext(file_path)[0] + '.txt'

        if class_id == -2:  # Check for blanks
            if not os.path.exists(label_file) or not os.path.getsize(label_file):
                return True
            return False

        if os.path.exists(label_file):
            with open(label_file, 'r') as file:
                for line in file:
                    if int(line.split()[0]) == class_id:
                        return True
        return False

    def update_list_view(self, image_files):
        # Filter out the placeholder image explicitly
        filtered_images = [img for img in image_files if os.path.basename(img) != 'default.png']
        logger.info(f"Updating ListView with {len(filtered_images)} files.")

        model = QStandardItemModel()
        for image_file in filtered_images:
            base_file = os.path.basename(image_file)
            item = QStandardItem(base_file)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make items uneditable
            model.appendRow(item)

        self.List_view.setModel(model)
        self.total_images.setText(f"Total Images: {len(filtered_images)}")
        logger.info("ListView updated successfully.")


    def on_list_view_clicked(self, index):
        # Get the item at the clicked index
        item = self.List_view.model().itemFromIndex(index)

        # Ensure the item is selected and not editable
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        file_name = index.data()

        # Get the index of the clicked item in the list
        self.current_img_index = index.row()

        file_path = os.path.join(self.image_directory, file_name)
        self.open_image(file_path)

        # Update img_index_number and display image
        self.img_index_number.setValue(self.current_img_index)  # Update QSpinBox value

        # Call toggle_label_visibility to handle label visibility
        if self.hide_labels:  # If labels are currently hidden, show them
            self.hide_labels = False

        # Scroll to the clicked item to make it visible in the ListView
        self.List_view.scrollTo(index, QAbstractItemView.PositionAtCenter)

        
    def img_index_number_changed(self, value):
        """
        Handles both index-based (if input is numeric) and filename-based image selection.
        Ensures the index does not exceed the filtered image count.
        """
        # Check if the value comes from QSpinBox (int) or QLineEdit (str)
        if isinstance(value, int):  # If value is from QSpinBox (index-based)
            if 0 <= value < len(self.filtered_image_files):
                self.current_img_index = value  # Update the current image index
                self.current_file = self.filtered_image_files[self.current_img_index]
                self.display_image(self.current_file)
            else:
                # Reset to the nearest valid value if out of range
                max_index = max(0, len(self.filtered_image_files) - 1)
                self.current_img_index = max_index
                self.img_index_number.setValue(max_index)
        elif isinstance(value, str):  # If value is from QLineEdit (filename-based)
            value = value.strip()  # Remove any leading/trailing spaces
            matching_files = [f for f in self.filtered_image_files if value in f]
            if matching_files:
                # If a matching file is found, display the first match
                self.current_file = matching_files[0]
                self.current_img_index = self.filtered_image_files.index(self.current_file)
                self.display_image(self.current_file)
            else:
                logger.warning(f"No image found with filename containing: {value}")

                
    def open_image(self, file_path):
        self.display_image(file_path)
        
    def display_all_images(self):
        # Repopulate the list of all image files from the image directory
        self.filtered_image_files = self.get_image_files(self.image_directory)

        # Ensure the placeholder image is included and prioritized
        placeholder_path = 'styles/images/default.png'  # Adjust path as needed
        if placeholder_path not in self.filtered_image_files:
            self.filtered_image_files.insert(0, placeholder_path)

        # Sort images by their corresponding label size (if label files exist)
        def get_label_size(image_path):
            label_path = os.path.splitext(image_path)[0] + '.txt'
            return os.path.getsize(label_path) if os.path.exists(label_path) else 0

        self.filtered_image_files.sort(key=get_label_size, reverse=True)  # Sort by label size, largest first

        # Ensure the placeholder image is still first after sorting
        if placeholder_path in self.filtered_image_files:
            self.filtered_image_files.remove(placeholder_path)
            self.filtered_image_files.insert(0, placeholder_path)

        # Create a new model for the list view and populate it with all image files
        model = QStandardItemModel()
        for img_file in self.filtered_image_files:
            item = QStandardItem(os.path.basename(img_file))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Ensure items are not editable
            model.appendRow(item)
        self.List_view.setModel(model)  # Update the list view model

        # Set the current image to the placeholder or the first image if it exists
        if self.filtered_image_files:
            self.current_file = self.filtered_image_files[0]
            self.display_image(self.current_file)



    def sorted_nicely(self, l):
        """ Sorts the given iterable in the way that humans expect."""
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)


    def get_image_files(self, directory, extensions=None):
        if extensions is None:
            extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}

        image_files = []
        for f in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, f)):
                ext = os.path.splitext(f)[1].lower()
                if ext in extensions and not self.is_placeholder_file(f):
                    image_files.append(os.path.join(directory, f).replace("\\", "/"))
        return image_files



    def preprocess_placeholder_image(self, image_path):
        """
        Preprocess the placeholder image to ensure it's in a valid format.

        Args:
            image_path (str): Path to the placeholder image.

        Returns:
            str: Path to a valid processed image, or None if failed.
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f" Placeholder image not found: {image_path}")
                return None

            with Image.open(image_path) as img:
                # Convert image to RGB format to prevent issues
                img = img.convert("RGB")

                # Generate a safe temporary copy
                temp_path = image_path.replace(".png", "_temp.png")

                # Save a temporary placeholder to avoid corrupting the original
                img.save(temp_path, format="PNG", icc_profile=None)

                logger.info(f" Placeholder image preprocessed and saved: {temp_path}")
                return temp_path

        except Exception as e:
            logger.error(f" Failed to preprocess placeholder image: {e}")
            return None


    def is_placeholder_file(self, file_path):
        """Check if a file is the placeholder image."""
        placeholder_file = 'styles/images/default.png'
        return os.path.basename(file_path) == os.path.basename(placeholder_file)




    def open_image_video(self):
        """
        Open an image directory, load images, initialize classes, and populate views.
        Deduplicate bounding boxes across the entire dataset during loading.
        Uses dotenv to store last directory persistently without interfering with deleted folders.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        # Initialize the progress bar
        self.label_progress.setValue(0)
        self.clear_annotations()

        # 🔹 Retrieve last used directory from .env file
        last_used_directory = os.getenv("ULTRADARKFUSION_LAST_DIR", None)

        # Ensure the directory exists; otherwise, fallback to the current working directory
        if last_used_directory and os.path.exists(last_used_directory):
            start_directory = last_used_directory
        else:
            start_directory = os.getcwd()  # Fallback directory if last directory is invalid

        # Open QFileDialog starting from the last known directory
        dir_name = QFileDialog.getExistingDirectory(None, "Open Image Directory", start_directory, options=options)

        placeholder_image_path = 'styles/images/default.png'  # Adjust this path if needed

        #  Preprocess the placeholder before anything else
        processed_placeholder = self.preprocess_placeholder_image(placeholder_image_path)

        if not dir_name:
            return  # Do nothing if no directory is selected

        # 🔹 Store last used directory in .env file
        self.last_image_directory = dir_name
        set_key(dotenv_path, "ULTRADARKFUSION_LAST_DIR", dir_name)  # Persist directory across restarts

        self.saveSettings()  # Save the settings after modifying them
        self.image_directory = dir_name
        logger.info(f"📂 Image Directory: {self.image_directory}")

        # 🔹 Load classes before anything else
        self.class_names = self.load_classes()
        if not self.class_names:
            logger.warning("⚠️ No classes found in classes.txt. Using default class ['person'].")
            self.class_names = ['person']

        # ✅ Ensure the classes dropdown is initialized correctly before use
        if not hasattr(self, "classes_dropdown"):
            self.classes_dropdown = QComboBox()  # Ensure dropdown is defined

        self.update_classes_dropdown(self.class_names)  # Ensure the dropdown is populated

        #  Load image files in the directory
        self.image_files = self.sorted_nicely(self.get_image_files(dir_name))

        #  Deduplicate bounding boxes
        if not hasattr(self, "dedup_worker") or not self.dedup_worker.isRunning():
            self.deduplicate_dataset(dir_name)
            self.dedup_worker = DeduplicationWorker(
                dir_name,
                self.get_image_files,
                self.remove_near_duplicate_bounding_boxes,
                parent=self  #  Pass the main window here
            )
            self.dedup_worker.progress.connect(self.label_progress.setValue)
            self.dedup_worker.finished.connect(lambda: logger.info("✅ Deduplication completed."))
            self.dedup_worker.start()

        #  Insert placeholder image at the beginning **only if it's valid**
        if processed_placeholder and processed_placeholder not in self.image_files:
            self.image_files.insert(0, processed_placeholder)

        total_images = len(self.image_files)
        if total_images == 0:
            self.total_images.setText("📸 Total Images: 0")
            QMessageBox.warning(None, 'No Images Found', "No image files found in the directory.")
            self.label_progress.setValue(0)  # Reset progress bar if no images found
            return

        # 🔹 Always start from the first image instead of remembering last opened image
        self.current_image_index = 0
        self.current_file = self.image_files[self.current_image_index]

        self.total_images.setText(f"📸 Total Images: {total_images}")
        self.display_image(self.current_file)
        self.initialize_yolo()

        # Populate list view with image files
        self.display_all_images()

        # 🔹 Ensure empty .txt and .json files exist only if needed
        self.create_empty_txt_and_json_files(dir_name, processed_placeholder)

        #  Process labels and calculate stats
        stats = self.process_labels_and_generate_stats()
        self.settings['stats'] = stats

        #  Set progress bar to 100% once loading is complete
        self.label_progress.setValue(100)

    def clear_annotations(self, force_full_reset=False):
        """Clear all annotations and optionally reset graphics scene."""
        if hasattr(self, 'screen_view') and self.screen_view.scene():
            self.screen_view.scene().clear()

        if force_full_reset or not hasattr(self, 'graphics_scene') or self.graphics_scene is None:
            # 💥 FULLY REBUILD
            self.graphics_scene = QGraphicsScene(self.screen_view)
            self.screen_view.setScene(self.graphics_scene)

        self.pixmap_item = None

        # Clear cached annotations
        for attr in ["all_frame_bounding_boxes", "all_frame_segmentations", "all_frame_obbs", "bounding_boxes"]:
            if hasattr(self, attr):
                getattr(self, attr).clear()

        if hasattr(self, "image_label"):
            self.image_label.clear()
        if hasattr(self, "current_file"):
            self.current_file = None
        if hasattr(self, "total_images"):
            self.total_images.setText("📸 Total Images: 0")
        if hasattr(self, "label_progress"):
            self.label_progress.setValue(0)

        logger.info("🧹 Cleared UI and graphics scene.")




    def deduplicate_dataset(self, image_directory, max_workers=None):
        """
        Deduplicates all bounding box, segmentation, and OBB label files in parallel.
        Uses label_progress to update the progress bar safely.
        """
        logger.info("🔁 Deduplicating annotations across the dataset (parallel mode)...")
        txt_files = [os.path.splitext(file)[0] + '.txt' for file in self.get_image_files(image_directory)]
        total_files = len(txt_files)

        if hasattr(self, 'label_progress'):
            self.label_progress.setValue(0)
            self.label_progress.setRange(0, total_files)

        max_workers = max_workers or os.cpu_count()

        completed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.deduplicate_single_label_file, txt_file): txt_file for txt_file in txt_files}

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"❌ Error during parallel deduplication: {e}")

                completed += 1

                if hasattr(self, 'label_progress'):
                    QTimer.singleShot(0, lambda c=completed: self.label_progress.setValue(c))

        logger.info("✅ Deduplication complete.")


    def deduplicate_single_label_file(self, txt_file):

        if not os.path.exists(txt_file):
            return

        try:
            with open(txt_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

            bboxes, obbs, segmentations, preserved_lines = [], [], [], []

            for line in lines:
                if not line.strip():
                    continue  # Skip blank lines

                bbox = BoundingBox.from_str(line)
                if bbox:
                    if bbox.obb:
                        obbs.append(bbox)
                    elif bbox.segmentation:
                        segmentations.append((bbox.class_id, bbox.segmentation))
                    else:
                        bboxes.append(bbox)
                else:
                    logging.warning(f"Skipping malformed line in {txt_file}: '{line}'")


            deduped_bboxes = self.remove_near_duplicate_bounding_boxes(bboxes, class_aware=True)
            deduped_obbs = self.remove_duplicate_obbs(obbs, iou_threshold=0.5)
            deduped_segmentations = self.remove_duplicate_segmentations(segmentations)

            deduped_lines = (
                [b.to_str() for b in deduped_bboxes if b] +
                [o.to_str() for o in deduped_obbs if o] +
                [f"{cid} {' '.join(f'{x:.6f}' for x in seg)}" for cid, seg in deduped_segmentations if seg]
            )

            # Remove any accidentally introduced blank lines and write to file
            with open(txt_file, 'w') as f:
                cleaned_lines = [line.strip() for line in deduped_lines if line.strip()]
                f.write('\n'.join(cleaned_lines) + '\n')

            logger.info(
                f"✅ Deduplicated {txt_file} "
                f"(BBoxes: {len(bboxes)} ➜ {len(deduped_bboxes)}, "
                f"OBBs: {len(obbs)} ➜ {len(deduped_obbs)}, "
                f"Segs: {len(segmentations)} ➜ {len(deduped_segmentations)})"
            )

        except Exception as e:
            logger.error(f"❌ Error deduplicating {txt_file}: {e}")


    def remove_near_duplicate_bounding_boxes(self, bounding_boxes, iou_threshold=0.5, class_aware=False):
        if not bounding_boxes:
            return []

        bounding_boxes = sorted(
            bounding_boxes, 
            key=lambda x: (len(x.keypoints) if x.keypoints else 0, float(x.confidence or 0)), 
            reverse=True
        )

        boxes = np.array([
            [
                bbox.x_center - bbox.width / 2,
                bbox.y_center - bbox.height / 2,
                bbox.x_center + bbox.width / 2,
                bbox.y_center + bbox.height / 2,
                bbox.class_id
            ]
            for bbox in bounding_boxes
        ])

        keep_indices = []
        suppressed = np.zeros(len(boxes), dtype=bool)

        for i in range(len(boxes)):
            if suppressed[i]:
                continue

            keep_indices.append(i)
            x1, y1, x2, y2, cls = boxes[i]
            rest = boxes[i+1:]

            if rest.size == 0:
                break

            xx1 = np.maximum(x1, rest[:, 0])
            yy1 = np.maximum(y1, rest[:, 1])
            xx2 = np.minimum(x2, rest[:, 2])
            yy2 = np.minimum(y2, rest[:, 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            area_i = (x2 - x1) * (y2 - y1)
            area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
            union = area_i + area_rest - inter

            iou = inter / np.maximum(union, 1e-7)

            if class_aware:
                class_mask = rest[:, 4] == cls
            else:
                class_mask = np.ones_like(iou, dtype=bool)

            duplicate_mask = (iou > iou_threshold) & class_mask
            suppressed[i+1:][duplicate_mask] = True

        deduped = [bounding_boxes[i] for i in keep_indices]

        logging.debug(f"Removed {len(bounding_boxes) - len(deduped)} duplicates.")
        return deduped



    def remove_duplicate_segmentations(self, segmentations, iou_threshold=0.95):
        unique_segs = []

        def poly_iou(poly1_flat, poly2_flat):
            try:
                from shapely.geometry import Polygon

                if len(poly1_flat) < 6 or len(poly2_flat) < 6:
                    return 0

                # Reshape from flat [x1, y1, x2, y2, ...] → [(x1, y1), (x2, y2), ...]
                poly1 = list(zip(poly1_flat[::2], poly1_flat[1::2]))
                poly2 = list(zip(poly2_flat[::2], poly2_flat[1::2]))

                p1 = Polygon(poly1)
                p2 = Polygon(poly2)

                if not p1.is_valid or not p2.is_valid or p1.area == 0 or p2.area == 0:
                    return 0

                return p1.intersection(p2).area / p1.union(p2).area

            except Exception as e:
                logger.warning(f"Error computing polygon IoU: {e}")
                return 0

        for current_id, (class_id_a, pts_a) in enumerate(segmentations):
            duplicate = False
            for class_id_b, pts_b in unique_segs:
                iou = poly_iou(pts_a, pts_b)
                if iou > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                unique_segs.append((class_id_a, pts_a))

        return unique_segs


    def remove_duplicate_obbs(self, obbs, iou_threshold=0.5):
        from shapely.geometry import Polygon

        def to_polygon(obb):
            try:
                coords = [(obb.obb[i], obb.obb[i+1]) for i in range(0, 8, 2)]
                return Polygon(coords)
            except Exception as e:
                logger.warning(f"Failed to convert OBB to polygon: {e}")
                return None

        unique = []
        for i, obb in enumerate(obbs):
            p1 = to_polygon(obb)
            if not p1 or not p1.is_valid:
                continue

            is_duplicate = False
            for u in unique:
                p2 = to_polygon(u)
                if not p2 or not p2.is_valid:
                    continue

                inter = p1.intersection(p2).area
                union = p1.union(p2).area
                iou = inter / union if union > 0 else 0

                if iou > iou_threshold and obb.class_id == u.class_id:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(obb)

        return unique


    def create_empty_txt_and_json_files(self, image_directory, placeholder_image_path):
        """
        Ensures that each image has a corresponding empty .txt file and initializes keypoints.json
        if not present. This guarantees a standard point list is always active.
        """
        image_files = self.get_image_files(image_directory)

        # ✅ Define default keypoints list (COCO-style)
        default_keypoints = [
            "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
        ]

        default_json_path = os.path.join(image_directory, "points.json")

        if not os.path.exists(default_json_path):
            keypoints = []
            for i, label in enumerate(default_keypoints):
                color = get_color(i, len(default_keypoints))
                keypoints.append({
                    "label": label,
                    "color": [color.red(), color.green(), color.blue(), 150],
                    "visible": True,
                    "ignore": False
                })

            skeleton = [
                [0, 1],  [0, 2],  [1, 3],  [2, 4],
                [0, 5],  [0, 6],  [5, 7],  [7, 9],
                [6, 8],  [8, 10], [5, 11], [6, 12],
                [11, 13],[13, 15],[12, 14],[14, 16]
            ]

            flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

            json_data = {
                "kpt_shape": [len(keypoints), 3],
                "keypoints": keypoints,
                "skeleton": skeleton,
                "flip_idx": flip_idx
            }

            try:
                with open(default_json_path, "w") as f:
                    json.dump(json_data, f, indent=4, separators=(",", ": "))
                logger.info(f"✅ Created default keypoint list: {default_json_path}")
            except Exception as e:
                logger.error(f"❌ Failed to create keypoint list: {e}")



        # ✅ Load that points.json file into the keypoint list table automatically
        self.load_keypoint_list_from_path(default_json_path)

        for image_file in image_files:
            if os.path.abspath(image_file) == os.path.abspath(placeholder_image_path):
                continue

            label_file, exists = self.get_label_file(image_file, return_existence=True)
            if not exists:
                self.save_labels_to_file(label_file, [], 'w')


                  
    def start_delete_timer(self):
        self.delete_timer.start()

    def stop_delete_timer(self):
        self.delete_timer.stop()

    def delete_current_image(self):
        if self.current_file is None:
            return

        # Ensure the placeholder image is never deleted
        placeholder_path = 'styles/images/default.png'
        if self.current_file == placeholder_path:
            logger.error("Cannot delete the placeholder image.")
            return

        if self.current_file in self.filtered_image_files:
            current_filtered_index = self.filtered_image_files.index(self.current_file)
            self.delete_files(self.current_file)
            self.filtered_image_files.remove(self.current_file)

            if self.current_file in self.image_files:
                self.image_files.remove(self.current_file)

            if len(self.filtered_image_files) > 0:
                if current_filtered_index >= len(self.filtered_image_files):
                    current_filtered_index = len(self.filtered_image_files) - 1
                self.current_file = self.filtered_image_files[current_filtered_index]
                self.display_image(self.current_file)
                self.current_img_index = current_filtered_index
                self.img_index_number.setValue(self.current_img_index)
            else:
                self.current_file = None
                self.current_img_index = -1
                self.display_placeholder()

            self.update_list_view(self.filtered_image_files)

        else:
            if self.current_file in self.image_files:
               logger.warning("The file is being handled as non-filtered and there's no specific action here.")
            else:
                logger.warning("Warning: No image currently loaded.")

    def delete_files(self, file_path):
        # Ensure the protected image is never deleted
        protected_file = 'styles/images/default.png'
        if file_path == protected_file:
            logger.error("Cannot delete the protected image.")
            return

        try:
            os.remove(file_path)  # Delete the image file.
        except FileNotFoundError:
           logger.warning(f"Warning: Image file {file_path} not found.")

        # Assume the label file has the same base name with a .txt extension.
        txt_file_path = os.path.splitext(file_path)[0] + '.txt'
        try:
            os.remove(txt_file_path)  # Delete the associated annotation file.
        except FileNotFoundError:
            logger.warning(f"Warning: Label file {txt_file_path} not found.")


    def is_scene_empty(self):
        return all(isinstance(item, QGraphicsPixmapItem) or isinstance(item, QGraphicsTextItem) for item in self.screen_view.scene().items())

    def closeEvent(self, event):
        if self.label_file:
            self.save_bounding_boxes(self.label_file, self.screen_view.scene(
            ).width(), self.screen_view.scene().height())
        event.accept()

    def closeEvent2(self, event):
        if self.current_file:  # only save if there's a current file
            self.settings['lastImage'] = self.current_file
        self.saveSettings()
        event.accept()

    def on_heatmap_toggle(self):
        self.display_image(self.last_logged_file_name)

    def on_heatmap_colormap_changed(self):
        if self.heatmap_Checkbox.isChecked():
            self.display_image(self.last_logged_file_name)

    def display_image(self, file_name=None, image=None):
        """
        This function handles displaying an image either from a file or a numpy array.
        """
        if file_name is None and image is None:
            logger.warning("File name and image are both None.")
            return None

        if file_name is not None:
            if not isinstance(file_name, str):
                logger.error("Invalid file name type.")
                return None

            if file_name != self.last_logged_file_name:
                logging.info(f"display_image: file_name={file_name}")
                self.last_logged_file_name = file_name

            # Load the image into a NumPy array
            image = cv2.imread(file_name)
            if image is None:
                logger.error(f"Failed to load image: {file_name}. Falling back to placeholder.")
                placeholder_path = 'styles/images/default.png'
                if os.path.exists(placeholder_path):
                    image = cv2.imread(placeholder_path)
                    if image is None:
                        logger.error("Failed to load placeholder image.")
                        return None
                else:
                    logger.warning("Placeholder image missing.")
                    return None

        if image is not None:
            if not isinstance(image, np.ndarray):
                logger.error("Image is not a NumPy array.")
                return None

            # Apply preprocessing
            processed_image, head_labels = self.apply_preprocessing(image)
            self.current_display_image = processed_image.copy()
            self.processed_image = processed_image.copy()
            # Ensure the processed image is in the correct format for QImage
            if len(processed_image.shape) == 2:  # Grayscale
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
            elif len(processed_image.shape) == 3 and processed_image.shape[2] == 4:  # BGRA
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2BGR)

            # Convert the processed NumPy array to a QImage and then to a QPixmap
            image_qimage = self.cv2_to_qimage(processed_image)
            self.image = QPixmap.fromImage(image_qimage)

            # Save the original pixmap size
            self.original_pixmap_size = self.image.size()

            # Create a QGraphicsPixmapItem with the QPixmap
            pixmap_item = QGraphicsPixmapItem(self.image)
            pixmap_item.setTransformationMode(Qt.SmoothTransformation)

            # Create a QGraphicsScene with the exact dimensions of the image
            if self.screen_view.scene():
                self.screen_view.scene().clear()  # Removes all previous items

            scene = QGraphicsScene(0, 0, self.image.width(), self.image.height())

            # Add the QGraphicsPixmapItem to the scene
            scene.addItem(pixmap_item)

            # Set up the scene and rect for screen_view
            self.set_screen_view_scene_and_rect(scene)

            #  Load both bounding boxes and segmentation data
            if file_name:
                label_file = self.replace_extension_with_txt(file_name)
                self.create_empty_file_if_not_exists(label_file)
                self.label_file = label_file

                #  Load both bounding boxes and segmentations
                # Load bounding boxes, segmentations, and OBBs
                bounding_boxes, segmentations, obb_boxes = self.load_labels(
                    label_file, self.image.width(), self.image.height()
                )

                # Pass bounding boxes, segmentations, and OBBs to your display function
                self.display_bounding_boxes(bounding_boxes, file_name, segmentations, obb_boxes)


                #  Force UI update to display segmentations properly
                self.screen_view.scene().update()

                has_labels = any([
                    bounding_boxes,
                    segmentations,
                    obb_boxes and any(len(pts) == 4 for _, pts, _ in obb_boxes)
                ])

                if not has_labels:
                   self.image = QPixmap.fromImage(image_qimage)


            # Synchronize list view selection with the displayed image
            self.sync_list_view_selection(file_name)

        return QPixmap.toImage(self.image)



    def flash_overlay_text(self, text, duration=1500, color=QColor(255, 255, 255), font_size=28):
        scene = self.screen_view.scene()
        if not scene:
            return

        # Remove existing flash text
        for item in scene.items():
            if isinstance(item, QGraphicsTextItem) and item.data(0) == "mode_flash":
                if item.scene():
                    scene.removeItem(item)

        text_item = QGraphicsTextItem(text)
        text_item.setData(0, "mode_flash")
        text_item.setDefaultTextColor(color)
        font = QFont("Arial", font_size, QFont.Bold)
        text_item.setFont(font)

        scene_width = scene.width()
        scene_height = scene.height()
        text_rect = text_item.boundingRect()
        x = (scene_width - text_rect.width()) / 2
        y = (scene_height - text_rect.height()) / 2
        text_item.setPos(x, y)
        text_item.setZValue(100)


        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(25)
        glow.setColor(color.darker(150))
        glow.setOffset(3, 3)
        text_item.setGraphicsEffect(glow)

        text_ref = weakref.ref(text_item)
        scene.addItem(text_item)

        def safely_remove_flash():
            item = text_ref()
            if item is not None and item.scene() is not None:
                try:
                    item.scene().removeItem(item)
                except RuntimeError as e:
                    logger.warning(f"⚠️ Text item already deleted: {e}")

        QTimer.singleShot(duration, safely_remove_flash)



    def set_screen_view_scene_and_rect(self, scene):
        """
        Set the scene and rectangle for the screen_view.
        """
        self.view_references.append(self.screen_view)
        self.screen_view.setScene(scene)
        self.screen_view.fitInView(QRectF(0, 0, self.image.width(), self.image.height()), Qt.KeepAspectRatio)
        self.screen_view.setSceneRect(QRectF(0, 0, self.image.width(), self.image.height()))

        self.graphics_scene = scene  # 💥 CRITICAL: make sure graphics_scene matches what screen_view is using




    def apply_super_resolution(self, file_name):
        """
        Apply super resolution to the image if the corresponding checkbox is checked.
        """
        if self.super_resolution_Checkbox.isChecked() and file_name is not None:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()

            # Dynamically resolve the path to the Sam folder
            base_dir = os.getcwd()  # Current working directory
            path_to_model = os.path.join(base_dir, "Sam", "FSRCNN_x4.pb")

            # Check if the model file exists
            if not os.path.isfile(path_to_model):
                logging.error(f"Error: Model file not found at {path_to_model}.")
                return

            try:
                # Load the model
                sr.readModel(path_to_model)
                sr.setModel("fsrcnn", 4)  # Use "espcn" or "lapsrn" for other models

                # Read the image
                img = cv2.imread(file_name, cv2.IMREAD_COLOR)
                if img is None:
                    logger.error("Error: Image file could not be read.")
                    return

                # Apply super resolution
                img_sr = sr.upsample(img)

                # Convert to QPixmap for display
                height, width, channel = img_sr.shape
                bytesPerLine = 3 * width
                qImg = QImage(img_sr.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                self.image = QPixmap.fromImage(qImg)

            except Exception as e:
                logger.error(f"Failed to apply super resolution. Error: {e}")



    def apply_grayscale(self):
        """
        Convert the image to grayscale if the corresponding checkbox is checked.
        """
        if self.grayscale_Checkbox.isChecked():
            image_qimage = self.image.toImage()
            image_cv = self.qimage_to_cv2(image_qimage)

            if image_cv is None:
                logger.error("Error: Failed to convert QImage to OpenCV format.")
                return

            # Convert the image to grayscale
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

            # Adjust brightness and contrast based on the value of the slider
            value = self.grey_scale_slider.value()
            alpha = value / 50.0
            beta = value
            image_cv = self.adjust_brightness_contrast(image_cv, alpha, beta)

            # 📝 Log grayscale adjustment percentage
            logger.info(f"Grayscale adjustment applied - Slider: {value}/100 → Intensity: {value}%")

            image_qimage = self.cv2_to_qimage(image_cv)

            if image_qimage is None:
                logger.error("Error: Failed to convert OpenCV image back to QImage.")
                return

            # Set the grayscale image as the QPixmap
            self.image = QPixmap.fromImage(image_qimage)



    def apply_edge_detection(self):
        # ✅ Always generate the processed edge image for snapping
        if hasattr(self, 'original_image'):
            image_cv = self.original_image.copy()
        else:
            image_qimage = self.image.toImage()
            image_cv = self.qimage_to_cv2(image_qimage)
            self.original_image = image_cv.copy()

        # Preprocess with Bilateral Filter (preserves edges)
        preprocessed = cv2.bilateralFilter(image_cv, 9, 75, 75)

        # ✅ First, run Canny for snapping (keep it RAW, not cleaned)
        edges_for_snapping = cv2.Canny(preprocessed, self.slider_min_value, self.slider_max_value)

        # ✅ Optional: if you want to keep the noise reduced for visuals
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges_clean = cv2.morphologyEx(edges_for_snapping, cv2.MORPH_CLOSE, kernel)

        # ✅ Use the raw edges for snapping logic
        self.processed_image = edges_for_snapping

        # ✅ Debug display mode only (optional visualization)
        if getattr(self, 'debug_edge_display_checkbox', None) and self.debug_edge_display_checkbox.isChecked():
            edges_colored = cv2.cvtColor(edges_clean, cv2.COLOR_GRAY2BGR)
            edges_colored[np.where((edges_colored != [0, 0, 0]).all(axis=2))] = [0, 255, 0]  # green edges

            image_overlayed = cv2.addWeighted(image_cv, 0.8, edges_colored, 0.2, 0)
            image_overlayed = cv2.cvtColor(image_overlayed, cv2.COLOR_BGR2RGBA)
            image_qimage = self.cv2_to_qimage(image_overlayed)

            if image_qimage:
                self.image = QPixmap.fromImage(image_qimage)
            else:
                logger.error("Failed to convert OpenCV image back to QImage.")

        else:
            # ✅ Always restore original clean image to display
            image_cv_rgba = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGBA)
            image_qimage = self.cv2_to_qimage(image_cv_rgba)

            if image_qimage:
                self.image = QPixmap.fromImage(image_qimage)
            else:
                logger.error("Failed to restore original image back to QImage.")




    def checkbox_clicked(self):
        self.display_image(self.current_file)



    def edge_slider_changed(self):
        self.slider_min_value = self.edge_slider_min.value()
        self.slider_max_value = self.edge_slider_max.value()

        if self.slider_min_value > self.slider_max_value:
            self.slider_min_value, self.slider_max_value = self.slider_max_value, self.slider_min_value  # swap if needed

        logging.info(f"🎚️ Min: {self.slider_min_value}, Max: {self.slider_max_value}")
        self.display_image(self.current_file)

    # Conversion utilities
    def qimage_to_cv2(self, img):
        img = img.convertToFormat(4)
        width, height = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


    def cv2_to_qimage(self, img):
        """
        Convert OpenCV image (NumPy array) to QImage.
        Handles both color and grayscale images.
        """
        if img is None:
            logger.error("Error: Input image is None.")
            return None

        if isinstance(img, cv2.UMat):
            img = img.get()

        # Ensure the image data is contiguous
        img = np.ascontiguousarray(img)

        if len(img.shape) == 2:
            # Grayscale image
            height, width = img.shape
            bytesPerLine = width
            return QImage(img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

        elif len(img.shape) == 3:
            # Color image
            height, width, channels = img.shape
            bytesPerLine = channels * width
            if channels == 3:
                return QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            elif channels == 4:
                return QImage(img.data, width, height, bytesPerLine, QImage.Format_ARGB32)

        logger.error(f"Error: Unsupported image format with {len(img.shape)} dimensions.")
        return None


    def adjust_brightness_contrast(self, image_cv, alpha, beta):
        return cv2.convertScaleAbs(image_cv, alpha=alpha, beta=beta)   

    def replace_extension_with_txt(self, file_name):
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
            file_name = file_name.replace(ext, ".txt")
        return file_name


    def display_image_with_text(self, scene, pixmap):
        # Set a high-contrast background color for the scene
        scene.setBackgroundBrush(QBrush(QColor(50, 50, 50)))  # Dark grey background for contrast

        # Add a thick glowing border around the pixmap using QGraphicsRectItem
        border_item = QGraphicsRectItem(0, 0, pixmap.width(), pixmap.height())
        border_item.setPen(QPen(QColor(0, 255, 0, 255), 10))  # Thick green border
        border_glow_effect = QGraphicsDropShadowEffect()
        border_glow_effect.setBlurRadius(30)
        border_glow_effect.setColor(QColor(0, 255, 0, 150))
        border_item.setGraphicsEffect(border_glow_effect)
        scene.addItem(border_item)

        # Overlay a semi-transparent background to further distinguish "BLANK" images
        overlay = QGraphicsRectItem(0, 0, pixmap.width(), pixmap.height())
        overlay.setBrush(QColor(0, 0, 0, 100))  # Semi-transparent black overlay
        overlay.setZValue(1)  # Ensure overlay is below the text
        scene.addItem(overlay)

        # Display the "BLANK" text prominently
        empty_text_item = QGraphicsTextItem("BLANK")
        empty_text_item.setDefaultTextColor(QColor(255, 0, 0))  # Bright red for high visibility

        # Set a bold, futuristic font
        font = QFont("Arial", 24, QFont.Bold)
        empty_text_item.setFont(font)

        # Position the text at the bottom center of the pixmap
        text_width = empty_text_item.boundingRect().width()
        text_height = empty_text_item.boundingRect().height()
        empty_text_item.setPos((pixmap.width() - text_width) / 2, pixmap.height() - text_height - 10)  # 10px margin from bottom

        # Create a drop shadow effect with pulsating glow
        effect = QGraphicsDropShadowEffect()
        effect.setOffset(5, 5)
        effect.setBlurRadius(35)
        effect.setColor(QColor(255, 0, 0, 200))  # Strong red shadow
        empty_text_item.setGraphicsEffect(effect)

        # Add the text to the scene on top of the overlay
        empty_text_item.setZValue(2)
        scene.addItem(empty_text_item)

        # Adjust the scene to fit the pixmap size
        scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

        # Ensure the animation is running
        animation = QPropertyAnimation(border_glow_effect, b"blurRadius")
        animation.setDuration(1000)
        animation.setStartValue(30)
        animation.setEndValue(50)
        animation.setLoopCount(-1)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        animation.start()

        # Optional: Store the animation reference to keep it alive
        self.animation = animation

        # Store the initial border color (if needed)
        self.initial_border_color = QColor(255, 255, 255, 255)




    def clear_empty_bounding_box_label(self):
        for item in self.screen_view.scene().items():
            if isinstance(item, QGraphicsTextItem):
                self.screen_view.scene().removeItem(item)
                del item
                break

    def create_empty_file_if_not_exists(self, file_path):
        if not os.path.isfile(file_path):
            with open(file_path, "w"):
                pass

    def remove_bbox_from_classes_txt(self, bbox_item):
        rects = [item for item in self.screen_view.scene(
        ).items() if isinstance(item, BoundingBoxDrawer)]
        found_bbox = self.find_bbox_in_rects(bbox_item)
        if found_bbox:
            rects.remove(found_bbox)
        else:
           logger.warning(f"bbox_item {bbox_item} not found in rects")
        self.save_bounding_boxes(self.current_file, self.screen_view.scene().width(), self.screen_view.scene().height())


    def create_bounding_box(self, x, y, w, h, label, confidence):
        """
        Create a bounding box and add it to the scene.
        """
        new_index = self.classes_dropdown.findText(label)
        if new_index != -1:
            self.classes_dropdown.setCurrentIndex(new_index)
            bbox_drawer = BoundingBoxDrawer(x, y, w, h, self, class_id=new_index, confidence=confidence)
            self.screen_view.scene().addItem(bbox_drawer)

            # Ensure label_file is initialized
            label_file = self.replace_extension_with_txt(self.current_file)
            if not label_file:
                logger.error("Error: Unable to determine the label file path.")
                return

            # Save bounding boxes without generating head labels here
            self.save_bounding_boxes(label_file, self.screen_view.scene().width(), self.screen_view.scene().height())
        else:
            logger.warning(f"Warning: '{label}' not found in classes dropdown.")




    def get_head_class_id(self):
        """
        Retrieve the class ID for head labels.
        This can be hardcoded or loaded dynamically based on your class labels.
        """
        head_class_name = "head"  # Replace with your actual head class name
        self.class_labels = self.load_classes()
        if head_class_name in self.class_labels:
            return self.class_labels.index(head_class_name)

        else:
            logging.error(f"Head class '{head_class_name}' not found in class labels.")
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
        if not self.heads_area.isChecked() or class_id == 1:
            return None

        head_class_id = self.get_head_class_id()
        if head_class_id is None:
            logging.warning("Skipping head labels due to missing head class ID.")
            return None

        # Ensure `box` is absolute (x1, y1, x2, y2) and not YOLO format
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        # Calculate the head bounding box in absolute coordinates
        head_x, head_y, head_w, head_h = self.calculate_head_area(x1, y1, w, h)

        #  **Normalize coordinates for YOLO format (relative to img size)**
        head_xc = (head_x + head_w / 2) / img_width
        head_yc = (head_y + head_h / 2) / img_height
        head_w /= img_width
        head_h /= img_height

        #  **Return YOLO-formatted label**
        return f"{head_class_id} {head_xc:.6f} {head_yc:.6f} {head_w:.6f} {head_h:.6f}"

        
        
    def apply_preprocessing(self, image, bounding_boxes=None, img_width=None, img_height=None):
        # ✅ Sanitize input image
        if image is None or image.size == 0:
            logging.warning("⛔ Image is empty. Skipping preprocessing.")
            return image, []

        # 🔧 Optional: grayscale
        if self.grayscale_Checkbox.isChecked():
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 🔧 Super-resolution (must be before color maps or overlays)
        if self.super_resolution_Checkbox.isChecked():
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = os.path.join(os.getcwd(), "Sam", "FSRCNN_x4.pb")
            if os.path.isfile(model_path):
                try:
                    sr.readModel(model_path)
                    sr.setModel("fsrcnn", 4)

                    if image.dtype != np.uint8:
                        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    if len(image.shape) != 3 or image.shape[2] != 3:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    image = sr.upsample(image)
                except Exception as e:
                    logging.error(f"Super-resolution failed: {e}")
            else:
                logging.warning(f"Model file not found at {model_path}. Skipping super-resolution.")

        # 🔧 Optional: edge outline
        if self.outline_Checkbox.isChecked():
            edges = cv2.Canny(image, self.slider_min_value, self.slider_max_value)
            self.processed_image = edges.copy()
            if getattr(self, 'debug_edge_display_checkbox', None) and self.debug_edge_display_checkbox.isChecked():
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                image = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

        # 🔧 Optional: heatmap last (for display only)
        if self.heatmap_Checkbox.isChecked():
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image_gray.dtype != np.uint8:
                image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            selected_map_name = self.heatmap_dropdown.currentText().strip().upper()
            colormap = self.heatmap_colormap_options.get(selected_map_name, cv2.COLORMAP_JET)
            image = cv2.applyColorMap(image_gray, colormap)

        return image, []



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



    def update_yolo_label_file(self, new_class_id):
        try:
            logger.info(f"Attempting to update YOLO label file at {self.label_file}")  # Debugging line
            with open(self.label_file, "r") as f:
                yolo_labels = f.readlines()

            found = False  # To check if a matching bounding box is found
            for i, line in enumerate(yolo_labels):
                bbox = BoundingBox.from_str(line)
                if bbox.is_same(self.selected_bbox):
                    found = True
                    logger.info("Matching bounding box found. Updating...")  # Debugging line
                    bbox.class_id = new_class_id
                    yolo_labels[i] = bbox.to_str()

            if not found:
                logger.warning("No matching bounding box found.")  # Debugging line

            with open(self.label_file, "w") as f:
                f.writelines(yolo_labels)

            logger.info("Successfully updated YOLO label file.")  # Debugging line
        except Exception as e:
            logger.error(f"An error occurred: {e}")  # Basic error handling

    def find_bbox_in_rects(self, bbox_item):
        for item in self.screen_view.scene().items():
            if isinstance(item, BoundingBoxDrawer):
                if (
                    item.rect().x() == bbox_item.rect().x()
                    and item.rect().y() == bbox_item.rect().y()
                    and item.rect().width() == bbox_item.rect().width()
                    and item.rect().height() == bbox_item.rect().height()
                ):
                    return item
        return None



    def toggle_label_visibility(self):
        """
        Toggles the visibility of class name labels for both bounding boxes and segmentation masks.
        """
        self.hide_labels = self.hide_label_checkbox.isChecked()
        scene = self.screen_view.scene()
        
        if scene is not None:
            for item in scene.items():
                # Handle bounding box labels
                if isinstance(item, BoundingBoxDrawer):
                    for child in item.childItems():
                        if isinstance(child, QGraphicsTextItem):
                            should_display = not self.hide_labels
                            child.setVisible(should_display)
                            item.labels_hidden = not should_display  
                            item.update()  

                # Handle segmentation labels
                elif isinstance(item, SegmentationDrawer):
                    if hasattr(item, "class_name_item"):  # Ensure segmentation has a label
                        item.class_name_item.setVisible(not self.hide_labels)



    def change_class_id(self, index):
        """
        Change the class ID of the selected bounding box safely.
        """
        if not hasattr(self.screen_view, 'selected_bbox') or self.screen_view.selected_bbox is None:
            logger.warning("⚠️ No bounding box selected.")
            return  

        if sip.isdeleted(self.screen_view.selected_bbox):
            logger.warning("⚠️ Selected bounding box was deleted.")
            return  

        self.screen_view.selected_bbox.set_class_id(index)

        class_name = self.classes_dropdown.itemText(index)  # 🔥 Fixed this line
        self.update_yolo_label_file(class_name)
        self.update_bbox_data()
        self.current_class_id = index


    def get_current_class_id(self):
        return self.classes_dropdown.currentIndex()

    @pyqtSlot()
    def clear_classes_dropdown(self):
        self.classes_dropdown.clear()

    @pyqtSlot(str)
    def add_item_to_classes_dropdown(self, item_text):
        """
        Add an item to the classes dropdown and ensure it is saved in the classes.txt file.
        """
        if not item_text:
            return  # Ignore empty strings

        #  Add to dropdown
        self.classes_dropdown.addItem(item_text)

        #  Determine correct directory (image or video mode)
        directory = self.image_directory if self.image_directory else self.output_path or os.getcwd()
        classes_file_path = os.path.join(directory, "classes.txt")

        #  Load existing classes
        existing_classes = self.load_classes(data_directory=directory)

        #  Only save if it's a new class
        if item_text not in existing_classes:
            try:
                with open(classes_file_path, "a") as f:
                    f.write(f"{item_text}\n")  # Append new class to file
                logger.info(f"Class '{item_text}' added to {classes_file_path}")
            except Exception as e:
                logger.error(f"Error writing to {classes_file_path}: {e}")


    def get_next_available_class_id(self):
        """
        Get the next available class ID based on the classes loaded from the classes.txt file.

        Returns:
            int: The next available class ID (default 0 if no classes are found).
        """
        # Load classes using the `load_classes` function
        class_labels = self.load_classes()

        # Check if classes are available
        if class_labels:
            # Convert class names to indices and find the next available ID
            class_ids = range(len(class_labels))
            return max(class_ids) + 1

        # Default to 0 if no classes are found
        return 0
    def update_classes_dropdown(self, classes):
        """
        Update the classes dropdown with a list of classes and synchronize the filter spinbox.
        """
        self.classes_dropdown.clear()  # Clear existing items
        self.class_visibility = {cls: True for cls in classes}  # Default visibility: True

        # Populate the dropdown with class names
        model = QStandardItemModel(self.classes_dropdown)
        for cls in classes:
            item = QStandardItem(cls)
            item.setCheckable(True)
            item.setCheckState(Qt.Checked)  # Default: Checked (visible)
            model.appendRow(item)

        self.classes_dropdown.setModel(model)
        self.classes_dropdown.setItemDelegate(CheckboxDelegate(self.classes_dropdown))

        # Sync the filter spinbox with the dropdown
        self.update_filter_spinbox(classes)  # 🔹 Ensure sync here

        # Connect changes to toggle_class_visibility
        model.itemChanged.connect(self.handle_class_visibility_change)

        logger.info(f"📌 Classes dropdown updated. Loaded {len(classes)} classes.")



    def handle_class_visibility_change(self, item):
        """
        Updates visibility for a class based on checkbox state.
        This applies to all bounding boxes across all frames.
        """
        class_name = item.text()
        self.class_visibility[class_name] = item.checkState() == Qt.Checked

        # Update visibility of bounding boxes in the current frame
        self.update_bbox_visibility()

        # Apply visibility to all loaded frames
        for file, bounding_boxes in self.all_frame_bounding_boxes.items():
            for bbox_tuple in bounding_boxes:
                rect, class_id, confidence = bbox_tuple  # Unpack the tuple
                if self.classes_dropdown.itemText(class_id) == class_name:
                    # Update visibility for the bounding box
                    bbox_tuple = (rect, class_id, confidence)  # Modify as needed
                    # Here, visibility would be applied when the bounding boxes are redrawn or used.
                    # If you store additional metadata, adjust accordingly.

        logger.info(f"Visibility for class '{class_name}' set to {self.class_visibility[class_name]}")


    def update_bbox_visibility(self):
        """
        Updates visibility of bounding boxes and segmentation masks in the current frame
        based on the stored class visibility settings.
        """
        for item in self.screen_view.scene().items():
            if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer)):  # Include SegmentationDrawer
                class_name = self.classes_dropdown.itemText(item.class_id)
                item.setVisible(self.class_visibility.get(class_name, True))




    def class_input_field_return_pressed(self):
        """
        Handle the return key press in the class input field to add a new class to classes.txt
        and update the dropdown menu.
        """
        #  Use correct directory for image or video mode
        directory = self.image_directory if self.image_directory else self.output_path or os.getcwd()

        if directory is None:
            logger.error("Error: No valid directory for saving classes.")
            return

        new_class = self.class_input_field.text().strip()
        if not new_class:
            return  # Do nothing if input is empty

        #  Load existing classes
        existing_classes = self.load_classes(data_directory=directory)

        if not isinstance(existing_classes, list):
            logger.error("Error: `load_classes` did not return a valid list.")
            return

        #  Check if class already exists
        if new_class in existing_classes:
            index = self.classes_dropdown.findText(new_class)
            if index >= 0:
                self.classes_dropdown.setCurrentIndex(index)
                logger.warning(f"Class '{new_class}' already exists. Selected in dropdown.")
            return

        #  Append new class to `classes.txt`
        classes_file_path = os.path.join(directory, "classes.txt")
        try:
            with open(classes_file_path, "a") as f:
                f.write(f"{new_class}\n")
        except Exception as e:
            logger.error(f"Error writing to {classes_file_path}: {e}")
            return

        #  Clear input field
        self.class_input_field.clear()

        #  Update dropdown and select new class
        updated_classes = existing_classes + [new_class]
        self.update_classes_dropdown(updated_classes)
        self.classes_dropdown.setCurrentIndex(self.classes_dropdown.count() - 1)

        logger.info(f"New class '{new_class}' added successfully.")

       
    #function to remove classes from classes.txt 
    def remove_class_button_clicked(self):
        """Remove the selected class from `classes.txt` and update the dropdown."""
        selected_class = self.classes_dropdown.currentText()
        
        if not selected_class:
            QMessageBox.warning(self, "Warning", "No class selected.")
            return

        #  Use correct directory for image or video mode
        directory = self.image_directory if self.image_directory else self.output_path or os.getcwd()
        classes_file_path = os.path.join(directory, "classes.txt")

        try:
            #  Load existing classes
            classes = self.load_classes(data_directory=directory)

            if selected_class in classes:
                classes.remove(selected_class)

                #  Rewrite `classes.txt` without the removed class
                with open(classes_file_path, "w") as f:
                    for cls in classes:
                        f.write(f"{cls}\n")

                #  Update dropdown
                self.update_classes_dropdown(classes)
                QMessageBox.information(self, "Information", f"Class '{selected_class}' removed.")
            else:
                QMessageBox.warning(self, "Warning", f"Class '{selected_class}' does not exist.")
        except FileNotFoundError:
            QMessageBox.warning(self, "Warning", "Classes file not found.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error removing class: {e}")


    # loading creating saving updating bboxes 
    
    
    def load_bounding_boxes(self, label_file, img_width, img_height):
        bounding_boxes = []

        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        bbox = BoundingBox.from_str(line)
                        if bbox:
                            bounding_boxes.append(bbox)

        # Optional JSON confidence scores
        label_file_json = label_file.replace('.txt', '.json')
        if os.path.exists(label_file_json) and os.path.getsize(label_file_json) > 0:
            try:
                with open(label_file_json, "r") as f:
                    data = json.load(f)
                annotations = data.get('annotations', []) if isinstance(data, dict) else data
            except json.JSONDecodeError:
                logger.error(f"Error reading JSON file: {label_file_json}")
                annotations = []
        else:
            annotations = [{} for _ in bounding_boxes]

        for bbox, annotation in zip(bounding_boxes, annotations):
            bbox.confidence = annotation.get('confidence', 0)

        return bounding_boxes  # Return full objects, not tuples



    def update_current_bbox_class(self):
        try:
            if self.selected_bbox is not None:
                # Get the index of the selected class from the dropdown
                new_class_id = self.classes_dropdown.currentIndex()

                # Update the class ID of the selected bounding box
                self.selected_bbox.set_class_id(new_class_id)

                # Update the YOLO label file with the new class ID
                self.save_bounding_boxes(self.current_file, self.screen_view.scene().width(), self.screen_view.scene().height(), remove_confidence=False)


                # Find the corresponding graphical representation of the selected bbox
                bbox_drawer_item = self.find_bbox_in_rects(self.selected_bbox)

                if bbox_drawer_item is not None:
                    # Get the color associated with the new class ID, default to white if not found
                    new_color = self.class_color_map.get(new_class_id, QColor(255, 255, 255))

                    # Set the pen color of the graphical bbox representation
                    bbox_drawer_item.setPen(QPen(new_color, 2))
                else:
                    # Handle the case where the graphical representation is not found
                    pass
            else:
                # Handle the case where there is no selected bbox
                pass
        except Exception as e:
            # Basic error handling: log the error
            logger.error(f"An error occurred: {e}")
            
    def get_label_file(self, image_file, return_existence=False):
        """
        Returns the label file path. If return_existence is True, also returns a boolean indicating whether the file exists.
        """
        if image_file is None:
            return (None, False) if return_existence else None

        try:
            base = os.path.basename(image_file)  # Get "image.jpg"
            name = os.path.splitext(base)[0]     # Get "image"

            # Get directory of the current file
            directory = os.path.dirname(image_file)

            # Construct the label filename
            label_filename = name + ".txt"
            label_file = os.path.join(directory, label_filename)

            if return_existence:
                label_exists = os.path.exists(label_file)
                return label_file, label_exists
            else:
                return label_file
        except Exception as e:
            logger.error(f"Error while getting label file: {e}")
            return (None, False) if return_existence else None   

    def display_bounding_boxes(self, rects, file_name, segmentations=None, obbs=None):
        scene = self.screen_view.scene()

        # 🟩 Display Bounding Boxes (and Keypoints)
        for index, rect_tuple in enumerate(rects):
            if len(rect_tuple) == 4:
                rect, class_id, confidence, keypoints = rect_tuple

                is_keypoint_data = (
                    isinstance(keypoints, list) and
                    all(isinstance(k, (list, tuple)) and len(k) == 3 for k in keypoints)
                )
                if not is_keypoint_data:
                    logger.warning(f"⚠️ Ignoring suspicious keypoints for {file_name}: {keypoints}")
                    keypoints = None

            elif len(rect_tuple) == 3:
                rect, class_id, confidence = rect_tuple
                keypoints = None
            elif len(rect_tuple) == 2:
                rect, class_id = rect_tuple
                confidence = None
                keypoints = None
            else:
                logger.warning(f"⚠️ Invalid bbox format in: {rect_tuple}")
                continue

            unique_id = f"{file_name}_{index}"

            rect_item = BoundingBoxDrawer(
                rect.x(), rect.y(), rect.width(), rect.height(),
                unique_id=unique_id,
                class_id=class_id,
                main_window=self,
                confidence=confidence
            )
            rect_item.file_name = file_name
            rect_item.setAcceptHoverEvents(True)
            rect_item.set_z_order(bring_to_front=False)

            if keypoints:
                points = [(x, y) for x, y, _ in keypoints]
                flags = [int(v) for _, _, v in keypoints]

                keypoint_drawer = KeypointDrawer(
                    points=points,
                    visibility_flags=flags,
                    main_window=self,
                    class_id=class_id,
                    file_name=file_name
                )

                rect_item.keypoint_drawer = keypoint_drawer
                scene.addItem(keypoint_drawer)

            scene.addItem(rect_item)
            self.bounding_boxes[f"{file_name}_{unique_id}"] = rect_item

        # 🔷 Display Segmentations
        if segmentations:
            for index, (class_id, points) in enumerate(segmentations):
                if not points:
                    continue
                unique_id = f"{file_name}_seg_{index}"
                seg_item = SegmentationDrawer(
                    main_window=self,
                    points=points,
                    class_id=class_id,
                    unique_id=unique_id
                )
                seg_item.setAcceptHoverEvents(True)
                seg_item.setZValue(0.2)
                seg_item.update_opacity()
                scene.addItem(seg_item)
                self.bounding_boxes[unique_id] = seg_item

        # 🟨 Display Oriented Bounding Boxes (OBBs)
        if obbs:
            for index, (class_id, obb_points, angle) in enumerate(obbs):
                unique_id = f"{file_name}_obb_{index}"
                obb_item = OBBDrawer(
                    obb_points=obb_points,
                    angle=angle,  # ✅ <--- this is the critical missing line
                    main_window=self,
                    class_id=class_id,
                    unique_id=unique_id,
                    file_name=file_name
                )

                obb_item.setAcceptHoverEvents(True)
                obb_item.setZValue(0.3)
                obb_item.update_opacity()
                scene.addItem(obb_item)
                self.bounding_boxes[unique_id] = obb_item




    def auto_save_bounding_boxes(self):
        """
        Automatically saves bounding boxes, segmentations, keypoints, and oriented bounding boxes (OBBs)
        with full precision, preventing premature saves if keypoint drawers are not fully initialized.
        """
        if self.current_file and not self.is_placeholder_file(self.current_file):
            scene = self.screen_view.scene()
            rects = [item for item in scene.items() if isinstance(item, BoundingBoxDrawer)]
            segmentations = [item for item in scene.items() if isinstance(item, SegmentationDrawer)]
            obbs = [item for item in scene.items() if isinstance(item, OBBDrawer)]

            if not rects and not segmentations and not obbs:
                return  # Silent skip if nothing to save

            total_kpts = self.keypoint_list.rowCount()
            for rect in rects:
                drawer = getattr(rect, "keypoint_drawer", None)
                if drawer and len(drawer.visibility_flags) < total_kpts:
                    logger.debug(f"Skipping auto-save: keypoints not fully initialized for class ID {rect.class_id}")
                    return

            # Save without logging unless there's an error
            try:
                self.save_bounding_boxes(self.current_file, scene.width(), scene.height(), log_save=False)
            except Exception as e:
                logger.error(f"❌ Failed auto-save for {self.current_file}: {e}")




    def save_bounding_boxes(self, image_file, img_width, img_height, scene=None, remove_confidence=True, extra_labels=None, log_save=True):
        """
        Save bounding boxes, keypoints, segmentations, and oriented bounding boxes (OBBs) without losing precision.

        Args:
            image_file (str): The image file associated with the bounding boxes.
            img_width (int): Width of the image.
            img_height (int): Height of the image.
            scene (QGraphicsScene, optional): The scene containing objects. Defaults to None.
            remove_confidence (bool, optional): Whether to remove confidence values. Defaults to True.
            extra_labels (list, optional): Extra labels to append. Defaults to None.
            log_save (bool, optional): Whether to log save events. Defaults to True.
        """
        if self.is_placeholder_file(image_file):
            return

        label_file = self.get_label_file(image_file)
        if not label_file:
            logger.error(f"❌ Could not determine label file for image: {image_file}")
            return

        scene = scene or self.screen_view.scene()

        bbox_lines = []
        keypoint_lines = []
        segmentation_lines = []
        obb_lines = []

        for item in scene.items():

            if isinstance(item, BoundingBoxDrawer) and not isinstance(item, SegmentationDrawer):
                if not hasattr(item, "keypoint_drawer") or not item.keypoint_drawer:
                    bbox = BoundingBox.from_rect(
                        QRectF(item.rect().x(), item.rect().y(), item.rect().width(), item.rect().height()),
                        img_width, img_height,
                        item.class_id,
                        item.confidence
                    )
                    bbox_str = bbox.to_str(remove_confidence=remove_confidence).strip()
                    if bbox_str:
                        bbox_lines.append(bbox_str)

            if isinstance(item, BoundingBoxDrawer) and hasattr(item, "keypoint_drawer") and item.keypoint_drawer:
                bbox = BoundingBox.from_rect(
                    QRectF(item.rect().x(), item.rect().y(), item.rect().width(), item.rect().height()),
                    img_width, img_height,
                    item.class_id,
                    item.confidence
                )
                drawer = item.keypoint_drawer
                keypoint_str = drawer.get_keypoint_string().strip()

                if keypoint_str:
                    keypoint_line = (
                        f"{item.class_id} "
                        f"{format(bbox.x_center, '.16f')} {format(bbox.y_center, '.16f')} "
                        f"{format(bbox.width, '.16f')} {format(bbox.height, '.16f')} "
                        f"{keypoint_str}"
                    )
                    keypoint_lines.append(keypoint_line.strip())

            if isinstance(item, SegmentationDrawer) and hasattr(item, "points") and item.points:
                points_str = " ".join(f"{format(x, '.16f')} {format(y, '.16f')}" for x, y in item.points)
                segmentation_lines.append(f"{item.class_id} {points_str}")

            if isinstance(item, OBBDrawer) and hasattr(item, "obb_points") and item.obb_points:
                obb_str = " ".join(f"{format(x, '.16f')} {format(y, '.16f')}" for x, y in item.obb_points)
                obb_lines.append(f"{item.class_id} {obb_str}")


        all_labels = bbox_lines + keypoint_lines + segmentation_lines + obb_lines
        if extra_labels:
            all_labels += [label.strip() for label in extra_labels if label.strip()]

        self.save_labels_to_file(label_file, all_labels, mode="w")

        if log_save:
            logger.info(f"✅ Saved {len(bbox_lines)} bboxes, {len(keypoint_lines)} keypoints, {len(segmentation_lines)} segmentations, and {len(obb_lines)} OBBs → {label_file}")
            logger.debug(f"📝 Label output preview for {label_file}:\n" + "\n".join(all_labels))




    def load_labels(self, label_file, img_width, img_height):
        """
        Loads bounding boxes, segmentations, keypoints, and oriented bounding boxes (OBBs) from a label file without clipping 
        or modifying the original points.
        """
        bounding_boxes = []
        segmentations = []
        obb_boxes = []

        if not os.path.exists(label_file):
            return bounding_boxes, segmentations, obb_boxes

        with open(label_file, 'r') as f:
            for line in f:
                cleaned_line = line.strip()
                if not cleaned_line:
                    logger.warning(f"⚠️ Found an empty line in {label_file}. Skipping it.")
                    continue

                try:
                    parts = list(map(float, cleaned_line.split()))
                except ValueError:
                    logger.warning(f"⚠️ Non-numeric line: {cleaned_line}")
                    continue

                if len(parts) < 2:
                    continue

                class_id = int(parts[0])
                values = parts[1:]
                num_values = len(values)

                # 🔷 CASE: SEGMENTATION
                if num_values >= 6 and num_values % 2 == 0:
                    is_probably_keypoints = (
                        (num_values - 4) % 3 == 0 and
                        all(i % 3 == 2 and values[i] in (0.0, 1.0, 2.0)
                            for i in range(6, num_values, 3))
                    )
                    if not is_probably_keypoints:
                        points = [(values[i], values[i + 1]) for i in range(0, num_values, 2)]
                        if len(points) >= 3:
                            segmentations.append((class_id, points))
                            logger.debug(f"🔷 Segmentation loaded with {len(points)} points → {label_file}")
                            continue

                # 🟨 CASE: OBB (9 coordinates: x1,y1,x2,y2,x3,y3,x4,y4,angle)
                if num_values == 9:
                    obb_points = [(values[i], values[i + 1]) for i in range(0, 8, 2)]
                    angle = values[8]
                    obb_boxes.append((class_id, obb_points, angle))
                    logger.debug(f"🟨 OBB loaded → {label_file}")
                    continue

                # 🟡 CASE: BBOX + KEYPOINTS
                if num_values >= 7 and (num_values - 4) % 3 == 0:
                    keypoints = values[4:]
                    vis_flags = keypoints[2::3]
                    if all(v in (0.0, 1.0, 2.0) and v == int(v) for v in vis_flags):
                        xc, yc, w, h = values[:4]
                        rect = self.yolo_to_rect(xc, yc, w, h, img_width, img_height)
                        keypoint_data = [tuple(keypoints[i:i + 3]) for i in range(0, len(keypoints), 3)]
                        bounding_boxes.append((rect, class_id, None, keypoint_data))
                        logger.debug(f"🟡 Keypoints loaded with {len(keypoint_data)} → {label_file}")
                        continue

                # 🟠 CASE: BBOX + CONFIDENCE
                if num_values == 5:
                    xc, yc, w, h, conf = values
                    rect = self.yolo_to_rect(xc, yc, w, h, img_width, img_height)
                    bounding_boxes.append((rect, class_id, conf))
                    continue

                # 🟢 CASE: REGULAR BBOX
                if num_values == 4:
                    xc, yc, w, h = values
                    rect = self.yolo_to_rect(xc, yc, w, h, img_width, img_height)
                    bounding_boxes.append((rect, class_id, None))
                    continue

                # ❌ UNKNOWN FORMAT
                logger.warning(f"⚠️ Unknown label format → {label_file} → {line}")

        if label_file not in self.logged_label_files:
            logger.info(f"📄 Processed: {label_file} | 🟢 {len(bounding_boxes)} boxes, 🔷 {len(segmentations)} segmentations, 🟨 {len(obb_boxes)} OBBs")
            self.logged_label_files.add(label_file)

        self.all_frame_bounding_boxes[self.current_file] = bounding_boxes
        self.all_frame_segmentations[self.current_file] = segmentations
        self.all_frame_obbs[self.current_file] = obb_boxes

        return bounding_boxes, segmentations, obb_boxes






    def set_selected(self, selected_bbox):
        if selected_bbox is not None:
            self.selected_bbox = selected_bbox
            # Update the class_input_field_label text
                
    def load_classes(self, data_directory=None, default_classes=None, create_if_missing=True):
        """
        Load class names from 'classes.txt'. If missing, create with default classes.
        """
        active_directory = data_directory or self.image_directory or os.getcwd()

        if not os.path.exists(active_directory):
            logger.error(f"❌ Error: Directory does not exist: {active_directory}")
            return []

        classes_file = os.path.join(active_directory, 'classes.txt')

        # Try to load existing classes.txt
        if os.path.exists(classes_file):
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    class_names = [
                        line.strip()
                        .replace('\u200b', '')    # Zero-width space
                        .replace('\xa0', '')      # Non-breaking space
                        .replace('\ufeff', '')    # Byte Order Mark (BOM)
                        for line in f if line.strip()
                    ]

                # Sync only if there's a change
                if hasattr(self, "class_names") and self.class_names == class_names:
                    logger.debug("✅ No changes detected in classes.txt. Skipping unnecessary updates.")
                    return class_names

                self.class_names = class_names
                self.id_to_class = {i: name for i, name in enumerate(class_names)}

                # 🔹 Update UI dropdown and filter spinbox
                self.update_classes_dropdown(class_names)

                logger.info(f"📄 Loaded classes from: {classes_file}")
                return class_names

            except Exception as e:
                logger.error(f"❌ Error reading {classes_file}: {e}")

        # If missing, create 'classes.txt' with default classes
        if create_if_missing and not os.path.exists(classes_file):
            default_classes = default_classes or ['person']
            try:
                with open(classes_file, 'w', encoding='utf-8') as f:
                    for cls in default_classes:
                        f.write(f"{cls}\n")

                self.class_names = default_classes
                self.id_to_class = {i: cls for i, cls in enumerate(default_classes)}

                # 🔹 Update UI dropdown and filter spinbox
                self.update_classes_dropdown(default_classes)

                logger.info(f"📄 Created 'classes.txt' at: {classes_file}")
                return default_classes
            except Exception as e:
                logger.error(f"❌ Error creating {classes_file}: {e}")

        return []


    def save_labels_to_file(self, file_path, labels, mode='w'):
        """
        Writes a list of label strings to a specified file, skipping empty lines.

        Parameters:
        - file_path (str): The path to the file where labels will be saved.
        - labels (list of str): The list of label strings to write to the file.
        - mode (str): The file mode; 'w' for write (overwrite) and 'a' for append.
        """
        try:
            with open(file_path, mode) as file:
                # ✅ Filter out empty lines and whitespace
                filtered_labels = [label.strip() for label in labels if label.strip()]
                if filtered_labels:
                    file.writelines(f"{label}\n" for label in filtered_labels)
        except IOError as e:
            logging.error(f"Failed to save labels to {file_path}: {e}")



    def update_bounding_boxes(self):
        # Clear existing bounding boxes
        for item in self.screen_view.scene().items():
            if isinstance(item, BoundingBoxDrawer):
                self.screen_view.scene().removeItem(item)
                del item

        # Add bounding boxes back to the scene with the updated scale
        rects = self.load_bounding_boxes(self.label_file, self.screen_view.scene(
        ).width(), self.screen_view.scene().height())
        self.display_bounding_boxes(rects) # type: ignore

    def load_files(self, current_file):
        # Get the parent folder of the current file
        folder = Path(current_file).parent

        # Search for image files with various extensions in the folder and sort them
        image_files = sorted(
            glob.glob(str(folder / "*.png"))
            + glob.glob(str(folder / "*.jpg"))
            + glob.glob(str(folder / "*.jpeg"))
            + glob.glob(str(folder / "*.bmp"))
            + glob.glob(str(folder / "*.gif"))
        )

        # Initialize a list to store pairs of image and label files
        paired_files = []

        # Iterate through the found image files
        for img_file in image_files:
            # Generate the corresponding label file name based on the image file name
            base_name = os.path.splitext(img_file)[0]
            txt_file = f"{base_name}.txt"

            # Check if the label file exists
            if os.path.isfile(txt_file):
                paired_files.append((img_file, txt_file))

        # Ensure that the current file's label file is in the list
        current_label_file = (
            current_file.replace(".png", ".txt")
            .replace(".jpg", ".txt")
            .replace(".jpeg", ".txt")
            .replace(".bmp", ".txt")
            .replace(".gif", ".txt")
        )

        if (current_file, current_label_file) not in paired_files:
            paired_files.append((current_file, current_label_file))

        return paired_files



    def clear_processed_logs(self):
        """Reset the set of logged label files when switching datasets."""
        self.logged_label_files.clear()





    def yolo_to_rect(self, x_center, y_center, width, height, img_width, img_height):
        """
        Converts YOLO format (relative) to absolute pixel coordinates.
        """
        rect_x = int((x_center - width / 2) * img_width)
        rect_y = int((y_center - height / 2) * img_height)
        rect_w = int(width * img_width)
        rect_h = int(height * img_height)

        return QRectF(rect_x, rect_y, rect_w, rect_h)

    def navigate_frame(self, direction):
        """Navigates to the next or previous frame based on the direction provided."""
        if not self.current_file:
            QMessageBox.warning(self, "Warning", "No current file selected.")
            return

        if self.current_file not in self.filtered_image_files:
            QMessageBox.warning(self, "Warning", "Current file not found in the filtered list.")
            return

        index = self.filtered_image_files.index(self.current_file)
        checkbox_checked = self.hide_label_checkbox.isChecked()

        # Determine the next or previous file based on direction
        if direction == 'next' and index + 1 < len(self.filtered_image_files):
            new_file = self.filtered_image_files[index + 1]
        elif direction == 'previous' and index - 1 >= 0:
            new_file = self.filtered_image_files[index - 1]
        else:
            QMessageBox.information(self, "Information", f"You have reached the {'end' if direction == 'next' else 'beginning'} of the list.")
            if self.auto_scan_checkbox.isChecked():
                self.auto_scan_checkbox.setChecked(False)
                self.stop_next_timer()
                self.stop_prev_timer()
            return

        # Save the bounding boxes for the current image before switching
        scene_width = self.screen_view.scene().width()
        scene_height = self.screen_view.scene().height()
        label_file = self.get_label_file(self.current_file)
        self.save_bounding_boxes(label_file, scene_width, scene_height)

        # ✅ Now switch the current file before loading new labels
        self.current_file = new_file
        self.settings['lastImage'] = new_file
        self.saveSettings()

        # Display the new image and update UI
        self.display_image(new_file)
        self.img_index_number.setValue(index + (1 if direction == 'next' else -1))

        # 🔥 Load labels for the new file
        self.load_labels(self.label_file, scene_width, scene_height)
        self.current_label_index = 0

        self.reinitialize_bounding_boxes()

        if checkbox_checked:
            self.hide_label_checkbox.setChecked(True)
            self.toggle_label_visibility()

        self.on_frame_change()
        self.update_bbox_visibility()

        if self.roi_checkbox.isChecked():
            self.update_roi(1)
        if self.roi_checkbox_2.isChecked():
            self.update_roi(2)


    def _reattach_bounding_boxes(self):
        """
        Ensures bounding boxes are properly re-added to the scene.
        This is needed to fix right-click deletion and hover issues when navigating between frames.
        """
        # Get all existing bounding boxes from the scene
        bboxes = [item for item in self.screen_view.scene().items() if isinstance(item, BoundingBoxDrawer)]
        
        # Re-add and configure each bounding box
        base_z = 1.0
        for bbox in bboxes:
            if bbox not in self.screen_view.scene().items():
                self.screen_view.scene().addItem(bbox)  # Re-add bounding boxes
            bbox.setZValue(base_z)  # Set incremental z-values
            base_z += 0.1
            bbox.setAcceptHoverEvents(True)  # Ensure hover detection

    def reinitialize_bounding_boxes(self):
        """
        Ensures bounding boxes are reattached properly after switching frames.
        This fixes hover detection and right-click delete issues when moving between frames.
        """
        # First reset all boxes to their normal state
        for item in self.screen_view.scene().items():
            if isinstance(item, BoundingBoxDrawer):
                item.setAcceptHoverEvents(True)
                item.setOpacity(item.normal_opacity)
                item.setSelected(False)

        # Now reattach with proper z-ordering
        self._reattach_bounding_boxes()
        
        # Force scene update
        self.screen_view.scene().update()

    # Updated next_frame and previous_frame to use the modified method
    def next_frame(self):
        # Stop scanning (if in progress)
        self.stop_next_timer()
        self.stop_prev_timer()

        # Change the direction
        self.scan_direction = 'next'
        
        # Navigate to the next frame
        self.navigate_frame('next')

        # Restart scanning if auto_scan is checked and not at the end
        if self.auto_scan_checkbox.isChecked():
            self.start_next_timer()

    def previous_frame(self):
        # Stop scanning (if in progress)
        self.stop_next_timer()
        self.stop_prev_timer()

        # Change the direction
        self.scan_direction = 'previous'
        
        # Navigate to the previous frame
        self.navigate_frame('previous')

        # Restart scanning if auto_scan is checked and not at the beginning
        if self.auto_scan_checkbox.isChecked():
            self.start_prev_timer()


    def update_timer_speed(self, value):
        # Update the timer interval based on the slider's value
        self.timer_interval = value

        # 🕒 Log the updated timer speed
        logger = logging.getLogger("UltraDarkFusionLogger")
        logger.info(f"🕒 Timer speed updated → Interval: {value} ms")

        # If the timers are running, restart them with the new interval
        if self.next_timer.isActive():
            self.start_next_timer()
        if self.prev_timer.isActive():
            self.start_prev_timer()

    # Timer handling methods to start/stop timers and use the new interval
    def start_next_timer(self):
        self.next_timer.stop()
        self.next_timer.start(self.timer_interval)

    def start_prev_timer(self):
        self.prev_timer.stop()
        self.prev_timer.start(self.timer_interval)

    def stop_next_timer(self):
        self.next_timer.stop()

    def stop_prev_timer(self):
        self.prev_timer.stop()




    def toggle_auto_scan(self, state):
        if state:
            # Start scanning in the current direction
            if self.scan_direction == 'next':
                self.start_next_timer()
            elif self.scan_direction == 'previous':
                self.start_prev_timer()
        else:
            # Stop any active scanning
            self.stop_next_timer()
            self.stop_prev_timer()
            
    def stop_on_user_input(self, event):
        # Stop both timers if any input is detected
        self.stop_next_timer()
        self.stop_prev_timer()

        # **Do not uncheck** the auto_scan_checkbox, just stop the timers
        # If auto_scan_checkbox is checked, we'll handle restarting in next_frame/previous_frame

        # Check if the event is a mouse or a key event and handle accordingly
        if isinstance(event, QMouseEvent):
            super().mousePressEvent(event)  # Call the appropriate event method for mouse press
        elif isinstance(event, QKeyEvent):
            super().keyPressEvent(event)  # Call the appropriate event method for key press



    def eventFilter2(self, obj, event):
        if event.type() in [QEvent.MouseButtonPress, QEvent.KeyPress]:
            self.stop_on_user_input(event)
        return super().eventFilter2(obj, event)
    
    def start_next_timer(self):
        # Start the next timer with the interval set by the slider
        self.next_timer.start(self.timer_interval)

    def start_prev_timer(self):
        # Start the previous timer with the interval set by the slider
        self.prev_timer.start(self.timer_interval)

    def stop_next_timer(self):
        self.next_timer.stop()

    def stop_prev_timer(self):
        self.prev_timer.stop()



    # part of the crop function adds noise to background


    def apply_glass_effect(self, image):
        try:
            h, w = image.shape[:2]

            # Apply a higher-quality Gaussian blur for the bubbled glass effect
            blurred = cv2.GaussianBlur(image, (15, 15), 0)
            bubbled = cv2.GaussianBlur(image, (21, 21), 30)

            # Generate random streaks pattern
            streaks = np.random.rand(h, w) * 255
            streaks = cv2.GaussianBlur(streaks, (31, 31), 30).astype(np.uint8)
            _, streaks = cv2.threshold(streaks, 220, 255, cv2.THRESH_BINARY)
            streaks = np.stack([streaks] * 3, axis=-1)

            # Generate random specular highlights
            glossy = np.random.rand(h, w) * 255
            glossy = cv2.GaussianBlur(glossy, (11, 11), 60).astype(np.uint8)
            _, glossy = cv2.threshold(glossy, 250, 255, cv2.THRESH_BINARY)
            glossy = np.stack([glossy] * 3, axis=-1)

            # Blend the original image with the glass effects
            alpha = 0.2
            beta = 0.2
            gamma = 0.05
            delta = 0.05

            result = cv2.addWeighted(image, 1 - alpha, blurred, alpha, 0)
            result = cv2.addWeighted(result, 1 - beta, bubbled, beta, 0)
            result = cv2.addWeighted(result, 1 - gamma, streaks, gamma, 0)
            result = cv2.addWeighted(result, 1 - delta, glossy, delta, 0)

            # Generate random reflection pattern
            reflection = np.random.rand(h, w) * 255
            reflection = cv2.GaussianBlur(reflection, (21, 21), 30).astype(np.uint8)
            reflection = np.stack([reflection] * 3, axis=-1)

            # Blend the result with the reflection
            zeta = 0.05
            result = cv2.addWeighted(result, 1 - zeta, reflection, zeta, 0)

            # Apply a color tint for the final image
            tint = np.array([200, 200, 255], dtype=np.uint8)
            tinted_layer = np.full_like(result, tint)
            result = cv2.addWeighted(result, 0.8, tinted_layer, 0.2, 0)

            return result

        except cv2.error as e:
            logger.error(f"OpenCV error occurred: {e}")
            logger.error("Please ensure the input image is not empty.")
            return None  # or you can return the original image if you prefer: return image

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return None  # or you can return the original image if you prefer: return image

    def glass_checkbox_state_changed(self, state):
        self.glass_effect = state == QtCore.Qt.Checked

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

    def apply_smoke_effect(self, image, intensity=0.05, speed=0.05):
        h, w, _ = image.shape

        # Increment the time for the rolling effect
        self.time += speed

        # Call generate_smoke_mask correctly without duplicate scale_factor
        smoke_mask = self.generate_smoke_mask(h, w, time_offset=self.time)

        # Adjust the intensity of the smoke
        smoke_mask = (smoke_mask * intensity * 255).astype(np.uint8)

        # Apply Gaussian blur to make the smoke effect smoother
        blur_radius = max(3, int(min(w, h) / 50))  # A larger blur for smoother smoke
        if blur_radius % 2 == 0:
            blur_radius += 1  # Ensure the radius is an odd number
        blurred_smoke = cv2.GaussianBlur(smoke_mask, (blur_radius, blur_radius), 0)

        # Convert the blurred mask to a 3-channel grayscale image for blending
        smoke_mask_colored = cv2.cvtColor(blurred_smoke, cv2.COLOR_GRAY2BGR)

        # Optional: Add a subtle color tint to the smoke (grayish-brown)
        tint = np.array([200, 200, 180], dtype=np.uint8)  # Light brown-gray
        smoke_tint_layer = np.full_like(smoke_mask_colored, tint)
        smoke_mask_colored = cv2.addWeighted(smoke_mask_colored, 0.8, smoke_tint_layer, 0.2, 0)

        # Apply the smoke effect using an alpha blend of the original image and the smoke mask
        smoke_effect_alpha = 0.4  # Lighter blend to match the image's subtle smoke
        smokey_image = cv2.addWeighted(image.astype(np.float32), 1 - smoke_effect_alpha,
                                    smoke_mask_colored.astype(np.float32), smoke_effect_alpha, 0)

        return smokey_image.astype(np.uint8)



    def generate_smoke_mask(self, h, w, scale_factor=1.0, time_offset=0.0, octaves=6):
        if not isinstance(h, int) or not isinstance(w, int):
            raise TypeError(f"Expected integers for h and w, but got {type(h)} and {type(w)}")

        # Initialize Perlin noise generator
        noise = PerlinNoise(octaves=octaves, seed=int(time_offset))

        # Generate mesh grid
        X, Y = np.meshgrid(np.arange(w), np.arange(h))

        # Create an empty noise array
        noise_array = np.zeros((h, w), dtype=np.float32)

        # Compute Perlin noise per pixel
        for i in range(h):
            for j in range(w):
                noise_val = noise([i / (20 * scale_factor), j / (20 * scale_factor)])
                noise_array[i, j] = noise_val

        # Normalize noise to [0,1]
        min_val, max_val = noise_array.min(), noise_array.max()
        if max_val - min_val > 0:  # Avoid division by zero
            noise_array = (noise_array - min_val) / (max_val - min_val)

        # Generate binary mask
        mask = noise_array > 0.5

        return mask



    def smoke_checkbox_changed(self, state):
        self.smoke_effect = state == Qt.Checked
        logger.info(f"Smoke effect changed: {self.smoke_effect}")

    def sight_picture_checkbox_changed(self, state):
        self.sight_picture = state == Qt.Checked



    def create_circular_mask(self, image, center, outer_radius_ratio=0.65, inner_radius_ratio=0.645, line_thickness=1, crosshair_length=50):
        h, w, _ = image.shape
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        outer_radius = outer_radius_ratio * min(h, w) // 2
        inner_radius = inner_radius_ratio * min(h, w) // 2

        # Create the circular ring mask
        circle_mask = (dist_from_center <= outer_radius) & (dist_from_center >= inner_radius)

        # Create vertical and horizontal crosshair lines
        vertical_line = (np.abs(X - center[0]) <= line_thickness) & (np.abs(Y - center[1]) <= crosshair_length)
        horizontal_line = (np.abs(Y - center[1]) <= line_thickness) & (np.abs(X - center[0]) <= crosshair_length)

        # Combine circle and crosshair masks
        mask = circle_mask | vertical_line | horizontal_line

        return mask 

    @staticmethod
    def is_polygon(coords):
        return len(coords) > 4
    
    def set_flip_images(self, state):
        self.flip_images = state

    def mosaic_checkbox_changed(self, state):
        self.mosaic_effect = (state == Qt.Checked)

    def pack_thumbnails(self, thumbnail_sizes, canvas_size):
        packer = newPacker(rotation=True)
        packer.add_bin(*canvas_size)
        for size, path in thumbnail_sizes:
            packer.add_rect(*size, rid=path)
        packer.pack()
        packed_thumbnails = []
        for rect in packer.rect_list():
            b, x, y, w, h, rid = rect
            packed_thumbnails.append((rid, x, y, w, h))
        return packed_thumbnails

    def create_mosaic_images(self):
        thumbnail_dir = self.thumbnail_dir
        canvas_size = (self.augmentation_size, self.augmentation_size)
        output_dir = os.path.join(thumbnail_dir, 'mosaic')
        max_images = self.augmentation_count

        extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
        thumbnail_paths = [
            os.path.join(thumbnail_dir, f)
            for f in os.listdir(thumbnail_dir)
            if os.path.splitext(f)[1].lower() in extensions
        ]

        (f"Found {len(thumbnail_paths)} thumbnails.")
        if not thumbnail_paths:
            logger.error("No thumbnails found. Please check the directory.")
            return

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created directory {output_dir}")

        processed_images_count = 0
        batch_size = 1000

        # Initialize a dictionary to track thumbnail usage
        thumbnail_usage = {path: 0 for path in thumbnail_paths}
        max_usage_limit = 5  # Set a max reuse limit

        def get_next_batch(batch_size):
            """Retrieve a batch of thumbnails with limited reuse."""
            available_thumbnails = [path for path, count in thumbnail_usage.items() if count < max_usage_limit]

            # If not enough available thumbnails, reset usage counts
            if len(available_thumbnails) < batch_size:
                thumbnail_usage.clear()
                thumbnail_usage.update({path: 0 for path in thumbnail_paths})
                available_thumbnails = [path for path, count in thumbnail_usage.items() if count < max_usage_limit]

            random.shuffle(available_thumbnails)
            current_batch = available_thumbnails[:batch_size]

            # Update usage counts
            for path in current_batch:
                thumbnail_usage[path] += 1

            return current_batch

        while processed_images_count < max_images:
            current_batch = get_next_batch(batch_size)

            thumbnail_sizes = []
            for path in current_batch:
                try:
                    with Image.open(path) as img:
                        thumbnail_sizes.append((img.size, path))
                except Exception as e:
                    logger.error(f"Failed to open image {path}: {e}")
                    continue

            packed_thumbnails = self.pack_thumbnails(thumbnail_sizes, canvas_size)
            if not packed_thumbnails:
                logger.error("No more thumbnails can fit on the canvas.")
                break

            canvas = Image.new('RGB', canvas_size, color='white')

            for rect in packed_thumbnails:
                try:
                    img_path, x, y, w, h = rect
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    logger.info(f"Placing image: {img_path} at position {(x, y)}, size {(w, h)}")

                    with Image.open(img_path) as thumbnail:
                        original_size = thumbnail.size

                        if original_size != (w, h) and original_size != (h, w):
                            thumbnail = thumbnail.resize((w, h), Image.Resampling.LANCZOS)
                        elif original_size == (h, w):
                            thumbnail = thumbnail.rotate(90, expand=True)
                            thumbnail = thumbnail.resize((w, h), Image.Resampling.LANCZOS)

                        canvas.paste(thumbnail, (x, y))

                except Exception as e:
                    logger.error(f"Failed to place image {img_path}: {e}")
                    continue

            unique_id = uuid.uuid4()
            output_image_path = os.path.join(output_dir, f"mosaic_{unique_id}.jpeg")
            canvas.save(output_image_path)
            logger.info(f"Saved {output_image_path}")

            output_annotation_path = os.path.join(output_dir, f"mosaic_{unique_id}.txt")
            with open(output_annotation_path, 'w') as file:
                for rect in packed_thumbnails:
                    try:
                        img_path, x, y, w, h = rect
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        x_center = (x + w / 2) / canvas_size[0]
                        y_center = (y + h / 2) / canvas_size[1]
                        width = w / canvas_size[0]
                        height = h / canvas_size[1]
                        file.write(f"0 {x_center} {y_center} {width} {height}\n")

                    except Exception as e:
                        logger.error(f"Failed to write annotation for {img_path}: {e}")
                        continue

            processed_images_count += 1

        logger.info(f"All done. Mosaic images created: {processed_images_count}")
        return processed_images_count




    @pyqtSlot(int)
    def update_progress_bar(self, value):
        self.label_progress.setValue(value)

    def set_progress_bar_maximum(self, max_value):
        self.label_progress.setMaximum(max_value)

    def process_images_triggered(self):
        """
        Trigger the process to apply effects and augmentations to images with proper handling of classes.
        """
        logger.info("process_images_triggered called")

        if not self.images_import:
            QMessageBox.warning(self, "Error", "No images to process.")
            return

        output_directory = os.path.dirname(self.images_import[0])
        selected_effects = []

        # Check for mosaic effect and other effects
        processed_images_count = 0  # Initialize counter here

        if self.mosaic_effect:
            logger.info("Creating mosaic images")
            processed_images_count += self.create_mosaic_images()  # Add the returned count here
            selected_effects.append("mosaic")

        if self.glass_effect:
            selected_effects.append("glass_effect")
        if self.motion_blur_effect:
            selected_effects.append("motion_blur")
        if self.flash_effect:
            selected_effects.append("flash_effect")
        if self.smoke_effect:
            selected_effects.append("smoke_effect")
        if self.sight_picture:
            selected_effects.append("sight_picture")
        if self.flip_images:
            selected_effects.append("flip_images")

        # Determine the final output folder name
        output_folder_name = "_".join(selected_effects) if selected_effects else "nonaugmented"
        output_path = os.path.join(output_directory, output_folder_name)

        # Ensure the output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Load class names using `load_classes`
        class_names = self.load_classes()
        if not class_names:
            QMessageBox.warning(self, "Error", "No valid classes found in classes.txt.")
            return

        # Image and Label mapping
        image_files_map = {os.path.splitext(os.path.basename(p))[0]: p for p in self.images_import}
        label_files_map = {os.path.splitext(os.path.basename(p))[0]: p for p in self.label_files}

        total_images = len(image_files_map)

        # Apply augmentations to each image
        for current_image, (image_base_name, image_path) in enumerate(image_files_map.items()):
            label_path = label_files_map.get(image_base_name)
            if label_path is not None and os.path.basename(label_path) != "classes.txt":
                self.apply_augmentations(current_image, image_path, label_path, output_path, output_folder_name, total_images)
                processed_images_count += 1  # Increment processed image count

        # Show the correct count of processed images in the message box
        QMessageBox.information(self, "Success", f"{processed_images_count} images have been successfully processed.")






    def apply_augmentations(self, current_image, image_path, label_path, output_path, output_folder_name, total_images):
        image = cv2.imread(image_path)
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Flip image and labels if enabled
        if self.flip_images:
            image = cv2.flip(image, 1)
            new_labels = []
            for label in labels:
                parts = label.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))

                # Handle bounding boxes (x_center y_center width height)
                if len(coords) == 4:
                    x_center, y_center, width, height = coords
                    x_center = 1 - x_center  # Flip horizontally
                    new_label = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

                # Handle polygons (x1 y1 x2 y2 ...)
                else:
                    flipped_coords = []
                    for i, val in enumerate(coords):
                        if i % 2 == 0:  # x-coordinate
                            flipped_coords.append(1 - val)
                        else:  # y-coordinate
                            flipped_coords.append(val)
                    coord_str = " ".join(f"{v:.6f}" for v in flipped_coords)
                    new_label = f"{class_id} {coord_str}\n"

                new_labels.append(new_label)
            labels = new_labels  # Replace with updated labels

        # If labels exist, use the first one to center effects (works for both box and polygon)
        if labels:
            first = labels[0].strip().split()
            class_id = int(first[0])
            coords = list(map(float, first[1:]))

            h, w, _ = image.shape

            if len(coords) == 4:
                x_center, y_center, width, height = coords
            else:
                # For polygons, compute the average of x and y coordinates
                xs = coords[::2]
                ys = coords[1::2]
                x_center = sum(xs) / len(xs)
                y_center = sum(ys) / len(ys)

            actual_x_center = int(x_center * w)
            actual_y_center = int(y_center * h)

            # Apply circular sight if enabled
            if self.sight_picture:
                center = (actual_x_center, actual_y_center)
                mask = self.create_circular_mask(image, center, crosshair_length=50)
                image[mask] = [0, 0, 0]  # Black pixels in the mask

        # Apply all other effects
        if self.smoke_effect:
            image = self.apply_smoke_effect(image)
            if image is None:
                return

        if self.flash_effect:
            image = self.apply_flashbang_effect(image)
            if image is None:
                return

        if self.motion_blur_effect:
            image = self.apply_motion_blur_effect(image)
            if image is None:
                return

        if self.glass_effect:
            image = self.apply_glass_effect(image)
            if image is None:
                return

        # Save the augmented image
        output_image_path = os.path.join(output_path, f"{output_folder_name}_{timestamp}_{current_image}.jpg")
        cv2.imwrite(output_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Save the updated label file
        output_label_path = os.path.join(output_path, f"{output_folder_name}_{timestamp}_{current_image}.txt")
        with open(output_label_path, 'w') as f:
            f.writelines(labels)

        # Emit progress update
        progress_percentage = int((current_image + 1) / total_images * 100)
        self.progress_signal.emit(progress_percentage)




    def import_images_triggered(self):
        directory = QFileDialog.getExistingDirectory(None, 'Select Image Directory')
        if not directory:
            return

        self.thumbnail_dir = directory
        self.images_import = self.get_image_files(directory)  # ✅ uses your logic

        # Collect label files (.txt only)
        self.label_files = [
            os.path.join(directory, f).replace("\\", "/")
            for f in os.listdir(directory)
            if f.lower().endswith(".txt")
        ]





    # Populate the GIF ComboBox with available GIFs in the styles/gifs folder
    def populateGifCombo(self):
        gif_directory = "styles/gifs"
        if os.path.exists(gif_directory):  # Ensure the directory exists
            gif_files = [f for f in os.listdir(gif_directory) if f.endswith(".gif")]
            self.gif_change.clear()
            self.gif_change.addItems(gif_files)

    # Update the GIF when a new one is selected from the ComboBox
    def onGifChange(self, index):
        if index != -1:  # Ensure the index is valid
            selected_gif = self.gif_change.currentText()
            gif_path = f"styles/gifs/{selected_gif}"
            if os.path.exists(gif_path):  # Ensure the GIF file exists
                self.movie.stop()
                self.movie.setFileName(gif_path)
                self.movie.start()

    # Populate the Style ComboBox with available stylesheets
    def populate_style_combo_box(self):
        style_folder = os.path.join(QtCore.QDir.currentPath(), 'styles')
        style_files = [file for file in os.listdir(style_folder) if file.endswith(('.qss', '.css', '.stylesheet'))] if os.path.exists(style_folder) else []

        qt_material_styles = list_themes()  # Optional: Add qt_material themes if available
        style_files.extend(qt_material_styles)
        self.styleComboBox.clear()
        self.styleComboBox.addItems(style_files)

        # Load settings and apply the last used theme
        self.settings = self.loadSettings()
        last_theme = self.settings.get('lastTheme', 'Default')
        last_theme_index = self.styleComboBox.findText(last_theme)
        self.styleComboBox.setCurrentIndex(last_theme_index if last_theme_index >= 0 else 0)

        self.apply_stylesheet()  # Apply the stylesheet
        self.update_gif_to_match_theme()  # Ensure the GIF matches the loaded theme

    # Apply the selected stylesheet to the application
    def apply_stylesheet(self):
        selected_style = self.styleComboBox.currentText()
        if selected_style == "Default":
            self.setStyleSheet("")  # No external styles applied
        elif selected_style in list_themes():
            apply_stylesheet(self, selected_style)  # Apply qt_material theme
        else:
            style_folder = os.path.join(QtCore.QDir.currentPath(), 'styles')
            file_path = os.path.join(style_folder, selected_style)
            try:
                with open(file_path, 'r', encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
            except Exception as e:
                logger.error(f"Failed to read stylesheet file: {e}")
                self.setStyleSheet("")  # Reset to default if there's an error


    # Update the GIF to match the current theme or style
    def update_gif_to_match_theme(self):
        selected_style = self.styleComboBox.currentText()
        gif_directory = "styles/gifs"

        # Automatically look for a GIF with the same name as the stylesheet
        style_base_name = os.path.splitext(selected_style)[0]  # Remove extension from style file
        corresponding_gif = f"{style_base_name}.gif"

        gif_path = os.path.join(gif_directory, corresponding_gif)
        if os.path.exists(gif_path):
            self.movie.stop()
            self.movie.setFileName(gif_path)
            self.movie.start()
        else:
            logger.warning(f"No matching GIF found for theme: {selected_style}")

    # Handle theme change
    def on_style_change(self):
        self.apply_stylesheet()  # Apply the selected stylesheet
        self.saveSettings()  # Save the selected style in settings
        self.update_gif_to_match_theme()  # Update the GIF to match the new style

    # Initialize the theme and GIF on startup
    def initialize_theme_and_gif(self):
        self.populate_style_combo_box()  # Populate themes
        self.populateGifCombo()  # Populate available GIFs
        self.update_gif_to_match_theme()  # Automatically load matching GIF


            # def for ultralytics train

    def browse_yaml_clicked(self):
        file_name = self.open_file_dialog(
            "Select YAML File", "YAML Files (*.yaml);;All Files (*)")
        if file_name:
            self.data_yaml_path = file_name
            self.yaml_label.setText(f"Data YAML: {file_name}")
        else:
            self.data_yaml_path = None
            self.yaml_label.setText("No Data YAML selected")

    def browse_pt_clicked(self):
        file_name = self.open_file_dialog("Select Model File", "Model Files (*.pt *.yaml);;All Files (*)")
        if file_name.endswith('.pt'):
            # Store .pt path for direct training or transfer learning
            self.pt_path = file_name  
            self.pt_label.setText(f"Model: {file_name}")  
            self.model_config_path = None  # Reset .yaml config path if a .pt is chosen
        elif file_name.endswith('.yaml'):
            # Store .yaml path for training config
            self.model_config_path = file_name
            self.pt_label.setText(f"Model Config: {file_name}")
            # Ask if the user wants to perform transfer learning
            transfer_learning_reply = QMessageBox.question(self, 'Transfer Learning',
                                                        'Do you want to perform transfer learning from a .pt file?',
                                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if transfer_learning_reply == QMessageBox.Yes:
                # If yes, prompt to select a .pt file for pretrained weights
                pretrained_model_file = self.open_file_dialog("Select Pre-trained Model File", "Model Files (*.pt);;All Files (*)")
                if pretrained_model_file:
                    self.pretrained_model_path = pretrained_model_file
                    self.pt_label.setText(f"Model Config: {self.model_config_path}; Pretrained: {pretrained_model_file}")
                else:
                    # If no file is chosen, clear the pretrained_model_path
                    self.pretrained_model_path = None
            else:
                # If no transfer learning, clear the pretrained_model_path
                self.pretrained_model_path = None
        # If no file is chosen, reset both paths
        else:
            self.model_config_path = None
            self.pretrained_model_path = None
            self.pt_label.setText("No Model selected")



    def open_file_dialog(self, caption, file_filter, multiple=False):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        if multiple:
            file_names, _ = QFileDialog.getOpenFileNames(
                self, caption, "", file_filter, options=options)
            return file_names
        else:
            file_name, _ = QFileDialog.getOpenFileName(
                self, caption, "", file_filter, options=options)
            return file_name

    def on_save_dir_clicked(self):
        logger.debug("on_save_dir_clicked called")  # Debugging message
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory", "", options=options)
        if directory:
            self.runs_directory = directory  # Update the save directory value


    def ultralytics_train_clicked(self):
        if not self.data_yaml_path:
            QMessageBox.warning(self, "Warning", "Data YAML file not selected. Please select a data YAML file before proceeding.")
            return

        #  Get selected task from combobox
        selected_task = self.task_combobox.currentText().lower()  # Get lowercase text (detect, segment, classify, etc.)

        #  Ensure valid task selection
        if selected_task not in ["detect", "segment", "classify", "pose", "obb"]:
            QMessageBox.critical(self, "Error", f"Invalid task selected: {selected_task}")
            return

        #  Set the command dynamically based on task
        command = [
            "yolo", selected_task, "train",  #  Dynamically insert the selected task
            f"data={self.data_yaml_path}",
            f"imgsz={self.imgsz_input.text()}",
            f"epochs={self.epochs_input.text()}",
            f"batch={self.batch_input.text()}",
            f"project={self.runs_directory}"
        ]

        #  Add model configurations
        if self.model_config_path:
            command.append(f"model={self.model_config_path}")
            if self.pretrained_model_path:
                command.append(f"pretrained={self.pretrained_model_path}")
        elif self.pt_path:
            command.append(f"model={self.pt_path}")

        #  Append additional settings
        if self.amp_true.isChecked():
            command.append("amp=True")
        if self.resume_checkbox.isChecked():
            command.append("resume=True")
        if self.freeze_checkbox.isChecked():
            command.append(f"freeze={self.freeze_input.value()}")
        if self.patience_checkbox.isChecked():
            command.append(f"patience={self.patience_input.text()}")

        #  Start the training process
        self.process = QProcess(self)
        self.process.setProgram("yolo")
        self.process.setArguments(command[1:])  # Exclude "yolo"

        #  Updated output handling
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)

        self.process.start()

        if not self.process.waitForStarted():
            QMessageBox.critical(self, "Error", "Failed to start the training process.")
            return

        logger.info(f" Training started with task: {selected_task}")

        #  Start TensorBoard if checked
        if self.tensorboardCheckbox.isChecked():
            tensorboard_log_dir = os.path.join(self.runs_directory, "train")
            self.tensorboard_process = QProcess(self)
            self.tensorboard_process.setProgram("tensorboard")
            self.tensorboard_process.setArguments(["--logdir", tensorboard_log_dir, "--port", "6007"])

            self.tensorboard_process.start()

            if not self.tensorboard_process.waitForStarted():
                QMessageBox.critical(self, "Error", "Failed to start TensorBoard.")
                return
            tensorboard_url = "http://localhost:6006"
            webbrowser.open(tensorboard_url)
            logger.info(" TensorBoard started and opened in web browser.")




    def handle_stdout(self):
        output = self.process.readAllStandardOutput().data().decode()

        #  Filter out unnecessary progress logs
        if "it/s" in output or "%" in output:
            print(output, end="\r")  # Show progress inline, avoid cluttering logs
        else:
            print(output)  # Just print it, no logging will be done here resizes the ui to much.



    def handle_stderr(self):
        error = self.process.readAllStandardError().data().decode()

        if "it/s" in error or "%" in error:
            print(error, end="\r")
        else:
            # Color the error in red
            print(f"\033[91m{error}\033[0m")  # Red color for errors





    def process_finished(self):
        logger.info("Training process finished.")
        self.process = None







    def load_model(self):
        home_directory = os.path.expanduser('~')
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', home_directory, "Model files (*.pt *.onnx)")

        if fname:
            logger.info(f"Loading model from file: {fname}")
            self.model = YOLO(fname)
            logger.info(f"Model loaded successfully from {fname}")
        else:
            logger.warning("No model file selected.")

    def initiate_conversion(self):
        # Check if a model is loaded
        if self.model is None:
            # Show a warning message if no model is loaded
            QMessageBox.warning(self, "No Model", 'No model loaded')
            logger.warning("Model conversion attempted without a loaded model.")
            return

        # Get the selected format for model conversion
        format_selected = self.convert_model.currentText()
        logger.info(f"Selected model conversion format: {format_selected}")

        # Check if the selected format is 'saved_model' (assuming it's the TensorFlow format)
        if format_selected.lower() == "saved_model":
            try:
                # Define the command for converting ONNX model to TensorFlow saved model
                command = [
                    "onnx2tf",
                    "-i", "model.onnx",
                    "-o", "model_saved_model",
                    "-b", "-ois", "-kt", "-kat", "-onwdt"
                ]
                logger.info("Starting ONNX to TensorFlow conversion using command: " + " ".join(command))

                # Run the conversion command using subprocess
                subprocess.run(command, check=True)

                # Show an information message indicating successful conversion
                QMessageBox.information(self, "Conversion Successful", f'Model converted to {format_selected}')
                logger.info(f"Model successfully converted to {format_selected}.")
            except subprocess.CalledProcessError:
                # Show an error message if the conversion process encounters an error
                QMessageBox.critical(self, "Conversion Error", "Failed to convert model")
                logger.error("Failed to convert model using onnx2tf.")
            except Exception as e:
                # Show an error message for any other exceptions
                QMessageBox.critical(self, "Conversion Error", str(e))
                logger.error(f"Error during conversion: {e}")
        else:
            try:
                # Export the model to the selected format
                self.model.export(format=format_selected)
                logger.info(f"Model successfully converted to {format_selected}")

                # Show an information message indicating successful conversion
                QMessageBox.information(self, "Conversion Successful", f'Model converted to {format_selected}')
            except Exception as e:
                # Show an error message if the conversion process encounters an error
                QMessageBox.critical(self, "Conversion Error", str(e))
                logger.error(f"Error exporting model to {format_selected}: {e}")

    def handle_conversion(self):
        if self.model is None:
            QMessageBox.warning(self, "No Model", 'No model loaded')
            logger.warning("Conversion attempted with no model loaded.")
            return

        format_selected = self.convert_model.currentText()
        logger.info(f"Handling conversion for format: {format_selected}")
        self.update_gui_elements(format_selected)  # Update GUI elements based on the selected format

        # Collect image size parameters
        imgsz_height = self.imgsz_input_H.text()
        imgsz_width = self.imgsz_input_W.text()
        try:
            imgsz_list = [int(imgsz_height), int(imgsz_width)]
            logger.info(f"Image size for conversion set to: Height={imgsz_height}, Width={imgsz_width}")
        except ValueError:
            QMessageBox.critical(self, "Conversion Error", f"Invalid image size values: Height={imgsz_height}, Width={imgsz_width}")
            logger.error(f"Invalid image size values: Height={imgsz_height}, Width={imgsz_width}")
            return

        # Initialize and collect parameters
        export_params = {
            'format': format_selected,
            'imgsz': imgsz_list,
            'device': '0'  # Default device is '0'
        }

        if self.half_true.isChecked() and self.half_true.isEnabled():
            export_params['half'] = True
            logger.info("Export parameter 'half' enabled.")
            
        if self.int8_true.isChecked() and self.int8_true.isEnabled():
            export_params['int8'] = True
            logger.info("Export parameter 'int8' enabled.")
            
        if self.simplify.isChecked() and self.simplify.isEnabled():
            export_params['simplify'] = True
            logger.info("Export parameter 'simplify' enabled.")

        if self.dynamic_true.isChecked() and self.dynamic_true.isEnabled():
            export_params['dynamic'] = True
            logger.info("Export parameter 'dynamic' enabled.")

        if self.batch_size_checkbox.isChecked() and self.batch_size_checkbox.isEnabled():
            export_params['batch'] = self.batch_input_spinbox.value()
            logger.info(f"Export parameter 'batch' set to: {export_params['batch']}")

        # Execute the export
        try:
            self.model.export(**export_params)
            QMessageBox.information(self, "Conversion Successful", f'Model converted to {format_selected}')
            logger.info(f"Model exported successfully to {format_selected} with parameters: {export_params}")
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", str(e))
            logger.error(f"Error exporting model: {e}")


    def is_parameter_supported(self, format_selected, param):
        supported_params = {
            'torchscript': {'optimize', 'batch', 'imgsz'},
            'onnx': {'half', 'int8', 'dynamic', 'simplify', 'batch', 'imgsz', 'opset'},
            'openvino': {'half', 'int8', 'batch', 'imgsz'},
            'engine': {'half', 'dynamic', 'simplify', 'workspace', 'int8', 'batch', 'imgsz'},
            'coreml': {'half', 'int8', 'nms', 'batch', 'imgsz'},
            'saved_model': {'int8', 'batch', 'keras', 'imgsz'},
            'pb': {'batch', 'imgsz'},
            'tflite': {'half', 'int8', 'batch', 'imgsz'},
            'edgetpu': {'batch', 'imgsz'},
            'tfjs': {'half', 'int8', 'batch', 'imgsz'},
            'paddle': {'batch', 'imgsz'},
            'ncnn': {'half', 'batch', 'imgsz'}
        }
        logger.debug(f"Checking if parameter '{param}' is supported for format '{format_selected}'")
        return param in supported_params.get(format_selected, set())

    def update_gui_elements(self, format_selected):
        # General enabling/disabling based on support
        self.half_true.setEnabled(self.is_parameter_supported(format_selected, 'half'))
        if not self.is_parameter_supported(format_selected, 'half'):
            self.half_true.setChecked(False)

        self.int8_true.setEnabled(self.is_parameter_supported(format_selected, 'int8'))
        if not self.is_parameter_supported(format_selected, 'int8'):
            self.int8_true.setChecked(False)

        self.simplify.setEnabled(self.is_parameter_supported(format_selected, 'simplify'))
        if not self.is_parameter_supported(format_selected, 'simplify'):
            self.simplify.setChecked(False)

        self.dynamic_true.setEnabled(self.is_parameter_supported(format_selected, 'dynamic'))
        if not self.is_parameter_supported(format_selected, 'dynamic') or self.half_true.isChecked():
            self.dynamic_true.setChecked(False)
            self.dynamic_true.setEnabled(False)

        self.batch_size_checkbox.setEnabled(self.is_parameter_supported(format_selected, 'batch'))
        if not self.is_parameter_supported(format_selected, 'batch'):
            self.batch_size_checkbox.setChecked(False)

        # Additional checks or reconfiguration based on current checkbox states
        if self.half_true.isChecked() and self.dynamic_true.isEnabled():
            self.dynamic_true.setEnabled(False)
            self.dynamic_true.setChecked(False)


    def format_changed(self):
        format_selected = self.convert_model.currentText()
        logger.debug(f"Format changed to: {format_selected}") 
        self.update_gui_elements(format_selected)



        # DEF FOR TXT MAKER
    def import_images(self):
        """
        Imports images from the selected directory and loads class names.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        directory = QFileDialog.getExistingDirectory(self, "Select Import Directory", options=options)

        if directory:
            # Store the correct directory for classes.txt
            self.image_directory = directory  # Ensure `self.image_directory` is set

            # Use `get_image_files()` to get absolute image paths
            self.images = self.get_image_files(directory)

            # Log output to ensure images are detected correctly
            if not self.images:
                QMessageBox.warning(self, 'Warning', '⚠️ No images found in the selected directory!')
            else:
                logger.info(f"📂 Found {len(self.images)} images in {directory}")

            # Load classes from `classes.txt` in the image directory
            self.classes = self.load_classes(data_directory=directory, default_classes=["person"])


    def output_paths(self):
        """
        Handles creation of train.txt and valid.txt.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        # Select between train.txt and valid.txt
        selected_option = self.dropdown.currentText()
        default_filename = selected_option if selected_option in ["valid.txt", "train.txt"] else ""
        default_path = os.path.join("", default_filename)

        save_file, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", default_path, "Text Files (*.txt);;All Files (*)", options=options
        )

        if save_file:
            output_dir = os.path.dirname(save_file).replace("\\", "/")

            #  Create train.txt
            if selected_option == "train.txt":
                train_txt_path = os.path.join(output_dir, "train.txt")
                with open(train_txt_path, "w", encoding="utf-8") as f:
                    for image in self.images:
                        abs_path = os.path.abspath(image).replace("\\", "/")  #  Ensure absolute paths
                        f.write(abs_path + "\n")  
                QMessageBox.information(self, 'Information', 'train.txt file has been created!')

            #  Create valid.txt
            elif selected_option == "valid.txt":
                valid_percent = int(self.valid_percent.currentText().replace('%', ''))

                # Shuffle images and select valid set
                images_copy = self.images[:]
                random.shuffle(images_copy)
                num_valid_images = int(len(images_copy) * (valid_percent / 100.0))

                #  Only select images with non-empty label files
                valid_images = []
                for image in images_copy:
                    txt_file = os.path.splitext(image)[0] + ".txt"
                    if os.path.exists(txt_file) and os.path.getsize(txt_file) > 0:
                        valid_images.append(image)
                    if len(valid_images) >= num_valid_images:
                        break

                # Write valid images with absolute paths
                with open(save_file, "w", encoding="utf-8") as f:
                    for image in valid_images:
                        abs_path = os.path.abspath(image).replace("\\", "/")  #  Ensure absolute paths
                        f.write(abs_path + "\n")  

                QMessageBox.information(self, 'Information', 'valid.txt file has been created!')

            #  Update YAML & Data Files
            self.create_and_update_yaml_and_data_files(output_dir, selected_option)


    def create_and_update_yaml_and_data_files(self, output_dir, selected_option):
        """
        Creates obj.data, obj.yaml, and obj.names with correct class count and paths.
        Ensures obj.names matches classes.txt.
        """
        output_dir = output_dir.replace("\\", "/")

        # 1. Use the correct dataset directory from `import_images()`
        dataset_dir = self.image_directory  # Ensures it pulls from the correct place

        if not dataset_dir:
            logger.error("❌ No dataset directory found! Make sure images are imported first.")
            return

        # Ensure dataset_dir is NOT an output directory
        if any(x in dataset_dir.lower() for x in ["output", "train.txt", "valid.txt", "obj.names", "obj.data"]):
            logger.warning(f"⚠️ Skipping incorrect dataset directory: {dataset_dir}")
            return

        # Load `classes.txt` from dataset directory
        self.classes = self.load_classes(data_directory=dataset_dir)

        if not self.classes:
            QMessageBox.critical(self, "Error", "❌ No valid classes found. Check 'classes.txt' in your dataset directory.")
            return

        class_numb = len(self.classes)

        # Define paths
        train_txt_path = os.path.join(output_dir, "train.txt").replace("\\", "/")
        valid_txt_path = os.path.join(output_dir, "valid.txt").replace("\\", "/") if selected_option == "valid.txt" else train_txt_path
        obj_names_file = os.path.join(output_dir, "obj.names").replace("\\", "/")
        data_file_path = os.path.join(output_dir, "obj.data").replace("\\", "/")
        obj_yaml_file = os.path.join(output_dir, "obj.yaml").replace("\\", "/")
        backup_dir = os.path.join(output_dir, "backup").replace("\\", "/")

        # 3. Ensure `obj.names` matches `classes.txt`
        try:
            with open(obj_names_file, "w", encoding="utf-8") as f:
                for class_name in self.classes:
                    f.write(class_name + "\n")

            logger.info(f"📄 Updated `obj.names` to match `classes.txt` (Loaded {len(self.classes)} classes)")

        except Exception as e:
            logger.error(f"❌ Failed to update `obj.names`: {e}")
            return

        # 4. Create `obj.data` for Darknet
        try:
            with open(data_file_path, "w", encoding="utf-8") as f:
                f.write(f"classes = {class_numb}\n")
                f.write(f"train = {train_txt_path}\n")
                f.write(f"valid = {valid_txt_path}\n")
                f.write(f"names = {obj_names_file}\n")
                f.write(f"backup = {backup_dir}\n")

            logger.info(f" Created `obj.data` with {class_numb} classes.")
        except Exception as e:
            logger.error(f"❌ Failed to create `obj.data`: {e}")
            return
        # 5. Create `obj.yaml` for yolo
        try:
            with open(obj_yaml_file, "w", encoding="utf-8") as f:
                f.write("# YOLOv8 configuration file\n\n")
                f.write(f"path: {output_dir}\n\n")
                f.write(f"train: {train_txt_path}\n")
                f.write(f"val: {valid_txt_path}\n\n")
                f.write(f"nc: {class_numb}\n")

                # 🔁 Always attempt to include keypoint config if available
                points_json_path = os.path.join(dataset_dir, "points.json")
                if os.path.exists(points_json_path):
                    try:
                        with open(points_json_path, "r", encoding="utf-8") as json_file:
                            points_data = json.load(json_file)
                            keypoints = points_data.get("keypoints", [])
                            num_keypoints = len(keypoints)
                            kpt_dims = 3

                            if num_keypoints > 0:
                                f.write(f"\nkpt_shape: [{num_keypoints}, {kpt_dims}]\n")

                                # ✍️ Write flip_idx if available
                                flip_idx = points_data.get("flip_idx", [])
                                if flip_idx:
                                    f.write(f"flip_idx: {flip_idx}\n")

                                # ✍️ Write skeleton if available
                                skeleton = points_data.get("skeleton", [])
                                if skeleton:
                                    f.write("skeleton:\n")
                                    for link in skeleton:
                                        f.write(f"  - {link}\n")

                                logger.info("🧠 Injected kpt_shape, flip_idx, and skeleton from points.json")

                    except Exception as e:
                        logger.warning(f"⚠️ Failed to parse keypoint config from points.json: {e}")
                else:
                    logger.info("ℹ️ No points.json found. Skipping keypoint structure.")

                # ✍️ Write class names
                f.write("\nnames:\n")
                for i, name in enumerate(self.classes):
                    f.write(f"  {i}: {name}\n")

            logger.info(f"✅ Created `obj.yaml` with {class_numb} classes.")

        except Exception as e:
            logger.error(f"❌ Failed to create `obj.yaml`: {e}")



    # def for train darknet

    def browse_file_clicked(self, title, file_types, key, multiple=False):
        file_names = self.open_file_dialog(
            title, file_types, multiple=multiple
        )
        if multiple:
            file_names = [
                file for file in file_names 
                if file.endswith(('.weights', '.conv.')) or re.match(r'.*\.conv\.\d+', file)
            ]
            file_label = "Files: " + ", ".join(file_names).rstrip()
        else:
            file_label = f"File: {file_names}"

        self.file_paths[key] = file_names
        getattr(self, f"{key}_label").setText(file_label)

    def browse_data_clicked(self):
        self.browse_file_clicked(
            "Select Data File", "Data Files (*.data);;All Files (*)", "data"
        )

    def browse_cfg_clicked(self):
        self.browse_file_clicked(
            "Select Config File", "Config Files (*.cfg);;All Files (*)", "cfg"
        )

    def browse_weights_clicked(self):
        self.browse_file_clicked(
            "Select Weights Files", "Weights Files (*.weights *.conv.*);;All Files (*)", "weights", multiple=True
        )


    def darknet_train(self):
        data_file = self.file_paths["data"]
        cfg_file = self.file_paths["cfg"]
        weights_file = self.file_paths["weights"][-1] if self.file_paths["weights"] else ""

        command = f"darknet detector train {data_file} {cfg_file} {weights_file}"
        if self.show_chart.isChecked():
            command += " -show_chart"
        else:
            command += " -dont_show"

        if self.cache.isChecked() and self.cache_input.text().isdigit():
            command += f" -cache {self.cache_input.text()}"

        if not self.map_check.isChecked():
            command += " -map "

        command += " -clear" if self.clear_check.isChecked() else ""
        command += " -random" if self.random.isChecked() else ""
        command += " -gpus 0,1 " if self.gpu1.isChecked() else ""
        command += " -gpus 0,1,2 " if self.gpu2.isChecked() else ""

        terminal_command = f'start cmd.exe @cmd /k "{command}"' if sys.platform == 'win32' else f'gnome-terminal -- {command}'
        subprocess.run(terminal_command, shell=True)
        logger.info(f"Command to be executed: {command}")




    def import_data(self):
        """Import text files and class configurations."""
        data_directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Data Directory')
        if not data_directory:
            logger.warning("No directory selected.")
            return

        data_directory = os.path.normpath(data_directory)
        self.text_files = glob.glob(os.path.join(data_directory, '*.txt'))
        self.image_directory = data_directory

        logger.info(f"Detected text files: {self.text_files}")

        # Load and cache class mapping once
        self.class_mapping = self.load_classes(data_directory)
        if self.class_mapping:
            num_classes = len(self.class_mapping)
            logger.info(f"Loaded {num_classes} classes.")
            self.update_classes_and_batches(num_classes)
        else:
            logger.error("Could not load classes.txt.")
            QtWidgets.QMessageBox.warning(self, "Error", "Could not load classes.txt.")






    def update_classes_and_batches(self, num_classes):
        """
        Update classes, max_batches, steps, and scales in the CFG and table.
        """
        max_batches = num_classes * 5000
        steps = ",".join([str(int(p * max_batches)) for p in self.STEP_PERCENTAGES])
        scales = self.DEFAULT_SCALES

        updates = {
            'classes': num_classes,
            'max_batches': max_batches,
        }

        # Update global (net) section for steps and scales
        net_updates = {
            'steps': steps,
            'scales': scales,
        }

        # Update each parameter in the relevant sections
        for param_name, value in updates.items():
            if param_name == 'classes':
                self.update_cfg_param(param_name, value, section="[yolo]")
            else:
                self.update_cfg_param(param_name, value, section="[net]")

        for param_name, value in net_updates.items():
            self.update_cfg_param(param_name, value, section="[net]")



    def calculate_anchors(self):
        """Calculate anchors and update CFG file."""
        if not hasattr(self, 'class_mapping') or not self.class_mapping:
            QtWidgets.QMessageBox.warning(self, "Error", "Classes are not loaded.")
            return

        if not self.text_files:
            QtWidgets.QMessageBox.warning(self, "Error", "No annotation files loaded.")
            return

        if not hasattr(self, 'filename') or not self.filename:
            QtWidgets.QMessageBox.warning(self, "Error", "Please open a .cfg file first.")
            return

        num_clusters = self.clusters_spinbox.value()
        width_value = self.width_spinbox.value()
        height_value = self.height_spinbox.value()

        annotation_dims = self.get_annotation_dimensions(valid_classes=set(range(len(self.class_mapping))))

        max_samples = 50000
        if len(annotation_dims) > max_samples:
            annotation_dims = random.sample(annotation_dims, max_samples)

        if not annotation_dims:
            QtWidgets.QMessageBox.warning(self, "Error", "No valid annotations found.")
            return

        X = np.array(annotation_dims)
        mini_kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=10000)
        mini_kmeans.fit(X)
        centroids = sorted(mini_kmeans.cluster_centers_, key=lambda c: c[0] * c[1])

        anchors_pixel = np.array([list(map(int, centroid * [width_value, height_value])) for centroid in centroids])
        anchors_str = ', '.join(map(str, anchors_pixel.flatten()))

        # Single method to update all CFG params consistently
        cfg_updates = {
            'anchors': anchors_str,
            'width': width_value,
            'height': height_value
        }

        for param, value in cfg_updates.items():
            self.update_cfg_param(param, value)

        # This line updates num= in all [yolo] sections immediately
        self.update_cfg_param('num', num_clusters, section="[yolo]", update_all=True)

        avg_iou = round(self.calculate_avg_iou(annotation_dims, centroids) * 100, 2)
        self.saveAnchorsSettings(avg_iou)

        if self.show_checkbox.isChecked():
            self.plot_anchors(centroids, avg_iou)





    def get_annotation_dimensions(self, valid_classes):
        """Retrieve dimensions, only considering valid classes."""
        annotation_dims = []
        for text_file in self.text_files:
            if text_file.endswith("classes.txt"):
                continue
            with open(text_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            if class_id in valid_classes:
                                width, height = float(parts[3]), float(parts[4])
                                annotation_dims.append((width, height))
                        except ValueError:
                            logger.warning(f"Invalid numeric values in {text_file}")
        return annotation_dims


    def update_cfg_param(self, param_name, value, section=None, update_all=False):
        """Update a parameter in the CFG file, optionally updating all occurrences."""
        if not hasattr(self, 'filename') or not self.filename:
            QtWidgets.QMessageBox.warning(self, "Error", "No .cfg file loaded.")
            return

        with open(self.filename, 'r') as f:
            lines = f.readlines()

        new_lines = []
        in_target_section = False
        updated_count = 0

        for line in lines:
            stripped_line = line.strip()

            # Check if entering a new section
            if stripped_line.startswith('['):
                if section:
                    in_target_section = (stripped_line == section)
                else:
                    in_target_section = True  # No section specified; global update

            if in_target_section and stripped_line.startswith(f"{param_name}="):
                new_lines.append(f"{param_name}={value}\n")
                updated_count += 1
                if not update_all:
                    in_target_section = False  # Exit section unless updating all occurrences
            else:
                new_lines.append(line)

        if updated_count == 0 and section:
            # Parameter wasn't found; append it at the end of the file within the section
            new_lines.append(f"{section}\n{param_name}={value}\n")

        with open(self.filename, 'w') as f:
            f.writelines(new_lines)

        self.update_table_param(param_name, value)
        logger.info(f"Updated '{param_name}' to '{value}' in {updated_count if updated_count else 'new'} occurrence(s).")






    def update_table_param(self, param_name, value):
        """Update a parameter in the cfg_table."""
        for row in range(self.cfg_table.rowCount()):
            param_item = self.cfg_table.item(row, 0)
            if param_item and param_name in param_item.text():
                self.cfg_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))
                logger.info(f"Updated '{param_name}' in cfg_table to {value}")


    def calculate_avg_iou(self, annotation_dims, centroids):
        """Calculate average IoU for anchors."""
        avg_iou = sum(max(self._iou(centroid, annotation) for centroid in centroids) for annotation in annotation_dims)
        return avg_iou / len(annotation_dims)

    def update_cfg_anchors(self, anchors):
        """
        Update anchors specifically in existing [yolo] sections of the CFG file.
        """
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()

            new_config = []
            inside_yolo_section = False
            yolo_section_count = 0

            for line in lines:
                stripped_line = line.strip()

                if stripped_line.startswith("[yolo]"):
                    inside_yolo_section = True
                    yolo_section_count += 1

                elif stripped_line.startswith('[') and inside_yolo_section:
                    inside_yolo_section = False

                if inside_yolo_section and stripped_line.startswith("anchors="):
                    new_config.append(f"anchors={anchors}\n")
                    continue

                new_config.append(line)

            # Check if extra [yolo] sections are being added
            if yolo_section_count == 0:
                raise ValueError("No [yolo] sections found in the CFG file!")

            with open(self.filename, 'w') as f:
                f.writelines(new_config)

            logger.info(f"Updated anchors in all {yolo_section_count} [yolo] sections.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred while updating anchors: {str(e)}")


    def clean_cfg_file(self):
        """
        Remove duplicate entries and maintain correct CFG structure.
        """
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()

            seen_params = set()
            clean_config = []
            inside_section = None

            for line in lines:
                stripped_line = line.strip()

                # Track section changes
                if stripped_line.startswith('['):
                    inside_section = stripped_line
                    clean_config.append(line)
                    continue

                # Skip duplicate parameters
                if stripped_line and "=" in stripped_line:
                    param_name = stripped_line.split('=')[0].strip()
                    if (inside_section, param_name) in seen_params:
                        continue
                    seen_params.add((inside_section, param_name))

                clean_config.append(line)

            # Write the cleaned config back to the file
            with open(self.filename, 'w') as f:
                f.writelines(clean_config)

            logger.info("Cleaned duplicates from the CFG file.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred while cleaning the CFG file: {str(e)}")


    def _iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes."""
        intersect_w = min(box1[0], box2[0])
        intersect_h = min(box1[1], box2[1])
        intersect = intersect_w * intersect_h
        union = box1[0] * box1[1] + box2[0] * box2[1] - intersect
        return intersect / union




    def plot_anchors(self, centroids, avg_iou):
        """Simplified anchor plotting."""
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_facecolor('white')

        colors = plt.cm.tab10(np.linspace(0, 1, len(centroids)))

        for i, centroid in enumerate(centroids):
            w, h = centroid
            rect = mpatches.Rectangle((0, 0), w, h, linewidth=2, edgecolor=colors[i], facecolor='none')
            ax.add_patch(rect)

        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title(f'Anchors (Avg IoU: {avg_iou:.2f}%)')
        plt.legend([f'Anchor {i+1}' for i in range(len(centroids))])
        plt.show()




    def saveAnchorsSettings(self, avg_iou):
        """Save the calculated anchor settings to a JSON file."""
        settings_file = 'settings.json'
        try:
            # Load existing settings from the file or initialize an empty dict
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
            else:
                settings = {}

            # Update settings with the new anchors and avg IoU
            settings['anchors'] = self.imported_anchors if self.imported_anchors else []
            settings['avg_iou'] = avg_iou

            # Write the updated settings to the file
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=4)

            logger.info(f"Anchor settings saved successfully with avg IoU: {avg_iou}%")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error saving anchor settings: {str(e)}")


    def main(self):
        """Main pipeline to import data and calculate anchors."""
        self.import_data()
        class_mapping = self.load_classes(self.image_directory)
        if not class_mapping:
            logger.warning("Classes not loaded. Exiting.")
            return

        if not self.text_files:
            logger.warning("No annotation files detected. Exiting.")
            return

        self.calculate_anchors()


    # yaml parser

    def clear_table(self):
        self.cfg_table.setRowCount(0)

    # Inside import_yaml method
    def import_yaml(self):
        """Handles the import of YAML configuration files."""
        if self.file_type != "yaml":
            self.clear_table()

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        initial_directory = self.default_yaml_path.text()
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Select YAML File", initial_directory, "YAML Files (*.yaml *.yml);;All Files (*)", options=options)

        if file_name:
            self.hide_activation_checkbox.setChecked(False)
            self.yaml_filename = file_name
            self.default_yaml_path.setText(file_name)
            
            # Parse YAML and populate the table
            with open(file_name, 'r', encoding="utf-8") as f:
                self.parsed_yaml = yaml.safe_load(f)

            self.file_type = "yaml"
            self.cfg_table.setColumnCount(2)
            self.cfg_table.setHorizontalHeaderLabels(["Key", "Value"])
            self.cfg_table.setRowCount(len(self.parsed_yaml))

            for row, (key, value) in enumerate(self.parsed_yaml.items()):
                self.cfg_table.setItem(row, 0, QTableWidgetItem(str(key)))
                self.cfg_table.setItem(row, 1, QTableWidgetItem(str(value)))

            self.filename = None  # Clear any previous CFG file reference
            self.cfg_open_label.setText("")

    def save_table_changes(self, row, column):
        key_item = self.cfg_table.item(row, 0)
        value_item = self.cfg_table.item(row, 1)

        if key_item is not None and value_item is not None:
            key = key_item.text()
            value = value_item.text()

            if self.file_type == "yaml" and self.yaml_filename is not None:
                self.parsed_yaml[key] = self.parse_value(value)
                with open(self.yaml_filename, 'w', encoding="utf-8") as f:
                    yaml.dump(self.parsed_yaml, f)

    def parse_value(self, value):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        return value

        # def for cfg editor
    def import_anchors(self):
        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self, "Error", "Please open a .cfg file before importing anchors.")
            return

    # Inside cfg_open_clicked method

    def cfg_open_clicked(self):
        """Handles opening a .cfg file."""
        self.hide_activation_checkbox.setChecked(False)

        if self.file_type != "cfg":
            self.clear_table()

        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly
        initial_directory = "C:/"
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Config File", initial_directory, "Config Files (*.cfg);;All Files (*)", options=options)

        if file_name:
            self.filename = file_name
            self.cfg_open_label.setText(f"Cfg: {file_name}")
            self.parse_cfg_file(file_name)
            self.yaml_filename = None  # Clear any YAML reference
            self.default_yaml_path.setText("")
            
    def toggle_activation_layers(self):
        # Check if 'activation_row_count' attribute exists
        if not hasattr(self, 'activation_row_count'):
            return

        for row in self.activation_row_count.values():
            if self.hide_activation_checkbox.isChecked():
                self.cfg_table.hideRow(row)
            else:
                self.cfg_table.showRow(row)

    def parse_cfg_file(self, file_name=None):
        if file_name is None:
            file_name = self.filename

        try:
            with open(file_name, 'r', encoding="utf-8") as f:
                config = f.read()

            self.activation_row_count = {}
            activation_count = 0
            self.cfg_table.setRowCount(0)
            self.cfg_table.setColumnCount(2)
            self.cfg_table.setHorizontalHeaderLabels(["Parameter", "Value"])
            self.cfg_table.setAlternatingRowColors(True)

            sections = re.findall(r"\[(.*?)\]\s*([^[]*)", config, re.DOTALL)

            # Count YOLO layers
            yolo_layers = len([sec for sec, _ in sections if sec == "yolo"])

            # Define masks based on the number of YOLO layers
            yolo_mask_values = {
                5: [[12, 13, 14], [9, 10, 11], [6, 7, 8], [3, 4, 5], [0, 1, 2]],
                4: [[9, 10, 11], [6, 7, 8], [3, 4, 5], [0, 1, 2]],
                3: [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                2: [[3, 4, 5], [0, 1, 2]],
                1: [[0, 1, 2]]
            }.get(yolo_layers, [])

            yolo_section_idx = 0

            if self.imported_anchors is not None:
                for idx, (section_type, section_content) in enumerate(sections):
                    if section_type == "yolo":
                        section_lines = section_content.strip().split("\n")
                        section_dict = {line.split("=")[0].strip(): line.split("=")[1].strip()
                                        for line in section_lines if "=" in line}

                        section_dict["anchors"] = ', '.join(
                            [f"{x},{y}" for x, y in self.imported_anchors])

                        sections[idx] = (section_type, '\n'.join(
                            [f"{key} = {value}" for key, value in section_dict.items()]))

            for idx, (section_type, section_content) in enumerate(sections):
                section_lines = section_content.strip().split("\n")
                section_dict = {line.split("=")[0].strip(): line.split("=")[1].strip()
                                for line in section_lines if "=" in line}

                # Handling net section
                if section_type == "net":
                    net_items = ["batch", "subdivisions", "width", "height", "saturation", "exposure", "hue",
                                "max_batches", "flip", "mosaic", "letter_box", "cutmix", "mosaic_bound",
                                "mosaic_scale", "mosaic_center", "mosaic_crop", "mosaic_flip", "steps",
                                "scales", "classes"]

                    for item in net_items:
                        if item in section_dict:
                            value_without_comment = section_dict[item].split('#')[0].strip()
                            if not value_without_comment:
                                continue

                            row_count = self.cfg_table.rowCount()
                            self.cfg_table.insertRow(row_count)
                            self.cfg_table.setItem(row_count, 0, QtWidgets.QTableWidgetItem(f"{item}_0"))
                            self.cfg_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(value_without_comment))

                            # Highlight specific parameters
                            if item in {"batch", "width", "height", "subdivisions", "max_batches"}:
                                self.cfg_table.item(row_count, 0).setBackground(QtGui.QColor(255, 255, 144))
                                self.cfg_table.item(row_count, 1).setBackground(QtGui.QColor(255, 255, 144))

                    self.net_dict = section_dict

                # Handling convolutional section
                elif section_type == "convolutional":
                    is_before_yolo = idx < len(sections) - 1 and sections[idx + 1][0] == "yolo"
                    conv_items = ["activation"]

                    for item in conv_items:
                        if item in section_dict and (is_before_yolo or item != "filters"):
                            row_count = self.cfg_table.rowCount()
                            self.cfg_table.insertRow(row_count)
                            self.cfg_table.setItem(row_count, 0, QtWidgets.QTableWidgetItem(f"{item}_{idx}"))

                            if item == "activation":
                                activation_combo = QtWidgets.QComboBox()
                                activation_combo.addItems(["leaky", "mish", "swish", "linear"])
                                activation_combo.setCurrentText(section_dict[item])
                                self.cfg_table.setCellWidget(row_count, 1, activation_combo)
                                self.activation_row_count[activation_count] = row_count
                                activation_count += 1
                            else:
                                self.cfg_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(section_dict[item]))

                # Handling yolo section
                elif section_type == "yolo":
                    yolo_items = ["mask", "anchors", "num", "classes", "ignore_thresh", "random"]

                    # Set mask values based on yolo_section_idx
                    if yolo_section_idx < len(yolo_mask_values):
                        mask_values = yolo_mask_values[yolo_section_idx]
                        section_dict["mask"] = ','.join(map(str, mask_values))

                    for item in yolo_items:
                        if item in section_dict:
                            if item == "num":
                                num_clusters = int(section_dict[item])
                                self.clusters_spinbox.setValue(num_clusters)

                            row_count = self.cfg_table.rowCount()
                            self.cfg_table.insertRow(row_count)
                            self.cfg_table.setItem(row_count, 0, QtWidgets.QTableWidgetItem(f"{item}_{idx}"))

                            if item == "anchors" and self.imported_anchors is not None:
                                self.cfg_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(
                                    ','.join([f"{x},{y}" for x, y in self.imported_anchors])))
                            else:
                                self.cfg_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(section_dict[item]))

                            # Highlight the 'classes' parameter
                            if item == "classes":
                                self.cfg_table.item(row_count, 0).setBackground(QtGui.QColor(255, 255, 144))
                                self.cfg_table.item(row_count, 1).setBackground(QtGui.QColor(255, 255, 144))

                    yolo_section_idx += 1

            self.cfg_table.resizeColumnsToContents()
            self.cfg_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred while parsing the CFG file: {str(e)}")

    def cfg_save_clicked(self):
        if self.filename:
            table_data = {}
            for row in range(self.cfg_table.rowCount()):
                param = self.cfg_table.item(row, 0).text()
                cell_widget = self.cfg_table.cellWidget(row, 1)
                value = cell_widget.currentText() if isinstance(cell_widget, QtWidgets.QComboBox) else self.cfg_table.item(row, 1).text()
                table_data[param] = value

            new_config = ""
            section_idx = -1
            previous_line_empty = False

            with open(self.filename, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                stripped_line = line.strip()

                # Start of a new section
                if stripped_line.startswith("["):
                    section_idx += 1

                    # Ensure exactly one empty line before a new section
                    if not previous_line_empty:
                        new_config += "\n"

                    new_config += f"{stripped_line}\n"
                    previous_line_empty = False
                    continue

                # Parameter line
                elif "=" in stripped_line and not stripped_line.startswith("#"):
                    param, old_value = stripped_line.split("=", 1)
                    param = param.strip()
                    new_value = table_data.get(f"{param}_{section_idx}", old_value.strip())
                    new_config += f"{param}={new_value}\n"
                    previous_line_empty = False
                    continue

                # Comment or other lines (preserved as is, except for excess whitespace)
                elif stripped_line.startswith("#"):
                    new_config += f"{stripped_line}\n"
                    previous_line_empty = False
                    continue

                # Empty lines
                if stripped_line == "":
                    if not previous_line_empty:  # Allow a single empty line
                        new_config += "\n"
                        previous_line_empty = True
                else:
                    previous_line_empty = False

            # Trim leading/trailing whitespace/newlines
            new_config = new_config.strip() + "\n"

            # Save updated config
            options = QtWidgets.QFileDialog.Options()
            save_file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Config File As", os.path.expanduser("~"),
                "Config Files (*.cfg);;All Files (*)", options=options)

            if save_file_name:
                if not save_file_name.endswith('.cfg'):
                    save_file_name += '.cfg'

                try:
                    with open(save_file_name, 'w', encoding='utf-8') as f:
                        f.write(new_config)

                    QtWidgets.QMessageBox.information(self, "Success", "Configuration file saved successfully.")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Error", str(e))




    def new_method(self, text):
        # Return a QTableWidgetItem instance with the text
        return QtWidgets.QTableWidgetItem(text)

     # combine txt files

    def on_combine_txt_clicked(self):
        if self.combine_txt_flag:
            logger.info("Function is already running.")
            return
        self.combine_txt_flag = True
        logger.info("Function called.")
        self.combine_txt_button.setEnabled(False)

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file1, _ = QFileDialog.getOpenFileName(
            self, "Select First File", "", "Text Files (*.txt);;All Files (*)", options=options
        )
        if not file1:
            self.combine_txt_flag = False
            logger.warning("File 1 prompt cancelled.")
            self.combine_txt_button.setEnabled(True)
            return
        file2, _ = QFileDialog.getOpenFileName(
            self, "Select Second File", "", "Text Files (*.txt);;All Files (*)", options=options
        )
        if not file2:
            self.combine_txt_flag = False
            logger.warning("File 2 prompt cancelled.")
            self.combine_txt_button.setEnabled(True)
            return
        output_file, _ = QFileDialog.getSaveFileName(
            self, "Save Combined File", "", "Text Files (*.txt);;All Files (*)", options=options
        )
        if not output_file:
            self.combine_txt_flag = False
            logger.warning("Output file prompt cancelled.")
            self.combine_txt_button.setEnabled(True)
            return

        try:
            with open(file1, "r") as f1, open(file2, "r") as f2, open(output_file, "w") as output:
                for line in f1:
                    output.write(line)
                for line in f2:
                    output.write(line)

            QMessageBox.information(
                self, "Success", "Files have been combined and saved successfully!")
            self.combine_txt_flag = False
            logger.info("Function finished successfully.")
            self.combine_txt_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while combining files: {e}")
            self.combine_txt_flag = False
            logger.error("Function finished with error.")
            self.combine_txt_button.setEnabled(True)


def run_pyqt_app():
    # Check if QApplication already exists
    app = QtWidgets.QApplication.instance()
    if app is None:  # If no instance exists, create a new one
        app = QtWidgets.QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()  # Ensure it is visible

    exit_code = app.exec_()  # Start the event loop

    # Cleanup
    app.deleteLater()
    sys.exit(exit_code)

if __name__ == "__main__":
    # Use cProfile to profile the execution of the 'run_pyqt_app()' function
    cProfile.run('run_pyqt_app()')
