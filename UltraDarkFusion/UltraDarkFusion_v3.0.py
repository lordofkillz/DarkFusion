import multiprocessing
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
import os
import cv2
# Import necessary libraries and modules for the application
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler
import concurrent.futures
import cProfile
import ctypes
from datetime import datetime, timedelta
from typing import List, Tuple
import glob
import json
import threading
import queue
import random
import re
from noise import snoise2 as perlin2
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
import copy
import psutil
import GPUtil
from threading import Thread
import functools
from PIL import Image
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtCore import (QCoreApplication, QEvent, QModelIndex, QObject,
                          QPoint, QRectF, QRunnable, Qt, QThread, QThreadPool,
                          QTimer, QUrl, pyqtSignal, pyqtSlot, QPointF,QModelIndex,Qt,QEvent,QPropertyAnimation, QEasingCurve)
from PyQt5.QtGui import (QBrush, QColor, QFont, QImage, QImageReader,
                         QImageWriter, QKeySequence, QMovie, QPainter, QPen,
                         QPixmap,  QStandardItem,
                         QStandardItemModel, QTransform, QLinearGradient,QIcon,QCursor,QStandardItemModel, QStandardItem,QMouseEvent,QKeyEvent)
from PyQt5.QtWidgets import (QApplication, QFileDialog,
                             QGraphicsDropShadowEffect, QGraphicsItem,
                             QGraphicsPixmapItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsTextItem, QGraphicsView,
                             QLabel, QMessageBox, QProgressBar,
                             QTableWidgetItem, QColorDialog, QMenu,QSplashScreen,QTableView, QVBoxLayout,QWidget,QHeaderView,QHBoxLayout,QCheckBox,QStyledItemDelegate,QStyle,QTabWidget,QStyleOptionButton ) 
from pytube import YouTube
from PyQt5 import QtWidgets, QtGui
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.cluster import KMeans
from qt_material import apply_stylesheet, list_themes
from segment_anything import sam_model_registry, SamPredictor
import pybboxes as pbx
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from noise import pnoise2
from dino import run_groundingdino
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
import webbrowser
import threading
from watchdog.events import FileSystemEventHandler
from PIL import Image
from rectpack import newPacker
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QAction
from sahi_predict_wrapper import SahiPredictWrapper
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QFileDialog,QComboBox,QDoubleSpinBox)
from torch.cuda.amp import autocast
from PyQt5.QtWidgets import QAbstractItemView
from PyQt5.QtCore import QItemSelectionModel
from PyQt5.QtCore import pyqtSlot, QModelIndex
from functools import lru_cache
from PyQt5.QtCore import QThread, pyqtSignal
import uuid
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from tqdm import tqdm
from torchvision import transforms
from PIL import ImageOps
from sklearn.cluster import MiniBatchKMeans
import validators
from urllib.parse import urlparse, urlunparse
from yt_dlp import YoutubeDL
from pytube.exceptions import VideoUnavailable
from PIL import ImageOps, Image
import gc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict
import tensorrt as trt
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Setup logger configuration
def setup_logger():
    logger = logging.getLogger('UltraDarkFusionLogger')

    # Remove any existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set up debug directory if not exists
    if not os.path.exists('debug'):
        os.makedirs('debug')

    # Set up file handler with a rotating file log
    file_handler = RotatingFileHandler(
        os.path.join('debug', 'my_app.log'),
        maxBytes=10000,
        backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    file_handler.setFormatter(file_formatter)

    # Set up console handler for logging without ANSI codes
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Set logging level and prevent propagation
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return logger

# Initialize logger
logger = setup_logger()

# Redirect TensorRT logs to UltraDarkFusionLogger
class TensorRTLogger(trt.ILogger):
    def __init__(self):
        super().__init__()

    def log(self, severity, msg):
        # Map TensorRT severity levels to Python logging levels
        if severity == trt.ILogger.Severity.ERROR:
            logger.error(f"TensorRT: {msg}")
        elif severity == trt.ILogger.Severity.WARNING:
            logger.warning(f"TensorRT: {msg}")
        elif severity == trt.ILogger.Severity.INFO:
            logger.info(f"TensorRT: {msg}")
        else:
            logger.debug(f"TensorRT: {msg}")

# Use custom TensorRT logger
tensorrt_logger = TensorRTLogger()

# Function to get engine input size
def get_engine_input_size(engine_path):
    try:
        with open(engine_path, "rb") as f, trt.Runtime(tensorrt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            binding_shape = engine.get_binding_shape(0)  # Get shape for the first input
            logger.info(f"Successfully retrieved input size from {engine_path}: {binding_shape}")
            return binding_shape  # Returns something like [1, 3, 416, 416]
    except Exception as e:
        logger.error(f"Failed to get engine input size from {engine_path}: {e}")
        return None




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
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window

        # Set up initial state and render settings
        self._setup_initial_state()
        self._setup_render_settings()

        # Create and configure sound player
        self.sound_player = QMediaPlayer()
        self.sound_player.setMuted(True)
        self.main_window.muteCheckBox.setChecked(True)  # type: ignore
        self.main_window.muteCheckBox.stateChanged.connect(self.mute_player)  # type: ignore

        # Initialize additional variables
        self.zoom_scale = 1.0
        self.show_crosshair = False
        self.right_click_timer = QTimer()
        self.right_click_timer.setInterval(100)
        self.right_click_timer.timeout.connect(self.remove_box_under_cursor)

        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setBlurRadius(10)
        self.setGraphicsEffect(shadow)

        # Enable mouse tracking and configure view settings
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(
            QPainter.Antialiasing | 
            QPainter.SmoothPixmapTransform | 
            QPainter.HighQualityAntialiasing | 
            QPainter.TextAntialiasing
        )
       
        self.setOptimizationFlags(QGraphicsView.DontSavePainterState)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # type: ignore
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # type: ignore
        self.setFrameShape(QGraphicsView.NoFrame)


    def _setup_initial_state(self):
        self.drawing = False
        self.start_point = None
        self.current_bbox = None
        self.selected_bbox = None
        self.moving_view = False
        self.last_mouse_position = None
        self.mode = "normal"
        self.bboxes = []
        self.clipboard = None
        self.copy_mode = False
        self.dragStartPos = None
        self.double_click_held = False
        self.create_bbox_on_click = True
        self.crosshair_position = QPointF()
        self.xy_lines_checkbox = self.main_window.xy_lines_checkbox  # type: ignore
        self.xy_lines_checkbox.toggled.connect(self.toggle_crosshair)
        self.main_window.crosshair_color.triggered.connect(self.pick_color)  # type: ignore 
        self.main_window.box_size.valueChanged.connect(self.update_bbox_size)  # type: ignore
        self.graphics_scene = QGraphicsScene(self)
    
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
        """Safely removes an item from the scene if it belongs to the scene."""
        if item and item.scene() == self.scene():
            self.scene().removeItem(item)
        else:
            print("Attempted to remove an item that does not belong to the current scene or is None.")

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        right_click_pos = self.mapToScene(event.pos())
        last_bbox = self._get_last_drawn_bbox()

        if last_bbox and last_bbox.contains(right_click_pos):
            delete_action = menu.addAction("Delete Last Drawn Box")
            action = menu.exec_(event.globalPos())

            if action == delete_action:
                self._remove_last_drawn_bbox(last_bbox)

    def remove_box_under_cursor(self):
        cursor_pos = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
        for item in self.scene().items(cursor_pos):
            if isinstance(item, BoundingBoxDrawer):
                self._play_sound_and_remove_bbox(item)
                if hasattr(self.main_window, 'auto_save_bounding_boxes'):
                    self.main_window.auto_save_bounding_boxes()  # type: ignore
                break

    def _get_last_drawn_bbox(self):
        if self.bboxes:
            return self.bboxes[-1]

    def _remove_last_drawn_bbox(self, bbox):
        if bbox:
            self.scene().removeItem(bbox)
            self.scene().removeItem(bbox.class_name_item)
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

        # Set zoom in and zoom out factors
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
            # Calculate dynamic minimum zoom scale based on image or scene dimensions
            min_zoom_scale = max(self.fitInView_scale, 0.1)  # Prevent zooming out too far
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
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)  # type: ignore
        self.fitInView_scale = self.transform().mapRect(QRectF(0, 0, 1, 1)).width()
        self.setTransform(QTransform().scale(self.zoom_scale, self.zoom_scale))
        self.update()

    def mousePressEvent(self, event):
        self.last_mouse_pos = self.mapToScene(event.pos())

        # Check if there is an image in the scene before proceeding
        if self.scene() is None or self.scene().itemsBoundingRect().isEmpty():
            print("No image loaded.")
            return

        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if isinstance(item, BoundingBoxDrawer):
                self._bring_to_front(item)
                item.set_selected(True)
                if self.selected_bbox and self.selected_bbox != item:
                    self.selected_bbox.set_selected(False)
                self.selected_bbox = item
            else:
                self._start_drawing(event)
        elif event.button() == Qt.RightButton:
            self._handle_right_button_press(event)
            self.right_click_timer.start()
        else:
            super().mousePressEvent(event)

    def _start_drawing(self, event):
        self.drawing = True
        self.start_point = self.mapToScene(event.pos())
        self.current_bbox = BoundingBoxDrawer(
            self.start_point.x(), self.start_point.y(), 0, 0,
            main_window=self.main_window, class_id=self.main_window.get_current_class_id()
        )
        self.scene().addItem(self.current_bbox)

    def _bring_to_front(self, item):
        item.setZValue(1)
        for other_item in self.scene().items():
            if other_item != item and isinstance(other_item, BoundingBoxDrawer):
                other_item.setZValue(0.5)
        item.setSelected(True)

    def _handle_right_button_press(self, event):
        click_pos = self.mapToScene(event.pos())
        tolerance = 10

        for item in self.scene().items():
            if isinstance(item, BoundingBoxDrawer):
                if item.rect().adjusted(-tolerance, -tolerance, tolerance, tolerance).contains(click_pos):
                    self._play_sound_and_remove_bbox(item)
                    break

    def _play_sound_and_remove_bbox(self, item):
        self.set_sound('sounds/shotgun.wav')
        self.sound_player.play()
        self.safe_remove_item(item)

    def save_bounding_boxes(self, label_file, scene_width, scene_height):
        try:
            with open(label_file, 'w') as file:
                for bbox in self.collect_bboxes_from_scene(scene_width, scene_height):
                    file.write(f"{bbox}\n")
            QMessageBox.information(None, "Success", "Bounding boxes saved successfully.")
            self._re_add_bounding_boxes()
        except IOError as e:
            QMessageBox.critical(None, "Error", f"Failed to save bounding boxes: {e}")

    def _re_add_bounding_boxes(self):
        for bbox in self.bboxes:
            self.scene().addItem(bbox)
            bbox.setZValue(0.5)
        if self.selected_bbox:
            self.selected_bbox.setZValue(1)

    def mouseMoveEvent(self, event):
        if self.show_crosshair:
            self._update_crosshair(event.pos())
        if self.moving_view:
            self._handle_moving_view(event)
        elif self.drawing and self.current_bbox:
            self._handle_drawing_bbox(event)
        elif self.selected_bbox and not self.drawing:
            self.selected_bbox.mouseMoveEvent(event)
        else:
            super().mouseMoveEvent(event)

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
            center_x = self.crosshair_position.x()
            center_y = self.crosshair_position.y()
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
            x, y = self._get_bbox_coordinates(end_point)
            width, height = self._get_bbox_dimensions(end_point, x, y)
            self.current_bbox.setRect(x, y, width, height)
        except RuntimeError as e:
            logging.error(f"Error while drawing bounding box: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in bounding box drawing: {e}")

    def _get_bbox_coordinates(self, end_point):
        x = max(0, min(self.start_point.x(), end_point.x()))
        y = max(0, min(self.start_point.y(), end_point.y()))
        return x, y

    def _get_bbox_dimensions(self, end_point, x, y):
        width = min(max(BoundingBoxDrawer.MIN_SIZE, abs(self.start_point.x() - end_point.x())), self.scene().width() - x)
        height = min(max(BoundingBoxDrawer.MIN_SIZE, abs(self.start_point.y() - end_point.y())), self.scene().height() - y)
        return width, height

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

        if event.button() == Qt.LeftButton:
            self._handle_left_button_release(event)
        if event.button() == Qt.RightButton:
            self.right_click_timer.stop()

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
            print(f"An error occurred: {e}")
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
        if self.selected_bbox and self.selected_bbox.boundingRect().isValid():
            self.selected_bbox.set_selected(False)
        self.selected_bbox = None
        self.viewport().setCursor(Qt.ArrowCursor)

    def itemAt(self, pos):
        return None

    def toggle_crosshair(self, checked):
        self.show_crosshair = checked
        self.viewport().update()




class BoundingBoxDrawer(QGraphicsRectItem):
    MIN_SIZE = 6
    MAX_SIZE = 100

    def __init__(self, x, y, width, height, main_window, class_id=None, confidence=None, unique_id=None):
        super().__init__(x, y, width, height)
        self.unique_id = unique_id
        self.main_window = main_window
        self.class_id = 0 if class_id is None else class_id
        self.confidence = confidence
        self.dragStartPos = None
        self.final_pos = None
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsFocusable)
        self.setPen(QPen(self.get_color(self.class_id, main_window.classes_dropdown.count()), 2))
        self.setZValue(0.5)
        self.class_name_item = QGraphicsTextItem(self)
        self.update_class_name_item()
        self.setAcceptHoverEvents(True)
        self.set_selected(False)
        self.hover_opacity = 1.0
        self.normal_opacity = 0.6
        self.flash_color = QColor(255, 0, 0)
        self.alternate_flash_color = QColor(0, 0, 255)
        self.flash_timer = QTimer()
        self.flash_timer.timeout.connect(self.toggle_flash_color)
        self.scroll_timer = QTimer()
        self.scroll_timer.setSingleShot(True)
        self.scroll_timer.timeout.connect(self.stop_flashing)
        self.setVisible(self.main_window.class_visibility.get(self.class_id, True))
        
        
    def start_flashing(self, interval, duration):
        self.flash_timer.start(interval)
        QTimer.singleShot(duration, self.stop_flashing)

    def stop_flashing(self):
        self.flash_timer.stop()
        self.setPen(QPen(self.get_color(self.class_id, self.main_window.classes_dropdown.count()), 2))

    def toggle_flash_color(self):
        current_color = self.pen().color()
        self.setPen(QPen(self.flash_color if current_color == self.alternate_flash_color else self.alternate_flash_color, 2))

    @staticmethod
    def get_color(class_id, num_classes):
        if num_classes == 0:
            logging.error("Number of classes should not be zero.")
            return QColor(255, 255, 255)  # Default color in case of error
        num_classes = min(num_classes, 100)
        hue_step = 360 / num_classes
        hue = (class_id * hue_step) % 360
        return QColor.fromHsv(hue, 255, 255)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self.main_window.shade_checkbox.isChecked():
            shade_value = self.main_window.shade_slider.value()
            color = self.pen().color()
            mapped_shade_value = int((shade_value / 100) * 255) if self.confidence is not None else shade_value
            shaded_color = QColor(color.red(), color.green(), color.blue(), mapped_shade_value)
            painter.setBrush(shaded_color)
            painter.drawRect(self.rect())
        if not self.main_window.hide_label_checkbox.isChecked():
            label_rect = self.class_name_item.boundingRect()
            label_rect.moveTopLeft(self.class_name_item.pos())
            painter.setBrush(QColor(0, 0, 0, 127))
            painter.drawRect(label_rect)

    def update_class_name_item(self):
        if not self.main_window:
            return
        full_text = self.get_formatted_class_text()
        self.class_name_item.setPlainText(full_text)
        self.update_class_color_and_position()
        self.update_class_name_item_font()

    def get_formatted_class_text(self):
        class_name = self.main_window.classes_dropdown.itemText(self.class_id)
        if self.confidence:
            return f"ID: {self.class_id} | {class_name} | Conf: {self.confidence*100:.2f}%"
        return f"ID: {self.class_id} | {class_name}"

    def update_class_color_and_position(self):
        offset = 14
        position_x, position_y = self.rect().x(), self.rect().y() - offset
        self.class_name_item.setPos(position_x, position_y)
        self.class_name_item.setDefaultTextColor(QColor(255, 255, 255))

    def hoverEnterEvent(self, event):
        self.setOpacity(self.hover_opacity)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setOpacity(self.normal_opacity)
        super().hoverLeaveEvent(event)

    def update_class_name_item_font(self):
        font = self.class_name_item.font()
        font.setPointSize(self.main_window.font_size_slider.value() + 1)
        self.class_name_item.setFont(font)

    def set_class_id(self, class_id):
        self.class_id = class_id
        self.update_class_name_item()

    def wheelEvent(self, event):
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            rect = self.rect()
            delta = 1 if event.delta() > 0 else -1  # Changed angleDelta to delta for compatibility
            new_width = min(max(self.MIN_SIZE, rect.width() + delta), self.scene().width() - rect.x())
            new_height = min(max(self.MIN_SIZE, rect.height() + delta), self.scene().height() - rect.y())
            new_x = max(0, min(rect.x() - delta / 2, self.scene().width() - new_width))
            new_y = max(0, min(rect.y() - delta / 2, self.scene().height() - new_height))
            self.setRect(new_x, new_y, new_width, new_height)
            self.main_window.save_bounding_boxes(self.main_window.label_file, self.scene().width(), self.scene().height())
            self.start_flashing(300, 1000)
        else:
            super().wheelEvent(event)

    def update_bbox(self):
        rect = self.rect()
        if self.scene():
            img_width = self.scene().width()
            img_height = self.scene().height()
            x_center = (rect.x() + rect.width() / 2) / img_width
            y_center = (rect.y() + rect.height() / 2) / img_height
            width = rect.width() / img_width
            height = rect.height() / img_height
            self.bbox = BoundingBox(self.class_id, x_center, y_center, width, height, self.confidence)

    def set_selected(self, selected):
        self.setSelected(selected)
        self.setZValue(1 if selected else 0.5)
        # Highlight bounding box when selected
        if selected:
            self.setPen(QPen(QColor(0, 255, 0), 2))  # Green color for selected box
        else:
            self.setPen(QPen(self.get_color(self.class_id, self.main_window.classes_dropdown.count()), 2))

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and not event.modifiers() & Qt.ControlModifier:
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.dragStartPos = event.pos() - self.rect().topLeft()
            self.setPen(QPen(QColor(0, 255, 0), 2))
        else:
            super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            if self.rect().contains(self.mapFromScene(event.scenePos())):
                self.set_selected(True)
                self.copy_and_create_new_bounding_box()
        else:
            super().mousePressEvent(event)

    def copy_and_create_new_bounding_box(self):
        copied_bbox = self
        new_bbox = BoundingBoxDrawer(
            copied_bbox.rect().x() + 10, copied_bbox.rect().y() + 10,
            copied_bbox.rect().width(), copied_bbox.rect().height(),
            self.main_window, copied_bbox.class_id
        )
        self.scene().addItem(new_bbox)
        new_bbox.setSelected(True)
        QTimer.singleShot(500, new_bbox.reset_color)

    def reset_color(self):
        self.setPen(QPen(self.get_color(self.class_id, self.main_window.classes_dropdown.count()), 2))

    def mouseMoveEvent(self, event):
        if self.dragStartPos is None:
            return
        newPos = event.pos() - self.dragStartPos
        newRect = QRectF(newPos, self.rect().size())
        if self.scene() is not None:
            maxLeft = 0
            maxTop = 0
            maxRight = self.scene().width() - newRect.width()
            maxBottom = self.scene().height() - newRect.height()
            newRect.moveLeft(max(maxLeft, min(newRect.left(), maxRight)))
            newRect.moveTop(max(maxTop, min(newRect.top(), maxBottom)))
        self.setRect(newRect)
        if self.isSelected:
            self.setZValue(1.0)
            for item in self.scene().items():
                if item != self and isinstance(item, BoundingBoxDrawer):
                    item.setZValue(0.5)
        else:
            self.setZValue(0.5)
        self.update_class_name_position()

    def update_class_name_position(self):
        offset = 14
        new_label_pos = QPointF(self.rect().x(), self.rect().y() - offset)
        self.class_name_item.setPos(new_label_pos)
        self.class_name_item.update()

    def mouseReleaseEvent(self, event):
        if self.dragStartPos is not None:
            self.final_rect = self.rect()
            self.dragStartPos = None
            self.update_bbox()  # Changed to update_bbox to avoid AttributeError
            self.setFlag(QGraphicsItem.ItemIsMovable, False)
            self.setPen(QPen(self.get_color(self.class_id, self.main_window.classes_dropdown.count()), 2))
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            self.set_selected(False)
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.setFlag(QGraphicsItem.ItemIsFocusable, True)
            self.setZValue(0.5)
        super().mouseReleaseEvent(event)

class BoundingBox:
    def __init__(self, class_id, x_center, y_center, width, height, confidence=None):
        self.class_id = class_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.confidence = confidence
        self.initial_mouse_pos = None

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
    def from_str(bbox_str):
        values = list(map(float, bbox_str.strip().split()))
        if len(values) >= 5:
            class_id, x_center, y_center, width, height = values[:5]
            confidence = values[5] if len(values) > 5 else None
            return BoundingBox(int(class_id), x_center, y_center, width, height, confidence)
        else:
            return None

    def to_str(self):
        bbox_str = "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
            self.class_id, self.x_center, self.y_center, self.width, self.height)
        if self.confidence is not None:
            bbox_str += " {:.6f}".format(self.confidence)
        return bbox_str



# Creates a graphical user interface (GUI) for the settings dialog.
class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        logging.info("SettingsDialog instance created.")
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("Settings")  # Set the window title
        self.resize(300, 200)  # Resize the dialog window

        layout = QtWidgets.QVBoxLayout(self)  # Create a vertical layout for the dialog

        scrollArea = QtWidgets.QScrollArea(self)  # Create a scrollable area
        scrollContent = QtWidgets.QWidget(scrollArea)  # Create a scrollable content widget
        scrollLayout = QtWidgets.QVBoxLayout(scrollContent)  # Create a vertical layout for scrollable content

        # Create hotkey input fields for each class
        self.classHotkeyInputs = {}
        for i in range(self.parent().classes_dropdown.count()):
            className = self.parent().classes_dropdown.itemText(i)
            self.classHotkeyInputs[className] = QtWidgets.QLineEdit(
                self.parent().settings.get('classHotkey_{}'.format(className), '')
            )
            scrollLayout.addWidget(QtWidgets.QLabel("Hotkey for class {}: ".format(className)))  # Add class label
            scrollLayout.addWidget(self.classHotkeyInputs[className])  # Add hotkey input field

        scrollContent.setLayout(scrollLayout)  # Set the layout for scrollable content
        scrollArea.setWidget(scrollContent)  # Set the scrollable content widget

        layout.addWidget(scrollArea)  # Add the scrollable area to the main layout

        # Create and add QLabel and QLineEdit widgets for various settings
        self.nextButtonInput = QtWidgets.QLineEdit(self.parent().settings.get('nextButton', ''))
        self.previousButtonInput = QtWidgets.QLineEdit(self.parent().settings.get('previousButton', ''))
        self.deleteButtonInput = QtWidgets.QLineEdit(self.parent().settings.get('deleteButton', ''))
        self.autoLabelInput = QtWidgets.QLineEdit(self.parent().settings.get('autoLabel', ''))

        layout.addWidget(QtWidgets.QLabel("Next Button:"))  # Add label for Next Button setting
        layout.addWidget(self.nextButtonInput)  # Add Next Button input field
        layout.addWidget(QtWidgets.QLabel("Previous Button:"))  # Add label for Previous Button setting
        layout.addWidget(self.previousButtonInput)  # Add Previous Button input field
        layout.addWidget(QtWidgets.QLabel("Delete Button:"))  # Add label for Delete Button setting
        layout.addWidget(self.deleteButtonInput)  # Add Delete Button input field

        saveButton = QtWidgets.QPushButton("Save")  # Create a "Save" button
        saveButton.clicked.connect(self.saveSettings2)  # Connect button click event to saveSettings method
        layout.addWidget(saveButton)  # Add the "Save" button to the layout

        self.setLayout(layout)  # Set the final layout for the settings dialog

    def saveSettings2(self):
        self.parent().settings['nextButton'] = self.nextButtonInput.text()  # Save Next Button setting
        self.parent().settings['previousButton'] = self.previousButtonInput.text()  # Save Previous Button setting
        self.parent().settings['deleteButton'] = self.deleteButtonInput.text()  # Save Delete Button setting

        # Save the hotkeys for each class
        for className, inputField in self.classHotkeyInputs.items():
            self.parent().settings['classHotkey_{}'.format(className)] = inputField.text()

        self.parent().saveSettings()  # Trigger the application to save all settings
        self.accept()  # Accept and close the settings dialog



# This class connects video processing with the main window.
class VideoProcessor(QtCore.QObject):
    progress_updated = pyqtSignal(int)  # Signal to indicate progress updates

    def process_video(self, video_path):
        pass  # Placeholder for video processing logic

    def get_image_extension(self, image_format):
        format_mapping = {"*.JPG": ".jpg", "*.JPEG": ".jpeg",
                          "*.GIF": ".gif", "*.BMP": ".bmp", "*.PNG": ".png"}
        return format_mapping.get(image_format, ".jpg")


# The video extractor class, a subclass of VideoProcessor.
class GUIVideoProcessor(VideoProcessor):
    def __init__(self):
        super().__init__()
        self.videos = []  # List of video file paths to be processed
        self.extract_all_frames = False  # Flag indicating whether to extract all frames
        self.custom_frame_count = None  # Custom frame count for extraction (if not 'None')
        self.custom_size = None  # Custom frame size (width, height)
        self.image_format = None  # Image format for saving extracted frames
        self.stop_processing = False  # Flag to stop the processing
        self.output_path = ""  # Path to save the extracted frames
        self.label_progress = QProgressBar()  # Progress bar for frame extraction

    def run(self):
        self.label_progress.reset()
        self.label_progress.setMinimum(0)
        self.label_progress.setMaximum(len(self.videos))
        for index, video in enumerate(self.videos):
            self.process_video(video)
            self.label_progress.setValue(index + 1)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Video frame extraction finished.")
        msg.exec_()

    def add_video(self, video):
        self.videos.append(video)

    def remove_video(self):
        return self.videos.pop()

    def set_extract_all_frames(self, value):
        self.extract_all_frames = value

    def set_custom_frame_count(self, count):
        self.custom_frame_count = count

    def set_custom_size(self, size):
        self.custom_size = size

    def set_image_format(self, image_format):
        self.image_format = image_format

    def set_output_path(self, path):
        self.output_path = path

    def stop(self):
        self.stop_processing = True

    def run(self):
        for video in self.videos:
            self.process_video(video)


    def process_video(self, video_path):
        # Open the video file for processing
        video = cv2.VideoCapture(video_path)

        # Get the total number of frames in the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine the frame extraction rate based on user settings
        if self.extract_all_frames:
            extract_rate = 1
        else:
            if self.custom_frame_count is not None and self.custom_frame_count > 0:
                extract_rate = self.custom_frame_count
            else:
                extract_rate = 1

        # Extract the video name from the file path
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Create an output directory for extracted frames
        output_dir = os.path.join(
            self.output_path, f'{video_name}_Extracted Frames')
        os.makedirs(output_dir, exist_ok=True)

        # Initialize variables for frame counting
        frame_count = 0

        # Loop through the video frames
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Resize the frame if a custom size is specified
            if self.custom_size is not None:
                frame = cv2.resize(frame, self.custom_size)

            # Check if the current frame should be extracted based on the rate
            if frame_count % extract_rate == 0:
                frame_output_path = f"{output_dir}/frame_{frame_count}{self.get_image_extension(self.image_format)}"
                cv2.imwrite(frame_output_path, frame,)

            # Update the frame count and progress
            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            self.progress_updated.emit(progress)

            # Check if processing should be stopped
            if self.stop_processing:
                break

        # Release the video file
        video.release()



# Custom exception class for handling invalid annotations
class InvalidAnnotationError(Exception):
    pass

# Custom exception class for handling invalid images
class InvalidImageError(Exception):
    pass

class ScanAnnotations:
    def __init__(self, parent):
        self.parent = parent
        self.valid_classes = []
        self.base_directory = ""
        self.total_images = 0
        self.total_labels = 0
        self.blank_labels = 0
        self.bad_labels = 0
        self.bad_images = 0
        self.review_folder = "review"
        self.metadata_queue = queue.Queue()
        self.metadata_thread = threading.Thread(target=self.metadata_removal_thread)
        self.metadata_thread.daemon = True
        self.metadata_thread.start()
    
    def remove_metadata(self, image_path, output_path=None):
        """
        Removes metadata from the image at `image_path`. If `output_path` is None,
        it overwrites the original image file with the cleaned version.
        """
        # Read the image
        image = cv2.imread(image_path)

        # Write the image back to the original file (or to output_path if specified)
        if output_path:
            cv2.imwrite(output_path, image)  # If an output path is provided, save the cleaned image there
        else:
            cv2.imwrite(image_path, image)  # Otherwise, overwrite the original image

    
    def metadata_removal_thread(self):
        """
        Background thread to handle metadata removal from images.
        It removes metadata but does not move the images to the review folder.
        """
        while True:
            try:
                # Get the next image to process
                image_path, output_path = self.metadata_queue.get(timeout=1)

                if os.path.exists(image_path):
                    # If output_path is None, it will overwrite the original file
                    self.remove_metadata(image_path, output_path)
                else:
                    logging.warning(f"File not found: {image_path}")

                # Mark the task as done
                self.metadata_queue.task_done()

            except queue.Empty:
                continue

    
    def process_image_for_metadata_removal(self, image_path):
        """
        Adds the image path to the queue for metadata removal.
        The image will be processed and overwritten in place.
        """
        self.metadata_queue.put((image_path, None))  # No output_path, so the image is overwritten in place

    
    def remove_orphan_json_files(self):
        json_files = glob.glob(os.path.join(self.base_directory, "*.json"))
        for json_file in json_files:
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            image_exists = any(
                os.path.exists(os.path.join(self.base_directory, f"{base_name}{ext}"))
                for ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
            )
            if not image_exists:
                os.remove(json_file)
                logging.info(f"Removed orphan JSON file: {json_file}")
    
    def is_valid_image(self, file_path):
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except (IOError, SyntaxError):
            return False
    
    def import_classes(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self.parent, "Import Classes", "", "Classes Files (classes.txt *.names)", options=options)
        if file_name:
            self.base_directory = os.path.dirname(file_name)
            with open(file_name, "r") as f:
                class_names = f.readlines()
                self.valid_classes = list(range(len(class_names)))
            os.makedirs(os.path.join(self.base_directory, self.review_folder), exist_ok=True)
    
    def check_annotation_file(self, file_path, image_folder):
        issues = []
        lines_to_keep = []
        should_move = False
        count_small_bboxes = 0
        height_zero = False
        class_index = -1
        img_file = None
    
        for ext in (".jpg", ".jpeg", ".png"):
            img_file = os.path.basename(file_path).replace(".txt", ext)
            if os.path.exists(os.path.join(image_folder, img_file)):
                break
    
        if img_file is None:
            return [f"Warning: No image file found for '{os.path.basename(file_path)}'"], lines_to_keep, True
    
        img_width, img_height = None, None
        img_path = os.path.join(image_folder, img_file)
    
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except FileNotFoundError as e:
                return [f"Warning: Error opening image file: {e}"], lines_to_keep, True
        else:
            return [f"Warning: No such file or directory: '{img_path}'"], lines_to_keep, True
    
        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                line_issues = []
    
                if len(tokens) != 5:
                    line_issues.append("Incorrect number of tokens")
                else:
                    try:
                        class_index = int(tokens[0])
                    except ValueError:
                        line_issues.append("Invalid class index format")
                        should_move = True
    
                    if class_index not in self.valid_classes:
                        line_issues.append("Invalid object class")
                        should_move = True
    
                    try:
                        x, y, width, height = map(float, tokens[1:])
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                            line_issues.append("Bounding box coordinates are not within the range of [0, 1]")
                            should_move = True
                        if not (width > 0 and height > 0):
                            line_issues.append("Width and height should be positive values")
                            should_move = True
                    except ValueError:
                        line_issues.append("Invalid bounding box values")
                        should_move = True
    
                if not line_issues and img_width is not None and img_height is not None:
                    abs_width = width * img_width
                    abs_height = height * img_height
    
                    if (width * height) >= self.max_label.value() / 100:
                        logging.info(f"Processing file: {file_path}")
                        logging.info(f"Problematic line: {line.strip()}")
    
                        line_issues.append("Bounding box takes up a large percentage of the frame")
                        should_move = True
    
                    if abs_height < self.box_size.value() / 100 * img_height:
                        height_zero = True
                        line_issues.append("Height is too small")
                    elif abs_width <= 1 and abs_height <= 1:
                        line_issues.append("Small bounding box (6x6 or smaller)")
                        count_small_bboxes += 1
                    else:
                        lines_to_keep.append(line)
    
                if line_issues:
                    issues.append((line.strip(), line_issues))
                    if not should_move:
                        should_move = True
    
        if lines_to_keep:
            with open(file_path, 'w') as f_out:
                f_out.writelines(lines_to_keep)
        else:
            should_move = True
    
        return issues, lines_to_keep, should_move
    
    def move_files(self, annotation_folder, review_folder, file_name):
        annotation_folder = Path(annotation_folder)
        review_folder = Path(review_folder)
        txt_path = annotation_folder / file_name
    
        if not txt_path.exists():
            logging.error(f"Error: {file_name} not found in {annotation_folder}.")
            return
    
        try:
            shutil.move(str(txt_path), str(review_folder / file_name))
        except FileNotFoundError as e:
            logging.error(f"Error: Unable to move {file_name} due to: {e}")
        except Exception as e:
            logging.error(f"Error: An unexpected error occurred: {e}")
    
        for ext in (".jpg", ".jpeg", ".png"):
            image_file = file_name.replace(".txt", ext)
            image_path = annotation_folder / image_file
    
            if image_path.exists():
                try:
                    shutil.move(str(image_path), str(review_folder / image_file))
                    logging.info(f"{image_file} was moved because it has no corresponding annotation.")
                except Exception as e:
                    logging.error(f"Error: Unable to move {image_file} due to: {e}")
                break
        else:
            logging.warning(f"No corresponding image file found for {file_name}.")
    
    def create_blanks_folder(self):
        blanks_folder_base = os.path.join(self.base_directory, "blanks")
        counter = 1
        blanks_folder = blanks_folder_base
        while os.path.exists(blanks_folder):
            blanks_folder = f"{blanks_folder_base}_{counter}"
            counter += 1
        os.makedirs(blanks_folder, exist_ok=True)
        return blanks_folder
    
    def move_blanks(self, blanks_folder, annotation_folder, image_folder, file_name):
        file_path = os.path.join(annotation_folder, file_name)
        if not os.path.isfile(file_path):
            logging.warning(f"File {file_name} not found in {annotation_folder}.")
            return
    
        try:
            with open(file_path, 'r') as f:
                pass
        except IOError:
            logging.warning(f"File {file_name} is being used by another process. Cannot move it.")
            return
    
        shutil.move(file_path, os.path.join(blanks_folder, file_name))
    
        for ext in (".jpg", ".jpeg", ".png"):
            img_file = file_name.replace(".txt", ext)
            if os.path.exists(os.path.join(image_folder, img_file)):
                shutil.move(os.path.join(image_folder, img_file), os.path.join(blanks_folder, img_file))
                break
        else:
            logging.warning(f"Image file {img_file} not found in {image_folder}.")
    
    def handle_blanks_after_review(self, annotation_folder, image_folder):
        reply = QMessageBox.question(self.parent, "Move Blanks",
                                    "Do you want to move empty text files and their corresponding images to a 'blanks' folder?",
                                    QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            blanks_folder = self.create_blanks_folder()
            txt_files = [file for file in os.listdir(annotation_folder) if file.endswith(".txt")]
            moved_count = 0

            for file_name in txt_files:
                file_path = os.path.join(annotation_folder, file_name)
                is_blank = False
                with open(file_path, 'r') as f:
                    is_blank = not f.read().strip()

                if is_blank:
                    self.move_blanks(blanks_folder, annotation_folder, image_folder, file_name)
                    moved_count += 1

            QMessageBox.information(self.parent, 'Information', f'{moved_count} blanks moved successfully!')
        else:
            QMessageBox.information(self.parent, 'Information', 'Blanks not moved.')

        reply2 = QMessageBox.question(self.parent, "Move by Label Size",
                                    "Do you want to move small, medium, and large files to subfolders based on their label size?",
                                    QMessageBox.Yes | QMessageBox.No)

        if reply2 == QMessageBox.Yes:
            subfolder_names = {"small": "small", "medium": "med", "large": "large"}
            for size in subfolder_names:
                subfolder_path = os.path.join(annotation_folder, subfolder_names[size])
                os.makedirs(subfolder_path, exist_ok=True)

            for file_name in txt_files:
                file_path = os.path.join(annotation_folder, file_name)
                label_size = None

                with open(file_path, 'r') as f:
                    lines = f.readlines()

                if lines:
                    tokens = lines[0].strip().split()
                    if len(tokens) == 5:
                        _, x, y, width, height = map(float, tokens)
                        label_size = self.categorize_label_size(width, height)

                if label_size in subfolder_names:
                    subfolder_path = os.path.join(annotation_folder, subfolder_names[label_size])
                    
                    # Handle each image extension separately
                    for ext in (".jpg", ".jpeg", ".png"):
                        img_file = file_name.replace(".txt", ext)
                        img_path = os.path.join(image_folder, img_file)
                        if os.path.isfile(img_path):
                            shutil.move(img_path, os.path.join(subfolder_path, img_file))
                            break
                        else:
                            logging.warning(f"Image file {img_file} not found in {image_folder}.")
                    
                    shutil.move(file_path, os.path.join(subfolder_path, file_name))

            QMessageBox.information(self.parent, 'Information', 'Files moved by label size successfully!')
        else:
            QMessageBox.information(self.parent, 'Information', 'Files not moved by label size.')

    
    def categorize_label_size(self, width, height):
        size = width * height
        if size <= 0.02:
            label_size = "small"
        elif size <= 0.07:
            label_size = "medium"
        else:
            label_size = "large"
        logging.info(f"Label size: {label_size}")
        return label_size

    def process_files(self, file_name, annotation_folder, image_folder, review_folder, statistics, label_sizes):
        try:
            logging.info(f"Processing file: {file_name}")
            file_path = os.path.join(annotation_folder, file_name)

            # Read and check annotation content
            if not os.path.exists(file_path):
                logging.warning(f"Annotation file {file_name} does not exist.")
                return

            # Get the issues, lines to keep, and whether to move the file
            issues, lines_to_keep, should_move = self.check_annotation_file(file_path, image_folder)

            if not lines_to_keep:
                self.blank_labels += 1
                logging.info(f"No valid lines in {file_name}, counting as blank label.")
            else:
                self.total_labels += len(lines_to_keep)
                logging.info(f"{len(lines_to_keep)} valid lines found in {file_name}")

            # Process lines to keep for label size categorization
            for line in lines_to_keep:
                tokens = line.strip().split()
                if len(tokens) == 5:
                    try:
                        _, x, y, width, height = map(float, tokens)
                        label_size = self.categorize_label_size(width, height)
                        label_sizes[label_size] += 1
                    except ValueError as ve:
                        logging.error(f"Value error processing line in {file_name}: {line.strip()} - {ve}")
                else:
                    logging.warning(f"Invalid line format in {file_name}: {line.strip()}")

            if issues:
                statistics[file_name] = issues
                if should_move:
                    self.bad_labels += 1
                    self.bad_images += 1
                    logging.info(f"Moving {file_name} and its corresponding image...")
                    self.move_files(annotation_folder, review_folder, file_name)
                    image_path = os.path.join(image_folder, file_name.replace(".txt", ".jpg", ".jpeg", ".png"))
                    if os.path.exists(image_path):
                        shutil.move(image_path, os.path.join(review_folder, os.path.basename(image_path)))
                    txt_path = os.path.join(annotation_folder, file_name)
                    if os.path.exists(txt_path):
                        shutil.move(txt_path, os.path.join(review_folder, os.path.basename(txt_path)))
                    logging.info(f"{file_name} and its corresponding image moved.")
        except Exception as e:
            logging.error(f"Error while processing {file_name}: {e}")



    def scan_annotations(self, progress_bar=None):
        if not self.valid_classes:
            logging.warning("No classes.txt or .names file found. Please import one first.")
            return

        if not self.base_directory:
            logging.warning("No base directory found. Please import a dataset first.")
            return

        annotation_folder = self.base_directory
        image_folder = self.base_directory
        review_folder = os.path.join(self.base_directory, "review")
        os.makedirs(review_folder, exist_ok=True)

        txt_files_set = {file for file in os.listdir(annotation_folder) if file.endswith(".txt") and file != "classes.txt" and not file.endswith(".names")}
        img_files_set = {file for file in os.listdir(image_folder) if file.lower().endswith((".jpg", ".jpeg", ".png"))}

        statistics = {}
        label_sizes = {"small": 0, "medium": 0, "large": 0}

        num_cores = os.cpu_count()
        max_workers = num_cores - 1 if num_cores > 1 else 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for file_name in txt_files_set:
                executor.submit(self.process_files, file_name, annotation_folder, image_folder, review_folder, statistics, label_sizes)

        self.remove_orphan_json_files()

        # Update total_images based on the processed files
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
                    ("label_sizes", label_sizes),
                    ("blank_labels", self.blank_labels),
                    ("bad_labels", self.bad_labels),
                    ("bad_images", self.bad_images),
                    ("problems", statistics)
                ]), f, indent=4)
            QMessageBox.information(self.parent, "Information", f"Statistics saved successfully to {save_path}!")
        except (IOError, OSError) as e:
            logging.error(f"An error occurred while saving the JSON file: {e}")
            QMessageBox.critical(self.parent, "Error", f"An error occurred while saving the JSON file: {e}")
        except PermissionError:
            logging.error("You do not have permission to write to this file.")
            QMessageBox.critical(self.parent, "Permission Error", "You do not have permission to write to this file.")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            QMessageBox.critical(self.parent, "Error", f"An unexpected error occurred: {e}")

        self.handle_blanks_after_review(annotation_folder, image_folder)




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
                    print(f"Downloaded video: {filename}")
                    self.update_progress.emit(self.row, 100)  # Set progress bar to 100%
                break

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(random.uniform(1, 3))
                else:
                    print(f"Download failed with error: {e}")
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

            print(f"Progress: {progress}% for row {self.row}")  # Debugging
            self.update_progress.emit(self.row, progress)  # Emit progress signal for UI

        elif d['status'] == 'finished':
            # When the download finishes, set progress to 100%
            print('Download finished, now processing...')
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
            print(f"Finished processing {self.file}")
            self.signals.finished.emit()  # Emit finished signal
            self.signals.finished.emit()  # Emit finished signal





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
            print(f"Error loading model: {e}")
            return None

    def run(self):
        if self.stable_diffusion_pipeline is None:
            print("Pipeline not loaded. Exiting...")
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
            print(f"Error reading image: {image_path}")
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
            print(f"Failed to inpaint image: {image_path}")
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
            print(f"Error in inpainting: {e}")
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
        self.main_window.preview_list.setRowCount(0)
        self.main_window.preview_list.setColumnCount(5)
        self.main_window.preview_list.setHorizontalHeaderLabels(['Image', 'Class Name', 'ID', 'Size', 'Bounding Box'])

        # Set up placeholder image
        placeholder_pixmap = QPixmap(128, 128)  # Create a 128x128 placeholder
        placeholder_pixmap.fill(Qt.gray)  # Fill the pixmap with a gray color

        if show_images:
            self.main_window.preview_list.setColumnWidth(0, 100)
        else:
            self.main_window.preview_list.setColumnWidth(0, 0)

        self.main_window.preview_list.setColumnWidth(1, 50)
        self.main_window.preview_list.setColumnWidth(2, 25)
        self.main_window.preview_list.setColumnWidth(3, 50)
        self.main_window.preview_list.setColumnWidth(4, 250)
        print("UI setup completed.")


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
            print(f"No label file found for {label_path}")
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

class CheckboxDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def paint(self, painter, option, index):
        # Get the item text and checkbox state
        text = index.data(Qt.DisplayRole)
        checked = index.data(Qt.CheckStateRole) == Qt.Checked

        # Draw the checkbox
        checkbox_style = QApplication.style()
        checkbox_option = QStyleOptionButton()
        checkbox_option.rect = option.rect
        checkbox_option.state = QStyle.State_Enabled | (QStyle.State_On if checked else QStyle.State_Off)
        checkbox_style.drawControl(QStyle.CE_CheckBox, checkbox_option, painter)

        # Draw the text
        text_rect = option.rect.adjusted(20, 0, 0, 0)  # Adjust to avoid overlap with checkbox
        painter.drawText(text_rect, Qt.AlignVCenter, text)

    def createEditor(self, parent, option, index):
        return None  # Disable editing in the dropdown

    def setModelData(self, editor, model, index):
        pass  # No editor needed

    def editorEvent(self, event, model, option, index):
        if event.type() == QEvent.MouseButtonPress:
            # Toggle the checkbox state
            current_state = index.data(Qt.CheckStateRole)
            new_state = Qt.Unchecked if current_state == Qt.Checked else Qt.Checked
            model.setData(index, new_state, Qt.CheckStateRole)
            return True
        return super().editorEvent(event, model, option, index)

from PyQt5.QtCore import QThread, pyqtSignal

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


ui_file: Path = Path(__file__).resolve().parent / "ultradarkfusion_v3.0.ui"
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
        
        
        # Video processor setup
        self.video_processor = GUIVideoProcessor()        

        # Set window title
        self.setWindowTitle("UltraDarkFusion")

        # Icon timer setup
        self.icon_timer = QTimer(self)
        self.icon_timer.timeout.connect(self.update_icon)
        self.icon_timer.start(500)
        self.current_icon_index = 0
        self.threadpool = QThreadPool()
        # Console output timer setup
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

        # Initialize the screen view and graphics scene
        self.view_references = []
        self.screen_view = CustomGraphicsView(main_window=self)
        self.screen_view.setBackgroundBrush(QBrush(Qt.black))
        self.setCentralWidget(self.screen_view)
        self.screen_view.setRenderHint(QPainter.Antialiasing)
        self.screen_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_scene = QGraphicsScene(self)
        self.selected_bbox = None
        
        # Now that screen_view is initialized, you can install the event filter


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
        self.class_input_field.returnPressed.connect(self.class_input_field_return_pressed)
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
        self.image_directory = None  # Directory will be set later
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
        self.image_directory = None  # Initialize as None or set a default path
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
        self.crop_all_button.clicked.connect(self.auto_label_yolo_button_clicked)
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

        # see download and extrct video
        self.video_upload.clicked.connect(self.on_add_video_clicked)
        self.remove_video_button.clicked.connect(self.on_remove_video_clicked)
        self.custom_frames_checkbox.stateChanged.connect(self.on_custom_frames_toggled)
        self.custom_size_checkbox.stateChanged.connect(self.on_custom_size_checkbox_state_changed)
        self.image_format.currentTextChanged.connect(self.on_image_format_changed)
        self.extract_button.clicked.connect(self.on_extract_button_clicked)
        self.stop_extract.clicked.connect(self.video_processor.stop)
        self.stop_extract.clicked.connect(self.on_stop_extract_clicked)
        self.image_format.setCurrentText("desired_value")
        self.dialog_open = False
        self.output_path = ""
        self.add_video_running = False
        self.custom_frames_checkbox.stateChanged.connect(self.update_checkboxes)
        self.custom_size_checkbox.stateChanged.connect(self.update_checkboxes)
        self.height_box.valueChanged.connect(self.on_size_spinbox_value_changed)
        self.width_box.valueChanged.connect(self.on_size_spinbox_value_changed)
        self.custom_input.textChanged.connect(self.on_custom_input_changed)
        self.video_processor.progress_updated.connect(self.update_progress)

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
        self.stop_extract.clicked.connect(self.on_stop_extract_clicked)
        self.current_image = None
        self.capture = None
        self.is_camera_mode = False
        self.current_file_name = None
        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.display_camera_input)
        self.location_button.clicked.connect(self.set_output_directory)
        self.input_selection.currentIndexChanged.connect(self.on_input_source_changed)
        self.setup_input_sources()       
        self.save_path = ""
        self.model = None
        self.weights_file_path = None    
    
        self.timer2.timeout.connect(self.display_camera_input)
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
        self.play_video_button.clicked.connect(self.on_play_video_clicked)

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
        

        self.remove_class.clicked.connect(self.remove_class_button_clicked)
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
        self.super_resolution_Checkbox.toggled.connect(self.checkbox_clicked)
        self.slider_min_value = 0
        self.slider_max_value = 255
        self.anti_slider.valueChanged.connect(self.slider_value_changed2)
        self.grayscale_Checkbox.stateChanged.connect(self.checkbox_clicked)
        self.bounding_boxes = {}

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
        
        self.dino_label.triggered.connect(self.on_dino_label_clicked)

        self.image_directory = None  # Initialize the attribute to store the image directory
        self.file_observer = None
        self.clear_json.triggered.connect(self.on_clear_json_clicked)

        # Save ROI related UI elements
        self.initial_geometry = self.saveGeometry()
        self.initial_state = self.saveState()
        self.roi_checkbox.stateChanged.connect(self.toggle_roi)        
        self.reset_layout_action = self.findChild(QAction, "reset_layout")
        if self.reset_layout_action:
            self.reset_layout_action.triggered.connect(self.resetLayout)
        # ROI sliders and spin boxes    
        self.actionSahi_label.triggered.connect(self.showSahiSettings)
        self.width_spin_slider.valueChanged.connect(self.update_roi)
        self.height_spin_slider.valueChanged.connect(self.update_roi)
        self.x_spin_slider.valueChanged.connect(self.update_roi)
        self.y_spin_slider.valueChanged.connect(self.update_roi)
        self.filter_button.clicked.connect(self.filter_bboxes)

        
        self.current_item_index = None  # CHECK THIS
        self.thumbnail_index = 0 
        self.time = 0.0          

        self.split_data_button = self.findChild(QPushButton, 'split_data')
        self.split_data_button.clicked.connect(self.launch_split_data_ui)
        self.image_quality_checkbox.toggled.connect(self.toggle_image_quality_analysis)
        self.image_quality_analysis_enabled = False 

        self.all_frame_bounding_boxes = {}  # Dictionary to store bounding boxes for all frames
        self.class_visibility = {}  # Store visibility state for each class
            
        self.show()
        
        
    def launch_split_data_ui(self):
        # Replace 'python' with the correct path if needed
        subprocess.Popen(["python", "splitdata.py"])

              
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

    

    def toggle_roi(self, state):
        """Toggle the ROI visibility based on checkbox state."""
        if state:
            self.update_roi()
        else:
            self.remove_roi()

    def validate_roi(self):
        """Ensure ROI and image dependencies are properly set up."""
        if not hasattr(self, 'roi_item') or not self.roi_item.isVisible():
            QMessageBox.warning(self, "No ROI Defined", "Please define and activate the ROI before filtering.")
            return False
        if not hasattr(self, 'image_directory') or not self.image_directory:
            QMessageBox.warning(self, "No Directory Selected", "Please select an image directory first.")
            return False
        return True

    def filter_bboxes(self):
        """Filter bounding boxes based on the active ROI, dynamically adjusted per image, with progress tracking."""
        if not self.validate_roi():
            return

        # Get the ROI rectangle from the user-defined settings
        roi_rect = self.roi_item.rect()
        total_images = len(self.image_files)
        
        # Initialize the progress bar
        self.label_progress.setMinimum(0)
        self.label_progress.setMaximum(total_images)
        self.label_progress.setValue(0)

        for idx, image_file in enumerate(self.image_files):
            if image_file.endswith("default.png"):
                continue

            # Load the image to get its actual dimensions
            image = QPixmap(image_file)
            if image.isNull():
                print(f"Failed to load image: {image_file}")
                continue

            # Get the dynamic ROI scaled to the current image dimensions
            image_width, image_height = image.width(), image.height()
            dynamic_roi = self.get_dynamic_roi(roi_rect, image_width, image_height)

            # Process the bounding boxes for the current image
            txt_file_path = self.replace_extension_with_txt(image_file)
            if os.path.exists(txt_file_path):
                self.filter_bboxes_in_file(txt_file_path, dynamic_roi, image_width, image_height)
            else:
                print(f"No annotation file found for {image_file}, skipping.")

            # Update progress bar
            self.label_progress.setValue(idx + 1)
            QtWidgets.QApplication.processEvents()  # Allow UI updates during processing

        QMessageBox.information(self, "Filter Complete", "Bounding boxes outside the ROI have been processed.")
        self.label_progress.setValue(total_images)  # Ensure the progress bar reaches the end


    def get_dynamic_roi(self, roi_rect, image_width, image_height):
        """
        Scale the ROI rectangle dynamically to fit the current image dimensions.

        Args:
            roi_rect: The static ROI rectangle from user settings.
            image_width: The width of the current image.
            image_height: The height of the current image.

        Returns:
            A tuple representing the scaled ROI (x_min, y_min, width, height).
        """
        # Scale ROI proportionally based on image dimensions
        x_scale = image_width / self.image.width()
        y_scale = image_height / self.image.height()

        scaled_x = int(roi_rect.x() * x_scale)
        scaled_y = int(roi_rect.y() * y_scale)
        scaled_width = int(roi_rect.width() * x_scale)
        scaled_height = int(roi_rect.height() * y_scale)

        return scaled_x, scaled_y, scaled_width, scaled_height


    def filter_bboxes_in_file(self, txt_file_path, roi, image_width, image_height, enforce_within_roi=False):
        """
        Filters bounding boxes in a YOLO `.txt` file based on a dynamically scaled ROI and ensures the output is in proper YOLO format.
        """
        x_roi, y_roi, width_roi, height_roi = roi

        # ROI absolute coordinates
        roi_x1, roi_y1 = x_roi, y_roi
        roi_x2, roi_y2 = x_roi + width_roi, y_roi + height_roi

        try:
            # Read bounding box data
            data = np.loadtxt(txt_file_path, delimiter=" ", ndmin=2)
            if data.size == 0:
                return  # No bounding boxes in the file

            # Parse YOLO format into absolute coordinates
            class_ids, x_centers, y_centers, widths, heights = data.T
            x1 = (x_centers - widths / 2) * image_width
            y1 = (y_centers - heights / 2) * image_height
            x2 = (x_centers + widths / 2) * image_width
            y2 = (y_centers + heights / 2) * image_height

            # Filter bounding boxes based on ROI
            if enforce_within_roi:
                # Fully within ROI
                condition = (x1 >= roi_x1) & (y1 >= roi_y1) & (x2 <= roi_x2) & (y2 <= roi_y2)
            else:
                # Any overlap with ROI
                condition = (x1 < roi_x2) & (x2 > roi_x1) & (y1 < roi_y2) & (y2 > roi_y1)

            filtered_data = data[condition]

            # Convert back to YOLO format
            if filtered_data.size > 0:
                filtered_class_ids = filtered_data[:, 0].astype(int)  # Ensure class IDs are integers
                filtered_x_centers = ((x1[condition] + x2[condition]) / 2) / image_width
                filtered_y_centers = ((y1[condition] + y2[condition]) / 2) / image_height
                filtered_widths = (x2[condition] - x1[condition]) / image_width
                filtered_heights = (y2[condition] - y1[condition]) / image_height

                yolo_data = np.column_stack((
                    filtered_class_ids, 
                    filtered_x_centers, 
                    filtered_y_centers, 
                    filtered_widths, 
                    filtered_heights
                ))

                # Save back to the file
                np.savetxt(txt_file_path, yolo_data, fmt="%d %.6f %.6f %.6f %.6f")
            else:
                # Clear the file if no valid bounding boxes remain
                open(txt_file_path, "w").close()

        except Exception as e:
            print(f"Error processing {txt_file_path}: {e}")



    def draw_roi(self, width, height, x, y):
        """Draw or update the ROI rectangle dynamically on the image."""
        pixmap_width = self.image.width()
        pixmap_height = self.image.height()

        # Constrain ROI to image bounds
        x = max(0, min(x, pixmap_width - width))
        y = max(0, min(y, pixmap_height - height))
        width = min(width, pixmap_width - x)
        height = min(height, pixmap_height - y)

        if not hasattr(self, 'roi_item') or self.roi_item not in self.screen_view.scene().items():
            self.roi_item = QGraphicsRectItem(x, y, width, height)
            pen = QPen(Qt.yellow)
            pen.setWidth(3)
            self.roi_item.setPen(pen)
            self.screen_view.scene().addItem(self.roi_item)
        else:
            self.roi_item.setRect(x, y, width, height)
            self.roi_item.show()


    def remove_roi(self):
        """Hide the ROI item instead of deleting it."""
        if hasattr(self, 'roi_item') and self.roi_item in self.screen_view.scene().items():
            self.roi_item.hide()

    def update_roi(self):
        """Update the ROI rectangle based on the current slider values."""
        if not hasattr(self, 'image') or self.image is None:
            if self.roi_checkbox.isChecked():
                print("Warning: No valid image loaded. ROI cannot be updated.")
            return

        # Calculate the center of the image
        width_offset = self.width_spin_slider.value() // 2
        height_offset = self.height_spin_slider.value() // 2
        center_x = self.image.width() // 2
        center_y = self.image.height() // 2

        x = center_x - width_offset + self.x_spin_slider.value()
        y = center_y - height_offset + self.y_spin_slider.value()

        # Draw or update the ROI
        self.draw_roi(width_offset * 2, height_offset * 2, x, y)




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
                self.update_roi()

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
                        print(f"Deleted {json_file}")
                    except Exception as e:
                        print(f"Error deleting {json_file}: {str(e)}")
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
                print(f'Model type changed to: {self.model_type}')
            else:
                print('No matching model type found.')

            print(f'SAM checkpoint changed to: {self.sam_checkpoint}')

    def on_check_button_click(self):
        self.overwrite = self.overwrite_var.isChecked()

    def process_directory(self):
        self.image_files = glob.glob(os.path.join(self.batch_directory, '*.[jp][pn]g'))

    def on_shadow_checkbox_click(self):
        self.shadow = self.shadow_var.isChecked()


    def on_noise_remove_checked(self, state):
        self.is_noise_remove_enabled = state == QtCore.Qt.Checked
        
    def generate_unique_name(self):
        """Generate a unique name using the current datetime and a UUID."""
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]  # Shortened UUID
        return f"{current_time}_{unique_id}"
              
                
    def copy_and_rename_files(self, image_path, txt_path, target_directory):
        """Copy and rename image and corresponding txt file with unique names."""
        os.makedirs(target_directory, exist_ok=True)

        unique_name = self.generate_unique_name()

        new_image_path = os.path.join(target_directory, f"{unique_name}{os.path.splitext(image_path)[1]}")
        try:
            shutil.copy2(image_path, new_image_path)
            logging.info(f"Copied image to {new_image_path}")
        except Exception as e:
            logging.error(f"Failed to copy image {image_path} to {new_image_path}: {e}")

        new_txt_path = os.path.join(target_directory, f"{unique_name}.txt")
        if os.path.exists(txt_path):
            try:
                shutil.copy2(txt_path, new_txt_path)
                logging.info(f"Copied annotation to {new_txt_path}")
            except Exception as e:
                logging.error(f"Failed to copy annotation {txt_path} to {new_txt_path}: {e}")
        else:
            logging.error(f"Annotation file not found: {txt_path}")

        return new_image_path, new_txt_path

    def predict_and_draw_yolo_objects(self, image, image_file_path):
        """
        Predict objects using YOLO model and draw bounding boxes on the image.
        Only displays regions segmented by SAM and generates adaptive shadow images
        based on object size if respective checkboxes are active.
        """
        if not self.predictor or not self.sam or not os.path.isfile(image_file_path):
            logging.error('Predictor, SAM, or image file is not available.')
            QMessageBox.critical(self, "Error", "Predictor, SAM, or image file is not available.")
            return None

        # Prepare image copies

        image_copy = image.copy()  # Original copy for shadow image generation
        image_with_boxes = image.copy()  # Separate copy for drawing bounding boxes

        # YOLO annotation file path
        yolo_file_path = os.path.splitext(image_file_path)[0] + ".txt"
        adjusted_boxes = []

        try:
            with open(yolo_file_path, "r") as f:
                yolo_lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.error(f"Failed to read YOLO file {yolo_file_path}: {e}")
            return None

        img_height, img_width = image.shape[:2]
        self.predictor.set_image(image)

        # Store masks for noise reduction and shadow generation
        masks_for_bboxes = []

        # Process each YOLO annotation
        for yolo_line in yolo_lines:
            try:
                values = list(map(float, yolo_line.split()))
                if len(values) != 5:
                    logging.warning(f"Invalid YOLO format in line: {yolo_line}")
                    continue

                class_index, x_center, y_center, w, h = values

                # Convert YOLO bbox to VOC format
                box = pbx.convert_bbox(
                    (x_center, y_center, w, h),
                    from_type="yolo",
                    to_type="voc",
                    image_size=(img_width, img_height)
                )
                x_min, y_min, x_max, y_max = map(int, box)
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(img_width - 1, x_max), min(img_height - 1, y_max)

                # Skip invalid bounding boxes
                if x_max <= x_min or y_max <= y_min:
                    logging.warning(f"Skipped invalid bounding box: {x_min, y_min, x_max, y_max}")
                    continue

                # Perform segmentation using SAM
                input_box = np.array([x_min, y_min, x_max, y_max])
                with torch.cuda.amp.autocast():
                    masks, _, _ = self.predictor.predict(box=input_box[None, :], multimask_output=False)
                mask = masks[0].astype(bool)

                # Draw mask and bounding box
                image_with_boxes, bbox = self.draw_mask_and_bbox(image_with_boxes, mask)
                if bbox == (0, 0, 0, 0):
                    logging.warning("No valid segmentation found for this bounding box.")
                    continue

                # Store mask for later background removal or shadow image creation
                masks_for_bboxes.append((mask, bbox))

                # Adjust bounding box and save updated coordinates in YOLO format if overwrite is enabled
                if self.overwrite:
                    x_min, y_min, x_max, y_max = bbox
                    voc_box = pbx.convert_bbox(
                        (x_min, y_min, x_max, y_max),
                        from_type="voc",
                        to_type="yolo",
                        image_size=(img_width, img_height)
                    )
                    adjusted_boxes.append(
                        f"{int(class_index)} {voc_box[0]:.6f} {voc_box[1]:.6f} {voc_box[2]:.6f} {voc_box[3]:.6f}\n"
                    )

            except Exception as e:
                logging.error(f"Error processing YOLO line: {e}")

        # Overwrite YOLO file with adjusted boxes, if necessary
        if self.overwrite and adjusted_boxes:
            try:
                with open(yolo_file_path, "w") as f:
                    f.writelines(adjusted_boxes)
            except IOError as e:
                logging.error(f"IO error when writing to {yolo_file_path}: {e}")

        # Apply noise reduction if enabled
        if self.is_noise_remove_enabled:
            self.apply_noise_reduction(image_copy, masks_for_bboxes, image_file_path)

        # Generate adaptive shadow image if enabled
        if self.shadow and masks_for_bboxes:
            self.create_shadow_image(image_copy, masks_for_bboxes, image_file_path)
        elif self.shadow:
            logging.warning("No valid masks found for shadow image creation.")

        # Return processed image with bounding boxes for display if screen update is enabled
        return image_with_boxes if self.screen_update.isChecked() else None

    def on_copy_paste_checkbox_click(self, state: int):
        """
        Slot to handle the state change of the copy_paste_checkbox.
        """
        self.copy_paste_enabled = state == QtCore.Qt.Checked
        logging.info(f"Copy-paste augmentation {'enabled' if self.copy_paste_enabled else 'disabled'}.")





    def apply_copy_paste_augmentation(self, progress_offset=0):
        """
        Performs copy-paste augmentation across all images with dynamic sizing, balanced distribution,
        and dynamic determination of the number of objects per frame.
        """
        if not self.image_directory or not os.path.exists(self.image_directory):
            QMessageBox.critical(self, "Error", "Please select a valid image directory first.")
            return

        copy_paste_dir = os.path.join(self.image_directory, "copy_and_paste")
        os.makedirs(copy_paste_dir, exist_ok=True)

        total_images = len(self.image_files)
        size_categories = ["small", "medium", "large"]
        min_size_threshold = 10  # Minimum object size threshold in pixels

        for idx, image_file in enumerate(self.image_files):
            image = cv2.imread(image_file)
            if image is None:
                logging.warning(f"Failed to load image: {image_file}. Skipping.")
                continue

            annotation_file = os.path.splitext(image_file)[0] + ".txt"

            # Load annotations if available
            annotations = []
            if os.path.exists(annotation_file):
                with open(annotation_file, "r") as f:
                    annotations = [line.strip() for line in f if line.strip()]

            existing_bboxes = []
            for ann in annotations:
                try:
                    _, x_center, y_center, w, h = map(float, ann.split())
                    img_h, img_w = image.shape[:2]
                    voc_bbox = pbx.convert_bbox(
                        (x_center, y_center, w, h),
                        from_type="yolo",
                        to_type="voc",
                        image_size=(img_w, img_h)
                    )
                    existing_bboxes.append(voc_bbox)
                except ValueError:
                    logging.warning(f"Invalid annotation line: {ann}. Skipping.")

            objects_to_add = max(0, np.random.randint(3, 6) - len(existing_bboxes))

            for _ in range(objects_to_add):
                source_image_file = np.random.choice(self.image_files)
                source_image = cv2.imread(source_image_file)
                if source_image is None:
                    logging.warning(f"Failed to load source image: {source_image_file}. Skipping.")
                    continue

                source_annotation_file = os.path.splitext(source_image_file)[0] + ".txt"
                if not os.path.exists(source_annotation_file):
                    continue

                with open(source_annotation_file, "r") as f:
                    source_annotations = [line.strip() for line in f if line.strip()]
                if not source_annotations:
                    continue

                random_bbox = np.random.choice(source_annotations)
                try:
                    class_index, x_center, y_center, w, h = map(float, random_bbox.split())
                    src_h, src_w = source_image.shape[:2]
                    voc_bbox = pbx.convert_bbox(
                        (x_center, y_center, w, h),
                        from_type="yolo",
                        to_type="voc",
                        image_size=(src_w, src_h)
                    )
                    x_min, y_min, x_max, y_max = map(int, voc_bbox)

                    mask = self.generate_segmentation_mask(source_image, (x_min, y_min, x_max, y_max))
                    object_segment = self.extract_segmented_object(source_image, mask, x_min, y_min, x_max, y_max)

                    if object_segment is None:
                        logging.warning("Segmentation failed, skipping object.")
                        continue

                    tgt_h, tgt_w = image.shape[:2]
                    obj_h, obj_w = object_segment.shape[:2]

                    scaling_factor = np.random.uniform(0.8, 1.5)
                    obj_h, obj_w = int(obj_h * scaling_factor), int(obj_w * scaling_factor)
                    obj_h = max(min_size_threshold, min(obj_h, tgt_h // 2))
                    obj_w = max(min_size_threshold, min(obj_w, tgt_w // 2))
                    object_segment = cv2.resize(object_segment, (obj_w, obj_h))

                    max_attempts = 50
                    valid_position = False
                    for _ in range(max_attempts):
                        paste_x = np.random.randint(0, tgt_w - obj_w)
                        paste_y = np.random.randint(0, tgt_h - obj_h)
                        new_bbox = (paste_x, paste_y, paste_x + obj_w, paste_y + obj_h)
                        if not self.check_overlap(new_bbox, existing_bboxes, min_spacing=10):
                            valid_position = True
                            break

                    if not valid_position:
                        logging.warning("Failed to place object without overlap after multiple attempts, skipping.")
                        continue

                    existing_bboxes.append(new_bbox)
                    image = self.overlay_object(image, object_segment, paste_x, paste_y)
                    new_x_center = (paste_x + obj_w / 2) / tgt_w
                    new_y_center = (paste_y + obj_h / 2) / tgt_h
                    new_w = obj_w / tgt_w
                    new_h = obj_h / tgt_h
                    annotations.append(f"{int(class_index)} {new_x_center:.6f} {new_y_center:.6f} {new_w:.6f} {new_h:.6f}")
                except ValueError:
                    logging.warning("Invalid bounding box format. Skipping.")

            output_image_name = f"copy_and_paste_{os.path.basename(image_file)}"
            output_image_path = os.path.join(copy_paste_dir, output_image_name)
            cv2.imwrite(output_image_path, image)

            output_annotation_name = output_image_name.replace(".png", ".txt").replace(".jpg", ".txt")
            output_annotation_path = os.path.join(copy_paste_dir, output_annotation_name)
            with open(output_annotation_path, "w") as f:
                f.write("\n".join(annotations))

            logging.info(f"Saved augmented image and annotations: {output_image_path}, {output_annotation_path}")
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
        with torch.cuda.amp.autocast():
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
        Ensures the creation of matching .txt files for YOLO annotations.
        """
        # Make a copy of the input image to preserve the original
        shadow_image = image.copy()

        for mask, (x_min, y_min, x_max, y_max) in masks_for_bboxes:
            # Extract the region of interest (ROI) defined by the bounding box
            object_region = shadow_image[y_min:y_max, x_min:x_max]

            # Convert the ROI to grayscale
            grayscale_region = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)

            # Replace the RGB channels in the masked area with the grayscale values
            for c in range(3):  # For each channel (B, G, R)
                object_region[..., c] = np.where(
                    mask[y_min:y_max, x_min:x_max],  # Apply only to the masked region
                    grayscale_region,  # Grayscale intensity values
                    object_region[..., c]  # Leave other areas unchanged
                )
            
            # Update the shadow_image with the modified grayscale ROI
            shadow_image[y_min:y_max, x_min:x_max] = object_region

        # Save the updated shadow image
        shadow_folder = os.path.join(os.path.dirname(image_file_path), "shadow")
        os.makedirs(shadow_folder, exist_ok=True)
        shadow_image_name = f"shadow_{self.generate_unique_name()}.png"
        shadow_image_path = os.path.join(shadow_folder, shadow_image_name)
        cv2.imwrite(shadow_image_path, shadow_image)

        # Copy the corresponding YOLO annotation file
        yolo_file_path = os.path.splitext(image_file_path)[0] + ".txt"
        shadow_txt_name = os.path.splitext(shadow_image_name)[0] + ".txt"
        shadow_txt_path = os.path.join(shadow_folder, shadow_txt_name)

        if os.path.exists(yolo_file_path):
            try:
                shutil.copy2(yolo_file_path, shadow_txt_path)
            except Exception as e:
                logging.error(f"Failed to copy annotation: {e}")

        return shadow_image_path, shadow_txt_path




    def draw_mask_and_bbox(self, image, mask, prioritize="largest"):
        """
        Draws a mask and bounding box for the intended object in overlapping scenarios.

        Args:
            image: Original image (numpy array).
            mask: Binary segmentation mask (numpy array).
            prioritize: Criteria to prioritize objects ('largest', 'topmost', 'bottommost').

        Returns:
            image_with_boxes: Updated image with drawn masks and bounding boxes.
            bbox: The best bounding box (x_min, y_min, x_max, y_max).
        """
        # Convert mask to uint8 if necessary
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Use morphology to clean and slightly expand the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Adjust kernel size as needed
        refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)  # Remove noise

        # Find contours in the refined mask
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logging.warning("No contours found in the mask.")
            return image, (0, 0, 0, 0)

        # Determine the prioritized contour based on criteria
        if prioritize == "largest":
            target_contour = max(contours, key=cv2.contourArea)  # Largest area
        elif prioritize == "topmost":
            target_contour = min(contours, key=lambda c: cv2.boundingRect(c)[1])  # Smallest y-coordinate
        elif prioritize == "bottommost":
            target_contour = max(contours, key=lambda c: cv2.boundingRect(c)[1])  # Largest y-coordinate
        else:
            logging.warning(f"Invalid prioritization method: {prioritize}")
            return image, (0, 0, 0, 0)

        # Compute the bounding box for the prioritized contour
        x_min, y_min, w, h = cv2.boundingRect(target_contour)
        x_max, y_max = x_min + w, y_min + h

        # Ensure bounding box stays within the image boundaries
        img_height, img_width = image.shape[:2]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_width - 1, x_max), min(img_height - 1, y_max)

        # Skip invalid bounding boxes
        if x_max <= x_min or y_max <= y_min:
            logging.warning("Invalid bounding box coordinates computed.")
            return image, (0, 0, 0, 0)

        # Draw the refined mask over the image
        alpha = 0.5  # Transparency factor for overlay
        overlay = image.copy()
        cv2.drawContours(overlay, [target_contour], -1, (0, 255, 0), thickness=cv2.FILLED)  # Fill the selected mask
        image_with_boxes = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw bounding box tightly around the mask
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red bounding box

        # Save debug visualization
        debug_path = "debug_overlapping_bbox.png"
        cv2.imwrite(debug_path, image_with_boxes)
        logging.info(f"Segmented mask and prioritized bounding box saved to {debug_path}")

        return image_with_boxes, (x_min, y_min, x_max, y_max)

    
    def apply_noise_reduction(self, image, masks_for_bboxes, image_file_path):
        """
        Apply noise reduction selectively to background areas, preserving foreground sharpness.
        """
        for mask, (x_min, y_min, x_max, y_max) in masks_for_bboxes:
            # Create an inverted mask to focus noise reduction on the background
            inverted_mask = ~mask[y_min:y_max, x_min:x_max]

            # Process each color channel
            for c in range(3):
                roi = image[y_min:y_max, x_min:x_max, c]  # Region of interest
                filtered_roi = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
                image[y_min:y_max, x_min:x_max, c] = np.where(inverted_mask, filtered_roi, roi)

        # Save noise-reduced image
        nr_folder = os.path.join(os.path.dirname(image_file_path), "noise_reduced")
        os.makedirs(nr_folder, exist_ok=True)
        nr_image_name = f"nr_{self.generate_unique_name()}.png"
        nr_image_path = os.path.join(nr_folder, nr_image_name)
        cv2.imwrite(nr_image_path, image)

        # Copy YOLO annotations
        yolo_file_path = os.path.splitext(image_file_path)[0] + ".txt"
        nr_txt_name = os.path.basename(nr_image_path).replace('.png', '.txt')
        nr_txt_path = os.path.join(nr_folder, nr_txt_name)
        shutil.copy2(yolo_file_path, nr_txt_path)

        return image



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
        self.current_image_index = -1  # Start from the first image
        self.stop_labeling = False  # Reset the stop_labeling flag at the start
        self.stop_batch = False

        # Update the image files from the directory before starting the process
        self.image_files = sorted(
            glob.glob(os.path.join(self.image_directory, "*.[pP][nN][gG]")) +
            glob.glob(os.path.join(self.image_directory, "*.[jJ][pP][gG]")) +
            glob.glob(os.path.join(self.image_directory, "*.[jJ][pP][eE][gG]")) +
            glob.glob(os.path.join(self.image_directory, "*.[bB][mM][pP]")) +
            glob.glob(os.path.join(self.image_directory, "*.[gG][iI][fF]")) +
            glob.glob(os.path.join(self.image_directory, "*.[tT][iI][fF]")) +
            glob.glob(os.path.join(self.image_directory, "*.[wW][eE][bB][pP]"))
        )

        total_images = len(self.image_files)  # Total number of images to process

        # Adjust total_steps based on whether copy-paste is enabled
        if self.copy_paste_enabled:
            total_steps = total_images * 2  # Processing images + augmentation
        else:
            total_steps = total_images

        self.label_progress.setRange(0, total_steps)

        if not self.image_directory or not self.sam or not self.predictor:
            QMessageBox.critical(self, "Error", "Please select a folder and model first")
            return

        # Start processing images
        for idx, image_file in enumerate(self.image_files):
            self.batch_running = True
            if self.stop_batch or self.stop_labeling:
                self.batch_running = False
                break
            file_path = image_file
            image = cv2.imread(file_path)

            if image is None:
                print(f"Failed to load image: {file_path}")
                continue  # Skip to the next image

            # Check for corresponding annotation file
            annotation_file = os.path.splitext(file_path)[0] + ".txt"
            if not os.path.exists(annotation_file):
                print(f"No annotation file found for image: {file_path}. Skipping.")
                continue  # Skip to the next image

            # Check if the annotation file is empty
            if os.path.getsize(annotation_file) == 0:
                print(f"Annotation file is empty for image: {file_path}. Skipping.")
                continue  # Skip to the next image

            # Process the image with YOLO and SAM, if annotation file is valid
            processed_image = self.predict_and_draw_yolo_objects(image, file_path)
            if processed_image is not None:
                self.show_image(processed_image)

            # Manually trigger garbage collection after each file
            gc.collect()

            self.label_progress.setValue(idx + 1)  # Update the progress bar
            QtWidgets.QApplication.processEvents()

        # After processing images, perform augmentation if enabled
        if self.copy_paste_enabled:
            self.apply_copy_paste_augmentation(progress_offset=total_images)

        # Final garbage collection after the batch is processed
        gc.collect()

        # Displaying completion message and finalizing progress bar
        if self.stop_labeling:
            QMessageBox.information(self, "Information", "Process was stopped!")
        else:
            QMessageBox.information(self, "Information", "Finished!")
        self.label_progress.setValue(total_steps)  # Finalize the progress bar
        QtWidgets.QApplication.processEvents()



    # function to auto label with dino.py
    def on_dino_label_clicked(self):
        self.dino_label.setEnabled(False)  # Disable the button to prevent multiple clicks

        try:
            if self.image_directory is not None:
                # Prompt the user to overwrite or not
                overwrite_reply = QMessageBox.question(
                    self, 'Overwrite Labels',
                    "Do you want to overwrite existing label files?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                overwrite = overwrite_reply == QMessageBox.Yes

                # Create and start the worker thread
                self.dino_worker = DinoWorker(self.image_directory, overwrite)
                self.dino_worker.finished.connect(self.on_dino_finished)
                self.dino_worker.error.connect(self.on_dino_error)
                self.dino_worker.start()
            else:
                QMessageBox.warning(self, 'Directory Not Selected', "Please select an image directory first.")
                self.open_image_video()  # Call the method that lets the user select the directory
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"An error occurred: {e}")
            self.dino_label.setEnabled(True)  # Re-enable the button if there's an error

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

    def load_class_names(self):
        class_names = []
        with open('classes.txt', 'r') as file:
            class_names = [line.strip() for line in file.readlines()]
        return class_names

    def get_all_labels(self):
        all_labels = []

        # Load class names from the classes.txt file
        self.class_names = self.load_class_names()

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
                                    print(f"Warning: Class index {class_index} in {txt_file} is out of range.")
                            except ValueError:
                                print(f"Error parsing line {line_number} in {txt_file}: '{line.strip()}'")

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
                    print(
                        f"No YOLO annotation found for {image_file}. Skipping...")
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
        print("download_videos is called")
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select download directory')
        if not directory:
            return  # Ensure a directory is selected
        
        for row in range(self.video_download.rowCount()):
            item = self.video_download.item(row, 0)  # Fetch the URL from the first column
            if item and item.text().strip():
                url = item.text().strip()
                print(f"Processing URL from row {row}: {url}")
                
                # Create and start the download thread
                download_thread = DownloadThread(url, row, directory)
                download_thread.update_status.connect(self.update_download_status)
                download_thread.update_progress.connect(self.update_download_progress)
                self.download_threads.append(download_thread)
                download_thread.start()
            else:
                print(f"No valid URL found in row {row}, column 0")

    def update_download_progress(self, row, progress):
        # Update the corresponding progress bar based on the row
        self.progress_bars[row].setValue(progress)

    def update_download_status(self, row, status):
        # Only update the status in the table if there is an error or specific message to show.
        if "Failed" in status:
            self.video_download.setItem(row, 1, QTableWidgetItem(status))  # Show errors only

    def enable_image_loading_mode(self):
        self.stop_program()



    def stop_program(self):
        self.timer2.stop()
        self.is_camera_mode = False
        if self.capture:
            self.capture.release()
            self.capture = None

    def get_input_devices(self):
        index = 0
        devices = []
        while True:
            # Force DirectShow
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            ret, _ = cap.read()
            cap.release()

            if not ret:
                break

            devices.append(index)
            index += 1

        return devices

    def setup_input_sources(self):
        self.input_selection.clear()
        # Add "null" so that user can pick "nothing"
        self.input_selection.addItem("null")
        # Add "Desktop"
        self.input_selection.addItem("Desktop")

        # Now add each camera device index
        devices = self.get_input_devices()
        for dev in devices:
            # dev is an integer like 0,1,2
            self.input_selection.addItem(str(dev))  # e.g. "0", "1", etc.

    def on_input_source_changed(self, index):
        # Stop the old capture
        self.stop_program()
        self.capture = None

        # Figure out what user picked
        current_text = self.input_selection.currentText()
        # Only start the timer if it’s NOT "null"
        if current_text != "null":
            self.timer2.start(0)


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

    def display_camera_input(self):
        """
        Display frames from desktop, camera, or video input.
        Perform YOLO inference and save frames if extraction is enabled.
        """
        try:
            # Get the input source
            current_text = self.input_selection.currentText()

            if current_text == "Desktop":
                screenshot = pyautogui.screenshot()
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif current_text.isdigit():
                # Camera input
                if not self.capture or not self.capture.isOpened():
                    self.capture = cv2.VideoCapture(int(current_text), cv2.CAP_DSHOW)
                ret, frame = self.capture.read()
                if not ret:
                    raise ValueError("Unable to read from camera.")
            else:
                # Video input
                ret, frame = self.capture.read()
                if not ret:
                    self.stop_video_playback()
                    return

            # Resize and crop frame if necessary
            frame = self.resize_frame(frame)
            frame = self.crop_frame(frame)

            # Perform YOLO inference
            if hasattr(self, 'model') and self.model:
                # Dynamically fetch model kwargs
                model_kwargs = self.get_model_kwargs()
                results = self.model(frame, **model_kwargs)
                annotated_frame = results[0].plot()
                
                # Save extracted frames on inference
                if self.extracting_frames:
                    self.save_frame_on_inference(frame, results)

                self.update_display(annotated_frame)
            else:
                self.update_display(frame)

        except Exception as e:
            print(f"Error displaying input: {e}")
            self.stop_program()

    def on_crop_images_checkbox_state_changed(self, state):
        """Enable or disable crop dimensions based on checkbox state."""
        self.update_crop_dimensions_enabled(state == QtCore.Qt.Checked)
        if state == QtCore.Qt.Checked:
            # Ensure immediate update if cropping is enabled
            self.display_camera_input()
        
    def resize_frame(self, frame):
        """Resize the frame if custom_size_checkbox is checked."""
        if self.custom_size_checkbox.isChecked():
            width = self.width_box.value()
            height = self.height_box.value()
            return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        return frame
    def update_crop_dimensions_enabled(self, enabled):
        """Enable or disable the width and height controls."""
        self.width_box.setEnabled(enabled)
        self.height_box.setEnabled(enabled)
        if enabled:
            # Set default values if not already set
            if self.width_box.value() == 0:
                self.width_box.setValue(640)  # Example default value
            if self.height_box.value() == 0:
                self.height_box.setValue(480)  # Example default value
    def crop_frame(self, frame):
        """Crop the frame if crop_images_checkbox is checked."""
        if not self.crop_images_checkbox.isChecked():
            return frame

        crop_width = self.width_box.value()
        crop_height = self.height_box.value()
        frame_height, frame_width = frame.shape[:2]

        # Validate dimensions
        if crop_width <= 0 or crop_height <= 0:
            print("Invalid crop dimensions. Ensure width and height are greater than 0.")
            return frame

        # Ensure we don't exceed the frame size
        crop_width = min(crop_width, frame_width)
        crop_height = min(crop_height, frame_height)

        # Calculate a center-crop region
        start_x = max(0, (frame_width - crop_width) // 2)
        start_y = max(0, (frame_height - crop_height) // 2)
        end_x = start_x + crop_width
        end_y = start_y + crop_height

        return frame[start_y:end_y, start_x:end_x]

    
    def on_crop_dimensions_changed(self):
        if self.crop_images_checkbox.isChecked():
            self.display_camera_input()  # Update the display immediately when crop dimensions change

    def update_crop_dimensions_enabled(self, enabled):
        self.width_box.setEnabled(enabled)
        self.height_box.setEnabled(enabled)
        
    def update_display(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.screen_view.width(), self.screen_view.height(), Qt.KeepAspectRatio)
        self.image = QPixmap.fromImage(p)
        scene = QGraphicsScene(0, 0, self.image.width(), self.image.height())
        pixmap_item = QGraphicsPixmapItem(self.image)
        pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        scene.addItem(pixmap_item)
        self.set_screen_view_scene_and_rect(scene)

    def save_frame(self, frame):
        try:
            os.makedirs(self.save_path, exist_ok=True)
            extension = self.get_image_extension()
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}{extension}"
            full_path = os.path.join(self.save_path, filename)
            cv2.imwrite(full_path, frame)
            print(f"Image saved successfully to {full_path}")
        except Exception as e:
            print(f"An error occurred while saving the frame: {e}")
            
    def save_frame_on_inference(self, frame, results):
        """
        Save frames only if YOLO detects objects that meet size and class criteria.
        """
        try:
            # Read class labels from `classes.txt`
            classes_file_path = os.path.join(self.save_path, 'classes.txt')
            with open(classes_file_path, 'r') as f:
                class_labels = [line.strip() for line in f.readlines()]
            class_indices = list(range(len(class_labels)))

            # Parse results
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            # Filter boxes by class indices
            filtered_boxes = [box for i, box in enumerate(boxes) if int(class_ids[i]) in class_indices]
            if not filtered_boxes:
                return  # Skip saving if no valid detections

            # Size filtering
            min_width = self.box_size.value() / 100 * frame.shape[1]
            max_width = self.max_label.value() / 100 * frame.shape[1]
            min_height = self.box_size.value() / 100 * frame.shape[0]
            max_height = self.max_label.value() / 100 * frame.shape[0]

            size_filtered_boxes = [
                box for box in filtered_boxes
                if min_width <= (box[2] - box[0]) <= max_width
                and min_height <= (box[3] - box[1]) <= max_height
            ]

            if size_filtered_boxes:  # Save frame if size criteria met
                os.makedirs(self.save_path, exist_ok=True)
                extension = self.get_image_extension()  # Dynamically get the extension
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}{extension}"
                filepath = os.path.join(self.save_path, filename)
                cv2.imwrite(filepath, frame)
                print(f"Frame saved to {filepath}")

        except Exception as e:
            print(f"Error saving frame on inference: {e}")




    def set_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.save_path = directory
            print(f"Output directory set to: {directory}")

            # Create classes.txt with default 'person' class
            classes_file_path = os.path.join(self.save_path, 'classes.txt')
            if not os.path.exists(classes_file_path):
                with open(classes_file_path, 'w') as f:
                    f.write('person\n')
                print(f"'classes.txt' created with default class 'person'.")

    @pyqtSlot()
    def on_extract_button_clicked(self):
        """
        Handle the 'Extract' button click: start extracting frames with or without inference.
        """
        if not self.capture:
            QMessageBox.warning(self, "Error", "No input source selected!")
            return

        if not self.save_path:
            QMessageBox.warning(self, "Error", "Please set the output directory first.")
            return

        self.extracting_frames = True
        self.timer2.start(30)  # Start timer for processing (adjust for desired FPS)
        print("Started extracting frames.")


    @pyqtSlot()
    def on_stop_extract_clicked(self):
        """
        Stop the extraction process.
        """
        self.extracting_frames = False
        self.timer2.stop()
        print("process stopped.")


    @pyqtSlot(int)
    def update_progress(self, progress):
        self.label_progress.setValue(progress)

    def update_checkboxes(self):
        self.height_box.setEnabled(self.custom_size_checkbox.isChecked())
        self.width_box.setEnabled(self.custom_size_checkbox.isChecked())

    def on_add_video_clicked(self):
        """
        Handle adding a video and loading the associated classes.
        """
        if self.add_video_running:
            return

        self.add_video_running = True
        if self.dialog_open:
            return
        self.dialog_open = True
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter(QCoreApplication.translate("Context", "Videos (*.mp4 *.avi)"))
        if file_dialog.exec_():
            filename = file_dialog.selectedFiles()[0]
            if len(self.video_processor.videos) < 5:
                self.video_processor.add_video(filename)
                row_position = self.video_table.rowCount()
                self.video_table.insertRow(row_position)
                self.video_table.setItem(row_position, 0, QTableWidgetItem(filename))
                print(f"Video added: {filename}")
            else:
                QMessageBox.information(self, "Information", "A maximum of 5 videos can be added at a time.")

        self.dialog_open = False
        self.add_video_running = False


    def on_remove_video_clicked(self):
        current_row = self.video_table.currentRow()
        if current_row != -1:
            self.video_processor.videos.pop(current_row)
            self.video_table.removeRow(current_row)

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
        self.video_processor.set_image_format(text)

    def get_image_extension(self):
        """
        Fetch the image extension based on the currently selected format.
        """
        format_mapping = {"*.JPG": ".jpg", "*.JPEG": ".jpeg", "*.GIF": ".gif", "*.BMP": ".bmp", "*.PNG": ".png"}
        return format_mapping.get(self.image_format.currentText(), ".jpg")


    def get_selected_video_path(self):
        current_row = self.video_table.currentRow()
        if current_row != -1:
            return self.video_table.item(current_row, 0).text()
        return None

    def process_video_file(self, video_path):
        """
        Process video frame-by-frame with YOLO inference and extraction logic.
        """
        try:
            output_dir = self.save_path or os.path.dirname(video_path)
            os.makedirs(output_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Error opening video file: {video_path}")

            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize and crop frame
                frame = self.resize_frame(frame)
                frame = self.crop_frame(frame)

                # Perform YOLO inference
                if hasattr(self, 'model') and self.model:
                    model_kwargs = self.get_model_kwargs()
                    results = self.model(frame, **model_kwargs)
                    self.save_frame_on_inference(frame, results)

                frame_index += 1

            cap.release()
            print(f"Video processing completed. Frames saved to: {output_dir}")

        except Exception as e:
            print(f"Error processing video file: {e}")




    def on_play_video_clicked(self):
        video_path = self.get_selected_video_path()
        if not video_path:
            print("No video selected")
            return

        try:
            self.capture = cv2.VideoCapture(video_path)
            if not self.capture.isOpened():
                print(f"Failed to open video: {video_path}")
                self.capture = None
                return

            # Start the timer for frame-by-frame playback
            self.timer2.timeout.connect(self.play_video_frame)
            self.timer2.start(30)  # 30 ms delay for approximately 33 FPS
            print(f"Playing video: {video_path}")

        except Exception as e:
            print(f"An error occurred during video playback: {e}")

    def play_video_frame(self):
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                self.stop_video_playback()
                print("Video playback completed.")
                return

            frame = self.resize_frame(frame)
            frame = self.crop_frame(frame)

            # Perform YOLO inference if weights are loaded
            if hasattr(self, 'model') and self.model:
                results = self.model(frame)
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame

            # Display the frame in the screen_view
            self.update_display(annotated_frame)
        else:
            self.stop_video_playback()

    def stop_video_playback(self):
        # Stop the timer to pause video playback
        self.timer2.stop()

        # Keep the last displayed frame on the screen
        print("Video playback paused (freeze frame).")


    def resume_video_playback(self):
        if self.capture and self.capture.isOpened():
            self.timer2.start(30)  # Restart the timer for frame-by-frame playback
            print("Video playback resumed.")
        


    # predict and reivew function       
    def process_images_cuda(self, image_files: List[str], target_size: Tuple[int, int] = (256, 256)) -> List[Image.Image]:
        """
        Process single or batch images using OpenCV CUDA.
        Args:
            image_files (List[str]): List of image file paths.
            target_size (Tuple[int, int]): Desired resize dimensions (width, height).
        Returns:
            List[Image.Image]: List of processed PIL images.
        """
        try:
            stream = cv2.cuda.Stream()
            gpu_images = []
            pil_images = []

            for image_file in image_files:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(cv2.imread(image_file))  # Upload image to GPU

                # Process image on GPU
                gpu_img_rgb = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2RGB, stream=stream)
                gpu_img_resized = cv2.cuda.resize(gpu_img_rgb, target_size, stream=stream)

                gpu_images.append(gpu_img_resized)

            stream.waitForCompletion()  # Ensure all operations are complete

            # Download images to CPU and convert to PIL
            pil_images = [Image.fromarray(gpu_img.download()) for gpu_img in gpu_images]

            return pil_images
        except Exception as e:
            print(f"Error processing images: {str(e)}")
            return []

    def lazy_image_batch_generator(self, image_files: List[str], batch_size: int):
        """
        Lazy generator for image batches.
        Args:
            image_files (List[str]): List of image file paths.
            batch_size (int): Number of images per batch.
        Yields:
            List[str]: Batch of image file paths.
        """
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
            print("No images found in the directory.")
            return

        batch_generator = self.lazy_image_batch_generator(image_files, batch_size)

        for batch in batch_generator:
            pil_images = self.process_images_cuda(batch)

            for image_file, pil_image in zip(batch, pil_images):
                label_file = os.path.splitext(image_file)[0] + '.txt'
                if not os.path.exists(label_file):
                    print(f"No label file found for {image_file}. Skipping.")
                    continue

                # Read label data for the image
                with open(label_file, 'r') as file:
                    labels = file.readlines()

                # Process each label and create thumbnails
                for label_index, label_data in enumerate(labels):
                    self.create_thumbnail_for_label(image_file, pil_image, label_data, label_index, thumbnails_directory)

        # Refresh UI after processing
        self.preview_list.setRowCount(0)  # Clear existing rows
        self.update_list_view(self.filtered_image_files)
        self.preview_list.resizeRowsToContents()
        self.preview_list.resizeColumnsToContents()

        print(f"Processing completed for {len(image_files)} images.")

            
    def create_thumbnail_for_label(self, image_file, pil_image, label_data, label_index, thumbnails_directory):
        """
        Create and save a thumbnail for a specific label of an image.
        Args:
            image_file (str): Path to the original image file.
            pil_image (PIL.Image): Processed PIL image.
            label_data (str): Label data string (bounding box and class).
            label_index (int): Index of the label in the image's label file.
            thumbnails_directory (str): Directory to save the thumbnails.
        """
        try:
            # Parse label data (class_id, x_center, y_center, width, height)
            parts = label_data.strip().split()
            if len(parts) != 5:
                print(f"Invalid label data: {label_data}")
                return

            class_id, x_center, y_center, width_ratio, height_ratio = map(float, parts)
            img_width, img_height = pil_image.size
            x_center *= img_width
            y_center *= img_height
            width = width_ratio * img_width
            height = height_ratio * img_height

            # Compute bounding box coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Crop the region corresponding to the bounding box
            cropped_image = pil_image.crop((x1, y1, x2, y2))

            # Save thumbnail
            base_file = os.path.splitext(os.path.basename(image_file))[0]
            thumbnail_filename = os.path.join(thumbnails_directory, f"{base_file}_{label_index}.jpeg")
            cropped_image.save(thumbnail_filename, "JPEG")
            print(f"Saved thumbnail: {thumbnail_filename}")

        except Exception as e:
            print(f"Error creating thumbnail for {image_file} label {label_index}: {str(e)}")


    def start_debounce_timer(self, value):
        self._image_size_value = min(value, self.MAX_SIZE)
        self._debounce_timer.start(300)

    def _perform_size_adjustment(self):
        """Resize thumbnails based on user input."""
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
        self.preview_list.setRowCount(0)
        QApplication.processEvents()  # Ensure the UI processes the row deletion
        print("Processing stopped and UI cleared.")
        if self.preview_list.rowCount() != 0:
            print(f"Error: Table still contains {self.preview_list.rowCount()} rows after clear attempt.")
        else:
            print("Table cleared successfully.")

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
                        print(f"Ignored event for row: {index.row()}, column: {index.column()}")
            return super().eventFilter(source, event)
        except Exception as e:
            print(f"An unexpected error occurred in eventFilter: {e}")
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
        print(f"Right-clicked on row: {index.row()}")
        row = index.row()
        try:
            image_item = self.preview_list.item(row, 0)
            label_item = self.preview_list.item(row, 4)
            if image_item is not None and label_item is not None:
                # Pass both the image file and label (bounding box) for deletion
                self.delete_item(row, image_item, label_item)
                
                # Save the updated bounding boxes immediately
                image_file = image_item.text()
                label_file = os.path.splitext(image_file)[0] + '.txt'
                img_width, img_height = self.get_image_dimensions(image_file)
                self.save_bounding_boxes(label_file, img_width, img_height)
                
                QApplication.processEvents()
            else:
                print("Image or label item is None.")
        except Exception as e:
            print(f"An unexpected error occurred in handle_right_click: {str(e)}")


    def delete_item(self, row, image_item, label_item):
        image_file = image_item.text()  # Image file path
        bbox_data = label_item.text()  # Bounding box data

        # Delete bounding box from the corresponding file
        self.update_label_file(image_file, bbox_data)

        # Remove the bounding box visually
        self.delete_thumbnail(image_file, label_item.data(Qt.UserRole))
        self.preview_list.removeRow(row)  # Remove row from table

        # Re-align indices and update the bounding box data
        self.realign_remaining_entries()
        self.update_thumbnail_indices(image_file)
        self.synchronize_list_view(image_file)


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
            print(f"Error getting dimensions for {image_file}: {str(e)}")
            return 0, 0


    def update_label_file(self, image_file, bbox_to_delete):
        label_file = os.path.splitext(image_file)[0] + '.txt'  # Corresponding label file

        try:
            # Read existing bounding boxes
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Normalize the bounding box to ensure consistent matching
            bbox_to_delete = " ".join(map(str, map(float, bbox_to_delete.split())))

            # Filter out the bounding box to delete
            with open(label_file, 'w') as f:
                for line in lines:
                    normalized_line = " ".join(map(str, map(float, line.strip().split())))
                    if normalized_line != bbox_to_delete:
                        f.write(line)

            print(f"Bounding box successfully deleted from {label_file}")
        except IOError as e:
            print(f"Error updating label file: {str(e)}")




    def delete_thumbnail(self, image_file, bbox_index):
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        thumbnail_png = os.path.join(self.thumbnails_directory, f"{base_filename}_{bbox_index}.png")
        thumbnail_jpeg = os.path.join(self.thumbnails_directory, f"{base_filename}_{bbox_index}.jpeg")

        if not self.attempt_delete_thumbnail(thumbnail_png):
            print(f"Thumbnail not found: {thumbnail_png}")
        if not self.attempt_delete_thumbnail(thumbnail_jpeg):
            print(f"Thumbnail not found: {thumbnail_jpeg}")

    def update_thumbnail_indices(self, image_file):
        """
        Update bounding box indices for thumbnails after a deletion.
        Args:
            image_file (str): Path to the image file.
        """
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        updated_bboxes = []

        label_file = os.path.splitext(image_file)[0] + '.txt'
        if os.path.exists(label_file):
            # Read the updated label file
            with open(label_file, 'r') as file:
                updated_bboxes = file.readlines()

        # Update thumbnails for the remaining bounding boxes
        for index, bbox in enumerate(updated_bboxes):
            old_thumbnail_path = os.path.join(self.thumbnails_directory, f"{base_filename}_{index}.jpeg")
            if os.path.exists(old_thumbnail_path):
                os.rename(
                    old_thumbnail_path,
                    os.path.join(self.thumbnails_directory, f"{base_filename}_{index}.jpeg")
                )


    def attempt_delete_thumbnail(self, thumbnail_path):
        if os.path.exists(thumbnail_path):
            try:
                os.remove(thumbnail_path)
                print(f"Successfully deleted {thumbnail_path}")
                return True
            except Exception as e:
                print(f"Failed to delete {thumbnail_path}: {str(e)}")
        else:
            print(f"Thumbnail not found: {thumbnail_path}")
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
            print("No image item found.")
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
            print(f"Attempting to flash bounding box with index: {bounding_box_index} for file: {image_file}")
            if bounding_box_index is not None:
                self.flash_bounding_box(bounding_box_index, image_file)
            else:
                print("Bounding box index not found.")
        else:
            print("Bounding box item not found.")
        self.synchronize_list_view(image_file)
    
    def flash_bounding_box(self, bbox_index, image_file):
        if bbox_index is None:
            print(f"Invalid bounding box index: {bbox_index} for file: {image_file}")
            return

        found = False
        unique_id = f"{image_file}_{bbox_index}"
        for rect_item in self.screen_view.scene().items():
            if isinstance(rect_item, BoundingBoxDrawer) and rect_item.unique_id == unique_id:
                rect_item.flash_color = QColor(*self.flash_color_rgb)
                rect_item.alternate_flash_color = QColor(*self.alternate_flash_color_rgb)
                rect_item.start_flashing(100, self.flash_time_value)  # Flash with interval of 100ms
                found = True
                break

        if found:
            print(f"Flashing initiated successfully for {unique_id}.")
        else:
            print(f"No matching bounding box found for {unique_id}")
            # Suggest re-synchronizing indices
            self.update_thumbnail_indices(image_file)



    def synchronize_list_view(self, image_file):
        file_basename = os.path.basename(image_file)
        self.List_view.clearSelection()
        list_view_model = self.List_view.model()
        for i in range(list_view_model.rowCount()):
            if list_view_model.item(i).text() == file_basename:
                matching_index = list_view_model.index(i, 0)  # Assuming the image file name is in the first column
                self.List_view.scrollTo(matching_index, QAbstractItemView.PositionAtCenter)
                self.List_view.selectionModel().select(matching_index, QItemSelectionModel.Select)
                print(f"ListView synchronized for {file_basename}")
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
                print("List_view model is not set.")
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
                
    def load_class_names(self, data_directory):
        """
        Load class names from classes.txt and map them to indices.
        """
        classes_file_path = os.path.join(data_directory, 'classes.txt')
        self.id_to_class = {}
        if os.path.exists(classes_file_path):
            with open(classes_file_path, 'r') as file:
                self.id_to_class = {i: line.strip() for i, line in enumerate(file)}
            print(f"Loaded classes: {self.id_to_class}")
        else:
            print(f"Warning: No classes.txt found in {data_directory}")



    def adjust_column_width(self, value):
        self._image_size_value = min(value, self.MAX_SIZE)
        self._perform_size_adjustment()


    def toggle_image_display(self):
        show_images = not self.dont_show_img_checkbox.isChecked()
        self.preview_list.setColumnWidth(0, 200 if show_images else 0)

    def extract_and_display_data(self):
        """
        Extract bounding boxes and display data filtered by the selected class in the filter_class_spinbox.
        """
        self.processing = True

        try:
            if self.image_directory is None:
                QMessageBox.warning(self.loader.main_window, "No Directory Selected", "Please select a directory before previewing images.")
                return

            data_directory = self.image_directory
            self.ui_loader.setup_ui(show_images=not self.dont_show_img_checkbox.isChecked())
            self.thumbnails_directory = os.path.join(data_directory, "thumbnails")
            os.makedirs(self.thumbnails_directory, exist_ok=True)
            print(f"Thumbnails directory confirmed at: {self.thumbnails_directory}")

            # Only process the filtered image files
            image_files = self.filtered_image_files if self.filtered_image_files else self.image_files

            if not image_files:
                QMessageBox.warning(self, "No Images Found", "No images found. Please load images before adjusting the slider.")
                return

            self.load_class_names(data_directory)
            current_value = self.image_size.value()
            self.image_size.setValue(current_value + 1)
            batch_size = self.batch_size_spinbox.value()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.label_progress.setMaximum(len(image_files))
            self.label_progress.setValue(0)

            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Keeps original thumbnail size while transforming
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            with ThreadPoolExecutor(max_workers=10) as executor:
                for i in range(0, len(image_files), batch_size):
                    batch_images = image_files[i:i + batch_size]
                    if not self.processing:
                        break

                    try:
                        images = [Image.open(img_file).convert('RGB') for img_file in batch_images]
                        original_sizes = [img.size for img in images]
                        tensor_images = torch.stack([transform(img) for img in images]).to(device)
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        continue

                    for idx, (image_file, original_size) in enumerate(zip(batch_images, original_sizes)):
                        try:
                            tensor_img = tensor_images[idx].unsqueeze(0).to(device)
                            tensor_img = tensor_img.squeeze().cpu()
                            transformed_img = transforms.ToPILImage()(tensor_img)

                            q_image = QImage(image_file)
                            pixmap = QPixmap.fromImage(q_image)

                            base_file = os.path.splitext(os.path.basename(image_file))[0]
                            label_file = f"{base_file}.txt"
                            label_path = os.path.join(data_directory, label_file)
                            if not os.path.exists(label_path):
                                print(f"Label file not found for {image_file}. Skipping.")
                                continue

                            with open(label_path, 'r') as file:
                                lines = file.readlines()
                        except Exception as e:
                            print(f"Error reading labels for {image_file}: {e}")
                            continue

                        original_width, original_height = original_size
                        for j, line in enumerate(lines):
                            try:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    print(f"Invalid label format in {label_file}: {line.strip()}")
                                    continue

                                class_id = int(parts[0])
                                current_filter_text = self.filter_class_spinbox.currentText()
                                if current_filter_text.startswith("All"):
                                    current_filter = -1
                                elif current_filter_text.startswith("Blanks"):
                                    current_filter = -2
                                else:
                                    current_filter = int(current_filter_text.split(":")[0])  # Extract the class index

                                # Skip if the class doesn't match the current filter (except for "All")
                                if current_filter != -1 and class_id != current_filter:
                                    continue

                                class_name = self.id_to_class.get(class_id)
                                if class_name is None:
                                    continue

                                x_center, y_center, width_ratio, height_ratio = map(float, parts[1:])
                                x_center *= original_width
                                y_center *= original_height
                                width = width_ratio * original_width
                                height = height_ratio * original_height

                                x1 = int(x_center - width / 2)
                                y1 = int(y_center - height / 2)
                                x2 = int(x_center + width / 2)
                                y2 = int(y_center + height / 2)

                                cropped_pixmap = pixmap.copy(x1, y1, x2 - x1, y2 - y1)

                                # Get dimensions of the cropped pixmap
                                width_cropped = cropped_pixmap.width()
                                height_cropped = cropped_pixmap.height()

                                # Decide on the extension based on width and height
                                if width_cropped <= 32 or height_cropped <= 32:
                                    extension = "png"
                                else:
                                    extension = "jpeg"

                                # Save the cropped image in its original size to the directory
                                thumbnail_filename = os.path.join(self.thumbnails_directory, f"{base_file}_{j}.{extension}")
                                executor.submit(cropped_pixmap.save, thumbnail_filename, extension.upper())

                                if not self.dont_show_img_checkbox.isChecked():
                                    # Resize cropped pixmap for display in the UI
                                    resized_pixmap = cropped_pixmap.scaled(
                                        128, 128,
                                        Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation
                                    )

                                    thumbnail_label = QLabel()
                                    thumbnail_label.setPixmap(resized_pixmap)
                                    thumbnail_label.setAlignment(Qt.AlignCenter)

                                    row_count = self.preview_list.rowCount()
                                    self.preview_list.insertRow(row_count)
                                    self.preview_list.setItem(row_count, 0, QTableWidgetItem(image_file))
                                    self.preview_list.setCellWidget(row_count, 0, thumbnail_label)
                                    self.preview_list.setItem(row_count, 1, QTableWidgetItem(class_name))
                                    self.preview_list.setItem(row_count, 2, QTableWidgetItem(str(class_id)))
                                    self.preview_list.setItem(row_count, 3, QTableWidgetItem(f"{int(width)}x{int(height)}"))
                                    self.preview_list.setItem(row_count, 4, QTableWidgetItem(line.strip()))

                                    bounding_box_item = QTableWidgetItem(line.strip())
                                    bounding_box_item.setData(Qt.UserRole, j)
                                    self.preview_list.setItem(row_count, 4, bounding_box_item)
                                    self.label_progress.setValue(self.label_progress.value() + 1)

                                    # Resize rows and columns immediately after inserting each item
                                    self.preview_list.resizeRowsToContents()
                                    self.preview_list.resizeColumnsToContents()

                                    QApplication.processEvents()
                            except Exception as e:
                                print(f"Error processing bounding box for {image_file}: {e}")
                                continue

                self._perform_size_adjustment()  # Final size adjustment after all data is loaded

        except Exception as e:
            print(f"An unexpected error occurred in extract_and_display_data: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")



                 


    # mute sounds

    def mute_player(self, state):
        # Check the state of the muteCheckBox
        is_muted = state == QtCore.Qt.Checked

        # Set the muted state of the sound_player
        self.sound_player.setMuted(is_muted)
        # settings

    def keyPressEvent(self, event):
        key = event.text()  # Get the actual key pressed (text)
        modifiers = QApplication.keyboardModifiers()  # Get the keyboard modifiers

        # Handle next and previous frame navigation
        if key == self.settings.get('nextButton'):
            if modifiers == Qt.NoModifier:
                # Stop any ongoing timers to avoid interference
                self.stop_next_timer()
                self.stop_prev_timer()

                # Navigate to the next frame
                self.next_frame()

                # Restart scanning if auto_scan is checked
                if self.auto_scan_checkbox.isChecked():
                    self.start_next_timer()

            elif modifiers == Qt.ControlModifier:
                # Quick navigation (without scanning)
                self.quick_next_navigation()

        elif key == self.settings.get('previousButton'):
            if modifiers == Qt.NoModifier:
                # Stop any ongoing timers to avoid interference
                self.stop_next_timer()
                self.stop_prev_timer()

                # Navigate to the previous frame
                self.previous_frame()

                # Restart scanning if auto_scan is checked
                if self.auto_scan_checkbox.isChecked():
                    self.start_prev_timer()

            elif modifiers == Qt.ControlModifier:
                # Quick navigation (without scanning)
                self.quick_previous_navigation()

        # Handling delete current image (Ctrl or no modifier)
        elif key == self.settings.get('deleteButton'):
            # Stop ongoing timers to avoid unintended behavior during delete
            self.stop_next_timer()
            self.stop_prev_timer()
            self.delete_current_image()  # Same action for Ctrl and no Ctrl

        else:
            # Add key to buffer and start timer for delayed processing
            self.keyBuffer += key
            if not self.timer.isActive():
                self.timer.start(300)


    def processKeyPresses(self):
        key = self.keyBuffer  # Process the buffered key(s)
        modifiers = QApplication.keyboardModifiers()  # Check modifiers again

        # Handling buffered next and previous frame navigation
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

        # Handling delete current image
        elif key == self.settings.get('deleteButton'):
            self.delete_current_image()  # Same action for Ctrl and no Ctrl

        # Handle auto label shortcut
        else:
            # Handling class hotkeys dynamically
            class_hotkeys = {k: v for k, v in self.settings.items() if k.startswith('classHotkey_')}
            class_name = None
            for class_key, hotkey in class_hotkeys.items():
                if hotkey == key:
                    class_name = class_key.split('classHotkey_')[-1]
                    break

            if class_name:
                print(f"Class name found: {class_name}")
                index = self.classes_dropdown.findText(class_name)
                if index >= 0:
                    self.classes_dropdown.setCurrentIndex(index)

        # Clear the key buffer after processing
        self.keyBuffer = ""


    def openSettingsDialog(self):
        print("Settings button was clicked.")
        settingsDialog = SettingsDialog(self)
        settingsDialog.exec_()

    def loadSettings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
        except FileNotFoundError:
            settings = self.defaultSettings()
        except json.JSONDecodeError:
            print("Error decoding JSON, using default settings")
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
            'batchSize': 1000
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


    def update_slider_values(self):
        # Code to handle slider value changes
        pass



    def slider_value_changed(self):
        if not all(hasattr(self, attr) for attr in ['current_file', 'weights_file_path', 'cfg_file_path']):
            logging.info("Missing one or more attributes: current_file, weights_file_path, cfg_file_path")
            return

        if not hasattr(self, 'net'):
            self.initialize_yolo()

        # Ensure there is a selected and valid image
        if not self.current_file or not os.path.exists(self.current_file):
            logging.info("No valid image selected")
            return

        # Update thresholds
        self.confidence_threshold = self.confidence_threshold_slider.value() / 100
        self.nms_threshold = self.nms_threshold_slider.value() / 100

        logging.info(f"Confidence Threshold: {self.confidence_threshold:.2f}, NMS Threshold: {self.nms_threshold:.2f}")

        # Restart the timer for processing the image
        self.process_timer.start(self.process_timer_interval)


    @functools.lru_cache(maxsize=None)  # No size limit for the cache
    def get_cached_image(self, file_path: str) -> Optional[np.ndarray]:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load image at {file_path}")
        return image

    @functools.lru_cache(maxsize=None)  # No size limit for the cache
    def get_cached_blob(self, file_path: str) -> Optional[np.ndarray]:
        image = self.get_cached_image(file_path)

        if image is None:
            return None  # If image loading failed, return None

        # Check if cropping is active
        if self.crop_true.isChecked():
            desired_height = self.crop_height.value()
            desired_width = self.crop_width.value()

            cropped_image = self.center_crop(image, desired_height, desired_width, file_path)  # pass file_path
            if cropped_image is not None:
                return cropped_image  # Return the cropped image

        return image  # Return the original image if cropping is not possible or not enabled
    def ensure_classes_loaded(self):
        if not hasattr(self, 'class_labels'):
            classes_file_path = os.path.join(self.image_directory, 'classes.txt')
            if not os.path.exists(classes_file_path):
                print("Classes file not found.")
                return False

            with open(classes_file_path, 'r') as classes_file:
                self.class_labels = [line.strip() for line in classes_file.readlines()]

        return True


    def open_weights(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Weights/Model", "", "Model Files (*.pt *.engine *.onnx *.weights)", options=options)

        if file_name:
            self.weights_file_path = file_name  # Store the weights file path
            try:
                if file_name.endswith('.engine'):
                    engine_input_size = get_engine_input_size(file_name)
                    if engine_input_size:
                        self.network_width.setValue(engine_input_size[2])
                        self.network_height.setValue(engine_input_size[3])
                        print(f"Set imgsz to {engine_input_size[2]}x{engine_input_size[3]} for `.engine` file.")
                    else:
                        print("Unable to determine input size from `.engine` file.")

                self.model = YOLO(file_name)  # Initialize the model
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
            return 'yolov8'
        elif file_path.endswith('.engine'):
            return 'yolov8_trt'
        elif file_path.endswith('.onnx'):
            return 'onnx'
        elif file_path.endswith('.weights'):
            return 'weights'
        else:
            return 'unknown'


    def auto_label_images2(self):
        logging.info("auto_label_images2 called")

        classes_file_path = os.path.join(self.image_directory, 'classes.txt')
        if not os.path.exists(classes_file_path):
            QMessageBox.critical(self, "Error", "Classes file not found.")
            return

        # Load class labels
        with open(classes_file_path, 'r') as classes_file:
            class_labels = [line.strip() for line in classes_file.readlines()]
        class_indices = list(range(len(class_labels)))

        # Ensure model is loaded
        if not hasattr(self, 'model'):
            QMessageBox.critical(self, "Error", "Model is not initialized.")
            return

        total_images = len(self.image_files)
        self.label_progress.setRange(0, total_images)

        overwrite = QMessageBox.question(
            self, 'Overwrite Labels', "Do you want to overwrite existing labels?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        for idx, image_file in enumerate(self.image_files):
            if self.stop_labeling:
                logging.info("Labeling stopped by user.")
                break

            # Cropping logic
            if self.crop_true.isChecked():
                image = self.read_image(image_file)
                desired_height = self.crop_height.value()
                desired_width = self.network_width.value()
                cropped_image = self.center_crop(image, desired_height, desired_width, image_file)

                if cropped_image is not None:
                    self.save_cropped_image(cropped_image, image_file)
                    image_file = os.path.join(self.current_cropped_directory, os.path.basename(image_file))

            # Apply preprocessing to the image
            original_image = cv2.imread(image_file)
            if original_image is None:
                logging.warning(f"Failed to read image: {image_file}. Skipping.")
                continue

            processed_image = self.apply_preprocessing(original_image)

            self.current_file = image_file
            label_filename = os.path.splitext(os.path.basename(image_file))[0] + '.txt'
            label_file = os.path.join(self.image_directory, label_filename)
            label_exists = os.path.exists(label_file)

            self.display_image(image_file)
            with Image.open(image_file) as img:
                img_width, img_height = img.size

            # Get model parameters using get_model_kwargs
            model_kwargs = self.get_model_kwargs()

            try:
                logging.info(f"Processing image: {image_file} with params: {model_kwargs}")
                results = self.model(processed_image, **model_kwargs)

                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                labeled_boxes = list(zip(boxes, class_ids))

            except AttributeError as e:
                logging.error(f"Error processing image: {e}")
                QMessageBox.critical(self, "Error", f"An error occurred while processing {image_file}: {e}")
                continue

            # Size filtering
            min_width_px, max_width_px = self.box_size.value() / 100 * img_width, self.max_label.value() / 100 * img_width
            min_height_px, max_height_px = self.box_size.value() / 100 * img_height, self.max_label.value() / 100 * img_height

            size_filtered_boxes = [
                (box, class_id) for box, class_id in labeled_boxes
                if min_width_px <= (box[2] - box[0]) <= max_width_px and
                min_height_px <= (box[3] - box[1]) <= max_height_px
            ]

            # Save size-filtered labels
            if overwrite == QMessageBox.Yes or not label_exists:
                with open(label_file, 'w') as f:
                    for box, class_id in size_filtered_boxes:
                        x1, y1, x2, y2 = box
                        xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                        xc, yc, w, h = xc / img_width, yc / img_height, w / img_width, h / img_height
                        f.write(f"{int(class_id)} {xc} {yc} {w} {h}\n")

                        # Head logic
                        if self.heads_area.isChecked() and class_id != 1:
                            head_class_id_str = self.class_id.text()
                            if head_class_id_str.strip():
                                try:
                                    head_class_id = int(head_class_id_str)
                                except ValueError:
                                    logging.warning(f"Invalid class ID '{head_class_id_str}'. Using the next available class ID.")
                                    head_class_id = self.get_next_available_class_id()
                            else:
                                logging.info("No class ID provided. Using the next available class ID.")
                                head_class_id = self.get_next_available_class_id()

                            head_x, head_y, head_w, head_h = self.calculate_head_area(x1, y1, x2 - x1, y2 - y1)
                            head_xc, head_yc, head_w, head_h = (
                                (head_x + head_w / 2) / img_width,
                                (head_y + head_h / 2) / img_height,
                                head_w / img_width,
                                head_h / img_height
                            )
                            f.write(f"{head_class_id} {head_xc} {head_yc} {head_w} {head_h}\n")
            else:
                logging.info(f"Skipping file {image_file} as it already has labels and overwrite is set to 'No'.")

            self.label_progress.setValue(idx + 1)
            QApplication.processEvents()

        self.label_progress.setValue(total_images)
        QMessageBox.information(self, "Information", "Finished!")
        
    def apply_preprocessing(self, image):
        """
        Apply selected preprocessing (grayscale, edge detection, super-resolution)
        to the image for inference.
        """
        if self.grayscale_Checkbox.isChecked():
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for model compatibility

        if self.outline_Checkbox.isChecked():
            edges = cv2.Canny(image, self.slider_min_value, self.slider_max_value)
            image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to 3 channels for display

        if self.super_resolution_Checkbox.isChecked():
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = os.path.join(os.getcwd(), "Sam", "FSRCNN_x4.pb")
            if os.path.isfile(model_path):
                sr.readModel(model_path)
                sr.setModel("fsrcnn", 4)
                image = sr.upsample(image)
            else:
                logging.warning(f"Super-resolution model not found at {model_path}. Skipping.")

        return image


    def get_model_kwargs(self):
        """
        Generate kwargs for YOLO inference, including dynamic input size adjustment.
        """
        conf_threshold = self.confidence_threshold_slider.value() / 100
        iou_threshold = self.nms_threshold_slider.value() / 100
        network_height = self.network_height.value()
        network_width = self.network_width.value()

        model_kwargs = {
            'conf': conf_threshold,
            'iou': iou_threshold,
            'imgsz': [network_width, network_height],  # Dynamic input size
            'batch': 1,
            'device': 0,
        }

        if self.fp_select_combobox.currentText() == "FP16":
            model_kwargs['half'] = True

        return model_kwargs
    
    def validate_input_size(self):
        """
        Validate if the configured input size matches the model's expected size.
        """
        if not hasattr(self, 'weights_file_path') or not self.weights_file_path.endswith('.engine'):
            QMessageBox.warning(self, "Error", "Validation can only be performed for `.engine` files.")
            return

        engine_input_size = get_engine_input_size(self.weights_file_path)  # Get expected dimensions from the `.engine` file
        if engine_input_size:
            expected_width, expected_height = engine_input_size[2], engine_input_size[3]
        else:
            QMessageBox.critical(self, "Error", "Unable to determine input size from `.engine` file.")
            return

        # Compare configured input size with expected size
        if (self.network_width.value() != expected_width or
                self.network_height.value() != expected_height):
            QMessageBox.warning(
                self, "Size Mismatch",
                f"Input size does not match the `.engine` file. "
                f"Expected: {expected_width}x{expected_height}, "
                f"Got: {self.network_width.value()}x{self.network_height.value()}"
            )
        else:
            QMessageBox.information(
                self, "Size Match",
                f"Input size matches the `.engine` file: {expected_width}x{expected_height}"
            )

    def process_current_image(self):
        if not hasattr(self, 'current_file'):
            print("No image file is currently opened.")
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
            print("Error: net is not initialized. Call initialize_yolo() first.")
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
            print('Using CUDA:', 'FP16' if self.yolo_floating_point else 'FP32')

            # Set the preferable target
            self.net.setPreferableTarget(target)

        elif cv2.ocl.haveOpenCL():
            # Check if OpenCL is available (usually for some CPUs and GPUs)
            print('Using OpenCL')

            # Set OpenCL as the backend
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

            # Set OpenCL as the target
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        else:
            # If no hardware acceleration is available, use CPU
            print('Using CPU')

            # Set default CPU backend
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

            # Set CPU as the target
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



    def initialize_yolo(self):
        try:
            # Initialize an empty list to hold class names
            self.classes = []

            if hasattr(self, 'weights_file_path') and hasattr(self, 'cfg_file_path'):
                file_extension = os.path.splitext(self.weights_file_path)[1]

                if file_extension == '.weights':
                    # Use OpenCV for .weights files
                    self.net = cv2.dnn.readNet(self.weights_file_path, self.cfg_file_path)

                    # Initialize floating point based on the ComboBox
                    current_text = self.fp_select_combobox.currentText()
                    self.yolo_floating_point = 0 if current_text == "FP32" else 1

                    # Apply the backend and target settings based on the initialized attributes
                    self.apply_backend_and_target()

                    # Get the names of the output layers
                    self.layer_names = self.net.getLayerNames()
                    unconnected_out_layers = self.net.getUnconnectedOutLayers()
                    print("Unconnected out layers:", unconnected_out_layers)

                    # Adjust your indexing to work with a numpy array of numbers
                    self.layer_names = [self.layer_names[i[0] - 1] for i in unconnected_out_layers]

                    # Load classes.txt file from the image directory
                    classes_file_path = os.path.join(self.image_directory, 'classes.txt')
                    if os.path.exists(classes_file_path):
                        with open(classes_file_path, 'r') as f:
                            self.classes = f.read().strip().split('\n')
                    else:
                        print("Error: classes.txt not found. Please ensure the classes.txt file is in the image directory.")

                else:
                    print("Unsupported file extension for weights file. Please use a .weights file.")
            else:
                print("Weights and/or CFG files not selected. Please ensure both files are selected for .weights models.")
        except Exception as e:
            print(f"Error initializing YOLO: {e}")


    def get_label_file(self, image_file):
        if image_file is None:
            return None  # Handle the case where image_file is None

        try:
            base = os.path.basename(image_file)  # Get "image.jpg"
            name = os.path.splitext(base)[0]  # Get "image"

            # Get directory of current file
            directory = os.path.dirname(self.current_file)

            # Construct label filename
            label_filename = name + ".txt"

            # Join directory with the new label filename
            label_file = os.path.join(directory, label_filename)

            return label_file
        except Exception as e:
            print(f"Error while getting label file: {e}")
            return None  # Handle any other errors gracefully


    def get_label_file_and_exists(self, image_path):
        # Replace the image extension with '.txt' to get the label file path
        label_file = os.path.splitext(image_path)[0] + '.txt'

        # Check if the label file exists
        label_exists = os.path.exists(label_file)

        return label_file, label_exists


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

            print(f"Saved YOLO label for cropped image: {cropped_label_file}")
        except Exception as e:
            print(f"Error saving YOLO label for cropped image: {e}")

    def process_image(self, overwrite):
        if not hasattr(self, 'current_file'):
            print("No image file is currently opened.")
            return

        label_file = self.get_label_file(self.current_file)

        if label_file is None:
            print("Label file could not be generated. Skipping image.")
            return

        if os.path.exists(label_file) and not overwrite:
            return

        image = self.read_image(self.current_file)

        # Perform preprocessing (cropping, grayscale, outline if enabled)
        if self.crop_true and self.crop_true.isChecked():
            desired_height = self.crop_height.value()
            desired_width = self.crop_width.value()
            image = self.center_crop(image, desired_height, desired_width, self.current_file)

        if self.grayscale_Checkbox and self.grayscale_Checkbox.isChecked():
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.merge((image, image, image))
            alpha = self.grey_scale_slider.value() / 50.0
            beta = self.grey_scale_slider.value()
            image = self.adjust_brightness_contrast(image, alpha, beta)

        if self.outline_Checkbox and self.outline_Checkbox.isChecked():
            edges = cv2.Canny(image, self.slider_min_value, self.slider_max_value)
            image[edges > 0, :3] = [255, 255, 255]

        # Overwrite bounding boxes during inference
        self.infer_and_display_bounding_boxes(image)
        
        # Save the detected bounding boxes
        self.save_bounding_boxes(label_file, self.screen_view.scene().width(), self.screen_view.scene().height())


    def auto_label_images(self):
        """
        Auto-label images using YOLO model with .weights and .cfg files.
        Includes overwrite functionality similar to auto_label_images2.
        """
        if self.model_type == 'weights' and (not hasattr(self, 'weights_file_path') or not hasattr(self, 'cfg_file_path')):
            QMessageBox.warning(self, "Missing Files", "Both .weights and .cfg files are required for this model type.")
            return

        # Validate the existence of the classes.txt file
        classes_file_path = os.path.join(self.image_directory, 'classes.txt')
        if not os.path.exists(classes_file_path):
            logging.error("Classes file not found.")
            return

        # Initialize YOLO if not already initialized
        if not hasattr(self, 'net'):
            self.initialize_yolo()

        # Load class labels
        with open(classes_file_path, 'r') as classes_file:
            class_labels = [line.strip() for line in classes_file.readlines()]

        total_images = len(self.image_files)
        self.label_progress.setRange(0, total_images)

        # Prompt user to overwrite existing labels
        overwrite = QMessageBox.question(
            self, 'Overwrite Labels',
            "Do you want to overwrite existing labels? If 'No', existing labels will be preserved.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        # Process each image
        for idx, image_file in enumerate(self.image_files):
            if self.stop_labeling:
                logging.info("Labeling stopped by user.")
                break

            self.current_file = image_file
            label_file = self.get_label_file(self.current_file)

            # Skip existing labels if overwrite is not selected
            if os.path.exists(label_file) and overwrite == QMessageBox.No:
                logging.info(f"Skipping {image_file} as label exists and overwrite is disabled.")
                continue

            # Load and preprocess the image
            original_image = self.read_image(self.current_file)
            if original_image is None:
                logging.error(f"Failed to load image {self.current_file}. Skipping.")
                continue

            # Apply cropping if enabled
            if self.crop_true.isChecked():
                desired_height = self.crop_height.value()
                desired_width = self.crop_width.value()
                cropped_image = self.center_crop(original_image, desired_height, desired_width, self.current_file)
                if cropped_image is not None:
                    self.current_file = self.save_cropped_image(cropped_image, self.current_file)
                    self.display_image(cropped_image)
                else:
                    self.display_image(original_image)

            self.display_image(self.current_file)

            # Perform YOLO inference and save labels
            self.process_image(overwrite)

            # Update progress bar
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

        classes_file_path = os.path.join(self.image_directory, 'classes.txt')
        try:
            with open(classes_file_path, 'r') as classes_file:
                self.class_labels = [line.strip() for line in classes_file.readlines()]
        except FileNotFoundError:
            logging.error(f"Could not find the file at {classes_file_path}")
            return False

        if self.weights_file_path.endswith(('.pt', '.engine', '.onnx')):
            print("PyTorch model detected. Running auto_label_images2.")
            self.auto_label_images2()
        elif self.weights_file_path.endswith('.weights'):
            print("Darknet model detected. Running auto_label_images.")
            if not hasattr(self, 'net'):
                self.initialize_yolo()

            cropping_active = self.crop_true.isChecked()
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

                if cropping_active:
                    original_image_copy = original_image.copy()
                    desired_height = self.crop_height.value()
                    desired_width = self.crop_width.value()
                    cropped_image = self.center_crop(original_image_copy, desired_height, desired_width)
                    if cropped_image is not None and cropped_image.shape[0] == desired_height and cropped_image.shape[1] == desired_width:
                        self.current_file = self.save_cropped_image(cropped_image, self.current_file)
                        self.display_image(cropped_image)
                    else:
                        self.display_image(original_image)
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

        return True



    def load_class_labels(self) -> List[str]:
        """
        Load class labels from a text file named 'classes.txt' located in the specified image directory.

        :return: A list of class labels, or an empty list if the image directory is not set,
                or 'classes.txt' file is not found in the image directory.
        """
        # Check if the image directory attribute is set
        if not hasattr(self, 'image_directory') or self.image_directory is None:
            logging.error("Image directory not selected.")
            return []

        # Construct the path to the 'classes.txt' file
        path = os.path.join(self.image_directory, 'classes.txt')

        try:
            # Attempt to open and read the 'classes.txt' file
            with open(path, 'r') as classes_file:
                return [line.strip() for line in classes_file.readlines()]
        except FileNotFoundError:
            # Log an error and return an empty list if the file is not found
            logging.error(f"Could not find the file at {path}")
            return []


    def infer_and_display_bounding_boxes(self, image) -> List[List[int]]:
        # No longer clearing bounding boxes; they will be replaced directly.
        
        if not hasattr(self, 'net'):
            print("Network not initialized. Please select weights file and cfg file.")
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
            print(f"Error: The directory {new_directory} does not exist.")
            return

        self.image_directory = new_directory
        print(f"Image directory set to {self.image_directory}.")
        # Trigger the stats calculation and display immediately
        stats = self.process_labels_and_generate_stats()
        self.settings['stats'] = stats
        self.display_stats()  # Immediately display updated stats

    # Creating plots without calling stats display
    def create_plot(self, plot_type='scatter'):
        """Unified function to create specific plots without refreshing stats."""
        if not hasattr(self, 'image_directory') or not self.image_directory:
            print("No image directory selected.")
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
        
        # Show a message box warning about performance impact
        if checked:
            QMessageBox.warning(self, "Performance Warning",
                                "Enabling image quality analysis may slow down computations significantly.")
        # Update stats to show "N/A" for quality metrics if disabled
        self.update_stats_with_na()

    def update_stats_with_na(self):
        """Updates the stats table to show 'N/A' for quality metrics if analysis is disabled."""
        stats = self.process_labels_and_generate_stats()
        if not self.image_quality_analysis_enabled:
            stats['Blurred Images'] = "N/A"
            stats['Underexposed Images'] = "N/A"
            stats['Overexposed Images'] = "N/A"
            stats['Low Contrast Images'] = "N/A"
        self.settings['stats'] = stats
        self.display_stats()
    # Process labels function with directory_path passed correctly
    def process_labels_and_generate_stats(self):
        """Load labels, calculate statistics, and prepare data for plotting."""
        directory_path = self.image_directory
        if not directory_path:
            print("Image directory is not set.")
            return {}

        # Initialize data lists for plotting
        self.all_center_points_and_areas = []  # List of ((x_center, y_center), area) tuples for scatter plot
        self.all_label_classes = []  # List of class IDs for bar plot

        # Load classes from classes.txt and map them to IDs
        class_names = self.load_classes()
        class_id_to_name = {i: name for i, name in enumerate(class_names)}

        # Initialize counts
        label_counts = defaultdict(int)
        pos_counts = defaultdict(int)
        size_counts = defaultdict(int)

        # Initialize quality metric counters
        blurred_images = 0
        underexposed_images = 0
        overexposed_images = 0
        low_contrast_images = 0

        labeled_images = 0
        total_labels = 0
        txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt') and f != 'classes.txt']

        for txt_file in txt_files:
            txt_file_path = os.path.join(directory_path, txt_file)

            with open(txt_file_path, 'r') as file:
                annotations = [line.strip() for line in file if line.strip()]
                if annotations:
                    labeled_images += 1
                    for line in annotations:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            center_x, center_y, width, height = map(float, parts[1:])
                            bbox_area = width * height
                            
                            # Add data to lists for plotting
                            self.all_center_points_and_areas.append(((center_x, center_y), bbox_area))
                            self.all_label_classes.append(class_id)
                            
                            # Update label counts
                            class_name = class_id_to_name.get(class_id, f"Class {class_id}")
                            label_counts[class_name] += 1
                            total_labels += 1

                            # Positional Bias
                            if center_x < 0.33: pos_counts['left'] += 1
                            elif center_x > 0.66: pos_counts['right'] += 1
                            else: pos_counts['center'] += 1

                            if center_y < 0.33: pos_counts['top'] += 1
                            elif center_y > 0.66: pos_counts['bottom'] += 1
                            else: pos_counts['middle'] += 1

                            # Size Bias
                            if bbox_area < 0.1: size_counts['small'] += 1
                            elif bbox_area < 0.3: size_counts['medium'] += 1
                            else: size_counts['large'] += 1

                # Quality metrics for each image (if enabled)
                if self.image_quality_analysis_enabled:
                    image_path = os.path.join(directory_path, txt_file.replace('.txt', '.jpg'))
                    if os.path.exists(image_path):
                        blur, brightness, contrast = self.analyze_image_quality(image_path)

                        # Count images that meet threshold criteria
                        if blur < 100:
                            blurred_images += 1
                        if brightness < 100:
                            underexposed_images += 1
                        elif brightness > 200:
                            overexposed_images += 1
                        if contrast < 50:
                            low_contrast_images += 1

        # Normalize Positional Counts
        normalized_pos_counts = {
            key: (count / total_labels) * 100 if total_labels else 0
            for key, count in pos_counts.items()
        }

        # Calculate Bias Score
        bias_score = self.calculate_bias_score(label_counts, pos_counts, size_counts)

        # Calculate Labeling Progress (%)
        labeling_progress = round((labeled_images / len(txt_files)) * 100, 1) if txt_files else 0

        # Calculate Class Balance Difference
        most_populated_class_count = max(label_counts.values()) if label_counts else 0
        class_balance_diff = {
            class_name: most_populated_class_count - count
            for class_name, count in label_counts.items()
        }

        # Dataset statistics
        stats = {
            'Total Images': len(txt_files),
            'Labeled Images': labeled_images,
            'Unlabeled Images': len(txt_files) - labeled_images,
            'Total Labels': total_labels,
            'Labels per Image (average)': round(total_labels / labeled_images, 1) if labeled_images else 0,
            'Labeling Progress (%)': labeling_progress,
            'Blurred Images': blurred_images if self.image_quality_analysis_enabled else "N/A",
            'Underexposed Images': underexposed_images if self.image_quality_analysis_enabled else "N/A",
            'Overexposed Images': overexposed_images if self.image_quality_analysis_enabled else "N/A",
            'Low Contrast Images': low_contrast_images if self.image_quality_analysis_enabled else "N/A",
            'Bias Score (%)': bias_score,
            'Class Counts': dict(label_counts),  # Convert defaultdict to dict for display
            'Class Balance Difference': class_balance_diff,  # Add balance difference
            'Normalized Positional Bias (%)': normalized_pos_counts,
            'Size Bias': dict(size_counts),
        }

        return stats





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
        bias_scores_tab = self.create_bias_scores_tab(stats)

        tab_widget.addTab(general_stats_tab, "General Stats")
        print("Added General Stats tab.")  # Debugging
        tab_widget.addTab(class_counts_tab, "Class Counts")
        print("Added Class Counts tab.")  # Debugging
        tab_widget.addTab(bias_scores_tab, "Bias Scores")
        print("Added Bias Scores tab.")  # Debugging

        # Ensure Tab Widget is added to Layout
        layout = QVBoxLayout()
        layout.addWidget(tab_widget)
        self.stats_widget.setLayout(layout)
        self.stats_widget.show()

    def create_general_stats_tab(self, stats):
        widget = QWidget()
        layout = QVBoxLayout()

        # General Stats Keys
        general_stats_keys = [
            'Total Images', 'Labeled Images', 'Unlabeled Images',
            'Total Labels', 'Labels per Image (average)', 'Labeling Progress (%)',
            'Blurred Images', 'Underexposed Images', 'Overexposed Images', 'Low Contrast Images',
            'Bias Score (%)'
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

        print("Class Counts tab created successfully.")  # Debugging
        return widget


    def create_bias_scores_tab(self, stats):
        widget = QWidget()
        layout = QVBoxLayout()

        # Positional Bias
        positional_bias = stats.get('Normalized Positional Bias (%)', {})
        size_bias = stats.get('Size Bias', {})

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Bias Type", "Value (%)"])

        for key, value in positional_bias.items():
            model.appendRow([QStandardItem(f"Positional ({key})"), QStandardItem(f"{value:.2f}%")])

        for key, value in size_bias.items():
            model.appendRow([QStandardItem(f"Size ({key})"), QStandardItem(str(value))])

        table = QTableView()
        table.setModel(model)
        self.style_table(table)

        layout.addWidget(table)
        widget.setLayout(layout)
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
            print("No data available for histogram plot.")
            return

        # Extract the area data from self.all_center_points_and_areas
        areas = [area for _, area in self.all_center_points_and_areas]

        fig, ax = plt.subplots()
        ax.hist(areas, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Label Area')
        plt.ylabel('Count')
        plt.title("Distribution of Label Areas")
        plt.show()


    def plot_bar(self):
        """Generate a bar plot showing class distribution."""
        from collections import Counter
        class_counts = Counter(self.all_label_classes)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        fig, ax = plt.subplots()
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
        ax.bar(classes, counts, color=colors)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title("Label Class Distribution")
        plt.show()

    def plot_data(self):
        """Plot data using scatter plot."""
        if not self.all_center_points_and_areas:
            print("No data available for scatter plot.")
            return

        x, y = zip(*[point[0] for point in self.all_center_points_and_areas])
        areas = [point[1] for point in self.all_center_points_and_areas]
        sizes = (np.array(areas) - np.min(areas)) / (np.max(areas) - np.min(areas)) * 50 + 10  # Normalize area sizes

        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, c=areas, cmap='viridis', alpha=0.7, s=sizes)
        plt.colorbar(scatter, label='Label Area', orientation='vertical')
        plt.title(f"Label Count: {len(self.all_center_points_and_areas)}")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()



    def load_classes(self):
        """Load class names from classes.txt."""
        if self.image_directory is None:
            print("Error: Image directory is not set. Please select an image directory.")
            return []
        
        classes_file_path = os.path.join(self.image_directory, 'classes.txt')
        try:
            with open(classes_file_path, 'r') as file:
                return file.read().splitlines()
        except FileNotFoundError:
            print(f"Error: classes.txt file not found in {self.image_directory}.")
            return []

    def analyze_image_quality(self, image_path):
        """Analyze blur, brightness, and contrast of an image using CPU."""
        # Load image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image at {image_path}")
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
                    print(f"Exception occurred for {path}: {e}")

        return results



    def calculate_bias_score(self, label_counts, pos_counts, size_counts):
        total_labels = sum(label_counts.values())
        if total_labels == 0:
            return 0

        # Class imbalance bias (Gini coefficient)
        gini_coefficient = self.calculate_gini_coefficient(label_counts)

        # Positional bias (example: percentage of center boxes vs. others)
        center_percentage = pos_counts.get('center', 0) / total_labels if total_labels else 0
        positional_bias = 1 - center_percentage  # Higher means more spread out, closer to 1 is less centered

        # Size bias (distribution of small, medium, large)
        small_percentage = size_counts.get('small', 0) / total_labels if total_labels else 0
        large_percentage = size_counts.get('large', 0) / total_labels if total_labels else 0
        size_bias = 1 - abs(small_percentage - large_percentage)  # Closer to 0 means equal distribution

        combined_bias_score = (gini_coefficient + positional_bias + size_bias) / 3
        return round(combined_bias_score * 100, 2)

    def calculate_gini_coefficient(self, label_counts):
        sorted_counts = sorted(label_counts.values())
        n = len(sorted_counts)
        cumulative_total = sum(sorted_counts)
        
        if cumulative_total == 0:
            return 0
        
        cumulative_sum = 0
        weighted_sum = 0
        
        for i, count in enumerate(sorted_counts, 1):
            cumulative_sum += count
            weighted_sum += cumulative_sum
        
        gini_coefficient = (2 * weighted_sum) / (n * cumulative_total) - (n + 1) / n
        return gini_coefficient




    def initialize_filter_spinbox(self):
        """
        Initialize the filter spinbox for class filtering.
        Includes options for 'All' (-1) and 'Blanks' (-2).
        Only enables spinbox after loading images.
        """
        self.filter_class_spinbox.setMinimum(-2)  # -2 for Blanks, -1 for All
        self.filter_class_spinbox.setMaximum(-1)  # Initially, no classes are loaded
        self.filter_class_spinbox.setEnabled(False)  # Disabled until images are loaded
        self.filter_class_spinbox.valueChanged.connect(self.filter_class)


    def update_filter_spinbox(self):
        """
        Populate the QComboBox with class names, indices, and special options.
        """
        self.filter_class_spinbox.blockSignals(True)  # Prevent triggering events during update
        self.filter_class_spinbox.clear()  # Clear existing entries

        # Add special options
        self.filter_class_spinbox.addItem("All (-1)")
        self.filter_class_spinbox.addItem("Blanks (-2)")

        # Add class indices and names
        for idx, class_name in enumerate(self.class_names):
            self.filter_class_spinbox.addItem(f"{idx}: {class_name}")

        self.filter_class_spinbox.blockSignals(False)  # Re-enable signals
        
    def on_filter_class_spinbox_changed(self, index):
        """
        Handle changes in the class filter ComboBox.
        """
        current_text = self.filter_class_spinbox.currentText()
        if current_text.startswith("All"):
            self.filter_class(-1)  # Special case: All
        elif current_text.startswith("Blanks"):
            self.filter_class(-2)  # Special case: Blanks
        else:
            class_index = int(current_text.split(":")[0])  # Extract the class index
            self.filter_class(class_index)


    def filter_class(self, filter_index):
        """
        Filter images based on the selected class index or blanks.
        """
        self.filtered_image_files = []
        placeholder_file = 'styles/images/default.png'

        if filter_index == -1:  # Show all
            self.filtered_image_files = self.image_files[:]
        elif filter_index == -2:  # Filter blanks
            for img_file in self.image_files:
                if img_file == placeholder_file:
                    continue
                label_file = os.path.splitext(img_file)[0] + '.txt'
                if not os.path.exists(label_file) or not os.path.getsize(label_file):
                    self.filtered_image_files.append(img_file)
        else:  # Filter by specific class
            for img_file in self.image_files:
                if img_file == placeholder_file:
                    continue
                label_file = os.path.splitext(img_file)[0] + '.txt'
                if os.path.exists(label_file):
                    with open(label_file, 'r') as file:
                        for line in file:
                            if int(line.split()[0]) == filter_index:
                                self.filtered_image_files.append(img_file)
                                break

        # Ensure the placeholder file is always first
        if placeholder_file not in self.filtered_image_files:
            self.filtered_image_files.insert(0, placeholder_file)

        # Update the list view and display the first image
        self.update_list_view(self.filtered_image_files)
        if self.filtered_image_files:
            self.current_file = self.filtered_image_files[0]
            self.display_image(self.current_file)


    def clear_class_boxes(self):
        """
        Clears bounding boxes for the selected class.
        """
        placeholder_file = 'styles/images/default.png'
        current_text = self.filter_class_spinbox.currentText()
        if current_text.startswith("All"):
            class_index = -1
        elif current_text.startswith("Blanks"):
            class_index = -2
        else:
            class_index = int(current_text.split(":")[0])  # Extract class index

        for img_file in self.filtered_image_files:
            if img_file == placeholder_file:  # Skip placeholder
                continue
            label_file = os.path.splitext(img_file)[0] + '.txt'
            if os.path.exists(label_file):
                with open(label_file, 'r+') as file:
                    lines = file.readlines()
                    file.seek(0)
                    file.truncate()
                    for line in lines:
                        if class_index == -2 or int(line.split()[0]) != class_index:
                            file.write(line)

        print(f"Cleared bounding boxes for class index: {class_index}")
        self.display_all_images()




    def move_filtered_images(self):
        """
        Moves filtered images based on the ComboBox-selected class.
        """
        placeholder_file = 'styles/images/default.png'
        current_text = self.filter_class_spinbox.currentText()
        if current_text.startswith("All"):
            class_index = -1
        elif current_text.startswith("Blanks"):
            class_index = -2
        else:
            class_index = int(current_text.split(":")[0])  # Extract class index

        class_name = "blanks" if class_index == -2 else (
            "all" if class_index == -1 else self.class_names[class_index]
        )

        class_folder = os.path.join(self.image_directory, class_name)
        os.makedirs(class_folder, exist_ok=True)

        for file_path in self.filtered_image_files:
            if file_path == placeholder_file:  # Skip placeholder
                continue
            self.move_file(file_path, class_folder)
            txt_file = os.path.splitext(file_path)[0] + '.txt'
            if os.path.exists(txt_file):
                self.move_file(txt_file, class_folder)

        print(f"Moved images for class index {class_index} to {class_folder}.")


    def move_file(self, file_path, destination_folder):
        """
        Moves a file to the specified destination folder.
        """
        try:
            # Ensure the destination folder exists
            os.makedirs(destination_folder, exist_ok=True)

            # Construct the destination path
            destination_path = os.path.join(destination_folder, os.path.basename(file_path))

            # Move the file
            shutil.move(file_path, destination_path)
            print(f"Moved {file_path} to {destination_path}")

        except Exception as e:
            print(f"Error moving {file_path} to {destination_folder}: {e}")


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
        """
        Update the ListView with image file names, even if thumbnails are missing.
        """
        print(f"Updating ListView with {len(image_files)} files.")

        model = QStandardItemModel()  # Create a new model for the ListView
        for image_file in image_files:
            base_file = os.path.basename(image_file)
            item = QStandardItem(base_file)  # Add the base file name
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make items uneditable
            
            thumbnail_file = os.path.join(self.thumbnails_directory, f"{base_file}.jpeg")
            if os.path.exists(thumbnail_file):
                # Optionally add custom styling if a thumbnail exists
                item.setBackground(Qt.green)
            else:
                # Highlight missing thumbnails (optional)
                item.setBackground(Qt.red)
                print(f"Thumbnail missing for {base_file}.")
            
            model.appendRow(item)

        self.List_view.setModel(model)  # Set the model for the ListView
        print("ListView updated successfully.")



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


    def update_list_view(self, image_files):
        model = QStandardItemModel()  # Create a model for the List_view
        for img_file in image_files:
            item = QStandardItem(os.path.basename(img_file))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make items uneditable
            model.appendRow(item)
        self.List_view.setModel(model)  # Set the model for the List_view
        
    def img_index_number_changed(self, value):
        """
        Handles both index-based (if input is numeric) and filename-based image selection.
        If the value is numeric, it treats it as an index (from QSpinBox).
        If the value is a string (from QLineEdit), it treats it as a filename.
        """

        # Check if the value comes from QSpinBox (int) or QLineEdit (str)
        if isinstance(value, int):  # If value is from QSpinBox (index-based)
            if 0 <= value < len(self.filtered_image_files):
                self.current_img_index = value  # Update the current image index
                self.current_file = self.filtered_image_files[self.current_img_index]
                self.display_image(self.current_file)
            else:
                print("Index out of range.")
        
        elif isinstance(value, str):  # If value is from QLineEdit (filename-based)
            value = value.strip()  # Remove any leading/trailing spaces
            matching_files = [f for f in self.filtered_image_files if value in f]
            if matching_files:
                # If a matching file is found, display the first match
                self.current_file = matching_files[0]
                self.current_img_index = self.filtered_image_files.index(self.current_file)
                self.display_image(self.current_file)
            else:
                print(f"No image found with filename containing: {value}")
                
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
            extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.webp'}
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in extensions
        ]


    def preprocess_placeholder_image(self,image_path):
        try:
            with Image.open(image_path) as img:
                img.save(image_path, icc_profile=None)
            print(f"Placeholder image preprocessed: {image_path}")
        except Exception as e:
            print(f"Failed to preprocess placeholder image: {e}")



    def open_image_video(self):
        """
        Open an image directory, load images, initialize classes, and populate views.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        # Initialize the progress bar to zero
        self.label_progress.setValue(0)

        # Read the last directory from the settings
        last_dir = self.settings.get('last_dir', "")

        dir_name = QFileDialog.getExistingDirectory(None, "Open Image Directory", last_dir, options=options)
        placeholder_image_path = 'styles/images/default.png'  # Adjust path as needed

        # Preprocess the placeholder image before anything else to prevent ICC profile issues
        if os.path.exists(placeholder_image_path):
            self.preprocess_placeholder_image(placeholder_image_path)

        if dir_name:
            # Save the selected directory to the settings
            self.settings['last_dir'] = dir_name
            self.saveSettings()  # Save the settings after modifying it
            # Update spinbox and dropdown after loading class names
 
            self.image_directory = dir_name
            print(f"Image Directory: {self.image_directory}")

            # Import classes.txt
            classes_file_path = os.path.join(self.image_directory, 'classes.txt')
            if os.path.exists(classes_file_path):
                with open(classes_file_path, 'r') as f:
                    class_names = [line.strip() for line in f.readlines()]
            else:
                # If classes.txt does not exist, create one with 'person' as the only class
                with open(classes_file_path, 'w') as f:
                    f.write('person\n')
                class_names = ['person']

            self.valid_classes = list(range(len(class_names)))
            self.class_names = class_names

            # Initialize class hotkeys for the current dataset
            for className in class_names:
                class_hotkey_setting = f'classHotkey_{className}'
                if class_hotkey_setting not in self.settings:
                    self.settings[class_hotkey_setting] = ''
                else:
                    # Remove existing hotkey assignment if it conflicts
                    current_hotkey = self.settings[class_hotkey_setting]
                    for other_class in self.settings:
                        if other_class.startswith('classHotkey_') and other_class != class_hotkey_setting:
                            if self.settings[other_class] == current_hotkey:
                                self.settings[other_class] = ''

            self.saveSettings()  # Save the settings after modifying the hotkeys

            # Update dropdown and spinbox for class filtering
            self.update_classes_dropdown(class_names)
            self.update_filter_spinbox()

            # Get all the image files in the directory
            self.image_files = self.sorted_nicely(self.get_image_files(dir_name))

            # Insert the placeholder image at the beginning of the list
            if placeholder_image_path not in self.image_files:
                self.image_files.insert(0, placeholder_image_path)

            total_images = len(self.image_files)
            if total_images > 0:
                # Try to get the last opened image from the settings
                last_image = self.settings.get('lastImage', "")
                if last_image and last_image in self.image_files and last_image != placeholder_image_path:
                    self.current_image_index = self.image_files.index(last_image)
                else:
                    self.current_image_index = 0

                self.current_file = self.image_files[self.current_image_index]
                self.total_images.setText(f"Total Images: {total_images}")
                self.display_image(self.current_file)
                self.initialize_yolo()

                # Populate list view with image files
                self.display_all_images()

                # Create empty .txt and .json files
                self.create_empty_txt_and_json_files(dir_name, placeholder_image_path)

                # Directly calculate stats and display without threading
                stats = self.process_labels_and_generate_stats()
                self.settings['stats'] = stats

                # Set progress bar to 100% once loading is complete
                self.label_progress.setValue(100)
            else:
                self.total_images.setText("Total Images: 0")
                QMessageBox.warning(None, 'No Images Found', "No image files found in the directory.")
                self.label_progress.setValue(0)  # Reset progress bar if no images found

    def create_empty_txt_and_json_files(self, image_directory, placeholder_image_path):
        image_files = self.get_image_files(image_directory)

        for image_file in image_files:
            # Skip creating .txt for the placeholder image
            if os.path.abspath(image_file) == os.path.abspath(placeholder_image_path):
                continue
            
            txt_file = os.path.splitext(image_file)[0] + '.txt'
            if not os.path.exists(txt_file):
                with open(txt_file, 'w') as f:
                    pass                    
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
            print("Cannot delete the placeholder image.")
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
                print("The file is being handled as non-filtered and there's no specific action here.")
            else:
                print("Warning: No image currently loaded.")

    def delete_files(self, file_path):
        # Ensure the protected image is never deleted
        protected_file = 'styles/images/default.png'
        if file_path == protected_file:
            print("Cannot delete the protected image.")
            return

        try:
            os.remove(file_path)  # Delete the image file.
        except FileNotFoundError:
            print(f"Warning: Image file {file_path} not found.")

        # Assume the label file has the same base name with a .txt extension.
        txt_file_path = os.path.splitext(file_path)[0] + '.txt'
        try:
            os.remove(txt_file_path)  # Delete the associated annotation file.
        except FileNotFoundError:
            print(f"Warning: Label file {txt_file_path} not found.")


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



    def display_image(self, file_name=None, image=None):
        """
        This function handles displaying an image either from a file or a numpy array.
        """
        if file_name is None and image is None:
            print("File name and image are both None.")
            return None

        if file_name is not None:
            if not isinstance(file_name, str):
                print("Invalid file name type.")
                return None

            # Check if the file_name is the same as the last logged file name
            if file_name != self.last_logged_file_name:
                logging.info(f"display_image: file_name={file_name}")
                self.last_logged_file_name = file_name

            # Load the image into a QPixmap
            self.image = QPixmap(file_name)

            if self.image.isNull():
                print(f"Failed to load image: {file_name}. Falling back to placeholder.")
                placeholder_path = 'styles/images/default.png'
                if os.path.exists(placeholder_path):
                    self.image = QPixmap(placeholder_path)
                    if self.image.isNull():
                        print("Failed to load placeholder image.")
                        return None
                else:
                    print("Placeholder image missing.")
                    return None

            self.original_pixmap_size = self.image.size()  # Save the original pixmap size

        if image is not None:
            if not isinstance(image, np.ndarray):
                print("Image is not a NumPy array.")
                return None

            # Convert the NumPy array to a QImage and then to a QPixmap
            image_qimage = self.cv2_to_qimage(image)
            self.image = QPixmap.fromImage(image_qimage)

        # Apply optional transformations (e.g., grayscale, edge detection)
        self.apply_super_resolution(file_name)
        self.apply_grayscale()
        self.apply_edge_detection()
        self.draw_crop_area()

        # Create a QGraphicsPixmapItem with the QPixmap
        pixmap_item = QGraphicsPixmapItem(self.image)
        pixmap_item.setTransformationMode(Qt.SmoothTransformation)

        # Create a QGraphicsScene with the exact dimensions of the image
        scene = QGraphicsScene(0, 0, self.image.width(), self.image.height())

        # Add the QGraphicsPixmapItem to the scene
        scene.addItem(pixmap_item)

        # Set up the scene and rect for screen_view
        self.set_screen_view_scene_and_rect(scene)

        # Handle bounding boxes display and text
        label_file = self.replace_extension_with_txt(file_name)
        self.create_empty_file_if_not_exists(label_file)
        self.label_file = label_file
        rects = self.load_bounding_boxes(label_file, self.image.width(), self.image.height())
        self.display_bounding_boxes(rects, file_name)

        if not rects:
            self.display_image_with_text(scene, self.image)

        # Synchronize list view selection with the displayed image
        self.sync_list_view_selection(file_name)

        return QPixmap.toImage(self.image)
    def set_screen_view_scene_and_rect(self, scene):
        """
        Set the scene and rectangle for the screen_view.
        """
        self.view_references.append(self.screen_view)
        self.screen_view.setScene(scene)
        self.screen_view.fitInView(QRectF(0, 0, self.image.width(), self.image.height()), Qt.KeepAspectRatio)
        self.screen_view.setSceneRect(QRectF(0, 0, self.image.width(), self.image.height()))

    def draw_crop_area(self):
        """
        Draw crop area if the crop checkbox is checked.
        """
        if self.crop_true.isChecked():
            painter = QPainter(self.image)
            h, w = self.image.height(), self.image.width()
            new_h, new_w = self.crop_height.value(), self.crop_width.value()
            start_x = w // 2 - new_w // 2
            start_y = h // 2 - new_h // 2

            pen = QPen(Qt.red)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(start_x, start_y, new_w, new_h)
            painter.end()

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
                print(f"Error: Model file not found at {path_to_model}.")
                return

            try:
                # Load the model
                sr.readModel(path_to_model)
                sr.setModel("fsrcnn", 4)  # Use "espcn" or "lapsrn" for other models

                # Read the image
                img = cv2.imread(file_name, cv2.IMREAD_COLOR)
                if img is None:
                    print("Error: Image file could not be read.")
                    return

                # Apply super resolution
                img_sr = sr.upsample(img)

                # Convert to QPixmap for display
                height, width, channel = img_sr.shape
                bytesPerLine = 3 * width
                qImg = QImage(img_sr.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                self.image = QPixmap.fromImage(qImg)

            except Exception as e:
                print(f"Failed to apply super resolution. Error: {e}")



    def apply_grayscale(self):
        """
        Convert the image to grayscale if the corresponding checkbox is checked.
        """
        if self.grayscale_Checkbox.isChecked():
            image_qimage = self.image.toImage()
            image_cv = self.qimage_to_cv2(image_qimage)

            if image_cv is None:
                print("Error: Failed to convert QImage to OpenCV format.")
                return

            # Convert the image to grayscale
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

            # Adjust brightness and contrast based on the value of the slider
            alpha = self.grey_scale_slider.value() / 50.0
            beta = self.grey_scale_slider.value()
            image_cv = self.adjust_brightness_contrast(image_cv, alpha, beta)

            image_qimage = self.cv2_to_qimage(image_cv)

            if image_qimage is None:
                print("Error: Failed to convert OpenCV image back to QImage.")
                return

            # Set the grayscale image as the QPixmap
            self.image = QPixmap.fromImage(image_qimage)


    def apply_edge_detection(self):
        """
        Apply edge detection to the image if the corresponding checkbox is checked.
        """
        if self.outline_Checkbox.isChecked():
            # Convert the current image to QImage
            image_qimage = self.image.toImage()

            # Convert QImage to OpenCV format (NumPy array)
            image_cv = self.qimage_to_cv2(image_qimage)

            # Check if the image is a UMat and convert it to a NumPy array if necessary
            if isinstance(image_cv, cv2.UMat):
                image_cv = image_cv.get()  # Convert UMat to NumPy array

            # Apply Canny edge detection
            edges = cv2.Canny(image_cv, self.slider_min_value, self.slider_max_value)

            # Convert edges to a 4-channel image (BGRA) to match the original image format
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGRA)

            # To overlay the edges on the original image, ensure dimensions match
            if edges_colored.shape[:2] != image_cv.shape[:2]:
                height, width = image_cv.shape[:2]
                edges_colored = cv2.resize(edges_colored, (width, height))

            # Use addWeighted to overlay edge detection result onto the original image
            image_cv = cv2.addWeighted(image_cv, 1, edges_colored, 0.5, 0)

            # Convert the final image back from BGRA to RGBA
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2RGBA)

            # Convert the final OpenCV image (NumPy array) back to QImage
            image_qimage = self.cv2_to_qimage(image_cv)

            # Update the QPixmap with the new image
            if image_qimage is not None:
                self.image = QPixmap.fromImage(image_qimage)
            else:
                print("Error: Failed to convert OpenCV image back to QImage.")









    def display_image_checkbox_toggled(self):
        # path of the currently displayed image.
        self.display_image(self.current_file)

    def checkbox_clicked(self):
        self.display_image(self.current_file)

    def slider_value_changed2(self, value):
        # Set the min and max values based on the slider value
        self.slider_min_value = value * 2
        self.slider_max_value = value * 3
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
        Convert OpenCV image (NumPy array or UMat) to QImage.
        Handles both color and grayscale images.
        """
        if img is None:
            print("Error: Input image is None.")
            return None

        if isinstance(img, cv2.UMat):
            img = img.get()

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

        print(f"Error: Unsupported image format with {len(img.shape)} dimensions.")
        return None

    def adjust_brightness_contrast(self, image_cv, alpha, beta):
        return cv2.convertScaleAbs(image_cv, alpha=alpha, beta=beta)   

    def replace_extension_with_txt(self, file_name):
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
            file_name = file_name.replace(ext, ".txt")
        return file_name
    def draw_crop_area(self):
            """
            Draw crop area if the crop checkbox is checked.
            """
            if self.crop_true.isChecked():
                painter = QPainter(self.image)
                h, w = self.image.height(), self.image.width()
                new_h, new_w = self.crop_height.value(), self.crop_width.value()
                start_x = w // 2 - new_w // 2
                start_y = h // 2 - new_h // 2

                pen = QPen(Qt.red)
                pen.setWidth(3)
                painter.setPen(pen)
                painter.drawRect(start_x, start_y, new_w, new_h)
                painter.end()

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
            print(f"bbox_item {bbox_item} not found in rects")
        self.save_bounding_boxes(self.label_file, self.screen_view.scene(
        ).width(), self.screen_view.scene().height())

    def create_bounding_box(self, x, y, w, h, label, confidence):
        new_index = self.classes_dropdown.findText(label)
        if new_index != -1:
            self.classes_dropdown.setCurrentIndex(new_index)
            bbox_drawer = BoundingBoxDrawer(x, y, w, h, self, class_id=new_index, confidence=confidence)
            self.screen_view.scene().addItem(bbox_drawer)

            # Check if the heads_area checkbox is checked
            if self.heads_area.isChecked() and new_index != 1:
                head_class_id_str = self.class_id.text()
                if head_class_id_str.strip():  # Check if the string is not empty
                    try:
                        head_class_id = int(head_class_id_str)
                    except ValueError:
                        print(f"Warning: Invalid class ID '{head_class_id_str}' provided in QLineEdit. Using the next available class ID.")
                        head_class_id = self.get_next_available_class_id()
                else:
                    print("No class ID provided. Using the next available class ID.")
                    head_class_id = self.get_next_available_class_id()

                head_x, head_y, head_w, head_h = self.calculate_head_area(x, y, w, h)
                head_bbox_drawer = BoundingBoxDrawer(head_x, head_y, head_w, head_h, self, class_id=head_class_id, confidence=confidence)
                self.screen_view.scene().addItem(head_bbox_drawer)

            label_file = self.replace_extension_with_txt(self.current_file)
            self.save_bounding_boxes(label_file, self.screen_view.scene().width(), self.screen_view.scene().height())
        else:
            print(f"Warning: '{label}' not found in classes dropdown.")


    def get_next_available_class_id(self):
        # Read classes from the file
        classes_file_path = os.path.join(self.image_directory, 'classes.txt')
        if os.path.exists(classes_file_path):
            with open(classes_file_path, 'r') as classes_file:
                class_labels = [line.strip() for line in classes_file.readlines()]

            # Convert class labels to integers and find the maximum class ID
            class_ids = [int(label) for label in class_labels if label.isdigit()]
            if class_ids:
                return max(class_ids) + 1  # Next available class ID
        return 0  # Default to 0 if no classes found or file doesn't exist

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



    def load_bounding_boxes(self, label_file, img_width, img_height):
        # Load bounding boxes from a .txt file
        bounding_boxes = []
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        bbox = BoundingBox.from_str(line)
                        if bbox:
                            bounding_boxes.append(bbox)

        # Load confidence scores from a corresponding .json file
        label_file_json = label_file.replace('.txt', '.json')
        if os.path.exists(label_file_json) and os.path.getsize(label_file_json) > 0:
            try:
                with open(label_file_json, "r") as f:
                    data = json.load(f)
                annotations = data.get('annotations', []) if isinstance(data, dict) else data
            except json.JSONDecodeError:
                print(f"Error reading JSON file: {label_file_json}")
                annotations = []
        else:
            annotations = [{} for _ in bounding_boxes]  # Empty dict for each bbox if no JSON data

        # Add confidence to bounding boxes
        for bbox, annotation in zip(bounding_boxes, annotations):
            bbox.confidence = annotation.get('confidence', 0)

        # Convert bounding boxes to a specific format for return
        return [(bbox.to_rect(img_width, img_height), bbox.class_id, bbox.confidence or 0) for bbox in bounding_boxes]


    def display_bounding_boxes(self, rects, file_name):
        for index, rect_tuple in enumerate(rects):
            # Generate or fetch a unique_id for each bounding box. Here, I am using the index as a simple example.
            # Replace with your actual logic to generate or fetch a unique identifier.
            unique_id = f"{file_name}_{index}"

            # Check the length of the tuple and create a BoundingBoxDrawer accordingly
            if len(rect_tuple) == 3:  # Tuple has three elements
                rect, class_id, confidence = rect_tuple
                rect_item = BoundingBoxDrawer(
                    rect.x(), rect.y(), rect.width(), rect.height(),
                    unique_id=unique_id,
                    class_id=class_id,
                    main_window=self,
                    confidence=confidence
                )
            elif len(rect_tuple) == 2:  # Tuple has two elements
                rect, class_id = rect_tuple
                rect_item = BoundingBoxDrawer(
                    rect.x(), rect.y(), rect.width(), rect.height(),
                    unique_id=unique_id,
                    class_id=class_id,
                    main_window=self
                )
            else:
                continue  # Skip tuples with unexpected size

            # Set the file_name attribute for the BoundingBoxDrawer
            rect_item.file_name = file_name

            # Add the BoundingBoxDrawer to the bounding_boxes dictionary using a unique key
            unique_key = f"{file_name}_{unique_id}"
            self.bounding_boxes[unique_key] = rect_item

            # Add the BoundingBoxDrawer to the scene
            self.screen_view.scene().addItem(rect_item)



    def save_bounding_boxes(self, label_file, img_width, img_height):
        """
        Save bounding boxes for the current image to the corresponding .txt file.
        Args:
            label_file (str): Path to the label file.
            img_width (int): Width of the image.
            img_height (int): Height of the image.
        """
        rects = [item for item in self.screen_view.scene().items() if isinstance(item, BoundingBoxDrawer)]

        bounding_boxes = [
            BoundingBox.from_rect(
                QRectF(rect.rect().x(), rect.rect().y(), rect.rect().width(), rect.rect().height()),
                img_width,
                img_height,
                rect.class_id,
                rect.confidence
            ) for rect in rects
        ]

        # Remove near-duplicate bounding boxes based on IoU
        bounding_boxes = self.remove_near_duplicate_bounding_boxes(bounding_boxes)

        # Save to .txt file
        try:
            with open(label_file, "w") as f:
                for bbox in bounding_boxes:
                    bbox_no_confidence = copy.copy(bbox)
                    bbox_no_confidence.confidence = None  # Remove confidence
                    f.write(bbox_no_confidence.to_str() + "\n")
            # Removed the print statement here
        except FileNotFoundError as fnf_error:
            logging.error(f"File not found: {fnf_error}")



    def remove_near_duplicate_bounding_boxes(self, bounding_boxes, iou_threshold=0.5):
        """
        Removes near-duplicate bounding boxes based on Intersection over Union (IoU).
        Bounding boxes with IoU above the threshold are considered duplicates.
        
        :param bounding_boxes: List of bounding box objects
        :param iou_threshold: Threshold to consider two boxes as duplicates (default 0.5)
        :return: List of unique bounding boxes
        """
        unique_bounding_boxes = []

        for i, bbox_a in enumerate(bounding_boxes):
            is_duplicate = False
            for j, bbox_b in enumerate(unique_bounding_boxes):
                iou = self.calculate_iou(bbox_a, bbox_b)
                if iou > iou_threshold and bbox_a.class_id == bbox_b.class_id:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_bounding_boxes.append(bbox_a)

        return unique_bounding_boxes


    def calculate_iou(self, bbox_a, bbox_b):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.

        :param bbox_a: First bounding box (assumed to have center coordinates and dimensions)
        :param bbox_b: Second bounding box
        :return: IoU value (float)
        """
        # Convert center coordinates to (x_min, y_min, x_max, y_max)
        x_min_a = bbox_a.x_center - bbox_a.width / 2
        y_min_a = bbox_a.y_center - bbox_a.height / 2
        x_max_a = bbox_a.x_center + bbox_a.width / 2
        y_max_a = bbox_a.y_center + bbox_a.height / 2

        x_min_b = bbox_b.x_center - bbox_b.width / 2
        y_min_b = bbox_b.y_center - bbox_b.height / 2
        x_max_b = bbox_b.x_center + bbox_b.width / 2
        y_max_b = bbox_b.y_center + bbox_b.height / 2

        # Determine the (x, y)-coordinates of the intersection rectangle
        x_left = max(x_min_a, x_min_b)
        y_top = max(y_min_a, y_min_b)
        x_right = min(x_max_a, x_max_b)
        y_bottom = min(y_max_a, y_max_b)

        # Calculate the area of overlap (intersection area)
        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

        # Calculate the area of both bounding boxes
        bbox_a_area = (x_max_a - x_min_a) * (y_max_a - y_min_a)
        bbox_b_area = (x_max_b - x_min_b) * (y_max_b - y_min_b)

        # Calculate the area of union
        union_area = bbox_a_area + bbox_b_area - intersection_area

        # Compute the IoU
        iou = intersection_area / union_area if union_area > 0 else 0

        return iou




    def update_yolo_label_file(self, new_class_id):
        try:
            print(f"Attempting to update YOLO label file at {self.label_file}")  # Debugging line
            with open(self.label_file, "r") as f:
                yolo_labels = f.readlines()

            found = False  # To check if a matching bounding box is found
            for i, line in enumerate(yolo_labels):
                bbox = BoundingBox.from_str(line)
                if bbox.is_same(self.selected_bbox):
                    found = True
                    print(f"Matching bounding box found. Updating...")  # Debugging line
                    bbox.class_id = new_class_id
                    yolo_labels[i] = bbox.to_str()

            if not found:
                print("No matching bounding box found.")  # Debugging line

            with open(self.label_file, "w") as f:
                f.writelines(yolo_labels)

            print("Successfully updated YOLO label file.")  # Debugging line
        except Exception as e:
            print(f"An error occurred: {e}")  # Basic error handling

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

    def update_current_bbox_class(self):
        try:
            if self.selected_bbox is not None:
                # Get the index of the selected class from the dropdown
                new_class_id = self.classes_dropdown.currentIndex()

                # Update the class ID of the selected bounding box
                self.selected_bbox.set_class_id(new_class_id)

                # Update the YOLO label file with the new class ID
                self.update_yolo_label_file(new_class_id)

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
            # Basic error handling: Print the error message
            print(f"An error occurred: {e}")
            



    def toggle_label_visibility(self):
        self.hide_labels = self.hide_label_checkbox.isChecked()
        scene = self.screen_view.scene()
        if scene is not None:
            for item in scene.items():
                if isinstance(item, BoundingBoxDrawer):
                    for child in item.childItems():
                        if isinstance(child, QGraphicsTextItem):
                            # Set visibility based only on the checkbox
                            should_display = not self.hide_labels
                            child.setVisible(should_display)
                            item.labels_hidden = not should_display  # existing line
                            item.update()  # existing line



    def auto_save_bounding_boxes(self):
        if self.label_file:
            self.save_bounding_boxes(self.label_file, self.screen_view.scene().width(), self.screen_view.scene().height())


    def set_selected(self, selected_bbox):
        if selected_bbox is not None:
            self.selected_bbox = selected_bbox
            # Update the class_input_field_label text



    def change_class_id(self, index):
        if self.screen_view.selected_bbox is not None:
            self.screen_view.selected_bbox.set_class_id(index)
            # Update the class ID in the YOLO label
            self.update_yolo_label_file(
                self.screen_view.selected_bbox.class_name)
            self.update_bbox_data()
            self.current_class_id = index

    def get_current_class_id(self):
        return self.classes_dropdown.currentIndex()

    @pyqtSlot()
    def clear_classes_dropdown(self):
        self.classes_dropdown.clear()

    @pyqtSlot(str)
    def add_item_to_classes_dropdown(self, item_text):
        self.classes_dropdown.addItem(item_text)

    def update_classes_dropdown(self, classes):
        """
        Update the classes dropdown with a list of classes.
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

        # Connect changes to toggle_class_visibility
        model.itemChanged.connect(self.handle_class_visibility_change)

        print(f"Classes dropdown updated: {classes}")

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

        print(f"Visibility for class '{class_name}' set to {self.class_visibility[class_name]}")


    def update_bbox_visibility(self):
        """
        Updates visibility of bounding boxes in the current frame
        based on the stored class visibility settings.
        """
        for item in self.screen_view.scene().items():
            if isinstance(item, BoundingBoxDrawer):
                class_name = self.classes_dropdown.itemText(item.class_id)
                item.setVisible(self.class_visibility.get(class_name, True))



    def class_input_field_return_pressed(self):
        if self.image_directory is None:
            print("Error: Image directory is not set. Please select an image directory.")
            return

        new_class = self.class_input_field.text()
        if new_class == "":
            return

        classes_file_path = os.path.join(self.image_directory, "classes.txt")

        try:
            with open(classes_file_path, "r") as f:
                existing_classes = [line.strip() for line in f.readlines()]

            # If the new class already exists, set it as the current class in the dropdown
            if new_class in existing_classes:
                index = self.classes_dropdown.findText(new_class)
                if index >= 0:  # if new_class is found in the dropdown
                    self.classes_dropdown.setCurrentIndex(index)
                    print("The class '{}' is selected.".format(new_class))
                return
        except FileNotFoundError:
            print(f"Error: 'classes.txt' not found in the directory {self.image_directory}.")
            return

        # Add the new class to the classes.txt file
        with open(classes_file_path, "a") as f:
            f.write(f"{new_class}\n")

        # Clear the class input field
        self.class_input_field.clear()

        # Update the classes dropdown menu
        self.update_classes_dropdown(existing_classes + [new_class])

        # Set the selected class in the classes dropdown menu
        self.classes_dropdown.setCurrentIndex(self.classes_dropdown.count() - 1)




    def remove_class_button_clicked(self):
        selected_class = self.classes_dropdown.currentText()

        if selected_class:
            classes_file_path = os.path.join(self.image_directory, "classes.txt")

            try:
                with open(classes_file_path, "r") as f:
                    classes = [line.strip() for line in f.readlines()]

                if selected_class in classes:
                    classes.remove(selected_class)

                    with open(classes_file_path, "w") as f:
                        for cls in classes:
                            f.write(f"{cls}\n")

                    # Update the classes dropdown
                    self.update_classes_dropdown(classes)

                    QMessageBox.information(
                        self, "Information", f"Class '{selected_class}' has been removed and the list has been updated."
                    )
                else:
                    QMessageBox.warning(
                        self, "Warning", f"Class '{selected_class}' does not exist."
                    )
            except FileNotFoundError:
                QMessageBox.warning(
                    self, "Warning", "Classes file not found. Please check your directory."
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"An error occurred while removing the class: {e}"
                )
        else:
            QMessageBox.warning(self, "Warning", "No class selected.")




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


    def load_labels(self, label_file):
        """
        Loads bounding boxes for the current frame and stores them
        in the `all_frame_bounding_boxes` dictionary.
        """
        # Load bounding boxes from the label file
        bounding_boxes = self.load_bounding_boxes(label_file, self.screen_view.scene().width(), self.screen_view.scene().height())

        # If no bounding boxes are loaded, create an empty entry
        if not bounding_boxes:
            bounding_boxes = []

        # Store bounding boxes for the current frame
        self.all_frame_bounding_boxes[self.current_file] = bounding_boxes

        # Update the labels attribute for the current frame
        self.labels = []
        for rect, class_id, confidence in bounding_boxes:
            x_center = rect.x() + rect.width() / 2
            y_center = rect.y() + rect.height() / 2
            width = rect.width()
            height = rect.height()
            self.labels.append((class_id, x_center, y_center, width, height, confidence))



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
            
            # Disable auto_scan when reaching the end or the beginning of the list
            if self.auto_scan_checkbox.isChecked():
                self.auto_scan_checkbox.setChecked(False)
                self.stop_next_timer()
                self.stop_prev_timer()

            return

        # Save the bounding boxes and settings
        self.save_bounding_boxes(self.label_file, self.screen_view.scene().width(), self.screen_view.scene().height())
        self.settings['lastImage'] = new_file
        self.saveSettings()

        # Display the new image and update UI elements
        self.display_image(new_file)
        self.img_index_number.setValue(index + (1 if direction == 'next' else -1))

        # Load labels and reset the label index
        self.load_labels(self.label_file)
        self.current_label_index = 0

        # Optionally zoom into the first label
        if self.zoom_mode:
            self.zoom_to_current_label()

        # Restore the hide labels checkbox state
        if checkbox_checked:
            self.hide_label_checkbox.setChecked(True)
            self.toggle_label_visibility()

        # Update the current file and trigger any other frame change events
        self.current_file = new_file
        self.on_frame_change()

        # Apply visibility settings
        self.update_bbox_visibility()

        # Redraw the ROI if the checkbox is checked
        if self.roi_checkbox.isChecked():
            self.update_roi()



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
            print(f"An OpenCV error occurred: {e}")
            print("Please ensure the input image is not empty.")
            return None  # or you can return the original image if you prefer: return image

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
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
        print(f"Flash effect changed: {self.flash_effect}")

    def apply_smoke_effect(self, image, intensity=0.05, speed=0.05):
        h, w, _ = image.shape

        # Increment the time for the rolling effect
        self.time += speed

        # Create a single light smoke mask with a more random and subtle pattern
        smoke_mask = self.generate_smoke_mask(h, w, scale_factor=1.0, time_offset=self.time)

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


    def generate_smoke_mask(self, h, w, scale_factor=1.0, time_offset=0.0, octaves=6, persistence=0.5, lacunarity=2.0):
        # Generate mesh grid for the image dimensions
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Create an empty array for noise
        noise = np.zeros((h, w), dtype=np.float32)

        # Directly iterate over the grid to compute Perlin noise
        for i in range(h):
            for j in range(w):
                noise[i, j] = pnoise2(
                    (X[i, j] / (20 * scale_factor) + time_offset),
                    (Y[i, j] / (20 * scale_factor)),
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=w,
                    repeaty=h,
                    base=0
                )

        # Normalize the noise to the range [0, 1]
        min_val, max_val = noise.min(), noise.max()
        if max_val - min_val > 0:  # Avoid division by zero
            noise = (noise - min_val) / (max_val - min_val)

        # Create a binary mask based on the Perlin noise values
        mask = noise > 0.5

        return mask


    def smoke_checkbox_changed(self, state):
        self.smoke_effect = state == Qt.Checked
        print(f"Smoke effect changed: {self.smoke_effect}")

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

        print(f"Found {len(thumbnail_paths)} thumbnails.")
        if not thumbnail_paths:
            print("No thumbnails found. Please check the directory.")
            return

        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory {output_dir}")

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
                    print(f"Failed to open image {path}: {e}")
                    continue

            packed_thumbnails = self.pack_thumbnails(thumbnail_sizes, canvas_size)
            if not packed_thumbnails:
                print("No more thumbnails can fit on the canvas.")
                break

            canvas = Image.new('RGB', canvas_size, color='white')

            for rect in packed_thumbnails:
                try:
                    img_path, x, y, w, h = rect
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    print(f"Placing image: {img_path} at position {(x, y)}, size {(w, h)}")

                    with Image.open(img_path) as thumbnail:
                        original_size = thumbnail.size

                        if original_size != (w, h) and original_size != (h, w):
                            thumbnail = thumbnail.resize((w, h), Image.Resampling.LANCZOS)
                        elif original_size == (h, w):
                            thumbnail = thumbnail.rotate(90, expand=True)
                            thumbnail = thumbnail.resize((w, h), Image.Resampling.LANCZOS)

                        canvas.paste(thumbnail, (x, y))

                except Exception as e:
                    print(f"Failed to place image {img_path}: {e}")
                    continue

            unique_id = uuid.uuid4()
            output_image_path = os.path.join(output_dir, f"mosaic_{unique_id}.jpeg")
            canvas.save(output_image_path)
            print(f"Saved {output_image_path}")

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
                        print(f"Failed to write annotation for {img_path}: {e}")
                        continue

            processed_images_count += 1

        print(f"All done. Mosaic images created: {processed_images_count}")
        return processed_images_count




    @pyqtSlot(int)
    def update_progress_bar(self, value):
        self.label_progress.setValue(value)

    def set_progress_bar_maximum(self, max_value):
        self.label_progress.setMaximum(max_value)

    def process_images_triggered(self):
        print("process_images_triggered called")

        if not self.images_import:
            QMessageBox.warning(self, "Error", "No images to process.")
            return

        output_directory = os.path.dirname(self.images_import[0])
        selected_effects = []

        # Check for mosaic effect and other effects
        processed_images_count = 0  # Initialize counter here

        if self.mosaic_effect:
            print("Creating mosaic images")
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

        # Generate a timestamp to append to filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Flip the image and labels if checked
        if self.flip_images:
            image = cv2.flip(image, 1)
            new_labels = []
            for label in labels:
                label_info = label.strip().split()
                class_id, x_center, y_center, width, height = map(float, label_info)
                x_center = 1 - x_center  # Flip the x_center for bounding box
                new_label = f"{int(class_id)} {x_center} {y_center} {width} {height}\n"  # Cast class_id to int
                new_labels.append(new_label)
            labels = new_labels  # Update the labels

        if labels:  # Check if labels are not empty
            # Parse the first label's information
            first_label_info = labels[0].strip().split()
            class_id, x_center, y_center, width, height = map(float, first_label_info)

            # Calculate actual x_center and y_center in pixels
            h, w, _ = image.shape
            actual_x_center = int(x_center * w)
            actual_y_center = int(y_center * h)

            # Apply circular sight picture if checked
            if self.sight_picture:
                center = (actual_x_center, actual_y_center)  # Center over the first labeled object
                mask = self.create_circular_mask(image, center, crosshair_length=50)

                # Where mask is True, make those pixels black
                image[mask] = [0, 0, 0]  # R, G, B

        # Apply smoke effect if checked
        if self.smoke_effect:
            image = self.apply_smoke_effect(image)
            if image is None:
                return

        # Apply flash effect if checked
        if self.flash_effect:
            image = self.apply_flashbang_effect(image)
            if image is None:
                return

        # Apply motion blur effect if checked
        if self.motion_blur_effect:
            image = self.apply_motion_blur_effect(image)
            if image is None:
                return

        # Apply glass effect if checked
        if self.glass_effect:
            image = self.apply_glass_effect(image)
            if image is None:
                return

        # Save the new image with a timestamp in the output folder
        output_image_path = os.path.join(output_path, f"{output_folder_name}_{timestamp}_{current_image}.jpg")
        cv2.imwrite(output_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Save YOLO label file with a timestamp
        output_label_path = os.path.join(output_path, f"{output_folder_name}_{timestamp}_{current_image}.txt")
        with open(output_label_path, 'w') as f:
            f.writelines(labels)

        # Update the progress
        progress_percentage = int((current_image + 1) / total_images * 100)
        self.progress_signal.emit(progress_percentage)




    def import_images_triggered(self):
        directory = QFileDialog.getExistingDirectory(None, 'Select Image Directory')
        if directory:
            self.thumbnail_dir = directory  # This should be set regardless of the files inside
            all_files = glob.glob(os.path.join(directory, '*'))

            self.images_import = []
            self.label_files = []

            for file_path in all_files:
                # Get file extension
                file_extension = os.path.splitext(file_path)[1].lower()

                # Check if it's an image
                if file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
                    self.images_import.append(file_path)

                # Check if it's a label file
                elif file_extension == '.txt':
                    self.label_files.append(file_path)





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
                print(f"Failed to read stylesheet file: {e}")
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
            print(f"No matching GIF found for theme: {selected_style}")

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
        print("on_save_dir_clicked called")  # Debugging print statement
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory", "", options=options)
        if directory:
            self.runs_directory = directory  # Update the save directory value



    def ultralytics_train_clicked(self):
        if not self.data_yaml_path:
            QMessageBox.warning(self, "Warning", "Data YAML file not selected. Please select a data YAML file before proceeding.")
            return
        
        # Base command setup
        command = "yolo train "  # Default to starting a new training session
        
        # Adjust command if resuming training
        if self.resume_checkbox.isChecked() and self.pt_path:
            command = f"yolo train resume model={self.pt_path} "  # Resume training command
        else:
            # Original command setup for new training sessions
            command += f"detect train data={self.data_yaml_path} "
            if self.model_config_path:
                command += f"model={self.model_config_path} "
                if self.pretrained_model_path:
                    command += f"pretrained={self.pretrained_model_path} "
            elif self.pt_path:
                command += f"model={self.pt_path} "
        
        # Append other training parameters
        imgsz_input = self.imgsz_input.text()
        epochs_input = self.epochs_input.text()
        batch_input = self.batch_input.text()
        command += f"imgsz={imgsz_input} epochs={epochs_input} batch={batch_input} "
        
        # Additional flags
        if self.half_true.isChecked():
            command += "half=True "
        if self.amp_true.isChecked():
            command += "amp=True "
        if self.freeze_checkbox.isChecked():
            freeze_layers = self.freeze_input.value()
            command += f"freeze={freeze_layers} "
        if self.patience_checkbox.isChecked():
            patience_value = self.patience_input.text()
            command += f"patience={patience_value} "
        
        command += f"project={self.runs_directory}"
        print("Training command:", command)

        # Construct the full command to run in a new terminal
        full_command = f"conda activate {os.environ['CONDA_DEFAULT_ENV']} && {command}"

        if os.name == 'nt':  # Windows
            subprocess.Popen(f'start cmd.exe /K "{full_command}"', shell=True)
        elif os.name == 'posix':  # macOS or Linux
            subprocess.Popen(f'gnome-terminal -- bash -c "{full_command}; exec bash"', shell=True)
        else:
            raise OSError("Unsupported operating system")

        # Automatically start TensorBoard if the checkbox is checked
        if self.tensorboardCheckbox.isChecked():
            tensorboard_log_dir = os.path.join(self.runs_directory, "train")  # Adjust this if necessary
            tb_command = f"tensorboard --logdir {tensorboard_log_dir}"
            if os.name == 'nt':  # Windows
                subprocess.Popen(f'start cmd.exe /K "{tb_command}"', shell=True)
            elif os.name == 'posix':  # macOS or Linux
                subprocess.Popen(f'gnome-terminal -- bash -c "{tb_command}; exec bash"', shell=True)
            else:
                raise OSError("Unsupported operating system")

            # Optionally open TensorBoard in the default web browser
            tensorboard_url = "http://localhost:6006"
            webbrowser.open(tensorboard_url)

            print("TensorBoard started and opened in web browser.")

        print("Training process started.")



    def load_model(self):
        home_directory = os.path.expanduser('~')
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', home_directory, "Model files (*.pt *.onnx)")


        if fname:
            self.model = YOLO(fname)

    def initiate_conversion(self):
        # Check if a model is loaded
        if self.model is None:
            # Show a warning message if no model is loaded
            QMessageBox.warning(self, "No Model", 'No model loaded')
            return

        # Get the selected format for model conversion
        format_selected = self.convert_model.currentText()

        # Check if the selected format is 'saved_model' (assuming it's the TensorFlow format)
        if format_selected.lower() == "saved_model":
            try:
                # Define the command for converting ONNX model to TensorFlow saved model
                # Assuming the input ONNX model is named 'model.onnx'
                command = [
                    "onnx2tf",
                    "-i", "model.onnx",
                    "-o", "model_saved_model",
                    "-b", "-ois", "-kt", "-kat", "-onwdt"
                ]

                # Run the conversion command using subprocess
                subprocess.run(command, check=True)

                # Show an information message indicating successful conversion
                QMessageBox.information(self, "Conversion Successful", f'Model converted to {format_selected}')
            except subprocess.CalledProcessError:
                # Show an error message if the conversion process encounters an error
                QMessageBox.critical(self, "Conversion Error", "Failed to convert model")
            except Exception as e:
                # Show an error message for any other exceptions
                QMessageBox.critical(self, "Conversion Error", str(e))
        else:
            try:
                # Export the model to the selected format
                self.model.export(format=format_selected)

                # Show an information message indicating successful conversion
                QMessageBox.information(self, "Conversion Successful", f'Model converted to {format_selected}')
            except Exception as e:
                # Show an error message if the conversion process encounters an error
                QMessageBox.critical(self, "Conversion Error", str(e))

    def handle_conversion(self):
        if self.model is None:
            QMessageBox.warning(self, "No Model", 'No model loaded')
            return

        format_selected = self.convert_model.currentText()
        self.update_gui_elements(format_selected)  # Update GUI elements based on the selected format

        # Collect image size parameters
        imgsz_height = self.imgsz_input_H.text()
        imgsz_width = self.imgsz_input_W.text()
        try:
            imgsz_list = [int(imgsz_height), int(imgsz_width)]
        except ValueError:
            QMessageBox.critical(self, "Conversion Error", f"Invalid image size values: Height={imgsz_height}, Width={imgsz_width}")
            return

        # Initialize and collect parameters
        export_params = {
            'format': format_selected,
            'imgsz': imgsz_list,
            'device': '0'  # Default device is '0'
        }

        if self.half_true.isChecked() and self.half_true.isEnabled():
            export_params['half'] = True

        if self.int8_true.isChecked() and self.int8_true.isEnabled():
            export_params['int8'] = True

        if self.simplify.isChecked() and self.simplify.isEnabled():
            export_params['simplify'] = True

        if self.dynamic_true.isChecked() and self.dynamic_true.isEnabled():
            export_params['dynamic'] = True

        if self.batch_size_checkbox.isChecked() and self.batch_size_checkbox.isEnabled():
            export_params['batch'] = self.batch_input_spinbox.value()

        # Execute the export
        try:
            self.model.export(**export_params)
            QMessageBox.information(self, "Conversion Successful", f'Model converted to {format_selected}')
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", str(e))


    def is_parameter_supported(self, format_selected, param):
        supported_params = {
            'torchscript': {'optimize', 'batch', 'imgsz'},  # Assuming TorchScript does not support 'half', 'int8', etc.
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
        print(f"Format changed to: {format_selected}")  # Debug print
        self.update_gui_elements(format_selected)



        # DEF FOR TXT MAKER
    def import_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        directory = QFileDialog.getExistingDirectory(
            self, "Select Import Directory", options=options)
        if directory:
            self.images = []
            for file in os.listdir(directory):
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".gif"):
                    self.images.append(os.path.join(
                        directory, file).replace("\\", "/"))

            # Look for classes.txt in the directory
            classes_file_path = os.path.join(directory, "classes.txt")
            if os.path.exists(classes_file_path):
                with open(classes_file_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
            else:
                self.classes = []

    def output_paths(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        # Determine if 'valid.txt' or 'train.txt' is selected and set the default filename accordingly
        selected_option = self.dropdown.currentText()
        default_filename = selected_option if selected_option in ["valid.txt", "train.txt"] else ""
        default_path = os.path.join("", default_filename)  # Set the default save directory if needed

        save_file, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", default_path, "Text Files (*.txt);;All Files (*)", options=options
        )

        if save_file:
            output_dir = os.path.dirname(save_file).replace("\\", "/")

            # Handle 'train.txt' creation
            if selected_option == "train.txt":
                train_txt_path = os.path.join(output_dir, "train.txt")
                with open(train_txt_path, "w", encoding="utf-8") as f:
                    for image in self.images:
                        f.write(image + "\n")
                QMessageBox.information(self, 'Information', 'train.txt file has been created!')

            # Handle 'valid.txt' creation
            elif selected_option == "valid.txt":
                valid_percent = int(self.valid_percent.currentText().replace('%', ''))

                # Create a copy of self.images to avoid modifying the original list
                images_copy = self.images[:]

                # Randomize the image list to avoid always selecting the first N images
                random.shuffle(images_copy)

                # Calculate the number of images to include in the valid.txt
                num_valid_images = int(len(images_copy) * (valid_percent / 100.0))

                # Filter images by checking if their corresponding .txt file is not empty
                valid_images = []
                for image in images_copy:
                    # Get the corresponding .txt file for each image
                    image_base = os.path.splitext(image)[0]
                    txt_file = image_base + ".txt"

                    # Check if the .txt file exists and is not empty
                    if os.path.exists(txt_file) and os.path.getsize(txt_file) > 0:
                        valid_images.append(image)

                    # Stop adding images if we've reached the number required
                    if len(valid_images) >= num_valid_images:
                        break

                # Write valid images to valid.txt
                with open(save_file, "w", encoding="utf-8") as f:
                    for image in valid_images:
                        f.write(image + "\n")

                QMessageBox.information(self, 'Information', 'valid.txt file has been created!')

            # Create or update other files like obj.data, obj.yaml, obj.names as necessary
            self.create_and_update_yaml_and_data_files(output_dir, selected_option)






    def create_and_update_yaml_and_data_files(self, output_dir, selected_option):
        """
        Helper method to create or update obj.yaml, obj.data, and obj.names files.
        This contains the logic from your original output_paths method for managing these files.
        """
        # Load existing class names from 'obj.names' if available
        obj_names_file = os.path.join(output_dir, "obj.names")
        if os.path.exists(obj_names_file):
            with open(obj_names_file, 'r', encoding="utf-8") as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            if selected_option == "valid.txt":
                QMessageBox.warning(self, 'Warning', 'The "obj.names" file does not exist. Please create it first.')
                return
            # Else, create a new 'obj.names' file
            with open(obj_names_file, "w", encoding="utf-8") as f:
                for class_name in self.classes:
                    f.write(class_name + "\n")

        # Update 'obj.data'
        class_numb = len(self.classes)
        train_txt_path = os.path.join(output_dir, "train.txt")
        valid_path = os.path.join(output_dir, "valid.txt") if selected_option == "valid.txt" else train_txt_path

        data_file_path = os.path.join(output_dir, "obj.data")
        with open(data_file_path, "w") as f:
            f.write("classes = " + str(class_numb) + "\n")
            f.write("train  = " + train_txt_path.replace("/", "\\") + "\n")
            f.write("valid  = " + valid_path.replace("/", "\\") + "\n")
            f.write("names = " + obj_names_file.replace("/", "\\") + "\n")
            f.write("backup = " + os.path.join(output_dir, "backup").replace("/", "\\") + "\n")

        # Update 'obj.yaml'
        obj_yaml_file = os.path.join(output_dir, "obj.yaml")
        with open(obj_yaml_file, "w", encoding="utf-8") as f:
            f.write("# YOLOv8 configuration file\n\n")
            f.write("# Dataset path\n")
            f.write("path: " + output_dir + "\n\n")
            f.write("# Training and validation set paths\n")
            f.write("train: " + train_txt_path.replace("\\", "/") + "\n")
            f.write("val: " + valid_path.replace("\\", "/") + "\n\n")
            f.write("# Class names and number of classes\n")
            f.write("names:\n")
            for i, name in enumerate(self.classes):
                f.write(f"  {i}: {name}\n")
            f.write("nc: " + str(class_numb) + "\n")

        # Create backup directory if it doesn't exist
        backup_dir = os.path.join(output_dir, "backup")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)


    # def for train darknet

    def browse_file_clicked(self, title, file_types, key, multiple=False):
        file_names = self.open_file_dialog(
            title, file_types, multiple=multiple)
        if multiple:
            file_names = [file for file in file_names if file.endswith(
                ('.weights', '.conv.')) or re.match('.*\.conv\.\d+', file)]
            file_label = "Files: " + ", ".join(file_names).rstrip()
        else:
            file_label = f"File: {file_names}"

        self.file_paths[key] = file_names
        getattr(self, f"{key}_label").setText(file_label)

    def browse_data_clicked(self):
        self.browse_file_clicked(
            "Select Data File", "Data Files (*.data);;All Files (*)", "data")

    def browse_cfg_clicked(self):
        self.browse_file_clicked("Select Config File",
                                 "Config Files (*.cfg);;All Files (*)", "cfg")

    def browse_weights_clicked(self):
        self.browse_file_clicked(
            "Select Weights Files", "Weights Files (*.weights *.conv.*);;All Files (*)", "weights", multiple=True)

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
        print(f"Command to be executed: {command}")





    def import_data(self):
        """Import only text files from the selected directory and update class configurations."""
        # Open a file dialog to select any directory (no hardcoded starting path)
        data_directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Data Directory')
        
        # Check if a directory was selected
        if not data_directory:
            return

        # Normalize the file path to ensure compatibility across platforms
        data_directory = os.path.normpath(data_directory)

        # Gather only text files (.txt) from the directory
        self.text_files = glob.glob(os.path.join(data_directory, '*.txt'))
        self.image_directory = data_directory

        # Print loaded text files for debugging purposes
        print(f"Detected text files: {self.text_files}")

        # Check for classes.txt and update configuration accordingly
        classes_txt_file = os.path.join(data_directory, 'classes.txt')
        if os.path.exists(classes_txt_file):
            with open(classes_txt_file, 'r') as f:
                class_lines = f.readlines()
                num_classes = len(class_lines)
                print(f"Detected {num_classes} classes from {classes_txt_file}")

            # Update classes and max_batches in the configuration and table
            self.update_classes_and_batches(num_classes)
        else:
            print(f"Could not find classes.txt in {data_directory}")



    def update_classes_and_batches(self, num_classes):
        """Update classes, max_batches, steps, and scales in both the CFG file and the table."""
        max_batches = num_classes * 5000
        steps = ",".join([str(int(p * max_batches)) for p in self.STEP_PERCENTAGES])
        scales = self.DEFAULT_SCALES

        # Update the CFG file
        self.update_cfg_param('classes', num_classes)
        self.update_cfg_param('max_batches', max_batches)
        self.update_cfg_param('steps', steps)
        self.update_cfg_param('scales', scales)

        # Update the cfg_table in real-time after the file is updated
        self.update_table_param('classes', num_classes)
        self.update_table_param('max_batches', max_batches)
        self.update_table_param('steps', steps)
        self.update_table_param('scales', scales)




    def calculate_anchors(self):
        """Calculate anchors and update CFG file, table, and plot anchors."""
        try:
            if not self.text_files:
                QtWidgets.QMessageBox.warning(self, "Error", "No annotation text files loaded.")
                return

            if not hasattr(self, 'filename') or not self.filename:
                QtWidgets.QMessageBox.warning(self, "Error", "Please open a .cfg file before calculating anchors.")
                return

            # Get current values from the spinboxes
            num_clusters = self.clusters_spinbox.value()
            width_value = self.width_spinbox.value()
            height_value = self.height_spinbox.value()

            # Collect annotation dimensions
            annotation_dims = self.get_annotation_dimensions()

            # Subsample annotations to speed up calculation (e.g., take 50k samples randomly)
            max_samples = 50000  # Adjust this number depending on your hardware
            if len(annotation_dims) > max_samples:
                annotation_dims = random.sample(annotation_dims, max_samples)
                print(f"Subsampled {max_samples} annotations for anchor calculation.")

            if not annotation_dims:
                QtWidgets.QMessageBox.warning(self, "Error", "No valid annotation dimensions found.")
                return

            # Use MiniBatchKMeans for faster clustering on large datasets
            X = np.array(annotation_dims)
            mini_kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=10000)
            mini_kmeans.fit(X)
            centroids = mini_kmeans.cluster_centers_
            centroids = sorted(centroids, key=lambda c: c[0] * c[1])

            # Convert centroids to pixel values
            anchors_pixel = np.array([list(map(int, centroid * [width_value, height_value])) for centroid in centroids])
            anchors_str = ', '.join(map(str, anchors_pixel.flatten()))

            # Update anchors, width, and height in the CFG file and table
            self.update_cfg_anchors(anchors_str)
            self.update_cfg_param('width', width_value)
            self.update_cfg_param('height', height_value)
            self.update_table_param('anchors', anchors_str)
            self.update_table_param('width', width_value)
            self.update_table_param('height', height_value)

            # Calculate and save the average IoU
            avg_iou = round(self.calculate_avg_iou(annotation_dims, centroids) * 100, 2)
            self.saveAnchorsSettings(avg_iou)

            # Plot the anchors
            self.plot_anchors(centroids, num_clusters, avg_iou)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")




    def get_annotation_dimensions(self):
        """Retrieve annotation dimensions from text files."""
        annotation_dims = []
        for text_file in self.text_files:
            if text_file.endswith("classes.txt"):
                continue
            with open(text_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            width = float(parts[3])
                            height = float(parts[4])
                            annotation_dims.append((width, height))
                        except ValueError:
                            print(f"Skipping line in {text_file}: width or height are not numbers")
                    else:
                        print(f"Skipping line in {text_file}: not enough parts")
        return annotation_dims


    def update_cfg_param(self, param_name, value):
        """Update a parameter in the CFG file."""
        if not hasattr(self, 'filename') or not self.filename:
            QtWidgets.QMessageBox.warning(self, "Error", f"Please open a .cfg file before updating {param_name}.")
            return

        try:
            # Read the CFG file
            with open(self.filename, 'r') as f:
                lines = f.readlines()

            new_config = ""
            param_updated = False

            # Update the relevant parameter in the file
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith(f"{param_name}="):
                    new_config += f"{param_name}={value}\n"
                    param_updated = True
                else:
                    new_config += line

            # Append the parameter if it wasn't found
            if not param_updated:
                new_config += f"{param_name}={value}\n"

            # Write the updated config back to the file
            with open(self.filename, 'w') as f:
                f.write(new_config)

            # Reflect the changes in the cfg_table
            self.update_table_param(param_name, value)  # <-- Ensure this is called after updating the cfg file

            print(f"Updated {param_name} to {value} in the CFG file and table.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred while updating the CFG file: {str(e)}")



    def update_table_param(self, param_name, value):
        """Update a parameter in the cfg_table."""
        for row in range(self.cfg_table.rowCount()):
            param_item = self.cfg_table.item(row, 0)
            if param_item and param_name in param_item.text():
                self.cfg_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))
                print(f"Updated '{param_name}' in cfg_table to {value}")


    def calculate_avg_iou(self, annotation_dims, centroids):
        """Calculate average IoU for anchors."""
        avg_iou = sum(max(self._iou(centroid, annotation) for centroid in centroids) for annotation in annotation_dims)
        return avg_iou / len(annotation_dims)


    def update_cfg_anchors(self, anchors):
        """Update anchors in the CFG file and table."""
        self.update_cfg_param('anchors', anchors)
        self.update_table_param('anchors', anchors)


    def _iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes."""
        intersect_w = min(box1[0], box2[0])
        intersect_h = min(box1[1], box2[1])
        intersect = intersect_w * intersect_h
        union = box1[0] * box1[1] + box2[0] * box2[1] - intersect
        return intersect / union




    def plot_anchors(self, centroids, num_clusters, avg_iou):
        """Plot anchors and display the average IoU."""
        if self.show_checkbox.isChecked():
            fig, ax = plt.subplots(figsize=(7, 7))
            
            # Set a neutral background color for the plot
            ax.set_facecolor('white')
            
            # Set up colors for the anchor boxes
            colors = plt.cm.get_cmap('tab10', num_clusters)
            patches = []

            # Draw each anchor box as a rectangle
            for i, centroid in enumerate(centroids):
                anchor_width, anchor_height = centroid
                rect = mpatches.Rectangle((0, 0), anchor_width, anchor_height, linewidth=2, 
                                        edgecolor=colors(i), facecolor='none')
                patches.append(rect)
                ax.add_patch(rect)

            # Set labels for the axes
            plt.xlabel('Width', fontsize=14)
            plt.ylabel('Height', fontsize=14)
            plt.title('Anchors', fontsize=16)

            # Move the legend outside the plot
            plt.legend(patches, [f'Anchor {i + 1}' for i in range(num_clusters)], 
                    loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

            # Add space for the legend by adjusting the layout
            plt.subplots_adjust(right=0.75)

            # Create a more subtle text display for width, height, and average IoU outside the plot
            width_value = self.width_spinbox.value()
            height_value = self.height_spinbox.value()
            fig.text(0.75, 0.85, f"Image Width: {width_value}", fontsize=12)
            fig.text(0.75, 0.80, f"Image Height: {height_value}", fontsize=12)
            fig.text(0.75, 0.75, f"Avg IoU: {avg_iou:.2f}%", fontsize=12, color='red')

            # Show the plot
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

            print(f"Anchor settings saved successfully with avg IoU: {avg_iou}%")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error saving anchor settings: {str(e)}")


    def main(self):
        self.import_data()
        if not self.image_files or not self.text_files:
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

    def parse_cfg_file(self, file_name=None, anchors_list=None):
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
            # Create a dictionary with the parameter values from the table widget
            table_data = {}
            for row in range(self.cfg_table.rowCount()):
                param = self.cfg_table.item(row, 0).text()
                item = self.cfg_table.item(row, 1)
                value = item.text() if item else ""
                table_data[param] = value

            # Calculate max_batches if 'max_batches' exists in the table
            for key in table_data.keys():
                if "max_batches" in key:
                    max_batches = int(table_data[key])
                    table_data[key] = str(max_batches)

            # Modify the relevant lines in the .cfg file
            new_config = ""
            section_idx = -1
            current_section = ""
            with open(self.filename, "r", encoding="utf-8") as f:
                for line in f:
                    stripped_line = line.strip()

                    if stripped_line.startswith("["):
                        section_idx += 1
                        current_section = stripped_line.strip("[]")
                        new_config += line
                    elif "=" in stripped_line:
                        # Safely split only at the first "="
                        param, value = stripped_line.split("=", 1)
                        param = param.strip()
                        new_value = table_data.get(f"{param}_{section_idx}", value.strip())
                        new_line = f"{param}={new_value}\n"
                        new_config += new_line
                    else:
                        new_config += line

            # Save the new configuration to file
            options = QtWidgets.QFileDialog.Options()
            save_file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Config File As", "C:/Users/YourUsername/Documents", "Config Files (*.cfg);;All Files (*)", options=options)

            if save_file_name:
                if not save_file_name.endswith('.cfg'):
                    save_file_name += '.cfg'

                try:
                    with open(save_file_name, 'w', encoding='utf-8') as f:
                        f.write(new_config)

                    QtWidgets.QMessageBox.information(self, "Success", "Configuration file saved successfully.")
                except PermissionError as e:
                    QtWidgets.QMessageBox.critical(self, "Error", f"Permission denied: {str(e)}")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")






    def new_method(self, text):
        # Return a QTableWidgetItem instance with the text
        return QtWidgets.QTableWidgetItem(text)

     # combine txt files

    def on_combine_txt_clicked(self):
        if self.combine_txt_flag:
            print("Function is already running.")
            return
        self.combine_txt_flag = True
        print("Function called.")
        self.combine_txt_button.setEnabled(False)

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file1, _ = QFileDialog.getOpenFileName(
            self, "Select First File", "", "Text Files (*.txt);;All Files (*)", options=options
        )
        if not file1:
            self.combine_txt_flag = False
            print("File 1 prompt cancelled.")
            self.combine_txt_button.setEnabled(True)
            return
        file2, _ = QFileDialog.getOpenFileName(
            self, "Select Second File", "", "Text Files (*.txt);;All Files (*)", options=options
        )
        if not file2:
            self.combine_txt_flag = False
            print("File 2 prompt cancelled.")
            self.combine_txt_button.setEnabled(True)
            return
        output_file, _ = QFileDialog.getSaveFileName(
            self, "Save Combined File", "", "Text Files (*.txt);;All Files (*)", options=options
        )
        if not output_file:
            self.combine_txt_flag = False
            print("Output file prompt cancelled.")
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
            print("Function finished successfully.")
            self.combine_txt_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while combining files: {e}")
            self.combine_txt_flag = False
            print("Function finished with error.")
            self.combine_txt_button.setEnabled(True)


def run_pyqt_app():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    exit_code = app.exec_()

    # Delete the application instance to prevent memory leaks
    app.deleteLater()
    sys.exit(exit_code)

if __name__ == "__main__":
    # Use cProfile to profile the execution of the 'run_pyqt_app()' function
    cProfile.run('run_pyqt_app()')

