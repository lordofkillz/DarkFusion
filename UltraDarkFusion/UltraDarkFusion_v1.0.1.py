import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import necessary libraries and modules for the application
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler
import concurrent.futures
import cProfile
import ctypes
from datetime import datetime, timedelta
from datetime import datetime
from typing import List, Tuple
import glob
import json
import threading
import queue
import os
import random
import re
from noise import snoise2 as perlin2
import shutil
import signal
import subprocess
import cv2
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
from matplotlib import image
import functools
from PIL import Image
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtCore import (QCoreApplication, QDir, QEvent, QModelIndex, QObject,
                          QPoint, QRectF, QRunnable, Qt, QThread, QThreadPool,
                          QTimer, QUrl, pyqtSignal, pyqtSlot, QPointF,QModelIndex)
from PyQt5.QtGui import (QBrush, QColor, QFont, QImage, QImageReader,
                         QImageWriter, QKeySequence, QMovie, QPainter, QPen,
                         QPixmap,  QStandardItem,
                         QStandardItemModel, QTransform, QLinearGradient,QIcon,QCursor)
from PyQt5.QtWidgets import (QApplication, QFileDialog,
                             QGraphicsDropShadowEffect, QGraphicsItem,
                             QGraphicsPixmapItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsTextItem, QGraphicsView,
                             QLabel, QMessageBox, QProgressBar,
                             QTableWidgetItem, QColorDialog, QMenu,QSplashScreen,QTableView, QVBoxLayout,QWidget,QHeaderView)
from pytube import YouTube
from PyQt5 import QtWidgets, QtGui
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.cluster import KMeans
from qt_material import apply_stylesheet, list_themes
from logging.handlers import RotatingFileHandler
from segment_anything import sam_model_registry, SamPredictor
import pybboxes as pbx
import io
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from dino import run_groundingdino
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtCore import QProcess
import webbrowser
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image, ImageDraw
from rectpack import newPacker



class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, directory, update_json_callback):
        self.directory = directory
        self.update_json_callback = update_json_callback

    def on_modified(self, event):
        if event.src_path.endswith('.txt') and event.src_path.startswith(self.directory):
            # Using a lambda to adapt the argument list
            callback = lambda path: self.update_json_callback(path)
            callback(event.src_path)

# Create or get the root logger
root_logger = logging.getLogger()
# Remove any existing handlers on the root logger to prevent duplicate messages
while root_logger.handlers:
    root_logger.removeHandler(root_logger.handlers[0])

# Set up logging configurations
start_time = datetime.now()
if not os.path.exists('debug'):
    os.makedirs('debug')

# Set up file handler for logging
file_handler = RotatingFileHandler(os.path.join('debug', 'my_app.log'), maxBytes=10000, backupCount=3)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: \x1b[31m%(message)s\x1b[0m')
file_handler.setFormatter(file_formatter)

# Set up console handler for logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s \x1b[31m%(message)s\x1b[0m')
console.setFormatter(console_formatter)

# Create a logger with a unique name
logger = logging.getLogger('UniqueLoggerName')
# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console)

# Set the logging level for this logger
logger.setLevel(logging.DEBUG)

# Prevent this logger from propagating messages to higher-level loggers
logger.propagate = False

class CustomLoggerAdapter(logging.LoggerAdapter):
    """
    A custom logger adapter class to prepend the current date and time to log messages.
    """
    def process(self, msg, kwargs) -> Tuple[str, dict]:
        return f'{datetime.now()} - {msg}', kwargs

# Wrap logger with the CustomLoggerAdapter
logger = CustomLoggerAdapter(logger, {})

app = QApplication([])

# Check for CUDA availability in PyTorch and OpenCV
pytorch_cuda_available = torch.cuda.is_available()
opencv_cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

# Create the initialization log message
init_msg = f"Initializing UltraDarkFusion... PyTorch CUDA: {'True' if pytorch_cuda_available else 'False'}, OpenCV CUDA: {'True' if opencv_cuda_available else 'False'}"

# Use the logger to log an initialization message with CUDA availability info.
logger.info(init_msg)

# The @functools.lru_cache(maxsize=None) decorator is employed to optimize the function's performance when it is called
# with the same arguments multiple times. In this specific case, it’s beneficial when cycling through images using timers
# as it prevents the redundant and potentially expensive I/O operation of loading the same image from disk repeatedly.
# The lru_cache will store the results of previous calls to load_image, and when the function is called with the same
# image_file argument, it will return the cached result instead of re-reading the image from disk.
#
# Additionally, by defining this function globally, it’s clear that it doesn’t rely on, or alter, any external state,
# making the behavior of the function predictable and the debugging process simpler if issues arise.

@functools.lru_cache(maxsize=None)
def load_image(image_file):
    try:
        pixmap = QPixmap(image_file)
        if pixmap.isNull():
            print(f"Failed to load image: {image_file}")
            return None
        return pixmap
    except Exception as e:
        print(f"Error reading image file {image_file}: {str(e)}")
        return None




# Set up a splash screen

class SplashScreen(QSplashScreen):
    def __init__(self, gif_path):
        pixmap = QPixmap()  # Create an empty QPixmap
        super(SplashScreen, self).__init__(pixmap, Qt.WindowStaysOnTopHint)

        # Create a QMovie from the provided GIF path and connect the frameChanged signal
        # to the set_frame method for dynamic updating.
        self.movie = QMovie(gif_path)
        self.movie.frameChanged.connect(self.set_frame)
        self.movie.start()

        # Create a QMediaPlayer for playing a sound and set the media content.
        self.sound_player = QMediaPlayer()
        self.sound_player.setMedia(QMediaContent(QUrl.fromLocalFile("sounds/machine2.wav")))

        # Play the sound.
        self.sound_player.play()

    def set_frame(self):
        self.setPixmap(self.movie.currentPixmap())

# Create an instance of the SplashScreen class with the provided GIF path.
splash = SplashScreen("styles/darknet3.gif")

# Show the splash screen.
splash.show()

# Allow Qt to process events and update the GUI.
app.processEvents()

# Set up a timer to close the splash screen after 2 seconds (2000 milliseconds).
timer = QTimer()
timer.singleShot(2000, splash.close)





class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None, main_window=None):
        super(CustomGraphicsView, self).__init__(parent)
        self.main_window = main_window

        # Set up initial state and render settings
        self._setup_initial_state()
        self._setup_render_settings()

        # Create a sound player for audio
        self.sound_player = QMediaPlayer()

        # Set the sound player to be muted by default
        self.sound_player.setMuted(True)

        # Set the mute checkbox state to Checked by default
        self.main_window.muteCheckBox.setChecked(True)

        # Connect the mute checkbox state change to the mute_player method
        self.main_window.muteCheckBox.stateChanged.connect(self.mute_player)

        # Initialize zoom scale and fitInView scale factor
        self.zoom_scale = 1.0
        self.fitInView_scale = 1.0

        # Track whether to display the crosshair
        self.show_crosshair = False

        # Add a drop shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setColor(QColor(0, 0, 0, 100))  # Set the shadow color and opacity
        shadow.setBlurRadius(10)  # Adjust the blur radius as needed
        self.setGraphicsEffect(shadow)

        # Enable mouse tracking
        self.setMouseTracking(True)

        # Set the transformation anchor and drag mode
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

        # Set render hints, cache mode, and optimization flags
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setCacheMode(QGraphicsView.CacheBackground)
        self.setOptimizationFlags(QGraphicsView.DontSavePainterState)

        # Set horizontal and vertical scroll bar policies
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # Set the frame shape to NoFrame
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.right_click_timer = QTimer()  # Timer to handle right-click hold functionality
        self.right_click_timer.setInterval(100)  # Set the timer interval to 100ms or any other suitable value
        self.right_click_timer.timeout.connect(self.remove_box_under_cursor)  # Connect the timer to the removal method
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
        self.moving_view = False
        self.moving_bbox = True
        self.create_bbox_on_click = True
        self.crosshair_position = QPoint()
        self.xy_lines_checkbox = self.main_window.xy_lines_checkbox
        self.xy_lines_checkbox.stateChanged.connect(self.toggle_crosshair)
        self.main_window.crosshair_color.clicked.connect(self.pick_color)
        self.main_window.box_size.valueChanged.connect(self.update_bbox_size)


    def contextMenuEvent(self, event):
        menu = QMenu(self)  # Create a context menu
        right_click_pos = self.mapToScene(event.pos())  # Convert event position to scene coordinates
        last_bbox = self._get_last_drawn_bbox()  # Get the last drawn bounding box

        # If there's a last drawn bounding box and the right-click position is within it
        if last_bbox and last_bbox.contains(right_click_pos):
            delete_action = menu.addAction("Delete Last Drawn Box")  # Add a delete action to the menu
            action = menu.exec_(event.globalPos())  # Execute the menu at the global position

            # If the user selects the delete action, remove the last drawn bounding box
            if action == delete_action:
                self._remove_last_drawn_bbox(last_bbox)

    def remove_box_under_cursor(self):
        # This method will be called repeatedly while the right mouse button is held down
        cursor_pos = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
        for item in self.scene().items(cursor_pos):
            if isinstance(item, BoundingBoxDrawer):
                self._play_sound_and_remove_bbox(item)
                break  # Remove only one box per interval

    def _get_last_drawn_bbox(self):
        if self.bboxes:
            return self.bboxes[-1]  # Return the last bounding box in the list

    def _remove_last_drawn_bbox(self, bbox):
        if bbox:
            self.scene().removeItem(bbox)  # Remove the bounding box from the scene
            self.scene().removeItem(bbox.class_name_item)  # Remove the class name item associated with the bounding box
            self.bboxes.remove(bbox)  # Remove the bounding box from the list of bounding boxes
            self._save_bboxes()  # Save the updated list of bounding boxes


    def update_bbox_size(self, value):
        BoundingBoxDrawer.MIN_SIZE = value  # Set the minimum size to the specified value
        BoundingBoxDrawer.MAX_SIZE = value  # Set the maximum size to the specified value

    def pick_color(self):
        color = QColorDialog.getColor()  # Open a color dialog to choose a color
        if color.isValid():
            # Save the chosen color in RGB format
            self.crosshair_color_rgb = color.getRgb()


    def wheelEvent(self, event):
        """
        Handle the wheel (mouse scroll) event for zooming the view.

        :param event: The wheel event triggered by scrolling the mouse wheel.
        """
        logging.info("Wheel event triggered.")  # Log that the wheel event has been triggered

        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            # Check if the Control modifier is detected
            logging.info("Control modifier detected; passing event to BoundingBoxDrawer.")
            super().wheelEvent(event)  # Pass the event to BoundingBoxDrawer to handle it
        else:
            zoomInFactor = 1.15
            zoomOutFactor = 1 / zoomInFactor

            # Save the scene position
            oldPos = self.mapToScene(event.pos())

            # Zoom in or out based on the direction of the wheel scroll
            if event.angleDelta().y() > 0:
                zoomFactor = zoomInFactor
                self.zoom_scale *= zoomInFactor
                logging.info("Zooming in; new zoom scale: %s", self.zoom_scale)
            else:
                # Prevent zooming out if it would result in a scale smaller than fitInView scale
                if self.zoom_scale * zoomOutFactor < self.fitInView_scale:
                    logging.warning("Zoom out prevented to maintain fitInView scale.")
                    return
                zoomFactor = zoomOutFactor
                self.zoom_scale *= zoomOutFactor
                logging.info("Zooming out; new zoom scale: %s", self.zoom_scale)

            self.scale(zoomFactor, zoomFactor)  # Apply the zoom factor to the view

            # Get the new position in the scene
            newPos = self.mapToScene(event.pos())

            # Adjust the center of the view based on the change in position
            delta = newPos - oldPos
            self.centerOn(self.mapToScene(self.viewport().rect().center()) - delta)
            logging.info("View adjusted; new center position: %s", self.mapToScene(self.viewport().rect().center()) - delta)


    def resizeEvent(self, event):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)  # Fit the view to the scene with aspect ratio
        self.fitInView_scale = self.transform().mapRect(QRectF(0, 0, 1, 1)).width()  # Calculate the fitInView scale
        self.setTransform(QTransform().scale(self.zoom_scale, self.zoom_scale))  # Apply the zoom scale
        self.update()  # Update the view immediately

    def _setup_render_settings(self):
        self.setBackgroundBrush(QColor(0, 0, 0))  # Set the background color to black
        self.setRenderHint(QPainter.Antialiasing)  # Enable antialiasing for smoother rendering
        self.setRenderHint(QPainter.SmoothPixmapTransform)  # Enable smooth pixmap transformation

    def mute_player(self, state):
        self.sound_player.setMuted(state == Qt.Checked)  # Mute or unmute the sound player

    def set_sound(self, sound_path):
        self.sound_player.setMedia(QMediaContent(QUrl.fromLocalFile(sound_path)))  # Set the media content with the specified sound file


    def mousePressEvent(self, event):
        self.last_mouse_pos = self.mapToScene(event.pos())  # Store the last clicked position

        if event.button() == Qt.LeftButton:
            self._handle_left_button_press(event)  # Handle left mouse button press
        elif event.button() == Qt.RightButton:
            self._handle_right_button_press(event)  # Handle right mouse button press
            self.right_click_timer.start()  # Start the timer when the right mouse button is pressed
        else:
            super().mousePressEvent(event)  # Call the base class method for other button press events


    def _handle_left_button_press(self, event):
        if event.modifiers() == Qt.ControlModifier:
            self._handle_control_modifier_with_selected_bbox(event)  # Handle control modifier with selected bounding box
        elif event.type() == QEvent.MouseButtonDblClick:
            self._handle_double_click(event)  # Handle double-click event
        else:
            self._handle_single_click(event)  # Handle single-click event

    def _handle_control_modifier_with_selected_bbox(self, event):
        if self.selected_bbox:
            self.initial_mouse_pos = event.pos()  # Store the initial mouse position
            self.setDragMode(QGraphicsView.NoDrag)  # Disable dragging

    def _handle_double_click(self, event):
        if not self.selected_bbox and self.scene():
            item = self.itemAt(event.pos())  # Get the item at the double-click position
            if isinstance(item, BoundingBoxDrawer):
                self.selected_bbox = item  # Select the clicked bounding box
                self.viewport().setCursor(Qt.OpenHandCursor)  # Set the cursor to open hand
            else:
                self.clear_selection()  # Clear the selection if clicking outside a bounding box

    def _handle_single_click(self, event):
        if not self.drawing and self.scene():
            item = self.itemAt(event.pos())  # Get the item at the click position
            self._handle_single_click_with_item(item, event)  # Handle click with the item
        else:
            self._handle_single_click_without_item()  # Handle click without an item (e.g., during drawing)

    def _handle_single_click_with_item(self, item, event):
        if isinstance(item, BoundingBoxDrawer):
            self._handle_single_click_with_bbox(item)  # Handle click with a bounding box item
        else:
            self._handle_single_click_without_bbox(event)  # Handle click without a bounding box item

    def _handle_single_click_with_bbox(self, item):
        if self.selected_bbox:
            self.selected_bbox.set_selected(False)  # Deselect the previously selected bounding box
        self.selected_bbox = item  # Select the clicked bounding box
        self.selected_bbox.set_selected(True)  # Set the clicked bounding box as selected
        self.viewport().setCursor(Qt.ArrowCursor)  # Set the cursor to arrow

    def _handle_single_click_without_bbox(self, event):
        if not self.selected_bbox and self.create_bbox_on_click:
            self.drawing = True  # Start drawing mode
            self.start_point = self.mapToScene(event.pos())  # Store the starting point of the new bounding box
            self._create_new_bbox()  # Create a new bounding box
            if self.main_window.hide_label_checkbox.isChecked():
                self.current_bbox.hide_labels()  # Hide labels for the new bounding box

    def _handle_single_click_without_item(self):
        if self.selected_bbox:
            self.selected_bbox.set_selected(False)  # Deselect the previously selected bounding box
        self.selected_bbox = None  # Clear the selected bounding box
        self.viewport().setCursor(Qt.ArrowCursor)  # Set the cursor to arrow


    def _handle_right_button_press(self, event):
        click_pos = self.mapToScene(event.pos())  # Convert the event position to scene coordinates
        tolerance = 10  # Tolerance for bounding box containment check

        # Iterate through all items in the scene
        for item in self.scene().items():
            if isinstance(item, BoundingBoxDrawer):
                # Check if the item's bounding box (with tolerance) contains the click position
                if item.rect().adjusted(-tolerance, -tolerance, tolerance, tolerance).contains(click_pos):
                    self._play_sound_and_remove_bbox(item)  # Play sound and remove the clicked bounding box
                    break

    def _create_new_bbox(self):
        self.current_bbox = BoundingBoxDrawer(
            self.start_point.x(),
            self.start_point.y(),
            0,
            0,
            main_window=self.parent(),
            class_id=self.main_window.get_current_class_id()
        )
        self.scene().addItem(self.current_bbox)  # Add the new bounding box to the scene

    def _play_sound_and_remove_bbox(self, item):
        self.set_sound('sounds/shotgun.wav')  # Set and play the sound
        self.sound_player.play()
        self.scene().removeItem(item)  # Remove the bounding box item from the scene
        self.scene().removeItem(item.class_name_item)  # Remove the associated class name item
        self._save_bboxes()  # Save the updated bounding box list
        if self.selected_bbox == item:
            self.selected_bbox = None  # Clear the selected bounding box if it was the removed one

    def _save_bboxes(self):
        self.main_window.save_bounding_boxes(
            self.main_window.label_file, self.scene().width(), self.scene().height())

    def mouseMoveEvent(self, event):
        if self.show_crosshair:
            self._update_crosshair(event.pos())  # Update the crosshair position if enabled
        if self.moving_view:
            self._handle_moving_view(event)  # Handle moving the view
        elif self.drawing and self.current_bbox:
            self._handle_drawing_bbox(event)  # Handle drawing a new bounding box
        elif self.selected_bbox and not self.drawing:
            self.selected_bbox.mouseMoveEvent(event)  # Handle mouse move on a selected bounding box
        else:
            super().mouseMoveEvent(event)  # Call the base class method for other mouse move events

    def _update_crosshair(self, pos):
        self.crosshair_position = pos  # Update the crosshair position
        self.viewport().update()  # Trigger a viewport update to redraw the crosshair

    def paintEvent(self, event):
        super().paintEvent(event)  # Call the base class paint event
        painter = QPainter(self.viewport())  # Create a QPainter for the viewport
        painter.setRenderHint(QPainter.Antialiasing)  # Enable antialiasing for smoother rendering
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # Enable smooth pixmap transformation

        if self.show_crosshair:
            # Set the crosshair color
            if hasattr(self, 'crosshair_color_rgb'):
                painter.setPen(QColor(*self.crosshair_color_rgb))
            else:
                painter.setPen(Qt.yellow)

            center_x = self.crosshair_position.x()
            center_y = self.crosshair_position.y()

            # Draw horizontal and vertical lines to create the crosshair
            painter.drawLine(center_x, self.viewport().rect().top(),
                            center_x, self.viewport().rect().bottom())
            painter.drawLine(self.viewport().rect().left(), center_y,
                            self.viewport().rect().right(), center_y)

        painter.end()  # Ensure to end the painter operation


    def _handle_moving_view(self, event):
        if self.last_mouse_position is None:
            self.last_mouse_position = event.pos()
        else:
            self._update_scrollbars(event)  # Update the scrollbars based on mouse movement

    def _update_scrollbars(self, event):
        delta = event.pos() - self.last_mouse_position
        self.last_mouse_position = event.pos()
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() - delta.x())  # Update the horizontal scrollbar
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().value() - delta.y())  # Update the vertical scrollbar

    def _handle_drawing_bbox(self, event):
        try:
            end_point = self.mapToScene(event.pos())
            x, y = self._get_bbox_coordinates(end_point)
            width, height = self._get_bbox_dimensions(end_point, x, y)
            self.current_bbox.setRect(x, y, width, height)  # This line appears to be the source of the error
        except RuntimeError as e:
            print(f"Caught RuntimeError: {e}")



    def _get_bbox_coordinates(self, end_point):
        x = max(0, min(self.start_point.x(), end_point.x()))
        y = max(0, min(self.start_point.y(), end_point.y()))
        return x, y

    def _get_bbox_dimensions(self, end_point, x, y):
        width = min(max(BoundingBoxDrawer.MIN_SIZE, abs(self.start_point.x() - end_point.x())),
                    self.scene().width() - x)
        height = min(max(BoundingBoxDrawer.MIN_SIZE, abs(self.start_point.y() - end_point.y())),
                    self.scene().height() - y)
        return width, height

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)  # Call the base class mouse release event

        if event.button() == Qt.LeftButton:
            self._handle_left_button_release(event)  # Handle left mouse button release
        if event.button() == Qt.RightButton:
            self.right_click_timer.stop()  # Stop the timer when the right mouse button is released
            
    def _handle_left_button_release(self, event):
        if self.selected_bbox:
            self._update_selected_bbox()  # Update the selected bounding box
        if self.drawing and self.current_bbox:
            self._handle_drawing_current_bbox()  # Handle the current bounding box being drawn
        if event.modifiers() == Qt.ControlModifier and self.selected_bbox:
            self._reset_after_bbox_move()  # Reset after moving a bounding box with the Control modifier
        elif self.selected_bbox:
            self._reset_selected_bbox()  # Reset the selected bounding box

    def _update_selected_bbox(self):
        self.selected_bbox.update_position_and_size()  # Update the position and size of the selected bounding box
        self.selected_bbox.update_bbox()  # Update the bounding box representation
        self.main_window.save_bounding_boxes(
            self.main_window.label_file, self.scene().width(), self.scene().height())  # Save the bounding box changes


    def _handle_drawing_current_bbox(self):
        try:
            # Check if self.current_bbox is not None and has a minimum size
            if self.current_bbox and self.current_bbox.rect().width() >= 6 and self.current_bbox.rect().height() >= 6:
                self._save_and_play_sound()  # Save and play a sound when a valid bounding box is created
            else:
                self._remove_current_bbox_items()  # Remove the current bounding box items if it doesn't meet the minimum size
        except RuntimeError as e:
            print(f"An error occurred: {e}")
        finally:
            self.drawing = False  # Exit drawing mode
            self.current_bbox = None  # Clear the current bounding box
            self.clear_selection()  # Clear any selection

    def _save_and_play_sound(self):
        self.main_window.set_selected(None)
        self.main_window.remove_bbox_from_classes_txt(self.current_bbox)
        self.main_window.save_bounding_boxes(
            self.main_window.label_file, self.main_window.screen_view.scene().width(),
            self.main_window.screen_view.scene().height())  # Save the bounding box changes
        self.set_sound('sounds/createcock.wav')  # Set and play a sound
        self.sound_player.play()

    def _remove_current_bbox_items(self):
        self.scene().removeItem(self.current_bbox)  # Remove the current bounding box item
        self.scene().removeItem(self.current_bbox.class_name_item)  # Remove the associated class name item

    def _reset_after_bbox_move(self):
        self.moving_view = False  # Exit moving view mode
        self.moving_bbox = False  # Exit moving bounding box mode
        self.initial_mouse_pos = None  # Clear the initial mouse position
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # Reset the drag mode
        self.clear_selection()  # Clear any selection

    def _reset_selected_bbox(self):
        self.selected_bbox.set_selected(False)  # Deselect the selected bounding box
        self.selected_bbox = None  # Clear the selected bounding box
        self.clear_selection()  # Clear any selection

    def clear_selection(self):
        if self.selected_bbox and self.selected_bbox.boundingRect().isValid():
            self.selected_bbox.set_selected(False)  # Deselect the selected bounding box
        self.selected_bbox = None  # Clear the selected bounding box
        self.viewport().setCursor(Qt.ArrowCursor)  # Set the cursor to arrow

    def itemAt(self, pos):

        return None

    def toggle_crosshair(self, state):
        self.show_crosshair = state == Qt.Checked  # Set the show_crosshair flag based on the checkbox state
        self.viewport().update()  # Trigger a viewport update to show or hide the crosshair


class BoundingBoxDrawer(QGraphicsRectItem):
    MIN_SIZE = 6
    MAX_SIZE = 100

    def __init__(self, x, y, width, height, main_window, class_id=None, confidence=None, unique_id=None):
        super().__init__(x, y, width, height)  # Initialize the base QGraphicsRectItem
        self.unique_id = unique_id  # Unique identifier for the bounding box
        width = min(width, main_window.image.width() - x)
        height = min(height, main_window.image.height() - y)
        super().__init__(x, y, width, height)  # Reinitialize the base QGraphicsRectItem
        self.main_window = main_window  # Reference to the main window
        self.dragStartPos = None  # Initialize the drag start position
        self.final_pos = None  # Initialize the final position (used for moving)
        self.class_id = 0 if class_id is None else class_id  # Class ID associated with the bounding box
        # Define confidence before calling update_class_name_item
        self.confidence = confidence  # Confidence level associated with the bounding box
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)  # Set item flags
        self.bbox = None  # Initialize the bounding box representation
        num_classes = self.main_window.classes_dropdown.count()
        color = self.get_color(self.class_id, num_classes)  # Get the bounding box color
        self.setPen(QPen(color, 0.9))  # Set the bounding box outline color and width
        self.setZValue(0.5)  # Set the Z-value for rendering order
        self.num_classes = self.main_window.classes_dropdown.count()  # Number of classes
        self.flash_timer = QTimer()  # Timer for flashing effect
        self.flash_timer.timeout.connect(self.toggle_flash_color)  # Connect timer to toggle_flash_color method
        self.scroll_timer = QTimer()  # Timer for stopping flashing
        self.scroll_timer.setSingleShot(True)
        self.scroll_timer.timeout.connect(self.stop_flashing)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)  # Disable item movement
        self.x = self.rect().x()  # Get the x-coordinate of the bounding box
        self.y = self.rect().y()  # Get the y-coordinate of the bounding box
        self.width = self.rect().width()  # Get the width of the bounding box
        self.height = self.rect().height()  # Get the height of the bounding box
        self.create_bbox_on_click = True  # Flag to determine whether to create a new bounding box on click
        self.shade_value = self.main_window.shade_slider.value()  # Get the shade value from the main window
        self.setFlag(QGraphicsItem.ItemIsMovable, True)  # Enable item movement
        self.orig_color = self.get_color(self.class_id, self.num_classes)  # Get the original bounding box color
        self.setAcceptHoverEvents(True)  # Enable hover events
        self.setZValue(1)  # Set the Z-value for rendering order
        self.main_window = main_window  # Reference to the main window
        self.class_id = class_id  # Class ID associated with the bounding box
        self.set_selected(False)  # Initialize the selection state as False
        self.flash_color = False  # Initialize the flash color state as False
        self.x = x  # x-coordinate of the bounding box
        self.y = y  # y-coordinate of the bounding box
        self.width = width  # Width of the bounding box
        self.height = height  # Height of the bounding box
        self.final_rect = self.rect()  # Initialize the final bounding box rectangle
        self.class_name_item = QGraphicsTextItem(self)  # Create a text item for class name
        self.class_name_item.setFlag(QGraphicsItem.ItemIsSelectable, False)  # Disable text item selection
        self.update_class_name_item()  # Update the class name item with the current class name
        self.update_bbox()  # Update the bounding box representation
        self.hover_opacity = 1.0  # Opacity value when hovering (adjust as needed)
        self.normal_opacity = 0.6  # Normal opacity value
        self.flash_color = QColor(0, 0, 0)  # Flash color (black)
        self.setAcceptHoverEvents(True)  # Enable hover events


    def get_confidence(self):
        return self.confidence if self.confidence is not None else 0

    def start_flashing(self):
        self.flash_timer.start(100)

    def flash(self, duration):
        self.start_flashing()
        QTimer.singleShot(duration, self.stop_flashing)

    def stop_flashing(self):
        self.flash_timer.stop()
        original_color = self.get_color(self.class_id, self.num_classes)
        self.setPen(QPen(original_color, 2))
        self.setPen(QPen(self.orig_color, 1))

    def toggle_flash_color(self):
        current_color = self.pen().color().getRgb()
        dark_color = QColor(0, 0, 139).getRgb()  # RGB values for dark blue
        flash_color = self.flash_color.getRgb()  # RGB values for flash color
        if current_color == dark_color:
            self.setPen(QPen(self.flash_color, 2))  # Set the flash color
        else:
            self.setPen(QPen(QColor(*dark_color), 2))  # Set the dark color


    @staticmethod
    def get_color(class_id, num_classes):
        if num_classes == 0:
            logging.error("Number of classes should not be zero.")
            return None  # Return None or some default color

        num_classes = min(num_classes, 100)  # Limit the number of classes to 100 for variety
        hue_step = 360 / num_classes
        hue = (class_id * hue_step) % 360
        saturation = 255
        value = 255
        color = QColor.fromHsv(hue, saturation, value)  # Generate a color based on class ID
        return color

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)

        # Apply shading to the bounding box if the shading checkbox is checked
        if self.main_window.shade_checkbox.isChecked():
            shade_value = self.main_window.shade_slider.value()

            if self.confidence is not None:  # Check if confidence value is set
                # Scale confidence to [0, 255]
                color = self.pen().color()
                mapped_shade_value = int((shade_value / 100) * 255)  # Map 0-100 to 0-255
                shaded_color = QColor(
                    color.red(), color.green(), color.blue(), mapped_shade_value)
            else:
                color = self.pen().color()
                shaded_color = QColor(
                    color.red(), color.green(), color.blue(), shade_value)

            painter.setBrush(shaded_color)
            painter.drawRect(self.rect())

        # Add a padding around the label text
        padding = 1

        # Get the bounding rect of the text and add padding
        label_rect = self.class_name_item.boundingRect()
        label_rect.setWidth(label_rect.width() + padding * 2)
        label_rect.setHeight(label_rect.height() + padding * 2)

        # Move the bounding rect to the correct position (taking into account the padding)
        label_rect.moveTopLeft(self.class_name_item.pos() - QPointF(padding, padding))

        # Apply shading and border to the label only if hide_label_checkbox is not checked
        if not self.main_window.hide_label_checkbox.isChecked():
            label_shade_value = self.main_window.shade_slider.value()  # Get the shade value
            label_color = self.pen().color()  # Get the color of the pen
            mapped_label_shade_value = int(
                (label_shade_value / 100) * 255)  # Map 0-100 to 0-255
            shaded_label_color = QColor(
                label_color.red(), label_color.green(), label_color.blue(), mapped_label_shade_value)  # Apply shading

            # Drawing a semi-transparent background for the label
            painter.setBrush(QColor(0, 0, 0, 127))  # Adjust color and transparency as needed
            label_rect = self.class_name_item.boundingRect()
            label_rect.moveTopLeft(self.class_name_item.pos())
            painter.drawRect(label_rect)


    def hide_labels(self):
        for child in self.childItems():
            if isinstance(child, QGraphicsTextItem):
                child.setVisible(False)

    def update_class_name_item(self):
        if not self.main_window:
            return
        full_text = self.get_formatted_class_text()
        self.update_class_name_item_text(full_text)
        self.update_class_color_and_position()
        self.update_class_name_item_font()

    def get_formatted_class_text(self):
        class_name = self.main_window.classes_dropdown.itemText(self.class_id)
        if self.confidence:
            confidence_str = f"Conf: {self.confidence*100:.2f}%"
            return f"ID: {self.class_id} | {class_name} | {confidence_str}"
        return f"ID: {self.class_id} | {class_name}"

    def update_class_name_item_text(self, full_text):
        self.class_name_item.setPlainText(full_text)

    def update_class_color_and_position(self):
        # Create and set rectangle for the colored tab
        color_tab = QGraphicsRectItem()
        color_tab.setBrush(QColor(255, 215, 0))
        color_tab.setRect(self.class_name_item.boundingRect())

        # Set position for class_name_item and color_tab
        offset = 14  # adjust as needed
        position_x, position_y = self.rect().x(), self.rect().y() - offset
        self.class_name_item.setPos(position_x, position_y)
        color_tab.setPos(position_x, position_y)

        # Set class_name_item text color
        self.class_name_item.setDefaultTextColor(QColor(255, 255, 255))

    def hoverEnterEvent(self, event):
        # Bring the item forward by setting a high Z-value
        self.setZValue(9999)
        self.setOpacity(self.hover_opacity)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        # Reset the Z-value to its original value
        self.setZValue(1)
        self.setOpacity(self.normal_opacity)
        super().hoverLeaveEvent(event)

    def update_class_name_item_font(self):
        font = self.class_name_item.font()
        font_size_slider_value = self.main_window.font_size_slider.value()
        font_point_size = font_size_slider_value + 1
        font.setPointSize(font_point_size)
        self.class_name_item.setFont(font)

    def set_class_id(self, class_id):
        self.class_id = class_id
        self.update_class_name_item()


    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()

        # Check if the Control key is pressed
        if modifiers == Qt.ControlModifier:
            # Get the current bounding box's size and position
            rect = self.rect()
            delta = 1 if event.delta() > 0 else -1

            # Calculate the new size of the bounding box while ensuring minimum and maximum values
            new_width = min(max(6, rect.width() + delta), self.scene().width() - rect.x())
            new_height = min(max(6, rect.height() + delta), self.scene().height() - rect.y())

            # Calculate the new position of the bounding box while considering the delta
            new_x = max(0, min(rect.x() - delta / 2, self.scene().width() - new_width))
            new_y = max(0, min(rect.y() - delta / 2, self.scene().height() - new_height))

            # Update the bounding box's size and position
            self.setRect(new_x, new_y, new_width, new_height)

            # Save the updated bounding box and trigger flashing
            self.main_window.save_bounding_boxes(
                self.main_window.label_file, self.scene().width(), self.scene().height())
            self.start_flashing()
            self.scroll_timer.start(500)
        else:
            super().wheelEvent(event)

    def update_bbox(self):
        # Update the bounding box's data based on its size and position within the scene
        rect = self.rect()
        if self.scene():
            img_width = self.scene().width()
            img_height = self.scene().height()
            x_center = (rect.x() + rect.width() / 2) / img_width
            y_center = (rect.y() + rect.height() / 2) / img_height
            width = rect.width() / img_width
            height = rect.height() / img_height

            # Create or update the bounding box object
            self.bbox = BoundingBox(
                self.class_id, x_center, y_center, width, height, self.confidence)


    def update_position_and_size(self):
        # Update the bounding box's position and size based on its current data
        self.update_bbox()

        if self.scene() is not None:
            # Calculate the normalized values for the bounding box's position and size
            self.bbox.x_center = (self.x + self.width /
                                  2) / self.scene().width()
            self.bbox.y_center = (self.y + self.height /
                                  2) / self.scene().height()
            self.bbox.width = self.width / self.scene().width()
            self.bbox.height = self.height / self.scene().height()

    def set_selected(self, selected):
        # Set the selected state of the bounding box within the scene
        if self.scene() is None:
            return
        self.setSelected(selected)  # Use setSelected to manage item selection

    def copy_and_create_new_bounding_box(self):
        # Create a copy of the selected bounding box and add a new one to the scene
        if self.isSelected():
            self.setPen(QPen(QColor(0, 255, 0), 1))  # Set the pen color for feedback
            copied_bbox = self
            new_bbox = BoundingBoxDrawer(copied_bbox.rect().x() + 10, copied_bbox.rect().y() + 10,
                                         copied_bbox.rect().width(), copied_bbox.rect().height(),
                                         self.main_window, copied_bbox.class_id)
            self.scene().addItem(new_bbox)
            new_bbox.setSelected(True)  # Select the new bounding box
            QTimer.singleShot(500, self.reset_color)  # Reset color after a delay

    def reset_color(self):
        original_color = self.get_color(self.class_id, self.num_classes)
        self.setPen(QPen(original_color, 2))

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and not event.modifiers() & Qt.ControlModifier:
            # Enable moving the bounding box and update appearance
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

    def mouseMoveEvent(self, event):
        """
        Handle the mouse move event for the bounding box.

        This method allows the user to move the bounding box within the scene bounds.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse move event.

        Notes:
            - If the `dragStartPos` is not set (i.e., it's `None`), no action is taken.
            - Calculate the new position (`newPos`) by subtracting the initial drag start position from the current mouse position.
            - Create a new rectangle (`newRect`) with the calculated position and the same size as the bounding box.
            - Ensure that the new rectangle stays within the scene bounds:
                - `maxLeft`, `maxTop`, `maxRight`, and `maxBottom` define the maximum allowed positions based on the scene size.
                - The `newRect` is moved within these limits to prevent it from going outside the scene.
            - Update the bounding box's position and size to match the adjusted `newRect`.

        Behavior:
            - If the bounding box is selected, it is brought to the front with a higher Z-value.
            - Other bounding boxes in the scene are set to a lower Z-value to ensure proper layering.

        """
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

        # Bring the selected item to the front with a higher Z-value
        if self.isSelected:
            self.setZValue(1.0)
            for item in self.scene().items():
                if item != self and isinstance(item, BoundingBoxDrawer):
                    item.setZValue(0.5)
        else:
            self.setZValue(0.5)


    def update_class_name_position(self):
        offset = 14  # Adjust this value as needed
        new_label_pos = QPointF(self.rect().x(), self.rect().y() - offset)
        self.class_name_item.setPos(new_label_pos)
        self.class_name_item.update()  # Update the graphics


    def mouseReleaseEvent(self, event):
        """
        Handle the mouse release event for the bounding box.

        This method performs actions when the mouse button is released, such as finalizing the bounding box's position
        and size after a drag operation.

        Notes:
            - If `dragStartPos` is not `None`, the bounding box's final position and size are set, and drag-related flags
            are cleared.
            - If the left mouse button is released with the Control key held down, the bounding box is deselected, made
            movable, selectable, and focusable again, and its stacking order is reset to the default value.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse release event.

        Returns:
            None.

        """
        if self.dragStartPos is not None:
            self.final_rect = self.rect()
            self.dragStartPos = None
            self.update_position_and_size()
            self.setFlag(QGraphicsItem.ItemIsMovable, False)
            original_color = self.get_color(self.class_id, self.num_classes)
            self.setPen(QPen(original_color, 2))

        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            self.set_selected(False)
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.setFlag(QGraphicsItem.ItemIsFocusable, True)
            self.setZValue(0.5)  # Ensure the released item goes back to the default stacking order
        super().mouseReleaseEvent(event)


# this class connects the labels to the main window


class BoundingBox:
    def __init__(self, class_id, x_center, y_center, width, height, confidence=None):
        self.class_id = class_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.confidence = confidence  # confidence score of the bounding box
        self.initial_mouse_pos = None

    @classmethod
    def from_rect(cls, rect, img_width, img_height, class_id, confidence=None):
        """
        Create a BoundingBox object from a QRectF object.

        Args:
            cls (BoundingBox): The BoundingBox class.
            rect (QRectF): The QRectF object representing the bounding box.
            img_width (int): The width of the image containing the bounding box.
            img_height (int): The height of the image containing the bounding box.
            class_id (int): The class ID associated with the bounding box.
            confidence (float, optional): The confidence score of the bounding box. Default is None.

        Returns:
            BoundingBox: A BoundingBox object.
        """
        x_center = (rect.x() + rect.width() / 2) / img_width
        y_center = (rect.y() + rect.height() / 2) / img_height
        width = rect.width() / img_width
        height = rect.height() / img_height
        return cls(class_id, x_center, y_center, width, height, confidence)

    def to_rect(self, img_width, img_height):
        """
        Convert the BoundingBox object to a QRectF object.

        Args:
            img_width (int): The width of the image containing the bounding box.
            img_height (int): The height of the image containing the bounding box.

        Returns:
            QRectF: A QRectF object representing the bounding box.
        """
        x = self.x_center * img_width - self.width * img_width / 2
        y = self.y_center * img_height - self.height * img_height / 2
        width = self.width * img_width
        height = self.height * img_height
        return QRectF(x, y, width, height)

    @staticmethod
    def from_str(bbox_str):
        """
        Create a BoundingBox object from a string representation.

        Args:
            bbox_str (str): A string containing space-separated values representing the bounding box.

        Returns:
            BoundingBox: A BoundingBox object created from the string representation.
        """
        values = list(map(float, bbox_str.strip().split()))
        if len(values) >= 5:
            class_id, x_center, y_center, width, height = values[:5]
            # Add check for extra confidence value
            confidence = values[5] if len(values) > 5 else None
            return BoundingBox(int(class_id), x_center, y_center, width, height, confidence)
        else:
            return None

    def to_str(self):
        """
        Convert the BoundingBox object to a string representation.

        Returns:
            str: A string representation of the bounding box.
        """
        # Add confidence to string representation
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
        layout.addWidget(QtWidgets.QLabel("autoLabel:"))  # Add label for autoLabel setting
        layout.addWidget(self.autoLabelInput)  # Add autoLabel input field

        saveButton = QtWidgets.QPushButton("Save")  # Create a "Save" button
        saveButton.clicked.connect(self.saveSettings)  # Connect button click event to saveSettings method
        layout.addWidget(saveButton)  # Add the "Save" button to the layout

        self.setLayout(layout)  # Set the final layout for the settings dialog

    def saveSettings(self):
        self.parent().settings['nextButton'] = self.nextButtonInput.text()  # Save Next Button setting
        self.parent().settings['previousButton'] = self.previousButtonInput.text()  # Save Previous Button setting
        self.parent().settings['deleteButton'] = self.deleteButtonInput.text()  # Save Delete Button setting
        self.parent().settings['autoLabel'] = self.autoLabelInput.text()  # Save autoLabel setting

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
        self.extract_progress = QProgressBar()  # Progress bar for frame extraction

    def run(self):
        self.extract_progress.reset()
        self.extract_progress.setMinimum(0)
        self.extract_progress.setMaximum(len(self.videos))
        for index, video in enumerate(self.videos):
            self.process_video(video)
            self.extract_progress.setValue(index + 1)
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
                cv2.imwrite(frame_output_path, frame)

            # Update the frame count and progress
            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            self.progress_updated.emit(progress)

            # Check if processing should be stopped
            if self.stop_processing:
                break

        # Release the video file
        video.release()


class WorkerSignals(QObject):
    update_stats = pyqtSignal(str)

class TrainingThread(QThread):

    def __init__(self, command, write_log, next_map):
        super(TrainingThread, self).__init__()
        self.command = command
        self.signals = WorkerSignals()
        self.write_log = write_log
        self.process = None
        self.map_counter = 0
        self.should_stop = False
        self.signals = WorkerSignals()
        self.next_map = next_map


    def stop(self):
        if self.process:
            # Send the CTRL+C signal to the process group on Windows
            ctypes.windll.kernel32.GenerateConsoleCtrlEvent(
                signal.CTRL_C_EVENT, 0)

    def _reader(self, pipe, signal):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = f"training_log_{timestamp}.txt"

        if self.write_log:
            with open(log_file, "a") as log:
                for line in pipe:
                    print(f"TrainingThread output: {line.strip()}")
                    signal.emit(line)
                    log.write(line)
                    self.update_stats(line)
        else:
            for line in pipe:
                print(f"TrainingThread output: {line.strip()}")
                signal.emit(line)
                self.update_stats(line)


    def run(self):
        """
        Execute the training command in a separate thread.

        This method launches the training process, reads its output streams (stdout and stderr),
        and updates the GUI with the process output. It also sends a newline character to start
        the training automatically.

        Note:
            This method should be run in a separate thread to avoid blocking the main GUI thread.

        """
        print(f"Executing command in TrainingThread: {self.command}")
        self.process = subprocess.Popen(self.command, shell=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=1, text=True)

        # Send a newline character to automatically start the training
        self.process.stdin.write("\n")
        self.process.stdin.flush()

        # Start separate threads to read stdout and stderr
        stdout_thread = threading.Thread(target=self._reader, args=(
            self.process.stdout, self.signals.update_stats))
        stderr_thread = threading.Thread(target=self._reader, args=(
            self.process.stderr, self.signals.update_stats))

        stdout_thread.start()
        stderr_thread.start()

        stdout_thread.join()
        stderr_thread.join()

        # Wait for the process to terminate and read its output
        stdout, stderr = self.process.communicate()

        # Print the output, if necessary
        print("Training process stdout:\n", stdout)
        print("Training process stderr:\n", stderr)

        self.process.wait()


    def update_stats(self, output_line):
        """
        Update training statistics and check for conditions to stop training.

        This method parses the output line from the training process and extracts relevant
        statistics, including mean average precision (mAP) and loss. It also checks if
        the training should be stopped based on mAP and iteration conditions.

        Args:
            output_line (str): A line of output from the training process.

        """
        print("update_stats called with:", output_line)
        log_lines = []
        map_pattern = re.compile(
            r"mean average precision \(mAP@0\.50\) = (\d+(\.\d+)?)")
        map_match = map_pattern.search(output_line)
        if map_match:
            map_value = float(map_match.group(1)) * \
                100  # Convert to percentage
            log_lines.append(f"mAP: {map_value:.2f}%")
            if map_value >= 98:
                self.map_counter += 1
            else:
                self.map_counter = 0

            if self.map_counter >= 10:
                self.stop()

        loss_iteration_pattern = re.compile(
            r"(\d+): (.*?), (.*?) avg loss, .*?, .*?, .*?, (.*?) hours left")

        loss_iteration_match = loss_iteration_pattern.search(output_line)
        if loss_iteration_match:
            current_iteration = int(loss_iteration_match.group(1).strip())
            total_loss_value = float(loss_iteration_match.group(3).strip())
            log_lines.append(
                f"Iteration: {current_iteration}, Total Loss: {total_loss_value:.2f}")
            self.signals.update_stats.emit(output_line)

        if self.should_stop:
            self.process.terminate()
            self.process.wait()

        # Print summary of log when training is stopped
        if self.process.returncode is not None:
            print("Training stopped with return code:", self.process.returncode)



# Custom exception class for handling invalid annotations
class InvalidAnnotationError(Exception):
    pass

# Custom exception class for handling invalid images
class InvalidImageError(Exception):
    pass

# Class to manage and process scan annotations
class ScanAnnotations:
    def __init__(self, parent):
        # Initialize the ScanAnnotations object with a parent object
        self.parent = parent
        self.valid_classes = []  # List to store valid classes for annotations
        self.base_directory = ""  # Base directory for scan data
        self.total_images = 0  # Total number of images processed
        self.total_labels = 0  # Total number of labels generated
        self.blank_labels = 0  # Number of labels with no annotations
        self.bad_labels = 0  # Number of labels with errors
        self.bad_images = 0  # Number of images with errors
        self.review_folder = "review"  # Folder for reviewing problematic data
        self.metadata_queue = queue.Queue()  # Queue for processing metadata removal
        self.metadata_thread = threading.Thread(target=self.metadata_removal_thread)
        self.metadata_thread.daemon = True
        self.metadata_thread.start()

    # Function to remove metadata from an image
    def remove_metadata(self, image_path, output_path):
        # Add your metadata removal logic here (e.g., using OpenCV)
        image = cv2.imread(image_path)
        cv2.imwrite(output_path, image)  # Save the image without metadata

    # Metadata removal thread for concurrent processing
    def metadata_removal_thread(self):
        while True:
            try:
                image_path, output_path = self.metadata_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Check if the image file exists before processing
            if os.path.exists(image_path):
                self.remove_metadata(image_path, output_path)
            else:
                print(f"File not found: {image_path}")

            self.metadata_queue.task_done()

    # Function to add images to the metadata removal queue
    def process_image_for_metadata_removal(self, image_path, output_path):
        self.metadata_queue.put((image_path, output_path))

    # Function to remove orphan JSON files
    def remove_orphan_json_files(self):
        # Get the list of all .json files in the base directory
        json_files = glob.glob(os.path.join(self.base_directory, "*.json"))

        # Iterate over all .json files
        for json_file in json_files:
            # Get the base name of the .json file (without the extension)
            base_name = os.path.splitext(os.path.basename(json_file))[0]

            # Check if there exists an image file with the same base name
            image_exists = any(
                os.path.exists(os.path.join(self.base_directory, f"{base_name}{ext}"))
                for ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
            )

            # If no corresponding image file exists, remove the .json file
            if not image_exists:
                os.remove(json_file)
                print(f"Removed orphan JSON file: {json_file}")

    def is_valid_image(self, file_path):
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except (IOError, SyntaxError):
            return False

    def import_classes(self, parent):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            parent, "Import Classes", "", "Classes Files (classes.txt *.names)", options=options)
        if file_name:
            self.base_directory = os.path.dirname(file_name)
            with open(file_name, "r") as f:
                class_names = f.readlines()
                self.valid_classes = list(range(len(class_names)))
            os.makedirs(os.path.join(self.base_directory, self.review_folder), exist_ok=True)

    def check_annotation_file(self, file_path, image_folder):
        # Initialize variables to track issues and other information
        issues = []
        lines_to_keep = []
        should_move = False
        count_small_bboxes = 0
        height_zero = False
        class_index = -1
        img_file = None

        # Find the image file associated with the annotation
        for ext in (".jpg", ".jpeg", ".png"):
            img_file = os.path.basename(file_path).replace(".txt", ext)
            if os.path.exists(os.path.join(image_folder, img_file)):
                break

        # Check if there's no associated image file
        if img_file is None:
            return [f"Warning: No image file found for '{os.path.basename(file_path)}'"], lines_to_keep, True

        img_width, img_height = None, None

        # Check if the image file exists and retrieve its dimensions
        if img_file is not None:
            img_path = os.path.join(image_folder, img_file)
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except FileNotFoundError as e:
                    return [f"Warning: Error opening image file: {e}"], lines_to_keep, True
            else:
                return [f"Warning: No such file or directory: '{img_path}'"], lines_to_keep, True

        # Read and process each line in the annotation file
        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                line_issues = []

                # Check if there are exactly 5 tokens in each line
                if len(tokens) != 5:
                    line_issues.append("Incorrect number of tokens")
                else:
                    try:
                        class_index = int(tokens[0])
                    except ValueError:
                        line_issues.append("Invalid class index format")
                        should_move = True

                    # Check if the class index is valid
                    if class_index not in self.valid_classes:
                        line_issues.append("Invalid object class")
                        should_move = True

                    try:
                        x, y, width, height = map(float, tokens[1:])
                        # Check if bounding box coordinates are within [0, 1] range
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                            line_issues.append("Bounding box coordinates are not within the range of [0, 1]")
                            should_move = True
                        # Check if width and height are positive values
                        if not (width > 0 and height > 0):
                            line_issues.append("Width and height should be positive values")
                            should_move = True
                    except ValueError:
                        line_issues.append("Invalid bounding box values")
                        should_move = True

                if not line_issues and img_width is not None and img_height is not None:
                    abs_width = width * img_width
                    abs_height = height * img_height

                    # Check if bounding box takes up a large percentage of the frame
                    if (width * height) >= self.max_label.value() / 100:
                        print(f"Processing file: {file_path}")
                        print(f"Problematic line: {line.strip()}")

                        line_issues.append("Bounding box takes up a large percentage of the frame")
                        should_move = True

                    # Check if height is too small
                    if abs_height < self.box_size.value() / 100 * img_height:
                        height_zero = True
                        line_issues.append("Height is too small")
                    # Check if bounding box is small (6x6 or smaller)
                    elif abs_width <= 6 and abs_height <= 6:
                        line_issues.append("Small bounding box (6x6 or smaller)")
                        count_small_bboxes += 1
                    else:
                        lines_to_keep.append(line)

                    # Write the lines to keep back to the annotation file
                    if len(lines_to_keep) > 0:
                        with open(file_path, 'w') as f_out:
                            f_out.writelines(lines_to_keep)
                    else:
                        if line_issues:
                            issues.append((line.strip(), line_issues))
                            if not should_move:
                                should_move = True

        # Check if height is zero, indicating an issue
        if height_zero:
            should_move = True

        return issues, lines_to_keep, should_move

    def move_files(self, annotation_folder, review_folder, file_name):
        # Convert folder paths to pathlib objects
        annotation_folder = Path(annotation_folder)
        review_folder = Path(review_folder)

        # Create a path to the .txt file
        txt_path = annotation_folder / file_name

        # Check if the .txt file exists
        if not txt_path.exists():
            print(f"Error: {file_name} not found in {annotation_folder}.")
            return  # Exit the function if the .txt file is not found

        try:
            # Move the .txt file to the review folder
            shutil.move(str(txt_path), str(review_folder / file_name))
        except FileNotFoundError as e:
            print(f"Error: Unable to move {file_name} due to: {e}")
        except Exception as e:
            # Catch any other exceptions
            print(f"Error: An unexpected error occurred: {e}")

        # Look for and move the corresponding image file
        for ext in (".jpg", ".jpeg", ".png"):
            image_file = file_name.replace(".txt", ext)
            image_path = annotation_folder / image_file

            if image_path.exists():
                try:
                    # Move the image file to the review folder
                    shutil.move(str(image_path), str(review_folder / image_file))
                    print(f"{image_file} was moved because...")
                except Exception as e:  # Catch any other exceptions
                    print(f"Error: Unable to move {image_file} due to: {e}")
                break
        else:  # This will execute if no image file was found
            print(f"No corresponding image file found for {file_name}.")



    def create_blanks_folder(self):
        # Define the base directory for the 'blanks' folder
        blanks_folder_base = os.path.join(self.base_directory, "blanks")

        # Initialize a counter for folder naming
        counter = 1

        # Generate a unique folder name by appending '_n' where n is a number
        blanks_folder = blanks_folder_base
        while os.path.exists(blanks_folder):
            blanks_folder = f"{blanks_folder_base}_{counter}"
            counter += 1

        # Create the 'blanks' folder if it doesn't exist
        os.makedirs(blanks_folder, exist_ok=True)

        # Return the path to the newly created 'blanks' folder
        return blanks_folder

    def move_blanks(self, blanks_folder, annotation_folder, image_folder, file_name):

        # Check if the annotation file exists in the annotation folder
        file_path = os.path.join(annotation_folder, file_name)

        if not os.path.isfile(file_path):
            print(f"File {file_name} not found in {annotation_folder}.")
            return

        try:
            # Attempt to open the annotation file to ensure it can be accessed without issues
            with open(file_path, 'r') as f:
                # Just checking if we can open the file without any issues.
                pass
        except IOError:
            print(f"File {file_name} is being used by another process. Cannot move it.")
            return

        # Move the annotation file to the 'blanks' folder
        shutil.move(file_path, os.path.join(blanks_folder, file_name))

        # Find the corresponding image file (e.g., .jpg, .jpeg, .png)
        img_file = None
        for ext in (".jpg", ".jpeg", ".png"):
            img_file = file_name.replace(".txt", ext)
            if os.path.exists(os.path.join(image_folder, img_file)):
                break

        # Construct the full path to the image file
        image_path = os.path.join(image_folder, img_file)

        # Check if the image file exists
        if os.path.isfile(image_path):
            # Move the image file to the 'blanks' folder
            shutil.move(image_path, os.path.join(blanks_folder, img_file))
        else:
            print(f"Image file {img_file} not found in {image_folder}.")


    def handle_blanks_after_review(self, annotation_folder, image_folder):
        """
        Ask the user if they want to move empty text files and their corresponding images to a 'blanks' folder.
        If confirmed, empty text files are moved, and the user is prompted to move files by label size to subfolders.

        Args:
            annotation_folder (str): The directory containing annotation files.
            image_folder (str): The directory containing image files.

        Returns:
            None
        """
        # Ask the user if they want to move empty text files and their corresponding images to a 'blanks' folder.
        reply = QMessageBox.question(self.parent, "Move Blanks",
                                    "Do you want to move empty text files and their corresponding images to a 'blanks' folder?",
                                    QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Create a 'blanks' folder to store empty text files and images
            blanks_folder = self.create_blanks_folder()

            # Get a list of all .txt files in the annotation folder
            txt_files = [file for file in os.listdir(annotation_folder) if file.endswith(".txt")]

            # Initialize a counter for moved files
            moved_count = 0

            # Iterate through each .txt file and check if it is empty
            for file_name in txt_files:
                file_path = os.path.join(annotation_folder, file_name)
                is_blank = False

                with open(file_path, 'r') as f:
                    is_blank = not f.read().strip()  # Check if the .txt file is empty

                if is_blank:
                    # Move the empty text file and its corresponding image (if exists) to the 'blanks' folder
                    self.move_blanks(blanks_folder, annotation_folder, image_folder, file_name)
                    moved_count += 1

            # Display a message indicating the number of blanks moved
            QMessageBox.information(
                self.parent, 'Information', f'{moved_count} blanks moved successfully!')
        else:
            # Display a message indicating that blanks were not moved
            QMessageBox.information(
                self.parent, 'Information', 'Blanks not moved.')

        # Ask if the user wants to move the files by label size
        txt_files = [file for file in os.listdir(annotation_folder) if file.endswith(".txt")]

        reply2 = QMessageBox.question(self.parent, "Move by Label Size",
                                    "Do you want to move small, medium, and large files to subfolders based on their label size?",
                                    QMessageBox.Yes | QMessageBox.No)

        if reply2 == QMessageBox.Yes:
            # Define subfolder names for different label sizes
            subfolder_names = {"small": "small", "medium": "med", "large": "large"}

            # Create subfolders for each label size in the annotation folder
            for size in subfolder_names:
                subfolder_name = subfolder_names[size]
                subfolder_path = os.path.join(annotation_folder, subfolder_name)
                os.makedirs(subfolder_path, exist_ok=True)

            # Iterate through each .txt file, categorize by label size, and move to the respective subfolder
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
                    subfolder_name = subfolder_names[label_size]
                    subfolder_path = os.path.join(annotation_folder, subfolder_name)

                    # Move the .txt file to the corresponding subfolder
                    shutil.move(file_path, os.path.join(subfolder_path, file_name))

                    # Move the corresponding image file (if exists)
                    img_file = file_name.replace(".txt", ".jpg")
                    img_path = os.path.join(image_folder, img_file)
                    if os.path.isfile(img_path):
                        shutil.move(img_path, os.path.join(subfolder_path, img_file))
                    else:
                        print(f"Image file {img_file} not found in {image_folder}.")

            # Display a message indicating that files were moved by label size
            QMessageBox.information(
                self.parent, 'Information', 'Files moved by label size successfully!')
        else:
            # Display a message indicating that files were not moved by label size
            QMessageBox.information(
                self.parent, 'Information', 'Files not moved by label size.')


    # Categorize label size based on width and height
    def categorize_label_size(self, width, height):
        size = width * height
        if size <= 0.02:
            label_size = "small"
        elif size <= 0.07:
            label_size = "medium"
        else:
            label_size = "large"

        print(f"Label size: {label_size}")
        return label_size

    def process_files(self, file_name, annotation_folder, image_folder, review_folder, statistics, label_sizes):
        """
        Process an annotation file and its corresponding image file (if exists).
        Check for issues in the annotation, categorize label sizes, and move problematic files to the review folder.

        Args:
            file_name (str): The name of the annotation file to be processed.
            annotation_folder (str): The directory containing annotation files.
            image_folder (str): The directory containing image files.
            review_folder (str): The directory where problematic files are moved.
            statistics (dict): A dictionary to store statistics and issues.
            label_sizes (dict): A dictionary to store label size categories and counts.

        Returns:
            None
        """
        try:
            print(f"Processing file: {file_name}")
            with open(os.path.join(annotation_folder, file_name)) as f:
                annotation_data = f.read()
        except IOError:
            print(f"{file_name} does not exist.")
            return
        file_path = os.path.join(annotation_folder, file_name)
        issues, lines_to_keep, should_move = self.check_annotation_file(
            file_path, image_folder)

        # Count total labels and blank labels
        if lines_to_keep:
            self.total_labels += len(lines_to_keep)
        else:
            self.blank_labels += 1

        # Increment total_images count
        self.total_images += 1

        # Categorize label sizes
        for line in lines_to_keep:
            tokens = line.strip().split()
            _, x, y, width, height = map(float, tokens)
            label_size = self.categorize_label_size(width, height)
            label_sizes[label_size] += 1

        if issues:
            if should_move:
                self.bad_labels += 1
                self.bad_images += 1
                statistics[file_name] = issues
                print(f"Moving {file_name}...")
                self.move_files(annotation_folder, review_folder, file_name)
                print(f"{file_name} moved.")
            else:
                statistics[file_name] = issues

        # Add the following code to move problematic files to the review folder
        if should_move:
            image_path = os.path.join(
                image_folder, file_name.replace(".txt", ".jpg"))
            if os.path.exists(image_path):
                shutil.move(image_path, os.path.join(
                    review_folder, os.path.basename(image_path)))
            txt_path = os.path.join(annotation_folder, file_name)
            if os.path.exists(txt_path):
                shutil.move(txt_path, os.path.join(
                    review_folder, os.path.basename(txt_path)))

    def scan_annotations(self, progress_bar):
        """
        Scan and process annotation files in the specified annotation and image folders.
        Move problematic files to the review folder, categorize label sizes, and save statistics.

        Args:
            progress_bar: A progress bar widget to update the scanning progress.

        Returns:
            None
        """
        if not self.valid_classes:
            print("No classes.txt or .names file found. Please import one first.")
            return

        if not self.base_directory:
            print("No base directory found. Please import a dataset first.")
            return

        # Initialize folders and data structures
        annotation_folder = self.base_directory
        image_folder = self.base_directory
        review_folder = os.path.join(self.base_directory, "review")
        os.makedirs(review_folder, exist_ok=True)

        txt_files_set = {file for file in os.listdir(annotation_folder) if file.endswith(
            ".txt") and file != "classes.txt" and not file.endswith(".names")}
        img_files_set = {file for file in os.listdir(
            image_folder) if file.lower().endswith((".jpg", ".jpeg", ".png"))}

        statistics = {}
        label_sizes = {"small": 0, "medium": 0, "large": 0}

        txt_files = [file for file in os.listdir(annotation_folder) if file.endswith(
            ".txt") and file != "classes.txt" and not file.endswith(".names")]

        num_cores = os.cpu_count()
        max_workers = num_cores - 1 if num_cores > 1 else 1

        # First loop to process annotation files
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for file_name in txt_files:
                executor.submit(self.process_files, file_name, annotation_folder,
                                image_folder, review_folder, statistics, label_sizes)

        # Call the method to remove orphan .json files
        self.remove_orphan_json_files()

        # Metadata removal loop
        for img_file in img_files_set:
            image_path = os.path.join(image_folder, img_file)
            txt_file = img_file.replace(".jpg", ".txt")
            txt_file_path = os.path.join(annotation_folder, txt_file)

            # Check if the .txt file exists for the image
            if os.path.exists(txt_file_path):
                # Add the image for metadata removal
                metadata_output_path = os.path.join(
                    review_folder, img_file)  # Change this path as needed
                self.process_image_for_metadata_removal(
                    image_path, metadata_output_path)

        # Handling files without corresponding annotations
        txt_files_without_img = txt_files_set - \
                    {os.path.splitext(img_file)[0] +
                    ".txt" for img_file in img_files_set}
        img_files_without_txt = img_files_set - \
                    {file.replace(".txt", os.path.splitext(
                        file)[1]) for file in txt_files_set}

        annotation_folder_path = Path(annotation_folder)
        review_folder_path = Path(review_folder)

        for file_name in txt_files_without_img:
            src_path = annotation_folder_path / file_name
            dst_path = review_folder_path / file_name
            print(
                f"{file_name} does not have a corresponding image file. Moving it to the 'review' folder.")
            try:
                shutil.move(str(src_path), str(dst_path))
            except FileNotFoundError as e:
                print(f"Error: Unable to move {file_name} due to: {e}")
            except Exception as e:
                # Catch any other exceptions
                print(f"Error: An unexpected error occurred: {e}")

        for img_file in img_files_without_txt:
            txt_file_path = os.path.join(
                annotation_folder, img_file.rsplit(".", 1)[0] + ".txt")
            if not os.path.exists(txt_file_path):  # Check if the .txt file exists
                print(
                    f"{img_file} does not have a corresponding annotation file. Moving it to the 'review' folder.")
                shutil.move(os.path.join(image_folder, img_file),
                            os.path.join(review_folder, img_file))
                statistics[img_file] = ["Annotation file does not exist"]

        # Specify a fixed path to save the statistics.json file directly
        save_path = os.path.join(image_folder, 'statistics.json')

        try:
            # Write statistics to JSON file
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
            # Display confirmation message
            QMessageBox.information(
                self.parent, "Information", f"Statistics saved successfully to {save_path}!")
        except (IOError, OSError) as e:
            print(f"An error occurred while saving the JSON file: {e}")
            QMessageBox.critical(
                self.parent, "Error", f"An error occurred while saving the JSON file: {e}")
        except PermissionError:
            print("You do not have permission to write to this file.")
            QMessageBox.critical(
                self.parent, "Permission Error", "You do not have permission to write to this file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            QMessageBox.critical(
                self.parent, "Error", f"An unexpected error occurred: {e}")

        # Call function to handle blanks after review
        self.handle_blanks_after_review(annotation_folder, image_folder)



class DownloadThread(QThread):
    update_status = pyqtSignal(int, str)  # Signal for status updates
    update_progress = pyqtSignal(int, int)  # Signal for progress updates

    def __init__(self, url, row, directory):
        QThread.__init__(self)
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
        try:
            yt = YouTube(self.url, on_progress_callback=self.emit_progress)
            stream = yt.streams.get_highest_resolution()

            self.update_status.emit(self.row, 'Downloading...')

            # Clean and format the filename
            title = self.clean_filename(yt.title)
            # Append file extension
            filename = f"{title}.{stream.subtype}"

            stream.download(output_path=self.directory, filename=filename)

            self.update_status.emit(self.row, 'Done')
        except Exception as e:
            print(f"Download failed with error: {e}")
            self.update_status.emit(self.row, f'Failed: {str(e)}')

    def emit_progress(self, stream, chunk, bytes_remaining):
        total_size = stream.filesize
        bytes_downloaded = total_size - bytes_remaining

        # Calculate download progress
        progress = int((bytes_downloaded / total_size) * 100)
        self.update_progress.emit(self.row, progress)



class ImageConverterRunnableSignals(QObject):
    finished = pyqtSignal()


class ImageConverterRunnable(QRunnable):
    def __init__(self, directory, file, target_format, target_directory):
        super(ImageConverterRunnable, self).__init__()
        self.signals = ImageConverterRunnableSignals()
        self.directory = directory
        self.file = file
        self.target_format = target_format
        self.target_directory = target_directory

    def run(self):
        try:
            # Use QImageReader and QImageWriter for the conversion
            reader = QImageReader(os.path.join(self.directory, self.file))
            image = reader.read()
            if image.isNull():
                print(f"Could not read {self.file}")
                raise Exception(f"Could not read {self.file}")

            target_file = os.path.join(
                self.target_directory, f"{os.path.splitext(self.file)[0]}.{self.target_format.lower()}")
            writer = QImageWriter(target_file, self.target_format.encode())
            if not writer.write(image):
                print(f"Could not write {self.file}")
                raise Exception(f"Could not write {self.file}")

            # If there's a .txt file with the same base name, copy it to the target directory
            txt_file = os.path.join(
                self.directory, f"{os.path.splitext(self.file)[0]}.txt")
            if os.path.isfile(txt_file):
                shutil.copy(txt_file, self.target_directory)
        finally:
            self.signals.finished.emit()


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

    def run(self):
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
        Process a single image using YOLO annotations.

        :param image_path: Path to the image file.
        :param yolo_annotation_path: Path to the YOLO annotation file.
        :param save_folder_path: Path to the folder where processed images will be saved.
        :param limit_percentage: The limit percentage for cropping.
        :param image_format: The image format to save (e.g., 'jpg', 'png').
        """
        # Read the image and its YOLO annotations
        img = cv2.imread(image_path)
        original_img = img.copy()
        with open(yolo_annotation_path, 'r') as f:
            yolo_data = f.readlines()

        img_height, img_width, _ = img.shape

        labeled_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # Define a margin as a percentage of the image size
        margin = int(0.010 * min(img_height, img_width))

        # Process each YOLO annotation
        for data in yolo_data:
            data = data.split()
            class_id = data[0]
            x_center, y_center, w, h = [float(x) for x in data[1:]]
            x_min = max(int((x_center - w/2) * img_width) - margin, 0)
            x_max = min(int((x_center + w/2) * img_width) + margin, img_width)
            y_min = max(int((y_center - h/2) * img_height) - margin, 0)
            y_max = min(int((y_center + h/2) * img_height) + margin, img_height)

            labeled_img[y_min:y_max,
                        x_min:x_max] = img[y_min:y_max, x_min:x_max]
            img[y_min:y_max, x_min:x_max] = 0
            mask[y_min:y_max, x_min:x_max] = 255

        # Fill the cropped area with context from the original image
        img = self.fill_cropped_area(original_img, labeled_img, img, mask)

        # Ensure the save folder exists
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save the negative image and its corresponding negative.txt file
        negative_filename = f'{base_filename}_no_labels_negative'
        negative_img = img.copy()
        self.save_image_with_limit(negative_img, save_folder_path,
                                negative_filename, limit_percentage, image_format, labeled_img)


    def save_image_with_limit(self, img, save_folder_path, filename, limit, image_format, labeled_img):
        # Decide whether to save as a negative image based on the limit
        save_negative_image = random.choices([True, False], weights=[limit, 100 - limit], k=1)[0]

        if save_negative_image:
            # Save the image and create an empty text file
            negative_filename = f'{save_folder_path}/blanks_{self.counter}.{image_format}'
            cv2.imwrite(negative_filename, img)
            with open(f'{save_folder_path}/blanks_{self.counter}.txt', 'w'):
                pass
            self.counter += 1  # Increment the counter


    def fill_cropped_area(self, original_img, labeled_img, img, mask):
        # Set inpainting method and parameters
        inpainting_methods = [cv2.INPAINT_TELEA, cv2.INPAINT_NS]
        inpaint_radius_values = [1, 3, 5]

        # Inpaint the image using different methods and parameters
        inpainted_images = []
        for inpainting_method in inpainting_methods:
            for inpaint_radius in inpaint_radius_values:
                inpainted_img = cv2.inpaint(
                    img, mask, inpaint_radius, inpainting_method)
                inpainted_images.append(inpainted_img)

        # Choose the best inpainted image based on SSIM
        best_ssim = -1
        best_inpainted_img = None
        for inpainted_img in inpainted_images:
            ssim = compare_ssim(original_img, inpainted_img, multichannel=True)
            if ssim > best_ssim:
                best_ssim = ssim
                best_inpainted_img = inpainted_img

        return best_inpainted_img


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


# Load the UI file and define the main class
ui_file: Path = Path(__file__).resolve().parent / "UltraDarkFusion_v1.0.0.ui"
with open(ui_file, "r", encoding="utf-8") as file:
    Ui_MainWindow: Type[QtWidgets.QMainWindow]
    QtBaseClass: Type[object]
    Ui_MainWindow, QtBaseClass = uic.loadUiType(file)

# Define the MainWindow class with signals
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    keyboard_interrupt_signal = pyqtSignal()
    progress_signal = pyqtSignal(int)
    clear_dropdown_signal = pyqtSignal()
    add_item_signal = pyqtSignal(str)
    rightClicked = pyqtSignal(QModelIndex)
    leftClicked = pyqtSignal(QModelIndex)
    def __init__(self):
        super().__init__()
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.icons = [
            QIcon('styles/1.png'),
            QIcon('styles/frame2.png'),
            # Add paths to other frames here
        ]

        self.setWindowIcon(self.icons[0])
        self.icon_timer = QTimer(self)
        self.icon_timer.timeout.connect(self.update_icon)
        self.icon_timer.start(500)
        self.current_icon_index = 0
        self.video_processor = GUIVideoProcessor()
        self.weights_files = []
        self.setWindowTitle("UltraDarkFusion")
        self.plot_labels.clicked.connect(self.on_plot_labels_clicked)
        self.next_button.clicked.connect(self.next_frame)
        self.previous_button.clicked.connect(self.previous_frame)
        self.current_file = None
        self.label_file = None
        self.next_timer = QTimer()
        self.next_timer.timeout.connect(self.next_frame)
        self.prev_timer = QTimer()
        self.prev_timer.timeout.connect(self.previous_frame)
        self.next_button.pressed.connect(self.start_next_timer)
        self.next_button.released.connect(self.stop_next_timer)
        self.previous_button.pressed.connect(self.start_prev_timer)
        self.previous_button.released.connect(self.stop_prev_timer)
        self.view_references = []
        self.screen_view = CustomGraphicsView(main_window=self)
        self.screen_view.setBackgroundBrush(QBrush(Qt.black))
        self.setCentralWidget(self.screen_view)
        self.screen_view.setRenderHint(QPainter.Antialiasing)
        self.screen_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_scene = QGraphicsScene(self)
        self.selected_bbox = None

        self.clear_dropdown_signal.connect(self.clear_classes_dropdown)
        self.add_item_signal.connect(self.add_item_to_classes_dropdown)
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self.auto_save_bounding_boxes)
        self.auto_save_timer.start(30000)  # Save every 30 seconds
        self.weights_file = None
        self.cfg_file = None

        self.class_input_field.returnPressed.connect(self.class_input_field_return_pressed)
        self.current_class_id = 0
        self.confidence_threshold_slider.valueChanged.connect(self.slider_value_changed)
        self.nms_threshold_slider.valueChanged.connect(self.slider_value_changed)
        self.auto_label_yolo_button.clicked.connect(self.auto_label_yolo_button_clicked)
        self.classes_dropdown.currentIndexChanged.connect(self.change_class_id)
        self.stop_labeling = False
        self.confidence_threshold_slider.setRange(0, 100)
        self.nms_threshold_slider.setRange(0, 100)
        self.confidence_threshold = self.confidence_threshold_slider.value() / 100
        self.nms_threshold = self.nms_threshold_slider.value() / 100
        self.process_timer = QTimer()
        self.process_timer.setSingleShot(True)
        self.process_timer.timeout.connect(self.process_current_image)
        self.process_timer_interval = 500  # Adjust the delay as needed
        
        self.delete_timer = QTimer(self)
        self.delete_timer.setInterval(100)  # The interval in milliseconds
        self.delete_timer.timeout.connect(self.delete_current_image)       
        self.delete_button.pressed.connect(self.delete_timer.start)
        self.delete_button.released.connect(self.delete_timer.stop)

        self.image_list = []
        self.class_to_id = {}
        self.zoom_mode = False
        self.original_transform = None

        self.settingsButton.clicked.connect(self.openSettingsDialog)
        self.settings = self.loadSettings()
        self.img_index_number.valueChanged.connect(self.img_index_number_changed)
        self.filter_class_input.textChanged.connect(self.filter_class)
        self.List_view.clicked.connect(self.on_list_view_clicked)
        self.filtered_image_files = []
        self.hide_label_checkbox.toggled.connect(self.toggle_label_visibility)
        self.hide_label_checkbox.setChecked(False)
        self.filter_blanks_checkbox.stateChanged.connect(self.handle_filter_blanks_state_change)
        self.image = None
        self.sound_player = QMediaPlayer()
        self.muteCheckBox.stateChanged.connect(self.mute_player)

        self.sound_player.setMuted(True)  # Set mute state to True by default

        self.muteCheckBox.setChecked(True)
        self.preview_button.clicked.connect(self.extract_and_display_data)
        self.id_to_class = {}
        self.image_size.setValue(150)
        self.image_size.valueChanged.connect(self.adjust_image_size)

        self.populate_style_combo_box()
        self.label_dict = {}
        self.image_directory = None
        self.rightClicked.connect(self.handle_right_click)
        self.preview_list.viewport().installEventFilter(self)
        self._last_call = time.time()
        self.auto_label.clicked.connect(self.auto_label_current_image)
        self.plot_counter = 0
        self.bounding_boxes = []
        self.leftClicked.connect(self.handle_left_click)
        self.image_cache = {}
        self.blob_cache = {}
        self.movie = QMovie('styles/darknet3.gif')

        self.image_label_2.setMovie(self.movie)
        self.movie.start()
        self.populateGifCombo()
        self.gif_change.currentIndexChanged.connect(self.onGifChange)
        self.video_upload.clicked.connect(self.on_add_video_clicked)
        self.remove_video_button.clicked.connect(self.on_remove_video_clicked)
        self.custom_frames_checkbox.stateChanged.connect(self.on_custom_frames_toggled)
        self.custom_size_checkbox.stateChanged.connect(self.on_custom_size_checkbox_state_changed)
        self.image_format.currentTextChanged.connect(self.on_image_format_changed)
        self.extract_button.clicked.connect(self.on_extract_clicked)
        self.stop_extract.clicked.connect(self.video_processor.stop)
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
        self.review_off.clicked.connect(self.stop_processing_and_clear)

        self.label_indices = {}

        self.processing = True
        self.current_img_index = 0
        self.cfg_dict = {}
        self.parsed_yaml = {}
        self.filename = ""
        self.file_type = ""

        self.import_button.clicked.connect(self.import_images)
        self.output_button.clicked.connect(self.output_paths)

        self.browse_data.clicked.connect(self.browse_data_clicked)
        self.browse_weights.clicked.connect(self.browse_weights_clicked)
        self.train_button.clicked.connect(self.train_button_clicked)
        self.weights_button.clicked.connect(self.open_weights)       
        self.import_data_button.clicked.connect(self.import_data)
        self.calculate_anchors_button.clicked.connect(self.calculate_anchors)
        self.browse_cfg.clicked.connect(self.browse_cfg_clicked)
        self.cfg_button.clicked.connect(self.open_cfg)


        
        self.model_input.clicked.connect(self.browse_pt_clicked)
        self.data_input.clicked.connect(self.browse_yaml_clicked)        
        self.runs_directory = ''  # Initial empty value for save directory
        self.save_runs_directory.clicked.connect(self.on_save_dir_clicked)
        self.ultralytics.clicked.connect(self.ultralytics_train_clicked)


        self.selected_pytorch_file = None

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
        self.combine_txt_button.clicked.connect(self.on_combine_txt_clicked)
        self.combine_txt_flag = False
        self.process_image_button.clicked.connect(self.on_button_click)
        self.images_import = []
        self.label_files = []
        self.scan_annotations = ScanAnnotations(self)
        self.import_images_button.clicked.connect(self.import_images_triggered)
        self.scan_button.clicked.connect(self.scan_annotations.scan_annotations)
        self.import_classes_button.clicked.connect(lambda: self.scan_annotations.import_classes(self))
        self.crop_button.clicked.connect(self.process_images_triggered)
        self.stop_training.clicked.connect(self.stop_training_button_clicked)
        self.map_counter = 0
        self.subfolder_counter = 1
        self.valid_classes = []
        self.base_directory = None
        self.images_button.clicked.connect(self.load_images_and_labels)
        self.adjust_label.clicked.connect(self.adjust_and_show_message)
        self.image_files = []
        self.image_paths = []
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
        self.progress_signal.connect(self.progress_aug.setValue)
        self.styleComboBox.activated.connect(self.apply_stylesheet)
        self.x_axis.valueChanged.connect(self.update_preview)
        self.y_axis.valueChanged.connect(self.update_preview)
        self.height_pos.valueChanged.connect(self.update_preview)
        self.width_position.valueChanged.connect(self.update_preview)
        self.remove_class.clicked.connect(self.remove_class_button_clicked)
        self.paste_url = self.findChild(QtWidgets.QLineEdit, 'paste_url')
        self.video_download = self.findChild(QtWidgets.QTableWidget, 'video_download')
        self.download_video = self.findChild(QtWidgets.QPushButton, 'download_video')

        self.download_video.clicked.connect(self.download_videos)
        self.remove_video = self.findChild(QtWidgets.QPushButton, 'remove_video')
        self.remove_video.clicked.connect(self.remove_selected_video)
        self.download_threads = []
        self.progress_bars = [
            self.findChild(QtWidgets.QProgressBar, 'video_{}'.format(i + 1)) for i in range(5)
        ]
        self.threadpool = QThreadPool()
        self.image_select.clicked.connect(self.select_directory)
        self.convert_image.clicked.connect(self.convert_images)

        self.model = None
       
        self.class_names = []
        self.class_colors = []
        self.keyBuffer = ""
        self.keyTime = time.time()
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.processKeyPresses)
        self.new_class_input = False
        self.outline_Checkbox.clicked.connect(self.checkbox_clicked)
        self.slider_min_value = 0
        self.slider_max_value = 255
        self.anti_slider.valueChanged.connect(self.slider_value_changed2)
        self.grayscale_Checkbox.stateChanged.connect(self.display_image_checkbox_toggled)
        self.bounding_boxes = {}
        self.select_images.clicked.connect(self.select_images_function)
        self.convert_class.clicked.connect(self.convert_class_function)
        self.move_all_button.clicked.connect(self.on_move_all_clicked)
        self.clear_all_button.clicked.connect(self.on_clear_all_clicked)
        self.call_stats_button.clicked.connect(self.display_stats)
        self.hide_labels = False
        self.current_image = None
        self.capture = None
        self.is_camera_mode = False
        self.current_file_name = None
        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.display_camera_input)
        self.display_image_button.clicked.connect(self.start_program)
        self.img_video_button.clicked.connect(self.open_image_video)
        self.location_button.clicked.connect(self.on_location_button_clicked2)
        self.input.currentIndexChanged.connect(self.change_input_device)
        self.img_video_button.clicked.connect(self.enable_image_loading_mode)
        self.extracting_frames = False
        self.skip_frames_count = 0
        self.custom_size = None
        self.stop_extract.clicked.connect(self.on_stop_extract_clicked)
        self.save_path = None
        self.populate_combo_box()
        self.display_help.currentIndexChanged.connect(self.load_file)
        self.grid_checkbox.stateChanged.connect(self.redraw_grid)
        self.grid_slider.valueChanged.connect(self.redraw_grid)
        self.grid_slider.setMinimum(16)
        self.fp_select_combobox.currentIndexChanged.connect(self.switch_floating_point_mode)
        self.crop_enabled = self.crop_true.isChecked()
        self.current_cropped_directory = None
        self.app = app
        self.yolo_files = []
        self.cuda_available = False
        self.crop_all_button.clicked.connect(self.auto_label_yolo_button_clicked)
        self.console_output_timer = QTimer(self)
        self.console_output_timer.timeout.connect(self.update_console_output)
        self.console_output_timer.start(1000)
        self.class_labels = self.load_class_labels()
        self.cpu_usage = 0
        self.gpu_usage = 0
        self.memory_usage = 0
        self.video_download.setRowCount(5)
        for i in range(5):
            self.video_download.setItem(i, 1, QtWidgets.QTableWidgetItem(''))
            self.video_download.setItem(i, 2, QtWidgets.QTableWidgetItem(''))
            self.video_download.setItem(i, 3, QtWidgets.QTableWidgetItem(''))
            self.video_download.setItem(i, 4, QtWidgets.QTableWidgetItem(''))
            self.video_download.setItem(i, 5, QtWidgets.QTableWidgetItem(''))
        self.start_gathering_metrics()
        pytorch_cuda_available = torch.cuda.is_available()
        opencv_cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.init_msg = f"PyTorch CUDA: {'True' if pytorch_cuda_available else 'False'}, OpenCV CUDA: {'True' if opencv_cuda_available else 'False'}"
        self.flip_images = False
        self.flip_checkbox.stateChanged.connect(self.set_flip_images)

        self.Load_pt_model.clicked.connect(self.load_model)
        self.convertButton.clicked.connect(self.handle_conversion)
        self.last_logged_file_name = None  # Initialize last_logged_file_name to None to prevent duplicate log entries when cycling through images using a timer for the next and previous buttons.
        self.flash_color.clicked.connect(self.pick_flash_color)

        self.process_folder_btn.clicked.connect(self.process_batch)
        self.overwrite_var.stateChanged.connect(self.on_check_button_click)
        self.model = None
        self.flash_color_rgb = (255, 255, 255, 255)
        self.batch_directory = ""
        self.batch_running = False
        self.current_bounding_boxes = []
        self.current_image_index = -1
        self.device = "cuda"
        self.image_files = []
        self.input_labels = []
        self.input_points = []
        self.model_type = "vit_b"
        self.sam_checkpoint = 'Sam/sam_vit_b_01ec64.pth'  # Path to your model checkpoint
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        self.temp_bbox = []
        self.overwrite = False
        self.update_image_files()
        self.yolo_cache = {}
        self.sam_files = os.listdir('Sam')
        self.sam_model.addItems(self.sam_files)  # Updated this line
        self.sam_model.currentIndexChanged.connect(self.on_dropdown_changed)  # Updated this line
        self.dino_label.clicked.connect(self.on_dino_label_clicked)
        self.image_directory = None  # Initialize the attribute to store the image directory
        self.file_monitoring_thread = None
        self.file_observer = None
        self.clear_json.clicked.connect(self.on_clear_json_clicked)
        self.super_resolution_Checkbox.clicked.connect(self.checkbox_clicked)
        self.mosaic_checkbox.stateChanged.connect(self.mosaic_checkbox_changed)
        self.size_input.valueChanged.connect(self.on_size_input_changed)
        self.size_number.valueChanged.connect(self.on_size_number_changed)
        self.augmentation_size = 416  # Default size
        self.augmentation_count = 100  # Default count
        self.mosaic_effect = False                  
        self.show()
        
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

        # Update self.model_type based on the selected file name
        if 'vit_h' in selected_file:
            self.model_type = 'vit_h'
        elif 'vit_b' in selected_file:
            self.model_type = 'vit_b'
        elif 'vit_l' in selected_file:
            self.model_type = 'vit_l'

        # Print the new model_type and sam_checkpoint values
        print(f'Model type changed to: {self.model_type}')
        print(f'SAM checkpoint changed to: {self.sam_checkpoint}')


    def on_check_button_click(self):
        self.overwrite = self.overwrite_var.isChecked()


    def process_directory(self):
        self.image_files = [file for file in os.listdir(self.batch_directory) if file.endswith((".jpg", ".png", ".jpeg"))]


    def predict_and_draw_yolo_objects(self, image, image_file_path):
        if not self.predictor or not self.sam or not os.path.isfile(image_file_path):
            logging.error('Predictor, SAM, or image file is not available.')
            QMessageBox.critical(self, "Error", "Predictor, SAM, or image file is not available.")
            return

        image_copy = image.copy()
        yolo_file_path = os.path.splitext(image_file_path)[0] + ".txt"
        adjusted_boxes = []

        # Read YOLO file or use cache
        try:
            yolo_lines = self.yolo_cache.get(yolo_file_path, None)
            if yolo_lines is None:
                with open(yolo_file_path, "r") as f:
                    yolo_lines = f.readlines()
                self.yolo_cache[yolo_file_path] = yolo_lines
        except Exception as e:
            logging.error(f'Failed to read yolo file: {e}')
            QMessageBox.critical(self, "Error", f"Failed to read yolo file: {e}")
            return

        img_height, img_width = image_copy.shape[:2]
        self.predictor.set_image(image_copy)

        for yolo_line in yolo_lines:
            class_index, x_center, y_center, w, h = map(float, yolo_line.strip().split())
            try:
                box = pbx.convert_bbox((x_center, y_center, w, h), from_type="yolo", to_type="voc", image_size=(img_width, img_height))
                input_box = np.array([box[0], box[1], box[2], box[3]]).reshape(1, 4)
                masks, _, _ = self.predictor.predict(box=input_box[None, :], multimask_output=False)
            except Exception as e:
                logging.error(f'Error during prediction or bounding box conversion: {e}')
                continue

            masked_image, (x_top_left, y_top_left, box_width, box_height) = self.draw_mask_and_bbox(image, masks[0], self.input_points)

            if box_width <= 0 or box_height <= 0:
                logging.warning(f"Invalid bounding box dimensions: {box_width}x{box_height}. Skipping this box.")
                continue

            x_bottom_right = x_top_left + box_width
            y_bottom_right = y_top_left + box_height
            new_coords = (x_top_left, y_top_left, x_bottom_right, y_bottom_right)

            if self.overwrite:
                try:
                    t = pbx.convert_bbox(new_coords, from_type="voc", to_type="yolo", image_size=(img_width, img_height))
                    s = "%d %.6f %.6f %.6f %.6f \n" % (class_index, t[0], t[1], t[2], t[3])
                    adjusted_boxes.append(s)
                except Exception as e:
                    logging.error(f'Error during bounding box conversion: {e}')
                    continue

            image = masked_image

        if self.overwrite:
            try:
                with open(yolo_file_path, "w") as f:
                    f.writelines(adjusted_boxes)
            except Exception as e:
                logging.error(f'Error writing to YOLO file: {e}')
                QMessageBox.critical(self, "Error", f"Error writing to YOLO file: {e}")

        return image


        

    def show_image(self, img):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, channel = img_rgb.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(img_rgb.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.screen_view.setScene(QtWidgets.QGraphicsScene())
        self.screen_view.scene().addPixmap(pixmap)
        self.screen_view.setSceneRect(QtCore.QRectF(pixmap.rect()))  # Convert QRect to QRectF
        self.screen_view.fitInView(self.screen_view.sceneRect(), QtCore.Qt.KeepAspectRatio)  # Fit the image within the view




    def process_batch(self):
        self.current_image_index = 0  # Start from the first image
        self.stop_labeling = False  # Reset the stop_labeling flag at the start
        self.stop_batch = False
        total_images = len(self.image_files)  # Total number of images to process
        self.label_progress.setRange(0, total_images)  

        if not self.image_directory or not self.sam or not self.predictor:  # Changed to self.image_directory
                QMessageBox.critical(self, "Error", "Please select a folder and model first")
                return

        # Ensure self.image_files is updated
        self.image_files = sorted(
            glob.glob(os.path.join(self.image_directory, "*.[pP][nN][gG]")) +
            glob.glob(os.path.join(self.image_directory, "*.[jJ][pP][gG]")) +
            glob.glob(os.path.join(self.image_directory, "*.[jJ][pP][eE][gG]")) +
            glob.glob(os.path.join(self.image_directory, "*.[bB][mM][pP]")) +
            glob.glob(os.path.join(self.image_directory, "*.[gG][iI][fF]")) +
            glob.glob(os.path.join(self.image_directory, "*.[tT][iI][fF]")) +
            glob.glob(os.path.join(self.image_directory, "*.[wW][eE][bB][pP]"))
        )

        # This is where you start processing images.
        for idx, file_name in enumerate(self.image_files):
            self.batch_running = True
            if self.stop_batch or self.stop_labeling:  # Check for stop_labeling flag alongside stop_batch
                self.batch_running = False
                break
            file_path = os.path.join(self.image_directory, file_name)
            image = cv2.imread(file_path)

            if image is None:
                print(f"Failed to load image: {file_path}")
                continue  # Skip to the next image

            processed_image = self.predict_and_draw_yolo_objects(image, file_path)
            if processed_image is not None:
                self.show_image(processed_image)

            self.label_progress.setValue(idx + 1)  # Update the progress bar
            QtWidgets.QApplication.processEvents()

        # Displaying completion message and finalizing progress bar
        if self.stop_labeling:
            QMessageBox.information(self, "Information", "Process was stopped!")
        else:
            QMessageBox.information(self, "Information", "Finished!")
        self.label_progress.setValue(total_images)  # Finalize the progress bar
        QtWidgets.QApplication.processEvents()


    def append_bbox(self):
        self.current_bounding_boxes.append(self.temp_bbox)
        self.temp_bbox = []
        self.input_points = []
        self.input_labels = []
        print(self.current_bounding_boxes)


    def draw_mask_and_bbox(self, image, mask, input_points):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logging.error("No contours found in the mask.")
            return image, (0, 0, 0, 0)

        img_height, img_width = image.shape[:2]
        x_min, y_min, x_max, y_max = img_width, img_height, 0, 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, max(x, 0))
            y_min = min(y_min, max(y, 0))
            x_max = max(x_max, min(x + w, img_width))
            y_max = max(y_max, min(y + h, img_height))

        if x_max <= x_min or y_max <= y_min:
            logging.error("Invalid bounding box coordinates computed.")
            return image, (0, 0, 0, 0)

        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Convert coordinates to integers
        x, y, w, h = map(int, [x_min, y_min, x_max - x_min, y_max - y_min])

        alpha = 0.2
        overlay = image.copy()
        cv2.fillPoly(overlay, contours, (255, 0, 0))
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        for point in input_points:
            self.draw_point(image, point[0], point[1], (255, 0, 0), 5)
        return image, (x, y, w, h)



    def update_image_files(self):
        if self.batch_directory:
            self.image_files = [f for f in os.listdir(self.batch_directory)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            

    def on_dino_label_clicked(self):
        self.dino_label.setEnabled(False)  # Disable the button to prevent multiple clicks
        try:
            if self.image_directory is not None:
                run_groundingdino(self.image_directory)
            else:
                QMessageBox.warning(self, 'Directory Not Selected', "Please select an image directory first.")
                self.open_image_video()  # Call the method that lets the user select the directory
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"An error occurred: {e}")
        finally:
            self.dino_label.setEnabled(True)  # Re-enable the button after processing is done

            
    # icon and console output
    def update_icon(self):
        # Update the window icon to the next frame
        self.current_icon_index = (self.current_icon_index + 1) % len(self.icons)
        self.setWindowIcon(self.icons[self.current_icon_index])

    def update_console_output(self):
        global start_time  # Declare start_time as a global variable
        now = datetime.now()
        elapsed_time = now - start_time
        elapsed_time_str = str(elapsed_time).split('.')[0]  # Remove microseconds for display

        # Number of last lines to read from the log file
        N = 1

        # Parse the last N log messages
        try:
            with open(os.path.join('debug', 'my_app.log'), 'r') as f:
                last_log_lines = f.readlines()[-N:]  # Get the last N lines in the log file
        except (FileNotFoundError, IndexError):
            last_log_lines = ["No log messages found"]

        last_log_messages = "".join(last_log_lines).strip()

        # Get CPU, GPU, and memory usage
        cpu_usage = self.cpu_usage
        gpu_usage = self.gpu_usage
        memory_usage = self.memory_usage

        text = f'Last log messages:<br>{last_log_messages}<br>' \
            f'<font color="red">UltraDarkLabel Initializing at: {start_time.strftime("%Y-%m-%d %H:%M:%S")}</font><br>' \
            f'{self.init_msg}<br>' \
            f'Running Time: {elapsed_time_str}<br>' \
            f'CPU Usage: {cpu_usage}%<br>' \
            f'GPU Usage: {gpu_usage}%<br>' \
            f'Memory Usage: {memory_usage}%'
        self.console_output.setText(text)
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

    # convert class ID

    def select_images_function(self):
        directory = QFileDialog.getExistingDirectory(
            self, 'Select Directory', os.getenv('HOME'))

        # Check if a directory was selected
        if not directory:
            QMessageBox.warning(self, "No Directory Selected", "Please select a valid directory.")
            return

        self.yolo_files = []
        try:
            for file in os.listdir(directory):
                if file.endswith('.txt') and file != 'classes.txt':
                    self.yolo_files.append(os.path.join(directory, file))
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"Failed to list files in the directory: {directory}")


    def convert_class_function(self):
        if not self.yolo_files:
            QMessageBox.warning(self, "No Files Loaded",
                "No YOLO files have been loaded. Please select a directory first.")
            return

        from_class = self.from_class.value()
        to_class = self.to_class.value()


        for yolo_file in self.yolo_files:
            with open(yolo_file, 'r') as file:
                lines = file.readlines()

            with open(yolo_file, 'w') as file:
                for line in lines:
                    parts = line.split()
                    current_class_id = int(parts[0])
                    if current_class_id == from_class:
                        parts[0] = str(to_class)
                    file.write(' '.join(parts) + '\n')

    def adjust_brightness_contrast(self, image_cv, alpha, beta):
        return cv2.convertScaleAbs(image_cv, alpha=alpha, beta=beta)

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


    def auto_label_images2(self):
                logging.info("auto_label_images2 called")

                classes_file_path = os.path.join(self.image_directory, 'classes.txt')
                if not os.path.exists(classes_file_path):
                    print("Classes file not found.")
                    return

                with open(classes_file_path, 'r') as classes_file:
                    class_labels = [line.strip() for line in classes_file.readlines()]
                class_indices = list(range(len(class_labels)))

                if not hasattr(self, 'model'):
                    print("Model is not initialized.")
                    return

                total_images = len(self.image_files)
                self.label_progress.setRange(0, total_images)

                overwrite = QMessageBox.question(
                    self, 'Overwrite Labels', "Do you want to overwrite existing labels?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )

                # Get network height and width from spinboxes
                network_height = self.network_height.value()
                network_width = self.network_width.value()

                # Determine floating point mode
                is_fp16 = self.fp_select_combobox.currentText() == "FP16"

                for idx, image_file in enumerate(self.image_files):
                    if self.stop_labeling:
                        print("Labeling stopped by user.")
                        break

                    # Cropping logic inside the loop
                    if self.crop_true.isChecked():
                        image = self.read_image(image_file)
                        desired_height = self.crop_height.value()
                        desired_width = self.network_width.value()
                        cropped_image = self.center_crop(image, desired_height, desired_width, image_file)

                        if cropped_image is not None:
                            # Save the cropped image
                            self.save_cropped_image(cropped_image, image_file)
                            # Update image_file to be the path of the cropped image
                            image_file = os.path.join(self.current_cropped_directory, os.path.basename(image_file))

                    self.current_file = image_file
                    label_filename = os.path.splitext(os.path.basename(image_file))[0] + '.txt'
                    label_file = os.path.join(self.image_directory, label_filename)
                    label_exists = os.path.exists(label_file)

                    # Get image dimensions
                    self.display_image(image_file)
                    with Image.open(image_file) as img:
                        img_width, img_height = img.size

                    conf_threshold = self.confidence_threshold_slider.value() / 100
                    iou_threshold = self.nms_threshold_slider.value() / 100

                    try:
                        model_kwargs = {
                            'conf': conf_threshold,
                            'iou': iou_threshold,
                            'imgsz': [network_width, network_height]
                        }
                        if is_fp16:
                            model_kwargs['half'] = True

                        if self.model_type in ['yolov8', 'yolov8_trt', 'onnx']:
                            logging.info(f"Processing with {self.model_type}: {image_file}, params={model_kwargs}")
                            results = self.model(image_file, **model_kwargs)
                            

                            predicted_labels = results.pred[0][:, -1].int().tolist()
                            boxes = results.pred[0][:, :-2].cpu().numpy()
                            # Filter boxes and labels based on class_indices
                            filtered_boxes = [box for i, box in enumerate(boxes) if predicted_labels[i] in class_indices]
                            filtered_labels = [label for label in predicted_labels if label in class_indices]
                            labeled_boxes = list(zip(filtered_boxes, filtered_labels))

                    except AttributeError:  # Handle cases where the model output structure is different
                        results = self.model(self.current_file, classes=class_indices)
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        class_ids = results[0].boxes.cls.cpu().numpy()
                        labeled_boxes = list(zip(boxes, class_ids))


                        # Calculate minimum and maximum width and height based on user settings
                        min_width_px, max_width_px = self.box_size.value() / 100 * img_width, self.max_label.value() / 100 * img_width
                        min_height_px, max_height_px = self.box_size.value() / 100 * img_height, self.max_label.value() / 100 * img_height

                        # Apply size filtering to the bounding boxes
                        size_filtered_boxes = []
                        for box, class_id in labeled_boxes:
                            x1, y1, x2, y2 = box
                            width = x2 - x1
                            height = y2 - y1

                            # Check if the detected object's dimensions meet the size criteria
                            if min_width_px <= width <= max_width_px and min_height_px <= height <= max_height_px:
                                size_filtered_boxes.append((box, class_id))

                
                            # Save the size-filtered labels to a text file
                            if overwrite == QMessageBox.Yes or not label_exists:
                                if confidence >= conf_threshold:
                                    with open(label_file, 'a') as f:
                                        f.write(f"{int(class_id)} {xc} {yc} {w} {h} {confidence:.4f}\n")
                                    for box, class_id in size_filtered_boxes:
                                        x1, y1, x2, y2 = box
                                        xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                                        xc, yc, w, h = xc / img_width, yc / img_height, w / img_width, h / img_height
                                        f.write(f"{int(class_id)} {xc} {yc} {w} {h}\n")

                                        # Validate head_class_id only if heads_area is checked and class_id is not 1
                                        if self.heads_area.isChecked() and class_id != 1:
                                            head_class_id_str = self.class_id.text()
                                            if head_class_id_str.strip():  # Check if the string is not empty
                                                try:
                                                    head_class_id = int(head_class_id_str)
                                                except ValueError:
                                                    print(f"Warning: Invalid class ID '{head_class_id_str}'. Using the next available class ID.")
                                                    head_class_id = self.get_next_available_class_id()
                                            else:
                                                print("No class ID provided. Using the next available class ID.")
                                                head_class_id = self.get_next_available_class_id()


                                            # Calculate and write the head bounding box
                                            head_x, head_y, head_w, head_h = self.calculate_head_area(x1, y1, x2 - x1, y2 - y1)
                                            # Convert head box coordinates to relative format
                                            head_xc, head_yc, head_w, head_h = (head_x + head_w / 2) / img_width, (head_y + head_h / 2) / img_height, head_w / img_width, head_h / img_height
                                            f.write(f"{head_class_id} {head_xc} {head_yc} {head_w} {head_h}\n")
                            else:
                                print(f"Skipping file {image_file} as it already has labels and overwrite is set to 'No'.")

                    # Update progress bar for each image processed
                    progress_value = (idx + 1) / total_images * 100
                    self.label_progress.setValue(progress_value)
                    QApplication.processEvents()

                # This block should be outside the for loop
                self.label_progress.setValue(total_images)
                QApplication.processEvents()
                QMessageBox.information(self, "Information", "Finished!")
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
            self.createnegprogress.setValue)
        self.processing_thread.start()
    # convert format

    def select_directory(self):
        self.directory = QFileDialog.getExistingDirectory(
            self, "Select Directory")
        if self.directory:
            # Define image file extensions that QImageReader can read
            valid_extensions = ('.bmp', '.gif', '.jpg', '.jpeg', '.png',
                                '.pbm', '.pgm', '.ppm', '.xbm', '.xpm', '.svg', '.tiff', '.tif')

            # Get all the image files in the directory
            self.files = [f for f in os.listdir(
                self.directory) if f.lower().endswith(valid_extensions)]

    def convert_images(self):
        # Check if the image directory is set
        if not hasattr(self, 'directory') or self.directory is None:
            QMessageBox.warning(self, "Error", "Image directory not selected.")
            return

        # Check if self.files is initialized and has files
        if not hasattr(self, 'files') or not self.files:
            QMessageBox.warning(self, "Error", "No files to convert.")
            return

        self.format_progress.setRange(0, len(self.files))
        self.format_progress.setValue(0)  # Reset the progress bar
        target_format = self.select_format.currentText()

        # Create a sub-directory with the selected format name
        self.target_directory = os.path.join(self.directory, target_format)
        os.makedirs(self.target_directory, exist_ok=True)

        for i, file in enumerate(self.files):
            runnable = ImageConverterRunnable(
                self.directory, file, target_format, self.target_directory)
            runnable.signals.finished.connect(self.update_progress_bar)
            self.threadpool.start(runnable)


    def update_progress_bar(self):
        self.format_progress.setValue(self.format_progress.value() + 1)

    # video download
    def remove_selected_video(self):
        current_row = self.video_download.currentRow()
        self.video_download.removeRow(current_row)


    def download_videos(self):
        print("download_videos is called")
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select download directory')
        if not directory:  # Ensure a directory is selected
            return
        for row in range(self.video_download.rowCount()):
            item = self.video_download.item(row, 0)
            if item is not None:  # Check if the item is not None
                url = item.text()
                if url:  # Only process rows with a URL entered
                    # Include directory parameter here
                    download_thread = DownloadThread(url, row, directory)
                    download_thread.update_status.connect(self.update_download_status)
                    download_thread.update_progress.connect(self.update_download_progress)
                    self.download_threads.append(download_thread)
                    download_thread.start()
            else:
                print(f"No item found in row {row}, column 0")


    def update_download_progress(self, row, progress):
        self.progress_bars[row].setValue(progress)

    def update_download_status(self, row, status):
        self.video_download.setItem(row, 1, QtWidgets.QTableWidgetItem(status))

    # main video extractor class
    def enable_image_loading_mode(self):
        self.stop_program()  # Stop any existing camera feed

    def start_program(self):
        self.setup_input_device()
        self.start_camera_mode()  # Enable camera mode when starting the program

    def stop_program(self):
        self.timer2.stop()
        self.is_camera_mode = False
        if self.capture:
            self.capture.release()
            self.capture = None

    def start_camera_mode(self):
        self.is_camera_mode = True
        self.timer2.start(0)  # for x ms interval, adjust as needed

    def get_input_devices(self):
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return arr

    def setup_input_device(self):
        # Clear existing items before adding new ones
        self.input.clear()
        for device_index in self.get_input_devices():
            self.input.addItem(str(device_index))

    def change_input_device(self, index):
        self.stop_program()  # stop previous video feed
        if self.capture:
            self.capture.release()
            self.capture = None
        # initialize capture to the selected device
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FPS, 200)  # Set the desired FPS
        self.start_camera_mode()  # start new video feed

    def set_camera_mode(self, is_camera):
        self.is_camera_mode = is_camera

    @pyqtSlot()
    def on_extract_clicked(self):
        print("Extract button clicked!")  # Debugging print
        self.extracting_frames = True

    @pyqtSlot()
    def on_stop_extract_clicked(self):
        self.extracting_frames = False
        print("Stopping extraction...")  # Debugging print

    @pyqtSlot(int)
    def on_skip_frames_changed(self, value):
        self.skip_frames_count = value

    @pyqtSlot(int, int)
    def on_custom_size_changed(self, width, height):
        self.custom_size = (width, height)


    def display_camera_input(self):
        try:
            if self.display_input.isChecked():
                # Capture the entire screen
                screen_shot = pyautogui.screenshot()

                # Convert it to a NumPy array
                frame = np.array(screen_shot)

                # Convert to BGR format for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Optionally, resize the frame if custom size is needed
                if self.custom_size_checkbox.isChecked():
                    width = self.width_box.value()
                    height = self.height_box.value()
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)

                # Save frame if extraction is ongoing
                if self.extracting_frames:
                    print("Extracting frames...")  # Debugging print
                    self.save_frame(frame)

                # Display the screenshot as needed, similar to how you'd display the camera input

                # Convert to Qt format and set the scene
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
            else:
                # Logic for displaying camera input without screen capturing

                # Capture a frame from the camera
                ret, frame = self.capture.read()
                if ret:
                    # Convert the frame to the RGB format
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
                else:
                    raise Exception("Unable to read from capture device")
        except Exception as e:
            print(f"An error occurred while displaying camera input: {e}")
            self.stop_program()


    def save_frame(self, frame):
            try:
                # Ensure the output directory exists
                os.makedirs(self.save_path, exist_ok=True)
                print(f"Saving to directory: {self.save_path}")  # Debugging print

                # Get the selected image extension
                extension = self.get_image_extension()

                # Define the filename with a timestamp and selected extension
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}{extension}"

                # Join the save path with the filename
                full_path = os.path.join(self.save_path, filename)
                print(f"Saving image to: {full_path}")  # Debugging print

                # Save the frame as an image in the selected format
                cv2.imwrite(full_path, frame)
                print(f"Image saved successfully.")  # Debugging print
            except Exception as e:
                print(f"An error occurred while saving the frame: {e}")


    def on_location_button_clicked2(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.save_path = directory

    @pyqtSlot(int)
    def update_progress(self, progress):
        # Update the progress bar with the new progress value
        self.extract_progress.setValue(progress)

    def update_checkboxes(self):
        if self.custom_size_checkbox.isChecked():
            self.height_box.setEnabled(True)
            self.width_box.setEnabled(True)
        else:
            self.height_box.setEnabled(False)
            self.width_box.setEnabled(False)

    def on_add_video_clicked(self):
        if self.add_video_running:
            return

        self.add_video_running = True
        if self.dialog_open:
            return
        self.dialog_open = True
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter(QCoreApplication.translate(
            "Context", "Videos (*.mp4 *.avi)"))
        if file_dialog.exec_():
            filename = file_dialog.selectedFiles()[0]
            # Check if there are already 5 videos in the list
            if len(self.video_processor.videos) < 5:
                self.video_processor.add_video(filename)
                # Add filename to the table widget
                row_position = self.video_table.rowCount()
                self.video_table.insertRow(row_position)
                self.video_table.setItem(
                    row_position, 0, QTableWidgetItem(filename))
            else:
                # Show a message box to inform the user
                QMessageBox.information(
                    self, "Information", "A maximum of 5 videos can be added at a time.")

        self.dialog_open = False
        self.add_video_running = False

    def on_custom_input_changed(self, text):
        """
        Handle changes in the custom input field for frame count.
        Sets the custom frame count for video processing based on the input text.
        If the input text is a digit, it sets that number as the frame count.
        If the input text is not a digit, it resets the frame count to None.

        :param text: The text input for custom frame count.
        """
        if text.isdigit():
            self.video_processor.set_custom_frame_count(int(text))
        else:
            self.video_processor.set_custom_frame_count(None)


    def on_remove_video_clicked(self):
        """
        Handle the event when the 'remove video' button is clicked.
        Removes the currently selected video from the processing list and updates the video table.
        """
        current_row = self.video_table.currentRow()
        if current_row != -1:
            self.video_processor.videos.pop(current_row)
            self.video_table.removeRow(current_row)

    def on_custom_frames_toggled(self, state):
        """
        Toggle the state of custom frames input.
        Enables or disables custom frame count based on the checkbox state.
        :param state: The state of the custom frames checkbox.
        """
        if state == QtCore.Qt.Checked:
            custom_input_text = self.custom_input.text()
            if custom_input_text.isdigit():
                self.video_processor.set_custom_frame_count(
                    int(custom_input_text))
            else:
                self.custom_frames_checkbox.setChecked(False)

    def on_original_size_toggled(self, state):
        """
        Toggle the state of the original size option.
        Sets the video processing to use the original size of the video if checked.
        :param state: The state of the original size checkbox.
        """
        self.video_processor.set_original_size(state == QtCore.Qt.Checked)

    def on_custom_size_checkbox_state_changed(self, state):
        """
        Handle the state change of the custom size checkbox.
        Enables or disables the height and width input boxes based on the checkbox state.
        :param state: The state of the custom size checkbox.
        """
        if state:
            self.height_box.setEnabled(True)
            self.width_box.setEnabled(True)
            self.on_size_spinbox_value_changed()
        else:
            self.height_box.setEnabled(False)
            self.width_box.setEnabled(False)

    def on_size_spinbox_value_changed(self):
        """
        Handle changes in the value of the size spinboxes.
        Sets the custom size for video processing based on the height and width values.
        """
        if self.custom_size_checkbox.isChecked():
            height = self.height_box.value()
            width = self.width_box.value()
            self.video_processor.set_custom_size((width, height))
        else:
            self.video_processor.set_custom_size(None)

    def on_image_format_changed(self, text):
        """
        Handle changes in the selected image format.
        Sets the image format for video frame extraction.
        :param text: The selected image format.
        """
        self.video_processor.set_image_format(text)

    def get_image_extension(self):
        """
        Get the file extension for the currently selected image format.
        Maps the selected format to its corresponding file extension.
        :return: The file extension for the selected image format.
        """
        format_mapping = {"*.JPG": ".jpg", "*.JPEG": ".jpeg",
                        "*.GIF": ".gif", "*.BMP": ".bmp", "*.PNG": ".png"}
        selected_format = self.image_format.currentText()
        return format_mapping.get(selected_format, ".jpg")

    def get_selected_video_path(self):
        """
        Get the file path of the currently selected video.
        Returns the path of the video that is currently selected in the video table.
        :return: The file path of the selected video or None if no video is selected.
        """
        current_row = self.video_table.currentRow()
        if current_row != -1:
            return self.video_table.item(current_row, 0).text()
        return None


    # New method to handle video file processing
    def process_video_file(self, video_path):
        # Use self.save_path if it is set, otherwise fall back to the original video directory
        output_dir = self.save_path if self.save_path else os.path.dirname(
            self.video_processor.videos[0])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f"Extracted frames_{timestamp}"

        output_path = os.path.join(output_dir, output_dir_name)
        os.makedirs(output_path, exist_ok=True)
        self.video_processor.set_output_path(output_path)

        self.video_processor.run()

    @pyqtSlot()
    def on_extract_clicked(self):
        """
        Handle the event when the 'extract' button is clicked.
        Initiates the frame extraction process based on the current GUI settings.
        If the 'display input' checkbox is checked, it sets a flag to start extracting frames.
        If the 'display input' checkbox is not checked, it retrieves the path of the selected video
        and starts processing it if a video is selected.
        """
        if self.display_input.isChecked():
            self.extracting_frames = True
        else:
            video_path = self.get_selected_video_path()  # Get selected video path
            if video_path:
                self.extracting_frames = True
                self.process_video_file(video_path)


     # to add gifs
    def populateGifCombo(self):
        self.gif_change.clear()
        gif_directory = "styles"
        gif_files = [f for f in os.listdir(
            gif_directory) if f.endswith(".gif")]
        self.gif_change.addItems(gif_files)
    # see the

    def onGifChange(self, index):
        if index != -1:
            selected_gif = self.gif_change.currentText()
            self.movie.stop()
            self.movie.setFileName(f"styles/{selected_gif}")
            self.movie.start()


    def stop_processing_and_clear(self):
        # Stop processing
        self.processing = False

        # Clear table
        self.preview_list.setRowCount(0)

    def eventFilter(self, source, event):
        try:
            if event.type() == QEvent.MouseButtonPress and source is self.preview_list.viewport():
                item = self.preview_list.itemAt(event.pos())
                if item is not None:
                    if event.button() == Qt.RightButton:
                        self.handle_right_click(self.preview_list.indexFromItem(item))
                    elif event.button() == Qt.LeftButton:
                        self.handle_left_click(self.preview_list.indexFromItem(item))
            return super().eventFilter(source, event)
        except Exception as e:
            print(f"An unexpected error occurred in eventFilter: {e}")
            return False  # Returning False allows other event handlers to handle this event


    # Adjust the image size based on the provided value
    def adjust_image_size(self, value):
        current_time = time.time()
        MAX_SIZE = 200

        # Limit the value to the maximum size
        if value > MAX_SIZE:
            value = MAX_SIZE
            self.slider.setValue(MAX_SIZE)

        # Prevent rapid calls within 0.8 seconds
        if current_time - self._last_call < 0.8:
            return

        # Adjust the column width in the preview list
        self.preview_list.setColumnWidth(0, value)
        # Extract and display data with the adjusted size
        self.extract_and_display_data(value)
        self._last_call = current_time

    # Highlight the thumbnail at the specified row
    def highlight_thumbnail(self, row):
        thumbnail_label = self.preview_list.cellWidget(row, 0)
        thumbnail_label.setStyleSheet("border: 2px solid red;")

    # Remove the highlighting from all thumbnails
    def unhighlight_all_thumbnails(self):
        for row in range(self.preview_list.rowCount()):
            thumbnail_label = self.preview_list.cellWidget(row, 0)
            thumbnail_label.setStyleSheet("")


    @pyqtSlot(QModelIndex)
    def handle_right_click(self, index):
        try:
            row = index.row()
            image_item = self.preview_list.item(row, 0)
            label_item = self.preview_list.item(row, 5)

            if image_item is not None and label_item is not None:
                image_file = image_item.text()
                label_file = image_file.rsplit('.', 1)[0] + '.txt'
                label_to_delete = label_item.text()

                with open(label_file, 'r') as f:
                    lines = f.readlines()

                with open(label_file, 'w') as f:
                    for line in lines:
                        if line.strip() != label_to_delete:
                            f.write(line)

                self.preview_list.removeRow(row)

                # Realign the remaining entries
                for i in range(row, self.preview_list.rowCount()):
                    item = self.preview_list.item(i, 1)
                    if item is not None:
                        item.setText(str(i))
            else:
                print("Image or label item is None.")
        except Exception as e:
            print(f"An unexpected error occurred in handle_right_click: {e}")

    def handle_left_click(self, index):
        row = index.row()
        # Log item fetching for debugging.
        item = self.preview_list.item(row, 0)
        image_file = item.text()
        self.display_image(image_file)
        self.unhighlight_all_thumbnails()
        self.highlight_thumbnail(row)
        # Toggle the hide_labels flag and update label visibility.
        if self.hide_labels:  # If labels are currently hidden, show them
            self.hide_labels = False
            self.toggle_label_visibility()


        # Retrieve the bounding box item from the preview_list
        bounding_box_item = self.preview_list.item(row, 5)
        if bounding_box_item is not None:
            # Retrieve the bounding box index stored as user data in the bounding_box_item
            bounding_box_index = bounding_box_item.data(Qt.UserRole)
            # Construct unique_key using image_file and bounding_box_index
            unique_key = f"{image_file}_{bounding_box_index}"
        else:
            return
        # Look up the bounding box details using the constructed unique_key
        bounding_box_details = self.bounding_boxes.get(unique_key)
        flash_duration = self.flash_time.value()

        if bounding_box_details is not None:
            # Find the specific bounding box item in the display image that matches the clicked thumbnail
            for rect_item in self.screen_view.scene().items():
                if isinstance(rect_item, BoundingBoxDrawer) and rect_item.unique_id == unique_key and rect_item.file_name == image_file:
                    # Flash the bounding box in yellow for the flash_duration milliseconds.
                    rect_item.flash_color = QColor(*self.flash_color_rgb)

                    rect_item.flash(duration=flash_duration)

                    # Stop the loop once the correct bounding box is found and flashed.
                    break

    def pick_flash_color(self):
        color = QColorDialog.getColor()  # Open a color dialog to choose a color
        if color.isValid():
            # Save the chosen color in RGB format
            self.flash_color_rgb = color.getRgb()

    def save_cropped_images(self, pixmap, x_center, y_center, image_file, i):
        # Check the state of the crop_img checkbox
        crop_images = self.crop_img.isChecked()

        # If crop_img is checked, proceed with cropping
        if crop_images:
            # Get the cropping dimensions from w_img and h_img spin boxes
            crop_width = self.w_img.value()
            crop_height = self.h_img.value()

            # Calculate cropping coordinates centered around the bounding box
            crop_x1 = max(0, int(x_center - crop_width // 2))
            crop_y1 = max(0, int(y_center - crop_height // 2))
            crop_x2 = min(pixmap.width(), crop_x1 + crop_width)
            crop_y2 = min(pixmap.height(), crop_y1 + crop_height)

            # Crop the image
            cropped_pixmap = pixmap.copy(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1)

            # Check the dimensions of the cropped pixmap
            if cropped_pixmap.width() == crop_width and cropped_pixmap.height() == crop_height:
                # Create a directory for cropped images if it doesn't exist
                cropped_directory = os.path.join(self.image_directory, "cropped")
                os.makedirs(cropped_directory, exist_ok=True)

                # Save the cropped image to the 'cropped' folder
                base_file_name = os.path.basename(image_file)
                base_file_name, ext = os.path.splitext(base_file_name)
                cropped_filename = os.path.join(
                    cropped_directory, f"{base_file_name}_{i}_cropped{ext}")
                cropped_pixmap.save(cropped_filename)



    # Extract and display image data based on the new size
    def extract_and_display_data(self, new_size):
        # Set the processing flag to True
        self.processing = True

        # Check if an image directory is selected, show a warning if not
        if self.image_directory is None:
            QMessageBox.warning(self, "No Directory Selected",
                                "Please select a directory before previewing images.")
            return

        # Determine the new size for images
        new_size = self.image_size.value() if self.image_size else 1
        data_directory = self.image_directory

        # Get a list of all files in the selected directory
        all_files = glob.glob(os.path.join(data_directory, '*'))
        images_files = [f for f in all_files if f.endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        txt_files = [f for f in all_files if f.endswith('.txt')]

        # Show a warning if no images are found
        if not images_files:
            QMessageBox.warning(
                self, "No Images Found", "No images found. Please load images before adjusting the slider.")
            return

        # Create a directory for thumbnails if it doesn't exist_create_new_bbox
        thumbnails_directory = os.path.join(data_directory, "thumbnails")
        os.makedirs(thumbnails_directory, exist_ok=True)

        # Clear the preview list and set column headers
        self.preview_list.setRowCount(0)
        self.preview_list.setColumnCount(6)
        self.preview_list.setHorizontalHeaderLabels(['Image', 'Class Name', 'ID', 'Size', 'Confidence', 'Bounding Box'])

        # Load class names from a 'classes.txt' file if it exists
        classes_file_path = os.path.join(data_directory, 'classes.txt')
        if os.path.exists(classes_file_path):
            with open(classes_file_path, 'r') as f:
                for class_id, line in enumerate(f.readlines(), start=1):  # Notice the start=1 argument
                    class_name = line.strip()
                    self.id_to_class[class_id - 1] = class_name  # Subtract 1 from class_id


        # Determine the number of CPU cores and set the maximum workers
        num_cores = os.cpu_count()
        max_workers = max(num_cores - 1, 1)
        self.preview_progress.setMaximum(len(images_files))
        self.preview_progress.setValue(0)
        self.preview_list.setColumnWidth(0, new_size)
        pixmap_cache = {}

        # Process images in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for image_file in images_files:
                if not self.processing:
                    break

                # Load the image as a pixmap
                pixmap = load_image(image_file)

                if pixmap is None:
                    continue

                base_file = os.path.splitext(image_file)[0]
                label_file = f"{base_file}.txt"
                if label_file not in txt_files:
                    continue

                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                except Exception as e:
                    print(f"Error reading label file {label_file}: {str(e)}")
                    continue

                json_file_path = f"{base_file}.json"
                if not os.path.exists(json_file_path):
                    print(f"No json file found at {json_file_path}")
                    continue

                with open(json_file_path, 'r') as json_file:
                    json_data = json.load(json_file)

                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Invalid label format in {label_file}")
                        continue

                    try:
                        class_id = int(parts[0])
                    except ValueError:
                        print(f"Invalid class_id in {label_file}")
                        continue

                    class_name = self.id_to_class.get(class_id)
                    if class_name is None:
                        print(f"No class found with id {class_id}")
                        continue

                    try:
                        x_center, y_center, width, height = map(
                            float, parts[1:])
                    except ValueError:
                        print(f"Invalid label values in {label_file}")
                        continue

                    image_width = pixmap.width()
                    image_height = pixmap.height()
                    x_center *= image_width
                    y_center *= image_height
                    width *= image_width
                    height *= image_height

                    if (
                        width <= 0
                        or height <= 0
                        or x_center - width / 2 < 0
                        or x_center + width / 2 > image_width
                        or y_center - height / 2 < 0
                        or y_center + height / 2 > image_height
                    ):
                        continue
                    # Call the save_cropped_images method
                    self.save_cropped_images(pixmap, x_center, y_center, image_file, i)

                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    cropped_pixmap = pixmap.copy(x1, y1, x2 - x1, y2 - y1)

                    base_file_name = os.path.basename(image_file)
                    base_file_name, ext = os.path.splitext(base_file_name)
                    thumbnail_filename = os.path.join(
                        thumbnails_directory, f"{base_file_name}_{i}{ext}")
                    cropped_pixmap.save(thumbnail_filename)

                    # Insert a new row into the preview list
                    self.preview_list.insertRow(self.preview_list.rowCount())

                    # Resize the cropped pixmap to the new size
                    resized_pixmap = cropped_pixmap.scaled(
                        new_size, new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                    # Create a label for the thumbnail
                    thumbnail_label = QLabel()
                    thumbnail_label.setPixmap(resized_pixmap)
                    thumbnail_label.setScaledContents(False)
                    thumbnail_label.setAlignment(Qt.AlignCenter)

                    row_count = self.preview_list.rowCount() - 1
                    self.preview_list.setItem(
                        row_count, 0, QTableWidgetItem(image_file))
                    self.preview_list.setCellWidget(
                        row_count, 0, thumbnail_label)
                    self.preview_list.setItem(
                        row_count, 1, QTableWidgetItem(class_name))
                    self.preview_list.setItem(
                        row_count, 2, QTableWidgetItem(str(class_id)))
                    self.preview_list.setItem(
                        row_count, 3, QTableWidgetItem(f"{int(width)}x{int(height)}"))

                    # Create a bounding box item
                    bounding_box_item = QTableWidgetItem(line.strip())
                    bounding_box_item.setData(Qt.UserRole, i)  # Store bounding box index i as user data
                    self.preview_list.setItem(row_count, 5, bounding_box_item)

                    # Generate a unique key for each bounding_box_item
                    unique_id = f"{image_file}_{i}"  # unique id based on image file name and index i
                    bounding_box_item = QTableWidgetItem(line.strip())
                    self.bounding_boxes[unique_id] = bounding_box_item  # Store bounding box item with unique_id

                    # Get the confidence value from the JSON data
                    confidence = 0
                    for annotation in json_data.get('annotations', []):
                        if annotation.get('id') == i:
                            confidence = annotation.get('confidence', 0)
                            break
                    else:
                        # Handle the case where no matching annotation is found
                        confidence = 0  # or some other default value


                    confidence = round(confidence, 2) if confidence else 0

                    confidence_percentage = f"{confidence * 100}%"
                    confidence_item = QTableWidgetItem(confidence_percentage)
                    if confidence >= 0.75:  # adjust threshold as needed
                        confidence_item.setBackground(QColor(0, 255, 0))  # set background to green
                    elif confidence >= 0.5:
                        confidence_item.setBackground(QColor(255, 255, 0))  # set background to yellow
                    else:
                        confidence_item.setBackground(QColor(255, 0, 0))  # set background to red
                    self.preview_list.setItem(row_count, 4, confidence_item)

                    # Update the progress bar and process events
                    self.preview_progress.setValue(
                        self.preview_progress.value() + 1)
                    QApplication.processEvents()

                    # Resize columns and rows in the preview list
                    self.preview_list.resizeColumnsToContents()
                    self.preview_list.resizeRowsToContents()


    # mute sounds

    def mute_player(self, state):
        # Check the state of the muteCheckBox
        is_muted = state == QtCore.Qt.Checked

        # Set the muted state of the sound_player
        self.sound_player.setMuted(is_muted)
        # settings

    def keyPressEvent(self, event):
        key = event.text()  # get the key pressed

        if key == self.settings.get('nextButton'):
            if QApplication.keyboardModifiers() == Qt.NoModifier:
                self.next_frame()
            elif QApplication.keyboardModifiers() == Qt.ControlModifier:
                self.quick_next_navigation()

        elif key == self.settings.get('previousButton'):
            if QApplication.keyboardModifiers() == Qt.NoModifier:
                self.previous_frame()
            elif QApplication.keyboardModifiers() == Qt.ControlModifier:
                self.quick_previous_navigation()
        elif key == self.settings.get('deleteButton'):
            if QApplication.keyboardModifiers() == Qt.NoModifier:
                self.delete_current_image()
            elif QApplication.keyboardModifiers() == Qt.ControlModifier:
                self.delete_current_image()

        else:
            self.keyBuffer += key

            if not self.timer.isActive():
                # set time interval as 500 ms this can be very trick!
                self.timer.start(300)

    def processKeyPresses(self):
        key = self.keyBuffer  # process the keys in the buffer

        if key == self.settings.get('nextButton'):
            if QApplication.keyboardModifiers() == Qt.NoModifier:

                self.next_frame()
            elif QApplication.keyboardModifiers() == Qt.ControlModifier:

                self.quick_next_navigation()

        elif key == self.settings.get('previousButton'):
            if QApplication.keyboardModifiers() == Qt.NoModifier:
                self.previous_frame()
            elif QApplication.keyboardModifiers() == Qt.ControlModifier:

                self.quick_previous_navigation()

        elif key == self.settings.get('deleteButton'):
            if QApplication.keyboardModifiers() == Qt.NoModifier:
                self.delete_current_image()
            elif QApplication.keyboardModifiers() == Qt.ControlModifier:
                self.delete_current_image()

        elif key == QKeySequence(self.settings.get('autoLabel')):
            print("Auto label current image")
            self.auto_label_current_image()

        else:
            class_hotkeys = {
                k: v for k, v in self.settings.items() if k.startswith('classHotkey_')
            }

            class_name = None
            for class_key, hotkey in class_hotkeys.items():
                if hotkey == key:
                    class_name = class_key.split('classHotkey_')[-1]
                    break

            if class_name:
                print("Class name found:", class_name)
                index = self.classes_dropdown.findText(class_name)
                if index >= 0:
                    self.classes_dropdown.setCurrentIndex(index)

        self.keyBuffer = ""  # reset the buffer after processing

    def openSettingsDialog(self):
        print("Settings button was clicked.")
        settingsDialog = SettingsDialog(self)
        settingsDialog.exec_()

    def loadSettings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                # Load last weights, cfg paths, and directory path if they exist
                self.weights_file_path = settings.get('lastWeightsPath', '')
                self.cfg_file_path = settings.get('lastCfgPath', '')
                self.pt_weights_file_path = settings.get(
                    'lastPTWeightsPath', '')
                return settings
        except FileNotFoundError:
            # Return default settings if the file doesn't exist
            return {
                'nextButton': '',
                'previousButton': '',
                'deleteButton': '',
                'autoLabel': '',    # Add auto-label feature to settings
                'lastWeightsPath': '',    # Add last weights path to settings
                'lastCfgPath': '',    # Add last cfg path to settings
                'lastPTWeightsPath': '',    # Add last PT weights path to settings
                'last_dir': '',    # Add last directory path to settings
                'lastImage': '',    # Add last image to settings
                'anchors': []   # Add default 'anchors' key
            }

    def saveSettings(self):
        # Save the current image to the settings
        self.settings['lastImage'] = self.current_file
        with open('settings.json', 'w') as f:
            json.dump(self.settings, f)

    def set_sound(self, sound_path):
        self.sound_player.setMedia(
            QMediaContent(QUrl.fromLocalFile(sound_path)))
        self.sound_player.play()

    # all part of the auto label function.
    def reset_progress_bar(self):
        self.label_progress.blockSignals(True)  # Block signals
        self.label_progress.setValue(0)  # Reset progress bar to 0 explicitly
        self.label_progress.blockSignals(False)  # Unblock signals
        QApplication.processEvents()

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
            None, "Open Weights/Model", "", "Weights/Model Files (*.weights *.pt *.engine *.onnx)", options=options)

        if file_name:
            # Clear the previous model
            if hasattr(self, 'model'):
                del self.model
                self.model = None
                self.model_type = None

            self.weights_file_path = file_name
            self.settings['lastWeightsPath'] = file_name

            # Check the file extension and load the model accordingly
            if file_name.endswith('.pt'):
                # Loading a PyTorch model
                self.load_pytorch_model(file_name)
            elif file_name.endswith('.engine'):
                # Loading a TensorRT engine
                self.load_tensorrt_model(file_name)
            elif file_name.endswith('.onnx'):
                # Loading an ONNX model
                self.load_onnx_model(file_name)

            self.saveSettings()
            self.loadSettings()


    def load_pytorch_model(self, model_path):
        try:
            self.model = YOLO(model_path)  # YOLO is a placeholder for the actual loading function
            self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_type = 'yolov8'
            logging.info("Loaded PyTorch model and moved to GPU if available.")
        except Exception as e:
            logging.error(f"Failed to load PyTorch model: {e}")
            self.model = None
            self.model_type = None


    def load_tensorrt_model(self, model_path):
        try:
            # Assuming the YOLO class or loading function accepts a 'task' argument
            self.model = YOLO(model_path, task='detect')  # Explicitly setting task to 'detect'
            self.model_type = 'yolov8_trt'
            logging.info("Loaded model as YOLOv8 with TensorRT for detection.")
        except Exception as e:
            logging.error(f"Failed to load TensorRT engine file: {e}")


    def load_onnx_model(self, model_path):
        try:
            # Initialize the YOLO object for ONNX models with task='detect'
            self.model = YOLO(model_path, task='detect')
            self.model_type = 'onnx'
            logging.info("Loaded ONNX model using Ultralytics YOLO.")            
        except Exception as e:
            logging.error(f"Failed to load ONNX model: {e}")
            self.model = None
            self.model_type = None


    def open_cfg(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open CFG", "", "Config Files (*.cfg)", options=options)
        if file_name:
            self.cfg_file_path = file_name
            self.settings['lastCfgPath'] = file_name
            self.saveSettings()
            self.loadSettings()  # Reload the settings
            self.initialize_yolo()  # Re-initialize YOLO

    def auto_label_current_image(self):
        try:
            if not hasattr(self, 'current_file') or not hasattr(self, 'weights_file_path') or not hasattr(self, 'cfg_file_path'):
                QMessageBox.warning(
                    self, "Error", "Please select an image, weights file, and cfg file first.")
                return

            if not hasattr(self, 'net'):
                print("Weight file not selected. Please choose a weight file.")
                self.initialize_yolo()
            # assume 'net' and 'layer_names' are set in the initialize_yolo function
            image = self.get_cached_image(self.current_file)
            # blob size is defined by network_height and width
            blob = cv2.dnn.blobFromImage(
                image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.layer_names)

            class_ids = []
            confidences = []
            boxes = []

            label_directory = os.path.dirname(self.image_directory)
            if not os.path.exists(label_directory):
                raise Exception("Label directory not found.")

            classes_file_path = os.path.join(
                self.image_directory, 'classes.txt')
            if not os.path.exists(classes_file_path):
                raise Exception("Classes file not found.")

            if not hasattr(self, 'net'):
                self.initialize_yolo()

            label_file = self.get_label_file(self.current_file)

            # Display the current image
            self.display_image(self.current_file)

            if self.cuda_available:
                self.process_current_image()  # Use CUDA for processing
            else:
                # Use multi-threading for processing
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.process_current_image)
                    future.result()  # Wait for the processing to complete

        except Exception as e:
            print("An error occurred: ", e)
            return self.image

        # After processing the current image, call toggle_label_visibility to handle label visibility
            if self.hide_labels:  # If labels are currently hidden, show them
                self.hide_labels = False
                self.toggle_label_visibility()

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
                        print("Error: classes.txt not found.")
                        QMessageBox.warning(
                            self, "Error", "Classes.txt file not found in the image directory.")

                else:
                    print("Unsupported file extension for weights file. Please use a .weights file.")
            else:
                QMessageBox.warning(self, "Error", "Weights and/or CFG files not selected.")
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
            os.makedirs(self.current_cropped_directory)

        cropped_file_path = os.path.join(self.current_cropped_directory, base_name)
        cv2.imwrite(cropped_file_path, image)

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
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Could not read image from {file_path}")
            return image
        except Exception as e:
            logging.error(f"Failed to read image: {e}")
            return None

    def save_yolo_label_for_cropped_image(self, label_file, cropped_image, cropped_region):
        try:
            # Load the original label data from the original label file
            with open(label_file, 'r') as f:
                original_label_data = f.readlines()

            # Create a new label file for the cropped region
            cropped_label_file = label_file.replace('.txt', '_cropped.txt')

            with open(cropped_label_file, 'w') as f:
                for line in original_label_data:
                    # Parse the original label line
                    parts = line.strip().split()
                    class_id = parts[0]
                    x_center, y_center, width, height = map(float, parts[1:])

                    # Convert absolute coordinates to relative coordinates within the cropped area
                    x_center_rel = (x_center - cropped_region[0]) / cropped_region[2]
                    y_center_rel = (y_center - cropped_region[1]) / cropped_region[3]
                    width_rel = width / cropped_region[2]
                    height_rel = height / cropped_region[3]

                    # Write the relative coordinates to the cropped label file
                    f.write(f"{class_id} {x_center_rel} {y_center_rel} {width_rel} {height_rel}\n")

            print(f"Saved YOLO label for cropped image: {cropped_label_file}")
        except Exception as e:
            print(f"Error saving YOLO label for cropped image: {e}")

    def process_image(self, overwrite):
        if not hasattr(self, 'current_file'):
            print("No image file is currently opened.")
            return

        label_file = self.get_label_file(self.current_file)

        # Check if label_file is None, and handle it gracefully
        if label_file is None:
            print("Label file could not be generated. Skipping image.")
            return

        if os.path.exists(label_file) and not overwrite:
            return

        # Load the image
        image = cv2.imread(self.current_file)

        # Check if cropping is enabled
        if self.crop_true.isChecked():
            desired_height = self.crop_height.value()
            desired_width = self.crop_width.value()
            image = self.center_crop(image, desired_height, desired_width, self.current_file)  # pass file_path here

        # If the grayscale_Checkbox is activated, convert image to grayscale
        if self.grayscale_Checkbox.isChecked():
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Convert the grayscale image back to 3-channel
            image = cv2.merge((image, image, image))

            # You can adjust brightness and contrast
            alpha = self.grey_scale_slider.value() / 50.0
            beta = self.grey_scale_slider.value()
            image = self.adjust_brightness_contrast(image, alpha, beta)

        # If the outline_Checkbox is activated, apply edge detection
        if self.outline_Checkbox.isChecked():
            edges = cv2.Canny(image, self.slider_min_value,
                              self.slider_max_value)
            image[edges > 0, :3] = [255, 255, 255]

        # Perform object detection and display the result
        self.infer_and_display_bounding_boxes(image)

        # Save the detected bounding boxes to a file
        self.save_bounding_boxes(label_file, self.screen_view.scene(
        ).width(), self.screen_view.scene().height())

    def auto_label_images(self):
        if not hasattr(self, 'image_directory') or not hasattr(self, 'weights_file_path') or not hasattr(self, 'cfg_file_path'):
            print("Please select an image directory, weights file, and cfg file first.")
            return

        classes_file_path = os.path.join(self.image_directory, 'classes.txt')
        if not os.path.exists(classes_file_path):
            print("Classes file not found.")
            return

        if not hasattr(self, 'net'):
            self.initialize_yolo()

        # Get the class labels from the classes.txt file
        with open(classes_file_path, 'r') as classes_file:
            class_labels = [line.strip() for line in classes_file.readlines()]

        if not hasattr(self, 'cfg_file_path'):
            print("Please select a CFG file first.")
            return

        # Update the class labels in the classes.txt file
        with open(classes_file_path, 'w') as classes_file:
            for class_label in class_labels:
                classes_file.write(f'{class_label}\n')

        total_images = len(self.image_files)
        self.label_progress.setRange(0, total_images)
        # Check if overwrite is required
        overwrite = QMessageBox.question(
            self, 'Overwrite Labels', "Do you want to overwrite existing labels?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        for idx, image_file in enumerate(self.image_files):
            if self.stop_labeling:
                break

            self.current_file = image_file
            label_file = self.get_label_file(self.current_file)

            # If overwrite is not set and label exists, only process images that were not perfectly labeled (NMS can help in this case)
            if os.path.exists(label_file) and overwrite == QMessageBox.No:
                # Open the label file and read its contents
                with open(label_file, 'r') as f:
                    label_data = f.read().strip()

                # Skip this image if it already has labels
                if label_data:
                    continue

            # If label does not exist or overwrite is set, process all images
            self.display_image(self.current_file)
            self.process_image(overwrite)
            # Toggle label visibility based on checkbox status
            if self.hide_labels:  # If labels are currently hidden, show them
                self.hide_labels = False
                self.toggle_label_visibility()
            # Update the progress bar
            self.label_progress.setValue(idx + 1)

            # Update the GUI while the loop is running
            QApplication.processEvents()

        # After finishing the loop, show a message box saying that auto-labeling is finished
        QMessageBox.information(self, "Auto-Labeling",
                                "Finished auto-labeling all images.")

        self.stop_labeling = False



    def auto_label_yolo_button_clicked(self) -> bool:
        """
        Handler for the 'Auto Label YOLO' button click event. It initializes necessary configurations,
        checks for necessary files, loads class labels, and triggers the auto-labeling process based on
        the model type (PyTorch or Darknet). It also handles image cropping if selected, and updates
        the progress bar and display while processing images.
        """
        logging.info("auto_label_yolo_button_clicked called")
        # Existing checks for directories and files
        if not hasattr(self, 'image_directory') or self.image_directory is None:
            logging.error("Image directory not selected.")
            return

        if not hasattr(self, 'weights_file_path') or self.weights_file_path is None:
            logging.error("Weights file path not selected.")
            return

        if not hasattr(self, 'cfg_file_path') or self.cfg_file_path is None:
            logging.error("CFG file path not selected.")
            return

        # Directly loading class labels here after confirming image_directory is set.
        try:
            classes_file_path = os.path.join(self.image_directory, 'classes.txt')
            with open(classes_file_path, 'r') as classes_file:
                self.class_labels = [line.strip() for line in classes_file.readlines()]
        except FileNotFoundError:
            logging.error(f"Could not find the file at {classes_file_path}")
            return
        # Check weights file extension and proceed accordingly
        if self.weights_file_path.endswith('.pt') or self.weights_file_path.endswith('.engine') or self.weights_file_path.endswith('.onnx'):

            print("PyTorch model detected. Running auto_label_images2.")
            self.auto_label_images2()
        elif self.weights_file_path.endswith('.weights'):
            print("Darknet model detected. Running auto_label_images.")

            # Check if 'net' attribute is available
            if not hasattr(self, 'net'):
                self.initialize_yolo()

            # Check if cropping is active
            cropping_active = self.crop_true.isChecked()

            total_images = len(self.image_files)
            self.label_progress.setRange(0, total_images)

            # Check if overwrite is required
            overwrite = QMessageBox.question(
                self, 'Overwrite Labels', "Do you want to overwrite existing labels?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            for idx, image_file in enumerate(self.image_files):
                if self.stop_labeling:
                    break

                self.current_file = image_file
                label_file = self.get_label_file(self.current_file)

                # If overwrite is not set and label exists, only process images that were not perfectly labeled (NMS can help in this case)
                if os.path.exists(label_file) and overwrite == QMessageBox.No:
                    # Open the label file and read its contents
                    with open(label_file, 'r') as f:
                        label_data = f.read().strip()

                    # Skip this image if it already has labels
                    if label_data:
                        continue

                # Load the original image
                original_image = cv2.imread(self.current_file)
                if original_image is None:
                    logging.error(f"Failed to load image at {self.current_file}")
                    continue


                # If cropping is active, make a copy of the original image
                if cropping_active:
                    original_image_copy = original_image.copy()
                    desired_height = self.crop_height.value()
                    desired_width = self.crop_width.value()
                    cropped_image = self.center_crop(original_image_copy, desired_height, desired_width)

                    # Check if cropping is successful and meets the criteria
                    if cropped_image is not None and cropped_image.shape[0] == desired_height and cropped_image.shape[1] == desired_width:
                        self.current_file = self.save_cropped_image(cropped_image, self.current_file)
                        # Display the cropped image
                        self.display_image(cropped_image)
                    else:
                        # If cropping failed or didn't meet the criteria, display the original image
                        self.display_image(original_image)
                else:
                    self.display_image(original_image)  # Display the original image

                # Display the current image (either the original or cropped)
                self.display_image(self.current_file)

                # Process the image (object detection and labeling)
                self.process_image(overwrite)

                # Toggle label visibility based on checkbox status
                if self.hide_labels:  # If labels are currently hidden, show them
                    self.hide_labels = False
                    self.toggle_label_visibility()

                # Update the progress bar
                self.label_progress.setValue(idx + 1)

                # Update the GUI while the loop is running
                QApplication.processEvents()

            # After finishing the loop, show a message box saying that auto-labeling is finished
            QMessageBox.information(self, "Auto-Labeling",
                                    "Finished auto-labeling all images.")

            self.stop_labeling = False


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
        self.clear_bounding_boxes()

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

    def load_classes(self):
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


    def show_labeled_stats(self):
        class_names = self.load_classes()
        label_counts = {class_name: 0 for class_name in class_names}
        class_confidences = {class_name: [] for class_name in class_names}
        
        json_files = [os.path.join(self.image_directory, f) for f in os.listdir(self.image_directory) if f.endswith('.json')]
        
        total_confidences = []
        total_images = len(json_files)
        labeled_images = 0
        total_labels = 0

        for json_file in json_files:
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)

                    # Check if 'annotations' key exists in the data
                    if 'annotations' in data:
                        for annotation in data['annotations']:
                            class_id = annotation['category_id']
                            confidence = annotation.get('confidence', 0)  # Default to 0 if not found
                            if class_id < len(class_names):
                                class_name = class_names[class_id]
                                label_counts[class_name] += 1
                                total_labels += 1
                                class_confidences[class_name].append(confidence)
                                total_confidences.append(confidence)

                labeled_images += 1

        total_confidences = [c for c in total_confidences if c is not None]
        average_confidence = round(sum(total_confidences) / len(total_confidences), 2) if total_confidences else 0

        unlabeled_images = total_images - labeled_images
        labels_per_image = round(total_labels / total_images, 2) if total_images > 0 else 0

        stats = {
            'Total Images': total_images,
            'Labeled Images': labeled_images,
            'Unlabeled Images': unlabeled_images,
            'Total Labels': total_labels,
            'Labels per Image (average)': labels_per_image,
            'Labeling Progress (%)': round((labeled_images / total_images) * 100, 2) if total_images > 0 else 0,
            'Average Confidence': average_confidence
        }

        for class_name, confidences in class_confidences.items():
            # Filter out None values from confidences list
            confidences = [c for c in confidences if c is not None]
            stats[f'{class_name} Label Count'] = label_counts[class_name]
            stats[f'{class_name} Confidence'] = round(sum(confidences) / len(confidences), 2) if confidences else 0


        self.settings['stats'] = stats
        self.saveSettings()

    def display_stats(self):
        stats = self.settings.get('stats', {})
        if not stats:
            QMessageBox.information(self, 'Labeling Statistics', 'No statistics available.')
            return

        # Create a new widget to hold the table view
        self.stats_widget = QWidget()
        self.stats_widget.setWindowTitle("Labeling Statistics")
        layout = QVBoxLayout()
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Statistic", "Value", "Confidence"])

        # Add general stats to the model
        general_stats_keys = [
            'Total Images', 'Labeled Images', 'Unlabeled Images',
            'Total Labels', 'Labels per Image (average)',
            'Labeling Progress (%)', 'Average Confidence'
        ]
        for key in general_stats_keys:
            item = QStandardItem(key)
            value = QStandardItem(str(stats.get(key, "N/A")))
            # For general stats, confidence is not applicable
            confidence = QStandardItem("N/A")  
            model.appendRow([item, value, confidence])

        # Add class-specific stats to the model
        for class_name in self.load_classes():
            class_label_key = f'{class_name} Label Count'
            class_confidence_key = f'{class_name} Confidence'
            label_count = QStandardItem(str(stats.get(class_label_key, "N/A")))
            confidence = QStandardItem(str(stats.get(class_confidence_key, "N/A")))
            model.appendRow([QStandardItem(class_name), label_count, confidence])

        # Create a QTableView, set the model, and add it to the layout
        table = QTableView()
        table.setModel(model)
        self.style_table(table)
        layout.addWidget(table)
        self.stats_widget.setLayout(layout)
        self.stats_widget.show()

    def style_table(self, table):
        # Adjust column widths
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        # Styling the table
        table.setStyleSheet("""
        QTableView {
            border: 1px solid #cccccc;
            font: 11pt;
        }
        QTableView::item {
            padding: 5px;
        }
        QHeaderView::section {
            background-color: #f0f0f0;
            padding: 4px;
            border: 1px solid #cccccc;
            font-weight: bold;
        }
        QTableView::item:nth-child(odd) {
            background-color: #c1bebc;
        }
        """)



    def clear_class_boxes(self, class_name):
        # Check if a class name is provided
        if not class_name:
            print("Please enter a class name.")
            return

        # Display class-to-ID mapping for reference
        print("Class to ID:", self.class_to_id)

        # Get the target class ID based on the provided class name
        target_class_id = self.class_to_id.get(class_name)

        # Check if a valid class ID is found for the provided class name
        if target_class_id is None:
            print(f"No class found with name {class_name}")
            return

        # Display the target class ID
        print(f"Target class ID for {class_name}: {target_class_id}")

        # Iterate through filtered image files and clear bounding boxes for the target class
        for file_path in self.filtered_image_files:
            print(f"Processing file: {file_path}")

            # Generate the corresponding label file path
            label_path = os.path.splitext(file_path)[0] + '.txt'

            # Open the label file for reading and writing
            with open(label_path, 'r+') as f:
                lines = f.readlines()
                print(f"Original lines: {lines}")

                # Reset the file position and truncate the file content
                f.seek(0)
                f.truncate()

                # Iterate through lines in the label file
                for line in lines:
                    parts = line.strip().split(' ')
                    current_class_id = int(parts[0])
                    print(f"Current class ID: {current_class_id}")

                    # Check if the current class ID matches the target class ID
                    if current_class_id != target_class_id:
                        print(f"Writing line: {line}")
                        f.write(line)

        # Display a message indicating that bounding boxes for the class have been cleared
        print(f"Cleared bounding boxes for class: {class_name}")


    def on_clear_all_clicked(self):
        class_name = self.filter_class_input.text()
        self.clear_class_boxes(class_name)
        # Update the displayed images after clearing the bounding boxes
        self.display_all_images()

    def on_move_all_clicked(self):
        class_name = self.filter_class_input.text()
        self.move_filtered_images(class_name)

    def move_filtered_images(self, class_name):
        if not class_name and not self.filter_blanks_checkbox.isChecked():
            print("Please enter a class name.")
            return

        if class_name:
            class_folder = os.path.join(self.image_directory, class_name)
            os.makedirs(class_folder, exist_ok=True)

        for file_path in self.filtered_image_files:
            file_name = os.path.basename(file_path)
            base_file = os.path.splitext(file_path)[0]
            label_path = base_file + '.txt'
            json_path = base_file + '.json'

            is_blank = False
            if self.filter_blanks_checkbox.isChecked() and os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    is_blank = (content == '')

            if is_blank:
                blank_folder = os.path.join(self.image_directory, 'blank')
                os.makedirs(blank_folder, exist_ok=True)
                dest_path = os.path.join(blank_folder, file_name)
            elif class_name:
                dest_path = os.path.join(class_folder, file_name)
            else:
                continue

            try:
                shutil.move(file_path, dest_path)
                if os.path.exists(label_path):
                    dest_label_path = os.path.splitext(dest_path)[0] + '.txt'
                    shutil.move(label_path, dest_label_path)
                if os.path.exists(json_path):
                    dest_json_path = os.path.splitext(dest_path)[0] + '.json'
                    shutil.move(json_path, dest_json_path)
                print(f"Moved {'blank image' if is_blank else file_name} to {'blank' if is_blank else class_name} folder.")
            except Exception as e:
                print(f"Error moving {'blank image' if is_blank else file_name}: {str(e)}")


    def on_list_view_clicked(self, index):
        # Get the item at the clicked index
        item = self.List_view.model().itemFromIndex(index)
        # Change the item flags to make it uneditable
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)

        file_name = index.data()

        # Get the index of the clicked item in the list
        self.current_img_index = index.row()

        file_path = os.path.join(self.image_directory, file_name)
        self.open_image(file_path)

        # Update img_index_number and display image
        self.img_index_number.setValue(
            self.current_img_index)  # Update QSpinBox

        # Call toggle_label_visibility to handle label visibility
        if self.hide_labels:  # If labels are currently hidden, show them
            self.hide_labels = False
            self.toggle_label_visibility()


    def img_index_number_changed(self, value):
        if 0 <= value < len(self.filtered_image_files):
            self.current_img_index = value  # Update the current image index
            self.current_file = self.filtered_image_files[self.current_img_index]
            self.display_image(self.current_file)
            self.img_index_number.setValue(self.current_img_index)

    def handle_filter_blanks_state_change(self, state):
        if state == Qt.Checked:
            self.filter_class('')
        else:
            self.display_all_images()

    def open_image(self, file_path):
        self.display_image(file_path)

    def filter_class(self, class_name):
        filtered_image_files = []

        # Iterate through image files in the image directory
        for img_file in self.image_files:
            base_file = os.path.splitext(img_file)[0]
            label_file = base_file + '.txt'

            # Check if a label file exists for the image
            if os.path.isfile(label_file):
                try:
                    with open(label_file, 'r') as f:
                        content = f.read().strip()

                        # Check if the "Filter Blanks" option is selected
                        if self.filter_blanks_checkbox.isChecked():
                            # Filter blank images (images with no annotations)
                            if content == '':
                                filtered_image_files.append(
                                    os.path.join(self.image_directory, img_file))
                        else:
                            # Check if a class name is provided for filtering
                            if class_name == '':
                                # Display all images if no class name is specified
                                self.display_all_images()
                                return

                            # Get the class ID for the specified class name
                            filter_id = self.class_to_id.get(class_name)

                            # Check if the specified class name exists in the class dictionary
                            if filter_id is None:
                                print(f"No class found with name {class_name}")
                                return

                            # Iterate through the lines of the label file
                            for line in content.splitlines():
                                class_id = int(line.split()[0])

                                # Filter images with the specified class ID
                                if class_id == filter_id:
                                    filtered_image_files.append(
                                        os.path.join(self.image_directory, img_file))
                                    break
                except FileNotFoundError:
                    print(f"No label file found for {img_file}")
            else:
                print(f"No label file for {img_file}")

        # Update the list view with filtered image files
        self.update_list_view(filtered_image_files)
        self.filtered_image_files = filtered_image_files

        # Set the current file to the first filtered image (if any)
        if filtered_image_files:
            self.current_file = filtered_image_files[0]
            self.display_image(self.current_file)


    def update_list_view(self, image_files):
        model = QStandardItemModel()
        for img_file in image_files:
            item = QStandardItem(os.path.basename(img_file))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            model.appendRow(item)
        self.List_view.setModel(model)

    def display_all_images(self):
        # Populate QListView with all image files
        # Make sure filtered_image_files matches image_files
        self.filtered_image_files = [os.path.join(
            self.image_directory, img_file) for img_file in self.image_files]

        model = QStandardItemModel()
        for img_file in self.filtered_image_files:
            item = QStandardItem(os.path.basename(img_file))
            # Make item uneditable
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            model.appendRow(item)
        self.List_view.setModel(model)
        # Reset the filtered_image_files list to match image_files
        self.filtered_image_files = self.image_files.copy()

    def sorted_nicely(self, l):
        """ Sorts the given iterable in the way that humans expect."""
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)
    
    def update_json_file(self,txt_file_path):
        json_file_path = txt_file_path.replace('.txt', '.json')

        # Attempt to read existing JSON data
        try:
            with open(json_file_path, 'r') as json_file:
                json_data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            # If no valid JSON data, initialize empty structure
            json_data = {
                "annotations": [],
                "categories": []
            }

        # Read and parse the .txt file
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()

        updated_annotations = []
        categories = set()

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue  # Skip invalid lines

            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])

            # Find corresponding existing annotation to keep confidence value
            existing_annotation = next((item for item in json_data["annotations"] if item["category_id"] == class_id and item["bbox"] == [x, y, width, height]), None)
            confidence = existing_annotation["confidence"] if existing_annotation else 0

            updated_annotations.append({
                "bbox": [x, y, width, height],
                "category_id": class_id,
                "iscrowd": 0,
                "confidence": confidence
            })

            categories.add(class_id)

        # Update categories in JSON data
        json_data["categories"] = [{"id": cat_id, "name": str(cat_id)} for cat_id in categories]
        # Update annotations in JSON data
        json_data["annotations"] = updated_annotations

        # Write the updated data to the .json file
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)   
    def stop_file_monitoring(self):
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
            self.file_observer = None

        if self.file_monitoring_thread and self.file_monitoring_thread != threading.current_thread():
            self.file_monitoring_thread.join()
            self.file_monitoring_thread = None

    def open_image_video(self):
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly

            # Read the last directory from the settings
            last_dir = self.settings.get('last_dir', "")

            dir_name = QFileDialog.getExistingDirectory(
                None, "Open Image Directory", last_dir, options=options)
            placeholder_image_path = 'styles/yolo.jpg'
            if dir_name:
                # Stop any existing file monitoring
                self.stop_file_monitoring()

                # Save the selected directory to the settings
                self.settings['last_dir'] = dir_name
                self.saveSettings()  # save the settings after modifying it

                self.image_directory = dir_name

                # Start file monitoring for the selected directory
                self.file_monitoring_thread = threading.Thread(target=self.start_file_monitoring, args=(dir_name,))
                self.file_monitoring_thread.daemon = True
                self.file_monitoring_thread.start()
                # Import classes.txt
                classes_file_path = os.path.join(
                    self.image_directory, 'classes.txt')
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

                # Initialize the hotkeys for the classes in the current dataset.
                for className in class_names:
                    class_hotkey_setting = 'classHotkey_{}'.format(className)
                    if class_hotkey_setting not in self.settings:
                        self.settings[class_hotkey_setting] = ''
                    else:
                        # If the hotkey is already assigned to a class, remove it from that class.
                        current_hotkey = self.settings[class_hotkey_setting]
                        for other_class in self.settings:
                            if other_class.startswith('classHotkey_') and other_class != class_hotkey_setting:
                                if self.settings[other_class] == current_hotkey:
                                    self.settings[other_class] = ''

                self.saveSettings()  # Save the settings after modifying the hotkeys
                # Update the classes dropdown after setting the class names
                self.update_classes_dropdown()

                # Get all the image files in the directory
                # Get all the image files in the directory
                self.image_files = self.sorted_nicely(
                    glob.glob(os.path.join(dir_name, "*.[pP][nN][gG]")) +
                    glob.glob(os.path.join(dir_name, "*.[jJ][pP][gG]")) +
                    glob.glob(os.path.join(dir_name, "*.[jJ][pP][eE][gG]")) +
                    glob.glob(os.path.join(dir_name, "*.[bB][mM][pP]")) +
                    glob.glob(os.path.join(dir_name, "*.[gG][iI][fF]")) +
                    glob.glob(os.path.join(dir_name, "*.[tT][iI][fF]")) +
                    glob.glob(os.path.join(dir_name, "*.[wW][eE][bB][pP]"))
                )

                # Insert the placeholder image at the beginning of the list
                if os.path.exists(placeholder_image_path):
                    if placeholder_image_path not in self.image_files:
                        self.image_files.insert(0, placeholder_image_path)

                if len(self.image_files) > 0:
                    # Try to get the last opened image from the settings
                    last_image = self.settings.get('lastImage', "")
                    if last_image and last_image in self.image_files and last_image != placeholder_image_path:
                        self.current_image_index = self.image_files.index(last_image)
                    else:
                        # Always start with the placeholder image if no last opened image is found or if the last opened image is the placeholder
                        self.current_image_index = 0

                    self.current_file = self.image_files[self.current_image_index]
                    self.total_images.setText("Total Images: {}".format(len(self.image_files)))

                    self.display_image(self.current_file)
                    self.initialize_yolo()

                    # Populate list view with image files
                    model = QStandardItemModel()
                    for img_file in self.image_files:
                        item = QStandardItem(os.path.basename(img_file))
                        model.appendRow(item)
                    self.List_view.setModel(model)
                    self.display_all_images()

                    # Create empty .txt and .json files
                    self.create_empty_txt_and_json_files(dir_name)

                    # Start a new thread to calculate and save the labeling statistics
                    threading.Thread(
                        target=self.show_labeled_stats, daemon=True).start()

                else:
                    self.total_images.setText(f"Total Images: 0")
                    QMessageBox.warning(None, 'No Images Found', "No image files found in the directory.")
                    print("No image files found in the directory.")

    def create_empty_txt_and_json_files(self, image_directory):
        image_files = glob.glob(os.path.join(image_directory, "*.[pP][nN][gG]")) + \
            glob.glob(os.path.join(image_directory, "*.[jJ][pP][gG]")) + \
            glob.glob(os.path.join(image_directory, "*.[jJ][pP][eE][gG]")) + \
            glob.glob(os.path.join(image_directory, "*.[bB][mM][pP]")) + \
            glob.glob(os.path.join(image_directory, "*.[gG][iI][fF]")) + \
            glob.glob(os.path.join(image_directory, "*.[tT][iI][fF]")) + \
            glob.glob(os.path.join(image_directory, "*.[wW][eE][bB][pP]"))

        for image_file in image_files:
            txt_file = os.path.splitext(image_file)[0] + '.txt'
            json_file = os.path.splitext(image_file)[0] + '.json'
            if not os.path.exists(txt_file):
                with open(txt_file, 'w') as f:
                    pass
            if not os.path.exists(json_file):
                with open(json_file, 'w') as f:
                    json.dump({}, f)  # creates an empty json file
                    
    def start_file_monitoring(self, directory):
        self.stop_file_monitoring()
        event_handler = FileChangeHandler(directory, self.update_json_file)
        self.file_observer = Observer()
        self.file_observer.schedule(event_handler, directory, recursive=False)
        self.file_observer.start()



    def start_delete_timer(self):
        self.delete_timer.start()

    def stop_delete_timer(self):
        self.delete_timer.stop()
        
    def delete_current_image(self):
        # Check if any image has been loaded, if not just return
        if self.current_file is None:
            return

        # First, check if the current file is in the list of image files
        if self.current_file in self.image_files:
            # Get the index of the current file
            current_file_index = self.image_files.index(self.current_file)

            # Remove the current file from the list
            self.image_files.remove(self.current_file)

            # Delete the image file
            try:
                os.remove(self.current_file)
            except FileNotFoundError:
                print(f"Warning: {self.current_file} not found.")

            # Delete the associated text file
            txt_file = self.replace_extension_with_txt(self.current_file)
            try:
                os.remove(txt_file)
            except FileNotFoundError:
                print(f"Warning: {txt_file} not found.")

            # If the current file is also in the filtered list, remove it from there too
            if self.current_file in self.filtered_image_files:
                # Get the index of the current file
                filtered_file_index = self.filtered_image_files.index(
                    self.current_file)
                # Remove the current file from the filtered list
                self.filtered_image_files.remove(self.current_file)

                # Adjust the current image index to reflect the deletion
                if filtered_file_index < self.current_img_index:
                    self.current_img_index -= 1
                elif self.current_img_index >= len(self.filtered_image_files):
                    self.current_img_index = len(self.filtered_image_files) - 1

            # Load the next image if there is one, else load the previous one (now last in the list)
            if len(self.image_files) > 0:
                self.current_file = self.image_files[current_file_index if current_file_index < len(
                    self.image_files) else -1]
                self.display_image(self.current_file)

                # Update the list view without re-filtering
                self.update_list_view(self.filtered_image_files)

                # Update the displayed image index
                self.img_index_number.setValue(self.current_img_index)
            else:
                # If there are no more images, clear the display
                self.screen_view.setScene(None)
                self.current_file = None
        else:
            print("Warning: No image currently loaded.")
        # start of scene view functions

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

    def clear_bounding_boxes(self):
        items_to_remove = []
        for item in self.screen_view.scene().items():
            if isinstance(item, BoundingBoxDrawer):
                items_to_remove.append(item)

        for item in items_to_remove:
            self.screen_view.scene().removeItem(item)
            del item

    def qimage_to_cv2(self, img):
        ''' Convert QImage to OpenCV format '''
        img = img.convertToFormat(4)
        width, height = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        arr = cv2.UMat(arr)  # Convert numpy array to UMat for GPU acceleration
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)



    def cv2_to_qimage(self, img):
        ''' Convert OpenCV image (numpy array or UMat) to QImage '''
        # If img is UMat, convert it to numpy array
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
            return QImage(img.data, width, height, bytesPerLine, QImage.Format_ARGB32)
        else:
            raise ValueError(f'img should have 2 or 3 dimensions, but has {len(img.shape)}')


    def redraw_grid(self):
        # You might need to pass the appropriate file_name here
        self.display_image(file_name=self.current_file_name)

    def display_image(self, file_name=None, image=None, keypoints=None, masks=None):
        if file_name is None and image is None:
            print("File name and image are both None.")
            return None

        if file_name is not None:
            if not isinstance(file_name, str):               
                return None

            # Check if the file_name is the same as the last logged file name
            if file_name != self.last_logged_file_name:
                logging.info(f"display_image: file_name={file_name}")
                # Update the last logged file name
                self.last_logged_file_name = file_name

            # Load the image into a QPixmap
            self.image = QPixmap(file_name)

            if self.image.isNull():
                print("Failed to load the image.")
                return None

            self.original_pixmap_size = self.image.size()  # Save the original pixmap size

        if image is not None:
            if not isinstance(image, np.ndarray):
                print("Image is not a NumPy array.")
                return None

            # Convert the NumPy array to a QImage and then to a QPixmap
            image_qimage = self.cv2_to_qimage(image)
            self.image = QPixmap.fromImage(image_qimage)
            
        if self.super_resolution_Checkbox.isChecked() and file_name is not None:
            # Initialize the super-resolution object
            sr = cv2.dnn_superres.DnnSuperResImpl_create()

            # Load the desired model
            path_to_model = "C:/EAL/Scripts/DarkFusion/UltraDarkFusion/Sam/FSRCNN_x4.pb"  # Replace with your FSRCNN model path
            sr.readModel(path_to_model)

            # Set the model and scale factor
            sr.setModel("fsrcnn", 4)  # Use "espcn" or "lapsrn" for other models

            # Read the image using OpenCV
            img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            
            # Upscale the image
            img_sr = sr.upsample(img)

            # Convert the result back to QPixmap to display in the UI
            height, width, channel = img_sr.shape
            bytesPerLine = 3 * width
            qImg = QImage(img_sr.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            self.image = QPixmap.fromImage(qImg)
        # If the grayscale_Checkbox is activated, convert image to grayscale
        if self.grayscale_Checkbox.isChecked():
            image_qimage = self.image.toImage()
            image_cv = self.qimage_to_cv2(image_qimage)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

            # Set the current image here
            self.current_image = image_cv

            # Adjust brightness and contrast based on the value of the slider
            alpha = self.grey_scale_slider.value() / 50.0
            beta = self.grey_scale_slider.value()
            image_cv = self.adjust_brightness_contrast(image_cv, alpha, beta)

            image_qimage = self.cv2_to_qimage(image_cv)
            self.image = QPixmap.fromImage(image_qimage)

        # If the outline_Checkbox is activated, apply edge detection
        if self.outline_Checkbox.isChecked():
            image_qimage = self.image.toImage()
            image_cv = self.qimage_to_cv2(image_qimage)
            image_cv = image_cv.get()

            edges = cv2.Canny(image_cv, self.slider_min_value, self.slider_max_value)

            # Convert edges to a 4 channel image to match the image_cv
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGRA)

            # To overlay the edges correctly on the original image, we make sure they have the same dimensions
            if edges_colored.shape[:2] != image_cv.shape[:2]:
                height, width = image_cv.shape[:2]
                edges_colored = cv2.resize(edges_colored, (width, height))

            # Here we use addWeighted to overlay the edge detection result onto the original image
            image_cv = cv2.addWeighted(image_cv, 1, edges_colored, 0.5, 0)

            # Convert from BGRA to RGBA
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2RGBA)

            image_qimage = self.cv2_to_qimage(image_cv)
            self.image = QPixmap.fromImage(image_qimage)
        # Draw Grid if grid_checkbox is checked helps visualize how the nn sees the image
        if self.grid_checkbox.isChecked():
            # Initialize painter
            painter = QPainter(self.image)

            # Get the value from the slider, ranging from 16 to 1024
            grid_size = self.grid_slider.value()

            # Avoid division by zero
            if grid_size > 0:
                # Calculate step size based on the image size and grid size
                step_x = self.image.width() // grid_size
                step_y = self.image.height() // grid_size

                # Draw vertical lines
                for x in range(0, self.image.width(), step_x):
                    painter.drawLine(x, 0, x, self.image.height())

                # Draw horizontal lines
                for y in range(0, self.image.height(), step_y):
                    painter.drawLine(0, y, self.image.width(), y)

            painter.end()
        # Draw cropping area if display_roi is checked and crop_true is checked
        if self.crop_true.isChecked() and self.crop_true.isChecked():
            # Initialize painter
            painter = QPainter(self.image)

            h, w = self.image.height(), self.image.width()
            new_h, new_w = self.crop_width.value(), self.crop_height.value()
            start_x = w // 2 - new_w // 2
            start_y = h // 2 - new_h // 2

            # Create a QPen object for drawing
            pen = QPen()
            pen.setColor(Qt.red)  # You can set a different color
            pen.setWidth(3)  # Set line width

            # Set the pen to the painter
            painter.setPen(pen)

            # Draw rectangle
            painter.drawRect(start_x, start_y, new_w, new_h)

            # End painting
            painter.end()
        # Create a QGraphicsPixmapItem with the QPixmap
        pixmap_item = QGraphicsPixmapItem(self.image)
        pixmap_item.setTransformationMode(
            Qt.SmoothTransformation)  # Enable smooth scaling

        # Create a QGraphicsScene with the exact dimensions of the image
        scene = QGraphicsScene(0, 0, self.image.width(), self.image.height())

        # Add the QGraphicsPixmapItem to the scene
        scene.addItem(pixmap_item)

        # Enable anti-aliasing for smoother display
        self.screen_view.setRenderHint(QPainter.Antialiasing, True)
        self.screen_view.setRenderHint(QPainter.SmoothPixmapTransform, True)


        # after setting up the new scene
        self.set_screen_view_scene_and_rect(scene)

        # Update the fitInView_scale and zoom_scale for the new image
        self.screen_view.fitInView_scale = self.screen_view.transform().mapRect(
            QRectF(0, 0, 1, 1)).width()
        self.screen_view.zoom_scale = self.screen_view.fitInView_scale

        label_file = self.replace_extension_with_txt(file_name)
        self.create_empty_file_if_not_exists(label_file)
        self.label_file = label_file
        rects = self.load_bounding_boxes(
            label_file, self.image.width(), self.image.height())
        self.display_bounding_boxes(rects, file_name)

        if not rects:
            self.display_image_with_text(scene, self.image)

        return QPixmap.toImage(self.image)  # Keep the return statement at the end

    def get_fit_transform(self):
        view_rect = self.screen_view.viewport().rect()
        scale = min(view_rect.width() / self.original_pixmap_size.width(),
                    view_rect.height() / self.original_pixmap_size.height())
        return QTransform().scale(scale, scale)

    def set_screen_view_scene_and_rect(self, scene):
        self.view_references.append(self.screen_view)
        self.screen_view.setScene(scene)
        self.screen_view.fitInView(
            QRectF(0, 0, self.image.width(), self.image.height()), Qt.KeepAspectRatio)
        self.screen_view.setSceneRect(
            QRectF(0, 0, self.image.width(), self.image.height()))

    def replace_extension_with_txt(self, file_name):
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
            file_name = file_name.replace(ext, ".txt")
        return file_name

    def display_image_with_text(self, scene, pixmap):
        # Add the image border
        border_item = SolidGradientBorderItem(pixmap.width(), pixmap.height())
        scene.addItem(border_item)

        # Display the empty bounding box text
        empty_text_item = QGraphicsTextItem("BLANK")
        empty_text_item.setDefaultTextColor(QColor(0, 255, 0))

        # Set a futuristic font
        font = QFont("Arial", 16, QFont.Bold)
        empty_text_item.setFont(font)

        # Position the text at the bottom center of the pixmap
        empty_text_item.setPos(pixmap.width() / 2 - empty_text_item.boundingRect().width() / 2,
                               pixmap.height() - empty_text_item.boundingRect().height())

        # Create a drop shadow effect and apply it to the text
        effect = QGraphicsDropShadowEffect()
        effect.setOffset(2, 2)
        effect.setBlurRadius(20)
        effect.setColor(QColor(0, 255, 0, 200))
        empty_text_item.setGraphicsEffect(effect)

        # Add the text to the scene
        scene.addItem(empty_text_item)

        # Adjust the scene to fit the pixmap size
        scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

        # Store the initial border color
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
        rects = [item for item in self.screen_view.scene(
        ).items() if isinstance(item, BoundingBoxDrawer)]
        bounding_boxes = [BoundingBox.from_rect(QRectF(rect.rect().x(), rect.rect().y(), rect.rect().width(
        ), rect.rect().height()), img_width, img_height, rect.class_id, rect.confidence) for rect in rects]

        # Save to txt file without confidence
        try:
            with open(label_file, "w") as f:
                for bbox in bounding_boxes:
                    bbox_no_confidence = copy.copy(bbox)
                    bbox_no_confidence.confidence = None  # Remove confidence
                    f.write(bbox_no_confidence.to_str() + "\n")
        except FileNotFoundError as fnf_error:
            print(
                f"Error: {fnf_error}. No such file or directory: {label_file}")
            return

        # Initialize annotations dictionary for .json file
        annotations_dict = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": []
        }
        image_id = 0
        annotation_id = 0

        # one image per call
        annotations_dict["images"].append({
            "file_name": label_file,
            "height": img_height,
            "width": img_width,
            "id": image_id
        })

        # each bounding box is of a separate category
        for bbox in bounding_boxes:
            rect = bbox.to_rect(img_width, img_height)
            annotations_dict["annotations"].append({
                "area": rect.width() * rect.height(),
                "bbox": [rect.x(), rect.y(), rect.width(), rect.height()],
                "category_id": bbox.class_id,
                "id": annotation_id,
                "image_id": image_id,
                "iscrowd": 0,  # not a crowd
                "confidence": bbox.confidence  # adding confidence
            })
            annotations_dict["categories"].append({
                "id": bbox.class_id,
                # a class_id can be used as name
                "name": str(bbox.class_id)
            })
            annotation_id += 1

        # Save to json file
        label_file_json = label_file.replace('.txt', '.json')
        with open(label_file_json, "w") as f:
            json.dump(annotations_dict, f)

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
        confidence_threshold = 0.3  # Set this to your desired confidence threshold
        scene = self.screen_view.scene()
        if scene is not None:
            for item in scene.items():
                if isinstance(item, BoundingBoxDrawer):
                    for child in item.childItems():
                        if isinstance(child, QGraphicsTextItem):
                            # Set visibility based on both the checkbox and the confidence
                            should_display = not self.hide_labels and item.get_confidence() >= confidence_threshold
                            child.setVisible(should_display)
                            item.labels_hidden = not should_display  # existing line
                            item.update()  # existing line


    def auto_save_bounding_boxes(self):
        if self.label_file:
            self.save_bounding_boxes(self.label_file, self.screen_view.scene(
            ).width(), self.screen_view.scene().height())

    def set_selected(self, selected_bbox):
        if selected_bbox is not None:
            self.selected_bbox = selected_bbox
            # Update the class_input_field_label text
        else:
            print("Warning: Selected bounding box not found.")



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

    def update_classes_dropdown(self):
        try:
            self.classes_dropdown.clear()  # Clear the existing items before adding new ones

            classes_file_path = os.path.join(self.image_directory, "classes.txt")
            with open(classes_file_path, "r") as f:
                classes = [line.strip() for line in f.readlines()]

            self.class_to_id = {}  # Initialize the class-to-ID mapping
            for i, cls in enumerate(classes):
                self.classes_dropdown.addItem(cls)
                self.class_to_id[cls] = i  # Update the class-to-ID mapping
        except Exception as e:
            print(f"An error occurred: {e}")  # Basic error handling

    def class_input_field_return_pressed(self):
        if self.image_directory is None:
            print("Error: Image directory is not set. Please select an image directory.")
            # Optionally, prompt the user to select an image directory here
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
            # Handle the error - perhaps inform the user through the UI
            return  # Exit the function as we can't proceed without the file

        # Add the new class to the classes.txt file
        with open(classes_file_path, "a") as f:
            f.write("{}\n".format(new_class))

        # Clear the class input field
        self.class_input_field.clear()

        # Update the classes dropdown menu
        self.update_classes_dropdown()

        # Set the selected class in the classes dropdown menu
        self.classes_dropdown.setCurrentIndex(
            self.classes_dropdown.count() - 1)



    def remove_class_button_clicked(self):
        # Get the currently selected class from the dropdown
        selected_class = self.classes_dropdown.currentText()

        if selected_class:
            # Define the path to the classes file
            classes_file_path = os.path.join(self.image_directory, "classes.txt")

            # Read the existing classes from the classes file
            with open(classes_file_path, "r") as f:
                classes = [line.strip() for line in f.readlines()]

            if selected_class in classes:
                # Remove the selected class from the list of classes
                classes.remove(selected_class)

                # Write the updated classes back to the classes file
                with open(classes_file_path, "w") as f:
                    for cls in classes:
                        f.write("{}\n".format(cls))

                # Show an information message indicating successful removal
                QMessageBox.information(
                    self, "Information", "Class '{}' has been removed.".format(selected_class))
            else:
                # Show a warning message if the selected class does not exist
                QMessageBox.warning(
                    self, "Warning", "Class '{}' does not exist.".format(selected_class))
        else:
            # Show a warning message if no class is selected
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
        self.display_bounding_boxes(rects)

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
        with open(label_file, "r") as f:
            lines = f.readlines()

        self.labels = []
        for line in lines:
            components = line.strip().split()
            class_id = int(components[0])
            x_center, y_center, width, height = map(float, components[1:5])

            # Load confidence score if available
            if len(components) > 5:
                confidence = float(components[5])
            else:
                confidence = None

            self.labels.append(
                (class_id, x_center, y_center, width, height, confidence))

        # We'll increment this to 0 with the first call to get_next_label_rect
        self.current_label_index = -1

    def load_initial_file(self, img_file):
        txt_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(txt_file):
            self.current_file = img_file
            self.label_file = txt_file

        else:
            print(f"Label file not found for {img_file}.")

    def next_frame(self):
        if self.current_file:
            if self.current_file in self.filtered_image_files:
                index = self.filtered_image_files.index(self.current_file)
                checkbox_checked = self.hide_label_checkbox.isChecked()  # Store the checkbox state
                self.save_bounding_boxes(self.label_file, self.screen_view.scene().width(), self.screen_view.scene().height())
                next_file = self.filtered_image_files[index + 1] if index + 1 < len(self.filtered_image_files) else None

                if next_file:
                    # Update 'lastImage' in settings
                    self.settings['lastImage'] = next_file
                    self.saveSettings()  # Save the settings after modifying it

                    self.display_image(next_file)
                    self.img_index_number.setValue(index + 1)  # Update the QSpinBox

                    # Load labels for the new file
                    self.load_labels(self.label_file)
                    # Always start with the first label in the next frame
                    self.current_label_index = 0

                    if self.zoom_mode:
                        # Zoom to the first label of the current image
                        self.zoom_to_current_label()

                    if checkbox_checked:
                        # Toggle on to hide labels in next frame
                        self.hide_label_checkbox.setChecked(True)
                        self.toggle_label_visibility()  # Refresh label visibility again

                    self.current_file = next_file

                else:
                    QMessageBox.information(self, "Information", "You have reached the end of the list.")
            else:
                QMessageBox.warning(self, "Warning", "Current file not found in the filtered list.")
        else:
            QMessageBox.warning(self, "Warning", "No current file selected.")

    def previous_frame(self):
        if self.current_file:
            if self.current_file in self.filtered_image_files:
                index = self.filtered_image_files.index(self.current_file)
                checkbox_checked = self.hide_label_checkbox.isChecked()  # Store the checkbox state
                prev_file = self.filtered_image_files[index - 1] if index - 1 >= 0 else None

                if prev_file:
                    # Update 'lastImage' in settings
                    self.settings['lastImage'] = prev_file
                    self.saveSettings()  # Save the settings after modifying it

                    self.display_image(prev_file)
                    self.img_index_number.setValue(index - 1)

                    # Load labels for the new file
                    self.load_labels(self.label_file)
                    # Always start with the first label in the previous frame
                    self.current_label_index = 0

                    if self.zoom_mode:
                        # Zoom to the first label of the current image
                        self.zoom_to_current_label()

                    if checkbox_checked:
                        # Toggle on to hide labels in previous frame
                        self.hide_label_checkbox.setChecked(True)
                        self.toggle_label_visibility()  # Refresh label visibility again

                    self.current_file = prev_file

                else:
                    QMessageBox.information(self, "Information", "You have reached the beginning of the list.")
            else:
                QMessageBox.warning(self, "Warning", "Current file not found in the filtered list.")
        else:
            QMessageBox.warning(self, "Warning", "No current file selected.")


    def start_next_timer(self):
        # Adjust the interval (in milliseconds) to control the speed
        self.next_timer.start(100)

    def stop_next_timer(self):
        self.next_timer.stop()

    def start_prev_timer(self):
        # Adjust the interval (in milliseconds) to control the speed
        self.prev_timer.start(100)

    def stop_prev_timer(self):
        self.prev_timer.stop()

    # Map Plot Code

    def on_plot_labels_clicked(self):
               # Check if an image directory has been selected
        if not hasattr(self, 'image_directory') or self.image_directory is None:
            print(
                "No image directory selected. Please select an image directory before plotting labels.")
            return

        self.create_plot()

    def load_yolo_labels(self, file_path):
        # Initialize lists to store center points, areas, and label classes
        list_of_center_points_and_areas = []
        label_classes = []

        # Generate the corresponding YOLO label file path based on the image file path
        txt_file_path = file_path[:-4] + ".txt"

        # Check if the YOLO label file exists and is not empty
        if os.path.exists(txt_file_path) and os.path.getsize(txt_file_path) != 0:
            with open(txt_file_path, "r") as f:
                # Read lines from the YOLO label file
                boxes = [line.strip() for line in f]

                # Parse each line to extract class, center point, and area information
                for box in boxes:
                    box = [float(val) for val in box.split()]
                    label_class = int(box[0])  # The first element is the class
                    label_classes.append(label_class)

                    # Calculate the center point and area size from box coordinates
                    center_point = (box[1] + box[3] / 2, box[2] + box[4] / 2)
                    area_size = box[3] * box[4]

                    # Store center point and area size in the list
                    list_of_center_points_and_areas.append((center_point, area_size))

        return list_of_center_points_and_areas, label_classes


    def load_all_yolo_labels(self, directory_path):
        # Create a list of image file paths in the specified directory with .jpg or .png extensions
        images = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(
            os.path.join(directory_path, f)) and (f.endswith(".jpg") or f.endswith(".png"))]

        # Initialize lists to store all center points, areas, and label classes
        all_center_points_and_areas = []
        all_label_classes = []

        # Use a ThreadPoolExecutor to parallelize loading YOLO labels for multiple images
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.load_yolo_labels, image) for image in images]

        # Iterate through the completed futures to collect results
        for future, image in zip(futures, images):
            try:
                result = future.result()
                if result is not None:
                    center_points_and_areas, label_classes = result

                    # Extend the lists with data from the current image
                    all_center_points_and_areas.extend(center_points_and_areas)
                    all_label_classes.extend(label_classes)
            except Exception as e:
                # Handle errors while processing individual files
                print(f"An error occurred while processing file {image}: {e}")

        # Sort the center points and areas by area size in descending order
        all_center_points_and_areas.sort(key=lambda x: x[1], reverse=True)

        return all_center_points_and_areas, all_label_classes


    def create_plot(self):
        if not hasattr(self, 'image_directory'):
            print("No image directory selected.")
            return

        directory_path = self.image_directory
        graphs_folder = os.path.join(directory_path, 'graphs')
        # Create the "graphs" subfolder if it doesn't exist
        os.makedirs(graphs_folder, exist_ok=True)
        print("Before calling load_all_yolo_labels()")
        list_of_center_points_and_areas, all_label_classes = self.load_all_yolo_labels(
            directory_path)
        print("After calling load_all_yolo_labels()")

        x, y = np.array([i[0][0] for i in list_of_center_points_and_areas]), np.array(
            [i[0][1] for i in list_of_center_points_and_areas])
        colors = np.array([i[1] for i in list_of_center_points_and_areas])
        sizes = (colors - colors.min()) / (colors.max() - colors.min()
                                           ) * 50 + 10  # Normalize area sizes for marker size

        plot_counter_mod = self.plot_counter % 3
        if plot_counter_mod == 0:
            # Bar plot
            from collections import Counter
            class_counts = Counter(all_label_classes)
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            fig, ax = plt.subplots()
            class_counts = Counter(all_label_classes)
            classes = list(class_counts.keys())
            counts = list(class_counts.values())

            # Create a list of colors
            colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

            # Pass the list of colors to the bar plot
            ax.bar(classes, counts, color=colors)

            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title(f"Label Class Distribution")

            # Save the plot to the "graphs" folder
            plt.savefig(os.path.join(graphs_folder, "label_class_distribution.png"))
        elif plot_counter_mod == 1:
            # Scatter plot
            fig, ax = plt.subplots()
            scatter = ax.scatter(
                x, y, c=colors, cmap='viridis', alpha=0.7, s=sizes)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.colorbar(scatter, label='Label Area', orientation='vertical')
            plt.title(f"Label Count: {len(list_of_center_points_and_areas)}")
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
        elif plot_counter_mod == 2:
            # Histogram
            fig, ax = plt.subplots()
            ax.hist(colors, bins=50, color='skyblue', edgecolor='black')
            plt.xlabel('Label Area')
            plt.ylabel('Count')
            plt.title(f"Distribution of Label Areas")
        plt.tight_layout()

        plot_file = os.path.join(
            graphs_folder, f'label_plot_{self.plot_counter}.png')
        plt.savefig(plot_file)
        plt.show()  # Show the plot

        self.plot_counter += 1


# adjust and view labels for a single image

    def update_preview(self):
        if not self.image_files or not self.label_files:
            return

        x_adjustment = self.x_axis.value()
        y_adjustment = self.y_axis.value()
        h_adjustment = self.height_pos.value()
        w_adjustment = self.width_position.value()

        image_file = self.image_files[0]
        label_file = self.label_files[0]

        img = cv2.imread(image_file)
        # Convert the image from BGR to RGB color format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        x_adjustment_frac = x_adjustment / width
        y_adjustment_frac = y_adjustment / height
        h_adjustment_frac = h_adjustment / height
        w_adjustment_frac = w_adjustment / width

        with open(label_file, 'r') as f:
            labels = [line.strip().split(' ') for line in f.readlines()]

        adjusted_labels = []
        for i, label in enumerate(labels):
            adjusted_label = [str(x) for x in [label[0],
                                               max(min(
                                                   float(label[1]) + x_adjustment_frac, 1.0), 0.0),
                                               max(min(
                                                   float(label[2]) + y_adjustment_frac, 1.0), 0.0),
                                               max(float(
                                                   label[3]) + h_adjustment_frac, 0.0),
                                               max(float(label[4]) + w_adjustment_frac, 0.0)]]
            adjusted_labels.append(adjusted_label)

            # Draw bounding boxes on the image
            class_id, x_center, y_center, box_width, box_height = [
                float(x) for x in adjusted_label]
            if box_width > 0 and box_height > 0:  # only draw if width and height are positive
                x_min = max(int((x_center - box_width / 2) * width), 0)
                y_min = max(int((y_center - box_height / 2) * height), 0)
                x_max = min(int((x_center + box_width / 2) * width), width - 1)
                y_max = min(
                    int((y_center + box_height / 2) * height), height - 1)
                cv2.rectangle(img, (x_min, y_min),
                              (x_max, y_max), (0, 255, 0), 2)

        # Convert the image to QPixmap and display it
        qimage = QImage(
            img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # Create a QGraphicsPixmapItem with the QPixmap
        pixmap_item = QGraphicsPixmapItem(pixmap)
        pixmap_item.setTransformationMode(
            Qt.SmoothTransformation)  # Enable smooth scaling

        # Create a QGraphicsScene with the exact dimensions of the image
        scene = QGraphicsScene(0, 0, pixmap.width(), pixmap.height())

        # Add the QGraphicsPixmapItem to the scene
        scene.addItem(pixmap_item)

        # Set the scene to screen_view
        self.screen_view.setScene(scene)

    def adjust_and_show_message(self):
        self.adjust_labels()
        self.show_adjustment_message()

    def load_images_and_labels(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if not folder:
            return

        image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        self.image_files = [img for ext in image_exts for img in glob.glob(
            os.path.join(folder, ext))]
        self.label_files = [os.path.splitext(
            img)[0] + '.txt' for img in self.image_files]
        self.label_files = [
            lbl for lbl in self.label_files if 'classes.txt' not in lbl]
        # Update the preview with the first image and label
        self.update_preview()

    def adjust_labels(self):
        x_adjustment = self.x_axis.value()
        y_adjustment = self.y_axis.value()
        h_adjustment = self.height_pos.value()
        w_adjustment = self.width_position.value()

        for i, (image_file, label_file) in enumerate(zip(self.image_files, self.label_files)):
            img = Image.open(image_file)
            width, height = img.size
            x_adjustment_frac = x_adjustment / width
            y_adjustment_frac = y_adjustment / height
            h_adjustment_frac = h_adjustment / height
            w_adjustment_frac = w_adjustment / width

            with open(label_file, 'r') as f:
                labels = [line.strip().split(' ') for line in f.readlines()]

            # Iterate over all labels and adjust each one
            for i, label in enumerate(labels):
                labels[i] = [str(x) for x in [label[0],
                                              float(label[1]) +
                                              x_adjustment_frac,
                                              float(label[2]) +
                                              y_adjustment_frac,
                                              float(label[3]) +
                                              h_adjustment_frac,
                                              float(label[4]) + w_adjustment_frac]]

            with open(label_file, 'w') as f:
                for label in labels:
                    f.write(' '.join(label) + '\n')

    def show_adjustment_message(self):
        x_adjustment = self.x_axis.value()
        y_adjustment = self.y_axis.value()
        h_adjustment = self.height_pos.value()
        w_adjustment = self.width_position.value()

        msg = QMessageBox()
        msg.setWindowTitle("Labels Adjusted")
        msg.setText(f"Labels have been adjusted with the following input:\n\n"
                    f"x_adjustment: {x_adjustment}\n"
                    f"y_adjustment: {y_adjustment}\n"
                    f"h_adjustment: {h_adjustment}\n"
                    f"w_adjustment: {w_adjustment}\n")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    # part of the crop function adds noise to background

    def apply_glass_effect(self, image):
        try:
            # Apply a higher-quality Gaussian blur for the bubbled glass effect.
            blurred = cv2.GaussianBlur(image, (15, 15), 0)
            # Create the bubbled image
            bubbled = cv2.GaussianBlur(image, (21, 21), 30)

            # Use a larger kernel size for the streaks pattern to create more grain.
            streaks = np.random.rand(*image.shape[:2]) * 255
            streaks = cv2.GaussianBlur(streaks, (31, 31), 30).astype(np.uint8)

            # Use a higher threshold for the streaks pattern to create fewer streaks.
            _, streaks = cv2.threshold(streaks, 220, 255, cv2.THRESH_BINARY)

            # Convert the single-channel 'streaks' image to a 3-channel image by stacking it along the channel axis.
            streaks = np.stack([streaks, streaks, streaks], axis=-1)

            # Use a smaller kernel size for the specular highlight to create a more focused highlight.
            glossy = np.random.rand(*image.shape[:2]) * 255
            glossy = cv2.GaussianBlur(glossy, (11, 11), 60).astype(np.uint8)

            # Use a higher threshold for the specular highlight to create a brighter highlight.
            _, glossy = cv2.threshold(glossy, 250, 255, cv2.THRESH_BINARY)

            # Convert the single-channel 'glossy' image to a 3-channel image by stacking it along the channel axis.
            glossy = np.stack([glossy, glossy, glossy], axis=-1)

            # Blend the original image, the blurred image, the bubbled image, the streaks, and the glossy effect.
            alpha = 0.2
            beta = 0.2
            gamma = 0.05
            delta = 0.05
            epsilon = 0.05
            result = cv2.addWeighted(
                image, 1 - alpha - beta - gamma - delta, blurred, alpha, 0)
            result = cv2.addWeighted(
                result, 1 - beta - gamma - delta, bubbled, beta, 0)
            result = cv2.addWeighted(
                result, 1 - gamma - delta, streaks, gamma, 0)
            result = cv2.addWeighted(result, 1 - delta, glossy, delta, 0)

            # Create a reflection-like effect.
            reflection = np.random.rand(*image.shape[:2]) * 255
            reflection = cv2.GaussianBlur(
                reflection, (21, 21), 30).astype(np.uint8)

            # Convert the single-channel 'reflection' image to a 3-channel image by stacking it along the channel axis.
            reflection = np.stack(
                [reflection, reflection, reflection], axis=-1)

            # Blend the result with the reflection.
            zeta = 0.05
            result = cv2.addWeighted(result, 1 - zeta, reflection, zeta, 0)

            # Apply a different color tint for the final image.
            tint = np.array([200, 200, 255], dtype=np.uint8)
            tint = np.full_like(result, tint)
            result = cv2.addWeighted(result, 0.8, tint, 0.2, 0)

            return result

        except cv2.error as e:
            print(f"An OpenCV error occurred: {e}")
            print("Please ensure the input image is not empty.")
            return None  # or you can return the original image if you prefer: return image

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None  # or you can return the original image if you prefer: return image

    def glass_checkbox_state_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.glass_effect = True
        else:
            self.glass_effect = False

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
        assert image is not None, "Image is None"
        assert 0.0 <= intensity <= 1.0, "Intensity should be between 0.0 and 1.0"
        assert image.dtype == np.uint8, "Image data type should be uint8"

        flash_mask = np.full(image.shape, (255, 255, 255), dtype=np.float32)
        flash_effect_alpha = intensity
        flash_image = cv2.addWeighted(image.astype(
            np.float32), 1 - flash_effect_alpha, flash_mask, flash_effect_alpha, 0)

        assert flash_image is not None, "Flash image is None"

        return flash_image.astype(np.uint8)

    def flash_checkbox_changed(self, state):
        self.flash_effect = state == Qt.Checked
        print(f"Flash effect changed: {self.flash_effect}")

    def apply_smoke_effect(self, image, intensity=0.1, radius_ratio=1.5, speed=0.1):
        h, w, _ = image.shape

        # Increment the time for the rolling effect
        self.time += speed

        # Create multiple smoke masks with varying size and opacity
        smoke_masks = []
        for scale_factor in np.linspace(0.5, 2.0, num=4):
            smoke_mask = self.generate_smoke_mask(
                h, w, scale_factor=scale_factor, time_offset=self.time)
            smoke_masks.append(smoke_mask * intensity)

        # Combine the masks and normalize
        combined_mask = np.sum(smoke_masks, axis=0)
        if np.count_nonzero(combined_mask) == 0:
            # The combined mask is empty, return an empty array
            return np.zeros_like(image)
        else:
            combined_mask = cv2.normalize(combined_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


        # Apply Gaussian blur to make the smoke effect smoother
        blur_radius = max(1, int(min(w, h) / 100))
        if blur_radius % 2 == 0:
            blur_radius += 1  # Make sure the radius is an odd number
        if blur_radius > 1:
            combined_mask = cv2.GaussianBlur(
                combined_mask, (blur_radius, blur_radius), 0)

        # Check if combined_mask is not empty
        if combined_mask.any():
            smoke_mask = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        else:
            smoke_mask = np.zeros_like(image)

        # Apply the smoke effect using an alpha blend of the original image and the smoke mask
        smoke_effect_alpha = 0.6
        smokey_image = cv2.addWeighted(image.astype(np.float32), 1 - smoke_effect_alpha, smoke_mask.astype(np.float32), smoke_effect_alpha, 0)


        return smokey_image.astype(np.uint8)

    def generate_smoke_mask(self, h, w, scale_factor=1.0, time_offset=0.0):
        # Initialize an empty noise array with the given height and width
        noise = np.zeros((h, w), dtype=np.float32)

        # Define parameters for Perlin noise generation
        octaves = 6  # Number of octaves in Perlin noise
        persistence = 0.5  # Persistence factor for Perlin noise
        lacunarity = 2.0  # Lacunarity factor for Perlin noise

        # Generate Perlin noise values for each pixel in the array
        for y in range(h):
            for x in range(w):
                noise[y, x] = perlin2(x / (20 * scale_factor) + time_offset,
                                    y / (20 * scale_factor), base=0)

        # Create a binary mask based on the Perlin noise values (values > 0.5 become True)
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

        outer_mask = dist_from_center <= outer_radius
        inner_mask = dist_from_center >= inner_radius

        circle_mask = np.logical_and(outer_mask, inner_mask)

        # Create the vertical line from the top to the bottom of the circle
        vertical_line = np.logical_and(center[1] - outer_radius <= Y, Y <= center[1] + outer_radius)
        # Create the horizontal line from left to right inside the circle
        horizontal_line = np.logical_and(center[0] - outer_radius <= X, X <= center[0] + outer_radius)

        vertical_thick_line = np.logical_and(center[0] - line_thickness <= X, X <= center[0] + line_thickness)
        horizontal_thick_line = np.logical_and(center[1] - line_thickness <= Y, Y <= center[1] + line_thickness)

        vertical_crosshair = np.logical_and(vertical_line, vertical_thick_line)
        horizontal_crosshair = np.logical_and(horizontal_line, horizontal_thick_line)

        # Combine circle and crosshair
        mask = circle_mask | vertical_crosshair | horizontal_crosshair

        return mask

    def set_flip_images(self, state):
        self.flip_images = state
        
    def mosaic_checkbox_changed(self, state):
        # Update the mosaic effect based on the checkbox state
        self.mosaic_effect = (state == Qt.Checked)
        
    def pack_thumbnails(self, thumbnail_paths, canvas_size):
        # Create a packer
        packer = newPacker(rotation=True)  # Allow rotation

        # Add the canvas (bin)
        packer.add_bin(*canvas_size)

        # Add each thumbnail (rectangle) to the packer
        for path in thumbnail_paths:
            img = Image.open(path)
            packer.add_rect(*img.size, rid=path)

        # Start packing
        packer.pack()

        # Get packed rectangles
        packed_thumbnails = []
        for rect in packer.rect_list():
            b, x, y, w, h, rid = rect
            packed_thumbnails.append((rid, x, y, w, h))

        return packed_thumbnails

    def create_mosaic_images(self):
        # Make sure necessary attributes are set
        thumbnail_dir = self.thumbnail_dir
        canvas_size = (self.augmentation_size, self.augmentation_size)  # Adjusted to be a tuple
        output_dir = os.path.join(thumbnail_dir, 'mosaic')
        max_images = self.augmentation_count

        # Define thumbnail_paths with more extensions
        extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
        thumbnail_paths = [os.path.join(thumbnail_dir, f) for f in os.listdir(thumbnail_dir)
                           if os.path.splitext(f)[1].lower() in extensions]

        print(f"Found {len(thumbnail_paths)} thumbnails.")
        if not thumbnail_paths:
            print("No thumbnails found. Please check the directory.")
            return

        random.shuffle(thumbnail_paths)

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory {output_dir}")

        used_thumbnails = set()
        image_count = 0
        while thumbnail_paths and (max_images is None or image_count < max_images):
            packed_thumbnails = self.pack_thumbnails(thumbnail_paths, canvas_size)
            if not packed_thumbnails:
                print("No more thumbnails can fit on the canvas.")
                break

            # Initialize the canvas
            canvas = Image.new('RGB', canvas_size, color='white')

            # Place thumbnails on the canvas
            for rid, x, y, w, h in packed_thumbnails:
                thumbnail = Image.open(rid)
                if thumbnail.size != (w, h):
                    thumbnail = thumbnail.rotate(90, expand=True)
                canvas.paste(thumbnail, (x, y))
                used_thumbnails.add(rid)

            thumbnail_paths = list(set(thumbnail_paths) - used_thumbnails)

            output_image_path = os.path.join(output_dir, f"mosaic_{image_count:04d}.png")
            output_annotation_path = os.path.join(output_dir, f"mosaic_{image_count:04d}.txt")
            canvas.save(output_image_path)
           
            print(f"Saved {output_image_path}")
            
            # Write YOLO annotations for each thumbnail
            with open(output_annotation_path, 'w') as file:
                for rid, x, y, w, h in packed_thumbnails:
                    x_center = (x + w / 2) / canvas_size[0]
                    y_center = (y + h / 2) / canvas_size[1]
                    width = w / canvas_size[0]
                    height = h / canvas_size[1]
                    file.write(f"0 {x_center} {y_center} {width} {height}\n")

            image_count += 1

            if not thumbnail_paths:
                thumbnail_paths = list(used_thumbnails)
                random.shuffle(thumbnail_paths)
                used_thumbnails.clear()

        print(f"All done. Mosaic images created: {image_count}")
        return image_count  # Add this line to return the count





    def set_progress_bar_maximum(self, max_value):
        self.progress_aug.setMaximum(max_value)



    def process_images_triggered(self):
        print("process_images_triggered called")

        if not self.images_import:
            QMessageBox.warning(self, "Error", "No images to process.")
            return

        output_directory = os.path.dirname(self.images_import[0])
        selected_effects = []

        # Check for mosaic effect and other effects
        if self.mosaic_effect:
            print("Creating mosaic images")
            processed_images_count = self.create_mosaic_images()  # Assign the returned count here
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
        processed_images_count = 0
        for current_image, (image_base_name, image_path) in enumerate(image_files_map.items()):
            label_path = label_files_map.get(image_base_name)
            if label_path is not None and os.path.basename(label_path) != "classes.txt":
                self.apply_augmentations(current_image, image_path, label_path, output_path, output_folder_name, total_images)
                processed_images_count += 1

        QMessageBox.information(self, "Success", f"{processed_images_count} images have been successfully processed.")


    def apply_augmentations(self, current_image, image_path, label_path, output_path, output_folder_name, total_images):
        image = cv2.imread(image_path)
        with open(label_path, 'r') as f:
            labels = f.readlines()


        # Flip the image and labels if checked.
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
            # Parse the first label's information.
            first_label_info = labels[0].strip().split()
            class_id, x_center, y_center, width, height = map(float, first_label_info)

            # Calculate actual x_center and y_center in pixels
            h, w, _ = image.shape
            actual_x_center = int(x_center * w)
            actual_y_center = int(y_center * h)

            # Apply circular sight picture if checked.
            if self.sight_picture:
                center = (actual_x_center, actual_y_center)  # Center over the first labeled object
                mask = self.create_circular_mask(image, center, crosshair_length=50)

                # Where mask is True, make those pixels black.
                image[mask] = [0, 0, 0]  # R, G, B
        # Apply smoke effect if checked.
        if self.smoke_effect:
            image = self.apply_smoke_effect(image)
            if image is None:
                return

        # Apply flash effect if checked.
        if self.flash_effect:
            image = self.apply_flashbang_effect(image)
            if image is None:
                return

        # Apply motion blur effect if checked.
        if self.motion_blur_effect:
            image = self.apply_motion_blur_effect(image)
            if image is None:
                return

        # Apply glass effect if checked.
        if self.glass_effect:
            image = self.apply_glass_effect(image)
            if image is None:
                return

        # Save the new image in the determined output folder
        output_image_path = os.path.join(output_path, f"{output_folder_name}_{current_image}.jpg")
        cv2.imwrite(output_image_path, image)

        # Save YOLO label file
        output_label_path = os.path.join(output_path, f"{output_folder_name}_{current_image}.txt")
        with open(output_label_path, 'w') as f:
            f.writelines(labels)



        # Update the progress.
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
                 

    # import stylesheet
    def populate_style_combo_box(self):
        # Your existing code to fetch custom styles
        style_folder = QDir.currentPath() + '/styles'
        style_files = [file for file in os.listdir(style_folder) if file.endswith(('.qss', '.css', '.ess', '.stylesheet'))]

        # Get the list of qt-material themes
        qt_material_styles = list_themes()
        style_files.extend(qt_material_styles)

        self.styleComboBox.addItems(style_files)


    def apply_stylesheet(self):
        selected_style = self.styleComboBox.currentText()
        if selected_style == "Default":
            # Apply the default GUI style
            self.setStyleSheet("")  # Set an empty style sheet
        elif selected_style in list_themes():  # Check if it's a qt-material theme
            apply_stylesheet(self.app, selected_style)
        else:
            style_folder = QDir.currentPath() + '/styles'
            file_path = os.path.join(style_folder, selected_style)
            with open(file_path, 'r', encoding="utf-8") as f:
                stylesheet = f.read()
            self.setStyleSheet(stylesheet)

        # def for ultralytics train

    def browse_yaml_clicked(self):
        file_name = self.open_file_dialog(
            "Select YAML File", "YAML Files (*.yaml);;All Files (*)")
        self.yaml_path = file_name
        self.yaml_label.setText(f"{file_name}")

    def browse_pt_clicked(self):
        file_name = self.open_file_dialog(
            "Select Model File", "Model Files (*.pt *.yaml);;All Files (*)")
        if file_name.endswith('.pt'):
            self.pt_path = file_name
            self.pt_label.setText(f" {file_name}")
        elif file_name.endswith('.yaml'):
            self.pt_path = file_name
            self.pt_label.setText(f"{file_name}")


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
        # Initialize variables
        data_input = None
        model_file = getattr(self, 'pt_path', None)
        
        # If resume_checkbox is checked, handle the resume functionality
        if self.resume_checkbox.isChecked():
            # Check if a model file has been selected; if not, show a warning.
            if not model_file or not model_file.endswith('.pt'):
                QMessageBox.warning(self, "No Model File Selected", "Please select a .pt file to resume training.")
                return
            else:
                normalized_model_path = model_file.replace("\\", "/")
                # Form the resume command
                command = f"yolo train resume model={normalized_model_path}"
                print("Resuming training process with command:", command)
        else:
            # Check if a YAML file has been selected; if not, show a warning.
            try:
                data_input = self.yaml_path
            except AttributeError:
                QMessageBox.warning(self, "No YAML File Selected", "Please select a YAML file before proceeding.")
                return

            normalized_model_path = model_file.replace("\\", "/")
            model_arg = f"model={normalized_model_path}"

            # Retrieve input values
            imgsz_input = self.imgsz_input.text()
            epochs_input = self.epochs_input.text()
            batch_input = self.batch_input.text()
            custom_project_name = "weights"  # Replace with your desired project name

            # Form the command string for training
            command = f"yolo detect train data={data_input} imgsz={imgsz_input} epochs={epochs_input} " \
                    f"batch={batch_input} {model_arg} project={self.runs_directory}"

            # Additional arguments based on user input
            if self.half_true.isChecked():
                command += " half=True"
            if self.amp_true.isChecked():
                command += " amp=True"
            if self.freeze_checkbox.isChecked():
                freeze_layers = self.freeze_input.value()  
                command += f" freeze={freeze_layers}"               
            if self.patience_checkbox.isChecked():
                patience_value = self.patience_input.text()  # assuming patience_input is a QLineEdit or similar
                command += f" patience={patience_value}"
            print("Starting training process with command:", command)

        # Initialize and start the training process
        train_process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)
        train_process.stdin.write(os.linesep.encode())  # Send Enter key press
        train_process.stdin.flush()

        # Automatically start TensorBoard if the checkbox is checked
        if self.tensorboardCheckbox.isChecked():
            tensorboard_log_dir = os.path.join(self.runs_directory, "train")  # Adjust this if necessary
            tb_command = f"tensorboard --logdir {tensorboard_log_dir}"
            tensorboard_process = subprocess.Popen(tb_command, shell=True, stdout=subprocess.PIPE)

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

        # Get image size parameters for height and width
        imgsz_height = self.imgsz_input_H.text()
        imgsz_width = self.imgsz_input_W.text()

        # Convert imgsz to a list containing two integers (height and width)
        try:
            imgsz_list = [int(imgsz_height), int(imgsz_width)]
        except ValueError:
            QMessageBox.critical(self, "Conversion Error", f"Invalid image size values: Height={imgsz_height}, Width={imgsz_width}")
            return

        # Get additional parameters based on the checkbox states
        half = self.half_true.isChecked()
        int8 = self.int8_true.isChecked()
        simplify = self.simplify.isChecked()
        dynamic = self.dynamic_true.isChecked()
        use_gpu = self.use_gpu_checkbox.isChecked()

        # Prepare parameters for the export method
        export_params = {
            'format': format_selected,
            'imgsz': imgsz_list,
            'half': half,
            'int8': int8,
            'simplify': simplify,
            'dynamic': dynamic,
            'device': 0 if use_gpu else 'cpu'  # Use GPU if checked, otherwise CPU
        }

        # Log the conversion command
        logging.info(f"Converting model to {format_selected} with parameters: {export_params}")

        # Prepare parameters for the export method
        export_params = {'format': format_selected, 'imgsz': imgsz_list}

        # Conditionally add parameters if they are supported and selected
        if self.is_parameter_supported(format_selected, 'half') and half:
            export_params['half'] = half
        if self.is_parameter_supported(format_selected, 'int8') and int8:
            export_params['int8'] = int8
        if self.is_parameter_supported(format_selected, 'simplify') and simplify:
            export_params['simplify'] = simplify
        if self.is_parameter_supported(format_selected, 'dynamic') and dynamic:
            export_params['dynamic'] = dynamic
        if self.is_parameter_supported(format_selected, 'device'):
            export_params['device'] = 0 if use_gpu else 'cpu'  # Use GPU if checked, otherwise CPU

        try:
            self.model.export(**export_params)
            QMessageBox.information(self, "Conversion Successful", f'Model converted to {format_selected}')
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", str(e))

    def is_parameter_supported(self, format_selected, param):
        supported_params = {
            'torchscript': {'optimize'},  # 'imgsz' is always supported
            'onnx': {'half', 'dynamic', 'simplify', 'opset'},  # 'imgsz' is always supported
            'openvino': {'half', 'int8'},  # 'imgsz' is always supported
            'engine': {'half', 'dynamic', 'simplify', 'workspace'},  # 'imgsz' is always supported
            'coreml': {'half', 'int8', 'nms'},  # 'imgsz' is always supported
            'saved_model': {'keras', 'int8'},  # 'imgsz' is always supported
            'pb': set(),  # 'imgsz' is always supported, no other parameters
            'tflite': {'half', 'int8'},  # 'imgsz' is always supported
            'edgetpu': set(),  # 'imgsz' is always supported, no other parameters
            'tfjs': {'half', 'int8'},  # 'imgsz' is always supported
            'paddle': set(),  # 'imgsz' is always supported, no other parameters
            'ncnn': {'half'},  # 'imgsz' is always supported
            # Add any other formats as necessary
        }
        return param in supported_params.get(format_selected, set())




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

            # If 'train.txt' is selected, remove existing configuration files, but keep 'valid.txt'
            if selected_option == "train.txt":
                for file_name in ["obj.data", "obj.names", "obj.yaml", "train.txt",]:
                    file_path = os.path.join(output_dir, file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
            
            # Save the image paths to the specified 'train.txt' or 'valid.txt'
            with open(save_file, "w", encoding="utf-8") as f:
                for image in self.images:
                    f.write(image + "\n")

            # Check if we are creating a 'valid.txt' file and if 'train.txt' already exists
            train_txt_path = os.path.join(output_dir, "train.txt")
            is_valid_txt = selected_option == "valid.txt"
            if is_valid_txt and not os.path.exists(train_txt_path):
                # If 'valid.txt' is selected but 'train.txt' does not exist, inform the user
                QMessageBox.warning(self, 'Warning', 'The "train.txt" file does not exist. Please create it first.')
                return

            # Load existing class names from 'obj.names' if available
            obj_names_file = os.path.join(output_dir, "obj.names")
            if os.path.exists(obj_names_file):
                with open(obj_names_file, 'r', encoding="utf-8") as f:
                    self.classes = [line.strip() for line in f.readlines()]
            else:
                # If 'obj.names' does not exist and we are creating 'valid.txt', we should not proceed
                if is_valid_txt:
                    QMessageBox.warning(self, 'Warning', 'The "obj.names" file does not exist. Please create it first.')
                    return
                # Else, create a new 'obj.names' file
                with open(obj_names_file, "w", encoding="utf-8") as f:
                    for class_name in self.classes:
                        f.write(class_name + "\n")

            # Update 'obj.data'
            class_numb = len(self.classes)
            data_file_path = os.path.join(output_dir, "obj.data")
            valid_path = os.path.join(output_dir, "valid.txt") if is_valid_txt else train_txt_path
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

            # Save settings
            self.settings["output_dir"] = output_dir
            self.settings["obj_names_file"] = obj_names_file
            self.settings["data_file_path"] = data_file_path
            self.settings["obj_yaml_file"] = obj_yaml_file
            self.settings["backup_dir"] = backup_dir
            self.saveSettings()  # Assuming you have defined a method to save these settings

            # Show confirmation message
            QMessageBox.information(self, 'Information', 'Output files have been created!')


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

    def train_button_clicked(self):
        data_file = self.file_paths["data"]
        cfg_file = self.file_paths["cfg"]
        weights_file = self.file_paths["weights"][-1] if self.file_paths["weights"] else ""

        command = f"darknet detector train {data_file} {cfg_file} "

        if weights_file:
            command += f" {weights_file}"

        # The rest of your flags as per your original function

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

        print(f"Command to be executed: {command}")

        # A 'self.results' is your QCheckBox for writing the log
        write_log = self.results.isChecked()
        # Instantiate the TrainingThread with the next_map QLabel
        self.training_thread = TrainingThread(
            command, write_log, self.next_map)  # Pass next_map here
        self.training_thread.signals.update_stats.connect(
            self.update_gui_stats)
        self.training_thread.start()

        # Load and display the chart image
        chart_file = "chart.png"
        pixmap = QPixmap(chart_file)
        if pixmap.isNull():
            # Image file doesn't exist yet, handle accordingly
            print(f"Chart image '{chart_file}' not found.")
        else:
            # Image file exists, display it in image_label_2
            self.image_label_2.setPixmap(pixmap)
            self.image_label_2.setScaledContents(True)

    # see map loss progress in gui
    def get_total_images(self, data_file):
        train_txt_path = None
        with open(data_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if "train" in line:
                train_txt_path = line.split("=")[-1].strip()
                break

        if not train_txt_path:
            print(f"train not found in {data_file}")
            return

        with open(train_txt_path, 'r') as f:
            num_images = len(f.readlines())

        return num_images

    def get_max_iterations_from_cfg(self):
        cfg_file = self.file_paths.get("cfg")

        if not cfg_file:
            print("Cfg file not found.")
            return

        max_iterations = None
        with open(cfg_file, "r") as f:
            cfg_lines = f.readlines()

        for line in cfg_lines:
            if "max_batches" in line:
                max_iterations = int(line.split("=")[-1].strip())
                break

        if max_iterations is None:
            print("max_batches not found in cfg file")
            return

        return max_iterations

    def update_gui_stats(self, output_line):
        print("update_gui_stats called with:", output_line)

        # For mAP:
        map_pattern = re.compile(
            r"mean average precision \(mAP@0\.50\) = (\d+(\.\d+)?)")
        map_match = map_pattern.search(output_line)
        if map_match:
            map_value = float(map_match.group(1)) * \
                100  # Convert to percentage
            self.map_label.setText(f"MAP: {map_value:.2f}%")

            if map_value < 50:
                self.map_label.setStyleSheet("color: red")
            elif 50 <= map_value < 70:
                self.map_label.setStyleSheet("color: orange")
            else:
                self.map_label.setStyleSheet("color: green")

        # For next mAP calculation iteration
        next_map_pattern = re.compile(
            r"next mAP calculation at (\d+) iterations")
        next_map_match = next_map_pattern.search(output_line)
        if next_map_match:
            next_map_iteration = int(next_map_match.group(1).strip())
            max_iterations = self.get_max_iterations_from_cfg()
            next_map_percentage = int(
                (next_map_iteration / max_iterations) * 100)
            self.next_map.setText(f"Next map: {next_map_percentage}%")

            # Check if the desired number of iterations has been reached (e.g., 500)
            if next_map_iteration >= 500:
                # Load and display the chart image if it exists
                chart_file = "chart.png"
                if os.path.exists(chart_file):
                    pixmap = QPixmap(chart_file)
                    self.image_label_2.setPixmap(pixmap)
                    self.image_label_2.setScaledContents(True)
                else:
                    print(f"Chart image '{chart_file}' not found.")

        # For loss and iterations:
        loss_iteration_pattern = re.compile(
            r"(\d+): (.*?), (.*?) avg loss, .*?, .*?, .*?, (.*?) hours left")
        loss_iteration_match = loss_iteration_pattern.search(output_line)
        if loss_iteration_match:
            current_iteration = int(loss_iteration_match.group(1).strip())
            total_loss_value = float(loss_iteration_match.group(3).strip())
            hours_left = float(loss_iteration_match.group(4).strip())

            self.loss_label.setText(f"Total Loss: {total_loss_value:.2f}")

            if total_loss_value >= 1.0:
                self.loss_label.setStyleSheet("color: red")
            else:
                self.loss_label.setStyleSheet("color: green")

            data_file = self.file_paths["data"]
            max_images = self.get_total_images(data_file)
            max_iterations = self.get_max_iterations_from_cfg()
            progress_percentage = int(
                (current_iteration / max_iterations) * 100)
            self.progress_label.setValue(progress_percentage)
            self.progress_label.setFormat(f"{progress_percentage:.2f}%")

            if progress_percentage < 33:
                self.progress_label.setStyleSheet(
                    "QProgressBar { text-align: center; } QProgressBar::chunk { background-color: red; }")
            elif 33 <= progress_percentage < 66:
                self.progress_label.setStyleSheet(
                    "QProgressBar { text-align: center; } QProgressBar::chunk { background-color: orange; }")
            else:
                self.progress_label.setStyleSheet(
                    "QProgressBar { text-align: center; } QProgressBar::chunk { background-color: green; }")

            # For hours left:
            # Step 1: Get the hours left value from the output_line
            # Step 2: Convert the hours left to a datetime object
            hours_left_timedelta = timedelta(hours=hours_left)
            end_time = datetime.now() + hours_left_timedelta


            # Step 4: Format the resulting datetime object as desired and update the QLabel widget
            end_time_str = end_time.strftime("%H:%M/%a").lower()
            self.time_stop.setText(f"Time Stop: {end_time_str}")

    def stop_training_button_clicked(self):
        if self.training_thread:
            self.training_thread.stop()
            try:
                time.sleep(1)  # Add a short delay
            except KeyboardInterrupt:
                pass  # Suppress the KeyboardInterrupt exception
            self.reset_gui()

    # stop training
    def reset_gui(self):
        # Clear the terminal
        os.system('cls')

        # Reset the GUI components to their initial state
        # For example, reset labels, progress bars, etc.

        self.loss_label.setText("Total Loss:")
        self.loss_label.setStyleSheet("")
        self.map_label.setText("mAP:")
        self.map_label.setStyleSheet("")
        self.progress_label.setValue(0)
        self.progress_label.setFormat("0.00%")
        self.progress_label.setStyleSheet("")
        self.next_map.setText("Next map: 0%")
        self.time_stop.setText("Time Stop: 0000/mon")

    # calculate anchors

    def import_data(self):
        data_directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Import Data Directory', 'c:\\')
        if not data_directory:
            return

        image_files = glob.glob(os.path.join(data_directory, '*.jpg')) + \
            glob.glob(os.path.join(data_directory, '*.png'))
        text_files = glob.glob(os.path.join(data_directory, '*.txt'))

        self.image_files = image_files
        self.text_files = text_files

    def calculate_anchors(self):
        try:
            if not self.image_files or not self.text_files:
                return

            if not hasattr(self, 'filename') or not self.filename:
                QtWidgets.QMessageBox.warning(
                    self, "Error", "Please open a .cfg file before calculating anchors.")
                return

            num_clusters = self.clusters_spinbox.value()
            width_value = self.width_spinbox.value()
            height_value = self.height_spinbox.value()
            print(
                f"num_of_clusters = {num_clusters}, width = {width_value}, height = {height_value}")

            annotation_dims = []
            image_counter = 0

            for text_file in self.text_files:
                # Ignore classes.txt file
                if text_file.endswith("classes.txt"):
                    continue

                with open(text_file, 'r') as f:
                    lines = f.readlines()
                    image_counter += 1  # Increment the image counter for each file

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                width = float(parts[3])
                                height = float(parts[4])
                                annotation_dims.append((width, height))
                            except ValueError:
                                print(
                                    f"Skipping line in {text_file}: width or height are not numbers")
                        else:
                            print(
                                f"Skipping line in {text_file}: not enough parts")

            print(f"Read labels from {image_counter} images")
            print(f"Loaded image: {image_counter} box: {len(annotation_dims)}")

            if not annotation_dims:
                return

            X = np.array(annotation_dims)
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
            kmeans.fit(X)
            centroids = kmeans.cluster_centers_
            centroids = sorted(centroids, key=lambda c: c[0] * c[1])

            # Calculate the average IoU
            avg_iou = 0
            for annotation in annotation_dims:
                closest_centroid = max(
                    centroids, key=lambda centroid: self._iou(centroid, annotation))
                avg_iou += self._iou(closest_centroid, annotation)
            avg_iou /= len(annotation_dims)

            print(f"Average IoU (loss): {avg_iou * 100:.2f}%")

            anchors_pixel = np.array(
                [list(map(int, centroid * [width_value, height_value])) for centroid in centroids])
            # save the percentage
            avg_iou *= 100
            avg_iou = round(avg_iou, 2)
            self.settings['anchors'] = anchors_pixel.tolist()
            # Pass the average IoU as a parameter
            self.saveAnchorsSettings(avg_iou)

            self.update_cfg_anchors(', '.join(map(str, anchors_pixel.flatten())))
            print(
                f"Saving anchors to settings.json\nanchors = {', '.join(map(str, anchors_pixel.flatten()))}")

            self.plot_anchors(centroids, num_clusters, avg_iou)

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"An error occurred: {str(e)}")


    def _iou(self, box1, box2):
        intersect_w = min(box1[0], box2[0])
        intersect_h = min(box1[1], box2[1])
        intersect = intersect_w * intersect_h
        union = box1[0] * box1[1] + box2[0] * box2[1] - intersect
        return intersect / union

    def plot_anchors(self, centroids, num_clusters, avg_iou):
        if self.show_checkbox.isChecked():
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_facecolor('Black')
            colors = plt.cm.get_cmap('rainbow', num_clusters)
            patches = []

            for i, centroid in enumerate(centroids):
                anchor_width, anchor_height = centroid
                x = anchor_width / 2
                y = anchor_height / 2
                x1 = x - anchor_width / 2
                y1 = y - anchor_height / 2
                rect = mpatches.Rectangle((x1, y1), anchor_width, anchor_height,
                                          linewidth=2, edgecolor=colors(i), facecolor='none')
                patches.append(rect)
                ax.add_patch(rect)

            # Add IoU text to the plot
            ax.text(0.05, 0.90, f"Average IoU (loss): {avg_iou:.2f}%", transform=ax.transAxes,
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            width_value = self.width_spinbox.value()
            height_value = self.height_spinbox.value()
            ax.text(0.05, 0.95, f"Width: {width_value}, Height: {height_value}", transform=ax.transAxes,
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.legend(
                patches, [f'Anchor {i + 1}' for i in range(num_clusters)], loc='upper right')
            plt.xlabel('Width')
            plt.ylabel('Height')
            plt.title('Anchors')
            plt.show()

    def saveAnchorsSettings(self, avg_iou):
        # Get anchors from settings and flatten them
        anchors = self.settings.get('anchors', [])
        flat_anchors = [item for sublist in anchors for item in sublist]

        settings = {
            'anchors': flat_anchors,
            'avg_iou': avg_iou
        }

        with open('settings.json', 'w') as f:
            json.dump(settings, f)

    def update_cfg_anchors(self, anchors):
        # Update the configuration file with the calculated anchors
        pass

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
        if self.file_type != "yaml":
            self.clear_table()
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        # Get the text from the default_yaml_path widget
        initial_directory = self.default_yaml_path.text()
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Select YAML File", initial_directory, "YAML Files (*.yaml *.yml);;All Files (*)", options=options)

        if file_name:
            self.hide_activation_checkbox.setChecked(
                False)  # Comment out or remove this line
            self.yaml_filename = file_name
            # Set the path to the YAML file in the default_yaml_path widget
            self.default_yaml_path.setText(file_name)
            with open(file_name, 'r', encoding="utf-8") as f:
                self.parsed_yaml = yaml.safe_load(f)
            self.file_type = "yaml"
            self.cfg_table.setColumnCount(2)
            self.cfg_table.setHorizontalHeaderLabels(["Key", "Value"])
            self.cfg_table.setRowCount(len(self.parsed_yaml))
            for row, (key, value) in enumerate(self.parsed_yaml.items()):
                self.cfg_table.setItem(row, 0, QTableWidgetItem(str(key)))
                self.cfg_table.setItem(row, 1, QTableWidgetItem(str(value)))
            self.filename = None  # Clear the CFG file when a YAML file is opened
            # Clear the CFG path from the cfg_open_label widget
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
        self.hide_activation_checkbox.setChecked(False)
        if self.file_type != "cfg":
            self.clear_table()
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly
        initial_directory = "C:/"  # Set the initial directory
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Config File", initial_directory, "Config Files (*.cfg);;All Files (*)", options=options)
        if file_name:
            self.filename = file_name
            self.cfg_open_label.setText("Cfg: " + file_name)
            self.parse_cfg_file(file_name)
            self.yaml_filename = None  # Clear the YAML file when a CFG file is opened
            # Clear the default_yaml_path widget
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
        if yolo_layers == 5:
            yolo_mask_values = [[12, 13, 14], [9, 10, 11],
                                [6, 7, 8], [3, 4, 5], [0, 1, 2]]
        elif yolo_layers == 4:
            yolo_mask_values = [[9, 10, 11], [6, 7, 8], [3, 4, 5], [0, 1, 2]]
        elif yolo_layers == 3:
            yolo_mask_values = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        elif yolo_layers == 2:
            yolo_mask_values = [[3, 4, 5], [0, 1, 2]]
        elif yolo_layers == 1:
            yolo_mask_values = [[0, 1, 2]]

        yolo_section_idx = 0

        if self.imported_anchors is not None:
            for idx, (section_type, section_content) in enumerate(sections):
                if section_type == "yolo":
                    section_lines = section_content.strip().split("\n")
                    section_dict = {line.split("=")[0].strip(): line.split(
                        "=")[1].strip() for line in section_lines if "=" in line}

                    section_dict["anchors"] = ', '.join(
                        [f"{x},{y}" for x, y in self.imported_anchors])

                    sections[idx] = (section_type, '\n'.join(
                        [f"{key} = {value}" for key, value in section_dict.items()]))

        for idx, (section_type, section_content) in enumerate(sections):
            section_lines = section_content.strip().split("\n")
            section_dict = {line.split("=")[0].strip(): line.split(
                "=")[1].strip() for line in section_lines if "=" in line}

            if section_type == "net":
                net_items = ["batch", "subdivisions", "width", "height", "saturation", "exposure", "hue",
                            "max_batches", "flip", "mosaic", "letter_box", "cutmix", "mosaic_bound",
                            "mosaic_scale", "mosaic_center", "mosaic_crop", "mosaic_flip", "steps",
                            "scales", "classes"]  # Add "max_batches" and "classes" to your net_items list

                for i, item in enumerate(net_items):
                    if item in section_dict:
                        value_without_comment = section_dict[item].split('#')[0].strip()

                        if not value_without_comment:
                            continue

                        row_count = self.cfg_table.rowCount()
                        self.cfg_table.insertRow(row_count)
                        self.cfg_table.setItem(row_count, 0, QtWidgets.QTableWidgetItem(f"{item}_0"))
                        self.cfg_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(value_without_comment))

                        # Highlighting specific rows based on the parameter
                        if item in {"batch","width", "height",  "subdivisions", "max_batches",}:
                            self.cfg_table.item(row_count, 0).setBackground(QtGui.QColor(255, 255, 144))  # Highlighting the parameter name cell
                            self.cfg_table.item(row_count, 1).setBackground(QtGui.QColor(255, 255, 144))  # Highlighting the value cell

                self.net_dict = section_dict
            elif section_type == "convolutional":
                is_before_yolo = idx < len(
                    sections) - 1 and sections[idx + 1][0] == "yolo"
                conv_items = ["activation"]

                for item in conv_items:
                    if item in section_dict and (is_before_yolo or item != "filters"):
                        row_count = self.cfg_table.rowCount()
                        self.cfg_table.insertRow(row_count)
                        self.cfg_table.setItem(
                            row_count, 0, QtWidgets.QTableWidgetItem(f"{item}_{idx}"))

                        if item == "activation":
                            activation_combo = QtWidgets.QComboBox()
                            activation_combo.addItems(
                                ["leaky", "mish", "swish", "linear"])
                            activation_combo.setCurrentText(section_dict[item])
                            self.cfg_table.setCellWidget(
                                row_count, 1, activation_combo)
                            self.activation_row_count[activation_count] = row_count
                            activation_count += 1
                        else:
                            self.cfg_table.setItem(
                                row_count, 1, QtWidgets.QTableWidgetItem(section_dict[item]))
            elif section_type == "yolo":
                yolo_items = ["mask", "anchors", "num",
                              "classes", "ignore_thresh", "random"]

                # Set mask values based on the yolo_section_idx
                mask_values = yolo_mask_values[yolo_section_idx]
                # convert list to comma-separated string
                section_dict["mask"] = ','.join(map(str, mask_values))

                if "anchors" in section_dict:
                    section_dict["anchors"] = ', '.join(
                        [x.strip() for x in section_dict["anchors"].split(',')])

                for item in yolo_items:
                    if item in section_dict:
                        if item == "num":
                            num_clusters = int(section_dict[item])
                            self.clusters_spinbox.setValue(num_clusters)

                        row_count = self.cfg_table.rowCount()
                        self.cfg_table.insertRow(row_count)
                        self.cfg_table.setItem(row_count, 0, QtWidgets.QTableWidgetItem(f"{item}_{idx}"))

                        if item == "anchors" and self.imported_anchors is not None:
                            self.cfg_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(','.join([f"{x},{y}" for x, y in self.imported_anchors])))
                        else:
                            self.cfg_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(section_dict[item]))

                        # Highlight the 'classes' parameter
                        if item == "classes":
                            self.cfg_table.item(row_count, 0).setBackground(QtGui.QColor(255, 255, 144))  # Highlighting the parameter name cell
                            self.cfg_table.item(row_count, 1).setBackground(QtGui.QColor(255, 255, 144))  # Highlighting the value cell

                # Safeguard to prevent yolo_section_idx from exceeding the range
                if yolo_section_idx >= len(yolo_mask_values) - 1:
                    yolo_section_idx = len(yolo_mask_values) - 1  # or reset to another value if necessary

                yolo_section_idx += 1



        self.cfg_table.resizeColumnsToContents()
        self.cfg_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)

    def cfg_save_clicked(self):
        if self.filename:
            # Create a dictionary with the parameter values from the table widget
            table_data = {}
            for row in range(self.cfg_table.rowCount()):
                param = self.cfg_table.item(row, 0).text()

                if param.startswith("activation"):
                    activation_count = {v: k for k, v in self.activation_row_count.items()}[
                        row]
                    value = self.cfg_table.cellWidget(row, 1).currentText()
                    table_data[f"{param}"] = value
                else:
                    item = self.cfg_table.item(row, 1)
                    value = item.text() if item else ""
                    table_data[param] = value
            # Find the corresponding max_batches value and update steps added a 3rd step remove if desired
            if "max_batches_0" in table_data:
                max_batches = int(table_data["max_batches_0"])
                steps_60 = int(max_batches * 0.6)
                steps_80 = int(max_batches * 0.8)
                steps_90 = int(max_batches * 0.9)
                table_data["steps_0"] = f"{steps_60},{steps_80},{steps_90}"
                # dont forget to remove one scale if your remove a step
                table_data["scales_0"] = ".1,.1,.1"

            # Find the corresponding classes value for each YOLO layer
            yolo_classes = {}
            for key, value in table_data.items():
                if "classes" in key:
                    section_idx = int(key.split('_')[-1])
                    yolo_classes[section_idx] = int(value)

            # Read the original configuration file line by line and modify the relevant lines
            new_config = ""
            section_idx = -1
            current_section = ""
            yolo_layer_indices = []
            conv_before_yolo = False
            with open(self.filename, "r", encoding="utf-8") as f:
                for line in f:
                    stripped_line = line.strip()

                    if stripped_line.startswith("["):
                        section_idx += 1

                        if conv_before_yolo:
                            # store the index of the convolutional layer before the YOLO layer
                            yolo_layer_indices.append(section_idx - 1)
                            conv_before_yolo = False
                        current_section = stripped_line.strip("[]")
                        new_config += line

                        if current_section == "convolutional":
                            conv_before_yolo = True
                        elif current_section == "yolo":
                            conv_before_yolo = False
                    elif "=" in stripped_line:
                        param, value = stripped_line.split("=")
                        param = param.strip()

                        if current_section == "net":
                            new_param = f"{param}_0"
                        else:
                            new_param = f"{param}_{section_idx}"

                        new_value = table_data.get(new_param, value.strip())

                        if param == "filters" and conv_before_yolo:
                            classes = yolo_classes.get(section_idx + 1)
                            if classes is not None:
                                new_value = (classes + 5) * 3
                                new_line = f"{param}={new_value}\n"
                                new_config += new_line
                                continue

                        new_line = f"{param}={new_value}\n"
                        new_config += new_line
                    elif stripped_line.startswith("#"):
                        new_config += line
                    else:
                        new_config += line

            # Add the save dialog code block here
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            save_file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Config File As", "", "Config Files (*.cfg);;All Files (*)", options=options)

            if save_file_name:
                # Update the filters in the convolutional layer before the YOLO layers based on the new classes

                # Add .cfg extension to the save_file_name if it doesn't have it
                if not save_file_name.endswith('.cfg'):
                    save_file_name += '.cfg'

                # Save the modified configuration to the selected file
                with open(save_file_name, 'w', encoding='utf-8') as f:
                    f.write(new_config)

                # Show a success message
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setWindowTitle("Success")
                msg.setText("Configuration file saved successfully.")
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec_()

    def update_cfg_anchors(self, anchors):
        for row in range(self.cfg_table.rowCount()):
            item_key = self.cfg_table.item(row, 0)
            if item_key and "anchors" in item_key.text():
                # Pass the string representation of the anchors
                new_var = self.new_method(str(anchors))
                self.cfg_table.setItem(row, 1, new_var)

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

