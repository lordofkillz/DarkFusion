"""Real Qt canvas measurements without loading models or modifying a dataset."""

import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets
from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPolygonF, QTransform
from PyQt5.QtWidgets import QApplication, QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView


class Box(QGraphicsRectItem):
    pass


class Segmentation(QGraphicsPolygonItem):
    def __init__(self, polygon):
        super().__init__(polygon)
        self.polygon = polygon


class Obb(QGraphicsPolygonItem):
    pass


def load_handlers():
    source = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    view = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CustomGraphicsView")
    names = {"_annotation_measurement_size", "_measurement_target_item", "_drawing_size_feedback",
             "_paint_drawing_size_feedback", "_paint_blank_image_overlay", "_update_crosshair", "viewportEvent", "mouseMoveEvent", "paintEvent"}
    view.body = [n for n in view.body if isinstance(n, ast.FunctionDef) and n.name in names]
    view.bases = [ast.Name(id="QGraphicsView", ctx=ast.Load())]
    settings = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SettingsDialog")
    callbacks = [n for n in settings.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"save_measurement_overlay_setting", "save_blank_image_overlay_setting", "save_pan_tool_setting"}]
    general = next(n for n in settings.body if isinstance(n, ast.FunctionDef) and n.name == "init_general_tab")
    # Exercise the actual Display widget construction, without unrelated controls.
    start = next(i for i, n in enumerate(general.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "display_group" for t in n.targets))
    end = next(i for i, n in enumerate(general.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "audio_group" for t in n.targets))
    general.body = ast.parse("layout = QtWidgets.QVBoxLayout(self)").body + general.body[start:end]
    settings.body = callbacks + [general]
    settings.bases = [ast.Attribute(value=ast.Name(id="QtWidgets", ctx=ast.Load()), attr="QDialog", ctx=ast.Load())]
    namespace = dict(QGraphicsView=QGraphicsView, QtWidgets=QtWidgets, QPointF=QPointF, QRectF=QRectF,
                     QEvent=QEvent, Qt=Qt, QColor=QColor, QPainter=QPainter, math=math,
                     BoundingBoxDrawer=Box, SegmentationDrawer=Segmentation, OBBDrawer=Obb,
                     annotation_labels_hidden=lambda parent: False)
    module = ast.fix_missing_locations(ast.Module(body=[view, settings], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["CustomGraphicsView"], namespace["SettingsDialog"]


View, Settings = load_handlers()


class AnnotationMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_view(self, zoom=1):
        view = View()
        view.resize(800, 600)
        view.setScene(QGraphicsScene(0, 0, 1000, 1000, view))
        view.setTransform(QTransform().scale(zoom, zoom))
        view.centerOn(500, 500)
        view.show_measurement_overlay = True
        view.show_crosshair = False
        view.crosshair_position = QPointF()
        view._measurement_cursor_inside = False
        view.drawing = view.moving_view = False
        view.current_bbox = view.current_segmentation = view.current_obb_drawer = None
        view._dragged_keypoint_handle = None
        view.main_window = SimpleNamespace(edit_mode_active=lambda: False, is_segmentation_mode=lambda: False)
        view._current_bbox_size_allowed = lambda: True
        view.show()
        self.app.processEvents()
        self.addCleanup(view.deleteLater)
        self.addCleanup(view.close)
        return view

    def hover(self, view, x=500, y=500):
        point = view.mapFromScene(QPointF(x, y))
        event = QMouseEvent(QEvent.MouseMove, QPointF(point), Qt.NoButton, Qt.NoButton, Qt.NoModifier)
        view.mouseMoveEvent(event)

    def test_hover_dimensions_stay_in_image_pixels_without_crosshair_lines(self):
        for zoom in (0.25, 1, 8):
            with self.subTest(zoom=zoom):
                view = self.make_view(zoom)
                item = Box(480, 490, 40, 20)
                view.scene().addItem(item)
                self.hover(view)
                self.assertEqual(view._drawing_size_feedback(), ("40.00 × 20.00 image px", True))

    def test_off_suppresses_hover_and_drawing_and_reenable_works(self):
        view = self.make_view()
        item = Box(480, 490, 40, 20)
        view.scene().addItem(item)
        self.hover(view)
        for drawing in (False, True):
            view.drawing = drawing
            view.current_bbox = item if drawing else None
            view.show_measurement_overlay = False
            self.assertIsNone(view._drawing_size_feedback())
            image = QImage(800, 600, QImage.Format_ARGB32)
            image.fill(Qt.transparent)
            painter = QPainter(image)
            try:
                self.assertIsNone(view._paint_drawing_size_feedback(painter))
                view.show_measurement_overlay = True
                self.assertIsNotNone(view._paint_drawing_size_feedback(painter))
            finally:
                painter.end()

    def test_smallest_visible_annotation_wins_and_deletion_clears_hover(self):
        view = self.make_view()
        large = Box(400, 400, 200, 200)
        small = Box(490, 490, 20, 20)
        view.scene().addItem(small)
        view.scene().addItem(large)
        large.setZValue(10)
        self.hover(view)
        self.assertIs(view._measurement_target_item(), small)
        small.hide()
        self.assertIs(view._measurement_target_item(), large)
        view.scene().removeItem(large)
        self.assertIsNone(view._measurement_target_item())

    def test_polygon_measurement_excludes_outline_and_empty_bounding_rect_area(self):
        view = self.make_view()
        polygon = Segmentation(QPolygonF([QPointF(480, 480), QPointF(520, 480), QPointF(480, 520)]))
        polygon.setPen(QPen(Qt.red, 8))
        view.scene().addItem(polygon)
        self.hover(view, 490, 490)
        self.assertEqual(view._drawing_size_feedback(), ("40.00 × 40.00 image px", True))
        self.hover(view, 518, 518)
        self.assertIsNone(view._drawing_size_feedback())

    def test_rotated_box_uses_edge_lengths_instead_of_axis_aligned_envelope(self):
        view = self.make_view(2)
        polygon = QPolygonF([QPointF(-40, -15), QPointF(40, -15), QPointF(40, 15), QPointF(-40, 15)])
        obb = Obb(QTransform().rotate(35).map(polygon))
        obb.setPos(500, 500)
        view.scene().addItem(obb)
        self.hover(view)
        self.assertEqual(view._drawing_size_feedback(), ("80.00 × 30.00 image px", True))

    def test_leaving_viewport_and_empty_background_clear_hover(self):
        view = self.make_view()
        view.scene().addItem(Box(480, 490, 40, 20))
        self.hover(view)
        self.assertIsNotNone(view._drawing_size_feedback())
        QApplication.sendEvent(view.viewport(), QEvent(QEvent.Leave))
        self.assertIsNone(view._drawing_size_feedback())
        self.hover(view, 600, 600)
        self.assertIsNone(view._drawing_size_feedback())

    def test_display_checkbox_applies_immediately_and_saves_both_states(self):
        parent = QtWidgets.QWidget()
        parent.settings = {"showMeasurementOverlay": False}
        parent.screen_view = self.make_view()
        saved = []
        parent.saveSettings = lambda: saved.append(dict(parent.settings))
        dialog = Settings(parent)
        dialog.save_hide_labels_setting = lambda value: None
        dialog.save_clear_frame_confirmation_setting = lambda value: None
        dialog.init_general_tab()
        checkbox = dialog.measurement_overlay_checkbox
        self.assertEqual(checkbox.parentWidget().title(), "Display")
        self.assertFalse(checkbox.isChecked())
        for checked in (True, False):
            checkbox.setChecked(checked)
            self.assertEqual(saved[-1]["showMeasurementOverlay"], checked)
            self.assertEqual(parent.screen_view.show_measurement_overlay, checked)
        dialog.close()
        parent.deleteLater()

    def test_blank_display_checkbox_is_separate_and_restores_saved_preference(self):
        parent = QtWidgets.QWidget()
        parent.settings = {"showBlankImageOverlay": False, "showMeasurementOverlay": True}
        saved, refreshed = [], []
        parent.saveSettings = lambda: saved.append(dict(parent.settings))
        parent.update_blank_overlay_state = lambda: refreshed.append(parent.settings["showBlankImageOverlay"])
        for initial, changes in ((False, (True, False)), (False, ())):
            dialog = Settings(parent)
            dialog.save_hide_labels_setting = lambda value: None
            dialog.save_clear_frame_confirmation_setting = lambda value: None
            dialog.init_general_tab()
            checkbox = dialog.blank_image_overlay_checkbox
            self.assertEqual(checkbox.parentWidget().title(), "Display")
            self.assertEqual(checkbox.isChecked(), initial)
            for checked in changes:
                checkbox.setChecked(checked)
                self.assertEqual(saved[-1]["showBlankImageOverlay"], checked)
                self.assertEqual(refreshed[-1], checked)
                self.assertTrue(parent.settings["showMeasurementOverlay"])
            dialog.close()
        parent.deleteLater()

    def test_pan_tool_checkbox_defaults_on_saves_preference_and_updates_puck(self):
        parent = QtWidgets.QWidget()
        parent.settings = {}
        saved = []
        parent.saveSettings = lambda: saved.append(dict(parent.settings))
        updates = []

        class Puck:
            enabled = True

            def update_visibility(self):
                updates.append(self.enabled)

        parent.screen_view = SimpleNamespace(pan_overlay=Puck())
        dialog = Settings(parent)
        dialog.save_hide_labels_setting = lambda value: None
        dialog.save_clear_frame_confirmation_setting = lambda value: None
        dialog.init_general_tab()
        checkbox = dialog.pan_tool_checkbox
        self.assertEqual(checkbox.parentWidget().title(), "Display")
        self.assertTrue(checkbox.isChecked())
        for checked in (False, True):
            checkbox.setChecked(checked)
            self.assertEqual(saved[-1]["panOverlayEnabled"], checked)
            self.assertEqual(parent.screen_view.pan_overlay.enabled, checked)
            self.assertEqual(updates[-1], checked)
        dialog.close()
        parent.deleteLater()


if __name__ == "__main__":
    unittest.main()
