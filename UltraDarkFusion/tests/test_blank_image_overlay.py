"""Blank-image badge behavior using the actual handlers and a model-free Qt scene."""

import ast
import math
import os
from pathlib import Path
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets, sip
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF, QTransform
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsRectItem


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"


class Box(QGraphicsRectItem):
    pass


class Segmentation(QGraphicsPolygonItem):
    pass


class Obb(QGraphicsPolygonItem):
    def remove_self(self):
        if self.scene() is not None:
            self.scene().removeItem(self)


def load_handlers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    main = classes["MainWindow"]
    main_names = {"clear_blank_overlay", "scene_has_annotations", "update_blank_overlay_state", "display_image_with_text"}
    main.body = [node for node in main.body if isinstance(node, ast.FunctionDef) and node.name in main_names]
    main.bases = [ast.Attribute(value=ast.Name(id="QtWidgets", ctx=ast.Load()), attr="QWidget", ctx=ast.Load())]
    view = classes["CustomGraphicsView"]
    view_names = {"_paint_blank_image_overlay", "paintEvent", "_clear_current_obb_drawer"}
    view.body = [node for node in view.body if isinstance(node, ast.FunctionDef) and node.name in view_names]
    view.bases = [ast.Attribute(value=ast.Name(id="QtWidgets", ctx=ast.Load()), attr="QGraphicsView", ctx=ast.Load())]
    settings = classes["SettingsDialog"]
    settings.body = [node for node in settings.body if isinstance(node, ast.FunctionDef) and node.name == "save_blank_image_overlay_setting"]
    settings.bases = [ast.Attribute(value=ast.Name(id="QtWidgets", ctx=ast.Load()), attr="QDialog", ctx=ast.Load())]
    namespace = {
        "QtCore": QtCore, "QtGui": QtGui, "QtWidgets": QtWidgets,
        "QPointF": QPointF, "QRectF": QRectF, "Qt": Qt, "QColor": QColor,
        "QFont": QFont, "QImage": QImage, "QPainter": QPainter,
        "QPen": QPen, "QPixmap": QPixmap, "QGraphicsPixmapItem": QGraphicsPixmapItem,
        "BoundingBoxDrawer": Box, "SegmentationDrawer": Segmentation, "OBBDrawer": Obb,
        "math": math, "sip": sip,
    }
    module = ast.fix_missing_locations(ast.Module(body=[main, view, settings], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["MainWindow"], namespace["CustomGraphicsView"], namespace["SettingsDialog"]


Window, View, Settings = load_handlers()


class BlankImageOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        windows_font = Path("C:/Windows/Fonts/segoeui.ttf")
        if windows_font.exists():
            QtGui.QFontDatabase.addApplicationFont(str(windows_font))
            cls.app.setFont(QFont("Segoe UI", 10))

    def make_view(self, image_size=(1000, 800), view_size=(800, 600), zoom=0.5):
        window = Window()
        window.settings = {}
        window.annotation_scene_active = True
        window.train_view_active = False
        window.current_file = "synthetic-image.png"
        window.is_placeholder_file = lambda path: path == "placeholder.png"
        window.saveSettings = Mock()
        view = View()
        window.screen_view = view
        view.main_window = window
        view.show_crosshair = False
        view.show_measurement_overlay = False
        view.drawing = view.drawing_obb = False
        view.current_bbox = view.current_segmentation = view.current_obb_drawer = None
        view._paint_drawing_size_feedback = lambda painter: None
        view.resize(*view_size)
        scene, image_item = self.replace_image(window, image_size)
        view.setTransform(QTransform().scale(zoom, zoom))
        view.centerOn(scene.sceneRect().center())
        view.show()
        self.app.processEvents()
        self.addCleanup(window.deleteLater)
        self.addCleanup(view.deleteLater)
        self.addCleanup(view.close)
        return window, view, scene, image_item

    def replace_image(self, window, size=(1000, 800), tagged=True):
        pixmap = QPixmap(*size)
        pixmap.fill(QColor(86, 107, 128))
        scene = QtWidgets.QGraphicsScene(0, 0, *size, window.screen_view)
        image_item = QGraphicsPixmapItem(pixmap)
        image_item.setZValue(-1000)
        if tagged:
            image_item.setData(0, "annotation_image")
        scene.addItem(image_item)
        window.image = pixmap
        window.screen_view.setScene(scene)
        return scene, image_item

    def paint_badge(self, view):
        image = QImage(view.viewport().size(), QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        try:
            rect = view._paint_blank_image_overlay(painter)
        finally:
            painter.end()
        return rect, image

    def visible_image_bounds(self, view, image_item):
        return view.mapFromScene(image_item.sceneBoundingRect()).boundingRect().intersected(view.viewport().rect())

    def assert_hidden(self, view):
        rect, image = self.paint_badge(view)
        self.assertIsNone(rect)
        empty = QImage(image.size(), image.format())
        empty.fill(Qt.transparent)
        self.assertEqual(image, empty)

    def test_no_annotations_enables_badge_without_mutating_image_or_scene_geometry(self):
        window, view, scene, image_item = self.make_view()
        original = image_item.pixmap().toImage().copy()
        original_rect = QRectF(scene.sceneRect())
        window.update_blank_overlay_state(scene)
        rect, overlay = self.paint_badge(view)
        self.assertIsNotNone(rect)
        self.assertIs(scene._blank_overlay_image_item, image_item)
        self.assertEqual(scene.sceneRect(), original_rect)
        self.assertEqual(image_item.pixmap().toImage(), original)
        self.assertEqual(len(scene.items()), 1)
        self.assertGreater(overlay.pixelColor(rect.center().toPoint()).alpha(), 0)
        self.assertEqual(overlay.pixelColor(view.viewport().rect().center()).alpha(), 0)

    def test_all_annotation_types_including_hidden_items_suppress_blank(self):
        polygon = QPolygonF([QPointF(100, 100), QPointF(200, 100), QPointF(150, 180)])
        for kind in ("box", "segmentation", "obb"):
            for hidden in (False, True):
                with self.subTest(kind=kind, hidden=hidden):
                    window, view, scene, _ = self.make_view()
                    window.update_blank_overlay_state(scene)
                    item = Box(100, 100, 100, 80) if kind == "box" else (Segmentation(polygon) if kind == "segmentation" else Obb(polygon))
                    scene.addItem(item)
                    item.setVisible(not hidden)
                    window.update_blank_overlay_state(scene)
                    self.assert_hidden(view)
                    # Direct image-load path obeys the same annotation rule.
                    window.display_image_with_text(scene, window.image)
                    self.assert_hidden(view)
                    scene.removeItem(item)
                    window.update_blank_overlay_state(scene)
                    self.assertIsNotNone(self.paint_badge(view)[0])

    def test_setting_off_is_enforced_on_both_entry_points_and_during_paint(self):
        window, view, scene, _ = self.make_view()
        window.display_image_with_text(scene, window.image)
        self.assertIsNotNone(self.paint_badge(view)[0])
        window.settings["showBlankImageOverlay"] = False
        self.assert_hidden(view)
        window.display_image_with_text(scene, window.image)
        self.assert_hidden(view)
        window.update_blank_overlay_state(scene)
        self.assert_hidden(view)
        window.settings["showBlankImageOverlay"] = True
        window.update_blank_overlay_state(scene)
        self.assertIsNotNone(self.paint_badge(view)[0])

    def test_settings_callback_persists_and_applies_both_states_immediately(self):
        window, view, scene, _ = self.make_view()
        window.update_blank_overlay_state(scene)
        settings = Settings(window)
        self.addCleanup(settings.deleteLater)
        for checked in (False, True, False):
            settings.save_blank_image_overlay_setting(checked)
            self.assertIs(window.settings["showBlankImageOverlay"], checked)
            if checked:
                self.assertIsNotNone(self.paint_badge(view)[0])
            else:
                self.assert_hidden(view)
        self.assertEqual(window.saveSettings.call_count, 3)

    def test_canceling_obb_restores_blank_after_clearing_drawing_state(self):
        window, view, scene, _ = self.make_view()
        window.update_blank_overlay_state(scene)
        drawing = Obb(QPolygonF([QPointF(100, 100), QPointF(150, 100), QPointF(130, 120)]))
        scene.addItem(drawing)
        view.current_obb_drawer = drawing
        view.drawing_obb = True
        window.clear_blank_overlay(scene)
        self.assert_hidden(view)
        view._clear_current_obb_drawer()
        self.assertIsNone(view.current_obb_drawer)
        self.assertFalse(view.drawing_obb)
        self.assertIsNone(drawing.scene())
        self.assertIsNotNone(self.paint_badge(view)[0])

    def test_placeholder_live_and_train_contexts_cannot_enable_badge(self):
        for context in ("placeholder", "live", "train"):
            with self.subTest(context=context):
                window, view, scene, _ = self.make_view()
                window.update_blank_overlay_state(scene)
                if context == "placeholder":
                    window.current_file = "placeholder.png"
                    window.annotation_scene_active = False
                elif context == "live":
                    window.annotation_scene_active = False
                else:
                    window.train_view_active = True
                self.assert_hidden(view)
                window.display_image_with_text(scene, window.image)
                self.assert_hidden(view)
                window.update_blank_overlay_state(scene)
                self.assert_hidden(view)

    def test_removed_deleted_or_untagged_image_never_uses_cached_pixmap(self):
        for operation in ("remove", "clear", "hide", "untagged"):
            with self.subTest(operation=operation):
                window, view, scene, image_item = self.make_view()
                window.update_blank_overlay_state(scene)
                if operation == "remove":
                    scene.removeItem(image_item)
                elif operation == "clear":
                    scene.clear()
                elif operation == "hide":
                    image_item.hide()
                else:
                    image_item.setData(0, None)
                    window.clear_blank_overlay(scene)
                # A paint may occur before the next state refresh. In particular,
                # scene.clear() leaves a Python reference to a deleted Qt item.
                self.assert_hidden(view)
                window.update_blank_overlay_state(scene)
                self.assert_hidden(view)

    def test_scene_navigation_cannot_leak_a_previous_blank_badge(self):
        window, view, old_scene, _ = self.make_view()
        window.update_blank_overlay_state(old_scene)
        scene, _ = self.replace_image(window)
        scene.addItem(Box(100, 100, 50, 50))
        self.assert_hidden(view)
        window.display_image_with_text(old_scene, window.image)
        self.assert_hidden(view)
        blank_scene, _ = self.replace_image(window)
        window.update_blank_overlay_state(blank_scene)
        self.assertIsNotNone(self.paint_badge(view)[0])

    def test_clearing_removes_legacy_overlay_items_without_touching_image_or_annotations(self):
        window, view, scene, image_item = self.make_view()
        window.update_blank_overlay_state(scene)
        annotation = Box(100, 100, 50, 50)
        scene.addItem(annotation)
        for _ in range(3):
            legacy = QGraphicsRectItem(0, 0, 50, 50)
            legacy.setData(0, "blank_overlay_item")
            scene.addItem(legacy)
        window.clear_blank_overlay(scene)
        self.assertIsNone(scene._blank_overlay_image_item)
        self.assertEqual(set(scene.items()), {image_item, annotation})
        self.assert_hidden(view)

    def test_badge_scales_with_visible_image_and_stays_capped_when_zoomed_or_resized(self):
        window, view, scene, image_item = self.make_view(view_size=(1200, 900), zoom=0.25)
        window.update_blank_overlay_state(scene)
        sizes = []
        for zoom in (0.25, 0.5, 2.0, 8.0):
            view.setTransform(QTransform().scale(zoom, zoom))
            view.centerOn(scene.sceneRect().center())
            self.app.processEvents()
            rect, _ = self.paint_badge(view)
            self.assertIsNotNone(rect)
            visible = QRectF(self.visible_image_bounds(view, image_item))
            self.assertTrue(visible.contains(rect))
            self.assertLessEqual(rect.width(), visible.width() * 0.45 + 1)
            self.assertLessEqual(rect.height(), visible.height() * 0.22 + 1)
            sizes.append(rect.size())
        self.assertGreater(sizes[1].height(), sizes[0].height())
        self.assertLessEqual(sizes[-1].height(), sizes[-2].height() + 1)
        view.resize(500, 300)
        view.centerOn(120, 160)
        self.app.processEvents()
        rect, _ = self.paint_badge(view)
        visible = QRectF(self.visible_image_bounds(view, image_item))
        self.assertIsNotNone(rect)
        self.assertTrue(visible.contains(rect))

    def test_tiny_displayed_images_hide_badge_instead_of_covering_image(self):
        window, view, scene, _ = self.make_view(image_size=(20, 15), zoom=1)
        window.update_blank_overlay_state(scene)
        self.assert_hidden(view)

    def test_paint_event_draws_badge_only_in_corner_and_clears_on_setting_change(self):
        window, view, scene, _ = self.make_view()
        window.update_blank_overlay_state(scene)
        self.app.processEvents()
        before = view.viewport().grab().toImage()
        badge, _ = self.paint_badge(view)
        self.assertIsNotNone(badge)
        window.settings["showBlankImageOverlay"] = False
        view.viewport().update()
        self.app.processEvents()
        after = view.viewport().grab().toImage()
        self.assertNotEqual(before, after)
        self.assertEqual(before.pixelColor(view.viewport().rect().center()), after.pixelColor(view.viewport().rect().center()))


if __name__ == "__main__":
    unittest.main()
