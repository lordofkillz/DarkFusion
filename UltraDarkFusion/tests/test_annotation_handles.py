"""Screen-sized edit handles, tested with Qt without loading models or the app."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QMouseEvent, QPen, QTransform
from PyQt5.QtWidgets import (
    QApplication, QGraphicsEllipseItem, QGraphicsItem, QGraphicsScene,
    QGraphicsView,
)


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"


def load_handle_classes():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    names = {
        "annotation_setting_number", "annotation_handle_radius",
        "BoxVertexHandle", "VertexHandle",
    }
    nodes = [node for node in tree.body if getattr(node, "name", "") in names]
    namespace = {
        "QGraphicsEllipseItem": QGraphicsEllipseItem, "QGraphicsItem": QGraphicsItem,
        "QBrush": QBrush, "QColor": QColor, "QPen": QPen, "Qt": Qt,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["BoxVertexHandle"], namespace["VertexHandle"]


HANDLE_CLASSES = load_handle_classes()


class AnnotationHandleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_handle(self, handle_class, zoom, radius=4):
        view = QGraphicsView()
        view.resize(400, 300)
        scene = QGraphicsScene(0, 0, 1000, 1000, view)
        view.setScene(scene)
        view.setTransform(QTransform().scale(zoom, zoom))
        view.centerOn(500, 500)
        updates = []
        drawer = SimpleNamespace(
            main_window=SimpleNamespace(settings={"annotationHandleSize": radius}),
            update_point=lambda index, point: updates.append((index, QPointF(point))),
        )
        handle = handle_class(drawer, 2)
        handle.setPos(500, 500)
        scene.addItem(handle)
        view.show()
        self.app.processEvents()
        self.addCleanup(view.deleteLater)
        self.addCleanup(view.close)
        return view, handle, updates

    def send_mouse(self, view, event_type, point, button, buttons):
        event = QMouseEvent(
            event_type, QPointF(point), QPointF(view.viewport().mapToGlobal(point)),
            button, buttons, Qt.NoModifier,
        )
        QApplication.sendEvent(view.viewport(), event)
        self.app.processEvents()

    def test_radius_and_scene_anchor_stay_constant_at_every_zoom(self):
        for handle_class in HANDLE_CLASSES:
            for zoom in (0.25, 1.0, 8.0):
                with self.subTest(handle=handle_class.__name__, zoom=zoom):
                    view, handle, _ = self.make_handle(handle_class, zoom)
                    device_transform = handle.deviceTransform(view.viewportTransform())
                    screen_rect = device_transform.mapRect(handle.rect())
                    self.assertAlmostEqual(screen_rect.width(), 8.0)
                    self.assertAlmostEqual(screen_rect.height(), 8.0)
                    self.assertEqual(handle.scenePos(), QPointF(500, 500))
                    self.assertEqual(handle.brush().color(), QColor(Qt.red))

    def test_native_view_picking_uses_screen_sized_hit_target(self):
        for handle_class in HANDLE_CLASSES:
            for zoom in (0.25, 1.0, 8.0):
                with self.subTest(handle=handle_class.__name__, zoom=zoom):
                    view, handle, _ = self.make_handle(handle_class, zoom)
                    center = view.mapFromScene(handle.scenePos())
                    self.assertIn(handle, view.items(center + QPoint(3, 0)))
                    self.assertNotIn(handle, view.items(center + QPoint(6, 0)))
                    scene_point = view.mapToScene(center + QPoint(3, 0))
                    self.assertIn(handle, view.scene().items(
                        scene_point, Qt.IntersectsItemShape, Qt.DescendingOrder,
                        view.viewportTransform(),
                    ))

    def test_drag_still_updates_vertex_in_image_coordinates(self):
        for handle_class in HANDLE_CLASSES:
            for zoom in (0.25, 1.0, 8.0):
                with self.subTest(handle=handle_class.__name__, zoom=zoom):
                    view, handle, updates = self.make_handle(handle_class, zoom)
                    original = QPointF(handle.scenePos())
                    # Grab off-center to ensure ignoring zoom does not make
                    # the vertex jump to the mouse or change its coordinates.
                    start = view.mapFromScene(original) + QPoint(2, 0)
                    end = start + QPoint(16, 8)
                    self.send_mouse(view, QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton)
                    self.assertIs(view.scene().mouseGrabberItem(), handle)
                    self.send_mouse(view, QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton)
                    self.send_mouse(view, QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton)
                    expected = original + QPointF(16 / zoom, 8 / zoom)
                    self.assertAlmostEqual(handle.scenePos().x(), expected.x())
                    self.assertAlmostEqual(handle.scenePos().y(), expected.y())
                    self.assertEqual(updates[-1][0], 2)
                    self.assertAlmostEqual(updates[-1][1].x(), expected.x())
                    self.assertAlmostEqual(updates[-1][1].y(), expected.y())
                    self.assertEqual(handle.brush().color(), QColor(Qt.red))

    def test_live_radius_update_remains_in_screen_pixels(self):
        for handle_class in HANDLE_CLASSES:
            view, handle, _ = self.make_handle(handle_class, 8.0)
            # Drawer refresh_drawing_style already applies radius changes with
            # this same local rect, so no zoom-dependent rewrite is needed.
            handle.setRect(-7, -7, 14, 14)
            for zoom in (0.25, 1.0, 8.0):
                view.setTransform(QTransform().scale(zoom, zoom))
                rect = handle.deviceTransform(view.viewportTransform()).mapRect(handle.rect())
                self.assertAlmostEqual(rect.width(), 14.0)
                self.assertAlmostEqual(rect.height(), 14.0)
                self.assertEqual(handle.scenePos(), QPointF(500, 500))


if __name__ == "__main__":
    unittest.main()
