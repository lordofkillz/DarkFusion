"""Optional DarkFusion art adapter. Qt5 only; no inference or annotation changes.

Installed beside UltraDarkFusion_v5.2.py. The single apply_stylesheet hook calls
apply_theme_assets(window, selected_style, style_folder). Switching to any other
theme restores the original icons and removes all decorative artwork.
"""
from pathlib import Path
import json
import logging
import weakref

from PyQt5 import QtCore, QtGui, QtWidgets, sip

_LOG = logging.getLogger(__name__)
_OBJECT_ICONS = {
    "video_upload": "add-video", "download_video": "download",
    "remove_video_button": "remove", "remove_video": "remove",
    "cuda_monitor_button": "monitor", "extract_button": "extract",
    "stop_extract": "stop", "play_video_button": "play",
    "next_button": "next", "previous_button": "previous",
    "forward_button": "next", "back_button": "previous",
    "location_button": "folder", "save_list_button": "save",
    "btn_save_file": "save", "img_video_button": "folder",
    "clear_current_frame_labels_button": "clear",
    "propagate_labels_button": "link", "propagate_range_button": "link",
    "video_label_frame_button": "annotate", "menuExtras": "settings",
}
_TEXT_ICONS = {
    "add video": "add-video", "download": "download", "remove": "remove",
    "system monitor": "monitor", "monitor": "monitor", "annotate": "annotate",
    "tools": "tools", "settings": "settings", "shortcuts": "shortcut",
    "keyboard shortcuts": "shortcut", "play / pause": "play", "play": "play",
    "stop": "stop", "extract": "extract", "save": "save", "save to": "folder",
    "next": "next", "prev": "previous", "clear frame": "clear",
    "label frame": "annotate", "collect": "add-video", "label": "annotate",
    "review": "review", "prepare / split": "split", "train / export": "train",
    "advanced": "tools", "help": "help", "file": "folder", "split": "split",
    "train": "train", "open folder": "folder", "auto scan": "scan",
}


def _alive(obj):
    return obj is not None and not sip.isdeleted(obj)


class _ArtLayer(QtWidgets.QWidget):
    """A purely visual child; never participates in the scene or intercepts input."""
    def __init__(self, parent, kind):
        super().__init__(parent)
        self.kind = kind
        self.pixmap = QtGui.QPixmap()
        self.base = QtGui.QColor("#0a1118")
        self.edge = QtGui.QColor("#345567")
        self.fit = "contain"
        self.focal_y = 0.5
        self.setObjectName("df_decorative_" + kind)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        parent.installEventFilter(self)
        self.setGeometry(parent.rect())

    def eventFilter(self, obj, event):
        if event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
            self.setGeometry(obj.rect())
            if self.kind == "header":
                self.lower()
        return False

    def set_art(self, file_path, palette):
        self.pixmap = QtGui.QPixmap(str(file_path))
        self.base = QtGui.QColor(palette["base"])
        self.edge = QtGui.QColor(palette["border"])
        self.fit = palette.get("headerFit", "right") if self.kind == "header" else "contain"
        self.focal_y = float(palette.get("headerFocalY", 0.5))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), self.base)
        if not self.pixmap.isNull():
            if self.kind == "header" and self.fit == "cover":
                # Fill the whole dock at any width; preserve image proportions.
                scale = max(self.width() / self.pixmap.width(),
                            self.height() / self.pixmap.height())
                size = QtCore.QSizeF(self.pixmap.width()*scale, self.pixmap.height()*scale)
                top = max(self.height()-size.height(),
                          min(0, self.height()/2-size.height()*self.focal_y))
                rect = QtCore.QRectF((self.width()-size.width())/2, top,
                                     size.width(), size.height())
            elif self.kind == "header":
                scale = self.height() / self.pixmap.height()
                size = QtCore.QSizeF(self.pixmap.width()*scale, self.height())
                rect = QtCore.QRectF(self.width()-size.width(), 0, size.width(), size.height())
            else:
                size = QtCore.QSizeF(self.pixmap.size())
                size.scale(QtCore.QSizeF(self.size()), QtCore.Qt.KeepAspectRatio)
                rect = QtCore.QRectF((self.width()-size.width())/2,
                                     (self.height()-size.height())/2,
                                     size.width(), size.height())
            painter.drawPixmap(rect, self.pixmap, QtCore.QRectF(self.pixmap.rect()))
        painter.setPen(QtGui.QPen(self.edge, 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class _ThemeAdapter(QtCore.QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.asset_dir = None
        self.palette_data = None
        self.header = None
        self.scene = None
        self.icons = weakref.WeakKeyDictionary()
        self.tabs = weakref.WeakKeyDictionary()
        self.header_panels = weakref.WeakKeyDictionary()
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh)
        self.scene_timer = QtCore.QTimer(self)
        self.scene_timer.setSingleShot(True)
        self.scene_timer.timeout.connect(self.update_idle)

    def use(self, asset_dir, palette_data):
        self.asset_dir = asset_dir
        self.palette_data = palette_data
        self.refresh()
        self.timer.start(650)  # runs after Designer inline-style clearing

    def refresh(self):
        if not _alive(self.window):
            return
        if self.asset_dir is None:
            self.restore()
            return
        window = self.window
        header_parent = getattr(window, "dockWidgetContents_3", None)
        if _alive(header_parent):
            for panel in header_parent.findChildren(QtWidgets.QWidget):
                if type(panel) in (QtWidgets.QWidget, QtWidgets.QFrame) and panel not in self.header_panels:
                    self.header_panels[panel] = panel.property('dfHeaderPanel')
                    panel.setProperty('dfHeaderPanel', True)
                    panel.style().unpolish(panel)
                    panel.style().polish(panel)
            if not _alive(self.header):
                self.header = _ArtLayer(header_parent, "header")
            self.header.set_art(self.asset_dir / "header.png", self.palette_data)
            self.header.show()
            self.header.lower()
        view = getattr(window, "screen_view", None)
        if isinstance(view, QtWidgets.QGraphicsView) and _alive(view):
            self.update_idle()
        candidates = window.findChildren(QtWidgets.QPushButton)
        candidates += window.findChildren(QtWidgets.QToolButton)
        candidates += window.findChildren(QtWidgets.QAction)
        for obj in candidates:
            # Qt menu bars replace titles with icons, so preserve menu titles.
            if isinstance(obj, QtWidgets.QAction) and obj.menu() is not None:
                continue
            key = _OBJECT_ICONS.get(obj.objectName())
            if key is None:
                key = _TEXT_ICONS.get(obj.text().replace("&", "").strip().lower())
            if key is None:
                continue
            icon = QtGui.QIcon(str(self.asset_dir / "icons" / (key + ".svg")))
            if icon.isNull():
                continue
            if obj not in self.icons:
                self.icons[obj] = (QtGui.QIcon(obj.icon()),
                                   obj.iconSize() if isinstance(obj, QtWidgets.QAbstractButton) else None,
                                   obj.minimumWidth() if isinstance(obj, QtWidgets.QAbstractButton) else None)
            obj.setIcon(icon)
            if isinstance(obj, QtWidgets.QAbstractButton):
                obj.setIconSize(QtCore.QSize(16, 16))
                if obj.text():
                    required = obj.fontMetrics().horizontalAdvance(obj.text().replace('&', '')) + 42
                    obj.setMinimumWidth(min(obj.maximumWidth(), max(obj.minimumWidth(), required)))
        for tabs in window.findChildren(QtWidgets.QTabWidget):
            originals = self.tabs.setdefault(tabs, {})
            for index in range(tabs.count()):
                key = _TEXT_ICONS.get(tabs.tabText(index).replace("&", "").strip().lower())
                if key:
                    page = tabs.widget(index)
                    if page not in originals:
                        originals[page] = QtGui.QIcon(tabs.tabIcon(index))
                    tabs.setTabIcon(index, QtGui.QIcon(str(self.asset_dir / "icons" / (key+".svg"))))

    def _scene_changed(self, _regions):
        # Coalesce scene notifications. No repeating timer on video or idle windows.
        if not self.scene_timer.isActive():
            self.scene_timer.start(0)

    def update_idle(self):
        """Update only the actual placeholder; never paint over a viewport."""
        view = getattr(self.window, "screen_view", None)
        if not _alive(view):
            return
        current_scene = view.scene()
        if current_scene is not self.scene:
            if _alive(self.scene):
                try:
                    self.scene.changed.disconnect(self._scene_changed)
                except (TypeError, RuntimeError):
                    pass
            self.scene = current_scene
            if _alive(current_scene):
                current_scene.changed.connect(self._scene_changed)
        if not _alive(current_scene):
            return
        placeholder = bool(current_scene.property("darkfusion_placeholder_scene"))
        if placeholder:
            # A capture path can add a frame to an existing scene. Decorative
            # scene metadata must never authorize replacing that real frame.
            if any(isinstance(item, QtWidgets.QGraphicsPixmapItem)
                   and item.data(0) != "placeholder_image" for item in current_scene.items()):
                return
        else:
            if current_scene.items() or getattr(self.window, "_live_display_active", False):
                return
            current_file = getattr(self.window, "current_file", None)
            if current_file and not self.window.is_placeholder_file(current_file):
                return
        path = self.window.placeholder_image_path()
        if placeholder and current_scene.property("darkfusion_placeholder_path") == path:
            return  # Do not turn scene-change notifications into a rebuild loop.
        self.window.display_placeholder()

    def restore(self):
        self.timer.stop()
        self.scene_timer.stop()
        for layer in (self.header,):
            if _alive(layer):
                layer.hide()
        for obj, (icon, size, minimum_width) in list(self.icons.items()):
            if _alive(obj):
                obj.setIcon(icon)
                if size is not None:
                    obj.setIconSize(size)
                    obj.setMinimumWidth(minimum_width)
        for tabs, originals in list(self.tabs.items()):
            if _alive(tabs):
                for page, icon in originals.items():
                    if _alive(page):
                        index = tabs.indexOf(page)
                        if index >= 0:
                            tabs.setTabIcon(index, icon)
        self.icons.clear()
        self.tabs.clear()
        for panel, original in list(self.header_panels.items()):
            if _alive(panel):
                panel.setProperty('dfHeaderPanel', original)
                panel.style().unpolish(panel)
                panel.style().polish(panel)
        self.header_panels.clear()
        self.update_idle()


def apply_theme_assets(window, selected_style, style_folder):
    """Called after normal stylesheet application; missing assets fail closed."""
    adapter = getattr(window, "_df_theme_adapter", None)
    if adapter is None:
        adapter = _ThemeAdapter(window)
        window._df_theme_adapter = adapter
    folder = Path(style_folder) / "df-assets"
    if selected_style.startswith("DF ") and selected_style.endswith(".qss") and folder.is_dir():
        for manifest in folder.glob("*/theme.json"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if data.get("styleName") == selected_style:
                    adapter.use(manifest.parent, data)
                    return
            except (OSError, ValueError):
                _LOG.warning("Could not read theme asset manifest %s", manifest)
    adapter.use(None, None)
