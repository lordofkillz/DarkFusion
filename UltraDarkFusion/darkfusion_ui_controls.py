"""Shared Qt controls and signal/numeric helpers used by DarkFusion dialogs."""
import contextlib
import math
import re
import weakref

from PyQt5 import QtCore, QtWidgets, sip
from PyQt5.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QLabel, QPushButton, QSpinBox, QStyle, QStyledItemDelegate, QWidget


def _take_layout_widget(self, layout, widget):
    if layout is None or widget is None:
        return False

    index = layout.indexOf(widget)
    if index >= 0:
        layout.takeAt(index)
        return True

    for i in range(layout.count()):
        item = layout.itemAt(i)
        child_layout = item.layout() if item is not None else None
        if child_layout is not None and self._take_layout_widget(child_layout, widget):
            return True

    return False


def snap_int_to_multiple(value, multiple=32, minimum=None, maximum=None):
    multiple = max(1, int(multiple))
    try:
        value = int(value)
    except Exception:
        value = int(minimum if minimum is not None else multiple)

    lower = int(minimum) if minimum is not None else None
    upper = int(maximum) if maximum is not None else None
    if lower is not None:
        lower = int(math.ceil(lower / multiple) * multiple)
    if upper is not None:
        upper = int(math.floor(upper / multiple) * multiple)
        if lower is not None and upper < lower:
            upper = lower

    snapped = int(round(value / multiple) * multiple)
    if lower is not None:
        snapped = max(lower, snapped)
    if upper is not None:
        snapped = min(upper, snapped)
    return snapped


@contextlib.contextmanager
def blocked_signals(widget):
    """Temporarily block Qt signals and restore the previous state."""
    if widget is None or not hasattr(widget, "blockSignals"):
        yield widget
        return

    try:
        previous_state = widget.blockSignals(True)
    except RuntimeError:
        yield widget
        return

    try:
        yield widget
    finally:
        try:
            widget.blockSignals(previous_state)
        except RuntimeError:
            pass


class CollapsibleSettingsGroup(QtWidgets.QGroupBox):
    """Shared themed category; collapsing it never disables its controls."""

    expandedChanged = QtCore.pyqtSignal(bool)

    def __init__(self, title, parent=None, expanded=True):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        self.setStyleSheet("CollapsibleSettingsGroup { margin-top: 0px; padding: 4px; }")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self.header = QtWidgets.QToolButton(self)
        self.header.setText(title)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setCheckable(True)
        self.header.setChecked(bool(expanded))
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.header.setFocusPolicy(Qt.StrongFocus)
        self.header.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.header)

        self.content = QtWidgets.QWidget(self)
        self.content.setProperty("settingsSectionContent", True)
        self.content.setStyleSheet(
            'QWidget[settingsSectionContent="true"] { background-color: transparent; }'
        )
        self.content.setVisible(bool(expanded))
        layout.addWidget(self.content)
        self.header.toggled.connect(self._set_expanded)

    def _set_expanded(self, expanded):
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)
        expanded_policy = getattr(self, "_expanded_size_policy", None)
        if expanded_policy is not None:
            policy = QtWidgets.QSizePolicy(expanded_policy)
            if not expanded:
                policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
            self.setSizePolicy(policy)
        self.updateGeometry()
        self.expandedChanged.emit(expanded)


class CollapsibleSectionsController(QtCore.QObject):
    """Apply the Settings category style without replacing any working controls.

    Wrap the original group, preserving its layout, object name, signals and
    explicit child visibility. This also covers dialogs created after startup.
    """

    BODY_STYLE = '''
        QWidget[collapsibleSectionBody="true"] {
            border: none; margin: 0px; padding: 0px; background: transparent;
        }
        QGroupBox[collapsibleSectionBody="true"]::title {
            width: 0px; height: 0px; padding: 0px; border: none;
            color: transparent; background: transparent;
        }
    '''

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self._installing = False
        # C++ parenting alone does not keep Python wrapper attributes alive.
        # Keep the original group/header pair until Qt destroys that tool.
        self._sections = {}
        self._app = QtWidgets.QApplication.instance()
        self._app.installEventFilter(self)
        self.install_tree(host)

    def _belongs_to_host(self, widget):
        current = widget
        while current is not None:
            if current is self.host:
                return True
            current = current.parent()
        return False

    def _source_title(self, group):
        if not isinstance(group, QtWidgets.QGroupBox):
            return str(group.property("collapsibleSectionTitle") or "")
        translator = getattr(self.host, "ui_translation", None)
        if translator is not None:
            source = translator.source_text(group, "title")
            if source:
                return source
        return group.title()

    def section_key(self, group):
        window = group.window()
        # Generated controls have stable object names. Dialogs built in Python
        # use their source category title instead, independent of translated UI.
        scope = window.objectName() or type(window).__name__
        if isinstance(window, QtWidgets.QDialog):
            translator = getattr(self.host, "ui_translation", None)
            title = translator.source_text(window, "window_title") if translator else ""
            scope += ":" + (title or window.windowTitle())
        parts = []
        current = group
        while current is not None and current is not window:
            if not isinstance(current, CollapsibleSettingsGroup) and not current.property("settingsSectionContent"):
                name = current.objectName()
                if name:
                    parts.append(name)
            current = current.parentWidget()
        if not group.objectName():
            parts.insert(0, self._source_title(group))
        return scope + "/" + "/".join(reversed(parts))

    def _remember(self, key, expanded):
        settings = self.host.settings
        saved = settings.get("collapsibleSections", {})
        states = dict(saved) if isinstance(saved, dict) else {}
        states[key] = bool(expanded)
        settings["collapsibleSections"] = states
        if hasattr(self.host, "queue_settings_save"):
            self.host.queue_settings_save(delay_ms=150)
        elif hasattr(self.host, "saveSettings"):
            self.host.saveSettings()

    def install_tree(self, root):
        if self._installing or not self._belongs_to_host(root):
            return
        self._installing = True
        try:
            groups = ([root] if isinstance(root, QtWidgets.QGroupBox) else [])
            groups += root.findChildren(QtWidgets.QGroupBox)
            groups += [frame for frame in root.findChildren(QtWidgets.QFrame)
                       if frame.property("collapsibleSectionTitle")]
            for section in groups:
                if (isinstance(section, CollapsibleSettingsGroup) and section.property("persistCollapseState")
                        and not getattr(section, "_collapse_state_registered", False)):
                    key = self.section_key(section)
                    saved = self.host.settings.get("collapsibleSections", {})
                    expanded = saved.get(key) if isinstance(saved, dict) else None
                    if isinstance(expanded, bool):
                        section.header.setChecked(expanded)
                    section.expandedChanged.connect(lambda value, key=key: self._remember(key, value))
                    section._collapse_state_registered = True
                    identity = id(section)
                    self._sections[identity] = (None, section)
                    section.setProperty("collapsibleSectionIdentity", str(identity))
                    section.destroyed.connect(self._forget_section)
            # Capture paths before wrapping any ancestors.
            candidates = [(group, self.section_key(group)) for group in groups if (
                not isinstance(group, CollapsibleSettingsGroup)
                and not getattr(group, "_collapsible_section", None)
                and not group.property("collapsibleSectionBody")
                and not group.property("collapsibleSectionSkip")
                and not (isinstance(group, QtWidgets.QGroupBox) and group.isCheckable())
                and self._source_title(group).strip()
                and group.layout() is not None and group.layout().count()
            )]
            for group, key in candidates:
                self._wrap_group(group, key)
        finally:
            self._installing = False

    def _wrap_group(self, group, key):
        parent = group.parentWidget()
        if parent is None or parent.layout() is None:
            return
        saved = self.host.settings.get("collapsibleSections", {})
        expanded = saved.get(key, True) if isinstance(saved, dict) else True
        expanded = expanded if isinstance(expanded, bool) else True
        hidden = group.isHidden()
        section = CollapsibleSettingsGroup(self._source_title(group), parent, expanded=expanded)
        section.setObjectName((group.objectName() or "category") + "_section")
        section.header.setObjectName(section.objectName() + "_header")
        section.header.setAccessibleName(section.header.text())
        section.header.setToolTip("Expand or collapse this category. Running tasks are not stopped.")
        section.content.setObjectName(section.objectName() + "_content")
        section._expanded_size_policy = QtWidgets.QSizePolicy(group.sizePolicy())
        policy = QtWidgets.QSizePolicy(section._expanded_size_policy)
        if not expanded:
            policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
        section.setSizePolicy(policy)
        item = parent.layout().replaceWidget(group, section, Qt.FindChildrenRecursively)
        if item is None:
            section.deleteLater()
            return
        section_layout = QtWidgets.QVBoxLayout(section.content)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(0)
        section_layout.addWidget(group)
        group.setProperty("collapsibleSectionBody", True)
        group.setStyleSheet(self.BODY_STYLE)
        group._collapsible_section = section
        section._source_group = group
        section.content._source_group = group
        group.setVisible(not hidden)
        section.setVisible(not hidden)
        section.expandedChanged.connect(lambda value, key=key: self._remember(key, value))
        identity = id(group)
        self._sections[identity] = (group, section)
        group.setProperty("collapsibleSectionIdentity", str(identity))
        group.destroyed.connect(self._forget_section)
        del item  # replaceWidget transfers ownership of the old layout item.

    @QtCore.pyqtSlot(QtCore.QObject)
    def _forget_section(self, destroyed):
        identity = destroyed.property("collapsibleSectionIdentity")
        if identity:
            self._sections.pop(int(identity), None)

    def eventFilter(self, obj, event):
        if self._installing:
            return False
        kind = event.type()
        if kind == QtCore.QEvent.Show and isinstance(obj, (QtWidgets.QDialog, QtWidgets.QMainWindow, QtWidgets.QGroupBox)):
            self.install_tree(obj)
        elif kind in (QtCore.QEvent.ShowToParent, QtCore.QEvent.HideToParent) and isinstance(obj, QtWidgets.QWidget):
            section = getattr(obj, "_collapsible_section", None)
            if section is not None:
                # Explicit feature hide/show still affects the whole category;
                # hiding an ancestor while collapsing must not hide the header.
                section.setVisible(not obj.isHidden())
        elif kind == QtCore.QEvent.LayoutRequest:
            group = getattr(obj, "_source_group", None)
            section = (getattr(group, "_collapsible_section", None)
                       if group is not None and not sip.isdeleted(group) else None)
            if (section is not None and not sip.isdeleted(section) and isinstance(group, QtWidgets.QGroupBox)
                    and group.title() and section.header.text() != group.title()):
                section.header.setText(group.title())
                section.header.setAccessibleName(group.title())
        return False


class MultipleOfSpinBox(QSpinBox):
    """Integer spin box that stores only multiples of a fixed step."""

    def __init__(self, multiple=32, parent=None):
        super().__init__(parent)
        self._multiple = max(1, int(multiple))
        self.setSingleStep(self._multiple)
        self.setKeyboardTracking(False)
        self.editingFinished.connect(self.snap_to_multiple)

    def _snap_value(self, value):
        return snap_int_to_multiple(
            value,
            self._multiple,
            minimum=self.minimum(),
            maximum=self.maximum(),
        )

    def setValue(self, value):
        super().setValue(self._snap_value(value))

    def stepBy(self, steps):
        self.setValue(int(self.value()) + int(steps) * self._multiple)

    def valueFromText(self, text):
        return self._snap_value(text)

    def snap_to_multiple(self):
        snapped = self._snap_value(self.value())
        if snapped != self.value():
            self.setValue(snapped)


class PercentSliderControl(QWidget):
    """Horizontal slider that exposes a 0.0-1.0 float value and shows percent text."""

    valueChanged = pyqtSignal(float)

    def __init__(self, minimum=0.0, maximum=1.0, value=0.0, parent=None):
        super().__init__(parent)
        self._minimum = float(minimum)
        self._maximum = float(maximum)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(self._to_int(self._minimum), self._to_int(self._maximum))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)
        self.slider.setTickPosition(QtWidgets.QSlider.NoTicks)

        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setMinimumWidth(48)
        self.value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.setValue(value)

    @staticmethod
    def _to_int(value):
        return int(round(float(value) * 100))

    @staticmethod
    def _to_float(value):
        return max(0.0, min(1.0, int(value) / 100.0))

    def setRange(self, minimum, maximum):
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self.slider.setRange(self._to_int(self._minimum), self._to_int(self._maximum))
        self.setValue(self.value())

    def setSingleStep(self, step):
        self.slider.setSingleStep(max(1, self._to_int(step)))

    def setDecimals(self, _decimals):
        pass

    def value(self):
        return round(max(self._minimum, min(self._maximum, int(self.slider.value()) / 100.0)), 2)

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def setValue(self, value):
        value = max(self._minimum, min(self._maximum, float(value)))
        self.slider.setValue(self._to_int(value))
        self._update_label()

    def setToolTip(self, text):
        super().setToolTip(text)
        self.slider.setToolTip(text)
        self.value_label.setToolTip(text)

    def _update_label(self):
        self.value_label.setText(f"{int(round(self.value() * 100))}%")

    def _on_slider_changed(self, _value):
        self._update_label()
        self.valueChanged.emit(self.value())


def theme_colors(stylesheet, manifest=None):
    """Resolve shared chrome colors from the active theme and its optional manifest."""
    # Ignore an earlier generated block when applying/scaling a theme again.
    source = re.sub(r"/\* DarkFusion dock chrome start \*/.*?/\* DarkFusion dock chrome end \*/",
                    "", stylesheet or "", flags=re.DOTALL)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    rules = re.findall(r"([^{}]+)\{([^{}]*)\}", source)

    def css_color(selectors, properties):
        result = None
        for selector, body in rules:
            if not set(part.strip() for part in selector.split(",")).intersection(selectors):
                continue
            for declaration in body.split(";"):
                name, _, value = declaration.partition(":")
                if name.strip().lower() not in properties:
                    continue
                matches = re.findall(r"#[0-9a-fA-F]{3,8}\b", value)
                candidate = QColor(matches[-1] if matches else value.strip())
                rgb = re.fullmatch(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)", value.strip())
                if not candidate.isValid() and rgb:
                    channels = [min(255, max(0, round(float(v)))) for v in rgb.groups()[:3]]
                    alpha = float(rgb.group(4)) if rgb.group(4) is not None else 255
                    if rgb.group(4) is not None and "." in rgb.group(4) and alpha <= 1:
                        alpha *= 255
                    candidate = QColor(*channels, min(255, max(0, round(alpha))))
                if candidate.isValid():
                    result = candidate
        return result

    def color(name, fallback):
        candidate = QColor(str((manifest or {}).get(name, "")))
        return candidate if candidate.isValid() else fallback

    def mix(first, second, amount):
        return QColor(*(round(a * (1 - amount) + b * amount)
                        for a, b in zip(first.getRgb()[:3], second.getRgb()[:3])))

    panel = color("panel", css_color({"QDockWidget"}, {"background", "background-color"})
                  or css_color({"QWidget", "QMainWindow"}, {"background", "background-color"}) or QColor("#171b21"))
    border = color("border", css_color({"QDockWidget"}, {"border", "border-color"})
                   or css_color({"QFrame", "QGroupBox"}, {"border", "border-color"})
                   or mix(panel, QColor("white") if panel.lightness() < 128 else QColor("black"), .22))
    accent = color("accent", css_color({"QPushButton:hover", "QLineEdit:focus"}, {"border", "border-color"})
                   or css_color({"QTabBar::tab:selected", "QMenu::item:selected"}, {"background", "background-color"})
                   or QColor("#65b7ff"))
    text = color("text", css_color({"QWidget", "QMainWindow", "QDialog", "QLabel"}, {"color"})
                 or QColor("#e6edf3" if panel.lightness() < 128 else "#17202b"))
    if text.alpha() == 0:
        text = QColor("#e6edf3" if panel.lightness() < 128 else "#17202b")
    return dict(panel=panel, border=border, accent=accent, text=text,
                base=color("base", css_color({"QWidget", "QMainWindow", "QDialog"}, {"background", "background-color"}) or panel),
                surface=color("surface", mix(panel, text, .06)),
                muted=color("muted", mix(text, panel, .35)),
                hover=mix(panel, accent, .22), separator=mix(panel, border, .45))


def dock_chrome_stylesheet(stylesheet, manifest=None):
    """Theme-aware dock boundaries only; never style a canvas or its contents."""
    colors = theme_colors(stylesheet, manifest)
    panel, border, accent = (colors[key] for key in ("panel", "border", "accent"))
    separator, hover = colors["separator"].name(), colors["hover"].name()
    border, accent = border.name(), accent.name()
    return f"""
/* DarkFusion dock chrome start */
QDockWidget {{
    border: none;
}}
QDockWidget::title {{
    border: 1px solid {border};
    padding: 5px 10px;
    text-align: left;
    font-weight: 600;
}}
QWidget[darkfusionDockBody="true"] {{
    border: 1px solid {border};
    border-top: none;
    border-bottom-left-radius: 3px;
    border-bottom-right-radius: 3px;
}}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background-color: {hover};
    border-color: {accent};
}}
QToolButton[darkfusionDockHelp="true"] {{
    background: transparent;
    border: 1px solid {border};
    border-radius: 3px;
    padding: 0px;
    margin: 0px;
    font-weight: 700;
}}
QToolButton[darkfusionDockHelp="true"]:hover {{
    background-color: {hover};
    border-color: {accent};
}}
QMainWindow::separator {{
    width: 6px;
    height: 6px;
    margin: 0px;
    background-color: {separator};
}}
QMainWindow::separator:hover {{ background-color: {accent}; }}
/* DarkFusion dock chrome end */
"""


def connect_weak_bound_signal(signal, callback):
    """Connect a Qt signal to a bound method without keeping the owner alive."""
    weak_callback = weakref.WeakMethod(callback)

    def wrapper(*args, **kwargs):
        method = weak_callback()
        if method is None:
            try:
                signal.disconnect(wrapper)
            except (TypeError, RuntimeError):
                pass
            return None

        try:
            owner = method.__self__
            if owner is None or sip.isdeleted(owner):
                try:
                    signal.disconnect(wrapper)
                except (TypeError, RuntimeError):
                    pass
                return None

            return method(*args, **kwargs)

        except RuntimeError:
            try:
                signal.disconnect(wrapper)
            except (TypeError, RuntimeError):
                pass
            return None

    signal.connect(wrapper)
    return wrapper


class DockHelpButtonController(QObject):
    """Place a working context-help button beside a dock's native controls."""

    def __init__(self, dock, callback, parent=None):
        super().__init__(parent or dock)
        self.dock = dock
        self.button = QtWidgets.QToolButton(dock)
        self.button.setText("?")
        self.button.setObjectName(f"{dock.objectName()}_help_button")
        self.button.setProperty("darkfusionDockHelp", True)
        self.button.setToolTip(f"About the {dock.windowTitle()} dock")
        self.button.setAccessibleName(f"Help for {dock.windowTitle()} dock")
        self.button.setAutoRaise(True)
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.setFixedSize(20, 20)
        self.button.clicked.connect(callback)
        dock.installEventFilter(self)
        dock.topLevelChanged.connect(self._on_top_level_changed)
        dock.windowTitleChanged.connect(self._on_title_changed)
        self.reposition()

    def _on_title_changed(self, title):
        self.button.setToolTip(f"About the {title} dock")
        self.button.setAccessibleName(f"Help for {title} dock")

    def _on_top_level_changed(self, floating):
        self.button.setVisible(not bool(floating))
        if not floating:
            QtCore.QTimer.singleShot(0, self.reposition)

    def reposition(self):
        if self.dock.isFloating():
            self.button.hide()
            return
        # Leave room for Qt's native float and close buttons.
        self.button.move(max(4, self.dock.width() - 70), 2)
        self.button.raise_()
        self.button.show()

    def eventFilter(self, watched, event):
        if watched is self.dock and event.type() in (
            QEvent.Resize,
            QEvent.Show,
            QEvent.LayoutRequest,
            QEvent.StyleChange,
        ):
            QtCore.QTimer.singleShot(0, self.reposition)
        return False


class RedBoxDelegate(QStyledItemDelegate):

    def paint(self, painter, option, index):
        """
        Custom painting to draw a red border around the selected item.
        """
        super().paint(painter, option, index)

        if option.state & QStyle.State_Selected:
            pen = QPen(QColor("red"), 2, Qt.SolidLine)
            painter.setPen(pen)

            rect = option.rect
            painter.drawRect(rect.adjusted(1, 1, -1, -1))  # Draw inside the item boundaries


class ClassColorButton(QPushButton):
    colorRequested = pyqtSignal(str)
    resetRequested = pyqtSignal(str)

    def __init__(self, class_name, parent=None):
        super().__init__(parent)
        self.class_name = class_name
        self.setFixedSize(24, 24)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setToolTip("Left-click to choose color. Right-click to reset to generated color.")

    def set_class_color(self, color):
        if not isinstance(color, QColor) or not color.isValid():
            color = QColor("#ffffff")

        self.setStyleSheet(
            "QPushButton {"
            f"background-color: {color.name()};"
            "border: 1px solid rgba(255, 255, 255, 120);"
            "border-radius: 12px;"
            "min-width: 24px;"
            "max-width: 24px;"
            "min-height: 24px;"
            "max-height: 24px;"
            "}"
            "QPushButton:hover {"
            "border: 1px solid rgba(255, 255, 255, 220);"
            "}"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.resetRequested.emit(self.class_name)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            self.colorRequested.emit(self.class_name)
            event.accept()
            return

        super().mousePressEvent(event)


class ROIEdgeSliders(QtCore.QObject):
    """Four sliders stuck to the edges of the display that drive the active ROI.

    Mapping: top=X offset, bottom=Width, left=Y offset, right=Height.
    Shown only while ROI is enabled; the center of the view stays clickable.
    """

    THICKNESS = 18

    def __init__(self, main_window, host):
        super().__init__(host)
        self.main_window = main_window
        self.host = host
        self.active_roi_id = 1
        self._syncing = False

        self.top = QtWidgets.QSlider(Qt.Horizontal, host)
        self.bottom = QtWidgets.QSlider(Qt.Horizontal, host)
        self.left = QtWidgets.QSlider(Qt.Vertical, host)
        self.right = QtWidgets.QSlider(Qt.Vertical, host)
        self._edges = {
            "x": self.top,
            "width": self.bottom,
            "y": self.left,
            "height": self.right,
        }
        for name, edge in self._edges.items():
            edge.setProperty("roiEdgeControl", True)
            edge.setObjectName("roi_edge_" + name)
            edge.setToolTip({"x": "Move the selected ROI left/right.",
                             "y": "Move the selected ROI up/down.",
                             "width": "Adjust the selected ROI width (image pixels).",
                             "height": "Adjust the selected ROI height (image pixels)."}[name])
            edge.hide()

        self.apply_theme()

        self.top.valueChanged.connect(lambda v: self._drive("x", v))
        self.bottom.valueChanged.connect(lambda v: self._drive("width", v))
        self.left.valueChanged.connect(lambda v: self._drive("y", v))
        self.right.valueChanged.connect(lambda v: self._drive("height", v))

        host.installEventFilter(self)

    def apply_theme(self):
        manifest = getattr(self.main_window, "_theme_manifest_for_style", None)
        colors = theme_colors(self.main_window.styleSheet(), manifest() if callable(manifest) else None)
        panel, border, accent = (colors[key].name() for key in ("panel", "border", "accent"))
        style = f'''
QSlider[roiEdgeControl="true"] {{ background: {panel}; border: none; border-radius: 4px; padding: 0px; margin: 0px; min-width: 0px; min-height: 0px; }}
QSlider[roiEdgeControl="true"]::groove:horizontal {{ height: 6px; background: {border}; border: none; margin: 0px; }}
QSlider[roiEdgeControl="true"]::groove:vertical {{ width: 6px; background: {border}; border: none; margin: 0px; }}
QSlider[roiEdgeControl="true"]::handle:horizontal {{ background: {accent}; border: 1px solid {border}; width: 16px; margin: -5px 0px; border-radius: 3px; }}
QSlider[roiEdgeControl="true"]::handle:vertical {{ background: {accent}; border: 1px solid {border}; height: 16px; margin: 0px -5px; border-radius: 3px; }}
QSlider[roiEdgeControl="true"]::sub-page:horizontal, QSlider[roiEdgeControl="true"]::sub-page:vertical {{ background: {border}; border: none; }}
QSlider[roiEdgeControl="true"]::add-page:horizontal, QSlider[roiEdgeControl="true"]::add-page:vertical {{ background: {border}; border: none; }}
'''
        for edge in self._edges.values():
            edge.setStyleSheet(style)

    def _real_slider(self, name):
        suffix = "" if self.active_roi_id == 1 else "_2"
        return getattr(self.main_window, f"{name}_spin_slider{suffix}", None)

    def set_active_roi(self, roi_id):
        self.active_roi_id = 1 if int(roi_id) == 1 else 2
        self._pull_from_real()

    def _drive(self, name, value):
        if self._syncing:
            return
        real = self._real_slider(name)
        if real is not None:
            real.setValue(int(value))

    def _pull_from_real(self):
        """Mirror the real ROI slider ranges/values onto the edge sliders."""
        self._syncing = True
        try:
            for name, edge in self._edges.items():
                real = self._real_slider(name)
                if real is None:
                    continue
                edge.blockSignals(True)
                edge.setRange(real.minimum(), real.maximum())
                edge.setValue(real.value())
                edge.blockSignals(False)
        finally:
            self._syncing = False
        # Left (Y) inverted so dragging up moves the ROI up; right (H) stays
        # normal so up = taller.
        self.left.setInvertedAppearance(True)
        self.right.setInvertedAppearance(False)

    def connect_real_sliders(self):
        """Keep edge sliders in step when the real sliders change elsewhere."""
        for roi_id in (1, 2):
            suffix = "" if roi_id == 1 else "_2"
            for name in ("x", "y", "width", "height"):
                real = getattr(self.main_window, f"{name}_spin_slider{suffix}", None)
                if real is not None:
                    real.valueChanged.connect(lambda _v, n=name, r=roi_id: self._on_real_changed(n, r))

    def _on_real_changed(self, name, roi_id):
        if roi_id != self.active_roi_id or self._syncing:
            return
        edge = self._edges.get(name)
        real = self._real_slider(name)
        if edge is None or real is None:
            return
        self._syncing = True
        edge.blockSignals(True)
        edge.setRange(real.minimum(), real.maximum())
        edge.setValue(real.value())
        edge.blockSignals(False)
        self._syncing = False

    def reposition(self):
        w = self.host.width()
        h = self.host.height()
        t = self.THICKNESS
        self.top.setGeometry(t, 0, max(1, w - 2 * t), t)
        self.bottom.setGeometry(t, h - t, max(1, w - 2 * t), t)
        self.left.setGeometry(0, t, t, max(1, h - 2 * t))
        self.right.setGeometry(w - t, t, t, max(1, h - 2 * t))

    def set_visible(self, visible):
        if visible:
            self._pull_from_real()
            self.reposition()
        for edge in self._edges.values():
            edge.setVisible(bool(visible))
            if visible:
                edge.raise_()

    def eventFilter(self, watched, event):
        if watched is self.host and event.type() in (QEvent.Resize, QEvent.Show):
            self.reposition()
        return super().eventFilter(watched, event)


def _format_slider_value_label(self, slider_name, value):
    if slider_name == "timmer_speed":
        return f"{int(value)} ms"
    if slider_name == "font_size_slider":
        return f"{int(value)} pt"
    if slider_name == "dot_size_slider":
        return f"{int(value)} px"
    if slider_name == "grey_scale_slider":
        return f"{int(value):+d}"

    slider = getattr(self, slider_name, None)
    if slider is not None and getattr(slider, "maximum", lambda: 0)() == 100:
        return f"{int(value)}%"

    return str(int(value))


def update_slider_value_label(self, slider_name, value=None):
    labels = getattr(self, "_slider_value_labels", {})
    label = labels.get(slider_name)
    slider = getattr(self, slider_name, None)
    if label is None or slider is None:
        return

    if value is None:
        value = slider.value()
    label.setText(self._format_slider_value_label(slider_name, value))


def update_slider_value_labels(self):
    for slider_name in getattr(self, "_slider_value_labels", {}):
        self.update_slider_value_label(slider_name)

