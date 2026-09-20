"""Cached UI translations with bounded GUI work and background-only I/O.

Combo boxes are translated when painted. Their model, currentText(), itemData(),
and emitted values remain the original application values.
"""
from collections import deque
import hashlib
import html
import json
import logging
from pathlib import Path
import queue
import re
import threading
import time
import weakref

from PyQt5 import QtCore, QtGui, QtWidgets, sip

log = logging.getLogger(__name__)
RESOURCE_SUFFIXES = {
    ".css", ".qss", ".stylesheet", ".ess", ".xml", ".gif", ".png", ".jpg",
    ".jpeg", ".bmp", ".svg", ".ico", ".webp", ".mp4", ".avi", ".mkv",
    ".pt", ".pth", ".onnx", ".engine", ".weights", ".cfg", ".yaml", ".yml",
    ".json", ".txt", ".csv", ".py", ".ui", ".dll", ".exe", ".bin",
}
PROTECTED_COMBOS = {
    "styleComboBox", "gif_change", "language_dropdown", "language_combo",
    "style_combo", "gif_combo", "classes_dropdown", "auto_head_target_class_combo",
    "filter_class_spinbox",
}
TECH_TERMS = re.compile(r"\b(?:DarkFusion|UltraDarkFusion|SAM\d*|YOLO[\w.-]*|CUDA|ONNX|TensorRT|DINO[\w.-]*|OpenCV|PyTorch|Ultralytics)\b")
PATH_IN_TEXT = re.compile(r"(?:[A-Za-z]:[\\/][^\n\"<>]+|https?://[^\s<>]+)")
TOKEN = re.compile(r"\{DF\d+\}")
_PROVIDER_BACKOFF = {}


def source_key(text):
    return "runtime_source_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def is_resource_text(text):
    text = str(text or "").strip()
    return bool(
        re.match(r"^(?:[A-Za-z]:[\\/]|[/\\]|https?://|file:)", text)
        or ("\n" not in text and Path(text).suffix.lower() in RESOURCE_SUFFIXES)
    )


def protected_combo(widget):
    name = widget.objectName()
    return (name in PROTECTED_COMBOS or "class" in name.lower()
            or bool(widget.property("translationSkip")))


def message_template(text):
    """Keep filenames, numbers, and technical names out of translation requests."""
    values = []

    def protect(match):
        values.append(match.group(0))
        return "{DF%d}" % (len(values) - 1)

    template = PATH_IN_TEXT.sub(protect, text)
    template = TECH_TERMS.sub(protect, template)
    template = re.sub(r"(?<![\w{])\d+(?:[.,]\d+)*(?:%)?(?![\w}])", protect, template)
    return template, values


def restore_template(text, values):
    return TOKEN.sub(lambda m: values[int(m.group(0)[3:-1])], text)


def online_translate(text, language):
    """Bounded requests with a second provider when the first is unavailable."""
    import requests
    from bs4 import BeautifulSoup

    for provider in ("google", "mymemory"):
        if time.monotonic() < _PROVIDER_BACKOFF.get(provider, 0):
            continue
        try:
            if provider == "google":
                with requests.get(
                    "https://translate.google.com/m",
                    params={"sl": "en", "tl": language, "q": text}, timeout=(3.0, 6.0),
                ) as response:
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, "html.parser")
                    result = soup.find("div", {"class": "result-container"}) or soup.find("div", {"class": "t0"})
                    if result is None:
                        raise ValueError("Google returned no translation")
                    translated = result.get_text().strip()
            else:
                # MyMemory's documented maximum is 500 UTF-8 bytes per segment.
                if len(text.encode("utf-8")) > 500:
                    continue
                with requests.get(
                    "https://api.mymemory.translated.net/get",
                    params={"q": text, "langpair": f"en|{language}"}, timeout=(3.0, 6.0),
                ) as response:
                    response.raise_for_status()
                    payload = response.json()
                    if str(payload.get("responseStatus")) != "200" or payload.get("quotaFinished"):
                        raise ValueError("MyMemory is unavailable or has reached its quota")
                    translated = html.unescape(payload["responseData"]["translatedText"]).strip()
            if not translated or sorted(TOKEN.findall(translated)) != sorted(TOKEN.findall(text)):
                raise ValueError("Translation changed protected placeholders")
            return translated
        except (requests.RequestException, ValueError, KeyError, TypeError):
            _PROVIDER_BACKOFF[provider] = time.monotonic() + 300.0
    raise RuntimeError("Translation providers are temporarily unavailable")


def read_pack(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid translation dictionary: {path.name}")
    return {key: text for key, text in value.items() if key and isinstance(text, str)}


def write_pack(path, changes):
    """Merge into the latest file, preserving edits made outside this process."""
    current = read_pack(path)
    updated = dict(current)
    updated.update(changes)
    if updated == current:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    try:
        temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class _Results(QtCore.QObject):
    ready = QtCore.pyqtSignal(str, object)


class _CatalogWorker:
    """Only plain data crosses the worker boundary; Qt widgets stay on the GUI."""
    def __init__(self, directory, results, online, translator):
        self.directory = Path(directory)
        self.results = results
        self.online = online
        self.translator = translator or online_translate
        self.commands = queue.Queue()
        self.requests = queue.Queue(maxsize=1)
        self.in_flight = None
        self.stop = threading.Event()
        self.language = "en"
        self.english = {}
        self.packs = {}
        self.pending = {}
        self.failed = set()
        self.dirty = {}
        self.retry_after = 0.0
        self.network_thread = None
        if self.online:
            self.network_thread = threading.Thread(
                target=self.translate_requests, name="DarkFusion translation provider", daemon=True
            )
            self.network_thread.start()
        self.thread = threading.Thread(target=self.run, name="DarkFusion translations", daemon=True)
        self.thread.start()

    def emit(self, code):
        if self.stop.is_set():
            return
        pack = self.packs.get(code, {})
        phrases = {}
        for key, source in self.english.items():
            value = pack.get(key)
            if value and value != source and not is_resource_text(source):
                # Canonical source entries take priority over old positional keys.
                if source not in phrases or key.startswith("runtime_source_"):
                    phrases[source] = value
        try:
            self.results.ready.emit(code, phrases)
        except RuntimeError:
            self.stop.set()

    def load(self, code):
        if not self.english:
            self.english = read_pack(self.directory / "en.json")
        if code not in self.packs:
            self.packs[code] = read_pack(self.directory / f"{code}.json")
        self.emit(code)

    def flush(self):
        for code, changes in list(self.dirty.items()):
            try:
                write_pack(self.directory / f"{code}.json", changes)
            except (OSError, ValueError) as exc:
                log.warning("Could not update %s translations: %s", code, exc)
            else:
                self.dirty.pop(code, None)

    def collect(self, code, texts):
        self.load(code) if code not in self.packs else None
        pack = self.packs[code]
        known = {source: pack[key] for key, source in self.english.items()
                 if pack.get(key) and pack[key] != source}
        pending = self.pending.setdefault(code, {})
        for text in texts:
            key = source_key(text)
            if key not in self.english:
                self.english[key] = text
                self.dirty.setdefault("en", {})[key] = text
            translated = known.get(text)
            if translated:
                if pack.get(key) != translated:
                    pack[key] = translated
                    self.dirty.setdefault(code, {})[key] = translated
            elif code != "en" and (code, text) not in self.failed:
                pending[text] = key
        self.emit(code)

    def translate_requests(self):
        """Provider latency never holds up catalog reads or language changes."""
        while not self.stop.is_set():
            try:
                code, text, key = self.requests.get(timeout=0.05)
            except queue.Empty:
                continue
            if self.stop.is_set():
                break
            translated, error = None, None
            try:
                translated = str(self.translator(text, code) or "").strip()
                if not translated or sorted(TOKEN.findall(translated)) != sorted(TOKEN.findall(text)):
                    raise ValueError("Translation changed protected placeholders")
            except Exception as exc:
                error = str(exc)
            if not self.stop.is_set():
                # Only the catalog thread touches dictionaries or emits Qt results.
                self.commands.put(("translated", code, (text, key, translated, error)))

    def accept_translation(self, code, result):
        text, key, translated, error = result
        self.in_flight = None
        # A new language selection can discover this phrase again while its
        # previous request is pending. Its one response satisfies both discoveries.
        self.pending.get(code, {}).pop(text, None)
        if error is not None:
            self.failed.add((code, text))
            self.retry_after = time.monotonic() + 30.0
            log.warning("Background translation unavailable; retaining cached text: %s", error)
        elif translated != text:
            self.packs[code][key] = translated
            self.dirty.setdefault(code, {})[key] = translated
            self.emit(code)
        else:
            self.failed.add((code, text))

    def run(self):
        last_flush = time.monotonic()
        # Finish queued catalog writes at shutdown, without waiting for a provider.
        while not self.stop.is_set() or not self.commands.empty():
            try:
                command = self.commands.get(timeout=0.05)
            except queue.Empty:
                command = None
            try:
                if command:
                    kind, code, payload = command
                    if kind == "load":
                        self.language = code
                        self.load(code)
                    elif kind == "collect":
                        self.collect(code, payload)
                    elif kind == "translated":
                        self.accept_translation(code, payload)
                # Prioritize language changes over more network requests.
                if (not self.stop.is_set() and self.commands.empty() and self.online
                        and self.in_flight is None and time.monotonic() >= self.retry_after):
                    code = self.language
                    pending = self.pending.get(code, {})
                    if pending:
                        text, key = next(iter(pending.items()))
                        pending.pop(text)
                        self.in_flight = (code, text, key)
                        self.requests.put_nowait(self.in_flight)
                if time.monotonic() - last_flush >= 0.5:
                    self.flush()
                    last_flush = time.monotonic()
            except (OSError, ValueError) as exc:
                log.warning("Could not load translation catalog: %s", exc)
        self.flush()


class _ComboDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, controller, combo):
        super().__init__(combo)
        self.controller = weakref.ref(controller)

    def displayText(self, value, locale):
        controller = self.controller()
        text = super().displayText(value, locale)
        return controller.translate_text(text) if controller else text


class TranslationController(QtCore.QObject):
    def __init__(self, owner, translation_dir, online=True, translator=None):
        super().__init__(owner)
        self.owner = weakref.ref(owner)
        self.language = "en"
        self.phrases = {}
        self._bindings = weakref.WeakKeyDictionary()
        self._widgets = weakref.WeakValueDictionary()
        self._queue = deque()
        self._queued = set()
        self._missing = set()
        self._submitted = set()
        self._applying = False
        self._closed = False
        self._results = _Results(self)
        self._results.ready.connect(self._catalog_ready, QtCore.Qt.QueuedConnection)
        self._worker = _CatalogWorker(translation_dir, self._results, online, translator)
        self._drain_timer = QtCore.QTimer(self)
        self._drain_timer.setSingleShot(True)
        self._drain_timer.timeout.connect(self._drain)
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(200)
        self._save_timer.timeout.connect(self._submit_missing)
        app = QtWidgets.QApplication.instance()
        app.installEventFilter(self)
        app.aboutToQuit.connect(self.close)
        self.enqueue(owner, tree=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._submit_missing()
        self._drain_timer.stop()
        self._save_timer.stop()
        QtWidgets.QApplication.instance().removeEventFilter(self)
        self._worker.stop.set()
        # No join on the GUI thread: an in-flight network request has a timeout.

    def set_language(self, code):
        if not re.fullmatch(r"[a-z]{2}(?:-[A-Za-z]{2,4})?", str(code)):
            code = "en"
        self.language = code
        self._submitted.clear()
        self._missing.clear()
        self._worker.commands.put(("load", code, None))
        for widget in list(self._widgets.values()):
            self.enqueue(widget)

    def _catalog_ready(self, code, phrases):
        if self._closed:
            return
        self.phrases[code] = phrases
        if code == self.language:
            # A background translation may arrive one phrase at a time. Reapply
            # only widgets whose recorded source text is represented in this
            # result instead of walking the entire UI for every phrase.
            changed_sources = set(phrases)
            for widget, bindings in list(self._bindings.items()):
                if any(record[0] in changed_sources for record in bindings.values()):
                    self.enqueue(widget)

    def translate_text(self, text, key=None):
        text = str(text or "")
        if not text or is_resource_text(text):
            return text
        phrases = self.phrases.get(self.language, {})
        if self.language != "en" and text in phrases:
            return phrases[text]
        template, values = message_template(text)
        if self.language != "en" and template in phrases:
            return restore_template(phrases[template], values)
        # Avoid recording counters, technical identifiers, or enormous data views.
        content = TOKEN.sub("", template).strip()
        if (len(text) <= 3000 and re.search(r"[A-Za-z]{2}", content)
                and not re.fullmatch(r"[A-Z0-9_. /+-]+", content)):
            if template not in self._submitted:
                self._missing.add(template)
                if not self._save_timer.isActive():
                    self._save_timer.start()
        return text

    def _submit_missing(self):
        if self._missing:
            missing, self._missing = self._missing, set()
            self._submitted.update(missing)
            self._worker.commands.put(("collect", self.language, missing))

    def enqueue(self, widget, tree=False):
        if self._closed or widget is None or sip.isdeleted(widget):
            return
        identity = id(widget)
        if (identity, tree) not in self._queued:
            self._queued.add((identity, tree))
            self._queue.append((weakref.ref(widget), identity, tree))
            if not self._drain_timer.isActive():
                self._drain_timer.start(0)

    def _drain(self):
        deadline = time.perf_counter() + 0.004
        while self._queue and time.perf_counter() < deadline:
            reference, identity, tree = self._queue.popleft()
            self._queued.discard((identity, tree))
            widget = reference()
            if widget is None or sip.isdeleted(widget):
                continue
            if tree:
                for child in widget.children():
                    if isinstance(child, (QtWidgets.QWidget, QtWidgets.QAction)):
                        self.enqueue(child, tree=True)
            self._apply_widget(widget)
        if self._queue:
            self._drain_timer.start(0)

    def source_text(self, widget, kind, index=None):
        record = self._bindings.get(widget, {}).get((kind, index))
        if record:
            if kind != "tab" or widget.tabText(index) == record[1]:
                return record[0]
        if kind == "tab":
            return widget.tabText(index)
        return widget.text() if hasattr(widget, "text") else ""

    def _field(self, widget, kind, getter, setter, index=None):
        current = getter()
        if not isinstance(current, str) or not current:
            return
        bindings = self._bindings.setdefault(widget, {})
        previous = bindings.get((kind, index))
        source = previous[0] if previous and current == previous[1] else current
        translated = self.translate_text(source)
        bindings[kind, index] = (source, translated)
        if current != translated:
            blocker = QtCore.QSignalBlocker(widget)
            setter(translated)
            del blocker

    def _apply_widget(self, widget):
        if self._applying or widget.property("translationSkip"):
            return
        self._applying = True
        try:
            self._widgets[id(widget)] = widget
            field = lambda kind, get, put, index=None: self._field(widget, kind, get, put, index)
            if isinstance(widget, (QtWidgets.QLabel, QtWidgets.QAbstractButton, QtWidgets.QAction)):
                field("text", widget.text, widget.setText)
            if isinstance(widget, QtWidgets.QGroupBox):
                field("title", widget.title, widget.setTitle)
            if isinstance(widget, QtWidgets.QWidget):
                field("tooltip", widget.toolTip, widget.setToolTip)
                field("whatsthis", widget.whatsThis, widget.setWhatsThis)
                if widget.isWindow() or isinstance(widget, QtWidgets.QDockWidget):
                    field("window_title", widget.windowTitle, widget.setWindowTitle)
            if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
                field("placeholder", widget.placeholderText, widget.setPlaceholderText)
            if isinstance(widget, QtWidgets.QMessageBox):
                field("message", widget.text, widget.setText)
                field("informative", widget.informativeText, widget.setInformativeText)
            if isinstance(widget, QtWidgets.QMenu):
                field("title", widget.title, widget.setTitle)
            if isinstance(widget, QtWidgets.QComboBox) and not protected_combo(widget):
                if not isinstance(widget.itemDelegate(), _ComboDelegate):
                    widget.setItemDelegate(_ComboDelegate(self, widget))
                # The model remains untouched, including editable values and roles.
                for index in range(widget.count()):
                    self.translate_text(widget.itemText(index))
                widget.update()
            if isinstance(widget, (QtWidgets.QTabWidget, QtWidgets.QToolBox)):
                for index in range(widget.count()):
                    if isinstance(widget, QtWidgets.QTabWidget):
                        field("tab", lambda i=index: widget.tabText(i), lambda text, i=index: widget.setTabText(i, text), index)
                    else:
                        field("toolbox", lambda i=index: widget.itemText(i), lambda text, i=index: widget.setItemText(i, text), index)
            if isinstance(widget, QtWidgets.QTableWidget):
                for index in range(widget.columnCount()):
                    header = widget.horizontalHeaderItem(index)
                    if header:
                        field("header", header.text, header.setText, index)
        except RuntimeError:
            pass  # A transient popup can be deleted before its queued update.
        finally:
            self._applying = False

    def eventFilter(self, obj, event):
        if self._closed or self._applying:
            return False
        kind = event.type()
        if kind == QtCore.QEvent.Paint and isinstance(obj, QtWidgets.QComboBox):
            if self.language != "en" and not protected_combo(obj) and not obj.isEditable():
                option = QtWidgets.QStyleOptionComboBox()
                obj.initStyleOption(option)
                translated = self.translate_text(obj.currentText())
                if translated != option.currentText:
                    option.currentText = translated
                    painter = QtWidgets.QStylePainter(obj)
                    painter.drawComplexControl(QtWidgets.QStyle.CC_ComboBox, option)
                    painter.drawControl(QtWidgets.QStyle.CE_ComboBoxLabel, option)
                    painter.end()
                    return True
        elif kind == QtCore.QEvent.Paint and isinstance(obj, (QtWidgets.QLabel, QtWidgets.QAbstractButton)):
            self._apply_widget(obj)
        elif kind in (QtCore.QEvent.Show, QtCore.QEvent.ChildPolished):
            child = event.child() if kind == QtCore.QEvent.ChildPolished else obj
            if isinstance(child, (QtWidgets.QWidget, QtWidgets.QAction)):
                self.enqueue(child, tree=True)
        elif kind in (QtCore.QEvent.ToolTipChange, QtCore.QEvent.WindowTitleChange, QtCore.QEvent.ActionChanged):
            self.enqueue(obj)
        return False
