"""Exercise language changes with real Qt widgets and no network/model imports."""

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets

from darkfusion_translation import TranslationController, source_key


class TranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)
        if Path("C:/Windows/Fonts/segoeui.ttf").exists():
            QtGui.QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
            cls.app.setFont(QtGui.QFont("Segoe UI", 10))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.owner = QtWidgets.QWidget()
        self.owner.setObjectName("mainWindow")
        self.layout = QtWidgets.QVBoxLayout(self.owner)
        self.controller = None
        self.release_events = []

    def tearDown(self):
        for event in self.release_events:
            event.set()
        if self.controller is not None:
            self.controller.close()
            # Production close deliberately never waits on network I/O. The test
            # must wait before removing a directory the final cache flush uses.
            self.controller._worker.thread.join(timeout=4)
            self.assertFalse(self.controller._worker.thread.is_alive())
            network_thread = self.controller._worker.network_thread
            if network_thread is not None:
                network_thread.join(timeout=4)
                self.assertFalse(network_thread.is_alive())
        self.owner.close()
        self.owner.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.app.processEvents()
        self.temporary.cleanup()

    def write_cache(self, phrases, extra_english=None, extra_french=None):
        english = {source_key(text): text for text in phrases}
        french = {source_key(text): translated for text, translated in phrases.items()}
        english.update(extra_english or {})
        french.update(extra_french or {})
        for language, data in (("en", english), ("fr", french)):
            (self.directory / f"{language}.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )

    def start(self, online=False, translator=None):
        self.owner.show()
        self.controller = TranslationController(
            self.owner, self.directory, online=online, translator=translator
        )
        self.controller.set_language("fr")
        return self.controller

    def until(self, predicate, message="asynchronous translation did not finish", timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        self.assertTrue(predicate(), message)

    def pump(self, duration=0.15):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)

    def add_label(self, text, name=""):
        label = QtWidgets.QLabel(text, self.owner)
        label.setObjectName(name)
        self.layout.addWidget(label)
        return label

    def test_cached_labels_actions_tooltips_and_switch_back_to_english(self):
        self.write_cache({
            "Label Frame": "Annoter l’image",
            "Draw labels on this frame": "Dessiner les annotations sur cette image",
            "About this tool": "À propos de cet outil",
            "Open Video": "Ouvrir une vidéo",
        })
        button = QtWidgets.QPushButton("Label Frame", self.owner)
        button.setToolTip("Draw labels on this frame")
        button.setWhatsThis("About this tool")
        self.layout.addWidget(button)
        action = QtWidgets.QAction("Open Video", self.owner)
        changed = []
        action.changed.connect(lambda: changed.append(True))
        self.start()
        self.until(lambda: button.text() == "Annoter l’image")
        self.assertEqual(button.toolTip(), "Dessiner les annotations sur cette image")
        self.assertEqual(button.whatsThis(), "À propos de cet outil")
        self.assertEqual(action.text(), "Ouvrir une vidéo")
        self.assertEqual(changed, [], "Language changes must not invoke action handlers")
        self.controller.set_language("en")
        self.until(lambda: button.text() == "Label Frame")
        self.assertEqual(action.text(), "Open Video")
        self.assertEqual(button.toolTip(), "Draw labels on this frame")

    def test_legacy_object_keys_still_work(self):
        self.write_cache({}, {"label_frame": "Label Frame"}, {"label_frame": "Annoter l’image"})
        label = self.add_label("Label Frame", "label_frame")
        self.start()
        self.until(lambda: label.text() == "Annoter l’image")

    def test_tabs_toolbox_headers_group_titles_and_input_placeholders(self):
        self.write_cache({
            "Settings": "Paramètres", "Training": "Entraînement",
            "Confidence": "Confiance", "Options": "Options françaises",
            "Search annotations": "Rechercher des annotations",
        })
        tabs = QtWidgets.QTabWidget(self.owner)
        tabs.addTab(QtWidgets.QWidget(), "Settings")
        toolbox = QtWidgets.QToolBox(self.owner)
        toolbox.addItem(QtWidgets.QWidget(), "Training")
        table = QtWidgets.QTableWidget(1, 1, self.owner)
        table.setHorizontalHeaderLabels(["Confidence"])
        table.setItem(0, 0, QtWidgets.QTableWidgetItem("user annotation"))
        group = QtWidgets.QGroupBox("Options", self.owner)
        field = QtWidgets.QLineEdit(self.owner)
        field.setPlaceholderText("Search annotations")
        field.setText("my-dataset/classes.txt")
        for widget in (tabs, toolbox, table, group, field):
            self.layout.addWidget(widget)
        self.start()
        self.until(lambda: tabs.tabText(0) == "Paramètres")
        self.assertEqual(toolbox.itemText(0), "Entraînement")
        self.assertEqual(table.horizontalHeaderItem(0).text(), "Confiance")
        self.assertEqual(group.title(), "Options françaises")
        self.assertEqual(field.placeholderText(), "Rechercher des annotations")
        self.assertEqual(field.text(), "my-dataset/classes.txt")
        self.assertEqual(table.item(0, 0).text(), "user annotation")

    def test_combo_translation_preserves_values_data_selection_and_signals(self):
        self.write_cache({"Greedy": "Glouton", "Hungarian": "Hongrois"})
        combo = QtWidgets.QComboBox(self.owner)
        combo.setObjectName("matching_method_combo")
        combo.addItem("Greedy", "greedy_algorithm")
        combo.addItem("Hungarian", "hungarian_algorithm")
        combo.setCurrentIndex(1)
        self.layout.addWidget(combo)
        changes = []
        combo.currentTextChanged.connect(lambda value: changes.append(("text", value)))
        combo.currentIndexChanged.connect(lambda value: changes.append(("index", value)))
        self.start()
        self.until(lambda: combo.itemDelegate().displayText("Hungarian", QtCore.QLocale()) == "Hongrois")
        self.assertEqual(combo.currentText(), "Hungarian")
        self.assertEqual(combo.itemText(0), "Greedy")
        self.assertEqual(combo.currentData(), "hungarian_algorithm")
        self.assertEqual(combo.currentIndex(), 1)
        self.assertEqual(combo.findText("Greedy"), 0)
        self.assertEqual(changes, [])
        self.controller.set_language("en")
        self.until(lambda: combo.itemDelegate().displayText("Hungarian", QtCore.QLocale()) == "Hungarian")
        self.assertEqual(combo.currentText(), "Hungarian")
        self.assertEqual(changes, [])

    def test_resource_class_and_language_combos_remain_original_even_with_bad_cache(self):
        examples = {
            "styleComboBox": ["Default", "ckp.css", "EAL.stylesheet"],
            "gif_change": ["None", "darkfusion.gif"],
            "model_picker": ["yolo11n.pt", "models/model.onnx"],
            "classes_dropdown": ["person", "car"],
            "language_dropdown": ["English", "Français"],
        }
        self.write_cache({text: f"BAD {text}" for values in examples.values() for text in values})
        combos = []
        for name, values in examples.items():
            combo = QtWidgets.QComboBox(self.owner)
            combo.setObjectName(name)
            for value in values:
                combo.addItem(value, value)
            self.layout.addWidget(combo)
            combos.append((combo, values, combo.itemDelegate()))
        marker = self.add_label("Settings")
        en_path, fr_path = self.directory / "en.json", self.directory / "fr.json"
        for path, value in ((en_path, "Settings"), (fr_path, "Paramètres")):
            data = json.loads(path.read_text(encoding="utf-8"))
            data[source_key("Settings")] = value
            path.write_text(json.dumps(data), encoding="utf-8")
        self.start()
        self.until(lambda: marker.text() == "Paramètres")
        for combo, values, original_delegate in combos:
            with self.subTest(selector=combo.objectName()):
                self.assertEqual([combo.itemText(index) for index in range(combo.count())], values)
                self.assertEqual([combo.itemData(index) for index in range(combo.count())], values)
                delegate = combo.itemDelegate()
                if hasattr(delegate, "displayText"):
                    for value in values:
                        self.assertEqual(delegate.displayText(value, QtCore.QLocale()), value)
                else:
                    self.assertIs(delegate, original_delegate)

    def test_new_dialog_message_and_changed_text_translate_without_language_reselection(self):
        self.write_cache({
            "Ready": "Prêt", "Finished": "Terminé", "Review": "Vérifier",
            "Delete selected annotations?": "Supprimer les annotations sélectionnées ?",
        })
        status = self.add_label("Ready")
        self.start()
        self.until(lambda: status.text() == "Prêt")
        status.setText("Finished")
        dialog = QtWidgets.QMessageBox(self.owner)
        dialog.setWindowTitle("Review")
        dialog.setText("Delete selected annotations?")
        # This Windows Qt offscreen plugin crashes inside QMessageBox.show(),
        # also with no controller. Exercise its real polish lifecycle instead;
        # the ordinary QDialog test below exercises live Show events separately.
        dialog.ensurePolished()
        self.until(lambda: dialog.text() == "Supprimer les annotations sélectionnées ?")
        self.until(lambda: status.text() == "Terminé")
        self.assertEqual(dialog.windowTitle(), "Vérifier")

    def test_duplicate_object_names_keep_each_source_phrase(self):
        self.write_cache({"Status": "État", "Memory": "Mémoire"})
        first = self.add_label("Status", "tileTitle")
        second = self.add_label("Memory", "tileTitle")
        self.start()
        self.until(lambda: first.text() == "État")
        self.assertEqual(second.text(), "Mémoire")
        self.controller.set_language("en")
        self.until(lambda: first.text() == "Status")
        self.assertEqual(second.text(), "Memory")

    def test_background_translation_does_not_block_heartbeat_and_persists_new_text(self):
        self.write_cache({}, {"existing_key": "Earlier text"}, {"existing_key": "Texte antérieur"})
        started = threading.Event()
        release = threading.Event()
        self.release_events.append(release)
        calls = []

        def translator(text, language):
            calls.append((text, language, threading.get_ident()))
            started.set()
            release.wait(3)
            return "Traduction nouvelle" if text == "New interface text" else f"fr:{text}"

        label = self.add_label("New interface text")
        timer = QtCore.QTimer(self.owner)
        timer.setInterval(5)
        ticks = []
        timer.timeout.connect(lambda: ticks.append(time.monotonic()))
        timer.start()
        self.start(online=True, translator=translator)
        self.until(started.is_set)
        before = len(ticks)
        self.pump(0.15)
        self.assertGreaterEqual(len(ticks) - before, 5)
        self.assertEqual(label.text(), "New interface text")
        release.set()
        self.until(lambda: label.text() == "Traduction nouvelle")

        def persisted():
            try:
                data = json.loads((self.directory / "fr.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            return data.get(source_key("New interface text")) == "Traduction nouvelle"

        self.until(persisted)
        self.assertTrue(calls)
        self.assertTrue(all(call[2] != threading.get_ident() for call in calls))
        english = json.loads((self.directory / "en.json").read_text(encoding="utf-8"))
        french = json.loads((self.directory / "fr.json").read_text(encoding="utf-8"))
        self.assertEqual(english[source_key("New interface text")], "New interface text")
        self.assertEqual(english["existing_key"], "Earlier text")
        self.assertEqual(french["existing_key"], "Texte antérieur")

    def test_cache_files_load_off_the_gui_thread(self):
        self.write_cache({"Settings": "Paramètres"})
        label = self.add_label("Settings")
        reads = []
        original_read = Path.open

        def record_read(path, *args, **kwargs):
            if path.parent == self.directory:
                reads.append(threading.get_ident())
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "open", record_read):
            self.start()
            self.until(lambda: label.text() == "Paramètres")
        self.assertTrue(reads, "The test must observe at least one cache read")
        self.assertNotIn(threading.get_ident(), reads)

    def test_network_failure_keeps_interface_usable_and_english_fallback(self):
        self.write_cache({})
        calls = []

        def translator(text, language):
            calls.append((text, language))
            raise TimeoutError("translation service unavailable")

        label = self.add_label("New feature")
        self.start(online=True, translator=translator)
        self.until(lambda: bool(calls))
        self.pump(0.2)
        self.assertEqual(label.text(), "New feature")
        self.assertLessEqual(sum(text == "New feature" for text, _ in calls), 1)
        self.controller.set_language("en")
        self.pump(0.05)
        self.assertEqual(self.controller.language, "en")

    def test_offline_mode_uses_cache_without_calling_translator(self):
        self.write_cache({"Settings": "Paramètres"})
        cached = self.add_label("Settings")
        missing = self.add_label("Not yet translated")
        calls = []
        self.start(online=False, translator=lambda text, language: calls.append((text, language)))
        self.until(lambda: cached.text() == "Paramètres")
        self.pump(0.1)
        self.assertEqual(missing.text(), "Not yet translated")
        self.assertEqual(calls, [])

    def test_slow_old_language_result_does_not_replace_new_selection(self):
        self.write_cache({})
        started = threading.Event()
        release = threading.Event()
        self.release_events.append(release)

        def translator(text, language):
            started.set()
            release.wait(3)
            return "Texte français"

        label = self.add_label("Late result")
        self.start(online=True, translator=translator)
        self.until(started.is_set)
        self.controller.set_language("en")
        release.set()
        self.pump(0.3)
        self.assertEqual(self.controller.language, "en")
        self.assertEqual(label.text(), "Late result")

    def test_new_language_cache_loads_while_previous_network_request_is_blocked(self):
        self.write_cache({"Settings": "Paramètres"})
        (self.directory / "de.json").write_text(
            json.dumps({source_key("Settings"): "Einstellungen"}), encoding="utf-8"
        )
        started = threading.Event()
        release = threading.Event()
        self.release_events.append(release)
        calls = []

        def translator(text, language):
            calls.append((text, language))
            started.set()
            release.wait(3)
            return f"{language}:{text}"

        label = self.add_label("Settings")
        self.add_label("Text missing from every catalog")
        timer = QtCore.QTimer(self.owner)
        timer.setInterval(5)
        ticks = []
        timer.timeout.connect(lambda: ticks.append(time.monotonic()))
        timer.start()
        self.start(online=True, translator=translator)
        self.until(started.is_set)
        self.assertEqual(label.text(), "Paramètres")
        self.assertNotIn("de", self.controller.phrases)
        before = len(ticks)
        self.controller.set_language("de")
        self.until(lambda: label.text() == "Einstellungen", timeout=0.75)
        self.pump(0.15)
        self.assertFalse(release.is_set())
        self.assertGreaterEqual(len(ticks) - before, 5)
        self.assertEqual(len(calls), 1, "Only one provider request may be outstanding")
        release.set()
        self.pump(0.15)
        self.assertEqual(label.text(), "Einstellungen")

    def test_close_returns_without_waiting_for_a_blocked_provider(self):
        self.write_cache({})
        started = threading.Event()
        release = threading.Event()
        self.release_events.append(release)
        calls = []

        def translator(text, language):
            calls.append((text, language))
            started.set()
            release.wait(3)
            return f"fr:{text}"

        label = self.add_label("Unfinished request")
        self.start(online=True, translator=translator)
        self.until(started.is_set)
        began = time.monotonic()
        self.controller.close()
        self.assertLess(time.monotonic() - began, 0.1)
        self.assertFalse(release.is_set())
        self.until(lambda: not self.controller._worker.thread.is_alive(), timeout=0.75)
        self.assertTrue(self.controller._worker.network_thread.is_alive())
        release.set()
        self.until(lambda: not self.controller._worker.network_thread.is_alive())
        self.assertEqual(label.text(), "Unfinished request")
        self.assertEqual(len(calls), 1)

    def test_destroyed_dialog_is_safe_when_background_result_arrives(self):
        self.write_cache({})
        started = threading.Event()
        release = threading.Event()
        self.release_events.append(release)

        def translator(text, language):
            started.set()
            release.wait(3)
            return f"fr:{text}"

        self.start(online=True, translator=translator)
        dialog = QtWidgets.QDialog(self.owner)
        dialog.setWindowTitle("Transient dialog")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Transient label", dialog))
        dialog.show()
        self.until(started.is_set)
        dialog.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        release.set()
        self.pump(0.3)
        survivor = self.add_label("Surviving label")
        self.until(lambda: survivor.text() == "fr:Surviving label")


if __name__ == "__main__":
    unittest.main()
