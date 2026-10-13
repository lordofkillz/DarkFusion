"""Compact provider setup, with optional ONNX troubleshooting overrides."""
from __future__ import annotations

from PyQt5 import QtCore, QtWidgets
import darkfusion_windows_ml as runtime

_workers = set()  # A closed dialog must not destroy a running setup thread.
PROVIDERS = (
    ("DirectML", "directml", "DmlExecutionProvider"),
    ("TensorRT RTX", "tensorrt_rtx", "NvTensorRTRTXExecutionProvider"),
    ("CUDA", "cuda", "CUDAExecutionProvider"),
    ("TensorRT (classic)", "tensorrt", "TensorrtExecutionProvider"),
    ("OpenVINO", "openvino", "OpenVINOExecutionProvider"),
    ("MIGraphX", "migraphx", "MIGraphXExecutionProvider"),
    ("QNN", "qnn", "QNNExecutionProvider"),
    ("Vitis AI", "vitisai", "VitisAIExecutionProvider"),
    ("WebGPU (experimental)", "webgpu", "WebGpuExecutionProvider"),
    ("ROCm", "rocm", "ROCMExecutionProvider"),
    ("CoreML", "coreml", "CoreMLExecutionProvider"),
    ("XNNPACK", "xnnpack", "XnnpackExecutionProvider"),
    ("CPU", "cpu", "CPUExecutionProvider"),
)


class ProviderWorker(QtCore.QThread):
    result = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str, float)

    def __init__(self, options, requested=None):
        super().__init__()
        self.options = dict(options)
        self.requested = requested

    def run(self):
        try:
            if self.requested is not None:
                options = dict(self.options, download_providers=True)
                runtime.prepare(self.requested, options, progress=self.progress.emit)
            self.result.emit(runtime.refresh(self.options))
        except Exception as exc:
            self.error.emit(str(exc))


class OnnxRuntimeSettings(QtWidgets.QGroupBox):
    changed = QtCore.pyqtSignal()

    def __init__(self, provider_combo, status_label, options=None, parent=None, *, backend_combo=None):
        super().__init__("ONNX provider setup", parent)
        self.provider_combo, self.status_label = provider_combo, status_label
        self.backend_combo = backend_combo
        self._catalog = {}
        self._available = set()
        self._status = None
        self._busy = False
        self._loading = True
        layout = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("Refresh providers")
        self.setup_button = QtWidgets.QPushButton("Download / prepare provider")
        toolbar.addWidget(self.refresh_button)
        toolbar.addWidget(self.setup_button)
        layout.addLayout(toolbar)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.device_label = QtWidgets.QLabel("Checking available GPUs and providers...")
        self.device_label.setWordWrap(True)
        layout.addWidget(self.device_label)

        self.model_info_label = QtWidgets.QLabel("Model settings are detected automatically. You do not need to choose where it was exported.")
        self.model_info_label.setWordWrap(True)
        layout.addWidget(self.model_info_label)

        self.download_check = QtWidgets.QCheckBox("Download compatible providers automatically")
        self.download_check.setToolTip("One-time setup, not a per-frame operation. Disable for offline or metered connections. Failed automatic setup retains DirectML/CPU.")
        self.experimental_check = QtWidgets.QCheckBox("Show experimental providers for manual testing")
        self.experimental_check.setToolTip("Opt-in only. Experimental providers are not selected by Automatic.")
        self.cpu_fallback_check = QtWidgets.QCheckBox("Allow CPU for unsupported graph operations")
        self.compare_check = QtWidgets.QCheckBox("Automatic: compare once and remember the fastest")
        self.compare_check.setToolTip("Warm-up and timed inference at first use, without rendering a preview or writing labels. Cached results are reused. Model, GPU/driver, provider, precision or input-size changes invalidate the result. Initial provider compilation may take longer.")
        layout.addWidget(self.compare_check)
        self.recompare_button = QtWidgets.QPushButton("Recheck speed on next model load")
        self.recompare_button.setToolTip("Use when other work affected the first timing. Keeps old cache records and does not touch images/labels.")
        layout.addWidget(self.recompare_button)

        self.advanced_toggle = QtWidgets.QToolButton()
        self.advanced_toggle.setText("Advanced / troubleshooting (usually not needed)")
        self.advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(QtCore.Qt.RightArrow)
        self.advanced_toggle.setCheckable(True)
        layout.addWidget(self.advanced_toggle)
        self.advanced_panel = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QVBoxLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.addWidget(self.download_check)
        advanced_layout.addWidget(self.cpu_fallback_check)
        advanced_layout.addWidget(self.experimental_check)
        advanced_note = QtWidgets.QLabel("Leave model overrides on Automatic for normal exports. Use these only for models with missing or unusual metadata; project classes and colors are unchanged.")
        advanced_note.setWordWrap(True)
        advanced_layout.addWidget(advanced_note)
        form = QtWidgets.QFormLayout()
        if self.backend_combo is not None:
            form.addRow("Compatibility loader:", self.backend_combo)
        self.policy_combo = self._combo([
            ("Maximum performance (hardware recommendation)", "max_performance"),
            ("Prefer GPU", "prefer_gpu"), ("Power efficient", "max_efficiency"),
            ("Prefer NPU", "prefer_npu"), ("Prefer CPU", "prefer_cpu"),
        ])
        form.addRow("Automatic policy:", self.policy_combo)
        self.device_spin = QtWidgets.QSpinBox()
        self.device_spin.setRange(0, 31)
        self.device_spin.setToolTip("Zero-based device index for a manually selected provider. Automatic policy selects its own device.")
        form.addRow("Manual device index:", self.device_spin)
        self.cache_edit = QtWidgets.QLineEdit()
        self.cache_edit.setPlaceholderText("Default: persistent per-model cache")
        form.addRow("Cache folder:", self.cache_edit)
        self.framework_combo = self._combo([("Automatic", "auto"), ("Ultralytics / YOLO", "ultralytics"), ("Darknet two-output ONNX", "darknet"), ("DAMO scores / pixel boxes (RGB 0..255)", "damo")])
        form.addRow("Model format override:", self.framework_combo)
        self.darknet_units_combo = self._combo([("Normalized xyxy (standard two-output export)", "normalized"), ("Input-image pixel xyxy", "pixels")])
        form.addRow("Darknet ONNX box units:", self.darknet_units_combo)
        self.damo_resize_combo = self._combo([("Official export preset / metadata", "auto"), ("Stretch to input size (Ns)", "stretch"), ("Keep aspect ratio, zero pad bottom/right (L)", "letterbox")])
        form.addRow("DAMO ONNX resize:", self.damo_resize_combo)
        self.output_combo = self._combo([
            ("Automatic", "auto"),
            ("Raw YOLO: boxes + class scores", "ultralytics"),
            ("Classic YOLO: boxes + objectness + class scores", "yolo_objectness"),
            ("End-to-end: boxes + confidence + class ID", "end2end"),
        ])
        form.addRow("Output format override:", self.output_combo)
        self.task_combo = self._combo([(label, value) for label, value in (
            ("Automatic", "auto"), ("Detection", "detect"), ("Segmentation", "segment"),
            ("Pose", "pose"), ("Oriented boxes", "obb"), ("Classification", "classify"),
        )])
        form.addRow("Task override:", self.task_combo)
        self.names_edit = QtWidgets.QLineEdit()
        self.names_edit.setPlaceholderText('Optional JSON class list, e.g. ["person", "enemy"]')
        self.names_edit.setToolTip("Class names in model ID order, not the project class order. Leave empty when the exported model contains names.")
        form.addRow("Model class names:", self.names_edit)
        self.plugin_name_edit = QtWidgets.QLineEdit()
        self.plugin_name_edit.setPlaceholderText("Optional standalone EP name")
        self.plugin_path_edit = QtWidgets.QLineEdit()
        self.plugin_path_edit.setPlaceholderText("Optional trusted plugin DLL (not a legacy provider DLL)")
        form.addRow("Custom EP name:", self.plugin_name_edit)
        form.addRow("Custom EP DLL:", self.plugin_path_edit)
        advanced_layout.addLayout(form)
        self.reset_model_button = QtWidgets.QPushButton("Reset model overrides to Automatic")
        self.reset_model_button.setToolTip("Clears only ONNX decoding overrides. Keeps your provider, cache, project classes, colors and dataset unchanged. Applies on the next model load.")
        advanced_layout.addWidget(self.reset_model_button)
        self.advanced_panel.hide()
        layout.addWidget(self.advanced_panel)
        note = QtWidgets.QLabel("Choose a provider to test it directly. Automatic comparison does not restrict manual choices. Changes apply when you load or apply the model.")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.set_options(options or {})
        self._loading = False
        self.refresh_button.clicked.connect(self.refresh_providers)
        self.setup_button.clicked.connect(self.setup_provider)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.reset_model_button.clicked.connect(self._reset_model_overrides)
        self.recompare_button.clicked.connect(self._recompare)
        self.provider_combo.currentIndexChanged.connect(self._update_enabled)
        if self.backend_combo is not None:
            self.backend_combo.currentIndexChanged.connect(self._update_enabled)
        for widget in (self.policy_combo, self.framework_combo, self.output_combo, self.task_combo, self.darknet_units_combo, self.damo_resize_combo):
            widget.currentIndexChanged.connect(self._change)
        self.device_spin.valueChanged.connect(self._change)
        for widget in (self.download_check, self.cpu_fallback_check, self.experimental_check, self.compare_check):
            widget.toggled.connect(self._change)
        self.experimental_check.toggled.connect(self._refresh_choices)
        for widget in (self.cache_edit, self.names_edit, self.plugin_name_edit, self.plugin_path_edit):
            widget.editingFinished.connect(self._change)
        self._update_enabled()
        QtCore.QTimer.singleShot(0, self.refresh_providers)

    @staticmethod
    def _combo(items):
        combo = QtWidgets.QComboBox()
        for label, value in items:
            combo.addItem(label, value)
        return combo

    def _change(self, *_args):
        if not self._loading:
            self._update_enabled()
            self._update_model_info()
            self.changed.emit()

    def _toggle_advanced(self, checked):
        self.advanced_panel.setVisible(checked)
        self.advanced_toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)

    def _update_model_info(self):
        overridden = (any(combo.currentData() != "auto" for combo in
                          (self.framework_combo, self.output_combo, self.task_combo, self.damo_resize_combo))
                      or self.darknet_units_combo.currentData() != "normalized" or bool(self.names_edit.text().strip()))
        self.model_info_label.setText(
            "Saved model overrides are active. Review or reset them under Advanced / troubleshooting."
            if overridden else "Model settings are detected automatically. You do not need to choose where it was exported.")
        self.reset_model_button.setEnabled(overridden)

    def _reset_model_overrides(self):
        value = self.options()
        for key in ("framework", "output_format", "task", "names", "darknet_box_units", "damo_resize"):
            value[key] = runtime.DEFAULT_OPTIONS[key]
        self.set_options(value)
        self._change()

    def _update_enabled(self, *_args):
        automatic = self.provider_combo.currentData() == "auto"
        onnx_enabled = self.backend_combo is None or self.backend_combo.currentData() != "ultralytics"
        self.provider_combo.setEnabled(onnx_enabled)
        self.policy_combo.setEnabled(onnx_enabled and automatic)
        self.device_spin.setEnabled(onnx_enabled and not automatic)
        compare = onnx_enabled and automatic and self.policy_combo.currentData() in {"max_performance", "prefer_gpu"}
        self.compare_check.setVisible(onnx_enabled and automatic)
        self.compare_check.setEnabled(compare)
        self.recompare_button.setVisible(onnx_enabled and automatic)
        self.recompare_button.setEnabled(self.compare_check.isEnabled() and self.compare_check.isChecked())

    def _recompare(self):
        self._comparison_epoch += 1
        self._change()
        self.status_label.setText("Provider speed will be compared again on the next ONNX model load.")

    def _refresh_choices(self, *_args):
        if not self._loading and self._status is not None:
            self._result(self._status)

    def options(self):
        return runtime.settings({
            "auto_policy": self.policy_combo.currentData(), "device_id": self.device_spin.value(),
            "download_providers": self.download_check.isChecked(), "allow_experimental": self.experimental_check.isChecked(),
            "cpu_fallback": self.cpu_fallback_check.isChecked(), "cache_dir": self.cache_edit.text().strip(),
            "compare_providers": self.compare_check.isChecked(),
            "comparison_epoch": self._comparison_epoch,
            "framework": self.framework_combo.currentData(), "output_format": self.output_combo.currentData(),
            "darknet_box_units": self.darknet_units_combo.currentData(),
            "damo_resize": self.damo_resize_combo.currentData(),
            "task": self.task_combo.currentData(), "names": self.names_edit.text().strip(),
            "plugin_name": self.plugin_name_edit.text().strip(), "plugin_path": self.plugin_path_edit.text().strip(),
        })

    def set_options(self, options):
        loading = self._loading
        self._loading = True
        value = runtime.settings(options)
        for key, widget in (("auto_policy", self.policy_combo), ("framework", self.framework_combo),
                            ("output_format", self.output_combo), ("task", self.task_combo), ("darknet_box_units", self.darknet_units_combo), ("damo_resize", self.damo_resize_combo)):
            index = widget.findData(value[key])
            widget.setCurrentIndex(max(0, index))
        self.device_spin.setValue(value["device_id"])
        self.download_check.setChecked(value["download_providers"])
        self.experimental_check.setChecked(value["allow_experimental"])
        self.cpu_fallback_check.setChecked(value["cpu_fallback"])
        self.compare_check.setChecked(value["compare_providers"])
        self._comparison_epoch = value["comparison_epoch"]
        for key, widget in (("cache_dir", self.cache_edit), ("names", self.names_edit),
                            ("plugin_name", self.plugin_name_edit), ("plugin_path", self.plugin_path_edit)):
            widget.setText(str(value[key]))
        self._loading = loading
        self._update_model_info()
        self._update_enabled()

    def _start(self, requested=None):
        if self._busy:
            return
        self._busy = True
        self.refresh_button.setEnabled(False)
        self.setup_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.status_label.setText("Preparing provider setup..." if requested else "Discovering providers / GPUs...")
        worker = ProviderWorker(self.options(), requested)
        _workers.add(worker)
        worker.result.connect(self._result)
        worker.error.connect(self._error)
        worker.progress.connect(self._progress)
        worker.finished.connect(self._finish)
        worker.finished.connect(lambda: _workers.discard(worker))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def refresh_providers(self):
        self._start()

    def setup_provider(self):
        alias = self.provider_combo.currentData() or "auto"
        canonical = next((name for _label, value, name in PROVIDERS if value == alias), alias)
        self._start(canonical)

    def _progress(self, name, progress):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(progress))
        self.status_label.setText(f"Setting up {name}: {progress:.0f}%")

    def _finish(self):
        self._busy = False
        self.refresh_button.setEnabled(True)
        self.setup_button.setEnabled(runtime.installed())
        self.progress_bar.hide()

    def _error(self, message):
        self.status_label.setText(f"Provider setup: {message}")

    def _result(self, status):
        self._status = status
        self._available = set(status["available"])
        self._catalog = {item["name"].lower(): item for item in status["catalog"]}
        selected = self.provider_combo.currentData() or "auto"
        blocker = QtCore.QSignalBlocker(self.provider_combo)
        self.provider_combo.clear()
        self.provider_combo.addItem("Automatic (recommended)", "auto")
        experimental = self.experimental_check.isChecked()

        def permitted(name):
            entry = self._catalog.get(name.lower())
            if name not in self._available and not entry:
                return False
            uncertified = bool(entry and not entry["certified"]) or runtime.is_experimental(name)
            return experimental or not uncertified

        def ready(name):
            entry = self._catalog.get(name.lower())
            return name in self._available and (not entry or bool(entry.get("registered")))

        def suffix(name):
            if ready(name):
                return ""
            entry = self._catalog.get(name.lower(), {})
            if entry.get("error"):
                return " (setup failed)"
            if entry.get("state") in {"READY", "NOT_READY"}:
                return " (setup required)"
            return " (download available)"

        known = set()
        for label, alias, name in PROVIDERS:
            if name == "TensorrtExecutionProvider" and "TensorRTPluginExecutionProvider" in self._available:
                name = "TensorRTPluginExecutionProvider"
            known.add(name)
            if not permitted(name):
                continue
            self.provider_combo.addItem(label + suffix(name), alias)
        for name in sorted((self._available | {entry["name"] for entry in status["catalog"]}) - known):
            if not permitted(name):
                continue
            self.provider_combo.addItem(name + suffix(name), name)
        index = self.provider_combo.findData(selected)
        if index < 0:
            alias = next((alias for _label, alias, name in PROVIDERS if name == selected), selected)
            if selected == "TensorRTPluginExecutionProvider":
                alias = "tensorrt"
            index = self.provider_combo.findData(alias)
        if index < 0:
            # Preserve a saved explicit request, not an entire list of unusable
            # providers. Discovery must not silently change the chosen backend.
            label = next((label for label, alias, name in PROVIDERS if selected in (alias, name)), str(selected))
            self.provider_combo.addItem(label + " (saved choice unavailable)", selected)
            index = self.provider_combo.count() - 1
            self.provider_combo.model().item(index).setEnabled(False)
        self.provider_combo.setCurrentIndex(index)
        del blocker
        self._update_enabled()
        friendly = {name: label for label, _alias, name in PROVIDERS}
        friendly["TensorRTPluginExecutionProvider"] = "TensorRT (classic)"
        names = [friendly.get(name, name) for name in status["available"] if permitted(name) and ready(name)]
        self.status_label.setText(", ".join(names) or "No installed providers detected")
        self.status_label.setToolTip("Installed providers are runtime-wide. Automatic validates and times compatible providers when a model first runs; this list is not a model-specific speed result.")
        gpu_names = sorted({str(device["name"]) for device in status["devices"] if "GPU" in device["type"]})
        self.device_label.setText("Detected GPU(s): " + (", ".join(gpu_names) or "none reported; inspect available providers above"))
