import inspect
import os
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_dataset_analysis_scope import app
from ui_ultradarkfusion_v5_2 import Ui_mainWindow


class PersistenceHarness(app.QtWidgets.QMainWindow):
    collect_persistent_ui_state = app.MainWindow.collect_persistent_ui_state
    setup_persistent_settings_bindings = app.MainWindow.setup_persistent_settings_bindings
    _widget_int_value = app.MainWindow._widget_int_value
    _widget_float_value = app.MainWindow._widget_float_value
    _widget_text_value = app.MainWindow._widget_text_value
    _combo_text_value = app.MainWindow._combo_text_value

    def __init__(self):
        super().__init__()
        self.settings = {}
        self._restoring_settings = False
        self.queue_settings_save = Mock()
        self.rapid_del_checkbox = app.QtWidgets.QCheckBox()
        self.outline_Checkbox = app.QtWidgets.QCheckBox()
        self.zoom_lock = app.QtWidgets.QCheckBox()
        self.debug_edge_display_checkbox = app.QtWidgets.QCheckBox()
        self.heatmap_Checkbox = app.QtWidgets.QCheckBox()
        self.screen_update = app.QtWidgets.QCheckBox()
        self.obb_checkbox = app.QtWidgets.QCheckBox()
        self.segmentation_checkbox = app.QtWidgets.QCheckBox()
        self.edge_slider_min = app.QtWidgets.QSlider()
        self.edge_slider_max = app.QtWidgets.QSlider()
        self.timmer_speed = app.QtWidgets.QSlider()
        self.timmer_speed.setMaximum(500)
        self.shade_slider = app.QtWidgets.QSlider()
        self.font_size_slider = app.QtWidgets.QSlider()
        self.dot_size_slider = app.QtWidgets.QSlider()
        self.box_size = app.QtWidgets.QSpinBox()
        self.max_label = app.QtWidgets.QSpinBox()
        self.heatmap_dropdown = app.QtWidgets.QComboBox()
        self.heatmap_dropdown.addItems(["JET", "VIRIDIS"])

    @staticmethod
    def selected_augmentation_output_format_value():
        return "source"

    @staticmethod
    def selected_nested_label_mode():
        return "hybrid"

    @staticmethod
    def current_roi_settings():
        return {}

    @staticmethod
    def current_layout_screen_metrics():
        return {}

    @staticmethod
    def normalize_inference_backend(value):
        return str(value or "auto")

    @staticmethod
    def _normalize_setting_path(value):
        return str(value or "")


class LabelingSettingsPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = app.QApplication.instance() or app.QApplication([])

    def setUp(self):
        self.window = PersistenceHarness()

    def tearDown(self):
        self.window.close()

    def test_runtime_minimum_label_control_accepts_one_pixel(self):
        host = app.QtWidgets.QMainWindow()
        ui = Ui_mainWindow()
        ui.setupUi(host)
        self.addCleanup(host.close)
        ui.box_size.setValue(1)
        self.assertEqual(ui.box_size.minimum(), 1)
        self.assertEqual(ui.box_size.value(), 1)

    def test_labeling_controls_queue_an_immediate_debounced_save(self):
        self.window.setup_persistent_settings_bindings()
        controls = (
            self.window.rapid_del_checkbox,
            self.window.outline_Checkbox,
            self.window.zoom_lock,
            self.window.debug_edge_display_checkbox,
            self.window.heatmap_Checkbox,
            self.window.screen_update,
            self.window.obb_checkbox,
            self.window.segmentation_checkbox,
        )
        for control in controls:
            control.setChecked(True)
        for slider, value in (
            (self.window.edge_slider_min, 15),
            (self.window.edge_slider_max, 80),
            (self.window.timmer_speed, 250),
            (self.window.shade_slider, 65),
            (self.window.font_size_slider, 8),
            (self.window.dot_size_slider, 12),
            (self.window.box_size, 6),
            (self.window.max_label, 90),
        ):
            slider.setValue(value)
        self.window.heatmap_dropdown.setCurrentIndex(1)

        self.assertEqual(self.window.queue_settings_save.call_count, 17)

    def test_collect_includes_all_labeling_values(self):
        self.window.rapid_del_checkbox.setChecked(True)
        self.window.outline_Checkbox.setChecked(True)
        self.window.zoom_lock.setChecked(True)
        self.window.debug_edge_display_checkbox.setChecked(True)
        self.window.heatmap_Checkbox.setChecked(True)
        self.window.screen_update.setChecked(True)
        self.window.obb_checkbox.setChecked(True)
        self.window.segmentation_checkbox.setChecked(True)
        self.window.edge_slider_min.setValue(12)
        self.window.edge_slider_max.setValue(77)
        self.window.timmer_speed.setValue(225)
        self.window.shade_slider.setValue(63)
        self.window.font_size_slider.setValue(7)
        self.window.dot_size_slider.setValue(14)
        self.window.box_size.setValue(5)
        self.window.max_label.setValue(85)
        self.window.heatmap_dropdown.setCurrentText("VIRIDIS")

        self.window.collect_persistent_ui_state()

        expected = {
            "rapidDeleteEnabled": True,
            "snappingEnabled": True,
            "zoomLockEnabled": True,
            "edgePreviewEnabled": True,
            "heatmapEnabled": True,
            "samShowPreview": True,
            "samGenerateObb": True,
            "samGenerateSegmentation": True,
            "edgeThresholdLow": 12,
            "edgeThresholdHigh": 77,
            "autoScanIntervalMs": 225,
            "heatmapColormap": "VIRIDIS",
            "shadeSlider": 63,
            "fontSizeSlider": 7,
            "keypointDotSize": 14,
            "minLabelSize": 5,
            "maxLabelPercent": 85,
        }
        for key, value in expected.items():
            self.assertEqual(self.window.settings.get(key), value, key)

    def test_restore_method_contains_matching_widget_mappings(self):
        source = inspect.getsource(app.MainWindow.restore_persistent_ui_state)
        for key in (
            "rapidDeleteEnabled", "snappingEnabled", "zoomLockEnabled",
            "edgePreviewEnabled", "edgeThresholdLow", "edgeThresholdHigh",
            "heatmapEnabled", "heatmapColormap", "autoScanIntervalMs",
            "samShowPreview", "samGenerateObb", "samGenerateSegmentation",
        ):
            self.assertIn(key, source)


if __name__ == "__main__":
    unittest.main()
