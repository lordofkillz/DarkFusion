"""Verify Label Frame works for every playback and live-capture source."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

SPEC = importlib.util.spec_from_file_location(
    "capture_frame_app_under_test", APP_DIR / "UltraDarkFusion_v5.2.py"
)
APP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = APP
SPEC.loader.exec_module(APP)


CAPTURE_METHODS = (
    "_video_annotation_source",
    "_current_label_frame_source",
    "_capture_annotation_source_details",
    "capture_annotation_paths",
    "_record_capture_annotation_frame",
    "label_current_video_frame",
    "is_video_url",
    "_clean_video_source_name",
    "_safe_video_name_from_source",
)


class SourceCombo:
    def __init__(self, text):
        self.text = text

    def currentText(self):
        return self.text


class CaptureFrameLabelingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frame = np.full((24, 32, 3), (12, 80, 190), dtype=np.uint8)

    def owner(self, source, output=True):
        owner = types.SimpleNamespace()
        for name in CAPTURE_METHODS:
            setattr(
                owner,
                name,
                types.MethodType(getattr(APP.MainWindow, name), owner),
            )
        owner.current_playback_original_source = source
        owner.current_playback_source = source
        owner.input_selection = SourceCombo(source)
        owner._last_video_frame_bgr = self.frame.copy()
        owner.output_path = str(self.root) if output else ""
        owner.video_source_cache = {}
        owner.normalize_path = lambda value: os.path.normpath(str(value)) if value else ""
        owner.get_image_extension = mock.Mock(return_value=".png")
        owner._ensure_video_dataset_metadata = mock.Mock()
        owner._open_video_frame_in_image_labeler = mock.Mock()
        owner.set_output_directory = mock.Mock(return_value=False)
        owner.current_video_frame_index = mock.Mock(return_value=12)
        owner.seek_video_frame = mock.Mock(return_value=True)
        owner.video_annotation_paths = mock.Mock()
        owner._record_video_annotation_frame = mock.Mock()
        return owner

    def test_live_sources_capture_matching_image_and_label_identity(self):
        cases = (
            ("Desktop", "desktop", "desktop_frame_0.png"),
            ("2", "camera", "camera_2_frame_0.png"),
            (
                "https://www.youtube.com/watch?v=abc123&token=private",
                "stream",
                "watch_frame_0.png",
            ),
            ("rtsp://user:password@example.com/live?token=private", "stream", "live_frame_0.png"),
        )
        for source, expected_kind, expected_name in cases:
            with self.subTest(source=source), mock.patch.object(APP, "QMessageBox"):
                owner = self.owner(source)
                self.assertTrue(owner.label_current_video_frame())
                opened = Path(owner._open_video_frame_in_image_labeler.call_args.args[0])
                self.assertEqual(opened.name, expected_name)
                self.assertTrue(opened.is_file())
                owner._ensure_video_dataset_metadata.assert_called_once_with(str(opened.parent))

                manifest_path = opened.parent / ".darkfusion" / "capture_annotations.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["source_kind"], expected_kind)
                self.assertEqual(manifest["workflow"], "snapshot_labeling")
                self.assertEqual(manifest["frames"]["0"]["image"], expected_name)
                self.assertEqual(
                    manifest["frames"]["0"]["label"],
                    opened.with_suffix(".txt").name,
                )
                self.assertNotIn("password", manifest["source"])
                self.assertNotIn("private", manifest["source"])

    def test_repeated_live_captures_share_session_and_never_overwrite(self):
        owner = self.owner("Desktop")
        with mock.patch.object(APP, "QMessageBox"):
            self.assertTrue(owner.label_current_video_frame())
            first = Path(owner._open_video_frame_in_image_labeler.call_args.args[0])
            self.assertTrue(owner.label_current_video_frame())
            second = Path(owner._open_video_frame_in_image_labeler.call_args.args[0])
        self.assertEqual(first.parent, second.parent)
        self.assertEqual(first.name, "desktop_frame_0.png")
        self.assertEqual(second.name, "desktop_frame_1.png")
        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())

    def test_live_capture_requests_output_folder_when_missing(self):
        owner = self.owner("Desktop", output=False)

        def choose_output():
            owner.output_path = str(self.root)
            return True

        owner.set_output_directory.side_effect = choose_output
        with mock.patch.object(APP, "QMessageBox"):
            self.assertTrue(owner.label_current_video_frame())
        owner.set_output_directory.assert_called_once_with()
        self.assertTrue(Path(owner._open_video_frame_in_image_labeler.call_args.args[0]).is_file())

    def test_local_video_keeps_exact_seekable_frame_workflow(self):
        source = self.root / "video.mp4"
        source.write_bytes(b"fixture")
        owner = self.owner(str(source))
        output = self.root / "video_Frames"
        image = output / "video_frame_12.png"
        label = output / "video_frame_12.txt"
        owner.video_annotation_paths.return_value = (str(output), str(image), str(label))

        with mock.patch.object(APP, "QMessageBox"):
            self.assertTrue(owner.label_current_video_frame())

        owner.seek_video_frame.assert_called_once_with(12, pause=True)
        owner._record_video_annotation_frame.assert_called_once_with(
            str(source), 12, str(image), str(label), sparse_manual=True
        )
        owner.set_output_directory.assert_not_called()
        self.assertTrue(image.is_file())
        self.assertEqual(owner._video_annotation_return_frame, 12)

    def test_choosing_output_location_does_not_start_extraction(self):
        owner = types.SimpleNamespace(
            output_path="",
            dialog_start_directory=mock.Mock(return_value=str(self.root)),
            remember_dialog_selection=mock.Mock(),
            set_extraction_ui_state=mock.Mock(),
            queue_settings_save=mock.Mock(),
            start_extraction_thread=mock.Mock(),
            start_camera_extraction_timer=mock.Mock(),
        )
        choose_location = types.MethodType(APP.MainWindow.set_output_directory, owner)
        with mock.patch.object(
            APP.QFileDialog,
            "getExistingDirectory",
            return_value=str(self.root),
        ):
            self.assertTrue(choose_location())
        self.assertEqual(owner.output_path, str(self.root))
        owner.start_extraction_thread.assert_not_called()
        owner.start_camera_extraction_timer.assert_not_called()

    def test_extract_location_prompt_returns_without_starting(self):
        owner = types.SimpleNamespace(
            output_path="",
            extraction_thread=None,
            _frame_extraction_url_resolver=None,
            _direct_extraction_resolver=None,
            qthread_is_running=mock.Mock(return_value=False),
            get_selected_video_path=mock.Mock(return_value=""),
            is_video_url=mock.Mock(return_value=False),
            set_output_directory=mock.Mock(return_value=True),
            prepare_runtime_for_frame_extraction=mock.Mock(),
            start_extraction_thread=mock.Mock(),
            start_camera_extraction_timer=mock.Mock(),
        )
        extract = types.MethodType(APP.MainWindow.on_extract_button_clicked, owner)
        with mock.patch.object(APP, "QMessageBox"):
            extract()
        owner.set_output_directory.assert_called_once_with()
        owner.prepare_runtime_for_frame_extraction.assert_not_called()
        owner.start_extraction_thread.assert_not_called()
        owner.start_camera_extraction_timer.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
