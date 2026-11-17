"""CI package startup check; requires an explicit, fresh test data directory."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

from ultradarkfusion import data_directory, prepare

if not os.environ.get("DARKFUSION_DATA_DIR"):
    raise RuntimeError("Set DARKFUSION_DATA_DIR to a fresh test-only directory.")
base = data_directory()
base.mkdir(parents=True, exist_ok=True)
settings = base / "settings.json"
if settings.exists():
    raise RuntimeError("Use a fresh test-only directory; existing settings were preserved.")
settings.write_text(json.dumps({"sam3StartupWarmup": False}), encoding="utf-8")
os.environ.update(QT_QPA_PLATFORM="offscreen", DARKFUSION_SETTINGS_PATH=str(settings),
                  DARKFUSION_MODEL_DIR=str(base / "models"), YOLO_AUTOINSTALL="False",
                  YOLO_CONFIG_DIR=str(base / "ultralytics-config"),
                  HF_HOME=str(base / "cache/huggingface"), TORCH_HOME=str(base / "cache/torch"))
workspace = prepare(check=True) / "UltraDarkFusion"
os.chdir(workspace)
sys.path.insert(0, str(workspace))
start = time.perf_counter()
import UltraDarkFusion as desktop
import cv2
from darkfusion_environment import APP_SETTINGS_PATH, model_directory

assert Path(desktop.__file__).resolve().parent == workspace
assert APP_SETTINGS_PATH == str(settings)
assert model_directory() == str(base / "models")
assert hasattr(cv2, "dnn_superres"), "Contrib OpenCV features must survive dependency installation"
report = {"app": str(desktop.__file__), "import_seconds": round(time.perf_counter()-start, 3),
          "settings": str(settings), "model_directory": model_directory(), "opencv_contrib": True}
app = desktop.QApplication([])
app.setQuitOnLastWindowClosed(True)
window = desktop.MainWindow()
app.main_window = window
window.show()
report["main_window_created"] = True

def finish():
    try:
        window.saveSettings()
        saved = json.loads(settings.read_text(encoding="utf-8"))
        assert saved.get("sam3StartupWarmup") is False
        assert "inferenceBackend" in saved
        assert not getattr(window, "_sam3_startup_warmup_active", False)
        report["settings_saved"] = True
        report["sam_warmup_disabled"] = True
        report["saved_settings_count"] = len(saved)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        # closeEvent may defer closing while device discovery finishes. Let its
        # normal retry and cleanup run; app.quit() would bypass those safeguards.
        accepted = window.close()
        report["initial_close_accepted"] = bool(accepted)


def timeout():
    report["error"] = "Desktop did not finish its normal shutdown within 60 seconds."
    app.exit(1)

desktop.QTimer.singleShot(3500, finish)
desktop.QTimer.singleShot(60000, timeout)
exit_code = app.exec_()
report["closed"] = not window.isVisible()
(base / "startup-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
sys.exit(1 if "error" in report or not report["closed"] else exit_code)
