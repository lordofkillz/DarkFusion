"""Verify the dependencies and local files needed to launch UltraDarkFusion."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "UltraDarkFusion"

REQUIRED_FILES = (
    "UltraDarkFusion_v5.2.py",
    "UltraDarkFusion_v5.2.ui",
    "ui_ultradarkfusion_v5_2.py",
    "dinov5_2.py",
    "sahi_predict_wrapperv5.py",
    "prediction_size_filter.py",
    "splash_utils.py",
    "darkfusion_command_runner.py",
    "darkfusion_ultralytics_batch.py",
    "darkfusion_ultralytics_cli.py",
    "darkfusion_ultralytics_train.py",
    "darkfusion_ultralytics_tune.py",
    "darkfusion_validation_review.py",
    "botsort.yaml",
    "Sam/FSRCNN_x4.pb",
    "styles/images/default.png",
    "styles/images/default_temp.png",
    "images/sight_overlays.json",
)

REQUIRED_IMPORTS = (
    "PyQt5",
    "GPUtil",
    "PIL",
    "cv2",
    "deep_translator",
    "diffusers",
    "dotenv",
    "groundingdino",
    "matplotlib",
    "mediapipe",
    "mss",
    "numpy",
    "onnx",
    "pandas",
    "plotly",
    "psutil",
    "pyautogui",
    "pyqtgraph",
    "qt_material",
    "sahi",
    "scipy",
    "shapely",
    "speech_recognition",
    "tensorrt",
    "torch",
    "torchvision",
    "transformers",
    "ultralytics",
    "yaml",
    "yt_dlp",
)


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        print(f"ERROR: Python 3.12 is required; found {sys.version.split()[0]}.")
        return 1

    missing_files = [name for name in REQUIRED_FILES if not (APP_ROOT / name).is_file()]
    import_failures: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # report every missing/broken binary binding
            import_failures.append(f"{module_name}: {exc}")

    if missing_files:
        print("ERROR: required repository files are missing:")
        for name in missing_files:
            print(f"  - UltraDarkFusion/{name}")

    if import_failures:
        print("ERROR: dependency imports failed:")
        for failure in import_failures:
            print(f"  - {failure}")

    if missing_files or import_failures:
        return 1

    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    optional_models = (
        "Sam/sam3.pt",
        "Sam/groundingdino_swint_ogc.pth",
        "yolo26n-reid.onnx",
    )
    missing_models = [name for name in optional_models if not (APP_ROOT / name).is_file()]
    if missing_models:
        print("Optional model files not installed:")
        for name in missing_models:
            print(f"  - UltraDarkFusion/{name}")
        print("See MODEL_SETUP.md.")

    print("UltraDarkFusion launch requirements verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
