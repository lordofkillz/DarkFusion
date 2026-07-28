"""Verify the dependencies and local files needed to launch UltraDarkFusion."""

from __future__ import annotations

import importlib
import json
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

MODEL_BUNDLE_URL = (
    "https://drive.google.com/file/d/"
    "1j9Y-WpUDjPt67_U43lafO-7dTkxLJuPS/view?usp=sharing"
)
FEATURE_MODELS = {
    "Sam/sam3.pt": 3_000_000_000,
    "Sam/groundingdino_swint_ogc.pth": 600_000_000,
}
TRANSLATION_CODES = (
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "id",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "ru",
    "th",
    "tr",
    "uk",
    "vi",
    "zh-CN",
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

    translation_failures: list[str] = []
    translation_dir = APP_ROOT / "translations"
    try:
        english = json.loads((translation_dir / "en.json").read_text(encoding="utf-8"))
        expected_keys = set(english)
    except (OSError, json.JSONDecodeError) as exc:
        expected_keys = set()
        translation_failures.append(f"en.json: {exc}")

    for code in TRANSLATION_CODES:
        path = translation_dir / f"{code}.json"
        if not path.is_file():
            translation_failures.append(f"{code}.json is missing")
            continue
        try:
            catalog = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            translation_failures.append(f"{code}.json: {exc}")
            continue
        if expected_keys and set(catalog) != expected_keys:
            missing_count = len(expected_keys - set(catalog))
            extra_count = len(set(catalog) - expected_keys)
            translation_failures.append(
                f"{code}.json: {missing_count} missing and {extra_count} extra keys"
            )

    if missing_files:
        print("ERROR: required repository files are missing:")
        for name in missing_files:
            print(f"  - UltraDarkFusion/{name}")

    if import_failures:
        print("ERROR: dependency imports failed:")
        for failure in import_failures:
            print(f"  - {failure}")

    if translation_failures:
        print("ERROR: bundled translation catalogs are incomplete:")
        for failure in translation_failures:
            print(f"  - {failure}")

    if missing_files or import_failures or translation_failures:
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

    model_problems: list[str] = []
    for name, minimum_size in FEATURE_MODELS.items():
        path = APP_ROOT / name
        if not path.is_file():
            model_problems.append(f"missing: UltraDarkFusion/{name}")
        elif path.stat().st_size < minimum_size:
            model_problems.append(
                f"incomplete: UltraDarkFusion/{name} "
                f"({path.stat().st_size:,} bytes)"
            )
    if model_problems:
        print("Required model bundle not fully installed:")
        for problem in model_problems:
            print(f"  - {problem}")
        print(f"Download: {MODEL_BUNDLE_URL}")
        print("Extract its Sam folder into UltraDarkFusion. See MODEL_SETUP.md.")
    else:
        print("SAM3 and GroundingDINO model files verified.")

    print(f"Bundled translations verified: {len(TRANSLATION_CODES)} languages.")
    print("UltraDarkFusion launch requirements verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
