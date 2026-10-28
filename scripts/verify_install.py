"""Verify the dependencies and local files needed to launch UltraDarkFusion."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "UltraDarkFusion"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

REQUIRED_FILES = (
    "../VERSION",
    "darkfusion_version.py",
    "UltraDarkFusion.py",
    "UltraDarkFusion.ui",
    "ui_ultradarkfusion.py",
    "darkfusion_dino.py",
    "darkfusion_sahi.py",
    "prediction_size_filter.py",
    "splash_utils.py",
    "darkfusion_command_runner.py",
    "darkfusion_environment.py",
    "darkfusion_gpu_memory.py",
    "darkfusion_image_loading.py",
    "darkfusion_file_reveal.py",
    "darkfusion_ultralytics_batch.py",
    "darkfusion_ultralytics_cli.py",
    "darkfusion_ultralytics_train.py",
    "darkfusion_ultralytics_tune.py",
    "darkfusion_validation_review.py",
    "darkfusion_teammate_review.py",
    "darkfusion_theme_assets.py",
    "darkfusion_onnx_runtime.py",
    "darkfusion_inference.py",
    "darkfusion_tensorrt.py",
    "darkfusion_onnx_tracking.py",
    "darkfusion_windows_ml.py",
    "darkfusion_onnx_settings.py",
    "darkfusion_onnx_performance.py",
    "darkfusion_media_tools.py",
    "darkfusion_desktop_capture.py",
    "darkfusion_video_capture.py",
    "darkfusion_video_decode.py",
    "darkfusion_video_display.py",
    "darkfusion_video_preparation.py",
    "darkfusion_video_window.py",
    "darkfusion_dataset_operations.py",
    "darkfusion_propagation.py",
    "darkfusion_object_clarity.py",
    "darkfusion_clarity_augmentation.py",
    "licenses/Real-ESRGAN.txt",
    "media-tools.json",
    "darkfusion_negative_crops.py",
    "darkfusion_dataset_statistics.py",
    "darkfusion_class_semantics.py",
    "darkfusion_model_compat.py",
    "darkfusion_model_security.py",
    "darkfusion_borderline_llm_verifier.py",
    "darkfusion_ollama_auto_setup.py",
    "darkfusion_ollama_memory.py",
    "darkfusion_review_feedback.py",
    "darkfusion_shape_verifier.py",
    "darkfusion_review_similarity.py",
    "darkfusion_review_settings.py",
    "darkfusion_translation.py",
    "darkfusion_training_size.py",
    "darkfusion_tune_monitor.py",
    "darkfusion_tune_settings.py",
    "darkfusion_training_runs.py",
    "darkfusion_extraction.py",
    "darkfusion_verified_images.py",
    "darkfusion_dinov3/config.py",
    "darkfusion_dinov3/loader.py",
    "darkfusion_dinov3/download.py",
    "darkfusion_dinov3/vision_transformer.py",
    "darkfusion_dinov3/LICENSE.md",
    "darkfusion_system_metrics.py",
    "training_eta.py",
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
    "clip",
    "diffusers",
    "dotenv",
    "groundingdino",
    "matplotlib",
    "mss",
    "numpy",
    "onnx",
    "onnxruntime",
    "pandas",
    "plotly",
    "psutil",
    "pynvml",
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
    "yt_dlp_ejs",
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


def verify_onnx_backend() -> list[str]:
    """Reject missing/overlapping ORT variants in the standard Windows runtime."""
    if sys.platform != "win32":
        return []
    failures: list[str] = []
    try:
        from darkfusion_windows_ml import get_ort
        onnxruntime = get_ort()
        version = importlib.metadata.version("onnxruntime-windowsml")
        for variant in ("onnxruntime", "onnxruntime-gpu", "onnxruntime-openvino", "onnxruntime-directml"):
            try:
                importlib.metadata.version(variant)
            except importlib.metadata.PackageNotFoundError:
                continue
            failures.append(
                f"{variant} overlaps onnxruntime-windowsml; use only the Windows ML variant."
            )
        if onnxruntime.__version__ != ".".join(version.split(".")[:3]):
            failures.append("ONNX Runtime module and Windows ML package versions do not match.")
        providers = onnxruntime.get_available_providers()
        if "DmlExecutionProvider" not in providers:
            failures.append("DirectML execution provider is missing from ONNX Runtime.")
        if not failures:
            print(f"Windows ML: {version}; DirectML available; providers: {', '.join(providers)}")
    except Exception as exc:
        failures.append(f"Windows ONNX backend requires onnxruntime-windowsml and its Windows App SDK runtime: {exc}")
    return failures


def verify_tensorrt_backend() -> list[str]:
    """TensorRT CUDA families share Python files and must not be stacked."""
    import re
    failures = []
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata.get("Name", "")).lower()
        if name == "tensorrt" or re.fullmatch(r"tensorrt-cu(?:12|13)(?:-bindings|-libs)?", name):
            packages[name] = distribution.version
    families = [family for family in ("tensorrt-cu12", "tensorrt-cu13")
                if any(name == family or name.startswith(family + "-") for name in packages)]
    if len(families) != 1:
        return ["TensorRT requires exactly one matching CUDA family; run install.ps1 to repair missing/overlapping packages."]
    family = families[0]
    version = packages.get(family)
    if not version or any(packages.get(name) != version for name in (family + "-bindings", family + "-libs")):
        failures.append("TensorRT package, bindings and native library versions do not match.")
    if "tensorrt" in packages and packages["tensorrt"] != version:
        failures.append("The unsuffixed TensorRT metapackage disagrees with the selected CUDA family.")
    try:
        import tensorrt
        if tensorrt.__version__ != version:
            failures.append("Imported TensorRT and installed package versions do not match.")
    except Exception as exc:
        failures.append(f"TensorRT import failed: {exc}")
    return failures


def verify_qt_material_backend() -> list[str]:
    """Importability alone does not prove the theme package supports PyQt5."""
    try:
        widgets = importlib.import_module("PyQt5.QtWidgets")
        material = importlib.import_module("qt_material")
        if not getattr(material, "GUI", False) or getattr(material, "QColorDialog", None) is not widgets.QColorDialog:
            return ["qt-material does not support the application's PyQt5 binding; reinstall the compatible version pinned in requirements.txt."]
    except Exception as exc:
        return [f"PyQt5/qt-material compatibility check failed: {exc}"]
    return []


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        print(f"ERROR: Python 3.12 is required; found {sys.version.split()[0]}.")
        return 1

    missing_files = [name for name in REQUIRED_FILES if not (APP_ROOT / name).is_file()]
    import_failures: list[str] = verify_onnx_backend() + verify_tensorrt_backend() + verify_qt_material_backend()
    if sys.platform == "win32":
        try:
            from darkfusion_media_tools import verify_bundled_media_tools
            media_failures = verify_bundled_media_tools(APP_ROOT)
            import_failures.extend(media_failures)
            if not media_failures:
                print("Bundled FFmpeg/FFprobe/Deno verified; system PATH is not required.")
        except Exception as exc:
            import_failures.append(f"Private media tools: {exc}")
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # report every missing/broken binary binding
            import_failures.append(f"{module_name}: {exc}")

    try:
        from ultralytics import YOLOE  # noqa: F401
    except Exception as exc:
        import_failures.append(f"ultralytics.YOLOE: {exc}")


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
        # Languages acquire new source keys independently in the background.
        # Missing translations intentionally fall back to the English UI.
        if not isinstance(catalog, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in catalog.items()
        ):
            translation_failures.append(f"{code}.json: expected a string-to-string catalog")

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

    from darkfusion_version import APP_VERSION
    print(f"DarkFusion: {APP_VERSION}")
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
