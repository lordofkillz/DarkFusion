#!/usr/bin/env python3
"""
Inspect skeleton metadata in YOLO model weights
"""
import os
import json
import sys
from pathlib import Path

# Keep the utility under tools while inspecting the application's model store.
SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(APP_DIR))

from ultralytics import YOLO

MODEL_DIR = APP_DIR / "weights"
MODELS_TO_CHECK = [
    "yolov8x-worldv2.pt",
    "yolov8n.pt",
    "yolo11x-pose.pt",
    "yolo11x-seg.pt",
    "yolo11x-obb.pt",
    "yolo26x.pt",
]

def check_skeleton_in_model(model_path):
    """Inspect a model for skeleton metadata"""
    if not os.path.exists(model_path):
        return {"exists": False, "path": model_path}

    print(f"\n{'='*70}")
    print(f"Checking: {os.path.basename(model_path)}")
    print(f"{'='*70}")

    result = {
        "path": model_path,
        "exists": True,
        "skeleton_found": False,
        "skeleton_source": None,
        "skeleton": None,
        "kpt_shape": None,
        "kpt_names": None,
        "model_task": None,
    }

    try:
        model = YOLO(model_path)
        result["model_task"] = getattr(model, "task", "unknown")

        # Check model.overrides
        if hasattr(model, "overrides"):
            skel = model.overrides.get("skeleton")
            if skel:
                result["skeleton"] = skel
                result["skeleton_source"] = "model.overrides"
                result["skeleton_found"] = True

        # Check model.model.args (the actual model object)
        model_obj = getattr(model, "model", None)
        if model_obj:
            args = getattr(model_obj, "args", None)
            if args and isinstance(args, dict):
                skel = args.get("skeleton")
                if skel and not result["skeleton"]:
                    result["skeleton"] = skel
                    result["skeleton_source"] = "model.model.args"
                    result["skeleton_found"] = True

        # Check ckpt
        if hasattr(model, "ckpt") and isinstance(model.ckpt, dict):
            skel = model.ckpt.get("skeleton")
            if skel and not result["skeleton"]:
                result["skeleton"] = skel
                result["skeleton_source"] = "model.ckpt"
                result["skeleton_found"] = True

        # Check direct attributes
        for attr in ("skeleton", "pose_skeleton", "kpt_skeleton", "kpt_names", "keypoint_names"):
            if hasattr(model, attr):
                value = getattr(model, attr)
                if value and not result["skeleton"] and "skeleton" in attr:
                    result["skeleton"] = value
                    result["skeleton_source"] = f"model.{attr}"
                    result["skeleton_found"] = True

                if attr in ("kpt_names", "keypoint_names") and value:
                    result["kpt_names"] = value

        # Check kpt_shape
        if hasattr(model, "kpt_shape"):
            result["kpt_shape"] = model.kpt_shape

        print(f"Task: {result['model_task']}")
        print(f"Keypoint names: {result['kpt_names']}")
        print(f"Keypoint shape: {result['kpt_shape']}")
        print(f"Skeleton found: {result['skeleton_found']}")
        if result['skeleton']:
            print(f"Skeleton source: {result['skeleton_source']}")
            print(f"Skeleton: {result['skeleton']}")
        else:
            print("❌ NO SKELETON METADATA FOUND")

    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Error loading model: {e}")

    return result

def main():
    print("\n" + "="*70)
    print("YOLO MODEL SKELETON METADATA INSPECTION")
    print("="*70)

    results = []
    for model_name in MODELS_TO_CHECK:
        model_path = MODEL_DIR / model_name
        result = check_skeleton_in_model(str(model_path))
        results.append(result)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    summary_file = SCRIPT_DIR / "skeleton_inspection_results.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    for result in results:
        status = "✅" if result.get("skeleton_found") else "❌"
        model_name = os.path.basename(result["path"])
        print(f"{status} {model_name:30} Task: {result.get('model_task', 'unknown'):12} Skeleton: {result.get('skeleton_found', False)}")

    print(f"\nResults saved to: {summary_file}")

if __name__ == "__main__":
    main()
