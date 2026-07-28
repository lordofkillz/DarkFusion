import argparse
import inspect
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.autobatch import check_train_batch_size


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SAFE_MAX_BATCH = 32
MIN_STEPS_PER_EPOCH = 4


def norm(path):
    return os.path.abspath(str(path)).replace("\\", "/") if path else ""


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def model_safe_batch_limit(model_path="", max_batch=SAFE_MAX_BATCH):
    name = Path(str(model_path or "")).stem.lower()
    limit = max(1, int(max_batch or SAFE_MAX_BATCH))
    if re.search(r"(^|[^a-z])x([^a-z]|$)|xlarge", name):
        return min(limit, 8)
    if re.search(r"(^|[^a-z])l([^a-z]|$)|large", name):
        return min(limit, 16)
    return limit


def safe_batch_limit(dataset_size=0, max_batch=SAFE_MAX_BATCH, model_path=""):
    try:
        limit = max(1, int(max_batch or SAFE_MAX_BATCH))
    except Exception:
        limit = SAFE_MAX_BATCH
    limit = model_safe_batch_limit(model_path, limit)

    try:
        dataset_size = int(dataset_size or 0)
    except Exception:
        dataset_size = 0

    if dataset_size > 0:
        if dataset_size >= MIN_STEPS_PER_EPOCH:
            dataset_limit = max(1, dataset_size // MIN_STEPS_PER_EPOCH)
        else:
            dataset_limit = dataset_size
        limit = min(limit, dataset_limit)
    return max(1, limit)


def clamp_batch(batch, dataset_size=0, model_path=""):
    try:
        batch = int(batch or 0)
    except Exception:
        batch = 0
    if batch <= 0:
        return 0, safe_batch_limit(dataset_size, model_path=model_path)

    limit = safe_batch_limit(dataset_size, model_path=model_path)
    return max(1, min(batch, limit)), limit


def dataset_root_from_yaml(data, yaml_dir):
    root = str((data or {}).get("path") or "").strip().strip('"').strip("'")
    if not root:
        return norm(yaml_dir)
    if os.path.isabs(root):
        return norm(root)
    return norm(os.path.join(yaml_dir, root))


def resolve_yaml_path(value, yaml_dir, dataset_root="", base_dir=""):
    if not value:
        return ""
    value = str(value).strip().strip('"').strip("'")
    if not value:
        return ""
    if os.path.isabs(value):
        return norm(value)
    for root in (base_dir, dataset_root, yaml_dir):
        root = norm(root)
        if not root:
            continue
        candidate = norm(os.path.join(root, value))
        if os.path.exists(candidate):
            return candidate
    return norm(os.path.join(dataset_root or yaml_dir, value))


def image_paths_from_train_entry(train_entry, yaml_dir, dataset_root=""):
    paths = []
    entries = train_entry if isinstance(train_entry, list) else [train_entry]
    for entry in entries:
        entry_path = resolve_yaml_path(entry, yaml_dir, dataset_root)
        if not entry_path:
            continue
        if os.path.isfile(entry_path) and entry_path.lower().endswith(".txt"):
            list_dir = os.path.dirname(entry_path)
            with open(entry_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        paths.append(resolve_yaml_path(text, yaml_dir, dataset_root, list_dir))
        elif os.path.isdir(entry_path):
            for root, _dirs, files in os.walk(entry_path):
                for name in files:
                    if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                        paths.append(norm(os.path.join(root, name)))
        elif os.path.isfile(entry_path) and os.path.splitext(entry_path)[1].lower() in IMAGE_EXTS:
            paths.append(entry_path)
    return list(dict.fromkeys(paths))


def label_candidates_for_image(image_path):
    path = Path(image_path)
    candidates = [path.with_suffix(".txt")]
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        idx = lowered.index("images")
        parts[idx] = "labels"
        candidates.append(Path(*parts).with_suffix(".txt"))
    return [norm(path) for path in candidates]


def dataset_object_stats(data_yaml):
    if not data_yaml or not os.path.exists(data_yaml):
        return 0, 1
    yaml_dir = os.path.dirname(norm(data_yaml))
    with open(data_yaml, "r", encoding="utf-8", errors="ignore") as f:
        data = yaml.safe_load(f) or {}
    dataset_root = dataset_root_from_yaml(data, yaml_dir)
    image_paths = image_paths_from_train_entry(data.get("train"), yaml_dir, dataset_root)
    max_labels = 0
    for image_path in image_paths:
        for label_path in label_candidates_for_image(image_path):
            if not os.path.exists(label_path):
                continue
            try:
                with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
                    count = sum(1 for line in f if line.strip())
                max_labels = max(max_labels, count)
                break
            except Exception:
                continue
    return len(image_paths), max(1, max_labels * 4)


def main():
    parser = argparse.ArgumentParser(description="DarkFusion Ultralytics batch calibration.")
    parser.add_argument("--model", required=True, help="Ultralytics model .pt or .yaml path.")
    parser.add_argument("--data", default="", help="Dataset YAML used to estimate max objects per batch.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--amp", default="true", help="Use AMP during calibration.")
    parser.add_argument("--fraction", type=float, default=0.60, help="CUDA memory fraction target, same behavior as Ultralytics batch=-1.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    args = parser.parse_args()

    result = {
        "generated_by": "DarkFusion Batch Calibrator",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": norm(args.model),
        "data": norm(args.data),
        "imgsz": int(args.imgsz),
        "amp": truthy(args.amp),
        "fraction": float(args.fraction),
        "dataset_size": 0,
        "max_num_obj": 1,
        "raw_batch": None,
        "batch_cap": None,
        "batch": None,
        "batch_note": "",
        "error": "",
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; auto-batch calibration requires a CUDA GPU.")

        dataset_size, max_num_obj = dataset_object_stats(args.data)
        result["dataset_size"] = int(dataset_size)
        result["max_num_obj"] = int(max_num_obj)

        yolo = YOLO(args.model)
        model = yolo.model.to("cuda")
        batch_kwargs = {
            "model": model,
            "imgsz": int(args.imgsz),
            "amp": truthy(args.amp),
            "batch": float(args.fraction),
            "max_num_obj": int(max_num_obj),
        }
        if "dataset_size" in inspect.signature(check_train_batch_size).parameters:
            batch_kwargs["dataset_size"] = int(dataset_size)
        raw_batch = int(check_train_batch_size(**batch_kwargs))
        batch, batch_cap = clamp_batch(raw_batch, dataset_size, args.model)
        result["raw_batch"] = raw_batch
        result["batch_cap"] = batch_cap
        result["batch"] = batch
        if raw_batch != batch:
            result["batch_note"] = (
                f"Clamped auto-batch from {raw_batch} to {batch} to avoid a full-dataset "
                f"or oversized training batch."
            )
    except Exception as e:
        result["error"] = str(e)
    finally:
        os.makedirs(os.path.dirname(norm(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
