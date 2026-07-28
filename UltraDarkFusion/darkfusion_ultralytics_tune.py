import argparse
import importlib.util
import os
import sys

import torch
import yaml
from ultralytics import YOLO


def parse_value(value):
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if "," in text and not os.path.exists(text):
        return [parse_value(part) for part in text.split(",")]
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_key_value_args(items):
    parsed = {}
    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed[key] = parse_value(value)
    return parsed


def tune_output_dir(kwargs):
    project = kwargs.get("project")
    name = kwargs.get("name") or "tune"
    if not project:
        return ""
    return os.path.abspath(os.path.join(str(project), str(name)))


def normalize_gpu_per_trial(value, use_ray):
    if not use_ray:
        return None

    text = "auto" if value is None else str(value).strip().lower()
    if text in {"", "auto", "default"}:
        return 1 if torch.cuda.is_available() else 0
    if text in {"none", "cpu", "false", "off"}:
        return 0
    try:
        return max(0, int(float(text)))
    except ValueError:
        raise ValueError(f"Invalid --gpu-per-trial value: {value}")


def save_ray_best_hyperparameters(result_grid, model, output_dir):
    if result_grid is None or not output_dir:
        return ""

    try:
        from ultralytics.utils.tuner import TASK2METRIC

        metric = TASK2METRIC.get(getattr(model, "task", None)) or "metrics/mAP50-95(B)"
        best = result_grid.get_best_result(metric=metric, mode="max")
        config = dict(getattr(best, "config", {}) or {})
    except Exception:
        try:
            best = result_grid.get_best_result()
            config = dict(getattr(best, "config", {}) or {})
        except Exception as e:
            print(f"Could not get Ray Tune best result: {e}")
            return ""

    blocked = {
        "data",
        "model",
        "imgsz",
        "epochs",
        "batch",
        "project",
        "name",
        "amp",
        "rect",
        "resume",
        "freeze",
        "patience",
        "pretrained",
        "distill_model",
        "dis",
        "device",
        "gpu_per_trial",
        "workers",
        "cache",
        "plots",
        "save",
        "val",
        "exist_ok",
    }
    best_hyp = {
        str(key): value
        for key, value in config.items()
        if str(key).strip().lower() not in blocked
    }
    if not best_hyp:
        return ""

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "best_hyperparameters.yaml")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_hyp, f, sort_keys=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="DarkFusion Ultralytics hyperparameter tuning launcher.")
    parser.add_argument("--model", required=True, help="Ultralytics model .pt or .yaml path.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of tuning trials.")
    parser.add_argument("--use-ray", action="store_true", help="Require Ray Tune.")
    parser.add_argument("--ray-if-available", action="store_true", help="Use Ray Tune only when ray[tune] is installed.")
    parser.add_argument(
        "--gpu-per-trial",
        default="auto",
        help="Ray Tune GPUs allocated per trial. Use auto, 0, or 1. Auto uses 1 when CUDA is available.",
    )
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Ultralytics train args as key=value pairs.")
    args = parser.parse_args()

    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    kwargs = parse_key_value_args(train_args)
    use_ray = bool(args.use_ray)
    if args.ray_if_available:
        use_ray = importlib.util.find_spec("ray") is not None

    if args.use_ray and importlib.util.find_spec("ray") is None:
        raise ModuleNotFoundError('Ray Tune requested but not installed. Install with: pip install "ray[tune]"')

    gpu_per_trial = normalize_gpu_per_trial(args.gpu_per_trial, use_ray)
    if use_ray and "device" not in {str(key).lower() for key in kwargs}:
        kwargs["device"] = 0 if gpu_per_trial else "cpu"

    print(f"DarkFusion tuning model: {args.model}")
    print(f"Iterations: {args.iterations}")
    print(f"Use Ray Tune: {use_ray}")
    if use_ray:
        print(f"Ray GPU per trial: {gpu_per_trial}")
    print("Train args:")
    for key, value in kwargs.items():
        print(f"  {key}={value}")

    model = YOLO(args.model)
    tune_kwargs = {"use_ray": use_ray, "iterations": max(1, int(args.iterations)), **kwargs}
    if use_ray:
        tune_kwargs["gpu_per_trial"] = gpu_per_trial
    result = model.tune(**tune_kwargs)

    print("\nTune finished.")
    if use_ray:
        best_path = save_ray_best_hyperparameters(result, model, tune_output_dir(kwargs))
        if best_path:
            print(f"Best hyperparameters saved: {best_path}")
    if result is not None:
        print(result)


if __name__ == "__main__":
    sys.exit(main())
