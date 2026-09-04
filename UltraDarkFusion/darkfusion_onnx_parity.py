"""Compare the standalone backend with Ultralytics on the same ONNX graph.

Ultralytics is imported only by this development-time validator, never by the
runtime backend. Example:

  python darkfusion_onnx_parity.py --model best.onnx --source image.png --provider cuda
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np
import cv2

from darkfusion_onnx_runtime import DarkFusionOnnxModel


def _array(value):
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _maximum_error(left, right):
    left, right = _array(left), _array(right)
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _mask_iou(left, right):
    left, right = _array(left), _array(right)
    if left is None or right is None or left.shape != right.shape:
        return None
    values = []
    for a, b in zip(left > 0, right > 0):
        union = np.logical_or(a, b).sum()
        values.append(float(np.logical_and(a, b).sum() / max(1, union)))
    return float(statistics.mean(values)) if values else 1.0


def _result_count(result):
    for name in ("boxes", "obb"):
        value = getattr(result, name, None)
        if value is not None:
            return len(value)
    return 1 if getattr(result, "probs", None) is not None else 0


def main():
    parser = argparse.ArgumentParser(description="Validate DarkFusion ONNX result and speed parity")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--provider", default="cuda")
    parser.add_argument("--task")
    parser.add_argument("--output-format", default="auto")
    parser.add_argument("--cache-dir")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--reference-device", help="Ultralytics comparator device; defaults to cpu for CPU EP, otherwise 0")
    parser.add_argument("--preload", action="store_true", help="Decode the source once, matching DarkFusion video-frame usage")
    args = parser.parse_args()

    # A validator must never mutate the environment it is measuring. In
    # particular, Ultralytics otherwise attempts to replace onnxruntime-gpu
    # with the CPU-only onnxruntime package for a CPU comparison.
    os.environ["YOLO_AUTOINSTALL"] = "False"
    from ultralytics import YOLO  # Development comparator only.

    custom = DarkFusionOnnxModel(
        args.model,
        providers=args.provider,
        task=args.task,
        output_format=args.output_format,
        cache_dir=args.cache_dir,
    )
    reference = YOLO(args.model, task=args.task)
    reference_device = args.reference_device or ("cpu" if custom.provider == "CPUExecutionProvider" else "0")
    source = args.source
    if args.preload:
        source = cv2.imread(args.source, cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(args.source)
    for _ in range(args.warmup):
        custom.predict(source, conf=args.conf, iou=args.iou)
        reference.predict(
            source, conf=args.conf, iou=args.iou, device=reference_device, verbose=False
        )

    custom_times, reference_times = [], []
    custom_result = reference_result = None
    for _ in range(args.runs):
        started = time.perf_counter()
        custom_result = custom.predict(source, conf=args.conf, iou=args.iou)[0]
        custom_times.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        reference_result = reference.predict(
            source, conf=args.conf, iou=args.iou, device=reference_device, verbose=False
        )[0]
        reference_times.append((time.perf_counter() - started) * 1000.0)

    report = {
        "model": args.model,
        "task": custom.task,
        "provider": custom.provider,
        "reference_device": reference_device,
        "runs": args.runs,
        "source_preloaded": args.preload,
        "milliseconds_median": {
            "darkfusion_onnx": statistics.median(custom_times),
            "ultralytics_onnx": statistics.median(reference_times),
        },
        "milliseconds_minimum": {
            "darkfusion_onnx": min(custom_times),
            "ultralytics_onnx": min(reference_times),
        },
        "last_reported_stage_ms": {
            "darkfusion_onnx": custom_result.speed,
            "ultralytics_onnx": reference_result.speed,
        },
        "counts": {
            "darkfusion": _result_count(custom_result),
            "ultralytics": _result_count(reference_result),
        },
        "max_error": {
            "boxes": _maximum_error(
                getattr(getattr(custom_result, "boxes", None), "data", None),
                getattr(getattr(reference_result, "boxes", None), "data", None),
            ),
            "obb": _maximum_error(
                getattr(getattr(custom_result, "obb", None), "data", None),
                getattr(getattr(reference_result, "obb", None), "data", None),
            ),
            "keypoints": _maximum_error(
                getattr(getattr(custom_result, "keypoints", None), "data", None),
                getattr(getattr(reference_result, "keypoints", None), "data", None),
            ),
            "probabilities": _maximum_error(
                getattr(getattr(custom_result, "probs", None), "data", None),
                getattr(getattr(reference_result, "probs", None), "data", None),
            ),
        },
        "top1": {
            "darkfusion": getattr(getattr(custom_result, "probs", None), "top1", None),
            "ultralytics": getattr(getattr(reference_result, "probs", None), "top1", None),
        },
        "mask_iou_mean": _mask_iou(
            getattr(getattr(custom_result, "masks", None), "data", None),
            getattr(getattr(reference_result, "masks", None), "data", None),
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
