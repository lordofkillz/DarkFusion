"""Generate a task-aware, per-image validation error report for DarkFusion."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

try:
    from shapely.geometry import Polygon
except Exception:
    Polygon = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def normalized(path):
    return os.path.abspath(os.path.expanduser(str(path or "")))


def yaml_names(data):
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return [str(item) for item in names or []]


def resolve_yaml_entry(entry, yaml_dir, root):
    entry = str(entry or "").strip()
    if not entry:
        return ""
    if os.path.isabs(entry):
        return normalized(entry)
    for base in (root, yaml_dir):
        candidate = normalized(os.path.join(base, entry))
        if os.path.exists(candidate):
            return candidate
    return normalized(os.path.join(root, entry))


def image_files_from_entry(entry, yaml_dir, root):
    files = []
    for value in entry if isinstance(entry, list) else [entry]:
        path = resolve_yaml_entry(value, yaml_dir, root)
        if os.path.isfile(path) and path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    item = line.strip()
                    if not item:
                        continue
                    candidate = item if os.path.isabs(item) else os.path.join(os.path.dirname(path), item)
                    candidate = normalized(candidate)
                    if os.path.isfile(candidate):
                        files.append(candidate)
        elif os.path.isfile(path) and Path(path).suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
        elif os.path.isdir(path):
            for current, _dirs, names in os.walk(path):
                for name in names:
                    candidate = os.path.join(current, name)
                    if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                        files.append(normalized(candidate))
    return sorted(dict.fromkeys(files))


def load_dataset(data_path, split):
    data_path = normalized(data_path)
    with open(data_path, "r", encoding="utf-8", errors="replace") as handle:
        data = yaml.safe_load(handle) or {}
    yaml_dir = os.path.dirname(data_path)
    root_value = str(data.get("path", "") or "").strip()
    root = resolve_yaml_entry(root_value, yaml_dir, yaml_dir) if root_value else yaml_dir
    entry = data.get(split)
    if entry is None and split == "val":
        entry = data.get("validation")
    if entry is None:
        raise ValueError(f"Dataset YAML has no '{split}' entry.")
    return image_files_from_entry(entry, yaml_dir, root), yaml_names(data), data


def label_path_for_image(image_path):
    path = Path(image_path)
    candidates = [path.with_suffix(".txt")]
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        replaced = list(parts)
        replaced[index] = "labels"
        candidates.insert(0, Path(*replaced).with_suffix(".txt"))
    for candidate in candidates:
        if candidate.exists():
            return normalized(candidate)
    return normalized(candidates[0])


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def bbox_from_points(points):
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [clamp01(min(xs)), clamp01(min(ys)), clamp01(max(xs)), clamp01(max(ys))]


def xywh_to_xyxy(values):
    cx, cy, width, height = [float(value) for value in values[:4]]
    return [
        clamp01(cx - width / 2.0),
        clamp01(cy - height / 2.0),
        clamp01(cx + width / 2.0),
        clamp01(cy + height / 2.0),
    ]


def bbox_iou(first, second):
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 1e-12 else 0.0


def polygon_iou(first, second):
    if Polygon is None or len(first or []) < 3 or len(second or []) < 3:
        return bbox_iou(bbox_from_points(first), bbox_from_points(second))
    try:
        poly_a = Polygon(first).buffer(0)
        poly_b = Polygon(second).buffer(0)
        if poly_a.is_empty or poly_b.is_empty:
            return 0.0
        union = poly_a.union(poly_b).area
        return float(poly_a.intersection(poly_b).area / union) if union > 1e-12 else 0.0
    except Exception:
        return bbox_iou(bbox_from_points(first), bbox_from_points(second))


def object_iou(first, second, task):
    if task in {"segment", "obb"} and first.get("points") and second.get("points"):
        return polygon_iou(first["points"], second["points"])
    return bbox_iou(first.get("bbox", [0, 0, 0, 0]), second.get("bbox", [0, 0, 0, 0]))


def parse_ground_truth(image_path, task, class_names):
    if task == "classify":
        folder_name = Path(image_path).parent.name
        try:
            class_id = class_names.index(folder_name)
        except ValueError:
            class_id = -1
        return [{"class_id": class_id, "class_name": folder_name}]

    label_path = label_path_for_image(image_path)
    objects = []
    if not os.path.isfile(label_path):
        return objects
    with open(label_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line_index, line in enumerate(handle):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                class_id = int(float(parts[0]))
                values = [float(value) for value in parts[1:]]
            except Exception:
                continue
            item = {
                "class_id": class_id,
                "class_name": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
                "label_line": line_index,
            }
            if task in {"detect", "pose"} and len(values) >= 4:
                item["bbox"] = xywh_to_xyxy(values[:4])
                if task == "pose" and len(values) > 4:
                    stride = 3 if (len(values) - 4) % 3 == 0 else 2
                    item["keypoints"] = [
                        [clamp01(values[index]), clamp01(values[index + 1]), values[index + 2] if stride == 3 else 1.0]
                        for index in range(4, len(values) - stride + 1, stride)
                    ]
            elif task == "segment" and len(values) >= 6 and len(values) % 2 == 0:
                item["points"] = [[clamp01(values[index]), clamp01(values[index + 1])] for index in range(0, len(values), 2)]
                item["bbox"] = bbox_from_points(item["points"])
            elif task == "obb" and len(values) >= 8:
                item["points"] = [[clamp01(values[index]), clamp01(values[index + 1])] for index in range(0, 8, 2)]
                item["bbox"] = bbox_from_points(item["points"])
            else:
                continue
            objects.append(item)
    return objects


def tensor_list(value):
    if value is None:
        return []
    try:
        return value.detach().cpu().tolist()
    except Exception:
        try:
            return np.asarray(value).tolist()
        except Exception:
            return []


def predictions_from_result(result, task, class_names):
    height, width = result.orig_shape
    width = max(1, int(width))
    height = max(1, int(height))
    predictions = []

    if task == "classify":
        probs = getattr(result, "probs", None)
        if probs is None:
            return predictions
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        return [{
            "class_id": class_id,
            "class_name": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
            "confidence": confidence,
        }]

    if task == "obb" and getattr(result, "obb", None) is not None:
        obb = result.obb
        classes = tensor_list(obb.cls)
        confidences = tensor_list(obb.conf)
        polygons = tensor_list(obb.xyxyxyxy)
        for index, points in enumerate(polygons):
            normalized_points = [[clamp01(point[0] / width), clamp01(point[1] / height)] for point in points]
            class_id = int(classes[index])
            predictions.append({
                "class_id": class_id,
                "class_name": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
                "confidence": float(confidences[index]),
                "points": normalized_points,
                "bbox": bbox_from_points(normalized_points),
            })
        return predictions

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return predictions
    classes = tensor_list(boxes.cls)
    confidences = tensor_list(boxes.conf)
    xyxy = tensor_list(boxes.xyxy)
    mask_polygons = list(getattr(getattr(result, "masks", None), "xyn", []) or [])
    keypoint_xy = tensor_list(getattr(getattr(result, "keypoints", None), "xyn", None))
    keypoint_conf = tensor_list(getattr(getattr(result, "keypoints", None), "conf", None))
    for index, bounds in enumerate(xyxy):
        class_id = int(classes[index])
        item = {
            "class_id": class_id,
            "class_name": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
            "confidence": float(confidences[index]),
            "bbox": [
                clamp01(bounds[0] / width),
                clamp01(bounds[1] / height),
                clamp01(bounds[2] / width),
                clamp01(bounds[3] / height),
            ],
        }
        if task == "segment" and index < len(mask_polygons):
            item["points"] = [[clamp01(point[0]), clamp01(point[1])] for point in np.asarray(mask_polygons[index]).tolist()]
        if task == "pose" and index < len(keypoint_xy):
            confidences_for_points = keypoint_conf[index] if index < len(keypoint_conf) else []
            item["keypoints"] = [
                [
                    clamp01(point[0]),
                    clamp01(point[1]),
                    float(confidences_for_points[point_index]) if point_index < len(confidences_for_points) else 1.0,
                ]
                for point_index, point in enumerate(keypoint_xy[index])
            ]
        predictions.append(item)
    return predictions


def pose_quality(ground_truth, prediction):
    gt_points = ground_truth.get("keypoints", [])
    pred_points = prediction.get("keypoints", [])
    if not gt_points or not pred_points:
        return None
    visible = [
        index for index, point in enumerate(gt_points)
        if index < len(pred_points) and (len(point) < 3 or float(point[2]) > 0)
    ]
    if not visible:
        return None
    bbox = ground_truth.get("bbox", [0, 0, 1, 1])
    scale = max(1e-6, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    distances = [
        math.hypot(gt_points[index][0] - pred_points[index][0], gt_points[index][1] - pred_points[index][1]) / scale
        for index in visible
    ]
    return float(sum(distances) / len(distances))


def make_issue(image_path, label_path, task, issue_type, gt=None, pred=None, overlap=None, detail=""):
    confidence = float((pred or {}).get("confidence", 0.0) or 0.0)
    severity = confidence
    if issue_type == "false_negative":
        severity = 1.0
    elif issue_type in {"wrong_class", "weak_localization", "poor_keypoints"}:
        severity = max(confidence, 1.0 - float(overlap or 0.0))
    reference = pred or gt or {}
    return {
        "id": uuid.uuid4().hex,
        "image_path": normalized(image_path),
        "label_path": normalized(label_path),
        "task": task,
        "type": issue_type,
        "severity": round(float(severity), 6),
        "class_id": int(reference.get("class_id", -1)),
        "class_name": str(reference.get("class_name", "")),
        "confidence": round(confidence, 6) if pred else None,
        "iou": round(float(overlap), 6) if overlap is not None else None,
        "ground_truth": gt,
        "prediction": pred,
        "detail": str(detail or ""),
        "review_status": "unreviewed",
    }


def compare_image(image_path, task, class_names, ground_truth, predictions, match_iou, good_iou):
    label_path = label_path_for_image(image_path)
    if task == "classify":
        gt = ground_truth[0] if ground_truth else None
        pred = predictions[0] if predictions else None
        if gt and pred and gt["class_id"] == pred["class_id"]:
            return []
        return [make_issue(
            image_path, label_path, task, "wrong_class", gt=gt, pred=pred,
            detail=f"Expected {gt.get('class_name') if gt else 'unknown'}, predicted {pred.get('class_name') if pred else 'none'}.",
        )]

    unmatched_gt = set(range(len(ground_truth)))
    unmatched_pred = set(range(len(predictions)))
    matches = []
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            if gt.get("class_id") == pred.get("class_id"):
                candidates.append((object_iou(gt, pred, task), gt_index, pred_index))
    for overlap, gt_index, pred_index in sorted(candidates, reverse=True):
        if overlap < match_iou or gt_index not in unmatched_gt or pred_index not in unmatched_pred:
            continue
        unmatched_gt.remove(gt_index)
        unmatched_pred.remove(pred_index)
        matches.append((gt_index, pred_index, overlap))

    issues = []
    # Pair spatially matching, differently classified leftovers as one class error.
    wrong_class_candidates = []
    for gt_index in unmatched_gt:
        for pred_index in unmatched_pred:
            overlap = object_iou(ground_truth[gt_index], predictions[pred_index], task)
            if overlap >= match_iou:
                wrong_class_candidates.append((overlap, gt_index, pred_index))
    for overlap, gt_index, pred_index in sorted(wrong_class_candidates, reverse=True):
        if gt_index not in unmatched_gt or pred_index not in unmatched_pred:
            continue
        unmatched_gt.remove(gt_index)
        unmatched_pred.remove(pred_index)
        issues.append(make_issue(
            image_path, label_path, task, "wrong_class",
            gt=ground_truth[gt_index], pred=predictions[pred_index], overlap=overlap,
            detail=f"Ground truth is {ground_truth[gt_index].get('class_name')}; prediction is {predictions[pred_index].get('class_name')}.",
        ))

    for gt_index, pred_index, overlap in matches:
        gt = ground_truth[gt_index]
        pred = predictions[pred_index]
        if overlap < good_iou:
            issues.append(make_issue(
                image_path, label_path, task, "weak_localization",
                gt=gt, pred=pred, overlap=overlap,
                detail=f"Matched {task} annotation has overlap {overlap:.3f}.",
            ))
        if task == "pose":
            keypoint_error = pose_quality(gt, pred)
            if keypoint_error is not None and keypoint_error > 0.1:
                issues.append(make_issue(
                    image_path, label_path, task, "poor_keypoints",
                    gt=gt, pred=pred, overlap=max(0.0, 1.0 - keypoint_error),
                    detail=f"Mean normalized keypoint error is {keypoint_error:.3f}.",
                ))

    for gt_index in sorted(unmatched_gt):
        issues.append(make_issue(
            image_path, label_path, task, "false_negative", gt=ground_truth[gt_index],
            detail="Ground-truth object was not matched by a prediction.",
        ))
    for pred_index in sorted(unmatched_pred):
        pred = predictions[pred_index]
        duplicate_matches = [
            (object_iou(gt, pred, task), gt)
            for gt in ground_truth
            if gt.get("class_id") == pred.get("class_id")
            and object_iou(gt, pred, task) >= match_iou
        ]
        duplicate = bool(duplicate_matches)
        duplicate_overlap, duplicate_gt = max(duplicate_matches, default=(None, None), key=lambda item: item[0])
        issue_type = "duplicate_prediction" if duplicate else "false_positive"
        issues.append(make_issue(
            image_path,
            label_path,
            task,
            issue_type,
            gt=duplicate_gt,
            pred=pred,
            overlap=duplicate_overlap,
            detail="Prediction duplicates an already matched object." if duplicate else "Prediction was not matched to ground truth.",
        ))
    return issues


def write_report(path, report):
    temporary = f"{path}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", default="detect", choices=["detect", "segment", "obb", "pose", "classify"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--good-iou", type=float, default=0.75)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    images, class_names, _data = load_dataset(args.data, args.split)
    dataset_total_images = len(images)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise ValueError("No validation images were found.")

    started = time.time()
    report = {
        "version": 1,
        "status": "running",
        "task": args.task,
        "model": normalized(args.model),
        "data": normalized(args.data),
        "split": args.split,
        "class_names": class_names,
        "settings": {
            "conf": args.conf,
            "match_iou": args.match_iou,
            "good_iou": args.good_iou,
            "imgsz": args.imgsz,
            "device": args.device,
        },
        "processed_images": 0,
        "total_images": len(images),
        "dataset_total_images": dataset_total_images,
        "scan_limit": max(0, int(args.max_images)),
        "stage": "loading_model",
        "summary": {},
        "issues": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_report(normalized(args.output), report)
    print(f"Loading {os.path.basename(args.model)} for {len(images)} images", flush=True)

    model = YOLO(normalized(args.model), task=args.task)
    report["stage"] = "reviewing_images"
    write_report(normalized(args.output), report)
    counts = {}
    processed = 0
    chunk_size = max(1, min(2048, int(args.chunk_size or 256)))
    for chunk_start in range(0, len(images), chunk_size):
        chunk = images[chunk_start:chunk_start + chunk_size]
        source = chunk[0] if len(chunk) == 1 else chunk
        stream = model.predict(
            source=source,
            stream=True,
            conf=max(0.001, min(1.0, args.conf)),
            imgsz=max(32, int(args.imgsz)),
            device=args.device,
            save=False,
            verbose=False,
        )
        for result_index, result in enumerate(stream):
            processed += 1
            # Ultralytics converts a list of paths into PIL images internally and
            # may expose synthetic result names such as ``image83.jpg``.  The
            # prediction stream preserves source order, so retain the matching
            # real dataset path instead of writing an unusable temporary name to
            # the validation review report.
            image_path = normalized(chunk[result_index]) if result_index < len(chunk) else normalized(result.path)
            ground_truth = parse_ground_truth(image_path, args.task, class_names)
            predictions = predictions_from_result(result, args.task, class_names)
            issues = compare_image(
                image_path,
                args.task,
                class_names,
                ground_truth,
                predictions,
                max(0.0, min(1.0, args.match_iou)),
                max(0.0, min(1.0, args.good_iou)),
            )
            report["issues"].extend(issues)
            for issue in issues:
                counts[issue["type"]] = counts.get(issue["type"], 0) + 1
            report["processed_images"] = processed
            report["summary"] = dict(sorted(counts.items()))
            if processed % 100 == 0:
                write_report(normalized(args.output), report)
                print(f"Reviewed {processed}/{len(images)} images; {len(report['issues'])} issues", flush=True)

    report["status"] = "complete"
    report["stage"] = "complete"
    report["issue_image_count"] = len({
        issue.get("image_path", "")
        for issue in report["issues"]
        if issue.get("image_path")
    })
    report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["elapsed_seconds"] = round(time.time() - started, 3)
    report["issues"].sort(key=lambda issue: (-float(issue.get("severity", 0.0)), issue.get("image_path", "")))
    write_report(normalized(args.output), report)
    print(f"Validation review complete: {len(report['issues'])} issues across {len(images)} images", flush=True)


if __name__ == "__main__":
    main()
