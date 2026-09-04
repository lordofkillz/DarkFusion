"""Find labeled video-game teammates by the name/marker above their head.

The scanner is deliberately read-only.  It writes Validation Review compatible
issues; DarkFusion performs any selected label removal and records a recovery
log beside the dataset metadata.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from darkfusion_validation_review import (
    apply_review_decisions,
    load_dataset,
    load_review_decisions,
    make_issue,
    normalized,
    parse_ground_truth,
    validation_issue_key,
    write_report,
)


NAME_TAG_PROMPTS = (
    "a readable gamer username nameplate directly above a video game player",
    "a teammate gamer tag made of letters next to a small squad icon",
    "a player name label identifying a friendly squad member",
)
NO_NAME_TAG_PROMPTS = (
    "a video game enemy without a username marker above their head",
    "a video game character with no name tag above them",
    "a player with floating damage numbers or score numbers above them",
    "ordinary video game HUD text and numbers overlapping a character",
)
COLORED_SYMBOL_PROMPTS = (
    "a small colored circle beside a teammate gamer name",
    "a colored diamond or squad symbol immediately next to a player nameplate",
    "a compact colored friendly-player icon attached to a gamer tag",
)
NO_COLORED_SYMBOL_PROMPTS = (
    "floating text or numbers with no colored teammate icon beside them",
    "a player behind a crosshair, weapon sight, reticle, or hit marker",
    "combat effects, health numbers, or damage indicators over a player",
    "a video game character partly covered by interface graphics or an overlay",
)


class TeammateMarkerClassifier:
    """Reusable, lazily loaded CLIP classifier for auto-label post-filtering."""

    def __init__(self, device="cuda"):
        requested = str(device or "cuda").lower()
        self.device = torch.device(
            "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"
        )
        model_name = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(
            model_name, local_files_only=True, use_fast=True
        )
        self.model = CLIPModel.from_pretrained(
            model_name, local_files_only=True
        ).to(self.device).eval()
        self.text_features = averaged_text_features(
            self.model, self.processor, self.device
        )

    def score(self, crops, batch_size=96):
        if not crops:
            return []
        return score_crops(
            self.model,
            self.processor,
            self.text_features,
            self.device,
            list(crops),
            max(1, int(batch_size or 96)),
        )


def marker_crop(image, ground_truth):
    """Crop the person's upper portion plus the nameplate band above the box."""
    bounds = list(ground_truth.get("bbox", []) or [])
    if len(bounds) < 4:
        return None
    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in bounds[:4]]
    box_width = max(1.0, (x2 - x1) * width)
    box_height = max(1.0, (y2 - y1) * height)
    left = max(0, round(x1 * width - box_width * 0.45))
    right = min(width, round(x2 * width + box_width * 0.45))
    top = max(0, round(y1 * height - box_height * 0.45))
    bottom = min(height, round(y1 * height + box_height * 0.55))
    if right - left < 8 or bottom - top < 8:
        return None
    return image.crop((left, top, right, bottom))


def has_paired_marker_layout(crop):
    """Require a compact colored icon beside a horizontal row of text-like strokes."""
    if crop is None:
        return False
    rgb = np.asarray(crop.convert("RGB"))
    if rgb.ndim != 3 or rgb.shape[0] < 8 or rgb.shape[1] < 12:
        return False

    # The top of the annotated player box is about 45% down marker_crop.
    # Keep a small allowance below it, but exclude the body, weapon, and most
    # center-screen sights from the structural test.
    band = rgb[:max(8, int(round(rgb.shape[0] * 0.58)))]
    band_h, band_w = band.shape[:2]
    scale = max(1.0, min(4.0, 320.0 / max(1, band_w)))
    if scale > 1.05:
        band = cv2.resize(
            band,
            (int(round(band_w * scale)), int(round(band_h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        band_h, band_w = band.shape[:2]

    hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
    colored = cv2.inRange(hsv, np.array((0, 65, 70)), np.array((179, 255, 255)))
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    icon_boxes = []
    band_area = float(band_h * band_w)
    contours, _ = cv2.findContours(colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        box_area = float(max(1, width * height))
        aspect = width / float(max(1, height))
        if not (max(3.0, band_area * 0.00008) <= area <= band_area * 0.035):
            continue
        if width > band_w * 0.22 or height > band_h * 0.48:
            continue
        if not (0.22 <= aspect <= 4.5) or area / box_area < 0.16:
            continue
        icon_boxes.append((x, y, width, height))
    if not icon_boxes:
        return False

    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    contrast = cv2.absdiff(gray, blurred)
    # Gamer tags are often only a few source pixels tall. Preserve their thin
    # strokes instead of opening them away like general scene texture.
    cutoff = max(8.0, float(np.percentile(contrast, 65.0)))
    strokes = np.where(contrast >= cutoff, 255, 0).astype(np.uint8)
    stroke_contours, _ = cv2.findContours(strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    character_boxes = []
    for contour in stroke_contours:
        x, y, width, height = cv2.boundingRect(contour)
        aspect = width / float(max(1, height))
        if not (band_h * 0.045 <= height <= band_h * 0.34):
            continue
        if width > band_w * 0.16 or not (0.06 <= aspect <= 1.8):
            continue
        character_patch = hsv[y:y + height, x:x + width]
        text_colored_pixels = (
            (character_patch[:, :, 2] >= 85)
            & (character_patch[:, :, 1] <= 145)
        )
        if float(text_colored_pixels.mean()) < 0.10:
            continue
        character_boxes.append((x, y, width, height))

    for icon_x, icon_y, icon_w, icon_h in icon_boxes:
        icon_left = icon_x
        icon_right = icon_x + icon_w
        icon_center_y = icon_y + icon_h / 2.0
        # Reject colored details on the player/body at the bottom of the crop.
        # A real overhead marker should sit above, or nearly on, the bbox top.
        if icon_center_y > band_h * 0.86:
            continue
        for side in ("left", "right"):
            row = []
            for x, y, width, height in character_boxes:
                center_y = y + height / 2.0
                if abs(center_y - icon_center_y) > band_h * 0.19:
                    continue
                if side == "left":
                    gap = icon_left - (x + width)
                else:
                    gap = x - icon_right
                if 0 <= gap <= band_w * 0.38:
                    row.append((x, y, width, height))
            if len(row) < 3:
                continue
            row.sort(key=lambda box: box[0])
            span = row[-1][0] + row[-1][2] - row[0][0]
            centers_y = [box[1] + box[3] / 2.0 for box in row]
            if span < band_w * 0.055 or span > band_w * 0.72:
                continue
            if max(centers_y) - min(centers_y) > band_h * 0.24:
                continue
            return True
    return False


def averaged_text_features(model, processor, device):
    groups = (
        NAME_TAG_PROMPTS,
        NO_NAME_TAG_PROMPTS,
        COLORED_SYMBOL_PROMPTS,
        NO_COLORED_SYMBOL_PROMPTS,
    )
    prompts = [prompt for group in groups for prompt in group]
    encoded = processor(text=prompts, return_tensors="pt", padding=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        features = model.get_text_features(**encoded)
        features = features / features.norm(dim=-1, keepdim=True)
        classes = []
        offset = 0
        for group in groups:
            classes.append(features[offset:offset + len(group)].mean(dim=0))
            offset += len(group)
        classes = torch.stack(classes)
        return classes / classes.norm(dim=-1, keepdim=True)


def score_crops(model, processor, text_features, device, crops, batch_size, return_details=False):
    scores = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size]
        pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = model.get_image_features(pixel_values=pixels)
            else:
                features = model.get_image_features(pixel_values=pixels)
            features = features.float()
            features = features / features.norm(dim=-1, keepdim=True)
            similarities = 100.0 * features @ text_features.T
            name_probabilities = torch.softmax(similarities[:, 0:2], dim=1)[:, 0]
            symbol_probabilities = torch.softmax(similarities[:, 2:4], dim=1)[:, 0]
            # Both pieces of evidence are mandatory. A high name score cannot
            # compensate for a missing colored symbol, or vice versa.
            combined = torch.minimum(name_probabilities, symbol_probabilities)
            layout = torch.tensor(
                [has_paired_marker_layout(crop) for crop in batch],
                dtype=combined.dtype,
                device=combined.device,
            )
            combined = combined * layout
        if return_details:
            scores.extend(
                (float(score), float(name), float(symbol))
                for score, name, symbol in zip(
                    combined.cpu(), name_probabilities.cpu(), symbol_probabilities.cpu()
                )
            )
        else:
            scores.extend(float(value) for value in combined.cpu())
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", default="detect", choices=["detect", "segment", "obb", "pose"])
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--review-history", default="")
    parser.add_argument("--batch-state", default="")
    parser.add_argument("--image-batch", type=int, default=256)
    parser.add_argument("--clip-batch", type=int, default=96)
    args = parser.parse_args()

    images, class_names, _data = load_dataset(args.data, args.split)
    dataset_total = len(images)
    scan_start = max(0, int(args.start_index or 0))
    if dataset_total and scan_start >= dataset_total:
        scan_start = 0
    if args.max_images > 0:
        images = images[scan_start:scan_start + args.max_images]
    else:
        images = images[scan_start:]
    if not images:
        raise ValueError("No dataset images were found.")

    next_start = scan_start + len(images)
    if next_start >= dataset_total:
        next_start = 0
    history_path = normalized(args.review_history) if args.review_history else ""
    decisions = load_review_decisions(history_path)
    suppressed_keys = set()
    threshold = max(0.5, min(0.999, float(args.threshold)))
    requested_device = str(args.device or "cuda").lower()
    device = torch.device("cuda" if requested_device != "cpu" and torch.cuda.is_available() else "cpu")
    started = time.time()
    report = {
        "version": 1,
        "status": "running",
        "stage": "loading_teammate_scanner",
        "scanner": "clip_teammate_marker",
        "model": "CLIP teammate marker scanner",
        "data": normalized(args.data),
        "split": args.split,
        "task": args.task,
        "class_names": class_names,
        "settings": {
            "threshold": threshold,
            "device": str(device),
            "require_name_and_colored_symbol_layout": True,
        },
        "processed_images": 0,
        "total_images": len(images),
        "dataset_total_images": dataset_total,
        "scan_limit": max(0, int(args.max_images)),
        "scan_start_index": scan_start,
        "next_start_index": next_start,
        "review_history_path": history_path,
        "suppressed_issue_count": 0,
        "summary": {},
        "issues": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    output = normalized(args.output)
    write_report(output, report)

    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading local CLIP teammate scanner on {device}", flush=True)
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True, use_fast=True)
    model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    text_features = averaged_text_features(model, processor, device)
    report["stage"] = "finding_teammate_labels"
    write_report(output, report)

    image_batch = max(1, int(args.image_batch or 256))
    clip_batch = max(1, int(args.clip_batch or 96))
    processed = 0
    for chunk_start in range(0, len(images), image_batch):
        chunk = images[chunk_start:chunk_start + image_batch]
        candidates = []
        crops = []
        for image_path in chunk:
            try:
                ground_truth = parse_ground_truth(image_path, args.task, class_names)
                if not ground_truth:
                    continue
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                    for gt in ground_truth:
                        crop = marker_crop(image, gt)
                        if crop is not None:
                            candidates.append((image_path, gt))
                            crops.append(crop)
            except Exception as error:
                print(f"Skipped {image_path}: {error}", flush=True)

        scores = score_crops(
            model, processor, text_features, device, crops, clip_batch, return_details=True
        ) if crops else []
        for (image_path, gt), (score, name_score, symbol_score) in zip(candidates, scores):
            if score < threshold:
                continue
            issue = make_issue(
                image_path,
                str(Path(image_path).with_suffix(".txt")),
                args.task,
                "teammate_marker",
                gt=gt,
                detail=(
                    "Both a gamer name and a nearby colored teammate symbol appear above this labeled player "
                    f"(name {name_score:.3f}, symbol {symbol_score:.3f}). "
                    "Verify it, then use Remove Teammate Label to delete only this annotation row."
                ),
            )
            issue["confidence"] = round(score, 6)
            issue["name_tag_confidence"] = round(name_score, 6)
            issue["colored_symbol_confidence"] = round(symbol_score, 6)
            issue["severity"] = round(score, 6)
            issue["issue_key"] = validation_issue_key(issue)
            if issue["issue_key"] in decisions:
                suppressed_keys.add(issue["issue_key"])
            else:
                report["issues"].append(issue)

        processed += len(chunk)
        report["processed_images"] = processed
        report["summary"] = {"teammate_marker": len(report["issues"])}
        report["suppressed_issue_count"] = len(suppressed_keys)
        decisions = load_review_decisions(history_path)
        apply_review_decisions(report, decisions, suppressed_keys)
        write_report(output, report)
        print(f"Scanned {processed}/{len(images)} images; {len(report['issues'])} teammate candidates", flush=True)

    decisions = load_review_decisions(history_path)
    apply_review_decisions(report, decisions, suppressed_keys)
    report["status"] = "complete"
    report["stage"] = "complete"
    report["issue_image_count"] = len({issue["image_path"] for issue in report["issues"]})
    report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["elapsed_seconds"] = round(time.time() - started, 3)
    report["issues"].sort(key=lambda issue: (-float(issue.get("confidence", 0.0)), issue["image_path"]))
    write_report(output, report)
    if args.batch_state:
        write_report(normalized(args.batch_state), {
            "version": 1,
            "scanner": "clip_teammate_marker",
            "data": normalized(args.data),
            "split": args.split,
            "task": args.task,
            "scan_limit": max(0, int(args.max_images)),
            "completed_start_index": scan_start,
            "completed_image_count": len(images),
            "next_start_index": next_start,
            "dataset_total_images": dataset_total,
            "completed_at": report["completed_at"],
        })
    print(f"Teammate review complete: {len(report['issues'])} candidates", flush=True)


if __name__ == "__main__":
    main()
