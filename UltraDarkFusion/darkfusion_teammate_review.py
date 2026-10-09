"""Find labeled video-game teammates by the name/marker above their head.

The scanner is deliberately read-only.  It writes Validation Review compatible
issues; DarkFusion performs any selected label removal and records a recovery
log beside the dataset metadata.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import os
import re
import threading
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

HUD_TEXT = {
    "ammo", "apex", "assist", "bamboozled", "banner", "cancel", "care package",
    "charging station", "close", "control", "damage", "decoy", "detected",
    "eliminated", "elimination", "enemy", "evo orb", "final round", "games",
    "get back", "hangar", "hold", "knocked", "locked on", "back", "low ammo",
    "max level", "open", "ping", "position revealed", "recover", "reload",
    "replicator", "respawn", "revive", "ring", "round", "scan", "shield",
    "sonar", "squad", "supplies", "toggle", "underdog bonus", "wild card",
    "safe zone", "weapon", "weapon upgrade", "upgrade", "objective", "capture",
    "interact", "ultimate", "ability", "inventory", "health", "armory",
}

_OCR_READERS = {}
_OCR_READER_LOCK = threading.Lock()


class TeammateMarkerClassifier:
    """Reusable, lazily loaded CLIP classifier for auto-label post-filtering."""

    def __init__(self, device="cuda"):
        requested = str(device or "cuda").lower()
        self.device = torch.device(
            "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"
        )
        self.processor = None
        self.model = None
        self.text_features = None
        self._model_lock = threading.Lock()
        self._ocr_reader = None
        self.last_ocr_error = ""
        self.last_evidence_counts = {"visual_marker": 0, "ocr_gamer_tag": 0}

    def _ensure_model(self):
        if self.model is not None:
            return
        with self._model_lock:
            if self.model is not None:
                return
            model_name = "openai/clip-vit-base-patch32"

            def load_cached_or_download(loader, **kwargs):
                try:
                    return loader.from_pretrained(model_name, local_files_only=True, **kwargs)
                except OSError:
                    # Only the first uncached marker candidate needs internet.
                    return loader.from_pretrained(model_name, local_files_only=False, **kwargs)

            processor = load_cached_or_download(CLIPProcessor, use_fast=True)
            model = load_cached_or_download(CLIPModel).to(self.device).eval()
            features = averaged_text_features(model, processor, self.device)
            self.processor = processor
            self.text_features = features
            self.model = model

    def score(self, crops, batch_size=96):
        if not crops:
            return []
        self._ensure_model()
        return score_crops(
            self.model,
            self.processor,
            self.text_features,
            self.device,
            list(crops),
            max(1, int(batch_size or 96)),
        )

    def score_details(self, crops, batch_size=96):
        if not crops:
            return []
        self._ensure_model()
        return score_crops(
            self.model,
            self.processor,
            self.text_features,
            self.device,
            list(crops),
            max(1, int(batch_size or 96)),
            return_details=True,
        )

    def _easyocr_reader(self):
        if self._ocr_reader is not None:
            return self._ocr_reader
        import easyocr
        cache_key = "cuda" if self.device.type == "cuda" else "cpu"
        with _OCR_READER_LOCK:
            reader = _OCR_READERS.get(cache_key)
            if reader is None:
                reader = easyocr.Reader(
                    ["en"],
                    gpu=self.device.type == "cuda",
                    download_enabled=True,
                    verbose=False,
                )
                _OCR_READERS[cache_key] = reader
        self._ocr_reader = reader
        return self._ocr_reader

    def friendly_indices(
        self,
        images,
        bounds_batches,
        *,
        threshold=0.90,
        batch_size=96,
        ocr=True,
        ocr_confidence=0.20,
        ocr_name_semantic=0.15,
        ocr_symbol_semantic=0.15,
    ):
        """Return friendly-player indices for each image using visual and OCR evidence.

        The visual path keeps the existing shape-agnostic colored-marker plus
        nameplate test. OCR is a second path for unfamiliar marker shapes: text
        must look like a gamer tag, be spatially tied to the proposed player,
        and have independent CLIP nameplate and colored-symbol evidence.
        """
        images = list(images or [])
        bounds_batches = list(bounds_batches or [])
        rejected = [set() for _image in images]
        self.last_ocr_error = ""
        self.last_evidence_counts = {"visual_marker": 0, "ocr_gamer_tag": 0}
        crops = []
        references = []
        normalized_images = []
        for image in images:
            if isinstance(image, Image.Image):
                normalized_images.append(image.convert("RGB"))
            elif isinstance(image, (str, os.PathLike)) and os.path.isfile(image):
                with Image.open(image) as opened:
                    normalized_images.append(opened.convert("RGB"))
            else:
                normalized_images.append(None)

        for image_index, image in enumerate(normalized_images):
            if image is None:
                continue
            boxes = bounds_batches[image_index] if image_index < len(bounds_batches) else []
            for box_index, bounds in enumerate(boxes or []):
                crop = marker_crop(image, {"bbox": bounds})
                if crop is not None:
                    crops.append(crop)
                    references.append((image_index, box_index, bounds))

        # Most predictions have no friendly marker at all.  Run the cheap,
        # local layout check first so ordinary auto-labeling never pays for a
        # CLIP/OCR pass.  This is deliberately conservative: it only permits
        # a candidate to advance when a compact colored symbol and a text-like
        # row are both present over the predicted player's head.
        screened = [
            (reference, crop)
            for reference, crop in zip(references, crops)
            if has_paired_marker_layout(crop)
        ]
        if not screened:
            return rejected

        screened_references = [reference for reference, _crop in screened]
        details = self.score_details(
            [crop for _reference, crop in screened], batch_size=batch_size
        )
        ocr_candidates = []
        threshold = max(0.50, min(0.999, float(threshold or 0.90)))
        for reference, detail in zip(screened_references, details):
            image_index, box_index, bounds = reference
            score, name_score, symbol_score = [float(value) for value in detail]
            if score >= threshold:
                rejected[image_index].add(box_index)
                self.last_evidence_counts["visual_marker"] += 1
                continue
            if (
                ocr
                and name_score >= float(ocr_name_semantic)
                and symbol_score >= float(ocr_symbol_semantic)
            ):
                ocr_candidates.append((image_index, box_index, bounds))

        if not ocr_candidates:
            return rejected

        ocr_crops = []
        ocr_references = []
        for image_index, box_index, bounds in ocr_candidates:
            context = wide_nameplate_ocr_crop(
                normalized_images[image_index], {"bbox": bounds}
            )
            if context is None:
                continue
            crop, anchor_x, anchor_y = context
            ocr_crops.append(crop)
            ocr_references.append((image_index, box_index, anchor_x, anchor_y))

        if not ocr_crops:
            return rejected
        try:
            reader = self._easyocr_reader()
            outputs = reader.readtext_batched(
                ocr_crops,
                # OCR only needs the short player-name strip.  Keep its input
                # deliberately small so it remains a quick post-filter rather
                # than a second full-frame inference pass.
                n_width=384,
                n_height=112,
                canvas_size=384,
                batch_size=min(max(1, int(batch_size or 96)), 16),
                workers=0,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_[]- ",
            )
        except Exception as error:
            self.last_ocr_error = str(error)
            return rejected

        for (image_index, box_index, anchor_x, anchor_y), results in zip(
            ocr_references, outputs
        ):
            text, _confidence = gamer_tag_from_ocr(
                results,
                112,
                float(ocr_confidence),
                image_width=384,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
            )
            if text:
                rejected[image_index].add(box_index)
                self.last_evidence_counts["ocr_gamer_tag"] += 1
        return rejected


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


def wide_nameplate_ocr_crop(image, ground_truth, output_size=(384, 112)):
    """Return a compact nameplate strip directly above one predicted player.

    The historical function name is retained for compatibility.  This crop is
    intentionally narrow: teammate OCR reads a gamer tag, never the screen's
    HUD or the rest of the game frame.
    """
    bounds = list(ground_truth.get("bbox", []) or [])
    if len(bounds) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bounds[:4]]
    width, height = image.size
    box_width = max(1.0, (x2 - x1) * width)
    box_height = max(1.0, (y2 - y1) * height)
    center_x = (x1 + x2) * width / 2.0
    top_y = y1 * height
    # A nameplate is generally centered at the head and only modestly wider
    # than the player box.  This is roughly one sixth of the previous OCR
    # search area, which keeps EasyOCR latency and GPU pressure low.
    left = max(0, round(center_x - max(box_width * 2.25, width * 0.10)))
    right = min(width, round(center_x + max(box_width * 2.25, width * 0.10)))
    top = max(0, round(top_y - max(box_height * 1.55, height * 0.09)))
    bottom = min(height, round(top_y + max(box_height * 0.18, height * 0.015)))
    if right - left < 8 or bottom - top < 8:
        return None
    output_width, output_height = output_size
    crop = image.crop((left, top, right, bottom)).resize(
        (output_width, output_height), Image.Resampling.LANCZOS
    )
    anchor_x = (center_x - left) / (right - left) * output_width
    anchor_y = (top_y - top) / (bottom - top) * output_height
    return np.asarray(crop), anchor_x, anchor_y


def gamer_tag_from_ocr(
    results,
    image_height,
    confidence_threshold,
    *,
    image_width=None,
    anchor_x=None,
    anchor_y=None,
):
    """Select readable player-name text close to the proposed player's head."""
    best_text = ""
    best_confidence = 0.0
    for bounds, text, confidence in results:
        cleaned = " ".join(str(text).split()).strip(" -_")
        lowered = cleaned.lower()
        letters = sum(character.isalpha() for character in cleaned)
        if float(confidence) < float(confidence_threshold) or letters < 4:
            continue
        if looks_like_hud_text(lowered):
            continue
        center_y = sum(float(point[1]) for point in bounds) / max(1, len(bounds))
        if center_y > image_height * 0.82:
            continue
        if anchor_x is not None and anchor_y is not None and image_width is not None:
            xs = [float(point[0]) for point in bounds]
            horizontal_gap = max(min(xs) - anchor_x, anchor_x - max(xs), 0.0)
            vertical_gap = anchor_y - center_y
            centered_above = (
                horizontal_gap <= image_width * 0.125
                and -image_height * 0.05 <= vertical_gap <= image_height * 0.48
            )
            adjacent_to_icon = (
                horizontal_gap <= image_width * 0.23
                and -image_height * 0.03 <= vertical_gap <= image_height * 0.25
            )
            if not (centered_above or adjacent_to_icon):
                continue
        if float(confidence) > best_confidence:
            best_text = cleaned
            best_confidence = float(confidence)
    return best_text, best_confidence


def looks_like_hud_text(text):
    """Reject exact and slightly clipped OCR readings of common game HUD text."""
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))
    if not normalized:
        return False
    padded = f" {normalized} "
    if any(f" {phrase} " in padded for phrase in HUD_TEXT):
        return True
    hud_tokens = {
        token
        for phrase in HUD_TEXT
        for token in re.findall(r"[a-z]+", phrase)
        if len(token) >= 4
    }
    for token in normalized.split():
        if len(token) < 4:
            continue
        for hud_token in hud_tokens:
            length_gap = abs(len(token) - len(hud_token))
            if length_gap > 1:
                continue
            if SequenceMatcher(None, token, hud_token).ratio() >= 0.84:
                return True
    return False


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

    print(f"Preparing lazy CLIP teammate scanner on {device}", flush=True)
    classifier = TeammateMarkerClassifier(str(device))
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

        scores = classifier.score_details(crops, batch_size=clip_batch)
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
