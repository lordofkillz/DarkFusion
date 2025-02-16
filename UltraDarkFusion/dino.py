import os
import logging
import argparse
import asyncio
import aiofiles
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast
import groundingdino
from groundingdino.util.inference import load_model, load_image, predict
import json

# Configure logging
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# Device setup
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Threshold tracking file
THRESHOLD_FILE = "class_thresholds.json"

def load_thresholds():
    """ Load previous thresholds or start fresh. """
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r") as f:
            return json.load(f)
    return {}

def save_thresholds(thresholds):
    """ Save updated class thresholds. """
    with open(THRESHOLD_FILE, "w") as f:
        json.dump(thresholds, f, indent=4)

def adjust_thresholds(detected_classes, class_thresholds):
    """
    Automatically adjusts thresholds:
    - If a class is detected **often**, lower its threshold slightly.
    - If a class is rarely detected, increase its threshold.
    """
    for cls in detected_classes:
        if cls in class_thresholds:
            class_thresholds[cls] = max(0.25, class_thresholds[cls] - 0.02)  # Reduce threshold slightly
        else:
            class_thresholds[cls] = 0.35  # Default confidence for new classes
    
    for cls in list(class_thresholds.keys()):
        if cls not in detected_classes:
            class_thresholds[cls] = min(0.6, class_thresholds[cls] + 0.02)  # Increase threshold slightly

    save_thresholds(class_thresholds)

async def write_to_disk(bbox_data_path, boxes, phrases, class_names, overwrite):
    """ Write bounding box data to disk, ensuring only current `classes.txt` classes are included. """
    boxes_to_write = []
    if not overwrite and bbox_data_path.exists():
        async with aiofiles.open(bbox_data_path, 'r') as f:
            async for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    existing_box = [float(part) for part in parts[1:]]
                    boxes_to_write.append((existing_box, parts[0]))

    new_boxes_with_ids = []
    for box, phrase in zip(boxes, phrases):
        normalized_phrase = phrase.lower().strip('.')
        if normalized_phrase in class_names:
            new_boxes_with_ids.append((box, str(class_names.index(normalized_phrase))))

    async with aiofiles.open(bbox_data_path, 'w') as f:
        for box, class_id in new_boxes_with_ids:
            xc, yc, w, h = box
            await f.write(f"{class_id} {xc} {yc} {w} {h}\n")

async def process_images(image_directory_path, model, TEXT_PROMPT, class_names, class_thresholds, TEXT_THRESHOLD, overwrite):
    """
    Process all images in a directory with GroundingDINO.
    Uses dynamic class-based thresholds.
    """
    image_directory = Path(image_directory_path)
    image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = [p for ext in image_formats for p in image_directory.glob(ext)]

    detected_classes = set()

    progress_bar = tqdm(total=len(image_paths), desc="Processing Images")
    for image_path in image_paths:
        try:
            image_source, image_tensor = load_image(str(image_path))
            image_tensor = image_tensor.to(DEVICE)

            with autocast():
                boxes, logits, phrases = predict(
                    model=model,
                    image=image_tensor,
                    caption=TEXT_PROMPT,
                    box_threshold=0.1,  # Start with a low threshold
                    text_threshold=TEXT_THRESHOLD,
                    device=DEVICE
                )

            filtered_boxes = []
            filtered_phrases = []
            for box, logit, phrase in zip(boxes, logits, phrases):
                normalized_phrase = phrase.lower().strip('.')
                if normalized_phrase in class_names:
                    dynamic_threshold = class_thresholds.get(normalized_phrase, 0.35)
                    if logit > dynamic_threshold:  # Apply auto-adjusted confidence threshold
                        filtered_boxes.append(box)
                        filtered_phrases.append(normalized_phrase)
                        detected_classes.add(normalized_phrase)

            if not filtered_boxes:
                logging.info(f"No valid detections for {image_path}, skipping.")
                continue

            bbox_data_path = image_path.with_suffix('.txt')
            await write_to_disk(bbox_data_path, filtered_boxes, filtered_phrases, class_names, overwrite)

        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")

        progress_bar.update(1)

    progress_bar.close()
    adjust_thresholds(detected_classes, class_thresholds)  # Auto-adjust thresholds after each run

def run_groundingdino(image_directory_path, overwrite):
    """Run GroundingDINO with auto-learning thresholds."""
    package_path = groundingdino.__path__[0]
    config_full_path = os.path.join(package_path, "config", "GroundingDINO_SwinT_OGC.py")
    model = load_model(config_full_path, "Sam/groundingdino_swint_ogc.pth", device=DEVICE)

    # Load class names from `classes.txt`
    with open(os.path.join(image_directory_path, 'classes.txt'), 'r') as f:
        class_names = [line.strip().lower() for line in f.readlines() if line.strip()]

    # Load previously learned thresholds or start fresh
    class_thresholds = load_thresholds()

    TEXT_PROMPT = '. '.join(class_names) + '.'
    TEXT_THRESHOLD = 0.35

    asyncio.run(process_images(image_directory_path, model, TEXT_PROMPT, class_names, class_thresholds, TEXT_THRESHOLD, overwrite))
    tqdm.write("Batch inference completed.")

def main():
    parser = argparse.ArgumentParser(description='Run GroundingDINO on a directory of images.')
    parser.add_argument('image_directory', type=str, help='Path to the image directory')
    args = parser.parse_args()

    overwrite = input("Do you want to overwrite existing label files? (yes/no): ").strip().lower() == 'yes'
    run_groundingdino(args.image_directory, overwrite)

if __name__ == '__main__':
    main()


