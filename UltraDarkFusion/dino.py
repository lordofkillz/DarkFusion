import os
import logging
import argparse
import asyncio
import aiofiles
import torch
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast
import groundingdino
from groundingdino.util.inference import load_model, load_image, predict
import json

# Constants
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
THRESHOLD_FILE = "class_thresholds.json"
TEXT_THRESHOLD = 0.35
DEFAULT_THRESHOLD = 0.35
MIN_THRESHOLD = 0.25
MAX_THRESHOLD = 0.6
BATCH_SIZE = 20

# Logging Setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_thresholds():
    """Load previous thresholds or start fresh."""
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r") as f:
            return json.load(f)
    return {}


def save_thresholds(thresholds):
    """Save updated class thresholds."""
    try:
        with open(THRESHOLD_FILE, "w") as f:
            json.dump(thresholds, f, indent=4)
        logger.info(f"Updated thresholds saved to {THRESHOLD_FILE}")
    except Exception as e:
        logger.error(f"Failed to save thresholds: {e}")



def adjust_thresholds(detected_classes, class_thresholds):
    """Adjust confidence thresholds dynamically based on detections."""
    logger.info(f"Before Adjustment: {json.dumps(class_thresholds, indent=2)}")
    logger.info(f"Detected classes in this run: {detected_classes}")

    for cls in detected_classes:
        class_thresholds[cls] = max(MIN_THRESHOLD, class_thresholds.get(cls, DEFAULT_THRESHOLD) - 0.02)

    for cls in list(class_thresholds.keys()):
        if cls not in detected_classes:
            class_thresholds[cls] = min(MAX_THRESHOLD, class_thresholds[cls] + 0.02)

    save_thresholds(class_thresholds)
    logger.info(f"After Adjustment: {json.dumps(class_thresholds, indent=2)}")



async def write_to_disk(bbox_data_path, boxes, phrases, class_names, overwrite):
    """Write bounding box data asynchronously to disk, ensuring correct format."""
    if not overwrite and bbox_data_path.exists():
        async with aiofiles.open(bbox_data_path, 'r') as f:
            existing_boxes = [line.strip() for line in await f.readlines()]
    else:
        existing_boxes = []

    new_boxes = []
    for box, phrase in zip(boxes, phrases):
        if phrase in class_names:
            class_id = class_names.index(phrase)
            formatted_box = ' '.join(map(str, box))  # Ensure floats are converted correctly
            new_boxes.append(f"{class_id} {formatted_box}\n")

    async with aiofiles.open(bbox_data_path, 'w') as f:
        await f.writelines(existing_boxes + new_boxes)


async def process_image(image_path, model, TEXT_PROMPT, class_names, class_thresholds, overwrite):
    """Process a single image with GroundingDINO."""
    try:
        image_source, image_tensor = load_image(str(image_path))
        image_tensor = image_tensor.to(DEVICE)

        with autocast():
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=TEXT_PROMPT,
                box_threshold=0.1,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE
            )

        detected_classes = set()
        filtered_boxes, filtered_phrases = [], []
        
        for box, logit, phrase in zip(boxes, logits, phrases):
            normalized_phrase = phrase.lower().strip('.')

            if normalized_phrase in class_names and logit.item() > class_thresholds.get(normalized_phrase, DEFAULT_THRESHOLD):
                filtered_boxes.append(box.tolist())  # Convert tensor to Python list
                filtered_phrases.append(normalized_phrase)
                detected_classes.add(normalized_phrase)

        if filtered_boxes:
            bbox_data_path = image_path.with_suffix('.txt')
            await write_to_disk(bbox_data_path, filtered_boxes, filtered_phrases, class_names, overwrite)

        return detected_classes

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return set()


async def process_images(image_directory_path, model, TEXT_PROMPT, class_names, class_thresholds, overwrite):
    """Process images concurrently in batches."""
    image_directory = Path(image_directory_path)
    image_paths = [p for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif') for p in image_directory.glob(ext)]

    detected_classes = set()
    progress_bar = tqdm(total=len(image_paths), desc="Processing Images")
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[i:i+BATCH_SIZE]
        results = await asyncio.gather(*[
            process_image(image_path, model, TEXT_PROMPT, class_names, class_thresholds, overwrite) for image_path in batch
        ])
        
        for result in results:
            detected_classes.update(result)
        progress_bar.update(len(batch))
    
    progress_bar.close()
    adjust_thresholds(detected_classes, class_thresholds)


def run_groundingdino(image_directory_path, overwrite):
    """Run GroundingDINO with dynamic thresholds."""
    package_path = groundingdino.__path__[0]
    config_full_path = os.path.join(package_path, "config", "GroundingDINO_SwinT_OGC.py")
    model = load_model(config_full_path, "Sam/groundingdino_swint_ogc.pth", device=DEVICE)
    
    with open(os.path.join(image_directory_path, 'classes.txt'), 'r') as f:
        class_names = [line.strip().lower() for line in f if line.strip()]
    
    class_thresholds = load_thresholds()
    TEXT_PROMPT = '. '.join(class_names) + '.'
    
    asyncio.run(process_images(image_directory_path, model, TEXT_PROMPT, class_names, class_thresholds, overwrite))
    logger.info("Batch inference completed.")


async def main():
    parser = argparse.ArgumentParser(description='Run GroundingDINO on a directory of images.')
    parser.add_argument('image_directory', type=str, help='Path to the image directory')
    args = parser.parse_args()
    
    overwrite = input("Do you want to overwrite existing label files? (yes/no): ").strip().lower() == 'yes'
    run_groundingdino(args.image_directory, overwrite)


if __name__ == '__main__':
    asyncio.run(main())