import os
import logging
import argparse
from pathlib import Path
import asyncio
import aiofiles
import cv2
import torch
from PIL import Image
import groundingdino
from groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm
from groundingdino.datasets import transforms as T
from torch.cuda.amp import autocast
import numpy as np

# Configure basic logging settings
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration for model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



async def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    # convert to format [x1, y1, x2, y2]
    box1 = [box1[0] - box1[2] / 2, box1[1] - box1[3] / 2,
            box1[0] + box1[2] / 2, box1[1] + box1[3] / 2]
    box2 = [box2[0] - box2[2] / 2, box2[1] - box2[3] / 2,
            box2[0] + box2[2] / 2, box2[1] + box2[3] / 2]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # compute the area of both the prediction and ground-truth rectangles
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou


async def write_to_disk(bbox_data_path, image, boxes, phrases, class_names, overwrite):
    """
    Write bounding box data to disk. If a duplicate 'grounding dino' box is found, it replaces the existing one.
    """
    # Prepare a list to hold all boxes that should be written to the file
    boxes_to_write = []

    # If we're not overwriting, load existing boxes and add them to the list
    if not overwrite and bbox_data_path.exists():
        async with aiofiles.open(bbox_data_path, 'r') as f:
            async for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    existing_box = [float(part) for part in parts[1:]]
                    boxes_to_write.append((existing_box, parts[0]))  # Box coordinates and class id

    # Prepare new boxes along with their class ids
    new_boxes_with_ids = []
    for box, phrase in zip(boxes, phrases):
        normalized_phrase = phrase.lower().strip('.')
        class_id = class_names.index(normalized_phrase) if normalized_phrase in class_names else -1
        if class_id != -1:
            new_boxes_with_ids.append((box, str(class_id)))

    # Now go through each new box and check for duplicates
    for new_box, class_id in new_boxes_with_ids:
        # Check against all existing boxes
        duplicate_found = False
        for i, (existing_box, existing_id) in enumerate(boxes_to_write):
            if await iou(existing_box, new_box) > 0.5:  # Assuming a 50% IoU threshold for duplicates
                # If it's a 'grounding dino', replace the existing box
                if "groundingdino" in phrases:
                    boxes_to_write[i] = (new_box, class_id)
                duplicate_found = True
                break  # Stop checking after finding a duplicate

        # If no duplicate was found, add the new box to the list
        if not duplicate_found:
            boxes_to_write.append((new_box, class_id))

    # Finally, write all boxes to the file
    async with aiofiles.open(bbox_data_path, 'w') as f:  # Always overwrite since we've merged boxes
        for box, class_id in boxes_to_write:
            xc, yc, w, h = box
            await f.write(f"{class_id} {xc} {yc} {w} {h}\n")





async def process_images(image_directory_path, model, TEXT_PROMPT, class_names, BOX_THRESHOLD, TEXT_THRESHOLD, overwrite):
    """
    Process all images in a given directory with a progress bar, based on overwrite flag.
    """
    image_directory = Path(image_directory_path)
    image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = [p for ext in image_formats for p in image_directory.glob(ext)]

    progress_bar = tqdm(total=len(image_paths), desc="Processing Images")
    for image_path in image_paths:
        try:
            image_source, image_tensor = load_image(str(image_path))
            
            # Send tensor to the correct device
            image_tensor = image_tensor.to(DEVICE)

            # Use autocast to handle FP16 conversion dynamically
            with autocast():
                boxes, logits, phrases = predict(
                    model=model,
                    image=image_tensor,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    device=DEVICE
                )
            
            bbox_data_path = image_path.with_suffix('.txt')
            # In your process_images function, make sure to pass the overwrite flag
            await write_to_disk(bbox_data_path, image_source, boxes, phrases, class_names, overwrite)

        
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
        
        progress_bar.update(1)
    progress_bar.close()

def run_groundingdino(image_directory_path, overwrite):
    package_path = groundingdino.__path__[0]
    config_full_path = os.path.join(package_path, "config", "GroundingDINO_SwinB_cfg.py")
    model = load_model(config_full_path, "Sam/groundingdino_swinb_cogcoor.pth", device=DEVICE)
    with open(os.path.join(image_directory_path, 'classes.txt'), 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip() != ""]

    TEXT_PROMPT = '.'.join(class_names) + '.'
    BOX_THRESHOLD = 0.40
    TEXT_THRESHOLD = 0.40

    asyncio.run(process_images(image_directory_path, model, TEXT_PROMPT, class_names, BOX_THRESHOLD, TEXT_THRESHOLD, overwrite))
    tqdm.write("Batch inference completed.")

def main():
    parser = argparse.ArgumentParser(description='Run GroundingDINO on a directory of images.')
    parser.add_argument('image_directory', type=str, help='Path to the image directory')
    args = parser.parse_args()
    
    # Ask user if they want to overwrite existing label files
    overwrite = input("Do you want to overwrite existing label files? (yes/no): ").strip().lower() == 'yes'
    
    run_groundingdino(args.image_directory, overwrite)

if __name__ == '__main__':
    main()
