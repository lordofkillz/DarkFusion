import os
import logging
import argparse
from pathlib import Path
import asyncio
import aiofiles
import cv2
import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import groundingdino
from groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm
from torch.cuda.amp import autocast
import numpy as np
import random

# Configure basic logging settings
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration for model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load pre-trained LLM for additional text interpretation
LLM_MODEL_NAME = "bert-base-uncased"
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForSequenceClassification.from_pretrained(LLM_MODEL_NAME).to(DEVICE)

def enhance_contrast(image):
    """
    Enhance image contrast using CLAHE for better visibility of small objects.
    """
    lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_image)

async def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    box1 = [box1[0] - box1[2] / 2, box1[1] - box1[3] / 2,
            box1[0] + box1[2] / 2, box1[1] + box1[3] / 2]
    box2 = [box2[0] - box2[2] / 2, box2[1] - box2[3] / 2,
            box2[0] + box2[2] / 2, box2[1] + box2[3] / 2]

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou


async def write_to_disk(bbox_data_path, boxes, phrases, class_names, overwrite):
    """
    Write bounding box data to disk.
    """
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
        class_id = class_names.index(normalized_phrase) if normalized_phrase in class_names else -1
        if class_id != -1:
            new_boxes_with_ids.append((box, str(class_id)))

    for new_box, class_id in new_boxes_with_ids:
        duplicate_found = False
        for i, (existing_box, existing_id) in enumerate(boxes_to_write):
            if await iou(existing_box, new_box) > 0.5:
                boxes_to_write[i] = (new_box, class_id)
                duplicate_found = True
                break
        if not duplicate_found:
            boxes_to_write.append((new_box, class_id))

    async with aiofiles.open(bbox_data_path, 'w') as f:
        for box, class_id in boxes_to_write:
            xc, yc, w, h = box
            await f.write(f"{class_id} {xc} {yc} {w} {h}\n")

async def process_images(image_directory_path, model, TEXT_PROMPT, class_names, BOX_THRESHOLD, TEXT_THRESHOLD, overwrite):
    """
    Process all images in a directory.
    """
    image_directory = Path(image_directory_path)
    image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = [p for ext in image_formats for p in image_directory.glob(ext)]

    progress_bar = tqdm(total=len(image_paths), desc="Processing Images")
    for image_path in image_paths:
        try:
            image_source, image_tensor = load_image(str(image_path))
            image_source = enhance_contrast(image_source)

            # Additional LLM processing for refined text interpretation
            llm_inputs = llm_tokenizer(TEXT_PROMPT, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            with torch.no_grad():
                llm_outputs = llm_model(**llm_inputs)

            image_tensor = image_tensor.to(DEVICE)
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
            await write_to_disk(bbox_data_path, boxes, phrases, class_names, overwrite)

        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")

        progress_bar.update(1)
    progress_bar.close()

def run_groundingdino(image_directory_path, overwrite):
    package_path = groundingdino.__path__[0]
    config_full_path = os.path.join(package_path, "config", "GroundingDINO_SwinT_OGC.py")
    model = load_model(config_full_path, "Sam/groundingdino_swint_ogc.pth", device=DEVICE)
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

    overwrite = input("Do you want to overwrite existing label files? (yes/no): ").strip().lower() == 'yes'
    run_groundingdino(args.image_directory, overwrite)

if __name__ == '__main__':
    main()
