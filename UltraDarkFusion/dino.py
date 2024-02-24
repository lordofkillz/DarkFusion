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
# Configure basic logging settings
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration for model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

async def write_to_disk(bbox_data_path, image, boxes, phrases, class_names):
    """
    Write bounding box data and image to disk.
    """
    # Save bounding box data
    async with aiofiles.open(bbox_data_path, 'w') as f:
        for box, phrase in zip(boxes, phrases):
            normalized_phrase = phrase.lower().strip('.')
            class_id = class_names.index(normalized_phrase) if normalized_phrase in class_names else -1
            if class_id != -1:
                xc, yc, w, h = box
                await f.write(f"{class_id} {xc} {yc} {w} {h}\n")

    # Convert image tensor to numpy array if it's a tensor
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    # Convert from RGB to BGR for OpenCV if image is RGB
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save image
    cv2.imwrite(str(bbox_data_path.with_suffix('.jpg')), image)

async def process_images(image_directory_path, model, TEXT_PROMPT, class_names, BOX_THRESHOLD, TEXT_THRESHOLD):
    """
    Process all images in a given directory with a progress bar.
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
            await write_to_disk(bbox_data_path, image_source, boxes, phrases, class_names)
        
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
        
        progress_bar.update(1)
    progress_bar.close()


def run_groundingdino(image_directory_path):
    package_path = groundingdino.__path__[0]
    config_full_path = os.path.join(package_path, "config", "GroundingDINO_SwinT_OGC.py")
    model = load_model(config_full_path, "Sam/groundingdino_swint_ogc.pth", device=DEVICE)
    with open(os.path.join(image_directory_path, 'classes.txt'), 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip() != ""]

    TEXT_PROMPT = '.'.join(class_names) + '.'
    BOX_THRESHOLD = 0.40
    TEXT_THRESHOLD = 0.40

    asyncio.run(process_images(image_directory_path, model, TEXT_PROMPT, class_names, BOX_THRESHOLD, TEXT_THRESHOLD))
    tqdm.write("Batch inference completed.")

def main():
    parser = argparse.ArgumentParser(description='Run GroundingDINO on a directory of images.')
    parser.add_argument('image_directory', type=str, help='Path to the image directory')
    args = parser.parse_args()
    
    run_groundingdino(args.image_directory)

if __name__ == '__main__':
    main()