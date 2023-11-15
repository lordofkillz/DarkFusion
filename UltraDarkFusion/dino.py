import os
import logging
import argparse
from pathlib import Path
import asyncio
import aiofiles
import groundingdino
from groundingdino.util.inference import load_model, load_image, predict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def write_to_disk(bbox_data_path, boxes, phrases, class_names):
    async with aiofiles.open(bbox_data_path, 'w') as f:
        for box, phrase in zip(boxes, phrases):
            normalized_phrase = phrase.lower().strip('.')
            class_id = class_names.index(normalized_phrase) if normalized_phrase in class_names else -1
            if class_id != -1:
                x, y, w, h = box
                await f.write(f"{class_id} {x} {y} {w} {h}\n")

async def process_image(model, image_path, TEXT_PROMPT, class_names, BOX_THRESHOLD, TEXT_THRESHOLD):
    try:
        image_source, image = load_image(str(image_path))
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        bbox_data_path = image_path.with_suffix('.txt')
        await write_to_disk(bbox_data_path, boxes, phrases, class_names)
        logging.info(f"Completed processing image: {image_path}")
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")

async def process_images(image_directory_path: str, model, TEXT_PROMPT, class_names, BOX_THRESHOLD, TEXT_THRESHOLD):
    image_directory = Path(image_directory_path)
    image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = [p for ext in image_formats for p in image_directory.glob(ext)]
    
    tasks = []
    for image_path in image_paths:
        task = asyncio.create_task(process_image(model, image_path, TEXT_PROMPT, class_names, BOX_THRESHOLD, TEXT_THRESHOLD))
        tasks.append(task)
    
    await asyncio.gather(*tasks)

def run_groundingdino(image_directory_path: str):
    package_path = groundingdino.__path__[0]
    config_full_path = os.path.join(package_path, "config", "GroundingDINO_SwinT_OGC.py")
    model = load_model(config_full_path, "sam/groundingdino_swint_ogc.pth")

    image_directory = Path(image_directory_path)
    classes_file_path = image_directory / 'classes.txt'
    
    if not classes_file_path.is_file():
        logging.error(f"No 'classes.txt' found in the directory: {classes_file_path}")
        return

    with open(classes_file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip() != ""]

    TEXT_PROMPT = '.'.join(class_names) + '.'
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.35
    image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = [p for ext in image_formats for p in image_directory.glob(ext)]

    # Assuming you're processing images in batches of 4
    batch_size = 100
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
    asyncio.run(process_images(image_directory_path, model, TEXT_PROMPT, class_names, BOX_THRESHOLD, TEXT_THRESHOLD))

    logging.info("Batch inference completed.")

def main():
    parser = argparse.ArgumentParser(description='Run GroundingDINO on a directory of images.')
    parser.add_argument('image_directory', type=str, help='Path to the image directory')
    args = parser.parse_args()
    
    run_groundingdino(args.image_directory)

if __name__ == '__main__':
    main()

