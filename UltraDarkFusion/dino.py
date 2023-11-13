import os
import logging
import argparse
from pathlib import Path
import cv2
import groundingdino
from groundingdino.util.inference import load_model, load_image, predict, annotate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_groundingdino(image_directory_path: str):
    package_path = groundingdino.__path__[0]
    config_full_path = os.path.join(package_path, "config", "GroundingDINO_SwinT_OGC.py")
    try:
        model = load_model(config_full_path, "sam/groundingdino_swint_ogc.pth")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    image_directory = Path(image_directory_path)
    classes_file_path = image_directory / 'classes.txt'
    
    if not classes_file_path.is_file():
        logging.error(f"No 'classes.txt' found in the directory: {classes_file_path}")
        return

    with open(classes_file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip() != ""]

    TEXT_PROMPT = '.'.join(class_names) + '.'

    BOX_THRESHOLD = 0.35  # Consider moving this to a config file
    TEXT_THRESHOLD = 0.35  # Consider moving this to a config file

    image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']  # Consider moving this to a config file

    image_paths = [p for ext in image_formats for p in image_directory.glob(ext)]

    for image_path in image_paths:
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
            with open(bbox_data_path, 'w') as f:
                for box, phrase in zip(boxes, phrases):
                    normalized_phrase = phrase.lower().strip('.')
                    class_id = class_names.index(normalized_phrase) if normalized_phrase in class_names else -1
                    if class_id != -1:
                        x, y, w, h = box
                        f.write(f"{class_id} {x} {y} {w} {h}\n")
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")

    logging.info("Batch inference and saving completed.")

def main():
    parser = argparse.ArgumentParser(description='Run GroundingDINO on a directory of images.')
    parser.add_argument('image_directory', type=str, help='Path to the image directory')
    args = parser.parse_args()
    
    run_groundingdino(args.image_directory)

if __name__ == '__main__':
    main()

