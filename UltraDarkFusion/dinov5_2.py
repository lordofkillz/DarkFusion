import os
import json
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import groundingdino
from groundingdino.util.inference import load_model, load_image, predict

# Constants
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CACHE_DIR = "./hf_cache"
THRESHOLD_FILE = "class_thresholds.json"
TEXT_THRESHOLD = 0.35
DEFAULT_THRESHOLD = 0.35
MIN_THRESHOLD = 0.25
MAX_THRESHOLD = 0.6
BBOX_THRESHOLD = 0.1
BATCH_SIZE = 20

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load CLIP
def load_clip_models():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=CACHE_DIR).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=CACHE_DIR)
    return clip_model, clip_processor

# Threshold handling
def load_thresholds():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r") as f:
            return json.load(f)
    return {}

def save_thresholds(thresholds):
    with open(THRESHOLD_FILE, "w") as f:
        json.dump(thresholds, f, indent=4)

def adjust_thresholds(detected_classes, class_thresholds):
    for cls in detected_classes:
        class_thresholds[cls] = max(MIN_THRESHOLD, class_thresholds.get(cls, DEFAULT_THRESHOLD) - 0.02)
    for cls in list(class_thresholds.keys()):
        if cls not in detected_classes:
            class_thresholds[cls] = min(MAX_THRESHOLD, class_thresholds[cls] + 0.02)
    save_thresholds(class_thresholds)

# File I/O
def write_to_disk(bbox_data_path, boxes, phrases, class_names, overwrite):
    existing_labels = set()
    if not overwrite and os.path.exists(bbox_data_path):
        with open(bbox_data_path, 'r') as f:
            existing_labels = {line.strip() for line in f if line.strip()}

    new_labels = {
        f"{class_names.index(phrase)} {' '.join(map(str, box))}".strip()
        for box, phrase in zip(boxes, phrases) if phrase in class_names
    }

    combined_labels = new_labels if overwrite else existing_labels.union(new_labels)
    combined_labels = sorted(combined_labels, key=lambda x: (int(x.split()[0]), float(x.split()[1])))

    with open(bbox_data_path, 'w') as f:
        for label in combined_labels:
            f.write(label + '\n')

# CLIP filtering
def rank_with_clip(image_path, boxes, phrases, clip_model, clip_processor, fusion_ratio=0.4):
    img = Image.open(image_path).convert("RGB")
    results = []

    for box, phrase in zip(boxes, phrases):
        x_center, y_center, w, h = box
        img_width, img_height = img.size
        x1 = int((x_center - w / 2) * img_width)
        y1 = int((y_center - h / 2) * img_height)
        x2 = int((x_center + w / 2) * img_width)
        y2 = int((y_center + h / 2) * img_height)
        cropped = img.crop((x1, y1, x2, y2))

        inputs = clip_processor(text=[phrase], images=cropped, return_tensors="pt", padding=True).to(DEVICE)
        score = clip_model(**inputs).logits_per_image.softmax(dim=1)[0][0].item()
        results.append((score, box, phrase))

    return sorted(results, reverse=True, key=lambda x: x[0])

# Image Processing
def process_image(image_path, model, TEXT_PROMPT, class_names, class_thresholds, overwrite,
                  clip_model, clip_processor, fusion_ratio=0.4):
    try:
        _, image_tensor = load_image(str(image_path))
        image_tensor = image_tensor.to(DEVICE)

        with torch.amp.autocast(device_type="cuda"):
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=TEXT_PROMPT,
                box_threshold=BBOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE
            )

        dino_detections = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            phrase = phrase.lower().strip('.')
            if phrase in class_names:
                dino_detections.append((logit.item(), box.tolist(), phrase))

        ranked = rank_with_clip(image_path, [b for _, b, _ in dino_detections], [p for _, _, p in dino_detections],
                                clip_model, clip_processor)

        valid_boxes, valid_phrases, detected_classes = [], [], set()
        for clip_score, box, phrase in ranked:
            dino_score = next((s for s, b, p in dino_detections if p == phrase and b == box), 0)
            fused_score = fusion_ratio * clip_score + (1 - fusion_ratio) * dino_score
            if fused_score > class_thresholds.get(phrase, DEFAULT_THRESHOLD):
                valid_boxes.append(box)
                valid_phrases.append(phrase)
                detected_classes.add(phrase)

        if valid_boxes:
            write_to_disk(image_path.with_suffix('.txt'), valid_boxes, valid_phrases, class_names, overwrite)

        return detected_classes
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return set()

# Image batch processing
def process_images(image_directory_path, model, TEXT_PROMPT, class_names, class_thresholds, overwrite,
                   clip_model, clip_processor):
    image_directory = Path(image_directory_path)
    image_paths = [p for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp') for p in image_directory.glob(ext)]

    detected_classes = set()
    progress_bar = tqdm(total=len(image_paths), desc="Processing Images")

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[i:i + BATCH_SIZE]
        for p in batch:
            detected_classes.update(
                process_image(p, model, TEXT_PROMPT, class_names, class_thresholds, overwrite,
                              clip_model, clip_processor)
            )
        progress_bar.update(len(batch))

    progress_bar.close()
    adjust_thresholds(detected_classes, class_thresholds)

# Main entry
def run_groundingdino(image_directory_path, overwrite):
    config_path = os.path.join(groundingdino.__path__[0], "config", "GroundingDINO_SwinT_OGC.py")
    model = load_model(config_path, "Sam/groundingdino_swint_ogc.pth", device=DEVICE)
    clip_model, clip_processor = load_clip_models()

    with open(os.path.join(image_directory_path, 'classes.txt'), 'r') as f:
        class_names = [line.strip().lower() for line in f if line.strip()]

    class_thresholds = load_thresholds()
    TEXT_PROMPT = '. '.join(class_names) + '.'

    process_images(image_directory_path, model, TEXT_PROMPT, class_names, class_thresholds, overwrite,
                   clip_model, clip_processor)
    logger.info("Batch inference completed.")

def main(image_directory):
    overwrite = input("Do you want to overwrite existing label files? (yes/no): ").strip().lower() == 'yes'
    run_groundingdino(image_directory, overwrite)
