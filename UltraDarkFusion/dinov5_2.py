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
import contextlib
from math import inf
import math
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
# New defaults for per-class tuning when a class isn't in JSON yet
FUSION_DEFAULTS = {
    "thr": DEFAULT_THRESHOLD,  # decision threshold
    "alpha": 0.40,             # weight for CLIP vs DINO (fused = alpha*clip + (1-alpha)*dino)
    "T_clip": 0.80,            # temperature for CLIP score calibration
    "T_dino": 1.10,            # temperature for DINO score calibration
    "min_area_frac": 0.0005,   # reject boxes smaller than this fraction of image area
    "max_ar": 8.0              # reject boxes with aspect ratio > this
}

# NMS
try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None

def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def _calibrate_prob(p, T):
    # Safely map p in (0,1) -> logit -> divide by T -> sigmoid
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = math.log(p/(1-p))
    return _sigmoid(z / max(T, 1e-6))

def _ensure_class_cfg(class_thresholds, phrase):
    if phrase not in class_thresholds or not isinstance(class_thresholds[phrase], dict):
        class_thresholds[phrase] = dict(FUSION_DEFAULTS)
    # backfill missing keys
    for k, v in FUSION_DEFAULTS.items():
        class_thresholds[phrase].setdefault(k, v)
    return class_thresholds[phrase]

def xywhn_to_xyxy_pix(box_xywhn, W, H):
    # box in normalized (x_center,y_center,w,h) -> pixel (x1,y1,x2,y2)
    xc, yc, w, h = box_xywhn
    x1 = (xc - w/2.0) * W
    y1 = (yc - h/2.0) * H
    x2 = (xc + w/2.0) * W
    y2 = (yc + h/2.0) * H
    return [max(0, x1), max(0, y1), min(W-1, x2), min(H-1, y2)]

def xyxy_pix_to_xywhn(xyxy, W, H):
    x1, y1, x2, y2 = xyxy
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    xc = (x1 + x2) / (2.0 * W)
    yc = (y1 + y2) / (2.0 * H)
    return [xc, yc, w, h]

def box_area_frac(xyxy, W, H):
    x1, y1, x2, y2 = xyxy
    if x2 <= x1 or y2 <= y1: 
        return 0.0
    return ((x2 - x1) * (y2 - y1)) / float(W * H)

def box_ar(xyxy):
    x1, y1, x2, y2 = xyxy
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    a = max(w/h, h/w)
    return a

def classwise_nms(xyxy_list, score_list, label_list, iou_thresh=0.55):
    if not xyxy_list:
        return []
    import torch
    xyxy = torch.tensor(xyxy_list, dtype=torch.float32)
    scores = torch.tensor(score_list, dtype=torch.float32)
    labels = torch.tensor(label_list, dtype=torch.int64)
    keep = []
    if tv_nms is None:
        # simple python NMS per class to avoid extra deps
        for c in labels.unique().tolist():
            idx = (labels == c).nonzero(as_tuple=True)[0].tolist()
            idx_sorted = sorted(idx, key=lambda i: scores[i], reverse=True)
            while idx_sorted:
                i = idx_sorted.pop(0)
                keep.append(i)
                remain = []
                for j in idx_sorted:
                    # IoU
                    xa = max(xyxy[i,0], xyxy[j,0]); ya = max(xyxy[i,1], xyxy[j,1])
                    xb = min(xyxy[i,2], xyxy[j,2]); yb = min(xyxy[i,3], xyxy[j,3])
                    inter = max(0, xb - xa) * max(0, yb - ya)
                    ai = (xyxy[i,2]-xyxy[i,0])*(xyxy[i,3]-xyxy[i,1])
                    aj = (xyxy[j,2]-xyxy[j,0])*(xyxy[j,3]-xyxy[j,1])
                    iou = inter / max(1e-6, ai + aj - inter)
                    if iou <= iou_thresh:
                        remain.append(j)
                idx_sorted = remain
    else:
        for c in labels.unique().tolist():
            idx = (labels == c).nonzero(as_tuple=True)[0]
            k = tv_nms(xyxy[idx], scores[idx], iou_thresh)
            keep += idx[k].tolist()
    return sorted(set(keep))



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
        cfg = _ensure_class_cfg(class_thresholds, cls)
        cfg["thr"] = max(MIN_THRESHOLD, cfg.get("thr", DEFAULT_THRESHOLD) - 0.02)
        class_thresholds[cls] = cfg
    for cls in list(class_thresholds.keys()):
        cfg = _ensure_class_cfg(class_thresholds, cls)
        if cls not in detected_classes:
            cfg["thr"] = min(MAX_THRESHOLD, cfg.get("thr", DEFAULT_THRESHOLD) + 0.02)
        class_thresholds[cls] = cfg
    save_thresholds(class_thresholds)


# File I/O
def write_to_disk(bbox_data_path, boxes, phrases, class_names, overwrite):
    tmp_path = Path(str(bbox_data_path) + ".tmp")
    existing_labels = set()

    if not overwrite and os.path.exists(bbox_data_path):
        with open(bbox_data_path, 'r') as f:
            existing_labels = {line.strip() for line in f if line.strip()}

    new_labels = {
        f"{class_names.index(phrase)} {' '.join(map(str, box))}".strip()
        for box, phrase in zip(boxes, phrases) if phrase in class_names
    }

    combined = new_labels if overwrite else existing_labels.union(new_labels)
    combined = sorted(combined, key=lambda x: (int(x.split()[0]), float(x.split()[1])))

    with open(tmp_path, 'w') as f:
        for line in combined:
            f.write(line + '\n')

    # atomic replace
    os.replace(tmp_path, bbox_data_path)


# CLIP filtering
def rank_with_clip(image_path, boxes, phrases, clip_model, clip_processor):
    if not boxes:
        return []

    img = Image.open(image_path).convert("RGB")
    crops, texts = [], []
    W, H = img.size

    for box, phrase in zip(boxes, phrases):
        xc, yc, w, h = box
        x1 = int((xc - w/2) * W); y1 = int((yc - h/2) * H)
        x2 = int((xc + w/2) * W); y2 = int((yc + h/2) * H)
        crops.append(img.crop((x1, y1, x2, y2)))
        texts.append(phrase)

    inputs = clip_processor(text=texts, images=crops, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        # similarity matrix: (N_images, N_texts)
        sims = clip_model(**inputs).logits_per_image
        probs = sims.softmax(dim=1)           # row-wise softmax over texts
        diag_scores = probs.diag().tolist()   # score of image i with its paired text i

    ranked = list(zip(diag_scores, boxes, phrases))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


# Image Processing
def process_image(image_path, model, TEXT_PROMPT, class_names, class_thresholds, overwrite,
                  clip_model, clip_processor, fusion_ratio=0.4, nms_iou=0.55):
    try:
        pil_img = Image.open(image_path).convert("RGB")
        W, H = pil_img.size

        _, image_tensor = load_image(str(image_path))
        image_tensor = image_tensor.to(DEVICE)

        amp_ctx = torch.amp.autocast if DEVICE.type == "cuda" else contextlib.nullcontext
        with amp_ctx(device_type="cuda") if DEVICE.type == "cuda" else amp_ctx():
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=TEXT_PROMPT,
                box_threshold=BBOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE
            )

        # Collect DINO detections for our classes
        dino_dets = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            phrase = phrase.lower().strip('.')
            if phrase in class_names:
                dino_dets.append((float(logit.item()), list(map(float, box.tolist())), phrase))

        if not dino_dets:
            return set()

        # Run batched CLIP scoring on those boxes
        ranked = rank_with_clip(image_path,
                                [b for _, b, _ in dino_dets],
                                [p for _, _, p in dino_dets],
                                clip_model, clip_processor)

        # Fuse scores with per-class calibration
        cand = []  # (xyxy_pix, fused_score, class_idx, phrase, clip_score, dino_score)
        for clip_score, box_xywhn, phrase in ranked:
            # find matching DINO score
            dino_score = 0.0
            for s, b, p in dino_dets:
                if p == phrase and b == box_xywhn:
                    dino_score = float(s)
                    break

            cfg = _ensure_class_cfg(class_thresholds, phrase)
            alpha = float(cfg.get("alpha", FUSION_DEFAULTS["alpha"]))
            T_clip = float(cfg.get("T_clip", FUSION_DEFAULTS["T_clip"]))
            T_dino = float(cfg.get("T_dino", FUSION_DEFAULTS["T_dino"]))
            thr = float(cfg.get("thr", FUSION_DEFAULTS["thr"]))

            clip_cal = _calibrate_prob(clip_score, T_clip)
            dino_cal = _calibrate_prob(dino_score, T_dino)
            fused = alpha * clip_cal + (1.0 - alpha) * dino_cal

            if fused >= thr:
                # build pixel xyxy for NMS & filters
                xyxy = xywhn_to_xyxy_pix(box_xywhn, W, H)
                cls_idx = class_names.index(phrase)
                cand.append((xyxy, fused, cls_idx, phrase, clip_cal, dino_cal))

        if not cand:
            return set()

        # Class-aware NMS
        xyxy_list = [c[0] for c in cand]
        score_list = [c[1] for c in cand]
        label_list = [c[2] for c in cand]
        keep_idx = classwise_nms(xyxy_list, score_list, label_list, iou_thresh=nms_iou)

        # Size and aspect-ratio sanity filters
        final_xywhn, final_phrases = [], []
        for i in keep_idx:
            xyxy, fused, cls_idx, phrase, _, _ = cand[i]
            cfg = _ensure_class_cfg(class_thresholds, phrase)
            min_area_frac = float(cfg.get("min_area_frac", FUSION_DEFAULTS["min_area_frac"]))
            max_ar = float(cfg.get("max_ar", FUSION_DEFAULTS["max_ar"]))

            if box_area_frac(xyxy, W, H) < min_area_frac:
                continue
            if box_ar(xyxy) > max_ar:
                continue

            # back to normalized xywh for YOLO txt
            xywhn = xyxy_pix_to_xywhn(xyxy, W, H)
            # clamp to [0,1]
            xywhn = [min(max(v, 0.0), 1.0) for v in xywhn]
            final_xywhn.append(xywhn)
            final_phrases.append(phrase)

        if final_xywhn:
            write_to_disk(image_path.with_suffix('.txt'), final_xywhn, final_phrases, class_names, overwrite)

        # Track which classes were detected for threshold adaptation
        return set(final_phrases)

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
