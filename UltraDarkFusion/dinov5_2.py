import os
import json
import math
import hashlib
import logging
import shutil
import gc
import time
from pathlib import Path

import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFile
from ultralytics import YOLO, YOLOE, SAM
import cv2
import numpy as np
import groundingdino
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_image, predict
from darkfusion_model_compat import load_groundingdino_model as load_model
from prediction_size_filter import prediction_size_allowed_xyxy

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------
# Constants / Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CUDA_DEVICE_STR = "cuda:0" if DEVICE.type == "cuda" else "cpu"

SAM_DIR = BASE_DIR / "Sam"
SAM_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = BASE_DIR / "hf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_FILE = BASE_DIR / "class_thresholds.json"

YOLOE_MODEL_NAME = os.getenv("YOLOE_MODEL_NAME", "yoloe-26s-seg.pt")
YOLOE_FP16 = os.getenv("YOLOE_FP16", "1").strip().lower() in {"1", "true", "yes", "y"}
YOLOE_ONNX_BACKEND = os.getenv("YOLOE_ONNX_BACKEND", "auto").strip().lower()
YOLOE_ONNX_PROVIDER = os.getenv("YOLOE_ONNX_PROVIDER", "auto").strip().lower()
DINO_FP16 = os.getenv("DINO_FP16", "1").strip().lower() in {"1", "true", "yes", "y"}
DINO_FP16_FALLBACK = os.getenv("DINO_FP16_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "y"}

USE_SAM3 = os.getenv("USE_SAM3", "1").strip().lower() in {"1", "true", "yes", "y"}
SAM3_MODEL_NAME = os.getenv("SAM3_MODEL_NAME", "sam3.pt")
SAM3_FP16 = os.getenv("SAM3_FP16", "1").strip().lower() in {"1", "true", "yes", "y"}
SAM3_IMGSZ = int(os.getenv("SAM3_IMGSZ", "1036"))
SAM3_PAD_RATIO = float(os.getenv("SAM3_PAD_RATIO", "0.05"))
SAM3_REJECT_INVALID = os.getenv("SAM3_REJECT_INVALID", "1").strip().lower() in {"1", "true", "yes", "y"}
SAM3_MIN_MASK_AREA = int(os.getenv("SAM3_MIN_MASK_AREA", "20"))
SAM3_MAX_AREA_MULT = float(os.getenv("SAM3_MAX_AREA_MULT", "3.0"))
SAM3_SAVE_SEGMENTS = os.getenv("SAM3_SAVE_SEGMENTS", "0").strip().lower() in {"1", "true", "yes", "y"}
PREDICT_IMGSZ = int(os.getenv("PREDICT_IMGSZ", "640"))
PREDICTION_MIN_SIZE_PX = float(os.getenv("PREDICTION_MIN_SIZE_PX", "0"))
PREDICTION_MAX_PERCENT = float(os.getenv("PREDICTION_MAX_PERCENT", "1"))
IGNORE_TEAMMATES = os.getenv("IGNORE_TEAMMATES", "0").strip().lower() in {"1", "true", "yes", "y"}
TEAMMATE_THRESHOLD = float(os.getenv("TEAMMATE_THRESHOLD", "0.90"))
_TEAMMATE_CLASSIFIER = None

YOLOE_CONF = float(os.getenv("YOLOE_CONF", "0.15"))
YOLOE_IOU = float(os.getenv("YOLOE_IOU", "0.45"))
MAX_DET = int(os.getenv("MAX_DET", "200"))

TEXT_THRESHOLD = float(os.getenv("TEXT_THRESHOLD", "0.35"))
BBOX_THRESHOLD = float(os.getenv("BBOX_THRESHOLD", "0.18"))

DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.45"))
MIN_THRESHOLD = float(os.getenv("MIN_THRESHOLD", "0.25"))
MAX_THRESHOLD = float(os.getenv("MAX_THRESHOLD", "0.65"))

DEFAULT_ALPHA_YOLOE = float(os.getenv("ALPHA_YOLOE", "0.35"))
DEFAULT_ALPHA_DINO = float(os.getenv("ALPHA_DINO", "0.65"))

DEFAULT_T_YOLOE = float(os.getenv("T_YOLOE", "1.00"))
DEFAULT_T_DINO = float(os.getenv("T_DINO", "1.05"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
REVIEW_MARGIN = float(os.getenv("REVIEW_MARGIN", "0.06"))
SAVE_REVIEW_PREVIEWS = os.getenv("SAVE_REVIEW_PREVIEWS", "1").strip().lower() in {"1", "true", "yes", "y"}

CROSS_CLASS_DEDUPE_IOU = float(os.getenv("CROSS_CLASS_DEDUPE_IOU", "0.80"))
PAIR_MERGE_IOU = float(os.getenv("PAIR_MERGE_IOU", "0.55"))

EMPTY_CACHE_EVERY_IMAGE = os.getenv("EMPTY_CACHE_EVERY_IMAGE", "1").strip().lower() in {"1", "true", "yes", "y"}
GC_EVERY_IMAGE = os.getenv("GC_EVERY_IMAGE", "1").strip().lower() in {"1", "true", "yes", "y"}
CPU_COOLDOWN_MS = int(os.getenv("CPU_COOLDOWN_MS", "0"))
VERIFY_IMAGES_FIRST = os.getenv("VERIFY_IMAGES_FIRST", "1").strip().lower() in {"1", "true", "yes", "y"}

FUSION_DEFAULTS = {
    "thr": DEFAULT_THRESHOLD,
    "alpha_yoloe": DEFAULT_ALPHA_YOLOE,
    "alpha_dino": DEFAULT_ALPHA_DINO,
    "T_yoloe": DEFAULT_T_YOLOE,
    "T_dino": DEFAULT_T_DINO,
    "min_area_frac": 0.0005,
    "max_ar": 8.0,
    # Fusion confidence policy. Agreement is trusted more than single-model hits.
    "single_source_mult": 0.70,
    "both_source_bonus": 0.08,
    "min_single_source_score": 0.50,
    "require_both_under": 0.40,
    "min_yoloe_score": 0.05,
    "min_dino_score": 0.05,
    "sam3_min_area": SAM3_MIN_MASK_AREA,
    "sam3_max_area_mult": SAM3_MAX_AREA_MULT,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None


# -------------------------
# Utility / stability
# -------------------------
def cleanup_memory():
    if GC_EVERY_IMAGE:
        gc.collect()
    if DEVICE.type == "cuda" and EMPTY_CACHE_EVERY_IMAGE:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def maybe_cooldown():
    if CPU_COOLDOWN_MS > 0:
        time.sleep(CPU_COOLDOWN_MS / 1000.0)


def ultralytics_fp16_kwargs(enabled):
    """Use Ultralytics' current FP16 predict argument."""
    return {"quantize": 16} if bool(enabled) else {}


def safe_open_image(image_path: Path):
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
    return w, h


def verify_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as im:
            im.verify()
        with Image.open(image_path) as im:
            im.convert("RGB")
        return True
    except Exception as e:
        logger.warning("Skipping invalid image %s: %s", image_path, e)
        return False


# -------------------------
# Math helpers
# -------------------------
def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _calibrate_prob(p: float, T: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    z = math.log(p / (1.0 - p))
    return _sigmoid(z / max(float(T), 1e-6))


def _ensure_class_cfg(class_thresholds: dict, cls: str) -> dict:
    if cls not in class_thresholds or not isinstance(class_thresholds[cls], dict):
        class_thresholds[cls] = dict(FUSION_DEFAULTS)
    # Preserve tuned values from settings created before the YOLOE upgrade.
    legacy_keys = {
        "alpha_world": "alpha_yoloe",
        "T_world": "T_yoloe",
        "min_world_score": "min_yoloe_score",
    }
    for old_key, new_key in legacy_keys.items():
        if new_key not in class_thresholds[cls] and old_key in class_thresholds[cls]:
            class_thresholds[cls][new_key] = class_thresholds[cls][old_key]
        class_thresholds[cls].pop(old_key, None)
    for k, v in FUSION_DEFAULTS.items():
        class_thresholds[cls].setdefault(k, v)
    return class_thresholds[cls]


# -------------------------
# Box helpers
# -------------------------
def xywhn_to_xyxy_pix(box_xywhn, W, H):
    xc, yc, w, h = box_xywhn
    x1 = (xc - w / 2.0) * W
    y1 = (yc - h / 2.0) * H
    x2 = (xc + w / 2.0) * W
    y2 = (yc + h / 2.0) * H
    return [max(0.0, x1), max(0.0, y1), min(W - 1.0, x2), min(H - 1.0, y2)]


def xyxy_pix_to_xywhn(xyxy, W, H):
    x1, y1, x2, y2 = xyxy
    w = max(0.0, (x2 - x1) / W)
    h = max(0.0, (y2 - y1) / H)
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
    return max(w / h, h / w)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


def merge_boxes(a, b):
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def classwise_nms(xyxy_list, score_list, label_list, iou_thresh=0.55):
    if not xyxy_list:
        return []

    xyxy = torch.tensor(xyxy_list, dtype=torch.float32)
    scores = torch.tensor(score_list, dtype=torch.float32)
    labels = torch.tensor(label_list, dtype=torch.int64)

    keep = []

    if tv_nms is None:
        for c in labels.unique().tolist():
            idx = (labels == c).nonzero(as_tuple=True)[0].tolist()
            idx_sorted = sorted(idx, key=lambda i: float(scores[i]), reverse=True)

            while idx_sorted:
                i = idx_sorted.pop(0)
                keep.append(i)
                remain = []
                for j in idx_sorted:
                    if box_iou(xyxy[i].tolist(), xyxy[j].tolist()) <= iou_thresh:
                        remain.append(j)
                idx_sorted = remain
    else:
        for c in labels.unique().tolist():
            idx = (labels == c).nonzero(as_tuple=True)[0]
            k = tv_nms(xyxy[idx], scores[idx], iou_thresh)
            keep += idx[k].tolist()

    return sorted(set(keep))


def cross_class_dedupe(candidates, iou_thresh=0.80):
    if not candidates:
        return []

    order = sorted(
        range(len(candidates)),
        key=lambda i: candidates[i]["fused"],
        reverse=True
    )

    keep = []

    for i in order:
        should_drop = False

        for kept_idx in keep:
            if box_iou(candidates[i]["xyxy"], candidates[kept_idx]["xyxy"]) >= iou_thresh:
                should_drop = True
                break

        if not should_drop:
            keep.append(i)

    return [candidates[i] for i in keep]


def _normalize_class_names(class_names):
    """
    Normalize class names while preserving incoming order.
    This matters because output YOLO class IDs follow this list index.
    """
    normalized = []
    seen = set()

    for name in class_names or []:
        clean = str(name).strip().lower()

        if not clean or clean in seen:
            continue

        normalized.append(clean)
        seen.add(clean)

    return normalized

# -------------------------
# Threshold file
# -------------------------
def load_thresholds():
    if THRESHOLD_FILE.exists():
        try:
            with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("Could not load threshold file %s: %s", THRESHOLD_FILE, e)
            return {}
    return {}


def save_thresholds(thresholds: dict):
    tmp = THRESHOLD_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=4)
    os.replace(tmp, THRESHOLD_FILE)


# -------------------------
# Disk I/O
# -------------------------
def write_to_disk(label_path: Path, boxes_xywhn, class_ids, overwrite):
    tmp_path = Path(str(label_path) + ".tmp")

    if overwrite or not label_path.exists():
        existing_items = []
    else:
        existing_items = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cid = int(parts[0])
                    box = list(map(float, parts[1:]))
                    existing_items.append((cid, box))
                except Exception:
                    continue

    new_items = list(zip(class_ids, boxes_xywhn))

    def same_box(a, b, tol=0.02):
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    merged = list(existing_items)

    for new_cid, new_box in new_items:
        replaced = False
        for i, (old_cid, old_box) in enumerate(merged):
            if old_cid == new_cid and same_box(old_box, new_box):
                merged[i] = (new_cid, new_box)
                replaced = True
                break
        if not replaced:
            merged.append((new_cid, new_box))

    merged.sort(key=lambda x: (x[0], x[1][0], x[1][1]))

    with open(tmp_path, "w", encoding="utf-8") as f:
        for cid, b in merged:
            f.write(f"{cid} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

    os.replace(tmp_path, label_path)


def save_review_preview(image_path: Path, out_dir: Path, xyxy_list, scores, labels, class_names):
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path).convert("RGB") as im:
        draw = ImageDraw.Draw(im)
        for (x1, y1, x2, y2), s, c in zip(xyxy_list, scores, labels):
            draw.rectangle([x1, y1, x2, y2], width=2)
            name = class_names[c] if 0 <= c < len(class_names) else str(c)
            draw.text((x1 + 2, y1 + 2), f"{name} {s:.2f}", fill=(255, 255, 255))
        im.save(out_dir / image_path.name)


# -------------------------
# YOLOE model prep
# -------------------------
def ensure_base_yoloe_pt(model_name: str) -> Path:
    local_pt = SAM_DIR / model_name
    if local_pt.exists():
        return local_pt

    logger.info("Base model not found locally. Downloading with Ultralytics: %s", model_name)
    model = YOLOE(model_name)

    src = None

    try:
        model_file = getattr(model.model, "pt_path", None)
        if model_file:
            p = Path(model_file)
            if p.exists():
                src = p
    except Exception:
        pass

    if src is None:
        try:
            p = Path(model.ckpt_path)
            if p.exists():
                src = p
        except Exception:
            pass

    if src is not None and src.exists():
        shutil.copy2(src, local_pt)
        logger.info("Copied base model to: %s", local_pt)
        return local_pt

    logger.warning("Could not locate downloaded .pt path. Falling back to model name.")
    return Path(model_name)


def _compiled_yoloe_class_names(model):
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        try:
            return [str(names[key]).strip().lower() for key in sorted(names, key=lambda key: int(key))]
        except (TypeError, ValueError):
            return [str(value).strip().lower() for value in names.values()]
    if isinstance(names, (list, tuple)):
        return [str(value).strip().lower() for value in names]
    return []


def _validate_compiled_yoloe_classes(model, class_names, model_path):
    embedded = _compiled_yoloe_class_names(model)
    expected = _normalize_class_names(class_names)
    if embedded and embedded != expected:
        raise ValueError(
            f"Compiled YOLOE model classes {embedded} do not match the selected DarkFusion "
            f"classes {expected}: {model_path}. Export the YOLOE model after setting this "
            "exact class list, or select the .pt model for live text prompts."
        )


def build_yoloe_predictor(class_names):
    class_names = _normalize_class_names(class_names)

    requested_model = str(YOLOE_MODEL_NAME or "yoloe-26s-seg.pt").strip()
    requested_path = Path(requested_model).expanduser()
    local_path = SAM_DIR / requested_model

    if requested_path.exists():
        model_path = requested_path
    elif local_path.exists():
        model_path = local_path
    elif requested_path.suffix.lower() == ".pt" or not requested_path.suffix:
        model_path = ensure_base_yoloe_pt(requested_model)
    else:
        model_path = requested_path

    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx" and YOLOE_ONNX_BACKEND != "ultralytics":
        from darkfusion_onnx_runtime import DarkFusionOnnxModel

        cache_key = hashlib.sha1(str(Path(model_path).resolve()).encode("utf-8")).hexdigest()[:12]
        cache_dir = BASE_DIR / ".darkfusion_cache" / "onnx_runtime" / cache_key
        logger.info(
            "Loading compiled YOLOE ONNX model with DarkFusion backend (%s): %s",
            YOLOE_ONNX_PROVIDER,
            model_path,
        )
        predictor = DarkFusionOnnxModel(
            str(model_path),
            providers=YOLOE_ONNX_PROVIDER,
            strict_provider=True,
            include_cpu_fallback=True,
            fp16=bool(YOLOE_FP16),
            cache_dir=cache_dir,
        )
        _validate_compiled_yoloe_classes(predictor, class_names, model_path)
        return predictor, "onnxruntime"

    if suffix in {".engine", ".onnx"}:
        logger.info("Loading prebuilt YOLOE runtime model with Ultralytics: %s", model_path)
        predictor = YOLO(str(model_path))
        return predictor, "engine" if suffix == ".engine" else "onnx"

    logger.info("Loading YOLOE PyTorch model: %s", model_path)
    predictor = YOLOE(str(model_path))
    predictor.set_classes(class_names)
    return predictor, "pt"


# -------------------------
# Detector inference
# -------------------------
@torch.inference_mode()
def run_yoloe_predict(predictor_model, predictor_kind, image_path: Path, class_names, image_source=None):
    source = image_source if image_source is not None else str(image_path)
    if predictor_kind == "pt":
        results = predictor_model.predict(
            source=source,
            imgsz=PREDICT_IMGSZ,
            conf=YOLOE_CONF,
            iou=YOLOE_IOU,
            max_det=MAX_DET,
            device=CUDA_DEVICE_STR,
            verbose=False,
            **ultralytics_fp16_kwargs(YOLOE_FP16 and DEVICE.type == "cuda"),
        )
    else:
        results = predictor_model.predict(
            source=source,
            imgsz=PREDICT_IMGSZ,
            conf=YOLOE_CONF,
            iou=YOLOE_IOU,
            max_det=MAX_DET,
            device=0 if DEVICE.type == "cuda" else "cpu",
            verbose=False,
            **ultralytics_fp16_kwargs(YOLOE_FP16 and DEVICE.type == "cuda"),
        )

    if not results:
        return []

    r = results[0]
    if predictor_kind != "pt":
        compiled_path = getattr(predictor_model, "model_path", None) or getattr(
            predictor_model, "ckpt_path", "compiled YOLOE model"
        )
        _validate_compiled_yoloe_classes(r, class_names, compiled_path)
    if r.boxes is None or len(r.boxes) == 0:
        return []

    xyxy = r.boxes.xyxy.detach().cpu().tolist()
    confs = r.boxes.conf.detach().cpu().tolist()
    classes = r.boxes.cls.detach().cpu().tolist()

    out = []
    for b, s, c in zip(xyxy, confs, classes):
        cls_idx = int(c)
        if 0 <= cls_idx < len(class_names):
            out.append({
                "xyxy": [float(v) for v in b],
                "score": float(s),
                "cls_idx": cls_idx,
                "source": "yoloe"
            })
    return out


@torch.inference_mode()
def dino_detect_one_class(model, image_tensor, cls_name, box_thresh, text_thresh):
    prompt = f"{cls_name}."

    use_fp16 = bool(DINO_FP16 and DEVICE.type == "cuda")

    def _predict_once(autocast_enabled: bool):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(autocast_enabled and DEVICE.type == "cuda"),
        ):
            return predict(
                model=model,
                image=image_tensor,
                caption=prompt,
                box_threshold=box_thresh,
                text_threshold=text_thresh,
                device=DEVICE
            )

    try:
        boxes, logits, _phrases = _predict_once(use_fp16)
    except Exception as e:
        if not (use_fp16 and DINO_FP16_FALLBACK):
            raise
        logger.warning("DINO FP16 failed for class '%s'; retrying FP32. Reason: %s", cls_name, e)
        boxes, logits, _phrases = _predict_once(False)

    out = []
    for b, s in zip(boxes, logits):
        out.append((float(s.item()), list(map(float, b.tolist()))))
    return out


@torch.inference_mode()
def run_dino_predict(dino_model, image_path: Path, class_names, image_source=None):
    if image_source is None:
        W, H = safe_open_image(image_path)
        _, image_tensor = load_image(str(image_path))
    else:
        image_pil = image_source.convert("RGB")
        W, H = image_pil.size
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.to(DEVICE, non_blocking=True)

    out = []

    for cls_idx, cls_name in enumerate(class_names):
        dino = dino_detect_one_class(
            model=dino_model,
            image_tensor=image_tensor,
            cls_name=cls_name,
            box_thresh=BBOX_THRESHOLD,
            text_thresh=TEXT_THRESHOLD
        )
        for dino_score, box_xywhn in dino:
            xyxy = xywhn_to_xyxy_pix(box_xywhn, W, H)
            out.append({
                "xyxy": [float(v) for v in xyxy],
                "score": float(dino_score),
                "cls_idx": cls_idx,
                "source": "dino"
            })

    try:
        del image_tensor
    except Exception:
        pass

    return out



# -------------------------
# SAM3 refinement
# -------------------------
def build_sam3_predictor():
    if not USE_SAM3:
        logger.info("SAM3 refinement disabled.")
        return None

    sam_path = SAM_DIR / SAM3_MODEL_NAME
    if not sam_path.exists():
        logger.warning("SAM3 model not found at %s. Continuing without SAM3.", sam_path)
        return None

    try:
        model = SAM(str(sam_path))
        logger.info(
            "Loaded SAM3: %s | imgsz=%d | fp16=%s",
            sam_path,
            SAM3_IMGSZ,
            bool(SAM3_FP16 and DEVICE.type == "cuda"),
        )
        return model
    except Exception as e:
        logger.warning("Could not load SAM3 from %s: %s", sam_path, e)
        return None


def _pad_bbox_xyxy(bbox_xyxy, img_w, img_h, pad_ratio=0.05):
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    if x2 <= x1 or y2 <= y1:
        return [x1, y1, x2, y2]

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * float(pad_ratio)))
    py = int(round(bh * float(pad_ratio)))

    return [
        max(0, x1 - px),
        max(0, y1 - py),
        min(img_w - 1, x2 + px),
        min(img_h - 1, y2 + py),
    ]


def _image_for_sam3(image_path: Path):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        return None
    if image_bgr.ndim == 2:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGB)
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_bgr

    return np.ascontiguousarray(image_rgb.copy())


@torch.inference_mode()
def sam3_snap_bbox_to_mask(sam3_model, image_rgb, bbox_xyxy, class_name="", imgsz=1036):
    if sam3_model is None or image_rgb is None:
        return None

    image_rgb = np.ascontiguousarray(image_rgb.copy())

    img_h, img_w = image_rgb.shape[:2]

    img_h, img_w = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    try:
        results = sam3_model.predict(
            source=image_rgb,
            bboxes=[[x1, y1, x2, y2]],
            imgsz=int(imgsz),
            device=0 if DEVICE.type == "cuda" else "cpu",
            verbose=False,
            **ultralytics_fp16_kwargs(SAM3_FP16 and DEVICE.type == "cuda"),
        )

        if not results:
            return None

        result = results[0]
        if not hasattr(result, "masks") or result.masks is None:
            return None

        masks = result.masks.data
        if masks is None or len(masks) == 0:
            return None

        mask = masks[0]
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()

        mask = np.asarray(mask)
        if mask.ndim > 2:
            mask = mask.squeeze()

        mask = (mask > 0.5).astype(np.uint8)

        if mask.shape[:2] != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        if int(np.count_nonzero(mask)) <= 0:
            return None

        return mask

    except Exception as e:
        logger.warning("SAM3 bbox refinement failed: %s", e)
        return None


def validate_sam3_mask_for_bbox(mask, original_bbox_xyxy, class_name="", cfg=None):
    if mask is None:
        return False

    cfg = cfg or {}
    try:
        mask_area = int(np.count_nonzero(mask))
    except Exception:
        return False

    min_area = int(cfg.get("sam3_min_area", SAM3_MIN_MASK_AREA))
    if mask_area < min_area:
        return False

    x1, y1, x2, y2 = [int(round(float(v))) for v in original_bbox_xyxy]
    bbox_area = max(1, abs(x2 - x1) * abs(y2 - y1))

    class_name = (class_name or "").strip().lower()
    max_area_mult = float(cfg.get("sam3_max_area_mult", SAM3_MAX_AREA_MULT))
    if class_name in {"person", "human", "player"}:
        max_area_mult = min(max_area_mult, 2.5)
    elif class_name in {"skateboard", "surfboard", "snowboard", "gun", "rifle", "knife", "tool"}:
        max_area_mult = min(max_area_mult, 1.8)
    elif class_name in {"car", "truck", "bus", "train"}:
        max_area_mult = max(max_area_mult, 3.5)

    if mask_area > bbox_area * max_area_mult:
        logger.debug(
            "SAM3 mask rejected for '%s': mask too large (%d > %.2fx %d).",
            class_name, mask_area, max_area_mult, bbox_area,
        )
        return False

    crop = mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return False

    inside_area = int(np.count_nonzero(crop))
    inside_ratio = inside_area / max(1, mask_area)

    if class_name in {"person", "human", "player"}:
        min_inside_ratio = 0.45
    elif class_name in {"skateboard", "surfboard", "snowboard", "gun", "rifle", "knife", "tool"}:
        min_inside_ratio = 0.60
    else:
        min_inside_ratio = 0.50

    if inside_ratio < min_inside_ratio:
        logger.debug(
            "SAM3 mask rejected for '%s': inside_ratio %.2f < %.2f.",
            class_name, inside_ratio, min_inside_ratio,
        )
        return False

    return True


def mask_to_xyxy(mask):
    if mask is None:
        return None

    mask = (np.asarray(mask) > 0).astype(np.uint8)
    points = cv2.findNonZero(mask)
    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)
    if w <= 1 or h <= 1:
        return None

    return [float(x), float(y), float(x + w), float(y + h)]


def refine_candidates_with_sam3(sam3_model, image_path: Path, candidates, class_names, class_thresholds, W, H):
    if sam3_model is None or not candidates:
        return candidates

    image_rgb = _image_for_sam3(image_path)
    if image_rgb is None:
        logger.warning("Could not read image for SAM3 refinement: %s", image_path)
        return candidates

    refined = []

    for c in candidates:
        cls_idx = int(c["cls_idx"])
        cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else ""
        cfg = _ensure_class_cfg(class_thresholds, cls_name)

        original_xyxy = c["xyxy"]
        prompt_xyxy = _pad_bbox_xyxy(original_xyxy, W, H, pad_ratio=SAM3_PAD_RATIO)

        mask = sam3_snap_bbox_to_mask(
            sam3_model=sam3_model,
            image_rgb=image_rgb,
            bbox_xyxy=prompt_xyxy,
            imgsz=SAM3_IMGSZ,
        )

        if not validate_sam3_mask_for_bbox(mask, original_xyxy, class_name=cls_name, cfg=cfg):
            if SAM3_REJECT_INVALID:
                continue
            refined.append(c)
            continue

        snapped_xyxy = mask_to_xyxy(mask)
        if snapped_xyxy is None:
            if SAM3_REJECT_INVALID:
                continue
            refined.append(c)
            continue

        if not _passes_geometry_filters(snapped_xyxy, cfg, W, H):
            if SAM3_REJECT_INVALID:
                continue
            refined.append(c)
            continue

        new_c = dict(c)
        new_c["xyxy"] = snapped_xyxy
        new_c["source"] = f'{c.get("source", "unknown")}+sam3'
        new_c["sam3_refined"] = True
        refined.append(new_c)

    return refined


# -------------------------
# Fusion
# -------------------------
def _passes_geometry_filters(xyxy, cfg, W, H):
    if not prediction_size_allowed_xyxy(
        xyxy,
        W,
        H,
        min_size_px=PREDICTION_MIN_SIZE_PX,
        max_percent=PREDICTION_MAX_PERCENT,
    ):
        return False
    if box_area_frac(xyxy, W, H) < float(cfg["min_area_frac"]):
        return False
    if box_ar(xyxy) > float(cfg["max_ar"]):
        return False
    return True


def _apply_single_source_policy(score, raw_score, cfg, source):
    """
    Penalize detections seen by only one detector.
    This is the main false-positive guard: single-source detections must be strong.
    """
    min_raw_key = "min_yoloe_score" if source == "yoloe" else "min_dino_score"
    if float(raw_score) < float(cfg.get(min_raw_key, 0.0)):
        return None

    adjusted = float(score) * float(cfg.get("single_source_mult", 0.70))

    if adjusted < float(cfg.get("min_single_source_score", 0.50)):
        return None

    return adjusted


def merge_same_class_candidates(yoloe_preds, dino_preds, class_names, class_thresholds, W, H):
    by_class_yoloe = {}
    by_class_dino = {}

    for p in yoloe_preds:
        by_class_yoloe.setdefault(p["cls_idx"], []).append(p)

    for p in dino_preds:
        by_class_dino.setdefault(p["cls_idx"], []).append(p)

    candidates = []
    borderline = []

    for cls_idx, cls_name in enumerate(class_names):
        cfg = _ensure_class_cfg(class_thresholds, cls_name)
        thr = float(cfg["thr"])
        alpha_yoloe = float(cfg["alpha_yoloe"])
        alpha_dino = float(cfg["alpha_dino"])
        T_yoloe = float(cfg["T_yoloe"])
        T_dino = float(cfg["T_dino"])
        require_both_under = float(cfg.get("require_both_under", 0.40))

        yoloe_detections = by_class_yoloe.get(cls_idx, [])
        dinos = by_class_dino.get(cls_idx, [])

        used_dino = set()

        for yoloe_detection in yoloe_detections:
            best_j = -1
            best_iou = 0.0

            for j, d in enumerate(dinos):
                if j in used_dino:
                    continue
                iou = box_iou(yoloe_detection["xyxy"], d["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            yoloe_cal = _calibrate_prob(yoloe_detection["score"], T_yoloe)

            if best_j >= 0 and best_iou >= PAIR_MERGE_IOU:
                d = dinos[best_j]
                used_dino.add(best_j)
                dino_cal = _calibrate_prob(d["score"], T_dino)

                fused = alpha_yoloe * yoloe_cal + alpha_dino * dino_cal
                fused = min(1.0, fused + float(cfg.get("both_source_bonus", 0.08)))

                merged_xyxy = merge_boxes(yoloe_detection["xyxy"], d["xyxy"])
                source = "both"
            else:
                dino_cal = 0.0
                fused = _apply_single_source_policy(
                    score=yoloe_cal,
                    raw_score=yoloe_detection["score"],
                    cfg=cfg,
                    source="yoloe",
                )
                if fused is None:
                    continue

                merged_xyxy = yoloe_detection["xyxy"]
                source = "yoloe"

            if not _passes_geometry_filters(merged_xyxy, cfg, W, H):
                continue

            item = {
                "xyxy": merged_xyxy,
                "fused": fused,
                "cls_idx": cls_idx,
                "yoloe_cal": yoloe_cal,
                "dino_cal": dino_cal,
                "source": source,
            }

            # Below this line, require detector agreement unless the single-source
            # detection was strong enough to pass _apply_single_source_policy().
            if source != "both" and fused < require_both_under:
                borderline.append(item)
            elif fused >= thr:
                candidates.append(item)
            elif abs(fused - thr) <= REVIEW_MARGIN:
                borderline.append(item)

        for j, d in enumerate(dinos):
            if j in used_dino:
                continue

            dino_cal = _calibrate_prob(d["score"], T_dino)
            fused = _apply_single_source_policy(
                score=dino_cal,
                raw_score=d["score"],
                cfg=cfg,
                source="dino",
            )
            if fused is None:
                continue

            merged_xyxy = d["xyxy"]

            if not _passes_geometry_filters(merged_xyxy, cfg, W, H):
                continue

            item = {
                "xyxy": merged_xyxy,
                "fused": fused,
                "cls_idx": cls_idx,
                "yoloe_cal": 0.0,
                "dino_cal": dino_cal,
                "source": "dino",
            }

            if fused < require_both_under:
                borderline.append(item)
            elif fused >= thr:
                candidates.append(item)
            elif abs(fused - thr) <= REVIEW_MARGIN:
                borderline.append(item)

    return candidates, borderline


# -------------------------
# Main image processing
# -------------------------
def filter_teammate_candidates(image_path, candidates, class_names, width, height):
    """Remove likely friendly-player candidates before DINO/YOLOE labels are saved."""
    global _TEAMMATE_CLASSIFIER
    if not IGNORE_TEAMMATES or not candidates:
        return candidates
    try:
        from darkfusion_teammate_review import TeammateMarkerClassifier

        if _TEAMMATE_CLASSIFIER is None:
            _TEAMMATE_CLASSIFIER = TeammateMarkerClassifier(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            bounds = []
            candidate_indices = []
            for index, candidate in enumerate(candidates):
                class_id = int(candidate.get("cls_idx", -1))
                class_name = class_names[class_id].lower() if 0 <= class_id < len(class_names) else ""
                if class_name and not any(
                    token in class_name for token in ("person", "player", "enemy", "character")
                ):
                    continue
                x1, y1, x2, y2 = [float(value) for value in candidate.get("xyxy", [])[:4]]
                bounds.append([
                    max(0.0, min(1.0, x1 / max(1, width))),
                    max(0.0, min(1.0, y1 / max(1, height))),
                    max(0.0, min(1.0, x2 / max(1, width))),
                    max(0.0, min(1.0, y2 / max(1, height))),
                ])
                candidate_indices.append(index)
        local_rejected = _TEAMMATE_CLASSIFIER.friendly_indices(
            [image],
            [bounds],
            threshold=max(0.50, min(0.999, float(TEAMMATE_THRESHOLD))),
            batch_size=96,
            ocr=True,
        )[0]
        rejected = {
            candidate_indices[local_index]
            for local_index in local_rejected
            if 0 <= int(local_index) < len(candidate_indices)
        }
        if _TEAMMATE_CLASSIFIER.last_ocr_error:
            logger.warning(
                "Teammate OCR unavailable for %s; visual marker filtering remains active: %s",
                image_path,
                _TEAMMATE_CLASSIFIER.last_ocr_error,
            )
        if rejected:
            logger.info(
                "Ignored %d teammate prediction(s) in %s at certainty %.3f",
                len(rejected), image_path.name, TEAMMATE_THRESHOLD,
            )
        return [candidate for index, candidate in enumerate(candidates) if index not in rejected]
    except Exception as error:
        logger.warning("Teammate filter unavailable for %s; keeping predictions: %s", image_path, error)
        return candidates


def _scaled_roi_bounds(roi_bounds, roi_reference_size, image_width, image_height):
    """Scale a DarkFusion display ROI into one dataset image's pixel space."""
    if not isinstance(roi_bounds, (list, tuple)) or len(roi_bounds) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in roi_bounds)
        ref_width, ref_height = roi_reference_size or (image_width, image_height)
        if ref_width <= 0 or ref_height <= 0:
            return None
        x *= image_width / float(ref_width)
        width *= image_width / float(ref_width)
        y *= image_height / float(ref_height)
        height *= image_height / float(ref_height)
        x = max(0.0, min(x, float(image_width)))
        y = max(0.0, min(y, float(image_height)))
        width = max(0.0, min(width, float(image_width) - x))
        height = max(0.0, min(height, float(image_height) - y))
        if width <= 0 or height <= 0:
            return None
        return x, y, x + width, y + height
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _scaled_roi_bounds_list(roi_bounds, roi_reference_size, image_width, image_height):
    if (isinstance(roi_bounds, (list, tuple)) and len(roi_bounds) == 4 and
            all(isinstance(value, (int, float)) for value in roi_bounds)):
        source_bounds = [roi_bounds]
    else:
        source_bounds = list(roi_bounds or [])
    return [
        scaled for scaled in (
            _scaled_roi_bounds(bounds, roi_reference_size, image_width, image_height)
            for bounds in source_bounds
        ) if scaled is not None
    ]


def _candidate_intersects_roi(candidate, roi):
    if roi is None:
        return True
    try:
        x1, y1, x2, y2 = (float(value) for value in candidate["xyxy"])
        left, top, right, bottom = roi
        box_left, box_right = min(x1, x2), max(x1, x2)
        box_top, box_bottom = min(y1, y2), max(y1, y2)
        return not (
            box_right < left or box_left > right or
            box_bottom < top or box_top > bottom
        )
    except (KeyError, TypeError, ValueError):
        return False


def _candidate_intersects_any_roi(candidate, rois):
    return not rois or any(_candidate_intersects_roi(candidate, roi) for roi in rois)


def _translate_and_clip_predictions(predictions, offset_x, offset_y, image_width, image_height):
    """Map crop-relative detector boxes into the retained full image."""
    mapped = []
    for prediction in predictions or []:
        try:
            x1, y1, x2, y2 = (float(value) for value in prediction["xyxy"])
            left = max(0.0, min(float(image_width), min(x1, x2) + float(offset_x)))
            top = max(0.0, min(float(image_height), min(y1, y2) + float(offset_y)))
            right = max(0.0, min(float(image_width), max(x1, x2) + float(offset_x)))
            bottom = max(0.0, min(float(image_height), max(y1, y2) + float(offset_y)))
        except (KeyError, TypeError, ValueError):
            continue
        if right <= left or bottom <= top:
            continue
        mapped_prediction = dict(prediction)
        mapped_prediction["xyxy"] = [left, top, right, bottom]
        mapped.append(mapped_prediction)
    return mapped


def process_image(
    image_path: Path,
    yoloe_model,
    yoloe_kind,
    dino_model,
    sam3_model,
    class_names,
    class_thresholds,
    overwrite,
    nms_iou=0.55,
    review_dir=None,
    preview_callback=None,
    roi_enabled=False,
    roi_bounds=None,
    roi_reference_size=None,
):
    detected = set()
    inference_image = None

    try:
        W, H = safe_open_image(image_path)
        rois = (
            _scaled_roi_bounds_list(roi_bounds, roi_reference_size, W, H)
            if roi_enabled else []
        )
        offset_x = 0
        offset_y = 0

        yoloe_preds = run_yoloe_predict(
            yoloe_model, yoloe_kind, image_path, class_names, image_source=inference_image
        )
        dino_preds = run_dino_predict(
            dino_model, image_path, class_names, image_source=inference_image
        )
        yoloe_preds = _translate_and_clip_predictions(yoloe_preds, offset_x, offset_y, W, H)
        dino_preds = _translate_and_clip_predictions(dino_preds, offset_x, offset_y, W, H)

        candidates, borderline = merge_same_class_candidates(
            yoloe_preds=yoloe_preds,
            dino_preds=dino_preds,
            class_names=class_names,
            class_thresholds=class_thresholds,
            W=W,
            H=H,
        )

        candidates = refine_candidates_with_sam3(
            sam3_model=sam3_model,
            image_path=image_path,
            candidates=candidates,
            class_names=class_names,
            class_thresholds=class_thresholds,
            W=W,
            H=H,
        )

        if not candidates:
            if preview_callback is not None and borderline:
                try:
                    preview_callback(
                        image_path,
                        [b["xyxy"] for b in borderline],
                        [b["fused"] for b in borderline],
                        [b["cls_idx"] for b in borderline],
                        class_names,
                    )
                except Exception as e:
                    logger.warning("Preview callback failed for %s: %s", image_path, e)
            if borderline and review_dir and SAVE_REVIEW_PREVIEWS:
                save_review_preview(
                    image_path,
                    Path(review_dir),
                    [b["xyxy"] for b in borderline],
                    [b["fused"] for b in borderline],
                    [b["cls_idx"] for b in borderline],
                    class_names,
                )
            if overwrite:
                write_to_disk(image_path.with_suffix(".txt"), [], [], True)
            return set()

        xyxy_list = [c["xyxy"] for c in candidates]
        score_list = [c["fused"] for c in candidates]
        label_list = [c["cls_idx"] for c in candidates]

        keep = classwise_nms(xyxy_list, score_list, label_list, iou_thresh=nms_iou)
        candidates = [candidates[i] for i in keep]
        candidates = cross_class_dedupe(candidates, iou_thresh=CROSS_CLASS_DEDUPE_IOU)
        candidates = filter_teammate_candidates(image_path, candidates, class_names, W, H)

        # Preview the complete full-image inference result. ROI is only a
        # write filter and must not hide otherwise valid detections in the UI.
        preview_candidates = list(candidates)
        label_candidates = (
            [candidate for candidate in candidates if _candidate_intersects_any_roi(candidate, rois)]
            if rois else candidates
        )

        final_xywhn = []
        final_cls = []

        for c in label_candidates:
            xywhn = xyxy_pix_to_xywhn(c["xyxy"], W, H)
            xywhn = [min(max(v, 0.0), 1.0) for v in xywhn]
            final_xywhn.append(xywhn)
            final_cls.append(c["cls_idx"])
            detected.add(class_names[c["cls_idx"]])

        if final_xywhn or overwrite:
            write_to_disk(image_path.with_suffix(".txt"), final_xywhn, final_cls, overwrite)

        if preview_callback is not None:
            preview_xyxy = [c["xyxy"] for c in preview_candidates]
            preview_scores = [c["fused"] for c in preview_candidates]
            preview_labels = [c["cls_idx"] for c in preview_candidates]
            try:
                preview_callback(image_path, preview_xyxy, preview_scores, preview_labels, class_names)
            except Exception as e:
                logger.warning("Preview callback failed for %s: %s", image_path, e)

        if borderline and review_dir and SAVE_REVIEW_PREVIEWS:
            save_review_preview(
                image_path,
                Path(review_dir),
                [b["xyxy"] for b in borderline],
                [b["fused"] for b in borderline],
                [b["cls_idx"] for b in borderline],
                class_names,
            )

        return detected

    except torch.cuda.OutOfMemoryError as e:
        logger.error("CUDA OOM on %s: %s", image_path, e)
        return set()

    except Exception as e:
        logger.error("Error processing %s: %s", image_path, e)
        return set()

    finally:
        if inference_image is not None:
            inference_image.close()
        cleanup_memory()
        maybe_cooldown()


def adjust_thresholds_simple(detected_classes, class_thresholds):
    changed = False
    for cls in detected_classes:
        cfg = _ensure_class_cfg(class_thresholds, cls)
        new_thr = max(MIN_THRESHOLD, min(MAX_THRESHOLD, float(cfg["thr"]) - 0.01))
        if new_thr != cfg["thr"]:
            cfg["thr"] = new_thr
            class_thresholds[cls] = cfg
            changed = True

    if changed:
        save_thresholds(class_thresholds)


def collect_image_paths(image_directory: Path):
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_paths.extend(image_directory.glob(ext))
    image_paths = sorted(image_paths)

    if VERIFY_IMAGES_FIRST:
        valid = []
        for p in image_paths:
            if verify_image(p):
                valid.append(p)
        image_paths = valid

    return image_paths


def process_images(
    image_directory_path,
    yoloe_model,
    yoloe_kind,
    dino_model,
    sam3_model,
    class_names,
    class_thresholds,
    overwrite,
    review_dir=None,
    progress_callback=None,
    preview_callback=None,
    roi_enabled=False,
    roi_bounds=None,
    roi_reference_size=None,
):
    image_directory = Path(image_directory_path)
    image_paths = collect_image_paths(image_directory)

    if not image_paths:
        logger.warning("No valid images found in %s", image_directory)
        return 0

    detected_classes = set()
    progress = tqdm(total=len(image_paths), desc="Auto-label")
    processed = 0
    if progress_callback is not None:
        try:
            progress_callback(0, len(image_paths))
        except Exception as e:
            logger.warning("Progress callback failed: %s", e)

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[i:i + BATCH_SIZE]

        for p in batch:
            if preview_callback is not None:
                try:
                    preview_callback(p, [], [], [], class_names)
                except Exception as e:
                    logger.warning("Preview callback failed for %s: %s", p, e)

            detected_classes.update(
                process_image(
                    image_path=p,
                    yoloe_model=yoloe_model,
                    yoloe_kind=yoloe_kind,
                    dino_model=dino_model,
                    sam3_model=sam3_model,
                    class_names=class_names,
                    class_thresholds=class_thresholds,
                    overwrite=overwrite,
                    nms_iou=0.55,
                    review_dir=review_dir,
                    preview_callback=preview_callback,
                    roi_enabled=roi_enabled,
                    roi_bounds=roi_bounds,
                    roi_reference_size=roi_reference_size,
                )
            )
            processed += 1
            if progress_callback is not None:
                try:
                    progress_callback(processed, len(image_paths))
                except Exception as e:
                    logger.warning("Progress callback failed: %s", e)
            else:
                progress.update(1)

    progress.close()
    adjust_thresholds_simple(detected_classes, class_thresholds)
    logger.info("Done.")
    return processed


# -------------------------
# Entry points
# -------------------------
def run_groundingdino(
    image_directory_path,
    overwrite,
    review_dir=None,
    class_names=None,
    dino_config_path=None,
    dino_weights_path=None,
    yoloe_model_name=None,
    runtime_overrides=None,
    progress_callback=None,
    preview_callback=None,
    roi_enabled=False,
    roi_bounds=None,
    roi_reference_size=None,
):
    global YOLOE_MODEL_NAME, YOLOE_FP16, YOLOE_ONNX_BACKEND, YOLOE_ONNX_PROVIDER
    global DINO_FP16, DINO_FP16_FALLBACK
    global YOLOE_CONF, YOLOE_IOU, TEXT_THRESHOLD, BBOX_THRESHOLD, BATCH_SIZE, PREDICT_IMGSZ
    global PREDICTION_MIN_SIZE_PX, PREDICTION_MAX_PERCENT, IGNORE_TEAMMATES, TEAMMATE_THRESHOLD
    global _TEAMMATE_CLASSIFIER
    global PAIR_MERGE_IOU, CROSS_CLASS_DEDUPE_IOU, DEFAULT_THRESHOLD

    runtime_overrides = runtime_overrides or {}
    override_keys = [
        "YOLOE_MODEL_NAME",
        "YOLOE_FP16",
        "YOLOE_ONNX_BACKEND",
        "YOLOE_ONNX_PROVIDER",
        "DINO_FP16",
        "DINO_FP16_FALLBACK",
        "YOLOE_CONF",
        "YOLOE_IOU",
        "TEXT_THRESHOLD",
        "BBOX_THRESHOLD",
        "BATCH_SIZE",
        "PREDICT_IMGSZ",
        "PREDICTION_MIN_SIZE_PX",
        "PREDICTION_MAX_PERCENT",
        "IGNORE_TEAMMATES",
        "TEAMMATE_THRESHOLD",
        "PAIR_MERGE_IOU",
        "CROSS_CLASS_DEDUPE_IOU",
        "DEFAULT_THRESHOLD",
    ]
    previous_values = {key: globals().get(key) for key in override_keys}

    def _restore_runtime_overrides():
        for key, value in previous_values.items():
            globals()[key] = value

    for key, value in runtime_overrides.items():
        if key in override_keys:
            globals()[key] = value

    if yoloe_model_name:
        YOLOE_MODEL_NAME = str(yoloe_model_name)

    config_path = dino_config_path or os.path.join(groundingdino.__path__[0], "config", "GroundingDINO_SwinT_OGC.py")
    weights = Path(dino_weights_path) if dino_weights_path else SAM_DIR / "groundingdino_swint_ogc.pth"

    if not weights.exists():
        logger.error("Missing weights: %s", weights)
        _restore_runtime_overrides()
        return False

    # Prefer passed-in class names from DarkFusion dropdown
    if class_names:
        class_names = [str(name).strip().lower() for name in class_names if str(name).strip()]
    else:
        dataset_path = Path(image_directory_path)
        classes_file = dataset_path / ".darkfusion" / "classes.txt"
        if not classes_file.exists():
            # Compatibility with datasets that have not been opened and
            # migrated by DarkFusion yet.
            classes_file = dataset_path / "classes.txt"
        if not classes_file.exists():
            logger.error("Missing classes file: %s", classes_file)
            _restore_runtime_overrides()
            return False

        with open(classes_file, "r", encoding="utf-8") as f:
            class_names = [line.strip().lower() for line in f if line.strip()]

    if not class_names:
        logger.error("No valid class names provided.")
        _restore_runtime_overrides()
        return False

    if review_dir is None:
        review_dir = str(Path(image_directory_path) / "_review_previews")

    logger.info("Loaded %d classes.", len(class_names))
    logger.info("Using device: %s", DEVICE)
    logger.info(
        "BATCH_SIZE=%d | YOLOE_CONF=%.3f | TEXT_THRESHOLD=%.3f | BBOX_THRESHOLD=%.3f | YOLOE_FP16=%s | DINO_FP16=%s",
        BATCH_SIZE, YOLOE_CONF, TEXT_THRESHOLD, BBOX_THRESHOLD, YOLOE_FP16, DINO_FP16
    )

    yoloe_model = None
    dino_model = None
    sam3_model = None
    try:
        yoloe_model, yoloe_kind = build_yoloe_predictor(class_names)
        logger.info("YOLOE predictor mode: %s", yoloe_kind)

        dino_model = load_model(config_path, str(weights), device=DEVICE)
        if hasattr(dino_model, "eval"):
            dino_model.eval()

        sam3_model = build_sam3_predictor()

        class_thresholds = load_thresholds()

        processed = process_images(
            image_directory_path=image_directory_path,
            yoloe_model=yoloe_model,
            yoloe_kind=yoloe_kind,
            dino_model=dino_model,
            sam3_model=sam3_model,
            class_names=class_names,
            class_thresholds=class_thresholds,
            overwrite=overwrite,
            review_dir=review_dir,
            progress_callback=progress_callback,
            preview_callback=preview_callback,
            roi_enabled=roi_enabled,
            roi_bounds=roi_bounds,
            roi_reference_size=roi_reference_size,
        )
        return bool(processed)
    finally:
        try:
            del yoloe_model
        except Exception:
            pass
        try:
            del dino_model
        except Exception:
            pass
        try:
            del sam3_model
        except Exception:
            pass
        _TEAMMATE_CLASSIFIER = None
        _restore_runtime_overrides()
        cleanup_memory()

def run_autolabel(image_directory_path, overwrite, review_dir=None, class_names=None):
    return run_groundingdino(
        image_directory_path,
        overwrite,
        review_dir=review_dir,
        class_names=class_names
    )


def main(image_directory):
    overwrite = input("Overwrite existing label files? (yes/no): ").strip().lower() == "yes"
    review_dir = str(Path(image_directory) / "_review_previews")
    run_groundingdino(image_directory, overwrite, review_dir=review_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py <image_directory>")
        raise SystemExit(2)

    main(sys.argv[1])
