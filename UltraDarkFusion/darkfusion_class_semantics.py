"""Conservative class-name verification for Dataset Analysis findings.

The DINOv3 scan finds annotations that look unusual within their saved class.
This module supplies a separate question that DINOv3 cannot answer: whether a
crop still looks like the class name in ``classes.txt``.  Results only suppress
very strong later-pass review candidates; they never edit dataset labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

from PIL import Image
from darkfusion_model_compat import pooled_features
from darkfusion_model_security import install_checkpoint_guard


MODEL_NAME = "google/siglip2-base-patch16-224"
MODEL_REVISION = "02c35f2c035e0ed4a367fb10a892c1fe2a3f364e"
FALLBACK_MODEL_NAME = "openai/clip-vit-base-patch32"
PROMPT_VERSION = "darkfusion-class-semantics-v4-siglip2-calibrated"

# These are deliberately gameplay-specific.  A crop that merely contains a
# human-shaped HUD element, nameplate, shadow, or effect must compete against
# that explanation before it can be protected as a valid class match.
DISTRACTOR_PROMPT_GROUPS = (
    (
        "video game background scenery with no labeled object",
        "terrain, a building, or scenery in a video game",
    ),
    (
        "a video game HUD icon or interface element",
        "text, numbers, a badge, or a nameplate on a game screen",
    ),
    (
        "a map marker, flag, objective icon, or revive icon",
        "a small face portrait or character icon in a game interface",
    ),
    (
        "a shadow, glow, explosion, or visual effect in a video game",
        "a weapon effect, reticle, or damage indicator",
    ),
)


def _article(class_name):
    return "an" if str(class_name).strip().lower()[:1] in "aeiou" else "a"


def class_prompts(class_name):
    """Return several natural descriptions rather than trusting one prompt."""
    name = " ".join(str(class_name or "object").replace("_", " ").split())
    article = _article(name)
    return (
        f"a clear image of {article} {name}",
        f"{article} {name} in a video game",
        f"the labeled object is {article} {name}",
        f"an example of the class {name}",
    )


def should_protect_class_match(result, *, scan_pass, minimum_pass=2):
    """Use only decisive semantic evidence, and never hide first-pass findings."""
    if int(scan_pass or 0) < int(minimum_pass):
        return False
    if not isinstance(result, dict) or result.get("error"):
        return False
    if bool(result.get("teammate_marker_layout", False)):
        return False
    probability = float(result.get("class_probability", 0.0) or 0.0)
    margin = float(result.get("semantic_margin", -1.0) or -1.0)
    similarity = float(result.get("class_similarity", -1.0) or -1.0)
    if result.get("backend") == "siglip2":
        # Calibrated against the current Apex review set. The top 60 later-pass
        # crops were checked both tightly and in their full frames; this stricter
        # three-part rule retained 50 of those player crops. Scores remain
        # relative to our class/distractor prompts rather than absolute truth.
        return probability >= 0.40 and margin >= -0.004 and similarity >= 0.02
    # Preserve the previously calibrated rule when an offline machine falls
    # back to the older CLIP model.
    return probability >= 0.60 and margin >= 0.004 and similarity >= 0.20


def false_positive_evidence(
    result, *, visual_similarity, visual_cutoff, scan_pass
):
    """Rank review candidates where visual and semantic evidence agree.

    The score only controls review order and wording. It never removes a label
    or suppresses a finding. That matters because effects, outlines, poses, and
    partial people can look semantically unusual while still being valid.
    
    Signals:
    - DINOv3: Visual similarity outlier detection (appearance consistency)
    - SigLIP2: Semantic class verification (labeled class makes sense?)
    - SAM3: Shape verification is done separately (shape matches class?)
    """
    try:
        similarity = float(visual_similarity)
        cutoff = float(visual_cutoff)
    except (TypeError, ValueError):
        similarity, cutoff = 1.0, 1.0
    dino_gap = max(0.0, cutoff - similarity)
    dino_anomaly = min(1.0, dino_gap / 0.15)
    
    try:
        pass_strength = 1.0 / max(1, int(scan_pass))
    except (TypeError, ValueError):
        pass_strength = 0.0

    semantic_available = isinstance(result, dict) and not result.get("error")
    semantic_mismatch = 0.0
    if semantic_available:
        probability = float(result.get("class_probability", 0.0) or 0.0)
        reference = 0.40 if result.get("backend") == "siglip2" else 0.60
        semantic_mismatch = max(0.0, min(1.0, (reference - probability) / reference))
    priority = (
        0.55 * semantic_mismatch
        + 0.30 * dino_anomaly
        + 0.15 * pass_strength
    )
    if semantic_available and priority >= 0.70:
        strength = "high"
    elif semantic_available and priority >= 0.55:
        strength = "moderate"
    else:
        strength = "review"
    return {
        "priority": round(priority, 6),
        "strength": strength,
        "semantic_mismatch": round(semantic_mismatch, 6),
        "dino_anomaly": round(dino_anomaly, 6),
    }


class ClassSemanticVerifier:
    """Batch SigLIP 2 inference with a cached CLIP fallback."""

    def __init__(
        self,
        cache_path,
        *,
        device="cuda",
        batch_size=32,
        context=0.08,
        status=None,
        cancelled=None,
    ):
        self.cache_path = os.fspath(cache_path)
        self.requested_device = str(device or "cuda")
        self.batch_size = max(1, int(batch_size or 32))
        self.context = max(0.0, min(0.35, float(context or 0.0)))
        self.status = status or (lambda _message: None)
        self.cancelled = cancelled or (lambda: False)
        self.device = None
        self.model = None
        self.processor = None
        self.model_name = MODEL_NAME
        self.model_revision = MODEL_REVISION
        self.backend = "siglip2"
        self.model_cache = str(Path(self.cache_path).parent / "models")
        self._torch = None
        self._connection = None

    def prepare(self):
        if self.model is not None:
            return self
        self._prepare_cache()
        import torch
        from huggingface_hub import snapshot_download
        install_checkpoint_guard()
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        use_cuda = self.requested_device.lower() != "cpu" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        Path(self.model_cache).mkdir(parents=True, exist_ok=True)
        self.status(f"Loading SigLIP 2 class verifier on {self.device}...")
        try:
            try:
                model_path = snapshot_download(
                    repo_id=MODEL_NAME,
                    revision=MODEL_REVISION,
                    cache_dir=self.model_cache,
                    local_files_only=True,
                )
            except (FileNotFoundError, OSError):
                if self.cancelled():
                    raise RuntimeError("Class semantic verification cancelled")
                self.status(
                    "Downloading SigLIP 2 class verifier (about 1.6 GB, once)..."
                )
                model_path = snapshot_download(
                    repo_id=MODEL_NAME,
                    revision=MODEL_REVISION,
                    cache_dir=self.model_cache,
                    allow_patterns=("*.json", "*.model", "*.safetensors"),
                )
            if self.cancelled():
                raise RuntimeError("Class semantic verification cancelled")
            self.processor = AutoProcessor.from_pretrained(
                model_path, local_files_only=True, use_fast=True,
                trust_remote_code=False,
            )
            self.model = AutoModel.from_pretrained(
                model_path, local_files_only=True, use_safetensors=True,
                trust_remote_code=False,
            )
        except Exception as siglip_error:
            if self.cancelled():
                raise RuntimeError("Class semantic verification cancelled") from siglip_error
            # Keep Dataset Analysis useful on offline systems that already have
            # DarkFusion's older CLIP dependency cached.
            self.status(
                "SigLIP 2 is unavailable; using the cached CLIP class verifier..."
            )
            self.model_name = FALLBACK_MODEL_NAME
            self.model_revision = ""
            self.backend = "clip_fallback"
            try:
                fallback_path = snapshot_download(
                    repo_id=FALLBACK_MODEL_NAME,
                    cache_dir=self.model_cache,
                    local_files_only=True,
                )
            except (FileNotFoundError, OSError):
                # Earlier DarkFusion versions used Hugging Face's default cache.
                fallback_path = snapshot_download(
                    repo_id=FALLBACK_MODEL_NAME, local_files_only=True,
                )
            self.processor = AutoProcessor.from_pretrained(
                fallback_path, local_files_only=True, use_fast=True,
                trust_remote_code=False,
            )
            self.model = AutoModel.from_pretrained(
                fallback_path, local_files_only=True, trust_remote_code=False,
            )
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.status(
            f"{('SigLIP 2' if self.backend == 'siglip2' else 'CLIP fallback')} "
            f"class verifier ready on {self.device}."
        )
        return self

    def _prepare_cache(self):
        if self._connection is not None:
            return
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.cache_path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS semantic_scores "
            "(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._connection.commit()

    @staticmethod
    def _normalized_bounds(bounds):
        values = list(bounds or [])[:4]
        if len(values) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(value) for value in values]
        except (TypeError, ValueError):
            return None
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        return x1, y1, x2, y2

    def _cache_key(self, record, *, model_name=None):
        image_file = os.path.abspath(os.fspath(record.get("image_file", "")))
        stat = os.stat(image_file)
        bounds = self._normalized_bounds(record.get("bounds"))
        identity = {
            "version": PROMPT_VERSION,
            "model": str(model_name or self.model_name),
            "image": os.path.normcase(image_file),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "bounds": [round(value, 7) for value in bounds] if bounds else None,
            "class_name": " ".join(str(record.get("class_name", "")).split()).lower(),
            "context": round(self.context, 4),
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _cached(self, cache_key):
        row = self._connection.execute(
            "SELECT payload FROM semantic_scores WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if not row:
            return None
        try:
            value = json.loads(row[0])
            return value if isinstance(value, dict) else None
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def _crop(self, record):
        bounds = self._normalized_bounds(record.get("bounds"))
        if bounds is None:
            return None
        with Image.open(record["image_file"]) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            x1, y1, x2, y2 = bounds
            box_width = max(1.0 / max(1, width), x2 - x1)
            box_height = max(1.0 / max(1, height), y2 - y1)
            x1 = max(0.0, x1 - box_width * self.context)
            x2 = min(1.0, x2 + box_width * self.context)
            y1 = max(0.0, y1 - box_height * self.context)
            y2 = min(1.0, y2 + box_height * self.context)
            pixels = (
                max(0, int(round(x1 * width))),
                max(0, int(round(y1 * height))),
                min(width, int(round(x2 * width))),
                min(height, int(round(y2 * height))),
            )
            if pixels[2] - pixels[0] < 3 or pixels[3] - pixels[1] < 3:
                return None
            return image.crop(pixels)

    @staticmethod
    def _has_teammate_marker_layout(record):
        """Use the established geometry check as a veto for person-like crops."""
        try:
            from darkfusion_teammate_review import (
                has_paired_marker_layout,
                marker_crop,
            )
            bounds = ClassSemanticVerifier._normalized_bounds(record.get("bounds"))
            if bounds is None:
                return False
            with Image.open(record["image_file"]) as opened:
                crop = marker_crop(opened.convert("RGB"), {"bbox": bounds})
            return bool(has_paired_marker_layout(crop))
        except Exception:
            # This veto is additional evidence. Its failure must not abort the
            # scan or turn a class match into proof that the label is good.
            return False

    def _text_features(self, class_names):
        torch = self._torch
        features = {}
        distractor_prompts = [
            prompt for group in DISTRACTOR_PROMPT_GROUPS for prompt in group
        ]
        prompts = []
        layout = {}
        for class_name in class_names:
            positive = list(class_prompts(class_name))
            start = len(prompts)
            prompts.extend(positive)
            positive_slice = (start, len(prompts))
            start = len(prompts)
            prompts.extend(distractor_prompts)
            negative_slices = []
            offset = start
            for group in DISTRACTOR_PROMPT_GROUPS:
                negative_slices.append((offset, offset + len(group)))
                offset += len(group)
            layout[class_name] = (positive_slice, negative_slices)
        encoded = self.processor(text=prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            text = pooled_features(self.model.get_text_features(**encoded)).float()
            text = text / text.norm(dim=-1, keepdim=True)
        for class_name, (positive_slice, negative_slices) in layout.items():
            positive = text[slice(*positive_slice)].mean(dim=0)
            positive = positive / positive.norm()
            negatives = []
            for start, stop in negative_slices:
                group = text[start:stop].mean(dim=0)
                negatives.append(group / group.norm())
            features[class_name] = (positive, torch.stack(negatives))
        return features

    def score_records(self, records):
        """Return one semantic evidence dictionary (or ``None``) per record."""
        records = list(records or [])
        self._prepare_cache()
        results = [None] * len(records)
        pending = []
        for index, record in enumerate(records):
            if self.cancelled():
                raise RuntimeError("Class semantic verification cancelled")
            try:
                cache_key = self._cache_key(record)
                cached = self._cached(cache_key)
            except (OSError, ValueError):
                cache_key, cached = "", None
            if cached is not None:
                cached["cached"] = True
                results[index] = cached
                continue
            try:
                crop = self._crop(record)
            except (OSError, ValueError):
                crop = None
            if crop is None:
                results[index] = {"error": "unavailable crop", "cached": False}
                continue
            pending.append((index, record, crop, cache_key))

        if not pending:
            return results
        self.prepare()
        if self.model_name != MODEL_NAME:
            fallback_pending = []
            for index, record, crop, _cache_key in pending:
                try:
                    cache_key = self._cache_key(
                        record, model_name=self.model_name
                    )
                    cached = self._cached(cache_key)
                except (OSError, ValueError):
                    cache_key, cached = "", None
                if cached is not None:
                    cached["cached"] = True
                    results[index] = cached
                else:
                    fallback_pending.append((index, record, crop, cache_key))
            pending = fallback_pending
            if not pending:
                return results
        class_names = list(dict.fromkeys(
            " ".join(str(record.get("class_name", "object")).split()) or "object"
            for _index, record, _crop, _key in pending
        ))
        text_features = self._text_features(class_names)
        torch = self._torch
        writes = []
        start = 0
        while start < len(pending):
            if self.cancelled():
                raise RuntimeError("Class semantic verification cancelled")
            batch = pending[start:start + self.batch_size]
            pixels = self.processor(
                images=[crop for _index, _record, crop, _key in batch],
                return_tensors="pt",
            )["pixel_values"].to(self.device)
            try:
                with torch.inference_mode():
                    if self.device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            image_features = self.model.get_image_features(pixel_values=pixels)
                    else:
                        image_features = self.model.get_image_features(pixel_values=pixels)
                    image_features = pooled_features(image_features).float()
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            except torch.cuda.OutOfMemoryError:
                del pixels
                torch.cuda.empty_cache()
                if self.device.type == "cuda" and self.batch_size > 1:
                    self.batch_size = max(1, self.batch_size // 2)
                    self.status(
                        f"Reducing class verification batch size to {self.batch_size} "
                        "because GPU memory is busy..."
                    )
                    continue
                raise
            for row, (index, record, _crop, cache_key) in enumerate(batch):
                class_name = " ".join(
                    str(record.get("class_name", "object")).split()
                ) or "object"
                positive, negatives = text_features[class_name]
                class_similarity = float(image_features[row] @ positive)
                distractor_scores = image_features[row] @ negatives.T
                distractor_similarity = float(distractor_scores.max())
                margin = class_similarity - distractor_similarity
                pair = torch.tensor(
                    [class_similarity, distractor_similarity],
                    device=self.device,
                    dtype=torch.float32,
                ) * 100.0
                probability = float(torch.softmax(pair, dim=0)[0])
                result = {
                    "model_name": self.model_name,
                    "backend": self.backend,
                    "class_probability": round(probability, 6),
                    "semantic_margin": round(margin, 6),
                    "class_similarity": round(class_similarity, 6),
                    "distractor_similarity": round(distractor_similarity, 6),
                    "teammate_marker_layout": False,
                    "cached": False,
                }
                if should_protect_class_match(result, scan_pass=2):
                    result["teammate_marker_layout"] = self._has_teammate_marker_layout(
                        record
                    )
                results[index] = result
                if cache_key:
                    writes.append((cache_key, json.dumps(result, separators=(",", ":"))))
            start += len(batch)
        if writes:
            self._connection.executemany(
                "INSERT OR REPLACE INTO semantic_scores(cache_key, payload) VALUES (?, ?)",
                writes,
            )
            self._connection.commit()
        return results

    def close(self):
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
        self.model = None
        self.processor = None
        torch = self._torch
        self._torch = None
        if torch is not None and self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.device = None

    def __enter__(self):
        return self.prepare()

    def __exit__(self, _type, _value, _traceback):
        self.close()
