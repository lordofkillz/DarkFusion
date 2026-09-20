"""Cached visual descriptors for the annotation-review worker (no GUI).

The score measures visual resemblance, not whether an annotation is incorrect.
DINOv3 downloads missing pinned checkpoints into Sam. The optional DINOv2 matcher downloads
official pinned safetensors. Importing this module never loads torch or a model.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
from PIL import Image

from darkfusion_dinov3.config import MODEL_CONFIGS, SOURCE_COMMIT
from darkfusion_dinov3.loader import cache_fingerprint, load_model
from darkfusion_dinov3.download import ensure_checkpoint


MODEL_ID = "facebook/dinov2-with-registers-base"
MODEL_REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"
CACHE_VERSION = "dinov2-cls-rgb-letterbox224-raw-orientation-v2"
IMAGE_SIZE = 224
EMBEDDING_SIZE = 768
DINO3_CACHE_VERSION = "dinov3-cls-rgb-letterbox224-raw-orientation-v1"
DEFAULT_MODEL_KEY = "dinov3_base"


class ReviewSimilarityError(RuntimeError):
    """The learned matcher could not run; callers must report the error."""


class ReviewSimilarityCancelled(ReviewSimilarityError):
    """The user cancelled the scan."""


class ReviewEmbeddingMatcher:
    """Encode records containing ``image_file`` and normalized ``bounds``.

    ``encode_records`` returns float32 unit vectors in input order, with None
    for unreadable images/invalid boxes. ``label_text`` is an optional additional
    cache identity. One instance belongs to one worker; independent workers can
    safely share its SQLite cache. Status callbacks receive a single string.
    Optional progress callbacks receive (completed records, total records),
    including cache hits and skipped records, separately for each encode call.
    """

    def __init__(self, cache_dir, status=None, cancelled=None, *,
                 batch_size=16, context=0.06, model_key=DEFAULT_MODEL_KEY,
                 models_dir=None, progress=None):
        self.cache_dir = Path(cache_dir)
        if model_key not in (*MODEL_CONFIGS, "visual"):
            raise ValueError(f"Unknown review similarity model: {model_key!r}")
        self.model_key = model_key
        self.models_dir = Path(models_dir) if models_dir is not None else Path(__file__).resolve().parent / "Sam"
        self.model_label = "DINOv2" if model_key == "visual" else MODEL_CONFIGS[model_key]["label"]
        self.embedding_size = EMBEDDING_SIZE if model_key == "visual" else MODEL_CONFIGS[model_key]["embedding_size"]
        self.checkpoint_path = None
        self._checkpoint_identity = self._verification_path = None
        if model_key != "visual":
            self.checkpoint_path = self.models_dir / MODEL_CONFIGS[model_key]["filename"]
            self._verification_path = self.cache_dir / f"{model_key}_checkpoint.json"
            self._checkpoint_identity = cache_fingerprint(
                self.checkpoint_path, model_key, self._verification_path)
        self.status = status or (lambda message: None)
        self.progress = progress or (lambda completed, total: None)
        self.cancelled = cancelled or (lambda: False)
        self.batch_size = max(1, min(32, int(batch_size)))
        self.context = float(context)
        if not math.isfinite(self.context) or not 0 <= self.context <= 0.5:
            raise ValueError("Crop context must be between 0 and 0.5.")
        self.device = "unloaded"
        self.stats = dict(cache_hits=0, encoded=0, skipped=0, images_read=0)
        self._model = self._processor = self._torch = None
        self._cache_disabled = False

    def _check_cancelled(self):
        if self.cancelled():
            raise ReviewSimilarityCancelled("Similarity scan cancelled.")

    def prepare(self):
        """Load on the review worker; download a missing checkpoint once."""
        self._check_cancelled()
        if self._model is not None:
            return self
        try:
            import torch
            self._torch = torch
            if self.model_key == "visual":
                model, processor = self._load_dinov2()
            else:
                if ensure_checkpoint(self.model_key, self.checkpoint_path,
                                     status=self.status, check_cancelled=self._check_cancelled):
                    self._checkpoint_identity = cache_fingerprint(
                        self.checkpoint_path, self.model_key, self._verification_path)
                self.status(f"Loading {self.model_label} from Sam...")
                model = load_model(self.model_key, self.checkpoint_path,
                                   memo_path=self._verification_path, status=self.status,
                                   check_cancelled=self._check_cancelled,
                                   expected_fingerprint=self._checkpoint_identity)
                processor = None
            self._check_cancelled()
            model.eval().requires_grad_(False)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                model.to(self.device)
            except torch.cuda.OutOfMemoryError:
                model.to("cpu")
                torch.cuda.empty_cache()
                self.device = "cpu"
                self.status(f"GPU memory is busy; using CPU for {self.model_label} similarity.")
            self._model, self._processor = model, processor
            mode = "GPU FP16" if self.device == "cuda" else "CPU"
            self.status(f"{self.model_label} similarity ready ({mode}).")
            return self
        except ReviewSimilarityCancelled:
            raise
        except Exception as exc:
            if self.model_key != "visual":
                raise ReviewSimilarityError(
                    f"Could not load {self.model_label}. Place the official checkpoint at "
                    f"{self.checkpoint_path}, or select DINOv2 / Appearance and shape (CPU) "
                    f"in Settings > Display > Review Preview. {exc}"
                ) from exc
            raise ReviewSimilarityError(
                "Could not load DINOv2 similarity. The first scan needs internet "
                f"access and a writable model cache. {exc}"
            ) from exc

    def _load_dinov2(self):
        from transformers import AutoImageProcessor, AutoModel
        model_cache = self.cache_dir / "models"
        model_cache.mkdir(parents=True, exist_ok=True)
        options = dict(cache_dir=str(model_cache), revision=MODEL_REVISION,
                       trust_remote_code=False, token=False)
        self.status("Loading DINOv2 similarity model from cache...")
        try:
            processor = AutoImageProcessor.from_pretrained(
                MODEL_ID, local_files_only=True, use_fast=False, **options)
            self._check_cancelled()
            model = AutoModel.from_pretrained(
                MODEL_ID, local_files_only=True, use_safetensors=True, **options)
        except OSError:
            self._check_cancelled()
            self.status("Downloading DINOv2 similarity model (about 350 MB, once)...")
            processor = AutoImageProcessor.from_pretrained(
                MODEL_ID, use_fast=False, **options)
            self._check_cancelled()
            model = AutoModel.from_pretrained(
                MODEL_ID, use_safetensors=True, **options)
        return model, processor

    def close(self):
        """Release model memory after the worker finishes."""
        self._model = self._processor = None
        if self._torch is not None and self.device == "cuda":
            self._torch.cuda.empty_cache()
        self.device = "unloaded"

    def _open_cache(self):
        if self._cache_disabled:
            return None
        connection = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                str(self.cache_dir / "object_embeddings.sqlite3"), timeout=0.25)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings "
                "(cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )
            connection.commit()
            return connection
        except (OSError, sqlite3.Error) as exc:
            if connection is not None:
                connection.close()
            self._disable_cache(exc)
            return None

    def _disable_cache(self, exc):
        if not self._cache_disabled:
            self.status("Similarity cache is unavailable; this scan will compute "
                        f"descriptors without saving them ({exc}).")
        self._cache_disabled = True

    @staticmethod
    def _file_identity(image_file):
        path = os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(image_file))))
        stat = os.stat(path)
        return path, stat.st_size, stat.st_mtime_ns

    def _cache_key(self, identity, bounds, label_text=""):
        if self.model_key == "visual":
            # Preserve every byte of the existing DINOv2 cache identity.
            payload = (CACHE_VERSION, MODEL_ID, MODEL_REVISION, self.context,
                       identity, bounds, str(label_text or ""))
        else:
            payload = (DINO3_CACHE_VERSION, self.model_key, MODEL_CONFIGS[self.model_key]["sha256"],
                       SOURCE_COMMIT, self.embedding_size, self._checkpoint_identity, self.context,
                       identity, bounds, str(label_text or ""))
        return hashlib.sha256(json.dumps(payload, ensure_ascii=True,
                                        separators=(",", ":")).encode("utf-8")).hexdigest()

    @staticmethod
    def _bounds(record):
        bounds = tuple(float(value) for value in record["bounds"])
        if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
            raise ValueError("Invalid object bounds")
        bounds = tuple(max(0.0, min(1.0, value)) for value in bounds)
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError("Empty object bounds")
        return bounds

    def _cached(self, connection, key):
        if connection is None or self._cache_disabled:
            return None
        try:
            row = connection.execute(
                "SELECT vector FROM embeddings WHERE cache_key = ?", (key,)).fetchone()
            if row is None:
                return None
            # A bad row is a miss, never a NaN score or a crash during review.
            if not isinstance(row[0], bytes) or len(row[0]) != self.embedding_size * 4:
                return None
            vector = np.frombuffer(row[0], dtype="<f4").copy()
            norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or not 0.99 <= norm <= 1.01:
                return None
            return vector / norm
        except (sqlite3.Error, ValueError, TypeError) as exc:
            self._disable_cache(exc)
            return None

    def _save(self, connection, pairs):
        if connection is None or self._cache_disabled or not pairs:
            return
        self._check_cancelled()
        try:
            with connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO embeddings (cache_key, vector) VALUES (?, ?)",
                    ((key, np.asarray(vector, dtype="<f4").tobytes())
                     for key, vector in pairs))
        except sqlite3.Error as exc:
            self._disable_cache(exc)

    def _crop(self, image, bounds):
        width, height = image.size
        x1, y1, x2, y2 = bounds
        margin_x, margin_y = (x2 - x1) * self.context, (y2 - y1) * self.context
        pixels = (max(0, math.floor((x1 - margin_x) * width)),
                  max(0, math.floor((y1 - margin_y) * height)),
                  min(width, math.ceil((x2 + margin_x) * width)),
                  min(height, math.ceil((y2 + margin_y) * height)))
        crop = image.crop(pixels)
        # Padding preserves the entire annotation and its aspect ratio, including
        # tall/thin objects that a standard center-crop would cut off.
        crop.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        if max(crop.size) < IMAGE_SIZE:
            scale = IMAGE_SIZE / max(crop.size)
            crop = crop.resize((max(1, round(crop.width * scale)),
                                max(1, round(crop.height * scale))), Image.Resampling.BICUBIC)
        mean = getattr(self._processor, "image_mean", [0.485, 0.456, 0.406])
        canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE),
                           tuple(round(float(value) * 255) for value in mean))
        canvas.paste(crop, ((IMAGE_SIZE - crop.width) // 2,
                            (IMAGE_SIZE - crop.height) // 2))
        return canvas

    def _forward(self, crops):
        torch = self._torch
        if self.model_key == "visual":
            inputs = self._processor(images=crops, return_tensors="pt",
                                     do_resize=False, do_center_crop=False)
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
        else:
            # Identical RGB/ImageNet normalization to the comparison and DINOv2,
            # with no dependency on a newer transformers release for DINOv3.
            pixels = np.stack([np.asarray(crop, dtype=np.float32) for crop in crops]) / 255.0
            pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                [0.229, 0.224, 0.225], dtype=np.float32)
            inputs = torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy()).to(self.device)
        autocast = torch.autocast("cuda", dtype=torch.float16) if self.device == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            if self.model_key == "visual":
                features = self._model(**inputs).last_hidden_state[:, 0]
            else:
                features = self._model.forward_features(inputs)["x_norm_clstoken"]
            # The model's normalized CLS token is its image retrieval descriptor;
            # register tokens are excluded rather than averaged into the result.
            vectors = torch.nn.functional.normalize(
                features.float(), p=2, dim=-1)
        return vectors.cpu().numpy().astype(np.float32, copy=False)

    def _encode_crops(self, crops):
        vectors = []
        offset = 0
        while offset < len(crops):
            self._check_cancelled()
            count = min(self.batch_size, len(crops) - offset)
            try:
                batch = self._forward(crops[offset:offset + count])
            except self._torch.cuda.OutOfMemoryError:
                if self.device != "cuda":
                    raise ReviewSimilarityError("Not enough memory to compute similarity.")
                self._torch.cuda.empty_cache()
                if count > 1:
                    self.batch_size = max(1, count // 2)
                    self.status(f"Reducing {self.model_label} batch size to {self.batch_size} for GPU memory.")
                else:
                    self._model.to("cpu")
                    self.device = "cpu"
                    self._torch.cuda.empty_cache()
                    self.status("GPU memory is busy; continuing similarity scan on CPU.")
                continue
            self._check_cancelled()
            if (batch.shape != (count, self.embedding_size)
                    or not np.isfinite(batch).all()
                    or np.any(np.linalg.norm(batch, axis=1) < 1e-8)):
                raise ReviewSimilarityError(f"{self.model_label} returned invalid similarity descriptors.")
            vectors.extend(batch)
            offset += count
        return vectors

    def encode_records(self, records):
        """Decode each image once and batch small crops across image boundaries.

        At most ``batch_size`` 224px crops and one full decoded image are retained.
        Cached records never need a model or image decode. Cancellation discards
        this call's partial result, while earlier committed cache batches survive.
        """
        records = list(records)
        results = [None] * len(records)
        initial_completed = sum(self.stats[name] for name in ("cache_hits", "encoded", "skipped"))
        last_progress_time = 0.0

        def report_progress(force=False):
            nonlocal last_progress_time
            now = time.monotonic()
            if force or now - last_progress_time >= 0.1:
                completed = sum(self.stats[name] for name in ("cache_hits", "encoded", "skipped"))
                self.progress(completed - initial_completed, len(records))
                last_progress_time = now

        self._check_cancelled()
        report_progress(force=True)
        grouped = OrderedDict()
        for index, record in enumerate(records):
            self._check_cancelled()
            try:
                path = os.path.normcase(os.path.realpath(os.path.abspath(
                    os.fspath(record["image_file"]))))
                bounds = self._bounds(record)
            except (KeyError, TypeError, ValueError, OSError):
                self.stats["skipped"] += 1
                continue
            grouped.setdefault(path, []).append((index, bounds, record.get("label_text", "")))
        connection = self._open_cache()
        snapshots = {}
        invalid_paths = set()
        encoded_indices, cached_indices = set(), set()
        queued_crops, queued_items = [], []

        def invalidate(path):
            if path in invalid_paths:
                return
            invalid_paths.add(path)
            entries = grouped[path]
            for index, _, _ in entries:
                results[index] = None
                if index in encoded_indices:
                    encoded_indices.remove(index)
                    self.stats["encoded"] -= 1
                if index in cached_indices:
                    cached_indices.remove(index)
                    self.stats["cache_hits"] -= 1
            self.stats["skipped"] += len(entries)

        def unchanged(path):
            if path in invalid_paths:
                return False
            try:
                matches = self._file_identity(path) == snapshots[path]
            except OSError:
                matches = False
            if not matches:
                invalidate(path)
            return matches

        def flush():
            nonlocal queued_crops, queued_items
            if not queued_crops:
                return
            self._check_cancelled()
            crops, items = queued_crops, queued_items
            queued_crops, queued_items = [], []
            try:
                vectors = self._encode_crops(crops)
                if len(vectors) != len(items):
                    raise ReviewSimilarityError("The similarity model returned an incomplete batch.")
                good_paths = {path for path in dict.fromkeys(item[2] for item in items) if unchanged(path)}
                pairs = [(key, vector) for (_, key, path), vector in zip(items, vectors) if path in good_paths]
                self._save(connection, pairs)
                for (index, _, path), vector in zip(items, vectors):
                    if path in good_paths:
                        results[index] = vector
                        encoded_indices.add(index)
                        self.stats["encoded"] += 1
            finally:
                for crop in crops:
                    crop.close()

        try:
            for image_number, (path, entries) in enumerate(grouped.items(), 1):
                self._check_cancelled()
                try:
                    identity = self._file_identity(path)
                except OSError:
                    self.stats["skipped"] += len(entries)
                    report_progress()
                    continue
                snapshots[path] = identity
                pending = []
                for index, bounds, label_text in entries:
                    key = self._cache_key(identity, bounds, label_text)
                    vector = self._cached(connection, key)
                    if vector is not None:
                        results[index] = vector
                        cached_indices.add(index)
                        self.stats["cache_hits"] += 1
                    else:
                        pending.append((index, bounds, key))
                if not pending:
                    report_progress()
                    continue
                self.prepare()
                # A first-use download changes the checkpoint identity. Store
                # descriptors under the installed file's identity immediately.
                pending = [(index, bounds, self._cache_key(identity, bounds,
                            records[index].get("label_text", "")))
                           for index, bounds, _ in pending]
                self.status(f"{self.model_label}: image {image_number}/{len(grouped)} "
                            f"({self.stats['cache_hits']} cached).")
                try:
                    with Image.open(path) as opened:
                        # Review displays raw QPixmap coordinates; applying EXIF
                        # rotation here would select a different annotation crop.
                        image = opened.convert("RGB")
                    self.stats["images_read"] += 1
                except (OSError, ValueError, Image.DecompressionBombError):
                    invalidate(path)
                    report_progress()
                    continue
                try:
                    for index, bounds, key in pending:
                        self._check_cancelled()
                        if path in invalid_paths:
                            break
                        queued_crops.append(self._crop(image, bounds))
                        queued_items.append((index, key, path))
                        if len(queued_crops) >= self.batch_size:
                            flush()
                finally:
                    image.close()
                report_progress()
            flush()
            # Also protect cached-only and earlier-batch results if an image was
            # replaced while another image/model was being processed.
            for path in snapshots:
                self._check_cancelled()
                unchanged(path)
            self._check_cancelled()
            report_progress(force=True)
            return results
        except ReviewSimilarityError:
            raise
        except Exception as exc:
            raise ReviewSimilarityError(f"{self.model_label} similarity scan failed: {exc}") from exc
        finally:
            for crop in queued_crops:
                crop.close()
            if connection is not None:
                connection.close()

    @staticmethod
    def score(reference, candidate):
        """Cosine resemblance scaled to [0, 1]; this is not a probability."""
        if reference is None or candidate is None:
            return 0.0
        left, right = np.asarray(reference), np.asarray(candidate)
        if (left.ndim != 1 or left.shape != right.shape or not left.size
                or not np.isfinite(left).all() or not np.isfinite(right).all()):
            return 0.0
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1e-12:
            return 0.0
        cosine = float(np.dot(left, right)) / denominator
        return float(np.clip((cosine + 1.0) * 0.5, 0.0, 1.0))
