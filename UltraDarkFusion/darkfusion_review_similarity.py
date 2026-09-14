"""Cached DINOv2 appearance descriptors for the object-review worker (no GUI).

The score measures visual resemblance, not whether an annotation is incorrect.
Models are downloaded from the official, pinned Hugging Face repository using
safetensors only. Importing this module does not import torch or download files.
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

import numpy as np
from PIL import Image


MODEL_ID = "facebook/dinov2-with-registers-base"
MODEL_REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"
CACHE_VERSION = "dinov2-cls-rgb-letterbox224-raw-orientation-v2"
IMAGE_SIZE = 224
EMBEDDING_SIZE = 768


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
    """

    def __init__(self, cache_dir, status=None, cancelled=None, *,
                 batch_size=16, context=0.06):
        self.cache_dir = Path(cache_dir)
        self.status = status or (lambda message: None)
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
        """Load the cached model, downloading it once when necessary."""
        self._check_cancelled()
        if self._model is not None:
            return self
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self._torch = torch
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
            self._check_cancelled()
            model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                model.to(self.device)
            except torch.cuda.OutOfMemoryError:
                model.to("cpu")
                torch.cuda.empty_cache()
                self.device = "cpu"
                self.status("GPU memory is busy; using CPU for DINOv2 similarity.")
            self._model, self._processor = model, processor
            mode = "GPU FP16" if self.device == "cuda" else "CPU"
            self.status(f"DINOv2 similarity ready ({mode}).")
            return self
        except ReviewSimilarityCancelled:
            raise
        except Exception as exc:
            raise ReviewSimilarityError(
                "Could not load DINOv2 similarity. The first scan needs internet "
                f"access and a writable model cache. {exc}"
            ) from exc

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
        payload = (CACHE_VERSION, MODEL_ID, MODEL_REVISION, self.context,
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
            if not isinstance(row[0], bytes) or len(row[0]) != EMBEDDING_SIZE * 4:
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
        inputs = self._processor(images=crops, return_tensors="pt",
                                 do_resize=False, do_center_crop=False)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        autocast = torch.autocast("cuda", dtype=torch.float16) if self.device == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            output = self._model(**inputs)
            # The model's normalized CLS token is its image retrieval descriptor;
            # register tokens are excluded rather than averaged into the result.
            vectors = torch.nn.functional.normalize(
                output.last_hidden_state[:, 0].float(), p=2, dim=-1)
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
                    self.status(f"Reducing DINOv2 batch size to {self.batch_size} for GPU memory.")
                else:
                    self._model.to("cpu")
                    self.device = "cpu"
                    self._torch.cuda.empty_cache()
                    self.status("GPU memory is busy; continuing similarity scan on CPU.")
                continue
            self._check_cancelled()
            if (batch.shape != (count, EMBEDDING_SIZE)
                    or not np.isfinite(batch).all()
                    or np.any(np.linalg.norm(batch, axis=1) < 1e-8)):
                raise ReviewSimilarityError("DINOv2 returned invalid similarity descriptors.")
            vectors.extend(batch)
            offset += count
        return vectors

    def encode_records(self, records):
        """Read each image at most once; cached records never require decoding.

        Caches commit only complete batches of descriptors. Cancellation leaves
        earlier entries usable, and always closes this call's SQLite connection.
        """
        records = list(records)
        results = [None] * len(records)
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
        try:
            for image_number, (path, entries) in enumerate(grouped.items(), 1):
                self._check_cancelled()
                try:
                    identity = self._file_identity(path)
                except OSError:
                    self.stats["skipped"] += len(entries)
                    continue
                pending = []
                for index, bounds, label_text in entries:
                    key = self._cache_key(identity, bounds, label_text)
                    vector = self._cached(connection, key)
                    if vector is not None:
                        results[index] = vector
                        self.stats["cache_hits"] += 1
                    else:
                        pending.append((index, bounds, key))
                if not pending:
                    continue
                self.prepare()
                self.status(f"Comparing objects: image {image_number}/{len(grouped)} "
                            f"({self.stats['cache_hits']} cached).")
                try:
                    with Image.open(path) as opened:
                        # Review displays raw QPixmap coordinates; applying EXIF
                        # rotation here would select a different annotation crop.
                        image = opened.convert("RGB")
                    self.stats["images_read"] += 1
                except (OSError, ValueError, Image.DecompressionBombError):
                    self.stats["skipped"] += len(pending)
                    continue
                try:
                    start = 0
                    while start < len(pending):
                        self._check_cancelled()
                        chunk = pending[start:start + self.batch_size]
                        crops = [self._crop(image, bounds) for _, bounds, _ in chunk]
                        vectors = self._encode_crops(crops)
                        try:
                            unchanged = self._file_identity(path) == identity
                        except OSError:
                            unchanged = False
                        if not unchanged:
                            # A concurrently replaced image must not receive a
                            # descriptor under its old timestamp/size identity.
                            for index, _, _ in entries:
                                results[index] = None
                            self.stats["skipped"] += len(entries)
                            break
                        self._save(connection, [(item[2], vector)
                                                for item, vector in zip(chunk, vectors)])
                        for (index, _, _), vector in zip(chunk, vectors):
                            results[index] = vector
                            self.stats["encoded"] += 1
                        start += len(chunk)
                finally:
                    image.close()
            return results
        except ReviewSimilarityError:
            raise
        except Exception as exc:
            raise ReviewSimilarityError(f"DINOv2 similarity scan failed: {exc}") from exc
        finally:
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
