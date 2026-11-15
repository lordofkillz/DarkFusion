"""Shared original images, label snapshots and bounded background loading.

Image and Qt dependencies load only when their readers are used.
"""

import io
import logging
import os
import threading
import weakref
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path


IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
    ".pbm",
    ".pgm",
    ".ppm",
    ".xbm",
    ".xpm",
    ".svg",
)

OUTPUT_IMAGE_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
    ".pbm",
    ".pgm",
    ".ppm",
)

NON_LABELING_IMAGE_SUFFIXES = (
    ".gif",
    ".svg",
    ".xbm",
    ".xpm",
)

DATASET_IMAGE_SUFFIXES = tuple(
    ext for ext in IMAGE_SUFFIXES if ext not in NON_LABELING_IMAGE_SUFFIXES
)


def normalize_image_extension(value, default=".jpg", valid_extensions=OUTPUT_IMAGE_SUFFIXES):
    text = str(value or "").strip().lower()
    if not text:
        return default

    if "." in text:
        text = os.path.splitext(text)[1] or text

    if not text.startswith("."):
        text = f".{text}"

    if text in valid_extensions:
        return text

    logging.getLogger("UltraDarkFusionLogger").warning(
        "Unsupported image format '%s'; using %s",
        value,
        default,
    )
    return default


def reflect_pad_pil(img, pad_right, pad_bottom):
    import numpy as np
    from PIL import Image

    np_img = np.array(img)
    if pad_right > 0:
        right_pad = np.flip(np_img[:, -pad_right:], axis=1)
        np_img = np.concatenate((np_img, right_pad), axis=1)
    if pad_bottom > 0:
        bottom_pad = np.flip(np_img[-pad_bottom:, :], axis=0)
        np_img = np.concatenate((np_img, bottom_pad), axis=0)
    return Image.fromarray(np_img)


def image_file_version(path):
    try:
        path = os.path.normcase(os.path.abspath(path))
        stat = os.stat(path)
        return path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino
    except (OSError, TypeError, ValueError):
        return None


class EditorLabelSource:
    """One version-checked label file shared by the scene, Review and saving."""

    MAX_BYTES = 1024 * 1024

    def __init__(self):
        self.clear()

    def clear(self):
        self._entry = (None, None)

    @property
    def _version(self):
        return self._entry[0]

    @property
    def _raw(self):
        return self._entry[1]

    def read_lines(self, path, *, latin1_fallback=False):
        version = image_file_version(path)
        cached_version, cached_raw = self._entry
        if version is not None and version == cached_version and cached_raw is not None:
            raw = cached_raw
        else:
            self.clear()
            if version is None or version[3] > self.MAX_BYTES:
                try:
                    with open(path, "r", encoding="utf-8") as stream:
                        return stream.readlines()
                except UnicodeDecodeError:
                    if not latin1_fallback:
                        raise
                    with open(path, "r", encoding="latin-1", errors="ignore") as stream:
                        return stream.readlines()
            with open(path, "rb") as stream:
                raw = stream.read(self.MAX_BYTES + 1)
            if len(raw) > self.MAX_BYTES:
                # The file grew during the read. Use the uncached reader once.
                try:
                    with open(path, "r", encoding="utf-8") as stream:
                        return stream.readlines()
                except UnicodeDecodeError:
                    if not latin1_fallback:
                        raise
                    with open(path, "r", encoding="latin-1", errors="ignore") as stream:
                        return stream.readlines()
            if version == image_file_version(path):
                self._entry = (version, raw)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            if not latin1_fallback:
                raise
            text = raw.decode("latin-1")
        # Match TextIOWrapper's universal newlines without treating other
        # Unicode whitespace as annotation row separators.
        return io.StringIO(text, newline=None).readlines()

    def matches(self, path, text):
        version, raw = self._entry
        return (raw is not None and version is not None
                and version == image_file_version(path) and raw == text.encode("utf-8"))


class OriginalImagePrefetch:
    """UI-owned futures containing only original arrays, never pixmaps or crops.

    One decoder, at most two requested neighbors, and up to four retained
    originals sharing a 64 MiB budget with pending decodes. Superseded work
    cannot publish anything into the UI or current image.
    """

    MAX_BYTES = 64 * 1024 * 1024
    MAX_RETAINED = 4

    def __init__(self, reader, version=image_file_version):
        self._reader = reader
        self._version = version
        self._executor = None
        self._pending = {}
        self._sizes = {}
        self._paths = {}
        self._retained = OrderedDict()

    def _trim_retained(self, reserved=0, keep=()):
        """Count retained originals and requested decodes against one budget."""
        keep = set(keep)
        total = reserved + sum(image.nbytes for version, image in self._retained.items() if version not in keep)
        for version in list(self._retained):
            if total <= self.MAX_BYTES and len(self._retained) <= self.MAX_RETAINED:
                break
            if version not in keep:
                total -= self._retained.pop(version).nbytes

    def remember(self, path, version, image):
        """Retain an already decoded original, never a thumbnail or Qt object."""
        if (version is None or image is None or not 0 < image.nbytes <= self.MAX_BYTES
                or self._version(path) != version):
            return
        pending = self._pending.pop(version, None)
        if pending is not None:
            pending.cancel()
        self._sizes.pop(version, None)
        self._paths.pop(version, None)
        image.setflags(write=False)
        self._retained[version] = image
        self._retained.move_to_end(version)
        self._trim_retained(sum(self._sizes.values()))

    @staticmethod
    def _size(path):
        # Other formats keep their existing synchronous decoder/plugin behavior.
        if Path(path).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            return 0
        from PIL import Image
        try:
            with Image.open(path) as image:
                return image.width * image.height * max(3, len(image.getbands()))
        except Exception:
            return 0

    @staticmethod
    def _decode(reader, version_fn, path, version):
        if version_fn(path) != version:
            return None
        image = reader(path)
        if image is None or image.nbytes > OriginalImagePrefetch.MAX_BYTES or version_fn(path) != version:
            return None
        return image

    def request(self, paths):
        requested = {}
        requested_sizes = {}
        budget = self.MAX_BYTES
        for path in paths:
            version = self._version(path)
            if version is None or version in requested:
                continue
            size = self._sizes.get(version)
            if version in self._retained:
                size = self._retained[version].nbytes
            if size is None:
                size = self._size(path)
            if not 0 < size <= budget:
                continue
            requested[version] = path
            requested_sizes[version] = size
            budget -= size
            if len(requested) == 2:
                break
        for version in list(self._pending):
            if version not in requested:
                future = self._pending.pop(version)
                self._sizes.pop(version, None)
                path = self._paths.pop(version, None)
                if future.done() and not future.cancelled():
                    try:
                        self.remember(path, version, future.result())
                    except Exception:
                        pass  # A failed speculative read has no reusable pixels.
                else:
                    future.cancel()
        self._trim_retained(sum(requested_sizes.values()), keep=requested)
        for version, path in requested.items():
            if version in self._retained:
                self._retained.move_to_end(version)
            elif version not in self._pending:
                self._sizes[version] = requested_sizes[version]
                self._paths[version] = path
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="original-image")
                self._pending[version] = self._executor.submit(
                    self._decode, self._reader, self._version, path, version)

    def take(self, path, version):
        image = self._retained.pop(version, None)
        if image is not None:
            return image if self._version(path) == version else None
        future = self._pending.pop(version, None)
        self._sizes.pop(version, None)
        self._paths.pop(version, None)
        if future is None:
            return None
        try:
            image = future.result()
        except Exception:
            return None  # Retry with the original synchronous reader.
        return image if self._version(path) == version else None

    def clear(self):
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        self._sizes.clear()
        self._paths.clear()
        self._retained.clear()

    def close(self, *, wait=True):
        self.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=True)
            self._executor = None

    def __del__(self):
        self.close(wait=False)


class EditorImageSource:
    """Share one unmodified editor image with its review crops on the UI thread.

    Preprocessing receives a separate array. Other review images and formats
    with different Qt decoding semantics retain their existing QPixmap loader.
    """

    def __init__(self):
        import cv2

        self._prefetch = OriginalImagePrefetch(lambda path: cv2.imread(path), self.file_version)
        self.clear()

    def clear(self):
        self._prefetch.clear()
        self._clear_current()

    def _clear_current(self):
        self._version = None
        self._bgr = None
        self._pixmap = None
        self._review_compatible = None

    def prefetch(self, paths):
        self._prefetch.request(paths)

    def close(self):
        self.clear()
        self._prefetch.close()

    file_version = staticmethod(image_file_version)

    @staticmethod
    def can_share_review(path):
        # Qt retains transparency/profiles and uses different EXIF defaults.
        # Inspect headers only; preserve its loader for those special cases.
        try:
            # Gameplay PNGs contain many small IDAT chunks. A bounded larger
            # buffer avoids a filesystem read/seek for every chunk on Windows.
            with open(path, "rb", buffering=512 * 1024) as stream:
                size = os.fstat(stream.fileno()).st_size
                if stream.read(8) == b"\x89PNG\r\n\x1a\n":
                    have_header = False
                    while stream.tell() + 12 <= size:
                        chunk = stream.read(8)
                        length, kind = int.from_bytes(chunk[:4], "big"), chunk[4:]
                        if length > size - stream.tell() - 4:
                            return False
                        if kind == b"IHDR":
                            if have_header or length != 13:
                                return False
                            info = stream.read(13)
                            if info[8] != 8 or info[9] not in (0, 2):
                                return False
                            have_header = True
                            stream.seek(4, os.SEEK_CUR)
                        elif kind == b"IEND":
                            return have_header and length == 0
                        elif kind in (b"IDAT", b"pHYs") and have_header:
                            stream.seek(length + 4, os.SEEK_CUR)
                        else:
                            # Metadata after IDAT must also keep Qt's original
                            # profile, transparency, EXIF and animation behavior.
                            return False
                    return False
                stream.seek(0)
                from PIL import Image

                with Image.open(stream) as header:
                    return (
                        header.format == "JPEG"
                        and header.mode in {"RGB", "L"}
                        and header.getexif().get(274, 1) == 1
                        and not any(key in header.info for key in ("transparency", "icc_profile", "gamma"))
                    )
        except Exception:
            return False

    def read(self, path):
        import cv2

        version = self.file_version(path)
        if version is not None and version == self._version and self._bgr is not None:
            return self._bgr.copy()

        # Neighbor futures contain full originals, never review thumbnails.
        image = self._prefetch.take(path, version) if version is not None else None
        if self._version is not None and self._bgr is not None:
            self._prefetch.remember(self._version[0], self._version, self._bgr)
        self._clear_current()
        if image is None:
            image = cv2.imread(path)
        if image is not None and image.size and version is not None and version == self.file_version(path):
            self._version = version
            self._bgr = image.copy() if image.flags.writeable else image
            self._bgr.setflags(write=False)
        # Preprocessing always gets writable pixels independent of retained originals.
        return image.copy() if image is not None and not image.flags.writeable else image

    def review_pixmap(self, path):
        from PyQt5.QtGui import QImage, QPixmap

        version = self.file_version(path)
        if version is not None and version == self._version and self._bgr is not None:
            if self._review_compatible is None:
                self._review_compatible = self.can_share_review(path)
            if self._review_compatible and version == self.file_version(path):
                if self._pixmap is None:
                    height, width = self._bgr.shape[:2]
                    image = QImage(
                        self._bgr.data, width, height, self._bgr.strides[0], QImage.Format_RGB888
                    ).rgbSwapped()
                    self._pixmap = QPixmap.fromImage(image)
                # Qt detaches this implicitly shared copy if a caller paints on it.
                return QPixmap(self._pixmap)
        return QPixmap(path)


def prepared_review_sources(images, batch_size):
    """Keep one list-source group ready using Ultralytics' exact PIL loader.

    A one-path source retains its existing OpenCV path. Exceptions and changed
    files fall back to the original source, so model.predict owns normal errors.
    The returned PIL images retain filenames and EXIF behavior. Nothing is saved.
    """
    from ultralytics.data.loaders import autocast_list

    def prepare(chunk):
        source = chunk[0] if len(chunk) == 1 else chunk
        versions = [image_file_version(path) for path in chunk]
        if len(chunk) == 1 or any(v is None for v in versions):
            return source, None
        sizes = [OriginalImagePrefetch._size(path) for path in chunk]
        if not all(sizes) or sum(sizes) > OriginalImagePrefetch.MAX_BYTES:
            return source, None
        return autocast_list(chunk), versions

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="review-image")
    future = None
    try:
        chunk = images[:batch_size]
        if not chunk:
            return
        future = executor.submit(prepare, chunk)
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            try:
                source, versions = future.result()
            except Exception:
                source, versions = (chunk[0] if len(chunk) == 1 else chunk), None
            next_chunk = images[start + batch_size:start + 2 * batch_size]
            future = executor.submit(prepare, next_chunk) if next_chunk else None
            if versions is not None and versions != [image_file_version(path) for path in chunk]:
                source = chunk
            yield chunk, source
    finally:
        if future is not None:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def prefetched_file_batches(items, batch_size, reader, *, prepare=None, executor=None,
                            extra_bytes_per_item=0, max_pending=4, include_versions=False):
    """Decode bounded file groups; provider execution stays on the caller.

    The queued originals share a 64 MiB estimate, independent of inference batch
    size. Changed files, unsupported/large groups and errors retain normal reads.
    The existing reader returns (original, path); results stay in input order.
    An optional CPU preparer uses a model-owned pool with its input/scratch
    estimate included in the loading budget. Default mode only decodes.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    limit = OriginalImagePrefetch.MAX_BYTES
    owns_executor = executor is None
    if owns_executor:
        executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference-image")
    pending = deque()
    cursor, reserved = 0, 0
    max_pending = max(1, int(max_pending))
    window = min(4, max_pending)

    def load(chunk, versions):
        if versions != [image_file_version(path) for path in chunk]:
            return None, None
        loaded = [reader(path) for path in chunk]
        return (prepare(loaded) if prepare is not None else loaded), versions

    try:
        while cursor < len(items) or pending:
            while cursor < len(items) and len(pending) < window:
                chunk = items[cursor:cursor + batch_size]
                versions = [image_file_version(path) for path in chunk]
                sizes = [OriginalImagePrefetch._size(path) for path in chunk]
                # Classification can briefly hold full-size RGB/Pillow copies
                # as well as the original. Reserve those before starting work.
                source_factor = 3 if prepare is not None else 1
                size = (sum(sizes) * source_factor + extra_bytes_per_item * batch_size) if all(sizes) and all(versions) else 0
                eligible = 0 < size <= limit
                if eligible and pending and reserved + size > limit:
                    break
                future = executor.submit(load, chunk, versions) if eligible else None
                pending.append((chunk, future, size if eligible else 0))
                reserved += size if eligible else 0
                cursor += len(chunk)
            chunk, future, size = pending.popleft()
            reserved -= size
            try:
                loaded, versions = future.result() if future is not None else (None, None)
            except Exception:
                loaded, versions = None, None
            if versions is not None and versions != [image_file_version(path) for path in chunk]:
                loaded = None
            yield (chunk, loaded, versions) if include_versions else (chunk, loaded)
            window = min(max_pending, window + 1)
    finally:
        for _, future, _ in pending:
            if future is not None:
                future.cancel()
        if owns_executor:
            executor.shutdown(wait=True, cancel_futures=True)
        else:
            # A reusable pool belongs to the model. Drain just this stream's
            # canceled/in-flight work before releasing its reader/preparer.
            for _, future, _ in pending:
                if future is not None:
                    try:
                        future.result()
                    except Exception:
                        pass


class ImagePreparationPool:
    """Model-owned CPU workers; each stream drains its own bounded preparation.

    The coordinator overlaps future scheduling/header work with inference.
    Original version tokens travel with the pixels through both queues.
    No Qt objects, provider Run calls or GPU buffers belong to these workers.
    """

    def __init__(self, *, initializer=None, max_workers=8):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, initializer=initializer,
                                            thread_name_prefix="inference-prepare")
        self._workers = max_workers
        self._cleanup = weakref.finalize(self, self._executor.shutdown, wait=True, cancel_futures=True)

    def close(self):
        self._cleanup()

    def batches(self, items, batch_size, reader, prepare, extra_bytes_per_item):
        iterator = prefetched_file_batches(
            items, batch_size, reader, prepare=prepare, executor=self._executor,
            extra_bytes_per_item=extra_bytes_per_item, max_pending=self._workers,
            include_versions=True)
        coordinator = ThreadPoolExecutor(max_workers=1, thread_name_prefix="input-coordinator")

        def advance():
            return next(iterator, None)

        try:
            future = coordinator.submit(advance)
            while True:
                record = future.result()
                if record is None:
                    return
                chunk, prepared, versions = record
                future = coordinator.submit(advance)
                if versions is not None and versions != [image_file_version(path) for path in chunk]:
                    prepared = None
                yield chunk, prepared
        finally:
            coordinator.shutdown(wait=True, cancel_futures=True)
            iterator.close()


class LatestFramePreparation:
    """Own CPU-only callables; never retain a model, session or Qt parent."""

    def __init__(self, initializer=None):
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, args=(initializer,),
                                        name="video-input-prepare", daemon=True)
        self._thread.start()

    def submit(self, frame, prepare):
        future = Future()
        with self._condition:
            if self._closed:
                future.cancel()
                return future
            self._clear_locked()
            self._pending = (future, frame, prepare)
            self._condition.notify()
        return future

    def _clear_locked(self):
        if self._pending is not None:
            self._pending[0].cancel()
            self._pending = None

    def clear(self):
        with self._condition:
            self._clear_locked()

    def close(self, wait=True):
        with self._condition:
            self._closed = True
            self._clear_locked()
            self._condition.notify()
        if wait:
            self._thread.join()

    def _run(self, initializer):
        initialization_error = None
        try:
            if initializer is not None:
                initializer()
        except Exception as exc:
            initialization_error = exc
        initializer = None
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                future, frame, prepare = self._pending
                self._pending = None
            if future.set_running_or_notify_cancel():
                try:
                    if initialization_error is not None:
                        raise initialization_error
                    future.set_result(prepare(frame))
                except Exception as exc:
                    future.set_exception(exc)
            # An idle worker must not retain the previous frame/payload/closure.
            future = frame = prepare = None


DATASET_CONVERSION_TARGET_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".pbm",
    ".pgm",
    ".ppm",
)


FORMAT_MAP = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".gif": "GIF",
    ".pbm": "PBM",
    ".pgm": "PGM",
    ".ppm": "PPM",
}


def image_format_for_extension(extension, default="JPEG"):
    extension = normalize_image_extension(extension, default=".jpg")
    return FORMAT_MAP.get(extension, default)


def normalize_augmentation_output_format(value, default="source"):
    text = str(value or "").strip().lower()
    if not text:
        return default

    if text in {"source", "same", "original"}:
        return "source"

    return normalize_image_extension(text, default=".jpg")


def cv2_write_params_for_extension(extension, quality=95):
    import cv2
    extension = normalize_image_extension(extension)

    if extension in (".jpg", ".jpeg"):
        return [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    if extension == ".png":
        compression = 0 if int(quality) >= 95 else 3
        return [int(cv2.IMWRITE_PNG_COMPRESSION), compression]

    if extension == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)]
        if hasattr(cv2, "IMWRITE_WEBP_METHOD"):
            params.extend([int(cv2.IMWRITE_WEBP_METHOD), 6])
        return params

    return []


def save_cv_image(image_path, image, quality=95):
    import cv2
    import numpy as np
    from PIL import Image
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        logging.getLogger("UltraDarkFusionLogger").warning(
            "Skipping invalid image save: %s",
            image_path,
        )
        return False

    extension = normalize_image_extension(os.path.splitext(image_path)[1])

    try:
        if extension == ".gif":
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb_image).save(image_path, format="GIF")
            return True

        params = cv2_write_params_for_extension(extension, quality=quality)
        ok = cv2.imwrite(image_path, image, params) if params else cv2.imwrite(image_path, image)
        if ok:
            return True

        logging.getLogger("UltraDarkFusionLogger").warning(
            "cv2.imwrite returned False for %s; retrying without params",
            image_path,
        )
        return bool(cv2.imwrite(image_path, image))
    except Exception as e:
        logging.getLogger("UltraDarkFusionLogger").warning(
            "Failed saving image %s: %s",
            image_path,
            e,
        )
        try:
            return bool(cv2.imwrite(image_path, image))
        except Exception:
            return False


def extraction_frame_paths(output_dir, prefix, frame_index, image_extension=".jpg"):
    """Build the shared, zero-based frame identity used by every extractor."""
    extension = normalize_image_extension(image_extension, default=".jpg")
    stem = f"{str(prefix or 'session')}_frame_{max(0, int(frame_index or 0))}"
    image_path = os.path.join(str(output_dir), stem + extension)
    label_path = os.path.join(str(output_dir), stem + ".txt")
    return image_path, label_path


def existing_labeled_extraction_image(image_path, label_path):
    """Return the existing paired image when a non-empty label must be protected."""
    try:
        if not os.path.isfile(label_path) or os.path.getsize(label_path) <= 0:
            return ""
        image_stem = os.path.splitext(image_path)[0]
        for extension in OUTPUT_IMAGE_SUFFIXES:
            candidate = image_stem + extension
            if os.path.isfile(candidate):
                return candidate
    except OSError:
        return ""
    return ""


def normalize_dataset_conversion_extension(value, default=".jpg"):
    return normalize_image_extension(
        value,
        default=default,
        valid_extensions=DATASET_CONVERSION_TARGET_SUFFIXES,
    )


def prepare_image_for_dataset_format(image, target_extension):
    import numpy as np
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None

    target_extension = normalize_dataset_conversion_extension(target_extension)

    if image.ndim == 2:
        return np.ascontiguousarray(image)

    if image.ndim != 3:
        return None

    channels = image.shape[2]

    if channels == 4 and target_extension in (".jpg", ".jpeg", ".bmp", ".pbm", ".pgm", ".ppm"):
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255.0)
        return np.ascontiguousarray((bgr * alpha + white * (1.0 - alpha)).astype(np.uint8))

    if channels in (1, 3, 4):
        return np.ascontiguousarray(image)

    return np.ascontiguousarray(image[:, :, :3])


def convert_dataset_image_file(source_path, target_path, target_extension, quality=95):
    import cv2
    import numpy as np
    from PIL import Image
    image = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        try:
            with Image.open(source_path) as pil_image:
                if pil_image.mode in ("RGBA", "LA"):
                    pil_image = pil_image.convert("RGBA")
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
                else:
                    pil_image = pil_image.convert("RGB")
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logging.getLogger("UltraDarkFusionLogger").warning(
                "Could not read image for conversion %s: %s",
                source_path,
                e,
            )
            return False

    prepared = prepare_image_for_dataset_format(image, target_extension)
    return save_cv_image(target_path, prepared, quality=quality)


class InvalidImageError(Exception): ...
