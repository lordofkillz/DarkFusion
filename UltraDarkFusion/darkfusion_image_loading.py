"""Bounded background loading of original images, independent of Qt thumbnails."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import weakref


def image_file_version(path):
    try:
        path = os.path.normcase(os.path.abspath(path))
        stat = os.stat(path)
        return path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino
    except (OSError, TypeError, ValueError):
        return None


class OriginalImagePrefetch:
    """UI-owned futures containing only original arrays, never pixmaps or crops.

    One decoder, at most two requested neighbors and 64 MiB of estimated decoded
    pixels. Superseded work cannot publish anything into the UI or current image.
    """

    MAX_BYTES = 64 * 1024 * 1024

    def __init__(self, reader, version=image_file_version):
        self._reader = reader
        self._version = version
        self._executor = None
        self._pending = {}
        self._sizes = {}

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
                self._pending.pop(version).cancel()
                self._sizes.pop(version, None)
        for version, path in requested.items():
            if version not in self._pending:
                self._sizes[version] = requested_sizes[version]
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="original-image")
                self._pending[version] = self._executor.submit(
                    self._decode, self._reader, self._version, path, version)

    def take(self, path, version):
        future = self._pending.pop(version, None)
        self._sizes.pop(version, None)
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

    def close(self, *, wait=True):
        self.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=True)
            self._executor = None

    def __del__(self):
        self.close(wait=False)


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
