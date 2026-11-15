"""Background dataset sorting, directory scans, file moves, exports and reveal jobs.

Filesystem operations and path policy stay in darkfusion_dataset_operations.
These workers preserve the main window's existing signal and constructor API.
"""

import concurrent.futures
import cv2
import glob
import logging
import mss
import numpy as np
import os
import shutil
import threading
import time
import torch
import traceback
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, QRunnable, QSettings, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImageReader, QImageWriter
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressBar
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from darkfusion_annotations import BoundingBox
from darkfusion_dataset_operations import (
    DATASET_METADATA_FILES, DatasetFileTransaction, dataset_metadata_path,
    delete_image_label_pair, label_file_version, natural_image_key, normalize_dataset_path,
    read_json_file, resolve_dataset_metadata_path, write_label_text_atomic,
)
from darkfusion_extraction import ExtractionCheckpoint, save_extraction_outputs
from darkfusion_image_loading import (
    DATASET_CONVERSION_TARGET_SUFFIXES, DATASET_IMAGE_SUFFIXES, IMAGE_SUFFIXES,
    OUTPUT_IMAGE_SUFFIXES, convert_dataset_image_file, existing_labeled_extraction_image,
    extraction_frame_paths, image_format_for_extension, normalize_dataset_conversion_extension,
    normalize_image_extension, save_cv_image,
)
from darkfusion_ui_controls import blocked_signals
from darkfusion_video_capture import transform_video_frame_by_settings
from pathlib import Path
from shapely.geometry import Polygon


logger = logging.getLogger("UltraDarkFusionLogger")


class DatasetNaturalSortWorker(QThread):
    """Sort an immutable snapshot without any filesystem calls or UI work."""
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image_files, directory, parent=None):
        super().__init__(parent)
        self.image_files = tuple(image_files)
        self.directory = directory

    def run(self):
        try:
            ordered = sorted(self.image_files, key=natural_image_key)
            if not self.isInterruptionRequested():
                self.completed.emit(ordered)
        except Exception as error:
            self.failed.emit(str(error))


class GridImageDeleteWorker(QThread):
    """Delete a checked snapshot; never access scene or widget state here."""
    progress = pyqtSignal(int, int)
    completed = pyqtSignal(object)

    def __init__(self, image_files, directory, parent=None):
        super().__init__(parent)
        self.image_files = tuple(image_files)
        self.directory = directory

    def run(self):
        deleted, failures = [], []
        total = len(self.image_files)
        for index, image_file in enumerate(self.image_files, 1):
            if self.isInterruptionRequested():
                break
            try:
                delete_image_label_pair(image_file)
                deleted.append(image_file)
            except Exception as error:
                failures.append(f"{os.path.basename(image_file)}: {error}")
                logger.error("Could not delete grid image pair %s: %s", image_file, error)
            if index % 25 == 0 or index == total:
                self.progress.emit(index, total)
        self.completed.emit({"deleted_files": deleted, "delete_failures": failures,
                             "directory": self.directory,
                             "canceled": self.isInterruptionRequested()})


class DatasetDirectoryScanWorker(QThread):
    """Read an image directory off the UI thread after its contents change."""

    completed = pyqtSignal(str, int, object, bool)
    failed = pyqtSignal(str, int, str)

    def __init__(self, directory, generation, known_files=None, parent=None):
        super().__init__(parent)
        self.directory = os.path.abspath(str(directory or ""))
        self.generation = int(generation)
        self.known_files = tuple(known_files or ())

    def run(self):
        try:
            image_files = []
            with os.scandir(self.directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_SUFFIXES:
                            image_files.append(os.path.abspath(entry.path).replace("\\", "/"))
                    except OSError:
                        continue
            known_set = {
                os.path.abspath(str(path)).replace("\\", "/")
                for path in self.known_files
                if path
            }
            scanned_set = set(image_files)
            changed = scanned_set != known_set
            self.completed.emit(
                self.directory.replace("\\", "/"),
                self.generation,
                image_files if changed else [],
                changed,
            )
        except Exception as exc:
            self.failed.emit(self.directory.replace("\\", "/"), self.generation, str(exc))


class FileRevealSignals(QObject):
    completed = pyqtSignal(str, bool, str)


class FileRevealWorker(QRunnable):
    def __init__(self, image_file):
        super().__init__()
        self.image_file = image_file
        self.signals = FileRevealSignals()

    def run(self):
        try:
            from darkfusion_dataset_operations import reveal_file
            selected = reveal_file(self.image_file)
            self.signals.completed.emit(self.image_file, selected, "")
        except Exception as error:
            self.signals.completed.emit(self.image_file, False, str(error))


class BulkMoveWorker(QThread):
    progress = pyqtSignal(int, int)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source_dir, target_dir, image_suffixes=None, max_workers=None, parent=None):
        super().__init__(parent)
        self.source_dir = self._normalize_path(source_dir)
        self.target_dir = self._normalize_path(target_dir)
        self.image_suffixes = tuple(str(ext).lower() for ext in (image_suffixes or IMAGE_SUFFIXES))
        self.max_workers = max_workers
        self._cancel_requested = False
        self._destination_lock = threading.Lock()
        self._reserved_destinations = set()

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            if not self.source_dir or not os.path.isdir(self.source_dir):
                self.failed.emit(f"Source directory does not exist: {self.source_dir}")
                return

            os.makedirs(self.target_dir, exist_ok=True)
            file_pairs = self._collect_file_pairs()
            total = len(file_pairs)
            moved = 0
            skipped = 0
            failures = []
            self.progress.emit(0, total)

            if total == 0:
                self.completed.emit({"moved": 0, "skipped": 0, "failed": [], "total": 0, "canceled": False})
                return

            worker_count = self._worker_count()
            pending = set()
            iterator = iter(file_pairs)
            pending_limit = max(worker_count * 4, 16)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                def fill_pending():
                    while not self._cancel_requested and len(pending) < pending_limit:
                        try:
                            pair = next(iterator)
                        except StopIteration:
                            break
                        pending.add(executor.submit(self._move_file_pair, pair))

                fill_pending()
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for future in done:
                        try:
                            if future.result():
                                moved += 1
                            else:
                                skipped += 1
                        except Exception as e:
                            failures.append(str(e))

                    processed = moved + skipped + len(failures)
                    if processed == total or processed % 50 == 0:
                        self.progress.emit(processed, total)

                    fill_pending()

            self.progress.emit(moved + skipped + len(failures), total)
            self.completed.emit({
                "moved": moved,
                "skipped": skipped,
                "failed": failures,
                "total": total,
                "canceled": self._cancel_requested,
            })
        except Exception as e:
            self.failed.emit(str(e))

    def _collect_file_pairs(self):
        pairs = []
        with os.scandir(self.source_dir) as entries:
            for entry in entries:
                if self._cancel_requested:
                    break

                if not entry.is_file():
                    continue

                if os.path.splitext(entry.name)[1].lower() not in self.image_suffixes:
                    continue

                img_src = self._normalize_path(entry.path)
                txt_src = os.path.splitext(img_src)[0] + ".txt"
                img_dst = self._normalize_path(os.path.join(self.target_dir, entry.name))
                txt_dst = os.path.splitext(img_dst)[0] + ".txt"
                pairs.append((img_src, txt_src, img_dst, txt_dst))

        return pairs

    def _worker_count(self):
        if self.max_workers:
            return max(1, int(self.max_workers))

        source_drive = os.path.splitdrive(self.source_dir)[0].lower()
        target_drive = os.path.splitdrive(self.target_dir)[0].lower()
        cpu_count = os.cpu_count() or 8

        if source_drive and target_drive and source_drive != target_drive:
            return max(2, min(8, cpu_count))

        return max(2, min(16, cpu_count))

    def _move_file_pair(self, pair):
        src_img, src_txt, requested_img, _requested_txt = pair

        if self._cancel_requested:
            return False

        dst_img, dst_txt = self._reserve_destination_pair(requested_img)
        image_moved = False
        label_moved = False
        label_created = False
        try:
            shutil.move(src_img, dst_img)
            image_moved = True
            if os.path.exists(src_txt):
                shutil.move(src_txt, dst_txt)
                label_moved = True
            else:
                Path(dst_txt).touch(exist_ok=False)
                label_created = True
            return True
        except Exception:
            # Keep the image and annotation together if the second half of the
            # move fails.  A missing source label is represented by an empty
            # destination label so every moved image remains a complete pair.
            try:
                if label_moved and os.path.exists(dst_txt) and not os.path.exists(src_txt):
                    shutil.move(dst_txt, src_txt)
                elif label_created and os.path.exists(dst_txt):
                    os.remove(dst_txt)
                if image_moved and os.path.exists(dst_img) and not os.path.exists(src_img):
                    shutil.move(dst_img, src_img)
            except Exception as rollback_error:
                logger.error("Could not roll back file-pair move for %s: %s", src_img, rollback_error)
            raise
        finally:
            with self._destination_lock:
                self._reserved_destinations.discard(os.path.normcase(dst_img))
                self._reserved_destinations.discard(os.path.normcase(dst_txt))

    def _reserve_destination_pair(self, requested_img):
        requested = Path(requested_img)
        with self._destination_lock:
            counter = 0
            while True:
                suffix = "" if counter == 0 else f"_{counter}"
                candidate_img = requested.with_name(
                    f"{requested.stem}{suffix}{requested.suffix}"
                )
                candidate_txt = candidate_img.with_suffix(".txt")
                keys = {
                    os.path.normcase(self._normalize_path(candidate_img)),
                    os.path.normcase(self._normalize_path(candidate_txt)),
                }
                if (
                    not candidate_img.exists()
                    and not candidate_txt.exists()
                    and not (keys & self._reserved_destinations)
                ):
                    self._reserved_destinations.update(keys)
                    return self._normalize_path(candidate_img), self._normalize_path(candidate_txt)
                counter += 1

    _normalize_path = staticmethod(normalize_dataset_path)


class FilteredExportWorker(QThread):
    progress = pyqtSignal(int, int)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, files_to_export, export_folder, class_index, class_name,
                 placeholder_paths=None, move_source=False, parent=None):
        super().__init__(parent)
        self.files_to_export = list(files_to_export or [])
        self.export_folder = self._normalize_path(export_folder)
        self.class_index = int(class_index)
        self.class_name = str(class_name or "export")
        self.placeholder_paths = set(placeholder_paths or [])
        self.move_source = bool(move_source)
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        total = len(self.files_to_export)
        copied = 0
        skipped = 0
        failures = []
        completed_sources = []
        try:
            os.makedirs(self.export_folder, exist_ok=True)
            self.progress.emit(0, total)

            if total == 0:
                self._write_classes_file()
                self.completed.emit({
                    "copied": 0,
                    "skipped": 0,
                    "failed": [],
                    "total": 0,
                    "canceled": False,
                    "export_folder": self.export_folder,
                    "class_name": self.class_name,
                    "operation": "move" if self.move_source else "copy",
                    "completed_sources": [],
                })
                return

            worker_count = self._worker_count()
            pending = set()
            iterator = iter(self.files_to_export)
            pending_limit = max(worker_count * 4, 16)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                def fill_pending():
                    while not self._cancel_requested and len(pending) < pending_limit:
                        try:
                            source = next(iterator)
                        except StopIteration:
                            break
                        pending.add(executor.submit(self._export_one, source))

                fill_pending()
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for future in done:
                        try:
                            result = future.result()
                            if result:
                                copied += 1
                                completed_sources.append(self._normalize_path(result))
                            else:
                                skipped += 1
                        except Exception as e:
                            failures.append(str(e))

                    processed = copied + skipped + len(failures)
                    if processed == total or processed % 50 == 0:
                        self.progress.emit(processed, total)

                    fill_pending()

            self._write_classes_file()
            self.progress.emit(copied + skipped + len(failures), total)
            self.completed.emit({
                "copied": copied,
                "skipped": skipped,
                "failed": failures,
                "total": total,
                "canceled": self._cancel_requested,
                "export_folder": self.export_folder,
                "class_name": self.class_name,
                "operation": "move" if self.move_source else "copy",
                "completed_sources": completed_sources,
            })
        except Exception as e:
            if completed_sources:
                failures.append(str(e))
                self.completed.emit({
                    "copied": copied,
                    "skipped": skipped,
                    "failed": failures,
                    "total": total,
                    "canceled": self._cancel_requested,
                    "export_folder": self.export_folder,
                    "class_name": self.class_name,
                    "operation": "move" if self.move_source else "copy",
                    "completed_sources": completed_sources,
                })
            else:
                self.failed.emit(str(e))

    def _export_one(self, src_image_path):
        if self._cancel_requested:
            return False

        normalized_src_image = self._normalize_path(src_image_path)

        if not normalized_src_image or normalized_src_image in self.placeholder_paths:
            return False

        if not os.path.exists(normalized_src_image):
            raise FileNotFoundError(f"Missing image file: {normalized_src_image}")

        os.makedirs(self.export_folder, exist_ok=True)
        dst_image_path = self._available_destination(normalized_src_image)
        dst_txt_path = os.path.splitext(dst_image_path)[0] + ".txt"

        src_txt_path = os.path.splitext(normalized_src_image)[0] + ".txt"
        if self.move_source:
            shutil.move(normalized_src_image, dst_image_path)
            try:
                if os.path.exists(src_txt_path):
                    shutil.move(src_txt_path, dst_txt_path)
                else:
                    Path(dst_txt_path).touch()
            except Exception:
                try:
                    if os.path.exists(dst_image_path) and not os.path.exists(normalized_src_image):
                        shutil.move(dst_image_path, normalized_src_image)
                except Exception as rollback_error:
                    logger.error(
                        "Could not roll back blank-image move %s: %s",
                        normalized_src_image,
                        rollback_error,
                    )
                raise
            return normalized_src_image

        shutil.copy2(normalized_src_image, dst_image_path)
        lines = []
        if os.path.exists(src_txt_path):
            with open(src_txt_path, "r", encoding="utf-8", errors="ignore") as file:
                lines = file.readlines()

        output_lines = []

        if self.class_index == -1:
            output_lines = []
        elif self.class_index == -2:
            output_lines = [line.rstrip("\n") for line in lines if line.strip()]
        else:
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                parts = stripped.split()
                try:
                    if int(parts[0]) == self.class_index:
                        parts[0] = "0"
                        output_lines.append(" ".join(parts))
                except Exception:
                    continue

        with open(dst_txt_path, "w", encoding="utf-8") as file:
            if output_lines:
                file.write("\n".join(output_lines) + "\n")

        return normalized_src_image

    def _available_destination(self, src_image_path):
        source = Path(src_image_path)
        candidate = Path(self.export_folder) / source.name
        counter = 1
        while candidate.exists() or candidate.with_suffix(".txt").exists():
            candidate = Path(self.export_folder) / f"{source.stem}_{counter}{source.suffix}"
            counter += 1
        return self._normalize_path(candidate)

    def _write_classes_file(self):
        classes_txt_path = os.path.join(self.export_folder, "classes.txt")
        with open(classes_txt_path, "w", encoding="utf-8") as file:
            file.write(f"{self.class_name}\n")

    def _worker_count(self):
        source_drive = ""
        for path in self.files_to_export:
            if path:
                source_drive = os.path.splitdrive(self._normalize_path(path))[0].lower()
                break

        target_drive = os.path.splitdrive(self.export_folder)[0].lower()
        cpu_count = os.cpu_count() or 8

        if source_drive and target_drive and source_drive != target_drive:
            return max(2, min(6, cpu_count))

        return max(2, min(8, cpu_count))

    _normalize_path = staticmethod(normalize_dataset_path)


class ClearLabelsWorker(QThread):
    """Clear a fixed label-file snapshot without doing disk I/O on the GUI thread."""

    progress = pyqtSignal(int, int)

    def __init__(self, targets, class_index, write_lines, parent=None):
        super().__init__(parent)
        self.targets = list(targets)
        self.class_index = int(class_index)
        self.write_lines = write_lines  # Filesystem-only atomic label writer.
        self.result = dict(processed=0, changed=[], failures=[], canceled=False)

    def run(self):
        total = len(self.targets)
        last_progress = time.monotonic()
        self.progress.emit(0, total)
        for index, (image_file, label_file) in enumerate(self.targets, start=1):
            if self.isInterruptionRequested():
                self.result["canceled"] = True
                break
            try:
                if self.class_index == -2 or not os.path.isfile(image_file) or not os.path.isfile(label_file):
                    continue
                source_version = label_file_version(label_file)
                try:
                    with open(label_file, "r", encoding="utf-8-sig") as stream:
                        lines = stream.readlines()
                except UnicodeDecodeError:
                    with open(label_file, "r", encoding="latin-1") as stream:
                        lines = stream.readlines()
                kept = []
                if self.class_index != -1:
                    for line in lines:
                        try:
                            remove = int(float(line.split()[0])) == self.class_index
                        except (ValueError, IndexError):
                            remove = False  # Keep unknown rows when clearing one class.
                        if not remove:
                            kept.append(line)
                if kept == lines:
                    continue
                if self.isInterruptionRequested():
                    self.result["canceled"] = True
                    break
                if not os.path.isfile(image_file) or label_file_version(label_file) != source_version:
                    raise OSError("The image or labels changed while clearing was prepared.")
                if not self.write_lines(label_file, kept):
                    raise OSError("The annotation file could not be replaced.")
                self.result["changed"].append(label_file)
            except Exception as error:
                logging.error("Clear Labels kept %s unchanged: %s", label_file, error)
                self.result["failures"].append(f"{os.path.basename(label_file)}: {error}")
            finally:
                self.result["processed"] = index
                now = time.monotonic()
                if index == total or now - last_progress >= .05:
                    self.progress.emit(index, total)
                    last_progress = now
        self.progress.emit(self.result["processed"], total)


class ClassLabelRemapWorker(QThread):
    """Remap dataset class IDs without blocking the Qt event loop."""

    progress = pyqtSignal(int, int)
    completed = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, directory, id_map, parent=None, *, classes_path=None, classes=None):
        super().__init__(parent)
        self.directory = str(directory or "")
        self.id_map = {int(key): value for key, value in dict(id_map or {}).items()}
        self.classes_path = classes_path
        self.classes = None if classes is None else list(classes)

    def _label_files(self):
        skip_names = {"classes.txt", "obj.names", "names.txt", "model_classes.txt"}
        image_stems = set()
        text_paths = []
        with os.scandir(self.directory) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                stem, suffix = os.path.splitext(entry.name)
                suffix = suffix.lower()
                if suffix in OUTPUT_IMAGE_SUFFIXES:
                    image_stems.add(stem)
                elif suffix == ".txt" and entry.name.lower() not in skip_names:
                    text_paths.append((stem, entry.path))
        return sorted(
            path for stem, path in text_paths
            if not image_stems or stem in image_stems
        )

    def run(self):
        transaction = None
        try:
            label_files = self._label_files()
            transaction = DatasetFileTransaction(self.directory)
            total = len(label_files)
            changed_files = 0
            self.progress.emit(0, total)

            for index, label_file in enumerate(label_files, start=1):
                with open(label_file, "rb") as stream:
                    original_bytes = stream.read()
                original_lines = original_bytes.decode("utf-8-sig").splitlines()

                new_lines = []
                changed = False
                for line in original_lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    try:
                        old_id = int(float(parts[0]))
                    except Exception:
                        new_lines.append(stripped)
                        continue

                    if old_id in self.id_map:
                        new_id = self.id_map[old_id]
                        if new_id != old_id:
                            changed = True
                        if new_id is None:
                            continue
                        parts[0] = str(int(new_id))
                    new_lines.append(" ".join(parts))

                if changed:
                    text = "\n".join(new_lines) + ("\n" if new_lines else "")
                    transaction.stage(label_file, text.encode("utf-8"), expected=original_bytes, check_expected=True)
                    changed_files += 1

                if index == total or index % 100 == 0:
                    self.progress.emit(index, total)

            if self.classes is not None:
                if not self.classes_path:
                    raise ValueError("A class-list path is required for a class-list change.")
                text = "\n".join(self.classes) + ("\n" if self.classes else "")
                transaction.stage(self.classes_path, text.encode("utf-8"))
            transaction.commit()
            self.completed.emit(changed_files)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if transaction is not None:
                transaction.close()


class ImageConverterRunnableSignals(QObject):
    finished = pyqtSignal()  # Signal emitted when a file is processed


class ImageConverterRunnable(QRunnable):

    def __init__(self, directory, file, target_format, target_directory):
        super().__init__()
        self.signals = ImageConverterRunnableSignals()
        self.directory = directory
        self.file = file
        self.target_extension = normalize_image_extension(target_format)
        self.target_format = image_format_for_extension(self.target_extension)
        self.target_directory = target_directory

    def run(self):
        try:
            reader = QImageReader(os.path.join(self.directory, self.file))
            image = reader.read()
            if image.isNull():
                raise Exception(f"Could not read {self.file}")

            target_file = os.path.join(
                self.target_directory, f"{os.path.splitext(self.file)[0]}{self.target_extension}"
            )
            writer = QImageWriter(target_file, self.target_format.encode())
            if not writer.write(image):
                raise Exception(f"Could not write {self.file}")

            txt_file = os.path.join(self.directory, f"{os.path.splitext(self.file)[0]}.txt")
            if os.path.isfile(txt_file):
                shutil.copy(txt_file, self.target_directory)
        finally:
            logger.info(f"Finished processing {self.file}")
            self.signals.finished.emit()  # Emit finished signal


class VideoProcessor(QObject):
    progress_updated = pyqtSignal(int)  # Signal to indicate progress updates

    def __init__(self):
        super().__init__()
        self.stop_processing = False
        self.image_extension = ".jpg"

    def set_image_extension(self, extension):
        self.image_extension = normalize_image_extension(extension)

    def process_video(self, video_path, output_dir, extract_rate=1, custom_size=None, image_extension=None):
        """
        Process a video, extracting frames at the specified rate.

        Args:
            video_path (str): Path to the input video.
            output_dir (str): Directory to save extracted frames.
            extract_rate (int): Extract every nth frame.
            custom_size (tuple): Custom resize dimensions (width, height).
        """
        image_extension = normalize_image_extension(image_extension or self.image_extension)
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error(f"Unable to open video file: {video_path}")
            video.release()
            return

        try:
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            os.makedirs(output_dir, exist_ok=True)

            while True:
                ret, frame = video.read()
                if not ret:
                    break

                if custom_size:
                    frame = cv2.resize(frame, custom_size, interpolation=cv2.INTER_LANCZOS4)

                if frame_count % extract_rate == 0:
                    frame_path, label_path = extraction_frame_paths(
                        output_dir, Path(video_path).stem, frame_count, image_extension
                    )
                    if not existing_labeled_extraction_image(frame_path, label_path):
                        save_cv_image(frame_path, frame)

                frame_count += 1
                progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
                self.progress_updated.emit(progress)

                if self.stop_processing:
                    logger.info("Video processing stopped.")
                    break
        finally:
            video.release()


class GUIVideoProcessor(VideoProcessor):

    def __init__(self, progress_bar: QProgressBar):
        super().__init__()
        self.videos = []
        self.extract_all_frames = False
        self.custom_frame_count = None
        self.custom_size = None
        self.output_path = ""
        self.label_progress = progress_bar
        self.stop_processing = False
        self.original_size = True

        self.progress_updated.connect(self.label_progress.setValue)

    def stop(self):
        """
        Stop the current processing operation.
        """
        self.stop_processing = True
        logger.info("Video processing stopped by user.")

    def add_video(self, video_path):
        if video_path not in self.videos:
            self.videos.append(video_path)
            logger.info(f"Added video: {video_path}")

    def remove_video(self):
        if self.videos:
            removed_video = self.videos.pop()
            logger.info(f"Removed video: {removed_video}")
            return removed_video
        else:
            logger.info("No videos to remove.")
            return None

    def set_output_path(self, path):
        self.output_path = path

    def set_extract_all_frames(self, value):
        self.extract_all_frames = value

    def set_custom_frame_count(self, count):
        self.custom_frame_count = count

    def set_custom_size(self, size):
        self.custom_size = size

    def set_original_size(self, value):
        self.original_size = bool(value)
        if self.original_size:
            self.custom_size = None

    def run(self):
        """
        Processes all videos added to the list.
        """
        if not self.output_path:
            QMessageBox.warning(None, "Warning", "Output directory not set!")
            return

        self.label_progress.reset()
        self.label_progress.setRange(0, 100)

        for idx, video_path in enumerate(self.videos):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(self.output_path, f"{video_name}_Frames")


            extract_rate = 1 if self.extract_all_frames else (self.custom_frame_count or 1)
            try:
                self.process_video(
                    video_path,
                    output_dir,
                    extract_rate,
                    self.custom_size,
                    self.image_extension,
                )
            except Exception as e:
                logger.error(f"Error processing video {video_path}: {e}")

            overall_progress = int(((idx + 1) / max(1, len(self.videos))) * 100)
            self.label_progress.setValue(overall_progress)

        QMessageBox.information(None, "Info", "Video frame extraction completed.")


class FrameExtractionThread(QThread):
    progress_range = pyqtSignal(int, int, str)
    progress_value = pyqtSignal(int)
    progress_format = pyqtSignal(str)
    extraction_finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    warning = pyqtSignal(str)
    checkpoint_resume_request = pyqtSignal(dict)

    def __init__(self, owner, config, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.config = dict(config or {})
        self._running = True
        self._lock = threading.Lock()
        self._resume_checkpoint = False
        self._checkpoint_frame_count = 0
        self._checkpoint_saved_count = 0
        self._resume_decision = threading.Event()
        self._checkpoint = ExtractionCheckpoint(self.config)
        self._last_completed_frame = -1
        self._saved_total = 0
        self._checkpoint_bucket = 0

    def stop(self):
        with self._lock:
            self._running = False

        # Stop must also release a job waiting for the resume prompt.
        self._resume_decision.set()

    def is_stopping(self):
        with self._lock:
            return not self._running

    def should_continue(self):
        if self.is_stopping():
            return False

        owner = self.owner
        if owner is not None and not getattr(owner, "extracting_frames", False):
            return False

        return True

    def set_resume_checkpoint(self, resume, frame_count, saved_count=0):
        """Set whether to resume from checkpoint and the frame number to start from."""
        with self._lock:
            self._resume_checkpoint = bool(resume)
            self._checkpoint_frame_count = max(0, int(frame_count or 0))
            self._checkpoint_saved_count = max(0, int(saved_count or 0)) if resume else 0
        self._resume_decision.set()

    def _checkpoint_path(self, output_dir):
        """Get path to checkpoint file."""
        return os.path.join(output_dir, '.extraction_checkpoint.json')

    def _load_checkpoint(self, output_dir):
        """Only offer checkpoints verified against this source and settings."""
        return self._checkpoint.load()

    def _save_checkpoint(self, output_dir, frame_count, saved_count):
        return self._checkpoint.save(int(frame_count), int(saved_count))

    def _record_completed_frame(self, frame_count, saved_count):
        self._last_completed_frame = int(frame_count)
        self._saved_total = self._checkpoint_saved_count + int(saved_count)
        bucket = self._saved_total // 50
        if bucket > self._checkpoint_bucket:
            self._save_checkpoint(self.config.get("output_dir"), frame_count, self._saved_total)
            self._checkpoint_bucket = bucket

    def run(self):
        summary = {
            "kind": self.config.get("kind", "video"),
            "frames_read": 0,
            "frames_saved": 0,
            "output_dir": self.config.get("output_dir", ""),
            "stopped": False,
        }

        output_dir = self.config.get("output_dir", "")
        try:
            checkpoint = self._load_checkpoint(output_dir)
            if checkpoint and not self._resume_decision.is_set():
                self.checkpoint_resume_request.emit(checkpoint)
                # A queued Qt signal does not wait for the GUI answer. Do not
                # open/decode/write anything until the user has decided.
                while self.should_continue() and not self._resume_decision.wait(0.1):
                    pass
            if not self.should_continue():
                return
            if self._resume_checkpoint:
                self._last_completed_frame = self._checkpoint_frame_count
                self._saved_total = self._checkpoint_saved_count
                self._checkpoint_bucket = self._saved_total // 50
            kind = self.config.get("kind")
            if kind == "desktop":
                self._extract_desktop(summary)
            elif kind == "camera":
                self._extract_capture(summary, cv2.CAP_DSHOW)
            else:
                self._extract_capture(summary, None)
        except Exception as e:
            self.error.emit(str(e))
            summary["error"] = str(e)
        finally:
            summary["stopped"] = self.is_stopping() or (self.owner is not None and not getattr(self.owner, "extracting_frames", False))
            summary["frames_saved_total"] = self._saved_total
            try:
                if summary.get("error") or summary["stopped"]:
                    # Include the final completed frame, not the failed write.
                    self._save_checkpoint(output_dir, self._last_completed_frame, self._saved_total)
                else:
                    self._checkpoint.complete()
            except OSError as error:
                summary["checkpoint_error"] = str(error)
                self.warning.emit(f"Could not update extraction resume checkpoint: {error}")
            self.extraction_finished.emit(summary)

    def _should_extract_frame(self, frame_count):
        try:
            step = int(self.config.get("frame_step", 1) or 1)
        except Exception:
            step = 1

        return int(frame_count) % max(1, step) == 0

    def _transform_frame(self, frame):
        # ROI only filters saved auto-labels; it never crops the retained frame.
        return transform_video_frame_by_settings(frame, dict(self.config))

    def _save_raw_frame(self, frame, frame_count, output_dir):
        save_frame = self._transform_frame(frame)
        if save_frame is None or not isinstance(save_frame, np.ndarray) or save_frame.size == 0:
            self.warning.emit(f"Frame {frame_count} became invalid after transform, skipping.")
            return False

        prefix = self.config.get("video_name") or "session"
        image_ext = normalize_image_extension(self.config.get("image_extension", ".jpg"))
        frame_filepath, annotation_filepath = extraction_frame_paths(
            output_dir, prefix, frame_count, image_ext
        )

        # A labeled frame is immutable during extraction: overwriting its image
        # after crop/resize settings change would silently invalidate the label.
        try:
            if existing_labeled_extraction_image(frame_filepath, annotation_filepath):
                return True
        except OSError:
            pass

        return save_extraction_outputs(frame_filepath, save_frame, save_cv_image)

    def _save_frame(self, frame, frame_count, output_dir):
        if not self._should_extract_frame(frame_count):
            return False

        if self.config.get("model_loaded"):
            owner = self.owner
            if owner is None:
                return False

            return bool(owner.save_frame_on_inference(
                frame=frame,
                results=None,
                frame_count=frame_count,
                output_dir=output_dir,
            ))

        return self._save_raw_frame(frame, frame_count, output_dir)

    def _extract_capture(self, summary, backend):
        source = self.config.get("capture_source", self.config.get("source"))
        output_dir = self.config.get("output_dir")
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)
        saved_count = read_count = 0
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video source: {self.config.get('original_source', source)}")
            if backend == cv2.CAP_DSHOW:
                properties = {
                    cv2.CAP_PROP_FRAME_WIDTH: 1920, cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                    cv2.CAP_PROP_FPS: float(self.config.get("capture_fps", 60)),
                    cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
                    cv2.CAP_PROP_BUFFERSIZE: 1 if self.config.get("buffer_mode") == "latency" else 5,
                }
                for key, value in properties.items():
                    try:
                        cap.set(key, value)
                    except Exception:
                        pass  # Capture drivers may ignore unsupported properties.
            try:
                total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            except (TypeError, ValueError, OverflowError):
                total_frames = 0
            has_known_total = total_frames > 0 and backend is None
            start_frame = max(0, int(self.config.get("start_frame") or 0))
            end_value = self.config.get("end_frame")
            end_frame = max(0, int(end_value)) if end_value is not None else None
            if end_frame is not None and end_frame < start_frame:
                start_frame, end_frame = end_frame, start_frame
            if has_known_total:
                start_frame = min(start_frame, total_frames - 1)
                end_frame = min(end_frame, total_frames - 1) if end_frame is not None else total_frames - 1
            if self._resume_checkpoint:
                start_frame = max(start_frame, self._checkpoint_frame_count + 1)
                logger.info("Resuming extraction from frame %s", start_frame)
            if backend is None and start_frame > 0:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
                    raise RuntimeError(f"Cannot seek to extraction frame {start_frame}; no frames were saved.")
                actual_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if actual_position != start_frame:
                    raise RuntimeError(f"Video seek reached frame {actual_position}, not {start_frame}; extraction stopped.")
            frame_count = start_frame
            total_work = max(1, end_frame - start_frame + 1) if has_known_total else 0
            self.progress_range.emit(0, 100 if has_known_total else 0, "%p%" if has_known_total else "Reading capture...")
            self.progress_value.emit(0)
            while self.should_continue() and (not has_known_total or frame_count < total_frames):
                if end_frame is not None and frame_count > end_frame:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    if self.is_stopping():
                        break
                    if backend == cv2.CAP_DSHOW:
                        raise RuntimeError("Camera/capture-card read failed. Extraction stopped; completed frames were kept.")
                    if has_known_total and frame_count < total_frames:
                        raise RuntimeError(f"Video read failed at frame {frame_count}; resume checkpoint was kept.")
                    break
                read_count += 1
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    raise RuntimeError(f"Invalid decoded frame at {frame_count}; extraction stopped.")
                if self._save_frame(frame, frame_count, output_dir):
                    saved_count += 1
                self._record_completed_frame(frame_count, saved_count)
                frame_count += 1
                self._emit_progress(read_count, total_work, has_known_total, saved_count)
        finally:
            cap.release()
            summary["frames_read"] = read_count
            summary["frames_saved"] = saved_count

    def _extract_desktop(self, summary):
        output_dir = self.config.get("output_dir")
        os.makedirs(output_dir, exist_ok=True)
        self.progress_range.emit(0, 0, "Reading desktop...")
        monitor_index = int(self.config.get("monitor_index", 1) or 1)
        frame_count = saved_count = 0
        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                monitor_index = max(0, min(monitor_index, len(monitors) - 1))
                monitor = monitors[monitor_index]
                while self.should_continue():
                    frame = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
                    if self._save_frame(frame, frame_count, output_dir):
                        saved_count += 1
                    self._record_completed_frame(frame_count, saved_count)
                    frame_count += 1
                    self._emit_progress(frame_count, 0, False, saved_count)
                    self.msleep(1)
        finally:
            summary["frames_read"] = frame_count
            summary["frames_saved"] = saved_count

    def _emit_progress(self, frame_count, total_frames, has_known_total, saved_count):
        if has_known_total and total_frames > 0:
            percent = int(min(100, (frame_count / total_frames) * 100))
            if percent != getattr(self, "_last_progress_percent", None):
                self._last_progress_percent = percent
                self.progress_value.emit(percent)
            return

        if saved_count and saved_count % 10 == 0 and saved_count != getattr(self, "_last_progress_saved", None):
            self._last_progress_saved = saved_count
            self.progress_format.emit(f"Saved {saved_count} frames...")


def on_clear_json_clicked(self):
    """
        This method is triggered when the 'Clear JSON' button is clicked.
        It deletes all .json files in the current image directory.
        """
    if self.image_directory:  # Check if the image directory is set
        json_files = glob.glob(os.path.join(self.image_directory, '*.json'))
        if json_files:  # Check if there are any .json files to delete
            for json_file in json_files:
                try:
                    os.remove(json_file)
                    logger.info(f"Deleted {json_file}")
                except Exception as e:
                    logger.error(f"Error deleting {json_file}: {str(e)}")
            QMessageBox.information(None, 'JSON Files Cleared', f"All JSON files have been deleted from {self.image_directory}.")
        else:
            QMessageBox.information(None, 'No JSON Files', "No JSON files found in the current directory.")
    else:
        QMessageBox.warning(None, 'Directory Not Set', "The image directory has not been set. Please open an image directory first.")


def on_button_click(self):
    return self.start_create_negative_images()


def setup_dataset_image_converter_controls(self):
    combo = getattr(self, "select_format", None)
    if isinstance(combo, QtWidgets.QComboBox):
        current = normalize_dataset_conversion_extension(combo.currentText())
        with blocked_signals(combo):
            combo.clear()
            for ext in DATASET_CONVERSION_TARGET_SUFFIXES:
                combo.addItem(ext)
            index = combo.findText(current)
            combo.setCurrentIndex(index if index >= 0 else combo.findText(".jpg"))
        combo.setToolTip("Target image format for current-dataset conversion.")

    if hasattr(self, "convert_image") and isinstance(self.convert_image, QtWidgets.QPushButton):
        self.convert_image.setText("Convert Dataset")
        self.convert_image.setToolTip("Convert images from the currently opened dataset only.")


def current_dataset_directory(self):
    def valid_directory(path):
        path = str(path or "").strip()
        if not path:
            return ""
        path = self.normalize_path(path)
        if os.path.isfile(path):
            path = self.normalize_path(os.path.dirname(path))
        if os.path.isdir(path):
            return path
        return ""

    for path in (
        getattr(self, "image_directory", ""),
        getattr(self, "last_image_directory", ""),
        getattr(self, "current_file", ""),
    ):
        directory = valid_directory(path)
        if directory:
            return directory

    for collection_name in ("filtered_image_files", "image_files"):
        for image_file in list(getattr(self, collection_name, []) or []):
            if not image_file or self.is_placeholder_file(image_file):
                continue
            directory = valid_directory(image_file)
            if directory:
                self.image_directory = directory
                return directory

    # ``last_dir`` is dialog/project history, not an active dataset.  Using
    # it here silently reactivated the previous  dataset when a tool (such
    # as the training evaluator) merely asked whether a dataset was open.
    # Only Open Image/Video or another explicit user dataset action may
    # establish the current GraphicsView dataset.
    return ""


def collect_current_dataset_conversion_files(self):
    dataset_dir = self.current_dataset_directory()
    if not dataset_dir:
        return []
    dataset_dir = self.normalize_path(dataset_dir)

    files = []
    seen = set()

    for image_file in list(getattr(self, "image_files", []) or []):
        if not image_file or self.is_placeholder_file(image_file):
            continue

        image_file = self.normalize_path(image_file)
        if image_file in seen or not os.path.isfile(image_file):
            continue

        try:
            common_dir = self.normalize_path(os.path.commonpath([dataset_dir, image_file]))
            if common_dir != dataset_dir:
                continue
        except Exception:
            continue

        if os.path.splitext(image_file)[1].lower() in DATASET_IMAGE_SUFFIXES:
            files.append(image_file)
            seen.add(image_file)

    if files:
        return files

    try:
        for entry in os.scandir(dataset_dir):
            if not entry.is_file():
                continue
            if self.is_placeholder_file(entry.name) or self.is_placeholder_file(entry.path):
                continue
            if os.path.splitext(entry.name)[1].lower() in DATASET_IMAGE_SUFFIXES:
                image_file = self.normalize_path(entry.path)
                if image_file not in seen:
                    files.append(image_file)
                    seen.add(image_file)
    except Exception as e:
        logger.error(f"Failed scanning dataset images for conversion: {e}")

    return sorted(files)


def dataset_conversion_output_path(self, source_path, output_dir, target_extension, stem_counts, used_paths):
    source = Path(source_path)
    stem = source.stem
    if stem_counts.get(stem, 0) > 1:
        stem = f"{stem}__{source.suffix.lower().lstrip('.')}"

    candidate = os.path.join(output_dir, f"{stem}{target_extension}")
    counter = 2
    while candidate in used_paths:
        candidate = os.path.join(output_dir, f"{stem}_{counter}{target_extension}")
        counter += 1

    used_paths.add(candidate)
    return candidate


def copy_dataset_conversion_metadata(self, dataset_dir, output_dir):
    copied = 0
    for name in DATASET_METADATA_FILES:
        source = resolve_dataset_metadata_path(dataset_dir, name, migrate=True)
        if not os.path.isfile(source):
            continue
        try:
            target = dataset_metadata_path(output_dir, name, create_parent=True)
            shutil.copy2(source, target)
            copied += 1
        except Exception as e:
            logger.warning(f"Failed copying dataset metadata {source}: {e}")

    for name in ("data.yaml", "dataset.yaml", "obj.names"):
        source = os.path.join(dataset_dir, name)
        if not os.path.isfile(source):
            continue
        try:
            shutil.copy2(source, os.path.join(output_dir, name))
            copied += 1
        except Exception as e:
            logger.warning(f"Failed copying dataset metadata {source}: {e}")

    return copied


def convert_current_dataset_images(
    self,
    target_extension=None,
    output_dir=None,
    source_files=None,
    overwrite=False,
    copy_labels=True,
):
    dataset_dir = self.current_dataset_directory()
    if not dataset_dir:
        QMessageBox.warning(self, "Dataset Converter", "Open an image dataset first.")
        return False

    target_extension = normalize_dataset_conversion_extension(
        target_extension
        or (
            self.select_format.currentText()
            if isinstance(getattr(self, "select_format", None), QtWidgets.QComboBox)
            else ".jpg"
        )
    )

    source_files = list(source_files or self.collect_current_dataset_conversion_files())
    if not source_files:
        QMessageBox.warning(self, "Dataset Converter", "No static dataset images were found to convert.")
        return False

    output_dir = output_dir or os.path.join(dataset_dir, f"converted_{target_extension.lstrip('.')}")
    output_dir = self.normalize_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stem_counts = Counter(Path(path).stem for path in source_files)
    used_paths = set()
    jobs = [
        (
            source_path,
            self.dataset_conversion_output_path(
                source_path,
                output_dir,
                target_extension,
                stem_counts,
                used_paths,
            ),
        )
        for source_path in source_files
    ]

    button = getattr(self, "convert_image", None)
    if isinstance(button, QtWidgets.QPushButton):
        button.setEnabled(False)

    self.set_label_progress(0, len(jobs), value=0, progress_format="%p%")
    logger.info(
        "Converting current dataset: %s image(s) -> %s (%s).",
        len(jobs),
        output_dir,
        target_extension,
    )

    converted = 0
    copied = 0
    skipped = 0
    failed = []

    def convert_one(source_path, target_path):
        try:
            if os.path.exists(target_path) and not overwrite:
                return "skipped", source_path, target_path

            if os.path.splitext(source_path)[1].lower() == target_extension:
                shutil.copy2(source_path, target_path)
                status = "copied"
            else:
                ok = convert_dataset_image_file(source_path, target_path, target_extension)
                status = "converted" if ok else "failed"

            if status in {"converted", "copied"} and copy_labels:
                source_label = os.path.splitext(source_path)[0] + ".txt"
                if os.path.isfile(source_label):
                    shutil.copy2(source_label, os.path.splitext(target_path)[0] + ".txt")

            return status, source_path, target_path

        except Exception as e:
            logger.error(f"Failed converting {source_path}: {e}")
            return "failed", source_path, target_path

    try:
        max_workers = max(1, min(8, (os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(convert_one, source, target) for source, target in jobs]
            total = len(futures)

            for index, future in enumerate(as_completed(futures), start=1):
                status, source_path, target_path = future.result()
                if status == "converted":
                    converted += 1
                elif status == "copied":
                    copied += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed.append(source_path)

                self.set_label_progress(0, total, value=index, progress_format="%p%")
                if index % 5 == 0:
                    QtWidgets.QApplication.processEvents()

        metadata_copied = self.copy_dataset_conversion_metadata(dataset_dir, output_dir) if copy_labels else 0

        message = (
            f"Output:\n{output_dir}\n\n"
                f"Converted: {converted}\n"
                f"Copied already target-format images: {copied}\n"
                f"Skipped existing: {skipped}\n"
                f"Metadata files copied: {metadata_copied}\n"
                f"Failed: {len(failed)}"
        )

        QMessageBox.information(self, "Dataset Converter", message)
        logger.info(message.replace("\n", " | "))
        return not failed

    finally:
        if isinstance(button, QtWidgets.QPushButton):
            button.setEnabled(True)
        self.reset_label_progress(100)


def convert_images(self):
    """Convert the current dataset using the selected format."""
    target_extension = normalize_dataset_conversion_extension(
        self.select_format.currentText()
        if isinstance(getattr(self, "select_format", None), QtWidgets.QComboBox)
        else ".jpg"
    )
    return self.convert_current_dataset_images(
        target_extension=target_extension,
        overwrite=False,
        copy_labels=True,
    )


def move_filtered_images(self):
    """
        Process filtered images into a class-specific folder.

        Behavior:
        - Blank images and their empty/missing labels are moved out of the dataset root
        - Other filters remain exports: source files stay and destination labels are generated

        Filter behavior:
        - Blanks: moves the image and its .txt; creates an empty .txt when missing
        - Size: copies image and copies label contents as-is
        - Normal class: copies image and writes only matching class lines, remapped to class 0
        """
    current_text = self.filter_class_spinbox.currentText()

    placeholder_candidates = {
        os.path.abspath('styles/images/default.png').replace("\\", "/"),
        os.path.abspath('styles/images/default_temp.png').replace("\\", "/"),
    }

    # Resolve filter
    if current_text.startswith("Blanks"):
        class_name = "Blanks"
        class_index = -1
    elif current_text.startswith("Size:"):
        class_name = current_text.replace("Size: ", "Size_").replace(" ", "_")
        class_index = -2
    else:
        try:
            class_index = int(current_text.split(":")[0].strip())
            class_name = self.class_names[class_index]
        except (ValueError, IndexError) as e:
            logger.warning(f"Invalid class filter format: {current_text} | {e}")
            return

    export_folder = os.path.join(self.image_directory, class_name)
    os.makedirs(export_folder, exist_ok=True)

    checked_files = []
    if hasattr(self, "grid_review_is_active") and self.grid_review_is_active():
        grid_model = getattr(self, "grid_thumbnail_model", None)
        checked_files = grid_model.checked_files() if grid_model is not None else []
    source_files = checked_files or list(self.filtered_image_files)
    files_to_export = [
        path for path in source_files
        if os.path.abspath(path).replace("\\", "/") not in placeholder_candidates
    ]
    if not files_to_export:
        QMessageBox.information(self, "Move Filtered", "No filtered images to export.")
        return

    active_worker = getattr(self, "_filtered_export_worker", None)
    if active_worker is not None:
        try:
            if active_worker.isRunning():
                QMessageBox.information(self, "Move Filtered", "A filtered export is already running.")
                return
        except RuntimeError:
            self._filtered_export_worker = None

    worker = FilteredExportWorker(
        files_to_export=files_to_export,
        export_folder=export_folder,
        class_index=class_index,
        class_name=class_name,
        placeholder_paths=placeholder_candidates,
        move_source=(class_index == -1),
        parent=self,
    )
    self._filtered_export_worker = worker
    self._filtered_export_grid_active = bool(
        hasattr(self, "grid_review_is_active") and self.grid_review_is_active()
    )
    worker.progress.connect(self.on_filtered_export_progress)
    worker.completed.connect(self.on_filtered_export_completed)
    worker.failed.connect(self.on_filtered_export_failed)
    worker.finished.connect(worker.deleteLater)

    if hasattr(self, "move_all_button"):
        self.move_all_button.setEnabled(False)

    selection_name = "checked" if checked_files else "filtered"
    operation = "Moving" if class_index == -1 else "Exporting"
    self._filtered_export_operation = operation
    self.set_label_progress(0, max(1, len(files_to_export)), value=0, progress_format=f"{operation} {selection_name}...")
    self.statusBar().showMessage(f"{operation} {len(files_to_export)} {selection_name} image(s)...", 3000)
    worker.start()


def on_filtered_export_progress(self, processed, total):
    operation = str(getattr(self, "_filtered_export_operation", "Exporting"))
    self.set_label_progress(
        0,
        max(1, int(total)),
        value=min(int(processed), max(1, int(total))),
        progress_format=f"{operation} {int(processed)}/{int(total)}",
    )


def on_filtered_export_completed(self, summary):
    summary = dict(summary or {})
    copied = int(summary.get("copied", 0))
    skipped = int(summary.get("skipped", 0))
    total = int(summary.get("total", 0))
    failed = list(summary.get("failed", []) or [])
    canceled = bool(summary.get("canceled", False))
    class_name = summary.get("class_name", "filtered")
    export_folder = summary.get("export_folder", "")
    operation = str(summary.get("operation", "copy"))
    completed_sources = list(summary.get("completed_sources", []) or [])

    if operation == "move" and completed_sources:
        self._remove_moved_images_from_review(completed_sources)

    if hasattr(self, "move_all_button"):
        self.move_all_button.setEnabled(True)

    self._filtered_export_worker = None
    self.reset_label_progress(100 if not failed else 0)

    action_past = "Moved" if operation == "move" else "Exported"
    action_log = "Moved" if operation == "move" else "Copied"
    if canceled:
        self.statusBar().showMessage(f"{action_past} {copied}/{total} before cancellation.", 3500)
    elif failed:
        preview = "\n".join(failed[:5])
        more = f"\n...and {len(failed) - 5} more." if len(failed) > 5 else ""
        skipped_note = f"\nSkipped: {skipped}" if skipped else ""
        QMessageBox.warning(
            self,
            "Move Filtered",
            f"{action_past} {copied}/{total} image(s) for '{class_name}', with {len(failed)} failure(s).{skipped_note}\n{preview}{more}",
        )
    else:
        skipped_note = f" Skipped {skipped}." if skipped else ""
        self.statusBar().showMessage(f"{action_past} {copied} image(s) for '{class_name}' to {export_folder}.{skipped_note}", 5000)

    logger.info(
        f"{action_log} {copied} images for '{class_name}' to {export_folder}."
        + (f" Skipped: {skipped}." if skipped else "")
        + (f" Failed: {len(failed)}." if failed else "")
    )
    self._filtered_export_grid_active = False
    self._filtered_export_operation = "Exporting"


def on_filtered_export_failed(self, message):
    if hasattr(self, "move_all_button"):
        self.move_all_button.setEnabled(True)

    self._filtered_export_worker = None
    self._filtered_export_grid_active = False
    self._filtered_export_operation = "Exporting"
    self.reset_label_progress(0)
    QMessageBox.critical(self, "Move Filtered", f"Filtered export failed:\n{message}")


def move_file(self, file_path, destination_folder):
    """
        Copies a file to the specified destination folder and returns the new path.
        """
    try:
        os.makedirs(destination_folder, exist_ok=True)

        destination_path = os.path.join(
            destination_folder,
            os.path.basename(file_path)
        ).replace("\\", "/")

        shutil.copy2(file_path, destination_path)

        logger.info(f"Copied {file_path} to {destination_path}")
        return destination_path

    except Exception as e:
        logger.error(f"Error copying {file_path} to {destination_folder}: {e}")
        return None


def is_file_matching_class_id(self, file_path, class_id):
    """
        Checks if a file matches the given class ID.
        """
    label_file = os.path.splitext(file_path)[0] + '.txt'

    if class_id == -2:  # Check for blanks
        if not os.path.exists(label_file) or not os.path.getsize(label_file):
            return True
        return False

    if os.path.exists(label_file):
        with open(label_file, 'r') as file:
            for line in file:
                if int(line.split()[0]) == class_id:
                    return True
    return False


def deduplicate_current_dataset_clicked(self):
    image_directory = getattr(self, "image_directory", None)
    if not image_directory or not os.path.isdir(image_directory):
        QMessageBox.information(self, "Deduplicate Annotations", "Open an image dataset first.")
        return

    duplicate_iou_threshold = self.get_duplicate_iou_threshold()
    reply = QMessageBox.question(
        self,
        "Deduplicate Annotations",
        (
            "This will clean near-duplicate boxes, segmentations, and OBBs in the currently open dataset.\n\n"
                f"Current base Duplicate IoU: {duplicate_iou_threshold:.2f}\n"
                "Cleanup also detects strong containment and close-center overlaps.\n\n"
                "After cleanup, open Dataset Analysis and press Start Scan to refresh the findings.\n\n"
                "Continue?"
        ),
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )

    if reply != QMessageBox.Yes:
        return

    sam_verify = (
        str(QSettings("UltraDarkFusion", "HealthCheck").value("sam_duplicate_verify", "false")).lower()
        in ("1", "true", "yes")
    )
    logger.info("User started label deduplication from Dataset Analysis.")
    summary = self.deduplicate_dataset(
        image_directory,
        iou_threshold=duplicate_iou_threshold,
        sam_verify=sam_verify,
    )

    if getattr(self, "current_file", None) and os.path.exists(self.current_file):
        self.display_image(self.current_file, rebuild_preview=False)

    removed = int(summary.get("labels_removed", 0))
    changed = int(summary.get("files_changed", 0))
    scanned = int(summary.get("files_scanned", 0))
    errors = int(summary.get("errors", 0))
    QMessageBox.information(
        self,
        "Deduplicate Annotations",
        (
            f"Cleanup complete.\n\n"
                f"Files checked: {scanned}\n"
                f"Files changed: {changed}\n"
                f"Duplicate labels removed: {removed}\n"
                f"Base duplicate IoU: {duplicate_iou_threshold:.2f}\n"
                f"SAM verified pairs: {int(summary.get('sam_pairs_checked', 0))}\n"
                f"SAM kept overlaps: {int(summary.get('sam_pairs_kept', 0))}\n"
                f"Files with errors: {errors}\n\n"
                "Opening Dataset Analysis. Press Start Scan to refresh the findings."
        ),
    )

    if hasattr(self, "scan_annotations"):
        self.scan_annotations.scan_annotations()


def deduplicate_dataset(self, image_directory, max_workers=None, iou_threshold=None, sam_verify=False):
    """
        Deduplicates all bounding box, segmentation, and OBB label files in parallel.
        Uses label_progress to update the progress bar safely.
        """
    sam_verify = bool(sam_verify)
    logger.info(
        "Deduplicating annotations across the dataset (%s mode).",
        "SAM-verified serial" if sam_verify else "parallel",
    )
    if iou_threshold is None:
        iou_threshold = self.get_duplicate_iou_threshold()
    iou_threshold = max(0.01, min(0.99, float(iou_threshold)))

    if sam_verify and torch.cuda.is_available():
        self._sam_force_fp16 = True
        self.sam_use_fp16 = True
        if hasattr(self, "settings"):
            self.settings["sam3Fp16"] = True

    if sam_verify and not self.ensure_sam_model_loaded():
        QMessageBox.warning(
            self,
            "SAM Verify",
            "SAM3 could not be loaded, so duplicate cleanup will use bbox geometry only.",
        )
        sam_verify = False

    self._dedupe_expected_keypoint_count = self.get_expected_keypoint_count_for_dedupe(image_directory)
    BoundingBox.EXPECTED_KEYPOINT_COUNT = self._dedupe_expected_keypoint_count or None

    txt_files = [os.path.splitext(file)[0] + '.txt' for file in self.get_image_files(image_directory)]
    total_files = len(txt_files)
    summary = {
        "files_scanned": total_files,
        "files_changed": 0,
        "labels_removed": 0,
        "bbox_removed": 0,
        "obb_removed": 0,
        "segmentation_removed": 0,
        "errors": 0,
        "iou_threshold": iou_threshold,
        "sam_pairs_checked": 0,
        "sam_pairs_kept": 0,
    }

    if hasattr(self, 'label_progress'):
        self.label_progress.setValue(0)
        self.label_progress.setRange(0, total_files)

    if total_files == 0:
        logger.info("No label files found for deduplication.")
        return summary

    completed = 0
    if sam_verify:
        for txt_file in txt_files:
            result = self.deduplicate_single_label_file(
                txt_file,
                iou_threshold,
                sam_verify=True,
            ) or {}
            summary["files_changed"] += int(result.get("changed", 0))
            summary["labels_removed"] += int(result.get("labels_removed", 0))
            summary["bbox_removed"] += int(result.get("bbox_removed", 0))
            summary["obb_removed"] += int(result.get("obb_removed", 0))
            summary["segmentation_removed"] += int(result.get("segmentation_removed", 0))
            summary["errors"] += int(result.get("errors", 0))
            summary["sam_pairs_checked"] += int(result.get("sam_pairs_checked", 0))
            summary["sam_pairs_kept"] += int(result.get("sam_pairs_kept", 0))

            completed += 1
            if hasattr(self, 'label_progress'):
                self.label_progress.setValue(completed)
                QApplication.processEvents()
            if completed % 5 == 0:
                self._safe_empty_cache()

        logger.info(
            "SAM-verified deduplication complete. Checked %d files, changed %d files, removed %d labels.",
            summary["files_scanned"],
            summary["files_changed"],
            summary["labels_removed"],
        )
        self._safe_empty_cache()
        return summary

    max_workers = max(1, min(int(max_workers or (os.cpu_count() or 4)), 8))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(self.deduplicate_single_label_file, txt_file, iou_threshold): txt_file
            for txt_file in txt_files
        }

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result() or {}
                summary["files_changed"] += int(result.get("changed", 0))
                summary["labels_removed"] += int(result.get("labels_removed", 0))
                summary["bbox_removed"] += int(result.get("bbox_removed", 0))
                summary["obb_removed"] += int(result.get("obb_removed", 0))
                summary["segmentation_removed"] += int(result.get("segmentation_removed", 0))
                summary["errors"] += int(result.get("errors", 0))
            except Exception as e:
                logger.error(f"Error during parallel deduplication: {e}")
                summary["errors"] += 1

            completed += 1

            if hasattr(self, 'label_progress'):
                self.label_progress.setValue(completed)
                QApplication.processEvents()

    logger.info(
        "Deduplication complete. Checked %d files, changed %d files, removed %d labels.",
        summary["files_scanned"],
        summary["files_changed"],
        summary["labels_removed"],
    )
    return summary


def delete_health_issues_from_dataset(self, health_issues, dataset_dir):
    """
        Delete flagged health issues (false positives, errors, etc.) from label files.
        Properly parses annotations and removes them using the labeling system.
        Returns detailed summary of deletions.
        """
    if not health_issues or not dataset_dir:
        return {"files_scanned": 0, "files_modified": 0, "annotations_removed": 0, "errors": []}

    logger.info("Deleting %d flagged health issues from dataset: %s", len(health_issues), dataset_dir)

    # Group issues by label file and line number
    issues_by_file = {}
    skipped_no_path = 0
    skipped_no_line = 0
    for issue in health_issues:
        label_path = str(issue.get("absolute_path", "")).strip()
        line_num = issue.get("line")
        if not label_path:
            skipped_no_path += 1
            continue
        if line_num is None:
            skipped_no_line += 1
            continue
        if label_path not in issues_by_file:
            issues_by_file[label_path] = set()
        issues_by_file[label_path].add(int(line_num))

    if skipped_no_path > 0 or skipped_no_line > 0:
        logger.warning("Skipped %d (no path) + %d (no line) issues", skipped_no_path, skipped_no_line)

    summary = {
        "files_scanned": len(issues_by_file),
        "files_modified": 0,
        "annotations_removed": 0,
        "lines_deleted": 0,
        "errors": [],
        "deleted_lines_by_file": {},
    }

    logger.info("Attempting to delete from %d label files", len(issues_by_file))

    for label_file, line_numbers in issues_by_file.items():
        try:
            if not os.path.isfile(label_file):
                msg = f"File not found: {label_file}"
                summary["errors"].append(msg)
                logger.warning(msg)
                continue

            # Read original file
            with open(label_file, "r", encoding="utf-8") as f:
                original_lines = f.readlines()

            original_count = len(original_lines)

            # Keep only lines NOT in the deletion set (1-indexed)
            kept_lines = []
            for idx, line in enumerate(original_lines, start=1):
                if idx not in line_numbers:
                    kept_lines.append(line)

            lines_removed = len(original_lines) - len(kept_lines)
            if lines_removed > 0:
                # A failed replacement leaves the original labels intact.
                write_label_text_atomic(label_file, "".join(kept_lines))

                summary["files_modified"] += 1
                summary["annotations_removed"] += lines_removed
                summary["lines_deleted"] += lines_removed
                summary["deleted_lines_by_file"][label_file] = sorted(
                    line for line in line_numbers if 1 <= line <= original_count
                )

                logger.info(
                    "✓ Deleted %d of %d lines from %s (%d remain)",
                    lines_removed,
                    original_count,
                    os.path.basename(label_file),
                    len(kept_lines)
                )
            else:
                logger.debug("No lines deleted from %s (0 of %d matched)", os.path.basename(label_file), original_count)
        except Exception as e:
            error_msg = f"{os.path.basename(label_file)}: {type(e).__name__}: {e}"
            summary["errors"].append(error_msg)
            logger.warning("Error deleting from %s: %s", label_file, e)

    logger.info(
        "Health issue deletion complete. Modified %d files, removed %d annotations.",
        summary["files_modified"],
        summary["annotations_removed"],
    )
    return summary


def _image_path_for_label_file(self, txt_file):
    base_path = os.path.splitext(txt_file)[0]
    for extension in DATASET_IMAGE_SUFFIXES:
        candidate = base_path + extension
        if os.path.exists(candidate):
            return candidate

        candidate = base_path + extension.upper()
        if os.path.exists(candidate):
            return candidate

    directory = os.path.dirname(txt_file) or "."
    stem = os.path.basename(base_path).lower()
    try:
        for entry in os.scandir(directory):
            if not entry.is_file():
                continue
            entry_stem, entry_ext = os.path.splitext(entry.name)
            if entry_stem.lower() == stem and entry_ext.lower() in DATASET_IMAGE_SUFFIXES:
                return entry.path
    except Exception:
        pass

    return None


@staticmethod
def _mask_pair_similarity(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return None

    mask_a = np.asarray(mask_a) > 0
    mask_b = np.asarray(mask_b) > 0
    if mask_a.shape != mask_b.shape or not mask_a.any() or not mask_b.any():
        return None

    intersection = float(np.count_nonzero(mask_a & mask_b))
    area_a = float(np.count_nonzero(mask_a))
    area_b = float(np.count_nonzero(mask_b))
    union = area_a + area_b - intersection
    if union <= 0.0:
        return None

    return {
        "iou": intersection / union,
        "containment": intersection / max(1.0, min(area_a, area_b)),
    }


def _sam_masks_indicate_duplicate(
    self,
    image,
    first_bbox,
    second_bbox,
    image_width,
    image_height,
    mask_cache,
    mask_iou_threshold=0.72,
    containment_threshold=0.88,
):
    if image is None or image_width <= 0 or image_height <= 0:
        return None

    masks = []
    for bbox in (first_bbox, second_bbox):
        bbox_xyxy = self._bbox_xyxy_from_box_obj(bbox, image_width, image_height)
        if bbox_xyxy is None:
            return None

        cache_key = (
            int(getattr(bbox, "class_id", -1)),
            round(float(getattr(bbox, "x_center", 0.0)), 6),
            round(float(getattr(bbox, "y_center", 0.0)), 6),
            round(float(getattr(bbox, "width", 0.0)), 6),
            round(float(getattr(bbox, "height", 0.0)), 6),
        )
        mask = mask_cache.get(cache_key)
        if mask is None:
            mask = self.sam_snap_bbox_to_mask(image, bbox_xyxy)
            if mask is None:
                return None
            mask_cache[cache_key] = mask

        masks.append(mask)

    similarity = self._mask_pair_similarity(masks[0], masks[1])
    if similarity is None:
        return None

    return {
        "duplicate": (
            similarity["iou"] >= mask_iou_threshold
            or similarity["containment"] >= containment_threshold
        ),
        **similarity,
    }


def deduplicate_single_label_file(self, txt_file, iou_threshold=None, sam_verify=False):

        if not os.path.exists(txt_file):
            return {
                "changed": 0,
                "labels_removed": 0,
                "bbox_removed": 0,
                "obb_removed": 0,
                "segmentation_removed": 0,
                "sam_pairs_checked": 0,
                "sam_pairs_kept": 0,
                "errors": 0,
            }

        try:
            BoundingBox.EXPECTED_KEYPOINT_COUNT = getattr(self, "_dedupe_expected_keypoint_count", 0) or None

            with open(txt_file, 'r', encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            bboxes, obbs, segmentations, preserved_lines = [], [], [], []
            polygon_parse_preference = self._polygon_label_parse_preference()
            image = None
            image_width = 0
            image_height = 0
            sam_pairs_checked = 0
            sam_pairs_kept = 0

            if sam_verify:
                image_path = self._image_path_for_label_file(txt_file)
                if image_path:
                    image = self._read_image_cv(image_path)
                    if image is not None:
                        image_height, image_width = image.shape[:2]

            for line in lines:
                if not line.strip():
                    continue

                bbox = BoundingBox.from_str(
                    line,
                    preferred_polygon_type=polygon_parse_preference,
                )
                if bbox:
                    if bbox.obb:
                        obbs.append(bbox)
                    elif bbox.segmentation:
                        segmentations.append((bbox.class_id, bbox.segmentation))
                    else:
                        bboxes.append(bbox)
                else:
                    preserved_lines.append(line)
                    logging.warning(f"Skipping malformed line in {txt_file}: '{line}'")

            # Prioritize keypoint boxes before deduplication
            bboxes.sort(key=self._dedupe_bbox_priority, reverse=True)

            deduped_bboxes = self.remove_near_duplicate_bounding_boxes(
                bboxes,
                iou_threshold=float(iou_threshold or self.get_duplicate_iou_threshold()),
                class_aware=True,
                image=image,
                image_width=image_width,
                image_height=image_height,
                sam_verify=bool(sam_verify and image is not None),
            )
            sam_pairs_checked = int(getattr(self, "_last_sam_dedupe_pairs_checked", 0))
            sam_pairs_kept = int(getattr(self, "_last_sam_dedupe_pairs_kept", 0))
            deduped_obbs = self.remove_duplicate_obbs(obbs, iou_threshold=0.5)
            deduped_segmentations = self.remove_duplicate_segmentations(segmentations)

            bbox_removed = max(0, len(bboxes) - len(deduped_bboxes))
            obb_removed = max(0, len(obbs) - len(deduped_obbs))
            segmentation_removed = max(0, len(segmentations) - len(deduped_segmentations))
            labels_removed = bbox_removed + obb_removed + segmentation_removed

            deduped_lines = (
                preserved_lines +
                [b.to_str() for b in deduped_bboxes if b] +
                [o.to_str() for o in deduped_obbs if o] +
                [f"{cid} {' '.join(f'{x:.6f}' for x in seg)}" for cid, seg in deduped_segmentations if seg]
            )

            cleaned_lines = [line.strip() for line in deduped_lines if line.strip()]
            changed = bool(labels_removed and cleaned_lines != lines)
            if changed:
                with open(txt_file, 'w', encoding="utf-8") as f:
                    f.write('\n'.join(cleaned_lines) + '\n')

            logger.info(
                f"Deduplicated {txt_file} "
                    f"(BBoxes: {len(bboxes)} ➜ {len(deduped_bboxes)}, "
                    f"OBBs: {len(obbs)} ➜ {len(deduped_obbs)}, "
                    f"Segs: {len(segmentations)} ➜ {len(deduped_segmentations)}, "
                    f"Preserved: {len(preserved_lines)})"
            )
            return {
                "changed": 1 if changed else 0,
                "labels_removed": labels_removed,
                "bbox_removed": bbox_removed,
                "obb_removed": obb_removed,
                "segmentation_removed": segmentation_removed,
                "sam_pairs_checked": sam_pairs_checked,
                "sam_pairs_kept": sam_pairs_kept,
                "errors": 0,
            }

        except Exception as e:
            logger.error(f"Error deduplicating {txt_file}: {e}")
            return {
                "changed": 0,
                "labels_removed": 0,
                "bbox_removed": 0,
                "obb_removed": 0,
                "segmentation_removed": 0,
                "sam_pairs_checked": 0,
                "sam_pairs_kept": 0,
                "errors": 1,
            }


def get_expected_keypoint_count_for_dedupe(self, image_directory=None):
    try:
        count = int(self.keypoint_list.rowCount())
        if count > 0:
            return count
    except Exception:
        pass

    candidates = []
    for directory in [image_directory, getattr(self, "image_directory", None), getattr(self, "output_path", None)]:
        if not directory:
            continue
        candidates.extend([
            resolve_dataset_metadata_path(directory, "points.json", migrate=True),
            resolve_dataset_metadata_path(directory, "pose.json", migrate=True),
        ])

    seen = set()
    for path in candidates:
        normalized = self.normalize_path(path) if hasattr(self, "normalize_path") else path
        if normalized in seen or not os.path.exists(normalized):
            continue
        seen.add(normalized)

        data = read_json_file(normalized, {})
        if not isinstance(data, dict):
            continue

        keypoints = data.get("keypoints")
        if isinstance(keypoints, list) and keypoints:
            return len(keypoints)

        kpt_shape = data.get("kpt_shape")
        if isinstance(kpt_shape, list) and kpt_shape:
            try:
                return max(0, int(kpt_shape[0]))
            except Exception:
                pass

    return 0


@staticmethod

def _dedupe_visible_keypoint_count(keypoints):
    count = 0
    for keypoint in keypoints or []:
        try:
            if len(keypoint) >= 3 and int(float(keypoint[2])) > 0:
                count += 1
        except Exception:
            continue
    return count


def _dedupe_bbox_priority(self, bbox):
    keypoints = getattr(bbox, "keypoints", None) or []
    area = float(getattr(bbox, "width", 0.0)) * float(getattr(bbox, "height", 0.0))
    return (
        self._dedupe_visible_keypoint_count(keypoints),
        1 if keypoints else 0,
        float(getattr(bbox, "confidence_safe", 0.0)),
        area,
    )


def _dedupe_keypoints_compatible(self, first_keypoints, second_keypoints, tolerance=0.02):
    first_keypoints = first_keypoints or []
    second_keypoints = second_keypoints or []

    first_visible = self._dedupe_visible_keypoint_count(first_keypoints)
    second_visible = self._dedupe_visible_keypoint_count(second_keypoints)

    if first_visible == 0 and second_visible == 0:
        return True

    if first_visible == 0 or second_visible == 0:
        return True

    if len(first_keypoints) != len(second_keypoints):
        return False

    distances = []
    for first, second in zip(first_keypoints, second_keypoints):
        try:
            first_visible_flag = int(float(first[2])) > 0
            second_visible_flag = int(float(second[2])) > 0
        except Exception:
            return False

        if not first_visible_flag or not second_visible_flag:
            continue

        try:
            dx = float(first[0]) - float(second[0])
            dy = float(first[1]) - float(second[1])
        except Exception:
            return False

        distances.append((dx * dx + dy * dy) ** 0.5)

    if not distances:
        return False

    return max(distances) <= tolerance


@staticmethod
def _bbox_center_distance(first_bbox, second_bbox):
    try:
        dx = float(first_bbox.x_center) - float(second_bbox.x_center)
        dy = float(first_bbox.y_center) - float(second_bbox.y_center)
        return (dx * dx + dy * dy) ** 0.5
    except Exception:
        return float("inf")


@staticmethod
def _bbox_area_ratio(first_bbox, second_bbox):
    try:
        first_area = max(1e-9, float(first_bbox.width) * float(first_bbox.height))
        second_area = max(1e-9, float(second_bbox.width) * float(second_bbox.height))
        return min(first_area, second_area) / max(first_area, second_area)
    except Exception:
        return 0.0


@staticmethod
def _dedupe_bbox_geometry_metrics(first_bbox, second_bbox):
    try:
        ax1 = float(first_bbox.x_center) - float(first_bbox.width) / 2.0
        ay1 = float(first_bbox.y_center) - float(first_bbox.height) / 2.0
        ax2 = float(first_bbox.x_center) + float(first_bbox.width) / 2.0
        ay2 = float(first_bbox.y_center) + float(first_bbox.height) / 2.0
        bx1 = float(second_bbox.x_center) - float(second_bbox.width) / 2.0
        by1 = float(second_bbox.y_center) - float(second_bbox.height) / 2.0
        bx2 = float(second_bbox.x_center) + float(second_bbox.width) / 2.0
        by2 = float(second_bbox.y_center) + float(second_bbox.height) / 2.0
    except Exception:
        return {
            "iou": 0.0,
            "containment": 0.0,
            "area_ratio": 0.0,
            "relative_center_distance": float("inf"),
        }

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    smaller_area = max(1e-9, min(area_a, area_b))
    larger_area = max(1e-9, max(area_a, area_b))
    center_distance = (
        (float(first_bbox.x_center) - float(second_bbox.x_center)) ** 2
        + (float(first_bbox.y_center) - float(second_bbox.y_center)) ** 2
    ) ** 0.5
    avg_w = max(1e-9, (abs(float(first_bbox.width)) + abs(float(second_bbox.width))) / 2.0)
    avg_h = max(1e-9, (abs(float(first_bbox.height)) + abs(float(second_bbox.height))) / 2.0)
    avg_diag = max(1e-9, (avg_w * avg_w + avg_h * avg_h) ** 0.5)

    return {
        "iou": inter / union if union > 0 else 0.0,
        "containment": inter / smaller_area,
        "area_ratio": smaller_area / larger_area,
        "relative_center_distance": center_distance / avg_diag,
    }


@staticmethod
def _dedupe_geometry_is_duplicate(metrics, iou_threshold):
    iou = float(metrics.get("iou", 0.0))
    containment = float(metrics.get("containment", 0.0))
    area_ratio = float(metrics.get("area_ratio", 0.0))
    rel_center = float(metrics.get("relative_center_distance", float("inf")))

    return (
        iou >= float(iou_threshold)
        or (containment >= 0.92 and area_ratio >= 0.18 and rel_center <= 0.45)
        or (containment >= 0.78 and area_ratio >= 0.35 and rel_center <= 0.35)
    )


def _box_only_keypoint_match_is_tight(self, first_bbox, second_bbox, pair_iou, pair_metrics=None):
    """
        Only collapse a keypoint label with a box-only label when geometry is tight.
        This avoids moving keypoints onto a neighboring overlap box in crowded scenes.
        """
    try:
        pair_iou = float(pair_iou)
    except Exception:
        pair_iou = 0.0

    if pair_iou < 0.85:
        pair_metrics = pair_metrics or self._dedupe_bbox_geometry_metrics(first_bbox, second_bbox)
        containment = float(pair_metrics.get("containment", 0.0))
        area_ratio = float(pair_metrics.get("area_ratio", 0.0))
        rel_center = float(pair_metrics.get("relative_center_distance", float("inf")))
        if not (containment >= 0.90 and area_ratio >= 0.65 and rel_center <= 0.25):
            return False

    center_distance = self._bbox_center_distance(first_bbox, second_bbox)
    area_ratio = self._bbox_area_ratio(first_bbox, second_bbox)
    return center_distance <= 0.035 and area_ratio >= 0.75


def remove_near_duplicate_bounding_boxes(
    self,
    bounding_boxes,
    iou_threshold=0.5,
    class_aware=True,
    image=None,
    image_width=0,
    image_height=0,
    sam_verify=False,
):
    """
        Removes near-duplicate bounding boxes using IoU, containment, and center proximity
        while preserving distinct keypoint sets.
        Boxes with different keypoints (or differing visibility) will NOT be merged,
        even if IoU > threshold.
        """
    self._last_sam_dedupe_pairs_checked = 0
    self._last_sam_dedupe_pairs_kept = 0

    if not bounding_boxes:
        return []

    # Sort boxes by confidence and keypoint count
    bounding_boxes = sorted(
        bounding_boxes,
        key=self._dedupe_bbox_priority,
        reverse=True
    )

    boxes = np.array([
        [
            bbox.x_center - bbox.width / 2,
            bbox.y_center - bbox.height / 2,
            bbox.x_center + bbox.width / 2,
            bbox.y_center + bbox.height / 2,
            bbox.class_id
        ]
        for bbox in bounding_boxes
    ])

    keep_indices = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    sam_mask_cache = {}
    use_sam_verify = bool(
        sam_verify
        and image is not None
        and int(image_width or 0) > 0
        and int(image_height or 0) > 0
    )

    for i in range(len(boxes)):
        if suppressed[i]:
            continue

        keep_indices.append(i)
        x1, y1, x2, y2, cls = boxes[i]
        rest = boxes[i + 1:]

        if rest.size == 0:
            break

        xx1 = np.maximum(x1, rest[:, 0])
        yy1 = np.maximum(y1, rest[:, 1])
        xx2 = np.minimum(x2, rest[:, 2])
        yy2 = np.minimum(y2, rest[:, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        area_i = (x2 - x1) * (y2 - y1)
        area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        union = area_i + area_rest - inter
        iou = inter / np.maximum(union, 1e-7)
        smaller_area = np.maximum(np.minimum(area_i, area_rest), 1e-7)
        larger_area = np.maximum(np.maximum(area_i, area_rest), 1e-7)
        containment = inter / smaller_area
        area_ratio = smaller_area / larger_area

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        rest_center_x = (rest[:, 0] + rest[:, 2]) / 2.0
        rest_center_y = (rest[:, 1] + rest[:, 3]) / 2.0
        center_distance = np.sqrt((center_x - rest_center_x) ** 2 + (center_y - rest_center_y) ** 2)
        avg_width = np.maximum(((x2 - x1) + (rest[:, 2] - rest[:, 0])) / 2.0, 1e-7)
        avg_height = np.maximum(((y2 - y1) + (rest[:, 3] - rest[:, 1])) / 2.0, 1e-7)
        avg_diag = np.maximum(np.sqrt((avg_width ** 2) + (avg_height ** 2)), 1e-7)
        relative_center_distance = center_distance / avg_diag

        geometry_duplicate_mask = (
            (iou >= iou_threshold)
            | ((containment >= 0.92) & (area_ratio >= 0.18) & (relative_center_distance <= 0.45))
            | ((containment >= 0.78) & (area_ratio >= 0.35) & (relative_center_distance <= 0.35))
        )

        if class_aware:
            class_mask = rest[:, 4] == cls
        else:
            class_mask = np.ones_like(iou, dtype=bool)

        duplicate_mask = geometry_duplicate_mask & class_mask

        # Preserve overlapping boxes when their keypoints describe different poses.
        for j, is_dup in enumerate(duplicate_mask):
            if not is_dup:
                continue
            current_idx = i
            compare_idx = i + 1 + j
            b1 = bounding_boxes[current_idx]
            b2 = bounding_boxes[compare_idx]
            b1_visible = self._dedupe_visible_keypoint_count(getattr(b1, "keypoints", None))
            b2_visible = self._dedupe_visible_keypoint_count(getattr(b2, "keypoints", None))

            if (b1_visible > 0) != (b2_visible > 0):
                pair_metrics = {
                    "iou": float(iou[j]),
                    "containment": float(containment[j]),
                    "area_ratio": float(area_ratio[j]),
                    "relative_center_distance": float(relative_center_distance[j]),
                }
                if not self._box_only_keypoint_match_is_tight(b1, b2, iou[j], pair_metrics):
                    duplicate_mask[j] = False
                    continue

            if not self._dedupe_keypoints_compatible(
                getattr(b1, "keypoints", None),
                getattr(b2, "keypoints", None)
            ):
                duplicate_mask[j] = False  # keep both if real keypoints describe different poses
                continue

            if use_sam_verify:
                sam_decision = self._sam_masks_indicate_duplicate(
                    image,
                    b1,
                    b2,
                    int(image_width),
                    int(image_height),
                    sam_mask_cache,
                )
                if sam_decision is not None:
                    self._last_sam_dedupe_pairs_checked += 1
                    if not sam_decision["duplicate"]:
                        duplicate_mask[j] = False
                        self._last_sam_dedupe_pairs_kept += 1
                        logging.debug(
                            "SAM kept overlapping boxes as separate objects "
                                "(mask_iou=%.2f, containment=%.2f).",
                            sam_decision["iou"],
                            sam_decision["containment"],
                        )

        suppressed[i + 1:][duplicate_mask] = True

    deduped = [bounding_boxes[i] for i in keep_indices]
    logging.debug(f"Removed {len(bounding_boxes) - len(deduped)} duplicates (preserved keypoint exclusivity).")
    return deduped


def remove_duplicate_segmentations(self, segmentations, iou_threshold=0.90, containment_ratio=0.70, min_area=1e-4):
    """
        Aggressively deduplicates masks by checking IoU and containment ratio.
        Keeps the largest, most complete version of each object.
        """
    from shapely.errors import TopologicalError

    grouped_by_class = {}
    final_segs = []

    for class_id, points in segmentations:
        poly = self.normalize_polygon(points)
        if not poly or poly.area < min_area:
            continue
        grouped_by_class.setdefault(class_id, []).append((poly, points))

    for class_id, entries in grouped_by_class.items():
        entries.sort(key=lambda x: x[0].area, reverse=True)
        kept = []

        for i, (poly_a, pts_a) in enumerate(entries):
            is_duplicate = False

            for poly_b, pts_b in kept:
                try:
                    inter_area = poly_a.intersection(poly_b).area
                    union_area = poly_a.union(poly_b).area
                    iou = inter_area / max(union_area, 1e-7)
                    containment = inter_area / max(poly_a.area, 1e-7)

                    if iou > iou_threshold or containment > containment_ratio:
                        is_duplicate = True
                        break

                except (ValueError, TopologicalError) as e:
                    logging.warning(f"Dedup comparison error: {e}")
                    continue

            if not is_duplicate:
                kept.append((poly_a, pts_a))

        final_segs.extend((class_id, pts) for _, pts in kept)

    return final_segs


@staticmethod

def normalize_polygon(points, tolerance=0.001):
    try:
        if len(points) < 6:
            return None
        poly = Polygon(list(zip(points[::2], points[1::2])))
        if not poly.is_valid or poly.area == 0:
            return None
        return poly.simplify(tolerance, preserve_topology=True)
    except Exception as e:
        logging.warning(f"Polygon normalization failed: {e}")
        return None


def remove_duplicate_obbs(self, obbs, iou_threshold=0.5):
    from shapely.geometry import Polygon

    def to_polygon(obb):
        try:
            coords = [(obb.obb[i], obb.obb[i+1]) for i in range(0, 8, 2)]
            return Polygon(coords)
        except Exception as e:
            logger.warning(f"Failed to convert OBB to polygon: {e}")
            return None

    unique = []
    for i, obb in enumerate(obbs):
        p1 = to_polygon(obb)
        if not p1 or not p1.is_valid:
            continue

        is_duplicate = False
        for u in unique:
            p2 = to_polygon(u)
            if not p2 or not p2.is_valid:
                continue

            inter = p1.intersection(p2).area
            union = p1.union(p2).area
            iou = inter / union if union > 0 else 0

            if iou > iou_threshold and obb.class_id == u.class_id:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(obb)

    return unique


@staticmethod

def calculate_iou(box1, box2):
    if not box1 or not box2:
        return 0.0

    x1_min = box1.x_center - box1.width / 2
    y1_min = box1.y_center - box1.height / 2
    x1_max = box1.x_center + box1.width / 2
    y1_max = box1.y_center + box1.height / 2

    x2_min = box2.x_center - box2.width / 2
    y2_min = box2.y_center - box2.height / 2
    x2_max = box2.x_center + box2.width / 2
    y2_max = box2.y_center + box2.height / 2

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area else 0.0


def on_combine_txt_clicked(self):
    if getattr(self, "combine_txt_flag", False):
        logger.info("Combine TXT function is already running.")
        return

    self.combine_txt_flag = True
    if hasattr(self, "combine_txt_button"):
        self.combine_txt_button.setEnabled(False)
    logger.info("Combine TXT process started.")

    try:
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        # Select two label files to combine
        file1, _ = QFileDialog.getOpenFileName(
            self, "Select First Label File", self.dialog_start_directory("dataset"), "Text Files (*.txt);;All Files (*)", options=options
        )
        if not file1:
            raise RuntimeError("First file selection cancelled.")
        self.remember_dialog_selection("dataset", file1)

        file2, _ = QFileDialog.getOpenFileName(
            self, "Select Second Label File", self.dialog_start_directory("dataset", file1), "Text Files (*.txt);;All Files (*)", options=options
        )
        if not file2:
            raise RuntimeError("Second file selection cancelled.")
        self.remember_dialog_selection("dataset", file2)

        # Choose where to save combined result
        output_file, _ = QFileDialog.getSaveFileName(
            self, "Save Combined Label File", self.dialog_start_directory("export", file1), "Text Files (*.txt);;All Files (*)", options=options
        )
        if not output_file:
            raise RuntimeError("Output file selection cancelled.")
        self.remember_dialog_selection("export", output_file)

        # Read both files safely
        combined_lines = []
        for path in (file1, file2):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        combined_lines.append(line)

        # Optional deduplication (avoid duplicate YOLO annotations)
        combined_lines = list(dict.fromkeys(combined_lines))

        # Write to output file
        with open(output_file, "w", encoding="utf-8") as out:
            out.write("\n".join(combined_lines) + "\n")

        QMessageBox.information(self, "Success", f"Labels combined successfully:\n{output_file}")
        logger.info(f"Combined TXT saved to {output_file}")

    except RuntimeError as cancel_reason:
        logger.warning(str(cancel_reason))
        QMessageBox.information(self, "Cancelled", str(cancel_reason))
    except Exception as e:
        logger.exception(f"Error combining files: {e}")
        QMessageBox.critical(self, "Error", f"An error occurred while combining files:\n{e}")
    finally:
        self.combine_txt_flag = False
        if hasattr(self, "combine_txt_button"):
            self.combine_txt_button.setEnabled(True)
        logger.info("Combine TXT function finished.")


def select_source_folder(self):
    folder = QFileDialog.getExistingDirectory(
        self, "Select Source Directory", self.dialog_start_directory("dataset", self.text_from.text())
    )
    if folder:
        self.remember_dialog_selection("dataset", folder)
        self.text_from.setText(folder)
        self.queue_settings_save()


def select_target_folder(self):
    folder = QFileDialog.getExistingDirectory(
        self, "Select Target Directory", self.dialog_start_directory("export", self.text_to.text())
    )
    if folder:
        self.remember_dialog_selection("export", folder)
        self.text_to.setText(folder)
        self.queue_settings_save()


def move_file_pair(self, src_img, src_txt, dst_img, dst_txt):
    shutil.move(src_img, dst_img)
    if os.path.exists(src_txt):
        shutil.move(src_txt, dst_txt)


def bulk_move_files(self, source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = os.listdir(source_dir)
    image_extensions = IMAGE_SUFFIXES


    file_pairs = []
    for f in files:
        if f.lower().endswith(image_extensions):
            img_src = os.path.join(source_dir, f)
            txt_src = os.path.splitext(img_src)[0] + ".txt"
            img_dst = os.path.join(target_dir, f)
            txt_dst = os.path.splitext(img_dst)[0] + ".txt"
            file_pairs.append((img_src, txt_src, img_dst, txt_dst))

    # Use ThreadPoolExecutor for speed and concurrency
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as executor:
        futures = [executor.submit(self.move_file_pair, img_src, txt_src, img_dst, txt_dst)
                for img_src, txt_src, img_dst, txt_dst in file_pairs]

        # Wait for all moves to complete
        for future in futures:
            future.result()


@pyqtSlot()

def start_moving_files(self):
    source_dir = self.text_from.text().strip()
    target_dir = self.text_to.text().strip()

    if not source_dir or not target_dir:
        QMessageBox.warning(self, "Warning", "Please select both source and target directories.")
        return

    source_dir = self.normalize_path(source_dir)
    target_dir = self.normalize_path(target_dir)

    if os.path.normcase(source_dir) == os.path.normcase(target_dir):
        QMessageBox.warning(self, "Warning", "Source and target directories cannot be the same.")
        return
    if not os.path.isdir(source_dir):
        QMessageBox.warning(self, "Move Files", f"Source directory does not exist:\n{source_dir}")
        return

    reply = QMessageBox.question(
        self, "Confirmation",
        f"Move all images and labels from:\n{source_dir}\nto:\n{target_dir}\n\nProceed?",
        QMessageBox.Yes | QMessageBox.No
    )

    if reply != QMessageBox.Yes:
        return

    active_worker = getattr(self, "_bulk_move_worker", None)
    if active_worker is not None:
        try:
            if active_worker.isRunning():
                QMessageBox.information(self, "Move Files", "A move job is already running.")
                return
        except RuntimeError:
            self._bulk_move_worker = None

    self.start_move_btn.setEnabled(False)
    self.start_move_btn.setText("Moving...")
    progress = getattr(self, "bulk_move_progress", None)
    if progress is not None:
        progress.setRange(0, 0)
        progress.setFormat("Scanning files...")
        progress.setToolTip(f"Source: {source_dir}\nTarget: {target_dir}")
        progress.show()

    worker = BulkMoveWorker(
        source_dir,
        target_dir,
        image_suffixes=IMAGE_SUFFIXES,
        parent=self,
    )
    self._bulk_move_worker = worker
    worker.progress.connect(self.on_bulk_move_progress)
    worker.completed.connect(self.on_bulk_move_completed)
    worker.failed.connect(self.on_bulk_move_failed)
    worker.finished.connect(lambda worker=worker: self._cleanup_worker_reference("_bulk_move_worker", worker))
    worker.start()


def on_bulk_move_progress(self, processed, total):
    progress = getattr(self, "bulk_move_progress", None)
    if progress is not None:
        progress.setRange(0, max(1, int(total)))
        progress.setValue(min(int(processed), max(1, int(total))))
        progress.setFormat(f"Moving {int(processed)}/{int(total)}")


def on_bulk_move_completed(self, summary):
    summary = dict(summary or {})
    moved = int(summary.get("moved", 0))
    skipped = int(summary.get("skipped", 0))
    total = int(summary.get("total", 0))
    failed = list(summary.get("failed", []) or [])
    canceled = bool(summary.get("canceled", False))
    details = "\n".join(failed[:20])
    if len(failed) > 20:
        details += f"\n...and {len(failed) - 20} more."
    self._bulk_move_last_summary = summary

    self.start_move_btn.setEnabled(True)
    self.start_move_btn.setText("Move Files")

    if canceled:
        message = f"Move canceled: {moved}/{total} image pair(s) moved. Skipped: {skipped}."
    elif failed:
        message = f"Moved {moved}/{total} image pair(s). Failed: {len(failed)}. Skipped: {skipped}."
        logger.warning("Standalone move finished with errors: %s", details)
    else:
        skipped_note = f" Skipped {skipped}." if skipped else ""
        message = f"Moved {moved} image pair(s).{skipped_note}"
    progress = getattr(self, "bulk_move_progress", None)
    if progress is not None:
        progress.setRange(0, max(1, total))
        progress.setValue(max(1, total) if not canceled else min(moved + skipped + len(failed), total))
        progress.setFormat(message)
        if failed:
            progress.setToolTip(details)
    self.statusBar().showMessage(message, 7000)


def on_bulk_move_failed(self, message):
    self.start_move_btn.setEnabled(True)
    self.start_move_btn.setText("Move Files")
    progress = getattr(self, "bulk_move_progress", None)
    if progress is not None:
        progress.setRange(0, 1)
        progress.setValue(0)
        progress.setFormat("Move failed; hover for details")
        progress.setToolTip(str(message))
    self.statusBar().showMessage(f"Move failed: {message}", 10000)
    logger.error("Standalone move failed: %s", message)


def run_bulk_move(self, source_dir, target_dir):
    try:
        self.bulk_move_files(source_dir, target_dir)
        QMessageBox.information(self, "Success", "Files moved successfully!")
    except Exception as e:
        QMessageBox.critical(self, "Error", f"An error occurred: {e}")
    finally:
        self.start_move_btn.setEnabled(True)
        self.start_move_btn.setText("Move Files")


def _compact_move_images_panel(self):
    if getattr(self, "_move_images_panel_compacted", False):
        return

    group = getattr(self, "groupBox_2", None)
    layout = self._prepare_group_grid(group)
    if layout is None:
        return

    names = ("move_from", "move_to", "text_from", "text_to", "start_move_btn")
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)

    widgets["text_from"].setPlaceholderText("Source folder")
    widgets["text_to"].setPlaceholderText("Target folder")

    source_row = self._row_layout()
    source_row.addWidget(widgets["move_from"])
    source_row.addWidget(widgets["text_from"], 1)

    target_row = self._row_layout()
    target_row.addWidget(widgets["move_to"])
    target_row.addWidget(widgets["text_to"], 1)

    layout.addLayout(source_row, 0, 0)
    layout.addLayout(target_row, 1, 0)
    layout.addWidget(widgets["start_move_btn"], 2, 0)
    widgets["start_move_btn"].setToolTip(
        "Move images and matching labels between these folders in the background. "
        "The source does not need to be open in the editor."
    )
    progress = QtWidgets.QProgressBar(group)
    progress.setObjectName("bulk_move_progress")
    progress.setRange(0, 100)
    progress.hide()
    layout.addWidget(progress, 3, 0)
    self.bulk_move_progress = progress

    self._move_images_panel_compacted = True


def _compact_image_format_panel(self):
    if getattr(self, "_image_format_panel_compacted", False):
        return

    group = getattr(self, "groupBox_3", None)
    layout = self._prepare_group_grid(group)
    if layout is None:
        return

    group.setTitle("Dataset Converter")
    names = ("select_format", "convert_image")
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)

    widgets["select_format"].setMaximumWidth(120)
    row = self._row_layout()
    row.addWidget(widgets["select_format"])
    row.addWidget(widgets["convert_image"], 1)
    layout.addLayout(row, 0, 0)

    self._image_format_panel_compacted = True


def get_duplicate_iou_threshold(self):
    widget = getattr(self, "dedupe_iou_spinbox", None)
    if widget is not None:
        return float(widget.value())

    try:
        return float(QSettings("UltraDarkFusion", "HealthCheck").value("duplicate_iou_threshold", 0.85))
    except Exception:
        return 0.85

