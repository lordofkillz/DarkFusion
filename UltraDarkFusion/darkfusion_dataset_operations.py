"""Filesystem-only dataset operations, safe to run away from the UI thread."""
from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
from bisect import bisect_left
from functools import lru_cache


class DatasetFileTransaction:
    """Stage small dataset metadata/label edits and roll back a failed batch."""
    def __init__(self, directory):
        parent = Path(directory).resolve() / ".darkfusion" / "class-remap"
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="operation-", dir=parent))
        self.records = []
        self.committed = []
        self.recovery_needed = False

    @staticmethod
    def _read(path):
        try:
            return Path(path).read_bytes()
        except FileNotFoundError:
            return None

    @staticmethod
    def _digest(content):
        return None if content is None else hashlib.sha256(content).hexdigest()

    def stage(self, path, content, *, expected=None, check_expected=False):
        target = Path(path).absolute()
        original = self._read(target)
        if check_expected and original != expected:
            raise OSError(f"Labels changed while class edits were prepared: {target}")
        if original == content:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        index = len(self.records)
        backup = self.directory / f"{index:08d}.original"
        staged = self.directory / f"{index:08d}.new"
        # Register paths before writing so a failed staging write is cleaned up.
        record = dict(target=str(target), backup=str(backup), staged=str(staged),
                      before=self._digest(original), after=self._digest(content))
        self.records.append(record)
        if original is not None:
            backup.write_bytes(original)
            stat = target.stat()
            os.utime(backup, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        staged.write_bytes(content)

    def commit(self):
        manifest = self.directory / "recovery.json"
        _atomic_bytes(str(manifest), json.dumps(self.records, indent=2).encode("utf-8"))
        for record in self.records:
            if self._digest(self._read(record["target"])) != record["before"]:
                raise OSError(f"Dataset file changed during class editing: {record['target']}")
        try:
            for record in self.records:
                if self._digest(self._read(record["target"])) != record["before"]:
                    raise OSError(f"Dataset file changed during class editing: {record['target']}")
                os.replace(record["staged"], record["target"])
                self.committed.append(record)
        except BaseException as error:
            failures = []
            for record in reversed(self.committed):
                try:
                    # Never roll back over a correction made by another process.
                    if self._digest(self._read(record["target"])) != record["after"]:
                        raise OSError("Newer changes detected; original backup retained")
                    if record["before"] is None:
                        os.remove(record["target"])
                    else:
                        os.replace(record["backup"], record["target"])
                except OSError as recovery_error:
                    failures.append(f"{record['target']}: {recovery_error}")
            if failures:
                self.recovery_needed = True
                raise OSError(f"{error}. Recovery files retained in {self.directory}: {'; '.join(failures)}") from error
            raise

    def close(self):
        if self.recovery_needed:
            return
        for record in self.records:
            for name in ("backup", "staged"):
                try:
                    Path(record[name]).unlink(missing_ok=True)
                except OSError:
                    logging.warning("Class-edit temporary file retained: %s", record[name])
        try:
            (self.directory / "recovery.json").unlink(missing_ok=True)
            self.directory.rmdir()
        except OSError:
            logging.warning("Class-edit temporary directory retained: %s", self.directory)


_delete_lock = threading.RLock()


@lru_cache(maxsize=8)
def _placeholder_roots(app_directory):
    normalize = lambda path: os.path.normcase(os.path.abspath(path))
    image_root = os.path.join(app_directory, "styles", "images")
    return (frozenset(normalize(os.path.join(image_root, name)) for name in ("default.png", "default_temp.png")),
            normalize(os.path.join(app_directory, "styles", "df-assets")))


def is_placeholder_path(path, app_directory):
    if not path:
        return False
    normalized = os.path.normcase(os.path.abspath(str(path)))
    placeholders, theme_root = _placeholder_roots(str(app_directory))
    return normalized in placeholders or (
        os.path.basename(normalized) == "idle-art.png"
        and os.path.dirname(os.path.dirname(normalized)) == theme_root
    )


def natural_image_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", os.path.basename(path))]


def excluded_video_frames(manifest):
    result = set()
    for value in manifest.get("deleted_frames", []) or []:
        try:
            frame = int(value)
            if frame >= 0:
                result.add(frame)
        except (TypeError, ValueError):
            continue
    return result


def sparse_video_navigation(context):
    manifest = (context or {}).get("manifest") or {}
    workflow = manifest.get("workflow")
    return workflow == "sparse_manual" or (not workflow and bool(manifest.get("frames")))


def update_health_report_after_deletion(issues, result):
    """Drop only committed rows, and reindex surviving findings in those files."""
    deleted = {os.path.normcase(os.path.abspath(path)): sorted(set(rows))
               for path, rows in result.get("deleted_lines_by_file", {}).items()}
    kept = []
    for issue in issues:
        path = issue.get("absolute_path")
        rows = deleted.get(os.path.normcase(os.path.abspath(path))) if path else None
        line = issue.get("line")
        if rows and isinstance(line, int):
            if line in rows:
                continue
            issue["line"] = line - bisect_left(rows, line)
        kept.append(issue)
    removed = len(issues) - len(kept)
    issues[:] = kept
    return removed


def _atomic_bytes(path, content):
    descriptor, temporary = tempfile.mkstemp(prefix=".darkfusion-write-", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            os.remove(temporary)


def delete_image_label_pair(image_path):
    """Stage both files before removing either; restore the pair on failure.

    A locked annotation must not leave its image deleted. The small label is
    retained in memory until image removal succeeds, allowing rollback even
    when the second unlink fails. Failed rollback paths remain on disk and are
    reported explicitly; an error never silently discards a staged file.
    """
    image_path = os.path.abspath(os.fspath(image_path))
    label_path = os.path.splitext(image_path)[0] + ".txt"
    directory = os.path.dirname(image_path)
    with _delete_lock:
        label_content = None
        try:
            with open(label_path, "rb") as stream:
                label_content = stream.read()
        except FileNotFoundError:
            pass
        manifest_path = os.path.join(directory, ".darkfusion", "video_annotations.json")
        manifest_before = manifest_after = None
        try:
            with open(manifest_path, "rb") as stream:
                manifest_before = stream.read()
            manifest = json.loads(manifest_before)
            match = re.fullmatch(re.escape(Path(manifest.get("source", "")).stem) + r"_frame_(\d+)",
                                 Path(image_path).stem, flags=re.IGNORECASE)
            if sparse_video_navigation({"manifest": manifest}) and match:
                deleted = excluded_video_frames(manifest)
                deleted.add(int(match.group(1)))
                manifest["deleted_frames"] = sorted(deleted)
                manifest_after = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        except FileNotFoundError:
            pass
        except (ValueError, TypeError, AttributeError):
            # Malformed provenance must not cause guessing or unrelated writes.
            pass

        staged = []
        label_backup = None
        manifest_written = False
        label_removed = False
        try:
            for original in (label_path, image_path):
                if original == label_path and label_content is None:
                    continue
                descriptor, temporary = tempfile.mkstemp(prefix=".darkfusion-delete-", suffix=".tmp", dir=directory)
                os.close(descriptor)
                try:
                    os.replace(original, temporary)
                except FileNotFoundError:
                    os.remove(temporary)
                    continue
                except BaseException:
                    os.remove(temporary)
                    raise
                staged.append((original, temporary))
            if manifest_after is not None:
                _atomic_bytes(manifest_path, manifest_after)
                manifest_written = True
            if label_content is not None and any(original == label_path for original, _ in staged):
                descriptor, label_backup = tempfile.mkstemp(prefix=".darkfusion-label-backup-", suffix=".tmp", dir=directory)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(label_content)
                    stream.flush()
                    os.fsync(stream.fileno())
            # The label goes first: its bytes can be restored cheaply if image
            # unlink fails. Never read/copy a full-resolution image for rollback.
            for original, temporary in staged:
                os.remove(temporary)
                if original == label_path:
                    label_removed = True
            if label_backup is not None:
                try:
                    os.remove(label_backup)
                except OSError:
                    logging.warning("Deleted image pair; label recovery copy could not be cleaned up: %s", label_backup)
            return True
        except BaseException as error:
            rollback_errors = []
            for original, temporary in reversed(staged):
                try:
                    if os.path.exists(temporary):
                        if os.path.exists(original):
                            raise OSError(f"Destination changed; recovery file retained at {temporary}")
                        os.rename(temporary, original)
                    elif original == label_path and label_removed and label_content is not None:
                        if os.path.exists(original):
                            raise OSError(f"Destination changed; could not restore {original}")
                        os.rename(label_backup, original)
                        label_backup = None
                except OSError as rollback_error:
                    recovery_path = label_backup if original == label_path and label_removed else temporary
                    rollback_errors.append(f"Could not restore {original}; recovery file: {recovery_path}: {rollback_error}")
            if manifest_written:
                try:
                    _atomic_bytes(manifest_path, manifest_before)
                except OSError as rollback_error:
                    rollback_errors.append(f"Could not restore video deletion metadata: {rollback_error}")
            if rollback_errors:
                raise OSError(f"{error}. Recovery needs attention: {'; '.join(rollback_errors)}") from error
            if label_backup is not None:
                try:
                    os.remove(label_backup)
                except OSError:
                    logging.warning("Label recovery copy retained: %s", label_backup)
            raise
