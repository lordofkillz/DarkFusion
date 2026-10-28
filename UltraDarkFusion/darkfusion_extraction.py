"""Checked extraction writes and source-bound, atomic resume checkpoints."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


class ExtractionWriteError(OSError):
    """A selected frame could not be committed; do not advance past it."""


def _temporary_path(target, suffix):
    descriptor, name = tempfile.mkstemp(prefix=".darkfusion-extract-", suffix=suffix, dir=target.parent)
    os.close(descriptor)
    return Path(name)


def save_extraction_outputs(image_path, image, image_writer, *, label_path=None, labels=None, label_writer=None, preserve_existing_labels=True):
    """Stage both outputs before replacing either, with rollback on commit failure.

    Label validation uses the normal DarkFusion writer. A failed writer never
    leaves a newly extracted image looking like an intentional negative sample.
    Existing outputs are restored if the second replacement fails. Failed
    rollback backups are retained and identified in the exception.
    """
    image_path = Path(image_path).absolute()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    protected_label = Path(label_path).absolute() if label_path is not None else image_path.with_suffix(".txt")

    def reviewed_pair():
        return preserve_existing_labels and image_path.is_file() and protected_label.is_file() and protected_label.stat().st_size > 0

    if reviewed_pair():
        return True
    outputs = [(image_path, image, image_writer)]
    if labels is not None:
        if label_path is None or label_writer is None:
            raise ValueError("A label path and writer are required for labeled extraction.")
        outputs.append((Path(label_path).absolute(), labels, label_writer))
    staged, backups, committed = {}, {}, []
    retained = set()
    try:
        for target, content, writer in outputs:
            # Correct image suffixes are needed by the encoder, but staged
            # images must not appear as dataset frames during a folder refresh.
            stage_directory = target.parent / ".darkfusion"
            stage_directory.mkdir(exist_ok=True)
            stage = _temporary_path(stage_directory / target.name, target.suffix)
            staged[target] = stage
            ok = writer(str(stage), content) if target == image_path else writer(str(stage), content, mode="w")
            if not ok:
                raise ExtractionWriteError(f"Could not save {'image' if target == image_path else 'labels'}: {target}")
            if not stage.is_file() or stage.stat().st_size == 0:
                raise ExtractionWriteError(f"Writer did not produce a complete output: {target}")
        # Review may have saved an annotation while inference/encoding ran.
        if reviewed_pair():
            return True
        for target, _content, _writer in outputs:
            if target.exists():
                backup = _temporary_path(target, ".backup")
                backups[target] = backup
                try:
                    os.replace(target, backup)
                except Exception:
                    backups.pop(target)
                    backup.unlink(missing_ok=True)
                    raise
            os.replace(staged[target], target)
            committed.append(target)
    except Exception as error:
        rollback_errors = []
        for target, _content, _writer in reversed(outputs):
            try:
                if target in backups:
                    os.replace(backups[target], target)
                elif target in committed:
                    target.unlink(missing_ok=True)
            except OSError as rollback_error:
                backup = backups.get(target)
                if backup is not None:
                    retained.add(backup)
                rollback_errors.append(f"{target}: {rollback_error}; backup: {backup}")
        detail = f"Extraction save failed: {error}"
        if rollback_errors:
            detail += ". Recovery files retained: " + "; ".join(rollback_errors)
        raise ExtractionWriteError(detail) from error
    finally:
        for path in [*staged.values(), *backups.values()]:
            if path not in retained:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
    return True


def extraction_job_fingerprint(config):
    """Bind a checkpoint to the source and settings, not just an output folder."""
    fields = (
        "kind", "original_source", "source", "output_dir", "video_name", "image_extension",
        "frame_step", "crop_enabled", "resize_enabled", "resize_policy", "width", "height",
        "network_width", "network_height", "model_size_available", "model_loaded",
        "inference_enabled", "start_frame", "end_frame", "inference_identity",
    )
    identity = {key: config.get(key) for key in fields}
    model_path = (config.get("inference_identity") or {}).get("model_path")
    if model_path and Path(model_path).is_file():
        stat = Path(model_path).stat()
        identity["model_file"] = [os.path.normcase(str(Path(model_path).resolve())), stat.st_size, stat.st_mtime_ns]
    source = config.get("capture_source", config.get("source"))
    if source is not None and not isinstance(source, int):
        path = Path(str(source))
        if "://" not in str(source) and path.is_file():
            stat = path.stat()
            identity["file"] = [os.path.normcase(str(path.resolve())), stat.st_size, stat.st_mtime_ns]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ExtractionCheckpoint:
    """Checkpoint only seekable file jobs; live sources have no historical frames."""
    def __init__(self, config):
        self.config = dict(config)
        self.path = Path(config.get("output_dir") or ".") / ".extraction_checkpoint.json"
        source = config.get("capture_source", config.get("source"))
        self.enabled = (config.get("kind") == "video" and isinstance(source, (str, os.PathLike))
                        and "://" not in str(source) and Path(source).is_file())
        self.fingerprint = extraction_job_fingerprint(config) if self.enabled else ""
        self.last_written = None

    def load(self):
        if not self.enabled:
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 2 or payload.get("job_fingerprint") != self.fingerprint:
                return None
            frame, saved = payload.get("last_frame_count"), payload.get("frames_saved_before")
            if type(frame) is not int or frame < 0 or type(saved) is not int or saved < 0:
                return None
            return payload
        except (OSError, ValueError):
            return None

    def save(self, frame, saved):
        if not self.enabled or frame < 0 or self.last_written == (frame, saved):
            return False
        payload = {"version": 2, "job_fingerprint": self.fingerprint,
                   "last_frame_count": int(frame), "frames_saved_before": int(saved),
                   "timestamp": datetime.now(timezone.utc).isoformat(), "extraction_dir": str(self.path.parent)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_path(self.path, ".json.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            self.last_written = (frame, saved)
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def complete(self):
        if self.load() is not None:
            self.path.unlink(missing_ok=True)
