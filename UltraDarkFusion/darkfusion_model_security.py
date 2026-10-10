"""Application-side checkpoint guard for CVE-2026-69112.

Reject unsafe indexes before Accelerate opens files. This does not claim the
installed upstream release has fixed the vulnerability.
"""
from __future__ import annotations
from functools import wraps
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import threading

_LOCK = threading.RLock()
_MAX_INDEX_BYTES = 4 * 1024 * 1024


def _regular_file(path):
    try:
        info = path.stat()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Checkpoint file is unavailable: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Checkpoint must be a regular file: {path}")
    return info


def validate_checkpoint(checkpoint):
    """Validate relative shard names, resolved containment, and regular files."""
    path = Path(checkpoint)
    if path.is_dir():
        indexes = sorted(path.glob("*.index.json"))
    else:
        _regular_file(path)
        indexes = [path] if path.name.endswith(".json") else []
    for index_path in indexes:
        if _regular_file(index_path).st_size > _MAX_INDEX_BYTES:
            raise ValueError("Checkpoint index is too large")
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        if not isinstance(index, dict):
            raise ValueError("Checkpoint index must be an object")
        weights = index.get("weight_map", index)
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Checkpoint weight_map must be a nonempty object")
        root = index_path.parent.resolve()
        allowed = root
        # Official HF snapshots may link only to blobs in their own model cache.
        if root.parent.name == "snapshots" and root.parent.parent.name.startswith("models--"):
            allowed = root.parent.parent
        values = list(weights.values())
        if not all(isinstance(value, str) for value in values):
            raise ValueError("Checkpoint shard names must be strings")
        for shard in set(values):
            if not shard or ":" in shard:
                raise ValueError("Invalid checkpoint shard name")
            portable = PurePosixPath(shard.replace("\\", "/"))
            if portable.is_absolute() or PureWindowsPath(shard).drive or ".." in portable.parts:
                raise ValueError(f"Checkpoint shard escapes its folder: {shard}")
            target = root.joinpath(*portable.parts)
            if not target.resolve().is_relative_to(allowed):
                raise ValueError(f"Checkpoint shard symlink escapes its model cache: {shard}")
            _regular_file(target)


def _guard(loader):
    @wraps(loader)
    def guarded(model, checkpoint, *args, **kwargs):
        validate_checkpoint(checkpoint)
        return loader(model, checkpoint, *args, **kwargs)
    guarded._darkfusion_checkpoint_guard = True
    return guarded


def install_checkpoint_guard():
    """Protect both loaders and their public aliases, once per process."""
    import accelerate
    import accelerate.big_modeling as big_modeling
    import accelerate.utils as utils
    import accelerate.utils.modeling as modeling
    with _LOCK:
        modules = (accelerate, big_modeling, utils, modeling)
        for name in ("load_checkpoint_in_model", "load_checkpoint_and_dispatch"):
            originals = {getattr(module, name) for module in modules if hasattr(module, name)}
            for original in originals:
                if getattr(original, "_darkfusion_checkpoint_guard", False):
                    continue
                guarded = _guard(original)
                for module in modules:
                    if getattr(module, name, None) is original:
                        setattr(module, name, guarded)
