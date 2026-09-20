"""Download pinned DINOv3 release assets on a worker, then install atomically."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import time

from .config import MODEL_CONFIGS
from .loader import CheckpointError, verify_checkpoint

RELEASE_URL = "https://github.com/lordofkillz/DarkFusion/releases/download/v5.2.1-windows.1"


def ensure_checkpoint(model_key, destination, *, status=None, check_cancelled=None):
    """Fetch only missing files. Existing checkpoints are verified by the loader.

    Call from a background worker: network reads can wait up to ten seconds.
    No credentials or remote Python code are used. An interrupted, oversized,
    truncated, or checksum-mismatched download never becomes a checkpoint.
    """
    destination = Path(destination)
    if destination.exists():
        return False
    status = status or (lambda message: None)
    check_cancelled = check_cancelled or (lambda: None)
    check_cancelled()
    spec = MODEL_CONFIGS[model_key]
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        status(f"Downloading {spec['label']} ({spec['size'] / 1_000_000:.0f} MB, once)...")
        with requests.get(f"{RELEASE_URL}/{spec['filename']}", stream=True,
                          timeout=(5, 10), headers={"Accept-Encoding": "identity"}) as response:
            response.raise_for_status()
            check_cancelled()
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != spec["size"]:
                raise CheckpointError("The DINOv3 download has an unexpected size.")
            digest = hashlib.sha256()
            received, updated = 0, 0.0
            with tempfile.NamedTemporaryFile(prefix=spec['filename'] + '.', suffix='.partial',
                                             dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                for chunk in response.iter_content(1024 * 1024):
                    check_cancelled()
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > spec['size']:
                        raise CheckpointError("The DINOv3 download exceeds the expected size.")
                    stream.write(chunk)
                    digest.update(chunk)
                    now = time.monotonic()
                    if now - updated >= 1:
                        status(f"Downloading {spec['label']}: {100 * received // spec['size']}%")
                        updated = now
            check_cancelled()
            if received != spec['size'] or digest.hexdigest() != spec['sha256']:
                raise CheckpointError("DINOv3 download checksum failed. Start the scan again to retry.")
            # Another worker may have finished while this one was downloading.
            if destination.exists():
                verify_checkpoint(model_key, destination, check_cancelled=check_cancelled)
            else:
                os.replace(temporary, destination)
                temporary = None
        status(f"{spec['label']} downloaded and verified.")
        return True
    except requests.RequestException as exc:
        # Provider exceptions may contain redirect URLs; keep user errors simple.
        raise CheckpointError(
            "DINOv3 could not download. Check the internet connection and retry, "
            f"or copy the official checkpoint to {destination}."
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
