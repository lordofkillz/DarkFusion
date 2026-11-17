"""Provision the required desktop checkpoints without importing ML or Qt.

The pip launcher calls this before opening the application. Source and EXE
launchers retain their existing model installation paths and workflows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
import zipfile

from darkfusion_dataset_operations import read_json_file, write_json_file_atomic
from darkfusion_dinov3.config import MODEL_CONFIGS
from darkfusion_dinov3.download import ensure_checkpoint


class ModelSetupError(RuntimeError):
    """Required checkpoint setup did not finish; existing files are preserved."""


def _signature(path):
    info = path.stat()
    return [info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _verified(path, spec, receipts):
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise ModelSetupError(f"Expected a regular checkpoint file: {path}")
    before = _signature(path)
    if before[0] != spec["size"]:
        raise ModelSetupError(f"Checkpoint size mismatch; existing file was preserved: {path}")
    record = {"stat": before, "sha256": spec["sha256"]}
    if receipts.get(path.name) != record:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if _signature(path) != before or digest != spec["sha256"]:
            raise ModelSetupError(f"Checkpoint checksum failed; existing file was preserved: {path}")
        receipts[path.name] = record
    return True


def _install_stream(source, destination, spec):
    """Verify while copying, then publish only a complete checkpoint."""
    temporary = None
    try:
        digest, count = hashlib.sha256(), 0
        with tempfile.NamedTemporaryFile(prefix=".model-", dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            while block := source.read(4 * 1024 * 1024):
                count += len(block)
                if count > spec["size"]:
                    raise ModelSetupError("Checkpoint exceeds its pinned size.")
                output.write(block)
                digest.update(block)
            output.flush()
            os.fsync(output.fileno())
        if count != spec["size"] or digest.hexdigest() != spec["sha256"]:
            raise ModelSetupError("Checkpoint checksum failed; no model was installed.")
        if destination.exists() or destination.is_symlink():
            raise ModelSetupError(f"A checkpoint appeared during setup; it was preserved: {destination}")
        # Setup holds the shared directory's cross-process lock. On Windows,
        # rename also refuses to overwrite an independently created destination.
        temporary.rename(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _bundle_specs(manifest):
    expected = {"sam3.pt", "groundingdino_swint_ogc.pth"}
    specs = {}
    for record in manifest["files"]:
        name = PurePosixPath(record["archive_path"]).name
        if (name not in expected or name in specs
                or record["archive_path"] != "Sam/" + name
                or record["target_path"] != "app/UltraDarkFusion/Sam/" + name
                or not isinstance(record["size_bytes"], int) or record["size_bytes"] <= 0
                or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"])):
            raise ModelSetupError("Invalid required-model bundle manifest.")
        specs[name] = {"size": record["size_bytes"], "sha256": record["sha256"],
                       "archive_path": record["archive_path"]}
    if set(specs) != expected or sum(s["size"] for s in specs.values()) != manifest["unpacked_size_bytes"]:
        raise ModelSetupError("Incomplete required-model bundle manifest.")
    return specs


def _safe_archive(archive):
    """Reject Windows aliases, links and file/folder collisions, even if unused."""
    seen, files, entries = set(), set(), {}
    for info in archive.infolist():
        name = info.orig_filename.replace("\\", "/").rstrip("/")
        parts = name.split("/")
        kind = stat.S_IFMT(info.external_attr >> 16)
        if (not name or info.flag_bits & 1 or info.external_attr & 0x400
                or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
                or any(not part or part in (".", "..") or part.endswith((".", " "))
                       or re.search(r'[<>:"|?*\x00-\x1f]', part)
                       or re.fullmatch(r"(?:CON|PRN|AUX|NUL|CLOCK\$|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?", part, re.I)
                       for part in parts)):
            raise ModelSetupError("Unsafe required-model archive entry.")
        key = name.casefold()
        if key in seen:
            raise ModelSetupError("Repeated required-model archive entry.")
        seen.add(key)
        entries[name] = info
        if not info.is_dir():
            files.add(key)
    for key in seen:
        if any("/".join(key.split("/")[:i]) in files for i in range(1, len(key.split("/")))):
            raise ModelSetupError("Conflicting required-model archive paths.")
    return entries


def _download_bundle(manifest, cache, status):
    """Resume our own partial archive when the server supports byte ranges."""
    import requests

    expected = {"size": manifest["size_bytes"], "sha256": manifest["sha256"]}
    archive = cache / "required-models.zip"
    if _verified(archive, expected, {}):
        return archive
    partial = cache / "required-models.zip.part"
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise ModelSetupError("Unsafe partial model download path.")
    received = partial.stat().st_size if partial.exists() else 0
    if received > expected["size"]:
        raise ModelSetupError("Partial model archive exceeds its pinned size; it was preserved.")
    try:
        if received < expected["size"]:
            headers = {"Accept-Encoding": "identity"}
            if received:
                headers["Range"] = f"bytes={received}-"
            status(f"Downloading required SAM3/GroundingDINO bundle ({expected['size'] / 1e9:.2f} GB)...")
            with requests.get(manifest["url"], stream=True, timeout=(15, 30), headers=headers) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    expected_range = f"bytes {received}-{expected['size'] - 1}/{expected['size']}"
                    if response.headers.get("Content-Range") != expected_range:
                        raise ModelSetupError("Model server returned an unexpected byte range.")
                elif response.status_code == 200:
                    received = 0
                else:
                    raise ModelSetupError("Model server returned an unexpected response.")
                length = response.headers.get("Content-Length")
                if ("text/html" in response.headers.get("Content-Type", "").lower()
                        or (length is not None and int(length) != expected["size"] - received)):
                    raise ModelSetupError("Model host did not return the pinned archive. Retry setup later.")
                updated = 0.0
                with partial.open("ab" if received else "wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        received += len(chunk)
                        if received > expected["size"]:
                            raise ModelSetupError("Model download exceeds its pinned size.")
                        output.write(chunk)
                        if time.monotonic() - updated >= 1:
                            status(f"Required model download: {100 * received // expected['size']}%")
                            updated = time.monotonic()
        if not _verified(partial, expected, {}):
            raise ModelSetupError("Required model archive is incomplete.")
        partial.rename(archive)
        return archive
    except requests.RequestException as exc:
        raise ModelSetupError("Model download interrupted. Run darkfusion again to resume setup.") from exc


def ensure_required_models(directory, *, manifest_path=None, source_directory=None, status=None):
    """Install all four required checkpoints; reuse unchanged verified files."""
    from filelock import FileLock

    status = status or (lambda message: None)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path or Path(__file__).resolve().parents[1] / "installer/windows/model-bundle.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    bundle = _bundle_specs(manifest)
    specs = dict(bundle)
    for spec in MODEL_CONFIGS.values():
        specs[spec["filename"]] = {"size": spec["size"], "sha256": spec["sha256"]}
    receipt_path = directory / ".verified-models.json"
    status("Checking required DarkFusion models...")
    with FileLock(str(directory / ".setup.lock")):
        receipts = read_json_file(receipt_path)
        missing = [name for name, spec in specs.items() if not _verified(directory / name, spec, receipts)]
        if source_directory is not None:
            source_directory = Path(source_directory).resolve()
            for name in missing[:]:
                source = source_directory / name
                if source.is_file() and not source.is_symlink():
                    status("Importing and verifying " + name + "...")
                    with source.open("rb") as stream:
                        _install_stream(stream, directory / name, specs[name])
                    missing.remove(name)
        needed_bundle = [name for name in missing if name in bundle]
        if needed_bundle:
            cache = directory / ".downloads"
            cache.mkdir(exist_ok=True)
            archive_path = _download_bundle(manifest, cache, status)
            # Hold the exact verified archive open through extraction.
            with archive_path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["sha256"]:
                    raise ModelSetupError("Required-model archive changed before extraction.")
                stream.seek(0)
                with zipfile.ZipFile(stream) as archive:
                    entries = _safe_archive(archive)
                    for name in needed_bundle:
                        spec = bundle[name]
                        info = entries[spec["archive_path"]]
                        if info.is_dir() or info.file_size != spec["size"]:
                            raise ModelSetupError("Incorrect checkpoint entry in model archive.")
                        status("Installing and verifying " + name + "...")
                        with archive.open(info) as source:
                            _install_stream(source, directory / name, spec)
            # This is our downloaded archive, never an imported user file.
            archive_path.unlink()
        for key, spec in MODEL_CONFIGS.items():
            if spec["filename"] in missing:
                ensure_checkpoint(key, directory / spec["filename"], status=status)
        for name, spec in specs.items():
            if not _verified(directory / name, spec, receipts):
                raise ModelSetupError("Required model was not installed: " + name)
        write_json_file_atomic(receipt_path, receipts)
    status(f"All required DarkFusion checkpoints verified: {directory}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--source-directory", type=Path, help="Import and verify existing checkpoints before downloading missing ones")
    args = parser.parse_args(argv)
    try:
        ensure_required_models(args.directory, source_directory=args.source_directory,
                               status=lambda message: print(message, flush=True))
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, "DarkFusion model setup: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
