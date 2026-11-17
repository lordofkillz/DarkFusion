"""Pip entry point for the existing DarkFusion desktop application.

No GUI, model, download, or application workspace is initialized on import.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib import resources
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile


def version() -> str:
    return resources.files(__name__).joinpath("VERSION").read_text(encoding="utf-8").strip()


def data_directory() -> Path:
    explicit = os.environ.get("DARKFUSION_DATA_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "DarkFusion/pip"


@contextmanager
def _workspace_lock(directory: Path):
    import msvcrt
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "prepare.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0"); stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _manifest(package: str) -> tuple[object, dict]:
    resource = resources.files(package)
    manifest = json.loads(resource.joinpath("manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("version") != version():
        raise RuntimeError(f"Install matching {package} and ultradarkfusion versions.")
    return resource, manifest


def _safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe application resource path")
    return path


def _file_matches(path: Path, record: dict) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]


def prepare(*, check: bool = False) -> Path:
    if sys.platform != "win32" or struct.calcsize("P") != 8:
        raise RuntimeError("This DarkFusion release supports 64-bit Windows with Python 3.12.")
    base = data_directory()
    destination = base / "versions" / version()
    segments = [_manifest("ultradarkfusion"), _manifest("ultradarkfusion_themes")]
    identity = hashlib.sha256(json.dumps([m for _, m in segments], sort_keys=True).encode()).hexdigest()
    marker = destination / "prepared.json"
    with _workspace_lock(base):
        if marker.is_file():
            if json.loads(marker.read_text(encoding="utf-8")).get("identity") != identity:
                raise RuntimeError("Installed files differ from this version's prepared workspace. Use a new DARKFUSION_DATA_DIR; existing files were preserved.")
            if check:
                for _, manifest in segments:
                    for name, record in manifest["files"].items():
                        target = destination.joinpath(*_safe_relative(name).parts)
                        if not _file_matches(target, record):
                            raise RuntimeError("Missing or changed application resource: " + name)
            if not (destination / "UltraDarkFusion/UltraDarkFusion.py").is_file():
                raise RuntimeError("Application entry point is missing; existing workspace was preserved.")
            return destination
        if destination.exists():
            raise RuntimeError("An incomplete workspace exists; choose a new DARKFUSION_DATA_DIR. Existing files were preserved.")
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=parent) as temporary:
            staged = Path(temporary) / "app"
            staged.mkdir()
            names = set()
            for resource, manifest in segments:
                with resource.joinpath("application.zip").open("rb") as stream, zipfile.ZipFile(stream) as archive:
                    for name, record in manifest["files"].items():
                        relative = _safe_relative(name)
                        key = str(relative).casefold()
                        if key in names:
                            raise ValueError("Repeated application resource path")
                        names.add(key)
                        target = staged.joinpath(*relative.parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        blob = "blobs/" + record["sha256"]
                        info = archive.getinfo(blob)
                        if info.file_size != record["size"] or info.is_dir():
                            raise ValueError("Incorrect application resource size")
                        with archive.open(info) as source, target.open("xb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                        if not _file_matches(target, record):
                            raise ValueError("Application resource checksum failed")
            (staged / "prepared.json").write_text(json.dumps({"identity": identity, "version": version()}), encoding="utf-8")
            staged.rename(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DarkFusion desktop annotation and training application")
    parser.add_argument("--version", action="store_true", help="Show the installed product version")
    parser.add_argument("--prepare", action="store_true", help="Prepare the application without loading Qt or models")
    parser.add_argument("--check", action="store_true", help="Verify all bundled application files")
    parser.add_argument("--setup-media", action="store_true", help="Install the existing hash-pinned FFmpeg/FFprobe/Deno tools")
    parser.add_argument("--setup-models", action="store_true", help="Install and verify all required SAM3, GroundingDINO and DINOv3 checkpoints")
    parser.add_argument("--models-from", type=Path, help="Import verified required checkpoints from an existing Sam folder before downloading missing ones")
    args, qt_args = parser.parse_known_args(argv)
    if args.version:
        print(version()); return 0
    try:
        workspace = prepare(check=args.check)
        if args.prepare or args.check:
            print(("Verified: " if args.check else "Prepared: ") + str(workspace / "UltraDarkFusion")); return 0
        if args.setup_media:
            return subprocess.call([sys.executable, "-s", str(workspace / "scripts/install-media-tools.py")], cwd=workspace)
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["DARKFUSION_SETTINGS_PATH"] = str(data_directory() / "settings.json")
        env["DARKFUSION_MODEL_DIR"] = str(data_directory() / "models")
        env.setdefault("HF_HOME", str(data_directory() / "cache/huggingface"))
        env.setdefault("TORCH_HOME", str(data_directory() / "cache/torch"))
        model_command = [sys.executable, "-s", str(workspace / "UltraDarkFusion/darkfusion_model_setup.py"),
                         "--directory", env["DARKFUSION_MODEL_DIR"]]
        if args.models_from is not None:
            model_command += ["--source-directory", str(args.models_from.expanduser().resolve())]
        result = subprocess.call(model_command, cwd=workspace / "UltraDarkFusion", env=env)
        if result or args.setup_models:
            return result
        return subprocess.call([sys.executable, "-s", str(workspace / "UltraDarkFusion/UltraDarkFusion.py"), *qt_args], cwd=workspace / "UltraDarkFusion", env=env)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, KeyError, ModuleNotFoundError) as error:
        print("DarkFusion: " + str(error), file=sys.stderr)
        return 1
