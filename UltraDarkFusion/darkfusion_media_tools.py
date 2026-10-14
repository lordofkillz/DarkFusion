"""Private, pinned media tools shared by playback, downloads, setup and QA.

Resolving tools never downloads anything, alters PATH, or starts a subprocess.
The install/build verifier deliberately ignores system tools so a developer's
Chocolatey/WinGet installation cannot hide an incomplete user package.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys


APP_ROOT = Path(__file__).resolve().parent
TOOL_PATHS = {
    "ffmpeg": "ffmpeg/ffmpeg.exe",
    "ffprobe": "ffmpeg/ffprobe.exe",
    "deno": "deno/deno.exe",
}


def bundle_directory(app_root: Path | str | None = None) -> Path:
    return Path(app_root or APP_ROOT).resolve() / "media_tools" / "windows-x64"


def read_media_manifest(app_root: Path | str | None = None) -> dict:
    root = Path(app_root or APP_ROOT).resolve()
    manifest = json.loads((root / "media-tools.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("platform") != "windows-x64":
        raise ValueError("Unsupported media-tool manifest.")
    packages = manifest.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ValueError("Media-tool manifest has no packages.")
    names, paths = set(), set()
    for package in packages:
        if package.get("name") in names or package.get("kind") not in {"zip", "file"}:
            raise ValueError("Invalid or duplicate media package.")
        names.add(package.get("name"))
        for record in [package, *package.get("files", [])]:
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))):
                raise ValueError("Invalid media-tool checksum.")
            if not isinstance(record.get("size"), int) or record["size"] <= 0:
                raise ValueError("Invalid media-tool size.")
        if not package.get("files"):
            raise ValueError("Media package has no files.")
        for record in package["files"]:
            relative = PurePosixPath(record["path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in record["path"] or ":" in record["path"]:
                raise ValueError("Unsafe media-tool destination.")
            key = record["path"].casefold()
            if key in paths:
                raise ValueError("Duplicate media-tool destination.")
            paths.add(key)
    if not set(TOOL_PATHS.values()).issubset(paths):
        raise ValueError("Media-tool manifest omits a required executable.")
    return manifest


def media_file_matches(path: Path, record: dict) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]


def verify_bundled_media_tools(app_root: Path | str | None = None, *, execute: bool = True) -> list[str]:
    """Verify bundled bytes and runnable versions, never a PATH fallback."""
    failures = []
    root = bundle_directory(app_root)
    try:
        manifest = read_media_manifest(app_root)
        for package in manifest["packages"]:
            for record in package["files"]:
                path = root / record["path"]
                if not path.resolve().is_relative_to(root) or not media_file_matches(path, record):
                    failures.append(f"Missing or checksum-mismatched bundled media file: {record['path']}")
        if failures or not execute:
            return failures
        versions = {package["name"]: package["version"] for package in manifest["packages"]}
        for tool, relative in TOOL_PATHS.items():
            expected = versions["ffmpeg" if tool == "ffprobe" else tool]
            command = [str(root / relative), "--version" if tool == "deno" else "-version"]
            result = subprocess.run(
                command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            first_line = result.stdout.splitlines()[0] if result.stdout else ""
            prefix = f"deno {expected}" if tool == "deno" else f"{tool} version {expected}"
            if not re.match(re.escape(prefix) + r"(?:[\s+-]|$)", first_line):
                failures.append(f"Unexpected bundled {tool} version: {first_line}")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        failures.append(f"Bundled media-tool verification failed: {exc}")
    return failures


def yt_dlp_options(app_root: Path | str | None = None) -> dict:
    """Use absolute private paths in every yt-dlp operation; no per-frame work."""
    root = bundle_directory(app_root)
    paths = {tool: root / relative for tool, relative in TOOL_PATHS.items()}
    if all(path.is_file() for path in paths.values()):
        return {
            "ffmpeg_location": str(paths["ffmpeg"].parent),
            "js_runtimes": {"deno": {"path": str(paths["deno"])}}
        }
    if sys.platform == "win32":
        missing = ", ".join(tool for tool, path in paths.items() if not path.is_file())
        raise RuntimeError(
            f"DarkFusion's bundled media tools are missing ({missing}). "
            "Run the latest DarkFusionSetup.exe into a new folder, or rerun "
            "fusion_install.bat for a source installation."
        )
    # Windows is the supported package target; retain manually managed tools on
    # other platforms without trying to execute a Windows binary.
    options = {}
    ffmpeg = shutil.which("ffmpeg")
    deno = shutil.which("deno")
    if ffmpeg:
        options["ffmpeg_location"] = str(Path(ffmpeg).resolve().parent)
    if deno:
        options["js_runtimes"] = {"deno": {"path": deno}}
    return options
