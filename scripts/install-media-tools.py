"""Install hash-pinned private FFmpeg/FFprobe/Deno without changing PATH or pip."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile


APP_ROOT = Path(__file__).resolve().parents[1] / "UltraDarkFusion"
sys.path.insert(0, str(APP_ROOT))
from darkfusion_media_tools import (  # noqa: E402
    bundle_directory, media_file_matches, read_media_manifest, verify_bundled_media_tools,
)


def download_package(package: dict, cache: Path, *, offline: bool) -> Path:
    parsed = urlparse(package["url"])
    if parsed.scheme != "https" or parsed.hostname not in {"github.com", "raw.githubusercontent.com"}:
        raise ValueError("Media tools must use an approved HTTPS download host.")
    cached = cache / (package["sha256"] + (".zip" if package["kind"] == "zip" else ".txt"))
    if media_file_matches(cached, package):
        return cached
    if cached.exists():
        raise ValueError(f"Checksum-mismatched cached download: {cached}")
    if offline:
        raise FileNotFoundError(f"Pinned media download is not cached: {package['name']}")
    cache.mkdir(parents=True, exist_ok=True)
    # Incomplete downloads cannot be mistaken for a valid cache entry.
    with tempfile.NamedTemporaryFile(dir=cache, prefix="download-", suffix=".part", delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        request = Request(package["url"], headers={"User-Agent": "DarkFusion-media-setup"})
        with urlopen(request, timeout=60) as response, staged.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > package["size"]:
                    raise ValueError("Media download exceeds its pinned size.")
                output.write(chunk)
        if not media_file_matches(staged, package):
            raise ValueError(f"Media download checksum mismatch: {package['name']}")
        staged.replace(cached)
        return cached
    finally:
        staged.unlink(missing_ok=True)


def install_media_tools(app_root: Path, cache: Path, *, offline: bool = False) -> None:
    manifest = read_media_manifest(app_root)
    destination = bundle_directory(app_root)
    destination.mkdir(parents=True, exist_ok=True)
    for package in manifest["packages"]:
        records = package["files"]
        if all(media_file_matches(destination / record["path"], record) for record in records):
            print(f"Already verified: {package['name']} {package['version']}", flush=True)
            continue
        print(f"Preparing private {package['name']} {package['version']}...", flush=True)
        archive_path = download_package(package, cache, offline=offline)
        # Each replacement is staged and verified on the same volume; unrelated
        # files and existing working binaries survive a failed download/extract.
        with tempfile.TemporaryDirectory(dir=destination, prefix=".media-stage-") as temporary:
            staging = Path(temporary)
            archive = zipfile.ZipFile(archive_path) if package["kind"] == "zip" else None
            try:
                for index, record in enumerate(records):
                    staged = staging / str(index)
                    if archive is None:
                        shutil.copyfile(archive_path, staged)
                    else:
                        entries = [item for item in archive.infolist() if item.filename == record["source"]]
                        if len(entries) != 1 or entries[0].is_dir() or entries[0].file_size != record["size"]:
                            raise ValueError(f"Invalid or repeated media archive entry: {record['source']}")
                        with archive.open(entries[0]) as source, staged.open("wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                    if not media_file_matches(staged, record):
                        raise ValueError(f"Extracted media checksum mismatch: {record['path']}")
                for index, record in enumerate(records):
                    target = destination / record["path"]
                    if not target.resolve().is_relative_to(destination) or target.is_symlink():
                        raise ValueError("Unsafe media installation path.")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staging / str(index), target)
            finally:
                if archive is not None:
                    archive.close()
    failures = verify_bundled_media_tools(app_root)
    if failures:
        raise RuntimeError("; ".join(failures))
    print("Private FFmpeg, FFprobe and Deno verified (no system tools required).", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-root", type=Path, default=APP_ROOT)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    app_root = args.app_root.resolve()
    if args.verify_only:
        failures = verify_bundled_media_tools(app_root)
        if failures:
            parser.exit(1, "\n".join(failures) + "\n")
        print("Bundled FFmpeg, FFprobe and Deno verified.")
        return 0
    if sys.platform != "win32":
        parser.exit(1, "The pinned media bundle targets Windows x64.\n")
    cache = args.cache_directory or app_root / ".darkfusion_cache" / "media-tools-setup"
    install_media_tools(app_root, cache.resolve(), offline=args.offline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
