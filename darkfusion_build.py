"""Build PyPI distributions from committed DarkFusion source and public assets.

The ordinary desktop implementations are copied byte-for-byte. Development
settings, models, datasets, build products and uncommitted translations never
enter the payload. This module is a PEP 517 backend and a release build command.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "distribution/ultradarkfusion"
THEMES = ROOT / "distribution/.themes-build"
ROOT_FILES = {"VERSION", "LICENSE.txt", "README.md", "USER_GUIDE.md", "MODEL_SETUP.md", "ONNX_RUNTIME_BACKEND.md", "SECURITY.md", "requirements.txt", "scripts/install-media-tools.py", "scripts/install-windows-ml-runtime.ps1", "installer/windows/model-bundle.json"}
TRT = {"tensorrt-cu13", "nvidia-modelopt", "onnx-graphsurgeon", "onnxconverter-common", "polygraphy", "onnxscript", "cppimport", "lief"}
BUILD_ONLY = {"setuptools", "wheel", "ninja", "pip"}


def product_version() -> str:
    value = (ROOT / "VERSION").read_text(encoding="utf-8-sig").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise ValueError("Use the product's three-part VERSION")
    return value


def prepare_requirements() -> None:
    version = product_version()
    core, native = [], []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        name = re.split(r"[\[<=>!~ @;]", line)[0].lower().replace("_", "-")
        if name in BUILD_ONLY:
            continue
        if name == "clip":
            core.append("clip-anytorch==2.6.0")
            continue
        if name in {"torch", "torchvision"}:
            line = line.replace("+cu130", "")
        if " @ " in line:
            raise ValueError("PyPI dependencies must not contain direct repository URLs")
        (native if name in TRT else core).append(line)
    core.append("ultradarkfusion-themes==" + version)
    for name, rows in [("pypi", core), ("tensorrt", native)]:
        target = ROOT / "distribution" / ("requirements-" + name + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _committed_files():
    """Stream the exact Git snapshot, never the live application directory."""
    process = subprocess.Popen(["git", "-C", str(ROOT), "archive", "--format=tar", "HEAD"], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = member.name
                if name not in ROOT_FILES and not name.startswith("UltraDarkFusion/"):
                    continue
                if not member.isfile():
                    if not member.isdir():
                        raise ValueError("Do not package linked application resources: " + name)
                    continue
                with archive.extractfile(member) as stream:
                    yield name, stream.read()
        if process.wait() != 0:
            raise RuntimeError("Cannot read committed application source")
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate(); process.wait()


def prepare_payloads() -> None:
    prepare_requirements()
    version = product_version()
    marker = PACKAGE / "manifest.json"
    git_available = (ROOT / ".git").exists()
    if not git_available:
        if not marker.is_file() or not (PACKAGE / "application.zip").is_file():
            raise RuntimeError("Source distribution is missing its checked application payload")
        if json.loads(marker.read_text())["version"] != version:
            raise RuntimeError("Source distribution payload has an incorrect version")
        return
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if previous.get("source_commit") == commit and previous.get("version") == version and (THEMES / "src/ultradarkfusion_themes/manifest.json").is_file():
            return
    PACKAGE.mkdir(parents=True, exist_ok=True)
    theme_package = THEMES / "src/ultradarkfusion_themes"
    theme_package.mkdir(parents=True, exist_ok=True)
    manifests = [{"schema": 1, "version": version, "source_commit": commit, "files": {}} for _ in range(2)]
    seen = [set(), set()]
    destinations = [PACKAGE, theme_package]
    with zipfile.ZipFile(PACKAGE / "application.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as core, zipfile.ZipFile(theme_package / "application.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as themes:
        for name, data in _committed_files():
            if name == "VERSION":
                if data.decode("utf-8-sig").strip() != version:
                    raise RuntimeError("Commit VERSION before creating a release payload")
            index = int(name.lower().endswith(".gif"))
            digest = hashlib.sha256(data).hexdigest()
            manifests[index]["files"][name] = {"sha256": digest, "size": len(data)}
            if digest not in seen[index]:
                (themes if index else core).writestr("blobs/" + digest, data)
                seen[index].add(digest)
    required = {"VERSION", "UltraDarkFusion/UltraDarkFusion.py", "UltraDarkFusion/ui_ultradarkfusion.py", "UltraDarkFusion/styles/default.css", "scripts/install-media-tools.py"}
    if not required.issubset(manifests[0]["files"]):
        raise RuntimeError("The committed application is incomplete")
    for path, manifest in zip(destinations, manifests):
        (path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        (path / "VERSION").write_text(version + "\n", encoding="utf-8")
    (theme_package / "__init__.py").write_text('"""Versioned DarkFusion animation resources; installed automatically."""\n', encoding="utf-8")
    (THEMES / "VERSION").write_text(version + "\n", encoding="utf-8")
    shutil.copyfile(ROOT / "LICENSE.txt", THEMES / "LICENSE.txt")
    (THEMES / "README.md").write_text("# UltraDarkFusion theme assets\n\nAutomatically installed by ultradarkfusion. Contains the original version-matched animation resources.\n", encoding="utf-8")
    (THEMES / "pyproject.toml").write_text('''[build-system]
requires = ["setuptools>=77", "wheel>=0.45"]
build-backend = "setuptools.build_meta"
[project]
name = "ultradarkfusion-themes"
dynamic = ["version"]
description = "Original version-matched theme animations for UltraDarkFusion"
readme = "README.md"
requires-python = ">=3.12,<3.13"
license = "AGPL-3.0-only"
license-files = ["LICENSE.txt"]
authors = [{name = "DarkFusion contributors"}]
[project.urls]
Repository = "https://github.com/lordofkillz/DarkFusion"
[tool.setuptools]
package-dir = {"" = "src"}
packages = ["ultradarkfusion_themes"]
include-package-data = true
[tool.setuptools.package-data]
ultradarkfusion_themes = ["application.zip", "manifest.json", "VERSION"]
[tool.setuptools.dynamic]
version = {file = ["VERSION"]}
''', encoding="utf-8")


def get_requires_for_build_wheel(config_settings=None):
    prepare_requirements()
    from setuptools import build_meta
    return build_meta.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    prepare_requirements()
    from setuptools import build_meta
    return build_meta.get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    prepare_requirements()
    from setuptools import build_meta
    return build_meta.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    prepare_payloads()
    from setuptools import build_meta
    return build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    prepare_payloads()
    from setuptools import build_meta
    return build_meta.build_sdist(sdist_directory, config_settings)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    prepare_payloads()
    destination = args.outdir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for project in [THEMES, ROOT]:
        subprocess.run([sys.executable, "-m", "build", str(project), "--outdir", str(destination)], check=True, cwd=ROOT)
    artifacts = []
    for path in sorted(destination.iterdir()):
        if path.suffix not in {".whl", ".gz"}:
            continue
        if path.stat().st_size >= 100_000_000:
            raise RuntimeError("PyPI artifact exceeds the 100 MB default: " + str(path))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        artifacts.append({"file": path.name, "size": path.stat().st_size, "sha256": digest})
    if len(artifacts) != 4:
        raise RuntimeError("Expected application and theme wheel/source distributions")
    (destination / "build-verification.json").write_text(json.dumps({"version": product_version(), "artifacts": artifacts}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
