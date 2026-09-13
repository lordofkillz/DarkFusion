"""Build offline native Windows installation media from a verified runtime ZIP.

Requires Python 3.12, Git, a conda-pack archive rooted at runtime/, and the two
compiled native executables. End users require none of these build tools.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unicode_python(archive: zipfile.ZipFile, work: Path, manifest_tool: Path) -> Path:
    """Enable Unicode paths for native extensions without changing system locale."""
    executable = work / "python.exe"
    executable.write_bytes(archive.read("runtime/python.exe"))
    manifest = work / "python.manifest"
    subprocess.run([str(manifest_tool), "-nologo", f"-inputresource:{executable};#1", f"-out:{manifest}"], check=True)
    tree = ET.parse(manifest)
    assembly = "urn:schemas-microsoft-com:asm.v3"
    code_page = "http://schemas.microsoft.com/SMI/2019/WindowsSettings"
    application = tree.getroot().find(f"{{{assembly}}}application")
    if application is None:
        application = ET.SubElement(tree.getroot(), f"{{{assembly}}}application")
    settings = application.find(f"{{{assembly}}}windowsSettings")
    if settings is None:
        settings = ET.SubElement(application, f"{{{assembly}}}windowsSettings")
    setting = settings.find(f"{{{code_page}}}activeCodePage")
    if setting is None:
        setting = ET.SubElement(settings, f"{{{code_page}}}activeCodePage")
    setting.text = "UTF-8"
    tree.write(manifest, encoding="utf-8", xml_declaration=True)
    subprocess.run([str(manifest_tool), "-nologo", "-manifest", str(manifest), f"-outputresource:{executable};#1"], check=True)
    return executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-archive", required=True, type=Path)
    parser.add_argument("--native-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--version", default="5.2")
    parser.add_argument("--manifest-tool", type=Path, help="Windows SDK x64 mt.exe (automatically discovered by default)")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    runtime_archive = args.runtime_archive.resolve()
    native = args.native_directory.resolve()
    backend = repo / "installer/windows/install-standalone.ps1"
    manifest_tool = args.manifest_tool
    if manifest_tool is None:
        sdk = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Windows Kits/10/bin"
        candidates = sorted(sdk.glob("*/x64/mt.exe"))
        manifest_tool = candidates[-1] if candidates else None
    if manifest_tool is None or not manifest_tool.is_file():
        raise SystemExit("Windows SDK x64 mt.exe is required to configure Unicode paths in the private Python runtime.")
    for path in (runtime_archive, native / "DarkFusionSetup.exe", native / "DarkFusion.exe", backend):
        if not path.is_file():
            raise SystemExit(f"Required build input is missing: {path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Output directory must be new or empty.")

    source_commit = git(repo, "rev-parse", "HEAD").strip()
    tracked = [name for name in git(repo, "ls-files", "-z").split("\0") if name]
    # Publish only reviewed, tracked source. Never sweep personal models/settings.
    app_files = [name for name in tracked if not name.startswith(("installer/", ".github/"))]
    required = {"UltraDarkFusion/UltraDarkFusion_v5.2.py", "scripts/verify_install.py", "LICENSE.txt"}
    if not required.issubset(app_files):
        raise SystemExit("Repository application files are incomplete.")
    for name in app_files:
        source = repo / name
        if not source.is_file() or source.is_symlink():
            raise SystemExit(f"Tracked source is missing or linked: {name}")

    with zipfile.ZipFile(runtime_archive) as archive:
        names = set(archive.namelist())
        if not {"runtime/python.exe", "runtime/Scripts/conda-unpack-script.py"}.issubset(names):
            raise SystemExit("Runtime archive is not a Windows conda-pack archive rooted at runtime/.")
        if any(not name.startswith("runtime/") for name in names):
            raise SystemExit("Runtime archive contains files outside runtime/.")

    output.mkdir(parents=True, exist_ok=True)
    payload = output / "payload.zip"
    print("Packaging the private runtime with Unicode path support...", flush=True)
    with tempfile.TemporaryDirectory(prefix=".runtime-build-", dir=output) as directory:
        work = Path(directory).resolve()
        assert work.parent == output  # Temporary cleanup stays inside this new build folder.
        with zipfile.ZipFile(runtime_archive) as source, zipfile.ZipFile(payload, "w", allowZip64=True) as target:
            python = unicode_python(source, work, manifest_tool)
            for entry in source.infolist():
                encoded = copy.copy(entry)
                encoded.compress_type = zipfile.ZIP_DEFLATED
                encoded._compresslevel = 1
                if entry.filename == "runtime/python.exe":
                    target.writestr(encoded, python.read_bytes())
                else:
                    with source.open(entry) as src, target.open(encoded, "w", force_zip64=True) as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
    print("Adding application source and native launcher...", flush=True)
    with zipfile.ZipFile(payload, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for name in app_files:
            archive.write(repo / name, "app/" + name)
        archive.write(native / "DarkFusion.exe", "DarkFusion.exe")
        archive.writestr("app/STANDALONE_BUILD.json", json.dumps({
            "source_commit": source_commit,
            "source_dirty": bool(git(repo, "status", "--porcelain")),
            "version": args.version,
        }, indent=2))
        unpacked_size = sum(item.file_size for item in archive.infolist())
        count = len(archive.infolist())

    print("Hashing the completed payload...", flush=True)
    manifest = {
        "schema_version": 1,
        "product": "DarkFusion",
        "version": args.version,
        "source_commit": source_commit,
        "sha256": sha256(payload),
        "archive_size_bytes": payload.stat().st_size,
        "unpacked_size_bytes": unpacked_size,
        "file_count": count,
    }
    (output / "payload.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(native / "DarkFusionSetup.exe", output / "DarkFusionSetup.exe")
    shutil.copy2(backend, output / "install-standalone.ps1")
    shutil.copy2(repo / "LICENSE.txt", output / "LICENSE.txt")
    (output / "START HERE.txt").write_text(
        "DarkFusion standalone installation for Windows 10 (1903+) / 11 x64\n\n"
        "Keep all files in this folder together. Double-click DarkFusionSetup.exe,\n"
        "choose a new writable folder, and click Install. Python, Conda, Git, and\n"
        "compiler tools are not needed on the destination computer.\n\n"
        "The setup package contains the application and its private Python/ML runtime.\n"
        "Keep your NVIDIA graphics driver installed for GPU features.\n"
        "SAM3 and GroundingDINO models are separate: see app/MODEL_SETUP.md after\n"
        "installation. Optional YouTube and Darknet tools may need Deno or Darknet.\n\n"
        "Launch the installed DarkFusion.exe or its shortcut. Install to a new folder\n"
        "to change locations; do not move the installed runtime. This version refuses\n"
        "to overwrite existing nonempty folders.\n\n"
        "The existing Python installation method remains available on GitHub:\n"
        "https://github.com/lordofkillz/DarkFusion\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Installation media ready: {output}", flush=True)


if __name__ == "__main__":
    main()
