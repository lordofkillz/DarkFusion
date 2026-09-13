# Optional native Windows installer

DarkFusion can be distributed with a C++ setup wizard and a private Python/ML
runtime. People using this distribution do not need to install Python, Conda,
Git, or build tools. The regular `fusion_install.bat` / `install.ps1` method
remains available for people who manage their own Python environment.

## Recommended installation

**[Download DarkFusionSetup.exe](https://github.com/lordofkillz/DarkFusion/releases/latest/download/DarkFusionSetup.exe)**,
open it, choose a new writable folder, and click **Install**. Setup downloads the
application/runtime automatically, checks the files, installs them and creates
your selected shortcuts. You only need to download the EXE yourself. When setup
finishes, click **Open DarkFusion** to launch the app immediately. The Start menu
shortcut is selected by default; a desktop shortcut is optional.

Allow approximately 25 GB free during setup and an internet connection for the
6.3 GB download. The installed application/runtime occupy about 11 GB. Download
progress appears in the window. Run setup again after an interrupted download
to resume it. Download files are removed after installation succeeds.

The [Python installation method](../README.md#python-installation) remains
available for people who manage their own environment. GitHub's source ZIP is
for that method; the setup EXE is linked above.

## Offline distribution (optional)

Maintainers can also supply a complete offline installation folder. The regular
online setup EXE does not require users to download or arrange these files.

If the complete distribution is supplied inside an outer ZIP, extract that ZIP
first. Keep the following files together and open **DarkFusionSetup.exe**.
Leave `payload.zip` intact; setup extracts it automatically.

```text
DarkFusionSetup.exe
install-standalone.ps1
payload.zip
payload.json
LICENSE.txt
START HERE.txt
```

Choose a new, writable folder on a local drive, such as
`D:\Applications\DarkFusion` or the suggested folder inside your user profile.
Allow approximately **11 GB** for the application/runtime, plus room for your
models, datasets, and generated files.
Setup verifies the package, extracts the application and runtime, configures
the runtime for the chosen location, and checks the required imports. It creates
the selected shortcuts after those checks pass.

Launch the installed **DarkFusion.exe** or its shortcut. The application uses
only its own Python environment; setup does not change system PATH, register
Conda, or install packages into another application's environment.

Windows 10 version 1903 or later, or Windows 11 x64, is required. Keep a compatible NVIDIA graphics driver for GPU
features. Application/runtime installation works offline from the complete
distribution. SAM3 and GroundingDINO checkpoints remain a separate download;
follow `app/MODEL_SETUP.md` inside the installation. Optional YouTube and Darknet
features still require their external Deno/Darknet tools.

This first installer supports fresh installations. It refuses to overwrite a
nonempty folder, including an existing DarkFusion installation. Choose a new
folder for another version and retain your datasets/models/settings as needed.
The app writes settings, logs and caches beside its files, so protected locations
such as Program Files are not supported. To change locations, install again at
the new location: a configured runtime must not simply be moved afterward.

## Build a distribution (maintainers)

Build on Windows x64 with Visual Studio 2022 C++ Build Tools and the Windows SDK.
Start with a dedicated Python 3.12 Conda environment installed and verified using
the repository's regular installer. Never package a personal environment that
contains unrelated applications, private packages, or credentials.

Create a separate packaging environment and install its build dependencies:

```powershell
python -m venv D:\DarkFusionBuild\build-env
D:\DarkFusionBuild\build-env\Scripts\python.exe -m pip install conda-pack==0.8.1 setuptools==78.1.1
```

Build the native executables and package the verified runtime:

```powershell
.\installer\native\build.ps1
.\installer\pack-runtime.ps1 `
  -EnvironmentPath C:\Miniconda3\envs\darkfusion-release `
  -OutputPath D:\DarkFusionBuild\runtime.zip `
  -BuildPython D:\DarkFusionBuild\build-env\Scripts\python.exe
```

The paths above are examples; use the actual build-machine locations. The setup
and launcher never use those build-machine paths to locate the installed runtime.
The environment's dependency check and conda-pack integrity checks must pass.

Commit the intended source changes before creating release media, then run:

```powershell
D:\DarkFusionBuild\build-env\Scripts\python.exe .\installer\build_distribution.py `
  --runtime-archive D:\DarkFusionBuild\runtime.zip `
  --native-directory .\installer\native\build `
  --output D:\DarkFusionBuild\distribution
```

The builder adds tracked application source, the license, and the native launcher
to a ZIP64 payload. It records the source commit, archive size, unpacked size, and
SHA-256 digest in `payload.json`. It excludes untracked models, datasets, settings,
and local diagnostics. Retain the complete distribution folder when sharing it;
the setup EXE alone does not contain the multi-gigabyte ML runtime.

The Windows SDK manifest tool enables UTF-8 paths in the bundled `python.exe`
while preserving its existing manifest settings. This lets native image libraries
read files from folders with international characters. It changes only the copy
inside the distribution; the source environment and Windows locale are preserved.
The native launcher also sets explicit private Qt plugin paths. Setup checks Qt
and MediaPipe startup before creating shortcuts.

The source tree contains installer build inputs. Prebuilt distribution files
must be built and shared separately; a source checkout alone is not the standalone
package. If publishing binaries, include this source commit and the bundled
third-party license files. Code signing can be applied to the native executables
before packaging when a signing certificate is available.

## Verification

For unattended installation into a disposable test directory:

```powershell
.\DarkFusionSetup.exe --quiet --install-dir 'D:\DarkFusion Install Test'
```

`DarkFusion.exe --verify` checks imports using the installed private runtime and
returns the check's exit code. `DarkFusion.exe --console` opens the application
with a diagnostic console. Setup logs its stages and any failure to the log path
shown by the wizard. Failed installation files are retained for diagnosis; setup
does not recursively delete a selected destination.

Runtime relocation uses [conda-pack](https://conda.github.io/conda-pack/).

## Publish the single-EXE installer (maintainers)

After building the offline distribution above, prepare the versioned runtime
downloads and compile their manifest and setup scripts into the native EXE:

```powershell
python .\installer\prepare_download.py `
  --distribution D:\DarkFusionBuild\distribution `
  --output D:\DarkFusionBuild\online `
  --release-tag v5.2.0-windows.1
.\installer\native\build.ps1 `
  -OutputDirectory D:\DarkFusionBuild\online\native `
  -OnlineManifest D:\DarkFusionBuild\online\download.json `
  -RunTests
```

Upload the generated `DarkFusion-runtime.*.bin` assets, `download.json`, and
`native/DarkFusionSetup.exe` to that exact GitHub release tag. The EXE embeds its
versioned download URLs and checksums, so upload all files before publishing the
release. The README download link targets only the setup EXE. Runtime assets are
fetched and verified automatically; end users do not assemble them.

Each runtime asset stays below GitHub's 2 GiB release-asset limit. Downloads are
cached on the selected installation drive, with range-based resume and SHA-256
verification. The installer preserves completed downloads when a connection
fails. It never downloads or uses a user's existing Conda installation.
