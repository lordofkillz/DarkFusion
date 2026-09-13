# Optional native Windows installer

DarkFusion can be distributed with a C++ setup wizard and a private Python/ML
runtime. People using this distribution do not need to install Python, Conda,
Git, or build tools. The regular `fusion_install.bat` / `install.ps1` method
remains available for people who manage their own Python environment.

## Install from a standalone distribution

Keep these files together and open **DarkFusionSetup.exe**:

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
