# Windows installer

Download [DarkFusionSetup.exe](https://github.com/lordofkillz/DarkFusion/releases/latest/download/DarkFusionSetup.exe).
It installs DarkFusion with its own Python/ML runtime; no Conda, Git, compiler,
or administrator elevation is needed.

Choose a new, writable folder on a local drive, such as `D:\Apps\DarkFusion`.
Setup does not overwrite nonempty folders or change existing training environments.
Allow at least 35 GB free during setup, keep your NVIDIA driver installed, and
stay connected while the runtime and required models download.

Windows ML also requires the matching Microsoft Windows App SDK shared runtime.
Setup checks for it and, if missing, downloads the Microsoft-signed runtime
installer. This prerequisite may need Windows to permit software deployment or
request a restart; it does not install GPU drivers or change a training environment.
Compatible GPU providers download on first use; DirectML is included. Offline
use requires completing this initial runtime/provider setup first.

Setup verifies downloads by size and SHA-256. Interrupted downloads are cached
on the selected drive and can resume. Runtime `.bin` release assets are managed
by setup; do not assemble them yourself. SAM3 and GroundingDINO are installed
automatically. Other feature-specific models download on first use; see
[MODEL_SETUP.md](../MODEL_SETUP.md).

Launch the installed `DarkFusion.exe` or its shortcut. Keep the installed
runtime in place; use setup in a new folder when changing locations.
Your settings and datasets should be preserved separately before updating.

For unattended setup:

```powershell
.\DarkFusionSetup.exe --quiet --install-dir 'D:\Apps\DarkFusion'
```

Quiet setup returns a nonzero exit code if installation fails. Diagnostic logs
are at `%TEMP%\DarkFusion-install-<process-id>.log`; the graphical wizard also
offers an Open Log button.

`DarkFusion.exe --verify` checks the installed runtime and application files.
`DarkFusion.exe --console` starts with a diagnostic console.
For annotation, review, and training instructions, see [USER_GUIDE.md](../USER_GUIDE.md).
The [source installation](../README.md) remains available.
