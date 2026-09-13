# Native Windows setup and launcher

These small C++ executables provide an optional standalone DarkFusion installation.
The existing Python/Conda installer remains available. DarkFusion itself still runs
in Python; the standalone release supplies a private, relocatable runtime and does
not depend on a user's Conda installation or PATH.

Build on Windows x64 with Visual Studio 2022 Build Tools, the **Desktop development
with C++** workload, and a Windows 10/11 SDK:

```powershell
.\installer\native\build.ps1 -RunTests
```

The two executables appear in `installer/native/build`. They use the static MSVC
runtime (`/MT`), so the native setup/launcher do not require a separately installed
Visual C++ runtime. The Python runtime's native libraries are packaged separately.

Both executables embed `UltraDarkFusion/styles/icons/DarkFusion.ico`, using the
existing gold `Df1.png` artwork at nine Windows icon sizes (16 through 256 pixels).
Setup also sets its window icons, and installed shortcuts use `DarkFusion.exe`
as their icon source. Online setup embeds the launcher it just built and installs
that verified launcher before creating shortcuts. The runtime archive remains
independently verified; `launcher_sha256` identifies the embedded launcher in
the completed installation state.

The release builder places these files together for installation:

```text
DarkFusionSetup.exe
install-standalone.ps1
payload.zip
payload.json
```

Setup offers a writable destination folder and optional shortcuts. Its default is
`%LOCALAPPDATA%\Programs\DarkFusion`; it runs without administrator elevation. The
application writes settings and data inside its installed application folder, so
the chosen destination must remain writable by the current user. Setup supports:

```powershell
.\DarkFusionSetup.exe --install-dir 'D:\My Apps\DarkFusion' --quiet
```

Quiet setup returns the backend's exit code. It creates a Start menu shortcut and
no desktop shortcut. Logs are written to `%TEMP%\DarkFusion-install-<process-id>.log`.
The graphical setup displays progress and offers a button to open the log.

The backend command contract is Windows PowerShell `-NoLogo -NoProfile
-NonInteractive -ExecutionPolicy Bypass -File install-standalone.ps1` with named
`-PackagePath`, `-ManifestPath`, `-InstallDirectory`, and `-LogPath` arguments, plus
optional `-DesktopShortcut` and `-StartMenuShortcut` switches. No shell command
interpolation is used for destination or package paths.

The installed layout is:

```text
DarkFusion.exe
runtime\python.exe
app\UltraDarkFusion\UltraDarkFusion_v5.2.py
app\scripts\verify_install.py
```

The launcher uses private `python.exe` without a console window. Keeping
`sys.executable` pointed at `python.exe` preserves training subprocess output. It
prefixes the private runtime directories on PATH, disables user Python packages,
clears inherited Python/Qt overrides, and uses the application directory as its
working directory. `DarkFusion.exe --console` opens a diagnostic console;
`DarkFusion.exe --verify` runs the bundled installation verifier and returns its
exit code. GPU drivers and optional model files remain separate requirements.

The native smoke tests cover real child-process argument quoting, Unicode and
shell-special-character paths, inherited environment isolation, missing files,
and quiet setup exit-code propagation through a disposable backend fixture.
