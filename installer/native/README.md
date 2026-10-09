# DarkFusion setup and launcher

The release supplies two Windows executables:

- `DarkFusionSetup.exe` downloads and installs the application and private runtime.
- `DarkFusion.exe` launches that installed application.

The launcher resolves its runtime relative to its own folder, isolates Python/Qt
from unrelated system environments, and opens the application without a console.

```powershell
.\DarkFusion.exe --verify
.\DarkFusion.exe --console
```

Use `--verify` for an installation check and `--console` for diagnostic output.
Keep `DarkFusion.exe`, `runtime`, and `app` together in the installed location.
See the [installer instructions](../README.md) for setup, downloads, and logs.
