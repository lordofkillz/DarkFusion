# PyPI publishing

The Python packages are `ultradarkfusion` and its automatically installed
`ultradarkfusion-themes` resource companion. These distributions contain Python
application source and artwork, not the Windows EXE installer or a bundled
Python/Anaconda runtime. Existing manual PowerShell/conda and EXE installation
paths remain available independently.

Both versions come from the repository's `VERSION`. The launcher is `darkfusion`.
The separate resource distribution keeps every existing animation available
without exceeding PyPI's per-file limit. The original desktop implementation is
copied from the committed Git snapshot; local settings, weights, datasets and
uncommitted translations are excluded.

## Account configuration

Register these GitHub Trusted Publishers in the PyPI account's Publishing page.
This authorization is separate from linking the GitHub account:

| Field | Application | Resource companion |
| --- | --- | --- |
| PyPI project name | `ultradarkfusion` | `ultradarkfusion-themes` |
| Owner | `lordofkillz` | `lordofkillz` |
| Repository | `DarkFusion` | `DarkFusion` |
| Workflow filename | `publish-pypi.yml` | `publish-pypi.yml` |
| Environment | `pypi` | `pypi` |

An account owner must approve this publishing access. No password or long-lived
PyPI API token needs to be sent to an assistant or committed to GitHub.

PyPI permits only one pending project per identical publisher configuration.
For the first release, register `ultradarkfusion` first, then dispatch the workflow
with **publish checked** and **bootstrap checked**. This exchanges an ephemeral
OIDC credential to activate that project without uploading distributions. Now
register the pending `ultradarkfusion-themes` publisher with the same fields.
The ordinary publish run activates the companion and publishes both together.
After the first release, both projects reuse this one workflow and environment.

## Build and publish

The workflow runs only when dispatched. First run it with **publish unchecked**
and **bootstrap unchecked**.
It builds all four distributions, checks their metadata and sizes, installs the
wheels in a clean Windows Python environment, checks dependencies, verifies the
prepared application bytes, opens the main window offscreen and verifies settings
saving and shutdown. This smoke check disables background model warmup in its
own test settings and does not load models or use the user's datasets.

Review the passing build and artifacts. Once both Trusted Publishers are
configured, dispatch the workflow with **publish checked** and **bootstrap unchecked**. The publish job uses
short-lived GitHub/PyPI credentials. It does not rebuild an EXE or run the manual
source installer. PyPI descriptions come from `distribution/README.md`.

Local build command (after committing the reviewed source):

```powershell
python -m pip install build twine
python darkfusion_build.py --outdir dist
python -m twine check --strict dist/*.whl dist/*.tar.gz
```

The GitHub build uses standard CPython, without Anaconda. Required SAM3,
GroundingDINO and both DINOv3 checkpoints are installed automatically by the
first full launch. The setup reuses the pinned installer manifest and DINOv3
downloader; importing the package and checking application files do not download
models or load ML frameworks. GPU drivers and system runtime prerequisites
remain specific to the user's machine.
