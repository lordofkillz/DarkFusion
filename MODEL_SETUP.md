# Model files

Model checkpoints and TensorRT/ONNX exports are intentionally excluded from
Git. They are large, hardware-specific, and standard GitHub repositories reject
individual files larger than 100 MB.

## Required DarkFusion model bundle

The recommended Windows **DarkFusionSetup.exe** downloads and installs this
bundle automatically before setup finishes. The installed checkpoints are at
`<installation folder>\app\UltraDarkFusion\Sam\sam3.pt` and
`<installation folder>\app\UltraDarkFusion\Sam\groundingdino_swint_ogc.pth`.
No manual model download is needed with that installer.

For a Python/source installation or the optional offline runtime distribution,
download the
[DarkFusion 5.2 required model bundle (3.84 GB)](https://drive.google.com/file/d/1j9Y-WpUDjPt67_U43lafO-7dTkxLJuPS/view?usp=sharing),
then extract its `Sam` folder into the repository's `UltraDarkFusion` folder.
For an offline runtime installation, extract it into
`<installation folder>\app\UltraDarkFusion` instead. On the standard Python
installation path, verify that these files exist:

```text
C:\DarkFusion\UltraDarkFusion\Sam\sam3.pt
C:\DarkFusion\UltraDarkFusion\Sam\groundingdino_swint_ogc.pth
```

Bundle SHA-256:
`2B26091FC94CE73A6EA1554D1E8CFE5DD3A5BE9F0EA8A66A8CA32DEC07FC9A5A`

UltraDarkFusion can launch without the large checkpoints, but SAM3 snapping,
SAM3 masks/augmentation, and GroundingDINO auto-labeling require them.

## Required paths by feature

| Feature | Expected file | Notes |
| --- | --- | --- |
| SAM3 snapping, masks, and SAM3 augmentation | `UltraDarkFusion/Sam/sam3.pt` | Required for SAM3 tools. |
| GroundingDINO auto-labeling | `UltraDarkFusion/Sam/groundingdino_swint_ogc.pth` | Swin-T OGC checkpoint. |
| FSRCNN 4x super-resolution | `UltraDarkFusion/Sam/FSRCNN_x4.pb` | Small support model included in the repository. |
| YOLO/YOLOE inference and training | Any `.pt`, `.onnx`, or `.engine` selected in the UI | Keep user models outside the repository. |
| ReID tracking | `UltraDarkFusion/yolo26n-reid.onnx` | Optional; only required for the matching tracker configuration. |
| Legacy Darknet inference | User-selected `.cfg`, `.weights`, and matching class names | Optional legacy backend. |

GroundingDINO publishes its checkpoints from the official project:
https://github.com/IDEA-Research/GroundingDINO

The published bundle contains the checkpoints used with the pinned DarkFusion
environment. Keep the file names and directory layout unchanged.

## Visual similarity review

**Settings > Display > Review Preview > Matching method** defaults to AI visual
matching with [Meta's DINOv2 Base with registers](https://huggingface.co/facebook/dinov2-with-registers-base).
The first similarity search automatically downloads approximately 350 MB from
the official repository. No account, API key, or additional Python packages are
required. The checkpoint is pinned to revision
`a1d738ccfa7ae170945f210395d99dde8adb1805` and loaded from safetensors.

Model files and object descriptors are cached under
`UltraDarkFusion/.darkfusion_cache/review_similarity`. Subsequent searches reuse
unchanged image/box descriptors; fully cached searches need neither internet nor
a loaded model. Image or annotation changes invalidate the corresponding cache
entries. GPU inference uses FP16; CPU inference is available when CUDA is absent
or GPU memory is busy.

The matcher preserves whole object crops and their aspect ratio, with a small
margin. It searches annotations of the selected class and ranks results by
visual resemblance. A similarity percentage is not confidence that a label is
incorrect. Review matches before using the existing bulk-removal actions.
**Appearance and shape (CPU)** remains available in the matching-method setting.
Use **Stop scan** in the status bar to cancel; completed cached work is retained.

## TensorRT engines

Do not distribute one `.engine` as a universal model. TensorRT engines depend
on the TensorRT version, GPU architecture, precision, input dimensions, and
whether the export is dynamic. Distribute the source `.pt` through a GitHub
Release or another model host, then export an engine on the target computer.

Example from the `fusion` environment:

```powershell
yolo export model="C:\path\to\best.pt" format=engine imgsz=640 half=True device=0
```

Select the resulting engine in DarkFusion after export.
