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

## Visual similarity review: DINOv3

**Settings > Display > Review Preview > Matching method** defaults to
**DINOv3 Base**. DINOv3 Large, DINOv2, and Appearance and shape (CPU) remain
available. Dataset Analysis also uses DINOv3 for visual class outliers.

The first uncached DINOv3 scan downloads the selected model in its background
worker. The interface remains usable and **Stop scan** cancels the work (an
active network read may take up to ten seconds to return). No account or token
is required. Subsequent scans reuse the checkpoint and cached descriptors.

| Model | Download | File in `UltraDarkFusion/Sam` |
| --- | --- | --- |
| DINOv3 Base | 343 MB | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
| DINOv3 Large | 1.21 GB | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` |

Downloads come from the versioned
[DarkFusion 5.2.1 release](https://github.com/lordofkillz/DarkFusion/releases/tag/v5.2.1-windows.1).
Each checkpoint is checked against its pinned size and full SHA-256 before
installation. Failed or cancelled downloads leave no model file; retry the scan
when connected. Existing files are preserved and verified before loading.
For an offline machine, copy the two named checkpoints into `Sam` yourself.
The release includes the [DINOv3 license](UltraDarkFusion/darkfusion_dinov3/LICENSE.md)
and [attribution](UltraDarkFusion/darkfusion_dinov3/NOTICE.md) accompanying Meta's
unmodified weights and vendored inference code. Upstream:
[Meta DINOv3](https://github.com/facebookresearch/dinov3).

Only the selected model downloads. These weights are separate from the existing
SAM3/GroundingDINO Google Drive bundle, which remains unchanged.

Descriptors are cached under `UltraDarkFusion/.darkfusion_cache/review_similarity`.
Unchanged, fully cached searches need neither internet nor a loaded model.
Image, annotation, model, and crop-setting changes invalidate relevant entries.
GPU inference uses FP16 with CPU fallback when CUDA is unavailable or VRAM is busy.
The optional DINOv2 method still downloads its pinned official safetensors model.

The matcher preserves object crops and aspect ratio, with a small margin. It
compares annotations of the same class. A similarity percentage means visual
resemblance, not confidence that a label is wrong; inspect matches before removal.

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
