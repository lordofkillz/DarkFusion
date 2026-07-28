# Model files

Model checkpoints and TensorRT/ONNX exports are intentionally excluded from
Git. They are large, hardware-specific, and standard GitHub repositories reject
individual files larger than 100 MB.

UltraDarkFusion itself launches without these optional checkpoints. Install the
files for the features you use.

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

SAM3 availability and licensing may depend on the model provider. Use a
checkpoint compatible with the pinned Ultralytics version and place it at the
exact path above.

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
