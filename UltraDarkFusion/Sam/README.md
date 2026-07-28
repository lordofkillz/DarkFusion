# Model directory

For complete DarkFusion functionality, download the
[required model bundle](https://drive.google.com/file/d/1j9Y-WpUDjPt67_U43lafO-7dTkxLJuPS/view?usp=sharing)
and copy its `Sam` folder into `UltraDarkFusion`.

- `sam3.pt` is required for SAM3 snapping, masks, and augmentation.
- `groundingdino_swint_ogc.pth` is required for GroundingDINO auto-labeling.
- `FSRCNN_x4.pb` is included for the optional 4x super-resolution tool.

The tokenizer/configuration files kept in this directory support the text-aware
model features. Large checkpoints are ignored by Git. See
[`MODEL_SETUP.md`](../../MODEL_SETUP.md) for the complete model layout.
