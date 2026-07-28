# Optional model directory

Place optional model files here:

- `sam3.pt` for SAM3 snapping, masks, and augmentation.
- `groundingdino_swint_ogc.pth` for GroundingDINO auto-labeling.
- `FSRCNN_x4.pb` is included for the optional 4x super-resolution tool.

The tokenizer/configuration files kept in this directory support the text-aware
model features. Large checkpoints are ignored by Git. See
[`MODEL_SETUP.md`](../../MODEL_SETUP.md) for the complete model layout.
