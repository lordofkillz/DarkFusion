# Generate Files + Parameters

The Trainer estimates a **starting setup**, then saves a valid Ultralytics
training configuration in `train_recommendations.yaml` beside the dataset YAML.
The explanation and measurements remain in the evaluation report. Only the
training split drives these estimates. Validation remains available to measure
accuracy independently.

## Image size

Leave **Candidates** set to **Auto** to compare 320–2048, the current ImgSz, and
a size calculated from the dataset. The former default list ending at 832 is
migrated to Auto. A custom list still works, with the current ImgSz included.
Candidates round upward to the model's maximum stride (at least 32).

For each image, the evaluator projects annotations through aspect-preserving
resizing: `scale = imgsz / max(image_width, image_height)`. It measures boxes by
their short side, rotated boxes by their actual short edge, segmentation by its
polygon bounds, and pose by both the box and the nearest distinct visible
keypoint spacing. EXIF rotation is included. Images and labels are paired by
their full paths, so repeated filenames in different folders stay separate.

The chosen size is the smallest candidate that retains:

- Two minimum model strides of object width for at least 90% of objects and
  at least 50% of the objects **within each class**.
- One minimum stride for 98% of objects, and four pixels for 99.5%.
- For segmentation, the main target also requires six mask cells using the
  selected `mask_ratio` (24 input pixels at the default ratio of four).
- For pose, one stride between the nearest visible keypoints for 90% of
  annotations that contain at least two distinct visible points.

These are DarkFusion heuristics, **not Ultralytics accuracy guarantees**. Each
target is capped at the detail actually present in the source image. Upscaling
a five-pixel object cannot recover a sixteen-pixel object's detail. Source
limitations are reported even if the retention target is met.

If no candidate meets the target, DarkFusion chooses the smallest candidate
within 0.5% of the best continuous detail-retention score.
The score balances average object retention with average retention per class;
it does not clamp difficult datasets to a zero-score tie. The report recommends
comparing larger sizes or tiles when detail remains insufficient. No valid
annotations means retaining the current size, with an explanation. Sparse
classes are flagged as low confidence.

Measurements describe input geometry before augmentation. They cannot measure
texture, blur, thin mask structures, augmentation effects, or model accuracy.
The reported squared size ratio estimates input pixel cost, not measured speed
or GPU memory. Confirm the choice using **Train View**, validation, and
**Calibrate Batch**.

## Other parameters

- Optimizer selection defaults to `optimizer=auto`, allowing the installed
  Ultralytics version to use its actual training workload. Explicit extra
  arguments, including optimizer, learning rate and mask ratio, are preserved.
- A checkpoint filename no longer triggers guessed learning rates or automatic
  freezing. Explicit layer-freezing settings are preserved.
- Batch remains a rough starting estimate, capped using training images. Use
  **Calibrate Batch** to profile the chosen model and resolution.
- Epoch budgets use training-image count: 200 below 500 images, 150 below
  2,000, otherwise 100; patience is 25% of that budget. These are starting
  budgets to review against validation curves, not tuned optima.
- Rect keeps the selected setting. Substantial square padding produces a
  suggestion to compare Rect, since changing it also affects training behavior.

## References

- [Ultralytics training settings](https://docs.ultralytics.com/modes/train/)
  (`imgsz`, `rect`, `optimizer`, `batch`, and validation).
- [Ultralytics image loading](https://docs.ultralytics.com/reference/data/base/#ultralytics.data.base.BaseDataset.load_image)
  (aspect-preserving training resize).
- [Ultralytics segmentation configuration](https://docs.ultralytics.com/usage/cfg/#train-settings)
  (`mask_ratio`).

The implementation is `darkfusion_training_size.py`, integrated by
`MainWindow.evaluate_training_setup`. Regression tests cover both the geometry
and generated recommendation YAML in `tests/test_training_size.py`.
