# Using UltraDarkFusion

This guide covers the normal DarkFusion workflow: collect images, create or
review labels, analyze the dataset, train a model, inspect its failures, and
repeat until the model works well for the intended scenario.

## 1. Install and launch

Follow the installation steps in [README.md](README.md). For SAM3 snapping,
SAM3 masks and augmentation, and GroundingDINO auto-labeling, also install the
[required model bundle](MODEL_SETUP.md).

Launch DarkFusion from the repository folder:

```powershell
.\run_darkfusion.bat
```

Use **Settings > Language** to change the interface language. All selectable
languages are stored locally under `UltraDarkFusion/translations`; changing
language does not require an internet connection.

DarkFusion enables **Settings > General > Auto-scale UI for screen size** by
default. It chooses a compact scale from the monitor's available logical
resolution, after Windows and Qt display scaling are applied. Turn it off to
select a manual scale from 75% to 125%, then choose **Apply UI Scale**.

Window and dock positions are restored only when the saved monitor size and
DPI are compatible with the current screen. If controls are clipped after a
monitor or Windows scaling change, use **Tools > Reset Layout** to rebuild a
responsive layout and replace the incompatible saved geometry.

## 2. Understand the workflow

A productive DarkFusion session normally follows this order:

1. Collect representative images.
2. Select the annotation task and define classes.
3. Auto-label or manually label the images.
4. Review every generated annotation.
5. Run Dataset Analysis and correct reported problems.
6. Open Trainer and evaluate the dataset.
7. Tune or train.
8. Review validation failures and test the resulting model on unseen video.
9. Add or correct data for the failure cases, then train again.

Keep a backup before running any operation that overwrites, deletes, or
automatically repairs labels.

## 3. Collect images

The training images should resemble the camera, resolution, viewpoint,
lighting, motion blur, distance, and object sizes where the model will
actually be used.

### Use an existing image folder

Load a folder that already contains images when the frames have been extracted
or photographed elsewhere. DarkFusion can work directly from the image source
folder and create matching YOLO label files.

### Extract frames from a local video

1. Open the video/frame extraction tools.
2. Select the local video.
3. Choose the destination folder and extraction interval.
4. Start extraction.
5. Load the resulting image folder for labeling.

Do not extract every frame from a long, mostly static video unless every frame
adds useful variation. Highly similar consecutive frames increase training
time and can make validation results look better than real-world performance.

### Download from a YouTube link

1. Open the download/extraction section.
2. Paste the YouTube URL.
3. Select the download or frame destination.
4. Download the video, then extract frames at a useful interval.

Only download material that you have permission to use.

### Capture a camera, screen, or image source

Select the appropriate capture source, camera, desktop region, or monitor.
Confirm the preview is showing the intended source before saving frames. Use
the same aspect ratio and visual conditions expected during inference whenever
possible.

## 4. Choose the task and classes

Select the correct task before labeling:

- **Detect** — axis-aligned bounding boxes.
- **Segment** — polygon masks.
- **Pose** — bounding boxes plus ordered keypoints.
- **OBB** — oriented bounding boxes.
- **Classify** — one class for an entire image.

Create the class list before a large auto-label run. Class IDs are positional:
class `0` is the first name, class `1` is the second, and so on. Do not reorder
the names after labeling unless the annotations are remapped too.

For pose datasets, define a consistent keypoint order and skeleton. For
segmentation, labels must contain polygon points rather than box-only rows.
Avoid mixing annotation formats in one dataset.

## 5. Create labels

### Manual labeling

Draw the geometry required by the selected task and assign its class. Zoom and
pan for precise edges. Snap assistance and SAM3 can help tighten boxes or
polygons, but the result still needs human review.

Use Next and Previous to move through images. The configurable skip control is
useful for moving quickly through videos while single navigation still moves
one image at a time.

### Auto-label with model weights

Open **Auto Label**, load the image folder, then select the model and settings.
DarkFusion supports these model paths:

- Ultralytics/PyTorch `.pt`
- ONNX `.onnx`
- TensorRT `.engine`
- Legacy Darknet `.weights` with its matching `.cfg` and class names

The model task must match the required output. A detection model produces
boxes; it cannot create genuine segmentation polygons or pose keypoints.
TensorRT engines are hardware and TensorRT-version specific, so an engine
should normally be exported on the computer that will run it.

Set confidence low enough to find difficult objects but high enough to keep
the review workload manageable. Start on a small sample before processing the
entire dataset.

### GroundingDINO

Use GroundingDINO when there is no suitable trained detector yet. Enter clear
class prompts that describe the visible objects. DINO is useful for producing
first-pass boxes, but its class mapping and object boundaries must be reviewed.
SAM3 can then help refine geometry where appropriate.

### SAHI slice labeling

Use SAHI when objects are small relative to a large image. SAHI divides the
image into overlapping tiles, runs the selected model on each tile, and merges
the results.

Important settings include:

- slice width and height;
- overlap ratio;
- confidence threshold;
- post-processing and match threshold;
- whether existing labels are overwritten;
- whether a standard full-image prediction is also performed.

More overlap can recover objects cut by tile boundaries, but it increases work
and can create more duplicate candidates.

### Compound multiple auto-label passes

Several labelers can contribute to the same dataset. For example:

1. Run a `.pt`, `.onnx`, or `.engine` detector.
2. Leave **Overwrite existing labels** disabled.
3. Run SAHI to recover small objects.
4. Run DINO for classes the detector misses.
5. Review and remove duplicate, wrong-class, or low-quality annotations.

Use overwrite only when the new pass is intended to replace the current label
files. Merging multiple passes without review can create duplicate boxes or
polygons.

## 6. Review annotations

Auto-label output is a starting point, not ground truth.

Check for:

- missed objects;
- false positives;
- wrong classes;
- loose or clipped boxes;
- inaccurate polygon edges;
- incorrect OBB rotation;
- missing, swapped, or invisible pose keypoints;
- duplicate annotations;
- labels on objects that should be ignored.

The preview panel shows individual annotation crops. Left-click a preview to
open its source image and flash the corresponding annotation on the main
viewer. Use this to move directly from a suspicious crop to the editable
label. The viewer automatically fits images while Next and Previous keep the
review inside the viewer.

If a category of object should always be ignored, include enough representative
images where it is present but unlabeled so the model learns it as background.
Keep the labeling rule consistent across the entire dataset.

## 7. Run Dataset Analysis

After labeling and manual review, open **Dataset Analysis** and select the
dataset root. It can identify problems such as:

- exact or near-duplicate images;
- missing, empty, or invalid labels;
- mixed detect/segment/pose/OBB annotation rows;
- out-of-range class IDs or coordinates;
- unusually small annotations;
- class imbalance and rare classes;
- image-size, stride, and task compatibility concerns.

Review the attached image or label for each issue before applying a fix.
Removing duplicates prevents the same scene from leaking across train and
validation splits. Re-run Dataset Analysis after repairs until the remaining
warnings are understood and intentional.

## 8. Prepare the training dataset

Keep training and validation data separate. Validation images should be
labeled ground truth and should not be near-duplicates of training images.
Whenever possible, reserve different video sections, scenes, or capture
sessions for validation.

Open **Trainer** and select the dataset and task. You can begin with:

- an official Ultralytics model downloaded from the
  [Ultralytics repository](https://github.com/ultralytics/ultralytics);
- another compatible local `.pt` model;
- a previous `best.pt` checkpoint for continued training.

Use a model variant that matches the task: detect, segment, pose, OBB, or
classify.

## 9. Evaluate before training

1. Turn on **Auto Create YAML** unless a carefully prepared dataset YAML is
   already being used.
2. Press **Evaluate Dataset**.
3. Confirm the inferred task, class names, split paths, annotation format,
   target sizes, and recommended image size.
4. Evaluate batch size or use automatic batch sizing.
5. Set workers according to available RAM, CPU, and storage performance.
6. Correct blocking dataset errors before starting a long run.

Larger image sizes preserve more pixels for small targets but consume more
VRAM and training time. Batch size mainly affects memory use and optimization;
it does not replace sufficient image resolution.

## 10. Tune or train

### Train directly

Use the evaluator recommendations as a starting point, select the epoch count,
and start training. Watch loss, precision, recall, and per-class validation
metrics rather than relying on one overall number.

Training runs in a separate process so the main interface can remain
responsive. Available controls include:

- **Stop after current epoch** for a clean checkpoint;
- **Stop now** for immediate termination;
- **Open run folder**;
- **Open best.pt**.

Closing the DarkFusion window does not necessarily stop a detached training
process. Use the training controls when the process itself should stop.

### Hyperparameter tuning

Tune when the dataset is already reasonably clean and there is enough compute
for multiple trials. Tuning can test learning rate, momentum, weight decay,
augmentation, and related settings. Use the best result as a candidate, then
perform a normal training run and validate it.

Tuning cannot repair incorrect labels, missing classes, duplicate leakage, or
an unrepresentative dataset.

## 11. Review the trained model

Use the Validation Review tools after training to inspect:

- false positives;
- missed ground-truth objects;
- wrong classes;
- weak localization or mask overlap;
- duplicate predictions;
- pose/keypoint errors.

Open a review result to return to the original image and annotation. Correct
bad ground truth, add hard negatives, or add more examples of weak scenarios.
Do not automatically change a correct label merely because an early model
disagrees with it.

The most useful improvement loop is:

1. train;
2. validate against trusted ground truth;
3. inspect failures;
4. correct labels or add representative examples;
5. re-run Dataset Analysis;
6. train again.

Finally, test `best.pt` or its exported ONNX/TensorRT version on video that was
not used for training or validation. This is the best check of whether the
model works in the actual scenario rather than only memorizing the dataset.

## 12. Practical rules

- Keep the annotation policy consistent.
- Prefer representative variety over thousands of nearly identical frames.
- Include negative/background images that resemble likely false positives.
- Review auto-generated labels before treating them as ground truth.
- Never mix task formats in the same label set.
- Back up labels before overwrite or automatic repair operations.
- Export TensorRT engines on the target hardware.
- Keep an untouched validation set for honest comparisons between runs.
