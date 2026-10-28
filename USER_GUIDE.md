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

Click a category's arrow or header to expand or collapse its controls. The same
category style is used throughout the workflow panels and grouped tool dialogs,
including Settings, capture/playback, labeling, dataset tools, and training/export.
DarkFusion remembers which sections you leave open. Collapsing a section only
hides its controls: it does not change settings or stop an active job. Scroll
the panel when expanded categories do not fit.

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

### Dock controls and help

The **Status** dock contains progress, annotation mode, class selection,
navigation, ROI, and shared **Label Tools**. Opacity, Rapid Delete, Zoom Lock,
Snapping, Visibility Options, and Auto Scan delay are available without leaving
the current workflow. The tools wrap onto two rows in narrower docks. The
separate Label tab has been removed; drawing still uses the main image viewer.
The **Workflow** dock contains Collect, Review, Prepare / Split, Train / Export,
and Advanced. Validation Review adds its own dock while reviewing model failures.
The Status dock fits its controls automatically, including the ROI selector and
wrapped tool rows. Review opens directly, without a duplicate inner Review tab.
Dock outlines and title dividers follow the selected theme. Drag the separator
between docks to resize them; it highlights on hover. Native float/restore and
close controls are unchanged, and the framing does not cover the image or video.

For clearer gameplay annotation images, open **Label Tools → Visibility Options**
and choose **Gameplay Clarity**. It lifts dark detail and strengthens local
contrast and existing edges, while tapering enhancement on bright effects and
saturated colors. The preset enables **Apply adjustments to annotation view only**:
the six adjustment sliders change the main annotation image while model inputs,
review thumbnails, source files and label coordinates keep their usual values.
Other preprocessing modes such as Enhance detail or Grayscale keep their existing
behavior. Use the existing **Opacity** slider to adjust label shading independently.
Hold **Hold for Original** to compare with the source.

**Enhance** can upscale the full image by 4× and adds processing time on each
frame. Turn it off to return to native resolution; its checkbox also reflects
the mode restored from your saved settings.

For small blurry objects, enable **Object crop clarity — Real-ESRGAN 4×** in
the same Visibility Options dialog. Choose **Review previews**, **Annotation
regions**, or both. Label crops are restored in the background and cached;
hover enlargement uses the enhanced preview too. **Draw clarity region** lets
you drag over an additional area without creating a label. Its blue border shows
**Enhancing…** during processing and **Clarity** when the restored section is ready;
Escape or right-click cancels. Drawn regions stay in the same place as you move between frames;
each frame is enhanced using its own pixels. Click **×** beside a region's caption
to delete that region, or use **Clear drawn regions** to remove all of them.
Region positions are remembered between sessions. Large regions can show less
detail than small crops. Selections stay aligned when Enhance or Hold for Original
changes the display size.
Hold **Hold for Original** to compare both the image and object previews.
Restored detail is an estimate, so use the original to check uncertain boundaries.
While Object Clarity is active for annotation regions, **SAM3 Snap** uses the
same completed clarity pixels when the drawn/edited box, polygon or OBB overlaps
an enhanced region. Snap reuses the existing restoration without another clarity
model pass. Turning clarity off, choosing Review previews only, or holding Original
restores the normal Snap input. Detector auto-label inputs follow their existing
preprocessing settings; source files and label coordinates are unchanged by viewing
clarity. It starts disabled; its first use
downloads the official 4.9 MB enhancement checkpoint if it is not already cached.

Saving identical label contents leaves the existing file and its modified time
intact. Actual edits still use the safe replacement path. Review keeps every
thumbnail and refreshes its crop, class, and annotation identity for the current
image; the live auto-label preview continues to update for each image.

In Review, left-click an object preview, then choose an action from the
**Annotation Actions** arrow. The main button shows and repeats your last choice
for the currently selected object, and remembers it between sessions.
**Find Similar Annotations**, **Change Class**, negative-crop actions, and other
annotation actions remain in the dropdown. The button stays disabled until an
object is selected. The scope below the bulk
buttons shows what **Clear Labels** and **Move Filtered** will affect. Clear
Labels uses all filtered images (not grid checkmarks); Move Filtered uses checked
images when present. Blanks are moved with their labels; class filters export
copies containing only that class's labels.

**Reveal File** selects the source image currently shown in the graphics view,
even when the Review selection differs. In grid mode it uses the selected
Review image.

Each dock has a real **?** button that explains the panel and can open this
guide. The neighboring dock button floats or restores the panel; it is not a
help button. The **X** hides the dock. Restore a hidden dock from the **View**
menu. Use **Tools > Reset Layout** if dock sizing becomes unusable.

Under **Prepare / Split > Prepare**, expand **Create Train / Valid** to choose
images and create the training files. Conversion, tiling, and moving images
have their own categories. Under **Augment**, shared percentage and
format controls are in **Augmentation Settings**; effects are in **OpenCV
Augmentation**, with **SAM3 Tools** separate. In **Advanced**, the parameter
table appears inside **CFG Editor** only after loading a CFG or YAML file.
**Save CFG As...** and **Hide Activation Layers** are also in the editor;
the **Anchors** category contains the anchor calculation controls.

**Move Images** uses its own **Source Folder** and **Target Folder**. The source
does not need to be loaded in the editor, and the active Review filter does not
select its files. Once confirmed, the move continues in the background when you
open another dataset. Its progress and result stay in the Move Images panel;
completion does not open a dialog over your work. Images and labels move as
pairs, with unique destination names when files already exist. Closing the app
requests cancellation and waits for in-flight pairs to finish safely.

**Collect** keeps Capture / Extract setup above Playback. **Extract** and
**Stop** stay at the bottom of the Workflow dock even when setup is collapsed
or scrolled. While extraction runs on another tab, this area shows its status,
**Stop**, and **Show Collect**. Choosing an output folder never starts extraction.
Use the top menu's **Dataset Analysis** entry for health checks and possible
false-positive review.

### Shadow and highlight augmentation

Under **Prepare / Split > Augment > SAM3 Tools**, enable **Shadow / Highlight**
and select **Shadows**, **Solid Highlight**, **Textured Highlight**, **Outline**,
or **Mixed**. Shadows vary darkness and retained color, sometimes shading only
part of the object. Solid Highlight creates a filled silhouette; Textured
Highlight retains surface detail; Outline keeps the original interior colors.
Mixed chooses one appearance per image, not a separate output for each style.

The **Color** swatch opens a color picker for highlights and outlines; red is
the default. The style and color are remembered independently of annotation
class colors. Existing settings default to Shadows, which does not use the swatch.
These effects apply to the existing labeled-object SAM3 masks, not a whole
bounding rectangle or an inferred hidden body. They do not classify enemies
versus teammates: use highlight styles on an appropriate dataset, keep real
unaltered examples, and check results against real footage.

Use the shared augmentation percentage and output format, then **Run SAM3**.
Variants are saved in the existing **shadow** subfolder with their annotation
coordinates unchanged (unless **Auto-Fit** is separately enabled). Original
images are not overwritten, and no additional SAM3 model pass is needed for
the appearance effects.

### Object clarity augmentation

Under **Prepare / Split > Augment**, enable **Object Clarity** in **OpenCV
Augmentation**, then choose **Run Augmentation**. It uses the same Real-ESRGAN
crop restoration as the annotation viewer, blended into existing boxes, polygons
or OBBs. In **SAM3 Tools**, **Object Clarity** followed by **Run SAM3** applies
restoration inside the labeled-object masks, including their holes and boundaries.
The existing **Object Detail** contrast/sharpening option remains independent.

Clarity variants are saved as new image/label pairs in **object_clarity** (or the
combined-effects folder for OpenCV). Image dimensions and label geometry are
preserved by this effect. Other selected augmentations, such as Flip, keep their
normal behavior. The shared percentage and output format still apply; choose PNG
for lossless output. Viewer-only clarity regions are independent of augmentation:
these tools use the dataset's annotations. Restored texture is estimated, so
compare the generated variants with their originals before training on them.

### Adjusting regions of interest (ROI)

The Status dock's **Enable ROI** button is a master switch: green means ROI is
enabled; clicking it again disables both regions. Enabling starts with **ROI 1**.
The dropdown below the button chooses which region the edge sliders adjust.
Selecting **ROI 2** adds the second region; it remains enabled when you switch
back to edit ROI 1. Turning the master off and back on starts with ROI 1 only,
but keeps both regions' last sizes and offsets for reuse.

Use the top/left edge sliders to move the selected ROI and the bottom/right
sliders to change its width/height. Its dimension badge shows image pixels
outside the boundary; there is no separate ROI window or numeric-entry step.
On video, a full-frame ROI has no free outside image area, so its badge sits at
the boundary. These overlays are display-only and are not saved into images.
ROI filtering keeps complete annotations touching either enabled region; it
does not crop images. **Prepare / Split > Prepare > Label Cleanup > Remove
Outside Labels** is a separate, explicit operation that removes saved labels
outside both regions. It is not an augmentation or a live inference setting.

For annotation clarity, enable an ROI and check **Clarity in ROI** beside the
ROI dropdown. Use the same edge sliders to position it. The border is amber
with **Enhancing…** while processing, then cyan with **Clarity active** once
the enhanced pixels are displayed. Both enabled ROIs are supported and each
new image uses its own pixels. This checkbox preserves existing inference
settings and works independently of Object crop clarity in Visibility Options.
Leave that crop option off when you want enhancement only inside the ROIs.
**Hold for Original** also compares ROI clarity; training pauses its GPU work.
**Draw clarity region** remains in Visibility Options for additional areas.

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

Changing workflow tabs does not stop playback or frame extraction. Extraction
keeps running, and its latest progress is restored when you return to
**Collect**, without overwriting another tab's progress. The completion notice
does not block other work; **Open Frames** opens the completed output only
when you choose it.

**Label Frame** still opens the displayed frame in the normal image editor;
removing the Label tab does not change either viewer or make live video directly
editable. Switching control tabs does not discard an editable frame. Returning
to an available Collect video surface saves edits before restoring playback.

Camera/capture-card extraction uses the same background worker and labeling
pipeline as other extraction sources. A detected frame is counted as saved only
after both its image and label are committed. A write failure stops extraction,
preserves existing outputs, and reports the error.

Interrupted local-video jobs can resume from a checkpoint that matches the
source file and extraction settings. Resume waits for your answer before
reading frames. Live camera, desktop, and direct-stream sources do not resume
historical frames. Old checkpoints without source/settings verification are
ignored rather than applied to a potentially different job.

Video keeps playing while its viewer is displayed. If you switch the viewer to
images or the grid, capture keeps advancing but unnecessary video rendering
and new live-inference work are skipped. Returning to **Collect** saves current
image edits and displays the latest video frame without restarting the source.
**Stop**, an actual input/dataset source change, and application shutdown still
stop and release the relevant capture/extraction resources.

### Download from a YouTube link

1. Open the download/extraction section.
2. Paste the YouTube URL.
3. Select the download or frame destination.
4. Download the video, then extract frames at a useful interval.

Only download material that you have permission to use.

The Windows installer includes private FFmpeg/FFprobe executables for merging
separate audio/video streams and Deno for YouTube's JavaScript challenges.
Playback, downloads and fallback caching use these same absolute tool paths;
there is no system PATH configuration to do. Source installation supplies the
same pinned bundle. If a tool is missing, rerun setup or follow
[the source repair instructions](UltraDarkFusion/INSTALL.md).

### Capture a camera, screen, or image source

Select the appropriate capture source, camera, desktop region, or monitor.
Confirm the preview is showing the intended source before saving frames. Use
the same aspect ratio and visual conditions expected during inference whenever
possible.

With video detection enabled, each completed prediction is displayed on the
exact frame that was analyzed. Capture continues normally with at most one
latest pending inference frame. New frames replace older pending work, not
the frame already being analyzed, and never accumulate in a backlog. The
annotated view therefore refreshes at the model's result rate; disabling
detection restores ordinary playback updates. Saved
labels, the frame position, and **Label Frame** refer to the displayed image.
Pausing or seeking refreshes that frame when needed, without an extra preview.

With **Detect** on, playback shows model predictions only; it does not layer
existing saved annotations over them. A frame with no predictions stays empty.
Turn **Detect** off to view saved labels during playback. This display choice
does not modify annotations or change **Label Frame**, labeling, or review.

The playback bar reports capture, inference, and render **FPS**; its tooltip
also reports processing times in milliseconds. **Drop** includes superseded
pending inference frames. GPU memory remains available in **System Monitor**
without additional VRAM polling from the playback bar. High display quality
retains its existing Area/Lanczos scaling; display resizing does not change
the model input or saved images.

In **Settings > Video**, **Website stream quality** controls the source
resolution for YouTube/website playback links, separately from on-screen
**Scaling**. The default prefers compatible H.264 video up to 1080p; choose
720p for lower bandwidth or Best available for more detail when offered by a
compatible stream. Higher resolution can increase decode and inference work.
Reopen the link after changing this setting. Local files and queued video
downloads are unaffected; this setting does not change playback speed or FPS.

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

In **Keypoints** mode, an existing or detected bounding box receives missing
point slots as `(0, 0, 0)` ghosts. They are editable placeholders, not predicted
coordinates or visible dots. An all-missing pose is ready for manual placement;
valid existing points remain intact. A pose model supplies actual points, while
a box-only detector can supply boxes plus ghosts. No object detection means no
new box or pose is invented. Auto-labeling preserves a compatible configured
keypoint list, including its names, order, colors and visibility settings.

## 5. Create labels

### Manual labeling

Draw the geometry required by the selected task and assign its class. Zoom and
pan for precise edges. Snap assistance and SAM3 can help tighten boxes or
polygons, but the result still needs human review.

Use Next and Previous to move through images. The configurable skip control is
useful for moving quickly through videos while single navigation still moves
one image at a time.

**Auto Scan** repeats Next or Previous in your chosen direction, using the Auto
Delay setting. The navigation keys also choose or reverse its direction. Each
step uses the same enabled tools and settings as manual navigation, including
**Propagate** and **Overwrite labels**. It waits for a frame's propagation and
saved labels before advancing again. Uncheck Auto Scan to stop repeating.
Propagate Range remains a separate operation.
**Auto Delay** sets the pause between completed steps; **0 ms** adds no pause.

Under **Settings → General → Display**, **Show annotation measurements**
shows width and height in image pixels while hovering over annotations or
drawing. Turning it off hides measurements in both cases.

**Show blank-image overlay** controls the small **BLANK** badge on images
without annotations. The badge scales with the visible image, stays in a
corner, and does not dim the frame. It hides while drawing and returns if the
drawing is canceled without adding a label. At extreme zoom-out it is omitted
if there is too little space to display it without covering the image.

The **Drawing** settings control outline thickness, hover emphasis, label
text, pose markers, and edit handles. Minimum and maximum annotation sizes
are checked in image pixels, including when zoomed in.

### Propagate corrected labels

Enable **Propagate** to carry your finished annotations to the next or previous
image and let SAM3 adjust their geometry. Correct the current frame before
continuing; those corrections become the next propagation seed. No preview is
required. **Propagate Range** handles an ordered image range or frames from a
local video, with forward/backward controls and stop-on-uncertainty safeguards.

When startup SAM3 warm-up is enabled and enough memory remains, the propagation
tracker warms in the background after Snap. This leaves **Propagate: Off** until
you enable it. Turning Propagate off keeps the same tracker ready for quick reuse;
training and other heavy GPU jobs can still unload it to free VRAM. If it was
released, enabling propagation reloads it in the background. Warming uses
synthetic frames and does not change your images or labels.

For an extracted image dataset, the range follows the active filtered image
list in its displayed order. Missing/deleted images are skipped, not recreated
from the original video. Sparse manual video labeling follows source frame
numbers but skips frames you explicitly deleted. **Until uncertain** means
continue until the end or a safety stop, not detect every new object in the
dataset. Tracking follows the starting annotations; it can stop at a scene
change or when those objects are lost. The final message retains the stopping
frame and reason rather than reporting an incomplete range as finished.

Range propagation reads and tracks images on demand in a background worker;
it does not first encode the entire folder into a temporary video. The progress
window shows loading, reading and tracking phases plus processed/total counts.
SAM3 keeps one continuous tracking session per direction. **Stop** requests a
safe stop: the window remains responsive while the current model or file-save
operation finishes. Already saved labels remain available for **Undo Last Batch**.
The normal **Propagate** toggle controls Next/Previous propagation only; it does
not need to be enabled to run **Propagate Range**.

**Overwrite labels** beside Propagate is off by default and saved per project.
When enabled, successful propagation replaces **all annotations on the target
frame**, including other classes; selected-object propagation can therefore
leave only that selected object. A failed or empty propagation result does not
clear existing labels. Corrections made while adjacent propagation is running
are protected rather than overwritten.

The range dialog offers **Existing labels > Overwrite existing labels** for the
same full-frame replacement. **Replace matching class** is narrower: it replaces
only overlapping annotations of the same class and retains unrelated labels.
Original files are backed up under the dataset's `.darkfusion/propagation`
folder. Use **Undo Last Batch** in Propagate Range to restore the last operation.
Keep a separate dataset backup before intentionally overwriting manual labels.

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

Exported `.onnx` models use Windows ML by default in labeling and video inference.
A manually selected backend/provider remains an explicit override. `.pt` and
`.engine` models keep their native Ultralytics loading paths.

The Windows ML **TensorRT RTX** provider uses its own native CUDA-graph replay
to reduce GPU submission overhead. This works with DarkFusion's ONNX input
pipeline and does not use the native `.engine` compatibility adapter. Automatic
provider selection compares speeds again after this change so it can account
for the faster RTX path; the fastest validated provider still depends on the
model, input size and hardware.

Compatible fixed-shape TensorRT `.engine` models use the current Ultralytics
backend's native CUDA-graph replay. DarkFusion does not add a second graph over
it. This reduces GPU submission overhead without changing project classes or
annotation formats. Dynamic engines and unsupported capture paths retain the
upstream execution path. A graph uses some additional GPU memory. DarkFusion's
older-release compatibility adapter is only used when Ultralytics has no native
graph implementation; `DARKFUSION_TENSORRT_CUDA_GRAPH=0` disables that adapter,
not Ultralytics' native implementation.

**Updating to TensorRT 11 requires rebuilding older TensorRT 10 engines.** Keep
the originals and export new copies from the original checkpoint on the target
GPU. The current requirements pin one CUDA 13 TensorRT family and include the
ModelOpt helpers needed for TensorRT 11 FP16/INT8 exports. Do not install
`nvidia-modelopt[onnx]` or `[all]` over this environment: those extras install a
conflicting ONNX Runtime variant. Use the repository installer and requirements
instead. Test the newly exported model before replacing an existing engine.

On Windows, **Automatic** prepares compatible certified Windows ML providers.
Maximum-performance mode compares their speed once per model/GPU/input size and
remembers the fastest numerically validated result. This adds first-use warm-up
and possibly compilation time, but no preview or repeated frame-by-frame benchmark.
The comparison does not change dataset annotations; it stores only timings and a
provider/device choice. Changed model, hardware, driver, runtime/provider or input
settings invalidate the result. Machine load can affect timings, so manual provider
selection remains available. Disable comparison to use the hardware recommendation.
Other power/NPU/CPU policies retain their requested hardware preference.
Use **Recompare speed on next model load** to rerun it after background work ends.

Expand the separate **ONNX / Windows ML** category in Auto Label to select or
refresh providers and GPUs, or download a compatible provider. It opens and
closes independently of **Shared Auto Label Settings**, which contains overwrite,
FP16 and teammate filtering. Use **Advanced / troubleshooting** to choose a policy/device, control cache,
download/offline and CPU-fallback behavior, or describe a metadata-free ONNX
export's task/output format/class names. The settings apply to Weights, compiled
YOLOE and SAHI when loading a new ONNX model. Native `.pt`, `.engine` and
Darknet `.weights`/`.cfg` loading and training are unchanged. See
[ONNX_RUNTIME_BACKEND.md](ONNX_RUNTIME_BACKEND.md) for model-contract limits and
the distinction between TensorRT RTX and classic CUDA/TensorRT plugins.

Set confidence low enough to find difficult objects but high enough to keep
the review workload manageable. Start on a small sample before processing the
entire dataset.

**SAM3 assist: refine boxes and fill short gaps** is optional under the Weights
tab and is off by default. It uses your existing SAM3 model to fit new YOLO
detection boxes to object masks. Uncertain masks keep the original YOLO box;
SAM3 does not change its class. ROI intersection, selected classes, size limits
and teammate-ignore still filter the final labels. Existing saved annotations
are preserved unless you enable overwrite. Segmentation, OBB and keypoint
outputs are unchanged.

For ordered, numbered frames from the same source, the same option also checks
short gaps of up to three images after the normal YOLO pass. It adds a missing
box only between strong YOLO detections of the same object, with agreement
between forward and backward SAM3 propagation. It can add an object to a frame
that already has other labels without replacing them. Scene changes, uncertain
tracking, conflicting boxes and changed/deleted files are skipped. **Undo Last
Batch** in Propagate Labels can undo these gap additions. There is no tracking
between unrelated or unnumbered images, no added preview, and no effect on live
inference. SAM3 adds processing time and is not a guarantee against gameplay
HUD false positives; review a small sample first.

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

The image viewer prepares nearby full-resolution originals in the background
to reduce waits during navigation. Review thumbnails and annotation crops keep
their existing rendering and image associations. Changed or replaced images
are read again before use.

Bulk YOLO image auto-labeling reuses the editor's full-resolution decode.
Preprocessing, review thumbnails, preview refreshes and label saving retain
their existing behavior.

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

In **Grid View**, use the corner checkboxes to choose images for batch deletion.
Click the first checkbox, then hold **Shift** and click the last checkbox to
check every image between them in the displayed order, including both ends.
Any other checked images stay checked. **Delete** removes the checked images
and their matching label files. Single-clicking a thumbnail outside its
checkbox still opens the image for labeling.

Checked-image deletion runs in the background and temporarily locks image-editing
controls while the selected pairs are processed. Failed pairs remain listed and
checked for retry; a locked image or label is rolled back rather than leaving a
partially deleted pair. Successful deletions still bypass the Recycle Bin.
If Windows also blocks rollback, the error lists retained recovery-file paths;
keep those files until recovery is complete.

Next/Previous in an extracted image dataset visits only surviving dataset images;
it does not re-extract gaps or deleted frames. Sparse manual video labeling still
opens adjacent source frames, but skips frames explicitly deleted in DarkFusion.
Use **Label Frame** explicitly if you want to restore one of those frames.

Dataset Analysis removes findings from its report only after their label changes
are saved successfully. Failed changes retain the original labels and findings.

The preview panel shows individual annotation crops. Left-click a preview to
open its source image and flash the corresponding annotation on the main
viewer. Use this to move directly from a suspicious crop to the editable
label. The viewer automatically fits images while Next and Previous keep the
review inside the viewer.

If a category of object should always be ignored, include enough representative
images where it is present but unlabeled so the model learns it as background.
Keep the labeling rule consistent across the entire dataset.

Use an annotation preview's context menu to find similar annotations. Adjust
the similarity threshold to narrow or broaden the matches, inspect the
results, and remove matching labels when appropriate. The default AI visual
matching uses DINOv3 Base to compare objects of the same class, including objects
with changes in lighting or background. A high similarity score means visual
resemblance; it does not establish that a label is incorrect.

While the search runs, the main progress bar shows its progress and the current
images remain available. When it finishes, **Review** opens and the image list
and annotation preview switch to the matching images and objects together.
A pending search is not an empty result: it must not blank the preview or remove
images from the list when
you edit a label. Completed results skip annotations deleted or changed during
the search. Opening a Dataset Analysis finding directly also enables this
workflow without first opening its dataset through File.

Choose the matching method in **Settings > General > Review Preview**.
The first DINOv3 Base search downloads approximately 343 MB; subsequent searches reuse
the model and cached object features. NVIDIA GPUs use FP16 inference, with CPU
support when CUDA is unavailable. **Appearance and shape (CPU)** keeps the
original matcher available. Use **Stop scan** in the status bar to cancel a
search. Thresholds may need adjustment when switching matching methods.

Negative-crop actions retain the selected region and available surrounding
context while excluding saved objects. Output goes into the dataset's
**blanks** folder with empty labels. Crops retain native resolution; only
dimensions shorter than 32 pixels are padded, without forcing a square.

## 7. Run Dataset Analysis

After labeling and manual review, open **Dataset Analysis** and select the
dataset root. It can identify problems such as:

- exact or near-duplicate images;
- missing, empty, or invalid labels;
- mixed detect/segment/pose/OBB annotation rows;
- out-of-range class IDs or coordinates;
- unusually small annotations;
- visual annotation outliers;
- class imbalance and rare classes;
- image-size, stride, and task compatibility concerns.

Review the attached image or label for each issue before applying a fix.
Removing duplicates prevents the same scene from leaking across train and
validation splits. Re-run Dataset Analysis after repairs until the remaining
warnings are understood and intentional.

Repeat scans automatically reuse image-integrity and quality measurements from
`.darkfusion/image_checks.sqlite3` inside the dataset folder. DarkFusion checks
the current image bytes before reuse, so replaced or repaired images are
rechecked. Labels are read again on every scan, and quality findings use your
current thresholds. The cache contains no labels; if storage is unavailable,
the image checks still run normally. No extra setting is required.

Image verification and quality checks prepare small groups in parallel while
the scan keeps its original image order and cancellation behavior. Content
hashing also runs on the loading workers to speed up repeat scans. Scans that
only need image dimensions continue reading headers without decoding pixels.

The possible-false-positive scan uses DINOv3 appearance comparisons, then the
enabled SigLIP 2 class-name and SAM3 shape checks. Decisive later-pass class
matches skip later verification work; the first-pass SigLIP rule remains
conservative. SAM3 shape references come only from annotations you explicitly
reviewed and kept in this dataset, not from the suspicious candidates being
judged. At least three independent usable reference masks per class are needed;
otherwise shape evidence stays unknown and the remaining checks continue.
The LLM is initialized only for unresolved eligible borderline cases. Uncertain
or conflicting class/shape/LLM evidence stays in review. These checks produce
review findings, not automatic label deletion or missing-label suggestions.
The LLM checks the individual annotation crop; invalid crops remain unverified
instead of silently using the whole image. Cached answers are tied to the full
image path, image content, crop, class, selected model and prompt version.
Timeouts, failed requests and malformed answers are not cached. Previous cache
formats are ignored automatically, so no manual cache clearing is needed.
SAM3 shape scores use real contour measurements and normalized geometry, not
raw pixel dimensions. Unchanged confirmed-good reference measurements are
reused across scans; changed images, annotations, SAM checkpoints or inference
settings invalidate them. Cache problems fall back to computation.

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

1. Turn on **Auto YAML** unless a carefully prepared dataset YAML is
   already being used.
2. Press **Generate Files + Parameters**.
3. Confirm the inferred task, class names, split paths, annotation format,
   target sizes, and recommended image size.
4. Evaluate batch size or use automatic batch sizing.
5. Set workers according to available RAM, CPU, and storage performance.
6. Correct blocking dataset errors before starting a long run.

**Recommended Args YAML > Browse** selects an existing training configuration
and remembers its path; it does not generate or overwrite that YAML. Use
**Generate Files + Parameters** to generate recommendations. Explicitly running
**Tune HParams** or **Calibrate Batch** can also update the selected configuration
with their results.

Training uses the currently selected configuration, not an earlier generated
copy. With **Recommended Args** off, tuning saves its hyperparameters inside the
selected dataset YAML and subsequent training reads them automatically. Changing
datasets or configurations while tuning runs does not redirect its results or
replace the new setup. Stop/Resume and the run-folder buttons continue to target
training, even when a validation or tuning job was launched more recently.

Larger image sizes preserve more pixels for small targets but consume more
VRAM and training time. Batch size mainly affects memory use and optimization;
it does not replace sufficient image resolution.

## 10. Tune or train

### Train directly

Use the evaluator recommendations as a starting point, select the epoch count,
and start training. Watch loss, precision, recall, and per-class validation
metrics rather than relying on one overall number.

The **Live Metrics** banner shows estimated time remaining, an approximate
finish time, and epoch progress once completed-epoch timings are available.
It uses recent epoch durations and updates when a run resumes or stops. The
estimate is based on the configured epoch target; early stopping can finish
sooner, and final validation or checkpoint saving can add time.

Open **System Monitor** to see GPU load, VRAM, CPU, RAM, drive usage, and
DarkFusion's resident memory (RSS). The CPU tile also shows the busiest logical
thread. Use its **Text size** control for larger text and **Keep Above** to
keep the monitor visible. GPU watts are NVIDIA GPU board power when available;
they do not measure the PSU or total system power.

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

### Export the trained model

Open the Trainer's **Export** tab and select a trained `.pt` checkpoint (or
**Use Best / Last**). DarkFusion calls the installed Ultralytics exporter;
using the same checkpoint, library version, and arguments uses the same export
implementation as `yolo export`.

Choose the target, width/height, supported precision, and CPU/GPU. Unchecked
**Simplify graph** and **Dynamic shapes** explicitly turn those options off.
Export defaults to **batch 1**, independently of the training batch. Leave
**Custom batch** unchecked for single-frame video inference. Enable it only
when you intentionally need a larger exported batch. Dynamic shapes are a
separate option; enabling Custom batch does not make the batch dynamic.
Unchecked **Embed NMS** keeps raw model output for runtime postprocessing.
For calibrated INT8 targets, select a representative dataset YAML.

**Extra arguments** accepts native `key=value` options and overrides the
controls above. For example, use `opset=17` for ONNX, or `workspace=4` for
TensorRT. Quote paths with spaces: `data="C:\datasets\my game\data.yaml"`.
Unsupported target options are rejected before export. Formats come from the
installed library's catalog; listing a format does not install its optional
SDK or mean that it supports Windows. Advanced mixed/dynamic INT8 choices
are enabled only for targets that support them.

Export runs in the background and leaves the active annotation/video model
unchanged. The completion message shows the actual output path. Test the
artifact on its intended inference backend, and check accuracy on held-out
data when changing precision. Build TensorRT engines on the target GPU.

## 11. Review the trained model

In the Trainer's **Validation Review** tab, **Metrics Only (mAP)** runs model
validation and reports accuracy scores without changing labels. **Find Validation
Problems** builds the image/object review queue for inspecting individual errors.
Both use the selected **Split**. For annotation datasets, **Entire Dataset**
scores the deduplicated union of the configured train/validation/test images;
its temporary input files stay in the metrics run folder. Choose **Validation**
to create a validation baseline comparable with training metrics. Training/test
scores are not used as validation baselines. Classification metrics require a
supported individual split rather than Entire Dataset.

For **Find Validation Problems**, leave **Images to scan** on **All** to review
the whole selected split. Prediction calls use small, bounded batches regardless
of dataset size; All does not send the whole dataset to the GPU together. The
report is updated during the scan. An inference failure retains a partial report
marked failed rather than reporting that the entire dataset completed.

GPU review scans can load the next small image group while processing the
current group. The inference batch limit and report-to-image mapping stay the
same; image loading does not write annotations.

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

## GPU memory during heavy jobs

Before training, evaluation, model export, Dataset Analysis, auto labeling, or SAM
augmentation, DarkFusion releases idle GPU models that the job does not need.
In-flight inference finishes before its model is released. Training and evaluation
reserve the GPU for their process lifetime, including when the Trainer window is
closed. Finish or stop an existing GPU job before starting training.

Optional Snap, propagation, clarity, and live detection models stay paused during
training. Propagation shows **Propagate: Paused**; turn it off to browse images
without propagation while training runs. Model selections and tool settings are
preserved. After the job, tools reload models when used, so their first operation
can take longer. Models are not automatically warmed again at job completion.

Dataset Analysis releases its worker models when the scan ends, including its
local Ollama model if one was used. The Ollama service stays running. GPU memory
owned by other applications is outside DarkFusion's control.

## 12. Practical rules

- Keep the annotation policy consistent.
- Prefer representative variety over thousands of nearly identical frames.
- Include negative/background images that resemble likely false positives.
- Review auto-generated labels before treating them as ground truth.
- Never mix task formats in the same label set.
- Back up labels before overwrite or automatic repair operations.
- Export TensorRT engines on the target hardware.
- Keep an untouched validation set for honest comparisons between runs.

## Language changes

Select the interface language in Settings. Cached translations apply immediately;
missing text is translated in the background and saved in `translations/*.json`.
English remains visible while a translation is unavailable. Language changes do
not rename files or change the values used by model, theme, GIF, or class selectors.
Online translation sends missing UI phrases to Google Translate with MyMemory as
a fallback. Paths, numbers, and recognized product/model terms are masked. Set
`translationOnlineEnabled` to `false` in settings.json for cached-only translation.

## DINOv3 on a new computer

Start a similarity search or visual outlier scan. The selected DINOv3 checkpoint
downloads once in the worker and is verified before use. Base is approximately
343 MB and Large is approximately 1.21 GB. Keep the app open until the download
finishes; **Stop scan** cancels it. See [Model setup](MODEL_SETUP.md) for offline
installation and the model license. DINOv2 and the CPU appearance method are also
available under Settings > Display > Review Preview.
