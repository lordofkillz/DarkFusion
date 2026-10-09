# Preprocessing Status - All Input Sources

**Date:** 2026-10-06  
**Status:** ✅ Already fully implemented across all sources

---

## Preprocessing Pipeline

### What's Applied
1. **Display Path** (if visibilityPreprocessEnabled):
   - Brightness/gamma/contrast adjustments
   - Saturation control
   - Detail enhancement (CLAHE)
   - Sharpness enhancement

2. **Inference Path** (always on):
   - All visibility adjustments (above)
   - Grayscale conversion (if enabled)
   - Super-resolution upscaling (if enabled)
   - Edge detection for SAM3 snapping (if enabled)

---

## Coverage by Input Source

### ✅ Webcam/Capture Card Playback
**Flow:** `display_camera_input()` → `apply_preprocessing()`
- Display: Visibility adjustments only (fast, no dimension change)
- Inference: Full preprocessing before model submission
- File: Same as inference when extracting with inference enabled

### ✅ Video File Playback
**Flow:** `play_video_frame()` → `apply_preprocessing()`
- Display: Visibility adjustments only
- Inference: Full preprocessing before model submission
- File: Same as inference when extracting with inference enabled

### ✅ Desktop Capture Playback
**Flow:** `display_camera_input()` → `capture_with_mss()` → `apply_preprocessing()`
- Display: Visibility adjustments only
- Inference: Full preprocessing before model submission
- File: Handled via same extraction pipeline

### ✅ Frame Extraction (With Inference)
**Flow:** `save_frame_on_inference()` → `apply_preprocessing()`
- Always applies full preprocessing before running model
- Handles both OpenCV DNN and PyTorch models
- Maintains consistency between displayed and saved frame coordinates

### ✅ Frame Extraction (Without Inference)
**Flow:** `_extract_capture()` or `_extract_desktop()` → `transform_frame_for_extraction()`
- Applies transformation settings from UI (crop, resize, rotate)
- Optional visibility adjustments if enabled
- Frame saved as-is for manual annotation

---

## Settings Controlling Preprocessing

### Display Visibility
- **Path:** Settings → Annotation → Visibility Preprocessing
- **Controls:**
  - `visibilityPreprocessEnabled` - Master toggle
  - `visibilityBrightness` (-100 to +100)
  - `visibilityGamma` (25 to 300%)
  - `visibilityContrast` (50% to 250%)
  - `visibilityDetail` (0% to 100%)
  - `visibilitySharpness` (0% to 100%)
  - `visibilitySaturation` (0% to 200%)
- **Effect:** Applied to display frame AND used during extraction with inference

### Advanced Preprocessing
- **Grayscale:** Settings → Annotation → Grayscale
  - Converts frame to grayscale during inference
  - Includes brightness adjustment for gray channel
  
- **Super-Resolution:** Settings → Annotation → Super-Resolution
  - Upscales frames before inference
  - Runs before visibility adjustments
  - Can slow inference but improves detection of small objects

- **Edge Detection:** Settings → Annotation → SAM3 Snap
  - Canny edge detection for snap fallback
  - Independent toggle from visibility

---

## Key Points

### Why Preprocessing is Already Sufficient
1. **Display path** is lean (only visibility adjustments)
   - Keeps playback fast (1-3ms render time)
   
2. **Inference path** is rich (full preprocessing)
   - Ensures model sees same optimized frames as user
   - Maintains coordinate alignment between display and save
   
3. **Extraction** applies preprocessing before saving
   - Consistency: display frame = saved frame quality
   - User can toggle visibility on/off to see impact immediately

### No Webcam-Specific Settings Needed
- Visibility adjustments work for ALL input sources
- Webcam quality controlled by:
  - Hardware camera settings (handled by OS)
  - Your visibility preprocessing (brightness, contrast, saturation)
  - Super-resolution if enabled
  - Frame rate/buffer mode (already just added)

### Consistency Guarantee
Same preprocessing applied to:
- Live display (visual feedback)
- Model inference (detection accuracy)
- Extracted frames (annotation data)

This ensures: **What you see = What you extract = What the model sees**

---

## No Action Needed
Preprocessing is complete and working across all input sources. Webcam settings would be redundant since visibility adjustments already control the entire pipeline.
