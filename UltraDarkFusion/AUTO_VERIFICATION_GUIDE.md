# Auto-Verification System: Clean as You Go

## How It Works

When you navigate between images in Label Maker using **Next/Previous shortcuts** (arrow keys), the system automatically marks the current image as "verified/cleaned" before moving to the next one.

### Workflow

```
Image 001 [flagged as FP]
    ↓ You fix the labels
    ↓ Press → (Right arrow)
    ↓ AUTO: Image 001 marked as verified
    ↓
Image 002 [next flagged item]
```

## Key Features

### 1. **Auto-Mark on Navigation**
- Triggered by: Left/Right arrow keys (or Previous/Next buttons)
- Marks the current image as "cleaned"
- Prevents Dataset Analysis from re-flagging it in future scans
- Completely transparent—no extra clicks needed

### 2. **Persistent Storage**
- Verified images stored in: `.darkfusion/verified_images.json`
- Located in your dataset root directory
- Tracks filename, timestamp, and metadata
- Survives between sessions

### 3. **Filter in Future Scans**
- Dataset Analysis automatically reads verified list
- Skips verified images in next scan
- Shows count of skipped images in summary
- Focus only on new/unreviewed issues

### 4. **Manual Control**
You can also manually track verified images:
```python
from darkfusion_verified_images import VerifiedImagesTracker

tracker = VerifiedImagesTracker()
tracker.mark_image_verified("D:\\cod\\cod_train", "images/001.jpg")
tracker.is_verified("D:\\cod\\cod_train", "images/001.jpg")  # True
tracker.get_verified_count("D:\\cod\\cod_train")  # Returns count
```

## Typical Iteration Workflow

**Iteration 1:**
```
Run Dataset Analysis Scan
└─ Found: 500 flagged FP candidates
   └─ Open in Label Maker
      └─ Fix image 1 → Press → [auto marked verified]
      └─ Fix image 2 → Press → [auto marked verified]
      └─ Fix image 3 → Press → [auto marked verified]
      └─ ... (continues for ~2 hours)
      └─ Cleaned: 300 true FPs

Run Dataset Analysis Scan again
└─ Previously verified images: SKIPPED (300 images)
└─ Found: 250 new/missed FP candidates
   └─ Open in Label Maker again
      └─ Fix remaining issues...
```

## What Gets Marked?

**Marked as verified:**
- ✓ Images you navigated away from in Label Maker
- ✓ Manual marks via `tracker.mark_image_verified()`
- ✓ Any images in the verified list

**NOT marked:**
- ✗ Images you refresh without navigating
- ✗ Images you don't touch
- ✗ Images from other label maker instances

## Resetting Verified List

If you need to restart (clear all verified marks):

```python
from darkfusion_verified_images import VerifiedImagesTracker

tracker = VerifiedImagesTracker()
tracker.clear_verified_list("D:\\cod\\cod_train")
```

Or simply delete the file:
```
D:\cod\cod_train\.darkfusion\verified_images.json
```

## Benefits

| Before | After |
|--------|-------|
| Manually filter out old issues | Auto-filtered in future scans |
| Re-check images you already cleaned | Focus only on new FPs |
| Slow iteration (need to skip manually) | Fast iteration (automatic filtering) |
| No persistence between sessions | Persisted cross-session tracking |

## Status Display

In Dataset Analysis results panel, you'll see:
```
Analysis Summary
────────────────
Images processed: 1,000
Verified (skipped): 300
New candidates: 250
Time: 24 minutes
```

The "Verified (skipped)" count shows how many images were filtered out because you already cleaned them.

## Edge Cases

### Case 1: Manual label maker (not from Dataset Analysis)
- System still tracks verified images
- On next Dataset Analysis scan, they're skipped
- Useful for integrating your own label maker tools

### Case 2: Different datasets
- Each dataset has its own verified list
- `.darkfusion/verified_images.json` is per-dataset
- No cross-contamination between datasets

### Case 3: Network/shared drives
- Verified list stored locally in dataset
- Works fine with network paths
- Path normalization handles Windows + Unix paths

### Case 4: Retraining the model
- Verified list is independent of model
- You can keep it or clear it
- Recommended: Keep it if retraining is only fine-tuning
- Clear it if retraining changes detection behavior significantly

## Troubleshooting

**Q: Verified images still show up in next scan**
- Check the JSON file exists: `.darkfusion/verified_images.json`
- Verify Dataset Analysis "Scan for Visual Outliers" is enabled
- Clear and restart if file got corrupted

**Q: Want to un-verify an image**
```python
tracker.remove_from_verified("D:\\cod\\cod_train", "images/001.jpg")
```

**Q: Lost my verified list**
- Restore from backup if available
- Or clear and restart: `tracker.clear_verified_list()`
- The system will rebuild over time as you clean again

## Performance Impact

- **Scan time**: No change (just filters fewer images)
- **Memory**: Negligible (JSON file is small)
- **Storage**: <1KB per 1,000 verified images
- **Navigation speed**: Same (verification is instant)

## Next Steps

1. **Open Dataset Analysis** with your YOLO auto-labeled dataset
2. **Run Scan** (uses 3-pass validation + foreground-aware scoring)
3. **Open results in Label Maker** (your existing workflow)
4. **Navigate with arrow keys** (auto-marks images as verified)
5. **Re-run Scan** (skips already-cleaned images)
6. **Iterate** until dataset converges

That's it! The system handles the bookkeeping automatically.
