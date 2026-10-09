# Copy/Paste Augmentation Issue - Game Footage Solution

**Problem:** Health bars, UI indicators, and contextual labels don't copy/paste well
**Root Cause:** SAM3 segments visual pixels, not semantic concepts. Isolated segments lose meaning.
**Example:** 
- Health bar alone ≠ player (needs context)
- UI flash isolated looks wrong
- Segments from low-contrast elements are poor quality

---

## Best Strategy for Game Footage

### ✅ **Solution: Class-Aware Copy/Paste**

Instead of using Copy/Paste for everything, **disable it for problematic classes** and rely on augmentations that work better:

```
Classes to DISABLE Copy/Paste:
❌ Health bars
❌ UI indicators
❌ Minimap elements
❌ Score text
❌ Cursor/crosshair
❌ Any low-contrast UI

Classes that ARE GOOD for Copy/Paste:
✅ Player character (high visual quality)
✅ Enemies (distinct objects)
✅ Weapons (clear boundaries)
✅ Items (defined objects)
```

### Why This Works

**Copy/Paste works when:**
- Object is self-contained visually
- Segmentation captures the actual entity
- Object looks correct when isolated

**Copy/Paste fails when:**
- Label is contextual (health bar = indicator, not visual entity)
- Segmentation is poor (low-contrast UI)
- Object needs surrounding context

---

## Augmentation Strategy for Game Footage

**Instead of relying on Copy/Paste (which breaks):**

```
✅ Noise (Gaussian)
   Works on: ANY class
   Why: Adds sensor/compression noise, no segmentation needed
   Effect: Helps model learn from noisy video streams

✅ Clean BG (Bilateral Filter)
   Works on: Objects with clear foreground/background contrast
   Why: Softens background, keeps labeled objects intact
   Effect: Teaches model to focus on foreground objects

✅ Shadow
   Works on: ANY labeled class
   Why: Renders realistic shadows based on object geometry
   Effect: Helps model recognize objects in different lighting

✅ Object Detail (CLAHE)
   Works on: ANY labeled class
   Why: Enhances local contrast inside object masks
   Effect: Teaches model fine detail recognition

✅ Negatives (Background Crops)
   Works on: Any image with unlabeled regions
   Why: Samples background regions without objects
   Effect: Teaches model what NOT to detect (false positive prevention)
```

### Recommended Workflow for Game Footage

```
Settings:
- Augmentation Percent: 80%
- Copy/Paste: DISABLED (or only for high-quality classes)
- Noise: ENABLED ✅ (works on all footage)
- Clean BG: ENABLED ✅ (works on players/enemies)
- Shadow: ENABLED ✅ (works on all)
- Object Detail: ENABLED ✅ (works on all)
- Negatives: ENABLED ✅ (background learning)

Result: Diverse augmentations without false positives from bad segmentation
```

---

## Implementation Options

### Option 1: **Simple - Disable Copy/Paste Entirely** (Recommended for now)
- Uncheck "Copy/Paste" in SAM3 tools
- Use the 5 augmentations above instead
- Fast, no code changes, proven effective

### Option 2: **Better - Per-Class Copy/Paste Toggle** (Requires code change)
- Add UI to select which classes use Copy/Paste
- Classes marked "no copy/paste" skip segmentation step
- Other augmentations still run normally
- More work, but prevents false positives

### Option 3: **Advanced - Segmentation Quality Filtering** (Complex)
- Use DINO confidence score to gate Copy/Paste
- Only copy/paste if segmentation quality high
- Slower (needs DINO inference), but more selective

---

## My Recommendation

**Use Option 1 now** (simple, effective):
1. Disable Copy/Paste checkbox
2. Enable Noise + Clean BG + Shadow + Object Detail + Negatives
3. Set augmentation percent to 80-100%
4. This gives you **5 solid augmentations** without segmentation issues

**Why this works for games:**
- ✅ Noise: Mimics video compression/artifacts from streaming
- ✅ Clean BG: Helps distinguish player from scene
- ✅ Shadow: Lights/shadows are game-critical for detection
- ✅ Object Detail: Fine details matter in game UI/characters
- ✅ Negatives: Prevents model labeling UI elements as objects

**Results:**
- No false positives from bad segmentation
- More diverse dataset
- Better real-world performance on game footage
- Faster augmentation (no SAM3 segmentation)

---

## Code Enhancement (If Desired Later)

If you want **per-class Copy/Paste control**, I can add:

1. **Settings storage:**
   ```python
   'copyPasteExcludedClasses': ['health_bar', 'minimap', 'ui_flash']
   ```

2. **UI selector:**
   - SAM3 panel → New checkbox group "Copy/Paste Class Filter"
   - Check classes to EXCLUDE from copy/paste
   - They'll still get other augmentations

3. **Logic update:**
   ```python
   if chosen_class in self.settings.get('copyPasteExcludedClasses', []):
       continue  # Skip this class, use other augmentations instead
   ```

**Cost:** ~50 lines of code + UI controls  
**Benefit:** Surgical control per class

---

## Summary

| Issue | Current | Solution |
|-------|---------|----------|
| Health bars copy badly | Copy/Paste active for all | Disable Copy/Paste for UI classes |
| Segmentation poor | No filtering | Use DINO confidence (advanced) |
| Limited augmentation | Over-reliant on Copy/Paste | Use 5-augmentation strategy |
| False positives in train | Bad segments included | Skip problematic classes |

**Immediate action:** 
1. Uncheck Copy/Paste
2. Enable Noise, Shadow, Object Detail, Negatives, Clean BG
3. Run augmentation at 80% percent
4. Your dataset will be much better for game footage without segmentation issues
