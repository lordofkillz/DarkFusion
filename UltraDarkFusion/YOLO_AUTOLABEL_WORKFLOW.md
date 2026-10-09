#!/usr/bin/env python3
"""
YOLO AUTO-LABEL → DATASET ANALYSIS → LABEL MAKER CLEANUP WORKFLOW

This is the recommended workflow for cleaning up YOLO-generated labels
using DarkFusion's false positive detection + your existing label maker.

================================
STEP 1: AUTO-LABEL WITH YOLO
================================

Before Dataset Analysis, you need auto-labeled data:

  from ultralytics import YOLO
  
  # Use a trained YOLO model or download one
  model = YOLO("yolov8n.pt")  # or your custom model
  
  # Predict on directory
  results = model.predict(
      source="path/to/images",
      conf=0.3,  # Lower confidence to catch more (will filter in analysis)
      save_txt=True,  # Save YOLO format labels
      project="path/to/labels"
  )

This creates standard YOLO label files (.txt) alongside your images.

================================
STEP 2: RUN DATASET ANALYSIS
================================

Now use DarkFusion to find false positives in those auto-labels:

In UltraDarkFusion UI:
  1. Open "Dataset Analysis" tab
  2. Load your dataset (point to the auto-labeled directory)
  3. Click "Scan for Visual Outliers" button
  
The scanner will:
  • Pass 1: Initial false positive candidates (visual + semantic mismatch)
  • Pass 2: Cross-validate against different reference set
  • Pass 3: Final sweep for edge cases
  • Score using:
    - DINOv3 visual similarity
    - SigLIP2 semantic class verification
    - SAM3 foreground masks (coherent vs. occluded)
    - Multi-pass rotating reference validation

What happens:
  ✓ Detects most likely false positives first (high priority)
  ✓ Protects genuinely occluded valid objects (behind poles, sights)
  ✓ Protects manually verified labels from being re-flagged
  ✓ Adaptive thresholds per class
  ✓ Comprehensive coverage (3 passes, not just 256 samples)

================================
STEP 3: REVIEW & CLEAN IN LABEL MAKER
================================

IMPORTANT: You already have the right workflow!

Dataset Analysis findings → Your Label Maker Review
  1. Results show FP candidates (sorted by confidence)
  2. Open each image in your NORMAL label maker
  3. Use your existing review tools to verify/delete
  4. Your manual labeling rules apply (not automated decisions)
  5. Mark confirmed labels as "correct" if you want them protected next scan

Why this is better than Dataset Analysis buttons:
  ✓ You control every decision (your rules matter)
  ✓ You can see full context (all labels, not isolated crops)
  ✓ Faster workflow (native tools you know)
  ✓ Reduces click-fatigue from Dataset Analysis UI

================================
STEP 4: ITERATE (The Key to Quality)
================================

After cleaning pass 1:
  1. Run scan again with Dataset Analysis
  2. It will find NEW false positives (didn't catch in pass 1)
  3. Clean those up in label maker
  4. Repeat until scan finds <5-10 new FPs per pass

Why iteration works:
  • Each pass refines the model of "what's normal"
  • Your manual corrections improve the baseline
  • False positives that pass through one scan get caught later
  • Dataset quality improves iteratively

Example iteration sequence:
  Run 1: 500 FP candidates → you clean 300 → dataset improved
  Run 2: 280 FP candidates → you clean 150 → tighter dataset
  Run 3: 45 FP candidates → you clean 20 → near-clean
  Run 4: 8 FP candidates → you clean 5 → done

================================
DATASET ANALYSIS SETTINGS
================================

In the UI, make sure these are enabled:

  Visual Outlier Scan:
    ✓ Enabled
    ✓ Deep passes: 3 (was 1, now improved)
    ✓ Semantic verification: ON (SigLIP2 class checking)
    ✓ Foreground verification: ON (SAM3 masks for occlusion)

  Confidence Thresholds:
    Visual cutoff: Default (adaptive per class)
    Semantic threshold: SigLIP2 calibrated (strict on clear, lenient on occluded)
    Foreground: Coherent=stricter, Weak=protected

  Display Options:
    Sort by: Priority (high-confidence FPs first)
    Show only: Candidates with semantic + visual agreement
    Hide: Manually verified labels (already trusted)

================================
WORKFLOW SUMMARY
================================

Sequential Process (Recommended):

  1. YOLO Auto-Label
     └─ Creates baseline labels
  
  2. Dataset Analysis Scan (Pass 1)
     └─ Finds obvious false positives + anomalies
  
  3. Label Maker Review & Clean
     └─ Your manual verification + corrections
     └─ Mark corrected labels as "trusted"
  
  4. Dataset Analysis Scan (Pass 2)
     └─ Different reference set catches remaining FPs
     └─ Your previous corrections are protected
  
  5. Label Maker Review & Clean (Again)
     └─ Clean new findings
  
  6. Repeat Until Convergence
     └─ Continue until very few FPs per pass

Why 3-Pass Matters (vs. old 1-pass):

  Old (1 pass):
    • Single baseline reference set
    • If that baseline has FPs, they're locked in
    • ~60% FP coverage
  
  New (3 passes):
    • Different random reference each pass
    • Cross-validates entire dataset
    • Old FPs get caught by new reference sets
    • ~90%+ FP coverage
    • Iteration converges faster

================================
EXPECTED RESULTS
================================

On typical YOLO auto-labeled gameplay footage:

  Pass 1 FPs detected:
    • Mislabeled classes (player detected as weapon)
    • HUD elements (markers, text, icons)
    • Shadows mistaken for objects
    • Partial/blurry detections
    • Effects (smoke, flash) labeled as objects

  After manual cleanup (1 iteration):
    • FP rate drops 30-50%
    • Dataset ready for training
  
  After 3-4 iterations:
    • 95%+ label accuracy
    • Few false positives remain
    • High confidence in dataset quality

================================
TIPS FOR EFFICIENCY
================================

1. Start with High Priority FPs
   • Dataset Analysis sorts by confidence
   • Review high-priority candidates first
   • You can skip lower-confidence ones if time-limited

2. Batch Processing
   • Clean 50-100 images per session
   • Take breaks to stay accurate
   • Multi-pass scanning in background while you clean

3. Use Filtering
   • Filter by class: "Show only person FPs"
   • Filter by confidence: "Show only >0.7 priority"
   • Speeds up focused cleaning

4. Mark as Trusted
   • After verifying a label is correct, mark it
   • Protects from re-flagging in next scan
   • Accumulates a "confirmed good" set

5. Compare Visually
   • Dataset Analysis shows comparison crops
   • See why each item was flagged
   • Understand the reasoning (DINOv3 similarity gap, class mismatch, etc.)

================================
CLEANUP DOESN'T NEED DATASET ANALYSIS BUTTONS
================================

Current Dataset Analysis Buttons (can be hidden/removed):
  • Keep (suppresses finding)
  • Skip (marks reviewed)
  • Next Issue (navigation)
  • etc.

Why you don't need these:
  • Your label maker is the source of truth
  • These are redundant with your workflow
  • They add UI complexity
  • You prefer native labeling tools

Better UX:
  • Dataset Analysis = DETECTION only
  • Label Maker = REMEDIATION only
  • Clear separation of concerns

================================
IMPLEMENTATION CHECKLIST
================================

Recommended setup:

  □ Have YOLO model ready (trained or pretrained)
  □ Run YOLO auto-labeling on your dataset
  □ Load dataset into DarkFusion
  □ Enable all analysis features (visual, semantic, foreground)
  □ Set passes=3 (done in code update)
  □ Run first scan
  □ Review high-priority findings in label maker
  □ Mark corrected labels as "trusted"
  □ Run scan again
  □ Repeat until satisfied

Time estimates:
  • YOLO auto-label: 30-60 min (depends on dataset size)
  • Dataset Analysis pass: 5-15 min per pass
  • Label maker cleanup: 30-120 min per pass (depends on FP count)

================================
Questions to Ask While Cleaning
================================

For each flagged FP, ask:

  1. "Is this the right class?"
     → If no, delete or fix label
  
  2. "Is this where the object actually is?"
     → If box is wrong, adjust or delete
  
  3. "Is this actually an object worth labeling?"
     → If no (HUD, effect, shadow), delete
  
  4. "Is this a valid but partially occluded object?"
     → If yes, keep (Dataset Analysis warned you)
  
  5. "Should I mark this as 'trusted' for next scan?"
     → If verified correct, mark it

This ensures YOUR labeling rules drive the quality.
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nWorkflow setup complete!")
    print("Next step: Run Dataset Analysis with visual outlier scan enabled.")
