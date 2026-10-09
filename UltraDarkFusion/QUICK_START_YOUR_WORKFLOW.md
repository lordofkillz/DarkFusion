#!/usr/bin/env python3
"""
QUICK START: Your Workflow (Auto-Label → Scan → Clean → Repeat)

This is your workflow optimized for speed and quality.
"""

WORKFLOW = """
================================
YOUR WORKFLOW (5 STEPS)
================================

STEP 1: Auto-label with YOLO (ONE TIME)
────────────────────────────────────────────────────────────────
  $ python -c "
  from ultralytics import YOLO
  
  model = YOLO('yolov8n.pt')  # or your custom model
  results = model.predict(
      source='D:/cod/cod_train/images',  # your images
      conf=0.3,  # low confidence (will filter)
      save_txt=True,
      project='D:/cod/cod_train/labels'
  )
  "
  
  ✓ Creates YOLO format .txt label files


STEP 2: Open in UltraDarkFusion
────────────────────────────────────────────────────────────────
  1. Launch UltraDarkFusion_v5.2.py
  2. Open Dataset tab (or use File → Open Dataset)
  3. Select D:/cod/cod_train (your auto-labeled dataset)
  4. Dataset Analysis section appears


STEP 3: Run False Positive Scan
────────────────────────────────────────────────────────────────
  In Dataset Analysis:
    • Click "Scan for Visual Outliers"
    • Wait for 3-pass scan to complete
      - Pass 1: Initial comprehensive scan
      - Pass 2: Different reference set (catches pass 1 FPs)
      - Pass 3: Final sweep
    
  Scan finds:
    ✓ High-priority FP candidates (visual + semantic mismatch)
    ✓ Protected: genuinely occluded valid objects
    ✓ Protected: manually verified labels


STEP 4: Review & Clean in Label Maker
────────────────────────────────────────────────────────────────
  For each flagged false positive:
  
    1. Click image in results → opens in your label maker
    2. Use your normal tools to verify/fix/delete
    3. Apply YOUR labeling rules (not automated)
    4. Mark as "trusted" if you confirm it's correct
    5. Move to next image
  
  This is the most important step:
    Your manual review ensures quality
    Your labeling rules drive the dataset


STEP 5: Repeat Scan
────────────────────────────────────────────────────────────────
  After cleaning 50-100 images:
    
    1. Run Dataset Analysis scan again
    2. It finds NEW false positives (didn't catch before)
    3. Clean those in label maker
    4. Repeat until very few FPs per scan
  
  Iteration improves:
    • Pass 1 → 500 FPs → you clean 300 → re-scan
    • Pass 2 → 250 FPs → you clean 150 → re-scan
    • Pass 3 → 40 FPs → you clean 25 → done


================================
KEY SETTINGS (Make Sure These Are On)
================================

In Dataset Analysis:

  Visual Outlier Scan:
    ☑ Enabled
    ☑ Semantic verification (SigLIP2)
    ☑ Foreground verification (SAM3 masks)
  
  Passes: 3 (increased from 1)
    • Pass 1: Comprehensive initial scan
    • Pass 2: Different references (catches pass 1 FPs)
    • Pass 3: Final validation
  
  Display:
    Sort by: Priority (high confidence FPs first)
    Show results by class


================================
WHAT YOU DON'T NEED
================================

You mentioned not using Dataset Analysis buttons:
  
  ✗ Keep button (redundant with label maker)
  ✗ Skip button (unnecessary)
  ✗ Next Issue button (you navigate via label maker)
  
BETTER:
  Dataset Analysis = "Find FPs"
  Label Maker = "Fix FPs"
  
  Clear separation. You're already doing this right.


================================
EXPECTED TIMELINE PER ITERATION
================================

Full dataset cleanup:

  Iteration 1:
    • Dataset Analysis pass 1-3: 10-20 min (depends on dataset size)
    • Label maker cleanup: 1-3 hours (depends on FP count)
    • Total: ~2-4 hours
  
  Iteration 2:
    • Analysis + cleanup: 1-2 hours (fewer FPs)
  
  Iteration 3:
    • Analysis + cleanup: 30-60 min (much cleaner)
  
  Done: ~5-7 hours for comprehensive cleanup


================================
ONE-COMMAND ITERATION CYCLE
================================

You could script this:

  for iteration in range(5):
      print(f"\\n=== ITERATION {iteration + 1} ===")
      
      # Step 1: Run scan
      print("Running Dataset Analysis scan...")
      app.scan_for_visual_outliers()  # (3-pass scan)
      
      # Step 2: Wait for you to clean
      print("\\nOpen Dataset Analysis, review flagged items in Label Maker")
      print("Mark corrected labels as 'trusted'")
      print("When done, press ENTER to run next scan...")
      input()
      
      # Step 3: Check if converged
      fp_count = app.visual_outlier_scan.get("queued_findings", 0)
      if fp_count < 10:
          print(f"\\n✓ Converged! Only {fp_count} FPs remaining.")
          break
      else:
          print(f"\\nFound {fp_count} FPs. Running next iteration...")


================================
INTEGRATION WITH YOUR YOLO WORKFLOW
================================

You probably have something like this:

  1. Train YOLO on dataset
  2. Inference on new footage → auto-labels
  3. DarkFusion Dataset Analysis → find FPs
  4. Manual cleanup → improved dataset
  5. Re-train YOLO → better model
  6. Repeat

This is exactly right! The loop improves both:
  • Dataset quality (fewer FPs)
  • Model quality (trained on cleaner data)


================================
PROTECTIONS BUILT-IN
================================

Things you manually verified won't be re-flagged:
  
  If you mark a label as "correct":
    • Dataset Analysis will protect it
    • Won't appear as FP in next scan
    • Only new anomalies get flagged
  
  Objects behind poles/sights:
    • SAM3 detects weak foreground (occlusion)
    • Lower visual similarity is expected
    • NOT flagged as anomaly
    • Your valid obscured labels stay safe


================================
RECOMMENDED FREQUENCY
================================

Suggested scanning schedule:

  New dataset (from YOLO):
    → Run scan immediately (1st iteration)
    → Clean thoroughly (all high-confidence FPs)
  
  After significant additions:
    → Run scan on new batch
    → Focus on classes with new examples
  
  Before final export:
    → Run 1-2 final scans
    → Ensure all obvious FPs are gone
  
  Ongoing monitoring:
    → Scan every 50 new images
    → Catch errors early


================================
YOUR SETUP
================================

Detected from your codebase:
  • 5090 GPU: Plenty of VRAM, can use full precision
  • SAM3 models: Available (foreground mask computation)
  • DINOv3: Available (visual anomaly detection)
  • SigLIP2: Available (semantic verification)
  
All tools ready for multi-pass scanning.


================================
NEXT STEPS
================================

1. Verify YOLO auto-labels are in place
2. Open UltraDarkFusion
3. Load your dataset
4. Click "Scan for Visual Outliers"
5. Review results in Label Maker
6. Mark verified labels as "trusted"
7. Re-run scan → iterate until clean

That's it. Your workflow is optimal!
"""

if __name__ == "__main__":
    print(WORKFLOW)
    print("\n" + "="*70)
    print("Ready to find and clean false positives!")
    print("="*70)
